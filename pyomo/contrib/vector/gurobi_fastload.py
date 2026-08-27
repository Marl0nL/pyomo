# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Transparent fast solver hand-off for classic linear models -- Gurobi backend.

This is the Gurobi twin of :mod:`pyomo.contrib.vector.fastload` (the HiGHS
``highs_fastload`` solver).  It reuses the *same* solver-neutral compile --
:func:`pyomo.contrib.vector.fastload.compile_to_highs_arrays`
(alias :data:`~pyomo.contrib.vector.fastload.compile_fastload_arrays`), which
walks a classic model once through the fast ``pyomo.repn`` visitors and emits a
single :class:`~pyomo.contrib.vector.fastload.FastLoadCompiled` (standard-form
range-row arrays: ``A``, row/col bounds, integrality, a linear cost, and an
optional objective Hessian) -- and hands the whole matrix to Gurobi in a handful
of *bulk-array* calls (``Model.addMVar`` + ``Model.addMConstr`` +
``Model.setMObjective``, gurobipy's native matrix API), instead of the classic
per-constraint ``addConstr`` load.  No user model change is required::

    from pyomo.contrib.solver.common.factory import SolverFactory
    results = SolverFactory('gurobi_fastload').solve(model)
    # or, via the legacy factory:
    pyomo.SolverFactory('gurobi_fastload').solve(model)

Scope (identical to ``highs_fastload``): **linear** continuous / MIP models, plus
a **convex-quadratic objective** (linear constraints).  A model that falls
outside this is rejected loudly:

* nonlinear terms / components the standard-form compiler cannot process are
  caught at compile time (pointing at a classic solver route);
* a **non-convex** quadratic objective is rejected by Gurobi's own PSD check
  (``NonConvex=0``), surfaced here as an :class:`IncompatibleModelError`;
* an **MIQP** (integer variable + quadratic objective) is rejected up front --
  Gurobi *can* solve convex MIQP, but ``NonConvex=0`` does not enforce
  convexity for integer models (a non-convex MIQP would still solve silently),
  so a correct MIQP path needs an explicit convexity gate that is out of scope
  for this backend.  Convex MIQP is the natural follow-up (see the PR body).

The fast path never silently produces a wrong answer: anything it cannot map
exactly onto Gurobi's convex matrix API fails loudly.
"""

from __future__ import annotations

import datetime
import io
import math
import time
from typing import Mapping, Sequence

from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.enums import ObjectiveSense
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer

from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager

from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
    get_infeasible_results,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoader
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    IncompatibleModelError,
)
from pyomo.contrib.solver.common.factory import SolverFactory

# Reuse the solver-neutral compile + the compiled-arrays container from the HiGHS
# fastload module (this backend adds no new compiler -- only an array->Gurobi
# builder and a Gurobi map-back).
from pyomo.contrib.vector.fastload import (
    compile_fastload_arrays,
    FastLoadCompiled,
    _ColumnarMapBackMixin,
)

# gurobipy is imported lazily inside the methods that need it (this module is
# imported at ``pyomo.contrib.vector`` import time to register the solver, and we
# do not want to force-import gurobipy then).

# Gurobi error code for a non-PSD objective Hessian (a non-convex QP under
# ``NonConvex=0``).  Surfaced as an IncompatibleModelError (convex QP only).
_GRB_ERROR_Q_NOT_PSD = 10020


# --------------------------------------------------------------------------- #
# FastLoadCompiled range rows -> Gurobi one-sided rows (sense + rhs + map-back)
# --------------------------------------------------------------------------- #
def _gurobi_rows(compiled: FastLoadCompiled):
    """Translate ``FastLoadCompiled`` range rows into Gurobi one-sided rows.

    Gurobi's ``addMConstr`` takes one sense char (``'<'`` / ``'>'`` / ``'='``)
    per row, so a genuinely two-sided range row (``lo <= A x <= hi`` with both
    finite and ``lo != hi``) is split into two rows sharing one
    :class:`ConstraintData` (a ``'<'`` row at ``hi`` and a ``'>'`` row at
    ``lo``).  The linear standard-form compile already splits ranges, so this
    only fires for models built with template-vectorized construction (the
    templated compile keeps genuine two-sided rows).

    Returns ``(sel, sense, rhs, con_of)``:

    * ``sel``   -- old-row index per Gurobi row (``None`` if no reordering
      needed, i.e. every row mapped 1:1 -- the common case);
    * ``sense`` -- list of sense chars, one per Gurobi row;
    * ``rhs``   -- list of right-hand-side floats, one per Gurobi row;
    * ``con_of``-- list of :class:`ConstraintData`, one per Gurobi row (for the
      dual map-back).
    """
    rl = compiled.row_lower
    ru = compiled.row_upper
    cons = [r[0] for r in compiled.rows]
    n = len(cons)

    sel, sense, rhs, con_of = [], [], [], []
    split = False
    for i in range(n):
        lo = rl[i]
        hi = ru[i]
        lo_f = np.isfinite(lo)
        hi_f = np.isfinite(hi)
        if lo_f and hi_f and lo == hi:
            sel.append(i)
            sense.append('=')
            rhs.append(float(lo))
            con_of.append(cons[i])
        elif lo_f and hi_f:
            # genuine two-sided range -> two Gurobi rows on the same constraint
            split = True
            sel.append(i)
            sense.append('<')
            rhs.append(float(hi))
            con_of.append(cons[i])
            sel.append(i)
            sense.append('>')
            rhs.append(float(lo))
            con_of.append(cons[i])
        elif hi_f:
            sel.append(i)
            sense.append('<')
            rhs.append(float(hi))
            con_of.append(cons[i])
        elif lo_f:
            sel.append(i)
            sense.append('>')
            rhs.append(float(lo))
            con_of.append(cons[i])
        # else: a free row (both open) carries no constraint -- drop it.
    # If every row mapped 1:1 in order, the selector is the identity; signal that
    # with None so the caller can hand Gurobi the compiled matrix untouched.
    if not split and len(sel) == n:
        sel = None
    return sel, sense, rhs, con_of


def _hessian_to_gurobi_Q(H):
    """Symmetric Gurobi objective matrix ``Q`` from the compiled Hessian.

    :class:`FastLoadCompiled` carries the objective Hessian in the HiGHS
    convention -- ``0.5 x' H x`` with ``H`` the (symmetric) Hessian stored as its
    lower triangle.  Gurobi's objective is ``x' Q x`` (no ``1/2`` factor), so the
    equivalent matrix is ``Q = 0.5 * H_symmetric``.  Reconstruct the full
    symmetric ``H`` from the stored lower triangle and halve it; passing a
    symmetric ``Q`` makes Gurobi's convexity (PSD) check act on the true Hessian.
    """
    from pyomo.common.dependencies import scipy

    d = H.diagonal()
    # H (lower) + H^T doubles the diagonal and mirrors the off-diagonals; undo the
    # double-counted diagonal to recover the symmetric Hessian.
    H_sym = H + H.transpose() - scipy.sparse.diags(d)
    return (0.5 * H_sym).tocsr()


# --------------------------------------------------------------------------- #
# Solution loader: map the captured Gurobi solution back onto Pyomo objects
# --------------------------------------------------------------------------- #
class FastLoadGurobiSolutionLoader(_ColumnarMapBackMixin, SolutionLoader):
    """Map a solved Gurobi model's solution back onto the original Pyomo objects.

    Columns are kept in the exact order the standard-form compiler produced them,
    so primal values / reduced costs index straight into the captured Gurobi
    solution vectors -- no per-object solver map is needed.  Columnar Vars map
    back in bulk via ``column_scatter`` (shared with ``highs_fastload`` through
    :class:`~pyomo.contrib.vector.fastload._ColumnarMapBackMixin`).  Duals map
    through the per-Gurobi-row ``con_of`` list (a two-sided range constraint
    contributes two rows; the larger-magnitude dual is kept, matching
    ``highs_fastload`` and the shipped GurobiDirect interface).

    The solution vectors are captured *eagerly* at postsolve (``col_value``,
    ``col_rc``, ``row_dual``) -- ``None`` where Gurobi does not provide them (no
    solution, or duals / reduced costs on a MIP).
    """

    def __init__(
        self,
        pyomo_model,
        columns,
        con_of,
        col_value,
        col_rc,
        row_dual,
        column_scatter=None,
    ):
        super().__init__()
        self._pyomo_model = pyomo_model  # for load_import_suffixes (dual / rc)
        self._columns = columns  # list[VarData|None], column-ordered
        self._column_scatter = column_scatter or ()
        self._con_of = con_of  # list[ConstraintData], one per Gurobi row
        self._col_value = col_value
        self._col_rc = col_rc
        self._row_dual = row_dual
        self._init_columnar_maps()

    def get_number_of_solutions(self) -> int:
        return 1 if self._col_value is not None else 0

    def _require_primal(self):
        if self._col_value is None:
            raise NoSolutionError()

    def load_vars(self, vars_to_load: Sequence[VarData] | None = None) -> None:
        self._require_primal()
        col_value = self._col_value
        if vars_to_load is None:
            self._load_all(col_value)
        else:
            for v in vars_to_load:
                j = self._col_for_var(v)
                if j is not None:
                    v.set_value(col_value[j], skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        self._require_primal()
        col_value = self._col_value
        if vars_to_load is None:
            return self._map_all(col_value)
        return self._map_selected(col_value, vars_to_load)

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        if self._col_rc is None:
            raise NoReducedCostsError()
        col_rc = self._col_rc
        if vars_to_load is None:
            return self._map_all(col_rc)
        return self._map_selected(col_rc, vars_to_load)

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        if self._row_dual is None:
            raise NoDualsError()
        row_dual = self._row_dual
        duals = {}
        want = None if cons_to_load is None else set(map(id, cons_to_load))
        for i, con in enumerate(self._con_of):
            if want is not None and id(con) not in want:
                continue
            d = row_dual[i]
            prev = duals.get(con)
            if prev is None or abs(d) > abs(prev):
                duals[con] = d
        return duals


# --------------------------------------------------------------------------- #
# The solver interface
# --------------------------------------------------------------------------- #
class FastLoadGurobi(SolverBase):
    """Direct (standard-form) Gurobi interface with a bulk matrix-API hand-off.

    Compiles a classic linear / convex-QP model to standard-form arrays once and
    hands the whole matrix to Gurobi through its native matrix API
    (``addMVar`` + ``addMConstr`` + ``setMObjective``) in a handful of calls,
    instead of the per-constraint ``addConstr`` load the persistent interface
    uses.  For load-bound models this reaches "the solver has the model" several
    times faster end-to-end, with no change to the user's model code.
    """

    CONFIG = BranchAndBoundConfig()
    name = 'gurobi_fastload'

    _available = None

    def available(self):
        if FastLoadGurobi._available is None:
            import importlib.util

            if importlib.util.find_spec('gurobipy') is None:
                FastLoadGurobi._available = Availability.NotFound
            else:
                # gurobipy imports fine without a usable license; a real solve
                # needs one.  Probe cheaply so ``available()`` reflects reality.
                try:
                    import gurobipy as gp

                    env = gp.Env(empty=True)
                    env.setParam('OutputFlag', 0)
                    env.start()
                    env.dispose()
                    FastLoadGurobi._available = Availability.FullLicense
                except Exception:
                    FastLoadGurobi._available = Availability.BadLicense
        return FastLoadGurobi._available

    def version(self):
        import gurobipy as gp

        return (gp.GRB.VERSION_MAJOR, gp.GRB.VERSION_MINOR, gp.GRB.VERSION_TECHNICAL)

    def solve(self, model, **kwds) -> Results:
        import gurobipy as gp

        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        tick = time.perf_counter()
        config = self.config(value=kwds, preserve_implicit=True)
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        StaleFlagManager.mark_all_as_stale()
        ostreams = [io.StringIO()] + config.tee

        grb_model = None
        try:
            try:
                with capture_output(TeeStream(*ostreams), capture_fd=True):
                    timer.start('compile')
                    compiled = compile_fastload_arrays(model, solver_name=self.name)
                    self._reject_out_of_scope(compiled)
                    timer.stop('compile')

                    timer.start('load')
                    grb_model, mvar, con_of = self._build_gurobi_model(compiled, config)
                    timer.stop('load')

                    timer.start('optimize')
                    grb_model.optimize()
                    timer.stop('optimize')

                results = self._postsolve(
                    grb_model, mvar, con_of, compiled, model, config
                )
            except gp.GurobiError as err:
                if err.errno == _GRB_ERROR_Q_NOT_PSD:
                    raise IncompatibleModelError(
                        f"The '{self.name}' fast solver hand-off could not solve the "
                        "quadratic objective: Gurobi reports the objective Q is not "
                        f"PSD ({err}).  This fast path supports convex QP only -- for "
                        "a minimize objective the Hessian must be positive "
                        "semidefinite (negative semidefinite for maximize)."
                    ) from err
                raise
        except InfeasibleConstraintException as err:
            results = get_infeasible_results(
                model=model,
                solver=self,
                config=config,
                err_msg='The problem was proven infeasible during compilation:\n'
                f'\t{err}',
            )
        finally:
            if grb_model is not None:
                grb_model.dispose()

        results.solver_log = ostreams[0].getvalue()
        tock = time.perf_counter()
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = tock - tick
        results.timing_info.timer = timer
        return results

    # ----------------------------------------------------------------------- #
    # Scope guard + model build
    # ----------------------------------------------------------------------- #
    def _reject_out_of_scope(self, compiled: FastLoadCompiled):
        """Reject MIQP loudly (integer variable + quadratic objective).

        Gurobi can solve *convex* MIQP, but ``NonConvex=0`` (this backend's
        convexity gate) does not enforce convexity for integer models -- a
        non-convex MIQP would still solve silently -- so a correct MIQP path
        needs an explicit convexity check that is out of scope here.
        """
        if compiled.is_quadratic and compiled.integrality.any():
            raise IncompatibleModelError(
                f"The '{self.name}' fast solver hand-off received a quadratic "
                "objective together with integer/binary variables (MIQP).  Gurobi "
                "can solve convex MIQP, but this fast path does not yet enforce the "
                "objective-convexity check that an integer model needs; use the "
                "classic Gurobi persistent interface for MIQP."
            )

    def _build_gurobi_model(self, compiled: FastLoadCompiled, config):
        """Build a Gurobi model from ``compiled`` via the bulk matrix API.

        Returns ``(grb_model, mvar, con_of)`` where ``mvar`` is the column
        :class:`gurobipy.MVar` and ``con_of`` is the per-Gurobi-row constraint
        list for the dual map-back.
        """
        import gurobipy as gp
        from gurobipy import GRB

        # Use the shared default environment (gurobipy manages its lifetime) so no
        # per-solve Env is leaked; OutputFlag on the model drives whether the
        # solver log reaches the captured streams.
        grb = gp.Model()
        grb.setParam('OutputFlag', 1 if config.tee else 0)
        # Convex QP only: make Gurobi reject a non-PSD objective Hessian rather
        # than silently solve the non-convex problem (a correctness guard).
        grb.setParam('NonConvex', 0)
        if config.threads is not None:
            grb.setParam('Threads', int(config.threads))
        if config.time_limit is not None:
            grb.setParam('TimeLimit', float(config.time_limit))
        if config.rel_gap is not None:
            grb.setParam('MIPGap', float(config.rel_gap))
        if config.abs_gap is not None:
            grb.setParam('MIPGapAbs', float(config.abs_gap))
        for key, opt in config.solver_options.items():
            grb.setParam(key, opt)

        n_col = compiled.n_col
        ginf = GRB.INFINITY
        col_lower = np.where(np.isneginf(compiled.col_lower), -ginf, compiled.col_lower)
        col_upper = np.where(np.isposinf(compiled.col_upper), ginf, compiled.col_upper)
        col_lower = col_lower.astype(np.float64)
        col_upper = col_upper.astype(np.float64)
        if compiled.integrality.any():
            vtype = np.where(compiled.integrality, GRB.INTEGER, GRB.CONTINUOUS)
        else:
            vtype = GRB.CONTINUOUS
        x = grb.addMVar(n_col, lb=col_lower, ub=col_upper, vtype=vtype)

        # --- constraints (one-sided rows; ranges split) --------------------- #
        sel, sense, rhs, con_of = _gurobi_rows(compiled)
        if con_of:
            A = compiled.A.tocsr()
            if sel is not None:
                A = A[np.asarray(sel, dtype=np.int64)]
            grb.addMConstr(A, x, np.asarray(sense), np.asarray(rhs, dtype=np.float64))

        # --- objective ------------------------------------------------------ #
        if compiled.has_objective:
            c = compiled.c.astype(np.float64)
            offset = float(compiled.c_offset)
            gsense = (
                GRB.MAXIMIZE
                if compiled.sense == ObjectiveSense.maximize
                else GRB.MINIMIZE
            )
            if compiled.is_quadratic:
                Q = _hessian_to_gurobi_Q(compiled.hessian)
                grb.setMObjective(Q, c, offset, x, x, x, gsense)
            else:
                grb.setMObjective(None, c, offset, None, None, x, gsense)

        grb.update()
        return grb, x, con_of

    # ----------------------------------------------------------------------- #
    # Postsolve: capture solution vectors + build Results
    # ----------------------------------------------------------------------- #
    def _postsolve(self, grb, mvar, con_of, compiled, model, config):
        import gurobipy as gp

        status = grb.Status
        has_feasible_solution = grb.SolCount > 0

        # Capture the solution vectors eagerly (the model is disposed after solve).
        col_value = None
        col_rc = None
        row_dual = None
        if has_feasible_solution:
            col_value = np.asarray(mvar.X, dtype=np.float64)
            # Reduced costs / duals exist for a continuous (LP/QP) model only.
            if not compiled.integrality.any():
                try:
                    col_rc = np.asarray(mvar.RC, dtype=np.float64)
                except (gp.GurobiError, AttributeError):
                    col_rc = None
                if con_of:
                    try:
                        row_dual = np.asarray(grb.getAttr('Pi'), dtype=np.float64)
                    except (gp.GurobiError, AttributeError):
                        row_dual = None

        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_config = config
        try:
            results.timing_info.gurobi_time = grb.Runtime
        except (gp.GurobiError, AttributeError):
            pass
        results.solution_loader = FastLoadGurobiSolutionLoader(
            model,
            compiled.columns,
            con_of,
            col_value,
            col_rc,
            row_dual,
            compiled.column_scatter,
        )

        if has_feasible_solution:
            if status == gp.GRB.OPTIMAL:
                results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        results.termination_condition = self._get_tc_map().get(
            status, TerminationCondition.unknown
        )

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        results.incumbent_objective = None
        results.objective_bound = None
        if compiled.has_objective:
            if has_feasible_solution:
                try:
                    obj_val = grb.ObjVal
                    if math.isfinite(obj_val):
                        results.incumbent_objective = obj_val
                except (gp.GurobiError, AttributeError):
                    pass
            try:
                results.objective_bound = grb.ObjBound
            except (gp.GurobiError, AttributeError):
                results.objective_bound = None

        if config.load_solutions:
            if has_feasible_solution:
                results.solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()

        return results

    # Gurobi status -> termination condition, mirroring the shipped GurobiDirect
    # interface (pyomo.contrib.solver.solvers.gurobi).  Built lazily so importing
    # this module does not import gurobipy.
    _TC_MAP = None

    @classmethod
    def _get_tc_map(cls):
        if cls._TC_MAP is None:
            from gurobipy import GRB

            TC = TerminationCondition
            cls._TC_MAP = {
                GRB.LOADED: TC.unknown,
                GRB.OPTIMAL: TC.convergenceCriteriaSatisfied,
                GRB.INFEASIBLE: TC.provenInfeasible,
                GRB.INF_OR_UNBD: TC.infeasibleOrUnbounded,
                GRB.UNBOUNDED: TC.unbounded,
                GRB.CUTOFF: TC.objectiveLimit,
                GRB.ITERATION_LIMIT: TC.iterationLimit,
                GRB.NODE_LIMIT: TC.iterationLimit,
                GRB.TIME_LIMIT: TC.maxTimeLimit,
                GRB.SOLUTION_LIMIT: TC.unknown,
                GRB.INTERRUPTED: TC.interrupted,
                GRB.NUMERIC: TC.unknown,
                GRB.SUBOPTIMAL: TC.unknown,
                GRB.USER_OBJ_LIMIT: TC.objectiveLimit,
            }
        return cls._TC_MAP


# --------------------------------------------------------------------------- #
# Registration (both the v2 SolverFactory and the legacy SolverFactory)
# --------------------------------------------------------------------------- #
def _register():
    if 'gurobi_fastload' not in SolverFactory:
        SolverFactory.register(
            name='gurobi_fastload',
            legacy_name='gurobi_fastload',
            doc='Direct Gurobi interface: standard-form compile + matrix-API '
            '(addMVar/addMConstr/setMObjective) bulk hand-off for classic '
            'linear / convex-QP models',
        )(FastLoadGurobi)


_register()
