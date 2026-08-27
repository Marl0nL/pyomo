# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Transparent fast solver hand-off for classic linear models (the load prize).

Phase-0 of the vectorized-construction project measured solver *load* as the
single largest stage of time-to-solution for classic Pyomo models -- 48-80% of
the coherent ``construct + load`` route at scale -- because the in-memory solver
interfaces extract the model one row at a time, re-running a per-row
``generate_standard_repn`` and one solver API call per constraint (#3888).

But a classic linear model already produces exactly the arrays a solver wants
*mid-pipeline*: :class:`~pyomo.repn.plugins.standard_form.LinearStandardFormCompiler`
walks every constraint once through the fast ``pyomo.repn.linear`` visitor and
emits a single scipy CSR/CSC matrix.  This module routes a classic model's solve
through that compile and then hands the whole matrix to HiGHS in one
``Highs.passModel`` call -- the same bulk hand-off Phase 1 built for vectorized
models (:func:`pyomo.contrib.vector.highs.load_highs`), now reused for
*unmodified* classic models.  No user model change is required: the fast path is
a drop-in solver registered under a normal name.

    from pyomo.contrib.solver.common.factory import SolverFactory
    results = SolverFactory('highs_fastload').solve(model)
    # or, via the legacy factory:
    pyomo.SolverFactory('highs_fastload').solve(model)

Scope: **linear** continuous / MIP models only.  A model with nonlinear terms
(or components the standard-form compiler cannot process) is rejected loudly with
a message pointing at the classic solver route -- it never silently produces a
wrong answer.  Vectorized (``pyomo.contrib.vector``) components in a mixed model
are handled through the standard-form compiler's scalarization contract (scoping
doc Sec 6.5): they materialize to classic data objects and load correctly, just
without the columnar fast path (use :func:`~pyomo.contrib.vector.highs.solve_highs`
for a pure-vector fast solve).
"""

from __future__ import annotations

import datetime
import io
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

# highspy is imported lazily inside the methods that need it (the module is
# imported at ``pyomo.contrib.vector`` import time to register the solver, and we
# do not want to force-import highspy then).
_inf = float('inf')


# --------------------------------------------------------------------------- #
# Standard-form compile -> range-row arrays for a direct HiGHS hand-off
# --------------------------------------------------------------------------- #
class FastLoadCompiled:
    """Range-row arrays + Pyomo map-back metadata for a compiled linear model.

    Produced by :func:`compile_to_highs_arrays` and consumed by
    :func:`build_highs_lp`.  The row form is HiGHS-native range rows
    (``row_lower <= A x <= row_upper``); fixed variables have already been
    substituted into the row bounds / objective offset by the standard-form
    compiler, so ``columns`` are exactly the free columns of ``A``.

    Attributes
    ----------
    A : scipy.sparse.csc_array   (n_row x n_col)
    row_lower, row_upper : np.ndarray (float64; +/- inf on an open side)
    col_lower, col_upper : np.ndarray (float64; +/- inf when unbounded)
    integrality : np.ndarray (bool; True where the column is integer/binary)
    c : np.ndarray (float64, length n_col)   -- objective coefficients
    c_offset : float
    sense : ObjectiveSense
    has_objective : bool
    columns : list[VarData]                  -- one per column of A (map-back)
    rows : list[(ConstraintData, int)]       -- one per row; int is the
        standard-form row multiplier (0 == equality, 1 == upper, -1 == lower)
    """

    __slots__ = (
        'A',
        'row_lower',
        'row_upper',
        'col_lower',
        'col_upper',
        'integrality',
        'c',
        'c_offset',
        'sense',
        'has_objective',
        'columns',
        'rows',
    )

    def __init__(
        self,
        A,
        row_lower,
        row_upper,
        col_lower,
        col_upper,
        integrality,
        c,
        c_offset,
        sense,
        has_objective,
        columns,
        rows,
    ):
        self.A = A
        self.row_lower = row_lower
        self.row_upper = row_upper
        self.col_lower = col_lower
        self.col_upper = col_upper
        self.integrality = integrality
        self.c = c
        self.c_offset = c_offset
        self.sense = sense
        self.has_objective = has_objective
        self.columns = columns
        self.rows = rows

    @property
    def n_col(self):
        return self.A.shape[1]

    @property
    def n_row(self):
        return self.A.shape[0]

    @property
    def nnz(self):
        return int(self.A.nnz)


def compile_to_highs_arrays(model):
    """Compile a classic linear ``model`` to HiGHS-ready range-row arrays.

    Reuses the stock :class:`LinearStandardFormCompiler` (mixed form: equalities
    kept as equalities, one-sided rows kept one-sided) -- the fast, single-pass
    matrix compile that the Phase-0 harness measured -- and converts its
    ``(constraint, multiplier)`` rows into HiGHS range rows.

    Raises :class:`IncompatibleModelError` if the model is not linear or contains
    components the standard-form compiler cannot process; the message points at
    the classic solver route so the user is never silently given a wrong answer.

    If the model was built with template-vectorized construction
    (:func:`pyomo.contrib.vector.template_vectorize.vectorized_construction`),
    every constraint family that templatizes is compiled by *vectorized*
    extraction (NumPy over the whole index set) and the rest by the classic
    per-row repn, over one shared column space -- construction and load both stay
    array-shaped end-to-end (Phase-3).
    """
    from pyomo.contrib.vector.template_vectorize import (
        model_has_templates,
        compile_templated_to_highs_arrays,
    )

    if model_has_templates(model):
        return compile_templated_to_highs_arrays(model)

    from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
    from pyomo.common.errors import InvalidConstraintError, InvalidExpressionError

    try:
        # set_sense=None keeps the model's own objective sense (we tell HiGHS the
        # sense directly), so the objective HiGHS reports is the true objective.
        info = LinearStandardFormCompiler().write(
            model, mixed_form=True, set_sense=None
        )
    except (InvalidConstraintError, InvalidExpressionError) as e:
        raise IncompatibleModelError(
            f"The '{FastLoadHighs.name}' fast solver hand-off only supports "
            "linear models; this model has nonlinear terms that the standard-form "
            f"compiler cannot process ({e}).  Use a classic nonlinear solver "
            "route (e.g. SolverFactory('ipopt'))."
        ) from e
    except ValueError as e:
        # categorize_valid_components raises ValueError listing components the
        # standard-form compiler does not understand (e.g. Piecewise, SOS).
        raise IncompatibleModelError(
            f"The '{FastLoadHighs.name}' fast solver hand-off cannot compile this "
            f"model to linear standard form: {e}  Use a classic solver route."
        ) from e

    if len(info.objectives) > 1:
        raise IncompatibleModelError(
            f"The '{FastLoadHighs.name}' fast solver hand-off supports at most one "
            f"objective (received {len(info.objectives)})."
        )

    columns = info.columns
    n_col = len(columns)
    A = info.A.tocsc()

    # --- column bounds + integrality (a single pass over the free columns) --- #
    col_lower = np.empty(n_col, dtype=np.float64)
    col_upper = np.empty(n_col, dtype=np.float64)
    integrality = np.zeros(n_col, dtype=bool)
    for j, v in enumerate(columns):
        lb, ub = v.bounds
        col_lower[j] = _ninf_none(lb)
        col_upper[j] = _pinf_none(ub)
        if not v.is_continuous():
            integrality[j] = True

    # --- range rows from the mixed-form (constraint, multiplier) rows -------- #
    rows = info.rows
    n_row = len(rows)
    rhs = np.asarray(info.rhs, dtype=np.float64)
    bt = np.fromiter((r[1] for r in rows), dtype=np.int8, count=n_row)
    # multiplier 0 == equality (lower == upper == rhs)
    #            1 == upper bound (A x <= rhs)      -> lower open
    #           -1 == lower bound (A x >= rhs)      -> upper open
    row_lower = np.where(bt == 1, -_inf, rhs)
    row_upper = np.where(bt == -1, _inf, rhs)

    # --- objective ---------------------------------------------------------- #
    has_objective = len(info.objectives) == 1
    if has_objective and info.c.shape[0]:
        c = np.asarray(info.c.todense()).reshape(-1)[:n_col].astype(np.float64)
        sense = ObjectiveSense(info.objectives[0].sense)
        c_offset = float(info.c_offset[0]) if len(info.c_offset) else 0.0
    else:
        c = np.zeros(n_col, dtype=np.float64)
        sense = ObjectiveSense.minimize
        c_offset = 0.0

    return FastLoadCompiled(
        A,
        row_lower,
        row_upper,
        col_lower,
        col_upper,
        integrality,
        c,
        c_offset,
        sense,
        has_objective,
        columns,
        list(rows),
    )


def _ninf_none(v):
    return -_inf if v is None else float(v)


def _pinf_none(v):
    return _inf if v is None else float(v)


def build_highs_lp(compiled: FastLoadCompiled):
    """Build a ``highspy.HighsLp`` from :class:`FastLoadCompiled` arrays."""
    import highspy

    hinf = highspy.kHighsInf
    A = compiled.A

    col_lower = np.where(np.isneginf(compiled.col_lower), -hinf, compiled.col_lower)
    col_upper = np.where(np.isposinf(compiled.col_upper), hinf, compiled.col_upper)
    row_lower = np.where(np.isneginf(compiled.row_lower), -hinf, compiled.row_lower)
    row_upper = np.where(np.isposinf(compiled.row_upper), hinf, compiled.row_upper)

    lp = highspy.HighsLp()
    lp.num_col_ = compiled.n_col
    lp.num_row_ = compiled.n_row
    lp.col_cost_ = compiled.c.astype(np.float64)
    lp.col_lower_ = col_lower.astype(np.float64)
    lp.col_upper_ = col_upper.astype(np.float64)
    lp.row_lower_ = row_lower.astype(np.float64)
    lp.row_upper_ = row_upper.astype(np.float64)
    lp.offset_ = float(compiled.c_offset)
    lp.sense_ = (
        highspy.ObjSense.kMaximize
        if compiled.sense == ObjectiveSense.maximize
        else highspy.ObjSense.kMinimize
    )
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.a_matrix_.num_col_ = compiled.n_col
    lp.a_matrix_.num_row_ = compiled.n_row
    if compiled.integrality.any():
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if flag else highspy.HighsVarType.kContinuous
            for flag in compiled.integrality
        ]
    return lp


# --------------------------------------------------------------------------- #
# Solution loader: map the HiGHS solution vectors back onto the Pyomo model
# --------------------------------------------------------------------------- #
class FastLoadHighsSolutionLoader(SolutionLoader):
    """Map a solved HiGHS model's solution back onto the original Pyomo objects.

    Columns and rows are kept in the exact order the standard-form compiler
    produced them, so primal values / reduced costs / duals index straight into
    the HiGHS solution vectors -- no per-object solver map is needed.
    """

    def __init__(self, solver_model, pyomo_model, columns, rows):
        super().__init__()
        self._solver_model = solver_model
        self._pyomo_model = pyomo_model
        self._columns = columns  # list[VarData], column-ordered
        self._rows = rows  # list[(ConstraintData, multiplier)], row-ordered
        self._sol = solver_model.getSolution()
        self._col_map = None  # id(var) -> column index (built lazily)

    def get_number_of_solutions(self) -> int:
        return 1 if self._sol.value_valid else 0

    def _require_primal(self):
        if not self._sol.value_valid:
            raise NoSolutionError()

    def _build_col_map(self):
        if self._col_map is None:
            self._col_map = {id(v): j for j, v in enumerate(self._columns)}
        return self._col_map

    def load_vars(self, vars_to_load: Sequence[VarData] | None = None) -> None:
        self._require_primal()
        col_value = self._sol.col_value
        if vars_to_load is None:
            for v, val in zip(self._columns, col_value):
                v.set_value(val, skip_validation=True)
        else:
            col_map = self._build_col_map()
            for v in vars_to_load:
                j = col_map.get(id(v))
                if j is not None:
                    v.set_value(col_value[j], skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        self._require_primal()
        col_value = self._sol.col_value
        if vars_to_load is None:
            return ComponentMap((v, col_value[j]) for j, v in enumerate(self._columns))
        col_map = self._build_col_map()
        res = ComponentMap()
        for v in vars_to_load:
            j = col_map.get(id(v))
            if j is not None:
                res[v] = col_value[j]
        return res

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        if not self._sol.dual_valid:
            raise NoReducedCostsError()
        col_dual = self._sol.col_dual
        if vars_to_load is None:
            return ComponentMap((v, col_dual[j]) for j, v in enumerate(self._columns))
        col_map = self._build_col_map()
        res = ComponentMap()
        for v in vars_to_load:
            j = col_map.get(id(v))
            if j is not None:
                res[v] = col_dual[j]
        return res

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        if not self._sol.dual_valid:
            raise NoDualsError()
        row_dual = self._sol.row_dual
        # A two-sided (range) constraint is split into two rows sharing one
        # ConstraintData; keep the larger-magnitude dual (matches GurobiDirect).
        duals = {}
        want = None if cons_to_load is None else set(map(id, cons_to_load))
        for i, (con, _mult) in enumerate(self._rows):
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
class FastLoadHighs(SolverBase):
    """Direct (standard-form) HiGHS interface with a bulk ``passModel`` hand-off.

    Compiles a classic linear model to standard-form arrays once and hands the
    whole matrix to HiGHS in a single ``passModel`` call, instead of the per-row
    ``set_instance`` load the persistent interface uses.  For load-bound models
    this reaches "the solver has the model" several times faster end-to-end,
    with no change to the user's model code.
    """

    CONFIG = BranchAndBoundConfig()
    name = 'highs_fastload'

    _available = None

    def available(self):
        if FastLoadHighs._available is None:
            import importlib.util

            FastLoadHighs._available = (
                Availability.FullLicense
                if importlib.util.find_spec('highspy') is not None
                else Availability.NotFound
            )
        return FastLoadHighs._available

    def version(self):
        import highspy

        try:
            return (
                highspy.HIGHS_VERSION_MAJOR,
                highspy.HIGHS_VERSION_MINOR,
                highspy.HIGHS_VERSION_PATCH,
            )
        except AttributeError:
            tmp = highspy.Highs()
            return (tmp.versionMajor(), tmp.versionMinor(), tmp.versionPatch())

    def solve(self, model, **kwds) -> Results:
        import highspy

        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        tick = time.perf_counter()
        config = self.config(value=kwds, preserve_implicit=True)
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        StaleFlagManager.mark_all_as_stale()
        ostreams = [io.StringIO()] + config.tee

        try:
            with capture_output(TeeStream(*ostreams), capture_fd=True):
                timer.start('compile')
                compiled = compile_to_highs_arrays(model)
                lp = build_highs_lp(compiled)
                timer.stop('compile')

                highs = highspy.Highs()
                highs.setOptionValue('log_to_console', False)
                if config.threads is not None:
                    highs.setOptionValue('threads', config.threads)
                if config.time_limit is not None:
                    highs.setOptionValue('time_limit', float(config.time_limit))
                if config.rel_gap is not None:
                    highs.setOptionValue('mip_rel_gap', float(config.rel_gap))
                if config.abs_gap is not None:
                    highs.setOptionValue('mip_abs_gap', float(config.abs_gap))
                for key, opt in config.solver_options.items():
                    highs.setOptionValue(key, opt)

                timer.start('load')
                highs.passModel(lp)
                timer.stop('load')

                timer.start('optimize')
                highs.run()
                timer.stop('optimize')

            results = self._postsolve(highs, compiled, model, config)
        except InfeasibleConstraintException as err:
            results = get_infeasible_results(
                model=model,
                solver=self,
                config=config,
                err_msg='The problem was proven infeasible during compilation:\n'
                f'\t{err}',
            )

        results.solver_log = ostreams[0].getvalue()
        tock = time.perf_counter()
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = tock - tick
        results.timing_info.timer = timer
        return results

    def _postsolve(self, highs, compiled, model, config):
        import highspy

        status = highs.getModelStatus()
        info = highs.getInfo()

        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_config = config
        results.timing_info.highs_time = highs.getRunTime()
        results.solution_loader = FastLoadHighsSolutionLoader(
            highs, model, compiled.columns, compiled.rows
        )

        has_feasible_solution = info.primal_solution_status == 2
        if status == highspy.HighsModelStatus.kOptimal:
            results.solution_status = SolutionStatus.optimal
        elif has_feasible_solution:
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
                results.incumbent_objective = info.objective_function_value
            if info.mip_node_count == -1:
                # LP: the incumbent is the proven bound at optimality.
                if (
                    has_feasible_solution
                    and results.termination_condition
                    == TerminationCondition.convergenceCriteriaSatisfied
                ):
                    results.objective_bound = info.objective_function_value
            else:
                results.objective_bound = info.mip_dual_bound

        if config.load_solutions:
            if has_feasible_solution:
                results.solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()

        return results

    # Model-status -> termination-condition, mirroring the persistent HiGHS
    # interface (pyomo.contrib.solver.solvers.highs).  Built lazily so importing
    # this module does not import highspy.
    _TC_MAP = None

    @classmethod
    def _get_tc_map(cls):
        if cls._TC_MAP is None:
            import highspy

            S = highspy.HighsModelStatus
            TC = TerminationCondition
            cls._TC_MAP = {
                S.kNotset: TC.unknown,
                S.kLoadError: TC.error,
                S.kModelError: TC.error,
                S.kPresolveError: TC.error,
                S.kSolveError: TC.error,
                S.kPostsolveError: TC.error,
                S.kModelEmpty: TC.emptyModel,
                S.kOptimal: TC.convergenceCriteriaSatisfied,
                S.kInfeasible: TC.provenInfeasible,
                S.kUnboundedOrInfeasible: TC.infeasibleOrUnbounded,
                S.kUnbounded: TC.unbounded,
                S.kObjectiveBound: TC.objectiveLimit,
                S.kObjectiveTarget: TC.objectiveLimit,
                S.kTimeLimit: TC.maxTimeLimit,
                S.kIterationLimit: TC.iterationLimit,
                S.kUnknown: TC.unknown,
            }
        return cls._TC_MAP


# --------------------------------------------------------------------------- #
# Registration (both the v2 SolverFactory and the legacy SolverFactory)
# --------------------------------------------------------------------------- #
def _register():
    if 'highs_fastload' not in SolverFactory:
        SolverFactory.register(
            name='highs_fastload',
            legacy_name='highs_fastload',
            doc='Direct HiGHS interface: standard-form compile + passModel '
            'bulk hand-off for classic linear models',
        )(FastLoadHighs)


_register()
