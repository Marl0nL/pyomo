# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Array-native persistent warm re-solve for classic linear models (the warm prize).

Phase-2 (:mod:`pyomo.contrib.vector.fastload`) cut the *cold* build+solve of a
classic linear model by replacing the persistent interface's per-row
``set_instance`` load with one bulk :meth:`Highs.passModel`.  It deliberately
left the *warm rolling* path -- ``construct once, then re-solve thousands of
times with slightly changed data`` (MPC / rolling-horizon / receding-horizon
control) -- untouched.

For that warm path the persistent APPSI HiGHS interface re-pushes every changed
coefficient to the solver **one scalar Python call at a time**, and re-evaluates
every changed coefficient with a **per-coefficient expression walk**
(``value(expr)`` then ``changeColCost`` / ``changeCoeff`` / ``changeRowBounds``,
once per coefficient, no dirty check -- ``pyomo/contrib/appsi/solvers/highs.py``).
On a model where a roll touches every objective price and every RHS forecast,
that per-coefficient evaluate-and-push loop is the single largest cost of the
warm tick, and profiling splits it *roughly evenly* between the value walks and
the scalar solver calls -- so batching only the solver side is not enough.

:class:`FastStepHighs` attacks **both** halves.  It compiles the model to
standard-form arrays **once** (reusing
:func:`~pyomo.contrib.vector.fastload.compile_to_highs_arrays`), hands the whole
matrix to a **retained, live** ``highspy.Highs`` via ``passModel``, and captures
each mutable objective coefficient / row bound / variable bound as an **affine
template over a parameter vector** -- ``values = M @ P + shift`` with ``M`` a
constant sparse map and ``P`` the current values of the model's mutable
``Param`` data.  A warm solve then:

1. reads the (small) parameter vector ``P`` once -- *not* one expression walk per
   coefficient;
2. expands every changed coefficient / bound with a single vectorized sparse
   ``M @ P`` (the evaluation side, vectorized); and
3. applies each group with one HiGHS **batch** call -- ``changeColsCost`` /
   ``changeRowsBounds`` / ``changeColsBounds`` (the solver side, batched),

preserving the warm simplex basis so the re-solve is a handful of iterations.

    stepper = FastStepHighs()
    stepper.set_instance(model)          # compile once + passModel + build templates
    res = stepper.solve()                # first solve

    for roll in horizon:
        # mutate the model's mutable Params / Var bounds in place, as usual ...
        res = stepper.solve()            # read P, M @ P, batch push, warm re-solve

The interface mirrors the persistent APPSI HiGHS solver (``set_instance`` then
repeated ``solve``) so an existing classic model with mutable ``Param`` data
adopts it without any model rewrite: mutate the Params, call ``solve``.

Array (mapping-free) update path
--------------------------------
A caller that already holds the roll's data as arrays can drive the solve
*without* touching the Pyomo model at all: pass the parameter vector directly as
``solve(param_values=P)`` (``P`` ordered by :attr:`FastStepHighs.parameters`),
optionally with a ``dirty`` boolean mask marking which parameters changed.
:class:`FastStepHighs` owns the LP row/column mapping and the coefficient
templates internally, so this mapping never has to live on the caller side.

Change-detection contract
-------------------------
``set_instance`` walks the objective and every constraint **once** with the same
symbolic repn the APPSI interface uses (``generate_standard_repn(...,
compute_values=False)``), records -- against the compiler's stable column/row
identity -- which objective coefficients, row bounds and variable bounds are
*mutable*, and decomposes each into its affine template over ``P``.  The
templates are self-checked at ``set_instance``: they must reproduce the compiled
standard-form arrays at the current ``P`` *and* at a random perturbation of
``P``, or the model is rejected loudly -- the same fail-loud posture as
``highs_fastload``.  (An entry that is not affine in the parameters, or that
references a fixed variable, is evaluated per-solve with ``value()`` as a
residual; correctness is preserved, only that entry is not vectorized.)

Value-aware static-matrix guard (constraint-matrix coefficients)
----------------------------------------------------------------
Many real rolling models carry *nominally* mutable ``Param`` coefficients on the
constraint matrix -- an interval duration in a state-of-charge recurrence, an
efficiency, a per-step gain -- whose **values do not actually change** between
warm solves (an equal-interval roll leaves the durations fixed).  Rejecting such
a model on the mutability *flag* alone -- the pre-guard behavior -- turned away a
large, warm-solvable class of models.

Instead of trusting the flag, :class:`FastStepHighs` **verifies the values**.  At
``set_instance`` a mutable matrix coefficient is captured into the
:class:`_MatrixGuard` -- the affine template over ``P`` for every affected
``A``-entry, plus the values HiGHS was loaded with (the same template machinery
the objective / RHS / bound groups use).  Each warm solve re-evaluates those
coefficients **vectorized** (one sparse ``M @ P``, no per-coefficient Python
walk) and compares them to that baseline:

* **unchanged** (exact by default, or a configurable tight tolerance) -- the
  matrix HiGHS holds is still correct, so the warm basis is kept; the comparison
  is the guard's only per-roll cost;
* **genuinely changed** -- never a stale-matrix solve.  The default
  ``on_matrix_change='error'`` fails loud, naming the offending Param /
  coefficient(s); the opt-in ``on_matrix_change='reload'`` instead rebuilds the
  whole standard-form matrix and reloads it (a fresh ``passModel``, basis reset)
  for that solve, then continues.

The guard's coefficient mapping (which mutable Params feed which ``A``-entries,
with what affine relation) is a standalone, reusable component: a later batch
matrix-update path can reuse it to *apply* a genuine change; this guard only
*detects* change.

Scope (fail loud, never a stale-matrix solve)
---------------------------------------------
* **Linear** continuous / MIP models only (inherited from the standard-form
  compile).  Nonlinear / unsupported structure is rejected at ``set_instance``.
* Supported warm updates: **objective coefficients, objective offset, constraint
  (row) bounds / RHS, and variable bounds**, all driven by mutable ``Param``
  values (or fixed-variable values).  This is exactly the rolling-horizon roll:
  prices move the objective, forecasts/limits move the RHS and bounds.
* **Constraint-matrix coefficients are value-guarded, not assumed static.**  A
  mutable matrix coefficient is accepted (see the value-guard section above); a
  coefficient that stays put warm-solves on the kept basis, one that genuinely
  changes fails loud or (opt-in) triggers a reload -- never a stale solve.  A
  coefficient that is genuinely *nonlinear in the parameters* (e.g. a product of
  two Params) cannot be templated affinely and is still rejected at
  ``set_instance`` by the template self-check.
* A **structure change** between solves (a constraint or variable added/removed,
  the objective swapped) is caught by a cheap fingerprint check and rejected --
  the caller must build a fresh :class:`FastStepHighs`.
"""

from __future__ import annotations

import datetime
import io
import time

from pyomo.common.dependencies import numpy as np, scipy
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer

from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.var import Var
from pyomo.core.expr import identify_mutable_parameters, identify_variables
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.numvalue import is_constant, value
from pyomo.repn.standard_repn import generate_standard_repn

from pyomo.contrib.solver.common.base import Availability
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
    get_infeasible_results,
)
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    IncompatibleModelError,
)

from pyomo.contrib.vector.fastload import (
    FastLoadCompiled,
    FastLoadHighsSolutionLoader,
    build_highs_lp,
    compile_to_highs_arrays,
)

_inf = float('inf')

# Tolerance for the set_instance self-check (templates must reproduce the
# compiled standard-form arrays).  Absolute + relative.
_SELFCHECK_ATOL = 1e-8
_SELFCHECK_RTOL = 1e-7


# --------------------------------------------------------------------------- #
# Affine template: values = M @ P + base, with a residual value() fallback
# --------------------------------------------------------------------------- #
class _AffineArray:
    """A length-``N`` value array expressed affinely over the parameter vector.

    ``compute(P)`` returns ``M @ P + base`` with the residual slots overwritten
    by direct ``value()`` evaluation.  ``M`` is a constant ``scipy.sparse``
    matrix (``N x n_params``); ``base`` folds every constant term (a fixed bound,
    an affine ``shift``, and +/- HiGHS-infinity for an open bound side).  Residual
    slots -- entries that are not affine in the parameters, or that reference a
    fixed variable -- carry a zero ``M`` row and are evaluated from their stored
    Pyomo expression on each ``compute``.
    """

    __slots__ = ('M', 'base', 'residual', 'n')

    def __init__(self, M, base, residual):
        self.M = M.tocsr()
        self.base = base
        self.residual = residual  # list[(pos, expr)]
        self.n = base.shape[0]

    def compute(self, P, hinf):
        out = self.M.dot(P) + self.base
        for pos, expr in self.residual:
            out[pos] = value(expr)
        np.clip(out, -hinf, hinf, out=out)
        return out

    def affected_rows(self, dirty_cols):
        """Row positions whose value can change when ``dirty_cols`` parameters do.

        Residual rows are always considered affected (they may read any Param).
        """
        if len(dirty_cols):
            sub = self.M[:, dirty_cols]
            rows = np.unique(sub.nonzero()[0])
        else:
            rows = np.empty(0, dtype=np.int64)
        if self.residual:
            rows = np.union1d(rows, np.array([p for p, _ in self.residual]))
        return rows

    def compute_rows(self, rows, P, hinf):
        """Compute the values at ``rows`` only (``rows`` an index array)."""
        out = self.M[rows].dot(P) + self.base[rows]
        if self.residual:
            resmap = {p: e for p, e in self.residual}
            for k, r in enumerate(rows):
                e = resmap.get(int(r))
                if e is not None:
                    out[k] = value(e)
        np.clip(out, -hinf, hinf, out=out)
        return out

    @property
    def has_residual(self):
        return bool(self.residual)


def _affine_over_params(e, param_index):
    """Decompose ``e`` into ``(shift, {param_pos: coef})`` if it is affine in the
    mutable parameters and references no (fixed) variable; else return ``None``.

    A non-constant derivative w.r.t. any parameter (e.g. a product of two
    parameters) means ``e`` is not affine -> ``None`` -> the caller keeps the
    entry as a residual.
    """
    # Fast path: a bare mutable parameter (``price[t]``) -- the dominant case --
    # maps to a single unit gather with no derivative work.
    if getattr(e, 'is_parameter_type', _false)():
        return 0.0, {param_index[id(e)]: 1.0}
    # A (fixed) variable in the expression makes the constant term depend on a
    # value that is not in the parameter vector -> evaluate as a residual.
    for _v in identify_variables(e, include_fixed=True):
        return None
    coefs = {}
    shift = float(value(e))
    for p in identify_mutable_parameters(e):
        d = differentiate(e, wrt=p)
        if not is_constant(d):
            return None
        c = float(value(d))
        coefs[param_index[id(p)]] = c
        shift -= c * float(value(p))
    return shift, coefs


def _false():
    return False


def _build_affine_array(slots, param_index, n_params, open_sign):
    """Build an :class:`_AffineArray` from a list of bound/coefficient ``slots``.

    Each slot is ``None`` (an open bound side -> ``open_sign * inf`` in ``base``),
    a plain ``float`` (a fixed value kept in ``base``), or a Pyomo expression
    (templated when affine in the parameters, else recorded as a residual and
    evaluated with ``value()`` every solve).
    """
    n = len(slots)
    rows = []
    cols = []
    data = []
    base = np.zeros(n, dtype=np.float64)
    residual = []
    for i, s in enumerate(slots):
        if s is None:
            base[i] = open_sign * _inf
        elif s.__class__ is float or s.__class__ is int:
            base[i] = float(s)
        else:
            aff = _affine_over_params(s, param_index)
            if aff is None:
                residual.append((i, s))
            else:
                shift, coefs = aff
                base[i] = shift
                for pos, c in coefs.items():
                    rows.append(i)
                    cols.append(pos)
                    data.append(c)
    M = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)), shape=(n, n_params)
    )
    return _AffineArray(M, base, residual)


# --------------------------------------------------------------------------- #
# The constraint-matrix value guard (a reusable coefficient mapping)
# --------------------------------------------------------------------------- #
class _MatrixGuard:
    """The value-aware static-matrix guard: which mutable Params feed which
    ``A``-entries, and the affine relation that produces them.

    A classic linear model can carry *nominally* mutable ``Param`` coefficients
    on its constraint matrix -- an interval duration in a state-of-charge
    recurrence, an efficiency, a per-step gain -- whose **values never actually
    change** between warm rolls (an equal-interval roll leaves the durations
    fixed).  The mutability *flag* is pessimistic; rejecting such a model on the
    flag alone (the pre-guard behavior) turns away a large class of real rolling
    models that are perfectly warm-solvable.

    This component replaces *trust-the-flag* with *verify-the-values*.  It
    records, against the compiler's stable ``A``-entry identity, every mutable
    matrix coefficient as an :class:`_AffineArray` over the parameter vector
    ``P`` (``coef_values = M @ P + shift`` -- the same template family the
    objective / RHS / bound groups use) together with the target ``(row, col)``
    index arrays and the coefficient values HiGHS was loaded with.  On each warm
    solve the affected coefficients are re-evaluated **vectorized** (one sparse
    ``M @ P``, no per-coefficient Python walk) and compared to that baseline:

    * unchanged -> the matrix HiGHS holds is still correct, keep the warm basis
      (the fast path);
    * changed -> never a stale-matrix solve (:class:`FastStepHighs` fails loud or,
      opt-in, reloads).

    The mapping (``rows`` / ``cols`` / ``affine``, keyed to the compiled matrix)
    is a standalone component on purpose: a later batch matrix-update path can
    reuse exactly this (which Params feed which ``A``-entries, with what affine
    relation) to *apply* a genuine change; this guard only **detects** change.

    ``provenance`` carries one ``(constraint_name, variable_name, coef_expr)``
    per entry so a fail-loud message can name the offending coefficient(s).
    """

    __slots__ = ('rows', 'cols', 'affine', 'baseline', 'provenance')

    def __init__(self, rows, cols, affine, baseline, provenance):
        self.rows = rows  # int32[N]: A row index per guarded entry
        self.cols = cols  # int32[N]: A col index per guarded entry
        self.affine = affine  # _AffineArray[N]: coefficient values from P
        self.baseline = baseline  # float64[N]: the values HiGHS was loaded with
        self.provenance = provenance  # list[(con_name, var_name, coef_expr)]

    @property
    def is_empty(self):
        return self.affine is None or self.affine.n == 0

    @property
    def has_residual(self):
        return self.affine is not None and self.affine.has_residual

    def current(self, P, hinf):
        """Re-evaluate every guarded coefficient at ``P`` (vectorized)."""
        return self.affine.compute(P, hinf)

    def changed_mask(self, current, atol, rtol):
        """Boolean mask of entries whose value moved off the loaded baseline.

        ``atol == rtol == 0`` (the default) is an exact comparison: any nonzero
        drift trips the guard.  A tight tolerance is opt-in.
        """
        return np.abs(current - self.baseline) > (atol + rtol * np.abs(self.baseline))

    def describe(self, changed, current, limit=8):
        """Human-readable list of the changed coefficients (for fail-loud)."""
        idx = np.nonzero(changed)[0]
        lines = []
        for k in idx[:limit]:
            con_name, var_name, coef = self.provenance[int(k)]
            params = sorted({p.name for p in identify_mutable_parameters(coef)})
            pnote = f" [Param(s): {', '.join(params)}]" if params else ""
            lines.append(
                f"  - constraint '{con_name}' coefficient on variable "
                f"'{var_name}': {self.baseline[k]!r} -> {current[k]!r}{pnote}"
            )
        if len(idx) > limit:
            lines.append(f"  - ... and {len(idx) - limit} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The mutable-update plan
# --------------------------------------------------------------------------- #
class _MutablePlan:
    """Everything needed to re-evaluate and batch-push a model's mutable data.

    ``params`` is the ordered parameter vector's backing ``ParamData`` list;
    each group carries the target index array (columns / rows) it pushes and the
    :class:`_AffineArray` templates that produce the values from ``P``.
    """

    __slots__ = (
        'params',
        'param_index',
        'obj_cols',
        'obj_affine',
        'obj_offset_affine',
        'row_idx',
        'row_lower_affine',
        'row_upper_affine',
        'col_idx',
        'col_lower_affine',
        'col_upper_affine',
        'matrix_guard',
    )

    def read_param_vector(self):
        return np.fromiter(
            (p.value for p in self.params), dtype=np.float64, count=len(self.params)
        )

    @property
    def has_residual(self):
        for a in (
            self.obj_affine,
            self.obj_offset_affine,
            self.row_lower_affine,
            self.row_upper_affine,
            self.col_lower_affine,
            self.col_upper_affine,
        ):
            if a is not None and a.has_residual:
                return True
        if self.matrix_guard is not None and self.matrix_guard.has_residual:
            return True
        return False


def _active_objective(model):
    objs = list(model.component_data_objects(Objective, active=True, descend_into=True))
    return objs[0] if objs else None


def _build_mutable_plan(model, compiled: FastLoadCompiled):
    """Derive the :class:`_MutablePlan` (parameter registry + affine templates).

    A mutable constraint-matrix coefficient is **not** rejected: it is captured
    into the :class:`_MatrixGuard` (the affine template over ``P`` for every
    affected ``A``-entry, plus the values HiGHS was loaded with), so a warm solve
    can *verify the values* rather than *trust the mutability flag*.  See the
    module docstring's "Value-aware static-matrix guard" section.
    """
    col_of = {id(v): j for j, v in enumerate(compiled.columns)}

    # id(constraint) -> the A-row index/indices it maps to (a range row splits
    # into two rows that share one body, so both carry the same coefficients).
    con_rows = {}
    for i, (con, _mult) in enumerate(compiled.rows):
        con_rows.setdefault(id(con), []).append(i)

    # --- collect the mutable slots (expressions / constants) per group ------ #
    obj_cols = []
    obj_slots = []
    obj_offset_slot = None
    if compiled.has_objective:
        obj = _active_objective(model)
        if obj is not None:
            repn = generate_standard_repn(
                obj.expr, quadratic=False, compute_values=False
            )
            if repn.nonlinear_expr is not None:
                raise IncompatibleModelError(
                    f"The '{FastStepHighs.name}' warm interface only supports linear "
                    "objectives."
                )
            for coef, v in zip(repn.linear_coefs, repn.linear_vars):
                j = col_of.get(id(v))
                if j is None:
                    raise IncompatibleModelError(
                        f"The '{FastStepHighs.name}' warm interface could not map an "
                        "objective variable onto the compiled column space."
                    )
                if not is_constant(coef):
                    obj_cols.append(j)
                    obj_slots.append(coef)
            if not is_constant(repn.constant):
                obj_offset_slot = repn.constant

    row_idx = []
    row_lower_slots = []
    row_upper_slots = []
    # Guarded matrix coefficients, collected per (row, col) A-entry.
    mat_rows = []
    mat_cols = []
    mat_slots = []
    mat_prov = []
    body_cache = {}
    for i, (con, mult) in enumerate(compiled.rows):
        cid = id(con)
        repn = body_cache.get(cid)
        if repn is None:
            repn = generate_standard_repn(
                con.body, quadratic=False, compute_values=False
            )
            if repn.nonlinear_expr is not None:
                raise IncompatibleModelError(
                    f"The '{FastStepHighs.name}' warm interface only supports linear "
                    "constraints."
                )
            # A *mutable* coefficient on a free variable is a mutable A-entry.
            # Rather than reject the model on the flag, capture it into the value
            # guard (one entry per A-row the constraint occupies -- a range row
            # occupies two, both with the same body coefficients).
            for coef, v in zip(repn.linear_coefs, repn.linear_vars):
                j = col_of.get(id(v))
                if j is not None and not is_constant(coef):
                    for ri in con_rows[cid]:
                        mat_rows.append(ri)
                        mat_cols.append(j)
                        mat_slots.append(coef)
                        mat_prov.append((con.name, v.name, coef))
            body_cache[cid] = repn
        offset = repn.constant

        if mult == 0:  # equality: lower == upper == rhs
            rhs = con.upper - offset
            live = None if is_constant(rhs) else rhs
            if live is not None:
                row_idx.append(i)
                row_lower_slots.append(live)
                row_upper_slots.append(live)
        elif mult == 1:  # A x <= rhs ; lower open
            rhs = con.upper - offset
            if not is_constant(rhs):
                row_idx.append(i)
                row_lower_slots.append(None)
                row_upper_slots.append(rhs)
        else:  # mult == -1 : A x >= rhs ; upper open
            rhs = con.lower - offset
            if not is_constant(rhs):
                row_idx.append(i)
                row_lower_slots.append(rhs)
                row_upper_slots.append(None)

    col_idx = []
    col_lower_slots = []
    col_upper_slots = []
    for j, v in enumerate(compiled.columns):
        lb = v.lower
        ub = v.upper
        lb_mut = lb is not None and not is_constant(lb)
        ub_mut = ub is not None and not is_constant(ub)
        if not lb_mut and not ub_mut:
            continue
        col_idx.append(j)
        col_lower_slots.append(lb if lb_mut else float(compiled.col_lower[j]))
        col_upper_slots.append(ub if ub_mut else float(compiled.col_upper[j]))

    # --- parameter registry (ordered P) ------------------------------------- #
    params = []
    param_index = {}

    def _register(slots):
        for s in slots:
            if s is None or s.__class__ is float or s.__class__ is int:
                continue
            for p in identify_mutable_parameters(s):
                if id(p) not in param_index:
                    param_index[id(p)] = len(params)
                    params.append(p)

    _register(obj_slots)
    if obj_offset_slot is not None:
        _register([obj_offset_slot])
    _register(row_lower_slots)
    _register(row_upper_slots)
    _register(col_lower_slots)
    _register(col_upper_slots)
    _register(mat_slots)
    n_params = len(params)

    # --- affine templates --------------------------------------------------- #
    plan = _MutablePlan()
    plan.params = params
    plan.param_index = param_index
    plan.obj_cols = np.fromiter(obj_cols, dtype=np.int32, count=len(obj_cols))
    plan.obj_affine = _build_affine_array(obj_slots, param_index, n_params, +1)
    plan.obj_offset_affine = (
        None
        if obj_offset_slot is None
        else _build_affine_array([obj_offset_slot], param_index, n_params, +1)
    )
    plan.row_idx = np.fromiter(row_idx, dtype=np.int32, count=len(row_idx))
    plan.row_lower_affine = _build_affine_array(
        row_lower_slots, param_index, n_params, -1
    )
    plan.row_upper_affine = _build_affine_array(
        row_upper_slots, param_index, n_params, +1
    )
    plan.col_idx = np.fromiter(col_idx, dtype=np.int32, count=len(col_idx))
    plan.col_lower_affine = _build_affine_array(
        col_lower_slots, param_index, n_params, -1
    )
    plan.col_upper_affine = _build_affine_array(
        col_upper_slots, param_index, n_params, +1
    )

    # --- matrix-coefficient template (the value guard) ---------------------- #
    matrix_affine = _build_affine_array(mat_slots, param_index, n_params, +1)
    mat_rows_arr = np.fromiter(mat_rows, dtype=np.int32, count=len(mat_rows))
    mat_cols_arr = np.fromiter(mat_cols, dtype=np.int32, count=len(mat_cols))
    if mat_rows:
        # The element-wise values HiGHS was loaded with (an absent A-entry -- a
        # coefficient that is currently zero -- reads back as 0.0).
        Acsr = compiled.A.tocsr()
        mat_loaded = (
            np.asarray(Acsr[mat_rows_arr, mat_cols_arr]).ravel().astype(np.float64)
        )
    else:
        mat_loaded = None

    # --- self-check: templates must reproduce the loaded matrix (at the current
    # P) and a fresh value() evaluation (at a random perturbation of P) -------- #
    groups = [
        (
            'objective coefficient',
            obj_slots,
            plan.obj_affine,
            +1,
            compiled.c[plan.obj_cols] if len(plan.obj_cols) else None,
        ),
        (
            'objective offset',
            [obj_offset_slot] if obj_offset_slot is not None else [],
            plan.obj_offset_affine,
            +1,
            np.array([compiled.c_offset]) if obj_offset_slot is not None else None,
        ),
        (
            'row lower bound',
            row_lower_slots,
            plan.row_lower_affine,
            -1,
            compiled.row_lower[plan.row_idx] if len(plan.row_idx) else None,
        ),
        (
            'row upper bound',
            row_upper_slots,
            plan.row_upper_affine,
            +1,
            compiled.row_upper[plan.row_idx] if len(plan.row_idx) else None,
        ),
        (
            'variable lower bound',
            col_lower_slots,
            plan.col_lower_affine,
            -1,
            compiled.col_lower[plan.col_idx] if len(plan.col_idx) else None,
        ),
        (
            'variable upper bound',
            col_upper_slots,
            plan.col_upper_affine,
            +1,
            compiled.col_upper[plan.col_idx] if len(plan.col_idx) else None,
        ),
        ('constraint matrix coefficient', mat_slots, matrix_affine, +1, mat_loaded),
    ]
    _self_check_plan(plan, groups)

    # --- matrix value guard --------------------------------------------------- #
    # Baseline the guard against the template's *own* value at the current P (not
    # the compiled matrix directly): the self-check above already tied the two
    # together to ~1e-8, and using the template's value makes an unchanged roll
    # compare bit-exact (no spurious trip from M @ P vs compiler arithmetic).
    if mat_rows:
        import highspy

        hinf = highspy.kHighsInf
        baseline = matrix_affine.compute(plan.read_param_vector(), hinf)
    else:
        baseline = mat_cols_arr.astype(np.float64)  # empty float64 array
    plan.matrix_guard = _MatrixGuard(
        mat_rows_arr, mat_cols_arr, matrix_affine, baseline, mat_prov
    )
    return plan


def _eval_slots(slots, open_sign, hinf):
    """Ground-truth (non-vectorized) evaluation of a slot list via ``value()``."""
    out = np.empty(len(slots), dtype=np.float64)
    for i, s in enumerate(slots):
        if s is None:
            out[i] = open_sign * _inf
        elif s.__class__ is float or s.__class__ is int:
            out[i] = float(s)
        else:
            out[i] = value(s)
    np.clip(out, -hinf, hinf, out=out)
    return out


def _self_check_plan(plan, groups):
    """Validate every affine template against ground truth at two ``P`` points.

    ``groups`` is a list of ``(name, slots, affine, open_sign, compiled_ref)``.
    The baseline point checks the templates reproduce the matrix HiGHS loaded;
    the perturbed point checks they still equal a fresh ``value()`` evaluation
    when the parameters move -- catching any non-affine template.  Failure raises
    :class:`IncompatibleModelError`.
    """
    import highspy

    hinf = highspy.kHighsInf
    P0 = plan.read_param_vector()

    def _run(P, ground_truth):
        for name, slots, affine, sign, cref in groups:
            if affine is None or affine.n == 0:
                continue
            got = affine.compute(P, hinf)
            ref = (
                _eval_slots(slots, sign, hinf) if ground_truth else _finite(cref, hinf)
            )
            _assert_template_close(got, ref, name)

    _run(P0, ground_truth=False)  # vs the loaded matrix
    if len(P0):
        rng = np.random.RandomState(0)
        Pp = P0 + rng.uniform(-1.0, 1.0, size=P0.shape) + 0.5
        # Write the transient probe values straight to the Param storage: a
        # perturbation may leave a Param's declared domain (e.g. a negative value
        # on a NonNegativeReals price), which the validating ``.value`` setter
        # rejects.  The values are restored immediately, so bypassing validation
        # here is safe and only affects this in-memory self-check.
        saved = [_pd_get(p) for p in plan.params]
        try:
            for p, v in zip(plan.params, Pp):
                _pd_set(p, float(v))
            _run(Pp, ground_truth=True)  # vs a fresh value() evaluation
        finally:
            for p, v in zip(plan.params, saved):
                _pd_set(p, v)


def _assert_template_close(got, ref, what):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape or not np.allclose(
        got, ref, atol=_SELFCHECK_ATOL, rtol=_SELFCHECK_RTOL, equal_nan=False
    ):
        bad = int(np.argmax(np.abs(got - ref))) if got.shape == ref.shape else 0
        raise IncompatibleModelError(
            f"The '{FastStepHighs.name}' warm interface could not reproduce the "
            f"{what} array (mismatch at position {bad}: template {got[bad]!r} vs "
            f"reference {ref[bad]!r}).  The model's mutable structure is not "
            "supported by the warm update path; use 'highs_fastload' for a fresh "
            "compile per solve."
        )


# --------------------------------------------------------------------------- #
# The persistent warm-solve engine
# --------------------------------------------------------------------------- #
class FastStepHighs:
    """Persistent, array-native warm re-solve for a classic linear model.

    Compile once, retain a live ``highspy.Highs``, and on each subsequent solve
    read the parameter vector, expand the changed coefficients / bounds with a
    vectorized sparse ``M @ P``, and batch-push them to the solver, keeping the
    warm simplex basis.  See the module docstring for the update contract and
    scope.
    """

    CONFIG = BranchAndBoundConfig()
    name = 'highs_faststep'

    #: Accepted ``on_matrix_change`` policies (see :meth:`set_instance`).
    _MATRIX_CHANGE_POLICIES = ('error', 'reload')

    _available = None

    def __init__(self, on_matrix_change='error', matrix_atol=0.0, matrix_rtol=0.0):
        self.config = self.CONFIG()
        self._model = None
        self._compiled = None
        self._plan = None
        self._highs = None
        self._loader = None
        self._fingerprint = None
        self._col_map = None
        self._n_solves = 0
        # Value-aware static-matrix guard policy.
        self._on_matrix_change = self._check_matrix_policy(on_matrix_change)
        self._matrix_atol = float(matrix_atol)
        self._matrix_rtol = float(matrix_rtol)

    @classmethod
    def _check_matrix_policy(cls, policy):
        if policy not in cls._MATRIX_CHANGE_POLICIES:
            raise ValueError(
                f"on_matrix_change must be one of {cls._MATRIX_CHANGE_POLICIES!r}; "
                f"got {policy!r}."
            )
        return policy

    # ------------------------------------------------------------------ #
    # availability / version
    # ------------------------------------------------------------------ #
    def available(self):
        if FastStepHighs._available is None:
            import importlib.util

            FastStepHighs._available = (
                Availability.FullLicense
                if importlib.util.find_spec('highspy') is not None
                else Availability.NotFound
            )
        return FastStepHighs._available

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

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def set_instance(
        self,
        model,
        *,
        on_matrix_change=None,
        matrix_atol=None,
        matrix_rtol=None,
        **options,
    ):
        """Compile ``model`` once, build the affine templates, and load HiGHS.

        Value-aware static-matrix guard options (may also be set on the
        constructor):

        on_matrix_change : {'error', 'reload'}
            What to do when a warm solve finds a constraint-matrix coefficient
            has genuinely changed since ``set_instance``.  ``'error'`` (the
            default) fails loud, naming the offending coefficient(s) -- a warm
            re-solve on the retained basis would use a stale matrix.  ``'reload'``
            transparently rebuilds the whole standard-form matrix and reloads it
            (basis reset) for that solve, then continues.  Neither ever solves a
            stale matrix.
        matrix_atol, matrix_rtol : float
            The comparison tolerance against the loaded coefficient values.  Both
            default to ``0.0`` -- an **exact** comparison, so any drift trips the
            guard.  Widen for a tight tolerance on models whose matrix Params
            round-trip with tiny numerical noise.
        """
        import highspy

        if not self.available():
            raise RuntimeError(
                f"The '{self.name}' warm interface requires highspy, which is not "
                "available in this environment."
            )
        if on_matrix_change is not None:
            self._on_matrix_change = self._check_matrix_policy(on_matrix_change)
        if matrix_atol is not None:
            self._matrix_atol = float(matrix_atol)
        if matrix_rtol is not None:
            self._matrix_rtol = float(matrix_rtol)
        if options:
            self.config = self.config(value=options, preserve_implicit=True)

        compiled = compile_to_highs_arrays(model)
        plan = _build_mutable_plan(model, compiled)  # includes the template self-check

        lp = build_highs_lp(compiled)
        highs = highspy.Highs()
        highs.setOptionValue('log_to_console', False)
        self._apply_config_options(highs)
        highs.passModel(lp)

        self._model = model
        self._compiled = compiled
        self._plan = plan
        self._highs = highs
        self._loader = None
        self._col_map = None
        self._fingerprint = _structure_fingerprint(model)
        self._n_solves = 0
        return self

    def _apply_config_options(self, highs):
        config = self.config
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

    # ------------------------------------------------------------------ #
    # solve
    # ------------------------------------------------------------------ #
    def solve(
        self,
        *,
        keep_basis=True,
        load_solutions=True,
        check_structure=True,
        update=True,
        raise_on_nonoptimal=None,
        param_values=None,
        dirty=None,
    ):
        """Warm re-solve: expand the changed data as arrays, keep the basis, solve.

        Parameters
        ----------
        keep_basis : bool
            Keep the retained simplex basis (warm start, the default).  ``False``
            discards it (``clearSolver``) for a cold re-solve -- the basis-reset
            equivalence path.
        load_solutions : bool
            Load the primal solution back onto the Pyomo variables.
        check_structure : bool
            Verify the model's structure fingerprint before solving (the default).
        update : bool
            Re-extract the model's mutable data and batch-push it before solving
            (the default).  Set ``False`` when the solver state has already been
            updated out of band (e.g. via :meth:`set_objective_coefficients`) and
            no further model re-extraction is wanted.
        param_values : array-like, optional
            The parameter vector ``P`` (ordered by :attr:`parameters`) to use
            *instead of* reading the model.  Lets a caller drive the solve from
            raw arrays with no Pyomo mutation; requires a fully templatable model
            (no residual entries).  Implies ``update``.
        dirty : bool array-like, optional
            With ``param_values``: a mask over ``P`` marking which parameters
            changed, so only the affected rows/columns are recomputed and pushed.
        """
        if self._highs is None:
            raise RuntimeError(f"'{self.name}'.solve() called before set_instance().")
        do_update = update or param_values is not None
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        tick = time.perf_counter()
        timer = HierarchicalTimer()
        ostreams = [io.StringIO()] + list(self.config.tee)

        try:
            with capture_output(TeeStream(*ostreams), capture_fd=True):
                if (
                    do_update
                    and self._n_solves > 0
                    and check_structure
                    and param_values is None
                ):
                    self._check_structure()
                if do_update:
                    timer.start('update')
                    self._push_updates(param_values, dirty)
                    timer.stop('update')
                if not keep_basis:
                    self._highs.clearSolver()
                timer.start('optimize')
                self._highs.run()
                timer.stop('optimize')
            results = self._postsolve(load_solutions, raise_on_nonoptimal)
        except InfeasibleConstraintException as err:
            results = get_infeasible_results(
                model=self._model,
                solver=self,
                config=self.config,
                err_msg='The problem was proven infeasible during update:\n' f'\t{err}',
            )

        self._n_solves += 1
        results.solver_log = ostreams[0].getvalue()
        tock = time.perf_counter()
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = tock - tick
        results.timing_info.timer = timer
        return results

    def _check_structure(self):
        fp = _structure_fingerprint(self._model)
        if fp != self._fingerprint:
            raise IncompatibleModelError(
                f"The '{self.name}' warm interface detected a structure change "
                f"since set_instance (fingerprint {self._fingerprint} -> {fp}): a "
                "constraint or variable was added/removed, or the objective was "
                "replaced.  Build a fresh FastStepHighs (or call set_instance "
                "again) -- the warm update path only supports data (Param/bound) "
                "changes on a fixed structure."
            )

    def _matrix_change_error(self, guard, changed, current, array_mode):
        details = guard.describe(changed, current)
        if array_mode:
            remedy = (
                " The array-driven (param_values) path cannot reload from the "
                "model, so a matrix change is always fatal here; drive the solve "
                "from the model (omit param_values) with on_matrix_change='reload' "
                "to rebuild instead."
            )
        else:
            remedy = (
                " Set on_matrix_change='reload' to rebuild and reload the matrix "
                "(basis reset) for this solve instead of failing."
            )
        return IncompatibleModelError(
            f"The '{self.name}' warm interface detected a genuine change in "
            f"{int(changed.sum())} constraint-matrix coefficient(s) since "
            "set_instance; a warm re-solve on the retained basis would use a "
            f"stale matrix.{remedy}\n{details}"
        )

    def _reload_model(self, P, hinf):
        """Rebuild + reload the whole matrix (the ``on_matrix_change='reload'``
        path): a fresh standard-form compile and ``passModel`` (basis reset).

        Deliberately *not* an incremental matrix edit -- applying a genuinely
        changed ``A`` as batch updates is a later stage that reuses the guard's
        coefficient mapping; this path just re-loads the whole model.  If the
        recompile shifts the matrix *shape* (e.g. a coefficient rolled to exactly
        zero and dropped a column/nonzero), the templates and mapping would no
        longer line up, so the whole instance is rebuilt from scratch.
        """
        compiled = compile_to_highs_arrays(self._model)
        if (
            compiled.n_col != self._compiled.n_col
            or compiled.n_row != self._compiled.n_row
            or compiled.nnz != self._compiled.nnz
        ):
            self.set_instance(self._model)
            return
        lp = build_highs_lp(compiled)
        self._highs.passModel(lp)
        self._compiled = compiled
        self._loader = None
        # Re-baseline the guard to the reloaded matrix (its current-P value), so
        # the *next* roll detects the next change.
        guard = self._plan.matrix_guard
        if guard is not None and not guard.is_empty:
            guard.baseline = guard.affine.compute(P, hinf)

    def _push_updates(self, param_values, dirty):
        import highspy

        hinf = highspy.kHighsInf
        highs = self._highs
        plan = self._plan

        if param_values is None:
            P = plan.read_param_vector()
        else:
            P = np.ascontiguousarray(param_values, dtype=np.float64)
            if P.shape != (len(plan.params),):
                raise ValueError(
                    f"param_values has length {P.shape} but the model has "
                    f"{len(plan.params)} mutable parameters."
                )
            if plan.has_residual:
                raise ValueError(
                    "param_values (array-driven) update requires a fully "
                    "templatable model, but this model has residual "
                    "(non-parameter-affine) entries; drive the solve from the "
                    "model instead (omit param_values)."
                )

        # --- value-aware static-matrix guard: never solve a stale matrix ------ #
        # Re-evaluate the mutable matrix coefficients (vectorized) and compare to
        # the values HiGHS holds.  Unchanged -> the retained matrix is still
        # correct, keep the warm basis (the fast path).  Changed -> fail loud, or
        # (opt-in) reload; never a stale-matrix solve.
        guard = plan.matrix_guard
        if guard is not None and not guard.is_empty:
            current = guard.current(P, hinf)
            changed = guard.changed_mask(current, self._matrix_atol, self._matrix_rtol)
            if changed.any():
                array_mode = param_values is not None
                if self._on_matrix_change == 'reload' and not array_mode:
                    # Rebuild + reload the whole standard-form matrix (basis
                    # reset) for this solve, then continue.  The fresh compile
                    # bakes in the current matrix *and* data, so the incremental
                    # push below is skipped.
                    self._reload_model(P, hinf)
                    return
                raise self._matrix_change_error(guard, changed, current, array_mode)

        if dirty is not None:
            dirty_cols = np.nonzero(np.asarray(dirty, dtype=bool))[0]
            self._push_dirty(highs, plan, P, hinf, dirty_cols)
            return

        # objective coefficients
        if len(plan.obj_cols):
            costs = plan.obj_affine.compute(P, hinf)
            highs.changeColsCost(len(plan.obj_cols), plan.obj_cols, costs)
        if plan.obj_offset_affine is not None:
            highs.changeObjectiveOffset(
                float(plan.obj_offset_affine.compute(P, hinf)[0])
            )
        # row bounds
        if len(plan.row_idx):
            lo = plan.row_lower_affine.compute(P, hinf)
            up = plan.row_upper_affine.compute(P, hinf)
            highs.changeRowsBounds(len(plan.row_idx), plan.row_idx, lo, up)
        # variable bounds
        if len(plan.col_idx):
            lo = plan.col_lower_affine.compute(P, hinf)
            up = plan.col_upper_affine.compute(P, hinf)
            highs.changeColsBounds(len(plan.col_idx), plan.col_idx, lo, up)

    def _push_dirty(self, highs, plan, P, hinf, dirty_cols):
        # objective coefficients
        if len(plan.obj_cols):
            rows = plan.obj_affine.affected_rows(dirty_cols)
            if len(rows):
                vals = plan.obj_affine.compute_rows(rows, P, hinf)
                highs.changeColsCost(len(rows), plan.obj_cols[rows], vals)
        if plan.obj_offset_affine is not None:
            if len(plan.obj_offset_affine.affected_rows(dirty_cols)):
                highs.changeObjectiveOffset(
                    float(plan.obj_offset_affine.compute(P, hinf)[0])
                )
        # row bounds (a row is pushed if either side is affected)
        if len(plan.row_idx):
            rows = np.union1d(
                plan.row_lower_affine.affected_rows(dirty_cols),
                plan.row_upper_affine.affected_rows(dirty_cols),
            ).astype(np.int64)
            if len(rows):
                lo = plan.row_lower_affine.compute_rows(rows, P, hinf)
                up = plan.row_upper_affine.compute_rows(rows, P, hinf)
                highs.changeRowsBounds(len(rows), plan.row_idx[rows], lo, up)
        # variable bounds
        if len(plan.col_idx):
            rows = np.union1d(
                plan.col_lower_affine.affected_rows(dirty_cols),
                plan.col_upper_affine.affected_rows(dirty_cols),
            ).astype(np.int64)
            if len(rows):
                lo = plan.col_lower_affine.compute_rows(rows, P, hinf)
                up = plan.col_upper_affine.compute_rows(rows, P, hinf)
                highs.changeColsBounds(len(rows), plan.col_idx[rows], lo, up)

    # ------------------------------------------------------------------ #
    # parameter-vector introspection (the array/mapping-free contract)
    # ------------------------------------------------------------------ #
    @property
    def parameters(self):
        """The ordered ``ParamData`` list backing the parameter vector ``P``.

        A caller that drives :meth:`solve` with ``param_values`` supplies an array
        in this order; :class:`FastStepHighs` owns the row/column mapping.
        """
        self._require_loaded()
        return list(self._plan.params)

    def read_param_vector(self):
        """The current parameter vector ``P`` read from the model (``float64``)."""
        self._require_loaded()
        return self._plan.read_param_vector()

    # ------------------------------------------------------------------ #
    # explicit array update API (raw index-addressed pushes)
    # ------------------------------------------------------------------ #
    def set_objective_coefficients(self, values, cols=None):
        """Set objective coefficients directly (bypassing the templates)."""
        self._require_loaded()
        values = np.ascontiguousarray(values, dtype=np.float64)
        cols = (
            np.arange(self._compiled.n_col, dtype=np.int32)
            if cols is None
            else np.ascontiguousarray(cols, dtype=np.int32)
        )
        self._highs.changeColsCost(len(cols), cols, values)

    def set_row_bounds(self, lower, upper, rows=None):
        """Set constraint (row) bounds directly.  ``+/- inf`` map to HiGHS inf."""
        import highspy

        self._require_loaded()
        lower = _finite(np.asarray(lower, dtype=np.float64), highspy.kHighsInf)
        upper = _finite(np.asarray(upper, dtype=np.float64), highspy.kHighsInf)
        rows = (
            np.arange(self._compiled.n_row, dtype=np.int32)
            if rows is None
            else np.ascontiguousarray(rows, dtype=np.int32)
        )
        self._highs.changeRowsBounds(len(rows), rows, lower, upper)

    def set_variable_bounds(self, lower, upper, cols=None):
        """Set variable (column) bounds directly.  ``+/- inf`` map to HiGHS inf."""
        import highspy

        self._require_loaded()
        lower = _finite(np.asarray(lower, dtype=np.float64), highspy.kHighsInf)
        upper = _finite(np.asarray(upper, dtype=np.float64), highspy.kHighsInf)
        cols = (
            np.arange(self._compiled.n_col, dtype=np.int32)
            if cols is None
            else np.ascontiguousarray(cols, dtype=np.int32)
        )
        self._highs.changeColsBounds(len(cols), cols, lower, upper)

    def column_index(self, var):
        """The HiGHS column index of a Pyomo ``VarData`` (or ``None``)."""
        self._require_loaded()
        if self._col_map is None:
            self._col_map = {id(v): j for j, v in enumerate(self._compiled.columns)}
        return self._col_map.get(id(var))

    def row_indices(self, con):
        """The HiGHS row index/indices a Pyomo ``ConstraintData`` maps to."""
        self._require_loaded()
        want = id(con)
        return [i for i, (c, _m) in enumerate(self._compiled.rows) if id(c) == want]

    def _require_loaded(self):
        if self._highs is None:
            raise RuntimeError(f"'{self.name}': set_instance() has not been called.")

    # ------------------------------------------------------------------ #
    # postsolve  (mirrors fastload's FastLoadHighs._postsolve)
    # ------------------------------------------------------------------ #
    def _postsolve(self, load_solutions, raise_on_nonoptimal=None):
        import highspy

        highs = self._highs
        compiled = self._compiled
        status = highs.getModelStatus()
        info = highs.getInfo()

        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_config = self.config
        results.timing_info.highs_time = highs.getRunTime()
        loader = FastLoadHighsSolutionLoader(
            highs, self._model, compiled.columns, compiled.rows
        )
        self._loader = loader
        results.solution_loader = loader

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

        raise_nonopt = (
            self.config.raise_exception_on_nonoptimal_result
            if raise_on_nonoptimal is None
            else raise_on_nonoptimal
        )
        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and raise_nonopt
        ):
            raise NoOptimalSolutionError()

        results.incumbent_objective = None
        results.objective_bound = None
        if compiled.has_objective:
            if has_feasible_solution:
                results.incumbent_objective = info.objective_function_value
            if info.mip_node_count == -1:
                if (
                    has_feasible_solution
                    and results.termination_condition
                    == TerminationCondition.convergenceCriteriaSatisfied
                ):
                    results.objective_bound = info.objective_function_value
            else:
                results.objective_bound = info.mip_dual_bound

        if load_solutions:
            if has_feasible_solution:
                results.solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()

        return results

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


def _structure_fingerprint(model):
    """A cheap, structure-only fingerprint of a model.

    O(number of component *containers*), not O(number of data objects): catches a
    constraint/variable added or removed and the objective being swapped.  It does
    not catch an equal-count swap; the documented contract is that structure is
    fixed across solves.
    """
    n_con = 0
    for c in model.component_objects(Constraint, active=True, descend_into=True):
        n_con += len(c)
    n_var = 0
    for x in model.component_objects(Var, active=True, descend_into=True):
        n_var += len(x)
    obj = _active_objective(model)
    return (n_con, n_var, id(obj))


def _pd_get(p):
    """Current value of a mutable ``ParamData``."""
    return value(p)


def _pd_set(p, v):
    """Write a value straight to a ``ParamData``'s storage, bypassing the
    validating ``.value`` setter (used only for the transient self-check probe,
    which may leave a Param's declared domain and is restored immediately)."""
    p._value = v


def _finite(arr, hinf):
    """Replace +/- inf in ``arr`` with +/- ``hinf`` (HiGHS infinity)."""
    arr = np.asarray(arr, dtype=np.float64)
    out = np.where(np.isneginf(arr), -hinf, arr)
    out = np.where(np.isposinf(out), hinf, out)
    return out
