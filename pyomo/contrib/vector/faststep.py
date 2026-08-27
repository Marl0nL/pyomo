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

Verified-static parameter folding (non-affine param participation)
------------------------------------------------------------------
The value guard above lets a *nominally-mutable* matrix coefficient through when
its value holds still, but it needs the coefficient to be **affine in the
parameter vector** to template it.  Real models routinely put practically-
constant mutable Params into **products and reciprocals** with other quantities
-- ``price[t] * duration`` in the objective, ``efficiency * duration`` and
``duration / efficiency`` in the matrix.  Such a coefficient is *not* affine in
the parameters (a product of two mutable Params has a non-constant partial), so
the affine self-check would reject the whole model even though ``duration`` /
``efficiency`` never actually change between rolls.

:class:`FastStepHighs` closes that gap by **folding** the verified-static
parameters.  At ``set_instance`` it classifies each mutable Param
(:func:`_classify_folded`): a Param that participates *non-affinely* (a
reciprocal ``1/eff``, or the structural-constant factor of a product coupling
many coefficients like a single ``duration`` multiplying every ``price[t]``) is
**folded** -- its current value substituted as a constant during template
construction -- while the affinely-participating Params stay templated.  After
folding, ``price[t] * duration`` becomes the affine ``duration_value *
price[t]`` over the remaining varying ``price[t]``, and template construction
succeeds; the model that was rejected now engages the warm path.

Every folded Param joins the value-guard watch list: each warm solve verifies
(vectorized) that the folded values are unchanged from ``set_instance``.  A
genuine change to a folded value means the templates are stale *by construction*
-- the default ``on_matrix_change='error'`` fails loud naming the Param, and the
opt-in ``on_matrix_change='reload'`` re-folds, re-templates, and reloads a fresh
model for that solve.  Never a silent stale solve.  A product of two *genuinely
varying* Params (no static factor to fold, e.g. a lone ``p*q``) is still rejected
loudly -- there is no correct affine template for it.  :attr:`folded_parameters`
/ :attr:`templated_parameters` / :meth:`classification_report` expose which
Params were folded vs templated (and an ``INFO`` log line reports it), so a user
can see why their model engaged and exactly what the guard is watching.

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
  changes fails loud or (opt-in) triggers a reload -- never a stale solve.
* **A coefficient non-affine in the parameters** (a product / reciprocal of
  Params, e.g. ``price*duration``) is handled by folding its verified-static
  factor (see the folding section above): the practically-constant Param is
  folded in as a watched constant so the coefficient becomes affine in the
  remaining varying Params.  A product of two *genuinely-varying* Params (no
  static factor to fold, a lone ``p*q``) has no correct affine template and is
  still rejected loudly at ``set_instance``.
* A **structure change** between solves (a constraint or variable added/removed,
  the objective swapped) is caught by a cheap fingerprint check and rejected --
  the caller must build a fresh :class:`FastStepHighs`.
"""

from __future__ import annotations

import datetime
import heapq
import io
import logging
import time

from pyomo.common.dependencies import numpy as np, scipy
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer

from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.var import Var
from pyomo.core.expr import identify_mutable_parameters, identify_variables
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.core.expr.numeric_expr import ProductExpression, NPV_ProductExpression
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
    build_highs_model,
    compile_to_highs_arrays,
)

logger = logging.getLogger(__name__)

_inf = float('inf')

# Tolerance for the set_instance self-check (templates must reproduce the
# compiled standard-form arrays).  Absolute + relative.
_SELFCHECK_ATOL = 1e-8
_SELFCHECK_RTOL = 1e-7

# The signatures below differentiate with reverse-mode *symbolic* differentiation
# (:func:`reverse_sd`, the engine behind ``differentiate(..., mode='reverse_
# symbolic')``).  It returns each partial as an *expression* (``d(price*dur)/dprice
# == dur``), so a partial that still references another varying parameter is
# detectable as non-constant.  The numeric mode (``reverse_ad``) instead bakes in
# the current values of the other params, so a genuinely non-affine product would
# yield a numerically-constant -- and therefore *wrong* -- partial.

# Sentinel returned by :func:`_affine_from_sig` for an entry that is genuinely
# non-affine in a *varying* parameter (a product/reciprocal of two varying params)
# -- distinct from ``None`` (a fixed-variable residual, evaluated per-solve).  A
# ``_NONAFFINE`` entry that survives fold classification is rejected loudly at
# ``set_instance`` (it can never be templated correctly).
_NONAFFINE = object()


# --------------------------------------------------------------------------- #
# Per-coefficient signature: the expensive symbolic walk, computed ONCE
# --------------------------------------------------------------------------- #
# Building the affine templates and classifying the fold set both need, for each
# mutable coefficient expression, the same facts: whether it is a bare parameter,
# a fixed-variable residual, or affine-in-parameters -- and, if affine, each
# parameter's symbolic partial (its value, and which parameters that partial still
# references).  The pre-refactor code walked every expression *twice* -- once in
# :func:`_classify_folded` to build the couplings, once in ``_affine_over_varying``
# to build the templates (plus a third ``identify_mutable_parameters`` walk to
# register the parameter vector) -- which made the per-coefficient ``differentiate``
# the dominant compile cost.  A :class:`_CoefSig` captures the walk once; fold
# classification, parameter registration, and template construction all read it.
_SIG_PARAM = 0  # a bare mutable parameter (price[t]) -- the dominant case
_SIG_RESIDUAL = 1  # references a (fixed) variable -- evaluated per-solve
_SIG_AFFINE = 2  # a linear-combination-of-parameters coefficient


class _CoefSig:
    """The once-computed symbolic signature of a mutable coefficient expression.

    ``all_params`` is the expression's mutable parameters in
    ``identify_mutable_parameters`` order (so the parameter vector registers in a
    stable order).  For an affine signature ``terms`` holds the per-parameter
    ``(param, d_value, dep_ids)`` triple -- ``d_value`` is the symbolic partial
    ``d(e)/d(param)`` evaluated at ``set_instance``, ``dep_ids`` the ids of the
    parameters that partial still references (empty ==> a constant partial ==>
    affine in that parameter) -- and ``base_value`` is ``value(e)`` at
    ``set_instance``.
    """

    __slots__ = ('kind', 'all_params', 'terms', 'base_value')

    def __init__(self, kind, all_params, terms, base_value):
        self.kind = kind
        self.all_params = all_params
        self.terms = terms
        self.base_value = base_value


def _coef_signature(e):
    """Compute the :class:`_CoefSig` for one mutable coefficient expression.

    Mirrors the branch order the pre-refactor ``_affine_over_varying`` /
    :func:`_classify_folded` used (bare parameter, then fixed-variable residual,
    then the affine per-parameter ``differentiate`` walk), so the templates and
    fold classification derived from it are byte-for-byte what those two
    independent walks produced.
    """
    if getattr(e, 'is_parameter_type', _false)():
        return _CoefSig(_SIG_PARAM, (e,), None, None)
    for _v in identify_variables(e, include_fixed=True):
        # A (fixed) variable makes the entry a residual (evaluated per-solve); it
        # still registers its parameters so their vector position stays stable.
        return _CoefSig(
            _SIG_RESIDUAL, tuple(identify_mutable_parameters(e)), None, None
        )
    params = tuple(identify_mutable_parameters(e))
    terms = []
    # Only differentiate when a mutable parameter is actually present: the
    # pre-refactor code called ``differentiate`` *inside* the per-parameter loop,
    # so a non-constant coefficient with no mutable parameter (e.g. an
    # ``NPV_Max``/``NPV_Min`` bound over constants, which reverse-mode cannot
    # differentiate) was never differentiated -- it is simply a constant shift.
    if params:
        # Structural fast path for the dominant coefficient shape -- a product of
        # two distinct bare parameters (``price*dt``) or a constant times one bare
        # parameter -- whose partial wrt each parameter factor is simply the other
        # factor (exactly the node reverse-mode differentiation returns).  Any
        # other structure (a sum, reciprocal, power, or shared factor) returns
        # ``None`` and falls back to the reverse-mode walk below.
        terms = _product_terms(e, params)
        if terms is None:
            # Reverse-mode symbolic differentiation computes the partial wrt
            # *every* node in one backward walk, so ``reverse_sd(e)`` once and
            # indexing per parameter replaces the pre-refactor
            # ``differentiate(e, wrt=p)`` per parameter -- which re-ran the whole
            # reverse-mode walk for each parameter.  The indexed result is
            # identical to ``differentiate(e, wrt=p, mode='reverse_symbolic')``
            # (which is exactly ``reverse_sd(e)[p]``, or ``0`` when ``p`` absent).
            derivs = reverse_sd(e)
            terms = []
            for p in params:
                d = derivs[p] if p in derivs else 0
                deps = _deriv_deps(d)
                terms.append((p, float(value(d)), deps))
    return _CoefSig(_SIG_AFFINE, params, terms, float(value(e)))


def _deriv_deps(d):
    """The mutable parameters a symbolic partial ``d`` still references, as an
    id frozenset.  A bare-parameter partial (the dominant ``d(price*dur)/dprice ==
    dur`` case) is its own single dependency when mutable, or none when an
    immutable parameter -- exactly what ``identify_mutable_parameters`` returns,
    without the walk."""
    if getattr(d, 'is_parameter_type', _false)():
        return frozenset() if is_constant(d) else frozenset((id(d),))
    return frozenset(id(pp) for pp in identify_mutable_parameters(d))


def _product_terms(e, params):
    """Structural affine terms for a two-factor product of bare parameters and
    constants, or ``None`` when ``e`` is not that shape (fall back to reverse_sd).

    For ``e = f0 * f1`` with each factor a bare parameter or a constant (and, when
    both factors are parameters, two *distinct* parameters), the partial wrt a
    parameter factor is exactly the other factor -- the same node reverse-mode
    differentiation returns -- so the affine term ``(param, value(other),
    deps(other))`` is read straight off the product, byte-for-byte what the
    reverse-mode path would produce, without a differentiation walk.
    """
    if (
        e.__class__ is not ProductExpression
        and e.__class__ is not NPV_ProductExpression
    ):
        return None
    args = e.args
    if len(args) != 2:
        return None
    a, b = args
    a_param = getattr(a, 'is_parameter_type', _false)()
    b_param = getattr(b, 'is_parameter_type', _false)()
    a_ok = a_param or a.__class__ is int or a.__class__ is float or is_constant(a)
    b_ok = b_param or b.__class__ is int or b.__class__ is float or is_constant(b)
    if not (a_ok and b_ok):
        return None
    if a_param and b_param and a is b:
        return None  # p*p -> the partial is 2p, not the "other" factor
    terms = []
    for p in params:
        if p is a:
            other = b
        elif p is b:
            other = a
        else:
            return None  # a mutable parameter that is not itself a factor
        terms.append((p, float(value(other)), _deriv_deps(other)))
    return terms


class _MaxName:
    """A string wrapper that orders in reverse, so popping a min-heap reproduces
    ``max(candidates, key=name)`` on a coverage tie (see :func:`_classify_folded`).
    Parameter names are unique, so this never has to fall through to a further
    tie-break."""

    __slots__ = ('s',)

    def __init__(self, s):
        self.s = s

    def __lt__(self, other):
        return self.s > other.s


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


def _affine_from_sig(sig, param_index, folded_ids):
    """Decompose a coefficient's :class:`_CoefSig` affinely over the varying params.

    ``folded_ids`` is the set of ``id(ParamData)`` that have been *folded* --
    treated as watched constants at their current values (see
    :func:`_classify_folded`).  A folded parameter contributes its current value
    to the constant term but never a template column, so a product like
    ``price * duration`` with ``duration`` folded decomposes to the affine
    ``duration_value * price`` over the remaining varying parameter ``price``.

    Same contract as the pre-refactor ``_affine_over_varying`` (which walked the
    expression directly, re-doing the ``differentiate`` the signature already
    holds):

    * ``(shift, {param_pos: coef})`` when the coefficient is affine in the varying
      parameters (folded parameters treated as constants);
    * ``None`` when it references a (fixed) variable -- a *residual*, evaluated
      per-solve with ``value()`` (correctness preserved, that entry not
      vectorized);
    * :data:`_NONAFFINE` when it is genuinely non-affine in a *varying* parameter
      (a product/reciprocal of two varying params that folding did not resolve) --
      the caller rejects the model loudly.
    """
    # Fast path: a bare mutable parameter (``price[t]``) -- the dominant case.
    if sig.kind == _SIG_PARAM:
        p = sig.all_params[0]
        if id(p) in folded_ids:
            return float(value(p)), {}  # a bare folded param is a constant
        return 0.0, {param_index[id(p)]: 1.0}
    # A (fixed) variable in the expression makes the constant term depend on a
    # value that is not in the parameter vector -> evaluate as a residual.
    if sig.kind == _SIG_RESIDUAL:
        return None
    coefs = {}
    shift = sig.base_value
    for p, dval, deps in sig.terms:
        pid = id(p)
        if pid in folded_ids:
            continue  # folded -> constant; its value is already folded into shift
        # ``e`` is affine in the varying param ``p`` iff its symbolic derivative
        # references *no varying parameter* (it may reference folded ones -- those
        # are constants).  A varying reference means a genuine coupling.
        for dep in deps:
            if dep not in folded_ids:
                return _NONAFFINE
        coefs[param_index[pid]] = dval
        shift -= dval * float(value(p))
    return shift, coefs


def _false():
    return False


def _build_affine_array(
    slots, sig_by_id, param_index, n_params, open_sign, folded_ids, what
):
    """Build an :class:`_AffineArray` from a list of bound/coefficient ``slots``.

    Each slot is ``None`` (an open bound side -> ``open_sign * inf`` in ``base``),
    a plain ``float`` (a fixed value kept in ``base``), or a Pyomo expression
    (templated when affine in the varying parameters, folded parameters treated
    as constants; a fixed-variable entry recorded as a residual and evaluated
    with ``value()`` every solve).  ``sig_by_id`` maps ``id(expr) -> _CoefSig``
    (the once-computed symbolic signature reused across every group).  ``what``
    names the group for a fail-loud message if an entry is genuinely non-affine in
    a varying parameter.
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
            aff = _affine_from_sig(sig_by_id[id(s)], param_index, folded_ids)
            if aff is None:
                residual.append((i, s))
            elif aff is _NONAFFINE:
                params = sorted({p.name for p in identify_mutable_parameters(s)})
                raise IncompatibleModelError(
                    f"The '{FastStepHighs.name}' warm interface found a {what} that "
                    "is non-affine in the varying parameters and could not be made "
                    "affine by folding a verified-static parameter (its factors "
                    f"[{', '.join(params)}] all vary): a product/reciprocal of two "
                    "genuinely-mutable parameters cannot be templated.  Declare one "
                    "factor immutable (or use 'highs_fastload' for a fresh compile "
                    "per solve)."
                )
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


def _classify_folded(sigs, hub_min=2):
    """Decide which mutable parameters to **fold** (treat as watched constants).

    ``sigs`` is the list of :class:`_CoefSig` signatures for the mutable
    coefficient / bound expressions across every template group (objective
    coefficients and offset, row bounds/RHS, variable bounds, matrix
    coefficients).  A parameter is *folded* -- its current
    value substituted as a constant during template construction -- when it
    participates **non-affinely** and folding it is what makes the remaining
    (varying) parameters affine.  Two reasons force a fold:

    * **Self-coupling (forced).**  A parameter that appears in its *own* symbolic
      derivative -- a reciprocal ``1/eff`` or a power ``dur**2`` -- is non-affine
      in itself and can never be templated; it must be folded.
    * **A hub coupling (greedy).**  Two varying parameters multiplied together
      (``price[t] * duration``) are mutually non-affine; folding either resolves
      it.  We fold the parameter appearing in the *most* such couplings -- the
      structural constant (a single ``duration`` coupling every ``price[t]``)
      rather than the many varying data parameters.  A coupling with no hub (two
      co-equal single-use parameters, e.g. a lone ``p*q``) is left unfolded and
      rejected downstream: with no evidence which factor is static, folding one
      would be a coin-flip that the value guard would trip every roll.

    Returns ``(folded_ids: set, names: dict[id -> Param.name])``, ``names`` holding
    only the fold candidates that were actually named (see below).  The fold set
    is a *best effort* to make the templates affine; the ``set_instance``
    self-check remains the hard correctness gate, and every folded parameter is
    watched by the value guard (a genuine change fails loud or, opt-in, reloads).
    """
    couplings = []  # per-expr {param_id: frozenset(dep param ids in d/dp)}
    pd_by_id = {}  # param_id -> ParamData (for lazy name lookup on fold candidates)
    for sig in sigs:
        if sig.kind == _SIG_PARAM:
            p = sig.all_params[0]
            couplings.append({id(p): frozenset()})
            pd_by_id[id(p)] = p
        elif sig.kind == _SIG_RESIDUAL:
            couplings.append({})  # fixed-variable residual: no param templating
        else:
            cmap = {}
            for p, _dval, deps in sig.terms:
                pd_by_id[id(p)] = p
                cmap[id(p)] = deps
            couplings.append(cmap)

    # ``getname`` on an indexed component is costly, and the greedy tie-break only
    # ever needs the name of the fold *candidates* (the parameters that reach the
    # heap) -- a small fraction of the parameters on a large model.  Resolve names
    # lazily, memoized, instead of naming every parameter up front.
    names = {}

    def _name(pid):
        nm = names.get(pid)
        if nm is None:
            pd = pd_by_id.get(pid)
            nm = pd.name if pd is not None else ''
            names[pid] = nm
        return nm

    # Forced folds: a parameter non-affine in itself (reciprocal / power).
    folded = set()
    for cmap in couplings:
        for pid, deps in cmap.items():
            if pid in deps:
                folded.add(pid)

    # Greedy hub folding for the remaining product couplings, maintained
    # *incrementally*.  The obvious implementation rebuilds the full candidate set
    # and rescans every coupling on each fold -- O(folds x couplings), which turns
    # quadratic on a model whose number of hub folds grows with its size (a
    # per-asset / per-zone structural constant).  Instead we track, per coupling,
    # the parameters currently *involved* in its unresolved conflict; folding a
    # parameter re-examines only the couplings it actually touched, and a lazy
    # max-heap keyed by ``(coverage, name)`` reproduces the exact same greedy
    # selection (coverage only ever shrinks as folds accumulate, and parameter
    # names are unique, so the argmax is identical to a full rescan).
    def _involved(cmap):
        inv = set()
        for pid, deps in cmap.items():
            if pid in folded:
                continue
            residual = deps - folded
            if residual:
                inv.add(pid)
                inv.update(residual)
        return inv

    cover = {}  # candidate param id -> set of conflict-expr indices it appears in
    involved = [None] * len(couplings)  # ei -> set of param ids (None if resolved)
    for ei, cmap in enumerate(couplings):
        if not cmap:
            continue
        inv = _involved(cmap)
        if inv:
            involved[ei] = inv
            for pid in inv:
                cover.setdefault(pid, set()).add(ei)

    heap = [(-len(s), _MaxName(_name(pid)), pid) for pid, s in cover.items()]
    heapq.heapify(heap)
    while heap:
        neg, _nm, pid = heapq.heappop(heap)
        s = cover.get(pid)
        if not s or pid in folded or -neg != len(s):
            continue  # empty, already folded, or a stale (superseded) coverage
        if len(s) < hub_min:
            break  # no hub -- leave the residual couplings to be rejected
        folded.add(pid)
        dirty = set()
        for ei in list(s):
            old = involved[ei]
            new = _involved(couplings[ei])
            involved[ei] = new or None
            for q in old - new:
                cover[q].discard(ei)
                dirty.add(q)
        for q in dirty:
            if q not in folded:
                heapq.heappush(heap, (-len(cover[q]), _MaxName(_name(q)), q))
    return folded, names


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

    ``provenance`` carries one ``(constraint, variable, coef_expr)`` per entry so
    a fail-loud message can name the offending coefficient(s).  The component
    ``.name`` is resolved lazily in :meth:`describe` (the fail-loud path only), not
    eagerly per entry at ``set_instance`` -- ``getname`` on indexed components is
    costly, and naming every guarded coefficient up front dominated the compile.
    """

    __slots__ = ('rows', 'cols', 'affine', 'baseline', 'provenance')

    def __init__(self, rows, cols, affine, baseline, provenance):
        self.rows = rows  # int32[N]: A row index per guarded entry
        self.cols = cols  # int32[N]: A col index per guarded entry
        self.affine = affine  # _AffineArray[N]: coefficient values from P
        self.baseline = baseline  # float64[N]: the values HiGHS was loaded with
        self.provenance = provenance  # list[(con, var, coef_expr)]

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
            con, var, coef = self.provenance[int(k)]
            con_name, var_name = con.name, var.name
            params = sorted({p.name for p in identify_mutable_parameters(coef)})
            pnote = f" [Param(s): {', '.join(params)}]" if params else ""
            lines.append(
                f"  - constraint '{con_name}' coefficient on variable "
                f"'{var_name}': {self.baseline[k]!r} -> {current[k]!r}{pnote}"
            )
        if len(idx) > limit:
            lines.append(f"  - ... and {len(idx) - limit} more")
        return "\n".join(lines)


def _build_param_reader(params):
    """Partition a parameter list into a columnar-aware bulk reader.

    Reading the parameter vector is on the warm-tick hot path (once per solve,
    plus the value-guard leg).  A classic ``ParamData`` reads its value from a
    plain slot; a *columnar* ``VectorParamData`` dereferences a weakref to its
    component and indexes the component's value column on every ``.value`` -- so a
    Python ``p.value`` loop over a large switch-ON parameter vector is markedly
    slower than the classic one (measured ~3x per read).

    This gathers every columnar parameter that shares a value column into one
    vectorized slice of that column, keyed once here, and leaves classic
    parameters to the per-object read.  Returns ``None`` when there are no
    columnar parameters, so a classic model keeps the original ``np.fromiter``
    path unchanged.
    """
    scalar = []
    col_by_comp = {}  # id(component) -> [component, dst_positions, src_positions]
    for k, p in enumerate(params):
        pos = getattr(p, '_pos', None)
        comp = p.parent_component() if pos is not None else None
        # A columnar VectorParamData exposes ``_pos`` and its component owns the
        # ``_value_arr`` value column; anything else is read per-object.
        if pos is not None and getattr(comp, '_value_arr', None) is not None:
            entry = col_by_comp.get(id(comp))
            if entry is None:
                entry = [comp, [], []]
                col_by_comp[id(comp)] = entry
            entry[1].append(k)
            entry[2].append(pos)
        else:
            scalar.append((k, p))
    if not col_by_comp:
        return None
    columnar = [
        (comp, np.asarray(dst, dtype=np.intp), np.asarray(src, dtype=np.intp))
        for comp, dst, src in col_by_comp.values()
    ]
    return scalar, columnar


def _gather_param_values(params, reader):
    """Read the current values of ``params`` into a float64 array.

    ``reader`` is ``None`` (all-classic: a simple per-object read) or the
    ``(scalar, columnar)`` partition from :func:`_build_param_reader`.  The
    columnar slice ``comp._value_arr[src]`` returns exactly the float64 values a
    per-object ``VectorParamData.value`` would (the value column is the single
    source of truth), so the result is bit-identical to the classic read.  The
    component's ``_value_arr`` is fetched at read time, so an in-place mutation --
    or a wholesale column replacement -- between ticks is always reflected.
    """
    n = len(params)
    if reader is None:
        return np.fromiter((p.value for p in params), dtype=np.float64, count=n)
    scalar, columnar = reader
    out = np.empty(n, dtype=np.float64)
    for k, p in scalar:
        out[k] = p.value
    for comp, dst, src in columnar:
        out[dst] = comp._value_arr[src]
    return out


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
        'folded_params',
        'folded_baseline',
        # Columnar-aware bulk readers (None => the simple per-Param path); built
        # once from ``params`` / ``folded_params`` at plan construction.
        '_param_reader',
        '_folded_reader',
    )

    def read_param_vector(self):
        return _gather_param_values(self.params, self._param_reader)

    def read_folded_vector(self):
        return _gather_param_values(self.folded_params, self._folded_reader)

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


def _classic_constraint_slots(compiled, resolve_col):
    """Collect the mutable constraint slots from a *classic* (non-templatized)
    compile.

    ``compiled.rows`` is ``[(ConstraintData, multiplier), ...]`` from the proven
    mixed-form path (a range row split into two ``+/-1`` rows that share one
    body).  Returns the seven parallel slot lists the plan tail consumes.  The
    body/slot logic is byte-for-byte the pre-refactor inline loop.
    """
    # id(constraint) -> the A-row index/indices it maps to (a range row splits
    # into two rows that share one body, so both carry the same coefficients).
    con_rows = {}
    for i, (con, _mult) in enumerate(compiled.rows):
        con_rows.setdefault(id(con), []).append(i)

    row_idx = []
    row_lower_slots = []
    row_upper_slots = []
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
                j = resolve_col(v)
                if j is not None and not is_constant(coef):
                    for ri in con_rows[cid]:
                        mat_rows.append(ri)
                        mat_cols.append(j)
                        mat_slots.append(coef)
                        # Store the components, not their names: ``.name`` is
                        # resolved lazily in _MatrixGuard.describe (fail-loud only).
                        mat_prov.append((con, v, coef))
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
    return (
        row_idx,
        row_lower_slots,
        row_upper_slots,
        mat_rows,
        mat_cols,
        mat_slots,
        mat_prov,
    )


def _refs_mutable_param(cd):
    """True if a constraint datum's body or either bound references a mutable
    Param.

    A templatized family's rows are structurally uniform (an index-conditional or
    otherwise non-uniform rule cannot templatize -- it falls back to the classic
    per-row path), so a single materialized row decides the whole family.  Uses
    the generator's first yield and never puts a ParamData in a set (a *columnar*
    ``VectorParamData`` is unhashable).
    """
    for _ in identify_mutable_parameters(cd.body):
        return True
    for bnd in (cd.lower, cd.upper):
        if bnd is not None and bnd.__class__ is not float and bnd.__class__ is not int:
            for _ in identify_mutable_parameters(bnd):
                return True
    return False


def _templated_constraint_slots(compiled, resolve_col):
    """Collect the mutable constraint slots from a *templatized* (switch-ON)
    compile.

    ``compiled.rows`` is ``[(family_or_data, local_row), ...]``: a templatized
    family contributes ``(IndexedConstraint, r)`` rows (there is no per-row
    ``.body`` -- the family carries the template), while a family that could not
    vectorize falls back to per-row ``(ConstraintData, 0)``.

    Two facts keep this both correct and cheap:

    * A templatized family always has a *static* coefficient matrix -- a mutable
      matrix coefficient is index-independent-constant-only on the vectorized
      path, so it forces the family onto the classic fallback.  A family that
      references no mutable Param at all therefore has fully static rows and is
      skipped *without materializing them* (the vectorized construction win
      survives into the warm compile).
    * A materialized row's symbolic ``.body`` / ``.lower`` / ``.upper`` feed the
      exact same slot machinery the classic path uses, and the plan self-check
      then validates every affine template against the compiled numeric arrays --
      so a wrong slot fails loud, never silently.

    Range direction is read from the constraint (``equality`` / open side) rather
    than a mixed-form multiplier: the templatized compile emits HiGHS-native range
    rows (one per constraint, never split), so ``compiled.rows[i][1]`` is a local
    row index, not a ``+/-1`` multiplier.
    """
    row_idx = []
    row_lower_slots = []
    row_upper_slots = []
    mat_rows = []
    mat_cols = []
    mat_slots = []
    mat_prov = []

    def _process_row(i, cd):
        repn = generate_standard_repn(cd.body, quadratic=False, compute_values=False)
        if repn.nonlinear_expr is not None:
            raise IncompatibleModelError(
                f"The '{FastStepHighs.name}' warm interface only supports linear "
                "constraints."
            )
        # A mutable matrix coefficient (only reachable on a classic-fallback row --
        # a templatized family's coefficients are static) is captured into the
        # value guard, exactly as on the classic path.
        for coef, v in zip(repn.linear_coefs, repn.linear_vars):
            j = resolve_col(v)
            if j is not None and not is_constant(coef):
                mat_rows.append(i)
                mat_cols.append(j)
                mat_slots.append(coef)
                mat_prov.append((cd, v, coef))
        offset = repn.constant
        lb = cd.lower
        ub = cd.upper
        if cd.equality:  # lower == upper == rhs
            rhs = ub - offset
            if not is_constant(rhs):
                row_idx.append(i)
                row_lower_slots.append(rhs)
                row_upper_slots.append(rhs)
        elif lb is None:  # A x <= ub ; lower open
            rhs = ub - offset
            if not is_constant(rhs):
                row_idx.append(i)
                row_lower_slots.append(None)
                row_upper_slots.append(rhs)
        elif ub is None:  # A x >= lb ; upper open
            rhs = lb - offset
            if not is_constant(rhs):
                row_idx.append(i)
                row_lower_slots.append(rhs)
                row_upper_slots.append(None)
        else:  # genuine two-sided range: lb <= A x <= ub
            lb_slot = lb - offset
            ub_slot = ub - offset
            lb_mut = not is_constant(lb_slot)
            ub_mut = not is_constant(ub_slot)
            if lb_mut or ub_mut:
                row_idx.append(i)
                row_lower_slots.append(lb_slot if lb_mut else float(value(lb_slot)))
                row_upper_slots.append(ub_slot if ub_mut else float(value(ub_slot)))

    def _materialize(family, index):
        cd = family[index]
        if hasattr(cd, 'template_expr'):
            cd.expr  # expand the stored template into a concrete per-index expression
        return cd

    N = len(compiled.rows)
    i = 0
    while i < N:
        con = compiled.rows[i][0]
        if hasattr(con, 'body'):
            # A non-vectorizable family fell back to a per-row ConstraintData.
            _process_row(i, con)
            i += 1
            continue
        # A templatized family occupies the contiguous run of rows keyed to it;
        # ``compiled.rows[k][1]`` is the local row index into ``con.index_set()``.
        fid = id(con)
        run = []
        while i < N and id(compiled.rows[i][0]) == fid:
            run.append(i)
            i += 1
        index_list = list(con.index_set())
        cd0 = _materialize(con, index_list[compiled.rows[run[0]][1]])
        if _refs_mutable_param(cd0):
            for ii in run:
                cd = _materialize(con, index_list[compiled.rows[ii][1]])
                _process_row(ii, cd)
        # else: a fully static family -- its rows never change across warm rolls,
        # so it contributes no slots and is not materialized past the probe row.
    return (
        row_idx,
        row_lower_slots,
        row_upper_slots,
        mat_rows,
        mat_cols,
        mat_slots,
        mat_prov,
    )


def _build_mutable_plan(model, compiled: FastLoadCompiled):
    """Derive the :class:`_MutablePlan` (parameter registry + affine templates).

    A mutable constraint-matrix coefficient is **not** rejected: it is captured
    into the :class:`_MatrixGuard` (the affine template over ``P`` for every
    affected ``A``-entry, plus the values HiGHS was loaded with), so a warm solve
    can *verify the values* rather than *trust the mutability flag*.  See the
    module docstring's "Value-aware static-matrix guard" section.
    """
    # Column resolver: classic columns are keyed by ``id(VarData)``; a *columnar*
    # Var contributes ``None`` column entries whose columns come from
    # ``column_scatter`` (keyed by ``(id(component), position)``), so a
    # materialized columnar VarData resolves through its ``_pos``.
    col_of = {id(v): j for j, v in enumerate(compiled.columns) if v is not None}
    scatter_of = {}
    for _comp, _solver_cols, _positions in compiled.column_scatter or ():
        _cid = id(_comp)
        for _j, _pos in zip(_solver_cols.tolist(), _positions.tolist()):
            scatter_of[(_cid, _pos)] = _j

    def _resolve_col(v):
        j = col_of.get(id(v))
        if j is not None:
            return j
        pos = getattr(v, '_pos', None)
        if pos is not None:
            return scatter_of.get((id(v.parent_component()), pos))
        return None

    # --- collect the mutable slots (expressions / constants) per group ------ #
    obj_cols = []
    obj_slots = []
    obj_offset_slot = None
    # Mutable Params that feed the (statically-loaded) objective Hessian.  The
    # warm path never pushes Hessian deltas, so every such Param must be a
    # watched constant: it is folded and value-guarded exactly like a static
    # matrix coefficient.  A genuine change to one trips the fold guard (fail-loud,
    # or -- opt-in -- a full re-fold + reload that rebuilds the Hessian).
    quad_param_slots = []
    if compiled.has_objective:
        obj = _active_objective(model)
        if obj is not None:
            repn = generate_standard_repn(
                obj.expr, quadratic=True, compute_values=False
            )
            if repn.nonlinear_expr is not None:
                raise IncompatibleModelError(
                    f"The '{FastStepHighs.name}' warm interface only supports linear "
                    "or convex-quadratic objectives; this objective has "
                    "higher-order nonlinear terms."
                )
            for coef, v in zip(repn.linear_coefs, repn.linear_vars):
                j = _resolve_col(v)
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
            # The Hessian is loaded once and held fixed across warm rolls: any
            # Param feeding a quadratic coefficient must therefore be static.
            for qcoef in getattr(repn, 'quadratic_coefs', ()) or ():
                if not is_constant(qcoef):
                    quad_param_slots.append(qcoef)

    # The constraint slots come from one of two collectors, matching the branch
    # ``compile_to_highs_arrays`` itself took: a templatized (switch-ON) compile
    # carries ``(family, local_row)`` rows whose bodies are not materialized per
    # row, so it needs the template-aware collector; a classic compile carries the
    # mixed-form ``(ConstraintData, multiplier)`` rows.  Both return the identical
    # slot lists, and everything below (folding / registry / affine templates /
    # self-check / value guard) is shared -- the self-check validates whichever
    # collector ran against the compiled numeric arrays.
    from pyomo.contrib.vector.template_vectorize import model_has_templates

    if model_has_templates(model):
        (
            row_idx,
            row_lower_slots,
            row_upper_slots,
            mat_rows,
            mat_cols,
            mat_slots,
            mat_prov,
        ) = _templated_constraint_slots(compiled, _resolve_col)
    else:
        (
            row_idx,
            row_lower_slots,
            row_upper_slots,
            mat_rows,
            mat_cols,
            mat_slots,
            mat_prov,
        ) = _classic_constraint_slots(compiled, _resolve_col)

    col_idx = []
    col_lower_slots = []
    col_upper_slots = []
    for j, v in enumerate(compiled.columns):
        if v is None:
            # A *columnar* Var owns this column: its bounds live in the
            # component's float arrays (no mutable-Param bound is representable
            # there), so the column is static -- nothing to template.
            continue
        lb = v.lower
        ub = v.upper
        lb_mut = lb is not None and not is_constant(lb)
        ub_mut = ub is not None and not is_constant(ub)
        if not lb_mut and not ub_mut:
            continue
        col_idx.append(j)
        col_lower_slots.append(lb if lb_mut else float(compiled.col_lower[j]))
        col_upper_slots.append(ub if ub_mut else float(compiled.col_upper[j]))

    # --- verified-static parameter folding (classification) ----------------- #
    # A mutable parameter that participates *non-affinely* (a product/reciprocal
    # with other quantities -- ``price*duration`` in the objective,
    # ``efficiency*duration`` / ``duration/efficiency`` in the matrix) cannot be
    # templated affinely and would otherwise sink the whole model to a fail-loud
    # rejection.  Classify each parameter: fold the non-affine (structurally
    # constant) ones as watched constants, keep the affinely-participating ones
    # varying.  After folding, ``price*duration`` becomes the affine
    # ``duration_value * price`` over the remaining varying ``price``.  Every
    # folded parameter is watched by the value guard (below): a genuine change to
    # a folded value means the templates are stale, so it fails loud (or, opt-in,
    # reloads) -- never a silent stale solve.
    all_slots = (
        obj_slots
        + ([obj_offset_slot] if obj_offset_slot is not None else [])
        + row_lower_slots
        + row_upper_slots
        + col_lower_slots
        + col_upper_slots
        + mat_slots
    )
    coef_exprs = [
        s
        for s in all_slots
        if s is not None and s.__class__ is not float and s.__class__ is not int
    ]
    # The expensive per-coefficient symbolic walk (``differentiate`` +
    # ``identify_mutable_parameters``), computed exactly once per distinct
    # expression and reused by classification, registration, and template
    # construction (the pre-refactor code walked each expression three times).
    sig_by_id = {}
    for s in coef_exprs:
        sid = id(s)
        if sid not in sig_by_id:
            sig_by_id[sid] = _coef_signature(s)
    folded_ids, _fold_names = _classify_folded([sig_by_id[id(s)] for s in coef_exprs])

    # --- parameter registry (ordered P, varying params only) ---------------- #
    params = []
    param_index = {}
    folded_params = []
    folded_seen = set()

    def _register(slots):
        for s in slots:
            if s is None or s.__class__ is float or s.__class__ is int:
                continue
            for p in sig_by_id[id(s)].all_params:
                pid = id(p)
                if pid in folded_ids:
                    if pid not in folded_seen:
                        folded_seen.add(pid)
                        folded_params.append(p)
                    continue
                if pid not in param_index:
                    param_index[pid] = len(params)
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

    # --- objective Hessian: its Params must be static (watched constants) ---- #
    # The Hessian is loaded once and never pushed as a delta across warm rolls,
    # so every Param feeding a quadratic coefficient is folded and value-guarded
    # (like a static matrix coefficient).  If such a Param also *varies* elsewhere
    # (it is a live template Param), the loaded Hessian would silently go stale
    # between rolls -- reject loudly rather than warm-solve a stale QP.
    for qcoef in quad_param_slots:
        for p in identify_mutable_parameters(qcoef):
            pid = id(p)
            if pid in param_index:
                raise IncompatibleModelError(
                    f"The '{FastStepHighs.name}' warm interface requires a static "
                    f"objective Hessian, but parameter '{p.name}' feeds a quadratic "
                    "objective coefficient and also varies elsewhere in the model. "
                    "A genuinely varying Hessian is not supported on the warm path; "
                    "call set_instance again to reload a changed Hessian."
                )
            if pid not in folded_seen:
                folded_seen.add(pid)
                folded_params.append(p)

    # --- affine templates (folded params baked in as constants) ------------- #
    plan = _MutablePlan()
    plan.params = params
    plan.param_index = param_index
    plan.folded_params = folded_params
    # Columnar-aware bulk readers for the two hot-path reads (a classic model
    # keeps the per-Param path -- these are ``None``).
    plan._param_reader = _build_param_reader(params)
    plan._folded_reader = _build_param_reader(folded_params)
    plan.folded_baseline = plan.read_folded_vector()
    plan.obj_cols = np.fromiter(obj_cols, dtype=np.int32, count=len(obj_cols))
    plan.obj_affine = _build_affine_array(
        obj_slots,
        sig_by_id,
        param_index,
        n_params,
        +1,
        folded_ids,
        'objective coefficient',
    )
    plan.obj_offset_affine = (
        None
        if obj_offset_slot is None
        else _build_affine_array(
            [obj_offset_slot],
            sig_by_id,
            param_index,
            n_params,
            +1,
            folded_ids,
            'objective offset',
        )
    )
    plan.row_idx = np.fromiter(row_idx, dtype=np.int32, count=len(row_idx))
    plan.row_lower_affine = _build_affine_array(
        row_lower_slots,
        sig_by_id,
        param_index,
        n_params,
        -1,
        folded_ids,
        'row lower bound',
    )
    plan.row_upper_affine = _build_affine_array(
        row_upper_slots,
        sig_by_id,
        param_index,
        n_params,
        +1,
        folded_ids,
        'row upper bound',
    )
    plan.col_idx = np.fromiter(col_idx, dtype=np.int32, count=len(col_idx))
    plan.col_lower_affine = _build_affine_array(
        col_lower_slots,
        sig_by_id,
        param_index,
        n_params,
        -1,
        folded_ids,
        'variable lower bound',
    )
    plan.col_upper_affine = _build_affine_array(
        col_upper_slots,
        sig_by_id,
        param_index,
        n_params,
        +1,
        folded_ids,
        'variable upper bound',
    )

    # --- matrix-coefficient template (the value guard) ---------------------- #
    matrix_affine = _build_affine_array(
        mat_slots,
        sig_by_id,
        param_index,
        n_params,
        +1,
        folded_ids,
        'constraint matrix coefficient',
    )
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

    # --- classification transparency ---------------------------------------- #
    if folded_params:
        logger.info(
            "%s: verified-static parameter folding engaged -- folded %d "
            "parameter(s) as watched constants (%s); %d parameter(s) remain "
            "templated (varying).  A genuine change to a folded value trips the "
            "value guard (fail-loud by default; on_matrix_change='reload' "
            "re-folds).",
            FastStepHighs.name,
            len(folded_params),
            _abbrev_names([p.name for p in folded_params]),
            len(params),
        )
    return plan


def _abbrev_names(names, limit=12):
    """A compact, deterministic rendering of a name list for a log/report line."""
    names = list(names)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", ... (+{len(names) - limit} more)"


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

        lp = build_highs_model(compiled)  # HighsModel (with Hessian) for a QP
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

    def _folded_change_error(self, plan, changed, current, array_mode):
        idx = np.nonzero(changed)[0]
        lines = []
        for k in idx[:8]:
            p = plan.folded_params[int(k)]
            lines.append(
                f"  - parameter '{p.name}': {plan.folded_baseline[k]!r} -> "
                f"{current[k]!r}"
            )
        if len(idx) > 8:
            lines.append(f"  - ... and {len(idx) - 8} more")
        details = "\n".join(lines)
        if array_mode:
            remedy = (
                " The array-driven (param_values) path cannot reload from the "
                "model, so a folded-parameter change is always fatal here; drive "
                "the solve from the model (omit param_values) with "
                "on_matrix_change='reload' to re-fold and rebuild instead."
            )
        else:
            remedy = (
                " Set on_matrix_change='reload' to re-fold, rebuild the templates, "
                "and reload the model (basis reset) for this solve instead of "
                "failing."
            )
        return IncompatibleModelError(
            f"The '{self.name}' warm interface detected a genuine change in "
            f"{int(changed.sum())} folded (verified-static) parameter(s) since "
            "set_instance; their set_instance values were substituted as constants "
            f"in the affine templates, so a warm re-solve would use stale "
            f"templates.{remedy}\n{details}"
        )

    def _reload_full(self):
        """Re-run ``set_instance``: re-classify the fold set at the model's
        current values, rebuild every affine template, and load a fresh model
        (basis reset).  Used when a *folded* (verified-static) parameter genuinely
        changed under ``on_matrix_change='reload'`` -- the templates themselves
        must change (a re-fold), so the lighter matrix-only :meth:`_reload_model`
        is not enough."""
        self.set_instance(self._model)

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
        lp = build_highs_model(compiled)  # HighsModel (with Hessian) for a QP
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

        # --- verified-static parameter fold guard: never solve stale templates - #
        # A folded parameter had its set_instance value baked into every template
        # that referenced it (a ``price*duration`` objective coefficient became
        # ``duration_value * price``).  If a folded value genuinely changed, those
        # templates are stale by construction -- verify (vectorized) before
        # touching the solver.  Unchanged -> the fast path; changed -> fail loud,
        # or (opt-in) re-fold + re-template + fresh passModel; never a stale solve.
        if plan.folded_params:
            fcur = plan.read_folded_vector()
            fchanged = np.abs(fcur - plan.folded_baseline) > (
                self._matrix_atol + self._matrix_rtol * np.abs(plan.folded_baseline)
            )
            if fchanged.any():
                array_mode = param_values is not None
                if self._on_matrix_change == 'reload' and not array_mode:
                    # Re-classify folds at the new values, rebuild the templates,
                    # and load a fresh model (basis reset) for this solve.  The
                    # fresh compile bakes in the current data, so the incremental
                    # push below is skipped.
                    self._reload_full()
                    return
                raise self._folded_change_error(plan, fchanged, fcur, array_mode)

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
    # fold classification transparency
    # ------------------------------------------------------------------ #
    @property
    def folded_parameters(self):
        """Names of the mutable parameters that were **folded** -- their
        ``set_instance`` values substituted as constants in the affine templates
        (because they participate non-affinely, e.g. ``price*duration``) and
        watched by the value guard for any subsequent change.  Empty when the
        model is fully affine in its parameters (nothing needed folding)."""
        self._require_loaded()
        return [p.name for p in self._plan.folded_params]

    @property
    def templated_parameters(self):
        """Names of the mutable parameters that remain **templated** (varying) --
        the parameter vector ``P`` the warm update path expands with ``M @ P``."""
        self._require_loaded()
        return [p.name for p in self._plan.params]

    def classification_report(self):
        """A readable dict of the fold classification: which parameters are folded
        (watched constants) vs templated (varying).  Lets a caller see *why* a
        model engaged the warm path and exactly what the value guard is watching.
        """
        self._require_loaded()
        folded = self.folded_parameters
        templated = self.templated_parameters
        return {
            'n_folded': len(folded),
            'n_templated': len(templated),
            'folded_parameters': folded,
            'templated_parameters': templated,
            'folding_engaged': bool(folded),
            'on_folded_change': self._on_matrix_change,
        }

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
            highs,
            self._model,
            compiled.columns,
            compiled.rows,
            # A switch-ON compile owns some columns via *columnar* Vars (a ``None``
            # in ``columns``); the bulk scatter maps their solution back without
            # materializing a per-index VarData.  Omitting it left columnar Vars
            # uninitialized after a warm solve.
            column_scatter=compiled.column_scatter,
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
