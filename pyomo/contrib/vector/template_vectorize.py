# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Template-vectorized construction for classic ``Constraint(index, rule=...)``.

Phase-3 of the vectorized-construction project (scoping doc Sec 6.2 ambition 3,
"your old code gets fast"): take a user's *existing* rule-based constraint family
and, when the rule templatizes, build the whole family's constraint matrix with
NumPy array ops -- never building one expression tree per index.  This is the
Phase-0 Spike-B "full vectorized instantiation" path: templatize the rule once,
extract the row skeleton once, then fill the CSR arrays over the entire index set
with NumPy.  Spike B proved this is 13-16x faster than the classic per-index repn
on the templatizable subset (scalar-affine incl. neighbours, unfiltered
sum-over-set); it also proved that *resolving the template per index* is 5-7x
SLOWER than classic, so this module never does that.

Two independent pieces cooperate:

1. **Construct** -- the opt-in :func:`vectorized_construction` context manager
   (or the ``PYOMO_VECTOR_CONSTRUCT`` environment variable) flips Pyomo's
   experimental ``TEMPLATIZE_CONSTRAINTS`` / ``TEMPLATIZE_OBJECTIVES`` switches so
   that a rule that templatizes is stored as a compact ``TemplateConstraintData``
   (no per-index expression tree), and a rule that does NOT templatize falls back
   to classic per-index construction, byte-identically (scoping doc Sec 6.5).
   Default OFF: with the switch off, stock Pyomo behaviour is untouched.

2. **Compile** -- :func:`compile_templated_to_highs_arrays` assembles the whole
   model's constraint matrix by *vectorized extraction* of every family that
   templatizes (this module) and the stock per-row repn for every family that
   does not, over one shared column space, and hands the result to the Phase-2
   ``highs_fastload`` HiGHS ``passModel`` bulk load.  Construction and load both
   stay array-shaped end-to-end, with no scalarization.

The templatizable subset is deliberately EXACTLY what Spike B proved:

* bodies that are a linear combination of ``coef * var[affine_index...]`` terms
  with numeric-constant coefficients, optionally inside an *unfiltered*
  ``sum(... for j in Set)`` (a ``TemplateSumExpression``);
* multiple distinct ``Var`` components (e.g. ``x[f, c] <= open[f]``);
* right-hand sides / bounds that are constants or mutable-``Param`` look-ups
  ``p[affine_index...]`` (evaluated vectorially);
* equality, one-sided inequality, and ranged relations.

Anything outside that -- index conditionals (``if i == 0``), filtered sums
(``for j in J if j != n``), modulo / non-affine indexing, or index-dependent
coefficients (``a[i, j] * x[j]``) -- raises :class:`NotVectorizable` and the
caller falls back to the classic per-row path for that family.  This partial
coverage with a mandatory scalarization fallback is the design Spike B mandated.
"""

from __future__ import annotations

import logging
import os

from pyomo.common.log import is_debug_set
from pyomo.common.dependencies import numpy as np, scipy
from pyomo.common.enums import ObjectiveSense

from pyomo.core.base import Var, Constraint, Objective
from pyomo.core.base.constraint import TemplateConstraintData, TemplateScalarConstraint
from pyomo.core.expr import numeric_expr as ne
from pyomo.core.expr import relational_expr as rel
from pyomo.core.expr.template_expr import (
    templatize_constraint,
    suppress_templatization_errors,
    IndexTemplate,
    GetItemExpression,
    TemplateSumExpression,
)

logger = logging.getLogger('pyomo.contrib.vector')

_inf = float('inf')
_ninf = -_inf


class NotVectorizable(Exception):
    """A templatized family cannot be extracted on the vectorized fast path.

    The caller falls back to the classic per-row repn for this family (the body
    uses a construct outside the Spike-B-proven subset: an index-dependent
    coefficient, a constant summed over a set, a non-affine index, etc.).
    """


# --------------------------------------------------------------------------- #
# Activation (opt-in; default OFF)
# --------------------------------------------------------------------------- #
def _set_templatize(flag):
    """Enable/disable the core ``TEMPLATIZE_CONSTRAINTS`` switch.

    Returns the prior ``(constraint_flag, objective_flag)`` so it can be
    restored.  Phase 3 templatizes **constraints** only -- the "your old code
    gets fast" milestone is about ``Constraint(index, rule=...)`` families.  We
    deliberately leave ``TEMPLATIZE_OBJECTIVES`` off: a scalar ``Objective(expr=
    sum(...))`` (the common case) templatizes trivially but then compiles through
    a large per-term code-generated evaluator that is *slower* than the classic
    objective walk, so enabling it would regress otherwise-untouched models.
    """
    import pyomo.core.base.constraint as _con
    import pyomo.core.base.objective as _obj

    prior = (_con.TEMPLATIZE_CONSTRAINTS, _obj.TEMPLATIZE_OBJECTIVES)
    _con.TEMPLATIZE_CONSTRAINTS = flag
    return prior


def _restore_templatize(prior):
    import pyomo.core.base.constraint as _con
    import pyomo.core.base.objective as _obj

    _con.TEMPLATIZE_CONSTRAINTS, _obj.TEMPLATIZE_OBJECTIVES = prior


class vectorized_construction:
    """Context manager enabling template-vectorized constraint construction.

    While the block is active, every ``Constraint(index, rule=...)`` (and
    rule-based ``Objective``) constructed on any model attempts templatization
    once; a rule that templatizes is stored as a compact template (no per-index
    expression tree), and one that does not falls back to classic construction.
    No user model code changes -- wrap the model build::

        with vectorized_construction():
            m = build_model(...)

    The switch is process-global for the duration of the block (Pyomo constructs
    components eagerly as they are assigned to the model), and is restored on
    exit, so it composes safely and never leaks into stock behaviour.

    Known limitation: a model built with the switch on cannot currently be
    ``clone()``d -- deep-copying a templatized ``TemplateSumExpression`` recurses
    (a pre-existing limitation of Pyomo's experimental template-expression
    feature, unrelated to this fast path).  The Phase-3 construct -> compile ->
    ``highs_fastload`` solve route never clones, so this does not affect the fast
    path; a workflow that clones the model should leave the switch off.
    """

    __slots__ = ('_enabled', '_prior')

    def __init__(self, enabled=True):
        self._enabled = enabled
        self._prior = None

    def __enter__(self):
        self._prior = _set_templatize(bool(self._enabled))
        return self

    def __exit__(self, *exc):
        _restore_templatize(self._prior)
        return False


def templatize_enabled_by_env():
    """True if ``PYOMO_VECTOR_CONSTRUCT`` requests template construction.

    Lets a benchmark / deployment turn the fast path on process-wide without any
    code change (``PYOMO_VECTOR_CONSTRUCT=1``).  Recognised truthy values:
    ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    """
    return os.environ.get('PYOMO_VECTOR_CONSTRUCT', '').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def apply_env_templatize():
    """If ``PYOMO_VECTOR_CONSTRUCT`` is set truthy, enable the switch globally.

    Called at ``pyomo.contrib.vector`` import so that setting the environment
    variable is sufficient to activate the fast path for the whole process.
    Returns True if it enabled the switch.
    """
    if templatize_enabled_by_env():
        _set_templatize(True)
        return True
    return False


# --------------------------------------------------------------------------- #
# Column mapping: (index tuple) -> column position, vectorized
# --------------------------------------------------------------------------- #
def _contiguous_int_rangeset(s):
    """Return ``(first, size)`` if ``s`` is a contiguous integer set, else None."""
    try:
        if not s.isfinite():
            return None
        n = len(s)
        first = s.first()
        last = s.last()
    except Exception:
        return None
    if not isinstance(first, int) or not isinstance(last, int):
        return None
    if last - first + 1 != n:
        return None  # not contiguous / step != 1
    return first, n


class _ColumnMapper:
    """Map a :class:`Var` component's index tuples to column positions (0..n-1).

    Column order is the component's own ``index_set()`` iteration order, which is
    the canonical order the standard-form column space uses.  For a variable
    indexed by a (product of) contiguous integer ``RangeSet``\\ s the mapping is a
    closed-form stride computation done vectorially; otherwise a one-time
    ``{index: position}`` dict is built (always correct, O(n) once).
    """

    __slots__ = ('var', 'n', '_mode', '_firsts', '_strides', '_sizes', '_pos')

    def __init__(self, var):
        self.var = var
        self.n = len(var)
        self._pos = None
        subsets = list(var.index_set().subsets())
        dims = [_contiguous_int_rangeset(s) for s in subsets]
        if all(d is not None for d in dims) and dims:
            sizes = [d[1] for d in dims]
            prod = 1
            for sz in sizes:
                prod *= sz
            if prod == self.n:
                self._mode = 'stride'
                self._firsts = [d[0] for d in dims]
                self._sizes = sizes
                strides = [1] * len(sizes)
                for k in range(len(sizes) - 2, -1, -1):
                    strides[k] = strides[k + 1] * sizes[k + 1]
                self._strides = strides
                return
        # dict fallback
        self._mode = 'dict'
        self._pos = {idx: p for p, idx in enumerate(var.index_set())}

    def pos(self, index):
        """Scalar: one index (tuple or scalar) -> column position (int).

        The fast per-term path for classic (non-vectorized) families; avoids the
        NumPy per-element overhead of :meth:`map` when mapping one variable at a
        time.
        """
        if self._mode == 'stride':
            if index.__class__ is not tuple:
                index = (index,)
            p = 0
            for idx, first, stride, size in zip(
                index, self._firsts, self._strides, self._sizes
            ):
                local = idx - first
                if local < 0 or local >= size:
                    raise KeyError(index)
                p += local * stride
            return p
        return self._pos[index]

    def map(self, dim_arrays):
        """Vectorized: list of per-dim int arrays (length m each) -> int col array."""
        if self._mode == 'stride':
            m = dim_arrays[0].shape[0]
            pos = np.zeros(m, dtype=np.int64)
            for arr, first, stride, size in zip(
                dim_arrays, self._firsts, self._strides, self._sizes
            ):
                local = arr - first
                if local.min() < 0 or local.max() >= size:
                    raise NotVectorizable(
                        f"index out of range for Var '{self.var.name}'"
                    )
                pos += local * stride
            return pos
        # dict path
        if len(dim_arrays) == 1:
            keys = dim_arrays[0].tolist()
        else:
            keys = list(zip(*(a.tolist() for a in dim_arrays)))
        pos = self._pos
        try:
            return np.fromiter((pos[k] for k in keys), dtype=np.int64, count=len(keys))
        except KeyError as e:
            raise NotVectorizable(
                f"index {e} not present in Var '{self.var.name}'"
            ) from e


# --------------------------------------------------------------------------- #
# Affine index-expression evaluation over an array of template values
# --------------------------------------------------------------------------- #
def _eval_affine(node, axis_vals):
    """Evaluate an affine index expression to an int array (or None if constant).

    ``axis_vals`` maps ``id(IndexTemplate) -> int array``.  Supports the affine
    index arithmetic templates preserve symbolically: the leaf template, integer
    constants, and +, -, * of those (``_1``, ``_1 - 1``, ``2*_1`` ...).  Returns
    ``None`` for a purely constant sub-expression (the caller broadcasts it).
    Raises :class:`NotVectorizable` on anything non-affine.
    """
    if isinstance(node, IndexTemplate):
        try:
            return axis_vals[id(node)]
        except KeyError:
            raise NotVectorizable("index references an unexpected template")
    if node.__class__ in (int, float):
        return None
    if isinstance(node, ne.NegationExpression):
        v = _eval_affine(node.args[0], axis_vals)
        return None if v is None else -v
    if isinstance(node, (ne.SumExpression, ne.LinearExpression, ne.NPV_SumExpression)):
        acc = None
        const = 0
        for a in node.args:
            v = _eval_affine(a, axis_vals)
            if v is None:
                const += _as_int_const(a)
            else:
                acc = v if acc is None else acc + v
        if acc is None:
            return None
        return acc + const if const else acc
    if isinstance(
        node,
        (ne.MonomialTermExpression, ne.ProductExpression, ne.NPV_ProductExpression),
    ):
        a, b = node.args
        va = _eval_affine(a, axis_vals)
        vb = _eval_affine(b, axis_vals)
        if va is None and vb is None:
            return None
        if va is None:
            return _as_int_const(a) * vb
        if vb is None:
            return va * _as_int_const(b)
        raise NotVectorizable("non-affine (variable * variable) index")
    # Any other node must be a numeric constant.
    try:
        _as_int_const(node)
        return None
    except Exception as e:
        raise NotVectorizable(f"non-affine index node {type(node).__name__}") from e


def _as_int_const(node):
    from pyomo.core.expr.numvalue import value as _value

    return int(_value(node))


def _resolve_index(node, axis_vals, m):
    """Return an int array of length ``m`` for one affine index expression."""
    v = _eval_affine(node, axis_vals)
    if v is None:
        return np.full(m, _as_int_const(node), dtype=np.int64)
    if np.isscalar(v):
        return np.full(m, int(v), dtype=np.int64)
    return np.asarray(v, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Family extraction
# --------------------------------------------------------------------------- #
def _get_template_info(con):
    """Return ``(relational_expr, index_templates)`` for a constraint family.

    Reuses the info stored by template construction when present (no re-work);
    otherwise templatizes the rule once here.  Raises whatever
    ``templatize_constraint`` raises for a non-templatizable rule -- the caller
    treats that as "use the classic path".
    """
    # A family constructed under TEMPLATIZE_CONSTRAINTS stores template_info on
    # each datum; reuse it to avoid re-running the rule in template mode.
    try:
        first = next(iter(con.values()))
    except StopIteration:
        first = None
    if first is not None and hasattr(first, 'template_expr'):
        info = first.template_expr()
        if info is not None:
            return info
    with suppress_templatization_errors():
        return templatize_constraint(con)


class _FamilyExtractor:
    """Vectorized CSR extraction for one templatized linear constraint family."""

    def __init__(self, con, col_offset, mappers):
        self.con = con
        self.col_offset = col_offset
        self.mappers = mappers
        self.rows = []
        self.cols = []
        self.data = []

    def _grid_axes(self, row_axis, row_ids, local_iters):
        """Build the row x (cartesian product of local sets) grid.

        Returns ``(m, row_index_of_entry, axis_vals)`` where ``axis_vals`` maps
        every template id (row + local) to its value array over the grid.
        """
        if not local_iters:
            return len(row_ids), row_ids, row_axis
        set_arrays = []
        tmpl_groups = []
        for tmpl_group, _set in local_iters:
            set_arrays.append(np.fromiter(iter(_set), dtype=np.int64))
            tmpl_groups.append(tmpl_group)
        mesh = np.meshgrid(row_ids, *set_arrays, indexing='ij')
        row_of_entry = mesh[0].ravel()
        axis_vals = {tid: arr[row_of_entry] for tid, arr in row_axis.items()}
        for gi, tmpl_group in enumerate(tmpl_groups):
            level = mesh[gi + 1].ravel()
            for t in tmpl_group:
                axis_vals[id(t)] = level
        return row_of_entry.shape[0], row_of_entry, axis_vals

    def add_var_term(self, coef, getitem, sign, row_axis, row_ids, local_iters):
        var = getitem.args[0]
        idx_exprs = getitem.args[1:]
        m, row_of_entry, axis_vals = self._grid_axes(row_axis, row_ids, local_iters)
        dim_arrays = [_resolve_index(ie, axis_vals, m) for ie in idx_exprs]
        mapper = self.mappers[id(var)]
        cols = mapper.map(dim_arrays) + self.col_offset[id(var)]
        self.rows.append(row_of_entry)
        self.cols.append(cols)
        self.data.append(np.full(m, float(coef) * sign))

    def walk(self, node, sign, row_axis, row_ids, local_iters):
        """Append variable terms; return a per-row (or scalar) constant array.

        A variable term becomes matrix entries; a constant / mutable-Param term
        becomes a right-hand-side contribution (moved to the row bounds).
        """
        if isinstance(node, GetItemExpression):
            comp = node.args[0]
            if comp.ctype is Var:
                self.add_var_term(1.0, node, sign, row_axis, row_ids, local_iters)
                return 0.0
            # Param (or other data component) look-up -> constant contribution.
            if local_iters:
                raise NotVectorizable("data look-up summed over a set")
            return sign * _eval_param_getitem(node, row_axis, len(row_ids))
        if isinstance(node, TemplateSumExpression):
            targs = node.template_args()
            summand = targs[0]
            sets = targs[1:]
            iters = list(zip(node.template_iters(), sets))
            return self.walk(summand, sign, row_axis, row_ids, local_iters + iters)
        if isinstance(node, ne.NegationExpression):
            return self.walk(node.args[0], -sign, row_axis, row_ids, local_iters)
        if isinstance(node, ne.MonomialTermExpression):
            coef, var_getitem = node.args
            if not isinstance(var_getitem, GetItemExpression):
                raise NotVectorizable("monomial term is not a templated variable")
            self.add_var_term(
                _const_coef(coef), var_getitem, sign, row_axis, row_ids, local_iters
            )
            return 0.0
        if isinstance(node, (ne.ProductExpression, ne.NPV_ProductExpression)):
            a, b = node.args
            a_is_var = isinstance(a, GetItemExpression) and a.args[0].ctype is Var
            b_is_var = isinstance(b, GetItemExpression) and b.args[0].ctype is Var
            if a_is_var and b_is_var:
                raise NotVectorizable("product of two variables (nonlinear)")
            if b_is_var:
                self.add_var_term(
                    _const_coef(a), b, sign, row_axis, row_ids, local_iters
                )
                return 0.0
            if a_is_var:
                self.add_var_term(
                    _const_coef(b), a, sign, row_axis, row_ids, local_iters
                )
                return 0.0
            # constant * constant
            if local_iters:
                raise NotVectorizable("constant product summed over a set")
            return sign * _eval_const_node(node, len(row_ids))
        if isinstance(
            node, (ne.SumExpression, ne.LinearExpression, ne.NPV_SumExpression)
        ):
            const = 0.0
            for a in node.args:
                const = const + self.walk(a, sign, row_axis, row_ids, local_iters)
            return const
        # A bare numeric constant / Param scalar.
        if local_iters:
            raise NotVectorizable("constant summed over a set")
        return sign * _eval_const_node(node, len(row_ids))


def _const_coef(node):
    """A coefficient must be a numeric constant (index-independent) for this path."""
    from pyomo.core.expr.numvalue import is_constant, value as _value

    if node.__class__ in (int, float):
        return float(node)
    if isinstance(node, IndexTemplate) or _mentions_template(node):
        raise NotVectorizable("index-dependent coefficient")
    try:
        if not is_constant(node):
            raise NotVectorizable("non-constant coefficient")
        return float(_value(node))
    except NotVectorizable:
        raise
    except Exception as e:
        raise NotVectorizable("uncomputable coefficient") from e


def _mentions_template(node):
    """True if the expression tree references any IndexTemplate."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, IndexTemplate):
            return True
        if isinstance(n, GetItemExpression):
            # its index args may contain templates
            stack.extend(n.args[1:])
            continue
        if hasattr(n, 'args') and not n.__class__ in (int, float, str):
            try:
                stack.extend(n.args)
            except (AttributeError, TypeError):
                pass
    return False


def _eval_const_node(node, n):
    """Evaluate an index-independent constant node, broadcast to length ``n``."""
    from pyomo.core.expr.numvalue import value as _value

    if node.__class__ in (int, float):
        return float(node)
    if _mentions_template(node):
        raise NotVectorizable("index-dependent constant term")
    try:
        return float(_value(node))
    except Exception as e:
        raise NotVectorizable("uncomputable constant term") from e


def _eval_param_getitem(node, row_axis, n):
    """Evaluate ``p[affine_index...]`` (a mutable Param look-up) over the rows.

    Returns a float array of length ``n`` (per-row RHS/bound contribution).
    """
    comp = node.args[0]
    idx_exprs = node.args[1:]
    dim_arrays = [_resolve_index(ie, row_axis, n) for ie in idx_exprs]
    if len(dim_arrays) == 1:
        keys = dim_arrays[0].tolist()
    else:
        keys = list(zip(*(a.tolist() for a in dim_arrays)))
    out = np.empty(n, dtype=np.float64)
    from pyomo.core.expr.numvalue import value as _value

    try:
        for i, k in enumerate(keys):
            out[i] = _value(comp[k])
    except Exception as e:
        raise NotVectorizable(f"could not evaluate look-up '{comp.name}': {e}") from e
    return out


def _as_row_array(val, n):
    if np.isscalar(val):
        return np.full(n, float(val), dtype=np.float64)
    return np.asarray(val, dtype=np.float64)


def extract_family(con, col_offset, mappers, template_info=None):
    """Extract ``(rows, cols, data, row_lb, row_ub, nrows)`` for one family.

    ``col_offset`` maps ``id(var component) -> global column offset``; ``mappers``
    maps ``id(var component) -> _ColumnMapper``.  Raises :class:`NotVectorizable`
    if the family's body is outside the proven subset.  The CSR triplet is over
    the global column space; ``row_lb``/``row_ub`` are float arrays (``+/- inf``
    on an open side).  Rows are in ``con.index_set()`` order.
    """
    if template_info is None:
        template_info = _get_template_info(con)
    expr, row_tmpls = template_info

    if not row_tmpls:
        # A scalar (unindexed) constraint: one row, nothing to vectorize over --
        # let the classic per-row path handle it (it is a single expression).
        raise NotVectorizable("scalar (unindexed) constraint")

    index_list = list(con.index_set())
    n = len(index_list)
    if n == 0:
        return (
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            0,
        )
    row_vals = np.array(
        [k if isinstance(k, tuple) else (k,) for k in index_list], dtype=np.int64
    )
    row_axis = {id(t): row_vals[:, k] for k, t in enumerate(row_tmpls)}
    row_ids = np.arange(n, dtype=np.int64)

    ex = _FamilyExtractor(con, col_offset, mappers)

    if isinstance(expr, rel.EqualityExpression):
        lhs, rhs = expr.args
        c_lhs = _as_row_array(ex.walk(lhs, 1.0, row_axis, row_ids, []), n)
        c_rhs = _as_row_array(ex.walk(rhs, -1.0, row_axis, row_ids, []), n)
        rhs_val = -(c_lhs + c_rhs)
        row_lb = rhs_val.copy()
        row_ub = rhs_val.copy()
    elif isinstance(expr, rel.InequalityExpression):
        lhs, rhs = expr.args
        c_lhs = _as_row_array(ex.walk(lhs, 1.0, row_axis, row_ids, []), n)
        c_rhs = _as_row_array(ex.walk(rhs, -1.0, row_axis, row_ids, []), n)
        # lhs <= rhs  ->  (lhs_vars - rhs_vars) <= (rhs_const - lhs_const)
        row_ub = -(c_lhs + c_rhs)
        row_lb = np.full(n, _ninf)
    elif isinstance(expr, rel.RangedExpression):
        lb_node, body, ub_node = expr.args
        c_body = _as_row_array(ex.walk(body, 1.0, row_axis, row_ids, []), n)
        lb_arr = _as_row_array(_eval_bound(lb_node, row_axis, n), n)
        ub_arr = _as_row_array(_eval_bound(ub_node, row_axis, n), n)
        row_lb = lb_arr - c_body
        row_ub = ub_arr - c_body
    else:
        raise NotVectorizable(f"unsupported relation {type(expr).__name__}")

    if ex.rows:
        rows = np.concatenate(ex.rows)
        cols = np.concatenate(ex.cols)
        data = np.concatenate(ex.data)
    else:
        rows = np.zeros(0, np.int64)
        cols = np.zeros(0, np.int64)
        data = np.zeros(0)
    return rows, cols, data, row_lb, row_ub, n


def _eval_bound(node, row_axis, n):
    """Evaluate a ranged-constraint bound node (constant or Param look-up)."""
    if isinstance(node, GetItemExpression) and node.args[0].ctype is not Var:
        return _eval_param_getitem(node, row_axis, n)
    return _eval_const_node(node, n)


# --------------------------------------------------------------------------- #
# Objective extraction
# --------------------------------------------------------------------------- #
def extract_objective(obj_data, c, col_offset, mappers, template_info):
    """Add a templatized scalar objective's cost into ``c``; return the offset.

    ``c`` is a dense float array over the global column space (modified in
    place).  Raises :class:`NotVectorizable` if the body is outside the proven
    subset.
    """
    body, _idx_tmpls = template_info
    ex = _FamilyExtractor(obj_data, col_offset, mappers)
    const = _as_row_array(ex.walk(body, 1.0, {}, np.zeros(1, dtype=np.int64), []), 1)
    for cols, dd in zip(ex.cols, ex.data):
        np.add.at(c, cols, dd)
    return float(const[0])


# --------------------------------------------------------------------------- #
# Whole-model assembly -> HiGHS-ready arrays (feeds highs_fastload passModel)
# --------------------------------------------------------------------------- #
def model_has_templates(model):
    """True if any active constraint family was built as a template family."""
    for con in model.component_objects(Constraint, active=True, descend_into=True):
        try:
            first = next(iter(con.values()))
        except StopIteration:
            continue
        if isinstance(first, (TemplateConstraintData, TemplateScalarConstraint)):
            return True
    return False


def compile_templated_to_highs_arrays(model):
    """Assemble ``model`` to HiGHS range-row arrays, vectorizing where possible.

    Every constraint family that templatizes is extracted with NumPy (this
    module); every family that does not is compiled with the stock per-row linear
    repn -- both over one shared column space built from the model's ``Var``
    components.  Returns a :class:`~pyomo.contrib.vector.fastload.FastLoadCompiled`
    so the Phase-2 ``highs_fastload`` solver can hand it to HiGHS via
    ``passModel`` unchanged.
    """
    from pyomo.contrib.vector.fastload import FastLoadCompiled

    # --- global column space over all Var components ---------------------- #
    var_comps = list(model.component_objects(Var, active=True, descend_into=True))
    col_offset = {}
    mappers = {}
    off = 0
    for v in var_comps:
        col_offset[id(v)] = off
        mappers[id(v)] = _ColumnMapper(v)
        off += len(v)
    n_var = off

    col_lower = np.full(n_var, _ninf)
    col_upper = np.full(n_var, _inf)
    integrality = np.zeros(n_var, dtype=bool)
    columns = [None] * n_var  # VarData, in column order (solution map-back)
    for v in var_comps:
        base = col_offset[id(v)]
        lo, hi, integ, vardata = _var_column_data(v)
        col_lower[base : base + len(v)] = lo
        col_upper[base : base + len(v)] = hi
        integrality[base : base + len(v)] = integ
        columns[base : base + len(v)] = vardata

    # --- objective -------------------------------------------------------- #
    c = np.zeros(n_var, dtype=np.float64)
    c_offset = 0.0
    sense = ObjectiveSense.minimize
    has_objective = False
    objs = list(model.component_objects(Objective, active=True, descend_into=True))
    if len(objs) > 1:
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        raise IncompatibleModelError(
            "template-vectorized fast load supports at most one objective "
            f"(received {len(objs)})."
        )
    if objs:
        obj = objs[0]
        has_objective = True
        sense = ObjectiveSense(obj.sense)
        obj_data = next(iter(obj.values())) if len(obj) else obj
        c, c_offset = _objective_cost(obj, obj_data, c, col_offset, mappers)

    # --- constraints ------------------------------------------------------ #
    data_parts, indices_parts = [], []
    indptr = [0]
    row_lb_parts, row_ub_parts = [], []
    rows_meta = []  # (ConstraintData, local_row) solution map-back
    for con in model.component_objects(Constraint, active=True, descend_into=True):
        if not con.active:
            continue
        A_fam, lb_fam, ub_fam, meta = _extract_or_classic(
            con, col_offset, mappers, n_var
        )
        if A_fam.shape[0] == 0:
            continue
        data_parts.append(A_fam.data)
        indices_parts.append(A_fam.indices)
        indptr.extend((A_fam.indptr[1:] + indptr[-1]).tolist())
        row_lb_parts.append(lb_fam)
        row_ub_parts.append(ub_fam)
        rows_meta.extend(meta)

    n_row = len(indptr) - 1
    if data_parts:
        A = scipy.sparse.csr_array(
            (
                np.concatenate(data_parts),
                np.concatenate(indices_parts),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(n_row, n_var),
        )
        row_lower = np.concatenate(row_lb_parts)
        row_upper = np.concatenate(row_ub_parts)
    else:
        A = scipy.sparse.csr_array((0, n_var))
        row_lower = np.zeros(0)
        row_upper = np.zeros(0)

    # --- fixed-variable substitution + unused-column elimination ---------- #
    (
        A,
        row_lower,
        row_upper,
        col_lower,
        col_upper,
        integrality,
        c,
        c_offset,
        columns,
    ) = _finalize_columns(
        A,
        row_lower,
        row_upper,
        col_lower,
        col_upper,
        integrality,
        c,
        c_offset,
        columns,
        var_comps,
        col_offset,
    )

    return FastLoadCompiled(
        A.tocsc(),
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
        rows_meta,
    )


def _var_column_data(v):
    """Return ``(lb, ub, integrality, vardata_list)`` arrays for a Var component.

    Reads bounds/domain from the (already-constructed classic) per-index VarData
    in the component's own index order -- these objects exist for a classic Var,
    so this materializes nothing new.
    """
    n = len(v)
    lo = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.float64)
    integ = np.zeros(n, dtype=bool)
    vardata = [None] * n
    for pos, idx in enumerate(v.index_set()):
        vd = v[idx]
        b_lo, b_hi = vd.bounds
        lo[pos] = _ninf if b_lo is None else float(b_lo)
        hi[pos] = _inf if b_hi is None else float(b_hi)
        if not vd.is_continuous():
            integ[pos] = True
        vardata[pos] = vd
    return lo, hi, integ, vardata


def _objective_cost(obj, obj_data, c, col_offset, mappers):
    """Fill objective cost vector ``c`` (vectorized if templatized, else classic)."""
    info = None
    if hasattr(obj_data, 'template_expr'):
        info = obj_data.template_expr()
    if info is not None:
        try:
            offset = extract_objective(obj_data, c, col_offset, mappers, info)
            return c, offset
        except NotVectorizable:
            c[:] = 0.0  # discard partial fill; redo classically
    # classic: walk the objective expression once
    from pyomo.repn.standard_repn import generate_standard_repn

    repn = generate_standard_repn(obj_data.expr, quadratic=False)
    if repn.nonlinear_expr is not None:
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        raise IncompatibleModelError(f"objective '{obj.name}' is not linear.")
    for coef, var in zip(repn.linear_coefs, repn.linear_vars):
        vc = var.parent_component()
        col = col_offset[id(vc)] + mappers[id(vc)].pos(var.index())
        c[col] += float(coef)
    return c, float(repn.constant)


def _extract_or_classic(con, col_offset, mappers, n_var):
    """Return ``(A_family_csr, lb, ub, meta)`` for one constraint family.

    Tries the vectorized extractor; on :class:`NotVectorizable` (or any failure
    to templatize the rule) falls back to the stock per-row linear repn.  ``meta``
    is a list of ``(ConstraintData, local_row)`` for solution map-back.
    """
    if len(con) == 0:
        return scipy.sparse.csr_array((0, n_var)), np.zeros(0), np.zeros(0), []
    try:
        info = _get_template_info(con)
        rows, cols, data, lb, ub, nr = extract_family(con, col_offset, mappers, info)
    except NotVectorizable:
        return _classic_family(con, col_offset, mappers, n_var)
    except Exception as e:
        # The rule did not templatize (index conditional / filtered sum /
        # modulo / ...): use the proven classic path for this family.
        if is_debug_set(logger):
            logger.debug(
                "family '%s' not vectorizable (%s: %s); using classic per-row repn"
                % (con.name, type(e).__name__, e)
            )
        return _classic_family(con, col_offset, mappers, n_var)
    A = scipy.sparse.coo_array((data, (rows, cols)), shape=(nr, n_var)).tocsr()
    A.sum_duplicates()
    meta = [(con, r) for r in range(nr)]
    return A, lb, ub, meta


def _classic_family(con, col_offset, mappers, n_var):
    from pyomo.repn.standard_repn import generate_standard_repn

    rows_l, cols_l, data_l = [], [], []
    lb_l, ub_l = [], []
    meta = []
    r = 0
    for idx in con:
        cd = con[idx]
        if not cd.active:
            continue
        if hasattr(cd, 'template_expr'):
            # A family that templatized but is outside the vectorizable subset
            # (or a scalar template constraint): materialize each row to a classic
            # ConstraintData first (its raw ``to_bounded_expression`` would
            # otherwise return the template).
            cd.expr
        lb, body, ub = cd.to_bounded_expression(evaluate_bounds=True)
        repn = generate_standard_repn(body, quadratic=False)
        if repn.nonlinear_expr is not None:
            from pyomo.contrib.solver.common.util import IncompatibleModelError

            raise IncompatibleModelError(f"constraint '{cd.name}' is not linear.")
        const = repn.constant
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            vc = var.parent_component()
            col = col_offset[id(vc)] + mappers[id(vc)].pos(var.index())
            rows_l.append(r)
            cols_l.append(col)
            data_l.append(float(coef))
        lb_l.append(_ninf if lb is None else float(lb) - float(const))
        ub_l.append(_inf if ub is None else float(ub) - float(const))
        meta.append((cd, 0))
        r += 1
    A = scipy.sparse.csr_array(
        scipy.sparse.coo_array(
            (
                np.asarray(data_l, dtype=np.float64),
                (
                    np.asarray(rows_l, dtype=np.int64),
                    np.asarray(cols_l, dtype=np.int64),
                ),
            ),
            shape=(r, n_var),
        )
    )
    A.sum_duplicates()
    return A, np.asarray(lb_l), np.asarray(ub_l), meta


def _finalize_columns(
    A,
    row_lower,
    row_upper,
    col_lower,
    col_upper,
    integrality,
    c,
    c_offset,
    columns,
    var_comps,
    col_offset,
):
    """Fixed-variable substitution + drop columns that appear nowhere.

    Mirrors the stock ``LinearStandardFormCompiler``: a fixed variable's
    contribution moves into the row bounds / objective offset (the #3851
    pitfall), and columns that appear in neither ``A`` nor the objective are
    eliminated.  Returns the trimmed arrays and the surviving ``columns`` list.
    """
    Acsc = A.tocsc()
    n_var = Acsc.shape[1]
    col_nnz = np.diff(Acsc.indptr)
    appears = (col_nnz > 0) | (c != 0.0)

    fixed = np.zeros(n_var, dtype=bool)
    fixed_val = np.zeros(n_var, dtype=np.float64)
    for vd, j in _iter_columns(columns):
        if vd is not None and vd.fixed:
            fixed[j] = True
            fixed_val[j] = 0.0 if vd.value is None else float(vd.value)

    fixed_appears = fixed & appears
    if fixed_appears.any():
        contrib = np.asarray(Acsc[:, fixed_appears] @ fixed_val[fixed_appears]).ravel()
        row_lower = np.where(np.isfinite(row_lower), row_lower - contrib, row_lower)
        row_upper = np.where(np.isfinite(row_upper), row_upper - contrib, row_upper)
        c_offset += float(c[fixed_appears] @ fixed_val[fixed_appears])

    keep = appears & ~fixed
    keep_cols = np.nonzero(keep)[0]
    A_keep = Acsc[:, keep_cols].tocsr()
    return (
        A_keep,
        row_lower,
        row_upper,
        col_lower[keep_cols],
        col_upper[keep_cols],
        integrality[keep_cols],
        c[keep_cols],
        c_offset,
        [columns[j] for j in keep_cols],
    )


def _iter_columns(columns):
    for j, vd in enumerate(columns):
        yield vd, j
