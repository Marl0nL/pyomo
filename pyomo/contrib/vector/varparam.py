# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Transparent columnar ``Var`` / ``Param`` construction under the Phase-3 switch.

Phase 3 (``template_vectorize``) made a classic ``Constraint(index, rule=...)``
family construct array-shaped; the per-index ``VarData`` / ``ParamData`` objects
stayed classic and, once the constraint families are vectorized, became the
dominant share of the remaining cold construct (profiled: on the templatizable
``resource_coupling`` model at 1e6 the switch-on construct is ~62 ms, of which
``Param`` construction is ~69% and ``Var`` ~7%; on a variable-heavy model where
``nnz ~ n_var`` the two together are ~76%).

This module closes that residual.  While the Phase-3 switch is active
(:func:`~pyomo.contrib.vector.template_vectorize.vectorized_construction` or the
``PYOMO_VECTOR_CONSTRUCT`` env var), a plain

    m.x = Var(index, bounds=..., domain=..., initialize=...)
    m.p = Param(index, initialize=...)          # mutable or immutable

whose arguments are *vectorizable* (scalars / mappings, not a genuinely
per-index callable) is constructed into NumPy columns -- reusing Phase-1's
:class:`~pyomo.contrib.vector.var._ColumnarVarMixin` (materialize-on-touch, the
identity contract, lazy scalarization) -- instead of one Python data object per
index.  A component whose arguments do *not* vectorize (a per-index callable
bound / initializer, a validation rule, per-index domains, ...) falls back to
byte-classic construction, silently and safely (the Spike-B design: partial
coverage + a mandatory fallback).

Activation is a pure monkeypatch of ``Var.construct`` / ``Param.construct``,
installed while the switch is on and removed when it is off, so **with the switch
off there is zero core-module behaviour change** -- the original methods are
literally in place.  The transparent components are ``IndexedVar`` /
``IndexedParam`` subclasses, so ``isinstance`` still holds and every classic
consumer keeps working via materialize-on-touch / scalarization.
"""

from __future__ import annotations

import logging
from weakref import ref as weakref_ref

from pyomo.common.dependencies import numpy as np
from pyomo.common.modeling import NOTSET
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.base.param import Param, IndexedParam, ParamData
from pyomo.core.base.set import _AnySet

from pyomo.contrib.vector.var import _ColumnarVarMixin, VectorVarData, domain_interval

logger = logging.getLogger('pyomo.contrib.vector')

_inf = float('inf')
_ninf = -_inf


# --------------------------------------------------------------------------- #
# Transparent columnar Var
# --------------------------------------------------------------------------- #
class TransparentVectorVar(IndexedVar):
    """A classic ``Var`` constructed into columns under the Phase-3 switch.

    An indexed ``Var`` whose ``bounds`` / ``domain`` / ``initialize`` arguments
    are vectorizable has its ``__class__`` swapped to this at construct time and
    its per-index ``VarData`` allocation replaced by :meth:`construct` here.  It
    IS an :class:`IndexedVar` -- ``isinstance`` holds, ``ctype`` is ``Var`` -- so
    any consumer that does not understand columns keeps working (touching
    ``m.x[i]`` materializes a ``VarData`` whose identity is stable; iterating
    scalarizes).

    The columnar behaviour (materialize-on-touch, scalarization, the fast-path
    array accessors, and the Phase-2 bulk-mutation / dirty-tracking API) is
    grafted from :class:`~pyomo.contrib.vector.var._ColumnarVarMixin` (see the
    graft loop below) rather than inherited, so this stays a *plain* ``IndexedVar``
    subclass -- the layout a live ``__class__`` swap from ``IndexedVar`` requires.
    """

    _ComponentDataClass = VectorVarData

    def __getitem__(self, index):
        # Fast columnar access: a known index materializes in O(1) (after the
        # one-time position map), skipping the classic ``_validate_index``
        # set-membership check that would otherwise make every ``m.x[i]`` in a
        # non-templatized objective / constraint slower than the classic dict
        # lookup -- which would negate the construction win.  Anything the fast
        # path does not recognise (a template / slice index, an out-of-range
        # key) defers to the classic ``IndexedVar.__getitem__`` (which yields the
        # variable ``GetItemExpression`` for templates, or the proper KeyError).
        try:
            return self._data[index]
        except KeyError:
            pass
        except TypeError:
            return IndexedVar.__getitem__(self, index)
        if self._pos_of is None:
            self._build_position_map()
        pos = self._pos_of.get(index)
        if pos is not None:
            obj = VectorVarData(self, pos)
            obj._index = index
            self._data[index] = obj
            return obj
        return IndexedVar.__getitem__(self, index)

    def construct(self, data=None):
        # This runs *after* the interceptor has swapped ``__class__`` from
        # IndexedVar; the classic ``Var.__init__`` already parsed the
        # initializers (``_rule_init`` / ``_rule_bounds`` / ``_rule_domain`` /
        # ``_dense`` / ``_units``) that we read here.
        if self._constructed:
            return
        assert data is None  # Var construction never accepts external data
        self._constructed = True
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        block = self.parent_block()
        self._domain = self._rule_domain(block, None, self)
        n = self._n = len(self._index_set)

        lb = ub = None
        if self._rule_bounds is not None:
            lb, ub = self._rule_bounds(block, None)
        self._lb_arr = _scalar_column(lb, n, self.name, 'lower bound')
        self._ub_arr = _scalar_column(ub, n, self.name, 'upper bound')
        self._value_arr = self._columnar_init_values(block, n)
        self._fixed_arr = np.zeros(n, dtype=bool)

        # Lazy scalar-access / scalarization + dirty-tracking state (mirrors
        # VectorVar.__init__; the interceptor bypassed that constructor).
        self._index_order = None
        self._pos_of = None
        self._scalarized = False
        self._scalarizing = False
        self._dirty_bounds = None

    def _columnar_init_values(self, block, n):
        """Fill the value column from ``initialize=`` (scalar / mapping / None)."""
        ri = self._rule_init
        if ri is None:
            return np.full(n, np.nan, dtype=np.float64)
        if ri.constant():
            val = ri(block, None)
            if val is None:
                return np.full(n, np.nan, dtype=np.float64)
            return np.full(n, float(val), dtype=np.float64)
        # contains_indices(): a mapping / sparse dict -- fill by position.
        arr = np.full(n, np.nan, dtype=np.float64)
        self._build_position_map()
        pos_of = self._pos_of
        for index in ri.indices():
            val = ri(block, index)
            if val is not None:
                arr[pos_of[index]] = float(val)
        return arr


# Graft the shared columnar behaviour onto TransparentVectorVar's own __dict__
# (highest MRO priority), so the identically-named classic ``Var`` /
# ``IndexedComponent`` methods (``set_values``, ``fix``, ``values``, ``__len__``,
# ...) never shadow the columnar versions.  ``construct`` is defined above and is
# not present on the mixin, so it is preserved.
for _name, _attr in _ColumnarVarMixin.__dict__.items():
    if _name in (
        '__dict__',
        '__weakref__',
        '__doc__',
        '__module__',
        '__slots__',
        '__qualname__',
    ):
        continue
    setattr(TransparentVectorVar, _name, _attr)
del _name, _attr


def _scalar_column(val, n, name, what):
    """Broadcast a *constant* bound value (scalar or None) to a length-n column.

    ``None`` -> NaN sentinel ("no explicit bound", matching classic VarData).
    Vectorizability already guaranteed a constant (non-callable) argument, so
    ``val`` here is a scalar or None.
    """
    if val is None:
        return np.full(n, np.nan, dtype=np.float64)
    try:
        return np.full(n, float(val), dtype=np.float64)
    except (TypeError, ValueError):
        # A non-scalar constant bound (array/sequence) -- broadcast if it fits.
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape == (n,):
            return arr.copy()
        raise ValueError(
            f"Var '{name}': vectorized {what} has shape {arr.shape}, "
            f"expected a scalar or ({n},)"
        )


def _var_is_vectorizable(v):
    """True if the indexed ``Var`` ``v`` can be built into columns.

    Requires: a plain ``IndexedVar`` (not ``VarList`` / already-columnar), a
    finite dense index, a homogeneous (constant) domain, constant bounds, and an
    ``initialize`` that is constant or a mapping -- i.e. NOT a genuinely per-index
    callable.  Anything else returns False and the caller uses classic
    construction for the whole component.
    """
    if type(v) is not IndexedVar:
        return False
    if not v._dense:
        return False
    if not v.index_set().isfinite():
        return False
    if not v._rule_domain.constant():
        return False
    rb = v._rule_bounds
    if rb is not None and not rb.constant():
        return False
    ri = v._rule_init
    if ri is not None and not (ri.constant() or ri.contains_indices()):
        return False
    return True


# --------------------------------------------------------------------------- #
# Transparent columnar Param
# --------------------------------------------------------------------------- #
class VectorParamData(ParamData):
    """An array-backed :class:`ParamData` view onto one column of a mutable
    columnar ``Param`` (the ``Param`` twin of :class:`VectorVarData`).

    The single ``_value`` storage slot inherited from :class:`ParamData` is
    shadowed by a property that reads / writes the parent component's value
    column at this object's position, so the array stays the single source of
    truth even after a datum is materialized -- bulk reads and scalar access can
    never drift.  A NaN in the column is the ``Param.NoValue`` sentinel.
    """

    __slots__ = ('_pos',)

    def __init__(self, component, pos):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._pos = pos

    @property
    def _value(self):
        v = self._component()._value_arr[self._pos]
        return Param.NoValue if v != v else float(v)

    @_value.setter
    def _value(self, val):
        c = self._component()
        if val is Param.NoValue or val is None:
            c._value_arr[self._pos] = np.nan
        else:
            c._value_arr[self._pos] = float(val)


class TransparentVectorParam(IndexedParam):
    """A classic ``Param`` constructed into a value column under the switch.

    Mutable: ``m.p[i]`` materialises a :class:`VectorParamData` (identity stable,
    cached in ``_data``) that delegates to the value column.  Immutable:
    ``m.p[i]`` returns the column value directly (no per-index object, as
    classic immutable Params also hold raw values).  Either way the per-index
    ``ParamData`` allocation of classic construction is skipped.  A consumer that
    iterates the Param scalarizes (the compatibility fallback).
    """

    def construct(self, data=None):
        if self._constructed:
            return
        # Flag mirrors Param.construct: None during construction (mutation OK),
        # True afterwards.  Columnar construction never takes external ``data``
        # (the interceptor only routes here when ``data is None``).
        self._constructed = None
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        block = self.parent_block()
        n = self._n = len(self._index_set)

        # Lazy scalar-access / scalarization state (set before building values,
        # which may itself trigger the position map for a mapping initializer).
        self._index_order = None
        self._pos_of = None
        self._scalarized = False
        self._scalarizing = False

        self._value_arr = self._columnar_param_values(block, n)
        self._validate_columnar_domain()
        self._constructed = True

    def _columnar_param_values(self, block, n):
        """Build the value column from ``initialize=`` and any ``default=``."""
        default = self._default_val
        base = np.nan if default is Param.NoValue else float(default)
        arr = np.full(n, base, dtype=np.float64)
        rule = self._rule
        if rule is None:
            return arr
        if rule.constant() and not rule.contains_indices():
            val = rule(block, None)
            arr[:] = np.nan if val is None else float(val)
            return arr
        # Mapping / sparse dict: fill provided keys, leave the rest at default.
        self._build_position_map()
        pos_of = self._pos_of
        for index in rule.indices():
            val = rule(block, index)
            if val is not None:
                arr[pos_of[index]] = float(val)
        return arr

    def _validate_columnar_domain(self):
        """Bulk-validate the value column against a numeric-interval domain.

        Mirrors classic per-index domain validation without materializing
        objects.  ``Any`` / non-interval domains impose no numeric constraint
        (vectorizability already excluded validation rules and united Params).
        """
        is_int, lo, hi = domain_interval(self.domain)
        vals = self._value_arr
        finite = ~np.isnan(vals)
        if not finite.any():
            return
        v = vals[finite]
        bad = None
        if lo is not None and (v < lo).any():
            bad = v[v < lo][0]
        elif hi is not None and (v > hi).any():
            bad = v[v > hi][0]
        elif is_int and not np.all(v == np.round(v)):
            bad = v[v != np.round(v)][0]
        if bad is not None:
            raise ValueError(
                "Invalid parameter value: %s = '%s'.\n\tValue not in parameter "
                "domain %s" % (self.name, bad, self.domain.name)
            )

    # ------------------------------------------------------------------ #
    # Position mapping (shared shape with _ColumnarVarMixin, kept local so
    # the Param need not inherit the Var mixin)
    # ------------------------------------------------------------------ #
    def _build_position_map(self):
        if self._pos_of is not None:
            return
        order = list(self._index_set)
        self._index_order = order
        self._pos_of = {idx: pos for pos, idx in enumerate(order)}

    def position_of(self, index):
        self._build_position_map()
        return self._pos_of[index]

    @property
    def n(self):
        return self._n

    @property
    def value_array(self):
        return self._value_arr

    def _columnar_param_values_at(self, keys):
        """Vectorized value read for a list of index keys (fast-path extractor).

        Returns a float64 array of the column values at ``keys`` (a mutable
        Param's live values, an immutable Param's fixed values).  ``keys`` is a
        list of index tuples / scalars in the caller's order.
        """
        self._build_position_map()
        pos_of = self._pos_of
        arr = self._value_arr
        try:
            positions = np.fromiter(
                (pos_of[k] for k in keys), dtype=np.int64, count=len(keys)
            )
        except KeyError as e:
            raise KeyError(f"index {e} not present in Param '{self.name}'") from e
        return arr[positions]

    # ------------------------------------------------------------------ #
    # Materialize-on-touch
    # ------------------------------------------------------------------ #
    def _getitem_when_not_present(self, index):
        pos = self.position_of(index)
        if self._mutable:
            obj = self._data[index] = VectorParamData(self, pos)
            obj._index = index
            return obj
        # Immutable: serve the value directly (do not populate _data).
        v = self._value_arr[pos]
        if v != v:  # NaN -> undefined; defer to the classic default/NoValue path
            return Param._getitem_when_not_present(self, index)
        return float(v)

    # ------------------------------------------------------------------ #
    # Scalarization contract (an unaware consumer iterating -> materialize)
    # ------------------------------------------------------------------ #
    def __len__(self):
        return self._n

    def _scalarize(self, reason="iterated"):
        if self._scalarized or self._scalarizing:
            return
        self._scalarizing = True
        try:
            self._build_position_map()
            if self._mutable:
                for index in self._index_order:
                    if index not in self._data:
                        self._getitem_when_not_present(index)
            else:
                arr = self._value_arr
                for pos, index in enumerate(self._index_order):
                    if index not in self._data:
                        v = arr[pos]
                        if v == v:  # skip undefined (NaN) entries
                            self._data[index] = float(v)
            self._scalarized = True
            logger.warning(
                "Param '%s' was scalarized (%s): a consumer that does not "
                "support columnar parameters triggered full materialization of "
                "its %d entries.  This is the compatibility fallback; the fast "
                "path is disabled for this component." % (self.name, reason, self._n),
                extra={'id': 'W-VEC02'},
            )
        finally:
            self._scalarizing = False

    def values(self, *args, **kwargs):
        self._scalarize()
        return super().values(*args, **kwargs)

    def items(self, *args, **kwargs):
        self._scalarize()
        return super().items(*args, **kwargs)

    def keys(self, *args, **kwargs):
        self._scalarize()
        return super().keys(*args, **kwargs)

    def __iter__(self):
        self._scalarize()
        return super().__iter__()


def _param_is_vectorizable(p):
    """True if the indexed ``Param`` ``p`` can be built into a value column.

    Requires: a plain ``IndexedParam``, a finite index, no per-index validation
    rule, no units, an ``initialize`` that is constant or a mapping (not a
    genuinely per-index callable), and a domain that imposes no per-index
    constraint we cannot check in bulk (``Any`` / a numeric interval).  A
    per-index callable ``default`` also falls back.
    """
    if type(p) is not IndexedParam:
        return False
    if not p.index_set().isfinite():
        return False
    if p._validate is not None:
        return False
    if p._units is not None:
        return False
    rule = p._rule
    if rule is not None and not (rule.constant() or rule.contains_indices()):
        return False
    # ``default`` must be a plain scalar (or NoValue); a callable / mapping
    # default is per-index -> classic.
    default = p._default_val
    if default is not Param.NoValue and type(default) not in (int, float):
        return False
    # Domain must be Any (no constraint) or a numeric interval we can bulk-check.
    dom = p.domain
    if not isinstance(dom, _AnySet):
        try:
            if dom.get_interval() is None:
                return False
        except Exception:
            return False
    return True


# --------------------------------------------------------------------------- #
# Activation: monkeypatch Var.construct / Param.construct while the switch is on
# --------------------------------------------------------------------------- #
_ORIG_VAR_CONSTRUCT = None
_ORIG_PARAM_CONSTRUCT = None


def _var_construct_interceptor(self, data=None):
    if self._constructed:
        return
    if _var_is_vectorizable(self):
        self.__class__ = TransparentVectorVar
        self.construct(data)
    else:
        _ORIG_VAR_CONSTRUCT(self, data)


def _param_construct_interceptor(self, data=None):
    if self._constructed:
        return
    if data is None and _param_is_vectorizable(self):
        self.__class__ = TransparentVectorParam
        self.construct(data)
    else:
        _ORIG_PARAM_CONSTRUCT(self, data)


def set_varparam_vectorize(flag):
    """Install (flag=True) or remove the Var/Param columnar construct patch.

    Returns the prior ``(installed, orig_var, orig_param)`` so a caller can
    restore it (nesting-safe, like the templatize switch).  Idempotent.
    """
    global _ORIG_VAR_CONSTRUCT, _ORIG_PARAM_CONSTRUCT
    installed = _ORIG_VAR_CONSTRUCT is not None
    prior = (installed, _ORIG_VAR_CONSTRUCT, _ORIG_PARAM_CONSTRUCT)
    if flag and not installed:
        _ORIG_VAR_CONSTRUCT = Var.construct
        _ORIG_PARAM_CONSTRUCT = Param.construct
        Var.construct = _var_construct_interceptor
        Param.construct = _param_construct_interceptor
    elif not flag and installed:
        Var.construct = _ORIG_VAR_CONSTRUCT
        Param.construct = _ORIG_PARAM_CONSTRUCT
        _ORIG_VAR_CONSTRUCT = None
        _ORIG_PARAM_CONSTRUCT = None
    return prior


def restore_varparam_vectorize(prior):
    """Restore the state returned by a previous :func:`set_varparam_vectorize`."""
    global _ORIG_VAR_CONSTRUCT, _ORIG_PARAM_CONSTRUCT
    installed, orig_var, orig_param = prior
    if installed:
        # Ensure the patch is in place with the recorded originals.
        _ORIG_VAR_CONSTRUCT = orig_var
        _ORIG_PARAM_CONSTRUCT = orig_param
        Var.construct = _var_construct_interceptor
        Param.construct = _param_construct_interceptor
    else:
        if _ORIG_VAR_CONSTRUCT is not None:
            Var.construct = _ORIG_VAR_CONSTRUCT
        if _ORIG_PARAM_CONSTRUCT is not None:
            Param.construct = _ORIG_PARAM_CONSTRUCT
        _ORIG_VAR_CONSTRUCT = None
        _ORIG_PARAM_CONSTRUCT = None


def is_columnar_var(v):
    """True if ``v`` is a columnar (array-backed) Var -- fast-path readable.

    Duck-typed so it recognises both the explicit Phase-1 ``VectorVar`` and the
    transparent :class:`TransparentVectorVar` without importing the former.
    """
    return (
        getattr(v, '_lb_arr', None) is not None
        and hasattr(v, 'effective_bounds')
        and hasattr(v, 'integrality')
    )


def is_columnar_param(p):
    """True if ``p`` is a transparently columnar Param (bulk-value readable)."""
    return isinstance(p, TransparentVectorParam)
