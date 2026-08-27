# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Columnar (array-backed) variables for the vectorized fast path.

:class:`VectorVar` stores the per-index variable data (bounds, value, fixed
flag) as parallel NumPy arrays instead of one Python :class:`VarData` object per
index.  This is the "columnar ``Var``" of the scoping document (§6.1, #202):
Phase-0 Spike A measured ~500x faster bulk allocation and ~0.14x the memory of
the classic per-index objects.

Design decisions locked in by the Phase-0 report (§6, §9 R1):

* **Materialize-on-touch, not pure flyweight.**  ``m.x[i]`` lazily creates a
  permanent :class:`VectorVarData` object, caches it in ``_data[i]``, and returns
  the *same* object on every subsequent access.  Because the object is permanent
  and cached, ``m.x[i] is m.x[i]`` holds by construction -- which is the load
  bearing correctness requirement, since repn and solver maps key on ``id()``.

* **Array-backed data objects.**  The materialized :class:`VectorVarData`
  delegates every bound/value/fixed read and write straight to the parent's
  arrays (it overrides the base :class:`VarData` storage slots with properties).
  The arrays therefore remain the single source of truth even after a datum is
  materialized, so the fast path (which reads the arrays in bulk) always agrees
  with scalar access -- there is no way for the two views to drift.

* **Homogeneous domain per component (Phase-1 scope).**  The whole ``VectorVar``
  shares one ``domain`` (e.g. ``NonNegativeReals``).  This matches idiomatic
  ``Var(index, domain=...)`` usage; per-index domain heterogeneity is a deferred
  Phase-2 feature (see the PR body).
"""

from __future__ import annotations

import logging
from weakref import ref as weakref_ref

from pyomo.common.dependencies import numpy as np
from pyomo.common.modeling import NOTSET
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.base.var import Var, VarData
from pyomo.core.base.set import Reals, SetInitializer
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.units_container import units

logger = logging.getLogger('pyomo.contrib.vector')

_inf = float('inf')
_ninf = -_inf


def domain_interval(domain):
    """Return ``(is_integer, domain_lb, domain_ub)`` for a (global) Set domain.

    ``domain_lb`` / ``domain_ub`` are numeric or ``None`` (unbounded); the
    interval is taken from :meth:`Set.get_interval` so it matches exactly what
    the classic :class:`VarData` uses to combine domain and explicit bounds.
    """
    interval = domain.get_interval()
    if interval is None:
        # Not a simple numeric interval (e.g. a discrete non-contiguous set).
        return False, None, None
    lo, hi, step = interval
    is_integer = step is not None and step != 0
    return is_integer, lo, hi


class VectorVarData(VarData):
    """An array-backed :class:`VarData` view onto one column of a VectorVar.

    Every storage slot inherited from :class:`VarData` (``_lb``, ``_ub``,
    ``_value``, ``_fixed``, ``_domain``) is shadowed by a property that
    reads/writes the parent component's NumPy arrays at this object's column
    position.  ``_stale`` is left as an ordinary (per-object) slot since it does
    not participate in the matrix representation.
    """

    __slots__ = ('_pos',)

    def __init__(self, component, pos):
        # Inlined ComponentData / VarData constructor: we set only the two
        # genuine slots (_component, _pos); every other "slot" is a property.
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._pos = pos
        self._stale = 0  # True

    # -- storage delegated to the parent component's columnar arrays --------- #
    @property
    def _lb(self):
        v = self._component()._lb_arr[self._pos]
        return None if v != v else float(v)  # NaN sentinel == "no explicit lb"

    @_lb.setter
    def _lb(self, val):
        # store None as NaN so the "no explicit bound" state round-trips
        self._component()._lb_arr[self._pos] = np.nan if val is None else val

    @property
    def _ub(self):
        v = self._component()._ub_arr[self._pos]
        return None if v != v else float(v)

    @_ub.setter
    def _ub(self, val):
        self._component()._ub_arr[self._pos] = np.nan if val is None else val

    @property
    def _value(self):
        v = self._component()._value_arr[self._pos]
        return None if v != v else float(v)

    @_value.setter
    def _value(self, val):
        self._component()._value_arr[self._pos] = np.nan if val is None else val

    @property
    def _fixed(self):
        return bool(self._component()._fixed_arr[self._pos])

    @_fixed.setter
    def _fixed(self, val):
        self._component()._fixed_arr[self._pos] = bool(val)

    @property
    def _domain(self):
        return self._component()._domain

    @_domain.setter
    def _domain(self, val):
        # Homogeneous-domain scope (Phase 1): only accept the component domain.
        if val is not self._component()._domain:
            raise NotImplementedError(
                "Per-index domains are not supported on VectorVar in Phase 1; "
                "the whole component shares one domain.  Use a classic Var, or "
                "declare a separate VectorVar for the differing domain."
            )


class VectorVar(IndexedComponent):
    """A columnar (array-backed) indexed variable.

    Parameters
    ----------
    *args :
        One or more Pyomo Sets giving the index (as for a classic ``Var``).
    domain : Set
        The (homogeneous) domain shared by every entry.  Default ``Reals``.
    bounds : tuple, optional
        ``(lower, upper)``.  Each of ``lower``/``upper`` may be ``None``, a
        scalar (broadcast to all entries), or a length-N array/sequence.
    initialize : scalar or array, optional
        Initial value(s).
    """

    _ComponentDataClass = VectorVarData

    def __init__(self, *args, **kwargs):
        domain = kwargs.pop('domain', None)
        within = kwargs.pop('within', None)
        self._domain_init = SetInitializer(
            domain if domain is not None else (within if within is not None else Reals)
        )
        self._bounds_arg = kwargs.pop('bounds', None)
        self._init_arg = kwargs.pop('initialize', None)
        self._units = kwargs.pop('units', None)
        if self._units is not None:
            self._units = units.get_units(self._units)
        kwargs.setdefault('ctype', Var)
        IndexedComponent.__init__(self, *args, **kwargs)

        # Populated at construct():
        self._domain = None
        self._n = 0
        self._lb_arr = None
        self._ub_arr = None
        self._value_arr = None
        self._fixed_arr = None
        # Lazily built (only on scalar access / scalarization):
        self._index_order = None  # list: position -> index
        self._pos_of = None  # dict: index -> position
        self._scalarized = False
        self._scalarizing = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def construct(self, data=None):
        if self._constructed:
            return
        self._constructed = True
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        self._domain = self._domain_init(self.parent_block(), None, self)
        n = self._n = len(self._index_set)

        def _as_array(val, default):
            if val is None:
                return np.full(n, default, dtype=np.float64)
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 0:
                return np.full(n, float(arr), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(
                    f"VectorVar '{self.name}': bound/initialize array has shape "
                    f"{arr.shape}, expected ({n},)"
                )
            return arr.copy()

        lb = ub = None
        if self._bounds_arg is not None:
            lb, ub = self._bounds_arg
        # Explicit bounds stored with NaN == "no explicit bound" (matches the
        # classic VarData _lb/_ub is None semantics).
        self._lb_arr = _as_array(lb, np.nan)
        self._ub_arr = _as_array(ub, np.nan)
        self._value_arr = _as_array(self._init_arg, np.nan)
        self._fixed_arr = np.zeros(n, dtype=bool)

    # ------------------------------------------------------------------ #
    # Position mapping (built lazily -- the fast path never needs it)
    # ------------------------------------------------------------------ #
    def _build_position_map(self):
        if self._index_order is not None:
            return
        order = list(self._index_set)
        self._index_order = order
        self._pos_of = {idx: pos for pos, idx in enumerate(order)}

    def position_of(self, index):
        """Column position (0..N-1) of ``index`` in this component."""
        self._build_position_map()
        return self._pos_of[index]

    def index_at(self, pos):
        self._build_position_map()
        return self._index_order[pos]

    # ------------------------------------------------------------------ #
    # Fast-path array accessors (no per-index Python object materialized)
    # ------------------------------------------------------------------ #
    @property
    def n(self):
        return self._n

    def effective_bounds(self):
        """Return ``(lb, ub)`` float64 arrays combining explicit + domain bounds.

        This replicates :meth:`VarData._resolve_bound_value` vectorially: the
        effective lower bound is the tighter (max) of the explicit lb and the
        domain lb, and symmetrically (min) for the upper bound.  ``None`` bounds
        become +/- inf so the arrays can go straight to a solver.
        """
        _, dom_lb, dom_ub = domain_interval(self._domain)
        lb = np.where(np.isnan(self._lb_arr), _ninf, self._lb_arr)
        ub = np.where(np.isnan(self._ub_arr), _inf, self._ub_arr)
        if dom_lb is not None:
            lb = np.maximum(lb, dom_lb)
        if dom_ub is not None:
            ub = np.minimum(ub, dom_ub)
        return lb, ub

    def integrality(self):
        """Bool array (length N); True where the entry is integer/binary."""
        is_int, _, _ = domain_interval(self._domain)
        return np.full(self._n, is_int, dtype=bool)

    @property
    def value_array(self):
        return self._value_arr

    @property
    def fixed_array(self):
        return self._fixed_arr

    def get_units(self):
        return self._units

    # ------------------------------------------------------------------ #
    # Materialize-on-touch
    # ------------------------------------------------------------------ #
    def _getitem_when_not_present(self, index):
        pos = self.position_of(index)
        obj = VectorVarData(self, pos)
        obj._index = index
        self._data[index] = obj
        return obj

    def _setitem_when_not_present(self, index, value=NOTSET):
        obj = self._getitem_when_not_present(index)
        if value is not NOTSET and value is not None:
            obj.set_value(value)
        return obj

    # ------------------------------------------------------------------ #
    # Length / iteration: force the compatibility (scalarization) contract
    # ------------------------------------------------------------------ #
    def __len__(self):
        # Logical length (never triggers materialization).  Making this equal
        # to len(index_set) is also what lets the base values()/items() walk
        # the *full* index (see keys(): the "dense" branch fires).
        return self._n

    def _scalarize(self, reason="iterated"):
        """Materialize every entry as a permanent VarData (the fallback path)."""
        if self._scalarized or self._scalarizing:
            return
        self._scalarizing = True
        try:
            self._build_position_map()
            for index in self._index_order:
                if index not in self._data:
                    self._getitem_when_not_present(index)
            self._scalarized = True
            logger.warning(
                "VectorVar '%s' was scalarized (%s): a consumer that does not "
                "support columnar variables triggered full materialization of "
                "its %d VarData objects.  This is the compatibility fallback "
                "(scoping doc §6.5); the fast path is disabled for this "
                "component." % (self.name, reason, self._n),
                extra={'id': 'W-VEC01'},
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

    def _pprint(self):
        from pyomo.core.base.var import value as _value

        headers = [
            ("Size", self._n),
            ("Index", self._index_set if self.is_indexed() else None),
            ("Domain", None if self._domain is None else self._domain.name),
            ("Columnar", True),
            ("Scalarized", self._scalarized),
        ]
        return (headers, (), None, None)
