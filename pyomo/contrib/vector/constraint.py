# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Explicit-array linear constraint families for the vectorized fast path.

:class:`VectorConstraint` is ambition 1 of the scoping document (§6.2): a whole
family of linear constraints stated as one object carrying CSR coefficient
arrays plus lower/upper bound arrays, over a column space spanned by one or more
:class:`VectorVar` components::

    m.balance = VectorConstraint(A=csr_matrix, x=[m.flow, m.stor],
                                 lb=b, ub=b)          # equality (lb == ub)
    m.cap     = VectorConstraint(A=csr_matrix, x=m.flow, ub=cap)   # A x <= cap

No per-index ``ConstraintData`` object and no per-index expression tree are ever
built on the fast path.  Scalar access (``m.balance[r]``) and iteration lazily
materialize a classic :class:`ConstraintData` for the requested row (the
compatibility / scalarization contract, scoping doc §6.5).
"""

from __future__ import annotations

import logging
from weakref import ref as weakref_ref

from pyomo.common.dependencies import numpy as np, scipy
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.relational_expr import inequality
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent,
    UnindexedComponent_set,
)
from pyomo.contrib.vector.var import VectorVar

logger = logging.getLogger('pyomo.contrib.vector')

_inf = float('inf')
_ninf = -_inf


class VectorConstraintData(ConstraintData):
    """A classic :class:`ConstraintData` lazily materialized from one CSR row.

    Only created when a consumer touches ``m.con[r]`` or iterates the family
    (the scalarization fallback).  The fast path never instantiates these.
    """

    __slots__ = ('_row',)

    def __init__(self, component, row):
        self._component = weakref_ref(component) if component is not None else None
        self._active = True
        self._row = row
        self._expr = component._build_row_expression(row)


class VectorConstraint(ActiveIndexedComponent):
    """A linear constraint family stored as explicit CSR + bound arrays.

    Parameters
    ----------
    A : scipy.sparse matrix or 2-D array
        The ``(n_rows, n_cols)`` coefficient matrix.  ``n_cols`` must equal the
        total number of columns spanned by ``x``.
    x : VectorVar or sequence of VectorVar
        The columnar variable(s) whose concatenated columns index ``A``.
    lb, ub : None, scalar, or length-n_rows array
        Row lower / upper bounds.  ``None`` means unbounded on that side.
    rhs : None, scalar, or length-n_rows array
        Convenience for equality rows: sets ``lb = ub = rhs`` (mutually
        exclusive with ``lb``/``ub``).
    """

    _ComponentDataClass = VectorConstraintData

    def __init__(self, *args, **kwargs):
        self._A_arg = kwargs.pop('A', None)
        x = kwargs.pop('x', None)
        self._lb_arg = kwargs.pop('lb', None)
        self._ub_arg = kwargs.pop('ub', None)
        self._rhs_arg = kwargs.pop('rhs', None)
        if self._A_arg is None or x is None:
            raise ValueError("VectorConstraint requires both 'A' and 'x'.")
        if self._rhs_arg is not None and (
            self._lb_arg is not None or self._ub_arg is not None
        ):
            raise ValueError("Specify either 'rhs' or 'lb'/'ub', not both.")
        self._xvars = [x] if isinstance(x, VectorVar) else list(x)
        kwargs.setdefault('ctype', Constraint)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

        self._nrows = 0
        self._A = None  # CSR (row-major), columns are GLOBAL over the x list
        self._row_lb = None  # float64, -inf allowed
        self._row_ub = None  # float64, +inf allowed
        self._col_split = None  # cumulative column offsets of each xvar block
        self._user_index = self._index_set is not UnindexedComponent_set
        self._scalarized = False
        self._scalarizing = False
        # Per-row active mask (masked deactivation, Phase-2).  True == the row is
        # enforced.  A deactivated row is *removed* from the one-shot standard
        # form (matching classic ``con[r].deactivate()``) and *relaxed* to
        # (-inf, +inf) on the persistent solve path (cheap, warm-basis friendly).
        self._row_active = None
        # Dirty-row tracking for the persistent (warm) re-solve path: positions
        # whose effective row bounds (RHS or active flag) changed since the last
        # ``pop_dirty_rows``.  ``None`` == "all dirty".
        self._dirty_rows = None

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
        for v in self._xvars:
            if not v._constructed:
                v.construct()

        A = self._A_arg
        if scipy.sparse.issparse(A):
            A = A.tocsr()
        else:
            A = scipy.sparse.csr_matrix(np.asarray(A, dtype=np.float64))
        A = A.astype(np.float64)
        A.eliminate_zeros()
        self._A = A
        self._nrows = A.shape[0]

        # Column space = concatenation of the x variables.
        sizes = [v.n for v in self._xvars]
        self._col_split = np.concatenate([[0], np.cumsum(sizes)])
        if A.shape[1] != int(self._col_split[-1]):
            raise ValueError(
                f"VectorConstraint '{self.name}': A has {A.shape[1]} columns but "
                f"x spans {int(self._col_split[-1])} columns."
            )

        n = self._nrows
        self._row_lb, self._row_ub = self._normalize_bounds(n)
        self._row_active = np.ones(n, dtype=bool)

        if not self._user_index:
            from pyomo.core.base.set import RangeSet

            rs = RangeSet(0, n - 1)
            rs.construct()
            self._index_set = rs

    def _normalize_bounds(self, n):
        def _vec(val, default):
            if val is None:
                return np.full(n, default, dtype=np.float64)
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 0:
                return np.full(n, float(arr), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(
                    f"VectorConstraint '{self.name}': bound array shape "
                    f"{arr.shape}, expected ({n},)"
                )
            return arr.copy()

        if self._rhs_arg is not None:
            rhs = _vec(self._rhs_arg, np.nan)
            return rhs.copy(), rhs.copy()
        lb = _vec(self._lb_arg, _ninf)
        ub = _vec(self._ub_arg, _inf)
        # NaN (from an all-None default) means unbounded on that side.
        lb = np.where(np.isnan(lb), _ninf, lb)
        ub = np.where(np.isnan(ub), _inf, ub)
        return lb, ub

    # ------------------------------------------------------------------ #
    # Fast-path array accessors
    # ------------------------------------------------------------------ #
    @property
    def nrows(self):
        return self._nrows

    @property
    def A(self):
        return self._A

    @property
    def row_lb(self):
        return self._row_lb

    @property
    def row_ub(self):
        return self._row_ub

    @property
    def xvars(self):
        return self._xvars

    @property
    def col_split(self):
        return self._col_split

    @property
    def row_active(self):
        return self._row_active

    def effective_row_bounds(self):
        """Return ``(lb, ub)`` with deactivated rows relaxed to ``(-inf, +inf)``.

        This is the solve-path view of the row mask: a masked-out row becomes
        vacuous (never binding), which is the persistent-path equivalent of the
        one-shot standard form dropping the row entirely.
        """
        if self._row_active.all():
            return self._row_lb, self._row_ub
        lb = np.where(self._row_active, self._row_lb, _ninf)
        ub = np.where(self._row_active, self._row_ub, _inf)
        return lb, ub

    # ------------------------------------------------------------------ #
    # Masked deactivation + RHS mutation + dirty tracking (Phase-2)
    # ------------------------------------------------------------------ #
    def _resolve_rows(self, where):
        """Normalize a ``where=`` row selector to an int position array (None=all)."""
        if where is None:
            return None
        arr = np.asarray(where)
        if arr.dtype == bool:
            if arr.shape != (self._nrows,):
                raise ValueError(
                    f"VectorConstraint '{self.name}': boolean 'where' mask has "
                    f"shape {arr.shape}, expected ({self._nrows},)."
                )
            return np.nonzero(arr)[0]
        return arr.astype(np.int64, copy=False).ravel()

    def _mark_rows_dirty(self, positions):
        if positions is None:
            self._dirty_rows = None
        elif self._dirty_rows is not None:
            self._dirty_rows.update(int(p) for p in positions)

    def deactivate_rows(self, where):
        """Mask off (deactivate) the selected rows; marks them dirty."""
        pos = self._resolve_rows(where)
        if pos is None:
            self._row_active[:] = False
        else:
            self._row_active[pos] = False
        self._mark_rows_dirty(pos)

    def activate_rows(self, where=None):
        """Re-activate the selected rows (default all); marks them dirty."""
        pos = self._resolve_rows(where)
        if pos is None:
            self._row_active[:] = True
        else:
            self._row_active[pos] = True
        self._mark_rows_dirty(pos)

    def set_row_active(self, mask):
        """Set the whole active mask from a length-nrows boolean array."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self._nrows,):
            raise ValueError(
                f"VectorConstraint '{self.name}': active mask has shape "
                f"{mask.shape}, expected ({self._nrows},)."
            )
        changed = np.nonzero(mask != self._row_active)[0]
        self._row_active = mask.copy()
        self._mark_rows_dirty(changed)

    def set_row_bounds(self, lb=NOTSET, ub=NOTSET, where=None):
        """Bulk-mutate row lower/upper bounds (RHS); marks rows dirty.

        ``lb``/``ub`` are scalars (broadcast) or arrays over the selection;
        ``None`` means unbounded on that side.  Pass only the side(s) to change.
        """
        pos = self._resolve_rows(where)
        count = self._nrows if pos is None else len(pos)

        def _vals(val):
            arr = np.asarray(_ninf if val is None else val, dtype=np.float64)
            if arr.ndim == 0:
                return np.full(count, float(arr), dtype=np.float64)
            if arr.shape != (count,):
                raise ValueError(
                    f"VectorConstraint '{self.name}': RHS array shape {arr.shape} "
                    f"does not match the {count} selected row(s)."
                )
            return arr

        if lb is not NOTSET:
            v = _vals(lb if lb is not None else _ninf)
            if pos is None:
                self._row_lb[:] = v
            else:
                self._row_lb[pos] = v
        if ub is not NOTSET:
            v = _vals(ub if ub is not None else _inf)
            if pos is None:
                self._row_ub[:] = v
            else:
                self._row_ub[pos] = v
        self._mark_rows_dirty(pos)

    def pop_dirty_rows(self):
        """Return dirty row positions (int array, or None=all) and clear."""
        d = self._dirty_rows
        self._dirty_rows = set()
        if d is None:
            return None
        return np.array(sorted(d), dtype=np.int64)

    def mark_all_rows_dirty(self):
        self._dirty_rows = None

    # ------------------------------------------------------------------ #
    # Row -> (VectorVar, position) mapping for a single (sparse) row
    # ------------------------------------------------------------------ #
    def _local_col_to_var(self, col):
        b = int(np.searchsorted(self._col_split, col, side='right') - 1)
        return self._xvars[b], int(col - self._col_split[b])

    def _build_row_expression(self, row):
        """Build a classic relational expression for one CSR row (scalarize)."""
        A = self._A
        s, e = A.indptr[row], A.indptr[row + 1]
        cols = A.indices[s:e]
        coefs = A.data[s:e]
        varlist = []
        for c in cols:
            v, pos = self._local_col_to_var(int(c))
            varlist.append(v[v.index_at(pos)])
        body = LinearExpression(
            constant=0, linear_coefs=list(coefs), linear_vars=varlist
        )
        lb = self._row_lb[row]
        ub = self._row_ub[row]
        has_lb = lb != _ninf and not np.isnan(lb)
        has_ub = ub != _inf and not np.isnan(ub)
        if has_lb and has_ub:
            if lb == ub:
                return body == float(lb)
            return inequality(float(lb), body, float(ub))
        if has_ub:
            return body <= float(ub)
        if has_lb:
            return float(lb) <= body
        # Unbounded both sides: a trivial constraint.  Represent as 0 <= body's
        # (feasible) form; the compiler skips these rows anyway.
        return inequality(None, body, None)

    # ------------------------------------------------------------------ #
    # Materialize-on-touch + scalarization contract
    # ------------------------------------------------------------------ #
    def is_indexed(self):
        return True

    def __len__(self):
        return self._nrows

    def _getitem_when_not_present(self, row):
        if not isinstance(row, (int, np.integer)) or row < 0 or row >= self._nrows:
            raise KeyError(row)
        obj = VectorConstraintData(self, int(row))
        obj._index = int(row)
        self._data[int(row)] = obj
        return obj

    def _setitem_when_not_present(self, index, value=NOTSET):
        raise NotImplementedError(
            "VectorConstraint rows are defined by the A/lb/ub arrays and cannot "
            "be assigned individually in Phase 1."
        )

    def _scalarize(self, reason="iterated"):
        if self._scalarized or self._scalarizing:
            return
        self._scalarizing = True
        try:
            for row in range(self._nrows):
                if row not in self._data:
                    self._getitem_when_not_present(row)
            self._scalarized = True
            logger.warning(
                "VectorConstraint '%s' was scalarized (%s): a consumer that does "
                "not support vectorized constraints triggered materialization of "
                "its %d rows into classic ConstraintData/LinearExpression "
                "objects (scoping doc §6.5).  The fast path is disabled for this "
                "component." % (self.name, reason, self._nrows),
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

    def _pprint(self):
        headers = [
            ("Size", self._nrows),
            ("Columns", None if self._col_split is None else int(self._col_split[-1])),
            ("Vectorized", True),
            ("Scalarized", self._scalarized),
        ]
        return (headers, (), None, None)
