# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Persistent warm re-solve for the vectorized fast path (Phase-2 mutability).

:class:`VectorPersistentHighs` loads an all-vector model into an in-process
HiGHS once (:func:`~pyomo.contrib.vector.highs.load_highs`) and keeps the solver
live.  Between solves the caller mutates the *columnar* components directly --
bounds (``m.x.setlb(...)`` / ``m.x[i].setub(...)``), fixed flags
(``m.x.fix(...)`` / ``m.x.unfix()``), row masks
(``m.con.deactivate_rows(...)``), and RHS (``m.con.set_row_bounds(...)``).  Each
mutation records the touched columns/rows *dirty*; :meth:`solve` expands only the
dirty subset and pushes it to HiGHS through the incremental batch APIs
(``changeColsBounds`` / ``changeRowsBounds``), retaining the warm simplex basis.

This is the array-native twin of :class:`~pyomo.contrib.vector.faststep.FastStepHighs`
(which drives the same HiGHS ``changeColsBounds`` path from classic-model mutable
``Param`` templates): here the update source is the VectorVar / VectorConstraint
arrays directly, so no template extraction is needed.

Structural-change discipline (mirroring FastStepHighs): a change to the *shape*
of the problem -- a VectorVar's length, a VectorConstraint's row count or matrix
nonzero count, or the objective's linear/quadratic character -- invalidates the
compiled model and is rejected loudly (:class:`PersistentStructureError`); the
caller must build a fresh handle.  Only value-level updates (bounds, fixed pins,
row masks, RHS) are pushed in place.  A masked-off row is relaxed to
``(-inf, +inf)`` via ``changeRowsBounds`` (vacuous, warm-basis friendly), the
persistent-path equivalent of the one-shot standard form dropping the row.
"""

from __future__ import annotations

from pyomo.common.dependencies import numpy as np

from pyomo.contrib.vector.highs import matrices_to_highs_model
from pyomo.contrib.vector.matrices import assemble

_inf = float('inf')
_ninf = -_inf


class PersistentStructureError(Exception):
    """Raised when a structural change makes the compiled model stale."""


class VectorSolveResult:
    """Lightweight result of one :meth:`VectorPersistentHighs.solve`."""

    __slots__ = ('status', 'termination', 'objective', 'col_value')

    def __init__(self, status, termination, objective, col_value):
        self.status = status
        self.termination = termination
        self.objective = objective
        self.col_value = col_value

    @property
    def optimal(self):
        return self.termination == 'optimal'

    def __repr__(self):
        return (
            f"VectorSolveResult(termination={self.termination!r}, "
            f"objective={self.objective!r})"
        )


class VectorPersistentHighs:
    """A persistent HiGHS handle for warm re-solves of a mutating vector model.

    Parameters
    ----------
    model :
        A pure-vector model (all active constraints/objective are
        :class:`~pyomo.contrib.vector.constraint.VectorConstraint` /
        :class:`~pyomo.contrib.vector.objective.VectorObjective`).  The model is
        assembled and loaded immediately.
    """

    def __init__(self, model):
        import highspy

        self._model = model
        mx = assemble(model)
        self._var_blocks = mx.var_blocks
        self._row_blocks = mx.row_blocks
        self._n_var = mx.n_var
        self._n_row = mx.n_row
        self._is_quadratic = mx.is_quadratic
        self._fingerprint = self._structure_fingerprint()

        m = matrices_to_highs_model(mx)
        h = highspy.Highs()
        h.silent()
        h.passModel(m)
        self._highs = h
        self._inf = highspy.kHighsInf
        self._n_solves = 0
        self._last = None
        # The load above already reflects every current bound / fixed pin / row
        # mask, so start from a clean slate: discard the "all dirty" initial
        # state on every component.
        self._clear_dirty()

    # ------------------------------------------------------------------ #
    # Structural guard
    # ------------------------------------------------------------------ #
    def _structure_fingerprint(self):
        """A cheap shape signature: per-var length, per-con (nrows, nnz), sizes."""
        vsig = tuple((id(v), int(v.n)) for v, _, _ in self._var_blocks)
        csig = tuple(
            (id(c), int(c.nrows), int(c.A.nnz)) for c, _, _ in self._row_blocks
        )
        return (self._n_var, self._n_row, vsig, csig, bool(self._is_quadratic))

    def _check_structure(self):
        if self._structure_fingerprint() != self._fingerprint:
            raise PersistentStructureError(
                f"The structure of model '{self._model.name}' changed since this "
                "VectorPersistentHighs was built (a variable length, constraint "
                "row count, matrix nonzero count, or the objective's "
                "linear/quadratic character differs).  The warm re-solve path "
                "only pushes value-level updates (bounds, fixed pins, row masks, "
                "RHS); build a fresh VectorPersistentHighs for the new structure."
            )

    def _clear_dirty(self):
        for v, _, _ in self._var_blocks:
            v.pop_dirty_bounds()
        for c, _, _ in self._row_blocks:
            c.pop_dirty_rows()

    # ------------------------------------------------------------------ #
    # Incremental push
    # ------------------------------------------------------------------ #
    def _var_effective_bounds(self, v):
        """Full-length ``(lb, ub)`` for ``v`` with fixed columns pinned."""
        lb, ub = v.effective_bounds()
        lb = np.array(lb, dtype=np.float64, copy=True)
        ub = np.array(ub, dtype=np.float64, copy=True)
        fixed = v.fixed_array
        if fixed.any():
            val = v.value_array
            pin = np.where(np.isnan(val), 0.0, val)
            lb = np.where(fixed, pin, lb)
            ub = np.where(fixed, pin, ub)
        return lb, ub

    def _gather_dirty_cols(self):
        idx_parts, lo_parts, up_parts = [], [], []
        for v, off, n in self._var_blocks:
            d = v.pop_dirty_bounds()
            if d is not None and len(d) == 0:
                continue
            lb, ub = self._var_effective_bounds(v)
            if d is None:
                sel = np.arange(n, dtype=np.int64)
            else:
                sel = d
            idx_parts.append(off + sel)
            lo_parts.append(lb[sel])
            up_parts.append(ub[sel])
        if not idx_parts:
            return None
        return (
            np.concatenate(idx_parts),
            np.concatenate(lo_parts),
            np.concatenate(up_parts),
        )

    def _gather_dirty_rows(self):
        idx_parts, lo_parts, up_parts = [], [], []
        for c, off, nr in self._row_blocks:
            d = c.pop_dirty_rows()
            if d is not None and len(d) == 0:
                continue
            lb, ub = c.effective_row_bounds()
            if d is None:
                sel = np.arange(nr, dtype=np.int64)
            else:
                sel = d
            idx_parts.append(off + sel)
            lo_parts.append(np.asarray(lb, dtype=np.float64)[sel])
            up_parts.append(np.asarray(ub, dtype=np.float64)[sel])
        if not idx_parts:
            return None
        return (
            np.concatenate(idx_parts),
            np.concatenate(lo_parts),
            np.concatenate(up_parts),
        )

    def _clip_inf(self, arr):
        out = np.array(arr, dtype=np.float64, copy=True)
        out[np.isneginf(out)] = -self._inf
        out[np.isposinf(out)] = self._inf
        return out

    def _push_updates(self):
        cols = self._gather_dirty_cols()
        if cols is not None:
            idx, lo, up = cols
            self._highs.changeColsBounds(
                len(idx), idx.astype(np.int32), self._clip_inf(lo), self._clip_inf(up)
            )
        rows = self._gather_dirty_rows()
        if rows is not None:
            idx, lo, up = rows
            self._highs.changeRowsBounds(
                len(idx), idx.astype(np.int32), self._clip_inf(lo), self._clip_inf(up)
            )

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve(
        self, *, keep_basis=True, load_solutions=True, update=True, check_structure=True
    ):
        """Warm re-solve: push the dirty updates, keep the basis, solve.

        Parameters
        ----------
        keep_basis : bool
            Retain the warm simplex basis (default).  ``False`` clears the solver
            for a cold re-solve.
        load_solutions : bool
            Scatter the primal solution back into each VectorVar's value array.
        update : bool
            Expand and push the dirty bounds/rows before solving (default).
        check_structure : bool
            Verify the shape fingerprint before solving (default).
        """
        import highspy

        if check_structure and self._n_solves > 0:
            self._check_structure()
        if update:
            self._push_updates()
        if not keep_basis:
            self._highs.clearSolver()
        self._highs.run()
        self._n_solves += 1

        status = self._highs.getModelStatus()
        info = self._highs.getInfo()
        optimal = status == highspy.HighsModelStatus.kOptimal
        has_feasible = info.primal_solution_status == 2
        termination = (
            'optimal' if optimal else ('feasible' if has_feasible else 'noSolution')
        )
        objective = info.objective_function_value if has_feasible else None

        col_value = None
        if has_feasible:
            col_value = np.array(self._highs.getSolution().col_value, dtype=np.float64)
            if load_solutions:
                self._load_solution(col_value)
        self._last = VectorSolveResult(status, termination, objective, col_value)
        return self._last

    def _load_solution(self, col_value):
        """Scatter the primal solution into each VectorVar's value array.

        Writes the arrays directly (not via ``set_values``) so the read-back does
        not re-dirty columns -- the solution is consistent with the pushed data.
        """
        for v, off, n in self._var_blocks:
            v._value_arr[:] = col_value[off : off + n]

    # ------------------------------------------------------------------ #
    @property
    def objective(self):
        return None if self._last is None else self._last.objective

    @property
    def highs(self):
        return self._highs

    @property
    def n_solves(self):
        return self._n_solves
