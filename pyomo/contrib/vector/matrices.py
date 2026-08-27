# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Assemble a whole vectorized model into standard-form arrays.

This is the ``LinearStandardFormCompiler`` splice of the scoping document
(§6.3): it collects the model's :class:`VectorVar` / :class:`VectorConstraint` /
:class:`VectorObjective` components and stacks their CSR arrays straight into a
single standard-form representation -- no per-index expression walk.

Two products come out of one assembly:

* :func:`assemble` builds :class:`VectorMatrices` -- the raw stacked arrays
  (range-row form ``lb <= A x <= ub``, column bounds, objective cost).  This is
  the fast path used by the direct HiGHS ``passModel`` hand-off (the "load
  prize", scoping doc §6.4).  It materializes no per-index Python object.

* :func:`compile_standard_form` converts those arrays into a
  :class:`LinearStandardFormInfo`-shaped object (the ``mixed_form`` row split,
  fixed-variable substitution and unused-column elimination performed exactly as
  the stock :class:`LinearStandardFormCompiler` does), so it can be compared to
  the classic path by the benchmark's standard-form equivalence oracle
  (Phase-1 correctness gate).
"""

from __future__ import annotations

import collections

from pyomo.common.dependencies import numpy as np, scipy
from pyomo.common.enums import ObjectiveSense
from pyomo.core.base import Var, Constraint, Objective

from pyomo.contrib.vector.var import VectorVar
from pyomo.contrib.vector.constraint import VectorConstraint
from pyomo.contrib.vector.objective import VectorObjective

_inf = float('inf')
_ninf = -_inf

RowEntry = collections.namedtuple('RowEntry', ['constraint', 'bound_type'])


class VectorPathDisabledError(Exception):
    """Raised when the fast path cannot be used (mixed / scalarized model)."""


class VectorMatrices:
    """Raw stacked standard-form arrays for a vectorized model.

    Attributes
    ----------
    n_var : int
    col_lower, col_upper : np.ndarray (float64, +/- inf for unbounded)
    integrality : np.ndarray (bool)
    col_value, col_fixed : np.ndarray  (fixed-var value / mask)
    A : scipy.sparse.csr_array   (n_row x n_var)
    row_lower, row_upper : np.ndarray (range-row bounds; +/- inf for open side)
    c : np.ndarray (float64, length n_var)
    c_offset : float
    sense : ObjectiveSense
    var_blocks : list[(VectorVar, offset, n)]
    row_blocks : list[(VectorConstraint, offset, nrows)]
    """

    def __init__(self, n_var, col_lower, col_upper, integrality, col_value,
                 col_fixed, A, row_lower, row_upper, c, c_offset, sense,
                 var_blocks, row_blocks, hessian=None, row_active=None):
        self.n_var = n_var
        self.col_lower = col_lower
        self.col_upper = col_upper
        self.integrality = integrality
        self.col_value = col_value
        self.col_fixed = col_fixed
        self.A = A
        self.row_lower = row_lower
        self.row_upper = row_upper
        # Per-row active mask (masked deactivation): True == enforced.  A
        # deactivated row is dropped by ``compile_standard_form`` (to match a
        # classic ``deactivate()``) and relaxed to (-inf,+inf) by the HiGHS
        # hand-off.  None == "all rows active".
        if row_active is None:
            row_active = np.ones(A.shape[0], dtype=bool)
        self.row_active = row_active
        self.c = c
        self.c_offset = c_offset
        self.sense = sense
        self.var_blocks = var_blocks
        self.row_blocks = row_blocks
        # Lower-triangular CSC of the full symmetric objective Hessian H
        # (``0.5 x'H x``) over the global column space, or None for a pure-linear
        # objective.  Sign is the true (unflipped) objective; the solver hand-off
        # flips it for a maximize objective.
        self.hessian = hessian

    @property
    def is_quadratic(self):
        return self.hessian is not None and self.hessian.nnz > 0

    @property
    def n_row(self):
        return self.A.shape[0]

    @property
    def nnz(self):
        return int(self.A.nnz)

    # -- lazy column / row identity (only needed for naming / the oracle) --- #
    def column_var_and_index(self, global_col):
        for v, off, n in self.var_blocks:
            if off <= global_col < off + n:
                return v, v.index_at(global_col - off)
        raise IndexError(global_col)

    def row_con_and_local(self, global_row):
        for con, off, nr in self.row_blocks:
            if off <= global_row < off + nr:
                return con, global_row - off
        raise IndexError(global_row)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def _collect(model):
    vvars, vcons, vobjs = [], [], []
    scalarized = []
    for v in model.component_objects(Var, active=True, descend_into=True):
        if isinstance(v, VectorVar):
            if v._scalarized:
                scalarized.append(v)
            vvars.append(v)
    for c in model.component_objects(Constraint, active=True, descend_into=True):
        if isinstance(c, VectorConstraint):
            if not c.active:
                continue
            if c._scalarized:
                scalarized.append(c)
            vcons.append(c)
        else:
            raise VectorPathDisabledError(
                f"Constraint '{c.name}' is not a VectorConstraint.  Mixed "
                "classic+vector models are not supported on the Phase-1 fast "
                "path; use the classic LinearStandardFormCompiler."
            )
    for o in model.component_objects(Objective, active=True, descend_into=True):
        if isinstance(o, VectorObjective):
            vobjs.append(o)
        else:
            raise VectorPathDisabledError(
                f"Objective '{o.name}' is not a VectorObjective."
            )
    if scalarized:
        names = ", ".join(sorted(s.name for s in scalarized))
        raise VectorPathDisabledError(
            "The fast path is disabled for scalarized component(s): "
            f"{names}.  Once a vectorized component has been scalarized "
            "(scoping doc §6.5) use the classic LinearStandardFormCompiler."
        )
    return vvars, vcons, vobjs


# --------------------------------------------------------------------------- #
# Fast assembly (no per-index Python objects)
# --------------------------------------------------------------------------- #
def assemble(model):
    """Build :class:`VectorMatrices` from a pure-vector model (the fast path)."""
    vvars, vcons, vobjs = _collect(model)

    # --- columns -------------------------------------------------------- #
    offset_of = {}
    off = 0
    col_lower_parts, col_upper_parts, integ_parts = [], [], []
    val_parts, fixed_parts, var_blocks = [], [], []
    for v in vvars:
        lb, ub = v.effective_bounds()
        col_lower_parts.append(lb)
        col_upper_parts.append(ub)
        integ_parts.append(v.integrality())
        val_parts.append(v.value_array)
        fixed_parts.append(v.fixed_array)
        offset_of[id(v)] = off
        var_blocks.append((v, off, v.n))
        off += v.n
    n_var = off
    if n_var:
        col_lower = np.concatenate(col_lower_parts)
        col_upper = np.concatenate(col_upper_parts)
        integrality = np.concatenate(integ_parts)
        col_value = np.concatenate(val_parts)
        col_fixed = np.concatenate(fixed_parts)
    else:
        col_lower = np.zeros(0)
        col_upper = np.zeros(0)
        integrality = np.zeros(0, dtype=bool)
        col_value = np.zeros(0)
        col_fixed = np.zeros(0, dtype=bool)

    # --- constraint matrix (stack, remapping local -> global columns) --- #
    data_parts, indices_parts, indptr = [], [], [0]
    row_lower_parts, row_upper_parts, row_active_parts, row_blocks = [], [], [], []
    roff = 0
    for con in vcons:
        A = con.A
        # Build a local-col -> global-col map: each xvar block maps to a
        # contiguous global range, so this is a vectorized gather.
        remap = np.empty(A.shape[1], dtype=np.int64)
        for b, xv in enumerate(con.xvars):
            lo = int(con.col_split[b])
            hi = int(con.col_split[b + 1])
            remap[lo:hi] = offset_of[id(xv)] + np.arange(hi - lo, dtype=np.int64)
        data_parts.append(A.data)
        indices_parts.append(remap[A.indices])
        # shift this family's indptr onto the running total
        indptr.extend((A.indptr[1:] + indptr[-1]).tolist())
        row_lower_parts.append(con.row_lb)
        row_upper_parts.append(con.row_ub)
        row_active_parts.append(con.row_active)
        row_blocks.append((con, roff, con.nrows))
        roff += con.nrows
    n_row = roff
    if data_parts:
        data = np.concatenate(data_parts)
        indices = np.concatenate(indices_parts)
        indptr = np.asarray(indptr, dtype=np.int64)
        A = scipy.sparse.csr_array((data, indices, indptr), shape=(n_row, n_var))
        row_lower = np.concatenate(row_lower_parts)
        row_upper = np.concatenate(row_upper_parts)
        row_active = np.concatenate(row_active_parts)
    else:
        A = scipy.sparse.csr_array((n_row, n_var))
        row_lower = np.zeros(0)
        row_upper = np.zeros(0)
        row_active = np.ones(0, dtype=bool)

    # --- objective ------------------------------------------------------ #
    c = np.zeros(n_var, dtype=np.float64)
    c_offset = 0.0
    sense = ObjectiveSense.minimize
    hessian = None
    if vobjs:
        if len(vobjs) > 1:
            raise VectorPathDisabledError("Multiple objectives are not supported.")
        obj = vobjs[0]
        for v, arr in obj.terms:
            o = offset_of[id(v)]
            c[o:o + v.n] += arr
        c_offset = obj.constant
        sense = obj.sense
        if obj.is_quadratic():
            hessian = _assemble_hessian(obj, offset_of, n_var)

    return VectorMatrices(
        n_var, col_lower, col_upper, integrality, col_value, col_fixed,
        A, row_lower, row_upper, c, c_offset, sense, var_blocks, row_blocks,
        hessian, row_active,
    )


def _assemble_hessian(obj, offset_of, n_var):
    """Assemble the lower-triangular CSC Hessian for ``0.5 x'H x`` over globals.

    Each ``(vrow, vcol, block)`` term contributes to the full symmetric Hessian:
    a diagonal block ``(v, v)`` its symmetrized sub-block; an off-diagonal
    ``(vi, vj)`` the coupling ``block`` at ``(vi, vj)`` plus its transpose at
    ``(vj, vi)``.  Only the lower triangle is returned (the HiGHS format).
    """
    rows, cols, data = [], [], []
    for vr, vc, B in obj.quadratic_terms:
        orr = offset_of[id(vr)]
        occ = offset_of[id(vc)]
        if vr is vc:
            Bsym = (B + B.transpose()) * 0.5
            coo = Bsym.tocoo()
            rows.append(coo.row + orr)
            cols.append(coo.col + occ)
            data.append(coo.data)
        else:
            coo = B.tocoo()
            rows.append(coo.row + orr)
            cols.append(coo.col + occ)
            data.append(coo.data)
            rows.append(coo.col + occ)
            cols.append(coo.row + orr)
            data.append(coo.data)
    if not rows:
        return None
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    d = np.concatenate(data)
    H = scipy.sparse.coo_matrix((d, (r, c)), shape=(n_var, n_var)).tocsr()
    H.sum_duplicates()
    Hl = scipy.sparse.tril(H).tocsc()
    Hl.sort_indices()
    Hl.eliminate_zeros()
    return Hl


# --------------------------------------------------------------------------- #
# Standard-form (mixed_form) for the equivalence oracle
# --------------------------------------------------------------------------- #
class VectorStandardFormInfo:
    """Mirror of :class:`pyomo.repn.plugins.standard_form.LinearStandardFormInfo`.

    Attributes match the stock compiler so the benchmark's equivalence oracle
    can compare the two directly: ``c`` (csc), ``c_offset``, ``A`` (csc),
    ``rhs``, ``rows`` (``RowEntry(constraint_data, +/-1|0)``), ``columns``
    (``VarData`` list), ``objectives``, ``eliminated_vars``.
    """

    def __init__(self, c, c_offset, A, rhs, rows, columns, objectives,
                 eliminated_vars):
        self.c = c
        self.c_offset = c_offset
        self.A = A
        self.rhs = rhs
        self.rows = rows
        self.columns = columns
        self.objectives = objectives
        self.eliminated_vars = eliminated_vars

    @property
    def x(self):
        return self.columns

    @property
    def b(self):
        return self.rhs


def compile_standard_form(model, mixed_form=True, set_sense=ObjectiveSense.minimize):
    """Compile a pure-vector model to a stock-compatible standard form.

    Replicates the stock compiler's behaviour (mixed-form row split,
    fixed-variable substitution, unused-column elimination) so the result is
    comparable up to row/column permutation.  Materializes the surviving
    columns/rows as classic data objects for naming (test-scale only).
    """
    if not mixed_form:
        raise NotImplementedError(
            "Phase 1 vector compiler supports mixed_form=True only."
        )
    from pyomo.common.errors import InfeasibleConstraintException

    mx = assemble(model)
    row_lb = mx.row_lower.astype(np.float64).copy()
    row_ub = mx.row_upper.astype(np.float64).copy()
    c_offset = float(mx.c_offset)
    Acsc = mx.A.tocsc()

    # Which columns actually appear (in A or in c)?  (Empty rows contribute no
    # entries, so this matches the stock compiler's active_var_mask.)
    col_nnz = np.diff(Acsc.indptr)
    active_mask = (col_nnz > 0) | (mx.c != 0.0)
    fixed_active = mx.col_fixed & active_mask

    # Fixed-variable substitution: move each fixed variable's contribution to
    # the row bounds / objective offset (scoping doc §6.3, the #3851 pitfall).
    if fixed_active.any():
        vals = np.where(np.isnan(mx.col_value), 0.0, mx.col_value)
        contrib = np.asarray(Acsc[:, fixed_active] @ vals[fixed_active]).ravel()
        row_lb = np.where(np.isfinite(row_lb), row_lb - contrib, row_lb)
        row_ub = np.where(np.isfinite(row_ub), row_ub - contrib, row_ub)
        c_offset += float(mx.c[fixed_active] @ vals[fixed_active])

    keep = active_mask & ~mx.col_fixed
    keep_cols = np.nonzero(keep)[0]
    A = Acsc[:, keep_cols].tocsr()
    c_keep = mx.c[keep_cols]

    # Masked-out (deactivated) rows are dropped entirely, matching a classic
    # ``con[r].deactivate()`` (the row disappears from the standard form).
    active = mx.row_active

    # After substitution/elimination, a row can be empty because it was
    # structurally empty OR because every one of its variables was fixed.  Both
    # are "constant constraints": drop them if feasible (body == 0), raise if
    # trivially infeasible -- exactly as the stock compiler does at repn time.
    # An empty row that is masked off never raises (it is not enforced).
    row_nnz = np.diff(A.indptr)
    empty = row_nnz == 0
    infeasible = empty & active & ((row_lb > 0.0) | (row_ub < 0.0))
    if infeasible.any():
        gi = int(np.nonzero(infeasible)[0][0])
        con, lr = mx.row_con_and_local(gi)
        raise InfeasibleConstraintException(
            f"model contains a trivially infeasible constraint, '{con.name}[{lr}]'"
        )
    keep_row = ~empty & active
    global_row_idx = np.nonzero(keep_row)[0]
    A = A[keep_row]
    row_lb = row_lb[keep_row]
    row_ub = row_ub[keep_row]

    if mx.sense is not None and set_sense is not None and set_sense != mx.sense:
        c_keep = -c_keep
        c_offset = -c_offset

    # -- mixed_form row split ------------------------------------------- #
    has_lb = np.isfinite(row_lb)
    has_ub = np.isfinite(row_ub)
    eq = has_lb & has_ub & (row_lb == row_ub)
    ub_rows = has_ub & ~eq
    lb_rows = has_lb & ~eq

    def _select(mask):
        idx = np.nonzero(mask)[0]
        return A[idx], idx

    A_eq, eq_idx = _select(eq)
    A_ubr, ub_idx = _select(ub_rows)
    A_lbr, lb_idx = _select(lb_rows)

    A_std = scipy.sparse.vstack([A_eq, A_ubr, A_lbr], format='csc') if (
        A_eq.shape[0] + A_ubr.shape[0] + A_lbr.shape[0]
    ) else scipy.sparse.csc_array((0, len(keep_cols)))
    rhs = np.concatenate([row_ub[eq_idx], row_ub[ub_idx], row_lb[lb_idx]])

    # Row metadata (materialize the owning constraint data objects).  The split
    # indices reference the *filtered* rows, so map back to the global row.
    rows = []
    for fi in eq_idx:
        con, lr = mx.row_con_and_local(int(global_row_idx[fi]))
        rows.append(RowEntry(con[lr], 0))
    for fi in ub_idx:
        con, lr = mx.row_con_and_local(int(global_row_idx[fi]))
        rows.append(RowEntry(con[lr], 1))
    for fi in lb_idx:
        con, lr = mx.row_con_and_local(int(global_row_idx[fi]))
        rows.append(RowEntry(con[lr], -1))

    # Columns (materialize VarData for each surviving global column).
    columns = []
    for gc in keep_cols:
        v, idx = mx.column_var_and_index(int(gc))
        columns.append(v[idx])

    # Objective as a (1 x n_cols) csc, matching the stock compiler.
    c_csc = scipy.sparse.csc_array(c_keep.reshape(1, -1)) if len(c_keep) else (
        scipy.sparse.csc_array((1, 0))
    )
    A_csc = A_std.tocsc()
    A_csc.sum_duplicates()
    A_csc.eliminate_zeros()

    objectives = list(model.component_objects(Objective, active=True))
    info = VectorStandardFormInfo(
        c_csc, np.array([c_offset]), A_csc, np.asarray(rhs),
        rows, columns, objectives, [],
    )
    return info
