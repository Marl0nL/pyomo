# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Direct HiGHS array hand-off for vectorized models (the load prize).

Phase-0 measured solver *load* as 48-80% of the classic time-to-solver -- the
single biggest stage -- because APPSI HiGHS extracts the model one row at a
time (#3888).  The fast path instead builds the standard-form arrays once
(:func:`pyomo.contrib.vector.matrices.assemble`) and hands them to HiGHS in bulk
via ``Highs.passModel`` -- exactly the ``array->HiGHS`` ceiling the Phase-0
harness measured (185 ms vs 11.6 s at 1e6 nnz).

The vectorized model is loaded as *range rows* (``row_lower <= A x <=
row_upper``), which HiGHS supports natively, so there is no per-constraint row
splitting.
"""

from __future__ import annotations

from pyomo.common.dependencies import numpy as np, scipy

from pyomo.contrib.vector.matrices import assemble, VectorMatrices, _collect


def matrices_to_highs_lp(mx: VectorMatrices):
    """Build a ``highspy.HighsLp`` from assembled :class:`VectorMatrices`."""
    import highspy

    inf = highspy.kHighsInf

    col_lower = mx.col_lower.astype(np.float64).copy()
    col_upper = mx.col_upper.astype(np.float64).copy()
    # Fixed variables: pin the column to its value (equivalent feasible region).
    if mx.col_fixed.any():
        vals = np.where(np.isnan(mx.col_value), 0.0, mx.col_value)
        col_lower = np.where(mx.col_fixed, vals, col_lower)
        col_upper = np.where(mx.col_fixed, vals, col_upper)
    col_lower = np.where(np.isneginf(col_lower), -inf, col_lower)
    col_upper = np.where(np.isposinf(col_upper), inf, col_upper)

    r_lower = mx.row_lower
    r_upper = mx.row_upper
    # Masked-out (deactivated) rows are relaxed to (-inf, +inf): vacuous, never
    # binding -- the persistent-path equivalent of dropping the row.
    if not mx.row_active.all():
        r_lower = np.where(mx.row_active, r_lower, -np.inf)
        r_upper = np.where(mx.row_active, r_upper, np.inf)
    row_lower = np.where(np.isneginf(r_lower), -inf, r_lower).astype(np.float64)
    row_upper = np.where(np.isposinf(r_upper), inf, r_upper).astype(np.float64)

    A = mx.A.tocsc()

    c = mx.c.astype(np.float64)
    maximize = str(mx.sense) == 'maximize' or int(mx.sense) == -1

    lp = highspy.HighsLp()
    lp.num_col_ = mx.n_var
    lp.num_row_ = mx.n_row
    lp.col_cost_ = c
    # Tell HiGHS the sense directly (rather than negating the cost), so it reports
    # the true objective and -- for a QP -- checks convexity for the right sense
    # (concave Hessian for maximize).
    lp.sense_ = (
        highspy.ObjSense.kMaximize if maximize else highspy.ObjSense.kMinimize
    )
    lp.offset_ = float(mx.c_offset)
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.a_matrix_.num_col_ = mx.n_var
    lp.a_matrix_.num_row_ = mx.n_row
    if mx.integrality.any():
        vt = np.where(
            mx.integrality,
            int(highspy.HighsVarType.kInteger),
            int(highspy.HighsVarType.kContinuous),
        )
        lp.integrality_ = [highspy.HighsVarType(int(v)) for v in vt]
    return lp


class QuadraticModelError(Exception):
    """Raised when a quadratic model cannot be handed to HiGHS (MIQP / etc.)."""


def _hessian_to_highs(Hl, n_var):
    """Build a ``highspy.HighsHessian`` from a lower-triangular CSC Hessian.

    The Hessian carries the true objective sign; ``matrices_to_highs_lp`` sets
    ``lp.sense_`` (rather than negating the cost), so HiGHS applies the sense to
    both the linear and quadratic parts and checks convexity for that sense.
    """
    import highspy

    hess = highspy.HighsHessian()
    hess.dim_ = n_var
    hess.format_ = highspy.HessianFormat.kTriangular
    hess.start_ = Hl.indptr.astype(np.int32)
    hess.index_ = Hl.indices.astype(np.int32)
    hess.value_ = Hl.data.astype(np.float64)
    return hess


def matrices_to_highs_model(mx: VectorMatrices):
    """Build a ``highspy`` model object to feed ``passModel``.

    Returns a bare ``HighsLp`` for a linear model, or a ``HighsModel`` carrying
    both the LP and the objective Hessian for a convex-QP model.  A MIQP
    (integer variable + quadratic objective) is rejected loudly: HiGHS cannot
    solve MIQP (verified empirically), so this never silently mis-solves.
    """
    import highspy

    lp = matrices_to_highs_lp(mx)
    if not mx.is_quadratic:
        return lp
    if mx.integrality.any():
        raise QuadraticModelError(
            "The vector fast path received a quadratic objective together with "
            "integer/binary variables (MIQP).  HiGHS cannot solve MIQP problems; "
            "use a MIQP-capable solver (e.g. Gurobi) for this model."
        )
    model = highspy.HighsModel()
    model.lp_ = lp
    model.hessian_ = _hessian_to_highs(mx.hessian, mx.n_var)
    return model


def load_highs(model):
    """Assemble ``model`` and load it into an in-process HiGHS via ``passModel``.

    Returns the ``highspy.Highs`` instance (model loaded, not solved).  Handles a
    convex-quadratic :class:`~pyomo.contrib.vector.objective.VectorObjective` by
    also passing its Hessian.
    """
    import highspy

    mx = assemble(model)
    m = matrices_to_highs_model(mx)
    h = highspy.Highs()
    h.silent()
    h.passModel(m)
    return h


def load_solution(model, highs):
    """Scatter a solved HiGHS solution back into the model's VectorVar arrays.

    The array-native read path (scoping doc Phase 2): the primal ``col_value``
    vector is split by the same column blocks :func:`assemble` uses and written
    straight into each :class:`~pyomo.contrib.vector.var.VectorVar`'s value
    array in bulk -- ``m.x.value_array`` (and ``m.x[i].value``) then read the
    solution with no per-index solver query.  Returns the number of columns
    loaded.
    """
    vvars, _, _ = _collect(model)
    col_value = np.array(highs.getSolution().col_value, dtype=np.float64)
    off = 0
    for v in vvars:
        n = v.n
        v._value_arr[:] = col_value[off:off + n]
        off += n
    return off


def info_has_feasible(highs):
    """True when HiGHS holds a feasible primal solution to read back."""
    return highs.getInfo().primal_solution_status == 2


def solve_highs(model, load_solutions=False):
    """Convenience: assemble, load, and solve; returns ``(highs, objective)``.

    With ``load_solutions=True`` the primal solution is scattered back into the
    model's VectorVar value arrays (:func:`load_solution`).

    A non-convex QP is surfaced clearly: HiGHS refuses a non-PSD Hessian at
    ``run`` time, which this reports rather than returning a bogus objective.
    """
    import highspy

    mx = assemble(model)
    m = matrices_to_highs_model(mx)
    h = highspy.Highs()
    h.silent()
    h.passModel(m)
    status = h.run()
    if mx.is_quadratic and str(status) != 'HighsStatus.kOk':
        raise QuadraticModelError(
            "HiGHS could not solve the quadratic model (status "
            f"{status}, model status {h.getModelStatus()}).  The most common "
            "cause is a non-convex objective Hessian (HiGHS solves convex QP "
            "only): for a minimize objective the Hessian must be positive "
            "semidefinite (negative semidefinite for maximize)."
        )
    if load_solutions and info_has_feasible(h):
        load_solution(model, h)
    info = h.getInfo()
    return h, info.objective_function_value
