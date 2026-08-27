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

from pyomo.contrib.vector.matrices import assemble, VectorMatrices


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

    row_lower = np.where(np.isneginf(mx.row_lower), -inf, mx.row_lower).astype(np.float64)
    row_upper = np.where(np.isposinf(mx.row_upper), inf, mx.row_upper).astype(np.float64)

    A = mx.A.tocsc()

    c = mx.c.astype(np.float64)
    if str(mx.sense) == 'maximize' or int(mx.sense) == -1:
        # HiGHS minimizes; flip a maximize objective.
        c = -c

    lp = highspy.HighsLp()
    lp.num_col_ = mx.n_var
    lp.num_row_ = mx.n_row
    lp.col_cost_ = c
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


def load_highs(model):
    """Assemble ``model`` and load it into an in-process HiGHS via ``passModel``.

    Returns the ``highspy.Highs`` instance (model loaded, not solved).
    """
    import highspy

    mx = assemble(model)
    lp = matrices_to_highs_lp(mx)
    h = highspy.Highs()
    h.silent()
    h.passModel(lp)
    return h


def solve_highs(model):
    """Convenience: assemble, load, and solve; returns ``(highs, objective)``."""
    h = load_highs(model)
    h.run()
    info = h.getInfo()
    return h, info.objective_function_value
