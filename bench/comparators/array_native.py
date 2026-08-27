"""Array-native comparators: the same LP built directly with numpy/scipy.

This is the tightest apples-to-apples baseline for the vectorization project: it
constructs *exactly the same* linear program as the Pyomo generator, but as
scipy sparse matrices assembled with vectorized numpy (Kronecker structure, no
per-index Python object), then loads it straight into a solver.  The gap between
this and the Pyomo construct+repn+write+load total is the pure overhead the fast
path (scoping doc §6.3/§6.4) aims to remove.

Two loaders:
  * ``load_highs`` - ``highspy.Highs.passModel`` from a CSC matrix.  No license
    limit, so it runs at every size (the full-scale array-native datapoint).
  * ``load_gurobi`` - gurobipy matrix API (``addMVar`` + ``addMConstr``).  The
    size-limited license caps this at ~2000 vars/constrs, so it runs on ``xs``
    only; that is the explicit "gurobipy matrix API" datapoint the plan asks for.

Only the two cleanly-vectorizable synthetics are implemented here (network flow
and facility location); the ragged supply-chain and the integer-heavy unit
commitment are Pyomo-only in Phase 0 (see the report for rationale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp

SUPPORTED = {"network_flow", "facility_location"}


@dataclass
class Matrices:
    """A linear (optionally integer) program in matrix form."""

    n_var: int
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    A_eq: sp.csr_matrix
    b_eq: np.ndarray
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    integrality: Optional[np.ndarray] = None  # bool mask, True = integer/binary
    # Canonical column names, one per variable, matching the Pyomo generator's
    # VarData names (e.g. "flow[0,1,3]", "open[2]").  Used by bench.equivalence to
    # align these columns with the Pyomo standard form by variable identity.
    col_names: Optional[list] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_con(self) -> int:
        return self.A_eq.shape[0] + self.A_ub.shape[0]

    @property
    def nnz(self) -> int:
        return int(self.A_eq.nnz + self.A_ub.nnz)


# --------------------------------------------------------------------------- #
# Matrix builders (vectorized, matching the Pyomo generators)
# --------------------------------------------------------------------------- #
def build_network_flow(params: Dict[str, Any]) -> Matrices:
    N = int(params["N"])
    T = int(params["T"])
    arcs = [(i, j) for i in range(N) for j in range(N) if i != j]
    A = len(arcs)
    arc_cap = 5.0 * N
    stor_cap = 10.0 * N

    # Node-arc incidence B (N x A): +1 if arc ends at n, -1 if arc starts at n.
    starts = np.array([i for (i, j) in arcs])
    ends = np.array([j for (i, j) in arcs])
    ka = np.arange(A)
    B = sp.coo_matrix(
        (
            np.concatenate([np.ones(A), -np.ones(A)]),
            (
                np.concatenate([ends, starts]),
                np.concatenate([ka, ka]),
            ),
        ),
        shape=(N, A),
    ).tocsr()

    I_T = sp.identity(T, format="csr")
    # Storage time-coupling D (T x T): row t has -1 at t and +1 at t-1.
    d_rows = np.concatenate([np.arange(T), np.arange(1, T)])
    d_cols = np.concatenate([np.arange(T), np.arange(0, T - 1)])
    d_val = np.concatenate([-np.ones(T), np.ones(T - 1)])
    D = sp.coo_matrix((d_val, (d_rows, d_cols)), shape=(T, T)).tocsr()
    I_N = sp.identity(N, format="csr")

    A_flow = sp.kron(B, I_T, format="csr")          # (N*T) x (A*T)
    A_stor = sp.kron(I_N, D, format="csr")          # (N*T) x (N*T)
    A_eq = sp.hstack([A_flow, A_stor], format="csr")

    # RHS demand, row layout (n-major, t-minor).  Import from the Pyomo model so
    # the two builders provably encode the same LP.
    from bench.models.network_flow import _demand

    b_eq = np.array([_demand(n, t, N) for n in range(N) for t in range(T)], dtype=float)

    n_flow = A * T
    n_stor = N * T
    n_var = n_flow + n_stor
    lb = np.zeros(n_var)
    ub = np.concatenate([np.full(n_flow, arc_cap), np.full(n_stor, stor_cap)])

    # Objective: arc cost repeated over T, storage 0.1.
    arc_cost = np.array([1.0 + ((i * 7 + j * 13) % 5) for (i, j) in arcs])
    c_flow = np.repeat(arc_cost, T)
    c = np.concatenate([c_flow, np.full(n_stor, 0.1)])

    # Column names matching the Pyomo generator's VarData names: flow is
    # arc-major/period-minor (col a*T + t), then stor node-major/period-minor.
    col_names = [f"flow[{i},{j},{t}]" for (i, j) in arcs for t in range(T)]
    col_names += [f"stor[{n},{t}]" for n in range(N) for t in range(T)]

    empty_ub = sp.csr_matrix((0, n_var))
    return Matrices(
        n_var=n_var,
        c=c,
        lb=lb,
        ub=ub,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=empty_ub,
        b_ub=np.zeros(0),
        integrality=None,
        col_names=col_names,
        meta={"N": N, "T": T, "arcs": A},
    )


def build_facility_location(params: Dict[str, Any]) -> Matrices:
    F = int(params["F"])
    C = int(params["C"])

    cap = np.array([3.0 + (f % 5) for f in range(F)], dtype=float)
    total_cap = cap.sum()
    scale = (1.3 * C) / total_cap if total_cap else 1.0
    cap = cap * scale

    # Variable layout: open[0..F-1], then x[f,c] at F + f*C + c.
    n_open = F
    n_x = F * C
    n_var = n_open + n_x

    ff, cc = np.meshgrid(np.arange(F), np.arange(C), indexing="ij")
    ff = ff.ravel()  # length F*C, f-major
    cc = cc.ravel()
    x_col = n_open + ff * C + cc  # global column of x[f,c]

    # serve (C rows): sum_f x[f,c] == 1.
    serve_rows = cc  # row = c
    serve_cols = x_col
    serve_val = np.ones(n_x)
    A_serve = sp.coo_matrix((serve_val, (serve_rows, serve_cols)), shape=(C, n_var)).tocsr()
    b_eq = np.ones(C)

    # link (F*C rows): x[f,c] - open[f] <= 0.
    link_row = np.arange(n_x)
    rows = np.concatenate([link_row, link_row])
    cols = np.concatenate([x_col, ff])            # x[f,c] col, open[f] col
    vals = np.concatenate([np.ones(n_x), -np.ones(n_x)])
    A_link = sp.coo_matrix((vals, (rows, cols)), shape=(n_x, n_var)).tocsr()

    # capacity (F rows): sum_c x[f,c] - cap[f]*open[f] <= 0.
    cap_rows = np.concatenate([ff, np.arange(F)])
    cap_cols = np.concatenate([x_col, np.arange(F)])
    cap_vals = np.concatenate([np.ones(n_x), -cap])
    A_cap = sp.coo_matrix((cap_vals, (cap_rows, cap_cols)), shape=(F, n_var)).tocsr()

    A_ub = sp.vstack([A_link, A_cap], format="csr")
    b_ub = np.zeros(n_x + F)

    lb = np.zeros(n_var)
    ub = np.concatenate([np.ones(n_open), np.ones(n_x)])  # open binary in [0,1], x in [0,1]

    open_cost = np.array([50.0 + 5.0 * (f % 6) for f in range(F)])
    assign_cost = np.array([1.0 + ((f * 3 + c * 7) % 11) for f in range(F) for c in range(C)])
    c_vec = np.concatenate([open_cost, assign_cost])

    integrality = np.zeros(n_var, dtype=bool)
    integrality[:n_open] = True  # open[f] binary

    # Column names matching the Pyomo generator: open[f] first, then x[f,c]
    # f-major/c-minor (col n_open + f*C + c).
    col_names = [f"open[{f}]" for f in range(F)]
    col_names += [f"x[{f},{c}]" for f in range(F) for c in range(C)]

    return Matrices(
        n_var=n_var,
        c=c_vec,
        lb=lb,
        ub=ub,
        A_eq=A_serve,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        integrality=integrality,
        col_names=col_names,
        meta={"F": F, "C": C},
    )


BUILDERS = {
    "network_flow": build_network_flow,
    "facility_location": build_facility_location,
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_highs(mx: Matrices):
    """Load the matrices into an in-process HiGHS via ``passModel`` (all sizes)."""
    import highspy

    inf = highspy.kHighsInf
    A = sp.vstack([mx.A_eq, mx.A_ub], format="csc")
    n_eq = mx.A_eq.shape[0]
    n_ub = mx.A_ub.shape[0]
    row_lower = np.concatenate([mx.b_eq, np.full(n_ub, -inf)])
    row_upper = np.concatenate([mx.b_eq, mx.b_ub])

    lp = highspy.HighsLp()
    lp.num_col_ = mx.n_var
    lp.num_row_ = n_eq + n_ub
    lp.col_cost_ = mx.c.astype(np.float64)
    lp.col_lower_ = mx.lb.astype(np.float64)
    lp.col_upper_ = mx.ub.astype(np.float64)
    lp.row_lower_ = row_lower.astype(np.float64)
    lp.row_upper_ = row_upper.astype(np.float64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.a_matrix_.num_col_ = mx.n_var
    lp.a_matrix_.num_row_ = n_eq + n_ub
    if mx.integrality is not None and mx.integrality.any():
        vt = np.where(
            mx.integrality,
            int(highspy.HighsVarType.kInteger),
            int(highspy.HighsVarType.kContinuous),
        )
        lp.integrality_ = [highspy.HighsVarType(int(v)) for v in vt]

    h = highspy.Highs()
    h.silent()
    status = h.passModel(lp)
    return h


def load_gurobi(mx: Matrices):
    """Load the matrices into gurobipy via the matrix API (xs only; size-limited)."""
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model()
    m.Params.OutputFlag = 0
    if mx.integrality is not None and mx.integrality.any():
        vtype = np.where(mx.integrality, GRB.BINARY, GRB.CONTINUOUS)
        x = m.addMVar(mx.n_var, lb=mx.lb, ub=mx.ub, vtype=list(vtype))
    else:
        x = m.addMVar(mx.n_var, lb=mx.lb, ub=mx.ub)
    if mx.A_eq.shape[0]:
        m.addMConstr(mx.A_eq, x, "=", mx.b_eq)
    if mx.A_ub.shape[0]:
        m.addMConstr(mx.A_ub, x, "<", mx.b_ub)
    m.setObjective(mx.c @ x, GRB.MINIMIZE)
    m.update()  # force model extraction (the "load" cost)
    return m
