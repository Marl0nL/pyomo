"""Dense multi-period network flow.

A complete directed graph on ``N`` nodes (so ``N*(N-1)`` arcs) over ``T`` time
periods, with per-node storage linking consecutive periods.  "Dense" means each
flow-balance constraint touches ~``2(N-1)`` flow variables, so the constraint
matrix has ~``2*N^2*T`` nonzeros — the high-degree regime where per-index tree
building and the interpreted repn walk hurt most.

Built the idiomatic Pyomo way (indexed ``Var`` + ``Constraint(index, rule=...)``)
so the baseline reflects the operator-overloading + per-index dispatch path the
vectorization project targets, not a hand-optimized ``LinearExpression``.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pyomo.environ as pyo

NAME = "network_flow"
DESCRIPTION = "Dense multi-period min-cost network flow (complete digraph)."
HAS_QUADRATIC = False

# Presets chosen so the linear constraint-matrix nonzeros land near the named
# target (nnz ~= 2*N^2*T).  ``xs`` is sized under gurobipy's 2000-var/-constr
# size-limited-license cap so the cross-system comparators can run on it.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"N": 8, "T": 6},        # ~672 nnz, ~432 vars, ~48 balance cons
    "1e4": {"N": 10, "T": 50},     # ~10k nnz
    "1e5": {"N": 20, "T": 125},    # ~100k nnz
    "1e6": {"N": 50, "T": 200},    # ~1.0M nnz
    "1e7": {"N": 100, "T": 500},   # ~10M nnz  (heavy: ~5M vars)
}


def _sink_demand(n: int, t: int) -> float:
    """Positive net demand at a sink node (n >= 1)."""
    return 1.0 + 0.25 * math.sin(0.3 * t + n)


def _demand(n: int, t: int, N: int) -> float:
    """Deterministic net demand that sums to zero across nodes each period.

    Node 0 is the single source; its supply is exactly the negated sum of the
    sink demands in the same period, so the flow problem is balanced and feasible
    (complete graph + ample arc capacity route it directly).  The RHS values do
    not affect construct/repn/write/load timing - the constraint structure is
    identical - they only make the small-size solve well-posed.
    """
    if n == 0:
        return -sum(_sink_demand(k, t) for k in range(1, N))
    return _sink_demand(n, t)


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    N = int(params["N"])
    T = int(params["T"])
    m = pyo.ConcreteModel(name=f"network_flow_{N}x{T}")

    m.nodes = pyo.RangeSet(0, N - 1)
    m.periods = pyo.RangeSet(0, T - 1)
    m.arcs = pyo.Set(
        initialize=[(i, j) for i in range(N) for j in range(N) if i != j],
        dimen=2,
    )

    arc_cap = 5.0 * N
    stor_cap = 10.0 * N

    m.flow = pyo.Var(m.arcs, m.periods, domain=pyo.NonNegativeReals, bounds=(0, arc_cap))
    m.stor = pyo.Var(m.nodes, m.periods, domain=pyo.NonNegativeReals, bounds=(0, stor_cap))

    def balance_rule(m, n, t):
        inflow = sum(m.flow[j, n, t] for j in m.nodes if j != n)
        outflow = sum(m.flow[n, j, t] for j in m.nodes if j != n)
        prev = m.stor[n, t - 1] if t > 0 else 0.0
        return inflow - outflow + prev - m.stor[n, t] == _demand(n, t, N)

    m.balance = pyo.Constraint(m.nodes, m.periods, rule=balance_rule)

    def arc_cost(i, j):
        return 1.0 + ((i * 7 + j * 13) % 5)

    m.obj = pyo.Objective(
        expr=sum(arc_cost(i, j) * m.flow[i, j, t] for (i, j) in m.arcs for t in m.periods)
        + 0.1 * sum(m.stor[n, t] for n in m.nodes for t in m.periods),
        sense=pyo.minimize,
    )
    return m
