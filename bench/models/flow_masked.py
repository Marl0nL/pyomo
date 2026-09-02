# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Dense multi-period network flow with a *vectorizable* right-hand side.

This is ``network_flow`` unchanged in structure -- a complete directed graph on
``N`` nodes (``N*(N-1)`` arcs) over ``T`` periods, with per-node storage linking
consecutive periods -- but the per-node net demand is **precomputed into a
mutable ``Param``** instead of being computed inside the rule.  That single
change is what lets the rule templatize: the balance body is now

    sum(m.flow[j, n, t] for j in m.nodes if j != n)     # filtered sum   (3b)
  - sum(m.flow[n, j, t] for j in m.nodes if j != n)     # filtered sum   (3b)
  + (m.stor[n, t - 1] if t > 0 else 0.0)                # conditional    (3b)
  - m.stor[n, t] == m.demand[n, t]                       # Param RHS      (Phase 3)

Every piece is inside the Phase-3b-proven subset, so the whole family
vectorizes.  ``network_flow`` (kept unchanged in the suite) is byte-identical to
this model *except* that it evaluates the demand -- ``1 + 0.25*sin(0.3*t + n)``,
a transcendental Python function of the index -- inside the rule, which raises
during templatization (float() of an index expression) and is out of the 3b
scope.  So ``flow_masked`` isolates that in-rule transcendental as the *only*
reason ``network_flow`` itself does not vectorize.

Built the idiomatic Pyomo way (indexed ``Var`` + ``Constraint(index, rule=...)``);
the demand values are identical to ``network_flow``'s so the two models have the
same feasible flow structure.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pyomo.environ as pyo

NAME = "flow_masked"
DESCRIPTION = "Multi-period min-cost flow, filtered-sum + conditional body (3b)."
HAS_QUADRATIC = False

# Same sizing as network_flow (nnz ~= 2*N^2*T) so the two are directly comparable.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"N": 8, "T": 6},
    "1e4": {"N": 10, "T": 50},
    "1e5": {"N": 20, "T": 125},
    "1e6": {"N": 50, "T": 200},
    "1e7": {"N": 100, "T": 500},
}


def _sink_demand(n: int, t: int) -> float:
    return 1.0 + 0.25 * math.sin(0.3 * t + n)


def _demand(n: int, t: int, N: int) -> float:
    """Deterministic net demand that sums to zero across nodes each period.

    Identical to ``network_flow._demand`` -- but here it is evaluated in Python
    at *build* time to fill a Param, never inside the rule.
    """
    if n == 0:
        return -sum(_sink_demand(k, t) for k in range(1, N))
    return _sink_demand(n, t)


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    N = int(params["N"])
    T = int(params["T"])
    m = pyo.ConcreteModel(name=f"flow_masked_{N}x{T}")

    m.nodes = pyo.RangeSet(0, N - 1)
    m.periods = pyo.RangeSet(0, T - 1)
    m.arcs = pyo.Set(
        initialize=[(i, j) for i in range(N) for j in range(N) if i != j], dimen=2
    )

    arc_cap = 5.0 * N
    stor_cap = 10.0 * N

    m.flow = pyo.Var(
        m.arcs, m.periods, domain=pyo.NonNegativeReals, bounds=(0, arc_cap)
    )
    m.stor = pyo.Var(
        m.nodes, m.periods, domain=pyo.NonNegativeReals, bounds=(0, stor_cap)
    )

    # The net demand, precomputed into a mutable Param (the vectorizable RHS).
    m.demand = pyo.Param(
        m.nodes,
        m.periods,
        initialize={
            (n, t): _demand(n, t, N) for n in range(N) for t in range(T)
        },
        mutable=True,
    )

    def balance_rule(m, n, t):
        inflow = sum(m.flow[j, n, t] for j in m.nodes if j != n)
        outflow = sum(m.flow[n, j, t] for j in m.nodes if j != n)
        prev = m.stor[n, t - 1] if t > 0 else 0.0
        return inflow - outflow + prev - m.stor[n, t] == m.demand[n, t]

    m.balance = pyo.Constraint(m.nodes, m.periods, rule=balance_rule)

    def arc_cost(i, j):
        return 1.0 + ((i * 7 + j * 13) % 5)

    m.obj = pyo.Objective(
        expr=sum(
            arc_cost(i, j) * m.flow[i, j, t] for (i, j) in m.arcs for t in m.periods
        )
        + 0.1 * sum(m.stor[n, t] for n in m.nodes for t in m.periods),
        sense=pyo.minimize,
    )
    return m
