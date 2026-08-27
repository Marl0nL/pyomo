"""Capacitated facility location, with an optional quadratic variant.

``F`` candidate facilities, ``C`` customers.  Binary ``open[f]`` plus continuous
assignment fractions ``x[f,c]``.  Constraint nonzeros ~= ``4*F*C``:

  * demand      - every customer fully served:   ``sum_f x[f,c] == 1``   (dense in F)
  * linking     - can't assign to a closed site:  ``x[f,c] <= open[f]``   (2 terms)
  * capacity    - per-facility throughput cap:     ``sum_c x[f,c] <= cap*open[f]``

The quadratic variant (``quadratic=True``) adds a separable congestion cost
``sum_{f,c} q * x[f,c]^2`` to the objective, exercising the quadratic repn / LP
quadratic-objective writer path (scoping doc R7: quadratic is the project's hard
ceiling).
"""

from __future__ import annotations

from typing import Any, Dict

import pyomo.environ as pyo

NAME = "facility_location"
DESCRIPTION = "Capacitated facility location (MILP; optional quadratic objective)."
HAS_QUADRATIC = "quadratic"  # per-params flag; see build_pyomo

# nnz ~= 4*F*C.  ``xs`` under the gurobipy size-limit cap.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"F": 10, "C": 25},        # ~1000 nnz, 260 vars, 285 cons
    "1e4": {"F": 25, "C": 100},      # ~10k nnz
    "1e5": {"F": 50, "C": 500},      # ~100k nnz
    "1e6": {"F": 100, "C": 2500},    # ~1M nnz
    "1e7": {"F": 200, "C": 12500},   # ~10M nnz
}


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    F = int(params["F"])
    C = int(params["C"])
    quadratic = bool(params.get("quadratic", False))
    m = pyo.ConcreteModel(name=f"facility_location_{F}x{C}{'_q' if quadratic else ''}")

    m.facilities = pyo.RangeSet(0, F - 1)
    m.customers = pyo.RangeSet(0, C - 1)

    # Deterministic per-facility capacity, sized so all demand can be met.
    cap = {f: 3.0 + (f % 5) for f in range(F)}
    total_cap = sum(cap.values())
    # Scale so total capacity comfortably exceeds C units of demand.
    scale = (1.3 * C) / total_cap if total_cap else 1.0
    cap = {f: cap[f] * scale for f in range(F)}

    m.open = pyo.Var(m.facilities, domain=pyo.Binary)
    m.x = pyo.Var(m.facilities, m.customers, domain=pyo.NonNegativeReals, bounds=(0, 1))

    def serve_rule(m, c):
        return sum(m.x[f, c] for f in m.facilities) == 1.0

    m.serve = pyo.Constraint(m.customers, rule=serve_rule)

    def link_rule(m, f, c):
        return m.x[f, c] <= m.open[f]

    m.link = pyo.Constraint(m.facilities, m.customers, rule=link_rule)

    def cap_rule(m, f):
        return sum(m.x[f, c] for c in m.customers) <= cap[f] * m.open[f]

    m.capacity = pyo.Constraint(m.facilities, rule=cap_rule)

    def assign_cost(f, c):
        return 1.0 + ((f * 3 + c * 7) % 11)

    open_cost = {f: 50.0 + 5.0 * (f % 6) for f in range(F)}
    obj = sum(open_cost[f] * m.open[f] for f in m.facilities) + sum(
        assign_cost(f, c) * m.x[f, c] for f in m.facilities for c in m.customers
    )
    if quadratic:
        q = 2.0
        obj = obj + q * sum(
            m.x[f, c] * m.x[f, c] for f in m.facilities for c in m.customers
        )
    m.obj = pyo.Objective(expr=obj, sense=pyo.minimize)
    return m
