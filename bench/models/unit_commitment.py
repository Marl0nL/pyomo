"""Unit commitment: mixed sparse/dense MILP.

``G`` generators over ``T`` periods.  Two structural regimes coexist:

  * dense rows  - per-period demand balance ``sum_g p[g,t] == D[t]`` (``G`` terms).
  * sparse rows - per-(g,t) max/min output, ramp, and startup-logic constraints
                  (2-3 terms each), plus a rolling min-up-time window.

This mix (a few wide rows, many narrow rows, integrality) is exactly the shape
where a purely dense array design would waste memory and a purely scalar design
is slow.  Matrix nonzeros ~= 13*G*T.

Idiomatic rule-based construction.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pyomo.environ as pyo

NAME = "unit_commitment"
DESCRIPTION = "Unit commitment MILP (mixed sparse/dense, binary commitment)."
HAS_QUADRATIC = False

# nnz ~= 13*G*T.  ``xs`` under the gurobipy size-limit cap.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"G": 6, "T": 10, "min_up": 3},     # ~120 (g,t); ~1.5k nnz? small
    "1e4": {"G": 20, "T": 40, "min_up": 3},   # ~10k nnz
    "1e5": {"G": 50, "T": 160, "min_up": 3},  # ~100k nnz
    "1e6": {"G": 100, "T": 800, "min_up": 3}, # ~1M nnz
    "1e7": {"G": 200, "T": 4000, "min_up": 3},# ~10M nnz
}


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    G = int(params["G"])
    T = int(params["T"])
    min_up = int(params.get("min_up", 3))
    m = pyo.ConcreteModel(name=f"unit_commitment_{G}x{T}")

    m.gens = pyo.RangeSet(0, G - 1)
    m.periods = pyo.RangeSet(0, T - 1)

    # Per-generator physical parameters (deterministic spread).
    pmax = {g: 50.0 + 10.0 * (g % 5) for g in range(G)}
    pmin = {g: 0.2 * pmax[g] for g in range(G)}
    ramp = {g: 0.5 * pmax[g] for g in range(G)}
    # Demand: a fraction of total capacity, with a daily-ish ripple.
    total_pmax = sum(pmax.values())

    def demand(t):
        return 0.55 * total_pmax * (1.0 + 0.15 * math.sin(0.2 * t))

    m.u = pyo.Var(m.gens, m.periods, domain=pyo.Binary)          # on/off
    m.su = pyo.Var(m.gens, m.periods, domain=pyo.NonNegativeReals, bounds=(0, 1))  # startup
    m.p = pyo.Var(m.gens, m.periods, domain=pyo.NonNegativeReals)  # output

    def demand_rule(m, t):
        return sum(m.p[g, t] for g in m.gens) == demand(t)

    m.demand = pyo.Constraint(m.periods, rule=demand_rule)

    def pmax_rule(m, g, t):
        return m.p[g, t] <= pmax[g] * m.u[g, t]

    m.max_out = pyo.Constraint(m.gens, m.periods, rule=pmax_rule)

    def pmin_rule(m, g, t):
        return m.p[g, t] >= pmin[g] * m.u[g, t]

    m.min_out = pyo.Constraint(m.gens, m.periods, rule=pmin_rule)

    def rampup_rule(m, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.p[g, t] - m.p[g, t - 1] <= ramp[g]

    m.ramp_up = pyo.Constraint(m.gens, m.periods, rule=rampup_rule)

    def rampdn_rule(m, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.p[g, t - 1] - m.p[g, t] <= ramp[g]

    m.ramp_dn = pyo.Constraint(m.gens, m.periods, rule=rampdn_rule)

    def startup_rule(m, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.su[g, t] >= m.u[g, t] - m.u[g, t - 1]

    m.startup = pyo.Constraint(m.gens, m.periods, rule=startup_rule)

    # Rolling min-up-time: if a unit starts at t, it stays on for min_up periods.
    def minup_rule(m, g, t):
        if t < min_up - 1:
            return pyo.Constraint.Skip
        return sum(m.su[g, k] for k in range(t - min_up + 1, t + 1)) <= m.u[g, t]

    m.min_up_time = pyo.Constraint(m.gens, m.periods, rule=minup_rule)

    cost = {g: 20.0 + 5.0 * (g % 7) for g in range(G)}
    start_cost = {g: 100.0 + 10.0 * (g % 4) for g in range(G)}
    m.obj = pyo.Objective(
        expr=sum(cost[g] * m.p[g, t] for g in m.gens for t in m.periods)
        + sum(start_cost[g] * m.su[g, t] for g in m.gens for t in m.periods),
        sense=pyo.minimize,
    )
    return m
