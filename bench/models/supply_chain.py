"""Ragged multi-echelon supply chain (sparse index sets).

Three echelons - ``S`` suppliers -> ``W`` warehouses -> ``R`` retailers - over
``T`` periods, but the lanes are a *ragged sparse subset* of the full cross
product: each supplier serves a random handful of warehouses, each warehouse a
random handful of retailers.  Warehouses carry inventory linking periods.

This is the case that a dense-rectangular design (e.g. xarray/linopy) handles
badly (it must allocate the full ``S*W`` / ``W*R`` grids and mask), and it is the
one the scoping doc (R5) insists must be in the suite "from day 1" so the fast
path is forced to handle real sparsity, not just dense boxes.

Lanes are generated with a fixed RNG seed so the model is reproducible.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import pyomo.environ as pyo

NAME = "supply_chain"
DESCRIPTION = "Ragged multi-echelon supply chain (sparse ragged lanes)."
HAS_QUADRATIC = False

# Sizes scale periods and node counts together.  nnz is roughly
# T * (2*|SW_lanes| + 2*|WR_lanes| + inventory terms); reported exactly by the
# harness.  ``xs`` under the gurobipy size-limit cap.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"S": 3, "W": 4, "R": 6, "deg": 2, "T": 5, "seed": 0},
    "1e4": {"S": 8, "W": 12, "R": 20, "deg": 3, "T": 40, "seed": 0},
    "1e5": {"S": 15, "W": 30, "R": 60, "deg": 4, "T": 120, "seed": 0},
    "1e6": {"S": 40, "W": 80, "R": 160, "deg": 5, "T": 300, "seed": 0},
    "1e7": {"S": 100, "W": 200, "R": 400, "deg": 6, "T": 800, "seed": 0},
}


def _ragged_lanes(
    src_ids: range, dst_ids: List[int], deg: int, rng: random.Random
) -> List[Tuple[int, int]]:
    """For each source, connect to a random 1..(2*deg) subset of destinations."""
    lanes: List[Tuple[int, int]] = []
    n_dst = len(dst_ids)
    for s in src_ids:
        k = rng.randint(1, min(max(1, 2 * deg), n_dst))
        chosen = rng.sample(dst_ids, k)
        for d in chosen:
            lanes.append((s, d))
    return lanes


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    S = int(params["S"])
    W = int(params["W"])
    R = int(params["R"])
    deg = int(params["deg"])
    T = int(params["T"])
    rng = random.Random(int(params.get("seed", 0)))

    warehouses = list(range(W))
    retailers = list(range(R))
    sw = _ragged_lanes(range(S), warehouses, deg, rng)          # supplier->warehouse
    wr = _ragged_lanes(range(W), retailers, deg, rng)           # warehouse->retailer

    # Ensure every retailer has at least one inbound lane (so demand is servable).
    wr_dst = {d for _, d in wr}
    for r in retailers:
        if r not in wr_dst:
            w = rng.randrange(W)
            wr.append((w, r))

    m = pyo.ConcreteModel(name=f"supply_chain_{S}_{W}_{R}x{T}")
    m.periods = pyo.RangeSet(0, T - 1)
    m.warehouses = pyo.Set(initialize=warehouses)
    m.SW = pyo.Set(initialize=sw, dimen=2)
    m.WR = pyo.Set(initialize=wr, dimen=2)

    # Precompute adjacency for O(1) rule bodies (idiomatic: build helper dicts).
    in_to_w: Dict[int, List[int]] = {w: [] for w in warehouses}
    out_from_w: Dict[int, List[int]] = {w: [] for w in warehouses}
    for (s, w) in sw:
        in_to_w[w].append(s)
    for (w, r) in wr:
        out_from_w[w].append(r)
    in_to_r: Dict[int, List[int]] = {r: [] for r in retailers}
    for (w, r) in wr:
        in_to_r[r].append(w)
    out_from_s: Dict[int, List[int]] = {s: [] for s in range(S)}
    for (s, w) in sw:
        out_from_s[s].append(w)

    lane_cap = 100.0
    inv_cap = 500.0

    m.ship_sw = pyo.Var(m.SW, m.periods, domain=pyo.NonNegativeReals, bounds=(0, lane_cap))
    m.ship_wr = pyo.Var(m.WR, m.periods, domain=pyo.NonNegativeReals, bounds=(0, lane_cap))
    m.inv = pyo.Var(m.warehouses, m.periods, domain=pyo.NonNegativeReals, bounds=(0, inv_cap))

    def demand(r, t):
        return 5.0 + ((r * 3 + t) % 7)

    # Warehouse inventory balance: inflow - outflow + prev_inv - inv == 0.
    def wbalance_rule(m, w, t):
        inflow = sum(m.ship_sw[s, w, t] for s in in_to_w[w])
        outflow = sum(m.ship_wr[w, r, t] for r in out_from_w[w])
        prev = m.inv[w, t - 1] if t > 0 else 0.0
        return inflow - outflow + prev - m.inv[w, t] == 0.0

    m.wbalance = pyo.Constraint(m.warehouses, m.periods, rule=wbalance_rule)

    # Retailer demand satisfaction.
    m.retailers = pyo.Set(initialize=retailers)

    def rdemand_rule(m, r, t):
        return sum(m.ship_wr[w, r, t] for w in in_to_r[r]) == demand(r, t)

    m.rdemand = pyo.Constraint(m.retailers, m.periods, rule=rdemand_rule)

    # Supplier capacity.
    m.suppliers = pyo.Set(initialize=list(range(S)))
    supply_cap = 2.0 * lane_cap

    def scap_rule(m, s, t):
        return sum(m.ship_sw[s, w, t] for w in out_from_s[s]) <= supply_cap

    m.scap = pyo.Constraint(m.suppliers, m.periods, rule=scap_rule)

    sw_cost = {(s, w): 1.0 + ((s + w) % 5) for (s, w) in sw}
    wr_cost = {(w, r): 1.0 + ((w + r) % 4) for (w, r) in wr}
    m.obj = pyo.Objective(
        expr=sum(sw_cost[s, w] * m.ship_sw[s, w, t] for (s, w) in sw for t in range(T))
        + sum(wr_cost[w, r] * m.ship_wr[w, r, t] for (w, r) in wr for t in range(T))
        + 0.05 * sum(m.inv[w, t] for w in warehouses for t in range(T)),
        sense=pyo.minimize,
    )
    return m
