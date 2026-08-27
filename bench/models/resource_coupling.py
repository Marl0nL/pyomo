"""Templatizable-heavy synthetic model: multi-zone resource coupling.

Deliberately built from the constraint shapes Phase-0 Spike B proved *do*
templatize -- unfiltered ``sum(... for j in Set)`` and scalar-affine
(incl. neighbour) bodies with constant coefficients and mutable-``Param`` bounds
-- so it exercises the Phase-3 template-vectorized construction fast path end to
end.  It is the "templatizable-heavy" counterpart to the existing synthetic
suite, whose idiomatic rules (filtered sums ``if j != n``, index conditionals
``if t > 0``) deliberately do NOT templatize and therefore stay on the classic
fallback.

``Z`` zones, ``J`` activities per zone.  Continuous ``x[z, j] >= 0``.  Two
constraint families, both templatizable:

* ``total[z]``   -- each zone meets its target:  ``sum_j x[z, j] == D[z]``
  (one row per zone; the unfiltered sum-over-set idiom).
* ``couple[z, i]`` -- fairness / coupling:  ``J * x[z, i] - sum_j x[z, j] <= cap[z, i]``
  (one row per activity; a scalar-affine *diagonal* term plus an unfiltered sum,
  so every activity variable appears in every coupling row of its zone).  This
  family carries the bulk of the nonzeros and is where per-index tree building
  hurts most -- ``nnz ~ Z * J^2`` while there are only ``Z * J`` variables, the
  high variable-reuse regime.

Both bodies have constant coefficients (``1``, ``-1``, ``J``) and mutable-``Param``
right-hand sides, exactly the proven subset.  The objective is a light linear
cost so that construction time is dominated by the (templatizable) constraint
families, which is the point of the fast path.
"""

from __future__ import annotations

from typing import Any, Dict

import pyomo.environ as pyo

NAME = "resource_coupling"
DESCRIPTION = "Multi-zone resource coupling (templatizable-heavy: sum + affine)."
HAS_QUADRATIC = False

# nnz ~= Z * J * (J + 2).  Chosen so the coupling family dominates and the
# variable-reuse ratio (nnz / n_vars ~ J) is high enough that skipping per-index
# tree construction is a large win.  ``xs`` stays under the gurobipy cap.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"Z": 2, "J": 20},  # ~880 nnz, 40 vars, 42 cons
    "1e4": {"Z": 10, "J": 30},  # ~10k nnz
    "1e5": {"Z": 20, "J": 70},  # ~100k nnz
    "1e6": {"Z": 100, "J": 100},  # ~1.02M nnz, 10k vars
    "1e7": {"Z": 250, "J": 200},  # ~10M nnz  (heavy)
}


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    Z = int(params["Z"])
    J = int(params["J"])
    m = pyo.ConcreteModel(name=f"resource_coupling_{Z}x{J}")

    m.zones = pyo.RangeSet(0, Z - 1)
    m.acts = pyo.RangeSet(0, J - 1)

    ub = 10.0
    m.x = pyo.Var(m.zones, m.acts, domain=pyo.NonNegativeReals, bounds=(0, ub))

    # Per-zone target (feasible: 0 <= D[z] <= J*ub); mid-range keeps it well posed.
    m.target = pyo.Param(
        m.zones, initialize={z: 0.5 * J * ub for z in range(Z)}, mutable=True
    )
    # Coupling cap (kept generous so the coupling rows are mostly non-binding,
    # i.e. the model stays feasible for any target split).
    m.cap = pyo.Param(
        m.zones,
        m.acts,
        initialize={(z, i): float(J * ub) for z in range(Z) for i in range(J)},
        mutable=True,
    )

    def total_rule(m, z):
        # Unfiltered sum-over-set equality (templatizes).
        return sum(m.x[z, j] for j in m.acts) == m.target[z]

    m.total = pyo.Constraint(m.zones, rule=total_rule)

    def couple_rule(m, z, i):
        # Diagonal scalar-affine term + unfiltered sum (templatizes); high reuse.
        return J * m.x[z, i] - sum(m.x[z, j] for j in m.acts) <= m.cap[z, i]

    m.couple = pyo.Constraint(m.zones, m.acts, rule=couple_rule)

    def cost(z, j):
        return 1.0 + ((z * 3 + j * 7) % 11)

    # Light linear objective (kept cheap so construction time reflects the
    # constraint families, not the objective).
    m.obj = pyo.Objective(
        expr=sum(cost(z, j) * m.x[z, j] for z in range(Z) for j in range(J)),
        sense=pyo.minimize,
    )
    return m
