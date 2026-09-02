# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""High-variable-reuse resource coupling with a *filtered* sum (Phase 3b).

Identical in shape and sizing to ``resource_coupling`` -- the templatizable-heavy
model whose ``nnz / n_vars ~ J`` puts the cost squarely in constraint (not
variable) construction -- but the coupling family uses a **filtered** sum that
excludes the diagonal activity::

    couple[z, i]:  (J - 1) * x[z, i] - sum(x[z, j] for j in acts if j != i) <= cap[z, i]

so the whole family exercises the Phase-3b filtered-sum extractor at the same
high reuse where skipping per-index tree construction is a large win.  Because
construct here is dominated by the constraint family (not the small ``Z * J``
variable set), it is where the filtered-sum construct speedup shows most clearly
-- the counterpart to ``flow_masked``, whose ``nnz / n_vars ~ 2`` makes its
construct Var-build-bound.
"""

from __future__ import annotations

from typing import Any, Dict

import pyomo.environ as pyo

NAME = "coupling_filtered"
DESCRIPTION = "Multi-zone coupling with a filtered sum (templatizable-heavy, 3b)."
HAS_QUADRATIC = False

SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"Z": 2, "J": 20},
    "1e4": {"Z": 10, "J": 30},
    "1e5": {"Z": 20, "J": 70},
    "1e6": {"Z": 100, "J": 100},
    "1e7": {"Z": 250, "J": 200},
}


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    Z = int(params["Z"])
    J = int(params["J"])
    m = pyo.ConcreteModel(name=f"coupling_filtered_{Z}x{J}")

    m.zones = pyo.RangeSet(0, Z - 1)
    m.acts = pyo.RangeSet(0, J - 1)

    ub = 10.0
    m.x = pyo.Var(m.zones, m.acts, domain=pyo.NonNegativeReals, bounds=(0, ub))

    m.target = pyo.Param(
        m.zones, initialize={z: 0.5 * J * ub for z in range(Z)}, mutable=True
    )
    m.cap = pyo.Param(
        m.zones,
        m.acts,
        initialize={(z, i): float(J * ub) for z in range(Z) for i in range(J)},
        mutable=True,
    )

    def total_rule(m, z):
        # Unfiltered sum-over-set equality (Phase 3 subset).
        return sum(m.x[z, j] for j in m.acts) == m.target[z]

    m.total = pyo.Constraint(m.zones, rule=total_rule)

    def couple_rule(m, z, i):
        # Diagonal term + *filtered* sum over the other activities (Phase 3b).
        return (J - 1) * m.x[z, i] - sum(
            m.x[z, j] for j in m.acts if j != i
        ) <= m.cap[z, i]

    m.couple = pyo.Constraint(m.zones, m.acts, rule=couple_rule)

    def cost(z, j):
        return 1.0 + ((z * 3 + j * 7) % 11)

    m.obj = pyo.Objective(
        expr=sum(cost(z, j) * m.x[z, j] for z in range(Z) for j in range(J)),
        sense=pyo.minimize,
    )
    return m
