"""Variable-heavy templatizable synthetic model (``nnz ~ n_var``).

The complement to ``resource_coupling`` (variable-*reuse*-heavy, where the
constraint nonzeros dominate): here every variable appears in only O(1)
constraints, so after Phase-3 templatizes the constraint families the residual
cold construct is dominated by the per-index ``Var`` / ``Param`` object
allocation -- exactly the regime the transparent columnar Var/Param construction
targets.  This mirrors the shape of large reconstruct-per-scenario models.

``N`` variables ``x[i] >= 0``.  Two templatizable families:

* ``cap[i]``  -- ``x[i] <= u[i]``            (mutable-Param upper bound; diagonal)
* ``link[i]`` -- ``x[i] - x[i-1] <= s``      (a scalar-affine neighbour coupling)

plus a light linear objective.  All bodies are in the Spike-B proven subset, so
both families vectorize; ``u`` is a mutable indexed ``Param`` initialized from a
dict (the classic per-index ``ParamData`` build this phase removes).
"""

from __future__ import annotations

from typing import Any, Dict

import pyomo.environ as pyo

NAME = "columnar_stress"
DESCRIPTION = "Variable-heavy templatizable model (nnz ~ n_var)."
HAS_QUADRATIC = False

# n_var == N, nnz ~= 3*N (diagonal cap + two-term link).  N chosen per decade.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"N": 200},
    "1e4": {"N": 3_300},
    "1e5": {"N": 33_000},
    "1e6": {"N": 330_000},
    "1e7": {"N": 3_300_000},
}


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    N = int(params["N"])
    m = pyo.ConcreteModel(name=f"columnar_stress_{N}")
    m.I = pyo.RangeSet(0, N - 1)

    m.u = pyo.Param(m.I, initialize={i: 5.0 + (i % 13) for i in range(N)}, mutable=True)
    m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=(0, 1000))

    def cap_rule(m, i):
        return m.x[i] <= m.u[i]

    m.cap = pyo.Constraint(m.I, rule=cap_rule)

    def link_rule(m, i):
        # Scalar-affine neighbour coupling (unfiltered; templatizes).  i == 0
        # would need an index conditional (out of subset), so start at 1 and let
        # the family index carry it -- the rule stays templatizable.
        return m.x[i] - m.x[i - 1] <= 2.0

    m.link = pyo.Constraint(pyo.RangeSet(1, N - 1), rule=link_rule)

    def cost(i):
        return 1.0 + ((i * 7) % 5)

    m.obj = pyo.Objective(expr=sum(cost(i) * m.x[i] for i in m.I), sense=pyo.maximize)
    return m
