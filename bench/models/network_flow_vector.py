"""Dense multi-period network flow, built the *vectorized* (fast-path) way.

This constructs the **same LP** as ``bench.models.network_flow`` (verified by the
standard-form equivalence oracle and by identical objective/nnz), but using
``pyomo.contrib.vector``: columnar ``VectorVar`` for the flow/storage variables,
one explicit-array ``VectorConstraint`` for the whole flow-balance family, and a
``VectorObjective`` -- no per-index ``VarData``/``ConstraintData`` and no
per-constraint expression tree.

The matrix arrays are assembled with the same vectorized numpy/scipy the
array-native comparator uses (Kronecker structure), so "construct" here reflects
the fast path's real cost: bulk columnar allocation + explicit CSR assembly.
"""

from __future__ import annotations

from typing import Any, Dict

import pyomo.environ as pyo

# Reuse the exact same size presets as the classic generator so the two backends
# line up column-for-column in the results tables.
from bench.models.network_flow import SIZES  # noqa: F401

NAME = "network_flow"
DESCRIPTION = "Dense multi-period min-cost network flow (vectorized fast path)."
HAS_QUADRATIC = False


def build_pyomo(params: Dict[str, Any]) -> pyo.ConcreteModel:
    from bench.comparators.array_native import build_network_flow
    from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective

    N = int(params["N"])
    T = int(params["T"])
    arc_cap = 5.0 * N
    stor_cap = 10.0 * N

    # Vectorized matrix build (numpy/scipy Kronecker structure): this is the
    # explicit-array construction the fast path is designed around.
    mx = build_network_flow(params)
    n_flow = mx.meta["arcs"] * T
    n_stor = N * T
    A_eq = mx.A_eq  # (N*T) x (n_flow + n_stor), CSR
    b_eq = mx.b_eq

    m = pyo.ConcreteModel(name=f"network_flow_vec_{N}x{T}")
    # Use the *same* index sets and variable names as the classic generator so
    # the columns line up by identity (arc-major/period-minor column order
    # matches array_native's kron(B, I_T)); this makes the standard-form
    # equivalence oracle able to map the two column spaces one-to-one.
    m.nodes = pyo.RangeSet(0, N - 1)
    m.periods = pyo.RangeSet(0, T - 1)
    m.arcs = pyo.Set(
        initialize=[(i, j) for i in range(N) for j in range(N) if i != j], dimen=2
    )
    # Columnar variables (bulk array allocation, no per-index objects).
    m.flow = VectorVar(
        m.arcs, m.periods, domain=pyo.NonNegativeReals, bounds=(0, arc_cap)
    )
    m.stor = VectorVar(
        m.nodes, m.periods, domain=pyo.NonNegativeReals, bounds=(0, stor_cap)
    )
    # One explicit-array constraint family for all flow-balance rows (equality).
    m.balance = VectorConstraint(A=A_eq, x=[m.flow, m.stor], lb=b_eq, ub=b_eq)
    # Linear objective as coefficient arrays.
    m.obj = VectorObjective(
        terms={m.flow: mx.c[:n_flow], m.stor: mx.c[n_flow:]}, sense=pyo.minimize
    )
    m.flow.construct()
    m.stor.construct()
    m.balance.construct()
    m.obj.construct()
    return m
