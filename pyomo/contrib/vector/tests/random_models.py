# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Matched random model builders for property-based equivalence testing.

Each case is a randomized linear program built *two* ways from the *same* arrays:

* :func:`build_vector` -- with :class:`VectorVar` / :class:`VectorConstraint` /
  :class:`VectorObjective` (the fast path), and
* :func:`build_classic` -- with idiomatic ``Var`` + ``Constraint(rule=...)`` +
  ``Objective`` (the classic path).

Because the two models share variable names (``y``, ``z``) and integer indices,
the equivalence oracle can match their columns by identity.
"""

from __future__ import annotations

from pyomo.common.dependencies import numpy as np, scipy


DOMAINS = None  # filled lazily to avoid importing pyo at module import


def _domains():
    global DOMAINS
    if DOMAINS is None:
        import pyomo.environ as pyo

        DOMAINS = [
            pyo.NonNegativeReals,
            pyo.Reals,
            pyo.Binary,
            pyo.Integers,
            pyo.NonPositiveReals,
        ]
    return DOMAINS


class Case:
    """A randomized LP specification shared by the vector and classic builders."""

    def __init__(self, rng, fixed=False):
        import pyomo.environ as pyo

        doms = _domains()
        self.ny = int(rng.integers(1, 5))
        self.nz = int(rng.integers(1, 5))
        n = self.ny + self.nz
        self.n = n
        self.R = int(rng.integers(1, 7))
        self.domy = doms[int(rng.integers(0, len(doms)))]
        self.domz = doms[int(rng.integers(0, len(doms)))]

        A = rng.integers(-3, 4, size=(self.R, n)).astype(float)
        A[rng.random((self.R, n)) < 0.35] = 0.0
        for r in range(self.R):  # guarantee >= 1 nonzero per row
            if not A[r].any():
                A[r, int(rng.integers(0, n))] = float(rng.choice([-2, -1, 1, 2, 3]))
        self.A = A

        self.yb = (float(rng.integers(-4, 1)), float(rng.integers(2, 10)))
        self.zb = (float(rng.integers(-4, 1)), float(rng.integers(2, 15)))

        rlb = np.zeros(self.R)
        rub = np.zeros(self.R)
        for r in range(self.R):
            k = int(rng.integers(0, 4))
            v = float(rng.integers(-6, 7))
            if k == 0:
                rlb[r] = rub[r] = v
            elif k == 1:
                rlb[r] = -np.inf
                rub[r] = v
            elif k == 2:
                rlb[r] = v
                rub[r] = np.inf
            else:
                rlb[r] = v
                rub[r] = v + float(rng.integers(1, 6))
        self.rlb = rlb
        self.rub = rub
        self.coy = rng.integers(-3, 4, size=self.ny).astype(float)
        self.coz = rng.integers(-3, 4, size=self.nz).astype(float)


def build_vector(case):
    import pyomo.environ as pyo
    from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective

    m = pyo.ConcreteModel()
    m.y = VectorVar(pyo.RangeSet(0, case.ny - 1), domain=case.domy, bounds=case.yb)
    m.z = VectorVar(pyo.RangeSet(0, case.nz - 1), domain=case.domz, bounds=case.zb)
    m.con = VectorConstraint(
        A=scipy.sparse.csr_matrix(case.A), x=[m.y, m.z], lb=case.rlb, ub=case.rub
    )
    m.obj = VectorObjective(
        terms={m.y: case.coy, m.z: case.coz}, sense=pyo.minimize
    )
    m.y.construct()
    m.z.construct()
    m.con.construct()
    m.obj.construct()
    return m


def build_classic(case):
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.y = pyo.Var(pyo.RangeSet(0, case.ny - 1), domain=case.domy, bounds=case.yb)
    m.z = pyo.Var(pyo.RangeSet(0, case.nz - 1), domain=case.domz, bounds=case.zb)
    A = case.A
    ny, nz, n = case.ny, case.nz, case.n
    rlb, rub = case.rlb, case.rub

    def allv(mm):
        return [mm.y[j] for j in range(ny)] + [mm.z[j] for j in range(nz)]

    def crule(mm, r):
        xs = allv(mm)
        body = sum(A[r, j] * xs[j] for j in range(n) if A[r, j] != 0)
        lo = rlb[r] if np.isfinite(rlb[r]) else None
        hi = rub[r] if np.isfinite(rub[r]) else None
        if lo is not None and hi is not None:
            return (body == lo) if lo == hi else pyo.inequality(lo, body, hi)
        if hi is not None:
            return body <= hi
        if lo is not None:
            return body >= lo
        return pyo.Constraint.Skip

    m.con = pyo.Constraint(pyo.RangeSet(0, case.R - 1), rule=crule)
    co = list(case.coy) + list(case.coz)
    m.obj = pyo.Objective(
        expr=sum(co[j] * allv(m)[j] for j in range(n)), sense=pyo.minimize
    )
    return m
