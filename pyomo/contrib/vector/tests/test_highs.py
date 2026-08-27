# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Direct HiGHS ``passModel`` hand-off: load + solve correctness.

This is the "load prize" of the project (scoping doc §6.4): the vectorized model
is handed to HiGHS as arrays in one ``passModel`` call.  These tests check that
the resulting solve agrees with a classic APPSI-HiGHS solve of the same model.
"""

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
    attempt_import,
)

highspy, highspy_available = attempt_import('highspy')


@unittest.skipUnless(
    numpy_available and scipy_available and highspy_available,
    "vector HiGHS hand-off requires numpy/scipy/highspy",
)
class TestHighsHandoff(unittest.TestCase):
    def test_passModel_lp_solve(self):
        import scipy.sparse as sp
        from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs

        # min x0 + x1 s.t. x0 + 2 x1 == 3 ; x0 - x1 <= 1 ; x >= 0
        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 1), domain=pyo.NonNegativeReals, bounds=(0, 10))
        A = sp.csr_matrix(np.array([[1.0, 2.0], [1.0, -1.0]]))
        m.c = VectorConstraint(A=A, x=m.x, lb=np.array([3.0, -np.inf]),
                               ub=np.array([3.0, 1.0]))
        m.obj = VectorObjective(terms={m.x: np.array([1.0, 1.0])}, sense=pyo.minimize)
        m.x.construct()
        m.c.construct()
        m.obj.construct()
        h, obj = solve_highs(m)
        self.assertAlmostEqual(obj, 1.5, places=6)

    def test_range_rows_native(self):
        # A ranged constraint 1 <= x0 + x1 <= 2 is loaded as a native range row.
        import scipy.sparse as sp
        from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective
        from pyomo.contrib.vector.highs import load_highs

        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 1), domain=pyo.NonNegativeReals, bounds=(0, 5))
        A = sp.csr_matrix(np.array([[1.0, 1.0]]))
        m.c = VectorConstraint(A=A, x=m.x, lb=np.array([1.0]), ub=np.array([2.0]))
        m.obj = VectorObjective(terms={m.x: np.array([1.0, 1.0])})
        m.x.construct()
        m.c.construct()
        m.obj.construct()
        h = load_highs(m)
        lp = h.getLp()
        self.assertEqual(lp.num_row_, 1)  # one range row, not split into two
        self.assertEqual(lp.num_col_, 2)

    def test_solve_matches_classic_appsi(self):
        import scipy.sparse as sp
        from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs

        appsi_highs = pyo.SolverFactory('appsi_highs')
        if not appsi_highs.available(exception_flag=False):
            self.skipTest("appsi_highs not available")

        rng = np.random.default_rng(99)
        for _ in range(20):
            n = int(rng.integers(3, 8))
            R = int(rng.integers(2, 6))
            A = rng.integers(-2, 3, size=(R, n)).astype(float)
            A[rng.random((R, n)) < 0.3] = 0.0
            for r in range(R):
                if not A[r].any():
                    A[r, int(rng.integers(0, n))] = 1.0
            ub = float(rng.integers(5, 12))
            xstar = rng.uniform(0, ub, size=n)
            rlb = np.zeros(R)
            rub = np.zeros(R)
            for r in range(R):
                b = float(A[r] @ xstar)
                k = int(rng.integers(0, 3))
                if k == 0:
                    rlb[r] = rub[r] = b
                elif k == 1:
                    rlb[r] = -np.inf
                    rub[r] = b + float(rng.uniform(0, 3))
                else:
                    rlb[r] = b - float(rng.uniform(0, 3))
                    rub[r] = np.inf
            co = rng.uniform(-2, 2, size=n)

            mv = pyo.ConcreteModel()
            mv.x = VectorVar(pyo.RangeSet(0, n - 1), domain=pyo.Reals, bounds=(0, ub))
            mv.c = VectorConstraint(A=sp.csr_matrix(A), x=mv.x, lb=rlb, ub=rub)
            mv.obj = VectorObjective(terms={mv.x: co}, sense=pyo.minimize)
            mv.x.construct()
            mv.c.construct()
            mv.obj.construct()
            _, ov = solve_highs(mv)

            mc = pyo.ConcreteModel()
            mc.x = pyo.Var(pyo.RangeSet(0, n - 1), domain=pyo.Reals, bounds=(0, ub))
            Ad = A

            def crule(m, r):
                body = sum(Ad[r, j] * m.x[j] for j in range(n) if Ad[r, j] != 0)
                lo = rlb[r] if np.isfinite(rlb[r]) else None
                hi = rub[r] if np.isfinite(rub[r]) else None
                if lo is not None and hi is not None:
                    return (body == lo) if lo == hi else pyo.inequality(lo, body, hi)
                if hi is not None:
                    return body <= hi
                return body >= lo

            mc.c = pyo.Constraint(pyo.RangeSet(0, R - 1), rule=crule)
            mc.obj = pyo.Objective(expr=sum(co[j] * mc.x[j] for j in range(n)))
            appsi_highs.solve(mc)
            oc = pyo.value(mc.obj)
            self.assertAlmostEqual(ov, oc, delta=1e-5 * max(1.0, abs(oc)))


if __name__ == "__main__":
    unittest.main()
