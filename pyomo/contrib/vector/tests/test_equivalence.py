# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Property-based standard-form equivalence: fast path vs classic path.

The Phase-1 correctness gate (scoping doc Phase 1 exit criterion): the vector
fast path and the classic path must produce identical standard forms, up to row
and column permutation, with the same bounds, on randomized models.
"""

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
)
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler


@unittest.skipUnless(numpy_available and scipy_available, "vector requires numpy/scipy")
class TestStandardFormEquivalence(unittest.TestCase):
    def _check(self, case):
        from pyomo.contrib.vector import compile_standard_form
        from pyomo.contrib.vector.tests.random_models import (
            build_vector,
            build_classic,
        )
        from pyomo.contrib.vector.tests.equivalence_oracle import assert_equivalent

        mv = build_vector(case)
        mc = build_classic(case)
        iv = compile_standard_form(mv, mixed_form=True)
        ic = LinearStandardFormCompiler().write(mc, mixed_form=True)
        assert_equivalent(self, iv, ic)

    def test_randomized_equivalence(self):
        from pyomo.contrib.vector.tests.random_models import Case

        rng = np.random.default_rng(20260827)
        for _ in range(250):
            self._check(Case(rng))

    def test_single_var_all_sense_types(self):
        # Equality, <=, >=, and ranged rows against a single VectorVar.
        from pyomo.contrib.vector import (
            VectorVar,
            VectorConstraint,
            VectorObjective,
            compile_standard_form,
        )
        from pyomo.contrib.vector.tests.equivalence_oracle import assert_equivalent
        import scipy.sparse as sp

        A = sp.csr_matrix(
            np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        )
        lb = np.array([2.0, -np.inf, 1.0, 3.0])
        ub = np.array([2.0, 4.0, np.inf, 7.0])
        c = np.array([1.0, -2.0, 3.0])

        mv = pyo.ConcreteModel()
        mv.x = VectorVar(pyo.RangeSet(0, 2), domain=pyo.NonNegativeReals, bounds=(0, 10))
        mv.con = VectorConstraint(A=A, x=mv.x, lb=lb, ub=ub)
        mv.obj = VectorObjective(terms={mv.x: c}, sense=pyo.minimize)
        mv.x.construct()
        mv.con.construct()
        mv.obj.construct()

        mc = pyo.ConcreteModel()
        mc.x = pyo.Var(pyo.RangeSet(0, 2), domain=pyo.NonNegativeReals, bounds=(0, 10))
        Ad = A.toarray()

        def crule(m, r):
            body = sum(Ad[r, j] * m.x[j] for j in range(3) if Ad[r, j] != 0)
            lo = lb[r] if np.isfinite(lb[r]) else None
            hi = ub[r] if np.isfinite(ub[r]) else None
            if lo is not None and hi is not None:
                return (body == lo) if lo == hi else pyo.inequality(lo, body, hi)
            if hi is not None:
                return body <= hi
            return body >= lo

        mc.con = pyo.Constraint(pyo.RangeSet(0, 3), rule=crule)
        mc.obj = pyo.Objective(expr=sum(c[j] * mc.x[j] for j in range(3)))

        iv = compile_standard_form(mv, mixed_form=True)
        ic = LinearStandardFormCompiler().write(mc, mixed_form=True)
        assert_equivalent(self, iv, ic)

    def test_trivial_feasible_row_dropped(self):
        # An all-zero constraint row that is trivially feasible is dropped
        # (matching the stock compiler).
        from pyomo.contrib.vector import (
            VectorVar,
            VectorConstraint,
            VectorObjective,
            compile_standard_form,
        )
        import scipy.sparse as sp

        A = sp.csr_matrix(np.array([[1.0, 1.0], [0.0, 0.0]]))
        mv = pyo.ConcreteModel()
        mv.x = VectorVar(pyo.RangeSet(0, 1), domain=pyo.NonNegativeReals, bounds=(0, 5))
        mv.con = VectorConstraint(A=A, x=mv.x, lb=np.array([1.0, -1.0]),
                                  ub=np.array([1.0, 1.0]))
        mv.obj = VectorObjective(terms={mv.x: np.array([1.0, 1.0])})
        mv.x.construct()
        mv.con.construct()
        mv.obj.construct()
        info = compile_standard_form(mv, mixed_form=True)
        # Only the first (real) row survives.
        self.assertEqual(info.A.shape[0], 1)

    def test_trivial_infeasible_row_raises(self):
        from pyomo.common.errors import InfeasibleConstraintException
        from pyomo.contrib.vector import (
            VectorVar,
            VectorConstraint,
            VectorObjective,
            compile_standard_form,
        )
        import scipy.sparse as sp

        A = sp.csr_matrix(np.array([[1.0, 1.0], [0.0, 0.0]]))
        mv = pyo.ConcreteModel()
        mv.x = VectorVar(pyo.RangeSet(0, 1), domain=pyo.NonNegativeReals, bounds=(0, 5))
        # second row: 0 == 5  -> trivially infeasible
        mv.con = VectorConstraint(A=A, x=mv.x, lb=np.array([1.0, 5.0]),
                                  ub=np.array([1.0, 5.0]))
        mv.obj = VectorObjective(terms={mv.x: np.array([1.0, 1.0])})
        mv.x.construct()
        mv.con.construct()
        mv.obj.construct()
        with self.assertRaises(InfeasibleConstraintException):
            compile_standard_form(mv, mixed_form=True)


if __name__ == "__main__":
    unittest.main()
