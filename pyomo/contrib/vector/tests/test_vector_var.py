# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the columnar (array-backed) VectorVar."""

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
from pyomo.core.base.var import Var, VarData


@unittest.skipUnless(numpy_available and scipy_available, "vector requires numpy/scipy")
class TestVectorVar(unittest.TestCase):
    def _make(self, **kw):
        from pyomo.contrib.vector import VectorVar

        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 4), **kw)
        m.x.construct()
        return m

    def test_is_var_typed(self):
        m = self._make(domain=pyo.NonNegativeReals, bounds=(0, 10))
        self.assertIs(m.x.ctype, Var)
        # component_objects(Var) finds it (without materializing data)
        found = list(m.component_objects(Var))
        self.assertIn(m.x, found)
        self.assertEqual(len(m.x._data), 0)  # nothing materialized yet

    def test_identity_materialize_on_touch(self):
        # m.x[i] is m.x[i] -- the load-bearing correctness requirement (R1).
        m = self._make(domain=pyo.NonNegativeReals, bounds=(0, 10))
        a = m.x[2]
        b = m.x[2]
        self.assertIs(a, b)
        self.assertIsInstance(a, VarData)
        # Touching one index does not scalarize the whole component.
        self.assertFalse(m.x._scalarized)
        self.assertEqual(len(m.x._data), 1)

    def test_scalar_access_reads_arrays(self):
        m = self._make(domain=pyo.NonNegativeReals, bounds=(1.0, 8.0),
                       initialize=3.0)
        self.assertEqual(m.x[0].lb, 1.0)
        self.assertEqual(m.x[0].ub, 8.0)
        self.assertEqual(m.x[0].value, 3.0)
        self.assertFalse(m.x[0].fixed)
        self.assertEqual(m.x[0].bounds, (1.0, 8.0))

    def test_scalar_write_through_to_arrays(self):
        # Array-backed: writing via the VarData updates the parent arrays, so
        # the fast path (which reads arrays) stays consistent.
        m = self._make(domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.x[3].set_value(4.5)
        m.x[3].fix()
        pos = m.x.position_of(3)
        self.assertEqual(m.x.value_array[pos], 4.5)
        self.assertTrue(m.x.fixed_array[pos])
        m.x[3].setlb(2.0)
        self.assertEqual(m.x[3].lb, 2.0)

    def test_per_index_bounds_array(self):
        from pyomo.contrib.vector import VectorVar

        m = pyo.ConcreteModel()
        lb = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ub = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        m.x = VectorVar(pyo.RangeSet(0, 4), domain=pyo.Reals, bounds=(lb, ub))
        m.x.construct()
        for i in range(5):
            self.assertEqual(m.x[i].bounds, (lb[i], ub[i]))

    def test_effective_bounds_combine_domain(self):
        # NonNegativeReals domain lb=0 tightens an explicit lb of -5 to 0.
        m = self._make(domain=pyo.NonNegativeReals, bounds=(-5.0, 100.0))
        elb, eub = m.x.effective_bounds()
        self.assertTrue(np.allclose(elb, 0.0))       # domain floor wins
        self.assertTrue(np.allclose(eub, 100.0))

    def test_binary_effective_bounds_and_integrality(self):
        m = self._make(domain=pyo.Binary)
        elb, eub = m.x.effective_bounds()
        self.assertTrue(np.allclose(elb, 0.0))
        self.assertTrue(np.allclose(eub, 1.0))
        self.assertTrue(m.x.integrality().all())
        self.assertTrue(m.x[0].is_binary())

    def test_integers_integrality(self):
        m = self._make(domain=pyo.Integers)
        self.assertTrue(m.x.integrality().all())
        self.assertTrue(m.x[0].is_integer())
        m2 = self._make(domain=pyo.Reals)
        self.assertFalse(m2.x.integrality().any())
        self.assertFalse(m2.x[0].is_integer())

    def test_len_is_logical_size(self):
        # __len__ returns the logical size without materializing anything.
        from pyomo.contrib.vector import VectorVar

        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 99), domain=pyo.Reals)
        m.x.construct()
        self.assertEqual(len(m.x), 100)
        self.assertEqual(len(m.x._data), 0)

    def test_multidimensional_index(self):
        from pyomo.contrib.vector import VectorVar

        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, 2)
        m.J = pyo.RangeSet(0, 3)
        m.x = VectorVar(m.I, m.J, domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.x.construct()
        self.assertEqual(m.x.n, 12)
        # position order matches index-set iteration (I outer, J inner)
        self.assertEqual(m.x.position_of((0, 0)), 0)
        self.assertEqual(m.x.position_of((0, 1)), 1)
        self.assertEqual(m.x.position_of((1, 0)), 4)
        self.assertIs(m.x[1, 2], m.x[1, 2])


if __name__ == "__main__":
    unittest.main()
