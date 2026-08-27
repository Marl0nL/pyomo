# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Lazy-scalarization compatibility contract (scoping doc §6.5).

Any consumer that does not recognize the vectorized components triggers
scalarization: the component materializes classic data objects on iteration,
marks itself scalarized (fast path disabled), and warns once.  The classic
behaviour that results must be identical to a natively-classic model.
"""

import logging

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler


@unittest.skipUnless(numpy_available and scipy_available, "vector requires numpy/scipy")
class TestScalarization(unittest.TestCase):
    def _small_vector_model(self):
        import scipy.sparse as sp
        from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective

        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 3), domain=pyo.NonNegativeReals, bounds=(0, 10))
        A = sp.csr_matrix(np.array([[1.0, 1.0, 0, 0], [0, 1.0, 1.0, 0], [0, 0, 1.0, 1.0]]))
        m.con = VectorConstraint(A=A, x=m.x, lb=np.array([1.0, -np.inf, 2.0]),
                                 ub=np.array([1.0, 3.0, np.inf]))
        m.obj = VectorObjective(terms={m.x: np.array([1.0, 2.0, 3.0, 4.0])})
        m.x.construct()
        m.con.construct()
        m.obj.construct()
        return m

    def test_scalar_constraint_access_materializes(self):
        m = self._small_vector_model()
        cd = m.con[0]
        self.assertIsInstance(cd, ConstraintData)
        lb, body, ub = cd.to_bounded_expression()
        self.assertEqual(lb, 1.0)
        self.assertEqual(ub, 1.0)
        # body is 1*x[0] + 1*x[1]; check its evaluated value at x=(2,3,...)
        m.x[0].set_value(2.0)
        m.x[1].set_value(3.0)
        self.assertEqual(pyo.value(cd.body), 5.0)
        self.assertIs(m.con[0], cd)  # cached (identity preserved)

    def test_iteration_triggers_scalarization_with_one_warning(self):
        m = self._small_vector_model()
        self.assertFalse(m.con._scalarized)
        with LoggingIntercept(level=logging.WARNING) as LOG:
            data = list(m.con.values())
            # iterate again: must NOT warn a second time
            _ = list(m.con.values())
        self.assertEqual(len(data), 3)
        self.assertTrue(m.con._scalarized)
        msgs = LOG.getvalue()
        self.assertEqual(msgs.count("was scalarized"), 1)

    def test_component_data_objects_finds_all(self):
        m = self._small_vector_model()
        cons = list(m.component_data_objects(Constraint, active=True))
        self.assertEqual(len(cons), 3)
        varsd = list(m.component_data_objects(pyo.Var, active=True))
        self.assertEqual(len(varsd), 4)

    def test_fast_path_disabled_after_scalarization(self):
        from pyomo.contrib.vector import compile_standard_form, VectorPathDisabledError

        m = self._small_vector_model()
        list(m.con.values())  # scalarize
        with self.assertRaises(VectorPathDisabledError):
            compile_standard_form(m, mixed_form=True)

    def test_stock_compiler_on_vector_matches_classic(self):
        # The ultimate compatibility proof: an unaware consumer (the stock
        # standard-form compiler) processes the vector model via scalarization
        # and produces an identical standard form to a natively-classic model.
        from pyomo.contrib.vector.tests.random_models import (
            Case,
            build_vector,
            build_classic,
        )
        from pyomo.contrib.vector.tests.equivalence_oracle import assert_equivalent

        rng = np.random.default_rng(4242)
        for _ in range(50):
            case = Case(rng)
            mv = build_vector(case)
            mc = build_classic(case)
            with LoggingIntercept():  # swallow the scalarization warnings
                iv = LinearStandardFormCompiler().write(mv, mixed_form=True)
            ic = LinearStandardFormCompiler().write(mc, mixed_form=True)
            assert_equivalent(self, iv, ic)


if __name__ == "__main__":
    unittest.main()
