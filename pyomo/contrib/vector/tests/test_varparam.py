# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for transparent columnar ``Var`` / ``Param`` construction (the Phase-3
switch extended to Var/Param).  Covers: the components go columnar under the
switch and byte-classic when off; the identity + scalarization contracts; the
mandatory fallback for genuinely per-index arguments; and standard-form + solve
equivalence between the switch on and off.
"""

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
from pyomo.core.base.var import Var, VarData, IndexedVar
from pyomo.core.base.param import Param, ParamData, IndexedParam

from pyomo.contrib.vector import vectorized_construction
from pyomo.contrib.vector.varparam import (
    TransparentVectorVar,
    TransparentVectorParam,
    VectorParamData,
    is_columnar_var,
    is_columnar_param,
    set_varparam_vectorize,
    restore_varparam_vectorize,
)

highspy_available = False
try:
    import highspy  # noqa: F401

    highspy_available = True
except ImportError:
    pass


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestColumnarConstruction(unittest.TestCase):
    def test_var_goes_columnar_under_switch(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 9)
            m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=(0, 10))
        self.assertIsInstance(m.x, TransparentVectorVar)
        self.assertIsInstance(m.x, Var)  # isinstance still holds
        self.assertIs(m.x.ctype, Var)
        self.assertTrue(is_columnar_var(m.x))
        # No per-index VarData materialized at construct.
        self.assertEqual(len(m.x._data), 0)
        self.assertEqual(len(m.x), 10)

    def test_switch_off_is_byte_classic(self):
        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, 9)
        m.x = pyo.Var(m.I, bounds=(0, 10))
        m.p = pyo.Param(m.I, initialize=1.0, mutable=True)
        self.assertIs(type(m.x), IndexedVar)
        self.assertIs(type(m.p), IndexedParam)
        self.assertFalse(is_columnar_var(m.x))
        self.assertFalse(is_columnar_param(m.p))
        # Classic build materializes every VarData.
        self.assertEqual(len(m.x._data), 10)

    def test_mutable_param_goes_columnar(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 4)
            m.p = pyo.Param(
                m.I, initialize={i: float(i) for i in range(5)}, mutable=True
            )
        self.assertIsInstance(m.p, TransparentVectorParam)
        self.assertIsInstance(m.p, Param)
        self.assertTrue(m.p.mutable)
        self.assertTrue(is_columnar_param(m.p))
        self.assertEqual(len(m.p._data), 0)
        for i in range(5):
            self.assertEqual(pyo.value(m.p[i]), float(i))

    def test_immutable_param_goes_columnar(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 4)
            m.p = pyo.Param(m.I, initialize={i: 2.0 * i for i in range(5)})
        self.assertIsInstance(m.p, TransparentVectorParam)
        self.assertFalse(m.p.mutable)
        # Immutable columnar Param serves raw values (no ParamData object).
        self.assertEqual(m.p[3], 6.0)
        self.assertNotIsInstance(m.p[3], ParamData)
        self.assertEqual([pyo.value(m.p[i]) for i in range(5)], [0, 2, 4, 6, 8])


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestContracts(unittest.TestCase):
    def _model(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 4)
            m.J = pyo.RangeSet(0, 2)
            m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals, bounds=(0, 7))
            m.p = pyo.Param(
                m.I, initialize={i: float(i) for i in range(5)}, mutable=True
            )
        return m

    def test_var_identity(self):
        m = self._model()
        self.assertIs(m.x[2, 1], m.x[2, 1])
        self.assertIsInstance(m.x[2, 1], VarData)
        self.assertEqual(m.x[0, 0].bounds, (0, 7))
        self.assertEqual(str(m.x[0, 0].domain), 'NonNegativeReals')

    def test_mutable_param_identity(self):
        m = self._model()
        self.assertIs(m.p[2], m.p[2])
        self.assertIsInstance(m.p[2], VectorParamData)
        # Array is the single source of truth: a scalar write is visible in bulk.
        m.p[2].value = 99.0
        self.assertEqual(m.p.value_array[2], 99.0)
        self.assertEqual(pyo.value(m.p[2]), 99.0)

    def test_var_scalarization_on_iter(self):
        m = self._model()
        with self.assertLogs('pyomo.contrib.vector', level='WARNING'):
            keys = list(m.x.keys())
        self.assertEqual(len(keys), 15)
        self.assertTrue(m.x._scalarized)
        # Every entry is now a real VarData, identity preserved.
        self.assertEqual(len(m.x._data), 15)
        self.assertIs(m.x[0, 0], m.x[0, 0])

    def test_param_scalarization_on_items(self):
        m = self._model()
        with self.assertLogs('pyomo.contrib.vector', level='WARNING'):
            d = dict(m.p.items())
        self.assertEqual(d[3].value, 3.0)

    def test_bulk_mutation_api(self):
        # The Phase-2 columnar mutation API (setlb/setub/fix) works transparently.
        m = self._model()
        m.x.setlb(1.0)
        m.x.setub(5.0, where=np.array([0, 1, 2]))
        lb, ub = m.x.effective_bounds()
        self.assertTrue(np.all(lb == 1.0))
        self.assertEqual(ub[0], 5.0)
        self.assertEqual(ub[14], 7.0)  # untouched keeps the domain/explicit ub


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestFallback(unittest.TestCase):
    """Genuinely per-index arguments fall back to byte-classic construction."""

    def test_per_index_bounds_rule_falls_back(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 4)
            m.x = pyo.Var(m.I, bounds=lambda m, i: (0, i + 1))
        self.assertIs(type(m.x), IndexedVar)
        self.assertFalse(is_columnar_var(m.x))
        self.assertEqual(m.x[3].ub, 4)

    def test_per_index_initialize_rule_falls_back(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 4)
            m.x = pyo.Var(m.I, initialize=lambda m, i: i * i, bounds=(0, 100))
        self.assertIs(type(m.x), IndexedVar)
        self.assertEqual(pyo.value(m.x[3]), 9)

    def test_per_index_domain_falls_back(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 3)
            m.x = pyo.Var(
                m.I, domain=lambda m, i: pyo.Binary if i % 2 else pyo.NonNegativeReals
            )
        self.assertIs(type(m.x), IndexedVar)

    def test_param_rule_falls_back(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 4)
            m.p = pyo.Param(m.I, initialize=lambda m, i: i * 10, mutable=True)
        self.assertIs(type(m.p), IndexedParam)
        self.assertEqual(pyo.value(m.p[3]), 30)

    def test_param_validate_falls_back(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 4)
            m.p = pyo.Param(
                m.I, initialize=1.0, validate=lambda m, v, i: v > 0, mutable=True
            )
        self.assertIs(type(m.p), IndexedParam)

    def test_scalar_var_not_transformed(self):
        with vectorized_construction():
            m = pyo.ConcreteModel()
            m.x = pyo.Var(bounds=(0, 1))
        # Scalars keep the classic ScalarVar path.
        self.assertNotIsInstance(m.x, TransparentVectorVar)

    def test_columnar_domain_validation(self):
        # A value outside the (bulk-checked) domain still raises, as classic.
        with self.assertRaises(ValueError):
            with vectorized_construction():
                m = pyo.ConcreteModel()
                m.I = pyo.RangeSet(0, 2)
                m.p = pyo.Param(
                    m.I,
                    domain=pyo.NonNegativeReals,
                    initialize={0: 1.0, 1: -3.0, 2: 2.0},
                    mutable=True,
                )


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestSwitchIsolation(unittest.TestCase):
    def test_switch_restores(self):
        prior = set_varparam_vectorize(True)
        try:
            m = pyo.ConcreteModel()
            m.I = pyo.RangeSet(0, 3)
            m.x = pyo.Var(m.I, bounds=(0, 1))
            self.assertIsInstance(m.x, TransparentVectorVar)
        finally:
            restore_varparam_vectorize(prior)
        # After restore, classic construction resumes.
        m2 = pyo.ConcreteModel()
        m2.I = pyo.RangeSet(0, 3)
        m2.x = pyo.Var(m2.I, bounds=(0, 1))
        self.assertIs(type(m2.x), IndexedVar)

    def test_nested_context(self):
        with vectorized_construction():
            with vectorized_construction(enabled=False):
                m = pyo.ConcreteModel()
                m.I = pyo.RangeSet(0, 3)
                m.x = pyo.Var(m.I, bounds=(0, 1))
                self.assertIs(type(m.x), IndexedVar)
            # Back to the outer (enabled) context.
            m2 = pyo.ConcreteModel()
            m2.I = pyo.RangeSet(0, 3)
            m2.x = pyo.Var(m2.I, bounds=(0, 1))
            self.assertIsInstance(m2.x, TransparentVectorVar)


def _build_lp(columnar):
    ctx = (
        vectorized_construction()
        if columnar
        else vectorized_construction(enabled=False)
    )
    with ctx:
        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, 5)
        m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=(0, 4))
        m.cap = pyo.Param(m.I, initialize={i: 3.0 + i for i in range(6)}, mutable=True)
        m.cover = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] <= m.cap[i])
        m.total = pyo.Constraint(expr=sum(m.x[i] for i in m.I) >= 5)
        m.obj = pyo.Objective(
            expr=sum((i + 1) * m.x[i] for i in m.I), sense=pyo.minimize
        )
    return m


@unittest.skipUnless(
    numpy_available and scipy_available and highspy_available,
    "solve equivalence requires highspy",
)
class TestEquivalence(unittest.TestCase):
    def _solve(self, m):
        from pyomo.contrib.solver.common.factory import SolverFactory

        SolverFactory('highs_fastload').solve(m)
        return pyo.value(m.obj), {i: pyo.value(m.x[i]) for i in m.I}

    def test_solve_equivalence_on_vs_off(self):
        obj_off, x_off = self._solve(_build_lp(False))
        obj_on, x_on = self._solve(_build_lp(True))
        self.assertAlmostEqual(obj_off, obj_on, places=6)
        for i in x_off:
            self.assertAlmostEqual(x_off[i], x_on[i], places=6)

    def test_solution_maps_back_without_scalarizing(self):
        m = _build_lp(True)
        self._solve(m)
        # Bulk map-back must not have scalarized the columnar Var.
        self.assertFalse(m.x._scalarized)
        self.assertTrue(is_columnar_var(m.x))

    def test_standard_form_matrix_equivalence(self):
        # The vectorized compile over columnar components matches the stock
        # standard form (row/col signature) of the classic build.
        from pyomo.contrib.vector import compile_templated_to_highs_arrays
        from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

        m_on = _build_lp(True)
        m_off = _build_lp(False)
        compiled = compile_templated_to_highs_arrays(m_on)
        info = LinearStandardFormCompiler().write(
            m_off, mixed_form=True, set_sense=None
        )
        # Same shape and nnz -> same problem structure.
        self.assertEqual(compiled.A.shape[0], info.A.shape[0])
        self.assertEqual(int(compiled.A.nnz), int(info.A.nnz))


if __name__ == "__main__":
    unittest.main()
