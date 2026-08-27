# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Phase-2 mutability: bounds, masked deactivation, fixed-variable substitution,
solution load-back, and persistent warm re-solve.

Every feature is checked *mutation-then-rewrite* against a classic reference: the
mutated vector model must produce the same standard form (up to permutation) as a
classic model with the identical mutation, and -- where a solver is available --
the same solve objective.  Component names/index tuples are shared so the
equivalence oracle can key columns on identity.
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

_arrays = numpy_available and scipy_available
_solve = _arrays and highspy_available


# --------------------------------------------------------------------------- #
# Shared small models (vector + classic, identical names/indices)
# --------------------------------------------------------------------------- #
def _vector_model():
    import scipy.sparse as sp
    from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective

    m = pyo.ConcreteModel()
    m.x = VectorVar(pyo.RangeSet(0, 3), domain=pyo.NonNegativeReals, bounds=(0, 10))
    m.x.construct()
    # One family of 3 rows over the 4 columns of x:
    #   r0: x0 + x1 <= 4
    #   r1: x1 + x2 == 3
    #   r2: x2 + x3 <= 5
    A = sp.csr_matrix(
        np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    )
    m.con = VectorConstraint(
        A=A, x=m.x, lb=np.array([-np.inf, 3.0, -np.inf]), ub=np.array([4.0, 3.0, 5.0])
    )
    m.con.construct()
    m.obj = VectorObjective(
        terms={m.x: np.array([-1.0, -2.0, -1.0, -1.0])}, sense=pyo.minimize
    )
    m.obj.construct()
    return m


def _classic_model(fix=None, deactivate_rows=(), ub=None, rhs=None):
    """Classic twin.  ``fix`` maps index->value; ``deactivate_rows`` drops rows;
    ``ub`` maps var index->new upper bound; ``rhs`` maps row->new upper bound."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(pyo.RangeSet(0, 3), domain=pyo.NonNegativeReals, bounds=(0, 10))
    if ub:
        for i, b in ub.items():
            m.x[i].setub(b)
    if fix:
        for i, v in fix.items():
            m.x[i].fix(v)
    rows = {
        0: (m.x[0] + m.x[1] <= (rhs.get(0, 4.0) if rhs else 4.0)),
        1: (m.x[1] + m.x[2] == 3.0),
        2: (m.x[2] + m.x[3] <= (rhs.get(2, 5.0) if rhs else 5.0)),
    }
    m.con = pyo.ConstraintList()
    for r in (0, 1, 2):
        if r in deactivate_rows:
            continue
        m.con.add(rows[r])
    m.obj = pyo.Objective(
        expr=-1 * m.x[0] - 2 * m.x[1] - 1 * m.x[2] - 1 * m.x[3], sense=pyo.minimize
    )
    return m


def _canon(info):
    from pyomo.contrib.vector.tests.equivalence_oracle import canonical_standard_form

    return canonical_standard_form(info)


def _vector_std(m):
    from pyomo.contrib.vector import compile_standard_form

    return _canon(compile_standard_form(m, mixed_form=True))


def _classic_std(m):
    from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

    return _canon(LinearStandardFormCompiler().write(m, mixed_form=True))


def _solve_classic(m):
    from pyomo.contrib.appsi.solvers.highs import Highs

    Highs().solve(m)
    return float(pyo.value(m.obj))


# --------------------------------------------------------------------------- #
# 1. Bulk variable mutation + dirty tracking
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_arrays, "vector requires numpy/scipy")
class TestBulkVarMutation(unittest.TestCase):
    def _var(self, **kw):
        from pyomo.contrib.vector import VectorVar

        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 4), domain=pyo.Reals, **kw)
        m.x.construct()
        m.x.pop_dirty_bounds()  # clear the initial "all dirty"
        return m.x

    def test_bulk_setlb_setub_all(self):
        x = self._var()
        x.setlb(1.0)
        x.setub(9.0)
        self.assertTrue(np.allclose(x._lb_arr, 1.0))
        self.assertTrue(np.allclose(x._ub_arr, 9.0))

    def test_where_boolean_mask(self):
        x = self._var()
        mask = np.array([True, False, True, False, True])
        x.setub(2.0, where=mask)
        self.assertTrue(
            np.array_equal(np.isnan(x._ub_arr), [False, True, False, True, False])
        )
        self.assertEqual(x._ub_arr[0], 2.0)

    def test_where_position_array(self):
        x = self._var()
        x.setlb(np.array([3.0, 7.0]), where=np.array([1, 3]))
        self.assertEqual(x._lb_arr[1], 3.0)
        self.assertEqual(x._lb_arr[3], 7.0)

    def test_none_clears_bound(self):
        x = self._var(bounds=(0.0, 5.0))
        x.pop_dirty_bounds()
        x.setub(None)
        self.assertTrue(np.all(np.isnan(x._ub_arr)))

    def test_dirty_tracking_subset(self):
        x = self._var()
        x.setub(2.0, where=np.array([1, 3]))
        d = x.pop_dirty_bounds()
        self.assertTrue(np.array_equal(d, [1, 3]))
        # popping again yields an empty set (clean)
        self.assertEqual(len(x.pop_dirty_bounds()), 0)

    def test_dirty_all_on_full_write(self):
        x = self._var()
        x.setlb(0.0)  # touches every column
        self.assertIsNone(x.pop_dirty_bounds())  # None == all dirty

    def test_per_element_view_marks_parent_dirty(self):
        x = self._var()
        x[2].setlb(1.5)
        d = x.pop_dirty_bounds()
        self.assertTrue(np.array_equal(d, [2]))

    def test_bulk_fix_unfix(self):
        x = self._var()
        x.fix(4.0, where=np.array([0, 2]))
        self.assertTrue(x._fixed_arr[0] and x._fixed_arr[2])
        self.assertEqual(x._value_arr[0], 4.0)
        d = x.pop_dirty_bounds()
        self.assertTrue(np.array_equal(d, [0, 2]))
        x.unfix(where=np.array([0]))
        self.assertFalse(x._fixed_arr[0])
        self.assertTrue(np.array_equal(x.pop_dirty_bounds(), [0]))

    def test_set_values_only_dirties_fixed(self):
        x = self._var()
        x.fix(where=np.array([1]))
        x.pop_dirty_bounds()
        x.set_values(np.arange(5.0))  # write all values
        d = x.pop_dirty_bounds()
        # only the fixed column (1) is a bounds change
        self.assertTrue(np.array_equal(d, [1]))


# --------------------------------------------------------------------------- #
# 2. Masked deactivation (row removal in std form; relax on solve)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_arrays, "vector requires numpy/scipy")
class TestMaskedDeactivation(unittest.TestCase):
    def test_stdform_drops_masked_row(self):
        m = _vector_model()
        m.con.deactivate_rows(np.array([1]))  # drop the equality row
        self.assertEqual(
            _vector_std(m), _classic_std(_classic_model(deactivate_rows=(1,)))
        )

    def test_reactivate_restores(self):
        m = _vector_model()
        m.con.deactivate_rows(np.array([0, 2]))
        m.con.activate_rows()  # all back
        self.assertEqual(_vector_std(m), _classic_std(_classic_model()))

    def test_set_row_active_mask(self):
        m = _vector_model()
        m.con.set_row_active(np.array([True, False, True]))
        self.assertEqual(
            _vector_std(m), _classic_std(_classic_model(deactivate_rows=(1,)))
        )

    def test_dirty_rows_tracked(self):
        m = _vector_model()
        m.con.pop_dirty_rows()
        m.con.deactivate_rows(np.array([2]))
        self.assertTrue(np.array_equal(m.con.pop_dirty_rows(), [2]))

    def test_effective_row_bounds_relaxes(self):
        m = _vector_model()
        m.con.deactivate_rows(np.array([1]))
        lb, ub = m.con.effective_row_bounds()
        self.assertEqual(lb[1], -np.inf)
        self.assertEqual(ub[1], np.inf)
        # active rows unchanged
        self.assertEqual(ub[0], 4.0)

    @unittest.skipUnless(_solve, "solve requires highspy")
    def test_masked_solve_matches_classic(self):
        from pyomo.contrib.vector.highs import solve_highs

        m = _vector_model()
        m.con.deactivate_rows(np.array([0]))  # relax the x0+x1<=4 row
        _, obj = solve_highs(m)
        ref = _solve_classic(_classic_model(deactivate_rows=(0,)))
        self.assertAlmostEqual(obj, ref, places=6)


# --------------------------------------------------------------------------- #
# 3. Fixed-variable substitution
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_arrays, "vector requires numpy/scipy")
class TestFixedSubstitution(unittest.TestCase):
    def test_stdform_bulk_fix_matches_classic(self):
        m = _vector_model()
        m.x.fix(2.0, where=np.array([0]))
        self.assertEqual(_vector_std(m), _classic_std(_classic_model(fix={0: 2.0})))

    def test_stdform_per_element_fix(self):
        m = _vector_model()
        m.x[3].fix(1.0)
        self.assertEqual(_vector_std(m), _classic_std(_classic_model(fix={3: 1.0})))

    def test_fix_multiple(self):
        m = _vector_model()
        m.x.fix(np.array([2.0, 1.0]), where=np.array([0, 3]))
        self.assertEqual(
            _vector_std(m), _classic_std(_classic_model(fix={0: 2.0, 3: 1.0}))
        )

    @unittest.skipUnless(_solve, "solve requires highspy")
    def test_fixed_solve_matches_classic(self):
        from pyomo.contrib.vector.highs import solve_highs

        m = _vector_model()
        m.x.fix(1.5, where=np.array([1]))
        _, obj = solve_highs(m)
        ref = _solve_classic(_classic_model(fix={1: 1.5}))
        self.assertAlmostEqual(obj, ref, places=6)


# --------------------------------------------------------------------------- #
# 4. Solution load-back (array-native read path)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_solve, "solve requires highspy")
class TestSolutionLoadBack(unittest.TestCase):
    def test_load_solution_scatters_into_arrays(self):
        from pyomo.contrib.vector.highs import solve_highs

        m = _vector_model()
        _, obj = solve_highs(m, load_solutions=True)
        # x.value_array now holds the primal solution; the objective it implies
        # matches the reported objective.
        c = np.array([-1.0, -2.0, -1.0, -1.0])
        self.assertAlmostEqual(float(c @ m.x.value_array), obj, places=6)
        # per-index view reads the same array slot
        self.assertAlmostEqual(m.x[1].value, m.x.value_array[1], places=12)

    def test_load_solution_returns_count(self):
        from pyomo.contrib.vector.highs import load_highs, load_solution

        m = _vector_model()
        h = load_highs(m)
        h.run()
        self.assertEqual(load_solution(m, h), 4)


# --------------------------------------------------------------------------- #
# 5. Persistent warm re-solve (bounds / fix / mask cycle)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_solve, "solve requires highspy")
class TestPersistentWarmResolve(unittest.TestCase):
    def test_bounds_mutation_matches_classic(self):
        from pyomo.contrib.vector import VectorPersistentHighs

        m = _vector_model()
        p = VectorPersistentHighs(m)
        r0 = p.solve()
        self.assertAlmostEqual(r0.objective, _solve_classic(_classic_model()), places=6)
        m.x.setub(1.0, where=np.array([1]))  # tighten x1
        r1 = p.solve()
        self.assertAlmostEqual(
            r1.objective, _solve_classic(_classic_model(ub={1: 1.0})), places=6
        )
        self.assertEqual(p.n_solves, 2)

    def test_fix_then_unfix_cycle(self):
        from pyomo.contrib.vector import VectorPersistentHighs

        m = _vector_model()
        p = VectorPersistentHighs(m)
        p.solve()
        m.x.fix(2.0, where=np.array([0]))
        r1 = p.solve()
        self.assertAlmostEqual(
            r1.objective, _solve_classic(_classic_model(fix={0: 2.0})), places=6
        )
        m.x.unfix(where=np.array([0]))
        r2 = p.solve()
        self.assertAlmostEqual(r2.objective, _solve_classic(_classic_model()), places=6)

    def test_masked_deactivation_warm(self):
        from pyomo.contrib.vector import VectorPersistentHighs

        m = _vector_model()
        p = VectorPersistentHighs(m)
        p.solve()
        m.con.deactivate_rows(np.array([0]))
        r1 = p.solve()
        self.assertAlmostEqual(
            r1.objective, _solve_classic(_classic_model(deactivate_rows=(0,))), places=6
        )
        m.con.activate_rows()
        r2 = p.solve()
        self.assertAlmostEqual(r2.objective, _solve_classic(_classic_model()), places=6)

    def test_rhs_mutation_warm(self):
        from pyomo.contrib.vector import VectorPersistentHighs

        m = _vector_model()
        p = VectorPersistentHighs(m)
        p.solve()
        m.con.set_row_bounds(ub=2.0, where=np.array([0]))  # x0+x1 <= 2
        r1 = p.solve()
        self.assertAlmostEqual(
            r1.objective, _solve_classic(_classic_model(rhs={0: 2.0})), places=6
        )

    def test_warm_readback_updates_value_arrays(self):
        from pyomo.contrib.vector import VectorPersistentHighs

        m = _vector_model()
        p = VectorPersistentHighs(m)
        r = p.solve()
        self.assertTrue(np.allclose(m.x.value_array, r.col_value))

    def test_structural_guard_fires(self):
        from pyomo.contrib.vector import VectorPersistentHighs, PersistentStructureError

        m = _vector_model()
        p = VectorPersistentHighs(m)
        p.solve()
        m.x._n += 1  # simulate a structural change
        try:
            self.assertRaises(PersistentStructureError, p.solve)
        finally:
            m.x._n -= 1

    def test_combined_mutation_sweep_matches_cold(self):
        # A multi-kind mutation then warm re-solve must match a fresh cold solve
        # of the identically-mutated model (the incremental push is exact).
        from pyomo.contrib.vector import VectorPersistentHighs
        from pyomo.contrib.vector.highs import solve_highs

        m = _vector_model()
        p = VectorPersistentHighs(m)
        p.solve()
        m.x.setub(3.0, where=np.array([2]))
        m.x.fix(1.0, where=np.array([3]))
        m.con.deactivate_rows(np.array([2]))
        r = p.solve()
        # cold: build fresh, apply the same mutation, one-shot solve
        m2 = _vector_model()
        m2.x.setub(3.0, where=np.array([2]))
        m2.x.fix(1.0, where=np.array([3]))
        m2.con.deactivate_rows(np.array([2]))
        _, cold = solve_highs(m2)
        self.assertAlmostEqual(r.objective, cold, places=6)


# --------------------------------------------------------------------------- #
# 6. Mixed classic+vector model: compatibility contract
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_arrays, "vector requires numpy/scipy")
class TestMixedModelContract(unittest.TestCase):
    def test_classic_constraint_disables_fast_path(self):
        from pyomo.contrib.vector import VectorPathDisabledError, assemble

        m = _vector_model()
        # add a classic constraint alongside the vector one
        m.extra = pyo.Constraint(expr=m.x[0] + m.x[3] <= 6)
        with self.assertRaises(VectorPathDisabledError):
            assemble(m)

    def test_scalarized_component_disables_fast_path(self):
        from pyomo.contrib.vector import VectorPathDisabledError, assemble

        m = _vector_model()
        m.x._scalarize(reason="test")  # force scalarization
        with self.assertRaises(VectorPathDisabledError):
            assemble(m)

    @unittest.skipUnless(_solve, "solve requires highspy")
    def test_scalarized_classic_solve_still_matches(self):
        # After scalarization the model is a classic model; a classic solve of it
        # must still match the reference objective (the fallback is correct, just
        # not fast).
        m = _vector_model()
        m.x._scalarize(reason="test")
        m.con._scalarize(reason="test")
        self.assertIsNotNone(m.obj.expr)  # objective scalarizes to a classic expr
        val = _solve_classic_model_inplace(m)
        self.assertAlmostEqual(val, _solve_classic(_classic_model()), places=6)


def _solve_classic_model_inplace(m):
    from pyomo.contrib.appsi.solvers.highs import Highs

    Highs().solve(m)
    return float(pyo.value(m.obj))


if __name__ == "__main__":
    unittest.main()
