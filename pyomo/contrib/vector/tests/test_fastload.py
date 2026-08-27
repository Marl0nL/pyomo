# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the transparent fast solver hand-off (``highs_fastload``).

The solver compiles an *unmodified* classic linear model to standard form, hands
the whole matrix to HiGHS via ``passModel``, solves, and maps the solution back
onto the model.  These tests check:

* the solver is registered under a normal name (v2 + legacy factories);
* solve results (objective, termination, primals, duals, reduced costs) are
  correct, including maximize + objective offset, MIP integrality, fixed
  variables, range constraints, and objective-free feasibility;
* solve-result **equivalence** with the persistent APPSI HiGHS interface on
  randomized models (the classic route), the Phase-2 correctness gate;
* the scope guard rejects nonlinear / unsupported models loudly.
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

_deps = numpy_available and scipy_available and highspy_available


def _fastload():
    from pyomo.contrib.solver.common.factory import SolverFactory

    return SolverFactory('highs_fastload')


@unittest.skipUnless(_deps, "highs_fastload requires numpy/scipy/highspy")
class TestFastLoadRegistration(unittest.TestCase):
    def test_registered_v2(self):
        import pyomo.contrib.vector  # noqa: F401  (registers the solver)
        from pyomo.contrib.solver.common.factory import SolverFactory

        self.assertIn('highs_fastload', SolverFactory)
        solver = SolverFactory('highs_fastload')
        self.assertEqual(solver.name, 'highs_fastload')

    def test_registered_legacy(self):
        import pyomo.contrib.vector  # noqa: F401

        # The legacy SolverFactory wrapper must resolve the same name.
        solver = pyo.SolverFactory('highs_fastload')
        self.assertTrue(solver.available(exception_flag=False))


@unittest.skipUnless(_deps, "highs_fastload requires numpy/scipy/highspy")
class TestFastLoadSolve(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition
        self.opt = _fastload()

    def test_simple_lp(self):
        # min x0 + x1  s.t.  x0 + 2 x1 == 3 ;  x0 - x1 <= 1 ;  0 <= x <= 10
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c1 = pyo.Constraint(expr=m.x[0] + 2 * m.x[1] == 3)
        m.c2 = pyo.Constraint(expr=m.x[0] - m.x[1] <= 1)
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1], sense=pyo.minimize)

        res = self.opt.solve(m)
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        self.assertAlmostEqual(res.incumbent_objective, 1.5, places=6)
        self.assertAlmostEqual(pyo.value(m.x[0]), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(m.x[1]), 1.5, places=6)

        duals = res.solution_loader.get_duals()
        self.assertIn(m.c1, duals)
        rc = res.solution_loader.get_reduced_costs()
        self.assertIn(m.x[0], rc)

    def test_maximize_with_offset(self):
        # max x + 10  s.t.  x <= 3  ->  objective 13 at x = 3
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None))
        m.c = pyo.Constraint(expr=m.x <= 3)
        m.obj = pyo.Objective(expr=m.x + 10, sense=pyo.maximize)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 13.0, places=6)
        self.assertAlmostEqual(pyo.value(m.x), 3.0, places=6)

    def test_mip_integrality(self):
        # max x + y  s.t.  x + y <= 3.5 ;  x, y integer in [0, 10]  ->  obj 3
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, 10))
        m.y = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x + m.y <= 3.5)
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
        res = self.opt.solve(m)
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        self.assertAlmostEqual(res.incumbent_objective, 3.0, places=6)
        self.assertEqual(pyo.value(m.x) + pyo.value(m.y), 3.0)

    def test_fixed_variable(self):
        # x fixed at 2; min x + y s.t. x + y >= 5 -> y = 3, obj = 5.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.y = pyo.Var(bounds=(0, 10))
        m.x.fix(2.0)
        m.c = pyo.Constraint(expr=m.x + m.y >= 5)
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 5.0, places=6)
        self.assertAlmostEqual(pyo.value(m.x), 2.0, places=6)
        self.assertAlmostEqual(pyo.value(m.y), 3.0, places=6)

    def test_range_constraint(self):
        # 1 <= x + y <= 2, min x + y -> obj 1 (two-sided row).
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 2))
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1.0, places=6)
        duals = res.solution_loader.get_duals()
        # The range constraint splits into two rows but maps back to one con.
        self.assertIn(m.c, duals)

    def test_no_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x >= 2)
        res = self.opt.solve(m)
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        self.assertIsNone(res.incumbent_objective)
        self.assertGreaterEqual(pyo.value(m.x), 2.0 - 1e-6)

    def test_infeasible(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.c = pyo.Constraint(expr=m.x >= 2)
        m.obj = pyo.Objective(expr=m.x)
        res = self.opt.solve(
            m, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )
        self.assertEqual(res.termination_condition, self.TC.provenInfeasible)

    def test_trivially_infeasible_at_compile(self):
        # Every var in the row is fixed and the row is violated: caught at
        # compile time (InfeasibleConstraintException) before HiGHS runs.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.x.fix(0.5)
        m.c = pyo.Constraint(expr=m.x >= 2)
        m.y = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.y)
        res = self.opt.solve(
            m, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )
        self.assertEqual(res.termination_condition, self.TC.provenInfeasible)

    def test_unbounded(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None))
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)
        res = self.opt.solve(
            m, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )
        self.assertIn(
            res.termination_condition,
            (self.TC.unbounded, self.TC.infeasibleOrUnbounded),
        )

    def test_scope_guard_nonlinear(self):
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x**2 <= 4)
        m.obj = pyo.Objective(expr=m.x)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_scope_guard_nonlinear_objective(self):
        # A genuinely higher-order nonlinear objective (beyond quadratic) is
        # rejected loudly.  (A convex-quadratic objective is now *supported* --
        # see test_quadratic.py; a non-convex quadratic is rejected there.)
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x <= 4)
        m.obj = pyo.Objective(expr=m.x**3, sense=pyo.maximize)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)


@unittest.skipUnless(_deps, "highs_fastload requires numpy/scipy/highspy")
class TestFastLoadEquivalence(unittest.TestCase):
    """Fast-route solve results match the classic (APPSI HiGHS) route."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition
        self.fast = _fastload()
        self.appsi = pyo.SolverFactory('appsi_highs')
        if not self.appsi.available(exception_flag=False):
            self.skipTest("appsi_highs not available")

    def test_matches_classic_on_random_models(self):
        from pyomo.contrib.vector.tests import random_models as rm

        rng = np.random.default_rng(2026)
        checked = 0
        for _ in range(40):
            case = rm.Case(rng)
            mc = rm.build_classic(case)
            # Classic reference (persistent APPSI HiGHS).
            try:
                self.appsi.solve(mc)
            except Exception:
                continue
            obj_classic = None
            try:
                obj_classic = pyo.value(mc.obj)
            except Exception:
                continue
            if obj_classic is None or not np.isfinite(obj_classic):
                continue

            # Fast route on a fresh identical model.
            mf = rm.build_classic(case)
            res = self.fast.solve(mf, raise_exception_on_nonoptimal_result=False)
            if res.termination_condition != self.TC.convergenceCriteriaSatisfied:
                # Both routes should agree on solvability; if the fast route did
                # not converge, the classic one produced a finite objective, so
                # this is a real mismatch.
                self.fail(
                    f"fast route did not converge (tc={res.termination_condition}) "
                    f"while classic route found objective {obj_classic}"
                )
            # Compare the HiGHS-reported objective: like the shipped GurobiDirect
            # interface, the standard-form compile drops columns that are all-zero
            # in A and c, so re-evaluating the original ``mf.obj`` expression can
            # reference an eliminated (zero-contribution) variable that was left
            # unvalued.  The solver-reported objective is the true objective.
            obj_fast = res.incumbent_objective
            self.assertAlmostEqual(
                obj_fast,
                obj_classic,
                delta=1e-5 * max(1.0, abs(obj_classic)),
                msg=f"objective mismatch: fast={obj_fast} classic={obj_classic}",
            )
            checked += 1
        self.assertGreater(checked, 5, "too few random cases were solvable")

    def test_primal_values_match_on_unique_lp(self):
        # A strictly-convex-feasible LP with a unique optimum: primal values (not
        # just the objective) must match the classic route to solver tolerance.
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1, 2], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c1 = pyo.Constraint(expr=m.x[0] + m.x[1] + m.x[2] == 6)
        m.c2 = pyo.Constraint(expr=m.x[0] - m.x[1] == 1)
        m.c3 = pyo.Constraint(expr=m.x[1] - m.x[2] == 1)
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])

        mref = m.clone()
        self.appsi.solve(mref)
        self.fast.solve(m)
        for i in (0, 1, 2):
            self.assertAlmostEqual(pyo.value(m.x[i]), pyo.value(mref.x[i]), places=6)


if __name__ == "__main__":
    unittest.main()
