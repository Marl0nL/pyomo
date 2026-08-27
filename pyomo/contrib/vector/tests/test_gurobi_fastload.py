# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the Gurobi fast solver hand-off (``gurobi_fastload``).

The Gurobi twin of ``highs_fastload``: it compiles an *unmodified* classic model
to standard form (the same solver-neutral compile the HiGHS backend uses) and
hands the whole matrix to Gurobi via its native matrix API
(``addMVar`` + ``addMConstr`` + ``setMObjective``), solves, and maps the solution
back onto the model.  These tests check:

* the solver is registered under a normal name (v2 + legacy factories);
* solve results (objective, termination, primals, duals, reduced costs) are
  correct -- maximize + offset, MIP integrality, fixed variables, range
  constraints, objective-free feasibility, a convex-QP objective;
* solve-result **equivalence** with ``highs_fastload`` *and* the classic
  reference on the randomized-model suite + a random convex QP -- the
  cross-backend correctness gate;
* the scope guards reject nonlinear / non-convex / MIQP models loudly.

**License note.**  The pip ``gurobipy`` wheel ships a *size-limited* license
(2000 variables and 2000 constraints).  Every model built here stays well under
that ceiling, so these are correctness/equivalence checks at licensed sizes only
-- they make no large-scale claim.  When ``gurobipy`` is absent, or its license
cannot solve a model, the affected tests *skip* (they never fail), so the suite
stays green without Gurobi.
"""

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
    attempt_import,
)

gurobipy, gurobipy_available = attempt_import('gurobipy')
highspy, highspy_available = attempt_import('highspy')
scipy_sparse, _ = attempt_import('scipy.sparse')

_have_np_scipy = numpy_available and scipy_available


def _gurobi_fastload():
    from pyomo.contrib.solver.common.factory import SolverFactory

    return SolverFactory('gurobi_fastload')


def _highs_fastload():
    from pyomo.contrib.solver.common.factory import SolverFactory

    return SolverFactory('highs_fastload')


def _v2_highs():
    from pyomo.contrib.solver.common.factory import SolverFactory

    return SolverFactory('highs')


def _gurobi_ok():
    """True iff ``gurobi_fastload`` has a usable (size-limited is fine) license."""
    if not (_have_np_scipy and gurobipy_available):
        return False
    try:
        import pyomo.contrib.vector  # noqa: F401  (registers the solver)

        return bool(_gurobi_fastload().available())
    except Exception:
        return False


def _highs_ok():
    if not (_have_np_scipy and highspy_available):
        return False
    try:
        import pyomo.contrib.vector  # noqa: F401

        return bool(_highs_fastload().available())
    except Exception:
        return False


_GUROBI = _gurobi_ok()
_HIGHS = _highs_ok()


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
@unittest.skipUnless(
    _have_np_scipy and gurobipy_available, "requires numpy/scipy/gurobipy"
)
class TestGurobiFastLoadRegistration(unittest.TestCase):
    def test_registered_v2(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.factory import SolverFactory

        self.assertIn('gurobi_fastload', SolverFactory)
        solver = SolverFactory('gurobi_fastload')
        self.assertEqual(solver.name, 'gurobi_fastload')

    def test_registered_legacy(self):
        import pyomo.contrib.vector  # noqa: F401

        solver = pyo.SolverFactory('gurobi_fastload')
        # Resolves through the legacy wrapper; availability is a runtime property
        # (a license may be absent), so we only assert the wrapper exists.
        self.assertIsNotNone(solver)


# --------------------------------------------------------------------------- #
# Core solve behavior
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_GUROBI, "gurobi_fastload requires a usable gurobipy license")
class TestGurobiFastLoadSolve(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition
        self.opt = _gurobi_fastload()

    def test_simple_lp(self):
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
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None))
        m.c = pyo.Constraint(expr=m.x <= 3)
        m.obj = pyo.Objective(expr=m.x + 10, sense=pyo.maximize)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 13.0, places=6)
        self.assertAlmostEqual(pyo.value(m.x), 3.0, places=6)

    def test_mip_integrality(self):
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
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 2))
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1.0, places=6)
        duals = res.solution_loader.get_duals()
        # The range constraint splits into two Gurobi rows but maps back to one con.
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
        self.assertIn(
            res.termination_condition,
            (self.TC.provenInfeasible, self.TC.infeasibleOrUnbounded),
        )

    def test_trivially_infeasible_at_compile(self):
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
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x <= 4)
        m.obj = pyo.Objective(expr=m.x**3, sense=pyo.maximize)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_scope_error_names_gurobi_backend(self):
        # The fail-loud message must name *this* backend, not the HiGHS one whose
        # compile it shares (the solver-neutral seam threads the solver name).
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x**2 <= 4)
        m.obj = pyo.Objective(expr=m.x)
        with self.assertRaises(IncompatibleModelError) as ctx:
            self.opt.solve(m)
        self.assertIn('gurobi_fastload', str(ctx.exception))


# --------------------------------------------------------------------------- #
# Convex-QP objective + fail-loud QP guards
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_GUROBI, "gurobi_fastload requires a usable gurobipy license")
class TestGurobiFastLoadQuadratic(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401

        self.opt = _gurobi_fastload()

    def test_analytic_convex_qp(self):
        # min 0.5(2 x0^2 + 4 x1^2) - 2 x0 - 8 x1 over a wide box -> x* = [1, 2].
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], bounds=(-100, 100))
        m.obj = pyo.Objective(
            expr=0.5 * (2 * m.x[0] ** 2 + 4 * m.x[1] ** 2) - 2 * m.x[0] - 8 * m.x[1],
            sense=pyo.minimize,
        )
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -9.0, places=5)
        self.assertAlmostEqual(pyo.value(m.x[0]), 1.0, places=5)
        self.assertAlmostEqual(pyo.value(m.x[1]), 2.0, places=5)

    def test_maximize_concave_qp(self):
        # max 6x - x^2 over [0, 10] -> x* = 3, obj = 9.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.obj = pyo.Objective(expr=6 * m.x - m.x**2, sense=pyo.maximize)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 9.0, places=5)
        self.assertAlmostEqual(pyo.value(m.x), 3.0, places=5)

    def test_offdiagonal_hessian(self):
        # min (x0 - x1)^2 + x0 + x1 s.t. x0 + x1 == 1 over [0,1]^2.
        # (x0-x1)^2 has an off-diagonal cross term -> exercises the symmetric-Q
        # reconstruction.  By symmetry the optimum is x0 = x1 = 0.5, obj = 1.
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], bounds=(0, 1))
        m.bal = pyo.Constraint(expr=m.x[0] + m.x[1] == 1)
        m.obj = pyo.Objective(
            expr=(m.x[0] - m.x[1]) ** 2 + m.x[0] + m.x[1], sense=pyo.minimize
        )
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1.0, places=5)
        self.assertAlmostEqual(pyo.value(m.x[0]), 0.5, places=5)
        self.assertAlmostEqual(pyo.value(m.x[1]), 0.5, places=5)

    def test_objective_only_variable(self):
        # A variable that appears only in the (quadratic) objective must still be
        # a column (the compiler eliminates it; the QP route re-adds it).
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], bounds=(0, None))
        m.z = pyo.Var(bounds=(-1, 1))
        m.bal = pyo.Constraint(expr=m.x[0] + m.x[1] == 1)
        m.obj = pyo.Objective(
            expr=0.5 * (m.x[0] ** 2 + m.x[1] ** 2) + m.z**2 - m.x[0], sense=pyo.minimize
        )
        self.opt.solve(m)
        self.assertAlmostEqual(pyo.value(m.z), 0.0, places=5)

    def test_nonconvex_fails_loud(self):
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-5, 5))
        m.c = pyo.Constraint(expr=m.x <= 4)
        m.obj = pyo.Objective(expr=-m.x**2, sense=pyo.minimize)  # concave -> non-convex
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_miqp_fails_loud(self):
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], domain=pyo.Integers, bounds=(0, 5))
        m.obj = pyo.Objective(expr=(m.x[0] - 2.4) ** 2 + (m.x[1] - 1.6) ** 2)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_quadratic_constraint_stays_out_of_scope(self):
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x**2 <= 4)
        m.obj = pyo.Objective(expr=m.x)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)


# --------------------------------------------------------------------------- #
# Cross-backend + classic-reference equivalence (licensed sizes only)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_GUROBI, "gurobi_fastload requires a usable gurobipy license")
class TestGurobiFastLoadEquivalence(unittest.TestCase):
    """gurobi_fastload agrees with highs_fastload and the classic reference."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition
        self.gurobi = _gurobi_fastload()

    def test_matches_classic_on_random_models(self):
        # Same randomized-LP suite the HiGHS equivalence gate uses; here the
        # reference is the classic persistent APPSI HiGHS route.
        from pyomo.contrib.vector.tests import random_models as rm

        appsi = pyo.SolverFactory('appsi_highs')
        if not appsi.available(exception_flag=False):
            self.skipTest("appsi_highs reference not available")

        rng = np.random.default_rng(20260828)
        checked = 0
        for _ in range(40):
            case = rm.Case(rng)
            mc = rm.build_classic(case)
            try:
                appsi.solve(mc)
                obj_classic = pyo.value(mc.obj)
            except Exception:
                continue
            if obj_classic is None or not np.isfinite(obj_classic):
                continue

            mg = rm.build_classic(case)
            res = self.gurobi.solve(mg, raise_exception_on_nonoptimal_result=False)
            if res.termination_condition != self.TC.convergenceCriteriaSatisfied:
                self.fail(
                    f"gurobi_fastload did not converge "
                    f"(tc={res.termination_condition}) while the classic route "
                    f"found objective {obj_classic}"
                )
            self.assertAlmostEqual(
                res.incumbent_objective,
                obj_classic,
                delta=1e-5 * max(1.0, abs(obj_classic)),
                msg=f"objective mismatch: gurobi={res.incumbent_objective} "
                f"classic={obj_classic}",
            )
            checked += 1
        self.assertGreater(checked, 5, "too few random cases were solvable")

    @unittest.skipUnless(_HIGHS, "highs_fastload cross-check unavailable")
    def test_matches_highs_fastload_on_random_models(self):
        # The direct cross-backend gate: the two fast paths, same compile, must
        # report the same objective on the same randomized models.
        from pyomo.contrib.vector.tests import random_models as rm

        highs = _highs_fastload()
        rng = np.random.default_rng(13131)
        checked = 0
        for _ in range(40):
            case = rm.Case(rng)
            mg = rm.build_classic(case)
            mh = rm.build_classic(case)
            # Compare objective/termination only; some random models are
            # infeasible/unbounded, so do not force a solution load.
            rg = self.gurobi.solve(
                mg, load_solutions=False, raise_exception_on_nonoptimal_result=False
            )
            rh = highs.solve(
                mh, load_solutions=False, raise_exception_on_nonoptimal_result=False
            )
            # Both backends should agree on *solvability* (whether a finite
            # optimum exists).  They may label an infeasible-or-unbounded model
            # differently (Gurobi's INF_OR_UNBD vs HiGHS's unbounded/infeasible),
            # so compare the convergence class rather than the exact condition.
            g_conv = rg.termination_condition == self.TC.convergenceCriteriaSatisfied
            h_conv = rh.termination_condition == self.TC.convergenceCriteriaSatisfied
            self.assertEqual(
                g_conv,
                h_conv,
                msg=f"solvability mismatch: gurobi={rg.termination_condition} "
                f"highs={rh.termination_condition}",
            )
            if not g_conv:
                continue
            self.assertAlmostEqual(
                rg.incumbent_objective,
                rh.incumbent_objective,
                delta=1e-5 * max(1.0, abs(rh.incumbent_objective)),
                msg=f"objective mismatch: gurobi={rg.incumbent_objective} "
                f"highs={rh.incumbent_objective}",
            )
            checked += 1
        self.assertGreater(checked, 5, "too few random cases were solvable")

    def test_unique_lp_primal_values_match_classic(self):
        appsi = pyo.SolverFactory('appsi_highs')
        if not appsi.available(exception_flag=False):
            self.skipTest("appsi_highs reference not available")

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1, 2], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c1 = pyo.Constraint(expr=m.x[0] + m.x[1] + m.x[2] == 6)
        m.c2 = pyo.Constraint(expr=m.x[0] - m.x[1] == 1)
        m.c3 = pyo.Constraint(expr=m.x[1] - m.x[2] == 1)
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])

        mref = m.clone()
        appsi.solve(mref)
        self.gurobi.solve(m)
        for i in (0, 1, 2):
            self.assertAlmostEqual(pyo.value(m.x[i]), pyo.value(mref.x[i]), places=6)

    @unittest.skipUnless(_HIGHS, "highs_fastload / v2 highs QP reference unavailable")
    def test_convex_qp_matches_reference(self):
        # A random convex QP solved by gurobi_fastload, highs_fastload, and the
        # classic v2-highs reference must agree on objective *and* (unique) primal.
        ref = _v2_highs()
        if not ref.available():
            self.skipTest("v2 highs QP reference not available")
        highs = _highs_fastload()

        rng = np.random.default_rng(97531)
        n = 5
        for _ in range(10):
            M = rng.normal(size=(n, n))
            Q = M.T @ M + np.eye(n)  # SPD -> strictly convex, unique optimum
            Q = 0.5 * (Q + Q.T)
            c = rng.normal(size=n)

            def build():
                mm = pyo.ConcreteModel()
                mm.I = pyo.RangeSet(0, n - 1)
                mm.x = pyo.Var(mm.I, bounds=(0.0, 1.0))
                mm.bal = pyo.Constraint(expr=sum(mm.x[i] for i in mm.I) == 1)
                quad = 0.5 * sum(
                    Q[i, j] * mm.x[i] * mm.x[j] for i in range(n) for j in range(n)
                )
                mm.obj = pyo.Objective(
                    expr=quad + sum(c[i] * mm.x[i] for i in mm.I), sense=pyo.minimize
                )
                return mm

            mg, mh, mr = build(), build(), build()
            rg = self.gurobi.solve(mg)
            rh = highs.solve(mh)
            ref.solve(mr)
            self.assertAlmostEqual(rg.incumbent_objective, pyo.value(mr.obj), places=5)
            self.assertAlmostEqual(
                rg.incumbent_objective, rh.incumbent_objective, places=5
            )
            for i in range(n):
                self.assertAlmostEqual(pyo.value(mg.x[i]), pyo.value(mr.x[i]), places=4)


if __name__ == "__main__":
    unittest.main()
