# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for convex-quadratic **objective** support on the vector fast path.

Covers the three hand-off routes (scoping doc §6, Phase-3 quadratic ambition,
the #1761 use case), all objective-quadratic only (constraints stay linear):

* the explicit-array API -- ``VectorObjective(..., quadratic=Q)`` solved through
  :func:`~pyomo.contrib.vector.highs.solve_highs`;
* the transparent classic route -- a classic Pyomo model with an
  ``x[i]*x[j]``-built quadratic objective solved through ``highs_fastload``;
* the persistent warm route -- :class:`FastStepHighs` with a *static* Hessian and
  a changing linear cost (the rolling-horizon portfolio path).

Equivalence gate: solves match the classic reference -- Pyomo's v2
``SolverFactory('highs')`` persistent interface (which solves QP; the legacy
``appsi_highs`` does not) -- on randomized convex QPs, plus an analytic
mini-problem with a known optimum.  Scope guards: MIQP and non-convex QPs are
rejected loudly; quadratic *constraints* stay out of scope (fail-loud).
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
scipy_sparse, _ = attempt_import('scipy.sparse')

_deps = numpy_available and scipy_available and highspy_available


def _v2_highs():
    from pyomo.contrib.solver.common.factory import SolverFactory

    return SolverFactory('highs')


def _reference_available():
    if not _deps:
        return False
    try:
        return bool(_v2_highs().available())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Randomized convex-QP generator (shared arrays; built two ways)
# --------------------------------------------------------------------------- #
def _random_convex_qp(rng, n=None):
    """A small feasible convex QP: ``min 0.5 x'Q x + c'x`` s.t. ``sum x = 1``,
    ``0 <= x <= 1``.  ``Q`` is SPD (``M'M + I``), so the problem is convex and
    always feasible (the simplex is nonempty)."""
    if n is None:
        n = int(rng.integers(2, 6))
    M = rng.normal(size=(n, n))
    Q = M.T @ M + np.eye(n)  # SPD
    Q = 0.5 * (Q + Q.T)  # exact symmetry
    c = rng.normal(size=n)
    return {'n': n, 'Q': Q, 'c': c}


def _build_vector_qp(case):
    from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective

    n = case['n']
    m = pyo.ConcreteModel()
    m.x = VectorVar(pyo.RangeSet(0, n - 1), bounds=(0.0, 1.0))
    A = scipy_sparse.csr_matrix(np.ones((1, n)))
    m.bal = VectorConstraint(A=A, x=m.x, rhs=np.array([1.0]))
    m.obj = VectorObjective(
        terms={m.x: case['c']},
        quadratic=scipy_sparse.csr_matrix(case['Q']),
        sense=pyo.minimize,
    )
    m.x.construct()
    m.bal.construct()
    m.obj.construct()
    return m


def _build_classic_qp(case):
    n = case['n']
    Q, c = case['Q'], case['c']
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, bounds=(0.0, 1.0))
    m.bal = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 1)
    quad = 0.5 * sum(
        Q[i, j] * m.x[i] * m.x[j] for i in range(n) for j in range(n)
    )
    m.obj = pyo.Objective(
        expr=quad + sum(c[i] * m.x[i] for i in m.I), sense=pyo.minimize
    )
    return m


# --------------------------------------------------------------------------- #
# Explicit-array vector QP (solve_highs)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_deps, "quadratic fast path requires numpy/scipy/highspy")
class TestVectorQuadraticObjective(unittest.TestCase):
    def test_analytic_unconstrained_optimum(self):
        # min 0.5 x'Q x + c'x over a wide box -> interior optimum x* = -Q^{-1} c.
        from pyomo.contrib.vector import VectorVar, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs

        Q = np.array([[2.0, 0.0], [0.0, 4.0]])
        c = np.array([-2.0, -8.0])
        xstar = np.linalg.solve(Q, -c)  # [1, 2]
        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 1), bounds=(-100.0, 100.0))
        m.obj = VectorObjective(
            terms={m.x: c}, quadratic=scipy_sparse.csr_matrix(Q), sense=pyo.minimize
        )
        m.x.construct()
        m.obj.construct()
        h, obj = solve_highs(m)
        xv = np.array(h.getSolution().col_value)
        self.assertTrue(np.allclose(xv, xstar, atol=1e-6))
        self.assertAlmostEqual(obj, 0.5 * xstar @ Q @ xstar + c @ xstar, places=6)

    @unittest.skipUnless(_reference_available(), "v2 highs QP reference unavailable")
    def test_matches_reference_on_random_qp(self):
        rng = np.random.default_rng(20260828)
        ref = _v2_highs()
        for _ in range(30):
            case = _random_convex_qp(rng)
            from pyomo.contrib.vector.highs import solve_highs

            mv = _build_vector_qp(case)
            h, obj = solve_highs(mv)
            xv = np.array(h.getSolution().col_value)

            mc = _build_classic_qp(case)
            ref.solve(mc)
            xr = np.array([pyo.value(mc.x[i]) for i in range(case['n'])])
            self.assertAlmostEqual(obj, pyo.value(mc.obj), places=5)
            self.assertTrue(np.allclose(xv, xr, atol=1e-4))

    def test_multi_block_hessian(self):
        # Two VectorVar blocks with a diagonal Hessian on each plus a cross block.
        from pyomo.contrib.vector import VectorVar, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs

        m = pyo.ConcreteModel()
        m.a = VectorVar(pyo.RangeSet(0, 1), bounds=(-10.0, 10.0))
        m.b = VectorVar(pyo.RangeSet(0, 1), bounds=(-10.0, 10.0))
        Qaa = scipy_sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))
        Qbb = scipy_sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))
        Qab = scipy_sparse.csr_matrix(np.array([[0.5, 0.0], [0.0, 0.5]]))
        m.obj = VectorObjective(
            terms={m.a: np.array([-1.0, -1.0]), m.b: np.array([-1.0, -1.0])},
            quadratic={(m.a, m.a): Qaa, (m.b, m.b): Qbb, (m.a, m.b): Qab},
            sense=pyo.minimize,
        )
        for comp in (m.a, m.b, m.obj):
            comp.construct()
        h, obj = solve_highs(m)
        # Reference: build the equivalent classic model and solve with v2 highs.
        if _reference_available():
            mc = pyo.ConcreteModel()
            mc.a = pyo.Var([0, 1], bounds=(-10, 10))
            mc.b = pyo.Var([0, 1], bounds=(-10, 10))
            quad = (
                sum(mc.a[i] ** 2 for i in (0, 1))
                + sum(mc.b[i] ** 2 for i in (0, 1))
                + 0.5 * sum(mc.a[i] * mc.b[i] for i in (0, 1))
            )
            mc.obj = pyo.Objective(
                expr=quad - sum(mc.a[i] + mc.b[i] for i in (0, 1)), sense=pyo.minimize
            )
            _v2_highs().solve(mc)
            self.assertAlmostEqual(obj, pyo.value(mc.obj), places=5)

    def test_maximize_concave(self):
        from pyomo.contrib.vector import VectorVar, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs

        # max 6x - x^2 over [0, 10] -> x* = 3, obj = 9.
        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 0), bounds=(0.0, 10.0))
        m.obj = VectorObjective(
            terms={m.x: np.array([6.0])},
            quadratic=scipy_sparse.csr_matrix([[-2.0]]),
            sense=pyo.maximize,
        )
        m.x.construct()
        m.obj.construct()
        h, obj = solve_highs(m)
        self.assertAlmostEqual(float(h.getSolution().col_value[0]), 3.0, places=5)
        self.assertAlmostEqual(obj, 9.0, places=5)

    def test_constant_offset_included(self):
        from pyomo.contrib.vector import VectorVar, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs

        # min 0.5(x-3)^2 = 0.5 x^2 - 3x + 4.5 -> optimum 0 at x = 3.
        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 0), bounds=(0.0, 10.0))
        m.obj = VectorObjective(
            terms={m.x: np.array([-3.0])},
            quadratic=scipy_sparse.csr_matrix([[1.0]]),
            constant=4.5,
            sense=pyo.minimize,
        )
        m.x.construct()
        m.obj.construct()
        _, obj = solve_highs(m)
        self.assertAlmostEqual(obj, 0.0, places=6)

    def test_miqp_fails_loud(self):
        from pyomo.contrib.vector import VectorVar, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs, QuadraticModelError

        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 1), domain=pyo.Integers, bounds=(0.0, 5.0))
        m.obj = VectorObjective(
            terms={m.x: np.array([-2.4, -1.6])},
            quadratic=scipy_sparse.eye(2, format='csr'),
            sense=pyo.minimize,
        )
        m.x.construct()
        m.obj.construct()
        with self.assertRaises(QuadraticModelError):
            solve_highs(m)

    def test_nonconvex_fails_loud(self):
        from pyomo.contrib.vector import VectorVar, VectorObjective
        from pyomo.contrib.vector.highs import solve_highs, QuadraticModelError

        # minimize a concave objective (negative-definite Hessian) -> non-convex.
        m = pyo.ConcreteModel()
        m.x = VectorVar(pyo.RangeSet(0, 0), bounds=(-5.0, 5.0))
        m.obj = VectorObjective(
            terms={m.x: np.array([0.0])},
            quadratic=scipy_sparse.csr_matrix([[-1.0]]),
            sense=pyo.minimize,
        )
        m.x.construct()
        m.obj.construct()
        with self.assertRaises(QuadraticModelError):
            solve_highs(m)


# --------------------------------------------------------------------------- #
# Transparent classic route (highs_fastload)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_deps, "quadratic fast path requires numpy/scipy/highspy")
class TestFastLoadQuadratic(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401  (registers the solver)
        from pyomo.contrib.solver.common.factory import SolverFactory

        self.fast = SolverFactory('highs_fastload')

    @unittest.skipUnless(_reference_available(), "v2 highs QP reference unavailable")
    def test_matches_reference_on_random_qp(self):
        rng = np.random.default_rng(4242)
        ref = _v2_highs()
        for _ in range(30):
            case = _random_convex_qp(rng)
            mf = _build_classic_qp(case)
            mr = _build_classic_qp(case)
            self.fast.solve(mf)
            ref.solve(mr)
            of = pyo.value(mf.obj)
            orr = pyo.value(mr.obj)
            self.assertAlmostEqual(of, orr, places=5)
            for i in range(case['n']):
                self.assertAlmostEqual(
                    pyo.value(mf.x[i]), pyo.value(mr.x[i]), places=4
                )

    def test_objective_only_variable_column_extension(self):
        # A variable that appears only in the (quadratic) objective -- not in any
        # constraint -- must still be a column (the compiler eliminates it, the
        # QP route re-adds it).
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], bounds=(0, None))
        m.z = pyo.Var(bounds=(-1, 1))  # objective-only
        m.bal = pyo.Constraint(expr=m.x[0] + m.x[1] == 1)
        m.obj = pyo.Objective(
            expr=0.5 * (m.x[0] ** 2 + m.x[1] ** 2) + m.z**2 - m.x[0],
            sense=pyo.minimize,
        )
        self.fast.solve(m)
        # z decouples: its optimum is 0 (unconstrained min of z^2 in [-1, 1]).
        self.assertAlmostEqual(pyo.value(m.z), 0.0, places=6)

    def test_fixed_variable_in_quadratic_term(self):
        # A fixed variable in a quadratic term folds into the linear/constant part
        # consistently with the constraint compiler's fixed-var elimination.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.p = pyo.Var(bounds=(-10, 10))
        m.p.fix(3.0)
        m.obj = pyo.Objective(
            expr=0.5 * m.x**2 + m.x * m.p, sense=pyo.minimize
        )  # min 0.5 x^2 + 3x -> x* = -3
        self.fast.solve(m)
        self.assertAlmostEqual(pyo.value(m.x), -3.0, places=5)

    def test_miqp_fails_loud(self):
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], domain=pyo.Integers, bounds=(0, 5))
        m.obj = pyo.Objective(expr=(m.x[0] - 2.4) ** 2 + (m.x[1] - 1.6) ** 2)
        with self.assertRaises(IncompatibleModelError):
            self.fast.solve(m)

    def test_nonconvex_fails_loud(self):
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-5, 5))
        m.c = pyo.Constraint(expr=m.x <= 4)
        m.obj = pyo.Objective(expr=-m.x**2, sense=pyo.minimize)  # concave -> non-convex
        with self.assertRaises(IncompatibleModelError):
            self.fast.solve(m)

    def test_quadratic_constraint_stays_out_of_scope(self):
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x**2 <= 4)  # quadratic constraint: not supported
        m.obj = pyo.Objective(expr=m.x)
        with self.assertRaises(IncompatibleModelError):
            self.fast.solve(m)


# --------------------------------------------------------------------------- #
# Persistent warm route (FastStepHighs)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_deps, "quadratic fast path requires numpy/scipy/highspy")
class TestFastStepQuadratic(unittest.TestCase):
    def _portfolio(self, mu0):
        n = len(mu0)
        rng = np.random.default_rng(7)
        M = rng.normal(size=(n, n))
        Q = M.T @ M + np.eye(n)
        Q = 0.5 * (Q + Q.T)
        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, n - 1)
        m.x = pyo.Var(m.I, bounds=(0, None))
        m.mu = pyo.Param(m.I, mutable=True, initialize=dict(enumerate(mu0)))
        m.bal = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 1)
        quad = 0.5 * sum(Q[i, j] * m.x[i] * m.x[j] for i in range(n) for j in range(n))
        m.obj = pyo.Objective(
            expr=quad - sum(m.mu[i] * m.x[i] for i in m.I), sense=pyo.minimize
        )
        m._Q = Q
        return m

    @unittest.skipUnless(_reference_available(), "v2 highs QP reference unavailable")
    def test_static_hessian_warm_rolls_match_reference(self):
        from pyomo.contrib.vector.faststep import FastStepHighs

        rolls = [
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([2.0, 0.5, 1.0, 1.5]),
            np.array([0.2, 2.0, 0.0, 3.0]),
        ]
        m = self._portfolio(rolls[0])
        step = FastStepHighs()
        step.set_instance(m)
        ref = _v2_highs()
        for mu in rolls:
            for i in range(len(mu)):
                m.mu[i] = mu[i]
            res = step.solve()
            xw = np.array([pyo.value(m.x[i]) for i in range(len(mu))])
            mr = self._portfolio(mu)
            ref.solve(mr)
            xr = np.array([pyo.value(mr.x[i]) for i in range(len(mu))])
            self.assertAlmostEqual(res.incumbent_objective, pyo.value(mr.obj), places=5)
            self.assertTrue(np.allclose(xw, xr, atol=1e-4))

    def test_static_mutable_hessian_param_loads_and_guards(self):
        # A mutable Param that feeds the Hessian is allowed (watched constant);
        # a genuine change to it fails loud (the fold guard).
        from pyomo.contrib.vector.faststep import FastStepHighs
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], bounds=(0, None))
        m.a = pyo.Param(mutable=True, initialize=2.0)
        m.bal = pyo.Constraint(expr=m.x[0] + m.x[1] == 1)
        m.obj = pyo.Objective(
            expr=0.5 * m.a * m.x[0] ** 2 + 0.5 * m.x[1] ** 2 - m.x[0],
            sense=pyo.minimize,
        )
        step = FastStepHighs()
        step.set_instance(m)  # a feeds the Hessian only -> folded, loads fine
        step.solve()
        m.a = 5.0  # genuine change to a Hessian parameter
        with self.assertRaises(IncompatibleModelError):
            step.solve()

    def test_varying_hessian_reload_rebuilds(self):
        from pyomo.contrib.vector.faststep import FastStepHighs

        def build(a):
            m = pyo.ConcreteModel()
            m.x = pyo.Var([0, 1], bounds=(0, None))
            m.a = pyo.Param(mutable=True, initialize=a)
            m.bal = pyo.Constraint(expr=m.x[0] + m.x[1] == 1)
            m.obj = pyo.Objective(
                expr=0.5 * m.a * m.x[0] ** 2 + 0.5 * m.x[1] ** 2 - m.x[0],
                sense=pyo.minimize,
            )
            return m

        m = build(2.0)
        step = FastStepHighs(on_matrix_change='reload')
        step.set_instance(m)
        step.solve()
        m.a = 5.0
        res = step.solve()  # reload rebuilds the Hessian at a = 5
        if _reference_available():
            mr = build(5.0)
            _v2_highs().solve(mr)
            self.assertAlmostEqual(
                res.incumbent_objective, pyo.value(mr.obj), places=5
            )

    def test_hessian_param_that_also_varies_is_rejected(self):
        # A Param feeding both the Hessian and a live (varying) linear coefficient
        # would silently stale the Hessian between rolls -> reject at set_instance.
        from pyomo.contrib.vector.faststep import FastStepHighs
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], bounds=(0, None))
        m.a = pyo.Param(mutable=True, initialize=2.0)
        m.bal = pyo.Constraint(expr=m.x[0] + m.x[1] == 1)
        # a appears in the linear cost (varying/templated) AND the Hessian.
        m.obj = pyo.Objective(
            expr=0.5 * m.a * m.x[0] ** 2 + m.a * m.x[1] - m.x[0], sense=pyo.minimize
        )
        step = FastStepHighs()
        with self.assertRaises(IncompatibleModelError):
            step.set_instance(m)


if __name__ == '__main__':
    unittest.main()
