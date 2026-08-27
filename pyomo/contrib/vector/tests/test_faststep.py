# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the array-native persistent warm re-solve (``FastStepHighs``).

The interface compiles a classic linear model once, retains a live HiGHS, and on
each subsequent solve re-evaluates the changed objective coefficients / row
bounds / variable bounds as arrays (affine templates over the model's mutable
``Param`` vector) and batch-pushes them, keeping the warm basis.  These tests
check:

* **the warm-solve equivalence gate** -- a rolling sequence of solves through
  ``FastStepHighs`` matches a per-roll fresh ``highs_fastload`` build+solve
  (objective, termination, and -- for unique-optimum LPs -- primal values),
  for both basis-kept and basis-reset runs, on the synthetic MPC model and on
  randomized update sequences;
* the **array (mapping-free) update path** (``solve(param_values=...)``) and the
  **dirty-mask** partial-update path match the model-driven path;
* the **explicit index-addressed API** (``set_objective_coefficients`` etc.);
* the scope guards fail loud (nonlinear, structure change between solves);
* the **value-aware static-matrix guard**: a nominally-mutable matrix
  coefficient whose value never changes is accepted and warm-solved (matching a
  fresh build); a coefficient that genuinely changes trips the guard -- fail-loud
  by default, or a documented ``on_matrix_change='reload'`` rebuild -- never a
  stale-matrix solve;
* residual (non-parameter-affine / fixed-variable) entries stay correct.
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


# --------------------------------------------------------------------------- #
# A small synthetic multi-asset energy MPC model with mutable roll data.
#   objective coef  <- price[t]      (mutable)
#   row RHS          <- dem[a,t]      (mutable, in the SoC recurrence)
#   row RHS          <- gcap[t]       (mutable, grid coupling)
#   variable bound   <- pmax[a,t]     (mutable, power upper bound)
# The constraint matrix (efficiency, +/-1 couplings) is static.
# --------------------------------------------------------------------------- #
def _mpc_model(A=3, T=5, mip=False):
    m = pyo.ConcreteModel()
    m.A = pyo.RangeSet(0, A - 1)
    m.T = pyo.RangeSet(0, T - 1)
    m.price = pyo.Param(m.T, initialize={t: 1.0 for t in range(T)}, mutable=True)
    m.dem = pyo.Param(
        m.A,
        m.T,
        initialize={(a, t): 0.5 for a in range(A) for t in range(T)},
        mutable=True,
    )
    m.gcap = pyo.Param(m.T, initialize={t: 3.0 * A for t in range(T)}, mutable=True)
    m.pmax = pyo.Param(
        m.A,
        m.T,
        initialize={(a, t): 5.0 for a in range(A) for t in range(T)},
        mutable=True,
    )
    eff = 0.95
    dom = pyo.NonNegativeIntegers if mip else pyo.NonNegativeReals
    m.p = pyo.Var(m.A, m.T, domain=dom)
    m.soc = pyo.Var(m.A, m.T, bounds=(0.0, 40.0))

    def socrule(mm, a, t):
        if t == 0:
            return mm.soc[a, t] == eff * mm.p[a, t] - mm.dem[a, t]
        return mm.soc[a, t] == mm.soc[a, t - 1] + eff * mm.p[a, t] - mm.dem[a, t]

    m.socc = pyo.Constraint(m.A, m.T, rule=socrule)
    m.grid = pyo.Constraint(
        m.T, rule=lambda mm, t: sum(mm.p[a, t] for a in mm.A) <= mm.gcap[t]
    )
    for a in range(A):
        for t in range(T):
            m.p[a, t].setub(m.pmax[a, t])
    m.obj = pyo.Objective(
        expr=sum(m.price[t] * m.p[a, t] for a in range(A) for t in range(T))
        + 0.01 * sum(m.soc[a, t] for a in range(A) for t in range(T)),
        sense=pyo.minimize,
    )
    return m


def _apply_roll(m, A, T, rng):
    for t in range(T):
        m.price[t] = float(rng.uniform(0.5, 3.0))
        m.gcap[t] = 3.0 * A * float(rng.uniform(0.7, 1.0))
    for a in range(A):
        for t in range(T):
            m.dem[a, t] = float(rng.uniform(0.0, 1.0))
            m.pmax[a, t] = float(rng.uniform(3.0, 6.0))


# --------------------------------------------------------------------------- #
# A model whose *constraint matrix* carries a nominally-mutable coefficient:
# a state-of-charge recurrence with a mutable ``dur`` (interval duration) on the
# power term -- the private external case's shape.  Under an equal-interval roll
# ``dur`` never changes (only price/demand move), so the matrix is static in
# value even though the flag says mutable.
# --------------------------------------------------------------------------- #
def _matrix_param_model(dur=1.0, T=4):
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.dur = pyo.Param(initialize=dur, mutable=True)  # matrix coefficient
    m.price = pyo.Param(m.T, initialize={t: 1.0 for t in range(T)}, mutable=True)
    m.dem = pyo.Param(m.T, initialize={t: 0.5 for t in range(T)}, mutable=True)
    m.p = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, 5))
    m.soc = pyo.Var(m.T, bounds=(0.0, 20.0))

    def socrule(mm, t):
        if t == 0:
            return mm.soc[t] == mm.dur * mm.p[t] - mm.dem[t]
        return mm.soc[t] == mm.soc[t - 1] + mm.dur * mm.p[t] - mm.dem[t]

    m.socc = pyo.Constraint(m.T, rule=socrule)
    m.obj = pyo.Objective(
        expr=sum(m.price[t] * m.p[t] for t in range(T))
        + 0.01 * sum(m.soc[t] for t in range(T)),
        sense=pyo.minimize,
    )
    return m


def _roll_matrix_model(m, rng):
    """An equal-interval roll: prices/demands move, ``dur`` (matrix) stays put."""
    T = len(m.T)
    for t in range(T):
        m.price[t] = float(rng.uniform(0.5, 3.0))
        m.dem[t] = float(rng.uniform(0.0, 1.0))


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepBasics(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def test_import_and_construct(self):
        from pyomo.contrib.vector import FastStepHighs

        s = FastStepHighs()
        self.assertEqual(s.name, 'highs_faststep')
        self.assertTrue(s.available())

    def test_solve_before_set_instance_raises(self):
        from pyomo.contrib.vector import FastStepHighs

        with self.assertRaises(RuntimeError):
            FastStepHighs().solve()

    def test_first_solve_matches_fastload(self):
        from pyomo.contrib.vector import FastStepHighs

        m = _mpc_model(A=3, T=4)
        s = FastStepHighs()
        s.set_instance(m)
        res = s.solve()
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        mref = _mpc_model(A=3, T=4)
        rref = _fastload().solve(mref)
        self.assertAlmostEqual(
            res.incumbent_objective, rref.incumbent_objective, places=6
        )

    def test_objective_coefficient_update(self):
        # min price*x s.t. x >= 1, x <= 10.  Rolling the price rolls the optimum
        # objective (x* = 1 while price > 0).
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.price = pyo.Param(initialize=2.0, mutable=True)
        m.x = pyo.Var(bounds=(1, 10))
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=m.price * m.x, sense=pyo.minimize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.obj), 2.0, places=6)
        m.price = 5.0
        s.solve()
        self.assertAlmostEqual(pyo.value(m.obj), 5.0, places=6)

    def test_domain_constrained_mutable_param(self):
        # A mutable Param with a restricted domain (NonNegativeReals): the
        # set_instance self-check perturbs the parameter vector and must not trip
        # the Param's domain validation on a transient probe value.
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.price = pyo.Param(
            [0, 1], domain=pyo.NonNegativeReals, initialize=1.0, mutable=True
        )
        m.cap = pyo.Param(domain=pyo.NonNegativeReals, initialize=5.0, mutable=True)
        m.x = pyo.Var([0, 1], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x[0] + m.x[1] <= m.cap)
        m.d = pyo.Constraint(expr=m.x[0] + m.x[1] >= 1)
        m.obj = pyo.Objective(expr=sum(m.price[i] * m.x[i] for i in (0, 1)))
        s = FastStepHighs()
        s.set_instance(m)  # must not raise a Param domain ValueError
        res = s.solve()
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        # The probe restores the Params exactly.
        self.assertEqual(pyo.value(m.price[0]), 1.0)
        self.assertEqual(pyo.value(m.cap), 5.0)

    def test_rhs_and_bound_update(self):
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.b = pyo.Param(initialize=3.0, mutable=True)
        m.ub = pyo.Param(initialize=10.0, mutable=True)
        m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, None))
        m.x.setub(m.ub)
        m.c = pyo.Constraint(expr=m.x >= m.b)
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 3.0, places=6)
        m.b = 7.0
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 7.0, places=6)
        # Tighten the upper bound below the RHS -> infeasible.
        m.ub = 5.0
        res = s.solve(load_solutions=False, raise_on_nonoptimal=False)
        self.assertEqual(res.termination_condition, self.TC.provenInfeasible)


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepEquivalence(unittest.TestCase):
    """The warm-solve equivalence gate (the correctness surface)."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition
        self.fastload = _fastload()

    def _fresh_obj_tc(self, A, T, mip, rng_state):
        mref = _mpc_model(A=A, T=T, mip=mip)
        _apply_roll(mref, A, T, np.random.default_rng(rng_state))
        r = self.fastload.solve(mref, raise_exception_on_nonoptimal_result=False)
        return r, mref

    def test_rolling_sequence_matches_fresh_lp(self):
        from pyomo.contrib.vector import FastStepHighs

        A, T = 4, 6
        for keep_basis in (True, False):
            m = _mpc_model(A=A, T=T)
            s = FastStepHighs()
            s.set_instance(m)
            s.solve(keep_basis=keep_basis)
            for roll in range(8):
                _apply_roll(m, A, T, np.random.default_rng(roll))
                res = s.solve(keep_basis=keep_basis)
                rref, mref = self._fresh_obj_tc(A, T, False, roll)
                self.assertEqual(
                    res.termination_condition,
                    rref.termination_condition,
                    msg=f"tc mismatch roll {roll} keep_basis={keep_basis}",
                )
                if res.termination_condition == self.TC.convergenceCriteriaSatisfied:
                    self.assertAlmostEqual(
                        res.incumbent_objective,
                        rref.incumbent_objective,
                        delta=1e-5 * max(1.0, abs(rref.incumbent_objective)),
                        msg=f"obj mismatch roll {roll} keep_basis={keep_basis}",
                    )

    def test_rolling_sequence_matches_fresh_mip(self):
        from pyomo.contrib.vector import FastStepHighs

        A, T = 3, 5
        m = _mpc_model(A=A, T=T, mip=True)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        for roll in range(6):
            _apply_roll(m, A, T, np.random.default_rng(100 + roll))
            res = s.solve()
            rref, _ = self._fresh_obj_tc(A, T, True, 100 + roll)
            self.assertEqual(res.termination_condition, rref.termination_condition)
            if res.termination_condition == self.TC.convergenceCriteriaSatisfied:
                # MIP: objective matches; variable values may differ by alt optima.
                self.assertAlmostEqual(
                    res.incumbent_objective,
                    rref.incumbent_objective,
                    delta=1e-5 * max(1.0, abs(rref.incumbent_objective)),
                )

    def test_primal_values_match_on_unique_lp(self):
        from pyomo.contrib.vector import FastStepHighs

        # A unique-optimum LP: primal values (not just objective) must match.
        m = pyo.ConcreteModel()
        m.p = pyo.Param([0, 1, 2], initialize={0: 1.0, 1: 2.0, 2: 3.0}, mutable=True)
        m.x = pyo.Var([0, 1, 2], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c1 = pyo.Constraint(expr=m.x[0] + m.x[1] + m.x[2] == 6)
        m.c2 = pyo.Constraint(expr=m.x[0] - m.x[1] == 1)
        m.c3 = pyo.Constraint(expr=m.x[1] - m.x[2] == 1)
        m.obj = pyo.Objective(expr=sum(m.p[i] * m.x[i] for i in range(3)))
        s = FastStepHighs()
        s.set_instance(m)
        rng = np.random.default_rng(7)
        for roll in range(5):
            vals = {i: float(rng.uniform(0.5, 4.0)) for i in range(3)}
            for i in range(3):
                m.p[i] = vals[i]
            s.solve()
            mref = m.clone()
            self.fastload.solve(mref)
            for i in range(3):
                self.assertAlmostEqual(
                    pyo.value(m.x[i]), pyo.value(mref.x[i]), places=6
                )

    def test_param_array_path_matches_model_path(self):
        from pyomo.contrib.vector import FastStepHighs

        A, T = 3, 5
        m = _mpc_model(A=A, T=T)
        s = FastStepHighs()
        s.set_instance(m)
        params = s.parameters
        s.solve()
        for roll in range(5):
            _apply_roll(m, A, T, np.random.default_rng(200 + roll))
            P = np.fromiter((p.value for p in params), float, len(params))
            res_arr = s.solve(param_values=P)
            # Same data via the model-driven path on a twin.
            rref, _ = self._fresh_obj_tc(A, T, False, 200 + roll)
            self.assertEqual(res_arr.termination_condition, rref.termination_condition)
            if res_arr.termination_condition == self.TC.convergenceCriteriaSatisfied:
                self.assertAlmostEqual(
                    res_arr.incumbent_objective,
                    rref.incumbent_objective,
                    delta=1e-5 * max(1.0, abs(rref.incumbent_objective)),
                )

    def test_dirty_mask_matches_full_update(self):
        from pyomo.contrib.vector import FastStepHighs

        # Change only a subset of parameters each roll and mark them dirty; the
        # result must equal a fresh build with the same (partially-changed) data.
        m = pyo.ConcreteModel()
        T = 6
        m.T = pyo.RangeSet(0, T - 1)
        m.price = pyo.Param(m.T, initialize=1.0, mutable=True)
        m.cap = pyo.Param(m.T, initialize=5.0, mutable=True)
        m.x = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(m.T, rule=lambda mm, t: mm.x[t] <= mm.cap[t])
        m.tot = pyo.Constraint(expr=sum(m.x[t] for t in m.T) >= 3)
        m.obj = pyo.Objective(expr=sum(m.price[t] * m.x[t] for t in m.T))
        s = FastStepHighs()
        s.set_instance(m)
        params = s.parameters
        pidx = {id(p): i for i, p in enumerate(params)}
        s.solve()
        rng = np.random.default_rng(0)
        for roll in range(5):
            # change all prices, only two caps
            for t in range(T):
                m.price[t] = float(rng.uniform(0.5, 3.0))
            for t in (1, 4):
                m.cap[t] = float(rng.uniform(2.0, 8.0))
            P = np.fromiter((p.value for p in params), float, len(params))
            dirty = np.zeros(len(params), dtype=bool)
            for t in range(T):
                dirty[pidx[id(m.price[t])]] = True
            for t in (1, 4):
                dirty[pidx[id(m.cap[t])]] = True
            res = s.solve(param_values=P, dirty=dirty)
            mref = m.clone()
            rref = _fastload().solve(mref)
            self.assertAlmostEqual(
                res.incumbent_objective,
                rref.incumbent_objective,
                delta=1e-6 * max(1.0, abs(rref.incumbent_objective)),
            )


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepExplicitAPI(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401

    def test_index_addressed_setters(self):
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x[0] + m.x[1] >= 2)
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1], sense=pyo.minimize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        j0 = s.column_index(m.x[0])
        j1 = s.column_index(m.x[1])
        self.assertEqual({j0, j1}, {0, 1})
        rows = s.row_indices(m.c)
        self.assertEqual(len(rows), 1)
        # Make x[0] cheaper via a direct coefficient push; the optimum shifts.
        # update=False so solve() does not re-extract the coefficients from the
        # (unchanged) model and clobber the explicit push.
        s.set_objective_coefficients(np.array([0.0, 1.0]), cols=np.array([j0, j1]))
        res = s.solve(update=False)
        # x[0] free (0 cost) up to 10 -> covers the >=2 requirement at 0 cost.
        # Check the solver objective (the model's own obj expr is unchanged).
        self.assertAlmostEqual(res.incumbent_objective, 0.0, places=6)

    def test_read_param_vector(self):
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.p = pyo.Param([0, 1], initialize={0: 2.0, 1: 3.0}, mutable=True)
        m.x = pyo.Var([0, 1], bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x[0] + m.x[1] >= 1)
        m.obj = pyo.Objective(expr=m.p[0] * m.x[0] + m.p[1] * m.x[1])
        s = FastStepHighs()
        s.set_instance(m)
        P = s.read_param_vector()
        self.assertEqual(len(P), 2)
        self.assertEqual(set(P.tolist()), {2.0, 3.0})


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepGuards(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        self.Err = IncompatibleModelError

    def test_guard_nonlinear(self):
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x[0] * m.x[1] <= 5)
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1])
        with self.assertRaises(self.Err):
            FastStepHighs().set_instance(m)

    def test_mutable_matrix_coefficient_accepted(self):
        # A *nominally* mutable matrix coefficient is no longer rejected at
        # set_instance: the value guard accepts it (verify-the-values, not
        # trust-the-flag) and warm-solves while the coefficient stays put.
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=2.0, mutable=True)
        m.x = pyo.Var([0, 1], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.a * m.x[0] + m.x[1] <= 5)
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1], sense=pyo.maximize)
        s = FastStepHighs()
        s.set_instance(m)  # must NOT raise
        res = s.solve()
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )

    def test_guard_structure_change(self):
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x[0] + m.x[1] <= 5)
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1], sense=pyo.maximize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        m.c2 = pyo.Constraint(expr=m.x[0] <= 3)  # structure change
        with self.assertRaises(self.Err):
            s.solve()

    def test_param_array_rejected_with_residual(self):
        # A fixed variable in a constraint body makes a residual entry; the
        # array-driven path must refuse (it cannot evaluate residuals from P).
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.b = pyo.Param(initialize=1.0, mutable=True)
        m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.f = pyo.Var(bounds=(0, 10))
        m.f.fix(2.0)
        m.c = pyo.Constraint(expr=m.x + m.f >= m.b)
        m.obj = pyo.Objective(expr=m.x)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        P = s.read_param_vector()
        with self.assertRaises(ValueError):
            s.solve(param_values=P)


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepResidual(unittest.TestCase):
    """A fixed variable whose value rolls is handled by the residual path."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401

    def test_fixed_variable_value_roll(self):
        from pyomo.contrib.vector import FastStepHighs

        # x + f >= 5 with f fixed; rolling f rolls the RHS via a residual entry.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.f = pyo.Var(bounds=(0, 10))
        m.f.fix(1.0)
        m.c = pyo.Constraint(expr=m.x + m.f >= 5)
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 4.0, places=6)  # 5 - 1
        m.f.fix(3.0)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 2.0, places=6)  # 5 - 3


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepMatrixGuard(unittest.TestCase):
    """The value-aware static-matrix guard (verify-the-values, not the flag)."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        self.TC = TerminationCondition
        self.Err = IncompatibleModelError

    def test_static_matrix_param_matches_fresh(self):
        # A mutable matrix coefficient that never changes: the rolling sequence
        # must match a per-roll fresh build, basis-kept and basis-reset.
        from pyomo.contrib.vector import FastStepHighs

        for keep_basis in (True, False):
            m = _matrix_param_model()
            s = FastStepHighs()
            s.set_instance(m)
            s.solve(keep_basis=keep_basis)
            for roll in range(6):
                _roll_matrix_model(m, np.random.default_rng(roll))
                res = s.solve(keep_basis=keep_basis)
                mref = _matrix_param_model()
                _roll_matrix_model(mref, np.random.default_rng(roll))
                rref = _fastload().solve(
                    mref, raise_exception_on_nonoptimal_result=False
                )
                self.assertEqual(
                    res.termination_condition,
                    rref.termination_condition,
                    msg=f"tc mismatch roll {roll} keep_basis={keep_basis}",
                )
                if res.termination_condition == self.TC.convergenceCriteriaSatisfied:
                    self.assertAlmostEqual(
                        res.incumbent_objective,
                        rref.incumbent_objective,
                        delta=1e-6 * max(1.0, abs(rref.incumbent_objective)),
                        msg=f"obj mismatch roll {roll} keep_basis={keep_basis}",
                    )

    def test_repeated_identical_roll_never_trips(self):
        # Exact comparison must not false-positive when the matrix param is
        # untouched across many solves (M @ P is self-consistent with itself).
        from pyomo.contrib.vector import FastStepHighs

        m = _matrix_param_model()
        s = FastStepHighs()  # default exact comparison
        s.set_instance(m)
        for _ in range(10):
            res = s.solve()
            self.assertEqual(
                res.termination_condition, self.TC.convergenceCriteriaSatisfied
            )

    def test_matrix_change_fails_loud_by_default(self):
        from pyomo.contrib.vector import FastStepHighs

        m = _matrix_param_model()
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        _roll_matrix_model(m, np.random.default_rng(0))
        s.solve()  # a static roll still fine
        m.dur = 2.0  # genuine matrix change
        with self.assertRaises(self.Err) as ctx:
            s.solve()
        msg = str(ctx.exception)
        self.assertIn("matrix", msg.lower())
        self.assertIn("dur", msg)  # names the offending Param
        self.assertIn("socc", msg)  # names the offending constraint

    def test_matrix_change_reload_rebuilds_and_continues(self):
        from pyomo.contrib.vector import FastStepHighs

        m = _matrix_param_model()
        s = FastStepHighs(on_matrix_change="reload")
        s.set_instance(m)
        s.solve()
        m.dur = 2.0  # genuine matrix change -> reload (basis reset), continue
        res = s.solve()
        mref = _matrix_param_model(dur=2.0)
        rref = _fastload().solve(mref)
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        self.assertAlmostEqual(
            res.incumbent_objective,
            rref.incumbent_objective,
            delta=1e-6 * max(1.0, abs(rref.incumbent_objective)),
        )
        # A subsequent static roll still matches a fresh build after the reload.
        _roll_matrix_model(m, np.random.default_rng(3))
        res2 = s.solve()
        mref2 = _matrix_param_model(dur=2.0)
        _roll_matrix_model(mref2, np.random.default_rng(3))
        rref2 = _fastload().solve(mref2)
        self.assertAlmostEqual(
            res2.incumbent_objective,
            rref2.incumbent_objective,
            delta=1e-6 * max(1.0, abs(rref2.incumbent_objective)),
        )

    def test_matrix_tolerance_accepts_tiny_drift(self):
        from pyomo.contrib.vector import FastStepHighs

        m = _matrix_param_model()
        s = FastStepHighs(matrix_atol=1e-6)
        s.set_instance(m)
        s.solve()
        m.dur = 1.0 + 1e-9  # within tolerance -> treated as unchanged
        res = s.solve()  # must not raise
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        m.dur = 1.2  # beyond tolerance -> trips
        with self.assertRaises(self.Err):
            s.solve()

    def test_range_constraint_matrix_coefficient(self):
        # A two-sided (range) row splits into two A-rows sharing the body; a
        # matrix-coefficient change must be caught on the shared coefficient.
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=2.0, mutable=True)
        m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.y = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(expr=(1.0, m.a * m.x + m.y, 8.0))  # range row
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        m.a = 4.0  # matrix change on the range row
        with self.assertRaises(self.Err):
            s.solve()

    def test_residual_matrix_coefficient_change_detected(self):
        # A coefficient that references a *fixed variable* is not affine in the
        # parameters -> a residual matrix entry evaluated with value() each
        # solve; the guard still detects a genuine change (here via the fixed
        # variable's value moving).
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.dur = pyo.Param(initialize=1.0, mutable=True)
        m.zfix = pyo.Var(bounds=(0, 10))
        m.zfix.fix(1.0)
        m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(expr=(m.dur + m.zfix) * m.x <= 8)  # coef (residual)
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 4.0, places=6)  # 8 / (1 + 1)
        m.zfix.fix(3.0)  # coefficient (dur + zfix) now 4 -> genuine change
        with self.assertRaises(self.Err):
            s.solve()

    def test_reload_handles_coefficient_to_zero_shape_shift(self):
        # A matrix coefficient rolling to exactly 0.0 drops its A-nonzero (a
        # matrix *shape* shift); the reload path must rebuild the whole instance
        # so the mapping stays consistent, and recover when it returns nonzero.
        from pyomo.contrib.vector import FastStepHighs

        def build(dur):
            m = pyo.ConcreteModel()
            m.dur = pyo.Param(initialize=dur, mutable=True)
            m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
            m.y = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
            m.c = pyo.Constraint(expr=m.dur * m.x + m.y == 5)
            m.d = pyo.Constraint(expr=m.x + m.y >= 1)
            m.obj = pyo.Objective(expr=m.x + 2 * m.y, sense=pyo.minimize)
            return m

        m = build(1.0)
        s = FastStepHighs(on_matrix_change="reload")
        s.set_instance(m)
        s.solve()
        for dur in (0.0, 2.0):  # nonzero -> zero (drop) -> nonzero (add)
            m.dur = dur
            res = s.solve()
            rref = _fastload().solve(build(dur))
            self.assertAlmostEqual(
                res.incumbent_objective,
                rref.incumbent_objective,
                delta=1e-6 * max(1.0, abs(rref.incumbent_objective)),
            )

    def test_bilinear_param_coefficient_rejected(self):
        # A coefficient bilinear in the parameters (p*q) is genuinely nonlinear
        # in P; the affine self-check cannot reproduce it under perturbation, so
        # the model is rejected loudly (never templated wrong).
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.p = pyo.Param(initialize=2.0, mutable=True)
        m.q = pyo.Param(initialize=1.0, mutable=True)
        m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.p * m.q * m.x <= 8)
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)
        with self.assertRaises(self.Err):
            FastStepHighs().set_instance(m)

    def test_array_mode_matrix_change_errors_even_with_reload(self):
        # The array-driven path cannot reload from the model, so a matrix change
        # is always fatal there -- even under on_matrix_change='reload'.
        from pyomo.contrib.vector import FastStepHighs

        m = _matrix_param_model()
        s = FastStepHighs(on_matrix_change="reload")
        s.set_instance(m)
        params = s.parameters
        s.solve()
        m.dur = 2.0  # matrix param changed in the model
        P = np.fromiter((p.value for p in params), float, len(params))
        with self.assertRaises(self.Err):
            s.solve(param_values=P)

    def test_invalid_on_matrix_change(self):
        from pyomo.contrib.vector import FastStepHighs

        with self.assertRaises(ValueError):
            FastStepHighs(on_matrix_change="bogus")
        m = _matrix_param_model()
        with self.assertRaises(ValueError):
            FastStepHighs().set_instance(m, on_matrix_change="bogus")


if __name__ == "__main__":
    unittest.main()
