# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for ``FastStepHighs`` warm re-solve over a *switch-ON* model.

A model built under :func:`vectorized_construction` (the Phase-3 construction
switch) carries *template-vectorized* constraint families and *columnar*
Var/Param stores rather than per-index data objects.  Before this integration,
``FastStepHighs.set_instance`` walked ``compiled.rows`` expecting a per-row
``ConstraintData.body`` and raised ``AttributeError: 'IndexedConstraint' object
has no attribute 'body'`` -- so a switch-ON model could not warm re-solve.

These tests check the closed gap:

* the exact failure no longer occurs -- a switch-ON model compiles and solves;
* the full warm loop over a switch-ON model matches the *switch-OFF* faststep
  run **solve-for-solve** (objective and, on unique-optimum LPs, primal values),
  for both basis-kept and basis-reset runs;
* every warm mechanism survives switch-ON: a templatized mutable RHS (equality
  and inequality), a mutable objective coefficient, verified-static parameter
  folding, the value-aware matrix guard (accept-and-verify, and fail-loud on a
  genuine change), and a classic-fallback Var carrying a mutable-Param bound;
* the solution scatters back onto *columnar* Vars after a warm solve.

The equivalence twin is built with the identical model source and the switch
off, so both compiles produce the same standard form; the reference solver is
therefore ``FastStepHighs`` itself (switch-off), per the integration's exit
criterion.
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


# --------------------------------------------------------------------------- #
# A synthetic rolling-horizon model that, under the switch, exercises every warm
# mechanism through the templatized representations:
#   * ``bal``  -- templatized equality, mutable RHS via ``dem[a,t]``
#   * ``grid`` -- templatized inequality, mutable RHS via ``gcap[t]``
#   * ``cap``  -- templatized *fully static* family (immutable ``pcap``): its
#                 rows never roll, so the warm plan must skip it
#   * ``p``    -- a Var whose upper bound is the mutable ``pmax[a,t]`` (a bounds=
#                 rule makes it fall back to a classic Var that can hold the
#                 mutable bound); ``soc`` stays columnar (static bounds)
#   * objective -- mutable coefficient via ``price[t]``
# Built with the identical source switch-on and switch-off, so the two compiles
# share one standard form.
# --------------------------------------------------------------------------- #
def _roll_model(switch, A=4, T=6, mip=False, fold=False):
    from pyomo.contrib.vector import vectorized_construction

    ctx = vectorized_construction() if switch else None
    if ctx is not None:
        ctx.__enter__()
    try:
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
        # A verified-static (structurally constant) duration folded into the
        # objective coefficient ``price[t] * dur``.
        m.dur = pyo.Param(initialize=2.0, mutable=True)
        # An immutable per-index capacity -> a fully static templatized family.
        m.pcap = pyo.Param(
            m.A,
            m.T,
            initialize={(a, t): 8.0 for a in range(A) for t in range(T)},
            mutable=False,
        )
        eff = 0.95
        dom = pyo.NonNegativeIntegers if mip else pyo.NonNegativeReals
        # A bounds= rule referencing a mutable Param keeps ``p`` a classic Var
        # (columnar Vars cannot hold a mutable bound), so ``pmax`` rolls the
        # variable's upper bound.
        m.p = pyo.Var(
            m.A, m.T, domain=dom, bounds=lambda mm, a, t: (0.0, mm.pmax[a, t])
        )
        m.soc = pyo.Var(m.A, m.T, bounds=(0.0, 40.0))

        # Templatized equality family (no index conditional) with mutable RHS.
        m.bal = pyo.Constraint(
            m.A,
            m.T,
            rule=lambda mm, a, t: mm.soc[a, t] == eff * mm.p[a, t] - mm.dem[a, t],
        )
        # Templatized inequality family with mutable RHS.
        m.grid = pyo.Constraint(
            m.T, rule=lambda mm, t: sum(mm.p[a, t] for a in mm.A) <= mm.gcap[t]
        )
        # Fully static templatized family (immutable capacity).
        m.cap = pyo.Constraint(
            m.A, m.T, rule=lambda mm, a, t: mm.p[a, t] <= mm.pcap[a, t]
        )

        def _coef(t):
            return (m.price[t] * m.dur) if fold else m.price[t]

        m.obj = pyo.Objective(
            expr=sum(_coef(t) * m.p[a, t] for a in range(A) for t in range(T))
            + 0.01 * sum(m.soc[a, t] for a in range(A) for t in range(T)),
            sense=pyo.minimize,
        )
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return m


def _apply_roll(m, A, T, rng):
    for t in range(T):
        m.price[t] = float(rng.uniform(0.5, 3.0))
        m.gcap[t] = 3.0 * A * float(rng.uniform(0.7, 1.0))
    for a in range(A):
        for t in range(T):
            m.dem[a, t] = float(rng.uniform(0.0, 1.0))
            m.pmax[a, t] = float(rng.uniform(3.0, 6.0))


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepTemplatizedGapClosed(unittest.TestCase):
    """The switch-ON model compiles and solves (the reproduced failure is gone)."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def test_switch_on_model_is_templatized(self):
        from pyomo.contrib.vector.template_vectorize import model_has_templates

        m = _roll_model(switch=True)
        # Guard the premise: we are actually exercising the templatized path.
        self.assertTrue(model_has_templates(m))

    def test_set_instance_and_solve_switch_on(self):
        from pyomo.contrib.vector import FastStepHighs

        m = _roll_model(switch=True)
        s = FastStepHighs()
        # Previously raised AttributeError('IndexedConstraint' ... 'body').
        s.set_instance(m)
        res = s.solve()
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )

    def test_columnar_var_solution_loads_back(self):
        from pyomo.contrib.vector import FastStepHighs
        from pyomo.contrib.vector.varparam import is_columnar_var

        m = _roll_model(switch=True)
        # ``soc`` is a columnar Var (its solution must scatter back after solve).
        self.assertTrue(is_columnar_var(m.soc))
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        # No "uninitialized VectorVarData" -- every columnar value is populated.
        vals = [pyo.value(m.soc[a, t]) for a in range(4) for t in range(6)]
        self.assertEqual(len(vals), 24)
        for v in vals:
            self.assertIsNotNone(v)


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepTemplatizedEquivalence(unittest.TestCase):
    """Switch-ON warm loop matches the switch-OFF faststep run, solve-for-solve."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def _assert_loop_matches(self, A, T, rolls, mip=False, fold=False, seed=0):
        from pyomo.contrib.vector import FastStepHighs

        for keep_basis in (True, False):
            mon = _roll_model(switch=True, A=A, T=T, mip=mip, fold=fold)
            moff = _roll_model(switch=False, A=A, T=T, mip=mip, fold=fold)
            son, soff = FastStepHighs(), FastStepHighs()
            son.set_instance(mon)
            soff.set_instance(moff)
            son.solve(keep_basis=keep_basis)
            soff.solve(keep_basis=keep_basis)
            for k in range(rolls):
                _apply_roll(mon, A, T, np.random.default_rng(seed + k))
                _apply_roll(moff, A, T, np.random.default_rng(seed + k))
                ron = son.solve(keep_basis=keep_basis)
                roff = soff.solve(keep_basis=keep_basis)
                self.assertEqual(
                    ron.termination_condition,
                    roff.termination_condition,
                    msg=f"tc mismatch roll {k} keep_basis={keep_basis}",
                )
                if ron.termination_condition == self.TC.convergenceCriteriaSatisfied:
                    self.assertAlmostEqual(
                        ron.incumbent_objective,
                        roff.incumbent_objective,
                        delta=1e-6 * max(1.0, abs(roff.incumbent_objective)),
                        msg=f"obj mismatch roll {k} keep_basis={keep_basis}",
                    )

    def test_rolling_sequence_matches_switch_off_lp(self):
        self._assert_loop_matches(A=4, T=6, rolls=10, seed=0)

    def test_rolling_sequence_matches_switch_off_mip(self):
        self._assert_loop_matches(A=3, T=5, rolls=6, mip=True, seed=100)

    def test_folding_rolling_matches_switch_off(self):
        # ``price[t] * dur`` -- ``dur`` folds as a watched constant; the varying
        # ``price`` templates over it, on both the switch-on and switch-off build.
        self._assert_loop_matches(A=3, T=5, rolls=8, fold=True, seed=200)

    def test_param_array_path_matches_switch_off(self):
        # The mapping-free array update path on a switch-ON model matches the
        # switch-OFF faststep run reading the same parameter vector.
        from pyomo.contrib.vector import FastStepHighs

        A, T = 3, 5
        mon = _roll_model(switch=True, A=A, T=T)
        moff = _roll_model(switch=False, A=A, T=T)
        son, soff = FastStepHighs(), FastStepHighs()
        son.set_instance(mon)
        soff.set_instance(moff)
        son.solve()
        soff.solve()
        for k in range(5):
            _apply_roll(mon, A, T, np.random.default_rng(300 + k))
            P = np.fromiter(
                (p.value for p in son.parameters), float, len(son.parameters)
            )
            ron = son.solve(param_values=P)
            _apply_roll(moff, A, T, np.random.default_rng(300 + k))
            roff = soff.solve()
            self.assertEqual(ron.termination_condition, roff.termination_condition)
            if ron.termination_condition == self.TC.convergenceCriteriaSatisfied:
                self.assertAlmostEqual(
                    ron.incumbent_objective,
                    roff.incumbent_objective,
                    delta=1e-6 * max(1.0, abs(roff.incumbent_objective)),
                )

    def test_unique_lp_primal_values_match_switch_off(self):
        from pyomo.contrib.vector import FastStepHighs, vectorized_construction

        def build(switch):
            ctx = vectorized_construction() if switch else None
            if ctx is not None:
                ctx.__enter__()
            try:
                m = pyo.ConcreteModel()
                m.I = pyo.RangeSet(0, 2)
                m.p = pyo.Param(m.I, initialize={0: 1.0, 1: 2.0, 2: 3.0}, mutable=True)
                m.rhs = pyo.Param(
                    m.I, initialize={0: 6.0, 1: 1.0, 2: 1.0}, mutable=True
                )
                m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=(0, 10))
                # A templatizable family with a mutable RHS.
                m.c = pyo.Constraint(m.I, rule=lambda mm, i: mm.x[i] <= mm.rhs[i] + 4.0)
                m.tot = pyo.Constraint(expr=m.x[0] + m.x[1] + m.x[2] == 6)
                m.obj = pyo.Objective(expr=sum(m.p[i] * m.x[i] for i in range(3)))
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
            return m

        mon, moff = build(True), build(False)
        son, soff = FastStepHighs(), FastStepHighs()
        son.set_instance(mon)
        soff.set_instance(moff)
        rng = np.random.default_rng(7)
        for _ in range(5):
            vals = {i: float(rng.uniform(0.5, 4.0)) for i in range(3)}
            for i in range(3):
                mon.p[i] = vals[i]
                moff.p[i] = vals[i]
            son.solve()
            soff.solve()
            for i in range(3):
                self.assertAlmostEqual(
                    pyo.value(mon.x[i]), pyo.value(moff.x[i]), places=6
                )


@unittest.skipUnless(_deps, "highs_faststep requires numpy/scipy/highspy")
class TestFastStepTemplatizedMatrixGuard(unittest.TestCase):
    """The value-aware matrix guard survives switch-ON (accept-and-verify + trip)."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def _guard_model(self, switch, T=4, dur=1.0):
        from pyomo.contrib.vector import vectorized_construction

        ctx = vectorized_construction() if switch else None
        if ctx is not None:
            ctx.__enter__()
        try:
            m = pyo.ConcreteModel()
            m.T = pyo.RangeSet(0, T - 1)
            m.dur = pyo.Param(initialize=dur, mutable=True)  # static-in-value coef
            m.price = pyo.Param(
                m.T, initialize={t: 1.0 for t in range(T)}, mutable=True
            )
            m.dem = pyo.Param(m.T, initialize={t: 0.5 for t in range(T)}, mutable=True)
            m.gcap = pyo.Param(m.T, initialize={t: 9.0 for t in range(T)}, mutable=True)
            m.p = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, 5))
            m.soc = pyo.Var(m.T, bounds=(0.0, 20.0))

            # Index-conditional recurrence -> a classic-fallback family carrying a
            # nominally-mutable (but static-in-value) matrix coefficient ``dur``.
            def socrule(mm, t):
                if t == 0:
                    return mm.soc[t] == mm.dur * mm.p[t] - mm.dem[t]
                return mm.soc[t] == mm.soc[t - 1] + mm.dur * mm.p[t] - mm.dem[t]

            m.socc = pyo.Constraint(m.T, rule=socrule)
            # Templatized inequality family with mutable RHS.
            m.grid = pyo.Constraint(m.T, rule=lambda mm, t: mm.p[t] <= mm.gcap[t])
            m.obj = pyo.Objective(
                expr=sum(m.price[t] * m.p[t] for t in range(T))
                + 0.01 * sum(m.soc[t] for t in range(T)),
                sense=pyo.minimize,
            )
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        return m

    def test_static_matrix_coef_accepted_and_matches_switch_off(self):
        from pyomo.contrib.vector import FastStepHighs

        mon = self._guard_model(switch=True)
        moff = self._guard_model(switch=False)
        son, soff = FastStepHighs(), FastStepHighs()
        son.set_instance(mon)  # matrix guard armed on the mutable-but-static ``dur``
        soff.set_instance(moff)
        rng = np.random.default_rng(9)
        for _ in range(8):
            for t in range(4):
                pv = float(rng.uniform(0.5, 3.0))
                dv = float(rng.uniform(0.0, 1.0))
                gv = float(rng.uniform(6.0, 12.0))
                mon.price[t] = moff.price[t] = pv
                mon.dem[t] = moff.dem[t] = dv
                mon.gcap[t] = moff.gcap[t] = gv
            ron = son.solve()
            roff = soff.solve()
            self.assertAlmostEqual(
                ron.incumbent_objective, roff.incumbent_objective, places=6
            )

    def test_matrix_coef_change_trips_guard_switch_on(self):
        from pyomo.contrib.vector import FastStepHighs
        from pyomo.contrib.solver.common.util import IncompatibleModelError

        m = self._guard_model(switch=True)
        s = FastStepHighs()  # default policy: fail loud on a real matrix change
        s.set_instance(m)
        s.solve()
        m.dur = 1.75  # a genuine matrix-coefficient change
        with self.assertRaises(IncompatibleModelError):
            s.solve()


if __name__ == '__main__':
    unittest.main()
