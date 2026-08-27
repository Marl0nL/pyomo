# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the **masked warm updates** on ``FastStepHighs`` -- row masks and
variable fixes that let a rolling-horizon MPC narrow its *active window* between
solves WITHOUT a structural change.

The heart of these tests is a semantic-equivalence gate that proves the claim the
feature rests on: solving the **full** compiled matrix with the out-of-window rows
relaxed to free and the out-of-window variables fixed to their boundary values
yields, **on the active window**, exactly the solution of the true
structurally-narrowed problem -- objective (up to the fixed-variable constant),
termination status, and in-window variable values, within solver tolerance.  The
reference is a *fresh, independently built* narrowed model (only the in-window
variables/rows, the boundary variable a fixed ``Param``) solved through
``highs_fastload`` -- the fresh-per-cycle "structurally narrow and re-solve" route
that masking replaces.

The sharp case is a **boundary-coupling row**: an in-window recurrence row that
references an out-of-window variable.  With that variable fixed, the row becomes
the correct boundary condition (the fixed value moves to the row's RHS), so the
row must stay *active* while the variable is *fixed* -- which is exactly what the
narrowing does.  Covered here: randomized models x randomized windows (LP and
MIP), degenerate windows (empty, full, single row), roll+narrow together, the
array-driven path, compatibility with the value guard / fold set / structure
fingerprint, and the ``on_matrix_change='reload'`` overlay round-trip.
"""

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
from pyomo.common.dependencies import attempt_import

highspy, highspy_available = attempt_import('highspy')

_deps = numpy_available and scipy_available and highspy_available


def _fastload():
    from pyomo.contrib.solver.common.factory import SolverFactory

    return SolverFactory('highs_fastload')


# --------------------------------------------------------------------------- #
# A multi-asset energy MPC over a horizon of length T.
#
#   vars     p[a,t] >= 0  (ub pmax[a,t]),  soc[a,t] in [0, cap]
#   recur    socc[a,t]: soc[a,t] == prev + eff*p[a,t] - dem[a,t]
#              prev = soc0            (t == 0)          -- initial condition
#                   = soc[a,t-1]      (t  > 0)          -- boundary coupling
#   grid     grid[t]:   sum_a p[a,t] <= gcap[t]         -- couples assets at time t
#   obj      min  sum  price[t]*p[a,t] + hold*soc[a,t]
#
# ``socc[a,t]`` is assigned to time ``t`` and couples ``t`` with ``t-1``; ``grid[t]``
# is assigned to time ``t`` and couples only time ``t``.  So the in-window rows of
# window [a,b) are exactly {socc[*,t], grid[t] : a <= t < b}, and the recurrence at
# the window's left edge (t == a) references the out-of-window boundary soc[*,a-1].
# --------------------------------------------------------------------------- #
_EFF, _CAP, _HOLD, _SOC0 = 0.95, 20.0, 0.01, 5.0


def _rand_data(A, T, seed):
    rng = np.random.default_rng(seed)
    return dict(
        price={t: float(rng.uniform(0.5, 3.0)) for t in range(T)},
        dem={(a, t): float(rng.uniform(0.0, 1.0)) for a in range(A) for t in range(T)},
        pmax={(a, t): float(rng.uniform(3.0, 6.0)) for a in range(A) for t in range(T)},
        gcap={t: float(rng.uniform(0.6, 1.0)) * 3.0 * A for t in range(T)},
    )


def build_full(A, T, seed=0, mip=False, data=None):
    """The full-horizon model (mutable Params so a roll can move the data)."""
    if data is None:
        data = _rand_data(A, T, seed)
    m = pyo.ConcreteModel()
    m.A = pyo.RangeSet(0, A - 1)
    m.T = pyo.RangeSet(0, T - 1)
    m.price = pyo.Param(m.T, initialize=data['price'], mutable=True)
    m.dem = pyo.Param(m.A, m.T, initialize=data['dem'], mutable=True)
    m.pmax = pyo.Param(m.A, m.T, initialize=data['pmax'], mutable=True)
    m.gcap = pyo.Param(m.T, initialize=data['gcap'], mutable=True)
    dom = pyo.NonNegativeIntegers if mip else pyo.NonNegativeReals
    m.p = pyo.Var(m.A, m.T, domain=dom)
    m.soc = pyo.Var(m.A, m.T, bounds=(0.0, _CAP))
    for a in range(A):
        for t in range(T):
            m.p[a, t].setub(m.pmax[a, t])

    def soc_rule(mm, a, t):
        prev = _SOC0 if t == 0 else mm.soc[a, t - 1]
        return mm.soc[a, t] == prev + _EFF * mm.p[a, t] - mm.dem[a, t]

    m.socc = pyo.Constraint(m.A, m.T, rule=soc_rule)
    m.grid = pyo.Constraint(
        m.T, rule=lambda mm, t: sum(mm.p[a, t] for a in mm.A) <= mm.gcap[t]
    )
    m.obj = pyo.Objective(
        expr=sum(
            m.price[t] * m.p[a, t] + _HOLD * m.soc[a, t]
            for a in range(A)
            for t in range(T)
        ),
        sense=pyo.minimize,
    )
    m._data = data
    return m


def build_narrowed(A, T, a, b, socbar, data, mip=False):
    """The genuine structurally-narrowed window [a,b): only in-window vars/rows,
    the boundary soc[*,a-1] a fixed ``Param`` (``socbar[asset]``).  Values are read
    as constants from ``data`` (the fresh-per-cycle narrowed build)."""
    m = pyo.ConcreteModel()
    m.A = pyo.RangeSet(0, A - 1)
    m.W = pyo.RangeSet(a, b - 1)
    dom = pyo.NonNegativeIntegers if mip else pyo.NonNegativeReals
    m.p = pyo.Var(m.A, m.W, domain=dom)
    m.soc = pyo.Var(m.A, m.W, bounds=(0.0, _CAP))
    for asset in range(A):
        for t in range(a, b):
            m.p[asset, t].setub(data['pmax'][asset, t])

    def soc_rule(mm, asset, t):
        if t == 0:
            prev = _SOC0
        elif t == a:
            prev = socbar[asset]
        else:
            prev = mm.soc[asset, t - 1]
        return mm.soc[asset, t] == prev + _EFF * mm.p[asset, t] - data['dem'][asset, t]

    m.socc = pyo.Constraint(m.A, m.W, rule=soc_rule)
    m.grid = pyo.Constraint(
        m.W, rule=lambda mm, t: sum(mm.p[asset, t] for asset in mm.A) <= data['gcap'][t]
    )
    m.obj = pyo.Objective(
        expr=sum(
            data['price'][t] * m.p[asset, t] + _HOLD * m.soc[asset, t]
            for asset in range(A)
            for t in range(a, b)
        ),
        sense=pyo.minimize,
    )
    return m


def _window_indices(stepper, m, A, T, a, b):
    """The out-of-window solver rows to mask and columns to fix for window [a,b)."""
    in_win = set(range(a, b))
    mask_rows = []
    fix_cols = []
    for t in range(T):
        if t in in_win:
            continue
        mask_rows += stepper.row_indices(m.grid[t])
        for asset in range(A):
            mask_rows += stepper.row_indices(m.socc[asset, t])
            fix_cols.append(stepper.column_index(m.p[asset, t]))
            fix_cols.append(stepper.column_index(m.soc[asset, t]))
    return np.array(mask_rows, dtype=np.int64), np.array(fix_cols, dtype=np.int64)


@unittest.skipUnless(_deps, "highs_faststep masking requires numpy/scipy/highspy")
class TestMaskedEquivalence(unittest.TestCase):
    """The semantic-correctness gate: masked-warm narrowing == a fresh
    structurally-narrowed build, on the active window."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def _check_window(self, A, T, a, b, seed, mip=False, keep_basis=True):
        from pyomo.contrib.vector import FastStepHighs

        m = build_full(A, T, seed=seed, mip=mip)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve(keep_basis=keep_basis)  # full solve -> feasible boundary point

        # boundary/initial condition = the full solution's out-of-window soc.
        socbar = {
            asset: (pyo.value(m.soc[asset, a - 1]) if a >= 1 else _SOC0)
            for asset in range(A)
        }
        # fix values = the full solution's out-of-window variable values.
        mask_rows, fix_cols = _window_indices(s, m, A, T, a, b)
        out = set(range(a, b))
        fix_vals = np.array(
            [
                pyo.value(v)
                for t in range(T)
                if t not in out
                for asset in range(A)
                for v in (m.p[asset, t], m.soc[asset, t])
            ],
            dtype=np.float64,
        )

        if mask_rows.size:
            s.deactivate_rows(mask_rows)
        if fix_cols.size:
            s.fix_variables(fix_cols, fix_vals)
        res = s.solve(keep_basis=keep_basis, raise_on_nonoptimal=False)
        const = s.masked_objective_constant()

        ref_model = build_narrowed(A, T, a, b, socbar, m._data, mip=mip)
        ref = _fastload().solve(ref_model, raise_exception_on_nonoptimal_result=False)

        self.assertEqual(
            res.termination_condition,
            ref.termination_condition,
            msg=f"status mismatch A={A} T={T} win=[{a},{b}) seed={seed} mip={mip}",
        )
        if res.termination_condition != self.TC.convergenceCriteriaSatisfied:
            return
        window_obj = res.incumbent_objective - const
        scale = max(1.0, abs(ref.incumbent_objective))
        self.assertLess(
            abs(window_obj - ref.incumbent_objective) / scale,
            1e-6,
            msg=(
                f"window-objective mismatch A={A} T={T} win=[{a},{b}) seed={seed} "
                f"mip={mip}: {window_obj} vs {ref.incumbent_objective}"
            ),
        )
        if not mip:
            # LP: this structure has a unique optimum -> in-window values match.
            for asset in range(A):
                for t in range(a, b):
                    self.assertAlmostEqual(
                        pyo.value(m.p[asset, t]),
                        pyo.value(ref_model.p[asset, t]),
                        places=5,
                        msg=f"p[{asset},{t}] mismatch win=[{a},{b}) seed={seed}",
                    )
                    self.assertAlmostEqual(
                        pyo.value(m.soc[asset, t]),
                        pyo.value(ref_model.soc[asset, t]),
                        places=5,
                        msg=f"soc[{asset},{t}] mismatch win=[{a},{b}) seed={seed}",
                    )

    def test_randomized_windows_lp(self):
        rng = np.random.default_rng(7)
        for seed in range(12):
            A, T = 3, 10
            a = int(rng.integers(0, T - 2))
            b = int(rng.integers(a + 1, T + 1))
            for keep_basis in (True, False):
                self._check_window(A, T, a, b, seed, mip=False, keep_basis=keep_basis)

    def test_randomized_windows_mip(self):
        rng = np.random.default_rng(21)
        for seed in range(8):
            A, T = 2, 8
            a = int(rng.integers(0, T - 2))
            b = int(rng.integers(a + 1, T + 1))
            self._check_window(A, T, a, b, seed, mip=True)

    def test_boundary_coupling_left_edge(self):
        # A window that starts at t=a>=1: the recurrence at t=a references the
        # out-of-window (fixed) soc[*,a-1] -- the boundary-coupling row.
        for a, b in ((1, 5), (3, 7), (5, 9)):
            self._check_window(A=3, T=10, a=a, b=b, seed=99)

    def test_degenerate_full_window(self):
        # No rows masked, no vars fixed -> the full problem; must be untouched.
        self._check_window(A=3, T=8, a=0, b=8, seed=3)

    def test_degenerate_single_row_window(self):
        for a in (0, 4, 7):
            self._check_window(A=2, T=8, a=a, b=a + 1, seed=5)

    def test_degenerate_empty_window(self):
        # Every row masked and every variable fixed -> the narrowed problem is
        # empty; both report the same (all-constant) objective and status.
        from pyomo.contrib.vector import FastStepHighs

        A, T = 2, 6
        m = build_full(A, T, seed=8)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        fixvals = np.array(
            [pyo.value(v) for v in list(m.p.values()) + list(m.soc.values())],
            dtype=np.float64,
        )
        all_rows = np.arange(s._compiled.n_row, dtype=np.int64)
        all_cols = np.arange(s._compiled.n_col, dtype=np.int64)
        s.deactivate_rows(all_rows)
        s.fix_variables(all_cols, fixvals)
        res = s.solve(raise_on_nonoptimal=False)
        # The objective is entirely constant (every variable pinned).
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        self.assertAlmostEqual(
            res.incumbent_objective - s.masked_objective_constant(), 0.0, places=6
        )


@unittest.skipUnless(_deps, "highs_faststep masking requires numpy/scipy/highspy")
class TestMaskedRolling(unittest.TestCase):
    """A rolling-horizon MPC: each cycle rolls the data AND slides the active
    window, and each masked-warm resolve must match a fresh narrowed build."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def _roll(self, m, A, T, seed):
        d = _rand_data(A, T, seed)
        for t in range(T):
            m.price[t] = d['price'][t]
            m.gcap[t] = d['gcap'][t]
            for a in range(A):
                m.dem[a, t] = d['dem'][a, t]
                m.pmax[a, t] = d['pmax'][a, t]
        m._data = d

    def test_roll_and_narrow_sequence(self):
        from pyomo.contrib.vector import FastStepHighs

        A, T, W = 3, 12, 5
        m = build_full(A, T, seed=0)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        for cyc in range(8):
            self._roll(m, A, T, seed=1000 + cyc)
            a = cyc % (T - W)
            b = a + W
            # slide the window: activate all, then mask/fix the out-of-window set.
            s.clear_window()
            # a full "priming" solve at the new data gives boundary values.
            s.solve()
            socbar = {
                asset: (pyo.value(m.soc[asset, a - 1]) if a >= 1 else _SOC0)
                for asset in range(A)
            }
            mask_rows, fix_cols = _window_indices(s, m, A, T, a, b)
            fix_vals = np.array(
                [
                    pyo.value(v)
                    for t in range(T)
                    if t not in set(range(a, b))
                    for asset in range(A)
                    for v in (m.p[asset, t], m.soc[asset, t])
                ],
                dtype=np.float64,
            )
            s.deactivate_rows(mask_rows)
            s.fix_variables(fix_cols, fix_vals)
            res = s.solve(raise_on_nonoptimal=False)
            window_obj = res.incumbent_objective - s.masked_objective_constant()

            ref_model = build_narrowed(A, T, a, b, socbar, m._data)
            ref = _fastload().solve(
                ref_model, raise_exception_on_nonoptimal_result=False
            )
            self.assertEqual(res.termination_condition, ref.termination_condition)
            if res.termination_condition == self.TC.convergenceCriteriaSatisfied:
                scale = max(1.0, abs(ref.incumbent_objective))
                self.assertLess(
                    abs(window_obj - ref.incumbent_objective) / scale,
                    1e-6,
                    msg=f"cycle {cyc} window obj mismatch",
                )


@unittest.skipUnless(_deps, "highs_faststep masking requires numpy/scipy/highspy")
class TestMaskedApi(unittest.TestCase):
    """The overlay API surface: round-trips, introspection, validation, the
    array-driven path, and non-interference with the guards / fingerprint."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def _simple(self):
        # min x + y  s.t.  x >= 2 ; y >= 3 ; x + y <= 100.  Optimum (2, 3), obj 5.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 50))
        m.y = pyo.Var(bounds=(0, 50))
        m.cx = pyo.Constraint(expr=m.x >= 2)
        m.cy = pyo.Constraint(expr=m.y >= 3)
        m.cap = pyo.Constraint(expr=m.x + m.y <= 100)
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
        return m

    def test_deactivate_reactivate_roundtrip(self):
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.obj), 5.0, places=6)
        # Relax the y >= 3 row: y can now drop to its lower bound 0.
        (ry,) = s.row_indices(m.cy)
        s.deactivate_rows([ry])
        s.solve()
        self.assertAlmostEqual(pyo.value(m.y), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(m.obj), 2.0, places=6)
        # Re-activate it: back to the original optimum.
        s.activate_rows([ry])
        s.solve()
        self.assertAlmostEqual(pyo.value(m.y), 3.0, places=6)
        self.assertAlmostEqual(pyo.value(m.obj), 5.0, places=6)

    def test_fix_unfix_roundtrip_and_constant(self):
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        jx = s.column_index(m.x)
        s.fix_variables([jx], 7.5)
        res = s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 7.5, places=6)
        self.assertAlmostEqual(pyo.value(m.y), 3.0, places=6)
        # objective coefficient of x is 1, so the constant is 1 * 7.5.
        self.assertAlmostEqual(s.masked_objective_constant(), 7.5, places=6)
        # reported obj = 7.5 (x) + 3.0 (y); window obj = reported - constant = 3.0.
        self.assertAlmostEqual(res.incumbent_objective, 10.5, places=6)
        self.assertAlmostEqual(
            res.incumbent_objective - s.masked_objective_constant(), 3.0, places=6
        )
        s.unfix_variables([jx])
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 2.0, places=6)
        self.assertAlmostEqual(s.masked_objective_constant(), 0.0, places=6)

    def test_fix_value_changes_between_solves(self):
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        jx = s.column_index(m.x)
        for v in (5.0, 8.0, 12.0, 2.0):
            s.fix_variables([jx], v)
            s.solve()
            self.assertAlmostEqual(pyo.value(m.x), v, places=6)

    def test_set_window_and_clear_window(self):
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        n_row, n_col = s._compiled.n_row, s._compiled.n_col
        active = np.ones(n_row, dtype=bool)
        (ry,) = s.row_indices(m.cy)
        active[ry] = False
        fixed = np.zeros(n_col, dtype=bool)
        values = np.zeros(n_col, dtype=np.float64)
        s.set_window(active_rows=active, fixed_cols=fixed, fixed_values=values)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.y), 0.0, places=6)
        s.clear_window()
        s.solve()
        self.assertAlmostEqual(pyo.value(m.y), 3.0, places=6)

    def test_introspection_properties(self):
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        # before engaging: all active, none fixed.
        self.assertTrue(s.active_rows.all())
        self.assertFalse(s.fixed_variables_mask.any())
        (ry,) = s.row_indices(m.cy)
        s.deactivate_rows([ry])
        jx = s.column_index(m.x)
        s.fix_variables([jx], 4.0)
        self.assertFalse(s.active_rows[ry])
        self.assertTrue(s.fixed_variables_mask[jx])
        self.assertAlmostEqual(s.fixed_values[jx], 4.0, places=12)

    def test_index_validation(self):
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        with self.assertRaises(IndexError):
            s.deactivate_rows([s._compiled.n_row])  # out of range
        with self.assertRaises(IndexError):
            s.fix_variables([s._compiled.n_col], 0.0)
        with self.assertRaises(ValueError):
            s.set_active_rows(np.ones(s._compiled.n_row + 1, dtype=bool))
        with self.assertRaises(ValueError):
            s.set_window(fixed_cols=np.zeros(s._compiled.n_col, dtype=bool))  # no vals

    def test_update_false_pushes_overlay(self):
        # A pending mask/fix delta is pushed even when update=False (no model
        # re-extraction) -- the overlay is state on the stepper, not on the model.
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        (ry,) = s.row_indices(m.cy)
        s.deactivate_rows([ry])
        s.solve(update=False)
        self.assertAlmostEqual(pyo.value(m.y), 0.0, places=6)

    def test_reactivate_after_mutable_rhs_rolled(self):
        # A masked row whose RHS is a mutable Param that rolls *while masked*: on
        # reactivation the row must enforce the NEW rhs, not the stale one (the
        # ``_eff`` restore array tracks the template through masked solves).
        from pyomo.contrib.vector import FastStepHighs

        m = pyo.ConcreteModel()
        m.b = pyo.Param(initialize=3.0, mutable=True)
        m.x = pyo.Var(bounds=(0, 100))
        m.c = pyo.Constraint(expr=m.x >= m.b)  # mutable-RHS row
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 3.0, places=6)
        (rc,) = s.row_indices(m.c)
        s.deactivate_rows([rc])
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 0.0, places=6)  # row relaxed
        m.b = 7.0  # roll the RHS while the row is masked
        s.activate_rows([rc])
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x), 7.0, places=6)  # enforces new rhs

    def test_dirty_param_path_composes_with_masking(self):
        from pyomo.contrib.vector import FastStepHighs

        m = build_full(2, 6, seed=6)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        P = s.read_param_vector()
        (g5,) = s.row_indices(m.grid[5])
        s.deactivate_rows([g5])
        c05 = s.column_index(m.p[0, 5])
        s.fix_variables([c05], 1.0)
        dirty = np.ones(len(P), dtype=bool)
        res = s.solve(param_values=P, dirty=dirty, raise_on_nonoptimal=False)
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        self.assertAlmostEqual(pyo.value(m.p[0, 5]), 1.0, places=6)

    def test_masking_does_not_trip_structure_guard(self):
        # Masking/fixing is purely solver-side; the Pyomo model (and thus the
        # structure fingerprint) is untouched, so no IncompatibleModelError.
        from pyomo.contrib.vector import FastStepHighs

        m = build_full(2, 6, seed=2)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        rows, cols = _window_indices(s, m, 2, 6, 2, 5)
        s.deactivate_rows(rows)
        fixvals = np.array(
            [
                pyo.value(v)
                for t in range(6)
                if t not in {2, 3, 4}
                for asset in range(2)
                for v in (m.p[asset, t], m.soc[asset, t])
            ],
            dtype=np.float64,
        )
        s.fix_variables(cols, fixvals)
        res = s.solve()  # check_structure defaults True; must not raise
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )

    def test_array_driven_path_composes_with_masking(self):
        # solve(param_values=...) drives the templated data from arrays; the mask
        # overlay must still apply on top.
        from pyomo.contrib.vector import FastStepHighs

        m = build_full(2, 6, seed=4)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        P = s.read_param_vector()
        rows, cols = _window_indices(s, m, 2, 6, 1, 4)
        s.deactivate_rows(rows)
        fixvals = np.array(
            [
                pyo.value(v)
                for t in range(6)
                if t not in {1, 2, 3}
                for asset in range(2)
                for v in (m.p[asset, t], m.soc[asset, t])
            ],
            dtype=np.float64,
        )
        s.fix_variables(cols, fixvals)
        res = s.solve(param_values=P, raise_on_nonoptimal=False)
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )

    def test_infeasible_window_matches(self):
        # Fix variables to values that violate an *active* row: the masked-warm
        # solve reports infeasible (fixing via equal bounds overrides the box, so
        # infeasibility must come from a row, exactly as a bad boundary condition
        # would in a real narrowed window).
        from pyomo.contrib.vector import FastStepHighs

        m = self._simple()
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        # cap row is x + y <= 100 (active); pin x=60, y=60 -> 120 > 100 -> infeasible.
        jx, jy = s.column_index(m.x), s.column_index(m.y)
        s.fix_variables([jx, jy], [60.0, 60.0])
        res = s.solve(load_solutions=False, raise_on_nonoptimal=False)
        self.assertEqual(res.termination_condition, self.TC.provenInfeasible)


@unittest.skipUnless(_deps, "highs_faststep masking requires numpy/scipy/highspy")
class TestMaskedWithGuards(unittest.TestCase):
    """Masking composes with the value guard, the fold set, and the reload path."""

    def setUp(self):
        import pyomo.contrib.vector  # noqa: F401
        from pyomo.contrib.solver.common.results import TerminationCondition

        self.TC = TerminationCondition

    def _matrix_param_model(self, T=6, dur=1.0):
        m = pyo.ConcreteModel()
        m.T = pyo.RangeSet(0, T - 1)
        m.dur = pyo.Param(initialize=dur, mutable=True)  # matrix coefficient
        m.price = pyo.Param(m.T, initialize={t: 1.0 for t in range(T)}, mutable=True)
        m.dem = pyo.Param(m.T, initialize={t: 0.5 for t in range(T)}, mutable=True)
        m.p = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, 5))
        m.soc = pyo.Var(m.T, bounds=(0.0, 20.0))

        def socrule(mm, t):
            prev = _SOC0 if t == 0 else mm.soc[t - 1]
            return mm.soc[t] == prev + mm.dur * mm.p[t] - mm.dem[t]

        m.socc = pyo.Constraint(m.T, rule=socrule)
        m.obj = pyo.Objective(
            expr=sum(m.price[t] * m.p[t] + 0.01 * m.soc[t] for t in range(T)),
            sense=pyo.minimize,
        )
        return m

    def test_masking_with_static_matrix_guard(self):
        # The value guard watches the (nominally-mutable, static) ``dur`` matrix
        # coefficient; masking rows must not disturb that -- an equal-interval roll
        # + a narrowed window still warm-solves.
        from pyomo.contrib.vector import FastStepHighs

        m = self._matrix_param_model(T=6)
        s = FastStepHighs()
        s.set_instance(m)
        s.solve()
        # ``dur`` is a nominally-mutable matrix coefficient (dur*p): the value guard
        # watches it.  Confirm the guard engaged, then mask/fix and re-solve.
        guard = s._plan.matrix_guard
        self.assertTrue(guard is not None and not guard.is_empty)
        (r3,) = s.row_indices(m.socc[3])
        s.deactivate_rows([r3])
        jp = s.column_index(m.p[3])
        s.fix_variables([jp], 1.0)
        res = s.solve()  # must not trip the folded/matrix guard
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )

    def test_reload_preserves_overlay(self):
        # on_matrix_change='reload': when a folded matrix coefficient genuinely
        # changes, the whole instance reloads (fresh passModel, basis reset).  A
        # live mask/fix overlay must survive the reload.
        from pyomo.contrib.vector import FastStepHighs

        m = self._matrix_param_model(T=6, dur=1.0)
        s = FastStepHighs(on_matrix_change='reload')
        s.set_instance(m)
        s.solve()
        # mask soc row 4 and fix p[4]; solve.
        (r4,) = s.row_indices(m.socc[4])
        s.deactivate_rows([r4])
        jp4 = s.column_index(m.p[4])
        s.fix_variables([jp4], 2.0)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.p[4]), 2.0, places=6)
        self.assertFalse(s.active_rows[r4])
        # Now genuinely change the folded matrix coefficient -> forces a reload.
        m.dur = 1.25
        res = s.solve()
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        # The overlay survived the reload: p[4] is still pinned, row 4 still masked.
        self.assertAlmostEqual(pyo.value(m.p[4]), 2.0, places=6)
        self.assertFalse(s.active_rows[r4])
        self.assertTrue(s.fixed_variables_mask[jp4])

    def _folded_model(self, T=6):
        # A model whose objective coefficient ``price[t] * dt`` is a product of two
        # mutable Params (non-affine): ``dt`` folds to a watched constant.  Changing
        # ``dt`` forces a *full* reload (re-fold + fresh set_instance).
        m = pyo.ConcreteModel()
        m.T = pyo.RangeSet(0, T - 1)
        m.dt = pyo.Param(initialize=0.25, mutable=True)
        m.price = pyo.Param(m.T, initialize={t: 1.0 for t in range(T)}, mutable=True)
        m.x = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, 5))
        m.floor = pyo.Constraint(m.T, rule=lambda mm, t: mm.x[t] >= 1.0)
        m.obj = pyo.Objective(
            expr=sum(m.price[t] * m.dt * m.x[t] for t in range(T)), sense=pyo.minimize
        )
        return m

    def test_full_reload_preserves_overlay(self):
        # on_matrix_change='reload' + a folded-value change -> _reload_full (a fresh
        # set_instance / re-fold).  A live mask/fix overlay must survive it.
        from pyomo.contrib.vector import FastStepHighs

        m = self._folded_model(T=6)
        s = FastStepHighs(on_matrix_change='reload')
        s.set_instance(m)
        self.assertIn('dt', s.folded_parameters)
        s.solve()
        (r2,) = s.row_indices(m.floor[2])
        s.deactivate_rows([r2])
        jx2 = s.column_index(m.x[2])
        s.fix_variables([jx2], 3.0)
        s.solve()
        self.assertAlmostEqual(pyo.value(m.x[2]), 3.0, places=6)
        # genuine folded-value change -> re-fold + fresh set_instance.
        m.dt = 0.5
        res = s.solve()
        self.assertEqual(
            res.termination_condition, self.TC.convergenceCriteriaSatisfied
        )
        # overlay survived the full reload.
        self.assertAlmostEqual(pyo.value(m.x[2]), 3.0, places=6)
        self.assertFalse(s.active_rows[r2])
        self.assertTrue(s.fixed_variables_mask[jx2])


if __name__ == "__main__":
    unittest.main()
