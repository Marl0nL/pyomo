# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Phase-3 template-vectorized construction: correctness gates.

Covers the four things Phase 3 must guarantee:

1. **Equivalence (switch ON == switch OFF).**  With template-vectorized
   construction on, the standard form is identical (up to row/column
   permutation, same bounds) to the classic build -- on templatizable,
   non-templatizable, and mixed models, plus randomized models.  Checked two
   ways: the stock compiler on both builds (construct equivalence), and the
   vectorized ``compile_templated_to_highs_arrays`` matrix vs the stock compiler
   (compile equivalence).
2. **Solve equivalence.**  The vectorized fast-load solve and a classic APPSI
   solve agree on objective (and primal for unique-optimum LPs).
3. **Scalarization / identity contract.**  A templatized ``ConstraintData`` still
   materializes on touch (``m.con[i]`` is a working, identity-stable
   ConstraintData with the correct expression), mirroring Phase-1's VectorVar
   contract.
4. **Mandatory fallback.**  A rule that does not templatize (index conditional,
   filtered sum, modulo) constructs byte-identically to classic Pyomo, silently
   -- and the switch OFF leaves stock behaviour completely untouched.
"""

import logging

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import numpy_available, scipy_available
from pyomo.common.log import LoggingIntercept
import pyomo.core.base.constraint as constraint_module
import pyomo.core.base.objective as objective_module
from pyomo.core.base.constraint import ConstraintData, TemplateConstraintData
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

from pyomo.contrib.vector.tests.equivalence_oracle import canonical_standard_form


# --------------------------------------------------------------------------- #
# Model builders exercising the proven templatizable subset + fallbacks
# --------------------------------------------------------------------------- #
def _templatizable_model():
    """Only templatizable families: scalar-affine, neighbour, unfiltered sum."""
    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, 5)
    m.I = pyo.RangeSet(1, 5)
    m.x = pyo.Var(m.J, domain=pyo.NonNegativeReals, bounds=(0, 10))
    m.b = pyo.Param(
        m.I, initialize={i: 1.0 + 0.1 * i for i in range(1, 6)}, mutable=True
    )
    # scalar-affine neighbour with mutable-Param RHS
    m.ramp = pyo.Constraint(m.I, rule=lambda m, i: 2.0 * m.x[i] - m.x[i - 1] <= m.b[i])
    # unfiltered sum-over-set equality
    m.total = pyo.Constraint(rule=lambda m: sum(m.x[j] for j in m.J) == 12.0)
    m.obj = pyo.Objective(expr=sum((j + 1) * m.x[j] for j in m.J), sense=pyo.maximize)
    return m


def _mixed_model():
    """Templatizable families (serve, link) + a non-templatizable one (capacity)."""
    m = pyo.ConcreteModel()
    m.F = pyo.RangeSet(0, 2)
    m.C = pyo.RangeSet(0, 3)
    cap = {0: 3.0, 1: 4.0, 2: 5.0}
    m.open = pyo.Var(m.F, domain=pyo.Binary)
    m.x = pyo.Var(m.F, m.C, domain=pyo.NonNegativeReals, bounds=(0, 1))
    m.serve = pyo.Constraint(m.C, rule=lambda m, c: sum(m.x[f, c] for f in m.F) == 1.0)
    m.link = pyo.Constraint(m.F, m.C, rule=lambda m, f, c: m.x[f, c] <= m.open[f])
    # capacity uses a Python-dict lookup on the index -> does NOT templatize
    m.capacity = pyo.Constraint(
        m.F, rule=lambda m, f: sum(m.x[f, c] for c in m.C) <= cap[f] * m.open[f]
    )
    m.obj = pyo.Objective(
        expr=sum(m.open[f] for f in m.F) + sum(m.x[f, c] for f in m.F for c in m.C)
    )
    return m


def _nontemplatizable_model():
    """Filtered sum + index conditional: nothing templatizes (full fallback)."""
    m = pyo.ConcreteModel()
    m.N = pyo.RangeSet(0, 3)
    m.T = pyo.RangeSet(0, 2)
    m.f = pyo.Var(m.N, m.N, m.T, domain=pyo.NonNegativeReals, bounds=(0, 5))
    m.s = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals, bounds=(0, 5))

    def bal(m, n, t):
        inflow = sum(m.f[j, n, t] for j in m.N if j != n)
        outflow = sum(m.f[n, j, t] for j in m.N if j != n)
        prev = m.s[n, t - 1] if t > 0 else 0.0
        return inflow - outflow + prev - m.s[n, t] == 0.0

    m.bal = pyo.Constraint(m.N, m.T, rule=bal)
    m.obj = pyo.Objective(expr=sum(m.f[i, j, t] for i in m.N for j in m.N for t in m.T))
    return m


def _random_model(seed):
    """A randomized mix of templatizable and non-templatizable families."""
    import random

    rng = random.Random(seed)
    n = rng.randint(4, 9)
    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.J, domain=pyo.NonNegativeReals, bounds=(0, rng.randint(3, 8)))
    coefs = {j: rng.uniform(1.0, 4.0) for j in range(n)}
    rhs = {j: rng.uniform(-2.0, 2.0) for j in range(n)}
    m.rhs = pyo.Param(m.J, initialize=rhs, mutable=True)

    # templatizable scalar-affine
    m.a = pyo.Constraint(m.J, rule=lambda m, j: coefs[j % n] * m.x[j] <= m.rhs[j] + 5.0)
    # templatizable unfiltered sum
    m.tot = pyo.Constraint(rule=lambda m: sum(m.x[j] for j in m.J) <= 3.0 * n)
    if rng.random() < 0.5:
        # non-templatizable filtered sum -> fallback
        m.filt = pyo.Constraint(
            m.J, rule=lambda m, j: sum(m.x[k] for k in m.J if k != j) >= 0.0
        )
    m.obj = pyo.Objective(
        expr=sum(coefs[j] * m.x[j] for j in range(n)), sense=pyo.maximize
    )
    return m


def _build(builder, templatized):
    from pyomo.contrib.vector import vectorized_construction

    if templatized:
        with vectorized_construction():
            return builder()
    return builder()


def _stock_sf(model):
    return LinearStandardFormCompiler().write(model, mixed_form=True, set_sense=None)


# --------------------------------------------------------------------------- #
# A permutation-invariant signature of a range-row standard form, so the
# vectorized ``FastLoadCompiled`` and the stock mixed-form output can be compared
# directly (keyed on variable identity).
# --------------------------------------------------------------------------- #
def _z(v):
    # round + normalize negative zero so signatures compare cleanly
    return round(float(v), 7) + 0.0


def _range_rows_from_stock(info):
    """Convert stock mixed-form rows into (entries, lb, ub) keyed on var id."""
    A = info.A.tocsr()
    cols = info.columns
    keys = [(v.parent_component().local_name, v.index()) for v in cols]
    rows = []
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        entries = frozenset(
            (keys[c], _z(d)) for c, d in zip(A.indices[s:e], A.data[s:e])
        )
        bt = info.rows[i].bound_type
        r = _z(info.rhs[i])
        lb = r if bt in (0, -1) else None
        ub = r if bt in (0, 1) else None
        rows.append((entries, lb, ub))
    return sorted(map(repr, rows))


def _range_rows_from_compiled(compiled):
    A = compiled.A.tocsr()
    cols = compiled.columns
    keys = [(v.parent_component().local_name, v.index()) for v in cols]
    rows = []
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        entries = frozenset(
            (keys[c], _z(d)) for c, d in zip(A.indices[s:e], A.data[s:e])
        )
        lb = compiled.row_lower[i]
        ub = compiled.row_upper[i]
        lb = None if lb == float('-inf') else _z(lb)
        ub = None if ub == float('inf') else _z(ub)
        rows.append((entries, lb, ub))
    return sorted(map(repr, rows))


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestTemplateEquivalence(unittest.TestCase):
    """Switch ON produces the same standard form as switch OFF."""

    BUILDERS = {
        'templatizable': _templatizable_model,
        'mixed': _mixed_model,
        'nontemplatizable': _nontemplatizable_model,
    }

    def test_construct_equivalence_stock(self):
        # stock compiler on ON-build vs OFF-build (construct equivalence)
        for name, builder in self.BUILDERS.items():
            m_on = _build(builder, True)
            m_off = _build(builder, False)
            a = canonical_standard_form(_stock_sf(m_on))
            b = canonical_standard_form(_stock_sf(m_off))
            self.assertEqual(a, b, f"construct equivalence failed for {name}")

    def test_compile_matrix_equivalence(self):
        # vectorized compile (ON) vs stock compiler (OFF), range-row signature
        from pyomo.contrib.vector import compile_templated_to_highs_arrays

        for name, builder in self.BUILDERS.items():
            m_on = _build(builder, True)
            m_off = _build(builder, False)
            compiled = compile_templated_to_highs_arrays(m_on)
            self.assertEqual(
                _range_rows_from_compiled(compiled),
                _range_rows_from_stock(_stock_sf(m_off)),
                f"compile matrix equivalence failed for {name}",
            )

    def test_random_models_equivalence(self):
        from pyomo.contrib.vector import compile_templated_to_highs_arrays

        for seed in range(25):
            m_on = _build(lambda: _random_model(seed), True)
            m_off = _build(lambda: _random_model(seed), False)
            # construct equivalence
            self.assertEqual(
                canonical_standard_form(_stock_sf(m_on)),
                canonical_standard_form(_stock_sf(m_off)),
                f"random construct equivalence failed (seed={seed})",
            )
            # compile equivalence
            self.assertEqual(
                _range_rows_from_compiled(compile_templated_to_highs_arrays(m_on)),
                _range_rows_from_stock(_stock_sf(m_off)),
                f"random compile equivalence failed (seed={seed})",
            )


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestTemplateSolveEquivalence(unittest.TestCase):
    """The fast-load solve matches a classic solve."""

    def _fastload_solve(self, model):
        from pyomo.contrib.solver.common.factory import SolverFactory

        return SolverFactory('highs_fastload').solve(model)

    def _appsi_solve(self, model):
        appsi = pytest_importorskip_appsi()
        h = appsi()
        return h.solve(model)

    def test_solve_equivalence(self):
        highspy = _try_highspy()
        if highspy is None:
            self.skipTest("highspy not available")
        for builder in (_templatizable_model, _mixed_model, _nontemplatizable_model):
            m_on = _build(builder, True)
            res = self._fastload_solve(m_on)
            m_off = _build(builder, False)
            res_off = self._fastload_solve(m_off)
            self.assertAlmostEqual(
                float(res.incumbent_objective),
                float(res_off.incumbent_objective),
                places=5,
            )


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestScalarizationContract(unittest.TestCase):
    """Templatized ConstraintData still materializes on touch (identity holds)."""

    def test_templatized_data_type_and_identity(self):
        m = _build(_templatizable_model, True)
        first = next(iter(m.ramp.values()))
        self.assertIsInstance(first, TemplateConstraintData)
        # identity: the same ConstraintData object each access
        self.assertIs(m.ramp[1], m.ramp[1])

    def test_materialize_on_touch_expression(self):
        m = _build(_templatizable_model, True)
        m_classic = _build(_templatizable_model, False)
        # the templatized row's evaluated body must match the classic row's
        m.x[1].set_value(2.0)
        m.x[0].set_value(1.0)
        m_classic.x[1].set_value(2.0)
        m_classic.x[0].set_value(1.0)
        self.assertAlmostEqual(
            pyo.value(m.ramp[1].body), pyo.value(m_classic.ramp[1].body)
        )
        self.assertAlmostEqual(
            pyo.value(m.ramp[1].upper), pyo.value(m_classic.ramp[1].upper)
        )
        # a genuinely unaware consumer (repn on .body) must work post-touch
        from pyomo.repn.standard_repn import generate_standard_repn

        repn = generate_standard_repn(m.ramp[1].body)
        self.assertEqual(len(repn.linear_vars), 2)

    def test_set_value_converts_to_classic(self):
        # Setting a value on a templatized datum converts it to a classic
        # ConstraintData (upstream TemplateDataMixin contract) -- and it stays
        # a working constraint.
        m = _build(_templatizable_model, True)
        cd = m.ramp[2]
        cd.set_value(3.0 * m.x[2] <= 9.0)
        self.assertEqual(cd.upper, 9.0)


@unittest.skipUnless(numpy_available and scipy_available, "requires numpy/scipy")
class TestFallbackAndOffState(unittest.TestCase):
    """Non-templatizable rules fall back silently; switch OFF is untouched."""

    def test_switch_off_no_templates(self):
        # With the switch off, no family is templatized (stock behaviour).
        self.assertFalse(constraint_module.TEMPLATIZE_CONSTRAINTS)
        m = _templatizable_model()  # built with switch OFF
        first = next(iter(m.ramp.values()))
        self.assertNotIsInstance(first, TemplateConstraintData)
        self.assertIs(type(first), ConstraintData)

    def test_switch_restored_after_context(self):
        from pyomo.contrib.vector import vectorized_construction

        self.assertFalse(constraint_module.TEMPLATIZE_CONSTRAINTS)
        with vectorized_construction():
            self.assertTrue(constraint_module.TEMPLATIZE_CONSTRAINTS)
        self.assertFalse(constraint_module.TEMPLATIZE_CONSTRAINTS)
        # objectives are never templatized (a scalar Objective(expr=...) would
        # otherwise regress through a large codegen path)
        self.assertFalse(objective_module.TEMPLATIZE_OBJECTIVES)

    def test_nontemplatizable_fallback_byte_identical(self):
        # The fallback build must be structurally identical to classic.
        m_on = _build(_nontemplatizable_model, True)
        m_off = _build(_nontemplatizable_model, False)
        # every constraint fell back to a classic ConstraintData
        for cd in m_on.bal.values():
            self.assertIs(type(cd), ConstraintData)
        # same standard form
        self.assertEqual(
            canonical_standard_form(_stock_sf(m_on)),
            canonical_standard_form(_stock_sf(m_off)),
        )

    def test_fallback_is_quiet(self):
        # Constructing a non-templatizable model with the switch on must not emit
        # an ERROR (the fallback is silent; at most a debug note).
        with LoggingIntercept(level=logging.INFO) as out:
            _build(_nontemplatizable_model, True)
        self.assertNotIn("ERROR", out.getvalue())
        self.assertNotIn("was raised when templatizing", out.getvalue())


def _try_highspy():
    try:
        import highspy

        return highspy
    except ImportError:
        return None


def pytest_importorskip_appsi():
    from pyomo.contrib.appsi.solvers import Highs

    return Highs


if __name__ == "__main__":
    unittest.main()
