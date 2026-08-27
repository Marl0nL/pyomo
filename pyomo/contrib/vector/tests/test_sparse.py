# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Sparse / ragged index sets on the vectorized fast path.

The scoping doc (R5) insists the fast path handle *ragged* sparsity -- variables
over a sparse subset ``arcs subset nodes x nodes`` -- not just dense boxes.  A
:class:`~pyomo.contrib.vector.var.VectorVar` over a sparse Pyomo ``Set`` already
stores one column per member and maps index tuples to positions through its
hash-to-position map (``position_of``); these tests pin that behaviour and check
that a ragged model built with the vector API produces the *same* standard form
(and solve) as the classic build with identical component names/indices.
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

# A ragged lane set: supplier 0 -> {0, 1}, supplier 1 -> {1} (not the full 2x2).
_LANES = [(0, 0), (0, 1), (1, 1)]
_CAP = 5.0


def _ragged_vector_model():
    import scipy.sparse as sp
    from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective

    m = pyo.ConcreteModel()
    m.periods = pyo.RangeSet(0, 1)
    m.SW = pyo.Set(initialize=_LANES, dimen=2)
    m.ship = VectorVar(m.SW, m.periods, domain=pyo.NonNegativeReals, bounds=(0, 10))
    m.ship.construct()

    # Supplier-capacity rows: for each (s, t), sum over lanes out of s.
    suppliers = sorted({s for s, _ in _LANES})
    out_lanes = {s: [(s2, w) for (s2, w) in _LANES if s2 == s] for s in suppliers}
    T = len(m.periods)
    pos = {idx: m.ship.position_of(idx) for idx in m.ship._index_set}

    rows, cols, data = [], [], []
    r = 0
    for s in suppliers:
        for t in range(T):
            for s2, w in out_lanes[s]:
                rows.append(r)
                cols.append(pos[(s2, w, t)])
                data.append(1.0)
            r += 1
    A = sp.coo_matrix((data, (rows, cols)), shape=(r, m.ship.n)).tocsr()
    m.scap = VectorConstraint(A=A, x=m.ship, ub=np.full(r, _CAP))
    m.scap.construct()

    c = np.array([-1.0] * m.ship.n)  # min -sum ship  ->  push shipments up
    m.obj = VectorObjective(terms={m.ship: c}, sense=pyo.minimize)
    m.obj.construct()
    return m


def _ragged_classic_model():
    m = pyo.ConcreteModel()
    m.periods = pyo.RangeSet(0, 1)
    m.SW = pyo.Set(initialize=_LANES, dimen=2)
    m.ship = pyo.Var(m.SW, m.periods, domain=pyo.NonNegativeReals, bounds=(0, 10))
    suppliers = sorted({s for s, _ in _LANES})
    out_lanes = {s: [(s2, w) for (s2, w) in _LANES if s2 == s] for s in suppliers}

    def scap_rule(m, s, t):
        return sum(m.ship[s2, w, t] for (s2, w) in out_lanes[s]) <= _CAP

    m.suppliers = pyo.Set(initialize=suppliers)
    m.scap = pyo.Constraint(m.suppliers, m.periods, rule=scap_rule)
    m.obj = pyo.Objective(
        expr=-sum(m.ship[s, w, t] for (s, w) in _LANES for t in m.periods),
        sense=pyo.minimize,
    )
    return m


@unittest.skipUnless(_arrays, "vector requires numpy/scipy")
class TestSparseIndex(unittest.TestCase):
    def test_position_map_over_sparse_2d_index(self):
        m = _ragged_vector_model()
        # 3 lanes x 2 periods = 6 columns; the missing (1,0) lane is absent.
        self.assertEqual(m.ship.n, 6)
        # index tuples flatten (s, w, t); the ragged member (1, 0, *) never exists.
        self.assertNotIn((1, 0, 0), m.ship._index_set)
        # position_of round-trips through index_at
        for idx in m.ship._index_set:
            self.assertEqual(m.ship.index_at(m.ship.position_of(idx)), idx)

    def test_ragged_standard_form_matches_classic(self):
        from pyomo.contrib.vector import compile_standard_form
        from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
        from pyomo.contrib.vector.tests.equivalence_oracle import (
            canonical_standard_form,
        )

        iv = compile_standard_form(_ragged_vector_model(), mixed_form=True)
        ic = LinearStandardFormCompiler().write(
            _ragged_classic_model(), mixed_form=True
        )
        self.assertEqual(canonical_standard_form(iv), canonical_standard_form(ic))

    def test_scalar_access_on_sparse_var(self):
        m = _ragged_vector_model()
        v = m.ship[0, 1, 0]  # a valid ragged lane
        self.assertEqual(v.bounds, (0.0, 10.0))
        self.assertIs(m.ship[0, 1, 0], v)  # identity (materialize-on-touch)
        with self.assertRaises(KeyError):
            m.ship[1, 0, 0]  # the absent lane

    @unittest.skipUnless(_solve, "solve requires highspy")
    def test_ragged_solve_matches_classic(self):
        from pyomo.contrib.vector.highs import solve_highs
        from pyomo.contrib.appsi.solvers.highs import Highs

        _, obj = solve_highs(_ragged_vector_model())
        mc = _ragged_classic_model()
        Highs().solve(mc)
        self.assertAlmostEqual(obj, float(pyo.value(mc.obj)), places=6)


if __name__ == "__main__":
    unittest.main()
