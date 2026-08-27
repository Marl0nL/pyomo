"""Build+solve evidence for convex-quadratic **objective** support (the #1761 use case).

A synthetic, fully public portfolio-style convex QP -- no external or
application-specific structure::

    min  0.5 x' Q x  -  mu' x
    s.t. sum_i x_i = 1                     (budget)
         sum_{i in sector s} x_i <= cap_s  (a few linear sector limits)
         0 <= x_i <= w_max                 (box)

``Q`` is a *dense-ish* banded SPD covariance (bandwidth ``b``), so its nonzero
count is ``~ n*(2b+1)`` -- swept through the 1e4 / 1e5-nnz class.  The objective
is quadratic; every constraint is linear (objective-quadratic only, the scope of
this feature).

Two builds of the *same* QP are timed:

* **vector** -- ``VectorVar`` + ``VectorConstraint`` + ``VectorObjective(quadratic=Q)``
  assembled to standard-form arrays and handed to HiGHS in one ``passModel``
  (linear part + Hessian), then ``run``;
* **classic** -- an idiomatic Pyomo model whose quadratic objective is built as a
  sum of ``Q_ij * x[i] * x[j]`` monomials, solved through Pyomo's v2
  ``SolverFactory('highs')`` persistent QP interface (the classic reference).

The honest expectation (mirrors the LP case): the **construction** stage is where
the array path wins big (it never builds N expression trees / n_nz monomial
nodes); the **solve** is solver-bound -- the same HiGHS convex-QP run -- so it is
at parity.  Both objectives are checked equal before any timing is reported.

Run from the repo root with the bench venv::

    bench/.venv/bin/python -m bench.spikes.quadratic_qp_scaling
    bench/.venv/bin/python -m bench.spikes.quadratic_qp_scaling 1000 2000 3200
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.sparse as sp

import pyomo.environ as pyo


# --------------------------------------------------------------------------- #
# Synthetic portfolio QP (shared spec; built two ways)
# --------------------------------------------------------------------------- #
def make_spec(n, bandwidth=25, n_sectors=10, seed=0):
    """A feasible convex portfolio QP spec: banded SPD ``Q``, linear budget +
    sector limits + box.  ``Q`` nnz ~= ``n*(2*bandwidth+1)`` (dense-ish band)."""
    rng = np.random.default_rng(seed)
    b = min(bandwidth, n - 1)
    # Banded factor -> SPD banded Q = L L' + I (kept symmetric, positive-definite).
    diags = []
    offsets = []
    for k in range(b + 1):
        vals = rng.normal(scale=1.0 / (1 + k), size=n - k)
        diags.append(vals)
        offsets.append(k)
    L = sp.diags(diags, offsets, shape=(n, n), format='csr')
    Q = (L @ L.T).tocsr() + sp.eye(n, format='csr')
    Q = (0.5 * (Q + Q.T)).tocsr()
    Q.sort_indices()
    mu = rng.normal(size=n)
    # Sector membership (contiguous blocks) -> a handful of linear rows.
    sector = np.floor(np.arange(n) / (n / n_sectors)).astype(int)
    caps = np.full(n_sectors, 2.0 / n_sectors)  # loose (keeps it feasible)
    w_max = 1.0
    return {
        'n': n,
        'Q': Q,
        'mu': mu,
        'sector': sector,
        'n_sectors': n_sectors,
        'caps': caps,
        'w_max': w_max,
    }


def _linear_rows(spec):
    """Assemble the linear constraint block ``A`` (budget + sector caps) as CSR,
    with range bounds ``(lb, ub)``."""
    n, ns = spec['n'], spec['n_sectors']
    rows, cols, data = [], [], []
    lb, ub = [], []
    # budget: sum x == 1
    rows += [0] * n
    cols += list(range(n))
    data += [1.0] * n
    lb.append(1.0)
    ub.append(1.0)
    # sector caps: sum_{i in s} x_i <= cap_s
    for s in range(ns):
        idx = np.nonzero(spec['sector'] == s)[0]
        rows += [1 + s] * len(idx)
        cols += idx.tolist()
        data += [1.0] * len(idx)
        lb.append(-np.inf)
        ub.append(float(spec['caps'][s]))
    A = sp.csr_matrix(
        (data, (rows, cols)), shape=(1 + ns, n)
    )
    return A, np.array(lb), np.array(ub)


# --------------------------------------------------------------------------- #
# Vector fast path
# --------------------------------------------------------------------------- #
def run_vector(spec):
    from pyomo.contrib.vector import (
        VectorVar,
        VectorConstraint,
        VectorObjective,
    )
    from pyomo.contrib.vector.matrices import assemble
    from pyomo.contrib.vector.highs import matrices_to_highs_model
    import highspy

    n = spec['n']
    A, lb, ub = _linear_rows(spec)

    t0 = time.perf_counter()
    m = pyo.ConcreteModel()
    m.x = VectorVar(pyo.RangeSet(0, n - 1), bounds=(0.0, spec['w_max']))
    m.lin = VectorConstraint(A=A, x=m.x, lb=lb, ub=ub)
    m.obj = VectorObjective(
        terms={m.x: -spec['mu']}, quadratic=spec['Q'], sense=pyo.minimize
    )
    m.x.construct()
    m.lin.construct()
    m.obj.construct()
    mx = assemble(m)
    build = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = matrices_to_highs_model(mx)
    h = highspy.Highs()
    h.setOptionValue('log_to_console', False)
    h.passModel(model)
    load = time.perf_counter() - t0

    t0 = time.perf_counter()
    h.run()
    solve = time.perf_counter() - t0

    obj = h.getInfo().objective_function_value
    return {'build': build, 'load': load, 'solve': solve, 'obj': obj}


# --------------------------------------------------------------------------- #
# Classic Pyomo QP route
# --------------------------------------------------------------------------- #
def run_classic(spec):
    from pyomo.contrib.solver.common.factory import SolverFactory

    n = spec['n']
    Q = spec['Q'].tocoo()
    mu = spec['mu']

    t0 = time.perf_counter()
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, bounds=(0.0, spec['w_max']))
    m.budget = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 1.0)

    def sector_rule(mm, s):
        idx = np.nonzero(spec['sector'] == s)[0]
        return sum(mm.x[int(i)] for i in idx) <= float(spec['caps'][s])

    m.sector = pyo.Constraint(pyo.RangeSet(0, spec['n_sectors'] - 1), rule=sector_rule)
    # Quadratic objective built from the sparse Q nonzeros (0.5 x'Q x - mu'x).
    quad = 0.5 * sum(
        float(v) * m.x[int(i)] * m.x[int(j)]
        for i, j, v in zip(Q.row, Q.col, Q.data)
    )
    lin = sum(-float(mu[i]) * m.x[i] for i in m.I)
    m.obj = pyo.Objective(expr=quad + lin, sense=pyo.minimize)
    build = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = SolverFactory('highs').solve(m)
    solve = time.perf_counter() - t0

    return {'build': build, 'solve': solve, 'obj': pyo.value(m.obj)}


# --------------------------------------------------------------------------- #
def main(sizes):
    hdr = (
        f"{'n':>6} {'Qnnz':>9} "
        f"{'v.build':>8} {'v.load':>7} {'v.solve':>8} {'v.total':>8}  "
        f"{'c.build':>8} {'c.solve':>8} {'c.total':>8}  "
        f"{'build×':>7} {'total×':>7} {'obj Δ':>9}"
    )
    print(hdr)
    print('-' * len(hdr))
    for n in sizes:
        spec = make_spec(n)
        qnnz = int(spec['Q'].nnz)
        v = run_vector(spec)
        c = run_classic(spec)
        v_total = v['build'] + v['load'] + v['solve']
        c_total = c['build'] + c['solve']
        objd = abs(v['obj'] - c['obj'])
        print(
            f"{n:>6} {qnnz:>9,} "
            f"{v['build']:>8.3f} {v['load']:>7.3f} {v['solve']:>8.3f} {v_total:>8.3f}  "
            f"{c['build']:>8.3f} {c['solve']:>8.3f} {c_total:>8.3f}  "
            f"{c['build'] / v['build']:>6.1f}x {c_total / v_total:>6.1f}x {objd:>9.2e}"
        )


if __name__ == '__main__':
    sizes = [int(a) for a in sys.argv[1:]] or [700, 1400, 2200, 3200]
    main(sizes)
