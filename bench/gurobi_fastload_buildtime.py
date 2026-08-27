"""Public build-time benchmark: classic Pyomo->Gurobi route vs ``gurobi_fastload``.

Mirrors the Phase-2 methodology (``bench/PHASE2_REPORT.md``): both routes
*construct the same classic model*, then differ only in the **hand-off** that
gets the model into Gurobi ("the solver has the model"):

* **classic route** -- the shipped v2 ``gurobi_persistent`` ``set_instance``:
  builds the Gurobi model one constraint at a time (per-row ``addConstr`` /
  ``addConstrs``), the per-row load called out in #3888;
* **fast route** -- ``gurobi_fastload``: the solver-neutral standard-form compile
  (:func:`pyomo.contrib.vector.fastload.compile_fastload_arrays`) handed to
  Gurobi's native matrix API (``addMVar`` + ``addMConstr`` + ``setMObjective``)
  in a handful of bulk calls.

The endpoint is identical, so ``construct + hand-off`` is directly comparable.

=====================================================================
LICENSE HONESTY -- read before quoting any number from this script.
=====================================================================
The pip ``gurobipy`` wheel ships a **size-limited** license: **2000 variables
and 2000 constraints**.  Every size run here stays strictly under that ceiling,
so these are *licensed-size* measurements only.  They show the hand-off cost and
the fast-route ratio at small sizes and confirm both routes agree on the
objective; they make **no** large-scale claim.  The architectural expectation --
a bulk matrix hand-off replacing a per-row load scales the way the HiGHS
``passModel`` route does (Phase-2) -- is stated but is **not locally
measurable** with this license.

Run (needs gurobipy + a usable license)::

    python bench/gurobi_fastload_buildtime.py
"""

from __future__ import annotations

import statistics
import time


# --------------------------------------------------------------------------- #
# Synthetic classic LP (scalable; stays under the size-limited license ceiling)
# --------------------------------------------------------------------------- #
def build_model(n):
    """A banded classic LP: ``n`` vars, ``n + 1`` constraints, ``~2n`` nonzeros.

    ``min sum c_i x_i`` s.t. ``x_i + x_{i+1} >= b_i`` (a coupling band, always
    feasible since ``x_i = 5`` satisfies every row) and one global budget row
    ``sum x_i <= 5 * n`` (a loose ``<=`` row so the matrix carries both senses),
    with ``0 <= x_i <= 5``.  Deterministic, so both routes build the same model.
    """
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, bounds=(0.0, 5.0))

    def band(mm, i):
        b = 1.0 + (i % 7) * 0.25
        if i == n - 1:
            return mm.x[i] >= b
        return mm.x[i] + mm.x[i + 1] >= b

    m.band = pyo.Constraint(m.I, rule=band)
    m.budget = pyo.Constraint(expr=sum(m.x[i] for i in m.I) <= 5.0 * n)
    m.obj = pyo.Objective(
        expr=sum((1.0 + (i % 5) * 0.1) * m.x[i] for i in m.I), sense=pyo.minimize
    )
    return m


# --------------------------------------------------------------------------- #
# Hand-off stages (no solve): "the solver has the model"
# --------------------------------------------------------------------------- #
def classic_handoff(model):
    """Classic per-row load: v2 ``gurobi_persistent`` ``set_instance`` (no solve)."""
    from pyomo.contrib.solver.common.factory import SolverFactory

    opt = SolverFactory('gurobi_persistent')
    opt.config.load_solutions = False
    opt.config.auto_updates.check_for_new_or_removed_constraints = False
    opt.set_instance(model)
    return opt


def fast_handoff(model):
    """Fast route: standard-form compile + Gurobi matrix-API build (no solve).

    Replicates ``FastLoadGurobi._build_gurobi_model`` (the timed hand-off the
    solver performs) without the solve, so it is comparable to ``classic_handoff``.
    """
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    from pyomo.common.enums import ObjectiveSense
    from pyomo.contrib.vector.fastload import compile_fastload_arrays
    from pyomo.contrib.vector.gurobi_fastload import _gurobi_rows, _hessian_to_gurobi_Q

    compiled = compile_fastload_arrays(model, solver_name='gurobi_fastload')

    grb = gp.Model()
    grb.setParam('OutputFlag', 0)
    grb.setParam('NonConvex', 0)

    ginf = GRB.INFINITY
    lo = np.where(np.isneginf(compiled.col_lower), -ginf, compiled.col_lower)
    hi = np.where(np.isposinf(compiled.col_upper), ginf, compiled.col_upper)
    if compiled.integrality.any():
        vtype = np.where(compiled.integrality, GRB.INTEGER, GRB.CONTINUOUS)
    else:
        vtype = GRB.CONTINUOUS
    x = grb.addMVar(
        compiled.n_col, lb=lo.astype(float), ub=hi.astype(float), vtype=vtype
    )

    sel, sense, rhs, con_of = _gurobi_rows(compiled)
    if con_of:
        A = compiled.A.tocsr()
        if sel is not None:
            A = A[np.asarray(sel, dtype=np.int64)]
        grb.addMConstr(A, x, np.asarray(sense), np.asarray(rhs, dtype=float))

    if compiled.has_objective:
        c = compiled.c.astype(float)
        gsense = (
            GRB.MAXIMIZE if compiled.sense == ObjectiveSense.maximize else GRB.MINIMIZE
        )
        if compiled.is_quadratic:
            grb.setMObjective(
                _hessian_to_gurobi_Q(compiled.hessian),
                c,
                float(compiled.c_offset),
                x,
                x,
                x,
                gsense,
            )
        else:
            grb.setMObjective(None, c, float(compiled.c_offset), None, None, x, gsense)
    grb.update()
    return grb


# --------------------------------------------------------------------------- #
# Objective equivalence (untimed correctness check)
# --------------------------------------------------------------------------- #
def objectives(model):
    """Solve the model both routes; return ``(classic_obj, fast_obj)``."""
    import pyomo.environ as pyo
    from pyomo.contrib.solver.common.factory import SolverFactory

    mc = model.clone()
    SolverFactory('gurobi_persistent').solve(mc)
    obj_classic = pyo.value(mc.obj)

    mf = model.clone()
    res = SolverFactory('gurobi_fastload').solve(mf)
    obj_fast = res.incumbent_objective
    return obj_classic, obj_fast


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _median_ms(fn, repeats):
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        obj = fn()
        samples.append((time.perf_counter() - t) * 1e3)
        # release the Gurobi model/env promptly
        try:
            obj.dispose()
        except Exception:
            pass
    return statistics.median(samples)


def run(sizes=(200, 500, 1000, 1500, 1800), repeats=5):
    print("gurobi_fastload build-time benchmark (LICENSED SIZES ONLY, <= 2000)")
    print("=" * 78)
    import gurobipy as gp

    print(f"gurobipy {gp.gurobi.version()}  |  size-limited license: 2000 vars/cons")
    print()
    hdr = (
        f"{'n_vars':>7} {'n_cons':>7} {'construct':>10} "
        f"{'classic HO':>11} {'fast HO':>9} {'HO speedup':>11} "
        f"{'build->solver x':>16} {'obj match':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for n in sizes:
        # construct (shared by both routes)
        construct_ms = _median_ms_construct(n, repeats)
        model = build_model(n)

        # measure hand-offs on a prebuilt model
        classic_ms = _median_ms(lambda: classic_handoff(model), repeats)
        fast_ms = _median_ms(lambda: fast_handoff(model), repeats)

        # objective equivalence
        oc, of = objectives(model)
        match = abs(oc - of) <= 1e-6 * max(1.0, abs(oc))

        # sizes
        from pyomo.core.base.constraint import Constraint
        from pyomo.core.base.var import Var

        nv = sum(len(v) for v in model.component_objects(Var, active=True))
        nc = sum(len(c) for c in model.component_objects(Constraint, active=True))

        ho_speedup = classic_ms / fast_ms if fast_ms else float('nan')
        e2e_classic = construct_ms + classic_ms
        e2e_fast = construct_ms + fast_ms
        e2e_x = e2e_classic / e2e_fast if e2e_fast else float('nan')

        print(
            f"{nv:>7} {nc:>7} {construct_ms:>9.1f}m {classic_ms:>10.1f}m "
            f"{fast_ms:>8.1f}m {ho_speedup:>10.2f}x {e2e_x:>15.2f}x "
            f"{('yes' if match else 'NO'):>10}"
        )

    print()
    print(
        "HO = hand-off (construct excluded).  build->solver x = "
        "(construct+classic)/(construct+fast)."
    )
    print(
        "LICENSE CAVEAT: sizes <= 2000 vars/cons (size-limited pip license); "
        "no large-scale claim is made or measurable locally."
    )


def _median_ms_construct(n, repeats):
    return _median_ms_plain(lambda: build_model(n), repeats)


def _median_ms_plain(fn, repeats):
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1e3)
    return statistics.median(samples)


if __name__ == "__main__":
    run()
