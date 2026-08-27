"""Mutation-cycle micro-benchmark for the persistent vector warm re-solve path.

Phase-2 mutability evidence (scoping doc §6.4, "persistent bound/RHS updates"):
a real mutating workload -- a fix/unfix/bounds/mask sweep on the ragged
supply-chain model -- driven two ways:

* **warm** -- one :class:`~pyomo.contrib.vector.VectorPersistentHighs`, mutate the
  columnar components in place (``setub`` / ``fix`` / ``deactivate_rows``) and
  re-solve, pushing only the dirty columns/rows through HiGHS'
  ``changeColsBounds`` / ``changeRowsBounds`` with the warm basis retained;
* **cold** -- rebuild the model and ``passModel`` from scratch each step (what
  you must do without a persistent path).

Both must agree on the objective at every step (the warm incremental push is
correct relative to a full recompile of the same mutated arrays); the warm path
should be markedly faster per step because it never re-assembles or re-loads the
whole matrix.

Run::

    python -m bench.mutation_cycle --sizes xs,1e4 --steps 12 \
        --out bench/results/phase2_mutation_cycle.json

Note: the ``supply_chain`` *generator* produces an infeasible instance at the
``1e5`` size and above (demand exceeds the ragged lanes' capacity for that
seed/degree -- a pre-existing property of ``bench.models.supply_chain``, not of
the vector path: the classic build is equally infeasible there, and the vector
build agrees).  This benchmark therefore sweeps the feasible sizes; the
build-to-solver speedup benchmark (``run_bench --backends pyomo,pyomo_vector``)
still covers ``1e5`` because time-to-solver does not solve.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

from pyomo.common.dependencies import numpy as np

from bench.models import supply_chain as _classic, supply_chain_vector as _vec
from pyomo.contrib.vector import (
    VectorPersistentHighs,
    assemble,
    matrices_to_highs_model,
)


def _apply_step(m, step: int) -> None:
    """Apply the deterministic mutation for ``step`` to a supply_chain model.

    Reproducible so the warm (incremental) and cold (rebuild) routes reach the
    same state at each step.  The sweep is designed to stay *feasible* and to be
    *reversible* (fixes are unfixed, masked rows reactivated a step later) so
    every step re-optimizes a genuinely different, solvable model -- exercising
    all three incremental HiGHS paths:

      * bounds  -- rotate ``ship_sw`` upper bounds in ``[50, 100]`` on a rotating
        column subset (``changeColsBounds``); shipments are far below 50 so this
        never starves demand;
      * fixed   -- pin one rotating ``ship_sw`` supply lane to a mid value ``30``
        (a forced inflow the warehouse absorbs via inventory/outflow) and release
        the previous one (``changeColsBounds`` pin/unpin); pinning a supply lane
        cannot starve downstream demand, so feasibility is preserved;
      * mask    -- deactivate one rotating ``scap`` row (a pure relaxation) and
        reactivate the previous one (``changeRowsBounds``).
    """
    n_sw = m.ship_sw.n
    nc = m.scap.nrows
    # bounds: rotate ship_sw ub within a safe [50, 100] band.
    where = np.arange(step % 5, n_sw, 5)
    m.ship_sw.setub(50.0 + (step % 6) * 10.0, where=where)
    # fixed: pin one rotating ship_sw supply lane to 30, release the previous.
    if n_sw:
        cur = (step * 7) % n_sw
        prev = ((step - 1) * 7) % n_sw
        m.ship_sw.unfix(where=np.array([prev]))
        m.ship_sw.fix(30.0, where=np.array([cur]))
    # mask: deactivate one rotating scap row, reactivate the previous one.
    if nc:
        cur = (step * 2) % nc
        prev = ((step - 1) * 2) % nc
        m.scap.activate_rows(np.array([prev]))
        m.scap.deactivate_rows(np.array([cur]))


def _solve_cold(params, upto_step: int):
    """Fresh build + apply steps 0..upto_step + passModel + solve.

    Returns ``(feasible, objective)`` -- feasibility is read from HiGHS' primal
    solution status (an infeasible model reports objective 0.0, so it must be
    checked, not trusted).
    """
    import highspy

    m = _vec.build_pyomo(params)
    for s in range(upto_step + 1):
        _apply_step(m, s)
    mx = assemble(m)
    h = highspy.Highs()
    h.silent()
    h.passModel(matrices_to_highs_model(mx))
    h.run()
    info = h.getInfo()
    feasible = info.primal_solution_status == 2
    return feasible, (info.objective_function_value if feasible else None)


def _median(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def run_size(size: str, params: Dict[str, Any], steps: int) -> Dict[str, Any]:
    # nnz report
    mx0 = assemble(_vec.build_pyomo(params))
    nnz = mx0.nnz

    # --- warm route: one persistent handle, mutate + re-solve in place ------ #
    m = _vec.build_pyomo(params)
    p = VectorPersistentHighs(m)
    p.solve()  # initial solve (basis established)
    warm_times, warm = [], []
    for s in range(steps):
        _apply_step(m, s)
        t = time.perf_counter()
        r = p.solve()
        warm_times.append(time.perf_counter() - t)
        warm.append((r.termination == 'optimal', r.objective))

    # --- cold route: rebuild + passModel + solve each step ------------------ #
    cold_times, cold = [], []
    for s in range(steps):
        t = time.perf_counter()
        feas, obj = _solve_cold(params, s)
        cold_times.append(time.perf_counter() - t)
        cold.append((feas, obj))

    # --- correctness: warm vs cold agree on feasibility + objective --------- #
    max_mismatch = 0.0
    agree = True
    for (wf, wo), (cf, co) in zip(warm, cold):
        if wf != cf:
            agree = False
            continue
        if wf and cf:
            rel = abs(wo - co) / max(1.0, abs(co))
            max_mismatch = max(max_mismatch, rel)
            if rel > 1e-6:
                agree = False
    n_feasible = sum(1 for wf, _ in warm if wf)

    warm_ms = _median(warm_times) * 1000.0
    cold_ms = _median(cold_times) * 1000.0
    return {
        "size": size,
        "nnz": int(nnz),
        "steps": steps,
        "feasible_steps": int(n_feasible),
        "warm_resolve_ms_median": round(warm_ms, 3),
        "cold_reload_ms_median": round(cold_ms, 3),
        "warm_speedup": round(cold_ms / warm_ms, 2) if warm_ms else None,
        "max_rel_objective_mismatch": float(max_mismatch),
        "warm_cold_agree": bool(agree),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="xs,1e4")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    results = []
    for size in args.sizes.split(","):
        size = size.strip()
        params = _classic.SIZES[size]
        res = run_size(size, params, args.steps)
        results.append(res)
        print(
            f"[{size}] nnz={res['nnz']} steps={res['steps']} "
            f"feasible={res['feasible_steps']}/{res['steps']} "
            f"warm={res['warm_resolve_ms_median']}ms "
            f"cold={res['cold_reload_ms_median']}ms "
            f"speedup={res['warm_speedup']}x "
            f"warm_cold_agree={res['warm_cold_agree']} "
            f"(max_rel_mismatch={res['max_rel_objective_mismatch']:.2e})"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mutation_cycle": results}, f, indent=2)
        print(f"# wrote {args.out}")
    return results


if __name__ == "__main__":
    main()
