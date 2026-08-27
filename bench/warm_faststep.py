"""Warm-tick benchmark: ``highs_faststep`` vs the persistent APPSI HiGHS route.

The Phase-4 exit-criteria evidence.  The cold build+solve benchmarks
(``run_bench.py`` / ``PHASE2``/``PHASE3``) measure the *once-per-process* stage.
This one measures the *warm rolling* stage that dominates cumulative compute in a
model-predictive-control / rolling-horizon deployment: ``construct once, then
re-solve thousands of times with slightly changed data``.

It rolls the synthetic :mod:`bench.models.rolling_mpc` model forward ``--rolls``
times and times, per roll, the **warm tick** -- the update + re-solve -- for:

* **appsi_persistent** -- the persistent APPSI HiGHS interface configured for its
  fastest warm path (structural scans off, ``update_params`` on): the per-
  coefficient ``value(expr)`` + scalar ``changeColCost`` / ``changeRowBounds``
  loop the Phase-4 evidence identified as the dominant warm-tick cost.
* **faststep_model** -- :class:`~pyomo.contrib.vector.faststep.FastStepHighs`
  driven from the model (mutate Params, ``solve``): vectorized affine-template
  evaluation (``M @ P``) + one batch ``changeColsCost`` / ``changeRowsBounds`` /
  ``changeColsBounds`` per group, warm basis kept.
* **faststep_array** -- the same, driven from a raw parameter-value array
  (``solve(param_values=P)``): the mapping-free path a private integration feeds.
* **faststep_valueguard** -- the value-aware static-matrix guard leg.  The model
  is rebuilt with a *nominally-mutable* constraint-matrix coefficient
  (``eff[a,t]`` on the state-of-charge recurrence) whose value never changes
  under the equal-interval roll.  The pre-guard interface rejected such a model
  outright; the value guard now accepts it, verifies the matrix is unchanged each
  roll (one vectorized ``M @ P`` + compare), and keeps the warm basis.  The table
  reports this leg's warm tick alongside the per-roll guard-check overhead and
  its ratio to the pure-static ``faststep_model`` tick (target: guard overhead
  ``< 10%`` of the warm tick; leg within ``~10%`` of the static path).

Every roll's objective is checked equal across all routes (the warm-solve
equivalence gate).  The table reports the median warm tick and the faststep
speedup; the exit target is ``faststep_model >= 1.4x`` the APPSI route at the
``1e5`` size.

Usage::

    bench/.venv/bin/python -m bench.warm_faststep --sizes 1e4,1e5 --rolls 30
    bench/.venv/bin/python -m bench.warm_faststep --sizes 1e5 --rolls 40 \
        --out bench/results/phase4_faststep.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List

import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.vector.faststep import FastStepHighs
from pyomo.contrib.vector.fastload import compile_to_highs_arrays

from bench.models import rolling_mpc


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def _appsi_persistent(model):
    """A persistent APPSI HiGHS solver configured for the fastest warm path."""
    from pyomo.contrib.appsi.solvers import Highs as AppsiHighs

    opt = AppsiHighs()
    uc = opt.update_config
    # Structure is fixed across rolls: skip the structural scans (the tuning the
    # warm-path evidence assumes) and only push the changed parameter data.
    uc.check_for_new_or_removed_constraints = False
    uc.check_for_new_or_removed_vars = False
    uc.check_for_new_or_removed_params = False
    uc.check_for_new_objective = False
    uc.update_constraints = False  # matrix static
    uc.update_named_expressions = False
    uc.update_vars = True  # variable bounds roll
    uc.update_params = True  # objective coefs + RHS roll
    opt.config.load_solution = True
    return opt


def _median_ms(xs: List[float]) -> float:
    return statistics.median(xs) * 1e3


# --------------------------------------------------------------------------- #
# one size
# --------------------------------------------------------------------------- #
def run_size(size: str, dims: Dict[str, Any], rolls: int, seed0: int) -> Dict[str, Any]:
    # structural stats (one instance, discarded)
    probe = rolling_mpc.build_pyomo(dims)
    comp = compile_to_highs_arrays(probe)
    n_col, n_row, nnz = comp.n_col, comp.n_row, comp.nnz
    del probe, comp

    # ----- appsi persistent ------------------------------------------------- #
    m_a = rolling_mpc.build_pyomo(dims)
    appsi = _appsi_persistent(m_a)
    appsi.solve(m_a)  # first solve (set_instance + solve)
    appsi_ticks = []
    appsi_objs = []
    for r in range(rolls):
        rolling_mpc.apply_roll(m_a, np.random.default_rng(seed0 + r))
        t0 = time.perf_counter()
        appsi.solve(m_a)
        appsi_ticks.append(time.perf_counter() - t0)
        appsi_objs.append(pyo.value(m_a.obj))

    # ----- faststep, model-driven ------------------------------------------- #
    m_f = rolling_mpc.build_pyomo(dims)
    fs = FastStepHighs()
    fs.set_instance(m_f)
    fs.solve()
    fs_ticks = []
    fs_objs = []
    for r in range(rolls):
        rolling_mpc.apply_roll(m_f, np.random.default_rng(seed0 + r))
        t0 = time.perf_counter()
        res = fs.solve()
        fs_ticks.append(time.perf_counter() - t0)
        fs_objs.append(res.incumbent_objective)

    # ----- faststep, array-driven (mapping-free) ---------------------------- #
    m_g = rolling_mpc.build_pyomo(dims)
    fa = FastStepHighs()
    fa.set_instance(m_g)
    params = fa.parameters
    fa.solve()
    fa_ticks = []
    fa_objs = []
    for r in range(rolls):
        rolling_mpc.apply_roll(m_g, np.random.default_rng(seed0 + r))
        P = np.fromiter((p.value for p in params), np.float64, len(params))
        t0 = time.perf_counter()
        res = fa.solve(param_values=P)
        fa_ticks.append(time.perf_counter() - t0)
        fa_objs.append(res.incumbent_objective)

    # ----- faststep, value-guard leg (nominally-mutable static matrix) ------- #
    # The model carries a mutable matrix coefficient (eff[a,t]) that never
    # changes under the roll; the value guard accepts it, verifies it each roll,
    # and keeps the warm basis.  We also time the guard check in isolation.
    import highspy

    hinf = highspy.kHighsInf
    m_v = rolling_mpc.build_pyomo(dims, mutable_matrix=True)
    vg = FastStepHighs()
    vg.set_instance(m_v)
    vg.solve()
    guard = vg._plan.matrix_guard  # the _MatrixGuard component
    n_guarded = int(guard.affine.n)
    vg_ticks = []
    vg_objs = []
    guard_ticks = []
    for r in range(rolls):
        rolling_mpc.apply_roll(m_v, np.random.default_rng(seed0 + r))
        t0 = time.perf_counter()
        res = vg.solve()
        vg_ticks.append(time.perf_counter() - t0)
        vg_objs.append(res.incumbent_objective)
        # Isolated guard-check cost: exactly the per-roll verification work.
        P = vg._plan.read_param_vector()
        g0 = time.perf_counter()
        cur = guard.current(P, hinf)
        guard.changed_mask(cur, vg._matrix_atol, vg._matrix_rtol)
        guard_ticks.append(time.perf_counter() - g0)

    # ----- equivalence gate ------------------------------------------------- #
    max_obj_dev = 0.0
    for oa, of, og, ov in zip(appsi_objs, fs_objs, fa_objs, vg_objs):
        scale = max(1.0, abs(oa))
        max_obj_dev = max(
            max_obj_dev,
            abs(of - oa) / scale,
            abs(og - oa) / scale,
            abs(ov - oa) / scale,
        )
    equivalent = max_obj_dev < 1e-6

    appsi_ms = _median_ms(appsi_ticks)
    fs_ms = _median_ms(fs_ticks)
    fa_ms = _median_ms(fa_ticks)
    vg_ms = _median_ms(vg_ticks)
    guard_ms = _median_ms(guard_ticks)
    return {
        "size": size,
        "A": dims["A"],
        "T": dims["T"],
        "n_col": n_col,
        "n_row": n_row,
        "nnz": nnz,
        "rolls": rolls,
        "appsi_persistent_ms": appsi_ms,
        "faststep_model_ms": fs_ms,
        "faststep_array_ms": fa_ms,
        "faststep_valueguard_ms": vg_ms,
        "guard_check_ms": guard_ms,
        "guard_overhead_frac": guard_ms / vg_ms if vg_ms else None,
        "valueguard_vs_static": vg_ms / fs_ms if fs_ms else None,
        "n_guarded_coeffs": n_guarded,
        "speedup_model": appsi_ms / fs_ms if fs_ms else None,
        "speedup_array": appsi_ms / fa_ms if fa_ms else None,
        "max_obj_deviation": max_obj_dev,
        "equivalent": equivalent,
    }


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def render_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    out.append(
        "| size | A x T | nnz | APPSI-persistent | faststep (model) | "
        "faststep (array) | **speedup (model)** | speedup (array) | equiv |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['size']} | {r['A']}x{r['T']} | {r['nnz']:,} | "
            f"{r['appsi_persistent_ms']:.2f} ms | {r['faststep_model_ms']:.2f} ms | "
            f"{r['faststep_array_ms']:.2f} ms | "
            f"**{r['speedup_model']:.2f}x** | {r['speedup_array']:.2f}x | "
            f"{'yes' if r['equivalent'] else 'NO'} |"
        )
    return "\n".join(out)


def render_guard_table(rows: List[Dict[str, Any]]) -> str:
    """The value-aware static-matrix guard evidence (the new leg)."""
    out = []
    out.append(
        "| size | guarded A-coeffs | faststep (static) | faststep (value-guard) | "
        "guard-check | guard overhead | leg vs static |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['size']} | {r['n_guarded_coeffs']:,} | "
            f"{r['faststep_model_ms']:.2f} ms | "
            f"{r['faststep_valueguard_ms']:.2f} ms | "
            f"{r['guard_check_ms']:.3f} ms | "
            f"{100.0 * r['guard_overhead_frac']:.1f}% | "
            f"{r['valueguard_vs_static']:.2f}x |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="1e4,1e5")
    ap.add_argument("--rolls", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default=None, help="write results JSON here")
    a = ap.parse_args()

    sizes = [s.strip() for s in a.sizes.split(",") if s.strip()]
    rows = []
    for size in sizes:
        if size not in rolling_mpc.SIZES:
            raise SystemExit(
                f"unknown size {size!r}; choose from {list(rolling_mpc.SIZES)}"
            )
        dims = rolling_mpc.SIZES[size]
        print(f"[{size}] A={dims['A']} T={dims['T']} rolls={a.rolls} ...", flush=True)
        row = run_size(size, dims, a.rolls, a.seed)
        rows.append(row)
        print(
            f"[{size}] nnz={row['nnz']:,} | APPSI {row['appsi_persistent_ms']:.2f} ms"
            f" | faststep model {row['faststep_model_ms']:.2f} ms"
            f" ({row['speedup_model']:.2f}x) | array {row['faststep_array_ms']:.2f} ms"
            f" ({row['speedup_array']:.2f}x) | value-guard "
            f"{row['faststep_valueguard_ms']:.2f} ms (guard "
            f"{100.0 * row['guard_overhead_frac']:.1f}% of tick, "
            f"{row['valueguard_vs_static']:.2f}x static) | "
            f"equivalent={row['equivalent']}",
            flush=True,
        )

    print("\n### warm tick vs APPSI-persistent\n")
    print(render_table(rows))
    print("\n### value-aware static-matrix guard\n")
    print(render_guard_table(rows) + "\n")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"model": "rolling_mpc", "rows": rows}, fh, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
