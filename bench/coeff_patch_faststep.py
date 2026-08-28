"""Coefficient-patch benchmark: the shrinking-first-interval MPC warm tick.

A receding-horizon MPC re-solves over a sliding window of the horizon.  The
*current control interval's remaining duration* shrinks as real time advances
within it -- a mutable ``dur[t]`` on the state-of-charge recurrence's charge term,
so the shrink is a small, **sparse constraint-matrix coefficient change** (the
``A`` entries of the current interval only).  That change genuinely trips
``highs_faststep``'s value guard every tick.  Three ways to absorb it:

* **patched-faststep** (``on_matrix_change='patch'``) -- patch just the changed
  matrix entries in place with per-entry ``changeCoeff`` and slide the window mask,
  keeping the warm simplex basis.  A pure warm update: no re-compile, no basis
  reset.
* **reload-per-tick faststep** (``on_matrix_change='reload'``) -- the guard's other
  non-fatal option: rebuild the whole standard-form matrix and ``passModel`` it
  (basis reset) every tick, then re-apply the window.
* **classic APPSI structural narrow** -- build a fresh model of just the active
  window (the receding-horizon-today baseline) and cold-solve it through the
  persistent APPSI HiGHS interface.

This bench slides a fixed-width window across the horizon, shrinks the window's
first-interval duration each cycle, rolls the price/demand data, and times **one
MPC cycle** each way at a day-length horizon (``day288``, T=288) and the ``1e5``
nonzero class.  It also isolates the **per-tick cost of the patch itself** (the
guard re-evaluation + the ``changeCoeff`` loop over the changed entries).  Every
cycle's window objective is checked equal across the three routes (the coefficient
-patch equivalence gate; the masked route's reported objective includes the fixed
-term constant, which :meth:`FastStepHighs.masked_objective_constant` subtracts).

The model is a generic multi-asset energy MPC (no application-specific structure);
the shrinking-first-interval pattern is the public stand-in for the load-bound
receding-horizon workload the patch path was built for.

Usage::

    bench/.venv/bin/python -m bench.coeff_patch_faststep --sizes day288,1e5 --cycles 24
    bench/.venv/bin/python -m bench.coeff_patch_faststep --sizes day288 --cycles 30 \
        --out bench/results/coeff_patch_faststep.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List, Tuple

import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.vector.faststep import FastStepHighs
from pyomo.contrib.vector.fastload import compile_to_highs_arrays

EFF = 0.95
SOC_MAX = 60.0
DT = 0.25  # nominal interval duration (hours)

SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"A": 3, "T": 40, "W": 8},
    "day288": {"A": 6, "T": 288, "W": 48},  # ~a day at 5-min steps
    "1e5": {"A": 40, "T": 640, "W": 64},  # ~1.0e5 nonzeros
}

BOUNDARY_SOC = 0.5 * SOC_MAX
OUT_P = 0.0
OUT_SOC = 0.0


def _median_ms(xs: List[float]) -> float:
    return statistics.median(xs) * 1e3


# --------------------------------------------------------------------------- #
# the shrinking-first-interval MPC model (mutable per-interval matrix duration)
# --------------------------------------------------------------------------- #
def build_model(A: int, T: int, durs=None) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.A = pyo.RangeSet(0, A - 1)
    m.T = pyo.RangeSet(0, T - 1)
    if durs is None:
        durs = {t: DT for t in range(T)}
    m.dur = pyo.Param(m.T, initialize=durs, mutable=True)  # matrix coefficient
    m.price = pyo.Param(m.T, initialize={t: 1.0 for t in range(T)}, mutable=True)
    m.gcap = pyo.Param(m.T, initialize={t: 4.0 * A for t in range(T)}, mutable=True)
    m.dem = pyo.Param(
        m.A,
        m.T,
        initialize={(a, t): 0.5 for a in range(A) for t in range(T)},
        mutable=True,
    )
    m.pmax = pyo.Param(
        m.A,
        m.T,
        initialize={(a, t): 5.0 for a in range(A) for t in range(T)},
        mutable=True,
    )
    m.p = pyo.Var(m.A, m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.A, m.T, bounds=(0.0, SOC_MAX))
    for a in range(A):
        for t in range(T):
            m.p[a, t].setub(m.pmax[a, t])

    def socrule(mm, a, t):
        prev = 0.0 if t == 0 else mm.soc[a, t - 1]
        return mm.soc[a, t] == prev + EFF * mm.dur[t] * mm.p[a, t] - mm.dem[a, t]

    m.socc = pyo.Constraint(m.A, m.T, rule=socrule)
    m.grid = pyo.Constraint(
        m.T, rule=lambda mm, t: sum(mm.p[a, t] for a in mm.A) <= mm.gcap[t]
    )
    m.obj = pyo.Objective(
        expr=sum(m.price[t] * m.p[a, t] for a in range(A) for t in range(T))
        + 0.01 * sum(m.soc[a, t] for a in range(A) for t in range(T)),
        sense=pyo.minimize,
    )
    return m


def apply_roll(m, A, T, rng) -> None:
    price = rng.uniform(0.5, 3.0, size=T)
    gcap = 4.0 * A * rng.uniform(0.7, 1.0, size=T)
    dem = rng.uniform(0.0, 1.0, size=(A, T))
    for t in range(T):
        m.price[t] = float(price[t])
        m.gcap[t] = float(gcap[t])
    for a in range(A):
        for t in range(T):
            m.dem[a, t] = float(dem[a, t])


# --------------------------------------------------------------------------- #
# window bookkeeping (static: computed once from the compiled mapping)
# --------------------------------------------------------------------------- #
def _index_maps(stepper, m, A, T):
    socc_row = {}
    grid_row = {}
    for t in range(T):
        (grid_row[t],) = stepper.row_indices(m.grid[t])
        for a in range(A):
            (socc_row[a, t],) = stepper.row_indices(m.socc[a, t])
    p_col = {
        (a, t): stepper.column_index(m.p[a, t]) for a in range(A) for t in range(T)
    }
    soc_col = {
        (a, t): stepper.column_index(m.soc[a, t]) for a in range(A) for t in range(T)
    }
    return socc_row, grid_row, p_col, soc_col


def _precompute_windows(stepper, m, A, T, W):
    n_row, n_col = stepper._compiled.n_row, stepper._compiled.n_col
    socc_row, grid_row, p_col, soc_col = _index_maps(stepper, m, A, T)
    windows = {}
    for a in range(1, T - W + 1):
        b = a + W
        in_win = set(range(a, b))
        active = np.zeros(n_row, dtype=bool)
        fixed = np.ones(n_col, dtype=bool)
        values = np.empty(n_col, dtype=np.float64)
        for asset in range(A):
            for t in range(T):
                values[p_col[asset, t]] = OUT_P
                values[soc_col[asset, t]] = OUT_SOC
        for t in range(T):
            if t in in_win:
                active[grid_row[t]] = True
                for asset in range(A):
                    active[socc_row[asset, t]] = True
                    fixed[p_col[asset, t]] = False
                    fixed[soc_col[asset, t]] = False
        for asset in range(A):
            values[soc_col[asset, a - 1]] = BOUNDARY_SOC
        windows[a] = (active, fixed, values)
    return windows, socc_row


# --------------------------------------------------------------------------- #
# the classic structural-narrowing route (fresh narrowed model + cold solve)
# --------------------------------------------------------------------------- #
def _build_narrowed(m, A, a, b) -> pyo.ConcreteModel:
    price = {t: pyo.value(m.price[t]) for t in range(a, b)}
    gcap = {t: pyo.value(m.gcap[t]) for t in range(a, b)}
    dur = {t: pyo.value(m.dur[t]) for t in range(a, b)}
    dem = {
        (asset, t): pyo.value(m.dem[asset, t])
        for asset in range(A)
        for t in range(a, b)
    }
    pmax = {
        (asset, t): pyo.value(m.pmax[asset, t])
        for asset in range(A)
        for t in range(a, b)
    }
    n = pyo.ConcreteModel()
    n.A = pyo.RangeSet(0, A - 1)
    n.W = pyo.RangeSet(a, b - 1)
    n.p = pyo.Var(n.A, n.W, domain=pyo.NonNegativeReals)
    n.soc = pyo.Var(n.A, n.W, bounds=(0.0, SOC_MAX))
    for asset in range(A):
        for t in range(a, b):
            n.p[asset, t].setub(pmax[asset, t])

    def socrule(nn, asset, t):
        prev = BOUNDARY_SOC if t == a else nn.soc[asset, t - 1]
        return nn.soc[asset, t] == prev + EFF * dur[t] * nn.p[asset, t] - dem[asset, t]

    n.socc = pyo.Constraint(n.A, n.W, rule=socrule)
    n.grid = pyo.Constraint(
        n.W, rule=lambda nn, t: sum(nn.p[asset, t] for asset in nn.A) <= gcap[t]
    )
    n.obj = pyo.Objective(
        expr=sum(
            price[t] * n.p[asset, t] + 0.01 * n.soc[asset, t]
            for asset in range(A)
            for t in range(a, b)
        ),
        sense=pyo.minimize,
    )
    return n


def _appsi_persistent():
    from pyomo.contrib.appsi.solvers import Highs as AppsiHighs

    opt = AppsiHighs()
    opt.config.load_solution = True
    return opt


# --------------------------------------------------------------------------- #
# isolated per-tick patch cost (guard re-eval + changeCoeff over changed entries)
# --------------------------------------------------------------------------- #
def _measure_patch_apply(fs, m, a0, A, reps=200) -> Tuple[float, float, int]:
    """Break the per-tick coefficient work into its two halves, timed separately:

    * **detect** -- read ``P``, re-evaluate the matrix guard (one vectorized
      ``M @ P``), and compare to the loaded baseline.  This is the value guard's
      pre-existing per-roll cost, paid whether or not anything changed.
    * **apply** -- the patch itself: the ``changeCoeff`` loop over just the changed
      entries (the target: microseconds-class for a handful).

    Returns ``(detect_seconds, apply_seconds, n_entries)``, medians, no solver run.
    """
    import highspy

    hinf = highspy.kHighsInf
    guard = fs._plan.matrix_guard
    saved = guard.baseline.copy()
    m.dur[a0] = 0.5 * DT
    P = fs._plan.read_param_vector()
    current = guard.current(P, hinf)
    changed = guard.changed_mask(current, 0.0, 0.0)
    n_changed = int(changed.sum())
    detect_ts, apply_ts = [], []
    for k in range(reps):
        m.dur[a0] = (0.3 + 0.6 * (k % 2)) * DT  # alternate to force a real change
        t0 = time.perf_counter()
        P = fs._plan.read_param_vector()
        current = guard.current(P, hinf)
        changed = guard.changed_mask(current, 0.0, 0.0)
        t1 = time.perf_counter()
        fs._patch_matrix(fs._highs, guard, changed, current)
        t2 = time.perf_counter()
        detect_ts.append(t1 - t0)
        apply_ts.append(t2 - t1)
    guard.baseline = saved  # restore so the caller's state is untouched
    m.dur[a0] = DT
    return statistics.median(detect_ts), statistics.median(apply_ts), n_changed


# --------------------------------------------------------------------------- #
# one size
# --------------------------------------------------------------------------- #
def run_size(
    size: str, dims: Dict[str, Any], cycles: int, seed0: int
) -> Dict[str, Any]:
    A, T, W = int(dims["A"]), int(dims["T"]), int(dims["W"])

    probe = build_model(A, T)
    comp = compile_to_highs_arrays(probe)
    n_col, n_row, nnz = comp.n_col, comp.n_row, comp.nnz
    del probe, comp

    # ----- patched-faststep: one live stepper, patch coefficients in place --- #
    m_p = build_model(A, T)
    fs_patch = FastStepHighs(on_matrix_change="patch")
    fs_patch.set_instance(m_p)
    windows, _socc_row = _precompute_windows(fs_patch, m_p, A, T, W)
    starts = sorted(windows)
    fs_patch.solve()

    # ----- reload-per-tick faststep: rebuild the whole matrix each tick ------ #
    m_r = build_model(A, T)
    fs_reload = FastStepHighs(on_matrix_change="reload")
    fs_reload.set_instance(m_r)
    windows_r, _ = _precompute_windows(fs_reload, m_r, A, T, W)
    fs_reload.solve()

    patch_ticks: List[float] = []
    reload_ticks: List[float] = []
    appsi_ticks: List[float] = []
    max_obj_dev = 0.0
    n_feas = 0

    for c in range(cycles):
        a = starts[c % len(starts)]
        b = a + W
        rng = np.random.default_rng(seed0 + c)
        shrink = float(np.random.default_rng(1000 + c).uniform(0.3, 0.95))

        # ---- patched-faststep cycle: shrink dur[a], roll, slide mask, solve -- #
        apply_roll(m_p, A, T, rng)
        m_p.dur[a] = shrink * DT
        active, fixed, values = windows[a]
        t0 = time.perf_counter()
        fs_patch.set_window(active_rows=active, fixed_cols=fixed, fixed_values=values)
        res_p = fs_patch.solve(raise_on_nonoptimal=False)
        patch_ticks.append(time.perf_counter() - t0)
        patch_obj = (
            None
            if res_p.incumbent_objective is None
            else res_p.incumbent_objective - fs_patch.masked_objective_constant()
        )

        # ---- reload-per-tick faststep cycle (same data) ---------------------- #
        apply_roll(m_r, A, T, np.random.default_rng(seed0 + c))
        m_r.dur[a] = shrink * DT
        active_r, fixed_r, values_r = windows_r[a]
        t0 = time.perf_counter()
        fs_reload.set_window(
            active_rows=active_r, fixed_cols=fixed_r, fixed_values=values_r
        )
        res_r = fs_reload.solve(raise_on_nonoptimal=False)
        reload_ticks.append(time.perf_counter() - t0)
        reload_obj = (
            None
            if res_r.incumbent_objective is None
            else res_r.incumbent_objective - fs_reload.masked_objective_constant()
        )

        # ---- classic APPSI structural narrow (fresh narrowed model) ---------- #
        appsi = _appsi_persistent()
        t0 = time.perf_counter()
        nm = _build_narrowed(m_p, A, a, b)
        try:
            appsi.solve(nm)
            appsi_obj = pyo.value(nm.obj)
        except Exception:
            appsi_obj = None
        appsi_ticks.append(time.perf_counter() - t0)

        # ---- equivalence gate on the window objective ------------------------ #
        if patch_obj is not None and appsi_obj is not None:
            n_feas += 1
            scale = max(1.0, abs(appsi_obj))
            max_obj_dev = max(max_obj_dev, abs(patch_obj - appsi_obj) / scale)
            if reload_obj is not None:
                max_obj_dev = max(max_obj_dev, abs(patch_obj - reload_obj) / scale)

    # ---- isolated per-tick patch cost (detect vs apply) -------------------- #
    detect_s, patch_apply_s, n_patched = _measure_patch_apply(
        fs_patch, m_p, starts[0], A
    )

    patch_ms = _median_ms(patch_ticks)
    reload_ms = _median_ms(reload_ticks)
    appsi_ms = _median_ms(appsi_ticks)
    return {
        "size": size,
        "A": A,
        "T": T,
        "W": W,
        "n_col": n_col,
        "n_row": n_row,
        "nnz": nnz,
        "cycles": cycles,
        "n_feasible_cycles": n_feas,
        "entries_patched_per_tick": n_patched,
        "guard_detect_us": detect_s * 1e6,
        "patch_apply_us": patch_apply_s * 1e6,
        "patched_faststep_ms": patch_ms,
        "reload_faststep_ms": reload_ms,
        "classic_appsi_ms": appsi_ms,
        "speedup_vs_reload": reload_ms / patch_ms if patch_ms else None,
        "speedup_vs_appsi": appsi_ms / patch_ms if patch_ms else None,
        "max_window_obj_deviation": max_obj_dev,
        "equivalent": max_obj_dev < 1e-6,
    }


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def render_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    out.append(
        "| size | A x T (win W) | nnz | entries/tick | patch apply | guard detect | "
        "classic APPSI | reload/tick | **patched warm** | **vs reload** | "
        "vs APPSI | equiv |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['size']} | {r['A']}x{r['T']} (W={r['W']}) | {r['nnz']:,} | "
            f"{r['entries_patched_per_tick']} | {r['patch_apply_us']:.1f} us | "
            f"{r['guard_detect_us']:.1f} us | "
            f"{r['classic_appsi_ms']:.2f} ms | {r['reload_faststep_ms']:.2f} ms | "
            f"{r['patched_faststep_ms']:.2f} ms | "
            f"**{r['speedup_vs_reload']:.2f}x** | {r['speedup_vs_appsi']:.2f}x | "
            f"{'yes' if r['equivalent'] else 'NO'} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="day288,1e5")
    ap.add_argument("--cycles", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--out", default=None, help="write results JSON here")
    a = ap.parse_args()

    sizes = [s.strip() for s in a.sizes.split(",") if s.strip()]
    rows = []
    for size in sizes:
        if size not in SIZES:
            raise SystemExit(f"unknown size {size!r}; choose from {list(SIZES)}")
        dims = SIZES[size]
        print(
            f"[{size}] A={dims['A']} T={dims['T']} W={dims['W']} cycles={a.cycles} ...",
            flush=True,
        )
        row = run_size(size, dims, a.cycles, a.seed)
        rows.append(row)
        print(
            f"[{size}] nnz={row['nnz']:,} | entries/tick={row['entries_patched_per_tick']} "
            f"| patch-apply {row['patch_apply_us']:.1f} us (detect "
            f"{row['guard_detect_us']:.1f} us) | classic-APPSI "
            f"{row['classic_appsi_ms']:.2f} ms | reload/tick "
            f"{row['reload_faststep_ms']:.2f} ms | patched-warm "
            f"{row['patched_faststep_ms']:.2f} ms "
            f"({row['speedup_vs_reload']:.2f}x reload, "
            f"{row['speedup_vs_appsi']:.2f}x APPSI) | "
            f"equivalent={row['equivalent']} "
            f"(feasible {row['n_feasible_cycles']}/{row['cycles']})",
            flush=True,
        )

    print("\n### Coefficient-patch MPC cycle: shrinking-first-interval warm re-solve\n")
    print(render_table(rows) + "\n")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"model": "coeff_patch_mpc", "rows": rows}, fh, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
