"""MPC-narrowing benchmark: masked warm re-solve vs a fresh structural narrow.

A rolling-horizon / receding-horizon MPC re-solves the same model over a *sliding
window* of the full horizon.  Two ways to do the per-cycle narrow:

* **classic structural narrow** -- build a fresh model of just the active window
  (only the in-window variables and rows, the boundary state a fixed initial
  condition) and solve it from scratch.  Every cycle pays Pyomo construction +
  a fresh compile/``set_instance`` + a cold solve.  This is what a receding
  horizon does today, and it is a *structural* change (different rows/columns
  each cycle) -- which is exactly why ``highs_faststep``'s structure fingerprint
  rejects it as a warm update.

* **masked warm narrow** -- keep the full compiled matrix loaded in a live
  ``FastStepHighs`` and, between solves, *mask* the out-of-window rows (relax them
  to free) and *fix* the out-of-window variables to the boundary state.  On the
  active window this is provably the same problem as the structural narrow (an
  in-window recurrence row that references a fixed out-of-window variable becomes
  the correct boundary condition), but it is a pure *data* update: the matrix and
  the fingerprint are untouched, so it rides the warm path with the basis kept.

This bench rolls the synthetic :mod:`bench.models.rolling_mpc` forward, slides a
fixed-width window across the horizon, and times **one MPC-narrowing cycle** each
way -- ``mask/fix + warm solve`` vs ``build narrowed model + fresh solve`` -- at a
day-length horizon (``day288``, T=288) and at the ``1e5``-nonzero class.  Every
cycle's *window* objective is checked equal across the two routes (the narrowing
equivalence gate; the masked route's reported objective includes the fixed-term
constant, which :meth:`FastStepHighs.masked_objective_constant` subtracts back).

Usage::

    bench/.venv/bin/python -m bench.mpc_narrow_faststep --sizes day288,1e5 --cycles 24
    bench/.venv/bin/python -m bench.mpc_narrow_faststep --sizes day288 --cycles 30 \
        --out bench/results/mpc_narrow_faststep.json
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

from bench.models import rolling_mpc

# Day-length horizon at 5-minute resolution (24*60/5 = 288 intervals); the
# receding-horizon shape the private load-bound case lives in.  Reuse the
# rolling_mpc structure, adding a day-class size next to the 1e5 nonzero class.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"A": 3, "T": 40, "W": 8},
    "day288": {"A": 6, "T": 288, "W": 48},  # ~6.9e3 nnz, a day at 5-min steps
    "1e5": {"A": 40, "T": 640, "W": 64},  # ~1.0e5 nnz
}

# The boundary (initial-condition) state the window is anchored on -- a nominal
# mid-charge state.  Both routes use the same value so the window problems match.
BOUNDARY_SOC = 0.5 * rolling_mpc.SOC_MAX
# Out-of-window variables that do not couple into the window (fixed to a benign
# value; they only shift the reported objective by a constant we subtract back).
OUT_P = 0.0
OUT_SOC = 0.0


def _median_ms(xs: List[float]) -> float:
    return statistics.median(xs) * 1e3


# --------------------------------------------------------------------------- #
# window bookkeeping (data-independent: computed once from the static mapping)
# --------------------------------------------------------------------------- #
def _index_maps(stepper, m, A, T):
    """``socc[a,t] -> row``, ``grid[t] -> row``, ``p[a,t] -> col``,
    ``soc[a,t] -> col`` for the whole horizon (one pass)."""
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


def _precompute_windows(stepper, m, A, T, W) -> Dict[int, Tuple[Any, Any, Any]]:
    """For every window start ``a`` in ``[1, T-W]`` (a>=1 so there is a real
    boundary), the ``(active_rows, fixed_cols, fixed_values)`` arrays the masked
    route feeds to :meth:`FastStepHighs.set_window`.  These depend only on the
    window position and the static row/column mapping, never on the rolling data,
    so an adapter precomputes them once -- as we do here."""
    n_row, n_col = stepper._compiled.n_row, stepper._compiled.n_col
    socc_row, grid_row, p_col, soc_col = _index_maps(stepper, m, A, T)
    windows = {}
    for a in range(1, T - W + 1):
        b = a + W
        in_win = set(range(a, b))
        active = np.zeros(n_row, dtype=bool)
        fixed = np.ones(n_col, dtype=bool)
        values = np.empty(n_col, dtype=np.float64)
        # default fixed value for every column (overwritten for the boundary).
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
        # boundary state: soc[asset, a-1] is referenced by the in-window
        # recurrence at t=a; pin it to the anchored initial condition.
        for asset in range(A):
            values[soc_col[asset, a - 1]] = BOUNDARY_SOC
        windows[a] = (active, fixed, values)
    return windows


# --------------------------------------------------------------------------- #
# the classic structural-narrowing route (fresh narrowed model, fresh solve)
# --------------------------------------------------------------------------- #
def _build_narrowed(m, A, a, b) -> pyo.ConcreteModel:
    """A fresh model of the window [a,b): only the in-window variables and rows,
    the boundary soc[*,a-1] a fixed initial condition.  Reads the CURRENT rolling
    data (Param values) off the full model ``m`` -- the per-cycle construction the
    classic route pays."""
    price = {t: pyo.value(m.price[t]) for t in range(a, b)}
    gcap = {t: pyo.value(m.gcap[t]) for t in range(a, b)}
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
    n.soc = pyo.Var(n.A, n.W, bounds=(0.0, rolling_mpc.SOC_MAX))
    for asset in range(A):
        for t in range(a, b):
            n.p[asset, t].setub(pmax[asset, t])

    eff = rolling_mpc.EFF

    def socrule(nn, asset, t):
        prev = BOUNDARY_SOC if t == a else nn.soc[asset, t - 1]
        return nn.soc[asset, t] == prev + eff * nn.p[asset, t] - dem[asset, t]

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
# one size
# --------------------------------------------------------------------------- #
def run_size(
    size: str, dims: Dict[str, Any], cycles: int, seed0: int
) -> Dict[str, Any]:
    A, T, W = int(dims["A"]), int(dims["T"]), int(dims["W"])

    probe = rolling_mpc.build_pyomo(dims)
    comp = compile_to_highs_arrays(probe)
    n_col, n_row, nnz = comp.n_col, comp.n_row, comp.nnz
    del probe, comp

    # ----- the masked warm route: one live FastStepHighs, full matrix ------- #
    m = rolling_mpc.build_pyomo(dims)
    fs = FastStepHighs()
    fs.set_instance(m)
    windows = _precompute_windows(fs, m, A, T, W)
    starts = sorted(windows)  # window start positions to cycle through
    fs.solve()  # prime

    from pyomo.contrib.solver.common.factory import SolverFactory

    fastload = SolverFactory("highs_fastload")

    masked_ticks: List[float] = []
    classic_appsi_ticks: List[float] = []
    classic_fastload_ticks: List[float] = []
    max_obj_dev = 0.0
    n_feas = 0

    for c in range(cycles):
        rolling_mpc.apply_roll(m, np.random.default_rng(seed0 + c))
        a = starts[c % len(starts)]
        b = a + W
        active, fixed, values = windows[a]

        # masked warm narrow: set the window + warm solve (timed together).
        t0 = time.perf_counter()
        fs.set_window(active_rows=active, fixed_cols=fixed, fixed_values=values)
        res = fs.solve(raise_on_nonoptimal=False)
        masked_ticks.append(time.perf_counter() - t0)
        masked_window_obj = (
            None
            if res.incumbent_objective is None
            else res.incumbent_objective - fs.masked_objective_constant()
        )

        # classic structural narrow (fresh APPSI-style narrowed solve): build the
        # narrowed model + cold solve, timed together.
        appsi = _appsi_persistent()
        t0 = time.perf_counter()
        nm = _build_narrowed(m, A, a, b)
        try:
            ra = appsi.solve(nm)
            appsi_obj = pyo.value(nm.obj)
        except Exception:
            appsi_obj = None
        classic_appsi_ticks.append(time.perf_counter() - t0)

        # a second classic baseline: fresh fastload compile + solve.
        t0 = time.perf_counter()
        nm2 = _build_narrowed(m, A, a, b)
        rf = fastload.solve(nm2, raise_exception_on_nonoptimal_result=False)
        classic_fastload_ticks.append(time.perf_counter() - t0)

        # equivalence gate on the window objective (when both feasible).
        if masked_window_obj is not None and appsi_obj is not None:
            n_feas += 1
            scale = max(1.0, abs(appsi_obj))
            max_obj_dev = max(max_obj_dev, abs(masked_window_obj - appsi_obj) / scale)

    masked_ms = _median_ms(masked_ticks)
    classic_appsi_ms = _median_ms(classic_appsi_ticks)
    classic_fastload_ms = _median_ms(classic_fastload_ticks)
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
        "masked_warm_ms": masked_ms,
        "classic_appsi_ms": classic_appsi_ms,
        "classic_fastload_ms": classic_fastload_ms,
        "speedup_vs_appsi": classic_appsi_ms / masked_ms if masked_ms else None,
        "speedup_vs_fastload": classic_fastload_ms / masked_ms if masked_ms else None,
        "max_window_obj_deviation": max_obj_dev,
        "equivalent": max_obj_dev < 1e-6,
    }


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def render_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    out.append(
        "| size | A x T (win W) | nnz | classic narrow (APPSI) | "
        "classic narrow (fastload) | masked warm | **speedup vs APPSI** | "
        "vs fastload | equiv |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['size']} | {r['A']}x{r['T']} (W={r['W']}) | {r['nnz']:,} | "
            f"{r['classic_appsi_ms']:.2f} ms | {r['classic_fastload_ms']:.2f} ms | "
            f"{r['masked_warm_ms']:.2f} ms | "
            f"**{r['speedup_vs_appsi']:.2f}x** | {r['speedup_vs_fastload']:.2f}x | "
            f"{'yes' if r['equivalent'] else 'NO'} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="day288,1e5")
    ap.add_argument("--cycles", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2000)
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
            f"[{size}] nnz={row['nnz']:,} | classic-APPSI "
            f"{row['classic_appsi_ms']:.2f} ms | classic-fastload "
            f"{row['classic_fastload_ms']:.2f} ms | masked-warm "
            f"{row['masked_warm_ms']:.2f} ms "
            f"({row['speedup_vs_appsi']:.2f}x APPSI, "
            f"{row['speedup_vs_fastload']:.2f}x fastload) | "
            f"equivalent={row['equivalent']} "
            f"(feasible {row['n_feasible_cycles']}/{row['cycles']})",
            flush=True,
        )

    print(
        "\n### MPC-narrowing cycle: masked warm re-solve vs a fresh structural narrow\n"
    )
    print(render_table(rows) + "\n")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"model": "rolling_mpc_narrow", "rows": rows}, fh, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
