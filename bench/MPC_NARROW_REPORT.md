# `highs_faststep` — MPC-narrowing report (masked warm re-solve)

**Project:** Vectorized Model Construction for Pyomo (see the Phase-4 report for the base warm interface)
**Feature:** masked warm updates on `FastStepHighs` — row masks + variable fixes that let a rolling-horizon MPC **narrow its active window** without a structural change
**Baseline:** the classic structural-narrowing route — build a fresh model of just the active window each cycle and solve it cold (a persistent APPSI HiGHS solve, and, as a faster second reference, a `highs_fastload` compile+solve).

## Environment

- Pyomo `6.10.2.dev0` (this worktree's branch)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- Reproducible with `bench/mpc_narrow_faststep.py` (§ Reproducing).

---

## 1. The problem

A receding-horizon MPC re-solves the same model over a **sliding window** of the
full horizon. Each cycle it activates only a window `[a, b)`:

- the in-window rows (recurrences + couplings at those time steps), and
- the in-window decision variables,

with the **boundary state** (`soc[·, a-1]`, the recurrence's initial condition at
the window's left edge) held at the current measured value.

Done the obvious way, narrowing is a **structural change**: a different, smaller
matrix each cycle. So it re-pays Pyomo construction, a fresh compile /
`set_instance`, and a cold solve — and it is exactly the kind of row/column change
that `highs_faststep`'s structure fingerprint rightly rejects as a warm update.

## 2. The mechanism — narrow by masking, not by rebuilding

Keep the **full** compiled matrix loaded in one live `FastStepHighs`, and between
solves push a small overlay:

- **row masks** — relax every out-of-window row to `-inf ≤ · ≤ +inf` (a free row
  imposes nothing), the warm-basis-friendly equivalent of the narrowed problem
  dropping the row (`changeRowsBounds`);
- **variable fixes** — pin every out-of-window variable to its boundary value via
  equal bounds `(v, v)` (`changeColsBounds`).

```python
stepper.set_window(active_rows=mask, fixed_cols=fixed, fixed_values=values)
res = stepper.solve()                          # warm re-solve on the kept basis
window_obj = res.incumbent_objective - stepper.masked_objective_constant()
```

The matrix — and thus the structure fingerprint, the value guard, and the fold
set — is **untouched**, so the whole update rides the warm path. On the active
window this is provably the *same problem* as the structural narrow: an in-window
recurrence row that references a **fixed** out-of-window variable becomes the
correct boundary condition (the fixed value moves to the row's RHS). See §4.

## 3. Results — one MPC-narrowing cycle, masked vs a fresh structural narrow

Each cycle rolls the [`rolling_mpc`](models/rolling_mpc.py) data forward, slides a
fixed-width window across the horizon, and times **one narrow+solve cycle** each
way. `day288` is a day-length horizon at 5-minute resolution (T=288); `1e5` is the
~10⁵-nonzero class. Median over the cycles.

| size | A × T (win W) | nnz | classic narrow (APPSI) | classic narrow (fastload) | **masked warm** | **speedup vs APPSI** | vs fastload | equiv |
|---|---|---|---|---|---|---|---|---|
| xs | 3×40 (W=8) | 477 | 3.80 ms | 2.21 ms | **0.93 ms** | **4.08×** | 2.37× | yes |
| day288 | 6×288 (W=48) | 6,906 | 26.28 ms | 9.42 ms | **3.25 ms** | **8.09×** | 2.90× | yes |
| 1e5 | 40×640 (W=64) | 102,360 | 240.64 ms | 70.25 ms | **34.78 ms** | **6.92×** | 2.02× | yes |

### Reading the numbers

- **vs the classic APPSI narrow (what a receding horizon does today):** masked
  warm is **4–8×** faster across the range. The classic route re-pays Pyomo
  construction + a fresh compile + a cold solve every cycle; the masked route
  pushes a bounds-only overlay and re-solves on the retained basis.
- **vs the fastest classic route (a fresh `highs_fastload` compile+solve):** still
  **2.0–2.9×** faster. This is the honest floor — `highs_fastload` already bulk-
  loads the narrowed model in one `passModel`, so the remaining win is purely the
  construction + cold-solve the warm path avoids.
- **At `1e5` the masked solve works the *full* matrix** (the out-of-window rows
  are relaxed, not removed), yet it still wins **6.92×** / **2.02×** — the retained
  basis makes the mostly-relaxed re-solve cheap, and it never re-pays construction.
  The margin narrows against `fastload` as the window solve shrinks relative to the
  full matrix, exactly as expected; reported so the tradeoff is visible.
- **The window objective matches every cycle** (`equivalent=yes`, all cycles
  feasible): masked-warm `window_obj` (reported objective minus the fixed-term
  constant) equals the freshly-narrowed model's objective to solver tolerance.

## 4. Why the window solution is exact (the correctness claim)

Partition the full variables into in-window `x_W` and out-of-window `x_O`, and the
rows into in-window `R_W` (kept active) and out-of-window `R_O` (relaxed). Fixing
`x_O = x̄_O` and relaxing `R_O` reduces the full problem, on the window, to

```
min  c_W·x_W                          (+ constant c_O·x̄_O)
s.t. A_WW x_W ∈ [l_W − A_WO x̄_O , u_W − A_WO x̄_O]     (R_W with the fixed cols on the RHS)
     x_W ∈ [xl_W, xu_W]
```

which is **exactly** the structurally-narrowed problem with boundary condition
`x̄_O` — for *any* `x̄_O`, because the relaxed rows `R_O` appear in neither problem.
So the masked-warm and structurally-narrowed problems have identical feasible
regions and objectives over `x_W`: same status, same optimum, and identical
in-window variable values wherever the optimum is unique. The only difference is a
constant: the masked solve reports the full-model objective, which includes
`c_O·x̄_O` (every fixed variable keeps its cost). `masked_objective_constant()`
returns that constant, so `reported − constant` is the pure window objective — the
convention the equivalence gate uses.

This is proven, not assumed: `pyomo/contrib/vector/tests/test_faststep_masked.py`
gates masked-warm narrowing against a **fresh, independently built** narrowed model
(solved through `highs_fastload`) across randomized models × randomized windows
(LP and MIP), the boundary-coupling left-edge case, and the degenerate windows
(empty, full, single row) — objective, status, and in-window values all within
solver tolerance.

## 5. Scope

- **Masks and fixes only.** This feature adds *row masks* and *variable fixes* as
  warm updates — no general structural-delta support (adding/removing rows or
  columns, changing a coefficient). Those remain a structure change and are still
  rejected by the fingerprint.
- **Adapter-friendly.** The window API takes plain arrays — `set_window(active_rows,
  fixed_cols, fixed_values)` (boolean masks + a fix-value array), or the granular
  `deactivate_rows` / `activate_rows` / `fix_variables` / `unfix_variables`. The
  window masks depend only on the position and the static row/column mapping (via
  `row_indices` / `column_index`), so an adapter precomputes them once and drives
  each cycle with array writes — no Pyomo mutation on the hot path.
- **Composes with everything.** Masking leaves the matrix (and the fingerprint,
  the value guard, and the fold set) untouched, and composes with the templated
  data roll, the array-driven (`solve(param_values=…)`) path, and the
  `on_matrix_change='reload'` rebuild (the overlay survives a reload).

## 6. Reproducing

```bash
bench/.venv/bin/python -m bench.mpc_narrow_faststep --sizes xs,day288,1e5 --cycles 20 \
    --out bench/results/mpc_narrow_faststep.json
```

`--sizes` selects from `xs` / `day288` / `1e5`; `--cycles` sets the number of
roll+narrow cycles the median is taken over. The committed JSON in
`bench/results/mpc_narrow_faststep.json` is a 20-cycle run on the environment above.

## 7. Deferrals

- **General structural deltas** (rows/columns genuinely added or removed between
  solves, or a matrix coefficient edited in place) are out of scope — masks and
  fixes only. Those stay a fresh-`set_instance` rebuild.
- **A shape-changing reload with a live overlay** drops the overlay with a warning
  (masks are index-addressed and cannot be remapped onto a different shape); the
  same-shape reload preserves it. Combining `on_matrix_change='reload'` with a
  window is otherwise fully supported.
- **Wiring a specific horizon mechanism** to this API (driving the window from an
  external controller's arrays and measuring the full real update+solve cycle) is a
  follow-up; the array-first API is shaped for exactly that adapter.
