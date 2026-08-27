# Pyomo Vectorized Construction — Phase 4 Report

**Project:** Vectorized Model Construction for Pyomo (see the project's *Vectorized Model Construction for Pyomo* scoping document and the Phase-0/2/3 reports)
**Phase:** 4 — array-native persistent **warm re-solve** for the rolling-horizon path
**Deliverable:** `highs_faststep` (`pyomo.contrib.vector.FastStepHighs`), a persistent HiGHS interface that compiles a classic linear model once, retains the live solver, and re-solves each roll by pushing the changed objective coefficients / RHS / bounds as **vectorized arrays** through HiGHS's batch APIs — keeping the warm basis.
**Baseline:** the persistent APPSI HiGHS interface (`pyomo.contrib.appsi.solvers.highs`), configured for its fastest warm path.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- Reproducible with `bench/warm_faststep.py` (§ Reproducing).

---

## 1. The insight, and what Phase 4 delivered

Phases 2–3 cut the **cold** build+solve of a classic linear model (the once-per-process stage) by replacing the persistent interface's per-row `set_instance` load with one bulk `passModel`. They **explicitly deferred the warm path** (`PHASE2 §5`: "no incremental persistent-solver path").

But the workloads this project targets — model-predictive control, rolling / receding-horizon optimization — do a cold build **once** and then **re-solve thousands of times** with slightly changed data as the horizon rolls forward. That warm rolling loop is ~100% of cumulative production compute, and it was untouched.

On that path the persistent APPSI HiGHS interface re-pushes every changed coefficient to the solver **one scalar Python call at a time**, after re-evaluating it with a **per-coefficient expression walk** — `value(expr)` then `changeColCost` / `changeCoeff` / `changeRowBounds`, once per coefficient, with no dirty check (`appsi/solvers/highs.py:108-147,583-593`). On a model where a roll touches every objective price and every RHS forecast, this evaluate-and-push loop is the single largest cost of the warm tick, and it splits **roughly evenly** between the value walks and the scalar solver calls — so batching only the solver side is not enough.

**Phase 4 attacks both halves.** `highs_faststep`:

```python
from pyomo.contrib.vector import FastStepHighs

stepper = FastStepHighs()
stepper.set_instance(model)      # compile once + passModel + build affine templates
stepper.solve()                  # first solve

for roll in horizon:
    # mutate the model's mutable Params / Var bounds in place, as usual ...
    stepper.solve()              # read P, M @ P, batch push, warm re-solve
```

1. **Compile once, retain the solver.** `set_instance` reuses the Phase-2
   `compile_to_highs_arrays` (linear-only, fail-loud) for the matrix + column/row
   identity and hands it to a retained `highspy.Highs` via `passModel`.
2. **Vectorized evaluation.** Each mutable objective coefficient / row bound /
   variable bound is captured at `set_instance` as an **affine template over the
   model's parameter vector** — `values = M @ P + shift`, with `M` a constant
   sparse map and `P` the current mutable-`Param` values. A roll reads the
   *small* vector `P` once and expands **every** changed coefficient with one
   sparse `M @ P` — not one `value(expr)` walk per coefficient.
3. **Batched solver update.** Each group is applied with a single HiGHS batch
   call — `changeColsCost` / `changeRowsBounds` / `changeColsBounds` — instead of
   one scalar call per coefficient.
4. **Warm basis kept.** The retained instance re-solves from the previous simplex
   basis (a handful of iterations); `keep_basis=False` gives a cold re-solve for
   the basis-reset equivalence check.

Delivered in `pyomo/contrib/vector/`:

* `faststep.py` — the `_MutablePlan` (parameter registry + affine templates), the
  `_AffineArray` evaluator (`M @ P + base`, with a `value()` residual fallback for
  any entry that is not affine in the parameters), and the `FastStepHighs`
  persistent engine + solution map-back (reusing the Phase-2 loader).
* `tests/test_faststep.py` — the warm-solve **equivalence gate** (rolling
  sequences vs per-roll fresh `highs_fastload` builds, LP + MIP, basis-kept and
  basis-reset), the array / dirty-mask update paths, the explicit index-addressed
  API, residual handling, and the fail-loud scope guards.
* `bench/models/rolling_mpc.py` + `bench/warm_faststep.py` — the synthetic
  rolling-horizon warm-tick benchmark and its runner.

No core module was changed; the whole feature is additive under `pyomo.contrib.vector`.

---

## 2. The mechanism (why it is faster)

| per warm roll | APPSI persistent (warm) | `highs_faststep` |
|---|---|---|
| detect changes | diff scan over mutable helpers | none (templates fixed at compile) |
| evaluate coefficients | **one `value(expr)` walk per coefficient** | **one sparse `M @ P` for the whole group** |
| push to solver | **one scalar `changeColCost`/`changeCoeff`/`changeRowBounds` per coefficient** | **one batch `changeColsCost`/`changeRowsBounds`/`changeColsBounds` per group** |
| basis | warm (kept) | warm (kept) |

Both routes keep the solver warm and solve the same problem; Phase 4 replaces the two per-coefficient Python loops (evaluate, push) with a read of the parameter vector, a vectorized matrix–vector product, and a batch solver call.

---

## 3. Results — synthetic rolling-horizon suite (the committed exit-criteria evidence)

Model: `rolling_mpc` — a multi-asset energy MPC (`A` assets × `T` intervals; a per-asset state-of-charge recurrence with static efficiency couples the intervals, a per-interval grid cap couples the assets). Each **roll** rewrites every price (objective), every demand and grid cap (RHS), and every power cap (variable bound); the constraint matrix is static. Matrix nonzeros ≈ `4·A·T`.

Median warm tick (update + warm re-solve) over the roll sequence; **speedup** = APPSI-persistent ÷ faststep. All objectives equal across routes to < 1e-6 relative (the equivalence gate).

| size | A × T | nnz | APPSI-persistent | faststep (model) | faststep (array) | **speedup (model)** | speedup (array) | equiv |
|---|---|---|---|---|---|---|---|---|
| xs  | 4×20   | 316     | 2.02 ms   | 0.83 ms   | 0.76 ms   | **2.43×** | 2.67× | yes |
| 1e4 | 12×220 | 10,548  | 52.95 ms  | 15.64 ms  | 14.54 ms  | **3.39×** | 3.64× | yes |
| 1e5 | 40×640 | 102,360 | 685.38 ms | 294.26 ms | 289.52 ms | **2.33×** | 2.37× | yes |

*(`xs`/`1e4` at 20 rolls, `1e5` at 30 rolls; seed 1000.)*

### Reading the numbers

* **At the `1e5`-nnz class the model-driven warm tick is 2.33× faster** than the persistent APPSI route, clearing the ≥ 1.4× exit target with margin. The array-driven (mapping-free) path is 2.37×.
* **The two halves both matter.** Because the evidence split the APPSI warm cost roughly evenly between per-coefficient `value()` walks and scalar solver calls, batching only the solver side would cap near ~1.4–1.7×; vectorizing the evaluation (`M @ P`) *and* batching the push is what reaches ~2.3×.
* **model vs array** are close here: the model-driven path's only extra cost is reading `P` from the Pyomo Params once per roll (a few ms at this size), which the array path skips by receiving `P` directly.
* The absolute warm tick is dominated by the update at this size (the LP itself is warm-started and cheap), which is exactly why an array-native update wins.

---

## 4. Warm-solve equivalence (correctness gate)

The correctness surface is **row/column identity across rolls**. Phase 4 pins it three ways:

1. **Matrix by construction.** The matrix loaded into HiGHS is the Phase-2
   `LinearStandardFormCompiler` standard form — identical to the classic route by
   construction — and it is never rebuilt; only data is pushed onto its fixed
   columns/rows.
2. **Template self-check at `set_instance`.** Every affine template must reproduce
   the compiled standard-form arrays *at the current parameters* **and** at a
   random perturbation of the parameter vector (a fresh `value()` evaluation) —
   catching any non-affine or mis-mapped template. A failure raises
   `IncompatibleModelError`; the model is rejected, never solved wrong.
3. **Rolling equivalence tests.** A rolling sequence of solves through
   `FastStepHighs` matches a per-roll fresh `highs_fastload` build+solve — objective
   and termination on LP and MIP, plus primal values on a unique-optimum LP — for
   **both** basis-kept and basis-reset runs, and across the model-driven,
   array-driven, and dirty-mask update paths. The benchmark additionally asserts
   objective equivalence across all three routes on every roll at every size.

---

## 5. Scope and guards (fail loud, never a stale-matrix solve)

* **Linear** continuous / MIP only (inherited from the standard-form compile).
  Nonlinear / unsupported structure is rejected at `set_instance`.
* Supported warm updates: **objective coefficients, objective offset, constraint
  (row) bounds / RHS, and variable bounds**, driven by mutable `Param` values (or
  fixed-variable values). This is exactly the rolling-horizon roll.
* **Constraint-matrix coefficients are treated as static.** A mutable coefficient
  on a *free* variable (the matrix `A` would change between rolls) is rejected at
  `set_instance` — updating `A` in place is out of scope, and silently ignoring it
  would risk a stale-matrix solve. On an equal-interval roll the matrix genuinely
  does not change (durations/efficiencies are constant), so the roll's update is
  fully batchable; a model that needs matrix updates uses `highs_fastload` for a
  fresh compile per solve. *(Deferred; see §7.)*
* A **structure change** between solves (a constraint/variable added or removed,
  the objective swapped) is caught by a cheap fingerprint check and rejected.
* An entry that is **not affine in the parameters, or references a fixed
  variable**, is evaluated per-solve with `value()` as a *residual* — correctness
  preserved, only that (typically rare) entry not vectorized.

---

## 6. The array (mapping-free) update path

A caller that already holds the roll's data as arrays can drive the solve without
touching the Pyomo model:

```python
P = ...                                   # values ordered by stepper.parameters
stepper.solve(param_values=P)             # faststep owns the LP row/col mapping
stepper.solve(param_values=P, dirty=mask) # only the changed parameters' rows
```

`FastStepHighs` owns the LP row/column mapping and the coefficient templates
internally, so that mapping never has to live on the caller side — the caller
supplies only raw per-parameter values (and, optionally, a `dirty` mask so only
the affected rows/columns are recomputed and pushed).

---

## 7. Deferrals

* **Constraint-matrix (A) coefficient updates.** Deferred deliberately: the target
  rolling pattern does not change the matrix (the report's motivating evidence
  notes durations/efficiencies are constant on an equal-interval roll), and
  supporting in-place `A` edits would add a per-coefficient path the batch design
  exists to avoid. Mutable matrix coefficients are rejected loudly rather than
  silently ignored. **Update:** the *rejection* was since replaced by a
  **value-aware static-matrix guard** — a nominally-mutable matrix coefficient
  whose value never changes is now accepted and warm-solved, and a genuine change
  fails loud (or, opt-in, reloads); see `bench/VALUEGUARD_REPORT.md`. In-place
  batched `A` *edits* remain deferred (the guard detects change; it does not yet
  apply it).
* **Finer dirty-set evaluation.** The `dirty` mask currently recomputes the full
  value of every *affected* row (any row touched by a changed parameter) and
  pushes those; it does not yet compute per-entry deltas. For the target pattern
  (every price/forecast rolls) the mask is all-true and this is moot; it is a
  refinement for sparser update patterns.

---

## 8. Reproducing

```bash
# recreate the venv (see bench/README.md), then from the repo root:
bench/.venv/bin/python -m bench.warm_faststep --sizes 1e4,1e5 --rolls 30 \
    --out bench/results/phase4_faststep.json

# the interface's unit + equivalence tests:
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/test_faststep.py
```

---

## 9. Deliverables checklist

- [x] `highs_faststep` persistent warm interface: compile once + `passModel`, retained live HiGHS, array-native batch update (`changeColsCost` / `changeRowsBounds` / `changeColsBounds`), warm basis, solution map-back (`pyomo/contrib/vector/faststep.py`)
- [x] Vectorized coefficient evaluation (affine templates `M @ P`), not per-coefficient `value()` walks
- [x] Change-detection contract: model-driven (mutate Params, solve) **and** array/mapping-free (`param_values` + `dirty`) update paths; faststep owns the row/col mapping
- [x] Scope guards: linear-only, mutable-matrix-coefficient rejection, structure-change detection — all fail loud
- [x] Warm-solve equivalence gate vs per-roll fresh classic build (LP + MIP, basis-kept + basis-reset, all update paths)
- [x] Synthetic rolling-horizon benchmark with a warm-tick table; ≥ 1.4× at the 1e5-nnz class (measured **2.33×**)
- [x] Zero core-module change; all Phase 1/2/3 tests still green
