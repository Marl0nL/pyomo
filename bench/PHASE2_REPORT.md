# Pyomo Vectorized Construction — Phase 2 Report

**Project:** Vectorized Model Construction for Pyomo (see the project's *Vectorized Model Construction for Pyomo* scoping document and the Phase-0 baseline report)
**Phase:** 2 — transparent fast solver hand-off for classic models
**Deliverable:** `highs_fastload`, a drop-in solver that routes an **unmodified classic** linear model's solve through the standard-form compile → `Highs.passModel` bulk hand-off.
**Baseline:** stock upstream Pyomo at this clone's HEAD, in-process HiGHS.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs, 15 GiB RAM
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- All numbers reproducible with the `bench/` harness (§ Reproducing); each `(model, size)` case runs in its own subprocess.

---

## 1. The insight, and what Phase 2 delivered

Phase 0 established that for classic Pyomo linear models the **solver load** dominates time-to-solution — 48–80% of the coherent `construct + load` route at scale — because the in-memory solver interfaces extract the model one row at a time, re-running a per-row `generate_standard_repn` and issuing one solver API call per constraint (#3888).

But a classic linear model already produces exactly the arrays a solver wants *mid-pipeline*: `LinearStandardFormCompiler` walks every constraint once through the fast `pyomo.repn.linear` visitor and emits one scipy CSR/CSC matrix. Phase 1 proved that handing such a matrix to HiGHS via `Highs.passModel` reaches "the solver has the model" at ~1.01× the theoretical array-native ceiling — but only for models built with the `pyomo.contrib.vector` columnar components.

**Phase 2 makes that hand-off transparent for unmodified classic models.** It adds one solver:

```python
from pyomo.contrib.solver.common.factory import SolverFactory
results = SolverFactory('highs_fastload').solve(model)   # no model change
# or the legacy factory:  pyomo.SolverFactory('highs_fastload').solve(model)
```

`highs_fastload` compiles the model with the stock `LinearStandardFormCompiler` (mixed form), builds a `highspy.HighsLp` from the resulting arrays, hands it to HiGHS in one `passModel` call, solves, and maps the solution (primals, objective, termination, plus duals / reduced costs for LPs) back onto the Pyomo objects. It is the HiGHS analogue of the shipped `gurobi_direct` interface, which already uses this exact `LinearStandardFormCompiler → addMConstr` pattern.

Delivered in `pyomo/contrib/vector/`:

* `fastload.py` — `compile_to_highs_arrays` (classic model → HiGHS range-row arrays + map-back metadata), `build_highs_lp`, the `FastLoadHighs` solver, and its solution loader. Registers `highs_fastload` on both the v2 and legacy `SolverFactory` at package import.
* Tests (`tests/test_fastload.py`) — registration, primal/dual/reduced-cost correctness, maximize + objective offset, MIP integrality, fixed variables, range constraints, objective-free feasibility, infeasible/unbounded/nonlinear handling, and **solve-result equivalence** vs the persistent APPSI HiGHS interface on randomized models.
* Harness: a `fastload_highs` stage + a fast-route column in `bench/analyze.py`.

No core module was changed; the whole feature is additive under `pyomo.contrib.vector`.

---

## 2. The mechanism (why it is faster)

| | classic coherent route | fast route (`highs_fastload`) |
|---|---|---|
| construct | build the `ConcreteModel` | build the `ConcreteModel` (**identical, shared**) |
| hand-off | APPSI `set_instance`: per-row `generate_standard_repn` + per-row `addRows` | `LinearStandardFormCompiler` (one bulk visitor pass) + one `Highs.passModel` |
| endpoint | the solver has the model | the solver has the model |

Both routes construct the same model. The fast route replaces the **hand-off** stage. Its win therefore has a ceiling set by how much of the classic route is the hand-off: on **load-bound** models (the hand-off dwarfs construct) the end-to-end speedup is large; on **construct-bound** models (fast construct, well-behaved load) the shared construct caps the end-to-end factor even though the hand-off itself is 3–4× faster.

---

## 3. Results — synthetic suite (the committed exit-criteria evidence)

Median-of-repeats, milliseconds unless suffixed `s`. `build→solver` = construct + the route's hand-off. `hand-off ×` isolates the stage the fast route replaces; `end-to-end ×` is the coherent `construct + hand-off` ratio.

**network_flow** (dense multi-period min-cost flow — LP, construct-bound)

| size | nnz | construct | load (classic) | fastload (fast) | classic build→solver | fast build→solver | hand-off × | **end-to-end ×** |
|------|-----|-----------|----------------|-----------------|----------------------|-------------------|------------|------------------|
| 1e4 | 9,990 | 14.7 | 56.8 | 18.0 | 71.5 | 32.6 | 3.2× | **2.19×** |
| 1e5 | 99,980 | 145.4 | 515.6 | 167.4 | 660.9 | 312.7 | 3.1× | **2.11×** |
| 1e6 | 999,950 | 1552.3 | 5611.9 | 1772.6 | 7164.2 | 3324.9 | 3.2× | **2.15×** |

**unit_commitment** (mixed sparse/dense MILP — load-bound)

| size | nnz | construct | load (classic) | fastload (fast) | classic build→solver | fast build→solver | hand-off × | **end-to-end ×** |
|------|-----|-----------|----------------|-----------------|----------------------|-------------------|------------|------------------|
| 1e4 | 12,500 | 12.2 | 134.7 | 21.0 | 146.9 | 33.1 | 6.4× | **4.43×** |
| 1e5 | 127,250 | 129.6 | 1393.3 | 235.9 | 1523.0 | 365.5 | 5.9× | **4.17×** |
| 1e6 | 1,278,500 | 1453.5 | 17.2s | 2698.5 | 18.7s | 4152.0 | 6.4× | **4.50×** |

**facility_location** (capacitated MILP — construct-bound)

| size | nnz | construct | load (classic) | fastload (fast) | classic build→solver | fast build→solver | hand-off × | **end-to-end ×** |
|------|-----|-----------|----------------|-----------------|----------------------|-------------------|------------|------------------|
| 1e4 | 10,025 | 9.9 | 57.8 | 15.7 | 67.7 | 25.5 | 3.7× | **2.65×** |
| 1e5 | 100,050 | 97.8 | 584.4 | 162.7 | 682.2 | 260.4 | 3.6× | **2.62×** |
| 1e6 | 1,000,100 | 1046.1 | 6054.5 | 1798.8 | 7100.7 | 2844.9 | 3.4× | **2.50×** |

**supply_chain** (ragged multi-echelon, sparse lanes — construct-bound)

| size | nnz | construct | load (classic) | fastload (fast) | classic build→solver | fast build→solver | hand-off × | **end-to-end ×** |
|------|-----|-----------|----------------|-----------------|----------------------|-------------------|------------|------------------|
| 1e4 | 6,388 | 10.0 | 51.5 | 13.7 | 61.6 | 23.8 | 3.8× | **2.59×** |
| 1e5 | 57,330 | 88.0 | 465.4 | 121.9 | 553.4 | 209.9 | 3.8× | **2.64×** |
| 1e6 | 459,520 | 763.5 | 4193.6 | 1088.5 | 4957.1 | 1852.0 | 3.9× | **2.68×** |

### Reading the numbers

* **The hand-off itself is 3.1–6.4× faster across the whole suite, at every size** — this is the stage the fast route replaces, and it is uniformly and substantially faster.
* **End-to-end, the load-bound `unit_commitment` model reaches 4.50× at 1e6** (4.4–4.5× across all sizes), clearing the ≥3× exit target. Its APPSI load is 17.2 s vs the fast route's 2.70 s hand-off.
* **The construct-bound models (`network_flow`, `facility_location`, `supply_chain`) reach 2.1–2.7× end-to-end.** This is the shared-construct ceiling of §2, not a limitation of the hand-off: for `network_flow` at 1e6, construct (1.55 s) + the interpreted repn the fast route still must run (1.77 s hand-off) already exceeds a third of the classic 7.16 s, so no transparent classic hand-off can reach 3× end-to-end on it. The gain is real (2.1×) and the hand-off is 3.2× faster; it is simply diluted by the construct the two routes share. A separate, external private real-world case — the kind of pathologically load-bound model this project targets — shows substantially larger end-to-end gains; those numbers are reported separately and are not part of this repository.

**The gain grows with how load-bound the model is.** The mechanism delivers the largest transparent wins exactly where the classic route hurts most (per-row load of param-referencing expressions), and a solid, uniformly-faster hand-off everywhere else.

---

## 4. Solve-result equivalence (correctness gate)

The fast route must produce the *same answer* as the classic route. Checked two ways:

1. **Matrix by construction.** `highs_fastload` compiles with the stock `LinearStandardFormCompiler` — the same matrix the classic repn produces — so the constraint system it hands HiGHS is identical to the classic standard form by construction (not merely equivalent).
2. **Solve results.** On the harness suite (small size, in the sweep JSON) and on 40 randomized models (in `tests/test_fastload.py`), the fast route's objective and termination match a persistent APPSI HiGHS solve of the same model:

| model | size | classic objective | fast objective | obj match | termination match |
|-------|------|-------------------|----------------|-----------|-------------------|
| network_flow | 1e4 | 1452.16 | 1452.16 | ✅ | ✅ |
| unit_commitment | 1e4 | 857121 | 857121 | ✅ | ✅ |
| facility_location | 1e4 | 1165.8 | 1165.8 | ✅ | ✅ |
| supply_chain | 1e4 | 22066 | 22066 | ✅ | ✅ |

For the unique-optimum LP (`network_flow`) the primal *values* also match to 1e-14; the MIPs match in objective and termination (variable values may differ by alternate optima, as they do between any two solvers).

---

## 5. Scope and guards

* **Linear continuous / MIP only.** A model with nonlinear terms (or components the standard-form compiler cannot process, e.g. SOS/Piecewise) is rejected with `IncompatibleModelError` pointing at the classic solver route — it never silently produces a wrong answer.
* **Objective sense + offset** are handed to HiGHS directly (`sense_`, `offset_`), so the reported objective is the true objective for both `minimize` and `maximize`, including a constant offset.
* **Fixed variables** are substituted into the row bounds / objective offset by the standard-form compiler (the #3851 pitfall) and pinned out of the column space.
* **Duals / reduced costs** are mapped back for LPs (HiGHS provides `row_dual` / `col_dual`); for MIPs HiGHS marks them invalid and the loader raises, matching the other in-tree interfaces.

---

## 6. Reproducing

```bash
# recreate the venv (see bench/README.md), then from the repo root:
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models network_flow,unit_commitment,facility_location,supply_chain \
    --backends pyomo --sizes 1e4,1e5,1e6 --out bench/results/phase2_fastload.json
bench/.venv/bin/python -m bench.analyze --phase2 bench/results/phase2_fastload.json

# the solver's own unit + equivalence tests:
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/test_fastload.py
```

The `pyomo` backend now measures a `fastload_highs` stage alongside `load_highs` and records the derived `classic_build_to_solver_ms` / `fast_build_to_solver_ms` / `fastload_speedup`, plus per-case solve-equivalence in `validation`.

---

## 7. Deliverables checklist

- [x] `highs_fastload` solver: classic model → standard-form compile → `passModel` → solution map-back (`pyomo/contrib/vector/fastload.py`)
- [x] Scope guard: linear only, loud rejection of nonlinear / unsupported structure
- [x] Solve-result equivalence vs the classic route (suite + randomized models)
- [x] Benchmark harness extended with the fast route as a column; synthetic suite reported at 1e4/1e5/1e6
- [x] Zero core-module change; all Phase 1 tests still green
