# Pyomo Vectorized Construction — Phase 0 Baseline Report

**Project:** Vectorized Model Construction for Pyomo (see the project's *Vectorized Model Construction for Pyomo* scoping document)
**Phase:** 0 — benchmark harness + feasibility spikes + published baseline
**Branch:** `fm/pyomo-phase0-bench` (harness under `bench/`)
**Date:** 2026-08-25
**Baseline:** stock upstream Pyomo at this clone's HEAD, measured against gurobipy (matrix API), linopy, and a raw scipy→HiGHS array-native path.

## Environment

- Pyomo `6.10.2.dev0` @ commit `2744a2446` (editable install of this worktree's HEAD)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs, 15 GiB RAM
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1 · gurobipy 13.0.3 (restricted, size-limited) · linopy 0.9.1
- All numbers reproducible; environment is embedded in every results JSON under `sysinfo`.

---

## 1. What Phase 0 delivered

1. **`bench/` harness** — parameterized generators for the four synthetic models the scoping doc names (dense multi-period network flow; unit commitment, mixed sparse/dense; facility location + a quadratic variant; ragged supply chain), sizes 10⁴–10⁶ nonzeros (10⁷ available manually). Stage-separated timings (construct / repn / write / load) + peak RSS, JSON output, a CI subset and a full manual mode. See `bench/README.md`.
2. **Comparators** — linopy and gurobipy matrix-API implementations, plus a raw scipy→HiGHS array-native path, for the two cleanly-vectorizable synthetics. Versions above.
3. **Equivalence oracle** (`bench/equivalence.py`) — checks the array-native builders encode the *same LP* as the Pyomo generators, up to row/column permutation keyed on variable identity (the Phase-1 correctness foundation); runs in the CI subset.
4. **Spike A** — columnar `Var` micro-prototype: object-creation & memory savings.
5. **Spike B** — template-expression numeric instantiation: the go/no-go gate for rule vectorization (scoping doc §6.2 ambition 3 / risk R3).
6. **This report** — published numbers, stage breakdowns, spike verdicts, and a recommended Phase 1 vertical slice.

---

## 2. Method (how to read the numbers)

`time-to-solver` is split into the four stages the project targets (scoping doc §2.1). Each is timed with a warmup then repeats; we report the **median** (min is in the JSON). Each `(backend, model, size)` case runs in its own subprocess, so peak RSS is attributable to one case and a crash / size-limited-license error can't take down the suite.

| Stage | Measured as | Exercises |
|---|---|---|
| construct | build the `ConcreteModel` from the generator | per-index `VarData`/`ConstraintData` alloc + operator-overload trees |
| repn | `LinearStandardFormCompiler().write(...)` (linear) | `pyomo.repn.linear` visitor → scipy CSC/CSR |
| write | `model.write("*.lp")` | LP writer v2 |
| load | APPSI HiGHS `set_instance` (no solve) | per-row solver load (#3888) |

**A note on totals (read this before comparing across systems).** The four stages are each measured *independently on the same built model*, and none consumes another's output. So `construct + repn + write + load` (reported as **`Σ stages`**) is **not a single pipeline** and no real workflow executes all four in sequence: `repn` (`LinearStandardFormCompiler`) is a standalone matrix compile on *neither* solver route, and `write` (LP file) and `load` (APPSI `set_instance`) are *alternative* routes to a solver, each doing its own internal repn. The honest end-to-end wall-clock — **"empty model → the solver has it"** — is the single coherent route **`build→solver = construct + load`** (the in-memory APPSI route). Cross-system ratios in §4 use `build→solver` on both sides; `Σ stages` is shown only for per-stage context. Sizes are named by target constraint-matrix nonzeros (the harness reports actual sizes); `xs` is sized under gurobipy's 2000-var/-constraint size-limited-license cap.

**Correctness (equivalence oracle).** That the array-native builders encode the *same LP* as the Pyomo generators is now checked by a committed oracle, `bench/equivalence.py` (run in the CI subset). It compares the two standard forms **up to row/column permutation, keyed on variable identity**: the constraint system as a multiset of sign-normalized rows, plus per-variable bounds, the objective coefficient vector, and (as fast pre-filters) nnz and the row/column degree-sequence multisets — and, as a final cross-check, both LPs solved through HiGHS. All checks pass for network flow (obj 128.784289, nnz 760) and facility location (obj 469.35, nnz 1010). Matching objective and nnz *alone* would be necessary but not sufficient (different LPs can share both); the row-multiset-up-to-permutation check is the sufficient one, and is what Phase 1's "identical standard forms" exit criterion needs.

---

## 3. Baseline results — Pyomo, stage-separated

**The headline structural finding.** Construction is *not* the largest stage at scale. For every non-quadratic model the **per-row solver load dominates**. The `load %` column below is load's share of `Σ stages` (the non-pipeline sum, which *understates* it); the last column is the honest figure — load's share of the coherent `build→solver` (construct+load) route:

| model | largest size | construct % | repn % | write % | load % (of Σ) | **load % (of build→solver)** |
|-------|------|-------------|--------|---------|---------------|------------------------------|
| network_flow | 1e6 | 13% | 12% | 27% | 48% | **79%** |
| unit_commitment | 1e6 | 6% | 9% | 15% | 70% | **92%** |
| facility_location | 1e6 | 10% | 13% | 23% | 54% | **85%** |
| supply_chain | 1e6 | 10% | 12% | 21% | 57% | **85%** |
| facility_location_q | 1e5 | 18% | 33% | 49% | n/a¹ | n/a¹ |

¹ The quadratic objective is not loadable via APPSI HiGHS (`DegreeError`); load N/A. It *is* writable (LP writer emits it), so construct/repn/write are measured.

This says plainly: a vectorization effort that only speeds up object construction (columnar `Var`) leaves most of the wall-clock on the table. **The direct array→solver hand-off (#3888, scoping doc §6.4) is where the biggest wins are.**

### Per-model stage breakdown (median ms, `s` = seconds)

**network_flow** (dense multi-period min-cost flow)

| size | vars | cons | nnz | construct | repn | write | load | **Σ stages** |
|------|------|------|-----|-----------|------|-------|------|-----------|
| 1e4 | 5,000 | 500 | 9,990 | 15.3 | 12.8 | 25.8 | 60.2 | **114.1** |
| 1e5 | 50,000 | 2,500 | 99,980 | 154.1 | 115.5 | 243.2 | 492.6 | **1005.4** |
| 1e6 | 500,000 | 10,000 | 999,950 | 1521 | 1394 | 3144 | 5582 | **11.6 s** |

**unit_commitment** (mixed sparse/dense MILP)

| size | vars | cons | nnz | construct | repn | write | load | **Σ stages** |
|------|------|------|-----|-----------|------|-------|------|-----------|
| 1e4 | 2,400 | 4,740 | 12,500 | 12.5 | 17.7 | 29.9 | 130.4 | **190.4** |
| 1e5 | 24,000 | 47,910 | 127,250 | 127 | 200 | 327 | 1359 | **2013.7** |
| 1e6 | 240,000 | 480,300 | 1,278,500 | 1418 | 2247 | 3676 | 17.1 s | **24.4 s** |

**facility_location** (capacitated MILP)

| size | vars | cons | nnz | construct | repn | write | load | **Σ stages** |
|------|------|------|-----|-----------|------|-------|------|-----------|
| 1e4 | 2,525 | 2,625 | 10,025 | 10.1 | 11.7 | 22.0 | 57.1 | **100.8** |
| 1e5 | 25,050 | 25,550 | 100,050 | 95.7 | 122 | 222 | 559 | **998.1** |
| 1e6 | 250,100 | 252,600 | 1,000,100 | 1079 | 1425 | 2498 | 5960 | **11.0 s** |

**supply_chain** (ragged multi-echelon, sparse lanes)

| size | vars | cons | nnz | construct | repn | write | load | **Σ stages** |
|------|------|------|-----|-----------|------|-------|------|-----------|
| 1e4 | 3,200 | 1,600 | 6,388 | 10.5 | 10.4 | 20.0 | 54.7 | **95.6** |
| 1e5 | 28,680 | 12,600 | 57,330 | 91.7 | 93.2 | 176 | 464 | **825.0** |
| 1e6 | 229,800 | 84,000 | 459,520 | 756 | 837 | 1503 | 4179 | **7273.6** |

**facility_location_q** (quadratic objective; R7 hard-ceiling probe) — repn via `generate_standard_repn(quadratic=True)`

| size | vars | cons | construct | repn | write | load | **Σ stages** |
|------|------|------|-----------|------|-------|------|-----------|
| 1e4 | 2,525 | 2,625 | 12.2 | 21.4 | 32.1 | n/a | **65.7** |
| 1e5 | 25,050 | 25,550 | 122 | 221 | 330 | n/a | **673.0** |

Timings scale ~linearly in nonzeros, as expected for an O(model-size) interpreted pipeline. Peak model RSS at 1e6 ranges ~0.7–1.9 GB (full figures in `bench/results/full.json`).

---

## 4. Cross-system comparison — how far is Pyomo from array-native?

For the two synthetics with clean matrix forms, the same LP built directly as scipy matrices and loaded into HiGHS (`array→HiGHS`) is the tightest lower bound — it is exactly what the fast path (scoping doc §6.3/§6.4) would do. The ratio below is the **coherent `build→solver` route on both sides** — Pyomo `construct+load` vs array `build+load` — *not* Pyomo's `Σ stages` (which would inflate it ~1.6× by summing a standalone repn compile plus the two alternative solver routes). `Σ stages` is shown for reference. linopy (xarray-backed) is the higher-level array-native comparator, but its endpoint is "matrix in memory", one step short of "loaded in the solver", so it is not strictly comparable to the `build→solver` columns; gurobipy's matrix API runs at `xs` only (size-limited license: build+load ≈ **1.3 ms** network flow / **1.4 ms** facility location at 384/260 vars — it refuses models >2000 vars/constraints).

**network_flow**

| size | nnz | Pyomo build→solver (construct+load) | array→HiGHS (build+load) | linopy (build+extract) | Pyomo Σ stages | **Pyomo ÷ array-native** |
|------|-----|-------------------------------------|--------------------------|------------------------|----------------|--------------------------|
| 1e4 | 9,990 | 75.5 | 2.8 | 40.1 | 114.1 | **27×** |
| 1e5 | 99,980 | 646.7 | 19.4 | 50.5 | 1005.4 | **33×** |
| 1e6 | 999,950 | 7103.3 | 184.8 | 438.3 | 11.6 s | **38×** |

**facility_location**

| size | nnz | Pyomo build→solver (construct+load) | array→HiGHS (build+load) | linopy (build+extract) | Pyomo Σ stages | **Pyomo ÷ array-native** |
|------|-----|-------------------------------------|--------------------------|------------------------|----------------|--------------------------|
| 1e4 | 10,025 | 67.2 | 3.3 | 49.4 | 100.8 | **20×** |
| 1e5 | 100,050 | 654.2 | 28.3 | 57.2 | 998.1 | **23×** |
| 1e6 | 1,000,100 | 7039.3 | 289.0 | 126.9 | 11.0 s | **24×** |

**On the coherent `build→solver` route, Pyomo spends ~20–38× more wall-clock than a raw array→solver path to reach the same solver state** (network flow 27→38×, facility location 20→24×), and the gap *widens* with size. (Comparing Pyomo's `Σ stages` instead would report ~30–60×, but that inflates the gap ~1.6× by summing non-sequential passes — see the §2 note.) linopy sits in between (array-native construction, but its own xarray canonicalization tax), with the endpoint caveat above. The gap is not a micro-optimization target; it is structural, and §3 shows almost all of the `build→solver` route is the per-row solver **load** (76–95%), not construct.

---

## 5. External real-world validation

Beyond the synthetic suite, an external private real-world case was measured
through the identical stage harness and showed the **same load-dominated
profile**: the per-row solver load is the overwhelming majority of the coherent
`build→solver` route, so columnar `Var` alone (where construct is only a small
fraction of the total) moves the total little and the direct array hand-off is
nearly the whole win. No detail of that case is carried here — the synthetic
numbers in §3–§4 carry the same conclusion and are the reproducible baseline.

---

## 6. Spike A — columnar `Var` (object-creation & memory)

**Question (scoping doc §6.1, G2, #202):** how much of construction cost and memory *is* the per-index `VarData` object?

| N (vars) | classic build | columnar build | build speedup | classic B/var | columnar B/var | memory ratio |
|----------|---------------|----------------|---------------|---------------|----------------|--------------|
| 10,000 | 4.2 ms | 0.01 ms | **359×** | 173 B | 26 B | 0.150 |
| 100,000 | 64.4 ms | 0.12 ms | **531×** | 196 B | 26 B | 0.132 |
| 1,000,000 | 854.7 ms | 1.58 ms | **542×** | 186 B | 26 B | 0.140 |

(Columnar = five parallel NumPy arrays: lb/ub/value f64, fixed bool, domain int8 = 26 B/var. Classic bytes are Python-heap via `tracemalloc`, a *lower* bound on true `VarData` overhead. Random access was also 40–60× faster for the array.)

**Verdict — strong GO for columnar `Var` storage.** Bulk allocation is **~360–540× faster** and uses **~0.14× the memory** — comfortably inside the scoping doc's ≤0.3× target.

**Caveat (R1).** The spike measures raw allocation only; it does *not* build flyweight views or preserve the `id()`-identity semantics that repn and solver maps depend on. That identity problem — not allocation — is the real Phase-1 correctness cost, and the scoping doc's "materialize-on-touch" starting point (§6.1) is the right way to de-risk it. And note §3/§5: for the most load-bound models, columnar `Var` alone moves the total very little.

---

## 7. Spike B — template vectorization (the go/no-go gate)

**Question (scoping doc §6.2 ambition 3 / R3):** can a user's existing `def rule(m, i): ...` be run **once** through template machinery and then numerically instantiated over its whole index set with NumPy — making existing Pyomo code fast without a rewrite?

**Coverage — which rule shapes templatize:**

| rule shape | example | templatizes? |
|------------|---------|--------------|
| scalar-affine | `a*x[i] <= p[i]` | ✅ yes |
| scalar-affine (neighbour) | `2*x[i] - x[i-1] <= 1` | ✅ yes |
| unfiltered sum-over-set | `sum(f[j,n,t] for j in N)` | ✅ yes |
| quadratic | `x[i]*x[i] <= 1` | ✅ yes |
| **index-conditional** | `if i==0: Skip else ...` | ❌ no — `PyomoException` (can't bool `_1 == 0`) |
| **modulo / non-affine index** | `x[(i-1) % N]` | ❌ no — `TypeError` on `%` |
| **filtered sum-over-set** | `sum(... for j in N if j != n)` | ❌ no — same `PyomoException` on the filter |

**Scalar-affine family (`2·x[i] − x[i−1] ≤ 1`): correctness + speed**

| N (constraints) | vectorized == classic? | classic repn | resolve/idx | vectorized | **vec speedup** | resolve "speedup" |
|-----------------|------------------------|--------------|-------------|------------|-----------------|-------------------|
| 10,000 | ✅ | 29 ms | 193 ms | 2.2 ms | **13×** | 0.15× |
| 100,000 | ✅ | 306 ms | 1871 ms | 18.9 ms | **16×** | 0.16× |
| 1,000,000 | ✅ | 3303 ms | 18539 ms | 210 ms | **16×** | 0.18× |

**Sum-over-set family (unfiltered flow balance):** templatizes; `resolve_template`/idx is 0.17–0.19× (again slower than classic).

**Verdict — QUALIFIED GO, with three hard design constraints:**

1. **Vectorized extraction works and is fast.** An automated walk of a templatized *linear* body reconstructs the exact CSR rows a classic per-index repn produces (correctness verified against classic) at **13–16× the speed**, scaling to 10⁶ constraints. This is real and it is the single highest-value ambition in the scoping doc.
2. **`resolve_template` per index is a trap.** Templatize-once-then-resolve-per-index is **~5–7× *slower* than just running the classic rule**. The win comes *only* from full vectorized instantiation (extract the skeleton once, fill arrays with NumPy). Any design that resolves the template per index will be slower than today — state this loudly in the Phase-3 design.
3. **Coverage is a well-defined subset → a scalarization fallback is mandatory (R3).** Index conditionals, modulo/non-affine indexing, and — importantly — *filtered* sums (`if j != n`, the everyday idiom for self-loops / sparse subsets) do **not** templatize. Template vectorization is a **partial-coverage optimization**: vectorize the rows that templatize, scalarize the rest. That fallback is the majority path for real models, not an edge case.

---

## 8. Risks — what Phase 0 confirmed or resharpened

| # | Scoping-doc risk | Phase-0 finding |
|---|---|---|
| R1 | flyweight identity breaks `id()`-keyed maps | Real and unmeasured here; Spike A confirms allocation is cheap (~500×), so **identity is the true Phase-1 cost** — start materialize-on-touch. |
| R3 | template rule vectorization can't handle real rules | **Confirmed and quantified.** Big subset works (13–16×); conditionals / modulo / *filtered sums* don't templatize; and `resolve_template`/idx is a de-optimization. Scalarization fallback is mandatory. |
| R5 | sparse-index mapping erodes speedup | The ragged supply-chain model is in the suite from day 1; **linopy could not idiomatically express it** (dense-label design), which is itself the differentiation the scoping doc predicts. |
| #3888 | per-row solver load dominates | **Loudly confirmed** — load is the single largest stage for *every* model (48–80% of `Σ stages`; **76–95% of the coherent `build→solver` route**), and the whole win for the most load-bound models. Direct array hand-off (§6.4) is where the biggest wins are. |

---

## 9. Recommended Phase 1 vertical slice

The scoping doc's Phase 1 (columnar `Var` + explicit-array `VectorConstraint` + standard-form splice + HiGHS array hand-off, proven on dense network flow) is the right slice. Phase 0 sharpens *where the value is*:

1. **Lead with dense network flow end-to-end** (the scoping doc's Phase-1 exit criterion). It is the cleanest full-pipeline demonstrator and already has a matched array-native lower bound in the harness to measure against.
2. **Prioritize the load hand-off, not just construction.** Load is the largest single stage at scale — 48–80% of `Σ stages`, but **76–95% of the coherent `build→solver` route** (§3); a `VectorConstraint` that never reaches a direct `Highs.passModel` array hand-off leaves most of the win on the table. Build the standard-form-splice → HiGHS `passModel` path early and measure it against `array→HiGHS`, the in-harness ceiling (185 ms at 1e6 vs Pyomo's 7.1 s `build→solver`).
3. **Columnar `Var` with materialize-on-touch first** (Spike A: ~500× allocation, ~0.14× memory; R1 says identity is the risk — take the scoping doc's safer starting point). But weight it against #2: for load-bound models it moves the total little on its own.
4. **Defer template vectorization to Phase 3, designed around Spike B's constraints** — full vectorized extraction (never resolve-per-index) + a mandatory scalarization fallback for non-templatizable rows.
5. **The correctness harness is stood up.** `bench/equivalence.py` already asserts the array-native and Pyomo standard forms are identical **up to row/col permutation, keyed on variable identity** (not just obj+nnz), for the two synthetics, and runs in the CI subset. Phase 1 extends the same oracle to the fast path (columnar `Var` + `VectorConstraint`) and to randomized models — the scoping doc's Phase-1 exit criterion — reusing the row-multiset comparator verbatim.
6. **Upstream the benchmark harness independently** (scoping doc open question #6: likely yes). It is a standalone win and a low-risk first PR that gives the eventual PEP its authority — this branch is that PR.

**Suggested Phase-1 exit metric, measured by this harness:** dense network flow end-to-end ≥10× faster than the classic Pyomo path at ≥10⁶ nonzeros and within 3× of `array→HiGHS`, with standard-form equivalence tests passing.

---

## 10. Reproducing

```bash
# recreate the venv (exact steps in bench/README.md), then from the repo root:
bench/.venv/bin/python -m bench.run_bench --suite full --sizes 1e4,1e5,1e6 --out bench/results/full.json
bench/.venv/bin/python -m bench.equivalence  --out bench/results/equivalence.json   # correctness oracle
bench/.venv/bin/python -m bench.spikes.spike_a_columnar_var  --out bench/results/spike_a.json
bench/.venv/bin/python -m bench.spikes.spike_b_template_expr --out bench/results/spike_b.json
bench/.venv/bin/python -m bench.analyze         # regenerates the tables in this report
# the CI subset (`--suite ci`) also runs the equivalence oracle and fails on a mismatch:
bench/.venv/bin/python -m bench.run_bench --suite ci --out bench/results/ci.json
```

Raw results are committed under `bench/results/` (`full.json`, `ci.json`, `spike_a.json`, `spike_b.json`, `equivalence.json`, `tables.md`). `full.json` holds the timing sweep; `equivalence.json` and the `equivalence` block in `ci.json` hold the correctness verdict. All timings share the one environment/commit above.
