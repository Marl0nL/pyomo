# Pyomo PEP: Vectorized Model Construction and Solver Hand-off

**Status:** Draft for discussion
**Type:** Design / Standards track (Pyomo 7 window, cf. #3882)
**Landing zone:** `pyomo.contrib.vector` (prototype), with a proposed `contrib → core` split (§9)
**Depends on:** the *Vectorized Model Construction for Pyomo* scoping document
**Evidence base:** the committed synthetic benchmark suite under `bench/` and its per-phase reports (`bench/PHASE0_REPORT.md`, `PHASE2_REPORT.md`, `PHASE3_REPORT.md`, `PHASE4_REPORT.md`, `VALUEGUARD_REPORT.md`, `STATICFOLD_REPORT.md`, `COMPILE_SCALING_REPORT.md`, `QUADRATIC_QP_REPORT.md`, `GUROBI_FASTLOAD_REPORT.md`)

Every number in this document is reproduced from a committed report in `bench/`; every API named exists in `pyomo/contrib/vector/` and its quickstart is exercised by the package's test suite (`pyomo/contrib/vector/tests/`). No claim here is aspirational — where a capability is deferred, it is stated as deferred.

---

## 1. Abstract

Pyomo builds optimization models one Python object at a time: every variable is a `VarData`, every constraint an operator-overload-built expression tree, and every writer or solver interface walks those trees in interpreted Python. For models with 10⁵–10⁷ constraints, getting the model *into a solver* dominates time-to-solution and is the single most cited reason users migrate to gurobipy, linopy, or JuMP.

This proposal describes a **benchmarked, additive fast path** — prototyped in `pyomo.contrib.vector` with **zero core-module behavioural change** — that:

1. stores indexed variables **columnarly** (NumPy arrays, not one `VarData` per index);
2. states a whole linear (or convex-quadratic) constraint/objective family as **explicit arrays** instead of N expression trees;
3. compiles a model — classic *or* vectorized — to **standard-form arrays once** and hands them to a solver in **one bulk call** (`Highs.passModel` / Gurobi's matrix API), replacing the per-row load that Phase-0 measurement showed to be the dominant cost;
4. makes **existing rule-based model code faster with no rewrite**, via an opt-in template-vectorized construction switch, for the rule shapes that templatize (and byte-identical classic fallback for those that do not);
5. adds a **persistent, array-native warm re-solve** interface for rolling-horizon / MPC workloads, pushing changed coefficients as vectorized batches while keeping the warm simplex basis.

The design is governed by one load-bearing rule — the **compatibility contract** (§6): any consumer that does not understand a fast-path component triggers *lazy scalarization* to classic objects, so the fast path is invisible to the rest of the ecosystem and the classic path is provably unchanged.

The headline evidence is in §7. In one line: on the coherent "empty model → solver has it" route, the classic Pyomo path spends **~20–38× more wall-clock than a raw array→solver path** on the public synthetics (Phase 0), and **76–95% of that route is per-row solver load**, not construction — so the transparent bulk hand-off is where the largest transparent wins are, and it delivers **up to 4.5× end-to-end on a load-bound model** with no model-code change (Phase 2), rising to **9.8×** when construction is also vectorized on a templatizable model (Phase 3).

---

## 2. Motivation: the public evidence this is wanted

The scoping document cites a cluster of long-open upstream issues that this work directly addresses:

| Issue | Symptom | How this proposal addresses it |
|---|---|---|
| **#202** — centralized `VarData` storage (open ~9y) | per-index object allocation is expensive at scale | columnar `VectorVar` (§5.1); Phase-0 Spike A: ~360–540× faster bulk allocation, ~0.14× memory |
| **#1073** — [PEP] revisit the canonical repn (open ~7y) | the interpreted tree walk is a redesignable bottleneck | array-native standard-form assembly + vectorized template extraction (§5.3, §5.5) |
| **#1761** — slow quadratic constraint creation (open ~5y) | building `Σ Qᵢⱼ xᵢxⱼ` one node at a time is slow | convex-QP objective as an array Hessian (§5.6); QP report: **37–58× construction** |
| **#2808** — `sum_product` with sparse coefficients (open ~3y) | interpreted per-term summation | array-native coefficient assembly (§5.2, §5.3) |
| **#3888** — improve the HiGHS interface | per-row solver load dominates | direct `passModel` bulk hand-off (§5.4); Phase 0 confirms load is 76–95% of the coherent route |
| **#309** — unknown-component handling (open ~8y) | ecosystem code silently mishandles new component types | the compatibility contract / lazy scalarization (§6) turns this into defined, tested behaviour |
| **#3882** — Pyomo 7 preparations | a major-version window is open | §9 proposes the `contrib → core` split for that window |

External pressure is the same the scoping document names: gurobipy's matrix API (`addMVar`/`addMConstr`) and linopy demonstrate that array-native model building reaches near-C build times in Python; the gap this proposal closes is exactly the interpreted per-object / per-row tax.

---

## 3. Where the time actually goes (Phase-0 baseline)

Phase 0 built a stage-separated benchmark harness (`bench/`) over four synthetic model families the scoping document names — dense multi-period **network flow**, mixed sparse/dense **unit commitment**, capacitated **facility location** (+ a quadratic variant), and ragged **supply chain** — at 10⁴–10⁶ nonzeros, and measured four stages independently on the same built model: **construct**, **repn**, **write** (LP), **load** (APPSI HiGHS `set_instance`, no solve).

The honest end-to-end wall-clock — "empty model → the solver has it" — is the single coherent route **`build→solver = construct + load`** (the in-memory route). Measured on that route, load is the dominant stage:

| model | construct % | load % of the coherent `build→solver` route |
|---|---|---|
| network_flow (1e6) | 13% | **79%** |
| unit_commitment (1e6) | 6% | **92%** |
| facility_location (1e6) | 10% | **85%** |
| supply_chain (1e6) | 10% | **85%** |

*(Source: `bench/PHASE0_REPORT.md` §3.)*

And the same LP built directly as scipy arrays and loaded into HiGHS (`array→HiGHS`, the in-harness lower bound the fast path targets) is far cheaper than the classic Pyomo route to reach the *same* solver state:

| model | size (nnz) | Pyomo `build→solver` | array→HiGHS | **Pyomo ÷ array-native** |
|---|---|---|---|---|
| network_flow | 999,950 | 7103.3 ms | 184.8 ms | **38×** |
| facility_location | 1,000,100 | 7039.3 ms | 289.0 ms | **24×** |

*(Source: `bench/PHASE0_REPORT.md` §4. The gap widens with size — 27→38× network flow, 20→24× facility location across 1e4→1e6.)*

**The structural conclusion:** a vectorization effort that only speeds up object construction (columnar `Var`) leaves most of the wall-clock on the table. The direct array→solver hand-off is where the biggest transparent wins are. This finding shaped the whole design that follows.

Two feasibility spikes fixed the design's harder decisions:

- **Spike A (columnar `Var`):** strong GO — ~360–540× faster bulk allocation, ~0.14× memory — but it confirmed that *object identity* (`id()`-keyed repn/solver maps), not allocation, is the real correctness cost, mandating the **materialize-on-touch** starting point (§5.1).
- **Spike B (template rule vectorization):** qualified GO. Full vectorized extraction of a templatizable linear family is **13–16× faster** than the classic per-index repn and reproduces the exact CSR rows; but *resolving the template per index is 5–7× slower than classic*, and a well-defined subset of rule shapes (index conditionals, filtered sums, modulo/non-affine indexing) **does not templatize** — so a mandatory scalarization fallback is required, not optional (§5.5).

---

## 4. Goals and non-goals

**Goals** (all realized in the prototype and benchmarked):

- **G1.** Columnar indexed-variable storage with transparent per-index materialization (#202).
- **G2.** Array-expression linear/quadratic constraint and objective families (no per-index trees).
- **G3.** A repn/writer fast path: standard-form arrays spliced straight into a bulk solver hand-off (#3888).
- **G4.** Full interoperability: mixing fast-path and classic components always works; anything needing a scalar view gets one (lazily).
- **G5.** "Your old code gets fast": template-vectorized construction of existing rule-based families where they templatize.
- **G6.** A persistent, array-native warm re-solve path for rolling-horizon workloads.

**Non-goals** (unchanged from the scoping document, and respected by the prototype):

- **N1.** No rewrite of the scalar expression system; no public-API break.
- **N2.** No general nonlinear vectorization. **Quadratic is the hard ceiling** — objective-quadratic only; quadratic *constraints* are out of scope (no HiGHS API).
- **N3.** No C/C++/Rust extension modules. The hypothesis under test — that NumPy-level vectorization closes most of the gap — is what the benchmarks measure.
- **N4.** No solver-algorithm work; solvers consume the output.
- **N5.** No transformation (GDP etc.) awareness beyond the "materialize to scalar" escape hatch.

---

## 5. Design overview: what exists today

Everything below lives under `pyomo/contrib/vector/` and is importable from `pyomo.contrib.vector`. The user-facing walkthrough of each route is `pyomo/contrib/vector/README.md`; this section is the design map.

### 5.1 Columnar variables — `VectorVar` (Phase 1)

`VectorVar(index, domain=..., bounds=..., initialize=...)` stores per-index bounds/value/fixed as parallel NumPy arrays (26 B/var vs. ~170–200 B/var for classic `VarData`, Spike A). Access is **materialize-on-touch**: `m.x[i]` lazily creates a *permanent* array-backed `VectorVarData` cached in `_data[i]`, so `m.x[i] is m.x[i]` holds by construction — the load-bearing requirement, since repn and solver maps key on `id()`. The materialized view delegates every read/write straight to the parent arrays, so bulk (fast-path) and scalar views can never drift. Bulk mutation (`setlb`/`setub`/`fix`/`unfix`/`set_values`, with a `where=` selector) records touched columns *dirty* for the warm re-solve path (§5.7). Domain is homogeneous per component in Phase 1 (per-index domains deferred).

### 5.2 Explicit-array constraints and objectives — `VectorConstraint`, `VectorObjective` (Phase 1)

`VectorConstraint(A=csr, x=[VectorVar, ...], lb=..., ub=...)` (or `rhs=` for equality) states a whole linear family as one object carrying CSR coefficient arrays over the concatenated column space of one or more `VectorVar`s, with lower/upper (or ranged) bound arrays. **No per-index `ConstraintData` and no per-index expression tree is built on the fast path.** Ragged / sparse index sets are handled by construction: the caller supplies the sparse `A`, so there is no dense-box assumption (the scoping document's differentiation from dense-labelled systems).

`VectorObjective(terms={VectorVar: coef_array, ...}, sense=..., constant=..., quadratic=Q)` stores the linear cost as per-block coefficient arrays and, optionally, a convex-quadratic part as a sparse Hessian `Q` in `c @ x + 0.5 · xᵀQx` (§5.6).

### 5.3 Standard-form assembly — `assemble` / `compile_standard_form`

`assemble(model)` splices the vector components into `VectorMatrices` — a single CSR `A`, column/row bound arrays, integrality, cost, and (if present) an objective Hessian — over a stable, negotiated column ordering. This is the array-native analogue of `LinearStandardFormCompiler`; fixed variables are substituted into row bounds / objective offset and pinned out of the column space (the #3851 correctness pitfall handled once, centrally).

### 5.4 Direct solver hand-off — `load_highs` / `solve_highs` (Phase 1)

`load_highs(model)` assembles and hands the arrays to an in-process HiGHS via **one `passModel`** call (range rows natively, no per-constraint row splitting); `solve_highs(model, load_solutions=True)` also solves and scatters the primal solution back into the `VectorVar` value arrays with no per-index solver query. This is the "load prize": Phase-1 measurement reached ~1.01× the `array→HiGHS` ceiling for models built the vector way.

### 5.5 Transparent fast hand-off for *classic* models — `highs_fastload` / `gurobi_fastload` (Phase 2)

Most users will never rewrite a model to the explicit-array API. `SolverFactory('highs_fastload').solve(model)` takes an **unmodified classic** linear (or convex-QP) model, compiles it once with the stock `LinearStandardFormCompiler`, builds a `highspy.HighsLp`/`HighsModel`, hands it over in one `passModel`, solves, and maps primals / duals / reduced costs back onto the Pyomo objects. It is the HiGHS analogue of the shipped `gurobi_direct` interface's `compile → addMConstr` pattern. `gurobi_fastload` is the Gurobi twin: the *same* solver-neutral compile, handed to Gurobi's native matrix API (`addMVar`/`addMConstr`/`setMObjective`). Both are registered on the v2 and legacy `SolverFactory` at package import; both reject nonlinear / unsupported structure loudly, pointing at a classic solver route — never a silently wrong answer.

### 5.6 Template-vectorized construction — `vectorized_construction` (Phase 3)

The "your old code gets fast" milestone. `with vectorized_construction(): m = build_model(...)` (or `PYOMO_VECTOR_CONSTRUCT=1`, both **default off**) makes every `Constraint(index, rule=...)` whose rule *templatizes* build its whole constraint matrix with NumPy — the Spike-B full-vectorized-instantiation path (templatize once, extract the row skeleton, fill the CSR over the entire index set), never resolving per index. `compile_templated_to_highs_arrays` then assembles the whole model — vectorized families + stock per-row repn for the rest — over one shared column space and feeds `highs_fastload`, so construct and load stay array-shaped end-to-end with no scalarization. A rule outside the proven subset (index conditionals, filtered sums, modulo/non-affine indexing, index-dependent coefficients) falls back to classic construction **byte-identically**. This capability builds on Pyomo core's experimental, default-off `TEMPLATIZE_CONSTRAINTS` / `LinearTemplateRepnVisitor`; the only core touch is a switch-gated broadening of the templatization fallback so a rule that raises during templatization falls back instead of crashing (§6, §8).

### 5.7 Persistent, array-native warm re-solve — `FastStepHighs`, `VectorPersistentHighs` (Phase 4)

Rolling-horizon / MPC workloads do a cold build **once** and then re-solve thousands of times with slightly changed data. `FastStepHighs` (`highs_faststep`) compiles a classic linear/QP model once, retains the live HiGHS, and on each roll reads the small mutable-`Param` vector `P`, expands **every** changed objective coefficient / RHS / bound with one sparse `M @ P` (an affine template captured at `set_instance`), and batch-pushes each group with a single HiGHS call (`changeColsCost` / `changeRowsBounds` / `changeColsBounds`), keeping the warm basis. It vectorizes *both* halves of the classic warm cost — the per-coefficient `value()` walks *and* the scalar solver calls. Two update paths: model-driven (mutate Params, `solve()`) and array/mapping-free (`solve(param_values=P, dirty=mask)`, where `FastStepHighs` owns the row/column mapping). `VectorPersistentHighs` is the columnar-component twin: it drives the same incremental HiGHS path from `VectorVar`/`VectorConstraint` mutation directly (no template extraction).

---

## 6. The compatibility contract (the load-bearing design rule)

> Any consumer that does not recognize a fast-path component triggers **lazy scalarization**: the component materializes classic data objects / expressions on iteration (or when its `.expr` / rows are requested), marks itself scalarized, logs a one-time warning, and thereafter behaves exactly like today's components.

This single rule (scoping document §6.5) is what makes the fast path safe to add:

- **Identity guarantees.** `VectorVar` materialize-on-touch means `m.x[i] is m.x[i]` always holds and a materialized datum is array-backed, so the bulk and scalar views are the *same* storage — they cannot disagree. `VectorConstraint[r]` / `VectorObjective.expr` lazily build a classic `ConstraintData` + `LinearExpression` for the requested row(s).
- **Lazy scalarization is tested behaviour, not a hope.** The package's `test_scalarization.py` / `test_sparse.py` / `test_mutation.py` assert that iterating or expression-touching a vector component produces classic objects and that the resulting standard form is identical to the classic build up to row/column permutation. This is the concrete resolution of #309 for the new component types.
- **Zero classic-path change — with evidence.** Every feature is additive under `pyomo/contrib/vector/`. The template switch's core touches (a switch-gated broadened templatization fallback; an opt-in log-suppression flag; materialize-on-touch `.body`/`.lower`/`.upper` on the template data classes) change behaviour *only* when a family is templatized, i.e. only under the switch. With the switch **off**, construction is byte-identical to stock Pyomo — Phase 3 verifies this two ways (the stock compiler on both switch-on and switch-off builds; and the adjacent core module test suite, **263 tests**, stays green — `bench/PHASE3_REPORT.md` §2c/§3). A non-templatizable-heavy model with the switch *on* is within noise of the switch-off build (0.96–1.01× construct — §7).

---

## 7. Benchmark evidence

All figures are median-of-repeats from the committed synthetic suite; each `(model, size)` case runs in its own subprocess. Environment (per every report): Pyomo `6.10.2.dev0`, Python 3.12.13, Linux x86-64, numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1. Reproduction commands are in each report's *Reproducing* section and §10 here.

### 7.1 Transparent hand-off for classic models (Phase 2, `highs_fastload`)

`build→solver = construct + hand-off`. The hand-off is the only stage the fast route replaces; construct is shared. *(Source: `bench/PHASE2_REPORT.md` §3.)*

| model (1e6) | classic `build→solver` | fast `build→solver` | hand-off × | **end-to-end ×** |
|---|---|---|---|---|
| unit_commitment (load-bound) | 18.7 s | 4.15 s | 6.4× | **4.50×** |
| supply_chain | 4.96 s | 1.85 s | 3.9× | **2.68×** |
| facility_location | 7.10 s | 2.84 s | 3.4× | **2.50×** |
| network_flow (construct-bound) | 7.16 s | 3.32 s | 3.2× | **2.15×** |

The **hand-off itself is 3.1–6.4× faster across the whole suite at every size.** End-to-end, the load-bound model clears the ≥3× target (4.50×); the construct-bound models reach 2.1–2.7×, capped not by the hand-off but by the construct the two routes *share* (§8). The gain grows with how load-bound the model is — precisely where the classic route hurts most.

**Correctness:** `highs_fastload` compiles with the stock `LinearStandardFormCompiler`, so its constraint system is identical to the classic standard form *by construction*; on the suite and 40 randomized models, its objective and termination match a persistent APPSI HiGHS solve (`bench/PHASE2_REPORT.md` §4).

### 7.2 Template-vectorized construction (Phase 3)

On `resource_coupling` (a templatizable-heavy synthetic: unfiltered sum-over-set + scalar-affine coupling — the Spike-B proven subset), with the switch as the only change: *(Source: `bench/PHASE3_REPORT.md` §2.)*

| size | construct OFF | construct ON | **construct ×** | classic coherent | phase-3 `build→solver` | **end-to-end vs classic** |
|---|---|---|---|---|---|---|
| 1e5 | 65.8 ms | 9.0 ms | 7.3× | 202.9 ms | 25.6 ms | **7.9×** |
| 1e6 | 661.5 ms | 59.1 ms | **11.2×** | 1995.4 ms | 204.6 ms | **9.8×** |

Construct is **11.2× faster** at 1e6 (skips ~1M expression-tree nodes) and the whole route is **9.8× faster than the classic coherent route**, with zero model-code changes. `network_flow` is the **documented out-of-subset case**: its balance rule uses a filtered sum (`if j != n`) and an index conditional (`if t > 0`), neither of which templatizes, so Phase 3 falls back to classic construction and it stays at the Phase-2 level (~2.13×) — retained in the suite as the honest fallback exemplar. A non-templatizable-heavy model with the switch on shows no material slowdown (unit_commitment 0.96×, supply_chain 1.01×, network_flow 1.00× construct); a mixed model (`facility_location`, ~75% of nonzeros vectorize) gets 1.7× construct / 1.55× vs Phase-2 at 1e5.

### 7.3 Persistent warm re-solve (Phase 4, `highs_faststep`)

Warm tick (update + warm re-solve) on the synthetic `rolling_mpc` model vs the persistent APPSI HiGHS interface on its fastest warm path; all objectives equal across routes to < 1e-6 relative. *(Source: `bench/PHASE4_REPORT.md` §3.)*

| size (nnz) | APPSI-persistent | faststep (model) | **speedup (model)** | speedup (array) |
|---|---|---|---|---|
| 10,548 | 52.95 ms | 15.64 ms | **3.39×** | 3.64× |
| 102,360 | 685.38 ms | 294.26 ms | **2.33×** | 2.37× |

Clears the ≥1.4× target with margin. Vectorizing the evaluation (`M @ P`) *and* batching the push is what reaches ~2.3× — batching only the solver side would cap near ~1.4–1.7×, because the classic warm cost splits roughly evenly between per-coefficient `value()` walks and scalar solver calls.

### 7.4 Robustness refinements on the warm path

Three refinements widen the class of models the warm path accepts, each with negligible per-roll cost and a hard correctness gate (fail-loud on genuine change; opt-in `on_matrix_change='reload'`):

- **Value-aware static-matrix guard** (`bench/VALUEGUARD_REPORT.md`): a nominally-mutable *matrix* coefficient whose value never changes is accepted and verified each roll rather than rejected on the mutability flag — per-roll overhead **0.1–0.2%** of the warm tick at scale; the guarded leg tracks the pure-static path (1.01–1.03×).
- **Verified-static parameter folding** (`bench/STATICFOLD_REPORT.md`): a practically-constant mutable param appearing *non-affinely* (a `price·duration` product, a `duration/efficiency` reciprocal) is folded to a watched constant so the remaining coefficients become affine and the previously-rejected model engages — **5.02× / 3.49×** over APPSI-persistent on the same non-affine model at 1e4 / 1e5.
- **Near-linear `set_instance` compile** (`bench/COMPILE_SCALING_REPORT.md`): the one-time compile was super-linear on folding-heavy models (`O(folds × nnz)`); an incremental greedy fold + single-signature walk makes it **near-linear** (byte-identical output), **19.6×** faster at 1024 hub folds, and brings the 1e6-nnz folding compile to **48.2 s** (≤ 60 s target; ~20 s for the plain variant).

### 7.5 Convex-quadratic objective (#1761)

Synthetic portfolio QP (banded SPD covariance), vector fast path vs the classic Pyomo QP route (v2 `SolverFactory('highs')`), objectives matched to ~1e-16 before timing. *(Source: `bench/QUADRATIC_QP_REPORT.md` §3.)*

- **Construction: 37–58× faster** — the classic route builds one expression-tree node per Hessian nonzero; the array path builds a scipy CSC and stacks it. This is the #1761 symptom removed at the root.
- **Solve: at parity** — same convex-QP HiGHS run (e.g. 24 ms vs 20 ms at n = 2200); the array path neither speeds up nor slows down the solver. **No claim is made that the solver runs faster.**

### 7.6 Gurobi hand-off — honest license scope

The pip `gurobipy` wheel ships a **size-limited license (2000 vars / 2000 constraints)**. Every `gurobi_fastload` measurement stays strictly under it: at licensed sizes the bulk matrix hand-off is **~3.8–5.4×** faster than the per-row `set_instance`, both routes agree on the objective, and the per-row column grows faster than the bulk one as size rises. **No claim is made beyond 2000 vars/constraints, where this license cannot solve.** With `gurobipy` absent the Gurobi tests skip and CI stays green. *(Source: `bench/GUROBI_FASTLOAD_REPORT.md`.)*

---

## 8. Limitations and deferrals (stated plainly)

- **Construct-bound models cap below 3× on the transparent Phase-2 route.** This is structural: the classic and fast routes *share* construction, so when construct + the interpreted repn the fast route still runs already exceeds a third of the classic total, no transparent classic hand-off can reach 3× (network_flow at 1e6). The gain is real (2.1×) but diluted by shared construct. Phase 3 lifts this only where the rules templatize.
- **Template coverage is a proven subset, not general.** Index conditionals, **filtered sums** (`for j in J if j != n` — the everyday sparse-subset idiom), modulo/non-affine indexing, and index-dependent coefficients do **not** templatize and fall back to classic construction. Per Spike B this fallback is the *majority* path for idiomatic real models, not an edge case. Extending the subset (a masked / gathered extractor for filtered sums and conditionals) is distinct, larger, deferred work.
- **Objective templatization is deliberately off.** A scalar `Objective(expr=sum(...))` templatizes trivially but then compiles through a per-term evaluator *slower* than the classic objective walk, so it would regress otherwise-untouched models; objectives are compiled classically.
- **Quadratic is the hard ceiling.** Objective-quadratic only, convex only. **Quadratic constraints are out of scope** (no HiGHS API). **MIQP and non-convex QP** are rejected loudly (HiGHS solves neither; verified empirically).
- **Warm-path scope.** `FastStepHighs` updates objective coefficients, RHS/row bounds, and variable bounds — the rolling-horizon roll. In-place *batched* constraint-matrix (`A`) edits are deferred: the guard/fold machinery *detects* a genuine coefficient change (fail-loud, or opt-in full reload) but does not yet *apply* it incrementally; the coefficient mapping is deliberately structured as a reusable component for that later stage. A **varying** Hessian on the warm QP path is likewise deferred (static Hessian supported; genuine change → fail-loud or reload).
- **`gurobi_faststep`** (the Gurobi warm twin) and **convex MIQP** are logged as separate follow-ups, not attempted here.
- **`clone()` under the template switch** is a known limitation (deep-copying an experimental `TemplateSumExpression` recurses — a pre-existing core limitation of template expressions).
- **Classic-path QP extraction is left as-is.** The vector path adds a bulk QP hand-off; it does not modify the classic per-element QP load.

None of these are silent: each unsupported case is rejected loudly at compile or `set_instance`, pointing at a classic route, so the fast path never mis-solves.

---

## 9. Migration path and a `contrib → core` split

The prototype is intentionally an **opt-in `contrib` package** — nothing changes for any model that does not import it, and the transparent routes are chosen at solve time (`SolverFactory('highs_fastload')`) or by an off-by-default switch. That makes an incremental adoption path natural for the Pyomo 7 window (#3882):

**Tier 1 — land as `contrib` (no core change; available now).**
The whole prototype: `VectorVar`/`VectorConstraint`/`VectorObjective`, `assemble`/`load_highs`, `highs_fastload`/`gurobi_fastload`, `FastStepHighs`/`VectorPersistentHighs`. These are additive and self-contained. The benchmark harness (`bench/`) is a standalone win and a low-risk first PR — it is what gives this proposal its authority (scoping open question #6).

**Tier 2 — promote the transparent solver interfaces toward the core solver registry.**
`highs_fastload` / `gurobi_fastload` are the direct answer to #3888 and are drop-in `SolverBase` implementations mirroring the shipped `gurobi_direct` compile→matrix pattern. They are the safest promotion: no new modelling surface, chosen explicitly by name, correctness gated against the persistent interfaces. Natural coordination point with the ongoing solver-interface redesign rather than a fork.

**Tier 3 — fold columnar storage toward `IndexedComponent` (the #202 question).**
Whether columnar storage becomes a `storage="columnar"` option on `Var` generally (or stays `Var`-specific machinery first) is the open API decision #202 has carried for years. The prototype's evidence — that materialize-on-touch preserves identity at ~500× cheaper allocation and 0.14× memory — is the datapoint that decision needs. Recommend `Var`-first, behind the compatibility contract, guarded by the same lazy-scalarization tests.

**Tier 4 — the template-vectorized construction switch.**
This depends on the experimental core template machinery (`TEMPLATIZE_CONSTRAINTS`, `LinearTemplateRepnVisitor`) maturing. Recommend it stays opt-in and `contrib`-driven until the templatizable subset is widened (filtered sums / conditionals) and the `clone()` limitation is resolved in core. The switch is the right long-term "single-API" story — old rule-based code that simply gets faster — but it should graduate last.

**Open questions for maintainers** (from the scoping document §10, still live): Suffix (dual/warmstart) mapping onto vector families; CUID/naming for never-materialized data; whether the benchmark harness is upstreamed as a CI performance gate independent of everything else (likely yes).

---

## 10. Reproducing

Recreate the harness venv (steps in `bench/README.md`), then from the repo root:

```bash
# Phase-0 baseline + spikes + correctness oracle
bench/.venv/bin/python -m bench.run_bench --suite full --sizes 1e4,1e5,1e6 --out bench/results/full.json
bench/.venv/bin/python -m bench.equivalence --out bench/results/equivalence.json
bench/.venv/bin/python -m bench.spikes.spike_a_columnar_var  --out bench/results/spike_a.json
bench/.venv/bin/python -m bench.spikes.spike_b_template_expr --out bench/results/spike_b.json

# Phase-2 transparent fastload
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models network_flow,unit_commitment,facility_location,supply_chain \
    --backends pyomo --sizes 1e4,1e5,1e6 --out bench/results/phase2_fastload.json

# Phase-3 template-vectorized construction
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models resource_coupling,facility_location,network_flow,unit_commitment,supply_chain \
    --backends pyomo,pyomo_template --sizes 1e4,1e5,1e6 --out bench/results/phase3_template.json

# Phase-4 warm re-solve + robustness variants
bench/.venv/bin/python -m bench.warm_faststep --sizes 1e4,1e5 --rolls 30 --out bench/results/phase4_faststep.json

# the whole fast-path test suite (correctness gates)
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/
```

The fast-path test suite (`pyomo/contrib/vector/tests/`) is the correctness backbone of every claim above: **167 tests** (142 pass + 25 Gurobi-skip when `gurobipy` is absent, as in the harness venv). See `pyomo/contrib/vector/README.md` for the user-facing walkthrough of each route.
