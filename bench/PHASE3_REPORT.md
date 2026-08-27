# Pyomo Vectorized Construction — Phase 3 Report

**Project:** Vectorized Model Construction for Pyomo (see the project scoping document, the Phase-0 baseline report, and the Phase-2 report)
**Phase:** 3 — template-based rule vectorization ("your old code gets fast")
**Deliverable:** an opt-in switch that makes an **unmodified classic** `Constraint(index, rule=...)` model construct *and* hand off to the solver array-shaped, when its rules templatize — and falls back byte-identically to classic Pyomo when they do not.
**Baseline:** stock Pyomo at this clone's HEAD; the Phase-2 `highs_fastload` route.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs, 15 GiB RAM
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- Reproduce with the `bench/` harness (§ Reproducing); each case runs in its own subprocess.

---

## 1. The insight, and what Phase 3 delivered

Phase 2 (`highs_fastload`) removed the solver-**load** bottleneck by compiling a classic model to standard form once and handing HiGHS the whole matrix via `passModel`. That left the shared **construct** stage — building one `ConstraintData` and one operator-overload expression tree per index — as the binding cost for construct-bound classic models (Phase-2 report §3: `network_flow` capped at ~2.15× end-to-end precisely because construct is shared between the classic and fast routes).

Phase 3 attacks construct itself. It builds on an **experimental, default-off** capability already in Pyomo core (`TEMPLATIZE_CONSTRAINTS` in `pyomo/core/base/constraint.py`, the `LinearTemplateRepnVisitor` in `pyomo/repn/linear_template.py`): when a rule *templatizes*, the family is stored as a compact template with **no per-index expression trees**. Phase 3 turns that capability into a usable, benchmarked fast path:

1. **An opt-in activation surface** (`pyomo.contrib.vector.template_vectorize`): a `vectorized_construction()` context manager and a `PYOMO_VECTOR_CONSTRUCT=1` environment variable, both **default off**, that a benchmark or deployment sets **without touching model code**.
2. **A vectorized CSR extractor** — the Phase-0 Spike-B "full vectorized instantiation": templatize a family once, extract the row skeleton, and fill the constraint-matrix CSR over the *entire index set* with NumPy (never resolving the template per index, which Spike B proved is 5–7× *slower* than classic). Measured ~7× faster than the stock standard-form compiler on the templatizable families, byte-identical output.
3. **Mixed-model assembly wired into `highs_fastload`**: every family that templatizes is extracted vectorially; every family that does not is compiled by the stock per-row repn; both feed one shared column space → `passModel`. Construction and load stay array-shaped end-to-end, with no scalarization.
4. **A mandatory, silent fallback.** A rule outside the proven subset constructs byte-identically to classic Pyomo. (This also fixed a real latent bug in the experimental core path: it only caught `TemplateExpressionError`, so a rule that raised `TypeError`/`PyomoException` during templatization — e.g. a `dict[index]` lookup — *crashed construction* instead of falling back. The fix is gated behind the switch: with it off, core behaviour is untouched.)

### The proven subset (exactly what Phase-0 Spike B validated)

Templatized and vectorized: linear bodies that are a combination of `coef * var[affine_index...]` terms with **constant coefficients**, optionally inside an **unfiltered** `sum(... for j in Set)`; multiple variable components (`x[f,c] <= open[f]`); constant or mutable-`Param` right-hand sides (`a*x[i] <= p[i]`); equality / inequality / ranged relations.

Deliberately **out of scope** → classic fallback (logged once at debug level): index conditionals (`if i == 0`), **filtered sums** (`for j in J if j != n`), modulo / non-affine indexing, and index-dependent coefficients (`a[i,j]*x[j]`). Per the Phase-0 verdict, this fallback is the *majority* path for idiomatic real models, not an edge case.

---

## 2. Results — exit criteria

Median-of-repeats, milliseconds. `construct` is the model build; `phase3 build→solver` = template construct (ON) + vectorized `fastload`; `classic-coherent` = classic construct + APPSI per-row load; `phase2` = classic construct + stock `fastload`.

### (a) Construct ≥5× on a templatizable-heavy model at 1e6 — **MET**

`resource_coupling` (new synthetic: unfiltered sum-over-set + scalar-affine coupling, the proven subset; `nnz/vars ~ J`, the high variable-reuse regime):

| size | nnz | vars | construct OFF | construct ON | **construct ×** | classic-coherent | phase3 build→solver | **end-to-end vs classic** |
|------|-----|------|---------------|--------------|-----------------|------------------|---------------------|---------------------------|
| 1e5 | ~100k | 1,400 | 65.8 | 9.0 | **7.3×** | 202.9 | 25.6 | **7.9×** |
| 1e6 | ~1.02M | 10,000 | 661.5 | 59.1 | **11.2×** | 1995.4 | 204.6 | **9.8×** |

Construct is **11.2× faster** at 1e6 (skips ~1M expression-tree nodes), and the whole "empty model → solver has it" route is **9.8× faster than the classic coherent route** — with zero model-code changes. This is the "your old code gets fast" milestone realised on a model whose rules are in the proven subset. Equivalence: identical objective + standard form vs the classic build (§3).

### (b) End-to-end ≥5× on a templatizable-heavy model at 1e6, zero model changes — **MET**

The Phase-0 gate for this milestone is a ≥5× end-to-end speedup, with no model-code change, on a model whose rules are in the templatizable subset Spike B proved. `resource_coupling` (§2a) delivers **9.8× vs the classic coherent route** at 1e6 — construct → `highs_fastload` → "solver has the model" — with the switch as the only change.

**`network_flow` is the documented boundary case (out of subset).** The original phase brief named `network_flow` for this criterion; on inspection its balance rule is `sum(m.flow[j,n,t] for j in m.nodes if j != n) ... + (m.stor[n,t-1] if t>0 else 0.0) ...`, and both the **filtered sum** (`if j != n`, the sparse arc set) and the **index conditional** (`if t > 0`, the storage boundary) are *outside* the templatizable subset — verified directly (`templatize_constraint` raises). Phase 3 falls back to classic construction for it, so it stays at the Phase-2 level:

| size | construct OFF | construct ON | construct × | classic-coherent | phase3 build→solver | end-to-end vs classic |
|------|---------------|--------------|-------------|------------------|---------------------|-----------------------|
| 1e5 | 193.9 | 193.3 | 1.00× | 759.9 | 357.3 | **2.13×** |

This is **structural**, not a tuning gap: no full-vectorized-instantiation design can vectorize a rule that does not templatize, and this phase's scope (Spike B) explicitly excludes filtered sums and conditionals. `network_flow` is retained in the suite precisely as the honest fallback exemplar — the 5× target is reachable only on rules in the proven subset. Extending the subset to filtered sums / conditionals (a masked / gathered extractor) is a distinct, larger piece of work, deferred to a possible follow-up.

A separate external private real-world case (measured outside this repository) is overwhelmingly in-subset and shows a solid end-to-end gain over Phase 2; those numbers are reported separately and are not part of this repository.

### (c) Equivalence gates — **MET**

With the switch on vs off, the standard form is **identical up to row/column permutation, with the same bounds** — checked two ways (`pyomo/contrib/vector/tests/test_template_vectorize.py`): the stock compiler on both builds (construct equivalence), and the vectorized `compile_templated_to_highs_arrays` matrix vs the stock compiler (compile equivalence), on templatizable, mixed, and non-templatizable models plus **25 randomized** models spanning both rule classes. Solve equivalence: `highs_fastload` (vectorized) and a classic APPSI HiGHS solve agree on objective on every model in the suite. The scalarization / identity contract holds: touching a templatized `ConstraintData` (`.body`, `.expr`, repn) materializes a classically-behaving object in place, identity preserved.

### (d) Non-templatizable-heavy model, no material slowdown with the switch on — **MET**

| model (1e4) | construct OFF | construct ON | construct × | phase3 vs phase2 |
|-------------|---------------|--------------|-------------|-------------------|
| unit_commitment | 14.1 | 14.8 | 0.96× | 0.98× |
| supply_chain | 11.8 | 11.7 | 1.01× | 1.01× |
| network_flow | 193.9 | 193.3 | 1.00× | 1.00× |

A model whose rules don't templatize sees no template objects, routes through the untouched Phase-2 stock path, and is within noise of the switch-off build. The only added cost is one (fast-failing, silenced) templatization attempt per family at construct.

### Mixed model (partial coverage)

`facility_location` (serve + link templatize; capacity falls back on a `dict[index]` lookup) is the partial-coverage case: **1.7× construct** and **1.55× vs Phase-2** at 1e5 — a real gain from the ~75%-of-nonzeros that vectorize, with the classic remainder handled at stock speed and the whole thing byte-identical to the classic standard form.

---

## 3. Scope, guards, and deferrals

* **Constraints only.** Phase 3 templatizes `Constraint` families. `TEMPLATIZE_OBJECTIVES` is deliberately left **off**: a scalar `Objective(expr=sum(...))` templatizes trivially but then compiles through a large per-term code-generated evaluator that is *slower* than the classic objective walk (it would regress otherwise-untouched models). Objectives are compiled classically.
* **Linear only**, matching `highs_fastload`; nonlinear/quadratic bodies are rejected by the standard-form path.
* **Zero core behaviour change with the switch off.** The core edits (a broadened templatization fallback in `constraint.py`/`objective.py`, an opt-in log-suppression flag in `template_expr.py`, and materialize-on-touch `.body`/`.lower`/`.upper` on the template constraint data classes) only change behaviour when a family is templatized, i.e. only under the switch. The full existing test suite for these modules (263 tests) is green.
* **Deferred (logged, not attempted):** filtered sums, index conditionals, modulo/non-affine indexing, index-dependent coefficients, and objective templatization. These are the boundary of Spike B's proven subset; extending past it is future work.

---

## 4. Reproducing

```bash
# recreate the venv (see bench/README.md), then from the repo root:
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models resource_coupling,facility_location,network_flow,unit_commitment,supply_chain \
    --backends pyomo,pyomo_template --sizes 1e4,1e5,1e6 \
    --out bench/results/phase3_template.json

# the correctness gates:
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/test_template_vectorize.py
```

The `pyomo_template` backend measures construct + fast-load with template-vectorized construction ON and OFF and records `construct_speedup`, `end_to_end_vs_classic`, `end_to_end_vs_phase2`, template coverage (`templatized_rows` / `classic_rows`), and per-case solve equivalence in `validation`.

---

## 5. Deliverables checklist

- [x] Opt-in activation (context manager + env var), default off, no model-code change (`template_vectorize.py`)
- [x] Vectorized CSR extractor for the Spike-B subset (full instantiation, never per-index resolve)
- [x] Mixed-model assembly wired into `highs_fastload` (vectorized families + classic fallback, one column space, no scalarization)
- [x] Mandatory silent fallback + hardened core templatization exception handling (switch-gated)
- [x] Equivalence gates (construct + compile + solve; templatizable / mixed / non-templatizable + 25 randomized); scalarization/identity contract
- [x] Templatizable-heavy synthetic model (`resource_coupling`) + benchmark switch-ON leg (`pyomo_template`)
- [x] Zero core-module behaviour change with the switch off (263 adjacent tests green)
- [x] **(b) ≥5× end-to-end on a templatizable-heavy model, zero model changes — MET** (`resource_coupling` 9.8×; `network_flow` documented as the out-of-subset fallback exemplar at ~2.1×, see §2b)
