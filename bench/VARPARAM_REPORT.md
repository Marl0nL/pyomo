# Pyomo Vectorized Construction — Transparent Columnar Var/Param Report

**Project:** Vectorized Model Construction for Pyomo (see the scoping doc, the Phase-0 baseline report, and the Phase-1..4 reports)
**Deliverable:** extend the Phase-3 opt-in construction switch so that an **unmodified classic** `Var(index, ...)` and `Param(index, ...)` with vectorizable arguments are built into **NumPy columns** (materialize-on-touch) instead of one Python data object per index — closing the residual cold construct that Phase-3 left behind once the *constraint* families were vectorized.
**Baseline:** the Phase-3 template-vectorized construction switch (constraints only); the Phase-2 `highs_fastload` route.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD, branched off `vectorisation`)
- Python 3.12, Linux x86-64 · numpy 2.5 · scipy 1.18 · highspy 1.15
- Each timing is the median of repeated cold builds in one process; reproduce with the `bench/` harness (§ Reproducing).

---

## 1. The residual, and what this change delivers

Phase 3 made a classic `Constraint(index, rule=...)` family construct array-shaped when its rule templatizes. That collapsed the constraint-tree cost, but **`Var` and `Param` construction stayed classic** — one `VarData` / `ParamData` object per index, plus per-index index-set validation. Profiling the post-Phase-3 cold construct at 1e6 (§2) confirmed these are now the dominant residual:

- On `resource_coupling` (the Phase-3 templatizable-heavy model), switch-on construct is ~60 ms at 1e6, of which **`Param` construction is ~69%** and `Var` ~7% (the model has high variable reuse — only 10k vars/params for ~1M constraint nonzeros — so the *object allocation*, not the matrix, is what is left).
- On a **variable-heavy** model (`nnz ~ n_var`), `Var` + `Param` construction together are ~76% of the post-Phase-3 construct.

This change routes such construction into the Phase-1 columnar machinery, under the **same** Phase-3 switch, default **OFF**:

1. **Transparent columnar `Var`** (`varparam.TransparentVectorVar`, an `IndexedVar` subclass reusing Phase-1's `_ColumnarVarMixin`): `bounds` / `domain` / `initialize` that are scalars / mappings are evaluated into NumPy columns; no `VarData` is created until an index is touched (`m.x[i]` materializes a stable-identity `VarData`; iterating scalarizes — the Phase-1 compatibility contract).
2. **Transparent columnar `Param`** (`varparam.TransparentVectorParam`): mutable Params materialize an array-backed `VectorParamData` on touch (identity stable); immutable Params serve raw values straight from the column. Domain validation is done in bulk.
3. **A mandatory, silent fallback.** A genuinely per-index callable (a `bounds`/`initialize`/`domain` rule, a `validate=` rule, a united Param, a non-interval domain) constructs **byte-identically to classic Pyomo** — the Spike-B design: partial coverage + a mandatory fallback.
4. **Array-native compile.** The Phase-3 whole-model compile (`compile_templated_to_highs_arrays`) reads columnar Var bounds/integrality/fixed and columnar Param values **in bulk** and maps the solution back with a **bulk scatter** (`column_scatter`) — so the construction win survives into the load with **no scalarization**. The map-back is shared by the HiGHS and Gurobi fast loaders.
5. **Zero core-module change.** Activation is a runtime monkeypatch of `Var.construct` / `Param.construct` installed only while the switch is on and removed when off, keyed on the existing `vectorized_construction()` context / `PYOMO_VECTOR_CONSTRUCT` env var. **No `pyomo/core/*` file is edited**, so with the switch off the original methods are literally in place.

---

## 2. Results — exit criteria

`construct` = building the model; `build+compile` = construct + the vectorized `compile_templated_to_highs_arrays` (the "empty model → solver-ready arrays" cost). `phase3-alone` = the Phase-3 switch with **only** constraints vectorized (classic Var/Param); `phase3+varparam` = this change.

### (a) Construct ≥2× at 1e6 on the templatizable-heavy model — **MET**

`resource_coupling` (median of 9), construct:

| size | classic | phase3-alone | **phase3+varparam** | **× vs phase3-alone** | × vs classic |
|------|---------|--------------|---------------------|-----------------------|--------------|
| 1e4 | 8.1 ms | 2.4 ms | 1.6 ms | 1.53× | 5.2× |
| 1e5 | 68 ms  | 8.9 ms | 3.4 ms | **2.65×** | 20× |
| 1e6 | 667 ms | 59.6 ms | 21.2 ms | **2.81×** | 31× |

At 1e6 the transparent columnar Var/Param construction is **2.81× faster than the Phase-3 switch alone** (31× vs classic) — it removes the ~40 ms of per-index `ParamData`/`VarData` allocation and index-set validation that Phase-3 left as the binding residual. The fast columnar `__getitem__` (skips per-access index validation) is what lets the win survive the *non-templatized* objective, which still reads every variable once.

End-to-end (build + compile), 1e6: phase3-alone ~133 ms → **phase3+varparam ~80 ms (≈1.65×)**; the remaining compile cost is the Phase-3 CSR extraction of the ~1M constraint nonzeros, which this change does not touch.

### (b) Standard-form + solve equivalence (switch on vs off) — **MET**

- **Matrix equivalence:** the vectorized compile over columnar components produces the same standard form (row/col signature + bounds) as the stock `LinearStandardFormCompiler` on the classic build — the existing Phase-3 equivalence gates (`test_template_vectorize.py`), now exercising columnar Var/Param, are green on templatizable / mixed / non-templatizable models plus **25 randomized** models.
- **Solve equivalence:** `highs_fastload` over a columnar-built model and over the classic build agree on the objective and every primal value; the columnar Var is **not scalarized** by the solve (bulk map-back). (`test_varparam.py::TestEquivalence`.)
- **Identity + scalarization contracts:** `m.x[i] is m.x[i]`, `m.p[i] is m.p[i]` (mutable); iterating a columnar component materializes every entry with identity preserved and logs the compatibility warning. (`test_varparam.py::TestContracts`.)

### (c) Switch off = byte-classic — **MET**

No `pyomo/core/*` file is modified; the construct patch is present only inside the switch. With the switch off, `Var`/`Param` are the stock `IndexedVar`/`IndexedParam` and the core `test_var.py` + `test_param.py` suites (851 tests) are green, as are `test_standard_form.py` and the adjacent core/solver suites.

### (d) All existing vector tests green — **MET**

The `pyomo/contrib/vector/tests/` suite is **163 passed, 25 skipped** (142 pre-existing + 21 new `test_varparam.py`); the Phase-1 refactor that extracted `_ColumnarVarMixin` kept the VectorVar/scalarization tests unchanged and green.

### Variable-heavy regime (`columnar_stress`, `nnz ~ n_var`)

The complement to `resource_coupling`: many rows, each variable in O(1) constraints, so the residual is the Var/Param object allocation **plus** the (non-templatized) objective build over every variable. Construct at 1e6: phase3-alone ~2.13 s → **phase3+varparam ~1.34–1.41 s (≈1.5–1.6×)**; build+compile ≈1.54×. The Var/Param construction itself is removed (~0.75 s saved), but here the templatized-constraint construct (660k rows) and the objective build dilute the ratio below 2× — an honest boundary: the ≥2× milestone lands where Var/Param construction is the dominant residual, which `resource_coupling` (few rows, light objective) is and `columnar_stress` is not. Solve equivalence + no-scalarization hold at scale.

---

## 3. Scope, guards, and deferrals

* **Vectorized subset (Var):** finite dense index; a constant (homogeneous) `domain`; constant `bounds`; `initialize` that is constant or a mapping. **Fallback → classic** for: a per-index callable `bounds`/`initialize`/`domain`, `dense=False`, a non-finite index, `VarList`, scalar `Var`.
* **Vectorized subset (Param):** finite index; `initialize` constant or a mapping; scalar `default`; `domain` that is `Any` or a numeric interval (bulk-validated). **Fallback → classic** for: a per-index callable rule/default, a `validate=` rule, a united Param, a non-interval domain.
* **Objectives are not templatized** (unchanged from Phase 3): a non-templatized `Objective(expr=sum(...))` reads each variable once; the fast columnar `__getitem__` keeps that read as cheap as the classic dict lookup, but it is not itself vectorized.
* **`clone()`** of a columnar-constructed model is not supported (same limitation as the Phase-3 templatized model); the construct→compile→solve fast path never clones.
* **Deferred (logged, not attempted):**
  - **`faststep` (warm re-solve) over a switch-ON model.** ~~`FastStepHighs.set_instance` on a model whose constraints were *templatized* fails today (`'IndexedConstraint' has no attribute 'body'`)~~ **— RESOLVED** (`FASTSTEP_TEMPLATIZED_REPORT.md`). The mutable-update plan now has a template-aware constraint-slot collector: a static templatized family is skipped without materializing its rows, a mutable-RHS family feeds the shared folding/guard/self-check machinery, and columnar Params are read back in one vectorized gather. A switch-ON build warm-solves bit-for-bit the switch-OFF run; `set_instance` is no slower and the warm tick is faster. `faststep` over a classic (switch-off) model is unaffected.
  - **Bulk map-back at 1e6+ vars** is done via `set_values` scatter (no per-index object); the *partial* `get_vars(vars_to_load=...)` / `get_reduced_costs(...)` cold paths materialize only the requested columnar `VarData`.
  - Per-index (heterogeneous) Var domains and array/dict `bounds` are not vectorized (→ classic fallback).

---

## 4. Reproducing

```bash
# construct + build→solver timing, template switch ON vs OFF, per size:
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models resource_coupling,columnar_stress \
    --backends pyomo,pyomo_template --sizes 1e4,1e5,1e6 \
    --out bench/results/varparam.json

# correctness gates:
bench/.venv/bin/python -m pytest \
    pyomo/contrib/vector/tests/test_varparam.py \
    pyomo/contrib/vector/tests/test_template_vectorize.py
```

`columnar_stress` (new, variable-heavy `nnz ~ n_var`) joins `resource_coupling` (variable-reuse-heavy) so both residual regimes are on the record.

---

## 5. Deliverables checklist

- [x] Transparent columnar `Var` construction under the Phase-3 switch (scalar/array/dict args; per-index callable → classic fallback), reusing Phase-1 `_ColumnarVarMixin` (materialize-on-touch, identity, scalarization)
- [x] Transparent columnar `Param` construction (mutable → array-backed `VectorParamData`; immutable → raw column values; bulk domain validation; per-index rule/validate/units → classic fallback)
- [x] Array-native compile: bulk columnar Var bounds/integrality/fixed + bulk columnar Param values + bulk-scatter solution map-back (HiGHS **and** Gurobi loaders), no scalarization
- [x] **(a)** construct ≥2× at 1e6 on `resource_coupling` (2.81×; 31× vs classic); end-to-end reported (~1.65×)
- [x] **(b)** standard-form + solve equivalence (on vs off), identity + scalarization contracts — green (incl. 25 randomized)
- [x] **(c)** switch off = byte-classic; zero `pyomo/core/*` change; core `test_var`/`test_param` (851) green
- [x] **(d)** existing vector suite green (163 passed, 25 skipped) + 21 new `test_varparam.py`
- [x] Variable-heavy synthetic model (`columnar_stress`) for the `nnz ~ n_var` regime + honest boundary framing
