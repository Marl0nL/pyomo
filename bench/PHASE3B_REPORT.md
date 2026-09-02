# Pyomo Vectorized Construction — Phase 3b Report

**Project:** Vectorized Model Construction for Pyomo (see the scoping document, the
Phase-0 baseline report, and the Phase-2 / Phase-3 reports)
**Phase:** 3b — extend template-vectorized construction to **filtered sums** and
**index conditionals**
**Deliverable:** the Phase-3 opt-in switch now also vectorizes rules containing
filtered generator sums (`sum(m.f[j, n] for j in N if j != n)`), index
conditionals in the body (`m.s[n, t-1] if t > 0 else 0`), and `Constraint.Skip`
under an index predicate — via a **masked / gathered** extractor — and still
falls back byte-identically to classic Pyomo for anything outside the proven
subset.
**Baseline:** stock Pyomo at this clone's HEAD; the Phase-2 `highs_fastload`
route; the Phase-3 template-vectorized construction.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs, ~15 GiB RAM
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- Same machine as the Phase-3 report. Reproduce with the `bench/` harness
  (§ Reproducing); medians of repeats, milliseconds.

---

## 1. What Phase 3b adds, and how

Phase 3 vectorized the **proven Spike-B subset**: linear bodies that are a
combination of `coef * var[affine_index...]` terms with constant coefficients,
optionally inside an **unfiltered** `sum(... for j in Set)`, with constant /
mutable-`Param` right-hand sides. It explicitly **deferred** filtered sums and
index conditionals to classic fallback, and documented `network_flow` as the
boundary exemplar that therefore stayed classic.

Phase 3b extends the *proven* subset to the two shapes that dominate idiomatic
sparse/temporal models, keeping full vectorized instantiation (Spike-B law 1: run
the rule through the template machinery **once**, extract a row skeleton, fill the
CSR with NumPy — never resolve the template per index):

1. **Filtered generator sums** whose filter is a comparison or a **conjunction**
   of comparisons on index values: `if j != n`, `if j < i`, `if t > 0`, `if j !=
   n and j >= 1`, chained `if a if b`. The predicate is captured symbolically
   during templatization (a `__bool__` hook on relational expressions, active only
   under the switch) and attached to the `TemplateSumExpression`; the extractor
   evaluates it as a **NumPy boolean mask over the sum grid** and gathers the
   surviving `(row, col, coef)` triples. This is exactly the sparse-arc pattern
   `for j in nodes if j != n` — the arc set membership expressed as a filter.

2. **Index conditionals in the body** — `(<term> if <index predicate> else
   <term/const>)` — and **`Constraint.Skip` under an index predicate**. Python
   evaluates only the taken branch, so a single templatization cannot see both.
   Phase 3b **replays** the rule once per truth-assignment (polarity combination)
   of the conditional predicates, each replay forcing them to fixed booleans and
   yielding an ordinary branch-free template; at extraction every row is routed —
   vectorially — to the template of the combination its index satisfies, and a
   combination that returns `Constraint.Skip` drops the row.

3. **Composition.** Filters and conditionals compose with each other and with
   everything Phase 3 already handled (multiple `Var` components, mutable-`Param`
   RHS, equality / inequality / ranged relations). The full `network_flow` *body*
   shape now vectorizes.

Both are **opt-in and default OFF** (same activation surface as Phase 3: the
`vectorized_construction()` context manager / `PYOMO_VECTOR_CONSTRUCT=1`), with
**zero stock behaviour change** when the switch is off.

### Mandatory, safe fallback (unchanged contract)

Anything still outside the subset falls back to classic per-index construction,
byte-identically, silently (one debug note at most):

* **Disjunctive / negated filters** (`if j < n or j > n`, `if not (j == n)`) —
  Python short-circuits `or`/`not`, so an always-True capture would silently drop
  a clause and build a *too-loose* mask. Phase 3b detects this (the number of
  comparison opcodes in the generator must equal the number of predicates
  captured) and falls back rather than produce a wrong result.
* **Set membership** `(i, j) in m.arcs` (raises on the unhashable IndexTemplate),
  **modulo / non-affine indexing**, **index-dependent coefficients**, **nested /
  data-dependent conditionals** (the replay's predicate set is inconsistent
  across polarities), and **objective templatization** — all remain classic.

The equivalence law holds through **every** path (the vectorized extractor, the
stock `LinearStandardFormCompiler` template codegen, and materialize-on-touch),
so a switch-on model is byte-identical to the classic build whether it vectorizes
or falls back.

---

## 2. Results — exit criteria

`construct` is the model build; `phase3 build→solver` = template construct (ON) +
vectorized `fastload`; `classic-coherent` = classic construct (OFF) + APPSI per-row
load; `phase2` = classic construct + stock `fastload`.

### (a) `network_flow` shape now templatizes; construct + end-to-end speedup

The Phase-3 brief named `network_flow` for this criterion. On inspection,
`network_flow` has **three** independent templatization blockers, not the two the
Phase-3 report named:

1. the **filtered sums** `for j in nodes if j != n` — **now covered** (3b);
2. the **index conditional** `m.stor[n, t-1] if t > 0 else 0` — **now covered** (3b);
3. its right-hand side `_demand(n, t, N)` = `1 + 0.25*sin(0.3*t + n)` (with an
   `if n == 0` Python branch) — a **transcendental Python function of the index**.

Blocker 3 is a genuine per-index Python computation (`float()` of an index
expression raises `TypeError` during templatization) and is **out of the 3b
scope** (index-dependent non-affine constant term / arbitrary Python callable,
Spike-B law 2). It fires *before* the rule returns, so `network_flow` **as
written cannot templatize** regardless of how well filters and conditionals are
handled — this is structural, not a tuning gap. Per the brief, we **do not tune
the model**; `network_flow` stays classic and is retained as the honest boundary
exemplar (§2e).

To measure the criterion on the **exact `network_flow` body**, `flow_masked` is
`network_flow` with the *single* change that isolates blocker 3: the demand is
**precomputed into a mutable `Param`** (values identical to `network_flow`), so
the RHS is a `Param` look-up (already in the Phase-3 subset). Every part of its
balance rule is now in the proven subset, and the whole family vectorizes:

**`flow_masked`** (filtered sums + storage conditional + sparse arcs + Param RHS,
`nnz ≈ 2·N²·T`, `nnz/vars ≈ 2` — a low-reuse regime):

| size | nnz | vars | construct OFF | construct ON | **construct ×** | classic-coherent | phase3 build→solver | **end-to-end vs classic** |
|------|-----|------|---------------|--------------|-----------------|------------------|---------------------|---------------------------|
| 1e4 | 9,990   | 5,000   | 17.4   | 10.1  | **1.72×** | 77.9   | 22.4   | **3.48×** |
| 1e5 | 99,980  | 50,000  | 159.7  | 88.7  | **1.80×** | 716.0  | 213.9  | **3.35×** |
| 1e6 | 999,950 | 500,000 | 1620.8 | 955.3 | **1.70×** | 7373.9 | 2673.3 | **2.76×** |

The **end-to-end target (≥3×) is met at 1e4 and 1e5** and reaches **2.76× at
1e6**, with **zero model-code changes** beyond moving the demand into a Param.
The **construct-stage 3× is *not* reached** — construct is **~1.7×** — and this is
reported honestly rather than tuned: `flow_masked` has `nnz/vars ≈ 2`, so its
construct is dominated by building the **500k-entry `flow` Var**, a cost the
switch does not vectorize (it vectorizes the *constraint* build). The constraint
family itself vectorizes fully (0 classic rows); the shared Var build is the
floor. The end-to-end win comes chiefly from the vectorized `fastload`
(1718 ms) replacing the APPSI per-row load (5753 ms).

**The construct-stage win is large when constraints, not variables, dominate
construct.** `coupling_filtered` is the same high-reuse shape as the Phase-3
`resource_coupling` (`nnz/vars ≈ J`) but with a **filtered** coupling sum
(`sum(x[z, j] for j in acts if j != i)`):

**`coupling_filtered`** (filtered sum, `nnz/vars ≈ J`):

| size | nnz | vars | construct OFF | construct ON | **construct ×** | classic-coherent | phase3 build→solver | **end-to-end vs classic** |
|------|-----|------|---------------|--------------|-----------------|------------------|---------------------|---------------------------|
| 1e4 | 9,300     | 300    | 7.8   | 1.4  | **5.6×**  | 24.9   | 4.0   | **6.2×**  |
| 1e5 | 99,400    | 1,400  | 68.2  | 3.3  | **20.7×** | 205.0  | 21.9  | **9.4×**  |
| 1e6 | 1,010,000 | 10,000 | 665.2 | 20.6 | **32.3×** | 1982.4 | 178.5 | **11.1×** |

On a filtered-sum model whose nonzeros dominate construct, Phase 3b delivers
**32× construct** and **11× end-to-end** at 1e6 — the "your old code gets fast"
milestone realised on the newly-covered filtered-sum shape, byte-identical to the
classic standard form.

### (b) No regression on previously-templatizable models

`resource_coupling` (the Phase-3 proven-subset headline) re-measured on this
machine, with the mask/gather generality now in the path:

| size | construct OFF | construct ON | construct × | end-to-end vs classic | Phase-3 report (e2e) |
|------|---------------|--------------|-------------|-----------------------|----------------------|
| 1e5 | 65.9  | 3.2  | **20.6×** | **9.65×**  | 7.9× |
| 1e6 | 637.0 | 18.4 | **34.6×** | **11.57×** | 9.8× |

The plain (unfiltered) templatizable path is **not slowed** by the added
predicate machinery — the extra work happens only when a filter or conditional is
actually present. The numbers reproduce (and, in this run, slightly exceed) the
Phase-3 report's speedups; the switch-OFF baselines match Phase 3 (65.9 vs 65.8,
637 vs 661 ms).

### (c) Equivalence gates — all green, including new shapes

`pyomo/contrib/vector/tests/test_template_vectorize.py` (17 tests) — with the
switch on vs off, the standard form is identical up to row/column permutation with
the same bounds, checked two ways (the stock compiler on both builds, and the
vectorized `compile_templated_to_highs_arrays` vs the stock compiler), on:

* `_filtered_sum_model` (`!=`, `<`, and conjunction filters),
* `_conditional_model` (`(term if pred else const)`, and two independent
  conditionals → `k=2` polarity combinations),
* `_skip_model` (`Constraint.Skip` under `t == 0`; the skipped rows are dropped in
  both builds — `len(con)` matches),
* `_filtered_conditional_model` (the network-flow body shape: filtered sums +
  conditional + sparse arcs + mutable-Param RHS),
* the Phase-3 `templatizable` / `mixed` models and **25 randomized** models,
* **deferrals** (`or` / `not` filters, nested conditionals) — asserted to fall
  back to classic `ConstraintData` **byte-identically**,
* **`network_flow` itself** — asserted to fall back byte-identically (the
  transcendental RHS), proving the boundary is handled, not broken.

Solve equivalence (`highs_fastload` vs a second `highs_fastload` build) agrees on
objective for every model. The scalarization / identity contract is unchanged:
touching a templatized `ConstraintData` re-runs the original rule for that index
(so filtered sums / conditionals materialize correctly per index) and converts the
datum to a classic object in place.

### (d) Non-templatizable-heavy model — no Phase-3b overhead

Construct on/off on models that do **not** templatize (the switch's only added
cost is a fast-failing templatization attempt per family):

| model (1e5) | construct OFF | construct ON | construct × |
|-------------|---------------|--------------|-------------|
| network_flow    | 149.9 | 154.0 | 1.03× |
| unit_commitment | 128.4 | 120.2 | 0.94× |
| supply_chain    | 96.7  | 92.8  | 0.96× |

Construct is within noise — the masked templatization fails fast (e.g.
`network_flow`'s RHS raises during the first, discovery, pass, before any replay).
**Phase 3b's constraint-side change adds no compile-time overhead either**: with
only the constraint switch and the columnar-`Var` feature disabled, the vectorized
`fastload` compile of an all-fallback `network_flow` is **18.77 ms vs 18.6 ms**
switch-off — identical. (A residual ~15–24 % fastload cost on fully-fallback
models is the **pre-existing Phase-3 columnar-`Var`** feature materializing under
the stock compiler when the constraints do not templatize; it is orthogonal to
this task and unchanged by it.)

### (e) `network_flow` — the honest boundary

`network_flow` (unchanged) stays classic and byte-identical to the switch-off
build, at the Phase-2 level:

| size | construct OFF | construct ON | construct × | templatized rows | note |
|------|---------------|--------------|-------------|------------------|------|
| 1e5 | 149.9  | 154.0  | 0.97× | 0 | filtered sums + conditional covered; RHS `sin(index)` out of scope |
| 1e6 | 1690.8 | 1675.6 | 1.01× | 0 | — |

Two of its three blockers (the filtered sums, the conditional) are now covered;
the third — an in-rule transcendental of the index — is a genuine per-index Python
computation that no full-vectorized-instantiation design can vectorize, and is
outside the 3b scope by construction. `flow_masked` (§2a) is the same model with
that one value precomputed into a Param, and it vectorizes fully — isolating the
in-rule transcendental as the *sole* reason `network_flow` itself does not.

---

## 3. Scope, guards, and deferrals

* **Covered (vectorized):** filtered sums with comparison / conjunction predicates
  on index values; index conditionals `(term if index-pred else term/const)`;
  `Constraint.Skip` under an index predicate; all composing with multiple Var
  components, mutable-Param RHS, and equality / inequality / ranged relations.
* **Deferred (classic fallback, tested):** disjunctive / negated filters
  (`or`, `not`), set-membership filters, modulo / non-affine indexing,
  index-dependent coefficients, nested / data-dependent conditionals, and any
  predicate that touches a variable. Objective templatization stays off.
* **Zero core behaviour change with the switch off.** The core edits — an opt-in
  `__bool__` hook on relational expressions, a capture-gated `IndexTemplate.__ne__`,
  an optional filter on `TemplateSumExpression`, a filter guard in the template
  codegen (`linear_template`), and a masked-family construct path
  (`Constraint._build_masked_template`) — are all gated on the switch; with it off
  the hook is `None` and behaviour is byte-identical to stock. The adjacent core
  test suites (`test_template_expr`, `test_linear_template`, `test_relational_expr`,
  `test_con`, `test_standard_form`, `test_numeric_expr`) are green (620 tests).

---

## 4. Reproducing

```bash
# recreate the venv (see bench/README.md), then from the repo root:

# (a) filtered-sum + conditional coverage + no-regression, small/medium:
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models flow_masked,coupling_filtered,resource_coupling,network_flow,unit_commitment,supply_chain \
    --backends pyomo_template --sizes 1e4,1e5 \
    --out bench/results/phase3b_small.json

# (a)/(b) at 1e6:
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models flow_masked,network_flow --backends pyomo_template --sizes 1e6 \
    --out bench/results/phase3b_1e6a.json
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models coupling_filtered,resource_coupling --backends pyomo_template --sizes 1e6 \
    --out bench/results/phase3b_1e6b.json

# the correctness gates:
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/test_template_vectorize.py
```

The `pyomo_template` backend records `construct_speedup`, `end_to_end_vs_classic`,
`end_to_end_vs_phase2`, template coverage (`templatized_rows` / `classic_rows`),
and per-case solve equivalence in `validation`. New bench models:
`bench/models/flow_masked.py`, `bench/models/coupling_filtered.py`.

---

## 5. Deliverables checklist

- [x] Filtered generator sums (comparison / conjunction predicates) vectorized via
      a masked / gathered extractor (never per-index resolution)
- [x] Index conditionals `(term if pred else term/const)` + `Constraint.Skip`
      under a predicate, via polarity replay
- [x] Composition with multiple Var components, mutable-Param RHS, ranged relations
- [x] Mandatory silent fallback for every deferral, incl. a **safe** rejection of
      `or`/`not`/short-circuit filters (never a wrong mask) — tested byte-identical
- [x] Equivalence gates for each new shape (filtered-sum, conditional, Skip,
      filtered+conditional) + deferrals + `network_flow` fallback; solve equivalence
- [x] Opt-in, default OFF, zero stock behaviour change (620 adjacent tests green)
- [x] Synthetic benchmark models (`flow_masked`, `coupling_filtered`) + numbers:
      **(a)** end-to-end ≥3× at 1e4/1e5 on the network-flow body (2.76× at 1e6;
      construct ~1.7×, Var-build-bound — reported, not tuned), **32× construct /
      11× end-to-end** on a high-reuse filtered-sum model;
      **(b)** no regression on `resource_coupling`;
      **(d)** no Phase-3b overhead on non-templatizable models
- [x] `network_flow` documented as the honest boundary: its filtered sums and
      conditional are now covered; its transcendental in-rule RHS is out of scope
