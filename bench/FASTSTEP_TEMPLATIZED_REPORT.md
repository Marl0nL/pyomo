# Faststep over Templatized (switch-ON) Models — Integration Report

**Project:** Vectorized Model Construction for Pyomo (see the Phase-0..4 reports, `VARPARAM_REPORT.md`, and the vector `README.md`).
**Deliverable:** let `FastStepHighs` (`highs_faststep`) warm re-solve a model built **with the Phase-3 construction switch** (`vectorized_construction()` / `PYOMO_VECTOR_CONSTRUCT`) — i.e. a model whose constraint families are *template-vectorized* and whose Vars/Params are *columnar*. Before this change, such a model got the program's best construction but could **not** use its best warm re-solve.
**Baseline:** `FastStepHighs` on a **switch-OFF** (byte-classic) build of the same model source — the closed-form reference for both correctness (solve-for-solve) and speed.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD, branched off `vectorisation`)
- Python 3.12, Linux x86-64 · numpy 2.5 · scipy 1.18 · highspy 1.15
- Timings are the median of repeated builds / warm rolls in one process. Reproduce with `bench/faststep_templatized.py` (§ Reproducing). All evidence is synthetic (public).

---

## 1. The failure mode (established first)

`set_instance` compiles the model to standard-form arrays and then builds the *mutable-update plan* — the affine templates over the model's mutable `Param` vector that let a warm solve push only the changed coefficients/bounds. The plan builder walked `compiled.rows` expecting a per-row `ConstraintData.body`:

```
AttributeError: 'IndexedConstraint' object has no attribute 'body'
   at _build_mutable_plan  ->  generate_standard_repn(con.body, ...)
```

A **templatized** compile (`compile_templated_to_highs_arrays`) does not materialize per-row bodies: `compiled.rows` carries `(family, local_row)` for a vectorized family (the `IndexedConstraint` itself, which has no `.body`), and `(ConstraintData, 0)` for a family that fell back to the classic per-row path. The classic plan builder assumed `(ConstraintData, multiplier)` throughout, so it crashed on the first templatized family — the compile itself already worked. This is a pre-existing Phase-3/PR-10 gap, reproducible with the switch alone.

Two facts about a templatized compile shape the fix:

- A templatized family always has a **static coefficient matrix** — a mutable matrix coefficient is not vectorizable (coefficients must be index-independent constants), so it forces the whole family onto the classic fallback. A templatized family's only mutable data is therefore its **right-hand side**.
- The templatized compile emits **HiGHS-native range rows** (one per constraint, never split), so `compiled.rows[i][1]` is a local row index, not a `±1` mixed-form multiplier.

---

## 2. What the change does

**One integration seam, no core-module edits, no new user features.** `_build_mutable_plan` now dispatches on the same `model_has_templates(model)` test the compiler itself uses:

1. **Templated constraint-slot collector** (`_templated_constraint_slots`). Walks `compiled.rows` family by family. A family that references **no mutable Param** has fully static rows and is **skipped without materializing them** (the vectorized-construction win survives into the warm compile — a single probe row decides the family, since a templatized family is structurally uniform). A family that does reference a mutable Param has its rows materialized (`con[idx].expr`) and fed to the *same* slot machinery the classic path uses; range direction is read from the constraint (`equality` / open side) instead of a multiplier. A classic-fallback row is walked directly. Everything downstream — verified-static folding, the parameter registry, the affine templates, the plan self-check, and the value guard — is **shared, unchanged**; the self-check validates the collected slots against the compiled numeric arrays, so a wrong slot fails loud, never silently.
2. **Columnar Var bounds.** A `None` column (owned by a columnar Var) is skipped in the bound scan — a columnar Var's bounds are static float columns with no mutable-Param bound to template.
3. **Columnar solution load-back.** `_postsolve` now passes `compiled.column_scatter` to the solution loader, so a warm solve scatters primals back onto columnar Vars in bulk (previously it left them uninitialized).
4. **Columnar-aware parameter read** (`_build_param_reader` / `_gather_param_values`). Reading the mutable-`Param` vector is on the warm-tick hot path. A columnar `VectorParamData.value` dereferences a weakref and indexes its component's value column on every read (~3× a classic slot read). The plan now gathers every columnar Param that shares a value column in **one vectorized slice** of that column, keyed once at `set_instance`; classic Params keep the per-object read, so a classic model is byte-unchanged. The result is bit-identical (the value column is the single source of truth).

---

## 3. Results — exit criteria

Model: a synthetic rolling-horizon MPC built from one identical source, switch-ON vs switch-OFF. It exercises the warm mechanisms through the templatized representations — a templatized equality with a mutable RHS (`dem`), a templatized inequality with a mutable RHS (`gcap`), a **fully static** templatized family (immutable `pcap`) that the plan skips, columnar Vars (`p`, `soc`), and a mutable objective coefficient (`price`). `A` assets × `T` horizon → `2·A·T` vars, `2·A·T + T` constraints.

**(a) Solve-for-solve equivalence.** The switch-ON warm loop matches the switch-OFF `FastStepHighs` run **bit-for-bit** every roll (objective, and — on unique-optimum LPs — primal values), for both basis-kept and basis-reset runs. Max objective difference over 15 rolls at every scale below: **0.0**.

**(b) Compile no slower, warm tick parity-or-better** (median; ratio = ON / OFF, lower is better for switch-ON):

| scale (A×T) | vars | cons | set_instance ON | OFF | **ON/OFF** | warm tick ON | OFF | **ON/OFF** |
|---|---|---|---|---|---|---|---|---|
| 20×50  | 2 000  | 2 050  | 71.3 ms  | 70.4 ms  | **1.01** | 1.05 ms | 1.42 ms | **0.74** |
| 40×100 | 8 000  | 8 100  | 318.1 ms | 321.4 ms | **0.99** | 2.64 ms | 4.12 ms | **0.64** |
| 60×150 | 18 000 | 18 150 | 744.8 ms | 783.8 ms | **0.95** | 5.64 ms | 8.93 ms | **0.63** |

- **set_instance** is at parity and **~0.95× at scale** — the static `cap` family is skipped without materializing its rows, and the templatized compile is vectorized.
- **warm tick is faster** — down to **~0.63×** at scale, from the one-shot columnar-Param gather versus the per-object read.

**(c) Tests.** The vector suite is green with additions: **173 passed, 25 skipped** (163 baseline + 10 new in `tests/test_faststep_templatized.py`); the classic `test_faststep.py` (42) is unchanged. Zero `pyomo/core/*` edits.

---

## 4. Scope, guards, and the one caveat

- **Every warm mechanism carries over, switch-ON, verified against the switch-OFF run:** mutable objective coefficient and offset; mutable row RHS from a templatized equality *and* inequality; the mapping-free array path (`solve(param_values=…)`); verified-static parameter **folding** (`price·dur`); the value-aware **matrix guard** (accept-and-verify a mutable-but-static coefficient, and fail loud on a genuine change); array *and* scalar Param mutations; basis-kept re-solves.
- **The one caveat — a mutable-`Param` *variable bound*.** A columnar Var cannot store a mutable-Param bound (its bound columns are `float64`); declaring one with a `bounds=` rule keeps that Var **classic** (the surrounding constraint families still templatize). This is a Phase-3/PR-10 representation limit, orthogonal to the constraint integration. It works and stays bit-exact, but the classic-fallback Var's per-index bound/body is read in Python during the compile (vs the C-level standard-form compiler switch-off), so `set_instance` is **modestly slower** in that mixed case (`--mutbound`: ON/OFF ≈ 1.42 / 1.18 / 1.05 across the three scales — the ratio shrinks as the static-family skip and the constraint families dominate; the **warm tick stays faster**, ≈ 0.71–0.79). If the rolling variable bound is instead the *program-driven* kind, use the PR-8 index-addressed API (`set_variable_bounds`), which is unaffected.

---

## 5. Reproducing

```bash
# headline (columnar, clean templatizable case)
bench/.venv/bin/python -m bench.faststep_templatized
# the mixed case (a classic-fallback Var carrying a mutable-Param bound)
bench/.venv/bin/python -m bench.faststep_templatized --mutbound
# tests
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/test_faststep_templatized.py
```
