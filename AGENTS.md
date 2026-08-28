# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Vectorized-construction program (this fork)

This fork carries a phased "Vectorized Model Construction for Pyomo" program under
`pyomo/contrib/vector/`; feature work branches off the `vectorisation` branch (not
`main`). Each phase has a report in `bench/` (`PHASE0..PHASE4_REPORT.md`) — read the
relevant one before touching a phase. Phases: 1 columnar components, 2
`highs_fastload` (cold `passModel` hand-off), 3 template construction, 4
`highs_faststep` (`FastStepHighs`, array-native persistent **warm** re-solve for
rolling-horizon / MPC). The Phase-3 construction switch (`vectorized_construction()`
/ `PYOMO_VECTOR_CONSTRUCT`) also does **transparent columnar Var/Param
construction** (`pyomo/contrib/vector/varparam.py`): a classic `Var`/`Param` with
vectorizable args builds into NumPy columns (materialize-on-touch), skipping the
per-index data object; per-index callables fall back to byte-classic. Activation
is a runtime monkeypatch of `Var.construct`/`Param.construct` — **no `pyomo/core/*`
edit** — so switch-off is byte-classic. See `bench/VARPARAM_REPORT.md`. `faststep`
**does** warm re-solve a templatized (switch-on) model: `_build_mutable_plan`
dispatches to a template-aware constraint-slot collector (static families skipped,
mutable-RHS families materialized, columnar Params bulk-read), bit-exact vs
switch-off, set_instance no slower + warm tick faster (`bench/FASTSTEP_TEMPLATIZED_REPORT.md`);
the one carry-over limit is that a columnar Var can't hold a mutable-`Param` bound
(declare it with a `bounds=` rule → that Var stays classic). The whole program is consolidated in two docs: the
PEP-style upstream proposal `docs/vector_proposal.md` (design, evidence,
compatibility contract, `contrib → core` migration split — every number cited from
a committed `bench/` report) and the user guide `pyomo/contrib/vector/README.md`
(quickstart per route); `bench/README.md` indexes the per-phase reports. Keep those
three in sync when a route's API or a headline number changes. `highs_faststep`
accepts mutable constraint-matrix
coefficients via a value-aware guard (verify the values each roll, not the
mutability flag) — see `bench/VALUEGUARD_REPORT.md` — and engages on models with
**non-affine** param participation (`price*duration`, `dur/eff`) by **folding**
the verified-static params as watched constants — see `bench/STATICFOLD_REPORT.md`
(`FastStepHighs.classification_report()` shows what folded). When a guarded (folded
or matrix-templated) coefficient **genuinely changes** every roll (the shrinking-
first-interval MPC), `on_matrix_change='patch'` makes it a **warm update** instead
of a reload: a partial in-place refold (classification unchanged, only the affected
templates' values re-derived + validated) plus per-entry `changeCoeff` on the
changed A-entries, basis kept; degrades to reload past `patch_max_entries` (auto
`max(4096, nnz//4)`) or on a Hessian-feeding fold. Opt-in (default stays fail-loud
`'error'`). See `bench/COEFF_PATCH_REPORT.md`. `highs_faststep` also
does **masked warm updates** — a solver-side overlay of **row masks** (relax rows
to free: `deactivate_rows`/`activate_rows`) + **variable fixes** (pin to a value via
equal bounds: `fix_variables`/`unfix_variables`, or `set_window`/`clear_window` for
both) that narrows a rolling MPC's **active window** without a structural change
(the matrix/fingerprint/guard/fold set are untouched, so it stays on the warm path).
On the window this is provably the structurally-narrowed problem (a fixed
out-of-window var on an in-window row = the boundary condition); the reported
objective includes the fixed vars' constant, `masked_objective_constant()` returns
it. Array-first (bool masks + fix-value arrays) for an external adapter. Equivalence
gated vs a fresh narrowed build in `tests/test_faststep_masked.py`; see
`bench/MPC_NARROW_REPORT.md`. All three routes also
support a **convex-quadratic objective** (`c@x + 0.5*x@Q@x`, the #1761 use case):
`VectorObjective(quadratic=Q)` explicit API, `highs_fastload` for a classic
`x[i]*x[j]` objective, and `highs_faststep` with a *static* Hessian (mutable Q
params are folded/guarded; varying-Q fails loud). Objective-quadratic only —
quadratic constraints, MIQP, and non-convex QP fail loud (HiGHS solves convex QP
only). See `bench/QUADRATIC_QP_REPORT.md`. CI on the fork is manual-trigger-only —
validate locally.

**Solver backends.** The cold hand-off has two backends over one solver-neutral
compile: `compile_to_highs_arrays` (alias `compile_fastload_arrays`) emits a
solver-agnostic `FastLoadCompiled` (standard-form range-row arrays); each backend
supplies only an array→solver builder + map-back. `highs_fastload` (`fastload.py`,
`build_highs_model`) and `gurobi_fastload` (`gurobi_fastload.py`, gurobipy's
`addMVar`/`addMConstr`/`setMObjective` matrix API) share scope: linear + convex-QP
objective; non-convex/MIQP/quadratic-constraint fail loud. When adding a backend,
reuse the compile and pass `solver_name=` so fail-loud messages name it. See
`bench/GUROBI_FASTLOAD_REPORT.md`. **gurobipy license note:** the pip wheel is
size-limited to **2000 vars/2000 cons** (errno 10010) — all Gurobi tests/benches
stay under it and *skip* when gurobipy/license absent; no large-scale claim is
locally measurable. Deferred: `gurobi_faststep` (warm) and convex-MIQP.

**Phase-2 mutability (real ragged/mutating models).** The columnar components are
mutable in bulk (position-space `where=`) and per-element (materialized view):
`VectorVar.setlb/setub/set_bounds/fix/unfix/set_values`,
`VectorConstraint.deactivate_rows/activate_rows/set_row_active/set_row_bounds`.
Masked-out rows are *dropped* from the one-shot standard form (matching classic
`deactivate()`) and *relaxed* on the solve path; fixed vars substitute out in
`compile_standard_form` / pin as column bounds in the HiGHS hand-off. Each mutation
records dirty columns/rows; `VectorPersistentHighs` (`persistent.py`) loads once and
warm re-solves by pushing only the dirty subset through HiGHS
`changeColsBounds`/`changeRowsBounds` (basis retained), with a fail-loud structural
guard and array-native solution load-back (`load_solution`, `VectorVar.value_array`).
The ragged `supply_chain` runs the full vector fast path via
`bench/models/supply_chain_vector.py` (register new vector models in
`run_bench._vector_models`); note the `supply_chain` *generator* is infeasible at
1e5+ (a pre-existing data property — the classic build is equally infeasible).

- Benchmarks run in a machine-local venv `bench/.venv` (recreate per `bench/README.md`;
  needs numpy/scipy/highspy, gurobipy optional). Cold-stage + vector-path benches via
  `bench/run_bench.py` (`--backends pyomo,pyomo_vector`), the warm-tick bench via
  `python -m bench.warm_faststep`, the mutation-cycle bench via `python -m bench.mutation_cycle`,
  the faststep-over-switch-on bench via `python -m bench.faststep_templatized` (`--mutbound` for the mixed case),
  the MPC-window-narrowing bench (masked warm vs fresh structural narrow) via `python -m bench.mpc_narrow_faststep`.
- Tests: `pytest pyomo/contrib/vector/tests/` (mutation: `test_mutation.py`,
  sparse/ragged index: `test_sparse.py`, faststep-over-switch-on: `test_faststep_templatized.py`).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
