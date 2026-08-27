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
rolling-horizon / MPC). `highs_faststep` accepts mutable constraint-matrix
coefficients via a value-aware guard (verify the values each roll, not the
mutability flag) — see `bench/VALUEGUARD_REPORT.md` — and engages on models with
**non-affine** param participation (`price*duration`, `dur/eff`) by **folding**
the verified-static params as watched constants — see `bench/STATICFOLD_REPORT.md`
(`FastStepHighs.classification_report()` shows what folded). All three routes also
support a **convex-quadratic objective** (`c@x + 0.5*x@Q@x`, the #1761 use case):
`VectorObjective(quadratic=Q)` explicit API, `highs_fastload` for a classic
`x[i]*x[j]` objective, and `highs_faststep` with a *static* Hessian (mutable Q
params are folded/guarded; varying-Q fails loud). Objective-quadratic only —
quadratic constraints, MIQP, and non-convex QP fail loud (HiGHS solves convex QP
only). See `bench/QUADRATIC_QP_REPORT.md`. CI on the fork is manual-trigger-only —
validate locally.

- Benchmarks run in a machine-local venv `bench/.venv` (recreate per `bench/README.md`);
  run cold-stage benches via `bench/run_bench.py`, the warm-tick bench via
  `python -m bench.warm_faststep`.
- Tests: `pytest pyomo/contrib/vector/tests/`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
