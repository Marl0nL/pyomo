# Convex-Quadratic Objective Support for the Vector Fast Path

**Project:** Vectorized Model Construction for Pyomo (see the Phase-0/2/3/4,
value-guard, and static-fold reports)
**Surface:** `pyomo.contrib.vector` — `VectorObjective(quadratic=Q)` (explicit
array API), `highs_fastload` (transparent classic route), and `highs_faststep`
(persistent warm re-solve).
**Change:** **capability**, not speed. Adds convex-quadratic **objective** support
(`c @ x + 0.5 * x @ Q @ x`) end-to-end — the scoping doc's Phase-3 quadratic
ambition and the direct target of issue #1761 ("slow quadratic constraint
creation"). Constraints stay linear; objective-quadratic only. All 90 existing
vector tests stay green; **zero core-module change** (every edit is inside
`pyomo/contrib/vector/`).

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12, Linux 6.17 x86-64
- numpy 2.5.2 · scipy 1.18.1 · **highspy 1.15.1**
- Reproducible with `bench/spikes/quadratic_qp_scaling.py` (§ Reproducing).

---

## 1. What HiGHS supports (verified empirically before designing)

HiGHS solves **convex LP + QP** with a quadratic *objective* only. Verified with
highspy 1.15.1:

| Case | HiGHS behavior | This feature |
|---|---|---|
| Convex QP objective (`c'x + 0.5 x'Hx`, H PSD for min) | `passModel(HighsModel)` with a `HighsHessian` (lower-triangular CSC of the full symmetric Hessian) | **supported** |
| Maximize concave (H NSD) | `lp.sense_ = kMaximize`; HiGHS checks concavity | **supported** |
| **Non-convex** QP (H indefinite for the sense) | `run()` → `kError`: *"Hessian … is not positive semidefinite … Cannot solve non-convex QP"* | **fail-loud**, message surfaced |
| **MIQP** (integer var + Hessian) | `run()` → `kError`: *"Cannot solve MIQP problems with HiGHS"* | **fail-loud**, detected early |
| **Quadratic constraint** | no HiGHS API at all | **out of scope**, fail-loud (unchanged) |

Convention: the Hessian `H` is the true second-derivative matrix; HiGHS applies
the `0.5` factor. From Pyomo's monomial repn, a diagonal `coef·x_i²` → `H_ii =
2·coef`; a cross `coef·x_i·x_j` → `H_ij = coef` (lower triangle once). For the
explicit-array API `Q` **is** `H` (the gurobipy/cvxpy convention).

Classic reference/oracle: Pyomo's **v2** `SolverFactory('highs')` (persistent
interface) solves classic QP; the legacy `appsi_highs` does **not** (raises
`DegreeError`). `LinearStandardFormCompiler` is linear-only (rejects a quadratic
objective), so the classic route compiles constraints linearly and adds the
objective Hessian over the same column space.

---

## 2. The three routes

1. **Explicit array API** — `VectorObjective(terms={m.x: c}, quadratic=Q)` on
   `VectorVar` blocks; `assemble` stacks the Hessian into `VectorMatrices`;
   `load_highs`/`solve_highs` pass it via `passModel`.
2. **Transparent classic** — a classic Pyomo model with an `x[i]*x[j]`-built
   quadratic objective solved through `SolverFactory('highs_fastload')`:
   `generate_standard_repn(quadratic=True)` extracts the Hessian; constraints and
   the column space come from the stock compiler (extended with objective-only
   variables); one bulk `passModel`.
3. **Persistent warm** — `FastStepHighs` loads the Hessian once and warm-re-solves
   a **static-Hessian** QP while the linear cost / RHS / bounds change each roll
   (the rolling-horizon portfolio path). A mutable `Param` feeding the Hessian is
   folded and value-guarded exactly like a static matrix coefficient: unchanged →
   warm solve; genuinely changed → fail-loud, or (opt-in `on_matrix_change=
   'reload'`) a re-fold + reload that rebuilds the Hessian.

Correctness gate: QP solves match the v2-HiGHS classic reference on randomized
convex QPs, plus analytic optima; 17 new tests in `tests/test_quadratic.py`
(explicit / fastload / faststep), all existing tests green.

---

## 3. Benchmark: synthetic portfolio QP (dense-ish banded `Q`)

`min 0.5 x'Qx − μ'x` s.t. budget `Σx = 1`, linear sector caps, box `0 ≤ x ≤ 1`.
`Q` is a banded SPD covariance (bandwidth 25), so `nnz(Q) ≈ n·51` — swept through
the **1e4 / 1e5-nnz class**. Two builds of the *same* QP: the **vector fast path**
(array assemble + bulk `passModel` + `run`) vs the **classic Pyomo QP route**
(monomial-built objective solved through v2 `SolverFactory('highs')`). Objectives
match to ~1e-16 before any timing is reported.

```
     n      Qnnz  v.build  v.load  v.solve  v.total   c.build  c.solve  c.total   build×  total×     obj Δ
   700    35,050    0.003   0.002    0.005    0.010     0.094    0.295    0.389    36.9x   38.7x  4.4e-16
  1400    70,750    0.004   0.004    0.009    0.018     0.187    0.589    0.776    46.8x   44.0x  4.4e-16
  2200   111,550    0.006   0.006    0.019    0.032     0.335    0.926    1.261    55.4x   39.9x  8.9e-16
  3200   162,550    0.008   0.009    0.031    0.048     0.452    1.394    1.846    57.6x   38.3x  0.0e+00
```
(times in seconds; `v.*` = vector fast path, `c.*` = classic Pyomo QP route.)

### The honest read

- **Construction is where the array path wins: 37–58×.** The classic route builds
  one expression-tree node per Hessian nonzero (`Σ Q_ij x_i x_j`); the array path
  builds a scipy CSC and stacks it — `O(nnz)` numpy, no per-monomial Python
  objects. This is exactly the #1761 symptom ("slow quadratic creation") removed
  at the root.
- **Solve (pure HiGHS run) is at parity — solver-bound, as expected.** At
  n = 2200 the vector `run()` is **24 ms**; the classic route's own HiGHS run
  (`results.timing_info.highs_time`) is **20 ms**. Same convex-QP solve, same
  floor. The array path adds nothing here and takes nothing away.
- **The classic route's `c.solve` (~0.3–1.4 s) is dominated by per-element model
  load, not the solve.** The persistent interface extracts the quadratic
  objective into HiGHS one element at a time (#3888, now in the QP setting); the
  vector path collapses it to a single bulk `passModel` (`v.load` ≈ 2–9 ms).

So the QP result mirrors the LP case precisely: **construction + load gains are
large and real; the solve itself is solver-bound and at parity.** No claim is made
that the solver runs faster — it runs the same.

---

## 4. Scope and deferrals

**In scope (landed):** convex-quadratic *objective* on all three routes; explicit
`Q` blocks (single- and multi-block); maximize (concave); constant offset; fixed
variables in quadratic terms (folded consistently with the constraint compiler);
objective-only variables (column extension). Fail-loud: MIQP, non-convex QP,
quadratic constraints, higher-order nonlinear objectives.

**Deferred (logged, not landed):**
- **Varying-Q on the warm path.** `faststep` supports a *static* Hessian; a
  genuinely varying `Q` (or a `Param` that feeds both the Hessian and a live
  linear template) fails loud, with `on_matrix_change='reload'` as the opt-in
  rebuild path. Pushing incremental Hessian deltas across warm rolls is future
  work.
- **MIQP / non-convex QP.** Outside HiGHS; would need a different solver backend.
- **Quadratic constraints.** Out of scope by design (no HiGHS API); the hard
  ceiling for this project is objective-quadratic.

## Reproducing

```
bench/.venv/bin/python -m bench.spikes.quadratic_qp_scaling            # default sweep
bench/.venv/bin/python -m bench.spikes.quadratic_qp_scaling 700 1400 2200 3200
```
Correctness (the equivalence gate) runs in the test suite:
```
python -m pytest pyomo/contrib/vector/tests/test_quadratic.py -q
```
