# `gurobi_fastload` — Gurobi array hand-off for the vector fast path

**Project:** Vectorized Model Construction for Pyomo (see the project scoping doc and the Phase-0 baseline report)
**Deliverable:** `gurobi_fastload`, a drop-in solver that routes an **unmodified classic** linear / convex-QP model's solve through the *same* standard-form compile the HiGHS fast path uses, handed to **Gurobi's native matrix API** (`Model.addMVar` + `addMConstr` + `setMObjective`) in a handful of bulk calls.
**Relation to Phase 2:** the Gurobi twin of `highs_fastload`. Today the vector fast path is HiGHS-only; many Pyomo users are Gurobi-first, and the bulk-array hand-off argument transfers directly (gurobipy's `addMVar`/`addMConstr`/`setMObjective` *are* its native matrix API).

---

## ⚠️ License honesty (read before quoting any number)

The pip `gurobipy` wheel ships a **size-limited** license: **2000 variables and 2000 constraints** (`GurobiError` errno 10010, *"Model too large for size-limited license"*, verified locally). **Every** measurement and equivalence check in this work stays strictly under that ceiling. Consequently:

* Correctness / cross-backend equivalence is established **at licensed sizes only**.
* The build-time table below is a **licensed-size** measurement. It shows the hand-off cost and the fast-route ratio at small sizes and confirms both routes agree on the objective — it makes **no** large-scale claim.
* The **architectural expectation** — that a bulk matrix hand-off replacing a per-row load scales the way the HiGHS `passModel` route did in Phase 2 (where the per-row `set_instance` cost grows super-linearly while the bulk hand-off tracks the array-native ceiling) — is stated here but is **not locally measurable** with this license.

When `gurobipy` (or a usable license) is absent, the whole test suite **skips** the Gurobi cases; nothing fails. CI stays green without Gurobi.

---

## What it is

`gurobi_fastload` is a `SolverBase` registered on both the v2 and legacy `SolverFactory` at `pyomo.contrib.vector` import, exactly like `highs_fastload`:

```python
from pyomo.contrib.solver.common.factory import SolverFactory
results = SolverFactory('gurobi_fastload').solve(model)   # no model change
# or the legacy factory:  pyomo.SolverFactory('gurobi_fastload').solve(model)
```

It is a **backend, not a new compiler.** It reuses the Phase-2 solver-neutral compile verbatim:

* `pyomo.contrib.vector.fastload.compile_to_highs_arrays` (aliased `compile_fastload_arrays`) walks the classic model once through the fast `pyomo.repn` visitors and emits a `FastLoadCompiled` — standard-form range-row arrays (`A`, row/col bounds, integrality, a linear cost, and an optional objective Hessian) **with no HiGHS specifics**. The HiGHS-only step was always `build_highs_model`; the Gurobi backend supplies its own array→Gurobi builder instead.
* The only seam added to the shared module is a one-line neutral alias plus an optional `solver_name` argument that labels the fail-loud error messages (so a Gurobi rejection names `gurobi_fastload`, not `highs_fastload`). No core module was changed; the whole feature is additive under `pyomo/contrib/vector/`.

### Mechanism (why it is faster)

| | classic Pyomo→Gurobi route | fast route (`gurobi_fastload`) |
|---|---|---|
| construct | build the `ConcreteModel` | build the `ConcreteModel` (**identical, shared**) |
| hand-off | v2 `gurobi_persistent` `set_instance`: per-row `addConstr` / `addConstrs` (#3888) | standard-form compile (one bulk visitor pass) + `addMVar`/`addMConstr`/`setMObjective` |
| endpoint | the solver has the model | the solver has the model |

Both routes construct the same model; the fast route replaces the **hand-off** stage. `construct + hand-off` is therefore directly comparable.

---

## Scope & fail-loud discipline (identical to `highs_fastload`)

**Supported:** linear continuous, linear MIP, and a **convex-quadratic objective** (linear constraints — the #1761 use case).

**Rejected loudly** (never a silently wrong answer):

* nonlinear terms / components the standard-form compiler cannot process → caught at compile, message points at a classic solver route;
* a **non-convex** quadratic objective → Gurobi's own PSD check (the backend sets `NonConvex=0`) refuses it (`GurobiError` errno 10020), surfaced as `IncompatibleModelError` — convex QP only;
* an **MIQP** (integer variable + quadratic objective) → rejected up front. *Note:* Gurobi **can** solve convex MIQP, but `NonConvex=0` was verified **not** to enforce convexity for integer models (a non-convex MIQP still solved), so a correct MIQP path needs an explicit convexity gate. That is deliberately out of scope here (see *Deferred*).

---

## Results — build-time comparison (licensed sizes only)

Median of 5 repeats. `construct` is shared by both routes. `classic HO` = v2 `gurobi_persistent` `set_instance` (per-row load, no solve); `fast HO` = the `gurobi_fastload` matrix-API build (no solve). `HO speedup` isolates the replaced stage; `build→solver ×` is the coherent `(construct + classic) / (construct + fast)` ratio. `obj match` solves both routes and compares objectives.

Synthetic banded LP (`bench/gurobi_fastload_buildtime.py`): `n` vars in `[0,5]`, `n` coupling rows `x_i + x_{i+1} ≥ b_i`, one budget row, one linear objective.

| n_vars | n_cons | construct | classic HO | fast HO | HO speedup | build→solver × | obj match |
|-------:|-------:|----------:|-----------:|--------:|-----------:|---------------:|:---------:|
|    200 |    201 |     0.9 ms |     9.6 ms |  2.5 ms |     3.83× |          3.06× | yes |
|    500 |    501 |     1.9 ms |    22.4 ms |  4.4 ms |     5.07× |          3.83× | yes |
|   1000 |   1001 |     4.2 ms |    43.2 ms |  8.2 ms |     5.29× |          3.84× | yes |
|   1500 |   1501 |     6.0 ms |    65.0 ms | 12.3 ms |     5.29× |          3.88× | yes |
|   1800 |   1801 |     7.2 ms |    77.7 ms | 14.4 ms |     5.40× |          3.93× | yes |

At licensed sizes the bulk matrix hand-off is **~3.8–5.4×** faster than the per-row `set_instance`, giving **~3.1–3.9×** on the coherent `construct + hand-off` route; both routes return the same objective. The per-row `classic HO` column grows faster than `fast HO` as `n` rises (the per-row load is the stage #3888 flags) — the licensed-size shadow of the same super-linear-vs-bulk gap Phase 2 measured for HiGHS. **No claim is made beyond 2000 vars/cons, where this license cannot solve.**

---

## Equivalence gates (all at licensed sizes)

`pyomo/contrib/vector/tests/test_gurobi_fastload.py`:

* **Core correctness** — simple LP (+ duals & reduced costs), maximize + objective offset, MIP integrality, fixed variables, range constraints (split into two Gurobi rows, one constraint on map-back), objective-free feasibility, infeasible / compile-infeasible / unbounded, and a convex-QP objective (analytic optimum, concave-maximize, off-diagonal Hessian, objective-only variable).
* **Cross-backend + classic-reference equivalence** — on the shared randomized-model suite (`random_models.py`) `gurobi_fastload` matches the classic APPSI-HiGHS reference *and* `highs_fastload` on objective/solvability; a random **convex QP** matches both `highs_fastload` and the v2-HiGHS QP reference on objective and (unique) primal.
* **Fail-loud guards** — nonlinear constraint, nonlinear (cubic) objective, non-convex QP, MIQP, quadratic constraint each raise `IncompatibleModelError` (and the message names `gurobi_fastload`).
* **Skip discipline** — with `gurobipy` blocked, all 25 Gurobi tests **skip** and the 107 existing vector tests still pass (verified).

Suite: **132 pass** with gurobipy present; **107 pass + 25 skip** with gurobipy absent — zero failures either way. Zero core-module change.

---

## Deferred (logged, not gold-plated)

* **`gurobi_faststep` (warm persistent re-solve).** The `FastStepHighs` analogue — gurobipy supports in-place attribute updates (`Model.setAttr` on `RHS`/`Obj`/bounds, `MVar` slices) that would carry the rolling-horizon warm-basis path — but it does not fall out for free from this backend's seam, so it is a separate PR.
* **Convex MIQP.** Gurobi solves it natively; the only missing piece is an explicit objective-convexity gate (an eigen/PSD check, since `NonConvex=0` does not enforce convexity for integer models). A one-stage follow-up once the check's cost model at scale is settled.

---

## Reproducing

```bash
# any Python 3.12 venv with pyomo (editable) + numpy + scipy + gurobipy (+ highspy for the cross-check)
python bench/gurobi_fastload_buildtime.py            # the table above
python -m pytest pyomo/contrib/vector/tests/test_gurobi_fastload.py -q
```

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs
- numpy 2.5.2 · scipy 1.18.1 · gurobipy 13.0.3 (size-limited pip license) · highspy 1.15.1 (cross-check reference)
