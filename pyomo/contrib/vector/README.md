# `pyomo.contrib.vector` — vectorized model construction & fast solver hand-off

A **benchmarked, additive fast path** for building large linear (and convex-quadratic)
Pyomo models and getting them into a solver quickly. Nothing here changes stock
Pyomo: every route is either an explicit opt-in component, a solver you select by
name, or an off-by-default switch. When a fast-path component is touched by code
that does not understand it, it transparently *scalarizes* back to classic Pyomo
objects (the compatibility contract — see the proposal, `docs/vector_proposal.md` §6).

This guide is the practical walkthrough of each route. For the design rationale,
the `contrib → core` migration proposal, and the full benchmark evidence, read
**`docs/vector_proposal.md`**. For the numbers behind each route, follow the
per-phase reports linked below and indexed in **`bench/README.md`**.

## Requirements

- **numpy** and **scipy** (required for every route).
- **highspy** (the HiGHS Python interface) for every HiGHS route.
- **gurobipy** only for `gurobi_fastload` (optional; absent → those paths just skip).

```python
import pyomo.contrib.vector   # importing registers highs_fastload / gurobi_fastload
```

## Which route do I want?

| You have… | …and want to… | Use | Report |
|---|---|---|---|
| a model you're building from scratch, in matrix form | maximum build+load speed, full array control | **explicit-array API** — `VectorVar` / `VectorConstraint` / `VectorObjective` + `solve_highs` | `bench/PHASE0_REPORT.md` |
| an **unmodified classic** linear/MIP/convex-QP model | solve it faster with **no model change** | **`SolverFactory('highs_fastload')`** (or `gurobi_fastload`) | `bench/PHASE2_REPORT.md`, `bench/GUROBI_FASTLOAD_REPORT.md` |
| classic `Constraint(index, rule=...)` code | make **construction** faster too, no rewrite | **`vectorized_construction()`** switch + `highs_fastload` | `bench/PHASE3_REPORT.md` |
| a rolling-horizon / MPC loop (re-solve thousands of times), classic **or** `vectorized_construction()`-built | warm re-solve pushing changed data as arrays | **`FastStepHighs`** | `bench/PHASE4_REPORT.md`, `bench/FASTSTEP_TEMPLATIZED_REPORT.md` |
| an MPC loop that **narrows to a sliding window** each cycle | re-solve the window on the warm path, no structural rebuild | **`FastStepHighs`** row masks + variable fixes | `bench/MPC_NARROW_REPORT.md` |
| an all-vector model you mutate between solves | warm re-solve driven by columnar mutation | **`VectorPersistentHighs`** | `bench/PHASE2_REPORT.md` |
| a convex-quadratic **objective** (`c·x + ½·xᵀQx`) | build & solve it array-native | `VectorObjective(quadratic=Q)` or `highs_fastload` | `bench/QUADRATIC_QP_REPORT.md` |

**Rule of thumb:** if you don't want to rewrite your model, reach for
`highs_fastload` first — it needs no code change and delivers the largest
transparent win on load-bound models. Rewrite to the explicit-array API only when
you're building the matrix yourself and want the construction win too.

---

## 1. Transparent fast hand-off for a classic model — `highs_fastload`

The zero-effort route: keep your model exactly as it is; change only the solver.

```python
import pyomo.environ as pyo
import pyomo.contrib.vector                                   # registers the solver
from pyomo.contrib.solver.common.factory import SolverFactory

m = pyo.ConcreteModel()
m.x = pyo.Var([0, 1], domain=pyo.NonNegativeReals, bounds=(0, 10))
m.c1 = pyo.Constraint(expr=m.x[0] + 2 * m.x[1] == 3)
m.c2 = pyo.Constraint(expr=m.x[0] - m.x[1] <= 1)
m.obj = pyo.Objective(expr=m.x[0] + m.x[1], sense=pyo.minimize)

res = SolverFactory('highs_fastload').solve(m)               # one bulk passModel
print(res.incumbent_objective)                                # 1.5
print(res.termination_condition)                              # convergenceCriteriaSatisfied
duals = res.solution_loader.get_duals()                       # LPs: duals + reduced costs
```

It compiles the model once with the stock `LinearStandardFormCompiler`, hands the
whole matrix to HiGHS in one `passModel`, solves, and maps primals / duals /
reduced costs back. The legacy factory works too: `pyo.SolverFactory('highs_fastload')`.

**When it wins:** the hand-off is 3.1–6.4× faster than the per-row APPSI load across
the synthetic suite; end-to-end up to **4.50×** on a load-bound model (`bench/PHASE2_REPORT.md`).
**Scope:** linear continuous / MIP, plus a convex-quadratic objective. Nonlinear
terms or components the standard-form compiler can't process (SOS/Piecewise) are
rejected loudly, pointing you at a classic solver route — never a wrong answer.

**Gurobi twin — `gurobi_fastload`** (needs `gurobipy`): identical usage, the same
solver-neutral compile handed to Gurobi's native matrix API.

```python
res = SolverFactory('gurobi_fastload').solve(m)               # requires gurobipy
```

> Note: the pip `gurobipy` wheel is size-limited to 2000 vars / 2000 constraints;
> the committed Gurobi benchmarks stay under that and make no larger claim
> (`bench/GUROBI_FASTLOAD_REPORT.md`).

---

## 2. Make your *rule-based* code build faster too — `vectorized_construction`

If your model uses `Constraint(index, rule=...)`, wrap the build in the switch:
rules that *templatize* build their whole constraint matrix with NumPy (no
per-index expression tree); rules that don't fall back to classic construction,
byte-identically. **Default off** — nothing changes unless you opt in.

```python
import pyomo.environ as pyo
from pyomo.contrib.vector import vectorized_construction, model_has_templates
from pyomo.contrib.solver.common.factory import SolverFactory

def build():
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 4)
    m.J = pyo.RangeSet(0, 3)
    m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals, bounds=(0, 10))
    m.cap = pyo.Param(m.I, initialize={i: 5.0 + i for i in range(5)}, mutable=True)
    def cap_rule(m, i):
        return sum(m.x[i, j] for j in m.J) <= m.cap[i]        # unfiltered sum + mutable RHS
    m.capcon = pyo.Constraint(m.I, rule=cap_rule)
    m.obj = pyo.Objective(expr=sum(m.x[i, j] for i in m.I for j in m.J),
                          sense=pyo.maximize)
    return m

with vectorized_construction():          # or set PYOMO_VECTOR_CONSTRUCT=1
    m = build()

print(model_has_templates(m))            # True — the family templatized
res = SolverFactory('highs_fastload').solve(m)   # construct + load stay array-shaped
```

Equivalently, set the environment variable process-wide with no code change at all:

```bash
PYOMO_VECTOR_CONSTRUCT=1 python your_model.py
```

**The templatizable subset** (exactly what Phase-0 Spike B proved): linear bodies
that are a combination of `coef * var[affine_index…]` terms with **constant**
coefficients, optionally inside an **unfiltered** `sum(… for j in Set)`; multiple
variable components (`x[f,c] <= open[f]`); constant or mutable-`Param` RHS;
equality / inequality / ranged relations.

**Out of subset → classic fallback** (logged once at debug level): index
conditionals (`if i == 0`), **filtered sums** (`for j in J if j != n`), modulo /
non-affine indexing (`x[(i-1) % N]`), and index-dependent coefficients
(`a[i,j]*x[j]`). These are common in real models, so the fallback is the norm, not
an edge case — a mixed model vectorizes the families that qualify and compiles the
rest classically, all byte-identical to the classic standard form.

**When it wins:** up to **11.2× construct / 9.8× end-to-end** on a templatizable-heavy
model at 1e6 nonzeros; **no material slowdown** (within noise) on a model whose
rules don't templatize (`bench/PHASE3_REPORT.md`).

*Known limitation:* a model built with the switch on cannot currently be `clone()`d
(a pre-existing core limitation of experimental template expressions).

---

## 3. Explicit-array construction — `VectorVar` / `VectorConstraint` / `VectorObjective`

When you're building the matrix yourself and want the construction win too. State
a whole variable/constraint/objective family as arrays; no per-index Python objects.

```python
import numpy as np, scipy.sparse as sp
import pyomo.environ as pyo
from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective
from pyomo.contrib.vector.highs import solve_highs

# min x0 + x1  s.t.  x0 + 2 x1 == 3 ;  x0 - x1 <= 1 ;  0 <= x <= 10
m = pyo.ConcreteModel()
m.x = VectorVar(pyo.RangeSet(0, 1), domain=pyo.NonNegativeReals, bounds=(0, 10))
A   = sp.csr_matrix(np.array([[1.0, 2.0], [1.0, -1.0]]))
m.c = VectorConstraint(A=A, x=m.x,
                       lb=np.array([3.0, -np.inf]),
                       ub=np.array([3.0,  1.0]))               # row 0 equality, row 1 <=
m.obj = VectorObjective(terms={m.x: np.array([1.0, 1.0])}, sense=pyo.minimize)
m.x.construct(); m.c.construct(); m.obj.construct()

h, obj = solve_highs(m, load_solutions=True)                   # assemble + passModel + run
print(obj, m.x[1].value)                                       # 1.5 1.5
```

Key points:

- **`VectorVar(index, domain=…, bounds=(lo, hi), initialize=…)`** — bounds/values may
  be `None`, a scalar (broadcast), or a length-N array. `bounds` accepts `-np.inf`/`np.inf`.
- **`VectorConstraint(A=…, x=…, lb=…, ub=…)`** — `A` is a scipy sparse (or dense) matrix
  whose columns span the concatenated `x` list; use `x=[m.flow, m.stor]` for several
  `VectorVar`s. Pass `rhs=` instead of `lb`/`ub` for equality rows. **Ragged / sparse
  index sets are handled by construction** — you supply the sparse `A`.
- **`VectorObjective(terms={var: coef_array, …}, sense=…, constant=…)`** — one coefficient
  array per `VectorVar` block; add `quadratic=Q` for a convex-QP objective (§5).
- Call **`.construct()`** on each component (the components are added to a
  `ConcreteModel` and constructed explicitly). `solve_highs(m, load_solutions=True)`
  writes the primal solution straight back into `m.x`'s value array (`m.x[i].value`).
- Materialize-on-touch: `m.c[r]` or iterating `m.c` lazily builds classic
  `ConstraintData`/`LinearExpression` objects (scalarization) — the fast path is
  disabled for that component thereafter, with a one-time warning.

---

## 4. Warm re-solve for rolling-horizon / MPC — `FastStepHighs`

Build once, re-solve thousands of times with slightly changed data (prices,
forecasts, bounds) while keeping the warm simplex basis. `FastStepHighs` reads the
changed mutable-`Param` values, expands every affected coefficient with a vectorized
`M @ P`, and batch-pushes it to a retained HiGHS.

```python
import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.vector import FastStepHighs

m = pyo.ConcreteModel()
m.p = pyo.Param([0, 1, 2], initialize={0: 1.0, 1: 2.0, 2: 3.0}, mutable=True)
m.x = pyo.Var([0, 1, 2], domain=pyo.NonNegativeReals, bounds=(0, 10))
m.c1 = pyo.Constraint(expr=m.x[0] + m.x[1] + m.x[2] == 6)
m.c2 = pyo.Constraint(expr=m.x[0] - m.x[1] == 1)
m.c3 = pyo.Constraint(expr=m.x[1] - m.x[2] == 1)
m.obj = pyo.Objective(expr=sum(m.p[i] * m.x[i] for i in range(3)))

stepper = FastStepHighs()
stepper.set_instance(m)                 # compile once + passModel + build templates
res = stepper.solve()                   # first solve
for roll in range(horizon):
    for i in range(3):
        m.p[i] = new_price(roll, i)     # mutate the model's mutable Params in place
    res = stepper.solve()               # read P, M @ P, batch push, warm re-solve
    print(res.incumbent_objective)
```

**Array (mapping-free) path** — drive the solve from raw arrays without touching the
Pyomo model; `FastStepHighs` owns the row/column mapping:

```python
P = ...                                          # values ordered by stepper.parameters
stepper.solve(param_values=P)                    # expand + batch-push, warm re-solve
stepper.solve(param_values=P, dirty=mask)        # only the changed parameters' rows
```

**Supported warm updates:** objective coefficients, objective offset, constraint
(row) bounds / RHS, and variable bounds — i.e. the rolling-horizon roll. The
constraint matrix `A` is treated as static (see the guard/fold knobs below).
**When it wins:** **2.33× at the 1e5-nnz class** (3.39× at 1e4) vs the persistent
APPSI HiGHS interface, clearing the ≥1.4× target with margin (`bench/PHASE4_REPORT.md`).

### Narrow the active window without a rebuild — masked warm updates

A rolling-horizon MPC often re-solves over a **sliding window** of the horizon:
each cycle only a window `[a, b)` is active, with the state at the window's edge
held at the current measurement. Narrowing the model structurally (a fresh,
smaller matrix each cycle) throws away the warm basis and re-pays construction —
and is exactly the row/column change `FastStepHighs`' fingerprint rejects.

Instead, keep the full matrix and narrow with a **solver-side overlay**: mask the
out-of-window rows off (relaxed to free, so they impose nothing) and fix the
out-of-window variables to their boundary values. On the active window this is
*provably* the structurally-narrowed problem — an in-window recurrence row that
references a fixed out-of-window variable becomes the correct boundary condition —
but the matrix (and the fingerprint, the value guard, and the fold set) is
untouched, so it rides the warm path.

```python
import numpy as np

# Precompute once from the static mapping (no data dependence):
active = np.ones(stepper.active_rows.shape, dtype=bool)      # True == row enforced
fixed  = np.zeros(stepper.fixed_variables_mask.shape, dtype=bool)
values = np.zeros_like(stepper.fixed_values)
# ... mark out-of-window rows inactive and out-of-window cols fixed to their
#     boundary values (use stepper.row_indices(con) / stepper.column_index(var)) ...

for cycle in horizon:
    update_params(m, cycle)                                  # roll the data as usual
    stepper.set_window(active_rows=active, fixed_cols=fixed, fixed_values=values)
    res = stepper.solve()                                    # warm re-solve, basis kept
    window_obj = res.incumbent_objective - stepper.masked_objective_constant()
```

Granular calls exist too — `deactivate_rows(rows)` / `activate_rows(rows)` and
`fix_variables(cols, values)` / `unfix_variables(cols)` — and `clear_window()`
restores the full model. The API is **array-first** (plain int / bool arrays) so an
adapter can drive the window from an external controller with no Pyomo mutation on
the hot path.

**Objective convention:** `solve()` reports the objective of the *full* model with
the fixed variables held at their pin values (the honest cost of what was solved).
That equals the narrowed window problem's objective **plus** the constant cost of
the fixed variables; `masked_objective_constant()` returns that constant, so
`reported − constant` is the pure in-window cost.

**When it wins:** one masked-warm narrow+solve cycle is **4–8× faster** than a fresh
structural narrow (a per-cycle APPSI build+solve), and **2.0–2.9×** faster than even
a fresh `highs_fastload` compile+solve, at the day-length and 1e5-nonzero classes,
with the window objective equal every cycle (`bench/MPC_NARROW_REPORT.md`).

### Composing with the construction switch (`vectorized_construction()`)

`FastStepHighs` warm-re-solves a model built **with the Phase-3 switch on** (§2)
exactly as it does a classic one — `set_instance` reads the templatized constraint
families and columnar Var/Param stores directly, so a switch-ON build gets the
program's best construction *and* its best warm re-solve in one pipeline. There is
nothing extra to call: build under `vectorized_construction()`, hand the model to
`set_instance`, and roll.

```python
from pyomo.contrib.vector import vectorized_construction, FastStepHighs

with vectorized_construction():          # templatized families + columnar Var/Param
    m = build_rolling_model()            # your classic Constraint(index, rule=...) code
stepper = FastStepHighs()
stepper.set_instance(m)                  # feeds templates/columns to the warm compile
for roll in range(horizon):
    update_params(m, roll)               # mutate mutable Params in place
    stepper.solve()                      # same warm loop, same results as switch-off
```

A switch-ON build warm-solves **bit-for-bit** the same sequence as the equivalent
switch-off build (objective and primals, both basis-kept and basis-reset), the
one-time `set_instance` compile is **no slower** than switch-off (at parity, ~0.95×
at scale — a fully static templatized family is skipped without materializing its
rows), and the **warm tick is faster** (~0.63× at scale: columnar Params are read
back in one vectorized gather per component rather than object-by-object). See
`bench/FASTSTEP_TEMPLATIZED_REPORT.md`. Every mechanism below (matrix guard,
folding, the array path) applies unchanged. One representation limit carries over
from the switch: a *columnar* Var cannot hold a mutable-`Param` bound, so declare
such a bound with a `bounds=` rule (which keeps that one Var classic) — the
surrounding constraint families still templatize, though that Var's per-index
compile makes `set_instance` modestly slower (the warm tick stays faster).

### Transparency knobs — the static-matrix guard & parameter folding

A rolling model often carries a *nominally* mutable coefficient whose value never
actually changes (an interval duration, an efficiency). `FastStepHighs` accepts
these and **verifies the values each roll** instead of rejecting on the mutability
flag; a coefficient that genuinely changes is caught — never a stale-matrix solve.

```python
# What to do when a guarded/folded coefficient genuinely changes mid-run:
stepper = FastStepHighs(on_matrix_change='error')     # default: fail loud, name the Param
stepper = FastStepHighs(on_matrix_change='reload')    # opt-in: rebuild + reload (basis reset)
# comparison tolerance against the loaded values (default exact):
stepper = FastStepHighs(matrix_atol=0.0, matrix_rtol=0.0)
```

`set_instance(model, on_matrix_change=…, matrix_atol=…, matrix_rtol=…)` accepts the
same options. A practically-constant mutable param that appears *non-affinely*
(a `price*duration` product, a `duration/efficiency` reciprocal) is **folded** to a
watched constant so the rest of the model becomes affine and engages the warm path.
Inspect the classification to see *why* your model engaged and what the guard watches:

```python
stepper.folded_parameters       # e.g. ['dt', 'eff[0,0]', ...]  (watched constants)
stepper.templated_parameters    # e.g. ['price[0]', 'dem[0,0]', ...]  (the varying P)
stepper.classification_report() # {'n_folded':.., 'n_templated':.., 'folding_engaged':True, ...}
```

The guard's per-roll cost is one vectorized compare — **0.1–0.2%** of the warm tick
at scale (`bench/VALUEGUARD_REPORT.md`, `bench/STATICFOLD_REPORT.md`). A product of
two *genuinely varying* params (no static factor) has no correct affine template and
is rejected loudly — declare one factor immutable, or use `highs_fastload` for a
fresh compile per solve.

### Columnar-mutation twin — `VectorPersistentHighs`

If your model is built with the explicit-array components (§3), drive the same warm
HiGHS path directly from columnar mutation (no template extraction):

```python
import numpy as np, scipy.sparse as sp, pyomo.environ as pyo
from pyomo.contrib.vector import (VectorVar, VectorConstraint, VectorObjective,
                                  VectorPersistentHighs)

m = pyo.ConcreteModel()
m.x = VectorVar(pyo.RangeSet(0, 1), domain=pyo.NonNegativeReals, bounds=(0, 10))
m.c = VectorConstraint(A=sp.csr_matrix(np.array([[1.0, 1.0]])), x=m.x, ub=np.array([4.0]))
m.obj = VectorObjective(terms={m.x: np.array([-1.0, -1.0])}, sense=pyo.minimize)
m.x.construct(); m.c.construct(); m.obj.construct()

ph = VectorPersistentHighs(m)           # assemble + load once, keep the solver live
r1 = ph.solve()                         # r1.objective == -4.0
m.x.setub(1.0)                          # bulk-mutate bounds (or fix/unfix, set_row_bounds, deactivate_rows)
r2 = ph.solve()                         # warm re-solve, basis kept  -> -2.0
```

Between solves you mutate the columnar components directly — `m.x.setlb/setub`,
`m.x.fix/unfix`, `m.con.set_row_bounds`, `m.con.deactivate_rows` (all with an
optional `where=` selector) — and each mutation records the touched columns/rows
*dirty*; `solve()` expands and pushes only the dirty subset. A **structural** change
(a variable length, row count, or nonzero count) is rejected loudly
(`PersistentStructureError`) — build a fresh handle for a new structure.

---

## 5. Convex-quadratic objective (`c·x + ½·xᵀQx`) — the #1761 use case

Both the explicit-array API and `highs_fastload` support a convex-quadratic
**objective** (constraints stay linear). `Q` is the Hessian in `½·xᵀQx` (the
gurobipy / cvxpy convention); HiGHS applies the `½`.

```python
import numpy as np, scipy.sparse as sp, pyomo.environ as pyo
from pyomo.contrib.vector import VectorVar, VectorConstraint, VectorObjective
from pyomo.contrib.vector.highs import solve_highs

n = 4
Mrand = np.random.default_rng(0).normal(size=(n, n))
Q = Mrand.T @ Mrand + np.eye(n)          # SPD -> convex
c = np.random.default_rng(1).normal(size=n)

m = pyo.ConcreteModel()
m.x = VectorVar(pyo.RangeSet(0, n - 1), bounds=(0.0, 1.0))
m.bal = VectorConstraint(A=sp.csr_matrix(np.ones((1, n))), x=m.x, rhs=np.array([1.0]))
m.obj = VectorObjective(terms={m.x: c}, quadratic=sp.csr_matrix(Q), sense=pyo.minimize)
m.x.construct(); m.bal.construct(); m.obj.construct()
_, obj = solve_highs(m)
```

The transparent route works on an ordinary classic model with an `x[i]*x[j]`-built
objective, too:

```python
res = SolverFactory('highs_fastload').solve(classic_qp_model)   # extracts the Hessian, one passModel
```

`FastStepHighs` also supports a **static-Hessian** QP with a changing linear cost
(the rolling portfolio path). **Construction is 37–58× faster** than the classic QP
route (the #1761 symptom removed at the root); the **solve itself is at parity** —
same convex-QP HiGHS run (`bench/QUADRATIC_QP_REPORT.md`). **MIQP** (integer +
quadratic) and **non-convex QP** are rejected loudly (HiGHS solves neither);
quadratic *constraints* are out of scope (no HiGHS API).

---

## Scope & fail-loud discipline (all routes)

- **Linear continuous / MIP**, plus a **convex-quadratic objective**. Everything else
  is rejected loudly at compile / `set_instance`, pointing at a classic route —
  never a silently wrong answer.
- Fixed variables are substituted into the row bounds / objective offset and pinned
  out of the column space (the #3851 pitfall, handled centrally).
- Duals / reduced costs are mapped back for LPs; for MIPs HiGHS marks them invalid
  and the loader raises, matching the in-tree interfaces.
- **Compatibility contract:** any consumer that doesn't understand a vector component
  triggers lazy scalarization to classic objects (a one-time warning), after which
  the component behaves exactly like today's — see `docs/vector_proposal.md` §6.

## Tests & further reading

- **Correctness backbone:** `pyomo/contrib/vector/tests/` — 167 tests (142 pass +
  25 Gurobi-skip when `gurobipy` is absent). Run: `python -m pytest pyomo/contrib/vector/tests/`.
- **Design + migration proposal:** `docs/vector_proposal.md`.
- **Benchmark harness & per-phase reports index:** `bench/README.md`.
