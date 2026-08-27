# Verified-Static Parameter Folding for `highs_faststep`

**Project:** Vectorized Model Construction for Pyomo (see the Phase-0/2/3/4 and
value-guard reports)
**Surface:** `highs_faststep` (`pyomo.contrib.vector.FastStepHighs`), the
array-native persistent **warm re-solve** interface.
**Change:** the affine self-check no longer rejects a model whose mutable Params
participate **non-affinely** (products / reciprocals).  Practically-constant
Params are **folded** -- substituted as watched constants during template
construction -- so the remaining coefficients become affine and the model
engages the warm path.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12, Linux 6.17 x86-64, 12 CPUs
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- Reproducible with `bench/warm_faststep.py` (§ Reproducing).

---

## 1. The problem: the value guard needs *affine* coefficients

Phase 4's `highs_faststep` expresses each mutable objective coefficient / row
bound / variable bound as an **affine template over the mutable-`Param` vector**
`P` -- `values = M @ P + shift` -- and the value guard (the previous change)
extended that to nominally-mutable **matrix** coefficients whose values hold
still.  Both rely on the coefficient being **affine in `P`**.

But real rolling models routinely put a practically-constant mutable Param into a
**product or reciprocal** with another quantity:

* an interval **duration** multiplying every electricity **price** in the
  objective (`price[t] * duration`);
* an **efficiency** times that duration on the state-of-charge recurrence
  (`efficiency * duration`), and a self-discharge **reciprocal**
  (`duration / efficiency`).

A product of two mutable Params is **not affine** in `P` (its partial w.r.t.
one factor still depends on the other), so the affine self-check could not
reproduce it under a parameter perturbation and **rejected the whole model** at
`set_instance` -- even though the duration and efficiency never actually change
between rolls.  The value guard was *necessary but not sufficient*: it verifies a
coefficient is static, but it cannot template a non-affine one in the first
place.

## 2. The change: fold the verified-static parameters

At `set_instance`, `FastStepHighs` now **classifies** each mutable Param
(`_classify_folded`) and **folds** the ones that participate non-affinely:

* **Forced folds.** A Param that appears in its *own* symbolic derivative -- a
  reciprocal `1/eff` or a power `dur**2` -- is non-affine in itself and can never
  be templated.  It must be folded.
* **Hub folds.** Two varying Params multiplied together (`price[t] * duration`)
  are mutually non-affine; folding either resolves it.  We fold the one appearing
  in the **most** such couplings -- the structural constant (a single `duration`
  coupling every `price[t]`) rather than the many varying data Params.  A
  coupling with **no hub** (two co-equal, single-use Params, a lone `p*q`) is left
  unfolded and rejected loudly: with no evidence which factor is static, folding
  one would be a coin-flip the guard would trip every roll.

A **folded** Param has its current value substituted as a constant during
template construction (across objective, matrix, RHS and bound templates alike),
so `price[t] * duration` becomes the affine `duration_value * price[t]` over the
remaining varying `price[t]` -- and the previously-rejected model engages.
Classification uses the **symbolic** derivative (`reverse_symbolic`) to detect
true non-affinity precisely; the decomposition and the `set_instance` self-check
(perturb the *varying* Params, hold the folded ones, verify the templates equal a
fresh `value()`) remain the hard correctness gate.

### Every folded Param is watched

A folded value baked into the templates must not change.  Each warm solve
verifies the folded values (vectorized -- the same compare the value guard uses)
against the `set_instance` baseline **before** touching the solver:

* **unchanged** -> the templates are still correct, keep the warm basis (the fast
  path -- the guard's only per-roll cost is this compare);
* **genuinely changed** -> never a stale-template solve.  The default
  `on_matrix_change='error'` fails loud naming the Param; the opt-in
  `on_matrix_change='reload'` **re-folds, re-templates, and reloads** a fresh
  model (basis reset) for that solve, then continues (the changed Param is
  re-classified at its new value and re-baselined, so a subsequent static roll is
  fast again).

### Classification transparency

`FastStepHighs` exposes which Params were folded vs templated so a user can see
*why* their model engaged and what the guard is watching:

```python
s = FastStepHighs(); s.set_instance(model)
s.folded_parameters        # ['dur', 'eff[0,0]', 'eff[0,1]', ...]
s.templated_parameters     # ['price[0]', 'dem[0,0]', ...]
s.classification_report()  # {'n_folded': .., 'n_templated': .., 'folding_engaged': True, ...}
```

An `INFO`-level log line reports the same at `set_instance`.

---

## 3. Results -- synthetic rolling suite

The synthetic `rolling_mpc` benchmark gains a **non-affine-param variant**
(`build_pyomo(dims, nonaffine_param=True)`): a practically-constant but *mutable*
interval duration `dt` enters the objective as `price[t] * dt` (a product) and a
self-discharge leak as `(dt / eff[a,t]) * soc` (a reciprocal in a mutable
`eff[a,t]`).  Neither coefficient is affine in `P`; the pre-folding interface
rejected this model.  Folding folds `dt` (the hub coupling every `price[t]`) and
every `eff[a,t]` (forced by the reciprocal), templates `price`/`dem`/`gcap`/
`pmax`, and warm-solves.

Median warm tick over the roll sequence; **speedup** = APPSI-persistent (on the
same non-affine-param model) ÷ faststep-fold; **fold vs static** = the fold leg ÷
the pure-static `faststep_model` tick; **reload** is the measured cost of one
folded-value change event under `on_matrix_change='reload'`.  Seed 1000.

| size | folded | templated | APPSI (fold model) | faststep (fold) | **speedup** | fold vs static | reload (1 event) | equiv |
|---|---|---|---|---|---|---|---|---|
| 1e4 | 2,641 | 5,720 | 79.24 ms | 15.79 ms | **5.02×** | 1.02× | 844.50 ms | yes |
| 1e5 | 25,601 | 52,480 | 940.13 ms | 269.51 ms | **3.49×** | 0.90× | 8097.74 ms | yes |

*(`xs` (4×20, 20 rolls): folded 81, templated 200, faststep-fold 0.91 ms,
3.15× APPSI, 1.06× static, reload 28 ms, equivalent.)*

The `fold vs static` column is a **cross-model** ratio (the non-affine-param
variant vs the base pure-static model) -- the two differ structurally (the fold
variant carries the reciprocal leak term), so the ratio lands slightly above or
below 1.0 from model geometry, not folding overhead.  The apples-to-apples cost
of folding is the per-roll verification: reading the folded vector and one
vectorized compare, negligible against the warm tick (the same order as the value
guard's `0.1-0.2%`).  The `speedup` column is apples-to-apples: APPSI-persistent
on the *same* non-affine model re-walks every `price*dt` / `dt/eff` coefficient
with `value()` each roll, which folding pays once at `set_instance`.

### Reading the numbers

* **Folding engages** -- the model the affine self-check would have rejected is
  now accepted and warm-solved, with `dt` and every `eff[a,t]` folded (watched
  constants) and the price/forecast Params templated.
* **The fold leg tracks the pure-static path** -- `1.02×` at 1e4 and `0.90×` at
  1e5 (a cross-model ratio, see below), i.e. the same performance class, never
  materially slower.  After folding, the *varying*-template work is the base
  model's; the folded values cost only a vectorized compare per roll.
* **The speedup over APPSI-persistent** on the *same* non-affine model is
  `5.02×` / `3.49×` -- larger than the pure-static leg's `3.45×` / `2.29×`,
  because APPSI must re-evaluate every non-affine `price*dt` / `dt/eff`
  coefficient with a per-coefficient `value()` walk each roll, a cost folding
  pays once at `set_instance` and never again on the fast path.
* **Equivalence holds** -- every roll's objective matches a per-roll fresh
  `highs_fastload` build to `< 1e-6` relative.
* **Reload is a one-time, bounded cost** -- a full re-fold + re-compile +
  `passModel` for the single change event, after which the fast path resumes.
* **Classification is a one-time `set_instance` cost.**  Folding classifies each
  non-affine coefficient with a symbolic derivative at `set_instance` (~4 s of the
  1e5 fold compile, a stress size with 25k+ product/reciprocal coefficients); it
  is paid **once per process** and amortized over the thousands of warm solves
  that follow, and is ~100× smaller at the realistic external-case scale.  The
  per-**roll** cost is only the folded-value compare.

---

## 4. Equivalence & fold gates (correctness)

`pyomo/contrib/vector/tests/test_faststep.py` pins folding:

1. **Engagement + report.** A model with `price*dur` (objective) and `eff*dur` /
   `dur/eff` (matrix) engages; `dur` and every `eff` fold, `price`/`dem` stay
   templated (`test_folding_engages_and_reports`).
2. **Rolling equivalence.** A rolling sequence matches per-roll fresh
   `highs_fastload` builds -- objective and termination -- for **both** basis-kept
   and basis-reset runs, and on the array-driven path
   (`test_folded_rolling_matches_fresh`, `test_folded_array_path_matches_model`).
3. **A folded change never solved stale.** A genuine change to a folded value
   trips the guard -- the default fail-loud (message names the Param), the
   `on_matrix_change='reload'` re-fold (matches a fresh build, and a subsequent
   static roll still matches), and the array-driven path (always fatal, cannot
   reload from arrays) (`test_folded_param_change_fails_loud`,
   `test_folded_param_change_reload`, `test_folded_change_array_mode_fatal`).
4. **Reciprocal forces a fold.** A lone `dur/eff` folds `eff` and warm-solves
   (`test_reciprocal_param_folds_and_solves`).
5. **No static factor -> still rejected.** A lone `p*q` of two co-equal varying
   Params has no fold and is rejected loudly, in the objective *and* the matrix
   (`test_lone_objective_bilinear_rejected`,
   `test_bilinear_param_coefficient_rejected`).
6. **Fully-affine models are unchanged.** The base MPC folds nothing
   (`test_no_folding_when_all_affine`).

All prior Phase-1/2/3/4 and value-guard tests stay green (**86** vector-contrib
tests total); **zero core-module change** -- the whole feature stays additive
under `pyomo.contrib.vector`.

---

## 5. Scope

* **Folded, not assumed constant.** A folded Param is watched every roll; a
  static value warm-solves on the kept basis, a genuine change fails loud or
  (opt-in) reloads -- never a stale-template solve.  Classification is a
  best-effort to make the templates affine; the `set_instance` self-check is the
  hard gate, and the guard is the runtime safety net.
* **Reload re-folds.** `on_matrix_change='reload'` on a folded-value change is a
  full re-`set_instance` (re-classify, re-template, fresh `passModel`) -- the
  templates themselves change, so the matrix-only reload is not enough.
* **A product of two genuinely-varying Params is still rejected.** There is no
  correct affine template for `p*q` when both vary; folding one would trip the
  guard every roll, so the model is rejected loudly (declare one factor immutable,
  or use `highs_fastload`).
* **Deferred:** applying a genuinely-changing folded/matrix coefficient as
  *batched* `A`/coefficient edits (reusing the guard's mapping) -- the reload path
  is the only change-handling here.  No Stage-1/2 array-hand-off work is in this
  change.

---

## 6. Reproducing

```bash
# recreate the venv (see bench/README.md), then from the repo root:
bench/.venv/bin/python -m bench.warm_faststep --sizes 1e4,1e5 --rolls 30 \
    --out bench/results/staticfold_faststep.json

# the folding unit + equivalence tests:
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/test_faststep.py
```

---

## 7. Deliverables checklist

- [x] `set_instance` classifies + **folds** verified-static (non-affine) mutable
      Params across all templates (objective / matrix / RHS / bounds); the
      previously-rejected product/reciprocal models engage (`_classify_folded`,
      `_affine_over_varying`)
- [x] Every folded Param joins the value-guard watch list; per-roll **vectorized**
      verify vs the `set_instance` baseline (reusing the PR-3 compare)
- [x] A genuine change never solved stale: fail-loud default (names the Param) +
      opt-in `on_matrix_change='reload'` (re-fold + re-template + fresh
      `passModel`, basis reset)
- [x] Classification transparency: `folded_parameters` / `templated_parameters` /
      `classification_report()` + an `INFO` log line
- [x] Equivalence gates: folded rolls match fresh builds (basis-kept + reset +
      array path); folded-change trips the guard (fail-loud + reload tested,
      reload re-folds correctly); a genuinely-varying `p*q` still rejected; all
      existing tests green (86); zero core-module change
- [x] Public synthetic benchmark: non-affine-param variant, fold leg within a few
      percent of the pure-static path, reload cost measured for one change event

*An external private real-world rolling case was profiled against this interface
separately; those findings are tracked through the private channel, not in this
public repository.*
