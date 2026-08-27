# Value-Aware Static-Matrix Guard for `highs_faststep`

**Project:** Vectorized Model Construction for Pyomo (see the Phase-0/2/3/4 reports)
**Surface:** `highs_faststep` (`pyomo.contrib.vector.FastStepHighs`), the array-native
persistent **warm re-solve** interface delivered in Phase 4.
**Change:** the constraint-matrix rejection at `set_instance` is replaced by a
**value-aware guard** — *verify the values* instead of *trust the mutability flag*.

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12, Linux 6.17 x86-64, 12 CPUs
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- Reproducible with `bench/warm_faststep.py` (§ Reproducing).

---

## 1. The problem: a pessimistic flag

Phase 4's `highs_faststep` compiles a classic linear model to standard-form arrays
once, retains a live HiGHS, and re-solves each roll by pushing the changed
objective coefficients / RHS / bounds as vectorized affine templates over the
model's mutable-`Param` vector, keeping the warm basis. It treated the
**constraint matrix `A` as static**, and — crucially — it *rejected any model
whose matrix carried a mutable-`Param` coefficient*, on the mutability **flag**
alone:

```
IncompatibleModelError: ... constraint 'socc' has a mutable coefficient on
variable 'p' ... Matrix-coefficient updates are out of scope ...
```

But many real rolling models carry *nominally* mutable matrix coefficients whose
**values never actually change** between warm solves — an interval duration in a
state-of-charge recurrence (`soc[t] == soc[t-1] + eff*dur*p[t] - dem[t]`), an
efficiency, a per-step gain. Under an equal-interval roll only the prices and
forecasts move; the durations are constant, so `A` is static *in value* even
though a `Param` in it is *flagged* mutable. The flag was turning away a large,
perfectly warm-solvable class of models.

## 2. The change: verify the values

Instead of trusting the flag, `set_instance` now **captures** each mutable matrix
coefficient into a reusable component, the **`_MatrixGuard`**:

* which `(row, col)` `A`-entries each mutable `Param` feeds, keyed to the
  compiler's stable matrix identity;
* the **affine template** `coef_values = M @ P + shift` that produces those
  coefficients from the parameter vector `P` — the *same* template family the
  objective / RHS / bound groups already use (`_AffineArray`, with a `value()`
  residual fallback for an entry that is not affine in the parameters, e.g. one
  that references a fixed variable);
* the coefficient values HiGHS was loaded with (the baseline).

Every warm solve then re-evaluates the guarded coefficients **vectorized** — one
sparse `M @ P`, no per-coefficient Python walk (the Phase-4 discipline) — and
compares them to the baseline:

* **unchanged** (exact by default, or a configurable tight tolerance) → the
  matrix HiGHS holds is still correct, so the warm basis is kept. This comparison
  is the guard's *only* per-roll cost.
* **genuinely changed** → **never a stale-matrix solve.** The default
  `on_matrix_change='error'` fails loud, naming the offending `Param` /
  coefficient(s); the opt-in `on_matrix_change='reload'` transparently rebuilds
  the whole standard-form matrix and reloads it (a fresh `passModel`, basis reset)
  for that solve, then continues.

The template self-check at `set_instance` is unchanged in spirit: the matrix
template must reproduce the loaded matrix at the current `P` **and** a fresh
`value()` evaluation at a random perturbation of `P`, or the model is rejected.
So a coefficient that is genuinely *nonlinear in the parameters* (a product of two
`Param`s, say) still can't be templated affinely and is still rejected loudly.

### Built for forward compatibility

The guard's coefficient mapping — which mutable `Param`s feed which `A`-entries,
with what affine relation — is a **standalone, reusable component**, not a
throwaway check. A later batch matrix-update path can reuse exactly this mapping
to *apply* a genuinely-changing coefficient as batched `A` edits; the guard here
only **detects** change. That is the single reason the mapping is structured as
`(rows, cols, affine, baseline, provenance)` rather than a boolean.

---

## 3. Results — synthetic rolling suite

The synthetic `rolling_mpc` benchmark gains a **nominally-mutable-matrix-param
variant** (`build_pyomo(dims, mutable_matrix=True)`): the charge efficiency
becomes a mutable `Param` `eff[a,t]` on the state-of-charge recurrence — one
guarded matrix coefficient per SoC constraint — that the equal-interval roll
never changes. The pre-guard interface *rejected* this model; the value guard
accepts it, verifies it each roll, and keeps the warm basis.

Median warm tick over the roll sequence; `guard-check` is the isolated per-roll
verification cost (`M @ P` + compare); `guard overhead` is that as a fraction of
the value-guard warm tick; `leg vs static` is the value-guard tick ÷ the
pure-static `faststep` tick. Seed 1000, 30 rolls.

| size | guarded A-coeffs | faststep (static) | faststep (value-guard) | guard-check | guard overhead | leg vs static |
|---|---|---|---|---|---|---|
| 1e4 | 2,640  | 15.39 ms  | 15.89 ms  | 0.034 ms | **0.2%** | 1.03× |
| 1e5 | 25,600 | 294.49 ms | 296.22 ms | 0.195 ms | **0.1%** | 1.01× |

*(xs (4×20, 80 guarded coeffs, 20 rolls): static 0.85 ms, value-guard 0.87 ms,
guard-check 0.024 ms = 2.7% of tick, 1.03×.)*

### Reading the numbers

* **The guard engages** — the model the flag would have rejected is now accepted
  and warm-solved (2,640 and 25,600 guarded coefficients at 1e4 / 1e5).
* **The per-roll verification cost is negligible** — `0.1–0.2%` of the warm tick
  at scale (2.7% on the tiny `xs` model, where the fixed vectorized-compare cost
  is a larger share of a sub-millisecond tick). Well inside the `< 10%` target.
* **The value-guard leg tracks the pure-static path** — `1.01–1.03×`, i.e. within
  ~1–3%, so accepting a nominally-mutable matrix costs essentially nothing when
  the matrix in fact holds still.
* **Equivalence holds** — every roll's objective matches the APPSI-persistent
  route (and the fresh-build reference) to `< 1e-6` relative, across the static,
  value-guard, model-driven and array-driven legs.

---

## 4. Equivalence & guard gates (correctness)

`pyomo/contrib/vector/tests/test_faststep.py` pins the guard four ways:

1. **Static-matrix equivalence.** A rolling sequence on a model with a
   nominally-mutable matrix coefficient (a mutable `dur` on a SoC recurrence,
   held constant across rolls) matches a per-roll fresh `highs_fastload`
   build+solve — objective and termination — for **both** basis-kept and
   basis-reset runs.
2. **No false positives.** Repeated identical solves (the matrix param untouched)
   never trip the exact-comparison guard: the baseline is the template's *own*
   `M @ P` value at `set_instance`, so an unchanged roll compares bit-exact.
3. **Change is caught.** A genuine mid-run change to a matrix coefficient trips
   the guard — the default fail-loud (message names the `Param` and the
   constraint), the `on_matrix_change='reload'` rebuild (matches a fresh build,
   and subsequent static rolls still match afterward), a two-sided (range) row
   whose two split `A`-rows share the coefficient, a residual (fixed-variable)
   coefficient re-evaluated with `value()`, and the array-driven path (which
   cannot reload from the model, so a matrix change is always fatal there).
4. **Nonlinear-in-params still rejected.** A coefficient bilinear in the
   parameters (`p*q`) is rejected at `set_instance` by the affine self-check.

All prior Phase-1/2/3/4 tests stay green; **zero core-module change** — the whole
feature stays additive under `pyomo.contrib.vector`.

---

## 5. Scope

* **Value-guarded, not assumed static.** A mutable matrix coefficient is accepted;
  a coefficient that stays put warm-solves on the kept basis, one that genuinely
  changes fails loud or (opt-in) reloads — never a stale solve.
* **Reload is a full re-compile, not an incremental `A` edit.**
  `on_matrix_change='reload'` does a fresh standard-form compile + `passModel`
  (basis reset). Applying a genuinely-changing coefficient as *batched* `A`
  updates — reusing the guard's coefficient mapping — is deliberately **out of
  scope here** (a later stage); the reload path is the only change-handling in
  this change.
* Everything else inherited from Phase 4 (linear-only, structure-fingerprint
  check, residual fallback) is unchanged.
* **A mutable param that appears as a *bilinear* (or otherwise non-affine)
  coefficient** — e.g. an interval duration that multiplies *both* a price and a
  power variable, so a coefficient is `price · duration` or `efficiency ·
  duration`. Such a coefficient is not affine in the parameter vector, so the
  affine self-check rejected the model at `set_instance` (objective *and*
  matrix). **Update:** this is now handled by **verified-static parameter
  folding** — a practically-constant mutable param is folded in as a constant
  across **all** templates (so `price · duration` becomes the affine
  `duration_value · price`) and value-guarded like the matrix coefficients here;
  see `bench/STATICFOLD_REPORT.md`. A product of two *genuinely-varying* params
  (no static factor) is still rejected loudly.

---

## 6. Reproducing

```bash
# recreate the venv (see bench/README.md), then from the repo root:
bench/.venv/bin/python -m bench.warm_faststep --sizes 1e4,1e5 --rolls 30 \
    --out bench/results/valueguard_faststep.json

# the guard's unit + equivalence tests:
bench/.venv/bin/python -m pytest pyomo/contrib/vector/tests/test_faststep.py
```

---

## 7. Deliverables checklist

- [x] `set_instance` accepts a model with mutable matrix coefficients; builds the
      reusable matrix-coefficient template/mapping (`_MatrixGuard`)
- [x] Per-roll **vectorized** re-evaluation + compare vs the loaded matrix; warm
      basis kept when unchanged (per-roll overhead `< 10%` of the warm tick —
      measured `0.1–0.2%` at scale)
- [x] Genuine change never solved stale: fail-loud default (names the offending
      `Param`/coefficients) + opt-in `on_matrix_change='reload'` (fresh
      `passModel`, basis reset)
- [x] Equivalence gates: static-matrix rolls match fresh builds; a changing
      coefficient trips the guard (both fail-loud and reload tested); all existing
      faststep/fastload/Phase-3 tests green; zero core-module change
- [x] Public synthetic benchmark: nominally-mutable-matrix-param variant, guard
      leg within ~1–3% of the pure-static faststep path
- [x] Matrix-coefficient mapping structured as a reusable component for a later
      batch matrix-update stage (this change detects; a follow-up applies)

*An external private real-world rolling case was profiled against this interface
separately; those findings are tracked through the private channel, not in this
public repository.*
