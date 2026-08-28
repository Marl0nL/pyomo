# `highs_faststep` — coefficient-patch report (warm matrix-coefficient updates)

**Project:** Vectorized Model Construction for Pyomo (see the Phase-4 report for the base warm interface, and the MPC-narrowing report for the masked window overlay this composes with)
**Feature:** `on_matrix_change='patch'` on `FastStepHighs` — a genuinely-changing **guarded** coefficient (a folded verified-static parameter, or a matrix-templated coefficient) becomes a first-class **warm update** (partial refold + sparse `changeCoeff`) instead of a fail-loud or a full reload.
**Baselines:** the guard's existing non-fatal option — `on_matrix_change='reload'` (rebuild the whole standard-form matrix and `passModel` it every tick, basis reset) — and the receding-horizon-today route, a fresh structural narrow solved cold through the persistent APPSI HiGHS interface.

## Environment

- Pyomo `6.10.2.dev0` (this worktree's branch)
- Python 3.12.13, Linux 6.17 x86-64, 12 CPUs
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.1
- Reproducible with `bench/coeff_patch_faststep.py` (§ Reproducing).

---

## 1. The problem — the honest asymmetry

The masked warm-update work (MPC-narrowing report) narrows a rolling-horizon MPC's
active window **without a structural change**, on the warm basis. But it left one
honest asymmetry, named in its own deferral list: *a matrix coefficient edited in
place* was still out of scope — a fold-participating or matrix-templated parameter
whose value **genuinely changes every tick** tripped the value guard and forced a
fail-loud (`'error'`) or a whole-matrix reload (`'reload'`) on that tick.

That is not a corner case. The production shape is a receding-horizon controller
whose **current control interval's remaining duration shrinks** as real time
advances within it: a mutable `dur[t]` on the state-of-charge recurrence's charge
term, `soc[a,t] == soc[a,t-1] + eff·dur[t]·p[a,t] − dem[a,t]`. Each tick the
window's first interval's duration shrinks — a small, **sparse** constraint-matrix
change (the `A` entries of the current interval only) that the guard rightly
refuses to let ride a stale-matrix warm solve. Under `'error'` the model is
rejected every tick; under `'reload'` it re-compiles and resets the basis every
tick. Both throw away the warm state for a handful of changed numbers.

## 2. The mechanism — patch, don't reload

`on_matrix_change='patch'` reuses the coefficient mapping the value guard already
built (which mutable Params feed which `A`-entries, with what affine relation) to
**apply** the change instead of only detecting it:

- **Partial refold (folded change).** The fold *classification is unchanged* — a
  patched fold is still folded. Only the values baked into the affected affine
  templates are re-derived, **in place**, for exactly the rows that folded value
  fed (`_AffineArray.refold_rows`). Because the fold set and parameter registry
  are the same ones `set_instance` used, the sparse column positions are
  identical; only the coefficients move, so the `M`-row edit is `O(affected nnz)`,
  not an `O(nnz)` rebuild. Every re-templated row is validated against a fresh
  `value()` at the model's current state — the `set_instance` self-check
  restricted to the affected subset — so a patch is never a stale (or wrong)
  template.
- **Sparse `changeCoeff` (matrix change).** The changed `A`-entries are pushed with
  per-entry `changeCoeff` (highspy exposes no batch matrix-entry update). The
  objective / bound groups ride the ordinary vectorized template push (which
  re-pushes every solve anyway).
- **The warm simplex basis is kept.** The re-solve is a handful of iterations.

The change-set is bounded by `patch_max_entries` (default auto `max(4096, nnz//4)`):
a larger change degrades to a reload with a one-line log note, never a pathological
entry-by-entry storm. A folded parameter feeding the loaded-once objective
**Hessian** likewise degrades (the Hessian is not templated). The patch **composes**
with the row-mask / variable-fix overlay (a coefficient patch and a window
narrowing apply in one warm step) and with the array-driven path for a matrix
(varying-parameter) change; a *folded* change stays fatal in array mode (the refold
re-derives templates from the model, which the array may not reflect).

`'patch'` is **opt-in** — the default stays the fail-loud `'error'`, so an existing
model's behavior is unchanged until it asks for the patch path.

## 3. Results — one shrinking-first-interval MPC cycle

Per cycle: shrink the window's first-interval duration, roll the price/demand data,
slide the window mask, warm solve. Timed three ways at a day-length horizon
(`day288`, T=288) and the `1e5`-nonzero class, 20 cycles, median.

| size | A × T (win W) | nnz | entries/tick | patch apply | guard detect | classic APPSI | reload/tick | **patched warm** | **vs reload** | vs APPSI | equiv |
|---|---|---|---|---|---|---|---|---|---|---|---|
| day288 | 6×288 (W=48) | 6,906 | 6 | **10.2 µs** | 593 µs | 26.36 ms | 36.72 ms | **3.54 ms** | **10.38×** | 7.45× | yes |
| 1e5 | 40×640 (W=64) | 102,360 | 40 | **47.3 µs** | 7.13 ms | 244.60 ms | 522.74 ms | **37.24 ms** | **14.04×** | 6.57× | yes |

### Reading the numbers

- **The patch itself is microseconds-class.** `patch apply` — the `changeCoeff`
  loop over the changed entries — is 10.2 µs for 6 entries and 47.3 µs for 40
  (≈ 1–1.7 µs/entry, matching the measured per-call `changeCoeff` cost). That is
  the target: microseconds for a handful of entries.
- **`guard detect` is the value guard's pre-existing cost, not the patch.** It is
  the one vectorized `M @ P` that re-verifies *every* guarded coefficient each tick
  (25,600 guarded `dur` entries at `1e5`) to find the handful that moved — the same
  cost paid on the fast path when nothing changes. The patch adds only the tiny
  `apply` on top.
- **10–14× the reload route, 6.5–7.5× the classic narrow.** The full warm tick
  (detect + apply + window slide + solve, basis kept) is an order of magnitude
  faster than rebuilding the matrix every tick, and several times faster than a
  fresh structural narrow through APPSI.
- **Equivalence is exact.** Every cycle's window objective matches the classic
  APPSI narrow *and* the reload route to `< 1e-6` (feasible 20/20 both sizes).

## 4. Why the patch is exact (the correctness claim)

- **Same decomposition as `set_instance`.** A refolded row is re-derived by the
  identical `_coef_signature` / `_affine_from_sig` path `set_instance` ran, reading
  the new folded values off the model, with the same fold set and parameter
  registry. The construction is the one the `set_instance` self-check already
  proved affine and matching; refolding only substitutes new constants into it.
- **Validated on the affected subset.** Each refolded row is checked against a
  fresh `value()` at the model's current state before the solve (the self-check,
  narrowed to the patched rows). A mismatch fails loud — never a wrong template.
- **Guard invariant maintained.** After a matrix patch, `guard.baseline` is set to
  what HiGHS now holds, so the next tick detects the next change against the truth.
- **Never a stale solve.** A change too dense for a per-entry patch, or one
  touching the un-templated Hessian, degrades to a reload; a folded change in array
  mode stays fatal. The fail-loud posture is preserved end to end.

The test suite proves this as an equivalence gate: rolling sequences where guarded
values change every solve (including the shrinking pattern, and folded + masked
combined) match fresh structurally-rebuilt references solve-for-solve, in both
basis modes; unchanged-value solves take the zero-cost path; and `'patch'` and
`'reload'` agree solve-for-solve on the same sequence.

## 5. Scope

- **In scope:** a genuinely-changing folded (verified-static) parameter, and a
  genuinely-changing matrix-templated coefficient, on a fixed structure — patched
  in place with the basis kept. Objective / bound changes ride the existing
  vectorized template push. Composes with the mask/fix window overlay.
- **Degrades to reload (never wrong):** a change-set larger than `patch_max_entries`;
  a folded parameter feeding the objective Hessian; a matrix change whose recompile
  would shift the matrix *shape* (a coefficient rolling to structural zero) — the
  reload path already rebuilds the whole instance for that.
- **Still fatal:** a folded change on the array-driven (`param_values`) path (the
  refold needs the model's varying params current); a structure change between
  solves (caught by the fingerprint); a coefficient with no correct affine template
  (a product of two genuinely-varying params — rejected at `set_instance`).

## 6. Reproducing

```bash
bench/.venv/bin/python -m bench.coeff_patch_faststep --sizes day288,1e5 --cycles 20 \
    --out bench/results/coeff_patch_faststep.json
```

`--sizes` selects from `xs` / `day288` / `1e5`; `--cycles` sets the number of
shrink+narrow+solve cycles the median is taken over. The committed JSON in
`bench/results/coeff_patch_faststep.json` is a 20-cycle run on the environment
above.

## 7. Deferrals

- **A batch matrix-entry push.** highspy 1.15.1 exposes only per-entry
  `changeCoeff`; the patch loop is bounded (`patch_max_entries`) and measured to
  beat a reload far past that cap, so a batch API is a pure upside if one lands
  upstream — no design change needed here.
- **A skippable refold validation.** The affected-subset `value()` re-check is
  always on (the module's fail-loud ethos, and cheap for a sparse change bounded by
  the degrade threshold). A `patch_validate=False` fast path for callers who trust
  the templates is a follow-up.
- **A dirty-set-aware matrix patch.** A matrix change is detected by re-verifying
  every guarded entry (the value guard's design); a `dirty`-mask restriction of
  that scan is possible but was not needed for the sparse MPC shape.
- **Auto-`'patch'` default.** The default stays `'error'` to preserve existing
  fail-loud behavior; promoting `'patch'` to a default is a separate policy call.
