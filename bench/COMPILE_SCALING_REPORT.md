# Near-Linear `set_instance` Compile for `highs_faststep`

**Project:** Vectorized Model Construction for Pyomo (see the Phase-0/2/3/4,
value-guard, and static-fold reports)
**Surface:** `highs_faststep` (`pyomo.contrib.vector.FastStepHighs`), the
array-native persistent **warm re-solve** interface.
**Change:** performance only. The *warm re-solve* path is untouched; only the
one-time **`set_instance` compile** is made near-linear. The affine templates,
fold classification, value-guard behavior, and solve results are **byte-for-byte
identical** to the pre-change implementation (verified bit-equal on base /
value-guard / folding / multi-hub models).

## Environment

- Pyomo `6.10.2.dev0` (editable install of this worktree's HEAD)
- Python 3.12, Linux 6.17 x86-64, 12 CPUs
- numpy 2.5.2 · scipy 1.18.1 · highspy 1.15.x
- Reproducible with `bench/spikes/faststep_compile_scaling.py` (§ Reproducing).

---

## 1. The problem: a fast warm tick behind a slow one-time compile

`highs_faststep` compiles a classic linear model to standard-form arrays **once**
(`set_instance`) and then warm-re-solves thousands of times, each tick a
vectorized `M @ P` template evaluation plus a batched HiGHS push. The warm tick
is the win. But the one-time compile did not scale: on models that engage
**verified-static parameter folding** it grew **super-linearly**, so at the
large horizons where the warm win is biggest the compile became the blocker —
seconds at a few hundred intervals, minutes at a few thousand.

`set_instance` does four things per compile: the standard-form compile
(shared with `highs_fastload`), **fold classification**, **affine-template
construction**, and a **self-check**. Profiling attributed the cost to the
middle two, in two distinct failure modes.

### 1a. Super-linear: the fold classifier's greedy hub loop — `O(folds × nnz)`

Verified-static folding classifies each mutable `Param`: a *hub* parameter that
couples many coefficients non-affinely (a single interval `duration` multiplying
every `price[t]`) is folded to a watched constant. The classifier picks hubs
greedily — repeatedly fold the parameter covering the most unresolved couplings.
The original implementation **rebuilt the full candidate set and rescanned every
coupling on each fold**: `O(folds × couplings)`. On a model whose *number* of
hub folds grows with its size — a **per-asset / per-zone** structural constant,
one hub each — that is `O(nnz²)`.

The generic single-hub `rolling_mpc` model hides this (one `duration` ⇒ one
fold). Isolating the fold count `H` at a **fixed** coefficient count exposes it:

| folded hubs `H` | obj terms (fixed) | compile **before** | exponent | compile **after** |
|---:|---:|---:|:--|---:|
| 16   | 8000 | 0.62 s | — | 0.31 s |
| 64   | 8000 | 0.84 s | `H^0.22` | 0.29 s |
| 256  | 7936 | 1.82 s | `H^0.55` | 0.27 s |
| 1024 | 7168 | 5.25 s | `H^0.77` | 0.27 s |

Before: rising toward `H^1` (the `O(folds × nnz)` term taking over). **After:
flat** (`H^-0.01`) — a **19.6×** speedup at `H = 1024`, and no longer growing.

### 1b. High linear constant: three symbolic walks per coefficient

Even single-hub, the constant was high. Every mutable coefficient expression was
symbolically walked **three times** — once to build the fold couplings, once to
build the affine template, once to register the parameter vector — and
reverse-mode differentiation was re-run **once per parameter** (a full backward
walk each), so the per-coefficient `differentiate` dominated the compile.

---

## 2. The fix (how, not what)

Nothing about *what* is compiled changes — the fold set, the template arrays
`M`/`base`, the value-guard baseline, and the solve results are bit-identical.
Only the work to produce them changes:

- **Incremental greedy fold.** Track, per coupling, the parameters involved in
  its unresolved conflict; folding a parameter re-examines only the couplings it
  actually touched, and a lazy max-heap reproduces the **identical**
  `(coverage, name)` greedy selection (coverage only ever shrinks, and parameter
  names are unique, so the arg-max matches a full rescan exactly). `O(couplings)`
  total → flat in the fold count.
- **One coefficient signature per expression**, computed once and reused by
  classification, registration, and template construction (was three walks).
- **`reverse_sd` once per expression**, indexed per parameter, instead of
  `differentiate(e, wrt=p)` re-running the whole reverse-mode walk for each
  parameter. The indexed result is identical to the per-parameter call.
- **Structural product fast path.** A bare two-parameter product (`price*dt`) —
  the dominant objective-coefficient shape — has partials that are simply the
  *other* factor, read straight off the product (byte-for-byte what reverse-mode
  returns) with no differentiation walk. Any other shape falls back.
- **Lazy name resolution** for guard provenance and fold candidates: `getname`
  on indexed components is costly and was paid for every coefficient up front;
  now only fold candidates that reach the heap, and only on a fail-loud, resolve
  a `.name`.

---

## 3. Result: near-linear compile

`bench/spikes/faststep_compile_scaling.py sweep` on the synthetic `rolling_mpc`
model (`A = 8` assets, mutable-param roll data), `set_instance` compile only:

**Plain variant** (no folding):

| intervals `T` | nnz | compile **before** | compile **after** | speedup |
|---:|---:|---:|---:|:--:|
| 288  | 9,208   | 0.32 s | 0.16 s | 2.0× |
| 1000 | 31,992  | 1.08 s | 0.58 s | 1.9× |
| 2160 | 69,112  | 2.41 s | 1.34 s | 1.8× |
| 5000 | 159,992 | 5.69 s | 3.02 s | 1.9× |

`34 → ~18 µs/nnz`, exponent `n^1.0` (was already linear; constant halved).

**Folding variant** (`nonaffine_param=True` — the super-linear stressor):

| intervals `T` | nnz | folded | compile **before** | compile **after** | speedup |
|---:|---:|---:|---:|---:|:--:|
| 288  | 9,208   | ~2.7k  | 0.69 s  | 0.38 s | 1.8× |
| 1000 | 31,992  | ~9k    | 2.52 s  | 1.51 s | 1.7× |
| 2160 | 69,112  | ~19k   | 5.44 s  | 3.27 s | 1.7× |
| 5000 | 159,992 | ~45k   | 12.91 s | 7.74 s | 1.7× |

`80 → ~46 µs/nnz`, exponent **`n^1.0`** (near-linear).

### Exit-criteria anchors (folding variant, `A = 12`)

| model | nnz | compile **after** | target |
|---|---:|---:|:--:|
| ~2160-interval class | 103,668 | **4.9 s** | ≤ 5 s ✓ |
| ~21600-interval / 1e6-nnz class | 1,036,788 | **48.2 s** | ≤ 60 s ✓ |

The plain variant at 1e6 nnz compiles in ~20 s. Both anchors pass with margin,
and compile is near-linear in nnz across the sweep.

---

## 4. Correctness: identical output, guarded

The change is performance-only and asserts it three ways:

- **Bit-equality gate (development).** The full `_build_mutable_plan` output —
  fold set, parameter order, every template `M`/`base`, guard rows/cols/baseline
  — is byte-for-byte identical before and after on base, value-guard, folding,
  and multi-hub models.
- **`set_instance` self-check (runtime).** Unchanged: every template must still
  reproduce the compiled standard-form arrays at the current `P` **and** at a
  random perturbation, or the model is rejected loudly.
- **New tests (`TestFastStepCompileScaling`).** The incremental greedy matches a
  brute-force replay of the original algorithm on randomized and many-hub
  coupling structures; the structural product fast path matches the reverse-mode
  reference on both fast-path and fall-back shapes; a many-hub model folds and
  warm-solves to the fresh-build optimum.

The existing 38 `highs_faststep` equivalence tests and the full 86-test vector
suite remain green (90 with the new tests). The warm bench
(`bench/warm_faststep.py`) reports `equivalent=True` on every leg with unchanged
warm ticks — the warm path was not touched.

---

## 5. Reproducing

```bash
# near-linearity of the compile vs interval count (plain + folding variants)
bench/.venv/bin/python -m bench.spikes.faststep_compile_scaling sweep

# the super-linear stressor: compile vs folded-hub count at fixed nnz
bench/.venv/bin/python -m bench.spikes.faststep_compile_scaling hubs

# warm-tick + equivalence (unchanged by this compile-only change)
bench/.venv/bin/python -m bench.warm_faststep --sizes 1e4 --rolls 20
```

Before/after figures were taken by running the same spike against this branch and
against its `vectorisation` base.
