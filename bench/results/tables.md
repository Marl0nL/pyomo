## Environment

- Pyomo `6.10.2.dev0` @ commit `2744a2446cc5`
- Python 3.12.13, Linux-6.17.7-ba29.fc43.x86_64-x86_64-with-glibc2.42, 12 CPUs
- numpy 2.5.2, scipy 1.18.1, highspy 1.15.1, gurobipy 13.0.3, linopy 0.9.1
- sweep complete: True

## Stage-dominance (largest size per model)

| model | size | construct % | repn % | write % | load % (of Σ) | **load % (of build→solver)** |
|-------|------|-------------|--------|---------|---------------|------------------------------|
| network_flow | 1e6 | 13% | 12% | 27% | 48% | 79% |
| unit_commitment | 1e6 | 6% | 9% | 15% | 70% | 92% |
| facility_location | 1e6 | 10% | 13% | 23% | 54% | 85% |
| facility_location_q | 1e5 | 18% | 33% | 49% | 0% | — |
| supply_chain | 1e6 | 10% | 12% | 21% | 57% | 85% |

_`% of Σ` divides by construct+repn+write+load (the non-pipeline sum), which **understates** load's true share; `% of build→solver` divides by the coherent construct+load route and is the honest dominance figure._


### network_flow

| size | vars | cons | nnz | construct | repn | write | load | Σ stages† | build→solver‡ | model RSS |
|------|------|------|-----|-----------|------|-------|------|-----------|---------------|-----------|
| 1e4 | 5,000 | 500 | 9990 | 15.3 | 12.8 | 25.8 | 60.2 | 114.1 | **75.5** | 97.41 MB |
| 1e5 | 50,000 | 2,500 | 99980 | 154.1 | 115.5 | 243.2 | 492.6 | 1005.4 | **646.7** | 215.63 MB |
| 1e6 | 500,000 | 10,000 | 999950 | 1521.4 | 1394.3 | 3144.2 | 5581.9 | 11.6s | **7103.3** | 1212.45 MB |

_All times median-of-repeats, milliseconds unless suffixed `s`._
_†`Σ stages` = construct+repn+write+load; a **sum of independently-measured, non-sequential passes**, NOT one pipeline (repn is a standalone compile; write and load are alternative routes to a solver)._
_‡`build→solver` = construct+load: the coherent in-memory (APPSI) route to 'the solver has it' — the number to compare across systems._


### unit_commitment

| size | vars | cons | nnz | construct | repn | write | load | Σ stages† | build→solver‡ | model RSS |
|------|------|------|-----|-----------|------|-------|------|-----------|---------------|-----------|
| 1e4 | 2,400 | 4,740 | 12500 | 12.5 | 17.7 | 29.9 | 130.4 | 190.4 | **142.9** | 114.25 MB |
| 1e5 | 24,000 | 47,910 | 127250 | 127.4 | 200.1 | 326.9 | 1359.2 | 2013.7 | **1486.6** | 223.56 MB |
| 1e6 | 240,000 | 480,300 | 1278500 | 1418.2 | 2246.5 | 3675.6 | 17.1s | 24.4s | **18.5s** | 1492.74 MB |

_All times median-of-repeats, milliseconds unless suffixed `s`._
_†`Σ stages` = construct+repn+write+load; a **sum of independently-measured, non-sequential passes**, NOT one pipeline (repn is a standalone compile; write and load are alternative routes to a solver)._
_‡`build→solver` = construct+load: the coherent in-memory (APPSI) route to 'the solver has it' — the number to compare across systems._


### facility_location

| size | vars | cons | nnz | construct | repn | write | load | Σ stages† | build→solver‡ | model RSS |
|------|------|------|-----|-----------|------|-------|------|-----------|---------------|-----------|
| 1e4 | 2,525 | 2,625 | 10025 | 10.1 | 11.7 | 22.0 | 57.1 | 100.8 | **67.2** | 127.11 MB |
| 1e5 | 25,050 | 25,550 | 100050 | 95.7 | 122.1 | 221.8 | 558.5 | 998.1 | **654.2** | 186.0 MB |
| 1e6 | 250,100 | 252,600 | 1000100 | 1079.2 | 1425.5 | 2498.0 | 5960.1 | 11.0s | **7039.3** | 980.68 MB |

_All times median-of-repeats, milliseconds unless suffixed `s`._
_†`Σ stages` = construct+repn+write+load; a **sum of independently-measured, non-sequential passes**, NOT one pipeline (repn is a standalone compile; write and load are alternative routes to a solver)._
_‡`build→solver` = construct+load: the coherent in-memory (APPSI) route to 'the solver has it' — the number to compare across systems._


### facility_location_q

| size | vars | cons | nnz | construct | repn | write | load | Σ stages† | build→solver‡ | model RSS |
|------|------|------|-----|-----------|------|-------|------|-----------|---------------|-----------|
| 1e4 | 2,525 | 2,625 | — | 12.2 | 21.4 | 32.1 | — | 65.7 | **—** | 89.3 MB |
| 1e5 | 25,050 | 25,550 | — | 121.9 | 221.0 | 330.1 | — | 673.0 | **—** | 194.21 MB |

_All times median-of-repeats, milliseconds unless suffixed `s`._
_†`Σ stages` = construct+repn+write+load; a **sum of independently-measured, non-sequential passes**, NOT one pipeline (repn is a standalone compile; write and load are alternative routes to a solver)._
_‡`build→solver` = construct+load: the coherent in-memory (APPSI) route to 'the solver has it' — the number to compare across systems._


### supply_chain

| size | vars | cons | nnz | construct | repn | write | load | Σ stages† | build→solver‡ | model RSS |
|------|------|------|-----|-----------|------|-------|------|-----------|---------------|-----------|
| 1e4 | 3,200 | 1,600 | 6388 | 10.5 | 10.4 | 20.0 | 54.7 | 95.6 | **65.2** | 93.71 MB |
| 1e5 | 28,680 | 12,600 | 57330 | 91.7 | 93.2 | 176.2 | 464.0 | 825.0 | **555.7** | 167.27 MB |
| 1e6 | 229,800 | 84,000 | 459520 | 755.8 | 836.7 | 1502.6 | 4178.5 | 7273.6 | **4934.3** | 707.58 MB |

_All times median-of-repeats, milliseconds unless suffixed `s`._
_†`Σ stages` = construct+repn+write+load; a **sum of independently-measured, non-sequential passes**, NOT one pipeline (repn is a standalone compile; write and load are alternative routes to a solver)._
_‡`build→solver` = construct+load: the coherent in-memory (APPSI) route to 'the solver has it' — the number to compare across systems._


## Comparators — network_flow

| size | nnz | Pyomo build→solver (construct+load) | array→HiGHS (build+load) | linopy (build+extract) | Gurobi (xs) | Pyomo Σ stages† | **Pyomo ÷ array-native‡** |
|------|-----|-----------------------------------|--------------------------|------------------------|-------------|-----------------|---------------------------|
| xs | — | — | — | — | 1.3 | — | — |
| 1e4 | 9990 | 75.5 | 2.8 | 40.1 | — | 114.1 | **27×** |
| 1e5 | 99980 | 646.7 | 19.4 | 50.5 | — | 1005.4 | **33×** |
| 1e6 | 999950 | 7103.3 | 184.8 | 438.3 | — | 11.6s | **38×** |

_‡Ratio is the coherent build→solver route on both sides (Pyomo construct+load vs array build+load), NOT Pyomo's `Σ stages` (which double-counts by summing a standalone repn compile plus the two alternative solver routes)._
_†`Σ stages` shown only for reference; see the per-model table's footnotes._
_linopy's endpoint is 'constraint matrix in memory', a step short of 'loaded in the solver', so it is not strictly comparable to the build→solver columns._


## Comparators — facility_location

| size | nnz | Pyomo build→solver (construct+load) | array→HiGHS (build+load) | linopy (build+extract) | Gurobi (xs) | Pyomo Σ stages† | **Pyomo ÷ array-native‡** |
|------|-----|-----------------------------------|--------------------------|------------------------|-------------|-----------------|---------------------------|
| xs | — | — | — | — | 1.4 | — | — |
| 1e4 | 10025 | 67.2 | 3.3 | 49.4 | — | 100.8 | **20×** |
| 1e5 | 100050 | 654.2 | 28.3 | 57.2 | — | 998.1 | **23×** |
| 1e6 | 1000100 | 7039.3 | 289.0 | 126.9 | — | 11.0s | **24×** |

_‡Ratio is the coherent build→solver route on both sides (Pyomo construct+load vs array build+load), NOT Pyomo's `Σ stages` (which double-counts by summing a standalone repn compile plus the two alternative solver routes)._
_†`Σ stages` shown only for reference; see the per-model table's footnotes._
_linopy's endpoint is 'constraint matrix in memory', a step short of 'loaded in the solver', so it is not strictly comparable to the build→solver columns._


## Equivalence oracle — array-native == Pyomo standard form

| model | size | vars | nnz | rows == (up to perm) | obj == | **equivalent** |
|-------|------|------|-----|----------------------|--------|----------------|
| facility_location | xs | 260 | 1,010 | ✅ | ✅ | ✅ yes |
| facility_location | 1e4 | 2,525 | 10,025 | ✅ | ✅ | ✅ yes |
| network_flow | xs | 384 | 760 | ✅ | ✅ | ✅ yes |
| network_flow | 1e4 | 5,000 | 9,990 | ✅ | ✅ | ✅ yes |

_Authoritative check: the constraint system as a multiset of sign-normalized rows, columns keyed by variable identity — equal iff the two LPs are the same standard form up to row/column permutation. `obj ==` solves both through HiGHS._


## Spike A — columnar Var

| N (vars) | classic build | columnar build | build speedup | classic B/var | columnar B/var | memory ratio |
|----------|---------------|----------------|---------------|---------------|----------------|--------------|
| 10,000 | 4.2 ms | 0.01 ms | 359× | 173 B | 26 B | 0.150 |
| 100,000 | 64.4 ms | 0.12 ms | 531× | 196 B | 26 B | 0.132 |
| 1,000,000 | 854.7 ms | 1.58 ms | 542× | 186 B | 26 B | 0.140 |


## Spike B — template vectorization

**Coverage — which rule shapes templatize:**

| rule shape | example | templatizes? | note |
|------------|---------|--------------|------|
| scalar-affine | `a*x[i] <= p[i]` | ✅ yes |  |
| scalar-affine (neighbour) | `2*x[i]-x[i-1] <= 1` | ✅ yes |  |
| sum-over-set | `sum(f[j,n,t] for j in N) balance` | ✅ yes |  |
| index-conditional | `if i==0: Skip else affine` | ❌ no | PyomoException: Cannot convert non-constant Pyomo expression |
| modulo index | `x[i]+x[(i-1)%N]` | ❌ no | TypeError: unsupported operand type(s) for %: 'NPV_SumExpres |
| quadratic | `x[i]*x[i] <= 1` | ✅ yes |  |

**Scalar-affine family (`2·x[i] − x[i−1] ≤ 1`): correctness + speed:**

| N (constraints) | vectorized == classic? | classic repn | resolve/idx | vectorized | **vec speedup** | resolve speedup |
|-----------------|------------------------|--------------|-------------|------------|-----------------|-----------------|
| 10,000 | ✅ | 29 ms | 193 ms | 2.2 ms | **13×** | 0.15× |
| 100,000 | ✅ | 306 ms | 1871 ms | 18.9 ms | **16×** | 0.16× |
| 1,000,000 | ✅ | 3303 ms | 18539 ms | 209.6 ms | **16×** | 0.18× |

**Sum-over-set family (unfiltered flow balance):**

| N | T | rows | templatizes? | classic repn | resolve/idx | resolve speedup |
|---|---|------|--------------|--------------|-------------|-----------------|
| 10 | 100 | 1,000 | ✅ | 17 ms | 90 ms | 0.19× |
| 20 | 250 | 5,000 | ✅ | 140 ms | 835 ms | 0.17× |



## Phase 2 — ragged supply-chain on the vector fast path + mutation cycle

**Ragged supply-chain (sparse multi-echelon lanes) — build-to-solver, vector API vs classic.**
Time-to-solver = construct + repn/assemble + HiGHS load (no solve). Vector build is
standard-form equivalent to the classic build (oracle Gate A) and solves to the same
objective (Gate C). `bench/results/phase2_supply_chain.json`.

| size | nnz | classic construct | classic repn | classic load | classic **TTS** | vector construct | vector assemble | vector load | vector **TTS** | **speedup** |
|------|-----|-------------------|--------------|--------------|-----------------|------------------|-----------------|-------------|----------------|-------------|
| xs   | 256    | 1.3 ms | 1.2 ms  | 3.9 ms  | 6.4 ms   | 1.4 ms  | 0.22 ms | 0.3 ms  | 1.9 ms  | 3.3× |
| 1e4  | 6,388  | 15.3 ms | 14.0 ms | 64.6 ms | 93.9 ms  | 6.9 ms  | 0.35 ms | 1.5 ms  | 8.8 ms  | **10.7×** |
| 1e5  | 57,330 | 124.8 ms | 124.6 ms | 551.1 ms | 800.4 ms | 51.4 ms | 1.17 ms | 10.4 ms | 62.9 ms | **12.7×** |

_The assemble stage (CSR stack, no tree walk) is ~100× cheaper than the classic repn walk;
the array `passModel` load is ~50× cheaper than per-row APPSI loading. This is the ragged
case linopy/xarray cannot express idiomatically (scoping doc R5) — no dense `S·W`/`W·R`
grid is ever allocated. Run: `python -m bench.run_bench --models supply_chain --backends pyomo,pyomo_vector --sizes 1e4,1e5`._

**Mutation cycle (fix / unfix / bounds / mask sweep) — persistent warm re-solve vs cold reload.**
One `VectorPersistentHighs` mutates the columnar components in place and pushes only the
dirty columns/rows through HiGHS `changeColsBounds`/`changeRowsBounds` (warm basis retained);
the cold route rebuilds + `passModel` each step. Both agree on every step's objective
(exact). `bench/results/phase2_mutation_cycle.json`.

| size | nnz | steps | warm re-solve (median) | cold reload (median) | **warm speedup** | objectives agree |
|------|-----|-------|------------------------|----------------------|------------------|------------------|
| xs   | 256   | 12 | 0.27 ms | 2.24 ms  | 8.4×  | ✅ (0 mismatch) |
| 1e4  | 6,388 | 12 | 1.02 ms | 18.95 ms | **18.6×** | ✅ (0 mismatch) |

_The `supply_chain` generator produces an infeasible instance at 1e5+ (a pre-existing data
property — the classic build is equally infeasible there); the mutation cycle therefore sweeps
the feasible sizes. Run: `python -m bench.mutation_cycle --sizes xs,1e4`._
