# pyomo-build-bench

Stage-separated model-construction benchmark harness for the **vectorized model
construction** project (scoping doc: *Vectorized Model Construction for Pyomo*).
This is the Phase-0 measurement foundation: every later speedup claim is checked
against numbers this harness produces.

It measures the wall-clock cost of getting a model from an empty container to
"the solver has it", broken into the four stages the project targets:

| Stage | What it measures | Pyomo entry point |
|---|---|---|
| **construct** | build the model (one `VarData`/`ConstraintData` per index, one expression tree per constraint) | the model generator |
| **repn** | canonicalize to a matrix | `LinearStandardFormCompiler` (linear) / `generate_standard_repn` (quadratic) |
| **write** | emit a solver file | LP writer v2 (`model.write("*.lp")`) |
| **load** | hand the model to an in-process solver, *no solve* | APPSI HiGHS `set_instance` (#3888) |

Peak RSS is captured per case; each `(backend, model, size)` runs in its own
subprocess so peak memory is attributable to that one case and a crash or
size-limited-license error can't take down the suite. Results (timings + memory +
structural stats + environment metadata) are written as JSON.

## What's in here

```
bench/
  run_bench.py            # orchestrator + per-case worker (CLI below)
  harness/
    timing.py             # warmup+repeats timers, peak/current RSS
    stages.py             # construct/repn/write/load stage runners for a Pyomo model
    sysinfo.py            # environment capture (versions, commit, machine)
  models/
    network_flow.py       # dense multi-period min-cost flow (complete digraph)
    unit_commitment.py    # mixed sparse/dense MILP (binary commitment)
    facility_location.py  # capacitated facility location (+ quadratic variant)
    supply_chain.py       # ragged multi-echelon supply chain (sparse lanes)
    network_flow_vector.py # network flow built with pyomo.contrib.vector (fast path)
  comparators/
    array_native.py       # same LP built with numpy/scipy -> HiGHS (all sizes) / Gurobi matrix API (xs)
    linopy_impl.py        # idiomatic linopy (network flow, facility location)
  spikes/
    spike_a_columnar_var.py   # columnar Var object-creation & memory savings
    spike_b_template_expr.py  # template-expression numeric instantiation (go/no-go gate)
  analyze.py              # turns the result JSONs into the report's markdown tables
  results/                # JSON outputs + tables.md live here (published baseline)
```

Model sizes are named by their target linear-constraint-matrix nonzero count
(`1e4`, `1e5`, `1e6`, `1e7`); `xs` is sized under gurobipy's 2000-var/-constr
size-limited-license cap so the matrix-API comparator can run. The harness
reports *actual* sizes, so the target names are approximate.

## Reports (per-phase index)

Each phase of the vectorized-construction project published a self-contained
report next to this file; every speedup claim in the design proposal
(`docs/vector_proposal.md`) and the user guide (`pyomo/contrib/vector/README.md`)
is reproduced from one of these. All share the one environment recorded in each
report's header (Pyomo `6.10.2.dev0`, Python 3.12, numpy 2.5.2 · scipy 1.18.1 ·
highspy 1.15.1); each `(model, size)` case runs in its own subprocess.

| Report | Topic | Headline |
|---|---|---|
| [`PHASE0_REPORT.md`](PHASE0_REPORT.md) | baseline + feasibility spikes | per-row solver **load is 76–95%** of the coherent `build→solver` route; Pyomo is ~20–38× a raw array→HiGHS path; Spike A (columnar `Var`) ~360–540× alloc / ~0.14× memory; Spike B (template) 13–16× extraction, with a mandatory scalarization fallback |
| [`PHASE2_REPORT.md`](PHASE2_REPORT.md) | `highs_fastload` — transparent classic hand-off | hand-off **3.1–6.4×** faster; end-to-end up to **4.50×** (load-bound), 2.1–2.7× (construct-bound, shared-construct ceiling) |
| [`PHASE3_REPORT.md`](PHASE3_REPORT.md) | template-vectorized construction | **11.2× construct / 9.8× end-to-end** on a templatizable model; byte-identical classic fallback; no material slowdown off-subset |
| [`PHASE4_REPORT.md`](PHASE4_REPORT.md) | `highs_faststep` — persistent warm re-solve | warm tick **2.33×** at 1e5 nnz (3.39× at 1e4) vs persistent APPSI HiGHS |
| [`VALUEGUARD_REPORT.md`](VALUEGUARD_REPORT.md) | value-aware static-matrix guard | accept a nominally-mutable matrix coefficient; per-roll overhead **0.1–0.2%** |
| [`STATICFOLD_REPORT.md`](STATICFOLD_REPORT.md) | verified-static parameter folding | engage models with non-affine (`price·dur`) params; **5.02× / 3.49×** over APPSI on the same model |
| [`COMPILE_SCALING_REPORT.md`](COMPILE_SCALING_REPORT.md) | near-linear `set_instance` compile | super-linear → **near-linear** (byte-identical output); 19.6× at 1024 folds |
| [`QUADRATIC_QP_REPORT.md`](QUADRATIC_QP_REPORT.md) | convex-quadratic objective (#1761) | **37–58× construction**; solve at parity (solver-bound) |
| [`GUROBI_FASTLOAD_REPORT.md`](GUROBI_FASTLOAD_REPORT.md) | `gurobi_fastload` — Gurobi matrix hand-off | **~3.8–5.4×** hand-off at licensed sizes only (2000-var/-constr cap; no larger claim) |

Raw result JSONs are under `bench/results/`. The design proposal
(`docs/vector_proposal.md`) synthesizes these into the upstream case and the
`contrib → core` migration split.

## Running

The harness runs in a dedicated virtualenv (`bench/.venv`, not committed) holding
this repo's Pyomo installed editable at HEAD plus numpy/scipy and the comparators
(highspy, gurobipy, linopy). Recreate it with:

```bash
uv venv --python 3.12 bench/.venv
uv pip install --python bench/.venv/bin/python -e .          # this repo's Pyomo, editable
uv pip install --python bench/.venv/bin/python numpy scipy highspy gurobipy linopy
```

> gurobipy uses a size-limited license unless you have a full one; the harness
> only runs it at `xs` for that reason (see below). numpy/scipy/highspy/linopy
> are permissively licensed and public.

Then, from the repo root:

```bash
# small CI-runnable subset (all models/backends at xs + 1e4):
bench/.venv/bin/python -m bench.run_bench --suite ci --out bench/results/ci.json

# full manual sweep (1e4..1e6; add 1e7 explicitly, it is heavy):
bench/.venv/bin/python -m bench.run_bench --suite full --sizes 1e4,1e5,1e6 \
    --out bench/results/full.json

# a single model / backend / size:
bench/.venv/bin/python -m bench.run_bench --models network_flow --sizes 1e4,1e5 \
    --backends pyomo --out bench/results/network_flow.json

# the feasibility spikes (self-contained):
bench/.venv/bin/python -m bench.spikes.spike_a_columnar_var --out bench/results/spike_a.json
bench/.venv/bin/python -m bench.spikes.spike_b_template_expr --out bench/results/spike_b.json
```

Backends: `pyomo`, `pyomo_vector`, `linopy`, `arraynative_highs`,
`arraynative_gurobi`

`pyomo_vector` is the Phase-1 vectorized fast path (`pyomo.contrib.vector`):
columnar `VectorVar` + explicit-array `VectorConstraint` + `assemble` splice +
direct HiGHS `passModel` hand-off.  It is implemented for `network_flow` and its
stages are `construct` / `repn` (assemble) / `load` (passModel) — no `write`
stage (array-native LP emission is Phase 2).  Reproduce the Phase-1 headline:

```bash
bench/.venv/bin/python -m bench.run_bench --models network_flow \
    --backends pyomo,pyomo_vector,arraynative_highs --sizes 1e4,1e5,1e6 \
    --out bench/results/phase1_network_flow.json
bench/.venv/bin/python -m bench.analyze --phase1 bench/results/phase1_network_flow.json
```
(comma-separated or `all`). gurobipy runs at `xs` only (size-limited license).

### Phase 2 — transparent fast solver hand-off

The `pyomo` backend also measures a `fastload_highs` stage: the Phase-2
`highs_fastload` solver (`pyomo.contrib.vector.fastload`) compiling an
**unmodified** classic model to standard form and handing it to HiGHS via
`passModel` — the endpoint is identical to `load_highs`, so
`construct + fastload_highs` is the fast-route coherent total to compare against
the classic `construct + load_highs`.  Each result records the derived
`classic_build_to_solver_ms` / `fast_build_to_solver_ms` / `fastload_speedup`
and per-case solve-equivalence in `validation`.  Reproduce the Phase-2 tables
(see `bench/PHASE2_REPORT.md`):

```bash
bench/.venv/bin/python -m bench.run_bench --suite full \
    --models network_flow,unit_commitment,facility_location,supply_chain \
    --backends pyomo --sizes 1e4,1e5,1e6 --out bench/results/phase2_fastload.json
bench/.venv/bin/python -m bench.analyze --phase2 bench/results/phase2_fastload.json
```

## Notes / caveats

- **Baseline = stock upstream Pyomo at this clone's HEAD** (editable install).
  The environment (Python, package versions, commit, machine) is embedded in
  every results file under `sysinfo`.
- **`peak_rss_mb`** is the process peak and includes the interpreter + import
  baseline; **`model_rss_mb`** (peak minus the post-import baseline) is the
  memory attributable to building and processing that one model — use it for
  cross-backend memory comparisons.
- The **quadratic** facility variant is the R7 hard-ceiling probe: its objective
  is not loadable via APPSI HiGHS (`DegreeError`), so its `load` stage is N/A;
  construct/repn/write are still measured.
- Comparator coverage is deliberately partial in Phase 0 — see the baseline
  report for the rationale (`bench/PHASE0_REPORT.md`).
