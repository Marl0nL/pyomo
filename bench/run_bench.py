"""pyomo-build-bench: stage-separated model-construction benchmark harness.

Measures the wall-clock cost of getting a model from an empty container to
"the solver has it", broken into stages, for:

  * four synthetic Pyomo models (network flow, unit commitment, facility
    location [+ quadratic variant], ragged supply chain),
  * array-native comparators (linopy, gurobipy matrix API, raw scipy->HiGHS).

Each (backend, model, size) case runs in its own subprocess so peak RSS is
attributable to that single case and a crash/size-limit in one case can't take
down the suite.

Usage
-----
  # Small CI-runnable subset:
  python -m bench.run_bench --suite ci --out bench/results/ci.json

  # Full manual sweep (sizes up to 1e6; add 1e7 explicitly):
  python -m bench.run_bench --suite full --out bench/results/full.json
  python -m bench.run_bench --suite full --sizes 1e4,1e5,1e6,1e7 --out ...

  # A single model/backend:
  python -m bench.run_bench --models network_flow --sizes 1e4,1e5 --backends pyomo

Internal worker mode (spawned per case):
  python -m bench.run_bench --single '<json-spec>'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional

# Ensure the repo root (parent of bench/) is importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bench.harness import timing, sysinfo  # noqa: E402


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #
def _pyomo_models() -> Dict[str, Any]:
    from bench.models import (
        network_flow,
        unit_commitment,
        facility_location,
        supply_chain,
    )

    reg = {
        "network_flow": network_flow,
        "unit_commitment": unit_commitment,
        "facility_location": facility_location,
        "supply_chain": supply_chain,
    }
    return reg


# Logical model "facility_location_q" reuses the facility module with the
# quadratic objective turned on.
_QUAD_ALIAS = {"facility_location_q": "facility_location"}

PYOMO_MODEL_NAMES = [
    "network_flow",
    "unit_commitment",
    "facility_location",
    "facility_location_q",
    "supply_chain",
]

COMPARATOR_MODEL_NAMES = ["network_flow", "facility_location"]

# Models with a pyomo.contrib.vector fast-path implementation (Phase 1).
VECTOR_MODEL_NAMES = ["network_flow"]

ALL_BACKENDS = [
    "pyomo",
    "pyomo_vector",
    "linopy",
    "arraynative_highs",
    "arraynative_gurobi",
]


def _resolve_pyomo(model: str):
    reg = _pyomo_models()
    base = _QUAD_ALIAS.get(model, model)
    mod = reg[base]
    is_quad = model.endswith("_q")
    return mod, is_quad


def _sizes_for(model: str) -> Dict[str, Dict[str, Any]]:
    mod, _ = _resolve_pyomo(model)
    return mod.SIZES


# --------------------------------------------------------------------------- #
# Worker: run one case in-process, return a result dict
# --------------------------------------------------------------------------- #
def run_pyomo_case(model: str, size: str, params: Dict[str, Any], repeats: int, warmup: int,
                   validate: bool) -> Dict[str, Any]:
    import pyomo.environ as pyo  # noqa: F401
    from bench.harness import stages

    mod, is_quad = _resolve_pyomo(model)
    build_params = dict(params)
    if is_quad:
        build_params["quadratic"] = True

    result: Dict[str, Any] = {"stages": {}}
    result["rss_before_build_mb"] = round(timing.current_rss_mb(), 2)

    # Stage 1: construct (fresh build each iteration).
    con_timing, m = timing.time_construct(
        "construct", lambda: mod.build_pyomo(build_params), repeats=repeats, warmup=warmup
    )
    result["stages"]["construct"] = con_timing.as_dict()

    # Structural stats + canonical nnz.
    st = stages.model_stats(m)
    nnz = stages.constraint_matrix_nnz(m) if not is_quad else stages.constraint_matrix_nnz(m)
    st["nnz"] = nnz
    result["stats"] = st

    # Stage 2: repn.
    if is_quad:
        result["stages"]["repn"] = timing.time_callable(
            "repn", lambda: stages.stage_repn_quadratic(m), repeats=repeats, warmup=warmup
        ).as_dict()
        result["repn_kind"] = "generate_standard_repn(quadratic=True)"
    else:
        result["stages"]["repn"] = timing.time_callable(
            "repn", lambda: stages.stage_repn(m), repeats=repeats, warmup=warmup
        ).as_dict()
        result["repn_kind"] = "LinearStandardFormCompiler"

    # Stage 3: write LP.
    tmpdir = tempfile.mkdtemp(prefix="bench_write_")
    lp_path = os.path.join(tmpdir, "m.lp")
    lp_size = {"bytes": None}

    def _write_lp():
        lp_size["bytes"] = stages.stage_write_lp(m, lp_path)

    try:
        result["stages"]["write_lp"] = timing.time_callable(
            "write_lp", _write_lp, repeats=repeats, warmup=warmup
        ).as_dict()
        result["lp_bytes"] = lp_size["bytes"]
    except Exception as e:
        result["stages"]["write_lp"] = {"error": f"{type(e).__name__}: {e}"}

    # Stage 4: load into HiGHS (no solve).
    def _load():
        h = stages.stage_load_highs(m)
        return h

    try:
        result["stages"]["load_highs"] = timing.time_callable(
            "load_highs", _load, repeats=max(3, repeats // 2), warmup=warmup
        ).as_dict()
    except Exception as e:
        result["stages"]["load_highs"] = {"error": f"{type(e).__name__}: {e}"}

    # Optional correctness validation (small sizes only): solve and record obj.
    if validate:
        try:
            res = stages.solve_highs(m)
            tc = str(getattr(res, "termination_condition", "n/a"))
            obj = None
            try:
                obj = float(pyo.value(m.obj))
            except Exception:
                pass
            result["validation"] = {"termination": tc, "objective": obj}
        except Exception as e:
            result["validation"] = {"error": f"{type(e).__name__}: {e}"}

    _finalize_total(result, ["construct", "repn", "write_lp", "load_highs"])
    return result


def run_pyomo_vector_case(model: str, size: str, params: Dict[str, Any],
                          repeats: int, warmup: int, validate: bool) -> Dict[str, Any]:
    """Fast-path (pyomo.contrib.vector) pipeline: construct -> assemble -> passModel.

    Stages mirror the classic ``pyomo`` backend so the results tables line up:

      construct  build the columnar VectorVar / VectorConstraint / VectorObjective
                 model (bulk array allocation + explicit CSR assembly)
      repn       ``assemble`` the components into one standard-form array stack
      load       hand the arrays to HiGHS via ``passModel`` (the load prize, #3888)

    There is no ``write`` stage: array-native LP-file emission is Phase 2 (scoping
    doc §6.3).  The endpoint ("the solver has the model") is the same as the
    classic backend's ``load`` stage, so construct+repn+load is the comparable
    end-to-end total.
    """
    if model not in VECTOR_MODEL_NAMES:
        return {"skipped": f"vector fast path not implemented for {model} (Phase 1)"}
    from bench.models import network_flow_vector
    from pyomo.contrib.vector import assemble, matrices_to_highs_lp
    import highspy

    result: Dict[str, Any] = {"stages": {}}
    result["rss_before_build_mb"] = round(timing.current_rss_mb(), 2)

    # Stage 1: construct (fresh vector model each iteration).
    con_timing, m = timing.time_construct(
        "construct", lambda: network_flow_vector.build_pyomo(params),
        repeats=repeats, warmup=warmup,
    )
    result["stages"]["construct"] = con_timing.as_dict()

    # Stage 2: repn (assemble the components into standard-form arrays).
    rep_timing, mx = timing.time_construct(
        "repn", lambda: assemble(m), repeats=repeats, warmup=warmup
    )
    result["stages"]["repn"] = rep_timing.as_dict()
    result["repn_kind"] = "pyomo.contrib.vector.assemble"

    # Structural stats straight from the arrays (no scalarization).
    result["stats"] = {
        "n_vars": int(mx.n_var),
        "n_constraints": int(mx.n_row),
        "nnz": int(mx.nnz),
    }

    # Stage 3: load into HiGHS via passModel (build the HighsLp + hand off).
    def _load():
        lp = matrices_to_highs_lp(mx)
        h = highspy.Highs()
        h.silent()
        h.passModel(lp)
        return h

    result["stages"]["load_highs"] = timing.time_callable(
        "load_highs", _load, repeats=max(3, repeats // 2), warmup=warmup
    ).as_dict()

    # Correctness validation (small sizes): standard-form equivalence vs the
    # stock compiler on the classic model, plus solve-objective agreement.
    if validate:
        result["validation"] = _validate_vector(model, params, m)

    _finalize_total(result, ["construct", "repn", "load_highs"])
    return result


def _validate_vector(model: str, params: Dict[str, Any], vector_model) -> Dict[str, Any]:
    """Check the fast path against the classic path at a small size.

    Three independent gates:

      A. the fast *splice* (``compile_standard_form``, no scalarization) matches
         the stock ``LinearStandardFormCompiler`` on the classic model;
      B. the committed harness oracle (``bench.equivalence``): the vector model,
         scalarized and read the classic per-constraint way, matches the
         array-native ground truth up to row/column permutation;
      C. the direct ``passModel`` solve objective matches a classic APPSI solve.
    """
    out: Dict[str, Any] = {}

    # A. Fast splice vs stock compiler (self-contained, keeps vector_model clean).
    try:
        from bench.models import network_flow as classic_nf
        from pyomo.contrib.vector import compile_standard_form
        from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
        from pyomo.contrib.vector.tests.equivalence_oracle import (
            canonical_standard_form,
        )

        classic = classic_nf.build_pyomo(params)
        iv = compile_standard_form(vector_model, mixed_form=True)
        ic = LinearStandardFormCompiler().write(classic, mixed_form=True)
        out["fast_splice_equivalent"] = (
            canonical_standard_form(iv) == canonical_standard_form(ic)
        )
    except Exception as e:
        out["fast_splice_error"] = f"{type(e).__name__}: {e}"

    # B. Committed harness equivalence oracle: (scalarized) vector model vs the
    # array-native ground truth (the artifact firstmate asked Phase 1 to consume).
    try:
        from bench import equivalence as eq
        from bench.comparators import array_native

        mx = array_native.BUILDERS[model](dict(params))
        fresh = network_flow_vector_build(params)  # scalarized by the oracle walk
        Pv = eq.pyomo_standard_form(fresh)
        Av = eq.array_standard_form(mx)
        out["oracle_rows_equal"] = bool(Pv["rows"] == Av["rows"])
        out["oracle_bounds_equal"] = bool(Pv["bounds"] == Av["bounds"])
        out["oracle_obj_equal"] = bool(Pv["obj_terms"] == Av["obj_terms"])
        out["oracle_var_names_equal"] = bool(Pv["var_names"] == Av["var_names"])
        out["oracle_equivalent"] = all(
            out[k] for k in (
                "oracle_rows_equal", "oracle_bounds_equal",
                "oracle_obj_equal", "oracle_var_names_equal",
            )
        )
    except Exception as e:
        out["oracle_error"] = f"{type(e).__name__}: {e}"

    # C. Solve-objective agreement (fast passModel vs classic APPSI HiGHS).
    try:
        from bench.models import network_flow as classic_nf
        from bench.harness import stages
        from pyomo.contrib.vector.highs import solve_highs
        import pyomo.environ as pyo

        _, ov = solve_highs(network_flow_vector_build(params))
        classic = classic_nf.build_pyomo(params)
        stages.solve_highs(classic)
        oc = float(pyo.value(classic.obj))
        out["objective_fast"] = round(ov, 6)
        out["objective_classic"] = round(oc, 6)
        out["objective_match"] = abs(ov - oc) <= 1e-5 * max(1.0, abs(oc))
    except Exception as e:
        out["solve_error"] = f"{type(e).__name__}: {e}"
    return out


def network_flow_vector_build(params):
    from bench.models import network_flow_vector

    return network_flow_vector.build_pyomo(params)


def run_linopy_case(model: str, size: str, params: Dict[str, Any], repeats: int,
                    warmup: int) -> Dict[str, Any]:
    from bench.comparators import linopy_impl

    if model not in linopy_impl.SUPPORTED:
        return {"skipped": f"linopy comparator not implemented for {model}"}
    builder = linopy_impl.BUILDERS[model]

    result: Dict[str, Any] = {"stages": {}}
    result["rss_before_build_mb"] = round(timing.current_rss_mb(), 2)
    build_timing, m = timing.time_construct(
        "build", lambda: builder(params), repeats=repeats, warmup=warmup
    )
    result["stages"]["build"] = build_timing.as_dict()
    result["stages"]["extract"] = timing.time_callable(
        "extract", lambda: linopy_impl.extract(m), repeats=repeats, warmup=warmup
    ).as_dict()
    result["stats"] = linopy_impl.stats(m)
    _finalize_total(result, ["build", "extract"])
    return result


def run_arraynative_case(model: str, size: str, params: Dict[str, Any], repeats: int,
                         warmup: int, solver: str) -> Dict[str, Any]:
    from bench.comparators import array_native

    if model not in array_native.SUPPORTED:
        return {"skipped": f"array-native comparator not implemented for {model}"}
    builder = array_native.BUILDERS[model]

    result: Dict[str, Any] = {"stages": {}}
    result["rss_before_build_mb"] = round(timing.current_rss_mb(), 2)
    build_timing, mx = timing.time_construct(
        "build_matrices", lambda: builder(params), repeats=repeats, warmup=warmup
    )
    result["stages"]["build_matrices"] = build_timing.as_dict()
    result["stats"] = {"n_vars": mx.n_var, "n_constraints": mx.n_con, "nnz": mx.nnz}

    if solver == "highs":
        loader = array_native.load_highs
        stage = "load_highs"
    elif solver == "gurobi":
        loader = array_native.load_gurobi
        stage = "load_gurobi"
    else:
        raise ValueError(solver)

    try:
        result["stages"][stage] = timing.time_callable(
            stage, lambda: loader(mx), repeats=max(3, repeats // 2), warmup=warmup
        ).as_dict()
        _finalize_total(result, ["build_matrices", stage])
    except Exception as e:
        result["stages"][stage] = {"error": f"{type(e).__name__}: {e}"}
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _finalize_total(result: Dict[str, Any], stage_order: List[str]) -> None:
    """Sum the median of every stage that succeeded.

    ``total_median_ms`` is the sum over *available* stages (so a model whose
    load stage is unsupported - e.g. the quadratic variant on APPSI HiGHS - still
    gets a meaningful construct+repn+write total); ``total_complete`` records
    whether every expected stage was measured.
    """
    total = 0.0
    ok = True
    for s in stage_order:
        d = result["stages"].get(s)
        if not d or "median_ms" not in d:
            ok = False
            continue
        total += d["median_ms"]
    result["total_median_ms"] = round(total, 4)
    result["total_complete"] = ok
    result["stage_order"] = stage_order


def run_single(spec: Dict[str, Any]) -> Dict[str, Any]:
    backend = spec["backend"]
    model = spec["model"]
    size = spec["size"]
    params = spec["params"]
    repeats = spec.get("repeats", 5)
    warmup = spec.get("warmup", 1)
    validate = spec.get("validate", False)

    out: Dict[str, Any] = {
        "backend": backend,
        "model": model,
        "size": size,
        "params": params,
        "repeats": repeats,
        "warmup": warmup,
        "ok": False,
        "error": None,
    }
    try:
        if backend == "pyomo":
            r = run_pyomo_case(model, size, params, repeats, warmup, validate)
        elif backend == "pyomo_vector":
            r = run_pyomo_vector_case(model, size, params, repeats, warmup, validate)
        elif backend == "linopy":
            r = run_linopy_case(model, size, params, repeats, warmup)
        elif backend == "arraynative_highs":
            r = run_arraynative_case(model, size, params, repeats, warmup, "highs")
        elif backend == "arraynative_gurobi":
            r = run_arraynative_case(model, size, params, repeats, warmup, "gurobi")
        else:
            raise ValueError(f"unknown backend {backend}")
        out.update(r)
        out["ok"] = "skipped" not in r
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    peak = timing.peak_rss_mb()
    out["peak_rss_mb"] = round(peak, 2)
    before = out.get("rss_before_build_mb")
    if before is not None:
        out["model_rss_mb"] = round(max(0.0, peak - before), 2)
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _default_sizes(suite: str) -> List[str]:
    if suite == "ci":
        return ["1e4"]
    return ["1e4", "1e5", "1e6"]


def _plan(suite: str, models: List[str], backends: List[str], sizes: List[str],
          repeats: int, warmup: int) -> List[Dict[str, Any]]:
    """Build the list of case specs to run."""
    specs: List[Dict[str, Any]] = []

    def small(size: str) -> bool:
        return size in ("xs", "1e4")

    def repeats_for(size: str) -> int:
        # Huge models are expensive to rebuild; the min-of-repeats statistic is
        # robust with fewer samples, so cap repeats at large sizes to bound
        # wall-clock and peak memory.
        if size in ("1e6", "1e7"):
            return min(repeats, 3)
        return repeats

    for backend in backends:
        if backend == "pyomo":
            model_pool = [m for m in models if m in PYOMO_MODEL_NAMES]
            for model in model_pool:
                model_sizes = _sizes_for(model)
                for size in sizes:
                    # A model may define only a subset of the size keys; skip
                    # any size this model doesn't define.
                    if size not in model_sizes:
                        continue
                    # The quadratic variant is the R7 hard-ceiling probe; its
                    # value is characterizing repn/write behaviour, not scaling.
                    # Cap it at 1e5 (1e6 quadratic is slow and low-signal).
                    if model == "facility_location_q" and size in ("1e6", "1e7"):
                        continue
                    specs.append({
                        "backend": backend, "model": model, "size": size,
                        "params": dict(model_sizes[size]),
                        "repeats": repeats_for(size), "warmup": warmup,
                        "validate": small(size),
                    })
        elif backend == "pyomo_vector":
            for model in [m for m in models if m in VECTOR_MODEL_NAMES]:
                model_sizes = _sizes_for(model)
                for size in sizes:
                    if size not in model_sizes:
                        continue
                    specs.append({
                        "backend": backend, "model": model, "size": size,
                        "params": dict(model_sizes[size]),
                        "repeats": repeats_for(size), "warmup": warmup,
                        "validate": small(size),
                    })
        elif backend in ("linopy", "arraynative_highs"):
            for model in [m for m in models if m in COMPARATOR_MODEL_NAMES]:
                model_sizes = _sizes_for(model)
                for size in sizes:
                    if size not in model_sizes:
                        continue
                    specs.append({
                        "backend": backend, "model": model, "size": size,
                        "params": dict(model_sizes[size]),
                        "repeats": repeats, "warmup": warmup,
                    })
        elif backend == "arraynative_gurobi":
            # Size-limited license: xs only.
            for model in [m for m in models if m in COMPARATOR_MODEL_NAMES]:
                model_sizes = _sizes_for(model)
                if "xs" not in model_sizes:
                    continue
                specs.append({
                    "backend": backend, "model": model, "size": "xs",
                    "params": dict(model_sizes["xs"]),
                    "repeats": repeats, "warmup": warmup,
                })
    return specs


def _run_case_subprocess(spec: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
        out_path = tf.name
    spec = dict(spec)
    spec["_out"] = out_path
    cmd = [sys.executable, "-m", "bench.run_bench", "--single", json.dumps(spec)]
    try:
        proc = subprocess.run(
            cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {**_spec_head(spec), "ok": False, "error": f"timeout>{timeout}s"}
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            with open(out_path) as fh:
                res = json.load(fh)
            os.unlink(out_path)
            return res
        except Exception:
            pass
    # Subprocess died before writing (OOM / segfault).
    tail = (proc.stderr or "")[-800:]
    return {
        **_spec_head(spec),
        "ok": False,
        "error": f"subprocess rc={proc.returncode}",
        "stderr_tail": tail,
    }


def _spec_head(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: spec[k] for k in ("backend", "model", "size", "params") if k in spec}


def _fmt_row(res: Dict[str, Any]) -> str:
    tag = f"{res.get('backend',''):>18s} {res.get('model',''):>20s} {res.get('size',''):>6s}"
    if res.get("skipped"):
        return f"{tag}  SKIP ({res['skipped']})"
    if not res.get("ok"):
        return f"{tag}  FAIL ({res.get('error')})"
    stats = res.get("stats", {})
    nnz = stats.get("nnz")
    total = res.get("total_median_ms")
    rss = res.get("peak_rss_mb")
    stage_bits = []
    for s, d in res.get("stages", {}).items():
        if isinstance(d, dict) and "median_ms" in d:
            stage_bits.append(f"{s}={d['median_ms']:.1f}")
        else:
            stage_bits.append(f"{s}=ERR")
    total_s = f"{total:.1f}ms" if total is not None else "n/a"
    return (f"{tag}  vars={stats.get('n_vars','?')} nnz={nnz} "
            f"total={total_s} rss={rss}MB | " + " ".join(stage_bits))


def orchestrate(args) -> int:
    models = args.models.split(",") if args.models and args.models != "all" else (
        PYOMO_MODEL_NAMES if not args.comparators_only else COMPARATOR_MODEL_NAMES
    )
    if args.backends == "all":
        backends = ALL_BACKENDS
    else:
        backends = args.backends.split(",")

    if args.sizes:
        sizes = args.sizes.split(",")
    else:
        sizes = _default_sizes(args.suite)
        # CI also exercises the comparators' xs (gurobi) — planner handles that.
        if args.suite == "ci":
            sizes = ["xs", "1e4"]

    specs = _plan(args.suite, models, backends, sizes, args.repeats, args.warmup)
    print(f"# planned {len(specs)} cases  (suite={args.suite}, backends={backends})")
    print(f"# sizes={sizes} models={models}")

    info = sysinfo.collect()

    # The CI subset runs the standard-form equivalence oracle (Pyomo vs
    # array-native, up to row/col permutation) so a correctness regression fails
    # the CI run, not just a timing drift.  It is cheap and size-independent, so
    # we run it once here rather than per timing case.
    equivalence_report = None
    if args.suite == "ci":
        try:
            from bench import equivalence

            equivalence_report = equivalence.check_all()
            ok = equivalence_report["all_equivalent"]
            print(f"# equivalence oracle: all_equivalent={ok} "
                  f"({len(equivalence_report['results'])} cases)", flush=True)
        except Exception as e:
            equivalence_report = {"error": f"{type(e).__name__}: {e}", "all_equivalent": False}
            print(f"# equivalence oracle FAILED: {equivalence_report['error']}", flush=True)

    def _payload(results):
        p = {
            "sysinfo": info,
            "suite": args.suite,
            "config": {
                "backends": backends, "models": models, "sizes": sizes,
                "repeats": args.repeats, "warmup": args.warmup,
            },
            "results": results,
            "complete": False,
        }
        if equivalence_report is not None:
            p["equivalence"] = equivalence_report
        return p

    def _write(results, complete):
        if not args.out:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        p = _payload(results)
        p["complete"] = complete
        with open(args.out, "w") as fh:
            json.dump(p, fh, indent=2)

    results: List[Dict[str, Any]] = []
    for i, spec in enumerate(specs, 1):
        timeout = args.timeout
        res = _run_case_subprocess(spec, timeout)
        results.append(res)
        print(f"[{i}/{len(specs)}] " + _fmt_row(res), flush=True)
        # Write after every case so a crash/OOM later never discards results.
        _write(results, complete=False)

    _write(results, complete=True)
    if args.out:
        print(f"# wrote {args.out}")
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"# done: {n_ok}/{len(results)} ok")
    # A CI-subset run is also a correctness gate: fail if the equivalence oracle
    # found the array-native and Pyomo standard forms are not the same LP.
    if equivalence_report is not None and not equivalence_report.get("all_equivalent"):
        print("# FAIL: equivalence oracle reported a mismatch")
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="pyomo-build-bench stage-separated harness")
    p.add_argument("--single", type=str, default=None,
                   help="internal worker mode: JSON case spec")
    p.add_argument("--suite", choices=["ci", "full"], default="ci")
    p.add_argument("--models", type=str, default="all",
                   help="comma list or 'all' (default: all pyomo models)")
    p.add_argument("--backends", type=str, default="all",
                   help=f"comma list or 'all' from {ALL_BACKENDS}")
    p.add_argument("--sizes", type=str, default=None,
                   help="comma list of size keys; default depends on --suite")
    p.add_argument("--comparators-only", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="per-case subprocess timeout (s)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    if args.single is not None:
        spec = json.loads(args.single)
        out_path = spec.pop("_out", None)
        res = run_single(spec)
        text = json.dumps(res, indent=2)
        if out_path:
            with open(out_path, "w") as fh:
                fh.write(text)
        else:
            print(text)
        return 0

    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
