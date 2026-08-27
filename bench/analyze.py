"""Turn the benchmark result JSONs into markdown tables for the report.

Reads ``full.json`` (the stage-separated sweep) plus the spike JSONs and prints
GitHub-flavoured markdown tables so the report quotes exactly the measured
numbers rather than hand-transcribed ones.

    bench/.venv/bin/python -m bench.analyze --full bench/results/full.json \
        --spike-a bench/results/spike_a.json --spike-b bench/results/spike_b.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

_SIZE_ORDER = ["xs", "1e4", "1e5", "1e6", "1e7"]


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _stage(res: Dict[str, Any], name: str) -> Optional[float]:
    d = res.get("stages", {}).get(name)
    if isinstance(d, dict) and "median_ms" in d:
        return d["median_ms"]
    return None


def _pipeline_ms(res: Dict[str, Any], backend: str) -> Optional[float]:
    """A COHERENT 'empty model -> the solver has it' wall-clock for one backend.

    The per-stage ``Σ stages`` sum is NOT a pipeline: for Pyomo it adds a
    standalone standard-form compile (``repn``, on neither solver route) to the LP
    ``write`` route AND the APPSI ``load`` route (which are *alternatives*).  The
    honest single pipeline is:

      * pyomo:  construct + load_highs      (the in-memory APPSI route)
      * array:  build_matrices + load_*     (== its Σ, already one route)
      * linopy: build + extract             (endpoint: matrix in memory)

    Use this, not ``Σ stages``, for cross-system ratios.
    """
    if backend == "pyomo":
        c, l = _stage(res, "construct"), _stage(res, "load_highs")
        return (c + l) if (c is not None and l is not None) else None
    # comparators already report a single-route total.
    return res.get("total_median_ms")


def _f(x: Optional[float], nd=1) -> str:
    if x is None:
        return "—"
    if x >= 10000:
        return f"{x/1000:.1f}s"
    return f"{x:.{nd}f}"


def _by(
    results: List[Dict[str, Any]], backend: str, model: str
) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in results:
        if r.get("backend") == backend and r.get("model") == model and r.get("ok"):
            out[r.get("size")] = r
    return out


def _sizes_present(d: Dict[str, Dict[str, Any]]) -> List[str]:
    return [s for s in _SIZE_ORDER if s in d]


def pyomo_stage_table(results: List[Dict[str, Any]], model: str) -> str:
    d = _by(results, "pyomo", model)
    if not d:
        return f"_(no pyomo results for {model})_\n"
    lines = [
        f"| size | vars | cons | nnz | construct | repn | write | load | Σ stages† | build→solver‡ | model RSS |",
        f"|------|------|------|-----|-----------|------|-------|------|-----------|---------------|-----------|",
    ]
    for s in _sizes_present(d):
        r = d[s]
        st = r.get("stats", {})
        total = r.get("total_median_ms")
        pipeline = _pipeline_ms(r, "pyomo")
        lines.append(
            f"| {s} | {st.get('n_vars','?'):,} | {st.get('n_constraints','?'):,} | "
            f"{(st.get('nnz') if st.get('nnz') is not None else '—')} | "
            f"{_f(_stage(r,'construct'))} | {_f(_stage(r,'repn'))} | "
            f"{_f(_stage(r,'write_lp'))} | {_f(_stage(r,'load_highs'))} | "
            f"{_f(total)} | **{_f(pipeline)}** | {r.get('model_rss_mb','?')} MB |"
        )
    return (
        "\n".join(lines) + "\n\n"
        "_All times median-of-repeats, milliseconds unless suffixed `s`._\n"
        "_†`Σ stages` = construct+repn+write+load; a **sum of independently-measured, "
        "non-sequential passes**, NOT one pipeline (repn is a standalone compile; write "
        "and load are alternative routes to a solver)._\n"
        "_‡`build→solver` = construct+load: the coherent in-memory (APPSI) route to 'the "
        "solver has it' — the number to compare across systems._\n"
    )


def comparator_table(results: List[Dict[str, Any]], model: str) -> str:
    py = _by(results, "pyomo", model)
    ln = _by(results, "linopy", model)
    an = _by(results, "arraynative_highs", model)
    gb = _by(results, "arraynative_gurobi", model)
    sizes = _sizes_present({**py, **ln, **an, **gb})
    lines = [
        f"| size | nnz | Pyomo build→solver (construct+load) | array→HiGHS (build+load) | linopy (build+extract) | Gurobi (xs) | Pyomo Σ stages† | **Pyomo ÷ array-native‡** |",
        f"|------|-----|-----------------------------------|--------------------------|------------------------|-------------|-----------------|---------------------------|",
    ]
    for s in sizes:
        nnz = None
        for src in (py, an, ln):
            if s in src:
                nnz = src[s].get("stats", {}).get("nnz")
                if nnz:
                    break
        py_pipe = _pipeline_ms(py[s], "pyomo") if s in py else None
        py_sigma = py.get(s, {}).get("total_median_ms")
        ln_t = ln.get(s, {}).get("total_median_ms")
        an_t = an.get(s, {}).get("total_median_ms")
        gb_t = gb.get(s, {}).get("total_median_ms")
        ratio = f"**{py_pipe/an_t:.0f}×**" if (py_pipe and an_t) else "—"
        lines.append(
            f"| {s} | {nnz if nnz else '—'} | {_f(py_pipe)} | {_f(an_t)} | {_f(ln_t)} | "
            f"{_f(gb_t) if gb_t else '—'} | {_f(py_sigma)} | {ratio} |"
        )
    return (
        "\n".join(lines) + "\n"
        "\n_‡Ratio is the coherent build→solver route on both sides (Pyomo construct+load "
        "vs array build+load), NOT Pyomo's `Σ stages` (which double-counts by summing a "
        "standalone repn compile plus the two alternative solver routes)._\n"
        "_†`Σ stages` shown only for reference; see the per-model table's footnotes._\n"
        "_linopy's endpoint is 'constraint matrix in memory', a step short of 'loaded in "
        "the solver', so it is not strictly comparable to the build→solver columns._\n"
    )


def fast_path_table(results: List[Dict[str, Any]], model: str) -> str:
    """Classic vs pyomo.contrib.vector fast path vs array-native ceiling.

    The end-to-end metric is empty model -> "the solver has it".  For every
    backend that is a coherent pipeline (no summing of independent
    non-sequential passes) that is:

      * classic  : construct + load  (APPSI HiGHS set_instance)
      * fast     : construct + repn(assemble) + load(passModel)
      * array    : build_matrices + load(passModel)   -- the in-harness ceiling
    """
    py = _by(results, "pyomo", model)
    vec = _by(results, "pyomo_vector", model)
    an = _by(results, "arraynative_highs", model)
    sizes = _sizes_present({**py, **vec, **an})
    if not vec:
        return f"_(no pyomo_vector results for {model})_\n"

    def coherent(r, stages):
        vals = [_stage(r, s) for s in stages]
        if any(v is None for v in vals):
            return None
        return sum(vals)

    lines = [
        "| size | nnz | classic (construct+load) | **fast (construct+repn+load)** | "
        "array→HiGHS (build+load) | **fast speedup vs classic** | fast / array-native |",
        "|------|-----|--------------------------|--------------------------------|"
        "--------------------------|-----------------------------|---------------------|",
    ]
    for s in sizes:
        nnz = None
        for src in (vec, py, an):
            if s in src:
                nnz = src[s].get("stats", {}).get("nnz")
                if nnz:
                    break
        py_coh = coherent(py[s], ["construct", "load_highs"]) if s in py else None
        vec_tot = vec[s].get("total_median_ms") if s in vec else None
        an_tot = an[s].get("total_median_ms") if s in an else None
        speed = f"**{py_coh/vec_tot:.1f}×**" if (py_coh and vec_tot) else "—"
        ratio = f"{vec_tot/an_tot:.2f}×" if (vec_tot and an_tot) else "—"
        lines.append(
            f"| {s} | {nnz if nnz else '—'} | {_f(py_coh)} | **{_f(vec_tot)}** | "
            f"{_f(an_tot)} | {speed} | {ratio} |"
        )
    return "\n".join(lines) + (
        "\n\n_Classic total is the *coherent* construct+load pipeline (the honest "
        "end-to-end baseline); classic repn/write are separate passes and are not "
        "summed here._\n"
    )


def fast_path_stage_table(results: List[Dict[str, Any]], model: str) -> str:
    """Per-stage breakdown of the fast path."""
    vec = _by(results, "pyomo_vector", model)
    if not vec:
        return f"_(no pyomo_vector results for {model})_\n"
    lines = [
        "| size | vars | cons | nnz | construct | repn (assemble) | load (passModel) | **total** | model RSS |",
        "|------|------|------|-----|-----------|-----------------|------------------|-----------|-----------|",
    ]
    for s in _sizes_present(vec):
        r = vec[s]
        st = r.get("stats", {})
        lines.append(
            f"| {s} | {st.get('n_vars','?'):,} | {st.get('n_constraints','?'):,} | "
            f"{st.get('nnz','—')} | {_f(_stage(r,'construct'))} | {_f(_stage(r,'repn'))} | "
            f"{_f(_stage(r,'load_highs'))} | **{_f(r.get('total_median_ms'))}** | "
            f"{r.get('model_rss_mb','?')} MB |"
        )
    return "\n".join(lines) + "\n"


def fastload_table(results: List[Dict[str, Any]], model: str) -> str:
    """Phase-2 transparent fast solver hand-off vs the classic coherent route.

    Both routes construct the *same* unmodified classic model; the classic route
    then does the per-row APPSI ``set_instance`` load, the fast route the
    standard-form compile + ``passModel`` bulk hand-off (``highs_fastload``).
    ``build→solver`` is construct + the route's hand-off ("the solver has the
    model"); the two speedups isolate the hand-off and the shared-construct
    end-to-end effect.
    """
    d = _by(results, "pyomo", model)
    if not d:
        return f"_(no pyomo results for {model})_\n"
    lines = [
        "| size | nnz | construct | load (classic) | fastload (fast) | classic build→solver | fast build→solver | hand-off × | **end-to-end ×** |",
        "|------|-----|-----------|----------------|-----------------|----------------------|-------------------|------------|------------------|",
    ]
    any_row = False
    for s in _sizes_present(d):
        r = d[s]
        load = _stage(r, "load_highs")
        fast = _stage(r, "fastload_highs")
        if fast is None:
            continue
        any_row = True
        st = r.get("stats", {})
        classic_b2s = r.get("classic_build_to_solver_ms")
        fast_b2s = r.get("fast_build_to_solver_ms")
        handoff = f"{load/fast:.1f}×" if (load and fast) else "—"
        e2e = r.get("fastload_speedup")
        e2e_s = f"**{e2e:.2f}×**" if e2e else "—"
        lines.append(
            f"| {s} | {(st.get('nnz') if st.get('nnz') is not None else '—')} | "
            f"{_f(_stage(r,'construct'))} | {_f(load)} | {_f(fast)} | "
            f"{_f(classic_b2s)} | {_f(fast_b2s)} | {handoff} | {e2e_s} |"
        )
    if not any_row:
        return f"_(no fastload_highs stage for {model})_\n"
    return (
        "\n".join(lines) + "\n\n"
        "_`load (classic)` = APPSI HiGHS `set_instance`; `fastload (fast)` = "
        "`LinearStandardFormCompiler` + `Highs.passModel`.  `build→solver` = "
        "construct + the route's hand-off.  `hand-off ×` isolates the stage the "
        "fast route replaces; `end-to-end ×` includes the shared construct._\n"
    )


def fastload_equivalence_table(results: List[Dict[str, Any]], models: List[str]) -> str:
    """Solve-equivalence of the fast route vs the classic (APPSI) route."""
    lines = [
        "| model | size | classic objective | fast objective | obj match | termination match |",
        "|-------|------|-------------------|----------------|-----------|-------------------|",
    ]
    for model in models:
        d = _by(results, "pyomo", model)
        for s in _sizes_present(d):
            v = d[s].get("validation") or {}
            if "objective_match" not in v and "termination_match" not in v:
                continue
            oc = v.get("objective")
            of = v.get("fastload_objective")
            oc_s = f"{oc:.6g}" if oc is not None else "—"
            of_s = f"{of:.6g}" if of is not None else "—"
            lines.append(
                f"| {model} | {s} | {oc_s} | {of_s} | "
                f"{'✅' if v.get('objective_match') else '—'} | "
                f"{'✅' if v.get('termination_match') else '❌'} |"
            )
    return "\n".join(lines) + "\n"


def stage_share(results: List[Dict[str, Any]]) -> str:
    """Which stage dominates, per model, at the largest available size."""
    lines = [
        "| model | size | construct % | repn % | write % | load % (of Σ) | **load % (of build→solver)** |",
        "|-------|------|-------------|--------|---------|---------------|------------------------------|",
    ]
    models = []
    seen = set()
    for r in results:
        if r.get("backend") == "pyomo" and r.get("model") not in seen:
            seen.add(r.get("model"))
            models.append(r.get("model"))
    for model in models:
        d = _by(results, "pyomo", model)
        sizes = _sizes_present(d)
        if not sizes:
            continue
        s = sizes[-1]
        r = d[s]
        parts = {
            k: _stage(r, v)
            for k, v in [
                ("construct", "construct"),
                ("repn", "repn"),
                ("write", "write_lp"),
                ("load", "load_highs"),
            ]
        }
        total = sum(v for v in parts.values() if v) or 1.0
        pipe = _pipeline_ms(r, "pyomo")
        load_pipe = (
            f"{parts['load']/pipe*100:.0f}%" if (parts["load"] and pipe) else "—"
        )
        lines.append(
            f"| {model} | {s} | {parts['construct']/total*100:.0f}% | "
            f"{parts['repn']/total*100:.0f}% | {parts['write']/total*100:.0f}% | "
            f"{(parts['load']/total*100 if parts['load'] else 0):.0f}% | {load_pipe} |"
        )
    return (
        "\n".join(lines) + "\n"
        "\n_`% of Σ` divides by construct+repn+write+load (the non-pipeline sum), which "
        "**understates** load's true share; `% of build→solver` divides by the coherent "
        "construct+load route and is the honest dominance figure._\n"
    )


def spike_a_table(rep: Dict[str, Any]) -> str:
    lines = [
        "| N (vars) | classic build | columnar build | build speedup | classic B/var | columnar B/var | memory ratio |",
        "|----------|---------------|----------------|---------------|---------------|----------------|--------------|",
    ]
    for r in rep["rows"]:
        c, v = r["classic"], r["columnar"]
        lines.append(
            f"| {r['N']:,} | {c['construct_s']*1000:.1f} ms | {v['construct_s']*1000:.2f} ms | "
            f"{r['speedup_construct']:.0f}× | {c['bytes_per_var']:.0f} B | {v['bytes_per_var']:.0f} B | "
            f"{r['memory_ratio_columnar_over_classic']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def spike_b_tables(rep: Dict[str, Any]) -> str:
    out = [
        "**Coverage — which rule shapes templatize:**\n",
        "| rule shape | example | templatizes? | note |",
        "|------------|---------|--------------|------|",
    ]
    for c in rep["coverage"]:
        mark = "✅ yes" if c["templatizes"] else "❌ no"
        note = "" if c["templatizes"] else c.get("error", "")[:60]
        out.append(f"| {c['shape']} | `{c['rule']}` | {mark} | {note} |")
    out.append(
        "\n**Scalar-affine family (`2·x[i] − x[i−1] ≤ 1`): correctness + speed:**\n"
    )
    out.append(
        "| N (constraints) | vectorized == classic? | classic repn | resolve/idx | vectorized | **vec speedup** | resolve speedup |"
    )
    out.append(
        "|-----------------|------------------------|--------------|-------------|------------|-----------------|-----------------|"
    )
    for r in rep["affine"]["rows"]:
        out.append(
            f"| {r['N']:,} | {'✅' if r['correct'] else '❌'} | {r['classic_s']*1000:.0f} ms | "
            f"{r['resolve_s']*1000:.0f} ms | {r['vectorized_s']*1000:.1f} ms | "
            f"**{r['speedup_vec_over_classic']:.0f}×** | {r['speedup_resolve_over_classic']:.2f}× |"
        )
    out.append("\n**Sum-over-set family (unfiltered flow balance):**\n")
    out.append(
        "| N | T | rows | templatizes? | classic repn | resolve/idx | resolve speedup |"
    )
    out.append(
        "|---|---|------|--------------|--------------|-------------|-----------------|"
    )
    for r in rep["sum_over_set"]["rows"]:
        rs = f"{r['resolve_s']*1000:.0f} ms" if r["resolve_s"] else "—"
        sp = (
            f"{r['speedup_resolve_over_classic']:.2f}×"
            if r["speedup_resolve_over_classic"]
            else "—"
        )
        out.append(
            f"| {r['N']} | {r['T']} | {r['n_constraints']:,} | {'✅' if r['templatizes'] else '❌'} | "
            f"{r['classic_s']*1000:.0f} ms | {rs} | {sp} |"
        )
    return "\n".join(out) + "\n"


def equivalence_table(rep: Dict[str, Any]) -> str:
    """Render the standard-form equivalence oracle's verdict."""
    res = rep.get("results", [])
    lines = [
        "| model | size | vars | nnz | rows == (up to perm) | obj == | **equivalent** |",
        "|-------|------|------|-----|----------------------|--------|----------------|",
    ]
    for r in res:
        rows_ok = r["checks"]["constraint_rows_equal_up_to_permutation"]
        obj_ok = r.get("obj_equal")
        obj_s = "✅" if obj_ok else ("—" if obj_ok is None else "❌")
        lines.append(
            f"| {r['model']} | {r['size']} | {r['n_vars']:,} | {r['nnz_pyomo']:,} | "
            f"{'✅' if rows_ok else '❌'} | {obj_s} | "
            f"{'✅ yes' if r['equivalent'] else '❌ NO'} |"
        )
    note = (
        "\n_Authoritative check: the constraint system as a multiset of sign-normalized "
        "rows, columns keyed by variable identity — equal iff the two LPs are the same "
        "standard form up to row/column permutation. `obj ==` solves both through HiGHS._\n"
    )
    return "\n".join(lines) + "\n" + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="bench/results/full.json")
    ap.add_argument("--spike-a", default="bench/results/spike_a.json")
    ap.add_argument("--spike-b", default="bench/results/spike_b.json")
    ap.add_argument("--equivalence", default="bench/results/equivalence.json")
    ap.add_argument(
        "--phase1",
        default="bench/results/phase1_network_flow.json",
        help="fast-path (pyomo_vector) sweep with classic + array-native",
    )
    ap.add_argument(
        "--phase2",
        default="bench/results/phase2_fastload.json",
        help="transparent fast solver hand-off (highs_fastload) sweep",
    )
    a = ap.parse_args()

    p1 = _load(a.phase1)
    if p1:
        res = p1["results"]
        print("## Phase 1 — vectorized fast path (network_flow)\n")
        print("### End-to-end: classic vs fast path vs array-native ceiling\n")
        print(fast_path_table(res, "network_flow"))
        print("\n### Fast-path stage breakdown\n")
        print(fast_path_stage_table(res, "network_flow"))
        print()

    p2 = _load(a.phase2)
    if p2:
        res = p2["results"]
        p2_models = [
            "network_flow",
            "unit_commitment",
            "facility_location",
            "supply_chain",
        ]
        print("## Phase 2 — transparent fast solver hand-off (highs_fastload)\n")
        print("### End-to-end: classic coherent route vs fast route\n")
        for model in p2_models:
            print(f"\n**{model}**\n")
            print(fastload_table(res, model))
        print("\n### Solve-result equivalence (fast route == classic route)\n")
        print(fastload_equivalence_table(res, p2_models))
        print()

    full = _load(a.full)
    if full:
        results = full["results"]
        info = full["sysinfo"]
        print("## Environment\n")
        print(
            f"- Pyomo `{info['packages'].get('Pyomo')}` @ commit `{(info.get('pyomo_commit') or '')[:12]}`"
        )
        print(
            f"- Python {info['python']}, {info['platform']}, {info['cpu_count']} CPUs"
        )
        pk = info["packages"]
        print(
            f"- numpy {pk.get('numpy')}, scipy {pk.get('scipy')}, highspy {pk.get('highspy')}, "
            f"gurobipy {pk.get('gurobipy')}, linopy {pk.get('linopy')}"
        )
        print(f"- sweep complete: {full.get('complete')}\n")

        print("## Stage-dominance (largest size per model)\n")
        print(stage_share(results))

        for model in [
            "network_flow",
            "unit_commitment",
            "facility_location",
            "facility_location_q",
            "supply_chain",
        ]:
            print(f"\n### {model}\n")
            print(pyomo_stage_table(results, model))

        for model in ["network_flow", "facility_location"]:
            print(f"\n## Comparators — {model}\n")
            print(comparator_table(results, model))

    eq = _load(a.equivalence)
    if eq is None and full and isinstance(full.get("equivalence"), dict):
        eq = full["equivalence"]
    if eq and eq.get("results"):
        print("\n## Equivalence oracle — array-native == Pyomo standard form\n")
        print(equivalence_table(eq))

    sa = _load(a.spike_a)
    if sa:
        print("\n## Spike A — columnar Var\n")
        print(spike_a_table(sa))
    sb = _load(a.spike_b)
    if sb:
        print("\n## Spike B — template vectorization\n")
        print(spike_b_tables(sb))


if __name__ == "__main__":
    main()
