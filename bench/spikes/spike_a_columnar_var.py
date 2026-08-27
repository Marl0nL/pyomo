"""Spike A (throwaway): columnar Var micro-prototype.

Question (scoping doc §6.1, G2, issue #202): how much of construction cost and
memory is the per-index ``VarData`` object itself?  If columnar (NumPy-array)
storage of bounds/value/fixed/domain is dramatically cheaper to allocate and
smaller in memory, that quantifies the ceiling for the columnar-Var work in
Phase 1.

We measure, at several N:

  * classic  - ``m.x = Var(RangeSet(N), bounds=..., domain=NonNegativeReals)``:
               wall time to construct and bytes allocated (``tracemalloc``).
  * columnar - the same data as five NumPy arrays (lb, ub, value, fixed,
               domain-code): wall time and ``nbytes``.
  * access   - random read of ``x[i].value`` (classic dict+attr) vs an array slot,
               to sanity-check that array access is not the bottleneck.

This is a micro-prototype, not a design: it does not build flyweight views or
preserve identity semantics (that is the Phase-1 correctness problem, R1).  It
only bounds the object-creation savings.
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from typing import Any, Dict, List

import numpy as np
import pyomo.environ as pyo


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure_classic(N: int) -> Dict[str, Any]:
    def build():
        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, N - 1)
        m.x = pyo.Var(m.I, bounds=(0.0, 10.0), domain=pyo.NonNegativeReals)
        return m

    wall = _time(build)

    gc.collect()
    tracemalloc.start()
    m = build()
    _cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # bytes attributable to the constructed model (current allocation).
    model_bytes = _cur

    # Access benchmark: sum values over random indices.
    idx = np.random.default_rng(0).integers(0, N, size=min(N, 100_000))
    xs = m.x

    def access():
        s = 0.0
        for i in idx:
            v = xs[int(i)].value
            s += v if v is not None else 0.0
        return s

    acc = _time(access, repeats=3)
    return {
        "N": N,
        "construct_s": wall,
        "bytes": model_bytes,
        "bytes_per_var": model_bytes / N,
        "vars_per_s": N / wall,
        "access_s_per_100k": acc * (100_000 / len(idx)),
    }


def measure_columnar(N: int) -> Dict[str, Any]:
    def build():
        lb = np.zeros(N, dtype=np.float64)
        ub = np.full(N, 10.0, dtype=np.float64)
        val = np.full(N, np.nan, dtype=np.float64)
        fixed = np.zeros(N, dtype=np.bool_)
        domain = np.zeros(N, dtype=np.int8)  # 0 = NonNegativeReals, code
        return lb, ub, val, fixed, domain

    wall = _time(build)
    lb, ub, val, fixed, domain = build()
    total_bytes = lb.nbytes + ub.nbytes + val.nbytes + fixed.nbytes + domain.nbytes

    idx = np.random.default_rng(0).integers(0, N, size=min(N, 100_000))

    def access():
        return float(np.nansum(val[idx]))

    acc = _time(access, repeats=3)
    return {
        "N": N,
        "construct_s": wall,
        "bytes": total_bytes,
        "bytes_per_var": total_bytes / N,
        "vars_per_s": N / wall,
        "access_s_per_100k": acc * (100_000 / len(idx)),
    }


def run(sizes: List[int] = None) -> Dict[str, Any]:
    if sizes is None:
        sizes = [10_000, 100_000, 1_000_000]
    rows = []
    for N in sizes:
        c = measure_classic(N)
        v = measure_columnar(N)
        rows.append({
            "N": N,
            "classic": c,
            "columnar": v,
            "speedup_construct": c["construct_s"] / v["construct_s"],
            "memory_ratio_columnar_over_classic": v["bytes"] / c["bytes"],
        })
    return {"spike": "A_columnar_var", "rows": rows}


def _print(report: Dict[str, Any]) -> None:
    print("=" * 92)
    print("SPIKE A - columnar Var micro-prototype (object-creation & memory savings)")
    print("=" * 92)
    print(f"{'N':>10} | {'classic build':>14} {'columnar build':>15} {'speedup':>8} | "
          f"{'classic B/var':>13} {'colB/var':>9} {'mem ratio':>9}")
    print("-" * 92)
    for r in report["rows"]:
        c, v = r["classic"], r["columnar"]
        print(f"{r['N']:>10} | {c['construct_s']*1000:>12.1f}ms {v['construct_s']*1000:>13.2f}ms "
              f"{r['speedup_construct']:>7.0f}x | {c['bytes_per_var']:>11.0f}B {v['bytes_per_var']:>8.1f}B "
              f"{r['memory_ratio_columnar_over_classic']:>8.3f}")
    print("-" * 92)
    print("access (per 100k random reads):")
    for r in report["rows"]:
        print(f"  N={r['N']:>9}: classic {r['classic']['access_s_per_100k']*1000:.2f}ms  "
              f"columnar {r['columnar']['access_s_per_100k']*1000:.3f}ms")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="10000,100000,1000000")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    rep = run(sizes)
    _print(rep)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"# wrote {a.out}")
