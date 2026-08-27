"""Compile-scaling evidence for ``highs_faststep`` ``set_instance``.

The warm re-solve (``FastStepHighs.solve``) is fast, but the *one-time compile*
(``set_instance``: standard-form compile + fold classification + affine-template
construction + self-check) used to scale super-linearly on models that fold many
verified-static parameters -- the fold classifier's greedy hub loop rebuilt its
whole candidate set and rescanned every coupling on each fold, i.e. ``O(folds x
nnz)``, degrading to ``O(nnz^2)`` when the fold count grew with the model.

This spike reproduces both the growth and the fix on the public synthetic
``rolling_mpc`` model (no external / application-specific structure):

* ``sweep``  -- ``set_instance`` compile time vs interval count (near-linearity);
* ``hubs``   -- compile time vs the number of folded *hub* parameters at a fixed
  nnz (the superlinear stressor: flat now, was ``~H^0.9``).

Run from the repo root with the bench venv::

    bench/.venv/bin/python -m bench.spikes.faststep_compile_scaling sweep
    bench/.venv/bin/python -m bench.spikes.faststep_compile_scaling hubs
"""

from __future__ import annotations

import math
import sys
import time

import pyomo.environ as pyo

from pyomo.contrib.vector.faststep import FastStepHighs
from pyomo.contrib.vector.fastload import compile_to_highs_arrays

from bench.models import rolling_mpc


def _time_compile(model):
    fs = FastStepHighs()
    t0 = time.perf_counter()
    fs.set_instance(model)
    return time.perf_counter() - t0


def sweep(A=8):
    """Compile time vs interval count for the plain and folding variants."""
    Ts = [288, 1000, 2160, 5000]
    for variant, kw in (("plain", {}), ("folding", {"nonaffine_param": True})):
        print(f"\n=== variant={variant}  A={A} ===")
        print(f"{'T':>7} {'nnz':>10} {'compile_s':>11} {'us/nnz':>9} {'exponent':>16}")
        prev = None
        for T in Ts:
            m = rolling_mpc.build_pyomo({"A": A, "T": T}, **kw)
            nnz = compile_to_highs_arrays(m).nnz
            dt = _time_compile(m)
            exp = "-"
            if prev:
                exp = "n^%.2f" % (math.log(dt / prev[0]) / math.log(nnz / prev[1]))
            print(f"{T:>7} {nnz:>10,} {dt:>11.3f} {dt / nnz * 1e6:>9.1f} {exp:>16}")
            prev = (dt, nnz)


def _hub_model(n_hubs, per_hub):
    """A folding model with ``n_hubs`` structural-constant durations ``dt[z]``,
    each multiplying every ``price[z, t]`` in its zone -- ``n_hubs`` hub folds at
    a total obj-term count of ``n_hubs * per_hub`` (held ~fixed as ``n_hubs``
    varies, isolating the fold-count effect from nnz)."""
    m = pyo.ConcreteModel()
    Z = range(n_hubs)
    Tz = range(per_hub)
    idx = [(z, t) for z in Z for t in Tz]
    m.x = pyo.Var(idx, domain=pyo.NonNegativeReals, bounds=(0, 10))
    m.price = pyo.Param(idx, initialize={k: 1.0 for k in idx}, mutable=True)
    m.dt = pyo.Param(list(Z), initialize={z: 0.25 for z in Z}, mutable=True)
    m.cap = pyo.Constraint(
        list(Z), rule=lambda mm, z: sum(mm.x[z, t] for t in Tz) <= 100.0
    )
    m.obj = pyo.Objective(
        expr=sum(m.price[z, t] * m.dt[z] * m.x[z, t] for z in Z for t in Tz),
        sense=pyo.minimize,
    )
    return m


def hubs(total=8000):
    """Compile time vs number of folded hubs, at ~fixed total obj terms."""
    print(f"fixed total obj terms ~= {total}; vary the hub (fold) count")
    print(f"{'hubs':>6} {'per_hub':>8} {'terms':>7} {'compile_s':>11} {'folded':>8} {'exponent':>14}")
    prev = None
    for H in [4, 16, 64, 256, 1024]:
        per = max(2, total // H)
        m = _hub_model(H, per)
        fs = FastStepHighs()
        t0 = time.perf_counter()
        fs.set_instance(m)
        dt = time.perf_counter() - t0
        n_folded = fs.classification_report()["n_folded"]
        exp = "-"
        if prev:
            exp = "H^%.2f" % (math.log(dt / prev[0]) / math.log(H / prev[1]))
        print(f"{H:>6} {per:>8} {H * per:>7} {dt:>11.3f} {n_folded:>8} {exp:>14}")
        prev = (dt, H)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif mode == "hubs":
        hubs()
    else:
        raise SystemExit("mode must be sweep|hubs")
