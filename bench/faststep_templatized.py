"""Faststep-over-templatized-models benchmark: switch-ON vs switch-OFF.

The evidence for the "faststep over a `vectorized_construction()`-built model"
integration.  A rolling-horizon model is built twice from one identical source --
once **with** the Phase-3 construction switch (templatized constraint families +
columnar Var/Param) and once **without** (byte-classic) -- and both are warm
re-solved through :class:`~pyomo.contrib.vector.faststep.FastStepHighs`.

For each scale it reports:

* **set_instance compile** (median of repeated builds) -- switch-ON vs switch-OFF,
  and the ON/OFF ratio.  Exit criterion: ON is **no slower** than OFF.
* **warm tick** (median per-roll update + re-solve) -- ON vs OFF, and the ratio.
  Exit criterion: **parity or better**.
* **solve-for-solve** max objective difference over the rolling sequence.  Exit
  criterion: the switch-ON warm loop matches the switch-OFF faststep run.

The synthetic model exercises the warm mechanisms through the templatized
representations on the clean templatizable case the switch is designed for:
columnar Vars (``p``, ``soc``), a templatized equality with a mutable RHS
(``dem``), a templatized inequality with a mutable RHS (``gcap``), a **fully
static** templatized family (immutable ``pcap``) that the warm plan skips without
materializing, and a mutable objective coefficient (``price``).

``--mutbound`` switches ``p`` to a ``bounds=`` rule over the mutable ``pmax``.
That is a legitimate warm case, but a columnar Var cannot hold a mutable-Param
bound, so the rule forces ``p`` onto the classic fallback -- whose per-index
bound/body is read in Python during the compile rather than by the C-level
standard-form compiler, making switch-ON ``set_instance`` modestly *slower* than
switch-OFF.  That cost is a property of the columnar-Var bound-representation
limit (a Phase-3/PR-10 constraint), not of the constraint templatization this
benchmark measures; the default (columnar) run is the headline.

Run from the ``bench/`` virtualenv::

    bench/.venv/bin/python -m bench.faststep_templatized
    bench/.venv/bin/python -m bench.faststep_templatized --scales 20x50,40x100,60x150
    bench/.venv/bin/python -m bench.faststep_templatized --mutbound   # mixed case
"""

import argparse
import statistics
import time

import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.vector import vectorized_construction, FastStepHighs


def build(switch, A, T, mutbound=False):
    ctx = vectorized_construction() if switch else None
    if ctx is not None:
        ctx.__enter__()
    try:
        m = pyo.ConcreteModel()
        m.A = pyo.RangeSet(0, A - 1)
        m.T = pyo.RangeSet(0, T - 1)
        m.price = pyo.Param(m.T, initialize={t: 1.0 for t in range(T)}, mutable=True)
        m.dem = pyo.Param(
            m.A,
            m.T,
            initialize={(a, t): 0.5 for a in range(A) for t in range(T)},
            mutable=True,
        )
        m.gcap = pyo.Param(m.T, initialize={t: 3.0 * A for t in range(T)}, mutable=True)
        m.pmax = pyo.Param(
            m.A,
            m.T,
            initialize={(a, t): 5.0 for a in range(A) for t in range(T)},
            mutable=True,
        )
        m.pcap = pyo.Param(
            m.A,
            m.T,
            initialize={(a, t): 8.0 for a in range(A) for t in range(T)},
            mutable=False,
        )
        eff = 0.95
        if mutbound:
            # A bounds= rule over the mutable ``pmax`` keeps ``p`` a *classic* Var
            # (a columnar Var cannot hold a mutable-Param bound).
            m.p = pyo.Var(
                m.A,
                m.T,
                domain=pyo.NonNegativeReals,
                bounds=lambda mm, a, t: (0.0, mm.pmax[a, t]),
            )
        else:
            # Static bounds keep ``p`` columnar (the clean templatizable case).
            m.p = pyo.Var(m.A, m.T, domain=pyo.NonNegativeReals, bounds=(0.0, 10.0))
        m.soc = pyo.Var(m.A, m.T, bounds=(0.0, 40.0))
        # STATIC templatized family (immutable cap) -> skipped by the warm plan.
        m.cap = pyo.Constraint(
            m.A, m.T, rule=lambda mm, a, t: mm.p[a, t] <= mm.pcap[a, t]
        )
        # mutable-RHS templatized equality (dynamics).
        m.bal = pyo.Constraint(
            m.A,
            m.T,
            rule=lambda mm, a, t: mm.soc[a, t] == eff * mm.p[a, t] - mm.dem[a, t],
        )
        # mutable-RHS templatized inequality (grid coupling).
        m.grid = pyo.Constraint(
            m.T, rule=lambda mm, t: sum(mm.p[a, t] for a in mm.A) <= mm.gcap[t]
        )
        m.obj = pyo.Objective(
            expr=sum(m.price[t] * m.p[a, t] for a in range(A) for t in range(T))
            + 0.01 * sum(m.soc[a, t] for a in range(A) for t in range(T)),
            sense=pyo.minimize,
        )
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return m


def roll(m, A, T, rng):
    for t in range(T):
        m.price[t] = float(rng.uniform(0.5, 3.0))
        m.gcap[t] = 3.0 * A * float(rng.uniform(0.7, 1.0))
    for a in range(A):
        for t in range(T):
            m.dem[a, t] = float(rng.uniform(0.0, 1.0))


def median_set_instance(switch, A, T, reps, mutbound):
    ts = []
    for _ in range(reps):
        m = build(switch, A, T, mutbound)
        s = FastStepHighs()
        t0 = time.perf_counter()
        s.set_instance(m)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def warm_ticks(switch, A, T, rolls, seed, mutbound):
    m = build(switch, A, T, mutbound)
    s = FastStepHighs()
    s.set_instance(m)
    rng = np.random.RandomState(seed)
    ts, objs = [], []
    for _ in range(rolls):
        roll(m, A, T, rng)
        t0 = time.perf_counter()
        r = s.solve()
        ts.append(time.perf_counter() - t0)
        objs.append(r.incumbent_objective)
    return statistics.median(ts), objs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scales",
        default="20x50,40x100,60x150",
        help="comma-separated AxT scales (assets x horizon)",
    )
    ap.add_argument("--reps", type=int, default=5, help="set_instance repetitions")
    ap.add_argument("--rolls", type=int, default=15, help="warm rolls per scale")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--mutbound",
        action="store_true",
        help="give p a mutable-Param bound (forces a classic-fallback Var)",
    )
    args = ap.parse_args()

    scales = []
    for tok in args.scales.split(","):
        a, t = tok.lower().split("x")
        scales.append((int(a), int(t)))

    print(
        "| scale (A×T) | vars | cons | set_instance ON | OFF | ON/OFF | warm tick ON | OFF | ON/OFF | max obj Δ |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for A, T in scales:
        nvar = 2 * A * T
        ncon = 2 * A * T + T
        on_si = median_set_instance(True, A, T, args.reps, args.mutbound)
        off_si = median_set_instance(False, A, T, args.reps, args.mutbound)
        on_tick, on_objs = warm_ticks(True, A, T, args.rolls, args.seed, args.mutbound)
        off_tick, off_objs = warm_ticks(
            False, A, T, args.rolls, args.seed, args.mutbound
        )
        maxd = max(abs(a - b) for a, b in zip(on_objs, off_objs))
        print(
            f"| {A}×{T} | {nvar} | {ncon} "
            f"| {on_si*1e3:.1f} ms | {off_si*1e3:.1f} ms | {on_si/off_si:.3f} "
            f"| {on_tick*1e3:.2f} ms | {off_tick*1e3:.2f} ms | {on_tick/off_tick:.3f} "
            f"| {maxd:.1e} |"
        )


if __name__ == "__main__":
    main()
