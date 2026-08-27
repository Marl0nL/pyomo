"""Timing and memory primitives for the vectorized-construction benchmark.

The whole harness reports a *stage-separated* breakdown of the wall-clock cost
of getting a model from an empty ``ConcreteModel`` to "the solver has it":

    construct -> repn -> write -> load

Each stage is timed with a warmup (to pay one-time import / JIT costs before we
start measuring) followed by ``repeats`` measured runs.  We report ``min`` (the
least-noisy estimate of the underlying cost) and ``median`` (robust central
estimate) in milliseconds; ``mean`` and ``stdev`` are kept for context.

Peak resident memory is captured per *process* via ``getrusage`` at the end of a
run.  Because ``ru_maxrss`` is a process-lifetime high-water mark, the runner
executes each (model, size, backend) case in its own subprocess so the reported
peak RSS is attributable to that single case's pipeline (see ``runner.py``).
"""

from __future__ import annotations

import gc
import resource
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Any, Dict, List


@dataclass
class StageTiming:
    """Timing summary for one stage, all times in milliseconds."""

    name: str
    repeats: int
    warmup: int
    min_ms: float
    median_ms: float
    mean_ms: float
    stdev_ms: float
    samples_ms: List[float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _summarize(name: str, samples: List[float], warmup: int) -> StageTiming:
    return StageTiming(
        name=name,
        repeats=len(samples),
        warmup=warmup,
        min_ms=min(samples),
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        stdev_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        samples_ms=[round(s, 4) for s in samples],
    )


def time_callable(
    name: str,
    fn: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: int = 1,
    setup: Optional[Callable[[], Any]] = None,
    gc_between: bool = True,
) -> StageTiming:
    """Time ``fn`` over ``warmup`` unmeasured + ``repeats`` measured calls.

    ``fn`` takes no arguments and its return value is discarded (hold any state
    it needs in a closure).  If ``setup`` is given it runs before every call
    (measured time excludes it) — used when a stage mutates its input and needs a
    fresh object each iteration (e.g. rebuilding the model for the construct
    stage).  ``perf_counter`` is the clock; GC is disabled *inside* the measured
    region and (optionally) collected between iterations so a collection pause
    never lands in a sample.
    """
    samples: List[float] = []
    total = warmup + repeats
    for i in range(total):
        if setup is not None:
            setup()
        if gc_between:
            gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
        finally:
            if gc_was_enabled:
                gc.enable()
        if i >= warmup:
            samples.append((t1 - t0) * 1000.0)
    return _summarize(name, samples, warmup)


def time_construct(
    name: str,
    build_fn: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: int = 1,
) -> tuple[StageTiming, Any]:
    """Time a fresh build each iteration and return (timing, last_built_object).

    ``build_fn`` must return a *newly constructed* object each call (the model /
    array container).  We keep the final built object so downstream stages can
    run against it without paying construction again.
    """
    holder: Dict[str, Any] = {}

    def _run():
        holder["obj"] = build_fn()

    timing = time_callable(name, _run, repeats=repeats, warmup=warmup)
    # One more build so the returned object corresponds to a clean, post-warmup
    # state (the timed loop's last object is fine, but we rebuild to be explicit
    # and avoid handing back an object that a `gc.collect` may have touched).
    holder["obj"] = build_fn()
    return timing, holder["obj"]


def peak_rss_mb() -> float:
    """Process peak resident set size in MiB (Linux ``ru_maxrss`` is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def current_rss_mb() -> float:
    """Best-effort current RSS in MiB by reading ``/proc/self/statm``."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * resource.getpagesize() / (1024.0 * 1024.0)
    except Exception:
        return float("nan")
