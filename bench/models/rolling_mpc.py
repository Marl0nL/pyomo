"""Rolling-horizon MPC: a synthetic warm-re-solve benchmark model.

A multi-asset energy / storage model-predictive-control problem -- the canonical
*warm rolling* workload: ``construct once, then re-solve thousands of times with
slightly changed data`` as the horizon rolls forward.  Between rolls only **data**
moves; the constraint matrix is static:

  * objective coefficient  <- ``price[t]``          (electricity price, mutable)
  * row RHS                <- ``dem[a,t]``           (per-asset demand, mutable)
  * row RHS                <- ``gcap[t]``            (grid import cap, mutable)
  * variable bound         <- ``pmax[a,t]``          (per-asset power cap, mutable)

``A`` assets over a horizon of ``T`` intervals.  Per asset a state-of-charge
recurrence ``soc[a,t] == soc[a,t-1] + eff*p[a,t] - dem[a,t]`` (static ``eff``)
couples the intervals; a per-interval grid constraint ``sum_a p[a,t] <= gcap[t]``
couples the assets.  Matrix nonzeros ~= ``4*A*T``.

This is a generic MPC/receding-horizon shape (no application-specific IP); it is
the public stand-in for the load-bound rolling workloads that motivated the
``highs_faststep`` warm interface.
"""

from __future__ import annotations

from typing import Any, Dict

import pyomo.environ as pyo

NAME = "rolling_mpc"
DESCRIPTION = "Multi-asset energy MPC (rolling-horizon warm re-solve)."
HAS_QUADRATIC = False

EFF = 0.95
SOC_MAX = 60.0

# nnz ~= 4*A*T.  Sizes named by target matrix nonzero count.
SIZES: Dict[str, Dict[str, Any]] = {
    "xs": {"A": 4, "T": 20},  # ~320 nnz
    "1e4": {"A": 12, "T": 220},  # ~1.1e4 nnz
    "1e5": {"A": 40, "T": 640},  # ~1.0e5 nnz
    "1e6": {"A": 80, "T": 3200},  # ~1.0e6 nnz
}


DT = 0.25  # interval duration (hours): a practically-constant, mutable Param


def build_pyomo(
    params: Dict[str, Any], mutable_matrix: bool = False, nonaffine_param: bool = False
) -> pyo.ConcreteModel:
    """Build the rolling-horizon MPC model.

    ``mutable_matrix=True`` puts the charge efficiency into a *mutable* ``Param``
    ``eff[a,t]`` on the state-of-charge recurrence -- a **nominally-mutable
    constraint-matrix coefficient**.  Under :func:`apply_roll` (an equal-interval
    roll) that Param never changes, so the matrix is static *in value* while the
    mutability *flag* says otherwise -- exactly the case the ``highs_faststep``
    value guard exists to accept.  ``False`` (the default) bakes the efficiency in
    as a constant, giving the pure-static-matrix model.

    ``nonaffine_param=True`` additionally introduces a practically-constant but
    *mutable* interval duration ``dt`` that participates **non-affinely**: the
    objective cost is ``price[t] * dt`` (a product of two mutable Params) and a
    self-discharge leak is ``(dt / eff[a,t]) * soc`` (a reciprocal in ``eff``).
    Neither coefficient is affine in the parameter vector, so the pre-folding
    interface rejected the model.  Verified-static folding folds ``dt`` (the hub
    coupling every ``price[t]``) and ``eff`` (forced by the reciprocal) as watched
    constants and keeps ``price`` / ``dem`` / ``gcap`` / ``pmax`` templated, so the
    model engages the warm path.  The charge term stays ``eff * p`` (with ``eff``
    folded to its constant value), so the feasible region matches the base model.
    Implies ``mutable_matrix`` (``eff`` is mutable).
    """
    A = int(params["A"])
    T = int(params["T"])
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
    m.gcap = pyo.Param(m.T, initialize={t: 4.0 * A for t in range(T)}, mutable=True)
    m.pmax = pyo.Param(
        m.A,
        m.T,
        initialize={(a, t): 5.0 for a in range(A) for t in range(T)},
        mutable=True,
    )

    if mutable_matrix or nonaffine_param:
        m.eff = pyo.Param(
            m.A,
            m.T,
            initialize={(a, t): EFF for a in range(A) for t in range(T)},
            mutable=True,
        )

        def _eff(mm, a, t):
            return mm.eff[a, t]

    else:

        def _eff(mm, a, t):
            return EFF

    if nonaffine_param:
        m.dt = pyo.Param(initialize=DT, mutable=True)  # folded (verified-static)

        def _dt(mm):
            return mm.dt

    else:

        def _dt(mm):
            return 1.0

    m.p = pyo.Var(m.A, m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.A, m.T, bounds=(0.0, SOC_MAX))

    def socrule(mm, a, t):
        # Charge stays eff*p (eff folds to its constant value), so the feasible
        # region matches the base model regardless of dt.
        charge = _eff(mm, a, t) * mm.p[a, t]
        prev = 0.0 if t == 0 else mm.soc[a, t - 1]
        if nonaffine_param:
            # A small self-discharge leak with a reciprocal (dt / eff) coefficient
            # on soc -- forces eff to fold (non-affine in itself).
            leak = (_dt(mm) / mm.eff[a, t]) * mm.soc[a, t] * 0.1
            return mm.soc[a, t] + leak == prev + charge - mm.dem[a, t]
        return mm.soc[a, t] == prev + charge - mm.dem[a, t]

    m.socc = pyo.Constraint(m.A, m.T, rule=socrule)
    m.grid = pyo.Constraint(
        m.T, rule=lambda mm, t: sum(mm.p[a, t] for a in mm.A) <= mm.gcap[t]
    )
    for a in range(A):
        for t in range(T):
            m.p[a, t].setub(m.pmax[a, t])

    m.obj = pyo.Objective(
        expr=sum(m.price[t] * _dt(m) * m.p[a, t] for a in range(A) for t in range(T))
        + 0.01 * sum(m.soc[a, t] for a in range(A) for t in range(T)),
        sense=pyo.minimize,
    )
    return m


def apply_roll(m, rng) -> None:
    """Mutate the model's rolling data in place (one horizon roll).

    Uses a NumPy Generator ``rng`` so a caller can reproduce the same roll for
    two model instances (faststep vs a fresh reference build)."""
    A = len(m.A)
    T = len(m.T)
    price = rng.uniform(0.5, 3.0, size=T)
    gcap = 4.0 * A * rng.uniform(0.7, 1.0, size=T)
    dem = rng.uniform(0.0, 1.0, size=(A, T))
    pmax = rng.uniform(3.0, 6.0, size=(A, T))
    for t in range(T):
        m.price[t] = float(price[t])
        m.gcap[t] = float(gcap[t])
    for a in range(A):
        for t in range(T):
            m.dem[a, t] = float(dem[a, t])
            m.pmax[a, t] = float(pmax[a, t])
