"""linopy comparators (xarray-backed array-native modelling).

linopy is the closest philosophical cousin to what the vectorization project
wants for Pyomo: dense-labelled array variables and constraints built with numpy
speed.  We implement the two cleanly-rectangular synthetics (network flow and
facility location) so we can compare Pyomo's construct+repn to linopy's
build+matrix-extraction at every size.

Metrics reported by the runner:
  * ``build``   - construct the linopy Model (add_variables/constraints/objective).
  * ``extract`` - materialize the constraint matrix (``m.matrices.A``), linopy's
                  canonicalization-to-matrix step (its repn analogue).

The ragged supply-chain and integer-heavy unit-commitment models are *not*
implemented here: linopy's dense-label design would force allocating the full
supplier x warehouse / warehouse x retailer grids and masking, which is exactly
the sparsity weakness the scoping doc (§5, R5) says Pyomo should differentiate
on.  That omission is itself a Phase-0 finding, recorded in the report.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

SUPPORTED = {"network_flow", "facility_location"}


def build_network_flow(params: Dict[str, Any]):
    import linopy
    import xarray as xr

    N = int(params["N"])
    T = int(params["T"])
    arcs = [(i, j) for i in range(N) for j in range(N) if i != j]
    A = len(arcs)
    arc_cap = 5.0 * N
    stor_cap = 10.0 * N

    m = linopy.Model()
    flow = m.add_variables(
        lower=0, upper=arc_cap, coords=[range(A), range(T)], dims=["arc", "t"], name="flow"
    )
    stor = m.add_variables(
        lower=0, upper=stor_cap, coords=[range(N), range(T)], dims=["node", "t"], name="stor"
    )

    # Dense node-arc incidence (network flow is a complete graph, so dense is
    # the honest representation here).
    Bnp = np.zeros((N, A))
    for k, (i, j) in enumerate(arcs):
        Bnp[j, k] += 1.0
        Bnp[i, k] -= 1.0
    B = xr.DataArray(
        Bnp, dims=["node", "arc"], coords={"node": list(range(N)), "arc": list(range(A))}
    )

    netflow = (B * flow).sum("arc")           # inflow - outflow, dims (node, t)
    prev = stor.shift(t=1)                      # stor[n, t-1], dropped at t=0
    lhs = netflow + prev - stor

    # RHS: reuse the Pyomo model's own demand function so this linopy model is
    # the *same* LP as the Pyomo/array-native builders (node 0 is the balancing
    # source: -sum of the sink demands, not a constant).  linopy is a timing-only
    # comparator here and the RHS does not affect its build/extract timing or the
    # constraint structure, but keeping it LP-faithful avoids a latent trap if the
    # linopy model is ever reused for correctness rather than timing.
    from bench.models.network_flow import _demand

    dem = np.array([[_demand(n, t, N) for t in range(T)] for n in range(N)])
    demda = xr.DataArray(
        dem, dims=["node", "t"], coords={"node": list(range(N)), "t": list(range(T))}
    )
    m.add_constraints(lhs == demda, name="balance")

    arc_cost = np.array([1.0 + ((i * 7 + j * 13) % 5) for (i, j) in arcs])
    cost_da = xr.DataArray(arc_cost, dims=["arc"], coords={"arc": list(range(A))})
    m.add_objective((cost_da * flow).sum() + 0.1 * stor.sum())
    return m


def build_facility_location(params: Dict[str, Any]):
    import linopy
    import xarray as xr

    F = int(params["F"])
    C = int(params["C"])
    cap = np.array([3.0 + (f % 5) for f in range(F)], dtype=float)
    cap = cap * ((1.3 * C) / cap.sum() if cap.sum() else 1.0)

    m = linopy.Model()
    opn = m.add_variables(binary=True, coords=[range(F)], dims=["f"], name="open")
    x = m.add_variables(
        lower=0, upper=1, coords=[range(F), range(C)], dims=["f", "c"], name="x"
    )
    m.add_constraints(x.sum("f") == 1, name="serve")
    m.add_constraints(x - opn <= 0, name="link")
    capda = xr.DataArray(cap, dims=["f"], coords={"f": list(range(F))})
    m.add_constraints(x.sum("c") - capda * opn <= 0, name="capacity")

    open_cost = xr.DataArray(
        np.array([50.0 + 5.0 * (f % 6) for f in range(F)]),
        dims=["f"],
        coords={"f": list(range(F))},
    )
    assign = xr.DataArray(
        np.array([[1.0 + ((f * 3 + c * 7) % 11) for c in range(C)] for f in range(F)]),
        dims=["f", "c"],
        coords={"f": list(range(F)), "c": list(range(C))},
    )
    m.add_objective((open_cost * opn).sum() + (assign * x).sum())
    return m


BUILDERS = {
    "network_flow": build_network_flow,
    "facility_location": build_facility_location,
}


def extract(model) -> Any:
    """Materialize the constraint matrix (linopy's canonicalization step)."""
    return model.matrices.A


def stats(model) -> Dict[str, Any]:
    A = model.matrices.A
    return {
        "n_vars": int(model.nvars),
        "n_constraints": int(model.ncons),
        "nnz": int(A.nnz),
    }
