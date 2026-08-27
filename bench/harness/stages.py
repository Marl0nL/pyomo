"""Stage runners for the Pyomo build pipeline.

The four stages the vectorized-construction project targets (scoping doc §2.1):

  1. construct  - build the ``ConcreteModel`` (one VarData/ConstraintData per
                  index, one operator-overload expression tree per constraint).
  2. repn       - canonicalize to a matrix: ``LinearStandardFormCompiler`` walks
                  every expression tree via the ``pyomo.repn.linear`` visitor and
                  emits scipy CSC/CSR.  This is the exact output target the fast
                  path (scoping doc §6.3) will splice arrays into, so it is the
                  right thing to baseline.
  3. write      - emit a solver file (LP writer v2 by default; NL optional).
  4. load       - hand the model to an in-process solver (APPSI HiGHS
                  ``set_instance``) WITHOUT solving: this is the per-row solver
                  load cost called out in #3888.

Stages 2-4 are pure functions of a fully constructed model, so the runner builds
the model once (timing stage 1) and then times 2-4 repeatedly on it.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

import pyomo.environ as pyo
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint


# --------------------------------------------------------------------------- #
# Model statistics
# --------------------------------------------------------------------------- #
def model_stats(model: pyo.ConcreteModel) -> Dict[str, Any]:
    """Structural size of a model: vars, constraints, matrix nonzeros.

    ``nnz`` is the number of structural nonzeros in the linear constraint matrix,
    taken from the standard-form ``A`` where the model is linear.  For models
    with a quadratic objective (facility-location quadratic variant) the linear
    constraint matrix is still well defined, so ``A.nnz`` remains the right
    "problem size" number and the quadratic terms are counted separately.
    """
    n_vars = 0
    n_binary = 0
    n_int = 0
    for v in model.component_objects(Var, active=True):
        for vd in v.values():
            n_vars += 1
            if vd.is_binary():
                n_binary += 1
            elif vd.is_integer():
                n_int += 1

    n_cons = 0
    for c in model.component_objects(Constraint, active=True):
        n_cons += len(c)

    stats: Dict[str, Any] = {
        "n_vars": n_vars,
        "n_binary": n_binary,
        "n_integer": n_int,
        "n_constraints": n_cons,
    }
    return stats


def constraint_matrix_nnz(model: pyo.ConcreteModel) -> Optional[int]:
    """Nonzeros in the linear constraint matrix, or None if it can't be built."""
    from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

    try:
        res = LinearStandardFormCompiler().write(model, mixed_form=True)
        return int(res.A.nnz)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Stage 2: repn (canonicalization to standard form)
# --------------------------------------------------------------------------- #
def stage_repn(model: pyo.ConcreteModel):
    """Compile the model to linear standard form (scipy sparse A, b, c).

    This is the modern matrix compiler that the fast path must be equivalent to.
    Returns the compiler result (has ``.A``, ``.b``, ``.c``, ``.rows``,
    ``.columns``).
    """
    from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

    return LinearStandardFormCompiler().write(model, mixed_form=True)


def stage_repn_quadratic(model: pyo.ConcreteModel) -> int:
    """Repn cost for models with quadratic terms the standard-form path rejects.

    Walks every constraint and the objective through ``generate_standard_repn``
    (the classic repn that handles quadratic).  Returns the number of expressions
    processed so the call can't be optimized away.
    """
    from pyomo.repn.standard_repn import generate_standard_repn

    n = 0
    for c in model.component_objects(Constraint, active=True):
        for cd in c.values():
            generate_standard_repn(cd.body, quadratic=True)
            n += 1
    for o in model.component_objects(pyo.Objective, active=True):
        for od in o.values():
            generate_standard_repn(od.expr, quadratic=True)
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Stage 3: write (solver file emission)
# --------------------------------------------------------------------------- #
def stage_write_lp(model: pyo.ConcreteModel, path: str) -> int:
    """Write an LP file (LP writer v2).  Returns file size in bytes."""
    model.write(path, io_options={"symbolic_solver_labels": False})
    return os.path.getsize(path)


def stage_write_nl(model: pyo.ConcreteModel, path: str) -> int:
    """Write an NL file (NL writer v2).  Returns file size in bytes."""
    model.write(path, io_options={"symbolic_solver_labels": False})
    return os.path.getsize(path)


# --------------------------------------------------------------------------- #
# Stage 4: load (per-row solver load, no solve)
# --------------------------------------------------------------------------- #
def stage_load_highs(model: pyo.ConcreteModel):
    """Load the model into an in-process HiGHS via APPSI ``set_instance``.

    This is stage 4 of scoping doc §2.1 (per-row solver loading, #3888) measured
    without the solve itself.  Returns the solver interface so the caller can
    read back the loaded dimensions.
    """
    from pyomo.contrib.appsi.solvers import Highs

    h = Highs()
    h.config.load_solution = False
    h.set_instance(model)
    return h


def solve_highs(model: pyo.ConcreteModel):
    """Full solve via APPSI HiGHS, for correctness validation only (untimed)."""
    from pyomo.contrib.appsi.solvers import Highs

    h = Highs()
    res = h.solve(model)
    return res
