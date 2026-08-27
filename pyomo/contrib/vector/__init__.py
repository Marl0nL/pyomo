# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""``pyomo.contrib.vector`` -- vectorized (columnar) model construction fast path.

The "Vectorized Model Construction for Pyomo" project (scoping doc + Phase-0
baseline report).  Provides:

Phase 1 -- explicit columnar components:

* :class:`VectorVar` -- columnar, array-backed indexed variable
  (materialize-on-touch).
* :class:`VectorConstraint` -- an explicit-array linear constraint family
  (``VectorConstraint(A=csr, x=..., lb=..., ub=...)``).
* :class:`VectorObjective` -- a linear (or convex-quadratic) objective stored as
  coefficient arrays plus an optional sparse Hessian (``c @ x + 0.5 * x @ Q @ x``,
  the #1761 use case).
* :func:`compile_standard_form` -- splice the vector components into a standard
  form comparable to :class:`~pyomo.repn.plugins.standard_form.LinearStandardFormCompiler`.
* :func:`load_highs` -- direct ``Highs.passModel`` array hand-off (the load
  prize); passes the objective Hessian for a convex-QP model.

Phase 2 -- transparent fast solver hand-off for unmodified classic models:

* :class:`FastLoadHighs` (``SolverFactory('highs_fastload')``) -- standard-form
  compile + ``passModel`` bulk load for a classic linear model.

Phase 3 -- template-vectorized construction ("your old code gets fast"):

* :func:`vectorized_construction` -- opt-in context manager (or the
  ``PYOMO_VECTOR_CONSTRUCT`` env var) that makes a classic
  ``Constraint(index, rule=...)`` construct array-shaped when its rule
  templatizes, and fall back to classic construction when it does not.
* :func:`compile_templated_to_highs_arrays` -- vectorized whole-model compile
  that feeds ``highs_fastload`` with no scalarization.

Phase 4 -- array-native persistent **warm re-solve** (the rolling-horizon path):

* :class:`FastStepHighs` -- a persistent HiGHS interface for MPC / rolling-horizon
  workloads: compile once + ``passModel``, retain the live solver, and re-solve
  each roll by pushing the changed objective coefficients / RHS / bounds as
  vectorized arrays (affine templates ``M @ P``) through HiGHS's batch APIs,
  keeping the warm basis.  Both a model-driven (mutate Params, ``solve``) and an
  array / mapping-free (``solve(param_values=...)``) update path.

The Phase-1 components fall back to classic per-index data objects
(*scalarization*) when touched by a consumer that does not understand them, per
the compatibility contract (scoping doc §6.5).
"""

from pyomo.contrib.vector.var import VectorVar, VectorVarData
from pyomo.contrib.vector.constraint import VectorConstraint, VectorConstraintData
from pyomo.contrib.vector.objective import VectorObjective
from pyomo.contrib.vector.matrices import (
    assemble,
    compile_standard_form,
    VectorMatrices,
    VectorStandardFormInfo,
    VectorPathDisabledError,
)
from pyomo.contrib.vector.highs import (
    load_highs,
    solve_highs,
    matrices_to_highs_lp,
    matrices_to_highs_model,
    QuadraticModelError,
)

# Importing fastload registers the ``highs_fastload`` solver with both the v2
# and legacy SolverFactory (the transparent classic-model fast hand-off).
from pyomo.contrib.vector.fastload import (
    FastLoadHighs,
    compile_to_highs_arrays,
    build_highs_lp,
    build_highs_model,
)

# Phase 4 -- array-native persistent warm re-solve for the rolling-horizon path.
from pyomo.contrib.vector.faststep import FastStepHighs

# Phase-3 template-vectorized construction ("your old code gets fast"): the
# opt-in switch + the vectorized whole-model compile that feeds highs_fastload.
from pyomo.contrib.vector.template_vectorize import (
    vectorized_construction,
    templatize_enabled_by_env,
    apply_env_templatize,
    model_has_templates,
    compile_templated_to_highs_arrays,
    extract_family,
    NotVectorizable,
)

# Honour PYOMO_VECTOR_CONSTRUCT=1 at import so setting the environment variable
# is sufficient to turn the fast path on process-wide (no code change).
apply_env_templatize()

__all__ = [
    'VectorVar',
    'VectorVarData',
    'VectorConstraint',
    'VectorConstraintData',
    'VectorObjective',
    'assemble',
    'compile_standard_form',
    'VectorMatrices',
    'VectorStandardFormInfo',
    'VectorPathDisabledError',
    'load_highs',
    'solve_highs',
    'matrices_to_highs_lp',
    'matrices_to_highs_model',
    'QuadraticModelError',
    'FastLoadHighs',
    'compile_to_highs_arrays',
    'build_highs_lp',
    'build_highs_model',
    'FastStepHighs',
    'vectorized_construction',
    'templatize_enabled_by_env',
    'apply_env_templatize',
    'model_has_templates',
    'compile_templated_to_highs_arrays',
    'extract_family',
    'NotVectorizable',
]
