# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""``pyomo.contrib.vector`` -- vectorized (columnar) model construction fast path.

Phase-1 vertical slice of the "Vectorized Model Construction for Pyomo" project
(scoping doc + Phase-0 baseline report).  Provides:

* :class:`VectorVar` -- columnar, array-backed indexed variable
  (materialize-on-touch).
* :class:`VectorConstraint` -- an explicit-array linear constraint family
  (``VectorConstraint(A=csr, x=..., lb=..., ub=...)``).
* :class:`VectorObjective` -- a linear objective stored as coefficient arrays.
* :func:`compile_standard_form` -- splice the vector components into a standard
  form comparable to :class:`~pyomo.repn.plugins.standard_form.LinearStandardFormCompiler`.
* :func:`load_highs` -- direct ``Highs.passModel`` array hand-off (the load prize).

All three components fall back to classic per-index data objects
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
from pyomo.contrib.vector.highs import load_highs, solve_highs, matrices_to_highs_lp

# Importing fastload registers the ``highs_fastload`` solver with both the v2
# and legacy SolverFactory (the transparent classic-model fast hand-off).
from pyomo.contrib.vector.fastload import (
    FastLoadHighs,
    compile_to_highs_arrays,
    build_highs_lp,
)

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
    'FastLoadHighs',
    'compile_to_highs_arrays',
    'build_highs_lp',
]
