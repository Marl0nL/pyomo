# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""A linear objective stored as coefficient arrays (vectorized fast path).

The fast path needs the objective cost vector as arrays, not as an expression
tree over N materialized variables (building that tree would materialize every
variable, defeating the columnar ``Var``).  :class:`VectorObjective` therefore
stores one coefficient array per :class:`VectorVar` block::

    m.obj = VectorObjective(terms={m.flow: c_flow, m.stor: c_stor},
                            sense=minimize)

The classic ``.expr`` (a :class:`LinearExpression`) is built lazily only if an
unaware consumer asks for it (scalarization, scoping doc §6.5).
"""

from __future__ import annotations

import logging
from weakref import ref as weakref_ref

from pyomo.common.dependencies import numpy as np
from pyomo.common.modeling import NOTSET
from pyomo.common.enums import ObjectiveSense
from pyomo.core.base import minimize
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.objective import Objective, ObjectiveData
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.contrib.vector.var import VectorVar

logger = logging.getLogger('pyomo.contrib.vector')


class VectorObjective(ObjectiveData, ActiveIndexedComponent):
    """A scalar linear objective backed by per-variable coefficient arrays."""

    def __init__(self, *args, **kwargs):
        terms = kwargs.pop('terms', None)
        sense = kwargs.pop('sense', minimize)
        constant = kwargs.pop('constant', 0.0)
        if terms is None:
            raise ValueError("VectorObjective requires 'terms' (a dict/list of "
                             "(VectorVar -> coefficient array)).")
        # Normalize terms into an ordered list of (VectorVar, coef_array).
        if isinstance(terms, dict):
            items = list(terms.items())
        else:
            items = list(terms)
        self._terms_arg = items
        self._const = float(constant)

        # ObjectiveData surface (this component is its own scalar data object).
        self._args_ = (None,)  # expr built lazily on scalarization
        self._component = weakref_ref(self)
        self._active = True
        self._sense = ObjectiveSense(sense)

        kwargs.setdefault('ctype', Objective)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)
        self._index = UnindexedComponent_index

        self._terms = None  # validated list of (VectorVar, float64 array)
        self._scalarized = False

    # ------------------------------------------------------------------ #
    def construct(self, data=None):
        if self._constructed:
            return
        self._constructed = True
        terms = []
        for v, coef in self._terms_arg:
            if not isinstance(v, VectorVar):
                raise TypeError(
                    "VectorObjective terms must map VectorVar -> array; got "
                    f"{type(v).__name__}."
                )
            if not v._constructed:
                v.construct()
            arr = np.asarray(coef, dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full(v.n, float(arr), dtype=np.float64)
            if arr.shape != (v.n,):
                raise ValueError(
                    f"VectorObjective coefficient array for '{v.name}' has shape "
                    f"{arr.shape}, expected ({v.n},)."
                )
            terms.append((v, arr))
        self._terms = terms
        # Register self as the single scalar data object so that
        # component_data_objects(Objective) finds it.
        self._data[None] = self

    # ------------------------------------------------------------------ #
    # Fast-path accessors
    # ------------------------------------------------------------------ #
    @property
    def terms(self):
        return self._terms

    @property
    def constant(self):
        return self._const

    @property
    def sense(self):
        return self._sense

    @sense.setter
    def sense(self, sense):
        self._sense = ObjectiveSense(sense)

    def is_minimizing(self):
        return self._sense == minimize

    # ------------------------------------------------------------------ #
    # Scalarization: build the classic LinearExpression lazily
    # ------------------------------------------------------------------ #
    def _build_expression(self):
        coefs = []
        varlist = []
        for v, arr in self._terms:
            nz = np.nonzero(arr)[0]
            for pos in nz:
                coefs.append(float(arr[pos]))
                varlist.append(v[v.index_at(int(pos))])
        return LinearExpression(
            constant=self._const, linear_coefs=coefs, linear_vars=varlist
        )

    @property
    def expr(self):
        if self._args_[0] is None:
            self._args_ = (self._build_expression(),)
            if not self._scalarized:
                self._scalarized = True
                logger.warning(
                    "VectorObjective '%s' was scalarized: a consumer requested "
                    "its expression, so a classic LinearExpression over all "
                    "referenced variables was built (scoping doc §6.5)."
                    % (self.name,),
                    extra={'id': 'W-VEC03'},
                )
        return self._args_[0]

    @expr.setter
    def expr(self, value):
        self._args_ = (value,)

    def set_value(self, expr):
        self._args_ = (expr,)

    def __len__(self):
        return 1 if self._constructed else 0

    def _pprint(self):
        n_terms = sum(v.n for v, _ in (self._terms or []))
        headers = [
            ("Size", 1),
            ("Sense", str(self._sense)),
            ("Terms", n_terms),
            ("Vectorized", True),
            ("Scalarized", self._scalarized),
        ]
        return (headers, (), None, None)
