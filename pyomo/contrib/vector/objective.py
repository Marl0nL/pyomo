# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""A linear (or convex-quadratic) objective stored as arrays (fast path).

The fast path needs the objective cost vector as arrays, not as an expression
tree over N materialized variables (building that tree would materialize every
variable, defeating the columnar ``Var``).  :class:`VectorObjective` therefore
stores one coefficient array per :class:`VectorVar` block::

    m.obj = VectorObjective(terms={m.flow: c_flow, m.stor: c_stor},
                            sense=minimize)

An optional **convex-quadratic** part is stored the same array-native way: one
sparse Hessian block per (VectorVar, VectorVar) pair, so the objective is
``c @ x + 0.5 * x @ Q @ x`` with ``Q`` a sparse symmetric matrix over the vector
components (the #1761 use case, scoping doc §6, Phase-3 quadratic ambition)::

    m.obj = VectorObjective(terms={m.x: c},
                            quadratic=Q,                # single-block 0.5 x'Q x
                            sense=minimize)
    # multi-block: quadratic={(m.x, m.x): Qxx, (m.x, m.y): Qxy}

``Q`` (or each block) is the **Hessian** in ``0.5 x'Q x`` -- exactly what a QP
solver wants (the gurobipy / cvxpy convention).  A diagonal block ``(v, v)`` is
the symmetric Hessian sub-block over ``v``; an off-diagonal block ``(vi, vj)``
is the coupling ``x_i' B x_j`` (the transpose half is implied).  The Hessian
only ever reaches a solver that supports convex QP (HiGHS): a non-convex ``Q``
or an integer variable (MIQP) is rejected loudly downstream, never silently
mis-solved.

The classic ``.expr`` (a :class:`LinearExpression`, or a quadratic expression
when a Hessian is present) is built lazily only if an unaware consumer asks for
it (scalarization, scoping doc §6.5).
"""

from __future__ import annotations

import logging
from weakref import ref as weakref_ref

from pyomo.common.dependencies import numpy as np, scipy
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
        quadratic = kwargs.pop('quadratic', None)
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
        self._quad_arg = quadratic
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
        self._quad = None  # validated list of (VectorVar, VectorVar, csr block)
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
        self._quad = self._validate_quadratic()
        # Register self as the single scalar data object so that
        # component_data_objects(Objective) finds it.
        self._data[None] = self

    # ------------------------------------------------------------------ #
    def _validate_quadratic(self):
        """Normalize the ``quadratic`` argument into ``[(vrow, vcol, csr), ...]``.

        Accepts either a single sparse/dense matrix (interpreted as the Hessian
        block ``(v, v)`` when the objective references exactly one VectorVar) or
        a dict mapping ``(VectorVar, VectorVar) -> block``.  Each block is stored
        as a float64 CSR matrix of shape ``(vrow.n, vcol.n)``; it is the Hessian
        sub-block in ``0.5 x'Q x`` (a diagonal block is symmetric; an off-diagonal
        block is the coupling ``x_i' B x_j``).
        """
        arg = self._quad_arg
        if arg is None:
            return []
        if isinstance(arg, dict):
            blocks = list(arg.items())
        else:
            # A bare matrix: only well-defined for a single-variable objective.
            vs = [v for v, _ in self._terms]
            if len(vs) != 1:
                raise ValueError(
                    "VectorObjective 'quadratic' given as a bare matrix requires "
                    "exactly one VectorVar in 'terms'; with multiple blocks use "
                    "quadratic={(vrow, vcol): block, ...}."
                )
            blocks = [((vs[0], vs[0]), arg)]

        out = []
        for key, block in blocks:
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError(
                    "VectorObjective 'quadratic' keys must be (VectorVar, "
                    f"VectorVar) pairs; got {key!r}."
                )
            vr, vc = key
            if not isinstance(vr, VectorVar) or not isinstance(vc, VectorVar):
                raise TypeError(
                    "VectorObjective 'quadratic' pairs must be VectorVar -> "
                    "VectorVar."
                )
            if not vr._constructed:
                vr.construct()
            if not vc._constructed:
                vc.construct()
            if scipy.sparse.issparse(block):
                B = block.tocsr()
            else:
                B = scipy.sparse.csr_matrix(np.asarray(block, dtype=np.float64))
            B = B.astype(np.float64)
            B.eliminate_zeros()
            if B.shape != (vr.n, vc.n):
                raise ValueError(
                    f"VectorObjective quadratic block for ('{vr.name}', "
                    f"'{vc.name}') has shape {B.shape}, expected ({vr.n}, {vc.n})."
                )
            if vr is vc:
                # Diagonal block: warn (do not silently correct) on asymmetry --
                # the caller owns the Hessian convention.
                if (abs(B - B.T) > 1e-12).nnz:
                    logger.warning(
                        "VectorObjective quadratic block for '%s' is not "
                        "symmetric; the Hessian is symmetrized as (Q + Q')/2 "
                        "when handed to the solver." % (vr.name,),
                        extra={'id': 'W-VEC04'},
                    )
            out.append((vr, vc, B))
        return out

    # ------------------------------------------------------------------ #
    # Fast-path accessors
    # ------------------------------------------------------------------ #
    @property
    def terms(self):
        return self._terms

    @property
    def quadratic_terms(self):
        """List of ``(vrow, vcol, csr_block)`` Hessian blocks (``0.5 x'Q x``)."""
        return self._quad

    def is_quadratic(self):
        return bool(self._quad)

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
        linear = LinearExpression(
            constant=self._const, linear_coefs=coefs, linear_vars=varlist
        )
        if not self._quad:
            return linear
        # Quadratic scalarization: build 0.5 x'Q x term-by-term from the sparse
        # Hessian blocks.  A diagonal block contributes 0.5 * sum Q_ij x_i x_j;
        # an off-diagonal (coupling) block contributes sum B_ij x_i x_j.
        expr = linear
        for vr, vc, B in self._quad:
            if vr is vc:
                # Symmetrize a diagonal block so scalarization agrees with the
                # solver hand-off (which also symmetrizes) on asymmetric input.
                B = (B + B.transpose()) * 0.5
                scale = 0.5
            else:
                scale = 1.0
            coo = B.tocoo()
            for i, j, val in zip(coo.row, coo.col, coo.data):
                xi = vr[vr.index_at(int(i))]
                xj = vc[vc.index_at(int(j))]
                expr = expr + (scale * float(val)) * xi * xj
        return expr

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
        q_nnz = sum(int(B.nnz) for _, _, B in (self._quad or []))
        headers = [
            ("Size", 1),
            ("Sense", str(self._sense)),
            ("Terms", n_terms),
            ("Quadratic nnz", q_nnz),
            ("Vectorized", True),
            ("Scalarized", self._scalarized),
        ]
        return (headers, (), None, None)
