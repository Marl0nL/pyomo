# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Standard-form equivalence oracle (self-contained, permutation invariant).

The Phase-1 correctness gate is: the vectorized fast path and the classic path
must produce **identical standard forms up to row/column permutation, with the
same bounds** (scoping doc Phase 1 exit criterion).

This module implements a canonical signature of a
:class:`~pyomo.repn.plugins.standard_form.LinearStandardFormInfo`-shaped object
that is invariant to row and column permutation, keyed on *variable identity*
(``(component_local_name, index)``).  Two standard forms are equivalent iff
their canonical signatures are equal.

The benchmark harness ships its own committed oracle for the benchmark-level
gate; this self-contained copy keeps the in-tree ``pyomo`` test suite free of any
dependency on ``bench/`` so it runs in CI.
"""

from __future__ import annotations

_TOL = 9  # rounding decimals for float comparison


def _col_key(v):
    pc = v.parent_component()
    return (pc.local_name, v.index())


def canonical_standard_form(info):
    """Return a permutation-invariant, identity-keyed signature of ``info``.

    ``info`` must expose ``A`` (scipy sparse), ``rhs``, ``rows`` (each with a
    ``bound_type``), ``columns`` (VarData), ``c`` (scipy sparse, 1 x n) and
    ``c_offset`` -- i.e. the stock ``LinearStandardFormInfo`` interface (which
    :class:`~pyomo.contrib.vector.matrices.VectorStandardFormInfo` mirrors).
    """
    A = info.A.tocsr()
    cols = info.columns
    keys = [_col_key(v) for v in cols]

    row_sigs = []
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        entries = tuple(
            sorted(
                (keys[c], round(float(d), _TOL))
                for c, d in zip(A.indices[s:e], A.data[s:e])
            )
        )
        row_sigs.append(
            (entries, info.rows[i].bound_type, round(float(info.rhs[i]), _TOL))
        )

    c = info.c.tocsr()
    c_sig = tuple(
        sorted(
            (keys[j], round(float(c[0, j]), _TOL))
            for j in range(c.shape[1])
            if c[0, j] != 0
        )
    )

    def _bnd(b):
        return None if b is None else round(float(b), _TOL)

    bound_sig = tuple(
        sorted(
            (keys[j], (_bnd(cols[j].bounds[0]), _bnd(cols[j].bounds[1])),
             bool(cols[j].is_integer()))
            for j in range(len(cols))
        )
    )

    offset = float(info.c_offset[0]) if hasattr(info.c_offset, '__len__') else float(
        info.c_offset
    )
    return (sorted(row_sigs), c_sig, bound_sig, round(offset, _TOL))


def assert_equivalent(test, info_a, info_b, msg=""):
    """unittest helper: assert two standard forms are equivalent."""
    ca = canonical_standard_form(info_a)
    cb = canonical_standard_form(info_b)
    if ca != cb:
        # Produce a focused diff over the four signature fields.
        fields = ['rows', 'objective', 'bounds', 'offset']
        detail = []
        for name, a, b in zip(fields, ca, cb):
            if a != b:
                detail.append(f"  {name} differ:\n    A={a}\n    B={b}")
        test.fail((msg + "\n" if msg else "") + "standard forms not equivalent:\n"
                  + "\n".join(detail))
