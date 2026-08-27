"""Spike B (feasibility gate): template-expression numeric instantiation.

This is the go/no-go gate for scoping-doc D2-ambition-3 / risk R3: can we take a
user's *existing* ``def rule(m, i): ...`` constraint rule, run it **once** through
Pyomo's template machinery (``templatize_constraint``), and then numerically
instantiate the whole constraint family over its index set with NumPy array ops
- never building one expression tree per index?  If yes, existing Pyomo model
code could be made fast without a rewrite (addresses #1761/#2808 at the root).

The spike answers three questions with evidence:

  1. COVERAGE - which common rule shapes templatize at all?  (Scalar-affine and
     the ``sum(... for j in Set)`` idiom vs index conditionals / modulo.)
  2. CORRECTNESS - can we programmatically walk a templatized *linear* body and
     reconstruct the exact CSR rows a classic per-index repn produces?
  3. SPEED - three build rates for the same family:
        (a) classic     : rule -> tree -> generate_standard_repn, per index.
        (b) resolve      : templatize once, then set index + resolve_template +
                           repn, per index (does reusing the template help?).
        (c) vectorized   : extract the template skeleton once, fill the CSR with
                           NumPy over the whole index array (the ceiling).

The verdict weighs the (c)/(a) speedup against the coverage limits.
"""

from __future__ import annotations

import gc
import json
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pyomo.environ as pyo
import scipy.sparse as sp

from pyomo.core.expr.template_expr import (
    templatize_constraint,
    resolve_template,
    IndexTemplate,
    GetItemExpression,
    TemplateSumExpression,
)
from pyomo.core.expr import numeric_expr as ne
from pyomo.repn.standard_repn import generate_standard_repn


def _time(fn: Callable[[], Any], repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------- #
# 1. Coverage catalog
# --------------------------------------------------------------------------- #
def coverage_catalog() -> List[Dict[str, Any]]:
    cases = []

    def check(label: str, build, shape: str):
        m = build()
        try:
            expr, idx = templatize_constraint(m.c)
            cases.append({"rule": label, "shape": shape, "templatizes": True,
                          "template": str(expr)[:80]})
        except Exception as e:
            cases.append({"rule": label, "shape": shape, "templatizes": False,
                          "error": f"{type(e).__name__}: {str(e)[:80]}"})

    def b_scalar():
        m = pyo.ConcreteModel(); m.I = pyo.RangeSet(0, 5); m.x = pyo.Var(m.I)
        m.p = pyo.Param(m.I, initialize={i: 1.0 for i in range(6)}, mutable=True)
        m.c = pyo.Constraint(m.I, rule=lambda m, i: 3.0 * m.x[i] <= m.p[i]); return m

    def b_affine():
        m = pyo.ConcreteModel(); m.I = pyo.RangeSet(1, 5); m.x = pyo.Var(pyo.RangeSet(0, 5))
        m.c = pyo.Constraint(m.I, rule=lambda m, i: 2.0 * m.x[i] - m.x[i - 1] <= 1.0); return m

    def b_sumset():
        m = pyo.ConcreteModel(); m.N = pyo.RangeSet(0, 4); m.T = pyo.RangeSet(0, 3)
        m.f = pyo.Var(m.N, m.N, m.T)
        def bal(m, n, t):
            return sum(m.f[j, n, t] for j in m.N) - sum(m.f[n, j, t] for j in m.N) == 0
        m.c = pyo.Constraint(m.N, m.T, rule=bal); return m

    def b_cond():
        m = pyo.ConcreteModel(); m.I = pyo.RangeSet(0, 5); m.x = pyo.Var(m.I)
        def r(m, i):
            if i == 0:
                return pyo.Constraint.Skip
            return m.x[i] - m.x[i - 1] <= 1.0
        m.c = pyo.Constraint(m.I, rule=r); return m

    def b_mod():
        m = pyo.ConcreteModel(); m.I = pyo.RangeSet(0, 5); m.x = pyo.Var(m.I)
        m.c = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] + m.x[(i - 1) % 6] <= 1.0); return m

    def b_quad():
        m = pyo.ConcreteModel(); m.I = pyo.RangeSet(0, 5); m.x = pyo.Var(m.I)
        m.c = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] * m.x[i] <= 1.0); return m

    check("a*x[i] <= p[i]", b_scalar, "scalar-affine")
    check("2*x[i]-x[i-1] <= 1", b_affine, "scalar-affine (neighbour)")
    check("sum(f[j,n,t] for j in N) balance", b_sumset, "sum-over-set")
    check("if i==0: Skip else affine", b_cond, "index-conditional")
    check("x[i]+x[(i-1)%N]", b_mod, "modulo index")
    check("x[i]*x[i] <= 1", b_quad, "quadratic")
    return cases


# --------------------------------------------------------------------------- #
# 2/3. Affine-linear family: extractor + vectorized instantiation
# --------------------------------------------------------------------------- #
def _eval_index_over(node, template: IndexTemplate, values: np.ndarray) -> np.ndarray:
    """Evaluate an affine index expression (of one IndexTemplate) over an array.

    Supports the leaf IndexTemplate, integer/float constants, and +, -, * of
    those - i.e. the affine index arithmetic the template preserves symbolically
    (``_1``, ``_1 - 1``, ``2*_1`` ...).  Raises on anything non-affine so the
    caller can fall back to scalarization.
    """
    if node is template:
        return values.astype(np.int64)
    if type(node) in (int, float):
        return np.full(values.shape, node)
    if isinstance(node, IndexTemplate):
        # A different template than the row template (e.g. sum-local) - unsupported here.
        raise ValueError("multiple index templates")
    if isinstance(node, ne.NegationExpression):
        return -_eval_index_over(node.args[0], template, values)
    if isinstance(node, (ne.SumExpression, ne.NPV_SumExpression, ne.LinearExpression)):
        acc = np.zeros(values.shape, dtype=np.int64)
        for a in node.args:
            acc = acc + _eval_index_over(a, template, values)
        return acc
    if isinstance(node, (ne.ProductExpression, ne.NPV_ProductExpression,
                         ne.MonomialTermExpression)):
        a, b = node.args
        return _eval_index_over(a, template, values) * _eval_index_over(b, template, values)
    # Fallback: try to read a constant value.
    try:
        return np.full(values.shape, float(pyo.value(node)))
    except Exception as exc:
        raise ValueError(f"non-affine index node {type(node).__name__}") from exc


def _collect_linear_terms(body) -> List[Tuple[float, GetItemExpression]]:
    """Flatten a linear template body into (constant_coeff, var_getitem) terms.

    Handles the node shapes a linear rule produces: SumExpression /
    LinearExpression, MonomialTermExpression, ProductExpression(const, getitem),
    NegationExpression, and a bare GetItem.  The coefficient must be a numeric
    constant (index-independent) for this fast path; a coefficient that depends
    on the index raises (a documented limitation of the spike, not the design).
    """
    terms: List[Tuple[float, GetItemExpression]] = []

    def rec(node, sign):
        if isinstance(node, GetItemExpression):
            terms.append((1.0 * sign, node))
            return
        if isinstance(node, ne.NegationExpression):
            rec(node.args[0], -sign)
            return
        if isinstance(node, ne.MonomialTermExpression):
            coef, var = node.args
            terms.append((float(pyo.value(coef)) * sign, var))
            return
        if isinstance(node, (ne.ProductExpression, ne.NPV_ProductExpression)):
            a, b = node.args
            # const * getitem  (either order)
            if isinstance(b, GetItemExpression):
                terms.append((float(pyo.value(a)) * sign, b))
                return
            if isinstance(a, GetItemExpression):
                terms.append((float(pyo.value(b)) * sign, a))
                return
            raise ValueError("product of two variables (nonlinear)")
        if isinstance(node, (ne.SumExpression, ne.LinearExpression)):
            args = node.args
            for a in args:
                rec(a, sign)
            return
        # A bare constant contributes to the RHS, not a variable term.
        try:
            float(pyo.value(node))
            return
        except Exception as exc:
            raise ValueError(f"unhandled node {type(node).__name__}") from exc

    rec(body, 1.0)
    return terms


def _var_position_map(var):
    """Return (offset, is_rangeset) so column = index - offset for a RangeSet var."""
    iset = var.index_set()
    # Fast path: contiguous integer RangeSet -> position is index - first.
    try:
        first = iset.first()
        last = iset.last()
        if (last - first + 1) == len(iset):
            return int(first), True
    except Exception:
        pass
    return None, False


def vectorized_instantiate(model, con_name: str) -> sp.csr_matrix:
    """Build the constraint CSR for a scalar-affine linear family via the template.

    Templatize once, extract the constant-coeff/affine-index skeleton, then fill
    row/col/data arrays with NumPy over the whole index set.  No per-index tree.
    Returns the constraint matrix over the family's own variable component's
    columns (single Var assumed for this spike family).
    """
    con = getattr(model, con_name)
    expr, indices = templatize_constraint(con)
    template = indices[0]
    body = expr.args[0]
    rhs_node = expr.args[1]
    terms = _collect_linear_terms(body)

    index_values = np.array(list(con.index_set()), dtype=np.int64)
    n_rows = len(index_values)

    # Assume a single variable component across all terms (true for this family).
    var = terms[0][1].args[0]
    offset, is_range = _var_position_map(var)
    if not is_range:
        raise ValueError("non-RangeSet var not supported in spike fast path")
    n_cols = len(var)

    rows_list, cols_list, data_list = [], [], []
    row_idx = np.arange(n_rows)
    for coeff, getitem in terms:
        idx_expr = getitem.args[1]  # single-dim index for this family
        col_index_vals = _eval_index_over(idx_expr, template, index_values)
        cols = col_index_vals - offset
        rows_list.append(row_idx)
        cols_list.append(cols)
        data_list.append(np.full(n_rows, coeff))
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    return A


def classic_csr(model, con_name: str, var) -> sp.csr_matrix:
    """Reference CSR built the classic way: per-index generate_standard_repn."""
    con = getattr(model, con_name)
    offset, _ = _var_position_map(var)
    n_cols = len(var)
    id_to_col = {id(var[i]): (i - offset) for i in var.index_set()}
    rows_list, cols_list, data_list = [], [], []
    for r, k in enumerate(con.index_set()):
        repn = generate_standard_repn(con[k].body, quadratic=False)
        for coef, v in zip(repn.linear_coefs, repn.linear_vars):
            rows_list.append(r)
            cols_list.append(id_to_col[id(v)])
            data_list.append(coef)
    A = sp.coo_matrix(
        (data_list, (rows_list, cols_list)),
        shape=(len(list(con.index_set())), n_cols),
    ).tocsr()
    return A


def _build_ramp_model(N: int) -> pyo.ConcreteModel:
    """Scalar-affine family: 2*x[i] - x[i-1] <= b[i], i in 1..N (ramp/storage shape)."""
    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, N)          # variable index 0..N
    m.x = pyo.Var(m.J)
    m.I = pyo.RangeSet(1, N)          # constraint index 1..N
    m.c = pyo.Constraint(m.I, rule=lambda m, i: 2.0 * m.x[i] - m.x[i - 1] <= 1.0)
    return m


def affine_family(sizes: List[int]) -> Dict[str, Any]:
    rows = []
    for N in sizes:
        m = _build_ramp_model(N)

        # (a) classic per-index repn.
        def classic():
            return classic_csr(m, "c", m.x)

        # (b) template + resolve per index + repn.
        expr, indices = templatize_constraint(m.c)
        template = indices[0]

        def resolve_path():
            n = 0
            for k in m.I:
                template.set_value(k)
                concrete = resolve_template(expr)
                generate_standard_repn(concrete.args[0], quadratic=False)
                n += 1
            return n

        # (c) vectorized instantiation.
        def vectorized():
            return vectorized_instantiate(m, "c")

        # Correctness: vectorized == classic.
        A_classic = classic()
        A_vec = vectorized()
        same = (A_classic != A_vec).nnz == 0

        t_classic = _time(classic, repeats=3)
        t_resolve = _time(resolve_path, repeats=3)
        t_vec = _time(vectorized, repeats=5)
        rows.append({
            "N": N,
            "n_constraints": N,
            "correct": bool(same),
            "classic_s": t_classic,
            "resolve_s": t_resolve,
            "vectorized_s": t_vec,
            "classic_rows_per_s": N / t_classic,
            "vectorized_rows_per_s": N / t_vec,
            "speedup_vec_over_classic": t_classic / t_vec,
            "speedup_resolve_over_classic": t_classic / t_resolve,
        })
    return {"family": "scalar_affine_ramp", "rows": rows}


# --------------------------------------------------------------------------- #
# Sum-over-set family (flow balance): templatize + resolve timing + ceiling
# --------------------------------------------------------------------------- #
def _build_flow_model(N: int, T: int) -> pyo.ConcreteModel:
    """Flow balance as an UNFILTERED sum over the node set.

    NOTE (a Spike-B finding): the idiomatic dense-flow rule uses a filtered sum
    ``sum(m.f[j, n, t] for j in m.N if j != n)`` to drop self-loops.  That
    ``if j != n`` filter compares the sum-local index template to the row index
    template and raises ``PyomoException`` under templatization - so the filtered
    idiom does NOT templatize.  We use the unfiltered form here (variables over
    the full N x N x T grid) precisely because it is the version that templatizes,
    to measure the resolve path on a sum-over-set body.
    """
    m = pyo.ConcreteModel()
    m.N = pyo.RangeSet(0, N - 1)
    m.T = pyo.RangeSet(0, T - 1)
    m.f = pyo.Var(m.N, m.N, m.T)

    def bal(m, n, t):
        return (sum(m.f[j, n, t] for j in m.N)
                - sum(m.f[n, j, t] for j in m.N) == 0.0)

    m.c = pyo.Constraint(m.N, m.T, rule=bal)
    return m


def sum_family(cases: List[Tuple[int, int]]) -> Dict[str, Any]:
    rows = []
    for (N, T) in cases:
        m = _build_flow_model(N, T)
        n_rows = N * T

        def classic():
            n = 0
            for k in m.c.index_set():
                generate_standard_repn(m.c[k].body, quadratic=False)
                n += 1
            return n

        templatizes = True
        try:
            expr, indices = templatize_constraint(m.c)
        except Exception as e:
            templatizes = False
            expr = None

        t_classic = _time(classic, repeats=3)

        t_resolve = None
        if templatizes:
            n_tmpl = len(indices)

            def resolve_path():
                cnt = 0
                for k in m.c.index_set():
                    kk = k if isinstance(k, tuple) else (k,)
                    for tmpl, val in zip(indices, kk):
                        tmpl.set_value(val)
                    resolve_template(expr)
                    cnt += 1
                return cnt

            try:
                t_resolve = _time(resolve_path, repeats=2)
            except Exception as e:
                t_resolve = None

        rows.append({
            "N": N, "T": T, "n_constraints": n_rows,
            "templatizes": templatizes,
            "classic_s": t_classic,
            "classic_rows_per_s": n_rows / t_classic,
            "resolve_s": t_resolve,
            "speedup_resolve_over_classic": (t_classic / t_resolve) if t_resolve else None,
        })
    return {"family": "sum_over_set_flow_balance", "rows": rows}


def run(affine_sizes=None, flow_cases=None) -> Dict[str, Any]:
    if affine_sizes is None:
        affine_sizes = [10_000, 100_000, 1_000_000]
    if flow_cases is None:
        flow_cases = [(10, 100), (20, 250)]
    return {
        "spike": "B_template_expr",
        "coverage": coverage_catalog(),
        "affine": affine_family(affine_sizes),
        "sum_over_set": sum_family(flow_cases),
    }


def _print(rep: Dict[str, Any]) -> None:
    print("=" * 96)
    print("SPIKE B - template-expression numeric instantiation (rule-vectorization gate)")
    print("=" * 96)
    print("\n1) COVERAGE - which rule shapes templatize?")
    for c in rep["coverage"]:
        mark = "YES" if c["templatizes"] else "no "
        detail = c.get("template", c.get("error", ""))
        print(f"   [{mark}] {c['shape']:<28} {c['rule']:<34} {detail[:40]}")

    print("\n2/3) SCALAR-AFFINE family (2*x[i]-x[i-1]<=1): correctness + speed")
    print(f"   {'N':>10} {'correct':>8} {'classic':>12} {'resolve':>12} {'vectorized':>12} "
          f"{'vec/classic':>12} {'resolve/classic':>16}")
    for r in rep["affine"]["rows"]:
        print(f"   {r['N']:>10} {str(r['correct']):>8} {r['classic_s']*1000:>10.1f}ms "
              f"{r['resolve_s']*1000:>10.1f}ms {r['vectorized_s']*1000:>10.2f}ms "
              f"{r['speedup_vec_over_classic']:>10.0f}x {r['speedup_resolve_over_classic']:>14.2f}x")

    print("\n   SUM-OVER-SET family (flow balance): templatization + resolve")
    for r in rep["sum_over_set"]["rows"]:
        rs = f"{r['resolve_s']*1000:.1f}ms" if r["resolve_s"] else "n/a"
        sp_ = f"{r['speedup_resolve_over_classic']:.2f}x" if r["speedup_resolve_over_classic"] else "n/a"
        print(f"   N={r['N']} T={r['T']} rows={r['n_constraints']}: templatizes={r['templatizes']} "
              f"classic={r['classic_s']*1000:.1f}ms resolve={rs} (resolve/classic={sp_})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--affine-sizes", type=str, default="10000,100000,1000000")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    sizes = [int(s) for s in a.affine_sizes.split(",")]
    rep = run(affine_sizes=sizes)
    _print(rep)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"# wrote {a.out}")
