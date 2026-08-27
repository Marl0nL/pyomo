"""Standard-form equivalence oracle: Pyomo model  ==  array-native matrices.

This is the correctness counterpart to the timing harness and the artifact the
vectorization project's Phase 1 needs: an automated check that two builders of
"the same LP" really do produce the **same standard form, up to row and column
permutation**, keyed on **variable identity** (not just matching shapes, nnz, or
optimal objective — any of which two genuinely different LPs can share).

For each supported synthetic (network flow, facility location) it compares:

  * the **Pyomo** model — ground truth built the classic per-constraint way
    (``generate_standard_repn`` on each ``ConstraintData``), against
  * the **array-native** matrices (``bench.comparators.array_native``), whose
    columns carry ``col_names`` matching the Pyomo VarData names.

The comparison, in increasing strength (earlier ones are fast pre-filters):

  1. variable-name sets equal, and per-variable ``(lb, ub, is_integer)`` equal;
  2. nnz and the row/column degree-sequence multisets equal (permutation
     invariants — cheap and catch gross structural drift);
  3. objective linear-coefficient vector (keyed by name) + constant equal;
  4. **authoritative:** the constraint system as a *multiset of normalized rows*
     is equal.  Each row is normalized to ``(sorted (name, coef) terms, rhs,
     sense)`` with columns keyed by variable name (so column order is irrelevant)
     and equality rows sign-canonicalized (so ``a==b`` and ``-a==-b`` collapse);
     equality of the two multisets is equivalence up to row permutation.
  5. optional: both LPs solved through HiGHS and their optimal objectives
     compared (reproduces the report's headline obj numbers).

Run standalone (exits non-zero on any mismatch — usable as a CI gate):

    python -m bench.equivalence --out bench/results/equivalence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

from bench.comparators import array_native as an

# Models with both a Pyomo generator and an array-native builder.
SUPPORTED = sorted(an.SUPPORTED)

_ROUND = 9  # decimals for coefficient / rhs comparison


def _terms_from_repn(repn) -> Dict[str, float]:
    """{var_name: coef} for a linear standard repn, duplicates summed, zeros dropped."""
    acc: Dict[str, float] = defaultdict(float)
    for coef, var in zip(repn.linear_coefs, repn.linear_vars):
        acc[var.name] += float(coef)
    return {n: v for n, v in acc.items() if abs(v) > 1e-12}


def _norm_row(terms: Dict[str, float], rhs: float, sense: str) -> Tuple:
    """Canonical, hashable key for one constraint row.

    ``sense`` is "EQ" or "LE".  EQ rows are sign-canonicalized (the whole row may
    be multiplied by -1) since ``a==b`` and ``-a==-b`` are the same constraint;
    LE rows are NOT sign-flipped (that would reverse the inequality).
    """
    items = sorted((n, round(c, _ROUND)) for n, c in terms.items() if abs(c) > 1e-12)
    rhs_r = round(float(rhs), _ROUND)
    if sense == "EQ" and items:
        # Flip sign so the first term (by name order) is positive -> canonical.
        if items[0][1] < 0:
            items = [(n, -c) for n, c in items]
            rhs_r = round(-rhs_r, _ROUND)
    return (tuple(items), rhs_r, sense)


def pyomo_standard_form(model: pyo.ConcreteModel) -> Dict[str, Any]:
    """Extract the Pyomo model's standard form the classic (ground-truth) way."""
    rows: Counter = Counter()
    row_deg: Counter = Counter()
    nnz = 0
    for con in model.component_objects(pyo.Constraint, active=True):
        for cd in con.values():
            repn = generate_standard_repn(cd.body, quadratic=False)
            terms = _terms_from_repn(repn)
            const = float(repn.constant)
            deg = len(terms)
            has_lo = cd.lower is not None
            has_hi = cd.upper is not None
            lo = float(pyo.value(cd.lower)) if has_lo else None
            hi = float(pyo.value(cd.upper)) if has_hi else None
            if cd.equality or (has_lo and has_hi and lo == hi):
                rows[_norm_row(terms, lo - const, "EQ")] += 1
                row_deg[deg] += 1
                nnz += deg
            else:
                if has_hi:  # body <= hi  ->  terms <= hi - const
                    rows[_norm_row(terms, hi - const, "LE")] += 1
                    row_deg[deg] += 1
                    nnz += deg
                if has_lo:  # body >= lo  ->  (-terms) <= -(lo - const)
                    neg = {n: -c for n, c in terms.items()}
                    rows[_norm_row(neg, -(lo - const), "LE")] += 1
                    row_deg[deg] += 1
                    nnz += deg

    # Objective (assume a single active objective, minimize).
    obj_terms: Dict[str, float] = {}
    obj_const = 0.0
    for o in model.component_objects(pyo.Objective, active=True):
        od = next(iter(o.values()))
        orepn = generate_standard_repn(od.expr, quadratic=False)
        obj_terms = _terms_from_repn(orepn)
        obj_const = float(orepn.constant)
        break

    # Bounds + integrality, keyed by variable name.
    bounds: Dict[str, Tuple] = {}
    col_deg_names: Counter = Counter()
    for v in model.component_objects(pyo.Var, active=True):
        for vd in v.values():
            lb = float(vd.lb) if vd.lb is not None else None
            ub = float(vd.ub) if vd.ub is not None else None
            bounds[vd.name] = (lb, ub, bool(vd.is_integer()))
    # column degree sequence (nonzeros per variable) from the constraint walk
    col_counts: Counter = Counter()
    for con in model.component_objects(pyo.Constraint, active=True):
        for cd in con.values():
            repn = generate_standard_repn(cd.body, quadratic=False)
            for n in _terms_from_repn(repn):
                col_counts[n] += 1
    col_deg = Counter(col_counts.values())

    return {
        "rows": rows,
        "row_deg": row_deg,
        "col_deg": col_deg,
        "nnz": nnz,
        "obj_terms": {n: round(c, _ROUND) for n, c in obj_terms.items()},
        "obj_const": round(obj_const, _ROUND),
        "bounds": bounds,
        "var_names": set(bounds),
    }


def array_standard_form(mx: an.Matrices) -> Dict[str, Any]:
    """Extract the array-native matrices into the same normalized standard form."""
    names = mx.col_names
    assert names is not None and len(names) == mx.n_var, "array builder must set col_names"

    rows: Counter = Counter()
    row_deg: Counter = Counter()
    nnz = 0

    def add_block(A: sp.spmatrix, b: np.ndarray, sense: str):
        nonlocal nnz
        Ac = A.tocsr()
        for r in range(Ac.shape[0]):
            s, e = Ac.indptr[r], Ac.indptr[r + 1]
            terms = {names[Ac.indices[k]]: float(Ac.data[k]) for k in range(s, e)}
            terms = {n: c for n, c in terms.items() if abs(c) > 1e-12}
            rows[_norm_row(terms, float(b[r]), sense)] += 1
            row_deg[len(terms)] += 1
            nnz += len(terms)

    add_block(mx.A_eq, mx.b_eq, "EQ")
    add_block(mx.A_ub, mx.b_ub, "LE")

    A_all = sp.vstack([mx.A_eq, mx.A_ub]).tocsc()
    col_deg = Counter(np.diff(A_all.indptr).tolist())

    obj_terms = {}
    for j, coef in enumerate(mx.c):
        if abs(coef) > 1e-12:
            obj_terms[names[j]] = round(float(coef), _ROUND)

    integ = mx.integrality if mx.integrality is not None else np.zeros(mx.n_var, bool)
    bounds = {
        names[j]: (
            float(mx.lb[j]) if np.isfinite(mx.lb[j]) else None,
            float(mx.ub[j]) if np.isfinite(mx.ub[j]) else None,
            bool(integ[j]),
        )
        for j in range(mx.n_var)
    }
    return {
        "rows": rows,
        "row_deg": row_deg,
        "col_deg": col_deg,
        "nnz": nnz,
        "obj_terms": obj_terms,
        "obj_const": 0.0,
        "bounds": bounds,
        "var_names": set(names),
    }


def _solve_obj_pyomo(model) -> Optional[float]:
    try:
        from pyomo.contrib.appsi.solvers import Highs

        h = Highs()
        h.config.load_solution = True
        h.solve(model)
        return round(float(pyo.value(model.obj)), 6)
    except Exception:
        return None


def _solve_obj_array(mx) -> Optional[float]:
    try:
        h = an.load_highs(mx)
        h.run()
        return round(float(h.getInfo().objective_function_value), 6)
    except Exception:
        return None


def check_model(name: str, params: Dict[str, Any], solve: bool = True) -> Dict[str, Any]:
    """Run the full oracle for one model/size; return a structured verdict."""
    from bench.models import network_flow, facility_location

    gens = {"network_flow": network_flow, "facility_location": facility_location}
    pmodel = gens[name].build_pyomo(dict(params))
    mx = an.BUILDERS[name](dict(params))

    P = pyomo_standard_form(pmodel)
    Aq = array_standard_form(mx)

    checks = {
        "var_names_equal": P["var_names"] == Aq["var_names"],
        "bounds_equal": P["bounds"] == Aq["bounds"],
        "nnz_equal": P["nnz"] == Aq["nnz"],
        "row_degree_multiset_equal": P["row_deg"] == Aq["row_deg"],
        "col_degree_multiset_equal": P["col_deg"] == Aq["col_deg"],
        "objective_equal": (P["obj_terms"] == Aq["obj_terms"]
                            and P["obj_const"] == Aq["obj_const"]),
        # authoritative:
        "constraint_rows_equal_up_to_permutation": P["rows"] == Aq["rows"],
    }
    result = {
        "model": name,
        "params": dict(params),
        "n_vars": len(P["var_names"]),
        "nnz_pyomo": P["nnz"],
        "nnz_array": Aq["nnz"],
        "checks": checks,
        "equivalent": all(checks.values()),
    }
    if solve:
        po, ao = _solve_obj_pyomo(pmodel), _solve_obj_array(mx)
        result["obj_pyomo"] = po
        result["obj_array"] = ao
        result["obj_equal"] = (po is not None and ao is not None
                               and abs(po - ao) <= 1e-6 * max(1.0, abs(po)))
    return result


# Small, cheap sizes for the oracle: correctness is size-independent, so we run
# an xs and a 1e4 case per model rather than the heavy 1e6 timing sizes.
_ORACLE_SIZES = ["xs", "1e4"]


def check_all(sizes: Optional[List[str]] = None, solve: bool = True) -> Dict[str, Any]:
    sizes = sizes or _ORACLE_SIZES
    from bench.models import network_flow, facility_location

    gens = {"network_flow": network_flow, "facility_location": facility_location}
    results = []
    for name in SUPPORTED:
        model_sizes = gens[name].SIZES
        for size in sizes:
            if size not in model_sizes:
                continue
            results.append({"size": size, **check_model(name, model_sizes[size], solve=solve)})
    return {
        "oracle": "standard_form_equivalence",
        "results": results,
        "all_equivalent": all(r["equivalent"] for r in results),
    }


def _print(report: Dict[str, Any]) -> None:
    print("=" * 84)
    print("EQUIVALENCE ORACLE - Pyomo standard form == array-native (up to row/col perm)")
    print("=" * 84)
    hdr = f"{'model':>18} {'size':>5} {'vars':>7} {'nnz':>8} {'rows==':>7} {'obj==':>7} {'EQUIV':>6}"
    print(hdr)
    print("-" * 84)
    for r in report["results"]:
        rows_ok = r["checks"]["constraint_rows_equal_up_to_permutation"]
        obj_ok = r.get("obj_equal")
        obj_s = "yes" if obj_ok else ("n/a" if obj_ok is None else "NO")
        print(f"{r['model']:>18} {r['size']:>5} {r['n_vars']:>7} {r['nnz_pyomo']:>8} "
              f"{str(rows_ok):>7} {obj_s:>7} {str(r['equivalent']):>6}")
        if not r["equivalent"]:
            failed = [k for k, v in r["checks"].items() if not v]
            print(f"{'':>32} FAILED checks: {failed}")
    print("-" * 84)
    print(f"all_equivalent: {report['all_equivalent']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="standard-form equivalence oracle")
    ap.add_argument("--sizes", type=str, default=None, help="comma list (default: xs,1e4)")
    ap.add_argument("--no-solve", action="store_true", help="skip the HiGHS obj cross-check")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args(argv)
    sizes = a.sizes.split(",") if a.sizes else None
    report = check_all(sizes=sizes, solve=not a.no_solve)
    _print(report)
    if a.out:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"# wrote {a.out}")
    # Non-zero exit on any mismatch so this is usable as a CI gate.
    return 0 if report["all_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
