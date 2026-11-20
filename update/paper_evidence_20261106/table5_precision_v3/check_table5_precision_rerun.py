"""Read saved LPs/witnesses and independently check Table 5 numerical results."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import scipy.sparse as sp


def check(cell):
    result = json.loads((cell / "result.json").read_text())
    for name, field in [("program.npz", "program_sha256"), ("witness.npz", "witness_sha256")]:
        assert hashlib.sha256((cell / name).read_bytes()).hexdigest() == result[field]
    with np.load(cell / "program.npz") as d:
        c, lv, uv, lc, uc = [d[k] for k in ["c", "lv", "uv", "lc", "uc"]]
        A = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    with np.load(cell / "witness.npz") as d:
        x, y, z = [d[k] for k in ["x", "y", "z"]]
    Ax = A @ x
    pv = np.maximum.reduce([lc - Ax, Ax - uc, np.zeros_like(Ax)])
    bv = np.maximum.reduce([lv - x, x - uv, np.zeros_like(x)])
    row_scale = 1 + abs(A) @ abs(x)
    finite_lower = np.isfinite(lc); finite_upper = np.isfinite(uc)
    row_scale = np.maximum(row_scale, np.where(finite_lower, abs(lc), 0))
    row_scale = np.maximum(row_scale, np.where(finite_upper, abs(uc), 0))
    stationarity = c - A.T @ y - z
    stat_scale = 1 + abs(c) + abs(A.T) @ abs(y) + abs(z)
    dual = 0.; invalid_support = 0.
    for v, lo, hi in [(y, lc, uc), (z, lv, uv)]:
        selected = np.where(v >= 0, lo, hi)
        finite = np.isfinite(selected)
        invalid_support = max(invalid_support, float(np.max(abs(v[~finite]), initial=0)))
        dual += float(v[finite] @ selected[finite])
    objective = float(c @ x)
    gap = abs(objective - dual) / max(1, abs(objective), abs(dual))
    values = dict(status=result["status"], objective=objective, dual_objective=dual,
                  relative_duality_gap=gap, maximum_row_violation=float(pv.max(initial=0)),
                  maximum_scaled_row_violation=float((pv / row_scale).max(initial=0)),
                  maximum_box_violation=float(bv.max(initial=0)),
                  maximum_scaled_stationarity=float((abs(stationarity) / stat_scale).max(initial=0)),
                  invalid_infinite_bound_dual_magnitude=invalid_support,
                  speedup=result["baseline_cycles"] / objective,
                  numerical_check=True, exact_rational_certificate=False)
    values["passed"] = bool(result["status"] == "Optimal" and gap < 1e-6
                            and values["maximum_scaled_row_violation"] < 1e-7
                            and values["maximum_box_violation"] < 1e-6
                            and values["maximum_scaled_stationarity"] < 1e-7
                            and invalid_support < 1e-7
                            and abs(objective - result["objective_cycles"]) < 1e-7 * max(1, abs(objective)))
    (cell / "independent_check.json").write_text(json.dumps(values, indent=2) + "\n")
    return values


if __name__ == "__main__":
    root = Path(sys.argv[1])
    checks = {p.name: check(p) for p in sorted((root / "runs").iterdir()) if (p / "result.json").exists()}
    (root / "independent_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({k: {f: v[f] for f in ["passed", "relative_duality_gap", "speedup"]} for k,v in checks.items()}, indent=2))
    assert checks and all(c["passed"] for c in checks.values())
