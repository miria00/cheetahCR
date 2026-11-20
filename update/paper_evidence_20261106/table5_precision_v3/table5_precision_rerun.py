"""Fresh fixed-resource, matched-width Table 5 LP experiments.

This is an analytical sensitivity of the historical Timeloop timing proxy.
It is not a native INT8 TPU execution or a precision-faithful BF16 simulator.
The deployed-topology column of the frozen tile library is repriced for both
resource budgets. Hardware-shape alternatives are excluded; no SRAM-sizing
selector or previous numerical solution is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import scipy.sparse as sp
import highspy

REPO = Path(__file__).resolve().parents[1]
for p in [REPO, REPO / "systems"]:
    sys.path.insert(0, str(p))

WORKLOADS = ["efficientnet_b7", "sd15_unet", "llama1_13b", "llama3_70b", "deepseek_v3"]
HARDWARE = {
    "tpuv3": dict(macs=65536, bandwidth_GBps=900., SRAM_MiB=32.),
    "compact_npu": dict(macs=32768, bandwidth_GBps=64., SRAM_MiB=16.),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inputs(root, workload, hardware, bits):
    """Preserve per-op Timeloop compute/stall decomposition at fixed rates.

    Cache inputs are the historical 16-bit proxy. Operand/activation byte
    coefficients scale with bit width; MAC issue costs do not. This does not
    add an INT8 packing-rate bonus or claim calibrated BF16 circuit timing.
    Full tensor sizes replace legacy element-count traffic lower bounds.
    """
    h = HARDWARE[hardware]
    graph = root / "inputs/graphs" / workload / "corgi_input.npz"
    cache = root / "inputs/tpuv3" / f"{workload}_pareto_for_tpuv3.npz"
    with np.load(graph) as d:
        tl = {k: d[k].copy() for k in ["T_max", "T_min", "t_I", "t_O", "t_W",
              "d_I", "d_O", "d_W", "B_i", "c_remat", "adjacency",
              "n_duplicates", "active_fraction", "is_dynamic"]}
        tl.update({"size_" + role: d["layer_sizes_" + name].copy()
                   for role, name in [("I", "input"), ("O", "output"), ("W", "weight")]})
        tl["order"] = d["execution_order"].copy()
        for k in ["tile_I", "tile_O", "tile_W", "peak_GB_tile_bytes"]:
            tl[k] = d[k].copy() if k in d else np.zeros_like(tl["T_min"])
    with np.load(cache, allow_pickle=True) as d:
        pc = {k: np.asarray(d[k]) for k in d.files if d[k].dtype != object}
    # Column zero is the deployed topology. The remaining historical columns
    # change physical PE/sys/VPU geometry and are not fixed-chip software
    # choices. Use only c0 in this control, with actual tile footprints.
    assert all(k in pc for k in ("tile_I_c", "tile_O_c", "tile_W_c"))
    source_configs = pc["T_min_c"].shape[1]
    for k in list(pc):
        if k.endswith("_c"):
            pc[k] = pc[k][:, :1].copy()
    pc["pe_macs_per_config"] = pc["pe_macs_per_config"][:1].copy()
    pc["configs_per_idx"] = pc["configs_per_idx"][:1].copy()
    q = bits / 16.
    compute_scale = np.asarray(pc["pe_macs_per_config"], float)[None, :] / h["macs"]
    traffic_scale = q * 900. / h["bandwidth_GBps"]
    p = {k: v.copy() for k, v in pc.items()}
    p["T_min_c"] = pc["T_min_c"] * compute_scale
    # The later, tile-qualified library already obeys this consistency rule.
    raw_memory_gap = pc["T_max_c"] - pc["T_min_c"]
    assert np.all(raw_memory_gap >= 0), "invalid archived timing decomposition"
    p["T_max_c"] = p["T_min_c"] + raw_memory_gap * traffic_scale
    tl["_cache_consistency"] = dict(
        rule="no timing repair; source T_max>=T_min required",
        source_configurations=source_configs, selected_column=0,
        selected_source_topology=pc["configs_per_idx"][0].tolist(),
        affected_coordinates=int(np.count_nonzero(raw_memory_gap < 0)),
        affected_reference_c0=int(np.count_nonzero(raw_memory_gap[:, 0] < 0)),
        original_reference_Tmax_sum=float(pc["T_max_c"][:, 0].sum()),
        clamped_reference_Tmax_sum=float(np.maximum(pc["T_max_c"][:, 0], pc["T_min_c"][:, 0]).sum()))
    for k in ["t_I_c", "t_O_c", "t_W_c"]:
        p[k] = pc[k] * traffic_scale
    for k in ["ws_c", "tile_I_c", "tile_O_c", "tile_W_c", "peak_GB_tile_bytes_c"]:
        if k in p:
            p[k] *= q
    for k in ["size_I", "size_O", "size_W", "tile_I", "tile_O", "tile_W", "peak_GB_tile_bytes", "B_i"]:
        tl[k] = np.asarray(tl[k], float) * q
    for role in "IOW":
        tl["d_" + role] = tl["size_" + role].copy()
    # Preserve the source's rematerialization penalty, repriced to this chip.
    tl["c_remat"] = np.asarray(tl["c_remat"], float) * 256. / h["macs"]
    tl["CGM"] = h["SRAM_MiB"] * 2**20
    tl["DRAM_BW"] = h["bandwidth_GBps"]
    if workload == "sd15_unet":
        with np.load(graph) as d:
            tl["fanout_edges"] = np.asarray(d["fanout_edges"], dtype=np.int64)
        ext = sorted({int(t) for s, t in tl["fanout_edges"] if s < 0})
        tl["source_map"] = {"timestep": [o for o in ext if o == 0],
                            "latent": [o for o in ext if o == 2],
                            "context": [o for o in ext if o not in (0, 2)]}
    baseline = float(p["T_max_c"][:, 0].sum())
    assert np.all(p["T_max_c"] >= p["T_min_c"])
    assert np.all(np.isfinite(p["T_min_c"]))
    return tl, p, baseline, {str(graph): sha(graph), str(cache): sha(cache)}


def run(root, workload, hardware, bits, limit):
    start = time.monotonic()
    cell = root / "runs" / f"{workload}__{hardware}__{bits}bit"
    cell.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "source"))
    from build_joint_lp_v21 import build_joint_lp_v21
    from build_joint_lp_v21_dag import build_joint_lp_v21_dag
    tl, p, baseline, bound_inputs = inputs(root, workload, hardware, bits)
    h = HARDWARE[hardware]
    build = build_joint_lp_v21_dag if workload == "sd15_unet" else build_joint_lp_v21
    prob = build(tl, p, bw_gbps=h["bandwidth_GBps"],
                 cgm_bytes=h["SRAM_MiB"] * 2**20,
                 traffic_bw_ref_gbps=h["bandwidth_GBps"])
    c, lv, uv, lc, uc = [np.asarray(x, dtype=np.float64) for x in
                         (prob.c, prob.l_v, prob.u_v, prob.l_c, prob.u_c)]
    idx = np.asarray(prob.A.indices)
    A = sp.coo_matrix((np.asarray(prob.A.data, float), (idx[:, 0], idx[:, 1])),
                      shape=prob.A.shape).tocsc()
    del prob, idx
    A.sum_duplicates(); A.sort_indices()
    nrow, ncol = A.shape
    np.savez_compressed(cell / "program.npz", c=c, lv=lv, uv=uv, lc=lc, uc=uc,
                        data=A.data, indices=A.indices, indptr=A.indptr,
                        shape=np.array(A.shape))
    solver = highspy.Highs()
    for name, val in {"output_flag": True, "threads": 2, "time_limit": float(limit),
                      "primal_feasibility_tolerance": 1e-7,
                      "dual_feasibility_tolerance": 1e-7,
                      "log_file": str(cell / "highs.log")}.items():
        assert solver.setOptionValue(name, val) == highspy.HighsStatus.kOk
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = ncol, nrow
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lv, uv
    lp.row_lower_, lp.row_upper_ = lc, uc
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    assert solver.passModel(lp) == highspy.HighsStatus.kOk
    solve_start = time.monotonic()
    solver.run()
    status = solver.modelStatusToString(solver.getModelStatus())
    sol = solver.getSolution()
    x, y, z = [np.asarray(a, float) for a in (sol.col_value, sol.row_dual, sol.col_dual)]
    np.savez_compressed(cell / "witness.npz", x=x, y=y, z=z)
    result = dict(workload=workload, hardware=hardware, operand_bits=bits,
                  hardware_resources=h, baseline_cycles=baseline,
                  objective_cycles=float(c @ x), speedup=float(baseline / (c @ x)),
                  status=status, solver="HiGHS " + solver.version(),
                  numerical_optimizer_calls=1, variables=ncol, rows=nrow, nonzeros=A.nnz,
                  solve_seconds=time.monotonic() - solve_start,
                  total_seconds=time.monotonic() - start,
                  source_inputs=bound_inputs, program_sha256=sha(cell / "program.npz"),
                  witness_sha256=sha(cell / "witness.npz"),
                  driver_sha256=sha(__file__),
                  cache_consistency=tl["_cache_consistency"],
                  lp_layout=tl["_v21_layout"],
                  full_decode_execution=False, integral_execution=False,
                  precision_faithful_native_TPU=False,
                  scope="Historical matrix-graph fixed-hardware LP, deployed c0 topology only; matched operand-width timing proxy; same rates for both precisions; actual archived tile footprints; no new Timeloop search.")
    write(cell / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workload", choices=WORKLOADS, required=True)
    ap.add_argument("--hardware", choices=HARDWARE, required=True)
    ap.add_argument("--bits", type=int, choices=[8, 16], required=True)
    ap.add_argument("--time-limit", type=float, default=600)
    a = ap.parse_args()
    run(a.root, a.workload, a.hardware, a.bits, a.time_limit)
