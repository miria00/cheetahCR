"""Independent input/width/fixed-topology audit for the Table 5 rerun.

Uses archived arrays and the saved sparse program, without importing the
producer or either LP builder. This checks the declared analytical contract;
it does not qualify floating-point arithmetic or a physical target chip.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import scipy.sparse as sp


def audit(root):
    rows = []
    for cell in sorted((root / "runs").iterdir()):
        result = json.loads((cell / "result.json").read_text())
        assert result["status"] == "Optimal"
        w, h, bits = result["workload"], result["hardware"], result["operand_bits"]
        hardware = {"tpuv3": (65536, 900., 32.),
                    "compact_npu": (32768, 64., 16.)}[h]
        macs, bandwidth, capacity = hardware
        assert result["hardware_resources"] == dict(
            macs=macs, bandwidth_GBps=bandwidth, SRAM_MiB=capacity)
        source = root / "inputs/tpuv3" / f"{w}_pareto_for_tpuv3.npz"
        graph = root / "inputs/graphs" / w / "corgi_input.npz"
        for path in (source, graph):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == result["source_inputs"][str(path)]
        with np.load(source) as d:
            minimum, maximum = d["T_min_c"][:, 0], d["T_max_c"][:, 0]
            expected_compute = minimum * float(d["pe_macs_per_config"][0]) / macs
            expected_stream = expected_compute + (maximum - minimum) * (bits / 16.) * 900. / bandwidth
            workspace = sum(d[k][:, 0] for k in ("tile_I_c", "tile_O_c", "tile_W_c")) * (bits / 16.)
        assert np.all(maximum >= minimum)
        assert np.max(workspace) <= capacity * 2**20
        assert result["lp_layout"]["C"] == 1
        L = result["lp_layout"]["L"]
        assert L == len(minimum)
        with np.load(graph) as d:
            assert L == len(d["T_min"])
        assert abs(expected_stream.sum() - result["baseline_cycles"]) <= 1e-12 * expected_stream.sum()
        with np.load(cell / "program.npz") as d:
            matrix = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
            # First L rows choose the only config; the next L rows enforce
            # its compute cost. Verify the actual saved coefficient matrix.
            compute = matrix[L:2*L, L:2*L]
            assert compute.nnz == L
            assert np.array_equal(compute.diagonal(), expected_compute)
            assert np.array_equal(matrix[:L, L:2*L].diagonal(), np.ones(L))
        with np.load(cell / "witness.npz") as d:
            assert np.allclose(d["x"][L:2*L], 1., rtol=0, atol=1e-9)
        rows.append(dict(cell=cell.name, passed=True, source_ops=L,
                         selected_topology_column=0, operand_bits=bits,
                         maximum_tile_workspace_bytes=float(workspace.max()),
                         baseline_cycles=float(expected_stream.sum()),
                         native_precision_validation=False))
    assert len(rows) == 20
    payload = dict(passed=True, cells=rows,
                   scope="Checks fixed topology and budgets, same-width costs, source identities, and actual saved LP compute rows; not a native dtype or implementation certificate.")
    (root / "contract_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed": True, "cells": len(rows)}))


if __name__ == "__main__":
    audit(Path(sys.argv[1]).resolve())
