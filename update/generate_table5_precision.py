"""Render the unadopted restricted-width diagnostic to an explicit output.

This experiment changed the original comparison protocol. It must not
overwrite the paper's Table 5; use generate_table5_original.py there.
"""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
DATA = HERE / "paper_evidence_20261106/table5_precision_v3/result.json"
WORKLOADS = [
    ("efficientnet_b7", "EfficientNet-B7"),
    ("sd15_unet", "SD1.5 UNet"),
    ("llama1_13b", "Llama-1 13B"),
    ("llama3_70b", "Llama-3 70B"),
    ("deepseek_v3", "DeepSeek-V3"),
]
COLUMNS = [("tpuv3", 8), ("compact_npu", 8),
           ("tpuv3", 16), ("compact_npu", 16)]


def render(data):
    cells = {(c["workload"], c["hardware"], c["operand_bits"]): c
             for c in data["cells"]}
    expected = {(w, h, b) for w, _ in WORKLOADS for h, b in COLUMNS}
    assert set(cells) == expected and len(data["cells"]) == 20
    rows = []
    for workload, label in WORKLOADS:
        values = []
        for hardware, bits in COLUMNS:
            c = cells[(workload, hardware, bits)]
            assert c["status"] == "Optimal" and c["independent_check"]["passed"]
            assert c["lp_layout"]["C"] == 1
            ratio = c["baseline_cycles"] / c["objective_cycles"]
            assert abs(ratio - c["speedup"]) < 1e-12
            values.append(f"${ratio:.3f}\\times$")
        rows.append(label + " & " + " & ".join(values) + r" \\")
    return r"""\begin{table}[htbp]
\centering\small
\caption{Fresh fixed-hardware LP comparisons for five workloads at matched
operand widths. Each entry is $T_{\rm streaming}/T_{\rm joint\,LP}$ on the
same modeled hardware and at the same width; these are fractional-objective
ratios, not recovered execution or measured chip speedups.}
\label{tab:fixed_hw}
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \multicolumn{2}{c}{INT8 model} & \multicolumn{2}{c}{BF16 model} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Workload & TPU-v3 & Compact NPU & TPU-v3 & Compact NPU \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
The two models fix 65,536/32,768 MAC units, 900/64~GB/s and 32/16~MiB SRAM,
respectively, at 1~GHz. Both precisions retain the same MAC issue-rate proxy;
8/16-bit stored-operand widths scale traffic and tile capacity. These are
width sensitivities of the inherited timing/accumulator model, not native
TPU-v3 INT8 or precision-faithful BF16 measurements. Twenty fresh HiGHS
solves use archived tile footprints and one frozen mapping template,
analytically repriced at each resource budget, with checked primal/dual
witnesses. Historical matrix graphs, workspace approximations and the
streaming-cost convention remain. A ratio below $1\times$ means the LP's
serial-reload/tensor-floor cost exceeds that streaming reference. No new
full-decode or quantization-accuracy claim is made.
\end{minipage}
\end{table}
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Explicit diagnostic output; the paper Table 5 is protected.')
    args = parser.parse_args()
    if args.output.resolve() == (HERE / 'wo24_fixed_hardware.tex').resolve():
        parser.error('This diagnostic is not an authorized replacement for Table 5.')
    args.output.write_text(render(json.loads(DATA.read_text())))
