"""Render the five original Table 5 rows, preserving their protocols.

The separate table5_precision_v3 diagnostic is not a like-for-like replacement.
"""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
DATA = HERE / 'paper_evidence_20261106/table5_original_contract/result.json'


def render(data):
    assert data['status'] == 'original_contract_restored'
    assert len(data['rows']) == 5
    rows = []
    for row in data['rows']:
        values = []
        for hardware in ['tpuv3', 'compact_npu']:
            cell = row['cells'][hardware]
            ratio = cell['reference_cycles'] / cell['objective_cycles']
            assert abs(ratio - cell['ratio']) < 1e-12
            values.append(f'${ratio:.3f}\\times$')
        rows.append(row['label'] + ' & ' + ' & '.join(values) + r' \\')
    return r'''\begin{table}[htbp]
\centering\small
\caption{Original fixed-hardware software-search comparisons for the five
selected workloads. Entries retain each stored experiment's workload,
implementation menu, memory policy and streaming-reference convention.
These are LP-objective ratios, not current decode co-design results.}
\label{tab:fixed_hw}
\begin{tabular}{@{}lrr@{}}
\toprule
Workload & Simulated TPU-v3 & Compact NPU \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
Ratios are $T_{\rm streaming}/T_{\rm joint\,LP}$ under the original
precision/timing proxies. TPU-v3 and compact NPU retain 900/64~GB/s and
65,536/32,768 MAC units at 1~GHz. The vision TPU and compact cells use
32/16~MiB SRAM. The original TPU LLM producer exposes an internal
1--256~MiB capacity selector; its stored 32-MiB hardware label alone does
not verify fixed SRAM. All original mapping choices remain.
Compact B7/13B/DeepSeek use stored iterative objectives (relative gap tags
0.020/0.125/0.019); the other cells retain their stored solver results. The original
graph and workspace approximations remain. These mixed precision
conventions are not relabeled as a verified INT8/BF16 pair.
\end{minipage}
\end{table}
'''


if __name__ == '__main__':
    (HERE / 'wo24_fixed_hardware.tex').write_text(render(json.loads(DATA.read_text())))
