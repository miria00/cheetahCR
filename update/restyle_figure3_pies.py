"""Remove annotation marks and put DeepSeek last; all numeric data stay fixed."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json

P = Path(__file__).resolve().parents[1]
OUT = P/'update/paper_evidence_20261106/figure3_pies_v10'
raw = (OUT/'inputs/parent_controls.json').read_bytes()
manifest = json.loads((OUT/'input_manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest() == manifest['parent_sha256']
parent = json.loads(raw)
result = deepcopy(parent)
order = ['llama1_13b', 'llama1_65b', 'llama3_70b', 'deepseek_v3']
result['panels'] = sorted(result['panels'], key=lambda p: order.index(p['workload']))
for panel in result['panels']:
    old = next(p for p in parent['panels'] if p['workload'] == panel['workload'])
    for new_slice, old_slice in zip(panel['slices'], old['slices']):
        new_slice['label'] = new_slice['label'].replace('†', '')
        assert {k: v for k, v in new_slice.items() if k != 'label'} == {
            k: v for k, v in old_slice.items() if k != 'label'}
    for key in ['reference_ns', 'final_ns', 'final_speedup_exact', 'stages']:
        assert panel[key] == old[key]
result['presentation'] = dict(panel_order=order, annotation_markers=False,
    subtitle='Decode · {speedup:.3f}×', SRAM_footer_removed=True,
    scope='Presentation only. Numeric shares, endpoints, source scope and methodological notes unchanged.')
result['parent_sha256'] = manifest['parent_sha256']
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for name, expected in result['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest() == expected
(OUT/'grouped_controls.json').write_text(json.dumps(result, indent=2)+'\n')
print('Presentation updated:', order)
