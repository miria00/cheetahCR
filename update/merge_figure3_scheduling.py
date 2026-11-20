"""Combine the two DeepSeek cumulative scheduling increments exactly."""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
import hashlib
import json

P = Path(__file__).resolve().parents[1]
OUT = P/'update/paper_evidence_20261106/figure3_pies_v9'
raw = (OUT/'inputs/parent_controls.json').read_bytes()
manifest = json.loads((OUT/'input_manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest() == manifest['parent_sha256']
parent = json.loads(raw)
assert parent['passed']
result = deepcopy(parent)
ds = next(p for p in result['panels'] if p['workload'] == 'deepseek_v3')
a, b = ds['slices'][3:]
assert [a['label'], b['label']] == ['Scheduling: prefetch', 'Scheduling: overlap']
merged = dict(label='Scheduling / concurrency',
    share_exact=str(F(a['share_exact'])+F(b['share_exact'])),
    added_speedup_exact=str(F(a['added_speedup_exact'])+F(b['added_speedup_exact'])),
    source_stages=a['source_stages']+b['source_stages'],
    table10_families=['Scheduling / concurrency'],
    scope='Sum of sequential cross-wave prefetch and compute/readout-overlap increments. '
          'This does not add overlapping feature-off ablations; original paid costs are unchanged.')
ds['slices'] = ds['slices'][:3]+[merged]
for now, old in zip(result['panels'], parent['panels']):
    assert len(now['slices']) == 4
    assert sum(F(s['share_exact']) for s in now['slices']) == 1
    assert sum(F(s['added_speedup_exact']) for s in now['slices']) == F(now['final_speedup_exact'])-1
    for key in ['reference_ns', 'final_ns', 'final_speedup_exact', 'stages']:
        assert now[key] == old[key]
    if now['workload'] != 'deepseek_v3':
        assert now == old
result['order_scope'] = ('Four display groups per workload. DeepSeek scheduling/concurrency sums '
    'its original sequential prefetch and compute/readout-overlap increments. Llama groups, '
    'underlying controls, comparison order, final speedups and reference remain unchanged.')
result['parent_sha256'] = manifest['parent_sha256']
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for name, expected in result['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest() == expected
(OUT/'grouped_controls.json').write_text(json.dumps(result, indent=2)+'\n')
print('DeepSeek scheduling/concurrency exact percentage:', 100*float(F(merged['share_exact'])))
