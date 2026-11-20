"""Map saved Figure 3 increments to Table 10; unmerge Llama's yellow group.

No new gains are estimated. The extra slice is the previously merged base
resident datapath increment, not an isolated weight/KV ablation.
"""
from pathlib import Path
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json

P = Path(__file__).resolve().parents[1]
OUT = P/'update/paper_evidence_20261106/figure3_pies_v7'
manifest = json.loads((OUT/'input_manifest.json').read_text())
for name, identity in manifest['sources'].items():
    assert hashlib.sha256((OUT/'inputs'/name).read_bytes()).hexdigest() == identity['sha256']
parent = json.loads((OUT/'inputs/parent_controls.json').read_text())
assert parent['passed']
panels = deepcopy(parent['panels'])

for panel in panels:
    if panel['workload'] == 'deepseek_v3':
        groups = [
            ('Placement / residency†', 0,
             ['Activation placement', 'Weight residency', 'MoE residency / offload',
              'Implementation / tiling', 'Hardware resources']),
            ('Hardware resources', 1, ['Hardware resources']),
            ('Fusion / storage', 2, ['Fusion / storage', 'Activation placement']),
            ('Scheduling: prefetch', 3, ['Scheduling / concurrency']),
            ('Scheduling: overlap', 4, ['Scheduling / concurrency']),
        ]
    else:
        groups = [
            ('Implementation / tiling', 1, ['Implementation / tiling']),
            ('HW resources: arrays', 2, ['Hardware resources']),
            ('HW resources: on-chip BW', 4, ['Hardware resources']),
            ('Weight residency + KV†', 0,
             ['Weight residency', 'KV retention', 'Hardware resources']),
            ('HW resources: DRAM BW', 3, ['Hardware resources']),
        ]
    panel['slices'] = []
    for label, i, families in groups:
        s = panel['stages'][i]
        panel['slices'].append(dict(label=label, source_stages=[i+1],
            table10_families=families, added_speedup_exact=s['added_speedup_exact'],
            share_exact=s['share_exact'], scope=s['scope']))
    assert len(panel['slices']) == 5
    assert sorted(i for x in panel['slices'] for i in x['source_stages']) == [1, 2, 3, 4, 5]
    assert sum(F(x['share_exact']) for x in panel['slices']) == 1
    assert sum(F(x['added_speedup_exact']) for x in panel['slices']) == F(panel['final_speedup_exact'])-1
    old = next(p for p in parent['panels'] if p['workload'] == panel['workload'])
    if panel['workload'] != 'deepseek_v3':
        merged = next(x for x in old['slices'] if x['label'] == 'Hardware + DRAM')
        assert sum(F(x['share_exact']) for x in panel['slices'][3:]) == F(merged['share_exact'])
    else:
        assert [x['share_exact'] for x in panel['slices']] == [x['share_exact'] for x in old['slices']]
    assert panel['stages'] == old['stages']

for name, sha in parent['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest() == sha
result = dict(passed=True, panels=panels, optimizer_calls=0,
    unchanged_figure2=parent['unchanged_figure2'], method=parent['method'],
    order_scope='Underlying cumulative comparisons unchanged. Legend is grouped, not chronological. '
                'The earlier Llama Hardware + DRAM display group is separated into its original '
                'base resident-datapath and DRAM-bandwidth increments.',
    terminology_source='Table 10, tab:current_features, frozen in inputs/table10_features.tex.',
    footnote='Dagger marks joint base groups including inherited hardware and mappings. '
             'Placement/residency and weight/KV labels do not claim isolated software gains '
             'or that all inherited policies are free selectors in the latest master.',
    source_scope='Saved checked cumulative controls, relabeled and regrouped with exact arithmetic. '
                 'No new optimizer, source-admission or physical-validation calls. '
                 'No shares chosen to balance the appearance of the pie.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'grouped_controls.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({p['workload']: {s['label']: 100*float(F(s['share_exact'])) for s in p['slices']}
                  for p in panels}, indent=2))
