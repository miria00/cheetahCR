"""Combine tiling/remapping with on-chip delivery, keeping hardware terms explicit."""
from pathlib import Path
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json

P = Path(__file__).resolve().parents[1]
OUT = P/'update/paper_evidence_20261106/figure3_pies_v8'
manifest = json.loads((OUT/'input_manifest.json').read_text())
raw = (OUT/'inputs/parent_controls.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == manifest['sources']['parent_controls.json']['sha256']
parent = json.loads(raw)
check_raw = (OUT/'remapping_check.json').read_bytes()
check = json.loads(check_raw)
assert check['passed'] and parent['passed']
panels = deepcopy(parent['panels'])
for p in panels:
    if p['workload'] == 'deepseek_v3':
        continue
    audit = check['workloads'][p['workload']]
    gain = F(p['final_speedup_exact'])-1
    parts = {r['resource']: F(r['hardware_added_speedup']) for r in audit['rows']}
    mapping = F(p['stages'][1]['added_speedup_exact']) + sum(F(r['remap_added_speedup']) for r in audit['rows'])
    groups = [
        ('Tiling + on-chip bandwidth', mapping+parts['local_GBps'],
         ['Implementation / tiling', 'Hardware resources']),
        ('Compute parallelism', parts['tiles'], ['Hardware resources']),
        ('Memory capacity + residency†', F(p['stages'][0]['added_speedup_exact']),
         ['Weight residency', 'KV retention', 'Hardware resources']),
        ('Off-chip DRAM bandwidth', parts['bandwidth'], ['Hardware resources']),
    ]
    p['slices'] = [dict(label=label, added_speedup_exact=str(delta),
                       share_exact=str(delta/gain), table10_families=families)
                   for label, delta, families in groups]
    p['mapping_diagnostic'] = audit
    p['detailed_attribution'] = dict(
        mapping_added_speedup=str(mapping),
        hardware_added_speedup={k: str(v) for k, v in parts.items()},
        base_added_speedup=p['stages'][0]['added_speedup_exact'])
    assert sum(F(x['share_exact']) for x in p['slices']) == 1
    assert sum(F(x['added_speedup_exact']) for x in p['slices']) == gain
    p['SRAM_capacity_held_fixed_in_resource_controls'] = True

for name, expected in parent['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest() == expected
result = dict(passed=True, panels=panels, optimizer_calls=0,
    unchanged_figure2=parent['unchanged_figure2'],
    method='Partition cumulative increments in S=T_B2/T, starting at S=1. For each Llama resource '
           'increase, first retain the previous mapping, then allow compatible remapping. Group '
           'the original menu extension and all remapping gains with the on-chip bandwidth gain. '
           'Remaining groups are compute parallelism, base memory/residency and DRAM bandwidth.',
    order_scope='Original resource order unchanged; intermediate fixed-mapping witnesses separate '
                'resource-rate and remapping effects. Display groups are not chronological. '
                'Llama has four groups; DeepSeek retains five. There is no arbitrary percentage split.',
    footnote='Memory capacity/residency includes inherited base hardware and mapping. It is not '
             'an isolated SRAM-capacity gain. SRAM capacity and bank/port reservations are fixed '
             'through the Llama tiling and resource comparisons; on-chip bandwidth is local_GBps. '
             'DRAM bandwidth refers to external DRAM, not FPGA BRAM.',
    remapping_check_sha256=hashlib.sha256(check_raw).hexdigest(),
    new_source_admission=False, new_physical_validation=False,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'grouped_controls.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({p['workload']: {s['label']: 100*float(F(s['share_exact'])) for s in p['slices']}
                  for p in panels}, indent=2))
