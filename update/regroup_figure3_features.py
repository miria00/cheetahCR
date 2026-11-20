"""Regroup checked increments; extend DeepSeek's hardware-control interval.

No percentage is assigned by appearance. Llama groups sum existing increments.
DeepSeek replaces the start of its sequence with the authenticated same-N16
control on earlier hardware, then retains every later cumulative endpoint.
"""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
import hashlib
import json

P=Path(__file__).resolve().parents[1]
OUT=P/'update/paper_evidence_20261106/figure3_pies_v6'
manifest=json.loads((OUT/'input_manifest.json').read_text())


def read(name):
    raw=(OUT/'inputs'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['sources'][name]['sha256']
    return json.loads(raw)


parent=read('parent_controls.json')
metrics=read('deepseek_metrics.json')
cert=read('deepseek_finite.json')
review=read('deepseek_metrics_review.json')
composition=read('wo23_composition.json')
for key,name in [('deepseek_metrics','deepseek_metrics.json'),
                 ('deepseek_finite','deepseek_finite.json'),
                 ('deepseek_metrics_review','deepseek_metrics_review.json')]:
    assert composition['input_identities'][key]['sha256']==manifest['sources'][name]['sha256']
assert parent['passed'] and metrics['passed'] and cert['passed'] and review['passed']
panels=deepcopy(parent['panels'])
ds=panels[0];assert ds['workload']=='deepseek_v3'
old_first=F(ds['stages'][0]['time_ns'])
assert old_first==F(metrics['HW_SW_cells_ns']['H1_S1'])
earlier=F(metrics['HW_SW_cells_ns']['H0_S1'])
row=next(x for x in cert['branch_rows'] if x['hardware']=='old_bank_control_N16')
assert earlier==F(row['exact_minimum_ns'])
assert F(row['maximum_capacity_violation'])==0
ds['stages'][0].update(label='Memory residency + mapping',time_ns=str(earlier),
    scope='Earlier P020 hardware with the qualified N16 mapping family and resident memory policy. '
          'This base group includes the inherited datapath; it is not an isolated memory-only effect.')
ds['stages'][1].update(scope='Combined WO23 and WO24 hardware refinements under the pre-fusion N16 '
    'family: earlier-hardware N16 control to selected WO24 hardware with producer fusion, prefetch '
    'and the second context disabled. Compatible mappings may be reselected. The intervening '
    'WO23 point is retained explicitly; no software-menu expansion is credited as hardware.',
    intermediate_time_ns=str(old_first))
ds['stages'][4]['label']='Compute/readout overlap'
ds['stages'][4]['scope']+=' The simple label denotes next-tile compute overlapping prior-result readout using the second FP32 context; bulk conversion and output stores remain charged.'
ref=F(ds['reference_ns']);previous=ref;speed=F(1);gain=F(ds['final_speedup_exact'])-1
for i,z in enumerate(ds['stages']):
    t=F(z['time_ns']);now=ref/t;assert 0<t<previous
    z.update(order=i+1,signed_time_saved_ns=str(previous-t),speedup_exact=str(now),speedup=float(now),
             added_speedup_exact=str(now-speed),share_exact=str((now-speed)/gain))
    previous,speed=t,now
assert previous==F(ds['final_ns'])
# The new first two groups exactly replace the prior first two groups.
assert sum(F(z['added_speedup_exact']) for z in ds['stages'][:2])==sum(
    F(z['added_speedup_exact']) for z in parent['panels'][0]['stages'][:2])

for p in panels:
    gain=F(p['final_speedup_exact'])-1
    if p['workload']=='deepseek_v3':
        groups=[(z['label'],[i]) for i,z in enumerate(p['stages'])]
    else:
        groups=[('Tiling extension',[1]),('Array parallelism',[2]),
                ('On-chip bandwidth',[4]),('Hardware + DRAM',[0,3])]
    slices=[];covered=[]
    for label,indices in groups:
        delta=sum(F(p['stages'][i]['added_speedup_exact']) for i in indices)
        slices.append(dict(label=label,source_stages=[i+1 for i in indices],
                           added_speedup_exact=str(delta),share_exact=str(delta/gain)))
        covered.extend(indices)
    assert sorted(covered)==list(range(5))
    assert sum(F(x['share_exact']) for x in slices)==1
    assert sum(F(x['added_speedup_exact']) for x in slices)==gain
    p['slices']=slices

for name,sha in manifest['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha
result=dict(passed=True,panels=panels,optimizer_calls=0,
    unchanged_figure2=manifest['unchanged_figure2'],
    method='Sum exact cumulative speedup increments within each displayed feature group; '
           'divide by final speedup minus one. Every underlying increment appears exactly once.',
    order_scope='Underlying stage order remains explicit in panels.stages. The Llama legend is '
                'grouped, not a new chronological sequence: source stages 1 and 4 are merged '
                'as Hardware + DRAM and placed last. The original conditional attribution is unchanged.',
    deepseek_change=dict(earlier_hardware_N16_time_ns=str(earlier),
                         intermediate_WO23_time_ns=str(old_first),
                         refined_WO24_pre_fusion_time_ns=ds['stages'][1]['time_ns'],
                         scope=metrics['controls_scope'],
                         pure_memory_isolation=False),
    label_mapping={'Compute/readout overlap':'Second FP32 accumulator context; overlap prior-result '
                   'readout and next-tile computation, with conversion/stores still paid.',
                   'On-chip bandwidth':'The selected hardware local_GBps service capacity used '
                   'in local-traffic pricing. This is an LP hardware-resource choice, not a new '
                   'software scheduler or a renamed DRAM link.',
                   'Hardware + DRAM':'Sum of the base datapath/weight-KV-residency group and '
                   'the DRAM-bandwidth increment; the merged category is not hardware-only.'},
    source_scope='Authenticated saved exact controls and independent review, with fresh rational '
                 'composition only. No original source-admission algorithms, numerical optimizer '
                 'or physical checks rerun. Performance endpoints remain unchanged.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'grouped_controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({p['workload']:{x['label']:100*float(F(x['share_exact'])) for x in p['slices']} for p in panels},indent=2))
