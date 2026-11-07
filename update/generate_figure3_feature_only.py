"""Feature-only visual allocations of the complete Baseline-2 time reduction.

Use a telescoping sequence of checked controls. The initial joint feature
bundle is a reference-to-control contrast, NOT an isolated feature ablation.
No remaining-runtime slice, negative clipping or sum of overlapping penalties.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
E = P / 'update/paper_evidence_20261106/figure3_latest12'
OLD = P / 'update/paper_evidence_20261106/figure3_pies_v1'
OUT = P / 'update/paper_evidence_20261106/figure3_pies_v2'
data = json.loads((E/'feature_groups_values.json').read_text())
bindings = {}


def read(path, expected=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, str(path)
    bindings[str(Path(path).relative_to(P))] = digest
    return json.loads(raw)


for original, info in data['sources'].items():
    read(E/info['file'], info['sha256'])


def frozen(original):
    v = data['sources'][original]
    return read(E/v['file'], v['sha256'])


def current(key):
    return frozen(str(P.parent/'Results/Blueprints/WO24/records'/(key+'.json')))


extra = read(OUT/'input_manifest.json')
for file, sha in extra['unchanged_figure2'].items():
    assert hashlib.sha256((P/file).read_bytes()).hexdigest() == sha, 'Figure 2 must remain unchanged'


def extra_input(key):
    v = extra['inputs'][key]
    return read(OUT/v['path'], v['sha256'])


replays = {case:read(OLD/'replays'/folder/'result.json') for case,folder in
           [('deepseek','deepseek'),('unet','unet_v2'),('weights','llama_weights_v2'),('b3','b3')]}
assert all(r['passed'] and r['optimizer_calls'] == 0 for r in replays.values())
panels = []


def add(key, label, phase, endpoints, groups, scopes):
    r = current(key)
    ts = [F(t) for t in endpoints]
    assert len(ts) == len(groups)+1 and 1 <= len(groups) <= 5
    if 'latency_ns' in r:
        assert ts[-1] == F(r['latency_ns']) and ts[0] == F(r['reference_ns'])
    ratio = ts[0]/ts[-1]
    assert abs(float(ratio)-r['reference_ratio']) < 1e-10
    saved = ts[0]-ts[-1]
    assert saved > 0
    weights = []
    for i, (group, scope) in enumerate(zip(groups, scopes)):
        delta = ts[i]-ts[i+1]
        assert delta > 0, 'A negative effect cannot be hidden in a positive pie.'
        weights.append(dict(label=group, before_ns=str(ts[i]), after_ns=str(ts[i+1]),
                            saved_ns=str(delta), share_exact=str(delta/saved),
                            speedup_gain_allocation_exact=str(delta/ts[-1]), scope=scope))
    assert sum((F(w['saved_ns']) for w in weights), F()) == saved
    assert sum((F(w['share_exact']) for w in weights), F()) == 1
    assert sum((F(w['speedup_gain_allocation_exact']) for w in weights), F()) == ratio-1
    # Largest remainder rounding makes the printed percentages sum to 100.00.
    exact_units = [F(w['share_exact'])*10000 for w in weights]
    units = [v.numerator//v.denominator for v in exact_units]
    rank = sorted(range(len(units)), key=lambda i:exact_units[i]-units[i], reverse=True)
    for i in rank[:10000-sum(units)]:
        units[i] += 1
    assert sum(units) == 10000
    for w,u in zip(weights,units):
        w['display_percent'] = f'{u//100}.{u%100:02d}'
    panels.append(dict(workload=key,label=label,phase=phase,reference_ns=str(ts[0]),
                       final_ns=str(ts[-1]), reference_ratio_exact=str(ratio),
                       total_saved_ns=str(saved), stages_ns=list(map(str,ts)), weights=weights))


ds = current('deepseek_v3')
ctrl = extra_input('deepseek_controls')
assert ctrl['passed'] and ctrl['all_controls_source_replayed']
attn = {k:F(v['attention_ns']) for k,v in ctrl['controls'].items()}
fixed = F(ds['latency_ns'])-attn['producer1_delivery1']
dataset = frozen('/tmp/cheetah_wo24_20261106_v1/authoritative_dataset_v1/result.json')
base = fixed+attn['producer0_delivery0']
assert base == F(dataset['ablations']['deepseek']['hardware_software_cells_ns']['new_hardware_old_software'])
add('deepseek_v3','DeepSeek-V3','Decode',
    [ds['reference_ns'],base,fixed+attn['producer1_delivery0'],ds['latency_ns']],
    ['Hardware, mapping\n& residency','Producer-layout fusion','Prefetch / two contexts'],
    ['Combined reference-to-control contrast: current chip, base mappings, resident arenas and inherited execution policies. These effects are not separately isolated or all newly free in the latest LP.',
     'Original rounded producer layout retained; addition precedes delivery extensions on the same hardware.',
     'Paid cross-wave prefetch and two FP32 contexts, conditional on producer fusion; no separate context credit.'])

ll = current('llama1_65b')
seq = read(OUT/'llama65_sequence.json')
assert seq['passed'] and seq['optimizer_calls'] == 0
assert F(seq['reference_ns']) == F(ll['reference_ns'])
add('llama1_65b','Llama-1 65B','Decode',
    [ll['reference_ns'],seq['old14_weight_off_ns'],seq['new25_weight_off_ns'],seq['new25_weight_on_ns']],
    ['Hardware, base maps\n& KV retention','Tiling refinement','Weight residency'],
    ['Combined reference-to-control contrast: current chip, original 14-map menu, fixed activation/KV placement and retained KV. Not an isolated hardware or KV ablation.',
     '14-to-25 mapping menu with all eligible weight retention disabled; paid import constants are independent of mapping.',
     'Enable eligible weight retention, preserving all first loads, kernel reads, initial KV and tail writes.'])

un = current('unet')
u = replays['unet']['details']['controls']
add('unet','SD1.5 UNet','Denoiser call',
    [un['reference_ns'],u['(0, 0)'],u['(0, 1)'],u['(1, 1)']],
    ['Hardware & base maps','Cast-storage fusion','Operand buffering'],
    ['Combined reference-to-control contrast: custom hardware and selected matrix recipes with cast fusion and buffering disabled; inherited owners and original boundaries remain.',
     'Cast-storage fusion enabled first, with buffering disabled.',
     'Buffering enabled after cast fusion; this order allocates their overlap once.'])

b3 = current('efficientnet_b3')
comparison = extra_input('b3_reference')
assert comparison['passed'] and not comparison['baseline_strengthened']
row = next(x for x in comparison['rows'] if x['workload']=='efficientnet_b3')
b3controls = frozen(b3['historical_record']['source_receipt'])
assert b3controls['passed']
b3off = F(str(b3controls['results']['materialized']['scheduled_ms']))*10**6
b3on = F(str(b3controls['results']['whole_owner']['scheduled_ms']))*10**6
assert abs(float(b3on/10**6)-replays['b3']['details']['totals_ms']['whole_owner']) < 1e-10
add('efficientnet_b3','EfficientNet-B3','Forward',
    [F(str(row['reference_forward_ms']))*10**6,b3off,b3on],
    ['Hardware & base maps','Tensor retention'],
    ['Combined reference-to-materialized-control contrast: inherited custom hardware and mapping modes. No isolated hardware/mapping split is available.',
     'Whole tensors retained in SRAM until their final consumers, on the same chip and mapping.'])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig = plt.figure(figsize=(10.8,3.95),facecolor='white')
for i,p in enumerate(panels):
    x = .012+i*.25
    fig.text(x+.115,.97,p['label'],ha='center',va='top',fontsize=12,fontweight='bold',color='#172536')
    fig.text(x+.115,.907,f"{p['phase']} · {float(F(p['reference_ratio_exact'])):.3f}× vs B2",
             ha='center',va='top',fontsize=10.1,color='#3d4c5c')
    ax = fig.add_axes([x,.325,.23,.55])
    values = [float(F(w['share_exact'])) for w in p['weights']]
    cs = ['#4075ae']+[('#8964ac' if 'retention' in w['label'].lower() or 'residency' in w['label'].lower()
                      else '#52abc9' if 'Tiling' in w['label']
                      else '#e5a745' if 'buffer' in w['label'].lower() or 'Prefetch' in w['label']
                      else '#55a67c') for w in p['weights'][1:]]
    wedges, _ = ax.pie(values,colors=cs,startangle=90,counterclock=False,
                       wedgeprops={'edgecolor':'white','linewidth':.8},normalize=True)
    for wedge,w in zip(wedges,p['weights']):
        if F(w['share_exact']) >= F(3,100):
            angle=math.radians((wedge.theta1+wedge.theta2)/2)
            ax.text(.62*math.cos(angle),.62*math.sin(angle),w['display_percent']+'%',
                    ha='center',va='center',fontsize=10.2,color='white',fontweight='bold')
    ax.set_aspect('equal')
    for j,(w,color) in enumerate(zip(p['weights'],cs)):
        y = .278-j*.078
        fig.add_artist(plt.Rectangle((x+.002,y-.008),.009,.020,transform=fig.transFigure,
                                     facecolor=color,edgecolor='none'))
        fig.text(x+.018,y,w['label'],ha='left',va='center',fontsize=8.6,color='#263442',linespacing=1.15)
        fig.text(x+.232,y,w['display_percent']+'%',ha='right',va='center',fontsize=8.6,color='#263442')
fig.text(.5,.017,'Visual estimates · 100% = modeled time saved vs Baseline 2 · joint effects grouped (see caption)',
         ha='center',va='bottom',fontsize=9,color='#425466')
stem=P/'fig3/feature_contributions_wo24'
for ext in ('pdf','svg','png'):
    fig.savefig(stem.with_suffix('.'+ext),dpi=250,facecolor='white',
                metadata={'Creator':'CHEETAH feature allocation'} if ext!='svg' else None)
plt.close(fig)
result=dict(passed=True, panels=panels,bindings=bindings,
    normalization='share_i=(T_i-T_next)/(T_reference-T_final); exact fractions telescope to 1. These also allocate the speedup gain S-1, not a sum of multiplicative feature speedups.',
    attribution_scope='Visual, order-dependent allocation with pooled initial hardware/mapping/memory contrasts. Not an independent causal feature decomposition or evidence that every inherited choice is free in the latest LP.',
    figure2='Unchanged historical figure. New pies use phase-labeled latest endpoints in the companion current-results table, not the historical Figure 2 bar values.',
    physical_scope='Analytical paper-method matrix and associated traffic; original precision proxy and research hardware qualifications remain. Not measured silicon or full native serving.',
    new_optimizer_calls=0, new_full_source_admission=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'plot_values.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({p['workload']:{w['label'].replace('\n',' '):w['display_percent']+'%' for w in p['weights']} for p in panels},indent=2))
