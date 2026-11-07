"""Four feature slices per decoder: visual estimates from checked controls.

Normalize positive conditional control savings. The user permits overlapping
effects for this illustration. Do not call this an exhaustive causal partition
of total Baseline-2 acceleration or add the individual speedup factors.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parents[1]
E=P/'update/paper_evidence_20261106/figure3_latest12'
OUT=P/'update/paper_evidence_20261106/figure3_pies_v3'
bindings={}


def read(p,expected=None):
    p=Path(p);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if expected:assert sha==expected,str(p)
    bindings[str(p.relative_to(P))]=sha
    return json.loads(raw)


d=read(OUT/'table_control_values.json')
for v in d['sources'].values():read(E/v['file'],v['sha256'])
old=read(P/'update/paper_evidence_20261106/figure3_pies_v2/input_manifest.json')
for f,sha in old['unchanged_figure2'].items():
    assert hashlib.sha256((P/f).read_bytes()).hexdigest()==sha
ds=read(P/'update/paper_evidence_20261106/figure3_pies_v2/inputs/deepseek_controls.json')
assert ds['passed'] and ds['all_controls_source_replayed']


def control(key,label):
    c=next(c for c in d['controls'][key]if c['label']==label)
    assert F(c['off_ns'])-F(c['on_ns'])==F(c['saved_ns'])
    return F(c['saved_ns']),c['scope']


def c(key,label,title,color):
    value,scope=control(key,label)
    return dict(label=title,color=color,saved_ns=str(value),scope=scope)


HW='#4075ae';TILE='#52abc9';KV='#b379bc';WEIGHT='#7b5ca2'
FUSION='#55a67c';PREFETCH='#e5a745';CONTEXT='#d77969'
attn={k:F(v['attention_ns'])for k,v in ds['controls'].items()}
prefetch=attn['producer1_delivery0']-attn['context2_off_prefetch_retained']
context=attn['context2_off_prefetch_retained']-attn['producer1_delivery1']
assert prefetch+context==control('deepseek_v3','Delivery')[0]
panels=[dict(workload='deepseek_v3',label='DeepSeek-V3',weights=[
    c('deepseek_v3','HW menu','Hardware change',HW),
    c('deepseek_v3','Producer','Producer-layout fusion',FUSION),
    dict(label='Cross-wave prefetch',color=PREFETCH,saved_ns=str(prefetch),
         scope='Enable prefetch with producer fusion on, before adding the second FP32 context; same chip and finite action replay.'),
    dict(label='Two accumulator contexts',color=CONTEXT,saved_ns=str(context),
         scope='Add second FP32 context with producer fusion and prefetch retained; same chip and finite action replay.')])]
for key,label,tiling in [('llama1_13b','Llama-1 13B','25 to 31 maps'),
                         ('llama1_65b','Llama-1 65B','14 to 25 maps'),
                         ('llama3_70b','Llama-3 70B','14 to 25 maps')]:
    kvlabel='GQA' if key=='llama3_70b' else 'KV'
    kvtitle='KV-head reuse (GQA)' if key=='llama3_70b' else 'KV retention'
    panels.append(dict(workload=key,label=label,weights=[
        c(key,'HW menu','Hardware change',HW),c(key,tiling,'Tiling extension',TILE),
        c(key,kvlabel,kvtitle,KV),c(key,'Weights','Weight residency',WEIGHT)]))

for panel in panels:
    key=panel['workload'];ws=panel['weights']
    assert 4<=len(ws)<=5 and all(F(w['saved_ns'])>0 for w in ws)
    total=sum((F(w['saved_ns'])for w in ws),F())
    ratio=d['performance'][key]['reference_ratio']
    panel.update(phase='Decode',reference_ratio=ratio,normalization_ns=str(total))
    exact=[]
    for w in ws:
        share=F(w['saved_ns'])/total
        w['share_exact']=str(share);exact.append(share*10000)
        # This is an explicitly illustrative allocation, not measured causality.
        w['illustrative_share_of_total_speedup_gain']=float(share)*(ratio-1)
    units=[v.numerator//v.denominator for v in exact]
    ranking=sorted(range(len(ws)),key=lambda i:exact[i]-units[i],reverse=True)
    for i in ranking[:10000-sum(units)]:units[i]+=1
    for w,u in zip(ws,units):w['display_percent']=f'{u//100}.{u%100:02d}'
    assert sum(F(w['share_exact'])for w in ws)==1
    assert sum(F(w['display_percent'])for w in ws)==100

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(10.8,4.10),facecolor='white')
for i,p in enumerate(panels):
    x=.012+i*.25
    fig.text(x+.115,.965,p['label'],ha='center',va='top',fontsize=12,fontweight='bold',color='#172536')
    fig.text(x+.115,.905,f"Decode · {p['reference_ratio']:.3f}× vs B2",ha='center',va='top',fontsize=10.1,color='#3d4c5c')
    ax=fig.add_axes([x,.335,.23,.53])
    values=[float(F(w['share_exact']))for w in p['weights']]
    wedges,_=ax.pie(values,colors=[w['color']for w in p['weights']],startangle=90,counterclock=False,
        wedgeprops={'edgecolor':'white','linewidth':.8},normalize=True)
    for wedge,w in zip(wedges,p['weights']):
        if F(w['share_exact'])>=F(10,100):
            angle=math.radians((wedge.theta1+wedge.theta2)/2)
            ax.text(.62*math.cos(angle),.62*math.sin(angle),w['display_percent']+'%',ha='center',va='center',
                fontsize=10.2,fontweight='bold',color='white')
    ax.set_aspect('equal')
    for j,w in enumerate(p['weights']):
        y=.287-j*.061
        fig.add_artist(plt.Rectangle((x+.002,y-.008),.009,.020,transform=fig.transFigure,facecolor=w['color'],edgecolor='none'))
        fig.text(x+.018,y,w['label'],ha='left',va='center',fontsize=8.1,color='#263442')
        fig.text(x+.232,y,w['display_percent']+'%',ha='right',va='center',fontsize=8.3,color='#263442')
fig.text(.5,.018,'Visual estimates · feature weights normalized to 100% · overlapping effects (see caption)',
    ha='center',va='bottom',fontsize=9,color='#425466')
stem=P/'fig3/feature_contributions_wo24'
for ext in ('pdf','svg','png'):
    fig.savefig(stem.with_suffix('.'+ext),dpi=250,facecolor='white',
        metadata={'Creator':'CHEETAH normalized feature estimates'}if ext!='svg'else None)
plt.close(fig)
result=dict(passed=True,panels=panels,bindings=bindings,
    normalization='Positive control time savings normalized to sum to one. Displayed percentages also sum to 100.00. These are visual estimates, not an exhaustive partition of the full reference-to-design saving.',
    overlap='Conditional effects can overlap; no sum of independent feature factors or causal coverage claim. DeepSeek prefetch and second-context increments form one ordered split of the delivery control.',
    reference_scope='Headlines are unchanged Baseline 2 analytical decode ratios from the current results table. Historical Figure 2 stays unchanged.',
    feature_scope='13B and 65B isolate the four requested control categories. 70B GQA is KV-read reuse, distinct from KV retention. DeepSeek provides the additional encoded fusion and delivery categories; its resident arenas remain inherited.',
    new_optimizer_calls=0,new_physical_or_full_source_admission=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'plot_values.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({p['workload']:{w['label']:w['display_percent']+'%'for w in p['weights']}for p in panels},indent=2))
