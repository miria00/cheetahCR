"""Grouped cumulative feature pies with authenticated hardware controls.

The 1x reference is excluded from the pie. Exact increments telescope to
S_final - 1; the percentages therefore partition the whole modeled speedup
gain. Fixed-order attribution assigns interactions to the later stage.
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
OUT = P/'update/paper_evidence_20261106/figure3_pies_v10'
path = OUT/'grouped_controls.json'
raw = path.read_bytes()
d = json.loads(raw)
assert d['passed'] and d['optimizer_calls']==0
for name,sha in d['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha

colors = {
    'Placement / residency':'#4075ae', 'Hardware resources':'#72a0ce',
    'Tiling + on-chip bandwidth':'#52abc9', 'Compute parallelism':'#315a94',
    'Off-chip DRAM bandwidth':'#e5a745', 'Memory capacity + residency':'#9771b5',
    'Fusion / storage':'#55a67c', 'Scheduling / concurrency':'#d77969'}

for panel in d['panels']:
    stages = panel['slices']
    assert len(stages)==4
    exact = [F(z['share_exact'])*100000 for z in stages]
    units = [x.numerator//x.denominator for x in exact]
    order = sorted(range(len(stages)),key=lambda i:exact[i]-units[i],reverse=True)
    for i in order[:100000-sum(units)]:
        units[i] += 1
    for z,u in zip(stages,units):
        z['color'] = colors[z['label']]
        z['display_percent'] = f'{u//1000}.{u%1000:03d}'
        z['display_label'] = z['label']
    assert sum(F(z['share_exact']) for z in stages)==1
    assert sum(F(z['display_percent']) for z in stages)==100
    assert sum(F(z['added_speedup_exact']) for z in stages)==F(panel['final_speedup_exact'])-1

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig = plt.figure(figsize=(10.8,4.10),facecolor='white')
legend_artists = []
for i,panel in enumerate(d['panels']):
    x = .012+i*.25
    fig.text(x+.115,.965,panel['label'],ha='center',va='top',fontsize=12,
             fontweight='bold',color='#172536')
    fig.text(x+.115,.905,f"Decode · {panel['final_speedup']:.3f}×",
             ha='center',va='top',fontsize=10.1,color='#3d4c5c')
    ax = fig.add_axes([x,.335,.23,.53])
    stages = panel['slices']
    wedges,_ = ax.pie([float(F(z['share_exact'])) for z in stages],
                     colors=[z['color'] for z in stages],startangle=90,counterclock=False,
                     wedgeprops={'edgecolor':'white','linewidth':.65},normalize=True)
    for wedge,z in zip(wedges,stages):
        if F(z['share_exact'])>=F(20,100):
            angle = math.radians((wedge.theta1+wedge.theta2)/2)
            ax.text(.51*math.cos(angle),.51*math.sin(angle),z['display_percent']+'%',
                    ha='center',va='center',fontsize=10.2,fontweight='bold',color='white')
    ax.set_aspect('equal')
    for j,z in enumerate(stages):
        y = .287-j*.047
        fig.add_artist(plt.Rectangle((x+.002,y-.008),.009,.020,transform=fig.transFigure,
                                    facecolor=z['color'],edgecolor='none'))
        label = fig.text(x+.018,y,z['display_label'],ha='left',va='center',
                         fontsize=7.3,color='#263442')
        percent = fig.text(x+.237,y,z['display_percent']+'%',ha='right',va='center',
                           fontsize=8,color='#263442')
        legend_artists.append((label,percent))
fig.text(.5,.006,'Grouped feature contributions · 100% of modeled speedup gain above 1×',
         ha='center',va='bottom',fontsize=9,color='#425466')
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for label,percent in legend_artists:
    assert label.get_window_extent(renderer).x1+2 < percent.get_window_extent(renderer).x0, label.get_text()
stem = P/'fig3/feature_contributions_wo24'
for ext in ('pdf','svg','png'):
    fig.savefig(stem.with_suffix('.'+ext),dpi=250,facecolor='white',
                metadata={'Creator':'CHEETAH grouped cumulative feature contributions'} if ext!='svg' else None)
plt.close(fig)
result = dict(passed=True,panels=d['panels'],method=d['method'],order_scope=d['order_scope'],
              input_sha256=hashlib.sha256(raw).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              percentages_sum_exactly_to_100=True,legend_overlap_check=True,
              new_optimizer_calls=0,physical_or_full_source_admission=False)
(OUT/'plot_values.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({p['workload']:{z['label']:z['display_percent']+'%' for z in p['slices']}
                  for p in d['panels']},indent=2))
