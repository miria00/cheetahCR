"""Cumulative hardware-resource controls on the selected resident decoders.

This repairs the v4 attribution question: disabling residency on an already
expanded custom chip and enabling it last is not a hardware/software split.
Here residency, memory capacity and host imports stay fixed. The intermediate
controls cap service on the selected chip; they are not TPU-v3 implementations.
"""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
from itertools import permutations
import hashlib
import json
import sys

P=Path(__file__).resolve().parents[1]
S=P.parent/'systems/cheetah_joint'
sys.path.insert(0,str(S))
from wo23_dense_banks_geometry_v2 import exact_minimum as exact25
from wo24_dense_banks import exact_minimum as exact31
from wo23_dense_mapping import TILES as TILES25
from wo24_dense_mapping import TILES as TILES31
from wo22_dense_staging import dma

OUT=P/'update/paper_evidence_20261106/figure3_pies_v5'
assert not (OUT/'cumulative_controls.json').exists()
bindings={}


def read(path,expected=None):
    path=Path(path);raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if expected:
        assert sha==expected,str(path)
    bindings[str(path)]=sha
    return json.loads(raw)


def finalize(key,label,record,stages):
    ref=F(record['reference_ns']);prev=ref;sp=F(1)
    for i,z in enumerate(stages):
        t=F(z['time_ns']);s=ref/t
        assert 0<t<prev,(key,i)
        z.update(order=i+1,signed_time_saved_ns=str(prev-t),speedup_exact=str(s),
                 speedup=float(s),added_speedup_exact=str(s-sp))
        prev,sp=t,s
    assert prev==F(record['latency_ns'])
    assert sp==F(record['reference_ratio_exact'])
    assert sum(F(z['added_speedup_exact']) for z in stages)==sp-1
    for z in stages:
        z['share_exact']=str(F(z['added_speedup_exact'])/(sp-1))
    return dict(workload=key,label=label,phase='Decode',reference_ns=str(ref),
                final_ns=str(prev),final_speedup_exact=str(sp),final_speedup=float(sp),stages=stages)


def independent_cost(source,hardware,tiles,selected):
    """Reconstruct every chosen kernel and original descriptor import."""
    h=hardware;total=F();counts=0;payload=0
    for z,ti in zip(source['profiles'],selected):
        matches=[m for m in z['options'] if m['matrix_tile']==ti]
        assert len(matches)==1
        m=matches[0];t=tiles[ti];gm,gn=h['array_geometry']
        bundle=((t.m+gm-1)//gm)*((t.n+gn-1)//gn)
        active=h['tiles']//bundle
        assert not m['jobs'] or active>0
        if m['jobs']:
            # Retained original bank-compatibility gates; lowering service
            # does not create extra storage or remove a port obligation.
            assert max(t.buffering*2*min(16,min(gm,t.m))*t.k,
                       t.buffering*2*t.k*min(16,min(gn,t.n)),
                       4*min(gm,t.m)*min(gn,t.n))<=h['operand_bank_bytes']
        else:
            assert m['workspace']<=262144
        cycles=((m['jobs']+active-1)//active)*(m['reduction_cycles']+gm+gn-2) if m['jobs'] else 0
        cost=(F(cycles)+F(m['vector'],h['vector_lanes'])+
              F(m['stream'],h['bandwidth'])+F(m['local'],h['local_GBps']))*m.get('multiplicity',1)
        total+=cost*z['counts'][1];counts+=z['counts'][1]
    for group,count_key in [('descriptors','decode_resident_count'),('KV_decode_writes','count')]:
        for z in source[group]:
            e,s=dma(z['address'],z['bytes'],h['host_interface_units'],h['staging_bytes'],h['staging_banks'])
            count=z[count_key]
            total+=count*(F(e+2*s)+F(z['bytes'],h['bandwidth']))
            payload+=count*z['bytes']
    return total,counts,payload


snapshot=read(OUT/'input_snapshot.json')
prior=read(P/'update/paper_evidence_20261106/figure3_pies_v4/cumulative_controls.json')
panels=[deepcopy(prior['panels'][0])]
assert panels[0]['workload']=='deepseek_v3'
audits={}
for n in (13,65,70):
    key=f'llama{3 if n==70 else 1}_{n}b'
    record=read(P.parent/'Results/Blueprints/WO24/records'/(key+'.json'))
    replay=read(record['source_replay']['original_path'],record['source_replay']['sha256'])
    sp=Path(f'/tmp/cheetah_wo{24 if n==13 else 23}_20261106_v1/dense_lane_v1/{n}/mapping_source_v1/source.json')
    source=read(sp,replay['bindings'][str(sp)])
    h=record['hardware'];assert replay['passed'] and h==replay['hardware']
    minimum,tiles,oldcount=(exact31,TILES31,25) if n==13 else (exact25,TILES25,14)
    caps=snapshot['diagnostic_caps']
    low=dict(tiles=caps['active_arrays'],bandwidth=caps['delivered_DRAM_GBps'],local_GBps=caps['delivered_local_GBps'])
    for k,v in low.items():
        assert 0<v<=h[k]
    assert low['local_GBps']>=32*h['staging_banks']
    factors=['tiles','bandwidth','local_GBps']
    cells={};selections={};counts=set();payloads=set()
    for mask in range(8):
        effective=dict(h)
        for i,k in enumerate(factors):
            effective[k]=h[k] if mask&(1<<i) else low[k]
        got=minimum(source,effective)
        independent,count,payload=independent_cost(source,effective,tiles,got['selected_tiles'])
        assert independent==F(got['decode_ns'])
        assert payload==got['timed_HOST_bytes']
        counts.add(count);payloads.add(payload)
        cells[str(mask)]=dict(time_ns=str(independent),effective_service={k:effective[k] for k in factors},
                              selected_tiles_sha256=hashlib.sha256(json.dumps(got['selected_tiles']).encode()).hexdigest())
        selections[str(mask)]=got['selected_tiles']
    assert len(counts)==len(payloads)==1
    assert F(cells['7']['time_ns'])==F(record['latency_ns'])==F(replay['phase_ns']['decode'])
    for mask in range(8):
        for i in range(3):
            assert F(cells[str(mask|1<<i)]['time_ns'])<=F(cells[str(mask)]['time_ns'])
    effective=dict(h,**low)
    old=minimum(source,effective,allowed_tiles=set(range(oldcount)))
    audit_old,count,payload=independent_cost(source,effective,tiles,old['selected_tiles'])
    assert audit_old==F(old['decode_ns']) and payload in payloads and count in counts
    first_scope=('Joint resident custom datapath: final memory capacity, weight and KV retention, '
                 'original first imports, formats, vector and host resources; old mapping menu. '
                 'Active arrays capped at 256, DRAM at 900 GB/s and local delivery at 8192 GB/s. '
                 'Not isolated software and not a TPU-v3 implementation.')
    rows=[('Resident datapath + KV',old['decode_ns'],first_scope,0,oldcount),
          ('Tiling extension',cells['0']['time_ns'],f'Extend {oldcount} to {len(tiles)} compatible mappings under the same service caps; residency and all owners stay fixed.',0,len(tiles)),
          ('Array parallelism',cells['1']['time_ns'],'Restore selected array availability; DRAM/local caps remain. Compatible mappings may be reselected.',1,len(tiles)),
          ('DRAM bandwidth',cells['3']['time_ns'],'Restore selected delivered DRAM bandwidth, retaining full array availability and the local-service cap.',3,len(tiles)),
          ('Local delivery bandwidth',cells['7']['time_ns'],'Restore selected local delivery bandwidth. This reaches the unchanged selected execution.',7,len(tiles))]
    stages=[dict(label=label,time_ns=t,scope=scope,resource_mask=mask,mapping_count=nm) for label,t,scope,mask,nm in rows]
    panel=finalize(key,f'Llama-{3 if n==70 else 1} {n}B',record,stages)
    panels.append(panel)
    # All six orders quantify order dependence without changing the figure's
    # declared sequence. Aggregate resource contribution is identical.
    orders=[];ref=F(record['reference_ns']);gain=ref/F(record['latency_ns'])-1
    for order in permutations(range(3)):
        mask=0;previous=ref/F(cells['0']['time_ns']);parts={}
        for i in order:
            mask|=1<<i;current=ref/F(cells[str(mask)]['time_ns'])
            parts[factors[i]]=str(current-previous);previous=current
        orders.append(dict(order=[factors[i] for i in order],added_speedup=parts))
    totals={sum(F(v) for v in z['added_speedup'].values()) for z in orders}
    assert len(totals)==1
    hw_gain=next(iter(totals));assert hw_gain==sum(F(z['added_speedup_exact']) for z in stages[2:])
    # A second endpoint diagnostic holds the final mapping choices fixed too.
    fixed_low,_,_=independent_cost(source,effective,tiles,selections['7'])
    assert fixed_low>=F(cells['0']['time_ns'])
    audits[key]=dict(installed_hardware=h,peak_live_DRAM_bytes=source['peak_live_DRAM_all_retained_bytes'],
        eligible_weight_count=replay['selected_weight_count'],resource_caps=low,all_eight_cells=cells,
        all_six_hardware_orders=orders,hardware_share_exact=str(hw_gain/gain),hardware_share=float(hw_gain/gain),
        final_mapping_fixed_low_service_time_ns=str(fixed_low),timed_HOST_bytes=next(iter(payloads)),
        original_decode_invocations=next(iter(counts)),
        invariants='Same installed memory, bank/port reservations, clock, host/staging path, original '
                   'source graph, arithmetic, KV owners, all eligible weight retention and paid first imports. '
                   'Only active-array availability and service rates are reduced. No area/power savings '
                   'are claimed for the capped control. Mapping choices may change within the stated menu.',
        interpretation='Conditional hardware-resource contribution including compatible remapping, '
                       'not a unique hardware-only fraction of the TPU-v3 comparison.')
    print(json.dumps({'workload':key,'stage_speedups':[x['speedup'] for x in stages],
                      'hardware_share_percent':100*float(hw_gain/gain),'all_resource_orders_checked':6}),flush=True)

# Verify the fixed-chip table cannot supply a current matched decode control.
anchors_path=Path('/home/miria/CHEETAH/rebuttal/grid_completion/highs_anchors.json')
anchors=read(anchors_path)['cells'];historical={}
for key in ['llama1_13b','llama1_65b','llama3_70b','deepseek_v3']:
    c=next(x for x in anchors if x['cell']==key+'@tpuv3(900,32)/joint')
    historical[key]={k:c[k] for k in ['cell','objective','naive_sum_Tmax','speedup_vs_naive','convention','src']}
for name,sha in snapshot['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha
for name in ['wo23_dense_banks_geometry_v2.py','wo24_dense_banks.py','wo23_dense_mapping.py',
             'wo24_dense_mapping.py','wo22_dense_staging.py']:
    path=S/name;bindings[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
result=dict(passed=True,panels=panels,hardware_audits=audits,bindings=bindings,optimizer_calls=0,
    unchanged_figure2=snapshot['unchanged_figure2'],historical_fixed_hardware=historical,
    method='S_i = original Baseline 2 time / cumulative stage time, S_0 = 1; '
           'slice i = (S_i-S_(i-1))/(S_final-1). Exact signed time savings are also retained.',
    order_scope='Declared order: joint resident datapath, tiling, arrays, DRAM bandwidth, local bandwidth. '
                'The later stage receives interactions with earlier choices. All six orders of the '
                'three hardware resources are separately recorded; aggregate resource gain is invariant. '
                'DeepSeek retains its separately checked five-stage sequence from v4.',
    correction='v4 added weight residency last on expanded hardware and assigned its enabling hardware '
               'effects to parameter classes. Those were valid sequential costs but did not isolate '
               'hardware versus software. No v4 percentages are reused as hardware contributions.',
    fixed_hardware_scope='Historical approximately 2x entries are fractional LP objectives with different '
                         'stored conventions, not the current full decode task. They are not used in '
                         'these percentages. New reduced-service controls are explicitly custom-chip controls.',
    source_scope='Exact original finite-kernel/descriptor cost replays, with independent chosen-cost '
                 'reconstruction and inherited source obligations. Analytical resource sensitivity, '
                 'not new OCS search, complete source/lifetime admission, physical timing or TPU execution.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'cumulative_controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'fresh_hardware_cells':24,'old_menu_controls':3,
                  'hardware_orders':18,'new_optimizer_calls':0}))
