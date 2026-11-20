"""FAST-style cumulative analytical controls, retaining the original Baseline 2.

Run with /home/miria/miniconda3/bin/python. No numerical optimizer is called.
Every stage retains the preceding additions. The first custom configuration
groups coupled hardware, base mappings and necessary memory policies; it is
not an isolated hardware attribution. Existing source admission is inherited.
"""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
import hashlib
import json
import sys

P = Path(__file__).resolve().parents[1]
S = P.parent / 'systems/cheetah_joint'
sys.path.insert(0, str(S))
from wo22_dense_staging import dma
from wo23_dense_banks_geometry_v2 import exact_minimum as exact25
from wo24_dense_banks import exact_minimum as exact31
from wo23_dense_mapping import alternative

E = P / 'update/paper_evidence_20261106/figure3_latest12'
OUT = P / 'update/paper_evidence_20261106/figure3_pies_v4'
OUT.mkdir(exist_ok=False)
bindings = {}


def read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, str(path)
    bindings[str(path)] = digest
    return json.loads(raw)


def stage(label, ns, scope, **extra):
    return dict(label=label, time_ns=str(ns), scope=scope, **extra)


def finish(key, label, record, stages):
    reference = F(record['reference_ns'])
    previous_time = reference
    previous_speedup = F(1)
    for i, item in enumerate(stages):
        current = F(item['time_ns'])
        assert 0 < current < previous_time, (key, i, current, previous_time)
        speedup = reference / current
        item.update(order=i+1, signed_time_saved_ns=str(previous_time-current),
                    speedup_exact=str(speedup), speedup=float(speedup),
                    added_speedup_exact=str(speedup-previous_speedup))
        previous_time, previous_speedup = current, speedup
    assert previous_time == F(record['latency_ns'])
    assert previous_speedup == F(record['reference_ratio_exact'])
    assert sum((F(z['added_speedup_exact']) for z in stages), F()) == previous_speedup-1
    assert sum((F(z['signed_time_saved_ns']) for z in stages), F()) == reference-previous_time
    for item in stages:
        item['share_exact'] = str(F(item['added_speedup_exact'])/(previous_speedup-1))
    return dict(workload=key, label=label, phase='Decode', reference_ns=str(reference),
                final_ns=str(previous_time), final_speedup_exact=str(previous_speedup),
                final_speedup=float(previous_speedup), stages=stages)


manifest = read(E / 'feature_groups_values.json')
v = manifest['sources']['/tmp/cheetah_wo24_20261106_v1/authoritative_dataset_v1/result.json']
dataset = read(E/v['file'], v['sha256'])
ds = read(P/'update/paper_evidence_20261106/figure3_pies_v2/inputs/deepseek_controls.json')
assert ds['passed'] and ds['all_controls_source_replayed']
record = read(P.parent/'Results/Blueprints/WO24/records/deepseek_v3.json')
cells = {k:F(v) for k,v in dataset['ablations']['deepseek']['hardware_software_cells_ns'].items()}
attention = {k:F(v['attention_ns']) for k,v in ds['controls'].items()}
final = F(record['latency_ns'])
fixed = final - attention['producer1_delivery1']
assert cells['new_hardware_new_software'] == final
assert fixed + attention['producer0_delivery0'] == cells['new_hardware_old_software']
panels = [finish('deepseek_v3', 'DeepSeek-V3', record, [
    stage('Datapath + base memory', cells['old_hardware_old_software'],
          'WO23 source-checked custom configuration: joint datapath, existing N16 mappings, resident arenas, original routing and KV policy. This grouped transition from Baseline 2 is not an isolated hardware or memory ablation.'),
    stage('Hardware refinement', cells['new_hardware_old_software'],
          'WO24 hardware with the old software policy; all earlier capabilities retained.'),
    stage('Producer-layout fusion', fixed+attention['producer1_delivery0'],
          'Enable rounded attention producer-layout storage fusion; no cross-wave prefetch or second accumulator.'),
    stage('Cross-wave prefetch', fixed+attention['context2_off_prefetch_retained'],
          'Enable paid cross-wave prefetch while retaining producer fusion and one FP32 accumulator context.'),
    stage('Double-buffered accumulators', final,
          'Add the second FP32 context with fusion and prefetch retained. Original output conversion, stores and reduction order remain paid.')])]

descriptor_ledgers = {}
for n in (13, 65, 70):
    key = f'llama{3 if n==70 else 1}_{n}b'
    r = read(P.parent/'Results/Blueprints/WO24/records'/(key+'.json'))
    replay = read(r['source_replay']['original_path'], r['source_replay']['sha256'])
    sp = Path(f'/tmp/cheetah_wo{24 if n==13 else 23}_20261106_v1/dense_lane_v1/{n}/mapping_source_v1/source.json')
    source = read(sp, replay['bindings'][str(sp)])
    h = r['hardware']
    assert replay['passed'] and replay['hardware'] == h
    minimum = exact31 if n==13 else exact25
    full = minimum(source,h)
    base = F(full['decode_ns'])
    assert base == F(r['latency_ns']) == F(replay['phase_ns']['decode'])
    old_count, new_count = (25,31) if n==13 else (14,25)
    grouped = {k:F() for k in ('attention','MLP','output')}
    details = []
    for z in source['descriptors']:
        if not z['eligible_weight']:
            continue
        name = z['name']
        group = ('attention' if '.self_attn.' in name else
                 'MLP' if '.mlp.' in name else 'output' if name=='lm_head.weight' else None)
        assert group is not None, name
        ext,local = dma(z['address'],z['bytes'],h['host_interface_units'],
                        h['staging_bytes'],h['staging_banks'])
        uses = int(z['decode_nonresident_count'])-int(z['decode_resident_count'])
        assert uses >= 0
        extra = uses*(F(ext+2*local)+F(z['bytes'],h['bandwidth']))
        # Independent chunk-calendar expansion, including every setup and
        # both staging passes, checks the per-import producer cost.
        addr,remaining = int(z['address']),int(z['bytes'])
        ee = ll = 0
        while remaining:
            length = min(remaining,h['staging_bytes']-addr%h['staging_bytes'])
            words = (length+31)//32
            ee += (words+h['host_interface_units']-1)//h['host_interface_units']+302
            ll += (words+h['staging_banks']-1)//h['staging_banks']+2
            remaining -= length
            addr += length
        assert (ee,ll) == (ext,local)
        grouped[group] += extra
        details.append(dict(name=name,group=group,address=z['address'],bytes=z['bytes'],
                            original_import_count=z['decode_nonresident_count'],
                            retained_import_count=z['decode_resident_count'],
                            extra_imports=uses,saved_ns=str(extra)))
    assert len(details) == replay['selected_weight_count']
    assert len({z['name'] for z in details}) == len(details)
    assert all(x>0 for x in grouped.values())
    total_extra = sum(grouped.values(),F())
    inherited = ('Current chip and original source policy, with all original KV owners, '
                 'first imports, stores, full-K arithmetic, mapping workspace and delivery costs. '
                 'The initial transition groups hardware, base tiling and KV retention; '
                 'these are not separately attributed.')
    if n != 70:
        old = minimum(source,h,allowed_tiles=set(range(old_count)))
        time = F(old['decode_ns'])+total_extra
        stages = [stage('Hardware + KV retention',time,inherited,
                        mapping_count=old_count,retained_weight_groups=[])]
        time = base+total_extra
        stages.append(stage('Tiling extension',time,
                            f'Extend {old_count} to {new_count} compatible logical mappings, '
                            'with weight retention still disabled and unchanged KV policy.',
                            mapping_count=new_count,retained_weight_groups=[]))
        retained = []
        for group,title in [('attention','Attention-weight residency'),
                            ('MLP','MLP-weight residency'),('output','Output-weight residency')]:
            time -= grouped[group]
            retained.append(group)
            stages.append(stage(title,time,
                                'Enable only this additional class of the original independent '
                                'weight selectors. All prior retained classes stay on; first '
                                'loads and every kernel weight read remain charged.',
                                retained_weight_groups=list(retained)))
    else:
        single = deepcopy(source)
        changed = 0
        for original,z in zip(source['profiles'],single['profiles']):
            if z['original_mode']['implementation'] != 'private_GQA_head_rows':
                continue
            assert z['binding']['row']['matrix']['m']==1
            assert z['binding']['row']['matrix']['batches']==64
            z['original_mode']['gqa_group']=1
            z['options']=[alternative(z,m['matrix_tile']) for m in original['options']]
            assert z['binding']==original['binding'] and z['counts']==original['counts']
            for before,after in zip(original['options'],z['options']):
                assert before['matrix_tile']==after['matrix_tile']
                assert before['multiplicity']==after['multiplicity']==1
                assert before['reduction_cycles']==after['reduction_cycles']
            changed += 1
        assert changed==1022
        old_single = minimum(single,h,allowed_tiles=set(range(old_count)))
        new_single = minimum(single,h)
        stages = [
            stage('Hardware + KV retention',F(old_single['decode_ns'])+total_extra,inherited,
                  mapping_count=old_count,gqa_group=1,retained_weight_groups=[]),
            stage('Tiling extension',F(new_single['decode_ns'])+total_extra,
                  'Extend 14 to 25 compatible mappings, with independent attention heads '
                  'and weight retention still disabled.',mapping_count=new_count,gqa_group=1,
                  retained_weight_groups=[]),
            stage('KV-head reuse (GQA)',base+total_extra,
                  'Enable the original group-eight attention implementation on the same chip '
                  'and 25-mapping menu. This is KV-read reuse, distinct from retention.',
                  mapping_count=new_count,gqa_group=8,retained_weight_groups=[]),
            stage('Attention-weight residency',base+grouped['MLP']+grouped['output'],
                  'Retain attention weights, with GQA and all preceding features enabled. '
                  'Initial loads and kernel reads remain paid.',retained_weight_groups=['attention']),
            stage('MLP/output-weight residency',base,
                  'Retain the remaining original MLP and output-head weights. All original '
                  'initial imports and kernel reads remain charged.',
                  retained_weight_groups=['attention','MLP','output'])]
    panels.append(finish(key,f'Llama-{3 if n==70 else 1} {n}B',r,stages))
    descriptor_ledgers[key] = dict(hardware=h,descriptors=details,
        group_savings_ns={k:str(v) for k,v in grouped.items()},all_weight_off_delta_ns=str(total_extra),
        capacity_argument='Every stage retains a subset of the final selected whole-weight owners. '
                          'It cannot exceed the original all-retained live inventory. Transient '
                          'imports use the already reserved staging path. Mapping minima check '
                          'bank compatibility; original KV, activation and output obligations stay fixed.',
        independence_argument='Original descriptor import counts and service depend only on descriptor '
                              'identity and hardware. Mapping and GQA alter kernel costs; each kernel '
                              'B read remains paid. Exact extra imports can therefore be added to '
                              'every selected mapping policy without a new overlap assumption.')
    print(json.dumps({'workload':key,'times_s':[float(F(x['time_ns'])/10**9) for x in stages],
                      'speedups':[x['speedup'] for x in stages]}),flush=True)

for name in ('wo22_dense_staging.py','wo23_dense_banks_geometry_v2.py','wo24_dense_banks.py',
             'wo23_dense_mapping.py','wo24_dense_mapping.py','wo14_gqa_candidate.py'):
    path=S/name
    bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
old = read(P/'update/paper_evidence_20261106/figure3_pies_v2/input_manifest.json')
for name,sha in old['unchanged_figure2'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha
result = dict(passed=True,panels=panels,descriptor_ledgers=descriptor_ledgers,bindings=bindings,
    unchanged_figure2=old['unchanged_figure2'],optimizer_calls=0,
    method='S_i = original Baseline 2 time / cumulative stage time; S_0 = 1. '
           'Slice i = (S_i-S_(i-1))/(S_final-1). Exact signed time savings are also retained. '
           'No standalone feature speedup factors or leave-one-out penalties are added.',
    order_scope='Fixed declared order; interactions are assigned to the later enabling stage. '
                'A stage may reselect compatible mappings within the enlarged menu. '
                'This is an ordered analytical attribution, not a unique causal partition.',
    first_stage_scope='Joint custom datapath, base mapping and memory policies. No counterfactual '
                      'TPU-v3 feature implementation or modified reference is inferred.',
    source_scope='Fresh exact finite-menu and descriptor cost controls using inherited source '
                 'contracts. Not new OCS search, full-model execution, lifetime admission or physical sign-off.',
    fast_source=dict(title='FAST: A Fast, Autonomous SoC Design Methodology',
                     url='https://arxiv.org/pdf/2105.12842',section='6.2.7',figure=15,
                     distinction='Figure 15 is cumulative. Table 6 separately removes components.'),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'cumulative_controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'panels':len(panels),'stages_per_panel':[len(p['stages']) for p in panels]}))
