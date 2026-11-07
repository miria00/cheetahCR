"""Bounded CPU replays for the four requested Figure 3 workloads; no optimizer."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
S = ROOT / 'systems/cheetah_joint'
sys.path[:0] = [str(S / 'tools'), str(S)]
E = P / 'update/paper_evidence_20261106/figure3_latest12'
data = json.loads((E / 'feature_groups_values.json').read_text())
bindings = {}


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    bindings[str(path)] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def frozen(original):
    info = data['sources'][original]
    path = E / info['file']
    result = read(path)
    assert bindings[str(path)] == info['sha256']
    return result


def current(key):
    return frozen(str(ROOT / 'Results/Blueprints/WO24/records' / (key + '.json')))


ap = argparse.ArgumentParser()
ap.add_argument('--case', choices=['deepseek', 'unet', 'llama_weights', 'llama_kv', 'b3'], required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--producer-python', default=sys.executable)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
start = time.monotonic()
details = {}
if a.case == 'deepseek':
    from wo24_deepseek_attention_controls import branch
    hid = current('deepseek_v3')['chip_id']
    original = Path('/tmp/cheetah_wo24_20261106_v1/deepseek_extension_v1/attention_controls_v1') / hid / 'result.json'
    old = read(original)
    result = branch((dict(id=hid, path=old['prepared_path'], sha256=old['prepared_sha256']), str(a.out)))
    fresh = read(result['path'])
    assert fresh['passed'] and fresh['all_controls_source_replayed']
    for name, control in old['controls'].items():
        assert F(fresh['controls'][name]['attention_ns']) == F(control['attention_ns'])
    for field in ['producer_conditional_saving_ns', 'delivery_conditional_saving_ns', 'context2_incremental_saving_ns']:
        assert fresh[field] == old[field]
    details = dict(scope='Fresh original finite attention action reoptimization and replay on selected hardware; fixed nonattention terms inherited.',
                   controls=5, hardware=hid, producer_saved_ns=fresh['producer_conditional_saving_ns'],
                   delivery_saved_ns=fresh['delivery_conditional_saving_ns'])
elif a.case in ('llama_weights', 'llama_kv'):
    root = Path('/tmp/cheetah_wo23_20261106_v1/dense_lane_v1/65')
    script = S / 'tools' / ('wo23_dense_feature_controls.py' if a.case == 'llama_weights' else 'wo23_dense_kv_control.py')
    args = [a.producer_python, str(script), '--replay', str(root / 'control_H1_new_v1_replay/result.json'),
            '--source', str(root / 'mapping_source_v1/source.json'), '--out', str(a.out / 'producer')]
    if a.case == 'llama_kv':
        args += ['--model', '65']
    with (a.out / 'producer.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=210)
    fresh = read(a.out / 'producer/result.json')
    old = read(root / ('features_selected_v1' if a.case == 'llama_weights' else 'KV_off_selected_v1') / 'result.json')
    assert fresh['passed'] and fresh['hardware'] == current('llama1_65b')['hardware']
    if a.case == 'llama_weights':
        assert fresh['full_decode_ns'] == old['full_decode_ns']
        assert fresh['without_weight_retention'] == old['without_weight_retention']
        saved = fresh['without_weight_retention']['signed_saved_ns']
    else:
        for field in ['decode_full_ns', 'decode_without_KV_retention_ns', 'restore_service_ns', 'reload_payload_bytes']:
            assert fresh[field] == old[field]
        saved = fresh['signed_saved_ns']
    details = dict(scope='Fresh original dense source-cost control, complete fixed-chip mapping minimum and original paid import/copy obligations.', saved_ns=saved)
elif a.case == 'unet':
    from wo23_unet_actions import actions
    root = Path('/tmp/cheetah_wo23_20261106_v1/unet_lane_v1')
    source = read(root / 'prepared_v2/source_program.json')
    candidate = read(root / 'finite_replay_v3/candidate.json')
    old = read(root / 'delivery_v1/result.json')
    g = source['graph']
    byid = {e['id']: e for e in g['events']}
    h = next(h['hardware'] for h in source['hardware'] if h['id'] == candidate['hardware_id'])
    allowed = {(0,0): {'serial'}, (0,1): {'serial','epilogue'},
               (1,0): {'serial','buffered'}, (1,1): {'serial','epilogue','buffered','buffered_cast'}}
    totals = {k: F() for k in allowed}
    count = 0
    for event in candidate['events']:
        if 'recipe' not in event.get('detail', {}):
            for k in totals: totals[k] += F(event['cost_ns'])
            continue
        info = event['detail']
        fresh = actions(info['recipe'], h, byid[info['matrix_id']], byid[info['cast_id']], g)
        old_mode = info['id'].split('/')[-1]
        for k in totals:
            buffer, cast = k
            keep_buffer = buffer and old_mode in ('buffered', 'buffered_cast')
            keep_cast = cast and old_mode in ('epilogue', 'buffered_cast')
            mode = 'buffered_cast' if keep_buffer and keep_cast else ('buffered' if keep_buffer else ('epilogue' if keep_cast else 'serial'))
            # Match the archived signed control: disable a selected mechanism;
            # do not enable a previously unselected one during this ablation.
            totals[k] += F(next(x['cost_ns'] for x in fresh if x['id'] == mode))
        count += 1
    assert count == 794
    for control in old['signed_controls']:
        assert totals[control['buffer'], control['cast']] == F(control['latency_ns'])
    assert totals[1,1] == F(current('unet')['latency_ns'])
    details = dict(scope='Fresh original cost/calendar functions for 794 selected matrix recipes and four cast/buffer controls; same hardware/mapping, fixed padding inherited.',
                   matrix_recipes=count, controls={str(k):str(v) for k,v in totals.items()})
elif a.case == 'b3':
    record = current('efficientnet_b3')
    original = record['historical_record']['source_receipt']
    old = frozen(original)
    assert old['passed']
    modes = old['results']
    off, on = (modes[k] for k in ('materialized', 'whole_owner'))
    assert [x['op'] for x in off['timeline']['per_op']] == [x['op'] for x in on['timeline']['per_op']]
    assert [x['mode'] for x in off['timeline']['per_op']] == [x['mode'] for x in on['timeline']['per_op']]
    totals = {}
    for mode, result in modes.items():
        assert result['passed'] and result['all_consumer_choices_binary']
        rows = result['timeline']['per_op']; assert len(rows) == 287
        previous = 0.; total = 0.
        for r in rows:
            assert r['active'] and r['n_inst'] == 1
            assert abs(r['start_ms'] - previous) < 1e-10
            assert r['end_ms'] >= r['start_ms']
            total += r['end_ms'] - r['start_ms']; previous = r['end_ms']
        assert abs(total-result['scheduled_ms']) < 1e-10
        totals[mode] = total
    assert abs(totals['whole_owner'] - record['latency_seconds']*1000) < 1e-10
    details = dict(scope='Fresh saved-timeline reconstruction of 287 original operations in both controls; original source admission, owner placement and mapper algorithms inherited.',
                   operations_per_control=287, totals_ms=totals, saved_ms=totals['materialized']-totals['whole_owner'])

receipt = dict(passed=True, case=a.case, wall_seconds=time.monotonic()-start,
               optimizer_calls=0, new_hardware_search=False, details=details, bindings=bindings,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(a.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(case=a.case,passed=True,seconds=receipt['wall_seconds'],details=details)))
