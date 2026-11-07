"""Recheck the 65B mapping/import sequence without a new numerical solve.

Every kernel weight read remains paid. The original producer makes descriptor
imports an additive constant independent of the mapping choice. This permits
the old and new mapping menus to be compared with weight retention disabled.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import sys

P = Path(__file__).resolve().parents[1]
S = P.parent / 'systems/cheetah_joint'
sys.path.insert(0, str(S))
from wo23_dense_banks_geometry_v2 import exact_minimum
from wo23_dense_mapping import OLD_TILES, TILES
from wo22_dense_staging import dma

E = P / 'update/paper_evidence_20261106/figure3_latest12'
OUT = P / 'update/paper_evidence_20261106/figure3_pies_v2'
manifest = json.loads((E / 'feature_groups_values.json').read_text())
bindings = {}


def read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, str(path)
    bindings[str(path)] = digest
    return json.loads(raw)


def frozen(original):
    v = manifest['sources'][original]
    return read(E / v['file'], v['sha256'])


dense = frozen('/tmp/cheetah_wo23_20261106_v1/dense_lane_v1/delivery_v1/result.json')
cells = next(x for x in dense['models'] if x['model'] == 65)['cells']
old = read(cells['H1_old']['path'], cells['H1_old']['sha256'])
new = read(cells['H1_new']['path'], cells['H1_new']['sha256'])
source_path = '/tmp/cheetah_wo23_20261106_v1/dense_lane_v1/65/mapping_source_v1/source.json'
source = read(source_path, new['bindings'][source_path])
assert old['passed'] and new['passed'] and old['hardware'] == new['hardware']
h = new['hardware']
assert (len(OLD_TILES), len(TILES)) == (14, 25)
old_min = exact_minimum(source, h, allowed_tiles=set(range(len(OLD_TILES))))
new_min = exact_minimum(source, h)
assert F(old_min['decode_ns']) == F(old['phase_ns']['decode'])
assert F(new_min['decode_ns']) == F(new['phase_ns']['decode'])
extra = F()
count = 0
for z in source['descriptors']:
    if not z['eligible_weight']:
        continue
    ext, stage = dma(z['address'], z['bytes'], h['host_interface_units'],
                     h['staging_bytes'], h['staging_banks'])
    uses = int(z['decode_nonresident_count']) - int(z['decode_resident_count'])
    assert uses >= 0
    extra += uses * (F(ext + 2 * stage) + F(z['bytes'], h['bandwidth']))
    count += 1
assert count == old['selected_weight_count'] == new['selected_weight_count']
weight = read(P / 'update/paper_evidence_20261106/figure3_pies_v1/replays/llama_weights_v2/producer/result.json')
assert extra == F(weight['without_weight_retention']['signed_saved_ns'])
old_off = F(old_min['decode_ns']) + extra
new_off = F(new_min['decode_ns']) + extra
assert new_off == F(weight['without_weight_retention']['decode_ns'])
assert old_off - new_off == F(old_min['decode_ns']) - F(new_min['decode_ns'])
for name in ['wo23_dense_banks_geometry_v2.py', 'wo23_dense_mapping.py',
             'wo22_dense_staging.py', 'wo21_dense_family_rect.py']:
    p = S / name
    bindings[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
result = dict(
    passed=True, optimizer_calls=0, model='llama1_65b',
    reference_ns=source['reference']['decode_ns'],
    old14_weight_off_ns=str(old_off), new25_weight_off_ns=str(new_off),
    new25_weight_on_ns=new_min['decode_ns'],
    mapping_saved_ns=str(old_off-new_off), weight_saved_ns=str(extra),
    eligible_descriptors=count, hardware=h, bindings=bindings,
    scope='Exact original kernel pricing plus every original descriptor import, setup, two staging passes and DRAM write; original KV retention and boundaries stay fixed. No new source-lifetime admission or physical qualification.',
    additive_import_proof='Descriptor import counts and DMA service depend on descriptor and hardware only. Disabling weight retention releases capacity, leaves each kernel B read and its mapping cost unchanged, and cannot increase the retained-owner peak.',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT / 'llama65_sequence.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed','old14_weight_off_ns','new25_weight_off_ns','new25_weight_on_ns','eligible_descriptors']}))
