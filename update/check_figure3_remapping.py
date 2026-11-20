"""Check whether compatible remapping explains the resource-attribution slices.

Each resource increase retains the previously selected mappings first, then
allows the existing compatible menu to be reselected. These are exact analytical
controls, not new optimizer calls or full execution/physical admission.
"""
from pathlib import Path
from fractions import Fraction as F
import ast
import hashlib
import json
import sys

P = Path(__file__).resolve().parents[1]
S = P.parent/'systems/cheetah_joint'
OUT = P/'update/paper_evidence_20261106/figure3_pies_v8'
raw = (OUT/'inputs/resource_controls.json').read_bytes()
manifest = json.loads((OUT/'input_manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest() == manifest['sources']['resource_controls.json']['sha256']
old = json.loads(raw)
for name in ['wo23_dense_banks_geometry_v2.py', 'wo24_dense_banks.py',
             'wo23_dense_mapping.py', 'wo24_dense_mapping.py', 'wo22_dense_staging.py']:
    q = S/name
    assert hashlib.sha256(q.read_bytes()).hexdigest() == old['bindings'][str(q)]
sys.path.insert(0, str(S))
from wo23_dense_banks_geometry_v2 import exact_minimum as m25
from wo24_dense_banks import exact_minimum as m31
from wo23_dense_mapping import TILES as T25
from wo24_dense_mapping import TILES as T31
from wo22_dense_staging import dma

# Reuse the independent arithmetic checker without executing the original
# campaign-producing module. Bind the exact function text in this receipt.
checker = P/'update/replay_figure3_resources.py'
tree = ast.parse(checker.read_text())
fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'independent_cost')
scope = dict(F=F, dma=dma)
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(checker), 'exec'), scope)
cost = scope['independent_cost']
results = {}
bindings = {}
for n in (13, 65, 70):
    key = f'llama{3 if n == 70 else 1}_{n}b'
    audit = old['hardware_audits'][key]
    h = audit['installed_hardware']
    path = Path(f'/tmp/cheetah_wo{24 if n == 13 else 23}_20261106_v1/dense_lane_v1/{n}/mapping_source_v1/source.json')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == old['bindings'][str(path)]
    bindings[str(path)] = old['bindings'][str(path)]
    source = json.loads(raw)
    mini, tiles = (m31, T31) if n == 13 else (m25, T25)
    panel = next(x for x in old['panels'] if x['workload'] == key)
    ref, gain = F(panel['reference_ns']), F(panel['final_speedup_exact'])-1
    prevh = dict(h, **audit['resource_caps'])
    prev = mini(source, prevh)
    checked, count, payload = cost(source, prevh, tiles, prev['selected_tiles'])
    assert checked == F(prev['decode_ns']) == F(audit['all_eight_cells']['0']['time_ns'])
    mapping = F(panel['stages'][1]['added_speedup_exact'])
    rows = []
    for mask, resource in [(1, 'tiles'), (3, 'bandwidth'), (7, 'local_GBps')]:
        newh = dict(prevh)
        newh[resource] = h[resource]
        opt = mini(source, newh)
        fixed, count_fixed, bytes_fixed = cost(source, newh, tiles, prev['selected_tiles'])
        best, count_best, bytes_best = cost(source, newh, tiles, opt['selected_tiles'])
        assert count == count_fixed == count_best
        assert payload == bytes_fixed == bytes_best
        assert best == F(opt['decode_ns']) == F(audit['all_eight_cells'][str(mask)]['time_ns'])
        before = F(prev['decode_ns'])
        assert before >= fixed >= best
        hardware, remap = ref/fixed-ref/before, ref/best-ref/fixed
        assert hardware+remap == ref/best-ref/before
        mapping += remap
        rows.append(dict(resource=resource, before_ns=str(before), fixed_mapping_ns=str(fixed),
            remapped_ns=str(best), hardware_added_speedup=str(hardware), remap_added_speedup=str(remap),
            remapping_percent_of_total_gain=100*float(remap/gain),
            original_selected_tiles=prev['selected_tiles'], remapped_selected_tiles=opt['selected_tiles'],
            changed_profiles=sum(a != b for a, b in zip(prev['selected_tiles'], opt['selected_tiles']))))
        prevh, prev = newh, opt
    assert F(prev['decode_ns']) == F(panel['final_ns'])
    results[key] = dict(rows=rows, menu_extension_and_remapping_percent=100*float(mapping/gain),
        original_decode_invocations=count, timed_HOST_bytes=payload,
        installed_SRAM_MiB=h['sram_mib'], SRAM_capacity_held_fixed=True)

result = dict(passed=True, workloads=results, source_bindings=bindings,
    checker_sha256=hashlib.sha256(checker.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    resource_control_sha256=manifest['sources']['resource_controls.json']['sha256'],
    optimizer_calls=0, new_source_admission=False, new_physical_validation=False,
    conclusion='Remapping does not explain the large bandwidth shares. Even including compatible '
               'remapping, the isolated menu/mapping increments remain small. A larger display group '
               'must explicitly include another measured resource or feature; SRAM capacity was fixed.')
(OUT/'remapping_check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v['menu_extension_and_remapping_percent'] for k, v in results.items()}, indent=2))
