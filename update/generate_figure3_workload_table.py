"""Generate the table below the retained Figure 3 from frozen experiment inputs."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
E = P / 'update/paper_evidence_20261106/figure3_latest12'
E.mkdir(parents=True, exist_ok=True)
sources = {}


def read(path):
    path = Path(path)
    name = hashlib.sha256(str(path).encode()).hexdigest()[:12] + '_' + path.name
    saved = E / 'inputs' / name
    saved.parent.mkdir(exist_ok=True)
    if not saved.exists():
        saved.write_bytes(path.read_bytes())
    payload = saved.read_bytes()
    sources[str(path)] = dict(file='inputs/' + name,
                             sha256=hashlib.sha256(payload).hexdigest())
    return json.loads(payload)


names = [('efficientnet_b' + str(i), 'EfficientNet-B' + str(i)) for i in range(1, 8)]
names += [('unet', 'SD1.5 UNet'), ('deepseek_v3', 'DeepSeek-V3'),
          ('llama1_13b', 'Llama-1 13B'), ('llama1_65b', 'Llama-1 65B'),
          ('llama3_70b', 'Llama-3 70B')]
records = {key: read(ROOT / 'Results/Blueprints/WO24/records' / (key + '.json'))
           for key, _ in names}
controls = {key: [] for key, _ in names}


def add(key, label, off, on, scope, group=None):
    off, on = F(off), F(on)
    record = records[key]
    # Controls must end at the current selected execution, not a different chip's point.
    expected = F(record['latency_ns']) if 'latency_ns' in record else F(str(record['latency_seconds'])) * 10**9
    assert abs(on - expected) <= F(1, 1000), (key, on, expected)
    controls[key].append(dict(label=label, group=group, off_ns=str(off), on_ns=str(on),
                             saved_ns=str(off-on), factor=str(off/on), scope=scope))


for number in (1, 3, 5, 7):
    key = 'efficientnet_b' + str(number)
    cnn = read(records[key]['historical_record']['source_receipt'])
    assert cnn['passed'] and all(v['passed'] for v in cnn['results'].values())
    for field in ('V', 'bw', 'memtech', 'pe_type', 'persistent_mib'):
        assert cnn['hardware_fix'][field] == records[key]['hardware'][field]
    assert cnn['hardware_fix']['tiles'] == records[key]['hardware']['tiles_int']
    add(key, 'Owners', F(str(cnn['results']['materialized']['scheduled_ms'])) * 10**6,
        F(str(cnn['results']['whole_owner']['scheduled_ms'])) * 10**6,
        cnn['scope'], 'storage')

b2 = read('/tmp/cheetah_wo14_20261105_v1/cnn/b2_actual_mode_response_v4/result.json')
assert all(v['passed'] for v in b2['results'].values())
add('efficientnet_b2', 'Owners/mode',
    F(str(b2['results']['materialized']['scheduled_ms'])) * 10**6,
    F(str(b2['results']['whole_owner']['scheduled_ms'])) * 10**6,
    b2['scope'], 'storage')

for number, path in [(4, '/tmp/cheetah_wo14_20261105_v1/cnn/b4_flattened_prepared_v4/family.json'),
                     (6, '/tmp/cheetah_wo14_20261105_v1/cnn/b6_admission/prepared_v4/family.json')]:
    family = read(path)
    add('efficientnet_b' + str(number), 'Spatial + buffers',
        family['old_total_ns'], family['all_new_total_ns'],
        'Joint mapping/buffering recipe on the retained fixed chip; no separate credits for its parts.', 'tiling')

unet = read('/tmp/cheetah_wo23_20261106_v1/unet_lane_v1/delivery_v1/result.json')
assert unet['passed']
uc = {(c['buffer'], c['cast']): c for c in unet['signed_controls']}
assert all(c['mapping_and_hardware_fixed'] for c in uc.values())
add('unet', 'Cast', uc[1, 0]['latency_ns'], uc[1, 1]['latency_ns'],
    'Buffering enabled; original output stores and rounding retained.', 'fusion')
add('unet', 'Buffers', uc[0, 1]['latency_ns'], uc[1, 1]['latency_ns'],
    'Cast/epilogue storage enabled; gains are conditional, not additive.', 'fusion')

dataset = read('/tmp/cheetah_wo24_20261106_v1/authoritative_dataset_v1/result.json')
ds = dataset['ablations']['deepseek']
branch = next(b for b in ds['branches'] if b['hardware'] == records['deepseek_v3']['chip_id'])
on = F(records['deepseek_v3']['latency_ns'])
for field, label in [('producer_saving_given_delivery_ns', 'Producer'),
                     ('delivery_saving_given_producer_ns', 'Delivery')]:
    add('deepseek_v3', label, on + F(branch[field]), on,
        'Same current chip, other feature retained, declared feature-off menu reoptimized.', 'fusion')
add('deepseek_v3', 'HW menu', ds['hardware_software_cells_ns']['old_hardware_new_software'], on,
    'Old versus new whole-chip branch with the new software menu available in both; hardware changes.', 'hardware')
fc = dataset['ablations']['llama13_old25_menu_reoptimization']['factorial_ns']
add('llama1_13b', '25 to 31 maps', fc['C'], fc['D'],
    'Old 25 versus new 31 mapping choices reoptimized on the selected WO24 chip.', 'tiling')
add('llama1_13b', 'HW menu', fc['B'], fc['D'],
    'Old versus new whole-chip branch with the new software menu available in both; hardware changes.', 'hardware')

dense = read('/tmp/cheetah_wo23_20261106_v1/dense_lane_v1/delivery_v1/result.json')
assert dense['passed']
for number, key in [(65, 'llama1_65b'), (70, 'llama3_70b')]:
    model = next(m for m in dense['models'] if m['model'] == number)
    off, on = (model['cells'][k] for k in ('H1_old', 'H1_new'))
    assert off['hardware'] == on['hardware'] == records[key]['hardware']
    add(key, '14 to 25 maps', off['decode_ns'], on['decode_ns'],
        'WO23 mapping-menu control on the same chip retained by WO24; other policies fixed.', 'tiling')
    add(key, 'HW menu', model['cells']['H0_new']['decode_ns'], on['decode_ns'],
        'WO23 old/new hardware branches with the new mapping menu in both; old 65B hardware exceeds the primary area bound.', 'hardware')
    feature = read(f'/tmp/cheetah_wo23_20261106_v1/dense_lane_v1/{number}/features_selected_v1/result.json')
    assert feature['passed'] and feature['hardware'] == records[key]['hardware']
    add(key, 'Weights', feature['without_weight_retention']['decode_ns'], feature['full_decode_ns'],
        'Same selected chip/mapping, restore all original nonresident weight imports and writes.', 'storage')
kv = read('/tmp/cheetah_wo23_20261106_v1/dense_lane_v1/65/KV_off_selected_v1/result.json')
assert kv['passed'] and kv['hardware'] == records['llama1_65b']['hardware']
add('llama1_65b', 'KV', kv['decode_without_KV_retention_ns'], kv['decode_full_ns'],
    'Same selected chip; canonical prefixes restored by qualified indexed-copy recipe; initial/tail writes still paid.', 'storage')

# Current-chip supplemental controls requested for the new four-decoder pies.
# The earlier frozen feature_groups_values.json is retained unchanged.
SUP = P / 'update/paper_evidence_20261106/figure3_pies_v3'
for number, key in [(13, 'llama1_13b'), (70, 'llama3_70b')]:
    extra = read(SUP / f'controls_{number}/result.json')
    assert extra['passed'] and extra['hardware'] == records[key]['hardware']
    for label, control in extra['controls'].items():
        if number == 70 and label == 'Weight residency':
            old = next(c for c in controls[key] if c['label'] == 'Weights')
            assert F(old['saved_ns']) == F(control['saved_ns'])
            continue
        short = {'Weight residency': 'Weights', 'KV retention': 'KV',
                 'KV-head reuse (GQA)': 'GQA'}[label]
        add(key, short, control['off_ns'], control['on_ns'], control['scope'],
            'tiling' if short == 'GQA' else 'storage')

groups = ('hardware', 'tiling', 'storage', 'fusion')


def cell(key, group):
    entries = [c for c in controls[key] if c['group'] == group]
    if entries:
        # Every number labels its own conditional control; do not combine factors.
        lines = [(c['label'].replace('14 to 25 maps', '14--25 maps') if len(entries)>1 else c['label']) + ' $' + f"{float(F(c['factor'])):.4f}" + r'\times$' for c in entries]
        if len(entries) == 1:
            lines = [entries[0]['label'], '$' + f"{float(F(entries[0]['factor'])):.4f}" + r'\times$']
        if key in ('efficientnet_b4', 'efficientnet_b6') and group == 'tiling':
            lines = [r'Spatial + buffers$^\dagger$',' $' + f"{float(F(entries[0]['factor'])):.4f}" + r'\times$']
        return r'\shortstack[l]{' + r'\\'.join(lines) + '}'
    if group == 'hardware':
        return 'Fixed; NR' if key.startswith('efficientnet') else 'Unchanged; NR'
    if group == 'tiling':
        if key == 'unet':
            return 'Matrix modes; NR'
        return 'Inherited; NR' if key.startswith('efficientnet') else 'Full-K; NR'
    if group == 'storage':
        return 'Owners; NR' if key.startswith('efficientnet') else ('Retention; NR' if key == 'llama1_13b' else 'Resident; fixed')
    if key in ('efficientnet_b4', 'efficientnet_b6'):
        return r'With tiling$^\dagger$'
    return 'No new recipe'

rows = []
for key, label in names:
    prefix = label + ' & $' + f"{float(records[key]['reference_ratio']):.3f}" + r'\times$'
    rows.append(prefix + ' & ' + ' & '.join(cell(key, group) for group in groups) + r' \\')
assert len(rows) == 12 and all(controls.values())

table = r'''% Generated by update/generate_figure3_workload_table.py; frozen evidence alongside it.
\begin{table}[H]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\caption{Per-workload contributions grouped by four types of LP decisions.
Each feature cell names the tested choice and its conditional speedup factor.
Total is the selected design's acceleration over unchanged Baseline 2
(modeled TPU-v3); the feature factors do not partition that total.}
\label{tab:figure3_breakdown}
\begin{tabularx}{\linewidth}{@{}l r *{4}{>{\raggedright\arraybackslash}X}@{}}
\toprule
Workload & Total & Hardware / datapath & Mapping / tiling & Placement / residency & Fusion / overlap \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabularx}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
EfficientNets: forward; UNet: one denoiser call; LLMs: decode. Software factors
are $T_{\rm off}/T_{\rm on}$ on the selected chip. Hardware factors compare
old/new chip branches with the same new software menu; the 65B old-chip control
exceeds the primary area bound. Mapping factors measure the stated menu
extension, not all tiling benefit. Owners denotes whole-tensor retention;
Producer denotes rounded attention-layout fusion; Delivery includes paid
prefetch/two contexts. Weights and KV are separate retention controls;
GQA measures private KV-head reuse, not KV retention or offload.
$^\dagger$B4/B6 mapping and buffering are one combined control, credited once.
Factors must not be added or multiplied; they are conditional, not an
exhaustive decomposition. NR means not isolated, not zero. Fixed/inherited
policies are not newly free selectors. Rematerialization belongs to the
placement/lifetime group, but no new decoder recipe or gain is established.
WO24 updates DeepSeek/13B; the other ten points retain earlier results.
All totals use analytical matrix-and-associated-traffic accounting and
separate chips; physical decoder qualification remains open.
\end{minipage}
\end{table}
'''
(P / 'update/figure3_table.tex').write_text(table)
(SUP / 'table_control_values.json').write_text(json.dumps(dict(sources=sources, groups=groups,
    performance={k: dict(reference_ratio=v['reference_ratio'], status=v['status']) for k, v in records.items()},
    controls=controls), indent=2) + '\n')
print(f'Wrote 12 workload rows, four feature columns, {sum(map(len,controls.values()))} controls; historical Figure 2 unchanged.')
