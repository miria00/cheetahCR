# Generate paper tables from preserved result records; no solver calls.
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, re

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
U = P / 'update'
E = U / 'paper_evidence_20261106'
E.mkdir(exist_ok=True)
sources = {}
def read_json(path):
    path = Path(path)
    # Freeze each report input on first use; later regeneration uses that copy.
    key = hashlib.sha256(str(path).encode()).hexdigest()[:12]+'_'+path.name
    saved = E/'inputs'/key
    saved.parent.mkdir(exist_ok=True)
    if not saved.exists():
        saved.write_bytes(path.read_bytes())
    data = saved.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)
def write(name, text):
    (U / name).write_text(text.strip()+'\n')
def take_table(text, label):
    for m in re.finditer(r'\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}', text, re.S):
        if '\\label{'+label+'}' in m.group():
            return m.group()
    raise ValueError(label)

records={}
names=[('efficientnet_b'+str(i),'EfficientNet-B'+str(i)) for i in range(1,8)]
names += [('unet','SD1.5 UNet'),('llama1_13b','Llama-1 13B'),('llama1_65b','Llama-1 65B'),('llama3_70b','Llama-3 70B'),('deepseek_v3','DeepSeek-V3')]
for key,_ in names:
    records[key]=read_json(ROOT/'Results/Blueprints/WO24/records'/f'{key}.json')
prefill={'llama1_13b':24.195974,'llama1_65b':14.432646,'llama3_70b':20.318050,'deepseek_v3':10.348224}
prefill_source=ROOT/'systems/cheetah_joint/audits/20261105/WO20_WORKLOAD_STATUS.md'
prefill_saved=E/'inputs'/'WO20_WORKLOAD_STATUS.md'
if not prefill_saved.exists(): prefill_saved.write_bytes(prefill_source.read_bytes())
sources[str(prefill_source)] = hashlib.sha256(prefill_saved.read_bytes()).hexdigest()
for val in prefill.values(): assert f'{val:.6f}' in prefill_saved.read_text()
rows=[]
for key,label in names:
    ratio=float(records[key]['reference_ratio'])
    if key in prefill: vals=['--',f'${prefill[key]:.6f}\\times$',f'${ratio:.6f}\\times$']
    else: vals=[f'${ratio:.6f}\\times$','--','--']
    rows.append(label+' & '+' & '.join(vals)+r' \\')
write('wo24_suite_summary.tex',r'''
% Sources and derived values: paper_evidence_20261106/values.json.
\begin{table}[H]
\centering\small
\caption{Predicted acceleration over unchanged paper Baseline 2 (modeled
TPU-v3). Latest decode-only research designs and previously checked prefill
designs are \emph{different chips}; the two phase columns are not one serving
design or an end-to-end speedup. Dashes mean inapplicable phases.}
\label{tab:current_phase_speedups}
\begin{tabular}{@{}lrrr@{}}
\toprule
Workload & Forward / denoiser & Prefill & Decode \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
All cells use matrix-and-associated-traffic accounting and retain the paper's
historical precision/timing proxy; they are not measured or precision-faithful
TPU-v3 serving results. Prefill: corrected WO18 Llamas and WO20 DeepSeek A512.
Decode: WO24, retaining the WO23 65B/70B executions. Vision: inherited WO14
EfficientNets and WO23 UNet. UNet is one batch-1, $64\!\times\!64$-latent,
context-77 denoiser call. All LLMs have 32 requests, 512 outputs and 511 decode
forwards; Llama-1 prompts have 1,536 tokens, the others 4,096.
Decode chips use relaxed memory/power assumptions and area proxies below
600~mm$^2$; physical chip qualification remains open.
\end{minipage}
\end{table}
''')

# All three Section 6.2 quantities, calculated on the exact same decode interval.
eff=[];effrows=[]
for key,label in [('deepseek_v3','DeepSeek-V3')]+names[8:11]:
    d=records[key];m=d['metrics'];h=d['hardware'];t=F(d['latency_ns'])/10**9
    power=F(str(d['chip_power_proxy_W']))
    if key=='deepseek_v3':
        traffic=m['totals'];b=[traffic[k] for k in ['DRAM_service_bytes','HOST_payload_bytes','local_service_bytes']]
        mfu=F(str(m['useful_BF16_MFU']))
    else:
        traffic=m['traffic_decode'];b=[traffic[k] for k in ['HBM','HOST','local_kernel']]
        mfu=F(str(m['full_decode_useful_MFU']))
    rates=[h[k] for k in ['bandwidth','host_GBps','local_GBps']]
    mbu=[F(v)/(F(str(rate))*10**9*t) for v,rate in zip(b,rates)]
    perwatt=F(16352)/t/power
    assert 0<=mfu<=1 and all(0<=x<=1 for x in mbu)
    eff.append(dict(model=key,seconds=str(t),mfu=str(mfu),mbu_dram_host_local=list(map(str,mbu)),chip_power_W=str(power),decode_tokens=16352,tokens_per_s_per_chip_proxy_W=str(perwatt)))
    effrows.append(label+' & '+f'{float(mfu)*100:.3f}'+' & '+' / '.join(f'{float(x)*100:.3f}' for x in mbu)+' & '+f'{float(power):.3f} & {float(perwatt):.3f}'+r' \\')
write('wo24_efficiency.tex',r'''
\begin{table}[htbp]
\centering\small
\caption{Numerical decode efficiency for the selected research chips.
MFU and MBU are percentages; the final column is decode tokens/s per watt
of the \emph{chip-power proxy}, not complete-system performance/TDP.}
\label{tab:decode_efficiency}
\begin{tabular}{@{}lrcrr@{}}
\toprule
Decoder & MFU & MBU: DRAM / host / local & Proxy W & Tokens/s/W \\
\midrule
'''+ '\n'.join(effrows)+r'''
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
$Q=32\cdot511=16{,}352$ newly produced decode tokens. Denominators use
the same matrix-and-associated-traffic interval as
Table~\ref{tab:current_phase_speedups}. MBU divides charged payload by the
installed delivered service rate and time; it excludes protocol padding and
unpriced standalone nonmatrix service. Dense local bytes cover the published
kernel ledger. DeepSeek's additional context-output local bytes are counted
once; FP32 router service is separate from BF16 MFU. Package/memory power is
incomplete, so no complete-system TDP or baseline-normalized energy-efficiency
claim follows from the final column.
\end{minipage}
\end{table}
''')

# Historical fixed-hardware objective comparisons: only supported cells.
R=Path('/home/miria/CHEETAH/rebuttal/grid_completion')
en=read_json(R/'effnet_cells.json'); anchors=read_json(R/'highs_anchors.json')['cells']
hi=read_json(R/'highs_highbw_cells.json'); pd=read_json(R/'pdlp_cells.json')
fixed={}
for c in en['cells']: fixed[(c['workload'],'tpuv3')]=dict(ratio=c['T_naive']/c['T_joint'],source='effnet_cells.json',cell=c['cell_key'],scope=c['convention_tag'])
for c in anchors:
    if c['cell'].endswith('@tpuv3(900,32)/joint'):
        fixed[(c['cell'].split('@')[0],'tpuv3')]=dict(ratio=c['naive_sum_Tmax']/c['objective'],source='highs_anchors.json',cell=c['cell'],scope=c['convention'])
for cs,fn in [(pd,'pdlp_cells.json'),(hi,'highs_highbw_cells.json')]:
    for c in cs:
        cell=c.get('cell_key','')
        if c.get('row')!='joint':continue
        hw=c.get('template');key=c.get('workload')
        if hw=='compact_npu' and '__bw64_cgm16_ref64_joint' in cell or hw=='tpuv3' and '__bw900_cgm32_ref900_joint' in cell:
            # Preserve the original LLM TPU anchors' own denominator convention.
            if hw=='tpuv3' and key.startswith(('llama','deepseek')):continue
            fixed[(key,hw)]=dict(ratio=c['T_naive']/c['T_joint'],source=fn,cell=cell,scope=c.get('convention_tag'),gap=c.get('gap_tag'))
fixedrows=[]
for key,label in names:
    k='sd15_unet' if key=='unet' else key
    vals=[]
    for hw in ['tpuv3','compact_npu']:
        v=fixed.get((k,hw));vals.append(f"${v['ratio']:.3f}\\times$" if v else 'NR')
    fixedrows.append(label+' & '+' & '.join(vals)+r' \\')
write('wo24_fixed_hardware.tex',r'''
\begin{table}[htbp]
\centering\small
\caption{Fixed-hardware software-search comparisons for all twelve model
families. These are \emph{historical LP-objective ratios} against streaming
on the same modeled chip, under each stored cell's original convention;
they are not new decode measurements or the co-design ratios of
Table~\ref{tab:current_phase_speedups}. NR: no supported stored result.}
\label{tab:fixed_hw}
\begin{tabular}{@{}lrr@{}}
\toprule
Workload & Modeled TPU-v3 & Compact NPU \\
\midrule
'''+ '\n'.join(fixedrows)+r'''
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
TPU-v3: 900~GB/s, 32~MiB; compact NPU: 64~GB/s, 16~MiB, 32,768 historical
INT8 MAC units at 1~GHz. Ratios are $T_{\rm streaming}/T_{\rm joint\,LP}$,
not integral execution certificates. Compact-NPU B7/13B/DeepSeek use stored
iterative objectives (reported relative gap tags 0.020/0.125/0.019);
other shown cells use stored HiGHS optima. EfficientNet TPU cells retain the
historical single-predecessor graph/workspace approximation. No compact-NPU
B1--B6/65B result or current full-decode fixed-hardware replay is supplied;
no cross-convention geometric mean is formed.
\end{minipage}
\end{table}
''')

dataset=read_json('/tmp/cheetah_wo24_20261106_v1/authoritative_dataset_v1/result.json')
rtl_receipt=read_json('/tmp/cheetah_wo24_20261106_v1/postseal_actual_v1/result.json')
ds_ab=dataset['ablations']['deepseek']
ds_branch=next(b for b in ds_ab['branches'] if b['hardware']==records['deepseek_v3']['chip_id'])
ll_ab=dataset['ablations']['llama13_old25_menu_reoptimization']['factorial_ns']
ablation_values=[
    (F(ds_branch['producer_saving_given_delivery_ns']), '1.348604', '1.021076'),
    (F(ds_branch['delivery_saving_given_producer_ns']), '4.210287', '1.065797'),
    (F(ds_branch['additional_two_context_saving_ns']), '2.870741', '1.044863'),
    (F(ll_ab['software_at_newHW']), '0.059881', '1.005735'),
]
for i,(saving,seconds,factor) in enumerate(ablation_values):
    on=F(records['llama1_13b' if i==3 else 'deepseek_v3']['latency_ns'])
    assert f'{float(saving/10**9):.6f}'==seconds
    assert f'{float((on+saving)/on):.6f}'==factor
assert f"{float(F(ds_ab['hardware_software_interaction_ns'])/10**6):.6f}"=='-107.107212'
assert f"{float(F(ll_ab['interaction_B_plus_C_minus_A_minus_D'])/10**6):.6f}"=='-0.633227'

write('wo24_features.tex',r'''
\begin{table}[H]
\centering\footnotesize
\caption{Encoded decision families and actual recovery coverage. The master
formulates broader freedoms than the latest decoder implementation menus.
``Fixed'' means an active inherited policy, not a free selector.}
\label{tab:current_features}
\begin{tabularx}{\linewidth}{@{}p{0.22\linewidth}XX@{}}
\toprule
Feature & DeepSeek decode & Llama decode \\
\midrule
Hardware resources & Selectable finite whole-chip branches & Selectable finite whole-chip branches \\
Implementation / tiling & Independent compatible full-K choices & Full-K choices; 13B menu grows from 25 to 31 \\
Activation placement & Resident arena fixed; rounded producer layout selectable & Original DRAM arenas and boundaries fixed \\
Weight residency & Original resident backing fixed; loads paid & Eligible whole-owner retention selectable; loads paid \\
Fusion / storage & Rounded attention producer-to-AV layout selected; inherited gate/up and SwiGLU & Inherited boundaries; no new fusion recipe \\
Tiling--placement coupling & Footprint, ports and shared resources checked; general lifetime exchange absent & Same restriction \\
KV retention & Original owners, appends and entry fixed and charged & Original owners/appends charged \\
MoE residency / offload & Original routing, groups and resident parameters fixed & Not applicable \\
Rematerialization & Formulated; new decoder implementation absent & Formulated; new decoder implementation absent \\
Scheduling / concurrency & Cross-N prefetch and two contexts selectable; arbitrary overlap absent & Ordinary serial DRAM kernels \\
Serving / phase choice & Decode-only; original request order and starting KV fixed & Same restriction \\
\bottomrule
\end{tabularx}
\medskip
\begin{tabular}{@{}llrr@{}}
\toprule
Workload & Disabled choice (other choices retained) & Time saved, s & Factor \\
\midrule
DeepSeek & Rounded producer-layout storage fusion & 1.348604 & $1.021076\times$ \\
DeepSeek & New delivery / concurrency recipes & 4.210287 & $1.065797\times$ \\
DeepSeek & Second context; prefetch still available & 2.870741 & $1.044863\times$ \\
Llama-1 13B & Six new full-K mappings & 0.059881 & $1.005735\times$ \\
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
Same-chip controls reoptimize the declared feature-off menu. Saved time is
$T_{\rm off}-T_{\rm on}$ and factor is $T_{\rm off}/T_{\rm on}$, with
$T_{\rm on}=63.988923$~s (DeepSeek) or $10.441348$~s (13B).
The second-context control is part of the delivery family, so these gains
must not be added. Fixed, unavailable and unisolated features receive no
invented individual speedup credit. Compatible alternatives remain selectable;
unselected alternatives are not declared absent.
\end{minipage}
\end{table}

\paragraph{Hardware--software interaction.}
For DeepSeek, old/old, old-hardware/new-software, new-hardware/old-software,
and new/new take 69.694835, 64.028837, 69.547814 and 63.988923~s.
The corresponding 13B times are 10.687989, 10.627475, 10.501228 and
10.441348~s. Both hardware-only minima choose the new chip within the
declared menu. The signed interaction
$I=T_{10}+T_{01}-T_{00}-T_{11}$ is $-107.107212$~ms (DeepSeek) and
$-0.633227$~ms (13B): combined savings are slightly less than the sum of the
separate changes. These finite-family controls do not certify unrestricted
placement, rematerialization or scheduling optima.
''')

values=dict(source_sha256=sources,performance=[dict(model=k,label=n,prefill_ratio=prefill.get(k),current_ratio=records[k]['reference_ratio'],current_phase='decode' if k in prefill else 'forward_or_denoiser') for k,n in names],efficiency=eff,ablations_checked_against_exact_source=True,ablation_savings_ns=[str(x[0]) for x in ablation_values],fixed_hardware={k+'@'+hw:v for (k,hw),v in fixed.items()},archived_tables=[6,7,10],deleted_tables=[4,5])
(E/'values.json').write_text(json.dumps(values,indent=2)+'\n')
print('Generated five current tables; exact ablation checks pass; evidence:',E)
