# Workload status after WO20

The later [decode-only native-offload appendix](WO20_NATIVE_OFFLOAD.md)
qualifies A288 at 35.245789× against an explicit 32-GB/s eager host reference,
or 17.883713× with authenticated ready-group reuse on that reference. Those
conditional native-operation ratios use a different model and boundary from
the unchanged Baseline 2 table below. A288 remains 0.383180× under Baseline 2.
No new native-objective optimum or unrestricted-master certificate is claimed.
The user's nine-row hardware/software feature table is reconciled with the
actual selectable and missing choices in [the menu crosswalk](WO20_FEATURE_MENU.md).
The subsequent [decode/denoiser hardware and software specifications](WO20_DECODE_HARDWARE_SOFTWARE.md)
add independently checked native accounting for all three Llamas and a fresh
same-result UNet audit. They add no new native-objective optimization; the
stronger historical Llama cohort controls remain unadopted because their
phase order differs. UNet's full native adapter remains incomplete.

The twelve input families are seven EfficientNets, the Stable Diffusion UNet,
three dense Llamas and DeepSeek V3. Ratios below mean the approved paper
Baseline 2 reference time divided by the custom-chip time. They are analytical
paper-method comparisons, not measured TPU-v3 performance. The historical
8-bit mapper proxy and approved serving exceptions remain; this is not a
precision-faithful BF16/FP32 TPU-v3 benchmark or end-to-end serving comparison.

| Input family | Forward | Prefill | Decode | Latest scope |
|---|---:|---:|---:|---|
| EfficientNet B1 | 11.848610× | — | — | Inherited WO14 summary |
| EfficientNet B2 | 6.679989× | — | — | Inherited WO14 summary |
| EfficientNet B3 | 11.427334× | — | — | Inherited WO14 summary |
| EfficientNet B4 | 9.620539× | — | — | Inherited WO14 summary |
| EfficientNet B5 | 9.936808× | — | — | Inherited WO14 summary |
| EfficientNet B6 | 7.803334× | — | — | Inherited WO14 summary |
| EfficientNet B7 | 7.799624× | — | — | Inherited WO14 summary |
| Stable Diffusion UNet | 2.201870× | — | — | One denoiser call; WO14 point rechecked in the WO20 appendix |
| Llama-1 13B | — | 24.195974× | 2.805212× | Corrected WO18, +15% power/shared-pin scenario |
| Llama-1 65B | — | 14.432646× | 1.274628× | Corrected WO18, +15% power/shared-pin scenario |
| Llama-3 70B | — | 20.318050× | 2.717197× | Corrected WO18, +15% power/shared-pin scenario |
| DeepSeek V3 | — | 10.348224× | 0.346338× | WO20 A512, hard 10× prefill constraint, +15% scenario |

CNN and UNet tasks have forward latency, not autoregressive prefill/decode.
UNet is batch 1 with 64×64 latent and context length 77; its time is not full
image generation. All LLM tasks have 32 actual requests and 512 outputs;
prefill produces the first output and decode contains 511 forwards. Llama-1
prompts have 1,536 tokens; Llama-3 and DeepSeek have 4,096. The CNN figures
retain their historical contracts and were not revalidated in WO18–WO20.
The later WO20 appendix rechecks the unchanged UNet point and its finite
certificate; it does not qualify full native UNet accounting.
The corrected WO18 Llama points replace the older WO14 serving figures for
the current accounting. Different networks use different selected chips;
within a serving pair one chip implements both phases.

Sources: [WO14](WO14_DELIVERY.md), [corrected WO18](WO18_DELIVERY.md), and
[WO20](WO20_DELIVERY.md). This summary joins published receipts and reports;
it does not rerun those earlier source checks.

DeepSeek's fastest admitted decode alternative in WO20 is A288: **8.982745×
prefill / 0.383180× decode**. It misses the strict prefill target. A512's
310,744.685663-ms prefill and 1,445,285.188862-ms decode protect prefill at a
decode cost. Its decode takes about 2.887 times the reference time. These are
paired alternatives, not phase values that can be combined into one chip.
The strict **10× in both phases remains unmet**. The larger historical
Table 10 decode ratio uses a different reference; see the
[reference audit](WO20_TABLE10_AUDIT.md).

OCS now solves a genuinely coupled clustered program for DeepSeek, but it is
**not the unrestricted full joint master LP**. WO19 demonstrated a
34.58-million-variable, 32-window program that fit the A100 memory budget;
that finer run did not meet the 0.1% objective criterion. The best admitted
eight-window family supplied the continuing incumbent. Memory fit did not
create missing source-qualified implementations.
The window refinement was manually chosen, not an automatic sensitivity-driven
clustering controller or a complete clustered image of every master decision.

WO20 couples seven declared whole-chip alternatives with original backing,
receiver/mapping costs, typed weight inventory and paid first loads, conditional
K caches, optional original V retention, workspace release and rounded
attention storage fusion. Each branch uses one chip across both phases and
its own phase constraints. The larger seven-chip controls had about 15.6
million variables. Under the final hard 10× prefill constraint, exact bounds
exclude six chips. The remaining fused program has **2,233,583 variables,
2,216,364 inequalities, one equality and 10,173,482 nonzeros**.

Its checked lower is 1,444,951.475178 ms decode; the admitted execution and
exact LP upper are 1,445,285.188862 ms. The relative interval is **0.023090%**.
This certificate covers that declared family. It combines an OCS dual freshly
repriced over the full new program with a separately recovered and source-
admitted feasible execution. It is not a claim that the raw floating-point
trajectory converged to the integral execution.

The adopted execution uses the paid workspace exchange, 1,203 weight stocks
per window and the 4-MiB rounded-attention fusion arena. Optional V retention
is available but selects zero. Fifteen expert mappings are qualified. Some
dominated mapping choices are eliminated exactly within fixed-state menus,
and complete prefill implementations are selected from finite templates.
They are not arbitrary independent per-event choices. The integrated expert
receiver remains serial receive followed by an ordinary DRAM kernel.

Authenticated expert memberships and cross-group order remain fixed. Temporal
policies are tied to eight windows; general scheduling, arbitrary lifetimes,
general rematerialization, arbitrary fusion, all layouts and unrestricted
hardware generation remain absent. Efficient paid encoded-weight delivery
and a broader source-qualified attention representation are unresolved.
The restricted optimum is still far from 10× decode; further numerical
tightening alone cannot close that gap.
