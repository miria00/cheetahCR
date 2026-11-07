# Four-decoder feature pies

Current figure: DeepSeek-V3, Llama-1 13B, Llama-1 65B and Llama-3 70B, all
decode. Each has four feature slices. Figure 2 is unchanged; the headline
ratios here use the latest phase-specific table beneath it.

The user explicitly permits overlapping effects for a visual estimate. We
normalize the positive analytical control time savings to exactly 100%, and
round the displayed percentages to sum to 100.00%. This provides relative
feature weights, not an exhaustive causal decomposition of the full Baseline 2
speedup. Applying those weights to the gain above 1x would be an illustrative
allocation only. No remaining-runtime wedge is present.

13B and 65B have hardware-change, tiling-extension, weight-residency and
KV-retention controls. The 13B controls were freshly repriced on its current
WO24 chip, with the original indexed KV copy and descriptor service functions.
70B's fourth control restores independent query-head matrix work instead of
group-eight KV-head reuse, using all 25 mappings on the current chip. This
is GQA read reuse, not KV retention/offload. Original reduction order, owners,
output coordinates and stores stay fixed. Semantic proofs are inherited.

DeepSeek has hardware-change, producer-layout fusion, cross-wave prefetch and
two-accumulator-context controls. Prefetch and second-context gains are an
ordered split of the checked combined delivery effect; producer fusion is
conditional on delivery. Its base resident arenas remain fixed/inherited.
A fifth nonzero slice is not invented where no separate control exists.

New numerical optimizer calls: zero. The bounded CPU checks reprice existing
source contracts, not a new hardware search or physical/full-source admission.
The 65B old hardware comparison retains its over-area qualification.

Current generator: `update/generate_figure3_decoder_pies.py`. The standard
`generate_figure3_pies.py` entry point invokes it. The twelve-row table also
includes the new 13B and 70B controls. Earlier pie versions remain preserved.
