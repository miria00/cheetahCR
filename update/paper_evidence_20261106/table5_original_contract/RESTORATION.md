# Table 5: original-protocol restoration and Elements recovery

The user requires a like-for-like replacement and advance notice of any
methodology change. The one-template table5_precision_v3 experiment remains
preserved as an unadopted diagnostic. The paper now shows the original five
requested workloads and their two original-convention columns. The requested
four INT8/BF16 columns are not complete and must not be filled from that
diagnostic or from relabeled historical values.

The Elements cleanup archives supplied five original source modules, three
prepatch caches, three patched caches and the original 13B result receipt.
The eleven source/cache files all pass content checks against archive
manifests; the result receipt is separately verified. The local candidate
LLM caches were prepatch. Executing the original patch_pareto_npz.py
preprocessing reconstructs the archived patched caches byte-for-byte.

Fresh original-protocol TPU LLM replay recovers 2.141924203x (13B),
2.508728535x (70B), and 2.218685813x (DeepSeek). The first is exactly the
stored objective; the latter two differ by about 1e-7 relative and reproduce
all displayed digits. Each passes the unchanged independent numerical
checker. Original B7 TPU replay is exactly 2.212671869x. UNet TPU replay is
1.498229852x, within 0.001989% of its stored objective; it also passes the
checker and retains the table's displayed value. These are fractional LP
results, not recovered native executions.

Ten CPU HiGHS calls were made, no GPU calls. The first seven include three
prepatch LLM attempts, which are rejected as original-result reproductions.
The two compact replay attempts have numerical stationarity/support-check
failures and are not adopted. Their differences from stored iterative
objectives are not attributed to precision. Three later calls use the
byte-verified original patched LLM caches.

The original compact mapping libraries for UNet, Llama-70B and DeepSeek
were not found in the inspected locations. All cleanup archive manifests
were searched. Broader mirror filename scans were bounded and partial;
this is not a claim that the files are absent everywhere on Elements.
No replacement library, new baseline, new SRAM constraint or new memory
floor was substituted. A matched precision extension needs the original
inputs and a documented precision-specific cost path.

Recovered producer scope: the TPU LLM source exposes its original 1--256
MiB CGM selector despite the stored 32-MiB chip label. This original rule
was retained and disclosed, not silently fixed. The vision and compact
V2.1 cells retain their own original capacity/traffic conventions.

All original numerical receipts, preprocessing attempts, source snapshots,
LP matrices, witnesses, checks and paper backups are retained. Paper
Figures 2 and 3 are byte-identical. The corrected paper builds to 88 pages
with Table 5 on page 12 and no undefined references or overfull boxes.
