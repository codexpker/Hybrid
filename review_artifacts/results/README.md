# Public aggregate results for code review

Start at `../EXPERIMENT_INDEX.md` and the machine-readable JSON/CSV index. This directory contains selected **aggregate** results from public benchmarks, not a new evaluation or a consolidated leaderboard.

- `*.metrics.json`: copied scalar fields from existing method summaries; no threshold or metric was recomputed.
- `*.highlights.json`: small transcriptions of key original report numbers with the exact source-file SHA256 and scope.
- `*__summary.json`, `*__per_seed.json`, `*__paired_seed_differences.json`: narrow public aggregate-column exports. Original seed2026 positive results are retained alongside negative seeds. `M`/one-to-one event metrics and `auxiliary_any_hit_*` are distinct definitions, not interchangeable.
- `EXPORT_MANIFEST.json`: output-file SHA256; the index maps each export to a logical source path and source SHA256. Raw sources remain outside this Git package.

Training permissions differ: source-only Z, target calibration, target-U, event-supervised F, and target-normal full-shot must remain separate. Native TimeRadar/LEFT offline scores cannot be represented as strict online alarms merely by clipping their time axis. Three training seeds on one entity are not three independent domains. Historical exposure is disclosed in each relevant entry.

Both final 2026-11-13 candidates are stopped, bounded NO-GO results: whole-channel blindspot training/scoring and mask-auxiliary training/native inference. The SMD confirmation's seed2026 improvement is real but the primary AP difference is negative in two of three seeds. No stopped version is labeled a validated current Ours. More general research hypotheses are not disproven by these local failures.

No model weights, raw observations, labels, per-point score banks, restricted upstream implementations, private application material, infrastructure configuration or credentials are included. Current related code paths are navigation links, not claims of byte-identical historical implementation; exact source reports and hashes preserve the experiment identities.
