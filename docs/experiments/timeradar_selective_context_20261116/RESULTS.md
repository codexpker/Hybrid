# Selective-context fixed development results

Completed:8 training runs of2000 updates,8 CAL/QUERY scoring runs,8 selected-checkpoint FIT diagnostics. Original comparison rows reused. Seed2024 only.

## Decision

Limited NO-GO for this fixed soft-weighting recipe. Selection changes actual context, decoder output and native L1 scores, but does not establish stable independent detection benefit over mean Cross and the matched capacity control. Stop this recipe without further local variants. Preserve the TimeRadar foundation and earlier local positives.

## Aggregation ablation (native full axis)

| Record | Arm | AP | ROC | C3 normal FPR (%) | C3 anomaly coverage (%) | C3 event F1 (%) |
|---|---|---:|---:|---:|---:|---:|
| PSM | self | 0.426359 | 0.678390 | 0.4602 | 1.9472 | 8.6207 |
| PSM | cross | 0.425339 | 0.680241 | 1.2418 | 4.2331 | 5.4455 |
| PSM | selective | 0.423699 | 0.680241 | 1.2387 | 4.2143 | 5.5838 |
| PSM | capacity_self | 0.426409 | 0.678637 | 0.4413 | 1.9284 | 9.3617 |
| machine-1-1 | self | 0.603667 | 0.905258 | 3.1004 | 46.3994 | 9.6386 |
| machine-1-1 | cross | 0.616477 | 0.905044 | 3.2522 | 49.7773 | 8.9385 |
| machine-1-1 | selective | 0.619314 | 0.905594 | 3.2288 | 49.5546 | 8.8889 |
| machine-1-1 | capacity_self | 0.603850 | 0.905252 | 3.0849 | 46.1767 | 9.6386 |
| machine-1-2 | self | 0.298940 | 0.922488 | 0.0997 | 16.4207 | 34.4828 |
| machine-1-2 | cross | 0.303594 | 0.922703 | 0.0911 | 17.1587 | 34.4828 |
| machine-1-2 | selective | 0.299965 | 0.921148 | 0.0867 | 16.2362 | 35.7143 |
| machine-1-2 | capacity_self | 0.299699 | 0.923056 | 0.1084 | 16.9742 | 33.3333 |
| machine-1-3 | self | 0.383640 | 0.887578 | 0.0000 | 5.3856 | 15.3846 |
| machine-1-3 | cross | 0.385731 | 0.889472 | 0.0000 | 5.5080 | 15.3846 |
| machine-1-3 | selective | 0.384068 | 0.889327 | 0.0087 | 5.3856 | 14.2857 |
| machine-1-3 | capacity_self | 0.383687 | 0.887712 | 0.0000 | 5.3856 | 15.3846 |

Selective-minus-mean AP(pp): PSM -0.1640; SMD1-1 +0.2837; SMD1-2 -0.3629; SMD1-3 -0.1663. C3 anomaly-point coverage decreases in all four cases. SMD1-1 long steady coverage drops51.342% to49.978%; the PSM long-steady increase5.956% to6.107% is retained as a limited positive, not a general sustained-detection gain.

Both new arms have17424 adapter parameters and811034 target trainables, exactly matched; old Self/mean Cross have810522. The added512 Q/K parameters are active. Fixed source-use checks hold encoder representations and RNG constant, remove donor0 values or force uniform weights, and observe changes in actual reconstructions/native scores. No causal/lag or irrelevant-channel-removal claim follows.

## Complete tables

- [Complete native comparison:24 rows](native_full_4x6.csv).
- [Matched external comparison:32 rows, including LEFT and HistoryMean512](matched_external_4x8.csv).
- [Aggregation ablation:16 rows](aggregation_ablation.csv).
- [Native paired differences](paired_native_differences.csv); [external paired differences](paired_external_differences.csv).

All strict/C3 point metrics, any-hit and one-to-one events, false-positive points/fragments, sustained coverage and longest misses remain in the tables. Event-level timing/UNKNOWN details remain in the raw archive and its112 hashed references. Native full and external common axes remain separate; all TimeRadar/LEFT scores are offline, not strict online alarms. LEFT incomplete SMD curriculum is unchanged and disclosed in the previous matched comparison. Same data permissions do not imply equal initialization or training budget. PSM and the three SMD machines are exposed development cases; one seed is not repeatability confirmation.

## Execution and verification

Actual new compute1.649843907488717 GPUh, including four failed initial FIT receipt-writing jobs(.0073196643h), eight successful FIT checks(.0126627909h), formal training/scoring and selected-FIT diagnostics. The only repair renamed a duplicate descriptive receipt field; technical run2 changes only output identity, not the scientific recipe. Failed originals are retained. No old models were retrained or rescored. CPU evaluation1.172s.

Bounded independent-responsibility audit artifact is present and SHA-verified:5d65ccd259738ca5633969d341e07465c6c05e18e53fdce5f8c1403babc22a5a. It covers16 raw banks, selected/final frozen states, paired initialization/FIT plans/mask streams, own CAL thresholds, selected-FIT source-use evidence,16 new evaluation rows and40 reused rows. Same-model-family responsibility separation, not external blinded review. The auditor later hit an account usage limit after delivering this artifact; no additional audit work is claimed.

No new variant, seed, dataset, threshold, lag mechanism or pollution/F experiment was started. Raw data, weights, private paths and upstream third-party source are excluded from this public result bundle. Public CSV uses LF normalization; raw files are unchanged.
