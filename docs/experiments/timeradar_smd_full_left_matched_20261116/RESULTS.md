# Frozen TimeRadar-native Cross adaptation: full SMD28 and matched LEFT results

**Completed / independently checked / primary accepted.** Scientific recipe remains `7480d5f215292ced9d0a9328eed274aff805584e`. Seed2024, all28 official entities retained; old3 raw/metrics reused, new25 completed. No query-driven changes or added seeds.

## Main finding

The reproducible AP gains on the original three machines remain valid local evidence. They do not extend to a stable benefit on the additional25: Cross-minus-Self AP is -0.0242 percentage points, 10 wins/0 ties/15 losses. Across all28, the mean is +0.0483pp but the paired median is -0.0194pp (13/0/15). Cross-minus-Head all28 is +0.0858pp (15/0/13). These are descriptive entity comparisons in one historically exposed SMD domain, not independent-domain or multi-seed confirmation.

The current recipe is retained with its positive and negative evidence; broad independent Cross benefit remains unsupported. There is no automatic new method, larger seed matrix, pollution/F run, or overall-route NO-GO. Current evidence does not justify treating Cross as a general paper-level contribution or immediately expanding computation without a new scientific decision.

## Fixed scope, permissions and axes

- Original3: SMD1-1/1-2/1-3. New25: every other official entity, no performance-based inclusion. Official source pin7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4; all84 raw files qualified by bytes/Git blob/SHA and shape. No missing/failed machine was omitted.
- For each original training record lengthN: FIT[0,floor(.6N)), SELECT[floor(.6N)+100,floor(.8N)), CAL[floor(.8N)+100,N). FIT-only standardization; physical learner files contain only observations and original indices. Complete original test is QUERY; labels are evaluation-only.
- Official initial weights, frozen body, Head/Self/Cross interfaces, 2000 updates, matching within-seed FIT/masks/initialization, normal SELECT rule, score seed2024, SELECT84551, L=1 and own-CAL q99higher strict`>` are unchanged. L=1 is native copy-score variance, not a forecast horizon or a newly mixed residual.
- Native four-method axis: [0,floor(testN/100)*100). H512 five-method common axis: [512,floor(testN/100)*100). The two axes are reported separately; all thresholds use their own full legitimate CAL, not QUERY or a different score unit.
- Native TimeRadar/Hybrid remain offline full-window and batch-dependent scores. Retrospective C3/latency does not establish strict online performance. H512 is past-only, but its inclusion does not make other methods causal.
- C3 confirms the current point after three adjacent exceedances without backfill. Any-hit and chronological one-to-one matching are distinct. Original event boundaries are retained; partially observed/boundary events remain detailed but ineligible for complete-event metrics. UNKNOWN is not normal or a full miss. Long events>=512, steady segment starts192 points after onset; only7/28 machines have eligible long events.

## Grouped native full-axis comparison

AP and ROC are unweighted entity macro means. AP median is across entities, not pooled points or seeds.
| Group | Method | AP macro | AP median | ROC macro |
| --- | --- | --- | --- | --- |
| old3 | TimeRadar zero | 0.416488 | 0.357354 | 0.896065 |
| old3 | Head | 0.428193 | 0.382512 | 0.904870 |
| old3 | Self | 0.428749 | 0.383640 | 0.905108 |
| old3 | Hybrid Cross | 0.435267 | 0.385731 | 0.905740 |
| new25 | TimeRadar zero | 0.417042 | 0.386244 | 0.850518 |
| new25 | Head | 0.418831 | 0.408661 | 0.851807 |
| new25 | Self | 0.419184 | 0.408198 | 0.851983 |
| new25 | Hybrid Cross | 0.418942 | 0.402318 | 0.851814 |
| all28 | TimeRadar zero | 0.416983 | 0.371799 | 0.855398 |
| all28 | Head | 0.419834 | 0.400836 | 0.857493 |
| all28 | Self | 0.420209 | 0.400811 | 0.857675 |
| all28 | Hybrid Cross | 0.420691 | 0.397584 | 0.857592 |

## Paired Cross increments (native full axis)

Values are percentage points. W/T/L uses the strict unrounded sign; no tolerance was chosen from results.
| Group | Comparison | Metric | Mean delta(pp) | Median delta(pp) | W/T/L |
| --- | --- | --- | --- | --- | --- |
| old3 | cross-minus-decoder_head | AP | 0.7075 | 0.4088 | 3/0/0 |
| old3 | cross-minus-decoder_head | ROC_AUC | 0.0870 | 0.0390 | 3/0/0 |
| old3 | cross-minus-self | AP | 0.6519 | 0.4654 | 3/0/0 |
| old3 | cross-minus-self | ROC_AUC | 0.0632 | 0.0215 | 2/0/1 |
| new25 | cross-minus-decoder_head | AP | 0.0111 | -0.0038 | 12/0/13 |
| new25 | cross-minus-decoder_head | ROC_AUC | 0.0006 | 0.0061 | 15/0/10 |
| new25 | cross-minus-self | AP | -0.0242 | -0.0402 | 10/0/15 |
| new25 | cross-minus-self | ROC_AUC | -0.0169 | 0.0044 | 13/0/12 |
| all28 | cross-minus-decoder_head | AP | 0.0858 | 0.0331 | 15/0/13 |
| all28 | cross-minus-decoder_head | ROC_AUC | 0.0099 | 0.0221 | 18/0/10 |
| all28 | cross-minus-self | AP | 0.0483 | -0.0194 | 13/0/15 |
| all28 | cross-minus-self | ROC_AUC | -0.0084 | 0.0062 | 15/0/13 |

## Every entity: native full-axis AP

All rows use seed2024; original3 are immutable accepted CSV rows.
| Machine | Zero | Head | Self | Cross | Cross-Self(pp) | Cross-Head(pp) |
| --- | --- | --- | --- | --- | --- | --- |
| machine-1-1 | 0.609689 | 0.602559 | 0.603667 | 0.616477 | 1.2810 | 1.3918 |
| machine-1-2 | 0.282421 | 0.299507 | 0.298940 | 0.303594 | 0.4654 | 0.4088 |
| machine-1-3 | 0.357354 | 0.382512 | 0.383640 | 0.385731 | 0.2091 | 0.3219 |
| machine-1-4 | 0.229186 | 0.241169 | 0.240390 | 0.239127 | -0.1263 | -0.2043 |
| machine-1-5 | 0.568655 | 0.571605 | 0.571695 | 0.565244 | -0.6451 | -0.6361 |
| machine-1-6 | 0.645547 | 0.647982 | 0.648901 | 0.654917 | 0.6015 | 0.6934 |
| machine-1-7 | 0.275398 | 0.272279 | 0.272331 | 0.272106 | -0.0226 | -0.0173 |
| machine-1-8 | 0.306428 | 0.302369 | 0.302565 | 0.305322 | 0.2756 | 0.2952 |
| machine-2-1 | 0.133835 | 0.141385 | 0.142078 | 0.144123 | 0.2045 | 0.2738 |
| machine-2-2 | 0.190392 | 0.193247 | 0.193713 | 0.194228 | 0.0515 | 0.0981 |
| machine-2-3 | 0.236304 | 0.245462 | 0.245414 | 0.245946 | 0.0532 | 0.0484 |
| machine-2-4 | 0.313246 | 0.294001 | 0.294136 | 0.300458 | 0.6322 | 0.6457 |
| machine-2-5 | 0.477906 | 0.487657 | 0.489017 | 0.496653 | 0.7636 | 0.8996 |
| machine-2-6 | 0.401647 | 0.408661 | 0.408198 | 0.409190 | 0.0992 | 0.0529 |
| machine-2-7 | 0.386244 | 0.413162 | 0.415970 | 0.402318 | -1.3653 | -1.0844 |
| machine-2-8 | 0.916013 | 0.908620 | 0.906337 | 0.904419 | -0.1918 | -0.4201 |
| machine-2-9 | 0.817010 | 0.819357 | 0.820151 | 0.819665 | -0.0485 | 0.0309 |
| machine-3-1 | 0.556900 | 0.566537 | 0.568157 | 0.567755 | -0.0402 | 0.1218 |
| machine-3-10 | 0.355913 | 0.393011 | 0.393424 | 0.392850 | -0.0573 | -0.0161 |
| machine-3-11 | 0.511749 | 0.455213 | 0.459231 | 0.457087 | -0.2144 | 0.1873 |
| machine-3-2 | 0.104408 | 0.100783 | 0.101264 | 0.100468 | -0.0796 | -0.0315 |
| machine-3-3 | 0.152449 | 0.157543 | 0.157987 | 0.156117 | -0.1870 | -0.1426 |
| machine-3-4 | 0.416215 | 0.413410 | 0.412951 | 0.413764 | 0.0813 | 0.0354 |
| machine-3-5 | 0.327495 | 0.334512 | 0.333126 | 0.332506 | -0.0620 | -0.2007 |
| machine-3-6 | 0.675487 | 0.681624 | 0.681554 | 0.680970 | -0.0584 | -0.0654 |
| machine-3-7 | 0.595622 | 0.581677 | 0.581462 | 0.581550 | 0.0088 | -0.0127 |
| machine-3-8 | 0.282275 | 0.291500 | 0.291624 | 0.291462 | -0.0162 | -0.0038 |
| machine-3-9 | 0.549733 | 0.547997 | 0.547918 | 0.545306 | -0.2613 | -0.2691 |

## Every entity: native full-axis ROC-AUC

| Machine | Zero | Head | Self | Cross | Cross-Self(pp) | Cross-Head(pp) |
| --- | --- | --- | --- | --- | --- | --- |
| machine-1-1 | 0.902007 | 0.904654 | 0.905258 | 0.905044 | -0.0214 | 0.0390 |
| machine-1-2 | 0.921863 | 0.922483 | 0.922488 | 0.922703 | 0.0215 | 0.0221 |
| machine-1-3 | 0.864324 | 0.887474 | 0.887578 | 0.889472 | 0.1893 | 0.1998 |
| machine-1-4 | 0.856978 | 0.884620 | 0.884903 | 0.885117 | 0.0215 | 0.0498 |
| machine-1-5 | 0.921044 | 0.932067 | 0.931340 | 0.930999 | -0.0341 | -0.1067 |
| machine-1-6 | 0.848345 | 0.842491 | 0.842994 | 0.845766 | 0.2772 | 0.3275 |
| machine-1-7 | 0.661248 | 0.650428 | 0.650473 | 0.650971 | 0.0499 | 0.0544 |
| machine-1-8 | 0.772984 | 0.771479 | 0.771829 | 0.772160 | 0.0330 | 0.0681 |
| machine-2-1 | 0.653179 | 0.656105 | 0.656163 | 0.659857 | 0.3695 | 0.3752 |
| machine-2-2 | 0.656511 | 0.669877 | 0.670937 | 0.671336 | 0.0399 | 0.1459 |
| machine-2-3 | 0.973115 | 0.976331 | 0.976347 | 0.976391 | 0.0044 | 0.0061 |
| machine-2-4 | 0.730755 | 0.716754 | 0.716693 | 0.718075 | 0.1382 | 0.1321 |
| machine-2-5 | 0.911557 | 0.915040 | 0.915465 | 0.914638 | -0.0826 | -0.0402 |
| machine-2-6 | 0.849224 | 0.851978 | 0.850936 | 0.849448 | -0.1488 | -0.2530 |
| machine-2-7 | 0.902821 | 0.907268 | 0.908610 | 0.902478 | -0.6132 | -0.4790 |
| machine-2-8 | 0.999299 | 0.999289 | 0.999277 | 0.999268 | -0.0009 | -0.0021 |
| machine-2-9 | 0.973334 | 0.973068 | 0.973107 | 0.972073 | -0.1034 | -0.0995 |
| machine-3-1 | 0.910297 | 0.922571 | 0.922913 | 0.924482 | 0.1569 | 0.1910 |
| machine-3-10 | 0.903383 | 0.910249 | 0.910569 | 0.910269 | -0.0300 | 0.0020 |
| machine-3-11 | 0.973686 | 0.972285 | 0.973107 | 0.972506 | -0.0601 | 0.0221 |
| machine-3-2 | 0.677156 | 0.655455 | 0.655698 | 0.653736 | -0.1962 | -0.1720 |
| machine-3-3 | 0.801562 | 0.802826 | 0.803656 | 0.797564 | -0.6092 | -0.5262 |
| machine-3-4 | 0.819761 | 0.795576 | 0.795354 | 0.795829 | 0.0475 | 0.0253 |
| machine-3-5 | 0.836374 | 0.845156 | 0.844974 | 0.844427 | -0.0548 | -0.0729 |
| machine-3-6 | 0.971964 | 0.973887 | 0.973818 | 0.974520 | 0.0702 | 0.0633 |
| machine-3-7 | 0.892898 | 0.896516 | 0.896492 | 0.896571 | 0.0079 | 0.0055 |
| machine-3-8 | 0.835603 | 0.845092 | 0.845353 | 0.843476 | -0.1876 | -0.1615 |
| machine-3-9 | 0.929871 | 0.928777 | 0.928571 | 0.933387 | 0.4816 | 0.4610 |

## Operating costs and sustained detection, native full-axis all28

These are entity macro means; they are not pooled event precision or equal realized false-positive rates. Full strict/C3 point P/R/F1, counts, any-hit, one-to-one P/R/F1, offsets, normal points/fragments, coverage and longest misses remain in [native_full_28x4.csv](native_full_28x4.csv).
| Method | Strict pointF1(%) | Strict FPR(%) | C3 FPR(%) | C3 fragments/1000 normal points | C3 eventP(%) | C3 eventR(%) | C3 eventF1(%) | C3 anomaly coverage(%) | Long steady(%,n=7) | Mean longest miss(points) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimeRadar zero | 30.5811 | 2.8242 | 1.7212 | 1.9461 | 25.7779 | 70.8276 | 26.1662 | 37.9894 | 20.6639 | 203.64 |
| Head | 31.2934 | 3.0893 | 1.9530 | 2.0551 | 27.3226 | 73.4665 | 26.5181 | 39.2390 | 23.1666 | 205.61 |
| Self | 31.1455 | 3.0766 | 1.9450 | 2.0494 | 27.2681 | 73.1590 | 26.3827 | 39.1187 | 22.9330 | 205.64 |
| Hybrid Cross | 31.1441 | 3.0327 | 1.9171 | 2.0172 | 27.6218 | 73.8308 | 26.9066 | 39.0768 | 22.7480 | 210.39 |

Cross versus Self: C3 eventF1 +0.5239pp and normal FPR -0.0279pp, while C3 coverage -0.0419pp and long steady coverage -0.1849pp; mean longest miss grows205.64→210.39 points. Thus the event/false-alarm tradeoff has a small positive component, but no sustained-coverage gain. Across327 eligible native-axis events, any-hit counts are Zero213/Head222/Self220/Cross222; one-to-one matched counts are211/220/218/220. These pooled counts are separate from entity macro recalls above.

## Five-method common-axis comparison, all28

Do not subtract these H512-axis metrics from native-full-axis metrics. Full per-machine five-method results are in [common_28x5.csv](common_28x5.csv).
| Method | AP | ROC | Strict FPR(%) | C3 FPR(%) | C3 eventF1(%) | C3 coverage(%) |
| --- | --- | --- | --- | --- | --- | --- |
| TimeRadar zero | 0.418801 | 0.854602 | 2.8528 | 1.7415 | 26.3766 | 38.0000 |
| Head | 0.421492 | 0.856740 | 3.1236 | 1.9776 | 26.7169 | 39.2506 |
| Self | 0.421868 | 0.856922 | 3.1104 | 1.9698 | 26.5861 | 39.1302 |
| Hybrid Cross | 0.422311 | 0.856827 | 3.0664 | 1.9416 | 27.0807 | 39.0881 |
| History Mean512 | 0.111306 | 0.661935 | 1.0370 | 0.2872 | 29.8225 | 6.5892 |

Cross beats H512 AP on28/28 machines on this axis and has much higher coverage. H512 has lower normal FPR and higher eventF1: C3 normal FPR0.2872% vsCross1.9416%; eventF1 29.8225% vs27.0807%. This is a cost tradeoff, not universal dominance or equal-FPR superiority.

## Fixed matched-data LEFT external comparison

[Full four-entity/six-method table](matched_left_external.csv) and [permission/metric details](LEFT_RESULTS.md). Same FIT/SELECT/FIT-only scaling is enforced, while source pretraining, initialization, native training budget and compute remain explicitly different. LEFT uses native20epoch/patience3, no old8000-update cap; PSM12400 updates, SMD1-1 2640 and SMD1-2/1-3 2200. Small SMD FIT does not finish the original1000+2000 curriculum; it was not extended to improve QUERY.
| Entity | LEFT AP / ROC | Cross AP / ROC | Head AP | Self AP | H512 AP |
| --- | --- | --- | --- | --- | --- |
| PSM | 0.571098 / 0.764667 | 0.426164 / 0.680347 | 0.427837 | 0.427206 | 0.316848 |
| machine-1-1 | 0.388016 / 0.837270 | 0.619073 / 0.904915 | 0.605361 | 0.606455 | 0.206087 |
| machine-1-2 | 0.398419 / 0.938496 | 0.307326 / 0.922357 | 0.303242 | 0.302738 | 0.105610 |
| machine-1-3 | 0.344046 / 0.883564 | 0.378949 / 0.888234 | 0.375789 | 0.376888 | 0.154823 |

LEFT wins PSM and SMD1-2 AP; Cross wins SMD1-1 and SMD1-3. False-alarm/coverage tradeoffs and common-axis event eligibility are retained; neither table establishes universal superiority. Historical different-permission LEFT-R/CATCH results remain separate; no new CATCH training was performed.

## PSM three-seed negative evidence retained

The following rows reuse the accepted native-axis three-seed archive; they are not recomputed or mixed with the external cropped-axis table.
| Seed | Cross AP | Cross-Self AP(pp) | Cross-Head AP(pp) | Self C3 FPR(%) | Cross C3 FPR(%) | Cross C3 coverage(%) |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | 0.425339 | -0.1020 | -0.1636 | 0.4602 | 1.2418 | 4.2331 |
| 2025 | 0.425140 | -0.2893 | -0.1563 | 0.4019 | 1.1378 | 3.8898 |
| 2026 | 0.425529 | 0.0023 | -0.1376 | 0.4917 | 1.3963 | 4.6470 |

Across these three PSM seeds, Cross-minus-Head AP is negative each time. Cross-minus-Self AP mean -0.1297pp (sampleSD0.1477pp); extra triggering/coverage comes with higher normal false alarms. The original three-machine SMD multi-seed positives are preserved in the [previous complete report](../timeradar_cross_variable_repeat_20261116/RESULTS.md); the new25 were not selectively assigned extra seeds.

## Execution, audit and decision

- Added SMD25:75 training jobs×2000 updates +100 scoring jobs =175 actual GPU jobs, all exit0,200 complete CAL/QUERY banks. Added fixed H512:50 CPU banks; original3 model/H512 assets reused. All28/28 are valid; no failure/asset omissions or replacement machines.
- Actual new GPU wall: TimeRadar expansion13.40385439203845h (training13.259069207972951, scoring0.1447851840655009) +LEFT0.4025681857616615h =13.806422577800111h. Old study costs are not re-billed. CPU full28 metrics9.672s; matched LEFT0.907s; H51267.844s. Audit adds0GPU.
- Resource exception: an unrelated job joined GPU0 near the last Self training. The original training finished2000 steps; a subsequent scoring preflight refusedGPU0 before launching (0GPUh). That one unstarted score ran once on freeGPU1 with unchanged config/checkpoint/scorer. Original198-bank/failure files and separate recovered200-bank provenance are both preserved. This was not a scientific rerun and no outside process was killed. Do not claim absolute GPU exclusivity for the entire run.
- Complete archive1643 files was mirrored with matching SHA. Original source/weights, old model and previous positive/negative results are unchanged. New code only transports fixed data, native LEFT and metrics/recovery identities; the scientific adapter recipe is unchanged.
- Key independent-responsibility audit PASS_BOUNDED:200 new+32 old banks,75 selected/final training identities/frozen state/matching plans,198→200 recovery,225 newly calculated rows and27 exactly reused rows,112/140 tables and all grouped/paired statistics. Official28 data and LEFT subchecks are included. Same model family, not external blinded review. Auditor script/asset-assumption failures were retained and corrected without modifying scientific products.
- Primary decision: retain the method with a clearly restricted evidence scope. Original3 repeatable AP gains do not generalize reliably to new25; small eventF1/false-alarm improvements do not establish better sustained detection. Current results support neither general Cross superiority nor lag/weak-anomaly/pollution-robustness claims. No automatic larger training matrix, new formula, U/F extension or alternative method was launched.

## Public artifacts

- [112 native full-axis rows](native_full_28x4.csv); [140 common-axis rows](common_28x5.csv).
- [Native group macro/median](native_full_macro_median.csv); [native paired per-machine deltas](native_full_paired_deltas.csv); [native paired group signs](native_full_paired_summary.csv).
- [Common-axis macro/median](five_method_common_macro_median.csv); [common-axis paired deltas](five_method_common_paired_deltas.csv); [common-axis paired signs](five_method_common_paired_summary.csv).
- [Matched LEFT external24 rows](matched_left_external.csv); [public provenance and primary acceptance](PROVENANCE_PUBLIC.json).
- Public CSV line endings follow Git LF normalization; raw archive CRLF and public Git-blob SHA are distinguished in provenance. All parsed fields are identical; raw scientific files are unchanged.
- Raw observations, model weights, private machine credentials/addresses and third-party source assets are not uploaded. Detailed raw banks/configs/indices/checkpoints/traces remain in the local/server research archive.
