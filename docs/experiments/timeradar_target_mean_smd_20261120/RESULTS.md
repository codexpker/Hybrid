# Fixed TimeRadar R / R-MeanT on SMD machine-3-2 and machine-3-3

Seed2025 only. These historically exposed, preassigned entities are a fixed SMD subset, not blind confirmation, two independent domains or the full28-machine benchmark. All methods use the official initial weights, source-FIT scaler, independent B1 norm0 calls, the same fixed ten complementary masks and native terminal99 variance score. Four decoder adaptations are new; noU has no training. Old norm1/batch32 scores are retained separately and not substituted.

## Complete ranking comparison

| Entity | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| machine-3-2 | noU | 0.036035 | 0.503761 | 0.068493 | 0.728434 |
| machine-3-2 | R | 0.037968 | 0.527994 | 0.136598 | 0.761325 |
| machine-3-2 | R_target_mean | 0.035829 | 0.517616 | 0.084654 | 0.737006 |
| machine-3-3 | noU | 0.043771 | 0.532765 | 0.066217 | 0.664504 |
| machine-3-3 | R | 0.092484 | 0.653673 | 0.132917 | 0.782451 |
| machine-3-3 | R_target_mean | 0.087649 | 0.626640 | 0.119191 | 0.753943 |

AP is Average Precision, not trapezoidal AUC-PR. VUS uses the same pinned official LEFT RangeAUC_volume_opt(windowSize100,thre250) implementation. Ranking uses native V; the AP/ROC columns repeated across strict/C3 raw CSV rows are the underlying score ranking, not a claim that confirmation changes AP. C3 low-FPR curves below are computed separately from past-three minima.

## Original CAL operation

| Entity | Method | Stream | Normal FP | FPR % | Recall % | Point F1 % | Matched/events | Any-hit | Event F1 % | Normal fragments | Complete-event coverage % | Long macro/pooled % | Longest miss | Hit-only median delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| machine-3-2 | noU | strict | 5 | 0.034483 | 2.032520 | 3.906250 | 1/8 | 1 | 16.666667 | 4 | 2.032520 | NA/NA | 123 | 0.0 |
| machine-3-2 | noU | confirmed3 | 1 | 0.006897 | 1.219512 | 2.400000 | 1/8 | 1 | 22.222222 | 1 | 1.219512 | NA/NA | 123 | 2.0 |
| machine-3-2 | R | strict | 2 | 0.013793 | 2.032520 | 3.952569 | 1/8 | 1 | 20.000000 | 2 | 2.032520 | NA/NA | 123 | 0.0 |
| machine-3-2 | R | confirmed3 | 1 | 0.006897 | 1.219512 | 2.400000 | 1/8 | 1 | 22.222222 | 1 | 1.219512 | NA/NA | 123 | 2.0 |
| machine-3-2 | R_target_mean | strict | 1 | 0.006897 | 2.032520 | 3.968254 | 1/8 | 1 | 22.222222 | 1 | 2.032520 | NA/NA | 123 | 0.0 |
| machine-3-2 | R_target_mean | confirmed3 | 1 | 0.006897 | 1.219512 | 2.400000 | 1/8 | 1 | 22.222222 | 1 | 1.219512 | NA/NA | 123 | 2.0 |
| machine-3-3 | noU | strict | 461 | 3.254271 | 3.103448 | 3.399433 | 9/16 | 9 | 15.254237 | 89 | 3.103448 | NA/NA | 304 | 0.0 |
| machine-3-3 | noU | confirmed3 | 340 | 2.400113 | 0.517241 | 0.650054 | 1/16 | 1 | 5.263158 | 20 | 0.517241 | NA/NA | 383 | 39.0 |
| machine-3-3 | R | strict | 182 | 1.284766 | 8.448276 | 12.083847 | 13/16 | 13 | 21.666667 | 79 | 8.448276 | NA/NA | 96 | 0.0 |
| machine-3-3 | R | confirmed3 | 75 | 0.529437 | 1.724138 | 3.007519 | 9/16 | 9 | 39.130435 | 20 | 1.724138 | NA/NA | 368 | 2.0 |
| machine-3-3 | R_target_mean | strict | 178 | 1.256530 | 7.931034 | 11.442786 | 13/16 | 13 | 21.311475 | 84 | 7.931034 | NA/NA | 134 | 0.0 |
| machine-3-3 | R_target_mean | confirmed3 | 60 | 0.423549 | 1.896552 | 3.379416 | 10/16 | 10 | 41.666667 | 21 | 1.896552 | NA/NA | 368 | 2.0 |

CAL is a separate published train record with no independent point labels: attack counts are NA, not test labels indexed by train timestamps. Each model uses its own q99/higher threshold and strict > on the same512 legal points. The nominal source role does not certify a normal1%FPR budget. QUERY is the complete pre-existing[8957,23703) axis (14,746 points), with true available_index=t from100-point endpoint windows. C3 requires three consecutive alarms with no backfill. The complete original test event catalog is retained; unobserved or boundary-ineligible events have NA complete coverage and miss statistics. An observed-hit count0 alone does not establish a fully unobserved event miss. One-to-one matching and any-hit remain separate. Long means duration>512, steady from event start+512; macro and pooled coverage use distinct denominators. Delay/miss units are sample points; no verified physical clock is asserted.

## Fixed paired changes

All rate/ranking differences below are percentage points; counts and miss differences retain sample/count units. Each contrast is within the same entity at seed2025: n=1, no sample SD or pooled entity significance claim.

| Entity | New minus control | Stream | AP pp | ROC pp | VUS-PR pp | VUS-ROC pp | FPR pp | Recall pp | Event F1 pp | Long macro pp | Long pooled pp | Matched delta | Any-hit delta | Longest-miss delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| machine-3-2 | R_target_mean−R | strict | -0.213843 | -1.037707 | -5.194458 | -2.431887 | -0.006897 | 0.000000 | 2.222222 | NA | NA | 0 | 0 | 0 |
| machine-3-2 | R_target_mean−noU | strict | -0.020582 | 1.385534 | 1.616069 | 0.857163 | -0.027586 | 0.000000 | 5.555556 | NA | NA | 0 | 0 | 0 |
| machine-3-2 | R_target_mean−R | confirmed3 | -0.213843 | -1.037707 | -5.194458 | -2.431887 | 0.000000 | 0.000000 | 0.000000 | NA | NA | 0 | 0 | 0 |
| machine-3-2 | R_target_mean−noU | confirmed3 | -0.020582 | 1.385534 | 1.616069 | 0.857163 | 0.000000 | 0.000000 | 0.000000 | NA | NA | 0 | 0 | 0 |
| machine-3-3 | R_target_mean−R | strict | -0.483578 | -2.703316 | -1.372590 | -2.850821 | -0.028237 | -0.517241 | -0.355191 | NA | NA | 0 | 0 | 38 |
| machine-3-3 | R_target_mean−noU | strict | 4.387809 | 9.387545 | 5.297429 | 8.943890 | -1.997741 | 4.827586 | 6.057238 | NA | NA | 4 | 4 | -170 |
| machine-3-3 | R_target_mean−R | confirmed3 | -0.483578 | -2.703316 | -1.372590 | -2.850821 | -0.105887 | 0.172414 | 2.536232 | NA | NA | 1 | 1 | 0 |
| machine-3-3 | R_target_mean−noU | confirmed3 | 4.387809 | 9.387545 | 5.297429 | 8.943890 | -1.976564 | 1.379310 | 36.403509 | NA | NA | 9 | 9 | -15 |

## Low-FPR supplement, unchanged definition

Empirical piecewise-linear ROC trapezoid area is integrated over[0,a], a=.01/.05, with interpolation only at the integration boundary. The second value is area/a, not McClish normalization or a convex hull. Cap1/3/5% recall is the maximum attainable whole-tie-group TPR, preferring minimum FPR and then highest cutoff on ties. No interpolated decision or event-based selection is used. The selected QUERY curve points are descriptive only and never replace CAL thresholds. All low-FPR fields, C3 cap event matching/any-hit/complete/long macro/pooled coverage/misses and both controls are in LOW_FPR_SUPPLEMENT.csv and LOW_FPR_PAIRS.csv.

| Entity | Method | Stream | Area01 / divided by.01 | Area05 / divided by.05 | cap1 FPR/TPR % | cap3 FPR/TPR % | cap5 FPR/TPR % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| machine-3-2 | noU | strict | 0.000239/0.023858 | 0.001598/0.031960 | 0.117241/2.439024 | 1.255172/3.252033 | 4.958621/4.065041 |
| machine-3-2 | noU | confirmed3 | 0.000133/0.013274 | 0.001071/0.021427 | 0.731034/1.626016 | 1.841379/2.439024 | 1.841379/2.439024 |
| machine-3-2 | R | strict | 0.000273/0.027334 | 0.002527/0.050541 | 1.000000/3.658537 | 1.737931/5.691057 | 3.772414/6.097561 |
| machine-3-2 | R | confirmed3 | 0.000126/0.012560 | 0.001048/0.020959 | 0.903448/1.626016 | 1.751724/2.439024 | 1.751724/2.439024 |
| machine-3-2 | R_target_mean | strict | 0.000219/0.021867 | 0.001597/0.031932 | 0.613793/2.439024 | 1.958621/3.658537 | 1.958621/3.658537 |
| machine-3-2 | R_target_mean | confirmed3 | 0.000123/0.012279 | 0.001092/0.021833 | 0.979310/1.626016 | 1.082759/2.439024 | 1.082759/2.439024 |
| machine-3-3 | noU | strict | 0.000150/0.015002 | 0.001379/0.027580 | 0.162361/1.724138 | 2.866017/2.931034 | 4.651984/4.655172 |
| machine-3-3 | noU | confirmed3 | 0.000000/0.000000 | 0.000457/0.009136 | 0.000000/0.000000 | 2.710716/1.206897 | 3.367217/1.896552 |
| machine-3-3 | R | strict | 0.000311/0.031106 | 0.006365/0.127306 | 0.981223/7.241379 | 2.971905/15.344828 | 4.955527/21.206897 |
| machine-3-3 | R | confirmed3 | 0.000198/0.019755 | 0.004172/0.083436 | 0.995341/4.482759 | 2.978964/10.689655 | 4.588451/12.413793 |
| machine-3-3 | R_target_mean | strict | 0.000370/0.036992 | 0.005987/0.119739 | 0.967104/7.413793 | 2.873076/15.172414 | 4.948468/19.310345 |
| machine-3-3 | R_target_mean | confirmed3 | 0.000215/0.021460 | 0.003705/0.074099 | 0.917690/3.965517 | 2.971905/9.310345 | 4.800226/11.379310 |


## Scientific decision and retained limits

The specified subset does not establish the PSM learning gain as a cross-task method advantage. R-MeanT is below R on AP, ROC-AUC, VUS-PR and VUS-ROC in both machines. On3-2, AP differences are-0.214pp versus R and-0.021pp versus noU; C3 hits/recall and longest miss are identical. On3-3, R-MeanT exceeds noU on ranking, but R provides stronger ranking gains: MeanT-R AP is-0.484pp and ROC-AUC-2.703pp. Thus beating noU there does not isolate the MeanT increment.

Retain the limited3-3 operating positive: at each method's original CAL threshold, C3 normal false positives fall75->60, matched events rise9->10/16, recall1.724->1.897% and eventF139.130->41.667%; normal fragments increase20->21 and longest miss remains368 points. Strict recall falls8.448->7.931%, longest miss rises96->134, and strict matched events remain13/16. Event discovery, point coverage, fragment counts and ranking therefore cannot be interchanged.

Low-FPR descriptions also remain mixed. Relative to R,3-2 strict normalized areas decrease at both bounds; C3 cap1/3/5 recalls are unchanged. On3-3 strict normalized area through1% rises0.589pp but through5% falls0.757pp. Its C3 cap1/3/5 recall changes-0.517/-1.379/-1.034pp despite a small positive area through1%. These points do not replace CAL thresholds. Full low-FPR event/coverage and both-control results are preserved in the CSVs.

Both fixed QUERY settings contain zero complete events longer than512 points. Long steady macro and pooled coverage, and their contrasts, are consequently NA. This is an observed qualification limit, not a zero-coverage result or a test of sustained long-fault generalization. All8/16 complete events and every original ineligible event remain in events.csv; none was removed to improve results.

Together with the accepted PSM/SWaT repetitions, the candidate remains a reproducible, task-limited PSM ranking/partial-budget learning result with a small3-3 CAL-operation positive, rather than an established broadly useful formal-method mechanism. The PSM2025 longest-miss reduction1228->879 did not repeat in2026/2027 (both arms1228); its event-discovery improvement did not repeat, and SWaT AP negatives repeated in all three seeds. No new machine, seed, coefficient, loss, threshold or model is queued. This decision does not discard the mature TimeRadar model, the public-data research goal, or the existing historical positive evidence.

TimeRadar, source replay and the classical error/disagreement decomposition retain their attribution. The two SMD machines are not a blind/full-SMD confirmation or contamination-causal experiment. The result does not establish a new Hybrid original contribution.

## Actual computation

Four new adaptations and12 necessary native readouts, plus two engineering gates, consumed3.228191597A40GPUh including startup and cleanup. All18 actual GPU jobs completed/cleared. An initial controller import failed before CUDA setup/model use because an existing dependency package was missing from deployment; its original log/runtime was retained and the same frozen dependency was supplied. Its GPU use is0 and CPU wall time was not separately instrumented. No successful training or inference was repeated. Original gate provenance wording is clarified by a separate immutable annotation rather than rewriting the gate/runtime. Full pretrained-containing weights, optimizers, scalers and numeric banks are archived outside publicGit; model/cache hashes and owned job costs are published.
