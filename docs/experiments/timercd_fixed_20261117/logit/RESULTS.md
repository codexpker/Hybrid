# TimeRCD fixed numerical-stability comparison: original logit difference

Completed cache-only evaluation. Every A/B entity uses archived abnormal-class minus normal-class head logits, never inverse-softmax probabilities or per-entity score selection. Official model/head/weights/window/normalization and caches remain unchanged. The original probability release and conclusions remain archived in the parent directory; this version corrects the interpretation of ranking degradation amplified by probability saturation. No new inference, training, threshold search or audit process.

## SMD28 common-axis method comparison

| Method | Valid n/28 | AP macro | AP median | ROC macro | Strict normal FPR (%) | C3 normal FPR (%) | C3 event P (%) | C3 event R (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%, n=7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_official_role | 28 | 0.487280 | 0.500909 | 0.838074 | 0.5759 | 0.2814 | 41.3127 | 56.8125 | 39.4073 | 30.2300 | 4.7697 |
| B_fit_only | 28 | 0.473616 | 0.455082 | 0.859199 | 7.2546 | 6.4250 | 30.9773 | 63.4739 | 28.4693 | 42.8880 | 29.1074 |
| zero | 28 | 0.418819 | 0.390820 | 0.854602 | 2.8528 | 1.7415 | 25.8244 | 70.9870 | 26.3766 | 38.0000 | 20.6639 |
| decoder_head | 28 | 0.421510 | 0.411386 | 0.856739 | 3.1236 | 1.9776 | 27.3606 | 73.6465 | 26.7169 | 39.2506 | 23.1666 |
| self | 28 | 0.421886 | 0.411146 | 0.856922 | 3.1104 | 1.9698 | 27.3063 | 73.3390 | 26.5861 | 39.1302 | 22.9330 |
| cross | 28 | 0.422329 | 0.406186 | 0.856826 | 3.0664 | 1.9416 | 27.6469 | 74.0026 | 27.0807 | 39.0881 | 22.7480 |
| history_mean512 | 28 | 0.111308 | 0.109026 | 0.661920 | 1.0371 | 0.2872 | 34.1991 | 46.9430 | 29.8225 | 6.5892 | 1.1447 |
| LEFT_native_same_roles | 3 | 0.376827 | 0.388016 | 0.886443 | 4.4799 | 4.1846 | 25.3333 | 48.8636 | 23.7177 | 20.8257 | 70.5901 |

LEFT has only3/28 SMD cases; its subset summary must not be compared as if it covered28 entities. Each baseline keeps its existing scoring expression and compatible results are reused. TimeRCD A/B rows in this table use logits only.

## All29 common-axis records: AP / ROC

| Entity | A_official_role | B_fit_only | zero | decoder_head | self | cross | history_mean512 | LEFT_native_same_roles |
|---|---|---|---|---|---|---|---|---|
| PSM | 0.359946 / 0.552157 | 0.303012 / 0.476555 | 0.432502 / 0.670965 | 0.427837 / 0.678159 | 0.427206 / 0.678538 | 0.426164 / 0.680347 | 0.316848 / 0.530450 | 0.571098 / 0.764667 |
| machine-1-1 | 0.180584 / 0.429793 | 0.450935 / 0.846280 | 0.612285 / 0.901876 | 0.605361 / 0.904564 | 0.606455 / 0.905165 | 0.619073 / 0.904915 | 0.206087 / 0.722219 | 0.388016 / 0.837270 |
| machine-1-2 | 0.454283 / 0.955374 | 0.448633 / 0.949967 | 0.286288 / 0.921543 | 0.303242 / 0.922140 | 0.302738 / 0.922145 | 0.307326 / 0.922357 | 0.105610 / 0.841062 | 0.398419 / 0.938496 |
| machine-1-3 | 0.601587 / 0.921610 | 0.058332 / 0.676537 | 0.351515 / 0.862250 | 0.375789 / 0.886172 | 0.376888 / 0.886283 | 0.378949 / 0.888234 | 0.154823 / 0.776147 | 0.344046 / 0.883564 |
| machine-1-4 | 0.521585 / 0.923060 | 0.203781 / 0.843539 | 0.223667 / 0.854000 | 0.233946 / 0.882162 | 0.233206 / 0.882453 | 0.231808 / 0.882659 | 0.132741 / 0.728227 | NA |
| machine-1-5 | 0.526362 / 0.920993 | 0.456732 / 0.892926 | 0.568672 / 0.920401 | 0.571616 / 0.931331 | 0.571706 / 0.930612 | 0.565255 / 0.930239 | 0.092879 / 0.836889 | NA |
| machine-1-6 | 0.696393 / 0.894334 | 0.869285 / 0.954487 | 0.645779 / 0.845255 | 0.648266 / 0.839420 | 0.649187 / 0.839920 | 0.655186 / 0.842746 | 0.184516 / 0.529503 | NA |
| machine-1-7 | 0.527504 / 0.824600 | 0.547950 / 0.783030 | 0.277814 / 0.660937 | 0.274647 / 0.650248 | 0.274697 / 0.650294 | 0.274470 / 0.650794 | 0.149209 / 0.543295 | NA |
| machine-1-8 | 0.336932 / 0.826037 | 0.312765 / 0.777077 | 0.307219 / 0.771991 | 0.303054 / 0.770390 | 0.303252 / 0.770742 | 0.306029 / 0.771075 | 0.086292 / 0.723381 | NA |
| machine-2-1 | 0.246989 / 0.696873 | 0.240495 / 0.661629 | 0.135436 / 0.652772 | 0.143046 / 0.655567 | 0.143734 / 0.655614 | 0.145794 / 0.659266 | 0.075144 / 0.583522 | NA |
| machine-2-2 | 0.258574 / 0.794332 | 0.255804 / 0.768934 | 0.192549 / 0.654380 | 0.195683 / 0.668751 | 0.196169 / 0.669847 | 0.196646 / 0.670194 | 0.129481 / 0.516280 | NA |
| machine-2-3 | 0.507598 / 0.922508 | 0.457383 / 0.947846 | 0.236370 / 0.972557 | 0.245490 / 0.975819 | 0.245443 / 0.975836 | 0.245977 / 0.975881 | 0.167200 / 0.817058 | NA |
| machine-2-4 | 0.573822 / 0.814096 | 0.453431 / 0.705491 | 0.314833 / 0.729295 | 0.295441 / 0.715229 | 0.295576 / 0.715166 | 0.301892 / 0.716391 | 0.139047 / 0.601959 | NA |
| machine-2-5 | 0.494220 / 0.946614 | 0.534617 / 0.951659 | 0.478696 / 0.910915 | 0.488595 / 0.914457 | 0.489935 / 0.914876 | 0.497454 / 0.914023 | 0.110076 / 0.677582 | NA |
| machine-2-6 | 0.468443 / 0.918469 | 0.446523 / 0.936834 | 0.402153 / 0.848959 | 0.409134 / 0.851636 | 0.408667 / 0.850575 | 0.409668 / 0.849084 | 0.090834 / 0.812923 | NA |
| machine-2-7 | 0.782247 / 0.944977 | 0.803110 / 0.958915 | 0.386759 / 0.902210 | 0.413639 / 0.906662 | 0.416425 / 0.907980 | 0.402705 / 0.901798 | 0.067527 / 0.531037 | NA |
| machine-2-8 | 0.977414 / 0.999824 | 0.935810 / 0.993298 | 0.916013 / 0.999284 | 0.908620 / 0.999273 | 0.906337 / 0.999261 | 0.904419 / 0.999252 | 0.085966 / 0.704908 | NA |
| machine-2-9 | 0.306047 / 0.607110 | 0.752834 / 0.930386 | 0.817286 / 0.973008 | 0.819686 / 0.972740 | 0.820481 / 0.972780 | 0.819991 / 0.971746 | 0.107976 / 0.628576 | NA |
| machine-3-1 | 0.692472 / 0.978193 | 0.814571 / 0.976257 | 0.557214 / 0.909892 | 0.566719 / 0.922054 | 0.568326 / 0.922389 | 0.567932 / 0.923965 | 0.132184 / 0.734164 | NA |
| machine-3-10 | 0.875662 / 0.957500 | 0.908464 / 0.980192 | 0.394881 / 0.907366 | 0.429616 / 0.913502 | 0.430025 / 0.913828 | 0.429044 / 0.913517 | 0.111202 / 0.583099 | NA |
| machine-3-11 | 0.591046 / 0.909264 | 0.538546 / 0.963238 | 0.511749 / 0.973202 | 0.455213 / 0.971776 | 0.459231 / 0.972613 | 0.457087 / 0.972001 | 0.128676 / 0.750968 | NA |
| machine-3-2 | 0.072996 / 0.593064 | 0.099660 / 0.718741 | 0.103240 / 0.672195 | 0.099528 / 0.649961 | 0.099995 / 0.650188 | 0.099228 / 0.648194 | 0.052214 / 0.489534 | NA |
| machine-3-3 | 0.294860 / 0.856879 | 0.079283 / 0.736954 | 0.154233 / 0.802511 | 0.159455 / 0.803975 | 0.159922 / 0.804813 | 0.158002 / 0.798708 | 0.076398 / 0.609209 | NA |
| machine-3-4 | 0.367225 / 0.786704 | 0.507597 / 0.876657 | 0.416953 / 0.819808 | 0.414082 / 0.795876 | 0.413625 / 0.795653 | 0.414444 / 0.796116 | 0.066339 / 0.543978 | NA |
| machine-3-5 | 0.669147 / 0.940746 | 0.152285 / 0.821183 | 0.327542 / 0.833797 | 0.334554 / 0.842656 | 0.333168 / 0.842465 | 0.332548 / 0.841910 | 0.127660 / 0.862140 | NA |
| machine-3-6 | 0.451423 / 0.901323 | 0.528571 / 0.942523 | 0.676120 / 0.971695 | 0.682233 / 0.973618 | 0.682151 / 0.973546 | 0.681599 / 0.974274 | 0.063841 / 0.543576 | NA |
| machine-3-7 | 0.229046 / 0.632264 | 0.374178 / 0.821254 | 0.595775 / 0.892284 | 0.581848 / 0.895896 | 0.581631 / 0.895871 | 0.581718 / 0.895951 | 0.049622 / 0.569942 | NA |
| machine-3-8 | 0.177807 / 0.593361 | 0.284579 / 0.664710 | 0.285862 / 0.834864 | 0.295487 / 0.844328 | 0.295620 / 0.844593 | 0.295396 / 0.842716 | 0.073286 / 0.523111 | NA |
| machine-3-9 | 0.761581 / 0.976164 | 0.745109 / 0.977016 | 0.550025 / 0.929608 | 0.548295 / 0.928503 | 0.548215 / 0.928295 | 0.545577 / 0.933128 | 0.149808 / 0.749470 | NA |

## Fixed expression and calibration

For every A/B CAL and QUERY bank, logit=archived head_logits[:,1]-head_logits[:,0], verified against the stored logit array. Each threshold is recomputed from that entity/permission's entire original legitimate CAL logit array using q99 with method=higher, followed by strict greater-than. Old probability thresholds are not copied. No QUERY value or label chooses the threshold. CAL/QUERY roles, valid masks, original event boundaries and padding remain unchanged. The new logit scores may be negative or very large and are never clipped or passed through sigmoid.

Primary scores for all A/B entities are uniformly logits. Both probability and logit operational metrics, including separate thresholds, strict/C3 counts, any-hit, one-to-one P/R/F1, false-positive points/fragments, coverage and longest misses, are kept side by side in probability_logit_operating_comparison.csv (58 common-axis rows). They are not spliced into one method or selected per machine. Original probability files are unchanged.

common_29x8.csv contains232 shared-axis rows. Pairwise deltas use these axes only: paired_all29_AP_ROC.csv includes every A/B record and reference (LEFT unavailable comparisons remain NA); the SMD paired tables retain all corresponding operating-metric differences and descriptive strict win/tie/loss counts. native_29x6.csv is a separate method-own/full-QUERY-axis table with174 rows, not a source for subtraction against common-axis rows. Original baselines are hash-reused; no old-model forward or training was repeated. All TimeRCD/TimeRadar/LEFT scores remain offline; no causal online delay claim. Each method keeps its earlier data permissions and context length.

## Corrected interpretation

On the same common axis, B SMD AP is .473616 rather than the archived probability .411833. It exceeds TimeRadar-zero .418819 and old meanCross .422329; the earlier inference that B had generally weak ranking was overstated by numerical probability compression. A AP changes only .487167 to .487280. The A-minus-B AP gap therefore narrows from7.5334 to1.3664 percentage points but does not disappear. B ROC .859199 exceeds A .838074 and is slightly above TimeRadar-zero .854602. These are full-method results, not evidence isolating a discriminative-loss mechanism.

Normalization sensitivity remains substantial in operational behavior after each expression is legally recalibrated. SMD B strict normal FPR7.2546% (probability7.1439%) and C3 FPR6.4250% (probability6.3188%) are much larger than A .5759%/.2814% and TimeRadar-zero2.8528%/1.7415%. B C3 anomaly coverage42.8880% exceeds zero38.0000%, while A30.2300% is lower. B long-steady coverage29.1074% versus A4.7697% and zero20.6639% remains a false-alarm/coverage tradeoff; only7/28 entities have eligible long events. Neither the best AP nor the improved event F1 establishes uniform sustained superiority.

PSM remains negative in ranking: A .359946/.552157 and B .303012/.476555 (AP/ROC), below zero .432502/.670965 and LEFT .571098/.764667. Its operational metrics are unchanged to the displayed precision: A/B C3 coverage5.5885%/4.9393% with C3 normal FPR2.1511%/1.7190%; these are not equal-cost event improvements over zero/LEFT. The full table retains both any-hit and one-to-one counts; they are not interchangeable.

## Verification and closure

The existing8 relevant unit tests passed. Primary raw-cache recomputation checked all116 new native/common rows for AP/ROC, exact own-CAL logit q99 threshold, strict FP, C3 FP and raw head/logit identity; no independent audit flow was added. The shared established event evaluator computes all complete-event and long-event metrics; original partial/UNKNOWN evidence is preserved in raw per-method details and event_detail_references.json. CPU evaluation4.594 seconds, new GPU time0. No labels are used to select representations, weights, normalization or thresholds.

This is a numerical scoring-expression version of the same external model, not a Hybrid innovation. The full official probability comparison is archived unchanged, not replaced. TimeRCD numerical diagnosis is now closed: no normalization, window, fusion, new baseline or Hybrid-training search follows this evaluation.

## Artifacts

- [Complete common-axis method table](common_29x8.csv).
- [Probability/logit operational metrics kept separate](probability_logit_operating_comparison.csv).
- [All29 entity paired AP/ROC differences](paired_all29_AP_ROC.csv).
- [SMD mean/median](common_SMD_macro_median.csv), [paired full-metric differences](common_SMD_paired_deltas.csv), [paired summaries](common_SMD_paired_summary.csv).
- [Separate full-QUERY/native axes](native_29x6.csv), [native-axis SMD summaries](native_SMD_macro_median.csv).
- [Public provenance](PROVENANCE_PUBLIC.json). Public CSV is LF-normalized; raw caches and original probability tables are unchanged. No raw data, weights or private server configuration is published.
