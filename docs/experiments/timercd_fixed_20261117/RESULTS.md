# TimeRCD official multivariate fixed benchmark comparison

Completed: PSM and all28 SMD entities, unchanged official model/head, W5000/B1, zero training updates. A=official whole-role normalization, B=project FIT-only normalization. All values below are local results, not paper-reported numbers. Both tracks are offline.

## SMD28 shared-axis summary

Each record uses the same scored-point intersection across its available methods. LEFT exists on only3/28 SMD machines; its subset is not directly compared against a28-machine average. AP/ROC are entity macro means, not pooled points.

| Method | Valid n/28 | AP macro | AP median | ROC macro | C3 normal FPR (%) | C3 event F1 (%) | C3 anomaly coverage (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_official_role | 28 | 0.487167 | 0.500851 | 0.838074 | 0.2814 | 39.4073 | 30.2300 |
| B_fit_only | 28 | 0.411833 | 0.426943 | 0.856560 | 6.3188 | 27.0804 | 42.4836 |
| zero | 28 | 0.418819 | 0.390820 | 0.854602 | 1.7415 | 26.3766 | 38.0000 |
| decoder_head | 28 | 0.421510 | 0.411386 | 0.856739 | 1.9776 | 26.7169 | 39.2506 |
| self | 28 | 0.421886 | 0.411146 | 0.856922 | 1.9698 | 26.5861 | 39.1302 |
| cross | 28 | 0.422329 | 0.406186 | 0.856826 | 1.9416 | 27.0807 | 39.0881 |
| history_mean512 | 28 | 0.111308 | 0.109026 | 0.661920 | 0.2872 | 29.8225 | 6.5892 |
| LEFT_native_same_roles | 3 | 0.376827 | 0.388016 | 0.886443 | 4.1846 | 23.7177 | 20.8257 |

## All29 records: common-axis AP / ROC

LEFT missing rows stayNA. All Head/Self/meanCross entries are frozen historical controls, not new Hybrid contributions.
| Entity | A_official_role | B_fit_only | zero | decoder_head | self | cross | history_mean512 | LEFT_native_same_roles |
|---|---|---|---|---|---|---|---|---|
| PSM | 0.359946 / 0.552157 | 0.303012 / 0.476555 | 0.432502 / 0.670965 | 0.427837 / 0.678159 | 0.427206 / 0.678538 | 0.426164 / 0.680347 | 0.316848 / 0.530450 | 0.571098 / 0.764667 |
| machine-1-1 | 0.180584 / 0.429793 | 0.417411 / 0.845914 | 0.612285 / 0.901876 | 0.605361 / 0.904564 | 0.606455 / 0.905165 | 0.619073 / 0.904915 | 0.206087 / 0.722219 | 0.388016 / 0.837270 |
| machine-1-2 | 0.452060 / 0.955374 | 0.447497 / 0.949967 | 0.286288 / 0.921543 | 0.303242 / 0.922140 | 0.302738 / 0.922145 | 0.307326 / 0.922357 | 0.105610 / 0.841062 | 0.398419 / 0.938496 |
| machine-1-3 | 0.601587 / 0.921610 | 0.047936 / 0.674246 | 0.351515 / 0.862250 | 0.375789 / 0.886172 | 0.376888 / 0.886283 | 0.378949 / 0.888234 | 0.154823 / 0.776147 | 0.344046 / 0.883564 |
| machine-1-4 | 0.521585 / 0.923060 | 0.167828 / 0.843433 | 0.223667 / 0.854000 | 0.233946 / 0.882162 | 0.233206 / 0.882453 | 0.231808 / 0.882659 | 0.132741 / 0.728227 | NA |
| machine-1-5 | 0.526362 / 0.920993 | 0.437797 / 0.892922 | 0.568672 / 0.920401 | 0.571616 / 0.931331 | 0.571706 / 0.930612 | 0.565255 / 0.930239 | 0.092879 / 0.836889 | NA |
| machine-1-6 | 0.696393 / 0.894334 | 0.868777 / 0.954486 | 0.645779 / 0.845255 | 0.648266 / 0.839420 | 0.649187 / 0.839920 | 0.655186 / 0.842746 | 0.184516 / 0.529503 | NA |
| machine-1-7 | 0.527654 / 0.824600 | 0.548657 / 0.783030 | 0.277814 / 0.660937 | 0.274647 / 0.650248 | 0.274697 / 0.650294 | 0.274470 / 0.650794 | 0.149209 / 0.543295 | NA |
| machine-1-8 | 0.336932 / 0.826037 | 0.312765 / 0.777077 | 0.307219 / 0.771991 | 0.303054 / 0.770390 | 0.303252 / 0.770742 | 0.306029 / 0.771075 | 0.086292 / 0.723381 | NA |
| machine-2-1 | 0.246409 / 0.696873 | 0.239843 / 0.661628 | 0.135436 / 0.652772 | 0.143046 / 0.655567 | 0.143734 / 0.655614 | 0.145794 / 0.659266 | 0.075144 / 0.583522 | NA |
| machine-2-2 | 0.258574 / 0.794332 | 0.253857 / 0.768933 | 0.192549 / 0.654380 | 0.195683 / 0.668751 | 0.196169 / 0.669847 | 0.196646 / 0.670194 | 0.129481 / 0.516280 | NA |
| machine-2-3 | 0.507481 / 0.922508 | 0.436475 / 0.947836 | 0.236370 / 0.972557 | 0.245490 / 0.975819 | 0.245443 / 0.975836 | 0.245977 / 0.975881 | 0.167200 / 0.817058 | NA |
| machine-2-4 | 0.573822 / 0.814096 | 0.453431 / 0.705491 | 0.314833 / 0.729295 | 0.295441 / 0.715229 | 0.295576 / 0.715166 | 0.301892 / 0.716391 | 0.139047 / 0.601959 | NA |
| machine-2-5 | 0.494220 / 0.946614 | 0.534610 / 0.951659 | 0.478696 / 0.910915 | 0.488595 / 0.914457 | 0.489935 / 0.914876 | 0.497454 / 0.914023 | 0.110076 / 0.677582 | NA |
| machine-2-6 | 0.472487 / 0.918472 | 0.448124 / 0.936834 | 0.402153 / 0.848959 | 0.409134 / 0.851636 | 0.408667 / 0.850575 | 0.409668 / 0.849084 | 0.090834 / 0.812923 | NA |
| machine-2-7 | 0.782247 / 0.944977 | 0.803110 / 0.958915 | 0.386759 / 0.902210 | 0.413639 / 0.906662 | 0.416425 / 0.907980 | 0.402705 / 0.901798 | 0.067527 / 0.531037 | NA |
| machine-2-8 | 0.973573 / 0.999818 | 0.146189 / 0.973764 | 0.916013 / 0.999284 | 0.908620 / 0.999273 | 0.906337 / 0.999261 | 0.904419 / 0.999252 | 0.085966 / 0.704908 | NA |
| machine-2-9 | 0.306047 / 0.607110 | 0.751893 / 0.930386 | 0.817286 / 0.973008 | 0.819686 / 0.972740 | 0.820481 / 0.972780 | 0.819991 / 0.971746 | 0.107976 / 0.628576 | NA |
| machine-3-1 | 0.692472 / 0.978193 | 0.811754 / 0.976250 | 0.557214 / 0.909892 | 0.566719 / 0.922054 | 0.568326 / 0.922389 | 0.567932 / 0.923965 | 0.132184 / 0.734164 | NA |
| machine-3-10 | 0.875662 / 0.957500 | 0.908464 / 0.980192 | 0.394881 / 0.907366 | 0.429616 / 0.913502 | 0.430025 / 0.913828 | 0.429044 / 0.913517 | 0.111202 / 0.583099 | NA |
| machine-3-11 | 0.591046 / 0.909264 | 0.058473 / 0.937200 | 0.511749 / 0.973202 | 0.455213 / 0.971776 | 0.459231 / 0.972613 | 0.457087 / 0.972001 | 0.128676 / 0.750968 | NA |
| machine-3-2 | 0.072381 / 0.593064 | 0.099660 / 0.718741 | 0.103240 / 0.672195 | 0.099528 / 0.649961 | 0.099995 / 0.650188 | 0.099228 / 0.648194 | 0.052214 / 0.489534 | NA |
| machine-3-3 | 0.294860 / 0.856879 | 0.061213 / 0.736656 | 0.154233 / 0.802511 | 0.159455 / 0.803975 | 0.159922 / 0.804813 | 0.158002 / 0.798708 | 0.076398 / 0.609209 | NA |
| machine-3-4 | 0.367225 / 0.786704 | 0.280082 / 0.871542 | 0.416953 / 0.819808 | 0.414082 / 0.795876 | 0.413625 / 0.795653 | 0.414444 / 0.796116 | 0.066339 / 0.543978 | NA |
| machine-3-5 | 0.669148 / 0.940746 | 0.045515 / 0.801041 | 0.327542 / 0.833797 | 0.334554 / 0.842656 | 0.333168 / 0.842465 | 0.332548 / 0.841910 | 0.127660 / 0.862140 | NA |
| machine-3-6 | 0.451423 / 0.901323 | 0.529392 / 0.942524 | 0.676120 / 0.971695 | 0.682233 / 0.973618 | 0.682151 / 0.973546 | 0.681599 / 0.974274 | 0.063841 / 0.543576 | NA |
| machine-3-7 | 0.229046 / 0.632264 | 0.372101 / 0.821252 | 0.595775 / 0.892284 | 0.581848 / 0.895896 | 0.581631 / 0.895871 | 0.581718 / 0.895951 | 0.049622 / 0.569942 | NA |
| machine-3-8 | 0.177807 / 0.593361 | 0.284563 / 0.664710 | 0.285862 / 0.834864 | 0.295487 / 0.844328 | 0.295620 / 0.844593 | 0.295396 / 0.842716 | 0.073286 / 0.523111 | NA |
| machine-3-9 | 0.761581 / 0.976164 | 0.763896 / 0.977063 | 0.550025 / 0.929608 | 0.548295 / 0.928503 | 0.548215 / 0.928295 | 0.545577 / 0.933128 | 0.149808 / 0.749470 | NA |

## Findings and costs

On SMD28, A-minus-TimeRadar-zero AP averages+6.8348 percentage points (20 wins/8 losses). B-minus-zero averages-0.6986pp (14/14). A has higher AP and lower normal false alarms, but lower ROC and lower sustained coverage: C3 coverage30.230% vszero38.000%. Eligible long-steady coverage (7 machines) is A4.770%, B29.107%, zero20.664%; B carries strict normal FPR7.144% versuszero2.853% (C3 B6.319% vszero1.741%). Neither permission establishes universal dominance.

PSM common-axis AP is A.359946/B.303012 versus TimeRadar-zero.432502 and LEFT.571098. A/B C3 event hits34/28 exceed zero2 and LEFT8, while C3 normal false-positive rates2.151%/1.719% exceed both references0%. More discoveries do not establish an equal-false-alarm advantage. Per-event offsets, any-hit versus one-to-one matching, false-positive points/fragments, long gaps and coverage are retained in full tables and raw event details.

A fits population mean/std on each entire supplied role, including later observations. CAL and QUERY are standardized separately, never concatenated. B uses raw FIT population mean/std once, std0 mapped to the official1e-8, and bypasses API normalization. The weights, model, head, precision, windows and padding are identical. A/B differences demonstrate sensitivity to statistical permissions, not an isolated causal effect of discriminative loss. Different context lengths/architectures and training permissions remain distinct from data-axis matching. B is a project adaptation, not native API.

Full native-axis results are in native_29x6.csv and are descriptive: TimeRCD retains every real tail point while old native TimeRadar stops at complete100-point windows. Only common_29x8.csv supports same-point cross-method pairing. That table has232 rows including25LEFT NA; native table174. Compatible old native116/common149 rows were reused, not inferred again. The three LEFT SMD cases and PSM retain native20epoch/random initialization and their earlier incomplete SMD curriculum; no LEFT retraining occurred.

Original probability, both head logits and class difference, original indices, full-window availability, repeated-tail/internal-patch padding and valid masks are saved in116 CAL/QUERY banks. Padding predictions never enter real-point metrics. Thresholds use each method's own legal full CAL q99higher strict>, no QUERY label or score-based model/threshold/window selection. A availability reflects whole-role statistics, B full-window context; neither is strict online.

## Reproduction and verification

Pinned author commit b031773e323bc0e6c9dcd8c9119b800ec4d70f8c. Official HF multivariate checkpoint revision0880420070a524efe14d7e163b34dcca90d2be1b;444841116bytes; SHA256 a497ac910f24f8fb347d690a21b60d74bc04785ebc777880553c158dea4434d0. See API.md for exact preprocessing/head and source attribution. Strict official checkpoint loader succeeded; actual API and cached A probabilities/raw logit differences matched bit-for-bit on FIT5017 for C25/C38, with true tail padding checks. Author source/model/head were not modified.

58 full inference jobs and2 FIT parity jobs all exited0. Actual occupied GPU time0.23740052507983314h, peak14,291,545,600 bytes; no training or failed GPU job. GPU0 carried unrelated work and was not displaced; GPU1–3 were released after completion. CPU metric evaluation4.297s. Root recomputed116 new native/common AP/ROC, own CAL thresholds, strict false-positive counts and raw head-logit differences from the banks; this is primary recomputation, not an external independent audit. An initial primary checker used the wrong common-axis field name; its check failed, was corrected to axis_start/axis_end_exclusive, and passed without changing scientific results. An earlier RoPE concern was an incorrect reading of a comment, withdrawn before actual API execution; no author modification was made. Network retrieval failures were retained separately from GPU jobs.

## Published values versus this benchmark

All comparison tables are local, fixed-protocol measurements. The official repository separately reports TSB-AD multivariate VUS-PR0.32; that is a different metric/dataset setting, not PSM/SMD AP and not inserted in our table. No unverified paper table values are filled in. Official reference: https://github.com/thu-sail-lab/Time-RCD .

TimeRCD remains an external method, never a renamed Hybrid contribution. The closed shallow-adaptation branch stays closed. This result neither isolates a discriminative-loss cause nor authorizes score fusion/new Hybrid training. No further experiment was started.

## Artifacts

- [Complete common-axis comparison](common_29x8.csv), [native full-axis results](native_29x6.csv).
- [SMD macro/median](common_SMD_macro_median.csv), [paired entity differences](common_SMD_paired_deltas.csv), [paired summaries](common_SMD_paired_summary.csv).
- [Native-axis macro/median](native_SMD_macro_median.csv).
- Public CSV is LF-normalized; raw banks/results remain unchanged. Raw observations, official weights/source and private server identities are excluded from this Git bundle.
