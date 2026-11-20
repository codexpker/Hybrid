# Target-mean / source-copy reconstruction: fixed native TimeRadar V comparison

Completed20 November2026; public SWaT31/PSM25, seed2025. `R-MeanT` denotes code arm `R_target_mean`: only U-FIT uses the differentiable ten-copy mean reconstruction MSE; SOURCE_FIT retains the original copy MSE. New model initialization is the official pretrained state, not the old R endpoint. Two new models, four new banks; old noU/TT/SS/R and compatible external results are reused unchanged.

## Complete method table

| Task | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| SWAT | noU | 0.795725 | 0.857852 | 0.576600 | 0.708990 |
| SWAT | TT | 0.767373 | 0.840286 | 0.635916 | 0.795740 |
| SWAT | SS | 0.759642 | 0.840201 | 0.629941 | 0.793154 |
| SWAT | R | 0.761478 | 0.838195 | 0.638054 | 0.789673 |
| SWAT | R-MeanT | 0.759388 | 0.841620 | 0.610312 | 0.783951 |
| PSM | noU | 0.550952 | 0.735048 | 0.484578 | 0.648161 |
| PSM | TT | 0.497243 | 0.712915 | 0.520744 | 0.714204 |
| PSM | SS | 0.534776 | 0.731992 | 0.520077 | 0.697853 |
| PSM | R | 0.551502 | 0.734425 | 0.552609 | 0.721470 |
| PSM | R-MeanT | 0.560142 | 0.742163 | 0.546871 | 0.714094 |

AP is Average Precision, not trapezoidal AUC-PR. The two new VUS states use the already pinned official LEFT RangeAUC_volume_opt(windowSize100,thre250). All old VUS values are reused. Native scoring remains terminal99 mean-channel ten-copy variance with correction1; E never enters the score.

The same method table's operating panel follows. Strict and C3 are separate; C3 requires three consecutive past/current alarms without backfill. Point recall includes all QUERY anomaly points, whereas event coverage uses complete eligible events. One-to-one is chronological first-unmatched overlap greedy matching, not maximum matching or any-hit. Delays and misses are sample points; delay medians condition on detected events, with nonhits separately censored in events.csv. Unknown/ineligible events retain NA coverage/misses; observed_alarm_points=0 does not establish a complete miss.

| Task | Method | Stream | Normal FP points | Normal FPR % | Point recall % | Point F1 % | Event P/R/F1 % | Matched/eligible | Any-hit | Normal fragments | Event coverage % | Long macro/pooled % | Longest miss | Hit-only median delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | noU | strict | 3759 | 1.303 | 69.169 | 78.141 | 7.812/45.455/13.333 | 10/22 | 10 | 122 | 69.169 | 11.725/82.587 | 3939 | 135.0 |
| SWAT | noU | confirmed3 | 3529 | 1.223 | 69.123 | 78.322 | 8.065/45.455/13.699 | 10/22 | 10 | 118 | 69.123 | 11.460/82.566 | 3941 | 137.0 |
| SWAT | TT | strict | 3945 | 1.367 | 66.743 | 76.278 | 9.009/45.455/15.038 | 10/22 | 10 | 108 | 66.743 | 11.455/79.685 | 5067 | 131.5 |
| SWAT | TT | confirmed3 | 3740 | 1.296 | 66.710 | 76.442 | 9.091/45.455/15.152 | 10/22 | 10 | 107 | 66.710 | 11.310/79.669 | 5069 | 133.5 |
| SWAT | SS | strict | 3901 | 1.352 | 66.875 | 76.411 | 9.174/45.455/15.267 | 10/22 | 10 | 106 | 66.875 | 11.458/79.809 | 5018 | 126.0 |
| SWAT | SS | confirmed3 | 3699 | 1.282 | 66.842 | 76.573 | 9.174/45.455/15.267 | 10/22 | 10 | 106 | 66.842 | 11.314/79.793 | 5020 | 128.0 |
| SWAT | R | strict | 4054 | 1.405 | 67.072 | 76.410 | 8.547/45.455/14.388 | 10/22 | 10 | 114 | 67.072 | 11.492/79.989 | 4949 | 122.0 |
| SWAT | R | confirmed3 | 3838 | 1.330 | 67.039 | 76.584 | 8.696/45.455/14.599 | 10/22 | 10 | 112 | 67.039 | 11.347/79.974 | 4951 | 124.0 |
| SWAT | R-MeanT | strict | 3815 | 1.322 | 66.499 | 76.223 | 8.475/45.455/14.286 | 10/22 | 10 | 114 | 66.499 | 11.411/79.417 | 5170 | 134.5 |
| SWAT | R-MeanT | confirmed3 | 3597 | 1.247 | 66.461 | 76.396 | 9.524/45.455/15.748 | 10/22 | 10 | 102 | 66.461 | 11.266/79.401 | 5172 | 136.5 |
| PSM | noU | strict | 14036 | 29.669 | 50.870 | 43.958 | 8.955/37.500/14.458 | 18/48 | 28 | 185 | 73.659 | 86.955/92.407 | 486 | 0.0 |
| PSM | noU | confirmed3 | 13765 | 29.096 | 48.878 | 42.894 | 7.059/12.500/9.023 | 6/48 | 17 | 88 | 73.195 | 86.453/92.114 | 694 | 0.0 |
| PSM | TT | strict | 1509 | 3.190 | 18.331 | 28.868 | 5.037/56.250/9.247 | 27/48 | 30 | 350 | 21.300 | 29.341/25.073 | 708 | 1.0 |
| PSM | TT | confirmed3 | 1022 | 2.160 | 14.249 | 23.725 | 5.618/20.833/8.850 | 10/48 | 11 | 102 | 18.258 | 24.441/22.016 | 1228 | 7.0 |
| PSM | SS | strict | 9089 | 19.212 | 47.442 | 47.532 | 6.720/52.083/11.905 | 25/48 | 31 | 336 | 68.107 | 68.984/85.545 | 534 | 0.0 |
| PSM | SS | confirmed3 | 8604 | 18.187 | 44.469 | 45.876 | 6.250/18.750/9.375 | 9/48 | 16 | 128 | 67.197 | 66.713/84.927 | 1228 | 0.0 |
| PSM | R | strict | 2380 | 5.031 | 32.080 | 44.023 | 4.396/58.333/8.175 | 28/48 | 28 | 462 | 43.917 | 46.044/57.285 | 534 | 0.0 |
| PSM | R | confirmed3 | 1698 | 3.589 | 28.073 | 40.738 | 4.933/22.917/8.118 | 11/48 | 11 | 153 | 41.008 | 42.016/54.195 | 1228 | 0.0 |
| PSM | R-MeanT | strict | 4987 | 10.541 | 40.795 | 48.157 | 4.471/62.500/8.345 | 30/48 | 32 | 510 | 57.021 | 57.394/73.870 | 400 | 0.0 |
| PSM | R-MeanT | confirmed3 | 4222 | 8.924 | 36.546 | 45.460 | 4.511/25.000/7.643 | 12/48 | 14 | 189 | 54.150 | 53.380/72.179 | 879 | 0.0 |

## Single-factor learning comparison

New minus control, n=1; no sample standard deviation is estimable. Rate differences below are percentage points. Count and longest-miss differences remain original units. All exact fields are in single_factor_pairs.csv.

| Task | New minus | Stream | AP pp | ROC pp | VUS-PR pp | VUS-ROC pp | FPR pp | Recall pp | Matched delta | Any-hit delta | Event F1 pp | Long macro pp | Long pooled pp | Longest-miss delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | R | strict | -0.209 | 0.342 | -2.774 | -0.572 | -0.083 | -0.573 | 0 | 0 | -0.103 | -0.081 | -0.573 | 221 |
| SWAT | noU | strict | -3.634 | -1.623 | 3.371 | 7.496 | 0.019 | -2.671 | 0 | 0 | 0.952 | -0.315 | -3.170 | 1231 |
| SWAT | TT | strict | -0.798 | 0.133 | -2.560 | -1.179 | -0.045 | -0.245 | 0 | 0 | -0.752 | -0.044 | -0.268 | 103 |
| SWAT | SS | strict | -0.025 | 0.142 | -1.963 | -0.920 | -0.030 | -0.377 | 0 | 0 | -0.981 | -0.048 | -0.392 | 152 |
| SWAT | R | confirmed3 | -0.209 | 0.342 | -2.774 | -0.572 | -0.084 | -0.578 | 0 | 0 | 1.149 | -0.081 | -0.573 | 221 |
| SWAT | noU | confirmed3 | -3.634 | -1.623 | 3.371 | 7.496 | 0.024 | -2.662 | 0 | 0 | 2.049 | -0.194 | -3.165 | 1231 |
| SWAT | TT | confirmed3 | -0.798 | 0.133 | -2.560 | -1.179 | -0.050 | -0.249 | 0 | 0 | 0.597 | -0.044 | -0.268 | 103 |
| SWAT | SS | confirmed3 | -0.025 | 0.142 | -1.963 | -0.920 | -0.035 | -0.381 | 0 | 0 | 0.481 | -0.048 | -0.392 | 152 |
| PSM | R | strict | 0.864 | 0.774 | -0.574 | -0.738 | 5.511 | 8.715 | 2 | 4 | 0.170 | 11.350 | 16.585 | -134 |
| PSM | noU | strict | 0.919 | 0.712 | 6.229 | 6.593 | -19.127 | -10.075 | 12 | 4 | -6.113 | -29.562 | -18.537 | -86 |
| PSM | TT | strict | 6.290 | 2.925 | 2.613 | -0.011 | 7.352 | 22.464 | 3 | 2 | -0.902 | 28.053 | 48.797 | -308 |
| PSM | SS | strict | 2.537 | 1.017 | 2.679 | 1.624 | -8.671 | -6.648 | 5 | 1 | -3.560 | -11.590 | -11.675 | -134 |
| PSM | R | confirmed3 | 0.864 | 0.774 | -0.574 | -0.738 | 5.335 | 8.474 | 1 | 3 | -0.475 | 11.363 | 17.984 | -349 |
| PSM | noU | confirmed3 | 0.919 | 0.712 | 6.229 | 6.593 | -20.172 | -12.331 | 6 | -3 | -1.379 | -33.073 | -19.935 | 185 |
| PSM | TT | confirmed3 | 6.290 | 2.925 | 2.613 | -0.011 | 6.764 | 22.297 | 2 | 3 | -1.206 | 28.939 | 50.163 | -349 |
| PSM | SS | confirmed3 | 2.537 | 1.017 | 2.679 | 1.624 | -9.263 | -7.922 | 3 | -2 | -1.732 | -13.333 | -12.748 | -349 |

## Scientific decision

PSM retains a limited positive: against R, point AP +0.864pp and ROC +0.774pp; C3 recall28.073%→36.546%, complete-event coverage41.008%→54.150%, long steady macro42.016%→53.380%, pooled54.195%→72.179%, and longest miss1228→879. Matched events11→12, any-hit11→14 are distinct counts. However C3 normal FPR3.589%→8.924% (+2524 normal alarm points), normal fragments+36, event F1.081181→.076433, and VUS-PR/ROC fall. Strict FPR5.031%→10.541%. This is an actual ranking increment independent of CAL, plus an operating trade-off, not low-false-alarm sustained-detection superiority.

SWaT has no corresponding clear detection increment: against R, AP -0.209pp, ROC +0.342pp, VUS-PR -2.774pp and VUS-ROC -.572pp. C3 normal FPR1.330%→1.247% and event F1 improves, but recall67.039%→66.461%, long macro11.347%→11.266%, pooled79.974%→79.401%, longest miss4951→5172, with matched/any-hit10 unchanged. Lower false alarms alone do not establish evidence preservation.

Against noU, PSM improves AP/ROC and lowers its very high false-alarm rate, but still loses recall and sustained coverage. Against SS, it improves ranking and lowers FPR but loses long coverage. SWaT adapted point AP/ROC remain below noU; both new VUS values remain above noU, preserving the original range-versus-point trade-off. The complete contrasts above retain these positive and negative findings rather than selecting one success metric.

**Decision:** the source/target constraint change produces a bounded PSM ranking/coverage positive with a substantial false-alarm cost, but no stable two-task preservation advantage. It is not established as an independently confirmed Hybrid core contribution or contamination robustness. Retain this exact version and both tasks without automatic loss coefficients, TT-mean, teacher, backbone, seed, data or threshold extensions. Neither official TimeRadar, replay nor the classical decomposition is claimed as original. This is not evidence that the removed variance term was the sole cause of historical failures.

## Same-forward training decomposition

| Task | Update interval | E_T | V_T full-window | E_S | V_S full-window | New loss |
| --- | --- | --- | --- | --- | --- | --- |
| SWAT | updates1-100 | 0.771768 | 0.032594 | 0.673468 | 0.034585 | 0.738181 |
| SWAT | updates1901-2000 | 0.630747 | 0.013254 | 0.555205 | 0.013034 | 0.598842 |
| PSM | updates1-100 | 0.791064 | 0.087087 | 0.603172 | 0.057175 | 0.722847 |
| PSM | updates1901-2000 | 0.479793 | 0.029753 | 0.450032 | 0.014464 | 0.471421 |

E/V are mean per-half full-window statistics from the same training forward, averaged over the named100 updates; these intervals contain different sampled windows and evolving weights, not a fixed-cohort counterfactual. V decreases during both new trainings; a variance increase was never a success criterion. Old R TRACE did not archive separate E/V, so no new old-model forward is run to manufacture that comparison.

CPU FP64 loss/gradient identity and real normal-source FP32 gates verify copyMSE=E+0.9V_fullwindow and, at identical parameters/windows/masks, L_R-L_new=0.45mean(V_T). The actual target objective computes E directly; it does not subtract large losses. Both source and target windows divide by64 once, with32+32 slots. The new4000-step traces satisfy loss=.5E_T+.5copy_S; decomposition errors remain within documented floating tolerances. This identity cannot be applied by subtracting the losses of already diverged R and R-MeanT trajectories. Training V_fullwindow is not the terminal native score.

## Existing external results, original permissions

| Task | External method | Readout permission | AP | ROC-AUC | VUS-PR | VUS-ROC | Original strict FPR % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | MLP_noU | CAUSAL_ENDPOINT | 0.783949 | 0.856207 | 0.543552 | 0.696882 | 97.186 |
| SWAT | CANDI_A | CAUSAL_ENDPOINT | 0.783885 | 0.856961 | 0.541710 | 0.695700 | 97.502 |
| SWAT | MLP_TT | CAUSAL_ENDPOINT | 0.776102 | 0.845163 | 0.514483 | 0.665694 | 97.057 |
| SWAT | MLP_R | CAUSAL_ENDPOINT | 0.782007 | 0.863262 | 0.541789 | 0.702566 | 96.612 |
| PSM | MLP_noU | CAUSAL_ENDPOINT | 0.606531 | 0.815061 | 0.627826 | 0.823338 | 4.327 |
| PSM | CANDI_A | CAUSAL_ENDPOINT | 0.606531 | 0.815059 | 0.627825 | 0.823337 | 4.327 |
| PSM | MLP_TT | CAUSAL_ENDPOINT | 0.492378 | 0.763049 | 0.538933 | 0.783649 | 3.494 |
| PSM | MLP_R | CAUSAL_ENDPOINT | 0.635775 | 0.828137 | 0.685514 | 0.852250 | 0.996 |
| SWAT | LEFT_noU | OFFLINE_CONTEXT | 0.607560 | 0.842267 | 0.606345 | 0.829634 | 0.988 |
| SWAT | LEFT_TT | OFFLINE_CONTEXT | 0.610342 | 0.884582 | 0.618301 | 0.881642 | 0.940 |
| SWAT | LEFT_R | OFFLINE_CONTEXT | 0.629347 | 0.885512 | 0.639265 | 0.884546 | 0.937 |
| PSM | LEFT_noU | OFFLINE_CONTEXT | 0.586600 | 0.776988 | 0.577678 | 0.780826 | 18.692 |
| PSM | LEFT_TT | OFFLINE_CONTEXT | 0.577794 | 0.744370 | 0.566977 | 0.750572 | 26.874 |
| PSM | LEFT_R | OFFLINE_CONTEXT | 0.593208 | 0.759509 | 0.590745 | 0.771875 | 17.047 |
| PSM | HistoryMean512 | CAUSAL_ENDPOINT | 0.303673 | 0.525589 | 0.332386 | 0.567946 | 3.018 |
| PSM | TimeRadar_native | OFFLINE_CONTEXT | 0.430540 | 0.688432 | 0.437702 | 0.677534 | NA |
| SWAT | HistoryMean512 | CAUSAL_ENDPOINT | 0.087661 | 0.227761 | 0.090893 | 0.232695 | NA |

External rows keep their accepted original input/scaler/training/CAL permissions in methods.csv; they are not same-backbone controls or new reproductions. LEFT uses192-point nonoverlapping full-window retrospective scores and normal-prefix statistics including SELECT; MLP/CANDI use10-point whole-window MSE at its endpoint and source-FIT statistics, with distinct objectives and exposure. Historical TimeRadar_native has its original offline/batched/normalization conditions and is distinct from the fixed B1 native-V noU. History Mean512 has only ranking where its CAL was incompatible; no invented operating point is added. The primary five methods share exact current512 CAL points; external historical CAL sets/thresholds are not assumed identical (including SWaT earlier38-point versus current34-point attack composition). Their original operating metrics remain descriptive, not a matched normal-FPR comparison. No pure time-frequency or pollution attribution follows from a cross-model gap.

## Curves and frozen permissions

[SWaT primary](curves/SWAT_TARGET_MEAN.png), [PSM primary](curves/PSM_TARGET_MEAN.png), [SWaT external](curves/SWAT_EXTERNAL.png), [PSM external](curves/PSM_EXTERNAL.png) show full PR, full ROC and normal-FPR0–5% ROC. Every original CAL marker is retained; outside-panel markers are explicitly listed. In PSM the new strict operation is outside5% FPR and is not moved into the panel. Curves can cross; no QUERY-derived operating threshold is adopted. Full numeric curve arrays remain in the results cache and are SHA-bound in PROVENANCE.

All five primary methods use QUERY SWaT[608535,944856)=336321 and PSM[23016,87744)=64728, original512 CAL observations, ownq99/higher/strict> threshold. SWaT CAL has34 attack points; PSM all512 are attack points. q99 is an empirical mixed-CAL operation, not a normal1% false-alarm guarantee. Complete events22/48, long events9/5; long duration>512, steady from original start+512. No private first-paper data, QUERY model selection, new donor or contamination dose was used. Information footprints remain causal100-point endpoint windows; these are exposed public development evaluations, not deployment certification or blind confirmation.

Only decoder793610/1636043 parameters update. Official PT, scaler, originalR PLAN/TASK bytes,32+32 streams, B1 own-order/masks84551/norm0, AdamW1e-4/betas(.9,.95)/eps1e-8/original literal-bias decay grouping/warmup200,2000 endpoint updates/128000 windows per model, and inference V are unchanged. All4000 order hashes match the corresponding oldR update. No SELECT checkpoint selection occurs.

## Actual compute and review

14 owned GPU jobs, allOS0/cleanup:2 normal engineering gates,2 trainings and10 native readouts. Actual total**2.747929552027425 A40 GPU hours**, including startup/cleanup, with no failed or repeated scientific work. Gate11.720744s, training5264.074430s, readout4616.751214s; total9892.546387s. Training body SWaT2655.299610s/PSM2601.472621s; peakallocated202.187/172.740MiB. No live GPU task remains.

22 affected CPU tests passed. Main evaluation CPUwall36.314353s; one same-family role-separated key audit checked45048 fields,4 new method-stream rows,16 contrasts and214 new raw event rows, maximum difference2.220446049250313e-16. The48 old method-stream rows and old VUS were reused. Review reuses the fixed official VUS code for two new states, not an independent mathematical rewrite; canonical savedFP32V is retained, with CPU copy-variance reconstruction checked only within tolerance. Independent later metadata/NumPy plus shared VUS wall29.1136s includes the6.4949s metadata subset (do not add twice); brief separately recorded overhead is not an exact all-command CPU total. A limited export check completes the same review, without model forward or extra GPU work.

Actual runtime bytes/config/environment and REUSE_BINDING remain immutable. The actual CONFIG preceded a metadata-only reuse_metrics key in the Git preparation routine; the separate REUSE_BINDING pins both old hashes without rewriting CONFIG, TASK or runtime. Training calculations match the fixed implementation; raw byte manifests, not text-normalized Git hashes, identify executed files. Full native endpoints/optimizer include upstream weights and stay outside publicGit; raw data, labels, scalers and full banks also stay outside. PROVENANCE records their identities and costs.
