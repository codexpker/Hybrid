# Fixed LEFT multiscale decoder conditioning comparison

[Complete public methods, structural ablation and cost](TABLES.md) · [Full precision results](METRICS.json) · [Normal training summary](training.csv) · [Independent check](review.json)

The three arms start from the same public configuration and seed on each task. The snapshot implementation protects prototype reads before the official in-place EMA and is shared across all arms. The matched native controls were trained in this implementation to keep gradients and training state comparable. The two active paths use the same existing `z_fuse` and `to_gb` with different inputs. The exact modulation is written in the manuscript. Each path activates 33,280 previously registered parameters. Active parameter counts are measured from Adam state.

The original LEFT training and native score are retained. Normal FIT and SELECT are [0, 105984) and [105984, 132481) on PSM, and [0, 396000) and [396000, 495000) on SWaT. The scaler uses the complete normal prefix. Each arm uses training stride 1, validation and test stride 192, batch128, Adam at an initial learning rate of 0.0001, the per-task native learning-rate schedule, at most 20 epochs and patience3. PSM uses 25 variables, schedule type2 and curriculum warmup/ramp 1000/2000. SWaT uses 31 variables, schedule type1 and warmup/ramp 500/1000. The maximum curriculum weight is 0.3 in both tasks. The normal-validation loss selects checkpoints using the author's early stopping implementation. Shared epochs have matching CPU sampler RNG fingerprints.

The real normal-window check reproduced native loss and score exactly with zero modulation. The two conditioned arms show nonzero projection gradients after curriculum warmup. Each pass makes one call to each prototype bank and increments the training step once. The check forces an eligible EMA batch to exercise the update path without changing any training configuration. Sealed inference preserves model tensors and step. SWaT omits 63 tail points and PSM omits 97. Each score becomes available at its W192 window end. PSM uses the test-relative axis [0, 87744). SWaT uses the absolute axis [495000, 944856). Both are offline whole-window comparisons. The matched LEFT arms share seed2025 and normal-data permissions. TimeRadar, CATCH and PCA retain the permissions recorded in their cached metadata. Historical costs are shown as NA when this table lacks a comparable measurement. Operation metrics are NA because a compatible independent calibration split has not been established on these full axes.

On PSM the prototype-conditioned method improved all four ranking metrics against both controls. The AP increase was 0.001301 versus native LEFT and 0.001291 versus observation conditioning. This single-seed difference is small. Both conditioned arms selected epoch 15, so their direct comparison shares the same selected update count and effective modulation capacity.

On SWaT, adding observation context decreased AP from 0.543866 to 0.490931. Prototype conditioning decreased it further to 0.140230. The prototype arm loses AP, ROC-AUC and both VUS metrics against both controls. The gaps against the capacity-matched control are -0.350700 AP, -0.261753 ROC-AUC, -0.314399 VUS-PR and -0.228073 VUS-ROC. Candidate selection used normal loss, which cannot locate the cause of the test ranking loss. This comparison supports a small PSM-specific increment and a large SWaT penalty. It does not support direct prototype conditioning as a generally effective modification of LEFT.

The six jobs consumed 13840.273 A40-seconds in training and 14.250 A40-seconds in readout, totalling **3.848479 GPU-hours**. Small front checks and startup were outside these measurements. All six checkpoints, scaler arrays and score banks passed local hash verification and remain in the workspace results directory and on the server. They are stored outside Git. No thresholds participate in the results. Public normal arrays are the only learning input. The independent CPU check reconstructed normal selection, active capacity, score axes, and AP/ROC-AUC from all six raw banks. VUS review covered the paired differences and reused the primary integration. The two archived TimeRadar block-0 score arrays were verified against their manifest axes and hashes. The third transferred TimeRadar cache had explicit indices already. No external model was trained to fill a missing row.

[Method section and results in the draft](../../../manuscript/next_generation_20261115/draft.md) · [LEFT](https://arxiv.org/abs/2602.08638) · [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html) · [FiLM](https://aaai.org/papers/11671-film-visual-reasoning-with-a-general-conditioning-layer/)

## SWaT post-hoc intervention at fixed weights

For each selected conditioned checkpoint, the intervention replaced the modulated token by the original token inside `_condition_ms` and sent it through the original unconditioned decoder. The existing cycle reconstruction retained its native memory mixing at $\lambda = 0.3$, and prototype reads and scoring components kept their original definitions. All checkpoints, scalers, model tensors, prototypes and inference steps were preserved. A replay with modulation enabled reproduced the archived per-channel and channel-mean scores bit for bit on the first 128 complete windows (24,576 points). Both cycle decoder outputs agreed bit for bit between the on and off passes on that batch. Each off bank covers all 449,856 points of the original absolute axis [495000, 944856), with the same window-end availability (395,235 normal and 54,621 attack-labeled points). The on and native scores are archived references.

| Method and modulation | AP | ROC-AUC | Spearman vs same-weight on | Normal mean / median [Q1, Q3] | Attack mean / median [Q1, Q3] |
|---|---:|---:|---:|---|---|
| Native LEFT (archived) | 0.543866 | 0.820853 | — | 0.018373 / 0.004172 [0.001579, 0.011844] | 0.169560 / 0.243303 [0.008895, 0.266562] |
| Observation on (archived) | 0.490931 | 0.791439 | — | 0.078042 / 0.083224 [0.003569, 0.125559] | 0.221958 / 0.287921 [0.108622, 0.316813] |
| Observation off (intervention) | 0.713724 | 0.838712 | 0.730442 | 0.786225 / 0.788536 [0.634079, 0.938120] | 1.499647 / 1.769836 [1.028541, 1.909630] |
| Prototype on (archived) | 0.140230 | 0.529687 | — | 0.133176 / 0.143518 [0.003816, 0.227532] | 0.171586 / 0.153201 [0.126602, 0.174096] |
| Prototype off (intervention) | 0.720355 | 0.838264 | 0.668258 | 1.369948 / 1.404068 [1.138807, 1.613332] | 2.242918 / 2.512061 [1.732975, 2.800809] |

Turning off direct modulation increased AP by **+0.222793** for observation conditioning and **+0.580124** for prototype conditioning. Both off variants exceeded native LEFT on AP and ROC-AUC at these fixed checkpoints. This intervention supports a negative effect of the immediate modulation on ranking in these trained SWaT models. It does not separate the effects of training-time co-adaptation or identify prototype collapse, excessive modulation strength, or anomaly over-reconstruction. The off scores are diagnostic results only. The selected checkpoints and formal results remain those of the original comparison.

The existing channel-mean readout gives channel $i$ a contribution of its raw score divided by 31. In both interventions, group-mean contributions rose on all 31 channels in both normal and attack-labeled periods. The average attack-minus-normal score gap rose from 0.143916 to 0.713421 for observation conditioning and from 0.038410 to 0.872970 for prototype conditioning. This describes contributions to the fixed score under system-level labels. Channel fault identities are unavailable from these labels.

<details>
<summary>All 31 channel contributions by system-level label (N normal, A attack)</summary>

| Channel | Native N / A | Observation N on → off | Observation A on → off | Prototype N on → off | Prototype A on → off |
|---|---:|---:|---:|---:|---:|
| FIT101 | 0.000399 / 0.000171 | 0.000506 → 0.028223 | 0.000545 → 0.025647 | 0.000491 → 0.054494 | 0.000224 → 0.040044 |
| LIT101 | 0.000047 / 0.000069 | 0.000142 → 0.022495 | 0.000391 → 0.038564 | 0.000114 → 0.039774 | 0.000651 → 0.069886 |
| P101 | 0.000440 / 0.000208 | 0.000552 → 0.026695 | 0.000623 → 0.028893 | 0.000520 → 0.053081 | 0.000280 → 0.043862 |
| AIT201 | 0.005155 / 0.045596 | 0.061084 → 0.152835 | 0.076335 → 0.147657 | 0.117101 → 0.216541 | 0.051004 → 0.185140 |
| AIT202 | 0.000145 / 0.001338 | 0.000471 → 0.064398 | 0.002159 → 0.057415 | 0.000474 → 0.112184 | 0.002950 → 0.100273 |
| AIT203 | 0.000015 / 0.000011 | 0.000163 → 0.011447 | 0.000092 → 0.005395 | 0.000107 → 0.015670 | 0.000098 → 0.010547 |
| FIT201 | 0.000389 / 0.000214 | 0.000503 → 0.026857 | 0.000632 → 0.029756 | 0.000471 → 0.053381 | 0.000287 → 0.045325 |
| P203 | 0.000430 / 0.000243 | 0.000543 → 0.026799 | 0.000656 → 0.028949 | 0.000511 → 0.053208 | 0.000315 → 0.043831 |
| P205 | 0.000374 / 0.000200 | 0.000461 → 0.027470 | 0.000514 → 0.022709 | 0.000437 → 0.055187 | 0.000239 → 0.035917 |
| DPIT301 | 0.000608 / 0.000349 | 0.000718 → 0.024142 | 0.000860 → 0.040589 | 0.000688 → 0.049060 | 0.000449 → 0.061327 |
| FIT301 | 0.000743 / 0.000493 | 0.000844 → 0.022042 | 0.001015 → 0.037312 | 0.000814 → 0.045311 | 0.000591 → 0.054472 |
| LIT301 | 0.000047 / 0.000088 | 0.000117 → 0.019321 | 0.000262 → 0.029203 | 0.000097 → 0.037376 | 0.000449 → 0.056375 |
| P301 | 0.000002 / 0.000005 | 0.000024 → 0.000340 | 0.000036 → 0.000413 | 0.000007 → 0.003333 | 0.000044 → 0.004023 |
| P302 | 0.000828 / 0.000583 | 0.000927 → 0.022396 | 0.001087 → 0.036089 | 0.000898 → 0.046089 | 0.000675 → 0.052901 |
| AIT401 | 0.000003 / 0.000005 | 0.000005 → 0.010297 | 0.000023 → 0.007454 | 0.000008 → 0.031842 | 0.000121 → 0.024447 |
| AIT402 | 0.000121 / 0.078636 | 0.000601 → 0.059753 | 0.081236 → 0.163447 | 0.000476 → 0.085906 | 0.039917 → 0.196598 |
| FIT401 | 0.000087 / 0.002408 | 0.000108 → 0.002831 | 0.003472 → 0.058472 | 0.000112 → 0.012122 | 0.012088 → 0.075144 |
| LIT401 | 0.000033 / 0.000158 | 0.000110 → 0.021020 | 0.001549 → 0.078290 | 0.000092 → 0.040131 | 0.000757 → 0.156121 |
| AIT501 | 0.000071 / 0.000412 | 0.000640 → 0.054911 | 0.010343 → 0.097026 | 0.000548 → 0.078641 | 0.004559 → 0.168794 |
| AIT502 | 0.000068 / 0.000778 | 0.000407 → 0.041338 | 0.001529 → 0.107826 | 0.000322 → 0.059131 | 0.004455 → 0.173232 |
| AIT503 | 0.000137 / 0.000067 | 0.000361 → 0.039084 | 0.000135 → 0.013097 | 0.000340 → 0.062546 | 0.000185 → 0.025465 |
| AIT504 | 0.000254 / 0.005560 | 0.000253 → 0.008080 | 0.005125 → 0.013910 | 0.000280 → 0.024990 | 0.006982 → 0.030367 |
| FIT501 | 0.000039 / 0.001398 | 0.000049 → 0.002719 | 0.001166 → 0.055981 | 0.000039 → 0.011885 | 0.007633 → 0.072383 |
| FIT502 | 0.000581 / 0.007753 | 0.000587 → 0.005218 | 0.007310 → 0.053566 | 0.000587 → 0.017508 | 0.003192 → 0.073484 |
| FIT503 | 0.000068 / 0.001949 | 0.000096 → 0.000593 | 0.002881 → 0.057918 | 0.000081 → 0.003498 | 0.009348 → 0.072458 |
| FIT504 | 0.000136 / 0.006518 | 0.000169 → 0.000863 | 0.006989 → 0.052740 | 0.000172 → 0.004897 | 0.010015 → 0.068990 |
| PIT501 | 0.000039 / 0.003791 | 0.000074 → 0.001139 | 0.003506 → 0.053557 | 0.000048 → 0.003144 | 0.004631 → 0.069747 |
| PIT502 | 0.000519 / 0.000572 | 0.000818 → 0.052007 | 0.001899 → 0.098095 | 0.000749 → 0.081132 | 0.001379 → 0.149936 |
| PIT503 | 0.000034 / 0.005297 | 0.000084 → 0.002043 | 0.004836 → 0.053331 | 0.000050 → 0.003067 | 0.003326 → 0.068311 |
| FIT601 | 0.003398 / 0.002408 | 0.003425 → 0.004540 | 0.002460 → 0.003227 | 0.003381 → 0.007409 | 0.002442 → 0.006705 |
| P602 | 0.003163 / 0.002281 | 0.003200 → 0.004328 | 0.002292 → 0.003119 | 0.003162 → 0.007408 | 0.002301 → 0.006812 |

</details>

The independent arithmetic check used tie-group AP and trapezoidal ROC from sorted points, rank covariance for Spearman, and the sum of channel contributions for each group mean. Maximum absolute difference across the checks was 1.35e-09, from the float32 channel mean. The two full off forwards took 0.559389 and 0.520555 A40-seconds (observation and prototype). The fixed-batch on replays took 0.638058 additional A40-seconds. Recorded forward cost totals 0.000477 A40 GPU-hours. CPU analysis took 5.150 seconds. Loading, startup, compression and a checkpoint-load preflight retry were outside the forward timings. That retry adjusted handling of archived profiling counters before any model forward.

The two score banks and exact replay source are retained outside Git in `results/left_decoder_context_20261124_v1_swat_modulation_off` on the research server. Score SHA-256 values are `301fd8f2f85c9d034933d3ffa116f9913f6b9df973f8cb05ee8aab58bd7fba6d` (observation off) and `009faceb214df0c488ce6edc1ec61407ce39518c754b9ddbd7e8c69438f9e426` (prototype off). Replay source SHA-256 is `6261b9431ba083ccdc6af38799276b20502c27f6bb434a7cdade13d6f9696adc`.

## Fixed conditioned-training comparison with unmodulated detection

The new method rule was recorded in commit `52bbc48ec26e946aab86218a52f57f9f4b67c4bc` after the SWaT intervention above and before the PSM off readouts. Normal training and checkpoint selection use the original conditioned paths. Detection removes only the direct multiscale modulation on both datasets. The cycle path uses its original memory mixing at $\lambda=0.3$. Scoring uses the original weights, scalers and native score. The two SWaT off banks and both domains' on and native banks come from the archived runs. This gives a fixed test of the rule on PSM and a reuse of the discovery evidence on SWaT.

The shared evaluation axes are PSM test-relative [0, 87744) and SWaT absolute [495000, 944856). LEFT's W192 scores are available at the end of their containing window and are assessed retrospectively. The level baseline uses beta-one SmoothL1 from each standardized observation to zero, averaged over channels. Its scaler is the same LEFT normal-prefix scaler including normal SELECT. Pointwise availability is recorded separately even though evaluation uses the common offline axis. Operation metrics are NA on these axes because independent compatible CAL is unavailable.

### Complete methods

| Dataset | Method | AP | ROC-AUC | VUS-PR | VUS-ROC | Training, scaling and context permissions |
| --- | --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT | 0.576639 | 0.776095 | 0.572611 | 0.785067 | Normal FIT and SELECT; normal-prefix statistics; W192 and window-end availability |
| PSM | Observation-conditioned training, detection off | 0.576335 | 0.755716 | 0.572616 | 0.762379 | Normal FIT and SELECT; normal-prefix statistics; W192 and window-end availability |
| PSM | Prototype-conditioned training, detection off | 0.579869 | 0.754757 | 0.568705 | 0.751577 | Normal FIT and SELECT; normal-prefix statistics; W192 and window-end availability |
| PSM | Pointwise level deviation | 0.438271 | 0.654233 | 0.446020 | 0.654290 | Zero training; same normal-prefix statistics; current point only |
| PSM | CATCH W192 seed0 | 0.445990 | 0.653418 | 0.536671 | 0.703657 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| PSM | CATCH W192 seed1 | 0.430264 | 0.641399 | 0.519273 | 0.690691 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| PSM | CATCH W192 seed2 | 0.433044 | 0.647248 | 0.526447 | 0.698733 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| PSM | PCA residual | 0.534404 | 0.793321 | 0.528991 | 0.766686 | target FIT-only scaler and PCA fitted on at most 50000 FIT points; score only uses x_t |
| PSM | TimeRadar official block0 seed2024 | 0.433265 | 0.670669 | 0.441259 | 0.678365 | Frozen pretrained model; native norm=0, B64, L1; prefix scaler includes SELECT; complete W100 and batch-shared order |
| PSM | TimeRadar common FIT scaler seed2024 | 0.455371 | 0.683823 | 0.458071 | 0.689163 | Frozen pretrained model; FIT-only scaler, B64, norm=0, L1; complete W100 and batch-shared order |
| SWaT | Native LEFT | 0.543866 | 0.820853 | 0.534601 | 0.809255 | Normal FIT and SELECT; normal-prefix statistics; W192 and window-end availability |
| SWaT | Observation-conditioned training, detection off | 0.713724 | 0.838712 | 0.544820 | 0.707896 | Normal FIT and SELECT; normal-prefix statistics; W192 and window-end availability |
| SWaT | Prototype-conditioned training, detection off | 0.720355 | 0.838264 | 0.509151 | 0.691949 | Normal FIT and SELECT; normal-prefix statistics; W192 and window-end availability |
| SWaT | Pointwise level deviation | 0.744800 | 0.831823 | 0.464536 | 0.644277 | Zero training; same normal-prefix statistics; current point only |
| SWaT | TimeRadar official block0 | 0.639299 | 0.821970 | 0.514061 | 0.766831 | Frozen pretrained model; native norm=0, B128, L1; prefix scaler includes SELECT; complete W100 and batch-shared order |
| SWaT | CATCH W192 | NA | NA | NA | NA | Compatible full-axis cache unavailable |
| SWaT | PCA residual | NA | NA | NA | NA | Compatible full-axis cache unavailable |

### Training and detection ablation

| Dataset | Comparison | ΔAP | ΔROC-AUC | ΔVUS-PR | ΔVUS-ROC |
| --- | --- | --- | --- | --- | --- |
| PSM | Observation-conditioned training, detection off − Observation-conditioned training, detection on | -0.000314 | -0.020311 | -0.000458 | -0.022620 |
| PSM | Prototype-conditioned training, detection off − Prototype-conditioned training, detection on | +0.001930 | -0.022233 | -0.006287 | -0.034739 |
| PSM | Observation-conditioned training, detection off − Native LEFT | -0.000304 | -0.020379 | +0.000005 | -0.022688 |
| PSM | Prototype-conditioned training, detection off − Native LEFT | +0.003230 | -0.021338 | -0.003906 | -0.033490 |
| PSM | Prototype-conditioned training, detection off − Observation-conditioned training, detection off | +0.003535 | -0.000959 | -0.003912 | -0.010802 |
| PSM | Observation-conditioned training, detection off − Pointwise level deviation | +0.138064 | +0.101484 | +0.126596 | +0.108088 |
| PSM | Prototype-conditioned training, detection off − Pointwise level deviation | +0.141599 | +0.100524 | +0.122684 | +0.097286 |
| SWaT | Observation-conditioned training, detection off − Observation-conditioned training, detection on | +0.222793 | +0.047273 | +0.087738 | -0.022111 |
| SWaT | Prototype-conditioned training, detection off − Prototype-conditioned training, detection on | +0.580124 | +0.308578 | +0.366469 | +0.190015 |
| SWaT | Observation-conditioned training, detection off − Native LEFT | +0.169857 | +0.017859 | +0.010218 | -0.101359 |
| SWaT | Prototype-conditioned training, detection off − Native LEFT | +0.176489 | +0.017412 | -0.025450 | -0.117306 |
| SWaT | Prototype-conditioned training, detection off − Observation-conditioned training, detection off | +0.006631 | -0.000448 | -0.035669 | -0.015947 |
| SWaT | Observation-conditioned training, detection off − Pointwise level deviation | -0.031077 | +0.006889 | +0.080284 | +0.063620 |
| SWaT | Prototype-conditioned training, detection off − Pointwise level deviation | -0.024446 | +0.006441 | +0.044615 | +0.047673 |

[PSM PR and ROC curves](figures/psm_pr_roc.svg) · [SWaT PR and ROC curves](figures/swat_pr_roc.svg) · [Full precision metrics and data bindings](unmodulated_metrics.json)

On PSM, switching off modulation yielded a small trade-off. Prototype-conditioned training with detection off gained 0.003230 AP over native LEFT, with −0.021338 ROC-AUC, −0.003906 VUS-PR and −0.033490 VUS-ROC. Observation-conditioned training with detection off was essentially tied with native LEFT on AP and VUS-PR and lower on ROC-AUC and VUS-ROC. Both off variants exceeded the level baseline on all four metrics.

The SWaT figures reuse the discovery scores. Both off variants exceed native LEFT on AP and ROC-AUC, and both have lower VUS-ROC. Observation conditioning slightly improves VUS-PR over native LEFT, while prototype conditioning is lower. The simple level baseline achieves the highest AP (0.744800), ahead of observation off (0.713724) and prototype off (0.720355). Both trained off variants exceed level deviation on ROC-AUC and both VUS metrics. The PR curves show why an AP improvement relative to native LEFT alone would overstate the evidence for learning a more useful anomaly ranking.

Prototype-conditioned training leads observation-conditioned training only on AP in both datasets (+0.003535 and +0.006631). ROC-AUC and both VUS measures favor observation conditioning. The data do not establish an independent prototype benefit across metrics. The PSM check did not reproduce the broad SWaT improvement, and the SWaT AP gain over native LEFT is smaller than the gain from a static level score under the same scaler. These results do not justify repeating the complete two-domain recipe with new seeds at this stage. They also leave open other static explanations for the observed ranking differences.

The two new PSM score banks (87,744 points, 25 channel scores each) passed checkpoint, scaler, score and availability checks. Measured A40 readout time was 0.690942 s for the first execution and 0.092070 s for the second, totaling 0.783013 A40-seconds (0.000218 GPU-hours). This includes the cold first forward and excludes loading and compression. CPU evaluation and plotting took 71.346 wall-seconds on the successful run. The initial CPU preflight was retried after differences in float32 channel summation order. The independent SmoothL1 check then matched all elementwise values on the first 4,096 points per task. The level scores retained their original NumPy channel mean. Independent tie-group AP and ROC-AUC recomputation on all 19 plotted rows gave maximum error 6.66e-16. VUS uses the existing LEFT implementation with `windowSize=100` and `thre=250`. The VUS contrasts are differences of these same-axis metrics.

The scored PSM outputs and both level baselines are stored outside Git in `results/left_context_unmodulated_20261124_v1`. PSM observation-off score SHA-256 is `4279535a196ae5677342a38e7f98ffd074a0981f00706631e5e6841c8a048cf2` and prototype-off is `f2787ea3e81cfbb3dc5ce44327118a827ba85d008622f7feb611ffa4be0fe5a6`.

## Seed2026 repetition of observation-assisted LEFT

The repetition fixes two complete methods. Native LEFT retains its original training and readout. Observation-assisted LEFT uses the added multiscale modulation in normal FIT and SELECT, selects by the same normal validation rule, and bypasses only that modulation in detection. The original time–frequency cycle, prototype reads, memory mixing at $\lambda=0.3$, scaler and scoring formula are retained. The two arms share a seed2026 initialization for their common components and matched training randomization within each task. Their CPU RNG fingerprints agree at all 20 common PSM epochs and all 19 common SWaT epochs. The seed2025 runs and source-of-context ablation are archived above. SWaT seed2025 supplied the post-hoc observation that motivated this rule.

The common axes are PSM test-relative [0, 87744) and SWaT absolute [495000, 944856). LEFT W192 scores become available at the last point of the containing window and are evaluated retrospectively. All operation metrics remain NA because there is no compatible independent CAL. External rows retain their own training, scaler and context permissions.

### Complete methods

| Dataset | Method | Seed | AP | ROC-AUC | VUS-PR | VUS-ROC | Permissions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT | 2025 | 0.576639 | 0.776095 | 0.572611 | 0.785067 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| PSM | Observation-assisted LEFT, modulation off at detection | 2025 | 0.576335 | 0.755716 | 0.572616 | 0.762379 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| PSM | Native LEFT | 2026 | 0.570185 | 0.771785 | 0.562230 | 0.778429 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| PSM | Observation-assisted LEFT, modulation off at detection | 2026 | 0.572256 | 0.765521 | 0.564671 | 0.771272 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| PSM | Pointwise level deviation | — | 0.438271 | 0.654233 | 0.446020 | 0.654290 | No parameter training, same normal-prefix scaler, current point only |
| PSM | CATCH W192 seed0 | Archive | 0.445990 | 0.653418 | 0.536671 | 0.703657 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| PSM | CATCH W192 seed1 | Archive | 0.430264 | 0.641399 | 0.519273 | 0.690691 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| PSM | CATCH W192 seed2 | Archive | 0.433044 | 0.647248 | 0.526447 | 0.698733 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| PSM | PCA residual | Archive | 0.534404 | 0.793321 | 0.528991 | 0.766686 | target FIT-only scaler and PCA fitted on at most 50000 FIT points; score only uses x_t |
| PSM | TimeRadar official block0 seed2024 | Archive | 0.433265 | 0.670669 | 0.441259 | 0.678365 | Frozen pretrained weights, native norm=0, B64, L1; prefix scaler includes SELECT; complete W100 and batch-shared order |
| PSM | TimeRadar common FIT scaler seed2024 | Archive | 0.455371 | 0.683823 | 0.458071 | 0.689163 | Frozen pretrained weights, FIT-only scaler, B64, norm=0, L1; complete W100 and batch-shared order |
| SWaT | Native LEFT | 2025 | 0.543866 | 0.820853 | 0.534601 | 0.809255 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| SWaT | Observation-assisted LEFT, modulation off at detection | 2025 | 0.713724 | 0.838712 | 0.544820 | 0.707896 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| SWaT | Native LEFT | 2026 | 0.586210 | 0.814548 | 0.522656 | 0.750588 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| SWaT | Observation-assisted LEFT, modulation off at detection | 2026 | 0.745122 | 0.844678 | 0.573343 | 0.793297 | Normal FIT and SELECT, normal-prefix scaler including SELECT, offline W192 |
| SWaT | Pointwise level deviation | — | 0.744800 | 0.831823 | 0.464536 | 0.644277 | No parameter training, same normal-prefix scaler, current point only |
| SWaT | TimeRadar official block0 | Archive | 0.639299 | 0.821970 | 0.514061 | 0.766831 | Frozen pretrained weights, native norm=0, B128, L1; prefix scaler includes SELECT; complete W100 and batch-shared order |
| SWaT | CATCH W192 | — | NA | NA | NA | NA | Compatible full-axis cache unavailable |
| SWaT | PCA residual | — | NA | NA | NA | NA | Compatible full-axis cache unavailable |

### Paired differences: observation-assisted LEFT minus native LEFT

| Dataset | Seed | ΔAP | ΔROC-AUC | ΔVUS-PR | ΔVUS-ROC |
| --- | --- | --- | --- | --- | --- |
| PSM | 2025 | -0.000304 | -0.020379 | +0.000005 | -0.022688 |
| PSM | 2026 | +0.002071 | -0.006264 | +0.002441 | -0.007157 |
| SWaT | 2025 | +0.169857 | +0.017859 | +0.010218 | -0.101359 |
| SWaT | 2026 | +0.158912 | +0.030130 | +0.050687 | +0.042708 |

On SWaT the candidate improves AP, ROC-AUC and VUS-PR relative to native LEFT in both seeds. VUS-ROC falls by 0.101359 in seed2025 and rises by 0.042708 in seed2026. The fixed pointwise level score reaches AP 0.744800. It exceeds the seed2025 candidate by 0.031077 and trails seed2026 by 0.000322. The trained candidate exceeds it on ROC-AUC and both VUS measures in both seeds. The AP gain relative to native LEFT is repeatable here, but these results leave the source of the gain beyond simple level deviation unresolved.

On PSM, the AP difference changes from −0.000304 to +0.002071 and VUS-PR changes are small. ROC-AUC and VUS-ROC fall in both seeds. The repeated pattern is a ranking trade-off with near-tied AP. Both candidate seeds exceed the fixed level baseline on all four measures. These two seeds provide descriptive evidence about training variation on the same development tasks. They do not establish statistical significance or independent-domain confirmation.

[PSM PR and ROC](repeat_2026/psm_pr_roc.svg) · [SWaT PR and ROC](repeat_2026/swat_pr_roc.svg) · [Full precision metrics and score identities](repeat_2026/METRICS.json)

### Training and evaluation cost

| Dataset | Seed2026 method | Epochs run | Selected epoch | Updates run | Training (A40 s) | Readout (A40 s) |
| --- | --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT | 20 | 19 | 16,540 | 974.416 | 0.800 |
| PSM | Observation-assisted LEFT, modulation off at detection | 20 | 19 | 16,540 | 1000.938 | 0.764 |
| SWaT | Native LEFT | 19 | 16 | 58,767 | 3838.619 | 3.935 |
| SWaT | Observation-assisted LEFT, modulation off at detection | 19 | 16 | 58,767 | 3986.365 | 3.998 |

The four new trainings used 2.722316 A40 GPU-hours. The four full readouts used 9.496 A40-seconds. Combined measured cost was 2.724954 A40 GPU-hours. CPU metrics, curves and numerical recomputation took 49.125 wall-seconds. Costs exclude process startup and reused runs.

An independent tie-group AP and trapezoidal ROC-AUC calculation on 17 new and reused score rows matched the reported values to within 6.66e-16. VUS uses the fixed LEFT evaluator (windowSize 100, thre 250). Score banks, checkpoints and scalers for all four new runs match their SHA-256 receipts. The new per-point and per-channel banks, input indices, availability indices, scalers and checkpoints are archived outside Git under `results/left_context_unmodulated_repeat_20261125_v1`.
