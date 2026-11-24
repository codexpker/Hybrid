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
