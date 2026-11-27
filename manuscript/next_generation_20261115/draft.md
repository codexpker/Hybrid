# Source transfer and mixed-unlabeled adaptation for multivariate time-series anomaly detection

## Abstract

A detector trained on normal source records can encounter a mixture of normal shifts and anomalies during unlabeled target adaptation. We evaluate how a complete time–frequency detector responds to this mixture using native LEFT on PSM, SWaT and all 28 SMD machines. The protocol separates normal-source training, target adaptation, calibration and evaluation, and distinguishes transferred initialization, ordinary target updating, source replay and reliability weighting. Under natural mixtures, window-weighted replay changes AP relative to ordinary replay by +0.002909, +0.001063 and −0.001477 on PSM, SWaT and SMD28, respectively. In a fixed exposure comparison, replacing matched target slots with fully anomalous instead of fully normal windows reduces PSM AP by 0.020624 for replay and 0.004491 for weighted replay. The corresponding difference in responses is +0.016132 on PSM and +0.002752 on SWaT. Window weighting also introduces operational trade-offs in recall, event matching and sustained coverage. The results establish task-dependent effects under the specified update budgets and exposure plans. [Natural and controlled results](../../docs/experiments/left_evidence_freeze_20261128/README.md)

## 1. Introduction

Multivariate anomaly detectors often learn by reconstructing observations judged likely to be normal. When a detector is transferred to a new domain, an adaptation prefix can contain both changing normal behavior and the anomalous behavior the detector should retain the ability to identify. Reconstruction alone gives a learner an incentive to explain both. A useful evaluation therefore asks what information is gained through updating, what detection ability is lost, and how those effects change when the training exposure is controlled.

We use LEFT as a complete detector with time, frequency and multi-scale reconstruction. Keeping the detector and readout fixed lets us compare update rules under the same architecture and input permissions. Ordinary target updating spends its budget on mixed windows. Replay replaces half that exposure with normal source windows. Window weighting redistributes the target half's loss contribution according to a frozen source score. Local weighting redistributes the same coefficients across time positions. These comparisons motivate three questions.

1. When does training on normal public sources help frozen transfer, normal target training or subsequent mixed-unlabeled learning?
2. At a fixed update budget, how do normal-source replay and reliability weighting change point ranking, event discovery and coverage?
3. Does a fixed additional abnormal exposure change these comparisons after matching the number and positions of replaced slots, and does correct weight-to-window correspondence matter?

The contributions are a reproducible three-domain training and evaluation framework, entity-level evidence on the transfer and adaptation trade-offs, and paired evidence on the response to the fixed additional exposure. The study preserves normal target training, frozen use and mixed adaptation as distinct permissions. The strongest conclusions concern the measured protocols and source–target folds. [Frozen contribution and evidence index](../../docs/experiments/left_evidence_freeze_20261128/README.md)

## 2. Related work

**Public detectors and transfer.** LEFT learns tri-view representations with cross-view consistency. TimeRadar, DADA and TimeRCD provide pretrained alternatives with different score definitions and contextual permissions. CATCH, PCA and pointwise deviation provide target-normal or simpler references. DACAD uses source supervision, anomaly injection and contrastive objectives for domain adaptation. Here the project source records are normal and the native detector runs with dimension-specific checkpoints. [LEFT](https://arxiv.org/abs/2602.08638) · [DACAD](https://arxiv.org/abs/2404.11269) · [External benchmark and permissions](../../docs/experiments/left_smd_reliability_20261126/SMD_RESULTS.md)

**Contaminated and changing data.** RiAD dynamically attenuates samples with uncertainty and outlierness using one-class statistics and adaptive reconstruction. CLEANet combines an encoder-neighborhood sigmoid weight with clustering-guided contrastive learning. Li et al. filter using early loss level and oscillation, and Latent Outlier Exposure alternates latent assignments and complementary losses. These methods establish the relevant precedents for robust sample weighting or selection. [RiAD](https://doi.org/10.1016/j.ins.2024.120222) · [CLEANet preprint](https://arxiv.org/html/2510.22619v1#S3.SS3) · [Li et al.](https://arxiv.org/pdf/2208.01841) · [Latent Outlier Exposure](https://proceedings.mlr.press/v162/qiu22b.html)

**Adaptation.** M2N2 uses model-predicted normal points to mask online updates. CANDI uses scores and reference similarity in sample curation and trains structured adaptation modules. Kumagai et al. adapt a robust reconstructor on unlabeled support and use labeled query ranking for the outer objective. The tested LEFT weighting instead uses one fixed scalar per target window, derived from frozen native point-score ranks with the target mass normalized within each batch. Mixing retained normal observations with new observations is standard experience replay. These ingredients define the tested protocol and its controls. [M2N2](https://ojs.aaai.org/index.php/AAAI/article/download/29210/30283) · [CANDI](https://doi.org/10.1609/aaai.v40i17.38524) · [Kumagai et al.](https://proceedings.mlr.press/v206/kumagai23a.html) · [Experience replay](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)

## 3. Problem, detector and adaptation

### Normal-source initialization and information permissions

Let $X_j\in\mathbb R^{192\times C}$ be a multivariate window within one legal record and role. Each source model sees numeric normal windows. PSM uses 25 variables and SWaT uses 31, with TEP FaultFree Training and SMD normal records as sources. The 38-channel SMD target fold uses TEP and HAI 21.03, excluding SMD from both source fitting and selection. The recorded source FIT positions use separate SELECT windows. TEP FIT uses runs 1–20 and SELECT runs 61–65. SMD source record splits are within-machine temporal splits. HAI train1/train3 are FIT and train2 is SELECT. Channel draws are without replacement within a window, sampled in a fixed order for that window. They do not assume shared sensor semantics across datasets. The three source jobs use 4,000 Adam updates at $10^{-4}$ with 16 windows from each source per step. Normal source validation selects step 4,000 for all three models. [Base source training and sample accounting](../../docs/experiments/left_multisource_20261126/RESULTS.md) · [HAI and 38-channel identity](../../docs/experiments/left_smd_reliability_20261126/SMD_RESULTS.md)

Frozen use Z has no target parameter updates. Normal continuation N uses target normal FIT and selection under the native target-normal protocol. Native LEFT fitted from scratch shares that target-normal protocol with N. Ordinary U starts from the same source checkpoint as replay. Both consume target values only from U-FIT. The held-out target QUERY is used only for evaluation, with no test-label selection of weights, thresholds or endpoints. PSM and SWaT use archived target normal-prefix statistics including normal SELECT. The SMD public axis likewise uses the normal prefix, whereas its independent U uses per-machine FIT-only statistics and a reserved nominal-normal CAL. We keep the two SMD score panels separate. [Recorded roles and statistics](../../docs/experiments/left_smd_reliability_20261126/SMD_RESULTS.md) · [Two-domain role contract](../../docs/experiments/left_multisource_20261126/RESULTS.md)

### Native fitting and readout

The implementation retains native LEFT's tri-view processing, prototype memory and curriculum. Per-window loss reductions include the time-to-frequency and frequency-to-time cycle losses, the multi-scale reconstruction loss and the full-rate consistency term. Writing $\mathcal L_\mathrm{cyc}$ for the sum of the two SmoothL1 cycle terms, $\mathcal L_\mathrm{ms}$ for native scale-weighted SmoothL1, and $\mathcal L_\mathrm{cons}$ for the consistency of the finest multi-scale output with the stopped-gradient inverse-frequency reconstruction, the native objective is
$$
\mathcal L_{\rm LEFT}=\alpha_{\rm cyc}\mathcal L_{\rm cyc}
+\alpha_{\rm ms}\mathcal L_{\rm ms}+\alpha_{\rm cons}\mathcal L_{\rm cons}.
$$
The original evaluation combines native reconstruction and association terms with the multi-scale score using the saved configuration. We average the native score over variables. The W192 nonoverlapping block is read in full, and all scores inside it become available at its end. This is retrospective detection. [Native model binding and unchanged score](../../docs/experiments/left_multisource_20261126/RESULTS.md) · [Scoring entry](../../src/hybrid_tsad/left_multisource_eval.py)

### Ordinary, replay and weighted adaptation

Each U job starts fresh Adam at $10^{-4}$ and performs 2,000 updates at batch size 32. Ordinary U uses two target half-batches. Replay uses the same first 16 target slots and eight normal windows from each source in the other half. Random-init U and source-only continuation identify two alternative sources of change. Gains from changing this batch composition at the fixed budget are gains of a substitution policy. [Matched replay code](../../src/hybrid_tsad/left_multisource.py)

Window weighting measures the original channel-mean point scores from the frozen parent on every target presentation in an adaptation plan. With one-based midranks across the plan's $M$ point presentations, define $r_{bit}=(\mathrm{midrank}(s_{bit})-0.5)/M$ and $u_{bit}=1.5-r_{bit}$. At step $b$, the weight of target window $i$ is
$$
w_{bi}=\frac{W^{-1}\sum_t u_{bit}}{(16W)^{-1}\sum_{i,t}u_{bit}},\qquad W=192.
$$
Source weights are one. With $\ell_\theta$ the native per-window loss contribution, the weighted data objective is $\big(\sum_i w_{bi}\ell_\theta(T_{bi})+\sum_j\ell_\theta(S_{bj})\big)/32$, with original scale reductions and native terms independent of data weighting retained. All windows still enter the native forward, including prototype admission. The coefficient weights a reduction and does not directly measure the gradient or EMA contribution of that window. A local control uses the time weights on the native grid and the same window mean in frequency and coarse terms. A permutation control preserves every target batch's weight multiset and total mass while changing assignments. The rank order is computed separately for each task and exposure condition. Deployment uses the same unweighted native score for all trained arms. [Weight code](../../src/hybrid_tsad/left_reliability.py) · [Matched SMD adaptation](../../src/hybrid_tsad/left_smd_targets.py)

## 4. Evaluation protocols and complete public results

We report AP, ROC-AUC, VUS-PR and VUS-ROC. AP is unadjusted average precision. VUS uses the archived LEFT implementation with window size 100 and 250 thresholds. The local benchmark separates complete legal grids from the external overlap and the independent U suffix. Matching timestamps do not equalize context, standardization or training permissions. The primary full axes are PSM test-relative $[0,87744)$, SWaT absolute $[495000,944856)$ and complete W192 blocks within each SMD test record. The external intersection starts at 99 and ends at the common W100/W192 grid limit per record. SMD results below first compute each metric within each of the 28 machines and then take an equal-weight mean. CATCH is a three-seed mean ± sample SD. The studied LEFT methods use seed2025. [Complete value and identity tables](../../docs/experiments/left_evidence_freeze_20261128/main_metrics.csv) · [Metric, axis and comparator permissions](../../docs/experiments/left_smd_reliability_20261126/SMD_RESULTS.md)

### Complete public grids

| Target | Method | Target fitting | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT, target-normal from scratch | target normal | 0.57664 | 0.77610 | 0.57261 | 0.78507 |
| PSM | Frozen multisource LEFT (Z) | none | 0.49639 | 0.70108 | 0.50096 | 0.70649 |
| PSM | Multisource target-normal training (N) | target normal | 0.55438 | 0.75671 | 0.55201 | 0.76627 |
| PSM | CATCH | target normal | 0.43643 ± 0.00839 | 0.64736 ± 0.00601 | 0.52746 ± 0.00874 | 0.69769 ± 0.00655 |
| PSM | PCA residual | target normal | 0.53440 | 0.79332 | 0.52899 | 0.76669 |
| PSM | Pointwise level deviation | none | 0.43827 | 0.65423 | 0.44602 | 0.65429 |
| PSM | Pretrained TimeRadar (native block grid) | none | 0.43327 | 0.67067 | 0.44126 | 0.67837 |
| SWaT | Native LEFT, target-normal from scratch | target normal | 0.54387 | 0.82085 | 0.53460 | 0.80926 |
| SWaT | Frozen multisource LEFT (Z) | none | 0.72392 | 0.81151 | 0.49756 | 0.66106 |
| SWaT | Multisource target-normal training (N) | target normal | 0.54250 | 0.85558 | 0.53098 | 0.84311 |
| SWaT | Pointwise level deviation | none | 0.74480 | 0.83182 | 0.46454 | 0.64428 |
| SWaT | Pretrained TimeRadar (native block grid) | none | 0.63930 | 0.82197 | 0.51406 | 0.76683 |
| SMD28 | Native LEFT, target-normal from scratch | target normal | 0.49751 | 0.87868 | 0.54151 | 0.91862 |
| SMD28 | Frozen multisource LEFT (Z) | none | 0.49690 | 0.86792 | 0.54191 | 0.91292 |
| SMD28 | Multisource target-normal training (N) | target normal | 0.51765 | 0.88301 | 0.56463 | 0.92087 |
| SMD28 | Pointwise level deviation | none | 0.42651 | 0.80337 | 0.42547 | 0.86664 |

### External common ticks

SWaT has no row on this external intersection in the fixed archive.

| Target | Method | Target fitting | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT, target-normal from scratch | target normal | 0.57689 | 0.77618 | 0.57412 | 0.78578 |
| PSM | Frozen multisource LEFT (Z) | none | 0.49671 | 0.70127 | 0.50229 | 0.70738 |
| PSM | Multisource target-normal training (N) | target normal | 0.55457 | 0.75672 | 0.55329 | 0.76690 |
| PSM | CATCH | target normal | 0.43684 ± 0.00844 | 0.64765 ± 0.00602 | 0.52717 ± 0.00869 | 0.69698 ± 0.00662 |
| PSM | PCA residual | target normal | 0.53444 | 0.79312 | 0.52873 | 0.76699 |
| PSM | Pretrained DADA | none | 0.39565 | 0.60496 | 0.43945 | 0.65197 |
| PSM | Pretrained TimeRCD (FIT-only) | none | 0.30193 | 0.47619 | 0.34850 | 0.53197 |
| PSM | Pretrained TimeRadar V (causal endpoint) | none | 0.53528 | 0.73098 | 0.48359 | 0.66478 |
| SMD28 | Native LEFT, target-normal from scratch | target normal | 0.49808 | 0.87855 | 0.54184 | 0.91853 |
| SMD28 | Frozen multisource LEFT (Z) | none | 0.49748 | 0.86779 | 0.54215 | 0.91282 |
| SMD28 | Multisource target-normal training (N) | target normal | 0.51818 | 0.88294 | 0.56510 | 0.92082 |
| SMD28 | Pretrained DADA | none | 0.41461 | 0.85136 | 0.47465 | 0.89970 |
| SMD28 | Pretrained TimeRCD (FIT-only) | none | 0.47295 | 0.85942 | 0.51514 | 0.90975 |
| SMD28 | Pretrained TimeRadar V (causal endpoint) | none | 0.44480 | 0.81321 | 0.46719 | 0.86996 |

Source initialization and subsequent fitting have different effects. On the complete PSM axis, N raises AP from 0.49639 for Z to 0.55438, while the target-normal model from scratch reaches 0.57664. On SWaT, frozen transfer reaches 0.72392 AP, while target normal continuation reaches 0.54250. At matched normal target training, changing the initialization gives small decreases in both PR metrics and improves the ROC metrics. On SMD28, N improves macro AP by 0.02014 over the matched from-scratch model, with 21 per-machine gains and seven losses. Source composition changes with the held-out target and is part of the fold definition. [Two-domain normal transfer](../../docs/experiments/left_multisource_20261126/RESULTS.md) · [SMD per-machine distributions](../../docs/experiments/left_smd_reliability_20261126/smd_paired_distribution.csv)

## 5. Natural mixed-unlabeled target adaptation

PSM and SWaT use the same fixed QUERY suffix across their mixed arms, respectively $[22656,87744)$ and $[608535,944856)$. SMD uses entity-specific suffixes with an independent CAL. A has its native target-normal metrics on the public grid. Its suffix metrics and SMD local-weighting results remain NA in the evaluated archive. Random-init target U is a separate control. All rows inside a domain block share the same scored points. [Exact source and score identities](../../docs/experiments/left_evidence_freeze_20261128/source_manifest.csv) · [All SMD entities](../../docs/experiments/left_evidence_freeze_20261128/smd_entities.csv)

| Target | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT, target-normal from scratch | NA | NA | NA | NA |
| PSM | Frozen multisource LEFT (Z) | 0.504296 | 0.720009 | 0.515853 | 0.731294 |
| PSM | Multisource ordinary target U | 0.576967 | 0.757871 | 0.585024 | 0.771524 |
| PSM | Multisource source replay U | 0.591946 | 0.755556 | 0.592012 | 0.771655 |
| PSM | Window-weighted source replay U | 0.594855 | 0.762450 | 0.594007 | 0.777015 |
| PSM | Local-weighted source replay U | 0.597741 | 0.762939 | 0.597100 | 0.778199 |
| PSM | Random-initialized target U | 0.580865 | 0.739244 | 0.574255 | 0.750188 |
| PSM | Normal source continuation | 0.524133 | 0.729912 | 0.534574 | 0.743078 |
| SWaT | Native LEFT, target-normal from scratch | NA | NA | NA | NA |
| SWaT | Frozen multisource LEFT (Z) | 0.791359 | 0.842321 | 0.630690 | 0.734086 |
| SWaT | Multisource ordinary target U | 0.779989 | 0.841665 | 0.633183 | 0.744684 |
| SWaT | Multisource source replay U | 0.786150 | 0.841968 | 0.633839 | 0.740272 |
| SWaT | Window-weighted source replay U | 0.787213 | 0.842038 | 0.631845 | 0.738376 |
| SWaT | Local-weighted source replay U | 0.787218 | 0.842086 | 0.631912 | 0.738421 |
| SWaT | Random-initialized target U | 0.788190 | 0.842215 | 0.633795 | 0.741378 |
| SWaT | Normal source continuation | 0.789719 | 0.842889 | 0.629091 | 0.736839 |
| SMD28 | Native LEFT, target-normal from scratch | NA | NA | NA | NA |
| SMD28 | Frozen multisource LEFT (Z) | 0.540538 | 0.871383 | 0.590123 | 0.916426 |
| SMD28 | Multisource ordinary target U | 0.555152 | 0.888150 | 0.608026 | 0.926151 |
| SMD28 | Multisource source replay U | 0.559843 | 0.895781 | 0.613119 | 0.929473 |
| SMD28 | Window-weighted source replay U | 0.558367 | 0.895324 | 0.611582 | 0.929960 |
| SMD28 | Local-weighted source replay U | NA | NA | NA | NA |
| SMD28 | Random-initialized target U | 0.545181 | 0.888176 | 0.597083 | 0.924072 |
| SMD28 | Normal source continuation | 0.540520 | 0.875003 | 0.588346 | 0.916472 |

Replay improves natural suffix AP over ordinary U by 0.014979 on PSM, 0.006160 on SWaT and 0.004692 in the SMD28 macro. Relative to replay, window weighting improves AP by 0.002909 on PSM and 0.001063 on SWaT, while the latter domain loses both VUS metrics. SWaT's two weighted arms remain below Z in AP. Window weighting loses 0.001477 macro AP on SMD28, with 12 machine gains, 16 losses and a paired median of −0.000344. The small VUS-ROC gain coexists with lower AP, ROC-AUC and VUS-PR in that domain. Local weighting improves PSM natural AP by 0.002886 over window weighting. The SWaT difference is approximately 0.000005. [PSM/SWaT mixed contrasts](../../docs/experiments/left_local_reliability_20261126/contrasts.csv) · [Per-machine SMD contrasts](../../docs/experiments/left_smd_reliability_20261126/smd_paired_distribution.csv)

## 6. Controlled abnormal exposure and operating cost

A retains natural U-FIT. B and C replace the same 3,200 of the 32,000 shared target slots with windows entirely labeled anomalous and normal, respectively, from U-FIT. These replacements account for 5% of all 64,000 presentations. The background retains its natural mixture. Public labels construct eligible donor windows offline. Learners consume only their numeric observations. B/C match slot count and location, while donor operating conditions can differ. Repeated donors do not create independent events. [Eligibility and exposure ledger](../../docs/experiments/left_memory_exposure_20261126/RESULTS.md)

With $\Delta_m=M(m,B)-M(m,C)$, window-weighting interaction against replay is $\Delta_\mathrm{window}-\Delta_\mathrm{replay}$. Exposure conditions refit their own frozen-score ranks over their complete target plans, so the interaction also includes changes in weights on common inputs. We report the absolute endpoint along with the interaction.

| Target | Method | A AP | B AP | C AP |
| --- | --- | --- | --- | --- |
| PSM | Multisource ordinary target U | 0.576967 | 0.526465 | 0.577491 |
| PSM | Multisource source replay U | 0.591946 | 0.575372 | 0.595996 |
| PSM | Window-weighted source replay U | 0.594855 | 0.595739 | 0.600230 |
| PSM | Local-weighted source replay U | 0.597741 | 0.594257 | 0.602739 |
| PSM | Permuted-window-weight source replay U | 0.589377 | 0.577787 | 0.597734 |
| PSM | Source-only EMA write restriction with replay | 0.591950 | 0.575299 | 0.595991 |
| SWaT | Multisource ordinary target U | 0.779989 | 0.778415 | 0.779960 |
| SWaT | Multisource source replay U | 0.786150 | 0.780606 | 0.786090 |
| SWaT | Window-weighted source replay U | 0.787213 | 0.784462 | 0.787194 |
| SWaT | Local-weighted source replay U | 0.787218 | 0.784685 | 0.787258 |
| SWaT | Permuted-window-weight source replay U | 0.786315 | 0.781040 | 0.786972 |
| SWaT | Source-only EMA write restriction with replay | 0.786155 | 0.780580 | 0.786086 |

| Target | Contrast | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| PSM | Window minus replay in B | +0.020367 | +0.024533 | +0.018039 | +0.022999 |
| PSM | B−C interaction | +0.016132 | +0.016224 | +0.015093 | +0.016351 |
| SWaT | Window minus replay in B | +0.003856 | +0.000770 | +0.001063 | -0.001231 |
| SWaT | B−C interaction | +0.002752 | +0.000558 | +0.001819 | +0.000357 |

On PSM, weighted replay raises B AP by 0.020367 relative to replay. The B−C AP loss shrinks from 0.020624 to 0.004491. The AP interaction is +0.016132. On SWaT, the B gain and AP interaction are +0.003856 and +0.002752, while B VUS-ROC falls by 0.001231. The result supports a conditional difference in AP response for this intervention. It leaves a broader robustness claim open. [All four absolute metrics](../../docs/experiments/left_evidence_freeze_20261128/abc_metrics.csv) · [Within-method and interaction contrasts](../../docs/experiments/left_evidence_freeze_20261128/abc_contrasts.csv)

Operating metrics use each model's own fixed higher-q99 CAL threshold and strict exceedance. The three-consecutive-point rule starts an alarm at the third point, without backfill. Event matching uses chronological first-unmatched overlap, with at most one match per predicted and reference event. Any-hit is counted separately. Sustained coverage is measured after the first 512 points in qualifying events and retained both by equal event weight and by pooled points. All 512 PSM CAL points and 38 of 512 SWaT CAL points carry attack labels. These thresholds provide retrospective operating points. Existing SMD CAL is nominal normal and independent of adaptation and its FIT statistics. [Full point, event and coverage metrics](../../docs/experiments/left_local_reliability_20261126/operations.csv) · [SMD operating panel](../../docs/experiments/left_smd_reliability_20261126/SMD_RESULTS.md)

In PSM B, window weighting reduces confirmed FPR from 0.1525 to 0.1341 and point recall from 0.4932 to 0.4694. The number of one-to-one matches drops from 24 to 21 of 48 and any-hit counts drop from 24 to 22. Long-event macro coverage falls from 0.7207 to 0.6833. In SWaT B, weighting increases FPR from 0.5387 to 0.6060 and long macro coverage from 0.2854 to 0.5677, while matched events decline from 14 to 12 of 22. In SMD28's natural suffix, both replay methods match 137 of 233 events after confirmation. Weighting reduces any-hit counts from 154 to 153 and increases false alarm fragments from 1,981 to 1,999, with small changes in coverage. [PSM/SWaT event records](../../docs/experiments/left_local_reliability_20261126/events.csv) · [SMD event totals](../../docs/experiments/left_smd_reliability_20261126/smd_event_totals.csv)

## 7. Controls and archived development results

**Weight correspondence and granularity.** In PSM B, correct weight assignments improve AP by 0.017951 over a within-batch permutation that preserves the coefficient multiset and target mass. On SWaT the corresponding difference is 0.003422 and VUS-ROC declines. The local allocation moves weight toward the normal points of partially anomalous windows, but its PSM AP gain over window weighting in A becomes a loss of 0.001481 in B. The permutation isolates correspondence from the multiset of weights within this plan. It does not compare dynamic and frozen policies. [Permutation contrast](../../docs/experiments/left_smd_reliability_20261126/permutation_contrasts.csv) · [Local allocation, gradients and complete results](../../docs/experiments/left_local_reliability_20261126/RESULTS.md)

**Prototype and optimization controls.** The explicit prototype-write restriction makes little difference to ranking. The traced study records zero eligible explicit EMA writes, while prototypes still move through Adam. The study therefore observes no harmful native explicit write for the restriction to prevent. Separate domain constraints suppress first-order conflicts but change AP over replay by only +0.000319 on PSM and −0.000221 on SWaT. Both controls are kept as boundary evidence. [EMA traces and controlled outcomes](../../docs/experiments/left_memory_exposure_20261126/RESULTS.md) · [Source-constraint results](../../docs/experiments/left_source_constraints_20261126/RESULTS.md)

**Detection-oriented source supervision.** Static proxy ranking (M1) did not meet the matched full-target U gate. U AP differences against matched normal continuation were −0.003363 on PSM and +0.000282 on SWaT. Some ROC or frozen Z VUS terms improved. The M2 proposal was stopped at the design review, so there is no trained M2 outcome. [M1 matched evaluation](../../docs/experiments/left_proxy_rank_20261127/RESULTS.md) · [M2 design review](../../docs/experiments/left_detection_oriented_source_20261127/REVIEW.md)

**Bounded input repair.** The archived transformation used $Y=X-\rho\,\operatorname{sign}(r)(|r|-\tau_q)_+$ with $r=X-\widehat X_0$ and a common corrected student input across all views. Sixteen matched source-proxy endpoints completed the fixed eight-arm study. R0 averaged 0.552964 proxy AP, the best finite repair 0.553008 and R7 0.553022. The preregistered source rule retained R0. The repair altered held-out clean coordinates and seldom activated at actual shift edits. The teacher diagnostic did not support near-complete copying of the source edits as an explanation. These data constrain the proxy-based motivation and contain no real target repair outcome. Proxy R0 and R7 checkpoints have their own source plan and are distinct from the true mixed-target methods reported above. [Original formula and protocol](../../docs/experiments/left_input_correction_20261127/PROTOCOL.md) · [Full archive](../../docs/experiments/left_input_correction_20261127/RESULTS.md)

## 8. Limitations, compute and reproducibility

The study's central matched runs use one project training seed. SMD28 machine differences measure variation over fixed evaluation entities; they do not measure training-seed uncertainty. Using new paired seeds and support draws could test whether the observed effects persist, especially around the PSM abnormal-exposure contrast. A direct frozen-versus-dynamic ablation would be needed to attribute benefits to freezing. None of the reported mechanisms constitutes a new detector with established transferability. All target datasets have prior development exposure, A/B/C applies one specific replacement rate and donor policy, source compositions vary by fold, and retrospective W192 scores carry window-end delay. The controlled tests cannot establish causal or physical fault relations. [Frozen scope and missing evidence](../../docs/experiments/left_evidence_freeze_20261128/README.md)

In the base PSM/SWaT archive, source fit, normal transfer, mixed fits and new reads used 1.42 A40 job-hours. The archived SMD expansion plus permutation work used 3.60 A40 hours, with failed setup excluded. The weighting development has frozen-scoring, training and readout costs reported separately, for example 6.63 seconds for the PSM exposure-rank preparation and 73.21 seconds for its A window-weighted update. These are per-study scopes. Reused models and banks are counted once in their own archives. This evidence assembly uses CPU-only data consolidation. [Base timing](../../docs/experiments/left_multisource_20261126/training.csv) · [Weighting costs](../../docs/experiments/left_local_reliability_20261126/costs.csv) · [SMD timing and accounting](../../docs/experiments/left_smd_reliability_20261126/SMD_RESULTS.md)

The versioned evidence tables record the original metric row, permission, seed and available checkpoint, score-bank, plan and weight hashes. A rebuild script reads the archived metric rows, verifies the SMD joins and recomputes interactions from the existing precision. Raw scores and original checkpoints remain in their recorded workspaces. No novel performance values are inferred from a missing archive. [Rebuild script](../../scripts/build_left_evidence_freeze.py) · [Identity manifest](../../docs/experiments/left_evidence_freeze_20261128/source_manifest.csv) · [Bound archive inputs](../../docs/experiments/left_evidence_freeze_20261128/inputs.json)

## 9. Conclusion

Cross-domain LEFT initialization and plain replay change detector behavior in ways that depend on the task and how detection is evaluated. Window weighting attenuates the AP penalty of additional abnormal exposure in the fixed PSM and SWaT comparisons. Its smaller natural gains, SMD28 losses and operating trade-offs delimit the claim. The evidence supports an empirical study of mixed adaptation and controlled exposure on the complete detector. The measured reconstruction, ranking and event outcomes remain separate when assessing potential methodological contributions.

### Historical material

The previous draft preserves the five-domain public panels, literature-reported CrossAD and ModernTCN, and the archived TimesFM conditioning, contiguous holdout, fixed-lag and other studies. They retain their original permissions and claims. [Previous draft and full historical tables](https://github.com/codexpker/Hybrid/blob/a1ada3751e8840212b75400e3291422a302a9ed0/manuscript/next_generation_20261115/draft.md)
