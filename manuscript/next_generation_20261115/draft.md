---
title: "Hybrid: Time–Frequency and Temporal Normality Modeling for Industrial Time-Series Anomaly Detection"
author: []
date: "Working draft, updated 18 November 2026"
lang: en
bibliography: references.bib
---

> **Research status, 18 November 2026.** Hybrid remains the intended proposed method for a methods paper. The shallow TimeRadar adaptation branch and density candidate retained below are historical, stopped designs. New completed evidence concerns native LEFT with source replay on SWaT and PSM; replay is an established mechanism, not a Hybrid invention. The current training hypothesis adds source-output preservation to that replay control, with pooled-output (P) and per-output (V) constraints. All four fixed-seed runs and bounded key-result review are complete. V has not established an independent detection increment; P retains a SWaT ranking positive with false-alarm cost and a PSM ranking negative. The new sections at the end define this hypothesis and the completed replay evidence; earlier architecture-specific sections remain historical records.

# Introduction

Industrial monitoring requires more than recognizing abrupt, high-amplitude departures from a nominal range. A weak deviation can be significant when it violates a normal response relationship, whereas a large change can be acceptable when several variables respond coherently to an operating transition. An anomaly may also persist after its initial transition: an effective monitor should discover the event and retain useful evidence during its abnormal steady state. These requirements are particularly demanding for multivariate streams in which responses are delayed, operating conditions change, and target-domain training records are unlabeled rather than independently certified as healthy.

Time–frequency representations offer a useful way to describe both local variation and temporal organization. TimeRadar learns a fractional time–frequency representation and combines reconstruction with contextual deviation learning [@timeradar2026]. LEFT learns interacting temporal, spectral, and multiscale views, followed by joint representation learning and reconstruction [@left2026]. These methods motivate a richer representation than a single instantaneous vector. However, a representation with good reconstruction or prediction accuracy does not automatically provide a useful anomaly ranking. A flexible reconstruction model may reconstruct a sustained abnormal state, and a predictor may improve normal accuracy without increasing the separation between abnormal responses and difficult normal transitions. These are empirical risks, rather than universal properties of either modeling family.

The central research question remains whether time-frequency and temporal information can distinguish weak and persistent departures from legitimate normal-state variation. The historical density candidate and the subsequently evaluated shallow TimeRadar adaptation branch have not established a generally effective solution. Hybrid remains the proposed-method identity for this research; no evaluated baseline, isolated favorable ablation or stopped implementation is promoted to the final method. The completed evidence below constrains future method decisions without itself supplying a new architecture.

Several boundaries matter. First, an exact density is not a semantic definition of normality: a high-likelihood input can still be undesirable, and a legitimate new operating state can have low likelihood. Second, a causal endpoint detector may use the current observation after it arrives, but it must not use later observations. This is different from predicting the current observation from strictly earlier inputs. Third, statistical cross-variable dependence is not evidence of physical causation or a verified directed lag mechanism. Finally, robustness to unlabeled training contamination requires a separate experiment after the detector has demonstrated useful baseline discrimination. Performance that remains unchanged near a detection floor does not establish robustness.

We evaluate these questions using fixed public-data settings with complementary purposes. PSM provides a multivariate development comparison, SWaT supports event and long-duration analyses, and a run-separated Tennessee Eastman Process (TEP) task provides a controlled industrial fault setting with a predefined conditional-response subset. All three have already influenced this research program and are explicitly treated as development or mechanism evidence. They cannot establish a new blind-confirmation claim. Strong simple controls, including historical means and history/conditional Ridge models where applicable, remain necessary alongside LEFT and TimeRadar.

The intended methodological contribution is a testable learning–criterion alignment for time–frequency and temporal normality modeling, not the invention of affine coupling, wavelets, or likelihood estimation. The experimental contribution must demonstrate a useful detection effect attributable to that design at an acceptable false-alarm cost. The present draft does not assert that either contribution has been established. The completed comparison and ablations will determine whether the candidate merits independent confirmation and subsequent multistage adaptation studies.

# Related Work

## Time–frequency and multiview anomaly representations

TimeRadar uses a learnable fractional time–frequency representation within its foundation-model anomaly-detection framework [@timeradar2026]. Its fractional transform spans different domain rotations; it should not be described as a frequency-only encoder. LEFT combines temporal, spectral, and multiscale tokens, selective cross-view interaction, and an analysis–synthesis reconstruction chain [@left2026]. CATCH instead examines frequency patching with channel-aware modeling [@catch2025]. These works show different ways of exploiting temporal and spectral structure. Hybrid does not claim that simply adding a transform or combining view-specific scores is new. Its current question concerns a joint normality criterion whose likelihood includes the transformation of the full observed window. The time-only ablation is required to determine whether the wavelet-domain transformations provide information beyond the temporal architecture and parameter count.

## Temporal dependencies and relationship anomalies

Forecasting and conditional-response models can expose departures that are not apparent from individual variables. Long-context forecasting also motivates temporal encoders, but a forecasting foundation model is not automatically an anomaly detector; TimesFM, for example, is presented as a forecasting model [@timesfm2024]. In our industrial relation task, conditional Ridge is a deliberately strong control: any proposed nonlinear detector must demonstrate value beyond a legal linear model before a cross-variable representation is credited. Self-lag denotes dependence on earlier values of the same variable, whereas cross-lag denotes dependence on earlier values of another variable. Neither a nonzero routing weight nor an active gradient proves that a detector identifies changes in those relationships. Mechanism claims require known relation-preserving and relation-breaking cases, including marginally similar controls and normal synchronous changes.

CAROTS uses causality-aware augmentations and contrastive learning to improve multivariate anomaly detection [@carots2025]. Its published method includes a complete augmentation, representation-learning, and detection procedure, rather than an arbitrary pairwise binary classifier. Accordingly, relationship-preserving augmentation and contrastive discrimination cannot be presented as novel merely by attaching them to a new backbone. The current Hybrid candidate does not resume the previously stopped relation-routing or isolated contrastive-loss variants. Instead, it tests whether joint temporal density modeling yields useful anomaly evidence; any additional claim about lag-specific detection remains conditional on controlled results.

## Normality learning, invertible models, and anomaly criteria

Reconstruction, prediction, prototype distance, and likelihood make different assumptions about normality. Their training losses can emphasize different features from those that control event detection. Real NVP introduced tractable density estimation through invertible affine transformations with exact log-determinants [@realnvp2017]. Wavelet Flow already combines wavelets with normalizing flows and models signal scales explicitly [@waveletflow2020]. Hybrid reuses these established principles; a Haar transform inside a flow is not its novelty. GANF also uses joint density estimation for time-series anomaly detection through a graph-factorized flow [@ganf2022], and MTGFlow studies multivariate detection with dynamic graphs and entity-aware normalizing flows [@mtgflow2023]. Thus neither window likelihood nor graph/dependency modeling is unprecedented.

The current candidate instead tests a particular alternating-coordinate joint normality model and its sustained-anomaly behavior under fixed information permissions. An invertible representation does not discard dimensions through an encoder bottleneck, but this property alone does not guarantee that the learned density orders anomalies correctly. The identity-transform Student-t control and the time-only and local-context ablations test whether learned dependencies and multiscale organization are useful, rather than attributing improvement to a generic heavy-tailed score or a change in score units. A publication-level distinction from existing flow methods would require such empirical evidence and an appropriate direct comparison; architectural resemblance alone is not treated as a contribution.

## Mixed-unlabeled adaptation and evaluation permissions

Mixed-unlabeled adaptation raises a different question from normal-reference learning. A target adaptation pool can contain unusual but legitimate states, anomalous episodes, and distributional changes relative to the source data. TSAD-C studies contaminated multivariate anomaly detection using a complete graph-conditional diffusion framework [@tsadc2025]. Such work motivates contamination-aware evaluation, but does not justify adding a robust loss when an experiment has not identified contamination-specific harm. We separate original-background adaptation, anomalous replacement, and matched normal replacement, and distinguish adaptation effects from calibration effects. A full contamination-robustness claim requires both detectable anomalies and a reproducible benefit under matched training exposure.

Information permissions also limit comparisons. A pretrained source detector, a target-trained detector, and an offline window detector answer different questions. Restricting an offline score to a common set of timestamps does not remove the future observations used to construct that score. We therefore report native offline and causal endpoint results separately, disclose target-training and standardization permissions, and use a shared evaluator only where the raw axes and labels are compatible. Published scores under different permissions are background references rather than interchangeable rows of a matched experiment.

# Problem Formulation

## Observations, causal windows, and anomaly events

Let $x_t\in\mathbb R^C$ denote the multivariate observation received at time $t$, and let $\mathcal F_t=\sigma(x_1,\ldots,x_t)$ denote the available observation history. An endpoint detector produces $s_t$ measurable with respect to $\mathcal F_t$. For a context length $H$, its input is

$$X_t=[x_{t-H+1},\ldots,x_t]\in\mathbb R^{H\times C}.$$

The endpoint permission permits the current observation $x_t$ after arrival and excludes $x_{t+1:}$. A predictor of $x_t$ has the stricter input permission $\mathcal F_{t-1}$; the two tasks must not be conflated. The present density model is an observed-window endpoint detector, not a masked-target predictor. Window normalization and every model statistic must respect the same permission.

Ground-truth anomaly events are maximal contiguous positive intervals in the fixed official label sequence, subject to the preregistered boundary and censoring rules. Their labels are used only for evaluation after model and calibration choices have been frozen. A weak anomaly refers to a small change relative to an explicitly defined training scale, not an uncertain label or a short event. A sustained anomaly is assessed through its event-level discovery and within-event coverage, including a predefined steady portion when one exists. Public fault labels alone do not identify which variable, lag, or physical relationship is abnormal.

## Historical formulation: stopped joint-density candidate

Channelwise location and scale $(\mu_c,\sigma_c)$ are estimated using legal FIT data only. With a fixed positive numerical floor on scale, the model receives $\widetilde X_{t,c}=(X_{t,c}-\mu_c)/\sigma_c$. This operation preserves within-window level relative to FIT; it does not subtract each window's own mean. Let $f_\theta:\mathbb R^{H\times C}\rightarrow\mathbb R^{H\times C}$ be invertible, and write $z_t=f_\theta(\widetilde X_t)$. With a factorized Student-t prior of fixed degrees of freedom $\nu=5$,

$$p_0(z)=\prod_{i=1}^{HC} \operatorname{StudentT}_{5}(z_i),$$

the window negative log-likelihood in standardized coordinates is

$$\ell_\theta(\widetilde X_t)=-\log p_0(f_\theta(\widetilde X_t))-\log\left|\det J_{f_\theta}(\widetilde X_t)\right|.$$

The fixed per-domain standardization contributes a constant density offset in physical coordinates. Its omission from standardized-space scores does not change within-domain ranking or an identically calibrated threshold, but raw likelihood values from separately standardized domains are not assumed comparable. The training objective and endpoint score are

$$\mathcal L_{\mathrm{FIT}}(\theta)=\frac{1}{|\mathcal T_{\mathrm{FIT}}|}\sum_{t\in\mathcal T_{\mathrm{FIT}}}\frac{\ell_\theta(\widetilde X_t)}{HC},\qquad s_t=\frac{\ell_\theta(\widetilde X_t)}{HC}.$$

The model is selected by the same normal SELECT likelihood, without QUERY labels. No reconstruction, prediction, reference-retrieval, or temporal accumulation score is added to $s_t$.

The current implementation design uses four affine coupling blocks, alternating time-domain coordinates and a two-level orthogonal Haar transform. A coupling keeps one partition unchanged and applies an invertible affine map to its complement:

$$u_a=v_a,\qquad u_b=v_b\odot\exp a_\theta(v_a)+b_\theta(v_a),$$

whose log-determinant is the sum of the log-scales $a_\theta$. Alternating checkerboard partitions expose different time–channel coordinates across blocks. The scale and translation functions use dilated temporal convolutions and cross-channel mixing. An orthogonal transform has unit absolute determinant, so it does not create an unaccounted likelihood term. The implemented two-level Haar transform requires $H$ divisible by four and uses exact adjacent-pair analysis and synthesis without padding. This finite observed-window map can mix coordinates within the available window; it is not a claim that every latent coordinate has a directed physical causal interpretation. Coordinatewise terms used to sum the prior and Jacobian are diagnostic allocations; after cross-channel transformations they are not per-channel marginal or conditional densities. Only the complete sum over all $H C$ coordinates defines the stated joint likelihood.

This construction differs from a fixed normal center because normality is learned as a transformed distribution rather than distance to a single reference. It differs from nearest-neighbor retrieval because no QUERY-dependent memory search defines the criterion. It differs from the stopped shared prediction–reconstruction design because the training and detection quantities are the same likelihood. Nevertheless, whole-window averaging may dilute a weak local departure or delay a warning, and maximum likelihood may allocate probability poorly under a new legitimate operating state. These are explicit falsifiable limitations, not effects removed by the architecture definition.

## Training regimes and future adaptation questions

We distinguish source training (Stage 1), source-calibrated zero-shot detection (Z-S), target-calibrated detection without parameter updates (Z-T), unlabeled target adaptation (U), and few-event supervision (F). Target CAL may define statistics and thresholds only within its authorized role; it never supplies gradients. Target FIT supplies U updates without anomaly labels. F supervision is counted in complete independent support events, with support, validation, and query events separated. A change from Z-S to target-trained learning changes the scientific question and is reported as such.

This first evaluation establishes only the candidate's basic detection capability under its stated training permissions. It does not establish multisource transfer, a contamination-robust U mechanism, or the value of few-event labels. Those questions remain in the research framework and require separate matched tests once the basic criterion is supported.

# Experimental Setup

## Historical joint-density candidate: fixed public-data settings

The data roles are inherited from the existing frozen assets; they are not reselected using candidate scores. Every input window must be contained in a legal role or its explicitly authorized causal context. No physical record is split into overlapping training and query windows, and no discontinuous records are concatenated. Existing guards are retained where they exceed the candidate's actual footprint. The candidate uses $H=192$ for PSM and SWaT and $H=100$ for TEP. These settings are fixed before candidate QUERY results are opened.

| Setting | Fixed training and selection assets | Fixed evaluation scope and interpretation |
|:--|:--|:--|
| PSM | FIT $[0,79488)$, SELECT $[80032,105984)$, CAL $[106528,132481)$; 25-channel FIT-only standardization | Existing complete test axis $[512,87841)$ and 71 complete events; external comparisons additionally use their exact common scored intersection. Repeatedly exposed development data. |
| SWaT | Source/normal-reference FIT $[0,296020)$, separate historical U-FIT $[296564,395237)$, SELECT $[395781,445117)$, CAL $[445661,495000)$; 31 retained official channels | Existing scored QUERY $[608535,944919)$ with 22 complete attacks, including nine longer than 512 samples. The reserved donor interval is not merged into normal-reference FIT. One operating record, not 22 independent runs. |
| TEP Rieth2017 V1 | Normal simulation runs 1–60 FIT, 61–80 SELECT, 81–100 CAL; run boundaries preserved | Fault runs 201–220, all 20 fault types; the predefined 4/11/14 subset is reported alongside all types. Existing target axis is samples 101–960, with fault onset at sample 161. A separate normal-test pool is retained. |

TEP's fixed response task uses XMEAS9 as target and the established XMV10/XMEAS21 conditional information. The density candidate scores their complete three-channel joint window; this differs from the target-XMEAS9 residual scored by the historical regression task. Their data axes and legal inputs can be matched, but their detection objects must remain explicit. A gain in joint-window detection alone cannot establish improved conditional-XMEAS9 modeling. XMV10 is a closed-loop manipulated variable and is not assumed to be a strictly exogenous cause. The three-minute sample interval permits conversion of index delay into elapsed time in this task. Shared normal prefixes across faults from the same simulation run are not counted repeatedly as independent false-alarm evidence. Fault-wise prevalence on the scored axis is high, so AP is interpreted together with ROC-AUC, normal false alarms, delay, and coverage rather than alone.

SWaT attack labels identify system-level attack intervals; they are not precise per-channel abnormality labels. Long-event analyses retain the nine predefined events and separately describe onset, sustained state, and recovery where the fixed boundaries permit. A low score during an event does not by itself prove that a sustained observable deviation exists in the retained channels. PSM, SWaT, and these TEP assets have historical development exposure. No result from this evaluation is described as a blind confirmation, and no favorable entity or fault is substituted after evaluation.

## Comparisons, information permissions, and ablations

Hybrid is the proposed method. LEFT, TimeRadar, and task-appropriate statistical models are external comparators; old Hybrid versions are internal historical controls. Native LEFT preserves its published learning and scoring chain. Its causal endpoint readout is named explicitly when used. Native retrospective LEFT and TimeRadar scores remain in an offline comparison track if their construction uses future observations relative to an earlier scored point. Reusing those scores on a shared axis does not change their information permission.

| Comparator | Purpose | Permission that must accompany each result |
|:--|:--|:--|
| Native LEFT / causal endpoint LEFT | Strong joint time–frequency reference | Target training, scaler, selected checkpoint, endpoint versus retrospective readout |
| TimeRadar | Pretrained time–frequency reference | Official weight identity, pretraining exposure, target statistics, native offline versus causal permission |
| History Mean 512 | Strong simple temporal control | Strictly earlier 512 observations; its own legal calibration and fixed residual scale |
| History and conditional Ridge | TEP history/conditional-response controls | Normal FIT runs, lag specification, SELECT rule, legal conditions, and own CAL |
| Fixed simple LEFT–Ridge combination | Tests whether joint modeling exceeds simple evidence combination | Matching axes and calibration roles; previously fixed rule, without QUERY weight selection |

Existing raw results are reused only when data identity, channel order, units, score axis, weight identity, and calibration role are compatible. A missing compatible row is marked unavailable; an author-reported best score under a different protocol is not inserted as a matched result. This draft does not require retraining completed external models.

The candidate's fixed internal comparisons are the full alternating-domain model, a parameter-matched time-only version without the Haar-coordinate blocks, and a local-context version with a shorter temporal receptive field. A fixed identity-transform Student-t score tests whether learned transformations are necessary. These controls distinguish learned density from the prior alone, multiscale coordinates from temporal capacity, and long context from a local likelihood model. They do not individually prove weak-anomaly or lag-change detection. Known controlled tasks, reported separately from public-data scores, are required for those claims. Stopped reference-center, masked-inference, SOC, routing, and shared-prediction candidates are not resumed.

Before opening the new QUERY results, we also fix the zero-parameter untrained full-model score as a control. All affine couplings and the input affine start at identity, leaving one net orthogonal Haar analysis followed by the same Student-t prior. This control is computed directly from observed windows without fitting or additional training; it separates learning from the change in an independent non-Gaussian prior under coordinate rotation. It is not an alternative score chosen from QUERY performance.

## Calibration, events, and sustained detection

Each method obtains a threshold from its own score units using the same authorized CAL locations and the frozen quantile rule. The existing rule is the 0.99 higher empirical quantile with a strict score-greater-than-threshold decision. A score-definition change therefore requires its own CAL evaluation; a historical numerical threshold is not transferable across units. CAL is not a QUERY-derived false-alarm guarantee, and its finite size and domain shift are disclosed. No QUERY labels choose a threshold, checkpoint, seed, horizon, aggregation weight, or entity.

TEP uses all 8,000 authorized normal CAL endpoints. PSM and SWaT use the same 512 pre-fixed mature CAL endpoints as their cached native controls. The new SWaT model can legally score earlier CAL endpoints with its shorter history, but the comparison retains the native control's original mature axis beginning at 446,173; the extra predictions are archived rather than substituted into calibration.

Pointwise Average Precision (AP) and ROC-AUC are computed on the exact common observed axis. AP is not renamed trapezoidal PR-AUC. Strict pointwise precision, recall, and F1 use the frozen calibrated threshold, without point adjustment. Three-consecutive-point confirmation is causal: confirmation occurs at the third scored positive point, without retrospectively marking the first two as detected. Both confirmed event discovery and confirmed anomalous-point coverage are reported.

Event evaluation uses the original event boundaries and a fixed one-to-one matching rule between predicted alarm fragments and true events. Precision, recall, and F1 under that matching are distinct from any-hit recall, which is reported separately. We report the first detection delay, undetected-event censoring, normal false-positive points, and false-alarm fragments per 1,000 observed normal points. Within-event coverage, predefined long-event steady-state macro and pooled coverage, and the longest consecutive miss quantify persistence rather than onset alone. Physical-time delay is supplied only when the sample interval is reliable. Unscored points are unknown, not automatically normal or missed. Discontinuous query segments are evaluated with their boundaries intact.

## Reproducibility, uncertainty, and decision criteria

The initial complete candidate and matched controls use the fixed first seed 2025. Configurations record source and data versions, role manifests, actual training updates, selection decisions, parameter counts, runtime, peak memory, and failure costs. Complete continuous scores, original timestamps or indices, labels for evaluation, and calibration inputs are retained. Numerical checks verify inverse reconstruction, the analytic log-determinant on a minimal input, finite gradients, causal endpoint visibility, and equality between the implemented training criterion and evaluation score. These are validity checks, not evidence of detection improvement.

TEP and PSM reuse the frozen normal training draw plans and maximum update counts of the cached target-trained comparison (3,760 and 12,400 updates). SWaT uses the same maximum of 20 epochs and normal-SELECT patience of three; actual stopping and selection can differ between objectives. The new SWaT implementation halves its learning rate after each epoch. The historical native implementation's type-1 schedule may have a one-epoch phase difference, so this row is not described as an identical per-update optimizer trajectory. The three new learned variants share their actual schedule and selection rule exactly.

A single seed on exposed development data supports screening, not a statistically confirmed generalization claim. Event-level paired summaries retain every event, while TEP uncertainty uses simulation run as a clustering unit; overlapping points are not independent replications. If the candidate demonstrates useful detection, subsequent confirmation freezes the implementation before new independent entities or runs are opened and includes repeated seeds. No significance claim is made from an event count that overstates the number of independent operating runs.

The candidate is supported only if an improvement relevant to weak or sustained anomaly detection can be attributed to the learning criterion or its temporal/multiscale structure and is not purchased through a severe increase in normal false alarms. It need not dominate every dataset or every metric. Conversely, a small AP change alone, an overall likelihood rescaling, improved training loss, or a threshold-only gain does not establish a methodological contribution. If the fixed complete design lacks a useful effect relative to the mature methods and simple controls, this candidate is stopped without a sequence of post hoc score, threshold, or architectural adjustments.


# Completed adaptation-branch evidence (17 November 2026)

This section records the evaluated implementation, not the final Hybrid method. Official TimeRadar weights, its complete forward/scoring chain and historical native configuration were retained. The native L=1 score is copy variance; no post-hoc reconstruction readout is substituted. Target FIT, normal SELECT, separate CAL, masks and checkpoints were fixed without QUERY labels. All TimeRadar/LEFT results here are offline retrospective; cropping or three-point confirmation does not make them strict online detections.

The original three SMD machines showed reproducible Cross-over-Self AP gains under three fixed training seeds. Those local positives remain valid. In the complete seed2024 SMD28 extension, however, Cross-minus-Self AP averaged +0.0483 percentage points with median -0.0194 and13 wins/15 losses. On the additional25 machines the mean was -0.0242 percentage points (10 wins/15 losses). SMD is one exposed domain; the machines are not independent datasets. PSM three-seed negatives and false-alarm costs remain in the evidence. The matched-data LEFT comparison was mixed, with different initialization and native training budgets explicitly retained, including the unfinished SMD curriculum.

Receiver-specific soft aggregation changed actual information sources, decoder outputs and native scores, but improved AP over mean Cross only on SMD1-1 and reduced three-point anomaly coverage in all four development cases. Exact-parameter controls did not support a stable independent selective-aggregation contribution. The fixed recipe was stopped. Cache-only E and E+0.9V diagnostics exposed some information omitted by V, but those advantages already existed before adaptation; they did not rescue the adaptation claim or become a selected new method.

The final matched experiment isolated whether the native decoder needed to update. It froze every native parameter and buffer, kept the decoder in eval mode, and retained autograd through its operations. Only16912 adapter parameters were trained. Four tasks used seed2024 and2000 updates with identical initialization, FIT draws, masks, SELECT and CAL rules. Earlier decoder-plus-adapter and zero-update results were reused.

| Development task | Zero AP | Decoder+Self AP | Adapter-only Self AP | Decoder+Cross AP | Adapter-only Cross AP |
|---|---:|---:|---:|---:|---:|
| PSM | .431166 | .426359 | .429517 | .425339 | .428966 |
| SMD1-1 | .609689 | .603667 | .595749 | .616477 | .601071 |
| SMD1-2 | .282421 | .298940 | .293192 | .303594 | .283690 |
| SMD1-3 | .357354 | .383640 | .357892 | .385731 | .361542 |

Freezing partly recovered PSM ranking but did not exceed zero-update AP; its Cross three-point false-positive rate fell to zero while anomaly coverage fell4.233% to0.122% and event hits fell11 to2. All three SMD adapter-only Cross AP values were below joint-update Cross. Cross-over-Self under the frozen decoder had two positive and two negative AP differences. These results do not identify decoder updating as the sole cause of the earlier limitations. Normal reconstruction improvement is not credited as detection improvement.

The branch status is **evaluated; no broad, stable independent increment established**. No additional aggregator, freezing schedule, regularizer, distillation, scoring combination or training-budget search is pursued within it. This does not reject the complete TimeRadar foundation, erase local positives or settle all possible cross-variable methods. Raw code, weights, scores, statistics and negative evidence remain bound to their original versions. Future method decisions are separate from this closed branch. The paper remains a methods paper; the completed diagnostics are research evidence, not a change to an evaluation-paper contribution.

Numerical sources and full permissions/ROC/event/false-alarm/coverage tables: [three-seed results](../../docs/experiments/timeradar_cross_variable_repeat_20261116/RESULTS.md), [SMD28 and matched LEFT](../../docs/experiments/timeradar_smd_full_left_matched_20261116/RESULTS.md), [selective aggregation](../../docs/experiments/timeradar_selective_context_20261116/RESULTS.md), [fixed error decomposition](../../docs/experiments/timeradar_error_decomposition_20261117/RESULTS.md), [decoder-freeze paired comparison](../../docs/experiments/timeradar_adapter_only_20261117/RESULTS.md). Auxiliary results do not replace the formal V comparisons. No new model or GPU job was run for this editorial update.

# Native source replay: completed evidence (18 November 2026)

We next examine an established learning control using the complete native LEFT model, rather than resuming the stopped shallow TimeRadar adapter. Each domain retains its own named normally trained initial checkpoint, scaler, prototypes and running training step. With seeds 2025, 2026 and 2027, R uses batches $[T,S]$, TT uses $[T,T']$, and SS uses $[S',S]$, where each half contains 128 windows of length 192. T is drawn from the fixed natural mixed target FIT segment; S comes only from actual source training FIT. Independent draw streams produce the primed halves. R and TT share T at each corresponding position; R and SS share S. All arms use a fresh Adam optimizer, learning rate $10^{-4}$, 2,000 complete native single-forward updates and the fixed endpoint. No QUERY labels choose training or checkpoints. noU is one fixed unadapted reference, not three independent repeats. These native whole-window scores are offline retrospective; three-point confirmation does not change their information permission.

| Paired R−TT effect | SWaT | PSM |
|:--|--:|--:|
| AP, percentage points | +2.609 ± 1.324 | +1.592 ± 0.111 |
| ROC-AUC, percentage points | +0.555 ± 0.602 | +1.526 ± 0.070 |
| Strict false-positive points | −154.000 ± 69.635 | −6853.667 ± 135.072 |
| Three-confirmed anomalous-point coverage, percentage points | +9.613 ± 9.702 | −19.781 ± 0.636 |
| Long steady-state pooled coverage, percentage points | +11.842 ± 11.957 | −16.022 ± 0.418 |

Entries are means and sample standard deviations of within-seed differences ($n=3$, ddof=1), not confidence intervals or independent operating-run replicates. SWaT coverage uses its frozen anomalous-point axis; PSM coverage in this table uses its 48 complete events, excluding boundary-truncated events from event metrics. Thus the two coverage columns summarize their respective predefined evaluation populations, not a pooled cross-domain effect. The denominators and raw axes are retained in the [SWaT three-seed report](../../docs/experiments/left_swat_replay_repeat_20261118/RESULTS.md) and [PSM three-seed report](../../docs/experiments/left_psm_replay_20261118/RESULTS.md).

R−TT AP is positive and actual normal false-positive counts are lower in all three seeds on both exposed tasks. Their operational effects differ: SWaT gains coverage relative to TT, whereas PSM loses strict recall, event discovery and sustained coverage. In PSM, R discovers 17/48 complete events under three-point confirmation in every seed, versus 26/48, 26/48 and 27/48 for TT. SWaT's nine adapted models all discover 15/22 events, and R still has substantial long misses. Relative to noU, SWaT replay has more false positives and less coverage; its AP improvement does not establish operational dominance. Source replay therefore supplies a reproducible, task-dependent learning control, not a general sustained-detection or contamination-robustness result.

PSM's fixed CAL role contains 3,848/3,848 attack-labeled points, and all 512 selected calibration points are attack-labeled. This composition was checked after training and score freezing; calibration was not relocated. Each model uses its own strict-greater-than `q99/higher` threshold on the same CAL locations. This mixed-CAL operating point cannot certify a normal 1% false-alarm budget. QUERY normal false positives remain measurable using evaluation labels, but their reduction can accompany the large recall and coverage reductions reported above. Ranking metrics and thresholded operational behavior must therefore remain separate. The repetitions share one initial model per domain, both tasks have development exposure, and natural target/source contrasts do not isolate a causal anomaly-contamination effect. These results do not establish an original Hybrid contribution.

# Current method hypothesis: preserve native source outputs

The current four-run experiment adds functional preservation to native R while leaving the native inference chain unchanged. A frozen copy of each domain's native initial model reads source FIT windows S only. In the same input-coordinate units, let its three $B\times192\times C$ reconstructions be $g_v(\theta_0;S)$ for `X_hat`, `X_hat_from_S` and `X_hat_ms`, and let the student return $g_v(\theta;S)$. They are outputs of shared, interacting pathways, not independent views. Writing $D=B\cdot192\cdot C$, $\Delta_v=g_v(\theta;S)-\operatorname{stopgrad}g_v(\theta_0;S)$ and $\overline\Delta=\frac13\sum_v\Delta_v$, we compare

$$P=\frac{\|\overline\Delta\|_F^2}{D},\qquad V=\frac{1}{3D}\sum_{v=1}^{3}\|\Delta_v\|_F^2,\qquad \mathcal L=\mathcal L_{\mathrm{native},R}+\lambda\mathcal L_{\mathrm{pres}},\quad\lambda=1,$$

where $\mathcal L_{\mathrm{pres}}$ is P or V. The identity

$$V-P=\frac{1}{3D}\sum_{v=1}^{3}\|\Delta_v-\overline\Delta\|_F^2\geq0$$

shows that V additionally penalizes dispersion between pathway drifts, whereas P permits their cancellation. V is pointwise at least as strong at the same coefficient; an observed difference cannot be attributed solely to separating views under equal regularization strength. Nor does this algebra imply any downstream detection improvement.

The added student forward is in eval behavior with gradients retained; the frozen teacher uses `no_grad`. Auxiliary execution restores buffers, the running step, heterogeneous per-module modes and CPU/CUDA RNG state, so these passes do not silently advance native replay state or its random trajectory. Teacher/auxiliary compute is an added training cost, even when update counts match R. Inference uses the original native model score, readout and calibration rule; neither teacher outputs nor preservation errors become an anomaly-score mixture. Full definitions and the state-restoration contract appear in [the current method definition](method_definition.md).

Functional preservation and rehearsal are established precedents. [Buzzega et al., NeurIPS 2020, §3/Eq5](https://proceedings.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) matches historical logits in Dark Experience Replay. This completed experiment fixes native reconstruction teachers and compares aggregated versus individual output drift; it is not a DER reproduction. Source replay, squared response matching and the native LEFT reconstruction pathways are not claimed as inventions. Whether this specific constraint yields a useful detection trade-off beyond R remains an empirical question.

**Completed fixed-seed results.** Each domain completed P and V from the same native initialization and seed2025 R plan. Native whole-window inference and each model's own fixed mixed-CAL quantile remain unchanged. The source-preservation term has a verified differentiable path; numerical/state checks do not establish efficacy.

| Domain | Arm | AP | ROC-AUC | VUS-PR | VUS-ROC | Strict FP | C3 complete-event coverage | C3 any-hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SWAT | noU | 0.607560 | 0.842267 | 0.606345 | 0.829634 | 5073 | 60.789% | 15/22 |
| SWAT | R | 0.629347 | 0.885512 | 0.639265 | 0.884546 | 5143 | 57.948% | 15/22 |
| SWAT | P | 0.635728 | 0.892502 | 0.642965 | 0.890241 | 5258 | 58.038% | 15/22 |
| SWAT | V | 0.625967 | 0.888787 | 0.627882 | 0.882386 | 5290 | 58.231% | 15/22 |
| PSM | noU | 0.583688 | 0.771345 | 0.575447 | 0.775647 | 7381 | 65.710% | 21/48 |
| PSM | R | 0.587891 | 0.748194 | 0.586017 | 0.761687 | 5294 | 61.901% | 17/48 |
| PSM | P | 0.578313 | 0.738939 | 0.577226 | 0.753827 | 5851 | 63.038% | 19/48 |
| PSM | V | 0.576862 | 0.738653 | 0.575583 | 0.753346 | 5566 | 61.834% | 17/48 |

V does not establish the hypothesized incremental benefit over R or P. Its AP changes relative to R are -0.3380 percentage points on SWaT and -1.1029 on PSM, with strict false positives increasing by147 and272. SWaT V has a small complete-event C3 coverage gain; PSM V does not. Aggregate preservation P retains a SWaT AP gain of0.6381 percentage points over R, with115 extra strict false-positive points and no added event discovery. On PSM its AP decreases0.9579 percentage points, while C3 discovery rises17 to19/48 at564 extra C3 normal false-positive points. These limited aggregate gains cannot be attributed to per-view preservation. Both P and V remain below noU in complete-event C3 coverage on both domains.

These are single-seed developmental results with mixed CAL; the all-attack PSM CAL limitation remains in force. No threshold, coefficient or readout was selected from QUERY. Actual measured GPU occupancy is0.473455 hours, plus a separate diagnostic wall-time upper bound of3.847 seconds. A role-separated same-family reviewer reproduced648 main numerical entries and640 CSV entries exactly, excluding an independent event-matching-F1/delay recompute. [The full report](../../docs/experiments/left_source_preservation_20261118/RESULTS.md) retains TT/SS controls, strict and C3 metrics, long steady-state macro/pooled coverage, longest misses, PR/ROC curves and complete score-table definitions. No further candidate or parameter search follows automatically. The paper remains a methods paper, but this fixed response-preservation experiment does not yet supply its original core contribution.
