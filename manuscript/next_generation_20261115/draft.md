---
title: "Hybrid: Time–Frequency and Temporal Normality Modeling for Industrial Time-Series Anomaly Detection"
author: []
date: "Working draft, updated 20 November 2026"
lang: en
bibliography: references.bib
---

> **Research status, 20 November 2026.** The native-V conditional quantile B/C candidate remains stopped. A separate public SWaT/PSM head-only TimeRadar TT/SS/R comparison has completed six fixed2000-update trainings,12 new banks and one bounded independent result recomputation; frozen A/noU is reused. PSM replay recovers ranking and part of coverage relative to TT with higher false alarms; SWaT does not establish a corresponding useful increment. Official TimeRadar and mature replay retain their attribution. These natural-mixture development results identify neither contamination causality nor an original Hybrid contribution. No successor is automatically launched.

# Introduction

Industrial monitoring requires more than recognizing abrupt, high-amplitude departures from a nominal range. A weak deviation can be significant when it violates a normal response relationship, whereas a large change can be acceptable when several variables respond coherently to an operating transition. An anomaly may also persist after its initial transition: an effective monitor should discover the event and retain useful evidence during its abnormal steady state. These requirements are particularly demanding for multivariate streams in which responses are delayed, operating conditions change, and target-domain training records are unlabeled rather than independently certified as healthy.

Time–frequency representations offer a useful way to describe both local variation and temporal organization. TimeRadar learns a fractional time–frequency representation and combines reconstruction with contextual deviation learning [@timeradar2026]. LEFT learns interacting temporal, spectral, and multiscale views, followed by joint representation learning and reconstruction [@left2026]. These methods motivate a richer representation than a single instantaneous vector. However, a representation with good reconstruction or prediction accuracy does not automatically provide a useful anomaly ranking. A flexible reconstruction model may reconstruct a sustained abnormal state, and a predictor may improve normal accuracy without increasing the separation between abnormal responses and difficult normal transitions. These are empirical risks, rather than universal properties of either modeling family.

The latest question is whether head-only natural-mixture reconstruction updates preserve native masked-reconstruction disagreement, and whether historical source replay changes that trade-off. The preceding fixed study of conditional quantiles of official TimeRadar V did not establish a benefit and remains stopped; its raw-variable forecasting and simple history controls are retained as historical evidence. The earlier conditional-energy study remains a separate stopped experiment, not a matching control for this formulation. Mature source replay and complete external models remain learning references rather than inventions or evidence inherited by the latest candidate.

Several boundaries matter. First, an exact density is not a semantic definition of normality: a high-likelihood input can still be undesirable, and a legitimate new operating state can have low likelihood. Second, a causal endpoint detector may use the current observation after it arrives, but it must not use later observations. This is different from predicting the current observation from strictly earlier inputs. Third, statistical cross-variable dependence is not evidence of physical causation or a verified directed lag mechanism. Finally, robustness to unlabeled training contamination requires a separate experiment after the detector has demonstrated useful baseline discrimination. Performance that remains unchanged near a detection floor does not establish robustness.

The fixed native-V study uses PSM for multivariate development comparison and SWaT for event and long-duration analyses. Earlier run-separated Tennessee Eastman Process (TEP) experiments supply historical conditional-response mechanism evidence; the native-V candidate is not evaluated on TEP. These public settings have already influenced the research program and are explicitly treated as development or mechanism evidence. They cannot establish a new blind-confirmation claim. Strong simple controls, including historical means and history/conditional Ridge models where applicable, remain necessary alongside LEFT and TimeRadar.

A defensible contribution must go beyond applying mature source replay: it requires an attributable detection increment with useful ranking, coverage and false-alarm trade-offs under matched permissions. The locked LEFT paired experiment separates abnormal replacement from normal replacement of the same target slots. Neither the stopped conditional-geometry study nor the subsequent normal-source V-conditioning comparison repeats that exposure experiment or inherits its protection effects. None establishes a new robust learning mechanism. No stopped architecture or isolated favorable metric is promoted to an original contribution.

# Related Work

## Time–frequency and multiview anomaly representations

TimeRadar uses a learnable fractional time–frequency representation within its foundation-model anomaly-detection framework [@timeradar2026]. Its fractional transform spans different domain rotations; it should not be described as a frequency-only encoder. LEFT combines temporal, spectral, and multiscale tokens, selective cross-view interaction, and an analysis–synthesis reconstruction chain [@left2026]. CATCH instead examines frequency patching with channel-aware modeling [@catch2025]. These works show different ways of exploiting temporal and spectral structure. Our locked adaptation studies retain LEFT's complete native learning and scoring chain as an external reference. A stopped Hybrid study tested frozen official TimeRadar response geometry inside a conditional energy criterion; the latest fixed study instead conditions native V on strictly historical inputs. Neither study claims the transform, multiview fusion, causal attention or replay itself as an invention.

## Temporal dependencies and relationship anomalies

Forecasting and conditional-response models can expose departures that are not apparent from individual variables. Long-context forecasting also motivates temporal encoders, but a forecasting foundation model is not automatically an anomaly detector. The original TimesFM is a forecasting model [@timesfm2024]; the specific lightweight modules used here come from the pinned TimesFM3 release [@timesfm3release2026], without its pretrained weights. Historical public TEP experiments retain conditional Ridge as a strong relationship-modeling control. Self-lag denotes dependence on earlier values of the same variable, whereas cross-lag denotes dependence on earlier values of another variable. Neither a nonzero routing weight nor an active gradient proves that a detector identifies changes in those relationships. Mechanism claims require known relation-preserving and relation-breaking cases, including marginally similar controls and normal synchronous changes; the present two-arm geometry comparison makes no directed-lag claim.

CAROTS uses causality-aware augmentations and contrastive learning to improve multivariate anomaly detection [@carots2025]. Its published method includes a complete augmentation, representation-learning, and detection procedure, rather than an arbitrary pairwise binary classifier. Accordingly, relationship-preserving augmentation and contrastive discrimination cannot be presented as novel merely by attaching them to a new backbone. The present conditional-geometry study does not resume stopped relation-routing or isolated contrastive-loss variants.

## Mixed-unlabeled adaptation and evaluation permissions

Mixed-unlabeled adaptation raises a different question from normal-reference learning. A target adaptation pool can contain unusual but legitimate states, anomalous episodes, and distributional changes relative to the source data. TSAD-C studies contaminated multivariate anomaly detection using a complete graph-conditional diffusion framework [@tsadc2025]. Such work motivates complete mixed-unlabeled method comparisons with matched data permissions. Contamination-specific harm need not be established before such comparisons, and method effectiveness must be judged using absolute detection performance as well as differential responses. We separate original-background adaptation, anomalous replacement, and matched normal replacement, and distinguish adaptation effects from calibration effects. A full contamination-robustness claim requires both detectable anomalies and a reproducible benefit under matched training exposure.

CANDI curates adaptation windows using reconstruction scores and similarity to normal reference representations, then updates input/output adapters around a frozen reconstruction backbone [@candi2026]. Our external comparison retains its complete MLP, FPM filtering and SANA adapters at the pinned released implementation. Its adaptation is restricted to the authorized U-FIT stream and then frozen for CAL and QUERY; this differs from the author's continuously adapting test stream. The normal reference is the source validation segment alone. CANDI's internal source-reference threshold is distinct from the final mixed-CAL alarm threshold. Each real sliding-window MSE is assigned to its actual endpoint and evaluated against project point labels, rather than broadcasting a window label or padding a score sequence. Differences from LEFT therefore include backbone, context, source training, standardization and readout permissions; cross-model differences cannot be attributed solely to filtering. CANDI remains an external method, not a Hybrid component or contribution.

Information permissions also limit comparisons. A pretrained source detector, a target-trained detector, and an offline window detector answer different questions. Restricting an offline score to a common set of timestamps does not remove the future observations used to construct that score. We therefore report native offline and causal endpoint results separately, disclose target-training and standardization permissions, and use a shared evaluator only where the raw axes and labels are compatible. Published scores under different permissions are background references rather than interchangeable rows of a matched experiment.

# Problem Formulation

## Observations, causal windows, and anomaly events

Let $x_t\in\mathbb R^C$ denote the multivariate observation received at time $t$, and let $\mathcal F_t=\sigma(x_1,\ldots,x_t)$ denote the available observation history. An endpoint detector produces $s_t$ measurable with respect to $\mathcal F_t$. For a context length $H$, its input is

$$X_t=[x_{t-H+1},\ldots,x_t]\in\mathbb R^{H\times C}.$$

The endpoint permission permits the current observation $x_t$ after arrival and excludes $x_{t+1:}$. A predictor of $x_t$ has the stricter input permission $\mathcal F_{t-1}$; the two tasks must not be conflated. These definitions distinguish permissions: the current conditional generator and score are causal, whereas the external native LEFT full-window readout is retrospective. Window normalization and every model statistic must respect the declared permission.

Ground-truth anomaly events are maximal contiguous positive intervals in the fixed official label sequence, subject to the preregistered boundary and censoring rules. QUERY labels are used only for evaluation after model and calibration choices have been frozen. The historical controlled exposure experiment separately permits FIT labels for donor construction, while keeping learner updates label-free. A weak anomaly refers to a small change relative to an explicitly defined training scale, not an uncertain label or a short event. A sustained anomaly is assessed through its event-level discovery and within-event coverage, including a predefined steady portion when one exists. Public fault labels alone do not identify which variable, lag, or physical relationship is abnormal.

# Completed native-V and strictly historical conditioning study

Let $h_t=x_{t-512:t-1}$ and $X_t=x_{t-99:t}$, with source-FIT-only normalization. For each actual observed window, frozen official TimeRadar produces ten complementary-mask reconstructions $r_{m,t,c}$. Every call has B=1, internal norm=0, native within-window order sharing, fixed isolated mask seed84551, and the unmodified criterion=None/L=1 variance:

$$V_t=\frac1C\sum_c\frac1{9}\sum_{m=1}^{10}(r_{m,t,c}-\bar r_{t,c})^2.$$

The ten masks measure reconstruction disagreement; they are not independent calibrated posterior samples. A uses $V_t$. B (MLP32/32) and C (official TimesFM3 ResidualBlock and mixing transformer, width128/two layers/four heads/patch16, causal temporal and contemporaneous variate attention) predict sorted $(q_{.05},q_{.50},q_{.95})$ for $z_t=V_t/s_V$, where $s_V=\max(Q^{FIT}_{.95}(V),10^{-12})$. The head reads ordered last-patch channel tokens; there are no official TimesFM weights or explicit directed-lag module. The equally weighted pinball objective and untruncated criterion are

$$\mathcal L_Q=\frac1{3N}\sum_{t,\tau}\max\{\tau(z_t-q_\tau(h_t)),(\tau-1)(z_t-q_\tau(h_t))\},\qquad S_t=\frac{V_t/s_V-q_{.95}(h_t)}{\max(q_{.95}(h_t)-q_{.05}(h_t),10^{-3})}.$$

The source global-quantile initialization uses zero final weights and FP32 biases. Its score is a positive affine transform of A; both full QUERY axes have identical step0 AP/ROC and strict alarms, with no new ties or reversed order. Stable sort permits ties, negative scores remain, and final score arithmetic is FP64. The current target is absent from the history predictor; it enters only the native V at observation arrival t. The99-point overlap between teacher and predictor histories is legal and disclosed.

D shares C's actual encoder initial tensors, predicts current source-normalized raw variables, minimizes their MSE and scores the mean per-channel squared residual. D uses neither V nor the quantile criterion, so it is a task reference rather than an equal-head or same-weight ablation. New trainable parameter counts are SWaT B509091/C587011/D698143 and PSM B410787/C584707/D655129, in addition to the separately frozen1,636,043-parameter TimeRadar needed by A/B/C.

## Source permissions, fixed selection and evaluation

The source/target roles and mixed CAL are unchanged from the preceding public task. B/C/D share complete20-epoch no-replacement FIT permutations, batch128 including the final partial batch, Adam1e-4 and FP32. SWaT completes3880 updates and494360 window exposures per model; PSM1040 and131840. Full normal SELECT is evaluated at0 and every epoch; the earliest strictly minimum own-task loss selects SWaT epochs B14/C20/D20 and PSM B16/C7/D19. All six selected models freeze before CAL/QUERY value readout, and all16 score banks seal before label evaluation. No R, pollution, additional seed, heterogeneous source pretraining or private data is used.

The fixed QUERY axes contain336321 SWaT and64728 PSM points; each model uses its own score at the same512 mixed CAL points for q99/higher/strict>. PSM CAL remains all attack-labeled. These empirical thresholds do not certify normal1%FPR. The temporal information footprint is causal endpoint, but the evaluation is retrospective, not a deployed trial. Native LEFT192-point whole-window scores, MLP/CANDI10-point scores and historical TimeRadar retain their differing source statistics, pretraining, target adaptation and input availability. Ranking-only caches receive no invented CAL threshold. [Complete same-axis tables, curves, artifacts and permissions](../../docs/experiments/timeradar_native_v_conditioning_20261119/RESULTS.md) remain the authoritative exports.

## Fixed results and scientific scope

| Domain | Arm | AP | ROC | VUS-PR | VUS-ROC | Strict normal FPR% | Strict recall% | C3 normal FPR% | C3 matched | C3 any-hit | C3 complete-event cover% | C3 long macro% | C3 long pooled% | Longest event miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | A | 0.7957 | 0.8579 | 0.5766 | 0.7090 | 1.303 | 69.169 | 1.223 | 10/22 | 10/22 | 69.123 | 11.460 | 82.566 | 3941 |
| SWAT | B | 0.5749 | 0.8365 | 0.4884 | 0.7445 | 0.728 | 1.528 | 0.617 | 8/22 | 8/22 | 1.488 | 0.056 | 0.077 | 31928 |
| SWAT | C | 0.1177 | 0.3983 | 0.1268 | 0.3813 | 67.960 | 87.211 | 67.432 | 19/22 | 20/22 | 87.008 | 63.921 | 91.558 | 1218 |
| SWAT | D | 0.7623 | 0.8385 | 0.5467 | 0.6960 | 94.966 | 97.982 | 94.910 | 5/22 | 21/22 | 97.959 | 88.889 | 99.461 | 721 |
| PSM | A | 0.5510 | 0.7350 | 0.4846 | 0.6482 | 29.669 | 50.870 | 29.096 | 6/48 | 17/48 | 73.195 | 86.453 | 92.114 | 694 |
| PSM | B | 0.3684 | 0.6422 | 0.3843 | 0.5857 | 39.775 | 63.999 | 38.933 | 8/48 | 18/48 | 84.271 | 99.832 | 99.902 | 399 |
| PSM | C | 0.2898 | 0.5875 | 0.3220 | 0.6076 | 0.321 | 0.155 | 0.008 | 0/48 | 0/48 | 0.000 | 0.000 | 0.000 | 4852 |
| PSM | D | 0.5679 | 0.7438 | 0.5755 | 0.7649 | 25.507 | 58.207 | 22.008 | 22/48 | 30/48 | 76.511 | 85.168 | 91.366 | 399 |

C−A AP is−.678044 on SWaT and−.261130 on PSM; C−B is−.457265 and−.078555. The SWaT C3 coverage increase comes with67.432% normal false alarms, while PSM C has no C3 anomaly hit. B also loses ranking on both tasks; its SWaT false-alarm reduction accompanies near-zero steady coverage. Conditional normal loss alone therefore does not validate anomaly discrimination, and CAL migration cannot explain the threshold-free ranking losses. These results do not isolate unsupported-history extrapolation, anomaly predictability or interval-width behavior as a unique cause.

Retain two bounded findings. Native A in SWaT has AP.795725 and C3 FPR1.223%/recall69.123%, but long-event macro11.460% versus pooled82.566% is strongly uneven. In PSM, D improves A by.016950 AP/.008797 ROC, with lower C3 normal FPR29.096%→22.008% and matched events6→22. D contains no V criterion, and its ranking remains below the compatible MLP references; this is not evidence of time–frequency/history synergy or an automatic new candidate. Full-point recall and complete-event point coverage use different denominators; any-hit and one-to-one matching are not substituted for each other. Missed events have censored/NA delay, not zero.

**Decision:** the fixed upper-tail V-conditioning candidate does not establish an attributable detection increment and is not automatically advanced to R. Official TimeRadar's concrete baseline evidence and D's limited reference positive remain; TimesFM here means actual lightweight official code reuse without pretrained weights, not validated foundation-model or directed-lag capability. No new seed, threshold, score, loss or successor is opened. The mixed-unlabeled/two-stage research objective remains, pending the main research decision.

Thirty-four owned GPU jobs all exit successfully with no repeated training or inference, totaling2.290003054 A40GPUh. One role-independent same-family review recomputes25,227 fields with maximum difference1.11e-16, sharing the pinned official VUS implementation and reusing old VUS. A corrected missing event-boundary export changes only a derived CSV, not scores or metrics. Raw V, ten-copy endpoints, orders/masks, predictions/quantiles, original footprints and runtime bytes remain archived; six project-trained selected checkpoints and their hashes are delivered separately from upstream weights. This does not confer a new Hybrid contribution or pollution-robustness claim.

# Native TimeRadar head probing on natural mixed target data

The conditional-V B/C recipe and its prospective R adaptation remain stopped. A separate fixed same-backbone experiment now tests native TimeRadar reconstruction-head probing with data replay. The official model's fractional-order network, embeddings, encoder and buffers are frozen; only the existing complex reconstruction decoder is updated. The native ten-copy terminal V criterion, SOURCE_FIT scaler, B=1 per-window order, norm0 and isolated84551 complementary masks are unchanged from A/noU. This is an established update scope [@timeradar2026] combined with mature replay, not a claimed original Hybrid mechanism or a restoration of former channel adapters.

For each logical batch of64 actual windows, the unchanged reconstruction objective is

$$\mathcal L_{\mathcal B}(\theta_{dec})=\frac1{64}\sum_{w\in\mathcal B}\frac1{10\cdot100\cdot C}\sum_{m,j,c}(\hat x^{(m)}_{w,j,c}-x_{w,j,c})^2.$$

Every native forward has B=1; gradients of64 scaled per-window losses are accumulated before one AdamW update. Frozen modules remain in eval, including backbone dropout. No independent windows are put into a native batch to average their fractional order. TT=[T,Tprime], SS=[Sprime,S] and R=[T,S] have32 windows per half, paired T and S positions, independently sampled Tprime/Sprime, and the same official initial state/fresh optimizer. Role-local nonoverlapping complete100-window pools follow the prior head sampler, with3960/900 source/target windows in SWaT and1059/175 in PSM. Repeated draws do not create independent events.

The fixed prior PSM probing configuration is logical batch64,2000 updates, AdamW1e-4/betas(.9,.95)/eps1e-8, weight decay.1 under the original name/LayerNorm grouping, and linear warmup200. Literal names containing 'bias' and LayerNorm parameters are exempt; complex rb*/ib* names are not. Use endpoint2000 without source SELECT, CAL or QUERY selection. Train only SOURCE_FIT and the existing natural U-FIT. This does not reproduce the LEFT256-batch training exposure or create a new normal-source initialization; noU remains the official pretrained TimeRadar plus fixed SOURCE_FIT normalization.

The primary learning contrasts are R−TT, with R−noU, R−SS and TT−noU retained. All methods share the same QUERY points,512 CAL observations and V criterion. Each uses its own q99/higher/strict> threshold; SWaT's selected mixed CAL has34 attack-labeled points and PSM512, as already recorded in the previous metrics. They do not certify a normal false-alarm budget. All new endpoints and scores must freeze before new label evaluation. Natural-mixture gains or losses cannot, without a separate intervention, be attributed specifically to anomaly contamination. Mature TimeRadar/replay retain their named external attribution; no innovation claim follows from training alone. All six endpoints and twelve new score banks were frozen before evaluation. The complete fixed comparison is now available below.


## Fixed native-V adaptation results

| Task | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| SWAT | noU | 0.795725 | 0.857852 | 0.576600 | 0.708990 |
| SWAT | TT | 0.767373 | 0.840286 | 0.635916 | 0.795740 |
| SWAT | SS | 0.759642 | 0.840201 | 0.629941 | 0.793154 |
| SWAT | R | 0.761478 | 0.838195 | 0.638054 | 0.789673 |
| PSM | noU | 0.550952 | 0.735048 | 0.484578 | 0.648161 |
| PSM | TT | 0.497243 | 0.712915 | 0.520744 | 0.714204 |
| PSM | SS | 0.534776 | 0.731992 | 0.520077 | 0.697853 |
| PSM | R | 0.551502 | 0.734425 | 0.552609 | 0.721470 |

The original A/noU native-V banks and all original thresholds remain archived; each new model calibrates its own V on the same512 mixed-CAL points. Results use SWaT336321 and PSM64728 identical QUERY points. PSM's CAL is entirely attack-labeled, so low normal false-alarm rates are empirical outcomes rather than a1% certification.

On PSM, R−TT improves AP5.426pp and ROC2.151pp; C3 long-event macro/pooled coverage improves24.441/22.016% to42.016/54.195%, while normal FPR rises2.160% to3.589%. Matched events increase10 to11 but any-hit remains11. R approximately restores noU AP (.551502 versus .550952), yet loses noU recall and long coverage; its C3 normal FPR falls29.096% to3.589%, long macro86.453% to42.016%, and longest miss increases694 to1228. Relative to source-only SS, R improves ranking and lowers FPR but loses coverage. Thus PSM supports a bounded replay ranking/coverage recovery against TT, not complete preservation or domination.

On SWaT, R−TT AP/ROC are -0.590/-0.209pp; small coverage gains cost additional false alarms and leave10 of22 matched events unchanged. All three adapted point AP/ROC values fall below noU. Improved VUS values are retained as distinct range-based outcomes, without overriding point or operating evidence.

The six trainings and30 readout jobs plus engineering checks consumed8.080724 A40 GPU hours, including startup/cleanup, with no failed or duplicate jobs. One role-separated same-family review matched16458 fields (maximum difference3.33e-16), including raw-score identity, paired exposure, calibration and complete-event accounting; shared official VUS code is disclosed. Full strict/C3 metrics, hit-only delays, censored misses, event macro/pooled coverage and cost/SHA records are in `docs/experiments/timeradar_native_v_replay_20261120/`. These are two exposed tasks, one seed, a mature backbone and replay policy; neither mixed-contamination robustness nor heterogeneous Stage1 training is established. No B/C R, further seed, loss or scoring variation follows automatically.


# Fixed target-mean / source-copy reconstruction comparison

This prospective single-factor experiment retains the preceding native TimeRadar head-only R initialization, paired T/S plan, effective batch64,2000 endpoint updates, optimizer and V inference. It changes only the target-window reconstruction constraint. No conditional-V B/C, old channel adapter, new backbone, score fusion or supervised target-label training is restored. Both public tasks, SWaT31 and PSM25, and seed2025 are fixed before new results.

For ten reconstructions $o_{w,m,j,c}$ of a single actual window, define

$$\bar o_{w,j,c}=\frac1{10}\sum_{m=1}^{10}o_{w,m,j,c},\qquad
E_w=\frac1{100C}\sum_{j,c}(\bar o_{w,j,c}-x_{w,j,c})^2,$$

$$V_w=\frac1{100C}\sum_{j,c}\frac1{9}\sum_{m=1}^{10}(o_{w,m,j,c}-\bar o_{w,j,c})^2.$$

Here $V_w$ is the full-window mean unbiased copy variance, not the terminal native detection score. The classical squared-error decomposition gives $L_{copy,w}=E_w+0.9V_w$. With32 target and32 source windows in the same logical batch, the new objective is

$$L_{new}=\frac1{64}\left[\sum_{w\in T}E_w+\sum_{w\in S}(E_w+0.9V_w)\right].$$

At identical parameters, windows and masks, the original R objective satisfies $L_R-L_{new}=0.45\,\operatorname{mean}_{w\in T}V_w$, including equality of parameter gradients. Training directly computes the mean-reconstruction MSE for target windows, rather than subtracting two large quantities. The mean remains differentiable, uses all100 positions and channels, and introduces no extra half weighting. Source windows retain the original ten-copy MSE. Each native forward remains B=1; each window contributes its loss divided by64 before a single optimizer update. $E_T,V_T,E_S,V_S$ are recorded from the same forwards and do not introduce another training pass.

The inference criterion remains the unchanged channel-mean, correction1 copy variance at position99. An increase in training variance does not establish improved anomaly discrimination. The hypothesis tested is whether removing direct target-copy agreement pressure, while retaining source-copy reconstruction, provides an actual detection increment over matched R. Classical decomposition, official TimeRadar and mature replay retain their original attribution; a new contribution would require the paired detection evidence, not merely this identity.

The primary contrast is new minus R; noU, TT and SS are retained unchanged. Common QUERY axes,512 mixed-CAL observations and each model's q99/higher/strict> threshold remain fixed. SWaT has34 attack-labeled selected CAL points and PSM512; no normal1% false-alarm guarantee follows. Point AP/ROC, fixed VUS, strict/C3 operating outcomes, one-to-one versus any-hit events, long-event macro/pooled coverage and longest misses will all be reported. QUERY curves are descriptive only. Results are pending; no protection, robustness or detection-gain claim is made before completion.

# Historical stopped conditional-energy candidate and fixed geometry comparison

Given strictly past observations $h_t=x_{t-512:t-1}$, a lightweight conditional generator produces $v_k=G_\theta(h_t,\epsilon_k)$, $k=1,\ldots,4$. It uses the official TimesFM3 residual and mixing blocks at width128, two layers, four heads and patch16, including variate attention. Shared8 and per-variate8 independent Gaussian noise allow joint current-value draws. No current target value enters this predictor. The generator has612626 trainable parameters; the separate official TimeRadar response operator retains1,636,043 frozen parameters. This is not the official330M TimesFM3 model, an official quantile forecast, or a restoration of the old Hybrid V1.

For each candidate v and the observed value, independently evaluate the native TimeRadar function on $[x_{t-99:t-1},v]$. Each B=1 call estimates its own fractional order, with native within-window variable sharing, and returns ten complementary-mask reconstructions. Masks use isolated seed84551; eval mode alone does not fix the random forward masks. Source-FIT mean/scale are fixed, and the native internal normalization is disabled. Defining the response as $r_h(v)$, arm A uses

$$g_h(v)=\left[v/\sqrt C,\;\operatorname{vec}r_h(v)/(s_r\sqrt{1000C})\right],$$

where $s_r$ is the aggregate RMS of complete SOURCE_FIT responses. Arm B uses only $v/\sqrt C$. Both minimize and report

$$s_t=\frac14\sum_{k=1}^4\|g_h(v_k)-g_h(x_t)\|_2-\frac1{24}\sum_{k\ne l}\|g_h(v_k)-g_h(v_l)\|_2.$$

This is the conventional energy score, equal to half the authors' implementation convention [@generative_scoring2024]. It is not a sum of independently calibrated reconstruction and forecast scores. Raw coordinates make the fixed response mapping injective; under finite first moments the population score is strictly proper for the corresponding conditional distribution. This mathematical property neither certifies anomaly semantics nor establishes contamination robustness. The frozen response parameters receive no gradients, while generated-input gradients pass through the actual fractional transform and reconstruction chain. No healthy counterfactual, memory retrieval or student-output preservation term is used.

We retain normal-source selected noU states and fixed2000-step R adaptations for both arms. Source training also has a2000-update cap, normal SELECT every100 including step0, logical batch16 and Adam1e-4. A/B share initial generator tensors, actual source/target draws and noise plans; R uses eight target and eight normal-source examples per optimizer update. It is a shared mature learning strategy, not an originality claim. These experiments are same-domain normal initialization followed by natural mixed adaptation, without a new heterogeneous multi-source pretraining stage or a new contamination-dose matrix.

## Historical conditional-energy evaluation setup

Only public SWaT31 and PSM25 enter the experiment. No first-paper laboratory, weak-label, experimental-run or enterprise records are used. The original roles are unchanged. Because512 past points plus the observed target require513 legal points, the shared QUERY axes are SWaT `[608535,944856)` and PSM `[23016,87744)`, containing336321 and64728 points. PSM's additional360-point crop is a footprint requirement, not a score-dependent choice. Candidate scores are available at t; cached LEFT whole-window scores retain their later window-end availability and are marked retrospective.

Each model receives its own q99/higher/strict-greater-than threshold using512 fixed points in the same legal mixed-CAL range: SWaT `[586056,607320)` and PSM `[18624,21888)`. PSM's all-attack CAL is retained rather than cleaned using labels. Its threshold is an empirical operation point, not a normal1% false-alarm guarantee. Compatible historical caches are recropped/recalibrated on these points without overwriting old tables. A cached method with no compatible new CAL bank contributes ranking only; its old threshold is not substituted. Query labels remain sealed until the entire fixed set of new models and scores is complete.

The primary contrasts are A−B separately before and after R, and each arm's R−noU. AP, ROC-AUC, fixed-implementation VUS-PR/VUS-ROC, strict point false alarms/recall, one-to-one events, any-hit, three-point coverage and long-event macro/pooled coverage remain separate outcomes. Normal SELECT improvement does not replace anomaly evaluation. Native LEFT+R, MLP, CANDI, complete TimeRadar and simple controls retain their actual training, statistics, context and compute differences. A single seed on two exposed development tasks does not establish replication, unseen-domain generalization or a directed-lag contribution.

# Historical fixed-geometry results and technical decision

The entire eight-model training set and16 CAL/QUERY score banks were frozen before QUERY-label evaluation. Table1 reports each geometry arm before and after source replay; all rates below are percentages except AP/ROC/VUS. Full external methods, strict/C3 event precision/recall/F1, false-alarm fragments, availability offsets, and all paired deltas are in the [complete result tables](../../docs/experiments/tf_conditional_energy_20261119/RESULTS.md) and machine-readable CSVs. SWaT contains22 complete eligible events including9 long events; PSM contains48 including5 long events on the new common axis.

| Domain | Arm/state | AP | ROC-AUC | VUS-PR | VUS-ROC | Strict normal FPR% | Strict recall% | C3 normal FPR% | C3 matched events | C3 complete-event coverage% | C3 long macro% | C3 long pooled% | Longest C3 miss(points) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | A_noU | 0.6513 | 0.8311 | 0.6539 | 0.8326 | 1.4192 | 66.1071 | 0.8917 | 14/22 | 63.8676 | 9.9598 | 78.0624 | 3931 |
| SWAT | B_noU | 0.6619 | 0.8324 | 0.6640 | 0.8346 | 31.2462 | 76.5724 | 16.8674 | 17/22 | 74.7933 | 21.3215 | 85.3775 | 2036 |
| SWAT | A_R | 0.3308 | 0.7896 | 0.3387 | 0.7970 | 0.7812 | 1.2684 | 0.1688 | 13/22 | 1.0486 | 0.1809 | 0.1238 | 31850 |
| SWAT | B_R | 0.4095 | 0.8138 | 0.4201 | 0.8211 | 1.1028 | 1.3668 | 0.1456 | 11/22 | 1.0402 | 0.0257 | 0.1161 | 31849 |
| PSM | A_noU | 0.3574 | 0.5777 | 0.3922 | 0.6200 | 2.3632 | 7.5492 | 0.1057 | 17/48 | 1.1749 | 0.4997 | 0.3902 | 3137 |
| PSM | B_noU | 0.3572 | 0.5775 | 0.3920 | 0.6199 | 2.3759 | 7.5435 | 0.1057 | 17/48 | 1.1749 | 0.4997 | 0.3902 | 3137 |
| PSM | A_R | 0.3641 | 0.5931 | 0.3974 | 0.6343 | 2.0038 | 6.5561 | 0.0613 | 14/48 | 0.8528 | 0.3317 | 0.2276 | 3693 |
| PSM | B_R | 0.3698 | 0.6023 | 0.4029 | 0.6430 | 2.0208 | 6.6077 | 0.0613 | 14/48 | 0.9096 | 0.3455 | 0.2764 | 3693 |

The primary A−B AP differences are−1.0594pp/noU and−7.8658pp/R in SWaT, and+0.0165pp/noU and−0.5707pp/R in PSM. The response geometry therefore does not establish independent ranking benefits that survive adaptation. Retain the real, bounded SWaT noU trade-off: strict false alarms fall31.2462%→1.4192% and C3 false alarms16.8674%→.8917%, with C3 complete-event coverage falling74.7933%→63.8676% and matched events17→14. These are model-specific mixed-CAL operation points, not equal-FPR comparisons or certified normal false-alarm budgets. A's9.9598% long macro coverage is much lower than its78.0624% pooled coverage; it does not establish reliable detection across long events.

Mixed adaptation does not preserve that sustained evidence. In SWaT A, C3 complete-event coverage falls63.8676%→1.0486%, long macro9.9598%→.1809% and pooled78.0624%→.1238%. Although A_R matches13 events versus B_R11, its event F1 is lower and C3 false alarms slightly higher; both have near-zero steady coverage. PSM replay improves the ranking of both arms modestly, but the raw-only arm improves more and event coverage declines. PSM's all-attack CAL is not cleaned or moved to rescue this operation point.

The full external comparison does not justify attributing SWaT conditional-model performance to TimeRadar geometry. SWaT A_noU AP.651273 exceeds LEFT_noU.607560 but is below its own matched B_noU.661866, and has lower ROC; on PSM A_noU.357405 is below native TimeRadar.430540 (offline ranking-only), LEFT_noU.586600 and MLP_noU.606531. Context, source scaling, pretraining priors, losses and exposure differ, so none of these cross-model gaps isolate a causal mechanism. Newly recalibrated LEFT operating results here must not be substituted into the earlier replay tables with different CAL points.

The SWaT degradation includes threshold-free ranking losses; CAL alone cannot explain it. This experiment does not distinguish abnormal history becoming predictable from other conditional-distribution learning changes, nor does R−noU isolate replay from additional updates without a matched TT arm. A proper conditional scoring rule and nonzero response gradients are not semantic anomaly guarantees. No new causal, lag-specific, multi-source or pollution-robust claim follows.

Eight training jobs completed successfully; one unstarted readout guard rejection was recovered without repeating successful work. Sixty-four unique readout shards merged into16 banks. Formal scale/train/readout occupancy was21.395340154 A40 GPUh; a5.841920s post-seal CUDA point check gives21.396962910 known owned GPUh. Initial profile import/exit overhead was incompletely instrumented and remains a separate timing limitation. The16 fixed CUDA points replayed exactly; official JMLR half-score maximum difference was1.49e-8. Role-separated same-family review independently recomputed45,070 fields/48 rows/2,498 event rows (max difference2.22e-16); VUS shared the pinned official implementation. This is not external double-blind review or independent retraining.

**Decision:** retain all arrays, matched states and the limited noU operating trade-off, but do not promote or expand this response-geometry candidate as the paper method. The public-data/two-stage research objective is unchanged. No extra seed, new loss, weight, score, threshold, architecture or successor candidate is automatically opened. The locked LEFT/replay and external-method evidence below remains attributed to its original methods.

# Locked external LEFT basis and paired additional-exposure experiment

This completed external comparison uses complete native LEFT with source replay R, its original objective, state updates, inference and per-model calibration. TT is the matched target-only control. Here A denotes each arm's unchanged natural target-FIT draw plan: $R_A=[T,S]$ and $TT_A=[T,T']$; these historical A/B/C exposure conditions differ from the current candidate's A/B geometry arms. Each plan has 2,000 updates with halves of 128 windows, giving 256,000 shared T slots and 512,000 total window exposures. Source replay is an established control, not the claimed invention of Hybrid.

The new $\{TT,R\}\times\{B,C\}$ experiment changes only the shared T half. Before learning, an evaluation-side constructor uses labels within the current authorized U-FIT segment to identify full 192-point abnormal donor windows for B and full 192-point normal donor windows for C. Both conditions replace the same 25,600 T slots: 10% of T exposures and 5% of total exposures. R and TT receive identical replacement locations and corresponding donor draws within each condition. The independent T′ and source S halves remain unchanged. The learner receives numerical windows and the native loss, without labels or a label-dependent update rule. This is label-assisted controlled data construction, not an operational method for discovering unknown contamination.

Direct checks of the frozen U-FIT label arrays find 3,468 all-abnormal and 82,903 all-normal stride-one windows in SWaT, and 649 all-abnormal and 12,852 all-normal windows in PSM. The abnormal windows originate from nine and five qualifying event intervals respectively, not that many independent window events. Donor reuse and achieved exposure fractions are recorded separately. In seed2025, SWaT used3,466 distinct B windows(maximum19 presentations) and21,982 C windows(maximum4); PSM used649 B(maximum63) and11,080 C(maximum10). Repeated seeds use their own fixed draws from the same pools; their complete reuse/exposure counts are retained separately. These repeated presentations do not increase the number of underlying event intervals. A full-window donor criterion does not make the natural background clean: original T may already contain anomalies, and unchanged T′ remains mixed. Equal additional abnormal donor exposure in the shared slots does not imply equal total contamination between TT and R. The registered replacement fraction counts window-exposure slots, not unique windows, independent events or the total anomalous-point fraction.

For each separately reported metric M, define $M_{a,k}$ for arm $a\in\{R,TT\}$ and condition $k\in\{A,B,C\}$. The paired interaction is

$$I_M=(M_{R,B}-M_{R,C})-(M_{TT,B}-M_{TT,C}).$$

We report the absolute outcomes and within-arm B−C, B−A and C−A alongside $I_M$. B−C contrasts abnormal with normal replacement under the same slot plan; C−A reveals effects of replacement and normal-donor composition relative to the natural draw. A positive interaction is favorable for a higher-is-better metric only in the context of the absolute outcomes; false-positive counts and delays have the opposite direction. No rule forces B to degrade, and a favorable interaction alone does not establish good detection, difficult-normal protection or general robustness. Native readout, CAL locations and threshold rules remain fixed; the existing all-attack PSM CAL limitation applies.

## Completed three-seed controlled-exposure evidence

The learning rule in that locked LEFT experiment is unchanged native LEFT evaluated on one complete replay batch, rather than separate target/source forwards or an additional preservation loss. For arm $a$ and exposure condition $k$,

$$\mathcal L_{a,k}(\theta)=\mathcal L_{\mathrm{LEFT}}\!\left(\theta;\operatorname{concat}(T_k,H_a)\right),\qquad H_R=S,\quad H_{TT}=T'.$$

Here $T_A=T$, while $T_B$ or $T_C$ replaces only the frozen shared target slots with the qualified donors. The complete native objective includes its original global terms, prototype and running-state updates. It is not generally interchangeable with two separately evaluated half-batch losses. Native inference is unchanged. The source replay mechanism and LEFT outputs retain their original attribution.

Seeds2025/2026/2027 share the same native initial model within each task. Each seed uses its own accepted A sampling plan, with isolated replacement streams SeedSequence([seed,20261118,10]).spawn(3). Thus these are joint training/intervention-sampling repetitions, not fresh initial pretraining, independent events or independent domains. Exactly16 B/C runs were added; all accepted seed2025 B/C, A, noU and SS results were reused. Every run uses freshAdam1e-4, batch256,2000 effective updates and the fixed endpoint. Source prototypes/scaler/running step were restored, and initial CPU/CUDA RNG matched that arm's same-seed A identity.

| Task | Arm | AP | ROC-AUC | Strict FP | Strict recall,% | C3 complete-event coverage,% | Long steady macro,% | Long steady pooled,% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SWAT | TT-B | 0.579275 +/- 0.001956 | 0.884994 +/- 0.000895 | 5531.3 +/- 61.3 | 15.422 +/- 5.102 | 13.710 +/- 5.179 | 4.061 +/- 0.695 | 15.222 +/- 6.378 |
| SWAT | TT-C | 0.591413 +/- 0.007259 | 0.883241 +/- 0.002352 | 5519.7 +/- 97.7 | 35.666 +/- 15.102 | 34.300 +/- 15.282 | 7.151 +/- 2.197 | 40.600 +/- 18.834 |
| SWAT | R-B | 0.626291 +/- 0.004468 | 0.890994 +/- 0.003508 | 5339.3 +/- 115.6 | 56.806 +/- 0.487 | 56.602 +/- 0.466 | 10.474 +/- 0.131 | 68.063 +/- 0.583 |
| SWAT | R-C | 0.626448 +/- 0.007292 | 0.888027 +/- 0.006709 | 5244.0 +/- 138.2 | 57.834 +/- 1.650 | 57.581 +/- 1.642 | 10.583 +/- 0.247 | 69.292 +/- 2.031 |
| PSM | TT-B | 0.579377 +/- 0.001396 | 0.740176 +/- 0.000340 | 10995.0 +/- 207.0 | 60.405 +/- 1.064 | 78.593 +/- 0.535 | 91.455 +/- 1.201 | 95.084 +/- 0.659 |
| PSM | TT-C | 0.571496 +/- 0.000716 | 0.731608 +/- 0.000425 | 12041.0 +/- 168.9 | 63.924 +/- 0.683 | 81.141 +/- 0.295 | 95.791 +/- 0.479 | 97.550 +/- 0.279 |
| PSM | R-B | 0.593355 +/- 0.000286 | 0.756274 +/- 0.000082 | 5319.0 +/- 348.7 | 41.888 +/- 1.109 | 61.193 +/- 1.008 | 59.814 +/- 1.533 | 81.480 +/- 0.615 |
| PSM | R-C | 0.587934 +/- 0.000728 | 0.747427 +/- 0.000537 | 5126.7 +/- 230.4 | 41.716 +/- 1.029 | 61.149 +/- 0.971 | 58.256 +/- 2.275 | 81.263 +/- 0.846 |

Values are means +/- sample SD($n=3$,ddof1), computed per model/seed on each task's unchanged scoring axis. The complete noU/SS/A/B/C tables, VUS, precision/recall/F1, fragments and offsets are retained in [the fixed replication report](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md) and [its per-seed table](../../docs/experiments/left_replay_exposure_repeat_20261118/PER_SEED_TABLE.md). NoU remains one frozen reference with SD=NA. Labels assist FIT donor construction but are absent from updates; QUERY labels do not select any recipe or threshold. The all-attack PSM CAL remains unchanged and cannot certify a normal false-alarm budget.

| Task | Paired effect | AP,pp | Strict FP,points | C3 complete-event coverage,pp | C3 long macro,pp | C3 long pooled,pp |
|---|---|---:|---:|---:|---:|---:|
| SWAT | RB-TTB | 4.7017 +/- 0.4782 | -192.0 +/- 57.7 | 42.892 +/- 5.521 | 6.412 +/- 0.770 | 52.841 +/- 6.806 |
| SWAT | I | 1.1982 +/- 1.0959 | 83.7 +/- 119.1 | 19.611 +/- 17.473 | 2.980 +/- 2.387 | 24.150 +/- 21.536 |
| PSM | RB-TTB | 1.3978 +/- 0.1155 | -5676.0 +/- 201.2 | -17.399 +/- 0.824 | -31.641 +/- 1.231 | -13.604 +/- 0.458 |
| PSM | I | -0.2460 +/- 0.0572 | 1238.3 +/- 107.2 | 2.593 +/- 0.137 | 5.894 +/- 1.217 | 2.683 +/- 0.480 |

All differences are formed within each seed before averaging. On SWaT, TT B-C AP is negative in all three seeds and I_AP is positive in all three. The C3 coverage interaction is also consistently positive, but ranges+2.987 to+37.824pp, so magnitude is uncertain. Direct R-B versus TT-B yields higher AP, higher coverage and192 +/-57.7 fewer strict false-positive points. The mean positive FP interaction(+83.7 strict points,+75.3 C3 points) is a change-of-response cost; it must not be misreported as higher absolute R-B false alarms than TT-B. All SWaT arms still have15/22 any-hit and15/22 one-to-one matched events: protection does not add discovered events.

SWaT R-B long steady macro coverage remains10.474 +/-0.132%, versus pooled68.063 +/-0.583%, and remains below noU/SS persistence. The seed2025 nine-event illustration shows why: E000022 contributes91.28% of steady points, and four long events have zero steady coverage across the six displayed TT/R A/B/C rows. This is not broad sustained-anomaly detection. [Full-resolution PR/low-FPR ROC explanations](../../docs/experiments/left_replay_exposure_repeat_20261118/curves/CURVE_FINDINGS.md) retain original CAL points; no curve-selected threshold is used. In seed2025, TT-B/C full AUCs are nearly identical, yet their low-FPR rises and original-point recalls differ sharply. This does not identify a unique training or calibration cause. [All324 long-event records](../../docs/experiments/left_replay_exposure_repeat_20261118/all_seed_nine_long_events.csv) retain all nine events across three seeds and strict/C3 streams.

On PSM, B-C AP is positive for both methods in every seed, so a corresponding ranking-specific contamination penalty is not observed. I_AP is consistently negative. Direct R-B still has better AP and fewer false positives than TT-B, while losing strict recall and complete-event coverage. R-B has18 one-to-one matches and18 any-hit events in each seed; TT-B has21 matches and22/22/24 any-hit events. Mean any-hit interaction(+5.333) differs from matched-count interaction(+1.000), and neither establishes stronger absolute R-B event detection. PSM's positive C3 coverage interaction coexists with an adverse+1262 +/-80.6 C3FP interaction and lower absolute coverage.

The accepted natural-mixture R-A versus TT-A ranking gains, controlled extra-exposure protection on SWaT and untested cross-domain/field robustness are distinct. Direct RB-TTB includes the pre-existing natural-adaptation advantage; its full magnitude must not be credited to new contamination protection. The paired B-C interaction is interpreted alongside that absolute comparison and the normal-replacement effect. Background second halves differ and labels are only event-time annotations, not physically affected-channel truth. These results support a bounded replay-retention finding, not a new Hybrid mechanism or contamination robustness in general. The experiment is locked after this replication; no dose, seed, loss or successor variant is added automatically. The first seed's original results remain in Appendix E and its immutable experiment report.

New successful job-wall GPU accounting is1.190615706 hours, plus a preserved0.193526-second prelaunch guard attempt with no CUDA process, for a conservative1.190669464-hour ledger. Only two never-started cases were recovered; none of the14 successful original jobs was rerun. Main CPU evaluation took126.313 seconds. A bounded same-family role-independent review checked new raw scores, event matching, offsets and paired summaries;8,192 comparisons differ by at most2.84e-14 and16,248 CSV comparisons match exactly. This is not external blind peer review.

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

# Complete external adaptation references and matched plain-MLP replay

We evaluated the complete released CANDI MLP, FPM and input/output SANA chain [@candi2026], pinned to commit `28c9679e503832f59e351208cde63657fcb51cad`. Each domain uses seed2025, one native normal-source training and one chronological pass over its existing natural mixed U-FIT, then freezes for CAL and QUERY. LEFT noU, TT-A and R-A are the accepted seed2025 cached results, not selected three-seed winners. A subsequent fixed comparison adds only four plain-MLP TT/R adaptations from the same accepted source models, without SANA or FPM. It reuses the identical192-point exposure endpoints as LEFT but presents each actual10-point suffix to the MLP. Both arms use fresh Adam1e-4 with defaults, batch256,2000 updates and fixed endpoints. This comparison does not change the locked LEFT+R learning rule or constitute a new Hybrid method.

| Domain | Method | AP | ROC-AUC | VUS-PR | VUS-ROC | Strict normal FPR | Strict recall | C3 event F1 | C3 complete-event coverage |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SWAT | LEFT_noU | 0.607560 | 0.842267 | 0.606345 | 0.829634 | 1.758% | 60.960% | 0.082645 | 60.789% |
| SWAT | LEFT_TT_A | 0.610342 | 0.884582 | 0.618301 | 0.881642 | 1.852% | 54.587% | 0.074257 | 54.369% |
| SWAT | LEFT_R_A | 0.629347 | 0.885512 | 0.639265 | 0.884546 | 1.782% | 58.327% | 0.073710 | 57.948% |
| SWAT | MLP_noU | 0.783949 | 0.856207 | 0.543552 | 0.696882 | 96.770% | 97.997% | 0.175439 | 97.980% |
| SWAT | MLP_TT_A | 0.776102 | 0.845163 | 0.514483 | 0.665694 | 96.220% | 98.014% | 0.185185 | 97.997% |
| SWAT | MLP_R_A | 0.782007 | 0.863262 | 0.541789 | 0.702566 | 96.132% | 98.018% | 0.119048 | 98.001% |
| SWAT | CANDI_A | 0.783885 | 0.856961 | 0.541710 | 0.695700 | 97.383% | 98.003% | 0.125000 | 97.984% |
| PSM | LEFT_noU | 0.583688 | 0.771345 | 0.575447 | 0.775647 | 15.602% | 43.613% | 0.104046 | 65.710% |
| PSM | LEFT_TT_A | 0.572727 | 0.733239 | 0.562617 | 0.740466 | 25.378% | 63.665% | 0.140468 | 80.955% |
| PSM | LEFT_R_A | 0.587891 | 0.748194 | 0.586017 | 0.761687 | 11.190% | 42.314% | 0.089474 | 61.901% |
| PSM | MLP_noU | 0.604606 | 0.810453 | 0.625606 | 0.819068 | 4.251% | 31.869% | 0.235294 | 49.062% |
| PSM | MLP_TT_A | 0.491079 | 0.755309 | 0.537025 | 0.776429 | 1.773% | 6.924% | 0.204082 | 6.462% |
| PSM | MLP_R_A | 0.631848 | 0.819876 | 0.681092 | 0.844464 | 0.898% | 4.348% | 0.302521 | 3.146% |
| PSM | CANDI_A | 0.604606 | 0.810451 | 0.625607 | 0.819065 | 4.253% | 31.869% | 0.235294 | 49.062% |

CANDI's SWaT AP changes0.78394872 to0.78388480 after adaptation; its C3 normal false-positive rate increases96.748% to97.356%. Any-hit remains21/22, while one-to-one matches fall5/22 to4/22 because long alarms can overlap several events without providing distinct detections. High coverage at this operating point is not evidence of useful false-alarm-controlled persistence. On PSM, AP changes0.604605815 to0.604605571, strict false positives increase by one, and C3 metrics are identical. Thus complete FPM/SANA adaptation has not demonstrated a useful incremental effect over its own plain source backbone in this fixed protocol.

PSM CANDI has higher ranking and fewer false alarms than LEFT R-A, with20/48 versus17/48 one-to-one C3 matches. However, complete-event coverage is49.062% versus61.901%; long-event steady macro coverage is23.761% versus59.734%, and the longest miss is1000 versus399 points. Its cross-model advantages largely predate adaptation. SWaT also retains higher CANDI AP but lower ROC/VUS and severe false-alarm cost. Neither backbone nor filtering is credited causally from these cross-model differences.

Permissions are explicit. CANDI uses a source-FIT-only scaler, source-SELECT checkpoint selection and the complete source-SELECT FPM pool. LEFT retains its original scaler including source validation. CANDI maps a real10-point sliding-window MSE to the current endpoint; LEFT uses192-point nonoverlapping full-window retrospective scores with availability at the window end. The common axes contain336321 SWaT and65088 PSM points, but do not equalize context or architecture. SWaT uses the released alpha5 preset; PSM has no dedicated main configuration and uses main defaults without QUERY tuning. The author's continuously adapting test stream is not reproduced by our U-FIT-adapt/QUERY-freeze protocol. SWaT's nonzero initial SANA gate also confounds insertion with subsequent updates when comparing against plain noU.

Each model uses its own q99/higher/strict-greater threshold on the same512 mixed-CAL points. SWaT contains38 labeled attack points there; all512 PSM CAL points are attack-labeled. These fixed empirical operating points do not certify a1% normal false-alarm budget. Internal FPM source thresholds remain separate. AP is not trapezoidal PR-AUC; VUS retains the pinned author implementation with windowSize100/thre250.

Actual CANDI acceptance is4005 hard and54513 moderate windows on SWaT, with1720 SGD steps; PSM accepts48 hard and6727 moderate windows, with48 steps and three pending hard windows left unflushed. Both source models train once for30 epochs and select epoch30 using normal validation only. The complete episode consumes0.1235334053 A40 hours, including two retained engineering failures and two recovery jobs; main CPU evaluation takes43.844 seconds. No source training or PSM adaptation is repeated. Selected-window anomaly counts and complete failed-phase training peaks were not archived and remain unavailable, rather than inferred. Readout-only peaks are not reported as training peaks.

A same-family role-independent bounded review checks1368 numerical/identity/contrast entries (maximum difference4.44e-16),1580 exact CSV entries, official source identities, scalers rebuilt from source FIT, calibration, events and recovery provenance. VUS shares the pinned official dependency; old LEFT VUS is reused. This is not external blind review. [The complete method table and implementation record](../../docs/experiments/candi_native_a_20261118/RESULTS.md) retains strict/C3 precision and recall, matching versus any-hit, long macro/pooled coverage, misses, offsets, costs and all limitations. CANDI remains an external reference; this result neither overturns the locked replay evidence nor supplies a Hybrid original contribution. No additional CANDI seed, B/C matrix or successor loss is started.

## What the matched backbone comparison establishes

| Domain | Backbone | R−TT AP (percentage points) | R−noU AP (percentage points) | R−TT C3 normal FP | R−TT C3 complete-event coverage (percentage points) |
|:--|:--|--:|--:|--:|--:|
| SWAT | LEFT | +1.900460 | +2.178684 | -168 | +3.579128 |
| SWAT | MLP | +0.590511 | -0.194179 | -305 | +0.004186 |
| PSM | LEFT | +1.516445 | +0.420339 | -6810 | -19.054387 |
| PSM | MLP | +14.076916 | +2.724260 | -300 | -3.316278 |

Plain MLP plus replay does not dominate the complete LEFT-plus-replay configuration. On SWaT, MLP R has higher AP but96.090% C3 normal false positives and5/22 one-to-one matches; LEFT R has1.512% and15/22, with57.948% complete-event coverage. On PSM, MLP R has higher AP/ROC/VUS and fewer false alarms, yet C3 complete-event coverage is3.146% versus61.901%, long-steady macro coverage1.380% versus59.734%, and the longest miss2025 versus399 points. These are observed full-configuration trade-offs at each model's own fixed mixed-CAL threshold, not equal-FPR or isolated representation comparisons. PSM's all-attack CAL especially limits normal-FPR interpretation.

Within the PSM MLP, replay raises AP from0.491079(TT) to0.631848 and reduces C3 false-positive points659 to359. However, matching falls20 to18/48 and coverage6.462% to3.146%. Event F1 rises because substantially fewer alarm fragments are produced, not because more events are found. Relative to its own noU, R loses coverage49.062% to3.146% despite higher ranking. On SWaT, the small R−TT AP gain coexists with unusably high normal alarms and no added distinct events. Neither fewer alarms nor higher AP alone establishes better adaptation.

The positive R−TT ranking contrast in both architectures supports source replay as a general learning control, rather than an exclusive property of time–frequency representations. It does not identify anomaly-specific protection in MLP, where no B/C experiment was performed. LEFT retains configuration-level value for the observed industrial operating point and sustained coverage, at higher cost; this cannot be attributed specifically to its frequency mechanism because context, scaling and native objectives differ. Mature LEFT and replay are not renamed as Hybrid originality.

MLP parameters are140584/94624 on SWaT/PSM, versus3742202/4519985 for the actual LEFT configurations. New MLP R training/readout takes9.863/3.559 seconds on SWaT and9.193/0.648 on PSM. Owned job walls are21.240/15.120 seconds, versus historical LEFT R274.168/247.965 seconds including startup/readout. These are recorded task costs, not context-normalized microbenchmarks. All four new jobs exit0 without retries or source retraining, consuming0.0204699198 A40 GPU hours; main CPU evaluation takes26.953 seconds. All12 MLP parameter tensors update, with the source weights/scaler/config retained and readout weights/RNG/optimizer frozen.

The full cached PR/ROC and0--5% FPR curves retain every original CAL point, including explicit out-of-panel points. They explain ranking versus the chosen empirical operation; no curve is used to replace thresholds. One same-family role-independent raw-result review passes2664 numerical comparisons (maximum2.22e-16) and2840 exact CSV comparisons, including complete endpoint plans, identities, raw metrics and events. The official VUS dependency is shared; old20 rows/VUS are reused. [Complete tables, curves, costs and limitations](../../docs/experiments/mlp_replay_20261119/RESULTS.md) retain all outcomes.

The fixed decision is to retain LEFT+R as the current sustained-coverage-oriented working basis and MLP noU/TT/R as mandatory low-cost strong controls. No universal winner or original Hybrid mechanism is established by this comparison. The baseline-backbone and ordinary-replay comparison now closes without added seeds, contamination ratios, losses, thresholds or an automatically generated successor. The methods-paper objective and the locked two-task three-seed replay evidence remain unchanged.

# Historical stopped P/V source-output preservation experiment

The completed four-run experiment added functional preservation to native R while leaving native inference unchanged. It is stopped; neither P nor V is included in the current baseline or replacement experiment. A frozen copy of each domain's native initial model reads source FIT windows S only. In the same input-coordinate units, let its three $B\times192\times C$ reconstructions be $g_v(\theta_0;S)$ for `X_hat`, `X_hat_from_S` and `X_hat_ms`, and let the student return $g_v(\theta;S)$. They are outputs of shared, interacting pathways, not independent views. Writing $D=B\cdot192\cdot C$, $\Delta_v=g_v(\theta;S)-\operatorname{stopgrad}g_v(\theta_0;S)$ and $\overline\Delta=\frac13\sum_v\Delta_v$, we compare

$$P=\frac{\|\overline\Delta\|_F^2}{D},\qquad V=\frac{1}{3D}\sum_{v=1}^{3}\|\Delta_v\|_F^2,\qquad \mathcal L=\mathcal L_{\mathrm{native},R}+\lambda\mathcal L_{\mathrm{pres}},\quad\lambda=1,$$

where $\mathcal L_{\mathrm{pres}}$ is P or V. The identity

$$V-P=\frac{1}{3D}\sum_{v=1}^{3}\|\Delta_v-\overline\Delta\|_F^2\geq0$$

shows that V additionally penalizes dispersion between pathway drifts, whereas P permits their cancellation. V is pointwise at least as strong at the same coefficient; an observed difference cannot be attributed solely to separating views under equal regularization strength. Nor does this algebra imply any downstream detection improvement.

The added student forward is in eval behavior with gradients retained; the frozen teacher uses `no_grad`. Auxiliary execution restores buffers, the running step, heterogeneous per-module modes and CPU/CUDA RNG state, so these passes do not silently advance native replay state or its random trajectory. Teacher/auxiliary compute is an added training cost, even when update counts match R. Inference uses the original native model score, readout and calibration rule; neither teacher outputs nor preservation errors become an anomaly-score mixture. Full definitions and the state-restoration contract appear in [the historical P/V definition](method_definition.md).

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

# Appendix A: stopped density candidate and original evaluation setup

This appendix retains the earlier mathematical definitions and experimental setup for historical traceability. They do not define the current LEFT-plus-replay baseline or its paired exposure experiment. References to the candidate in this appendix refer only to that stopped design.

## Normality learning, invertible models, and anomaly criteria

Reconstruction, prediction, prototype distance, and likelihood make different assumptions about normality. Their training losses can emphasize different features from those that control event detection. Real NVP introduced tractable density estimation through invertible affine transformations with exact log-determinants [@realnvp2017]. Wavelet Flow already combines wavelets with normalizing flows and models signal scales explicitly [@waveletflow2020]. Hybrid reuses these established principles; a Haar transform inside a flow is not its novelty. GANF also uses joint density estimation for time-series anomaly detection through a graph-factorized flow [@ganf2022], and MTGFlow studies multivariate detection with dynamic graphs and entity-aware normalizing flows [@mtgflow2023]. Thus neither window likelihood nor graph/dependency modeling is unprecedented.

The stopped candidate tested a particular alternating-coordinate joint normality model and its sustained-anomaly behavior under fixed information permissions. An invertible representation does not discard dimensions through an encoder bottleneck, but this property alone does not guarantee that the learned density orders anomalies correctly. The identity-transform Student-t control and the time-only and local-context ablations test whether learned dependencies and multiscale organization are useful, rather than attributing improvement to a generic heavy-tailed score or a change in score units. A publication-level distinction from existing flow methods would require such empirical evidence and an appropriate direct comparison; architectural resemblance alone is not treated as a contribution.

## Historical formulation: stopped joint-density candidate

Channelwise location and scale $(\mu_c,\sigma_c)$ are estimated using legal FIT data only. With a fixed positive numerical floor on scale, the model receives $\widetilde X_{t,c}=(X_{t,c}-\mu_c)/\sigma_c$. This operation preserves within-window level relative to FIT; it does not subtract each window's own mean. Let $f_\theta:\mathbb R^{H\times C}\rightarrow\mathbb R^{H\times C}$ be invertible, and write $z_t=f_\theta(\widetilde X_t)$. With a factorized Student-t prior of fixed degrees of freedom $\nu=5$,

$$p_0(z)=\prod_{i=1}^{HC} \operatorname{StudentT}_{5}(z_i),$$

the window negative log-likelihood in standardized coordinates is

$$\ell_\theta(\widetilde X_t)=-\log p_0(f_\theta(\widetilde X_t))-\log\left|\det J_{f_\theta}(\widetilde X_t)\right|.$$

The fixed per-domain standardization contributes a constant density offset in physical coordinates. Its omission from standardized-space scores does not change within-domain ranking or an identically calibrated threshold, but raw likelihood values from separately standardized domains are not assumed comparable. The training objective and endpoint score are

$$\mathcal L_{\mathrm{FIT}}(\theta)=\frac{1}{|\mathcal T_{\mathrm{FIT}}|}\sum_{t\in\mathcal T_{\mathrm{FIT}}}\frac{\ell_\theta(\widetilde X_t)}{HC},\qquad s_t=\frac{\ell_\theta(\widetilde X_t)}{HC}.$$

The model is selected by the same normal SELECT likelihood, without QUERY labels. No reconstruction, prediction, reference-retrieval, or temporal accumulation score is added to $s_t$.

The historical implementation design uses four affine coupling blocks, alternating time-domain coordinates and a two-level orthogonal Haar transform. A coupling keeps one partition unchanged and applies an invertible affine map to its complement:

$$u_a=v_a,\qquad u_b=v_b\odot\exp a_\theta(v_a)+b_\theta(v_a),$$

whose log-determinant is the sum of the log-scales $a_\theta$. Alternating checkerboard partitions expose different time–channel coordinates across blocks. The scale and translation functions use dilated temporal convolutions and cross-channel mixing. An orthogonal transform has unit absolute determinant, so it does not create an unaccounted likelihood term. The implemented two-level Haar transform requires $H$ divisible by four and uses exact adjacent-pair analysis and synthesis without padding. This finite observed-window map can mix coordinates within the available window; it is not a claim that every latent coordinate has a directed physical causal interpretation. Coordinatewise terms used to sum the prior and Jacobian are diagnostic allocations; after cross-channel transformations they are not per-channel marginal or conditional densities. Only the complete sum over all $H C$ coordinates defines the stated joint likelihood.

This construction differs from a fixed normal center because normality is learned as a transformed distribution rather than distance to a single reference. It differs from nearest-neighbor retrieval because no QUERY-dependent memory search defines the criterion. It differs from the stopped shared prediction–reconstruction design because the training and detection quantities are the same likelihood. Nevertheless, whole-window averaging may dilute a weak local departure or delay a warning, and maximum likelihood may allocate probability poorly under a new legitimate operating state. These are explicit falsifiable limitations, not effects removed by the architecture definition.

## Training regimes and future adaptation questions

We distinguish source training (Stage 1), source-calibrated zero-shot detection (Z-S), target-calibrated detection without parameter updates (Z-T), unlabeled target adaptation (U), and few-event supervision (F). Target CAL may define statistics and thresholds only within its authorized role; it never supplies gradients. Target FIT supplies U updates without anomaly labels. F supervision is counted in complete independent support events, with support, validation, and query events separated. A change from Z-S to target-trained learning changes the scientific question and is reported as such.

This first evaluation establishes only the candidate's basic detection capability under its stated training permissions. It does not establish multisource transfer, a contamination-robust U mechanism, or the value of few-event labels. Those questions remain in the research framework and require separate matched tests once the basic criterion is supported.

## Historical experimental setup

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



# Appendix B: closed shallow TimeRadar adaptation branch

## Completed adaptation-branch evidence (17 November 2026)

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

# Appendix E: initial seed2025 controlled-exposure result, retained

**Completed fixed exposure results.** All eight seed2025 jobs finished2000 updates and their original native CAL/QUERY readouts. NoU, SS and both A runs were reused. CAL roles and the all-attack PSM limitation were retained.

| Domain | Arm/condition | AP | ROC-AUC | Strict FP | Strict recall | C3 complete-event coverage | C3 any-hit |
|---|---|---:|---:|---:|---:|---:|---:|
| SWAT | TT-A | 0.610342 | 0.884582 | 5344 | 54.587% | 54.369% | 15/22 |
| SWAT | TT-B | 0.577183 | 0.885519 | 5467 | 10.829% | 8.835% | 15/22 |
| SWAT | TT-C | 0.599753 | 0.885894 | 5499 | 49.358% | 48.186% | 15/22 |
| SWAT | R-A | 0.629347 | 0.885512 | 5143 | 58.327% | 57.948% | 15/22 |
| SWAT | R-B | 0.624099 | 0.886953 | 5234 | 56.914% | 56.697% | 15/22 |
| SWAT | R-C | 0.622314 | 0.880514 | 5095 | 58.614% | 58.225% | 15/22 |
| PSM | TT-A | 0.572727 | 0.733239 | 12006 | 63.665% | 80.955% | 26/48 |
| PSM | TT-B | 0.579550 | 0.740236 | 11019 | 60.341% | 78.596% | 22/48 |
| PSM | TT-C | 0.571787 | 0.731490 | 12061 | 63.918% | 81.173% | 27/48 |
| PSM | R-A | 0.587891 | 0.748194 | 5294 | 42.314% | 61.901% | 17/48 |
| PSM | R-B | 0.593225 | 0.756182 | 5559 | 42.753% | 62.147% | 18/48 |
| PSM | R-C | 0.588581 | 0.748001 | 5341 | 42.680% | 62.128% | 17/48 |

SWaT TT's B-C AP difference is-2.2570 percentage points and its three-confirmed complete-event coverage changes48.186% to8.835%. R's AP changes+0.1785 percentage points, with coverage58.225% to56.697%. Hence I_AP=+2.4355 percentage points and I_C3coverage=+37.8236 percentage points. All four arms discover15/22 events, and R-B has139 more strict normal false-positive points than R-C. R-B remains worse than R-A in AP/coverage and below noU/SS in sustained coverage. These absolute costs qualify the retention effect.

PSM B-C AP is positive for both R(+0.4643 percentage points) and TT(+0.7764), so I_AP=-0.3120 percentage points does not indicate reduced ranking harm. I_C3coverage=+2.5962 percentage points coexists with R-B's lower absolute coverage and event discovery than TT-B(18/48 versus22/48), and an adverse I_C3FP of+1312 points. TT-B/C have21 one-to-one matches each despite22/27 any-hit events; R-B/C have18/17 matches. Thus the any-hit interaction(+6) is not the matched-count interaction(+1). A favorable interaction is not equivalent to stronger absolute detection. Thresholded effects include model-dependent CAL changes under the fixed rule, and PSM CAL does not certify a normal false-alarm budget.

B-C adds exactly4,915,200 event-labeled time-position exposures in each arm(5% of98,304,000 total time exposures), while B-A is smaller because the overwritten natural T slots already contain some positive labels. TT's unchanged second half remains mixed and R's is nominal normal history. Event labels do not identify physically affected channels or exact gradient contribution through native spectral/multiscale losses. The full B-C/B-A/C-A and interaction tables, VUS, one-to-one event scores, long steady pooled/macro coverage, misses, delays, donor audit and costs are retained in [the fixed result report](../../docs/experiments/left_replay_exposure_20261118/RESULTS.md).

This single-seed, exposed-task stress experiment supports a limited SWaT retention effect relative to TT, not a cross-task robustness claim or a new Hybrid mechanism. Eight jobs consumed0.584922 GPU hours and the main CPU evaluation47.672 seconds. Bounded same-family role-separated review independently reproduced1624 main numerical entries and1592 CSV entries, including event matching and offline offsets, with maximum difference0. All four donor banks were independently reconstructed exactly. This does not provide external peer review or B/C seed replication. No dose, seed, loss or follow-up variant is automatically added. LEFT and replay retain their original attribution. The earlier three-seed A results remain learning controls and do not supply replication for B/C.
