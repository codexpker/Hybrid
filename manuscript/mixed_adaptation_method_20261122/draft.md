# Protecting Cross-Channel Time–Frequency Evidence During Mixed-Unlabeled Adaptation

Method draft · 2026-11-22. The proposed mechanism has no experimental results yet. The [source-replay manuscript](../source_replay_submission_20261121/manuscript.md) remains an archived account of completed experiments. Only public multivariate time-series data enter this study.

## 1. Introduction

An anomaly detector may need to adapt after a system changes operating regime. The observations available for this update can contain both ordinary transitions and abnormal intervals, with no labels to distinguish them. A reconstruction objective can improve its fit to recurring patterns in that mixture. When an abnormal pattern recurs, the same objective can also make the pattern easier to reconstruct. The resulting reduction in training loss gives little information about whether the adapted detector still separates unusual behavior from ordinary change.

Time–frequency models supply several ways to measure that separation. TimeRadar learns a fractional transform and uses contextual deviations during pretraining. LEFT jointly models temporal, spectral, and multiscale structure and maps their discrepancies to timestamp-level anomaly scores. These models motivate studying what adaptation does to an existing detector’s evidence. The archived TimeRadar–TimesFM experiments found different responses to some real events, while controlled comparisons did not establish the expected perturbation-specific advantages. That evidence gives little basis for using a combined prediction and reconstruction score. [TimeRadar, method and evaluation](https://arxiv.org/html/2602.19068v1) · [LEFT, method and evaluation](https://arxiv.org/html/2602.08638v1) · [Frozen controlled comparisons](../../docs/experiments/controlled_complementarity_20261121/RESULTS.md).

Our starting observation comes from matched LEFT updates on two exposed public tasks. Replacing the second target half of a batch with nominally normal source windows improved average precision relative to target-only updates in all three paired seeds on both SWaT and PSM. Mean paired improvements were 2.609 and 1.592 percentage points, respectively. Confirmed event-point coverage improved on average in SWaT and fell in PSM. The subsequent reconstruction-response experiments penalized changes in the outputs of a fixed source model. The corresponding aggregate and per-view penalties did not produce a consistent improvement. These findings motivate a distinction between fitting source windows, reproducing their reconstruction outputs, and retaining a score contrast that depends on how channels vary together. [SWaT paired results](../../docs/experiments/left_swat_replay_repeat_20261118/paired_summary.csv) · [PSM paired results](../../docs/experiments/left_psm_replay_20261118/paired_summary.csv) · [Reconstruction-response experiments](../../docs/experiments/left_source_preservation_20261118/RESULTS.md).

We propose **source score-contrast constrained adaptation** on the complete LEFT detector. A four-way intervention on a nominally normal source window balances the perturbation seen by each channel while varying whether two channels are rotated together or in opposite directions within a localized time–frequency subspace. The resulting contrast is calculated from the score used for detection. During a mixed-unlabeled update, we constrain the parameter increment in directions that would reduce this contrast to first order. This leaves other directions available for the original learning objective. The method tests whether preserving a source-supported *joint score response* contributes to target detection beyond source replay or taking smaller optimizer steps. It makes no assumption that the source probes are genuine faults or that a normal system must preserve every phase relationship across operating regimes.

## 2. Closest methods and scope of difference

| Reference | Verified learning and evaluation setting | Relationship to the proposed method |
|---|---|---|
| LEFT | Tri-view fusion, analysis–synthesis and multiscale reconstruction, prototype associations and native per-timestamp scoring. The paper evaluates public benchmark discrimination and component ablations. [Paper](https://arxiv.org/html/2602.08638v1) · [Pinned code](https://github.com/DezhengWang/Left/tree/3fadb4811797075e233076b9faceb0d8ec59b697) | Reuse the complete detector, training loss and score. The proposed update additionally constrains a joint response of the deployed score under four source-only inputs. |
| TimeRadar | Learnable fractional orders, rotation and inverse reconstruction, with CDL based on patch deviations within context. Evaluates pretrained zero-shot and few-shot probing alongside methods trained on target data. [Paper](https://arxiv.org/html/2602.19068v1) | Retain as an external detector. The conditional quantity here is a derivative of LEFT's deployed score under paired source perturbations during adaptation. There is no new fractional transform or context-deviation head. |
| CANDI and M2N2 | CANDI uses normal validation references to curate target examples through FPM and updates input/output SANA components around a frozen detector. Its paper studies sequential test-time adaptation and ablates sample selection and the adapter. M2N2 uses trend normalization and updates on low-score observations attributed to normality. [CANDI paper](https://doi.org/10.1609/aaai.v40i17.38524) · [CANDI methods and ablations](https://arxiv.org/html/2604.01845v1) · [M2N2 paper](https://arxiv.org/html/2312.11976) | Every scheduled source and target window contributes through LEFT's own learning rule. A source contrast constrains the joint increment. The query model is frozen. The archived local CANDI result also uses fit-then-freeze and retains its own sample selection and 10-point input. |
| DACAD and TSAD-C | DACAD uses labeled source data, synthetic anomalies and contrastive/domain alignment objectives with an unlabeled target. TSAD-C learns from contaminated training series using conditional diffusion for decontamination followed by graph-based temporal and spatial modeling. [DACAD paper](https://arxiv.org/html/2404.11269) · [TSAD-C paper](https://proceedings.mlr.press/v286/ho25a.html) | Source probes test the native score without fitting a classifier or inferring individual contaminant identities. TSAD-C stays a literature comparator. Its fixed local implementation had no usable compatible benchmark. [Integration record](../../docs/research_scope.md) |
| A-GEM | Projects updates to avoid increasing average reference-task loss to first order. Continual learning benchmarks study retention with episodic memory. [ICLR paper](https://openreview.net/pdf/68b34388f383ea8919ffc158ee58a351da811add.pdf) | Credit the half-space projection to this family. The technical hypothesis concerns the reference functional: a balanced mixed difference of native time–frequency detection scores instead of reference-task loss. We apply the constraint to the actual Adam proposal, including its preconditioning. |

**Relation to our archived experiments.** Plain replay controls which observations generate the native learning update. It imposes no condition on how their learned scores respond to a relation change. MeanT removes the target-side across-copy variance term from TimeRadar's reconstruction loss while keeping its variance readout. The archived teacher policies penalize reconstruction differences on unchanged source inputs, either after aggregation or per view. The new method has no teacher or target-specific reconstruction term and does not hold any absolute reconstruction fixed. Old reference-center, shared-prediction and score-fusion runs changed reference values, prediction, or readout. Here the measured interaction uses source-only inputs to constrain a parameter increment with the original test-time score. Historical FRFT, long-history and explicit Self/Cross-lag experiments remain separate. This method tests localized cross-channel structure and makes no claim about delayed causal effects or long-history modeling. [MeanT results](../../docs/experiments/timeradar_target_mean_repeat_20261120/RESULTS.md) · [Preservation experiment](../../docs/experiments/left_source_preservation_20261118/RESULTS.md).

## 3. Method

### 3.1 Base detector and task

Let $X\in\mathbb R^{192\times C}$ be a window in the archived LEFT scaler. Let $q_{\theta,b}(X)\in\mathbb R^{192}$ be the channel-mean of LEFT's native point scores. Model parameters are $\theta$. The readout state $b$ includes non-gradient prototype buffers and the native running step. Each task restores the existing source-validation-selected initial checkpoint, scaler and runtime state. Source replay uses only `SOURCE_FIT`. The scaler follows the published local LEFT protocol and includes the released source validation observations. `U-FIT` contains the later mixed target observations. The model receives no point labels from either adaptation or scoring roles. In this first test each task initializes and adapts within its own dataset.

The native training objective is $\mathcal L_{\mathrm{LEFT}}(\theta;[T_j,S_j])$, evaluated once on the original mixed batch of 128 target and 128 source windows. Its cycle, multiscale, consistency, regularization and prototype-update paths remain intact. All existing trainable LEFT parameters are eligible to move. The scaler, window grid and score configuration are fixed.

### 3.2 Source intervention and score contrast

Use the central index set $I=[48,144)$ and write $u=0,\ldots,95$ inside it. Define $w_u=\sin^2(\pi u/95)$, $f_1(u)=\cos(2\pi\,4u/96)$, $f_2(u)=\sin(2\pi\,4u/96)$. For each $f_i$, subtract its $w$-weighted mean before multiplying by $w$. Orthonormalize the resulting two columns in the stated order with a thin QR and positive diagonal of the triangular factor. Extend by zero outside $I$ to form $Q\in\mathbb R^{192\times2}$. Its columns are orthonormal and zero-mean. The intervention acts in this localized two-dimensional subspace, with approximate band localization given by the taper.

Choose one channel pair $c,d$ per task before training. For channel $i$ let $v_i=Q^\top X_{:,i}$. On complete `SOURCE_FIT` windows in a nonoverlapping 192-point grid anchored at the role start, choose the distinct pair with largest absolute normalized pooled inner product

$$
\Gamma_{i j}=\frac{\sum_X v_i^\top v_j}{\sqrt{\sum_X\|v_i\|^2\sum_X\|v_j\|^2}}.
$$

Exclude zero-energy denominators and break ties by the archived channel order. Freeze the pair for every seed. No model scores or query labels enter this choice. An absent nonzero pair is an explicit missing-structure case.

Let $G(\phi)$ be the standard 2-D rotation of angle $\phi=\pi/12$. For each of four signs $(a,e)\in\{+1,-1\}^2$, replace the two columns by

$$
X^{a,e}_{:,i}=X_{:,i}+Q\big[G(s_i\phi)-I_2\big]v_i,
\qquad (s_c,s_d)=(a,e),
$$

leaving all other columns fixed. For either sign, each edited channel keeps its window mean and squared norm, and its perturbation energy is $4\sin^2(\phi/2)\|v_i\|^2$. All edits vanish outside the same core. Common rotations preserve $v_c^\top v_d$. Averaging the inner product over opposite rotations gives $\cos(2\phi)v_c^\top v_d$. These are identities of the projected components. The transformations can still modify the exact marginal histogram or alter relationships with untouched channels. They supply a controlled change of local coordination, with no physical fault label or certified natural regime attached to any constructed window.

Write $A(X)=|I|^{-1}\sum_{u\in I}q_{\theta,b}(X)_u$. For each source window define

$$
d_{\theta,b}(X)=\frac{A(X^{+,-})+A(X^{-,+})-A(X^{+,+})-A(X^{-,-})}{2}.
$$

With the four inputs exactly paired, a common score reduction and a single readout state, the mixed difference cancels every response of the form $F(a)+G(e)+k$. This includes nonlinear responses to either perturbation separately. A nonzero value measures nonadditivity with respect to these two perturbation factors. Joint normalization, prototype association and other model coupling can all contribute. The same-sign and opposite-sign arms are constructed controls with unspecified health status. Their contrast supplies neither a time-lag estimate nor a causal interpretation. Positive $d$ is the prespecified orientation to check on separated source validation blocks. Negative and near-zero values remain in the analysis.

### 3.3 Constrain the actual parameter increment

Use the first eight windows of $S_j$ each step. Let $M_j(\theta;b^+)$ be their mean $d_{\theta,b^+}(X)$, with $b^+$ captured **after** the native training forward. Execute the readout probes as one 32-window eval pass using the same score function as inference, with gradients enabled for the student parameters. The auxiliary pass has no independent optimizer, EMA update or retained RNG change. Compute $g_j=\nabla_\theta M_j(\theta;b^+)$. Adam's native gradient and moments determine the unmodified proposal $p_j$. Keep the moments from this native proposal and project the **actual parameter increment** in Euclidean parameter space:

$$
\delta_j=\arg\min_{\delta}\tfrac12\|\delta-p_j\|_2^2
\quad\text{subject to }g_j^\top\delta\geq 0,
\qquad
\delta_j=\begin{cases}
p_j-\dfrac{g_j^\top p_j}{\|g_j\|^2}g_j, & g_j^\top p_j<0,\\
p_j, & g_j^\top p_j\ge 0.
\end{cases}
$$

When the proposal violates the constraint, the nearest feasible point is on its boundary. Moving along $g_j$ with coefficient $-g_j^\top p_j/\|g_j\|^2$ lands on that boundary and leaves the orthogonal component intact. The second branch includes $g_j=0$. Use float64 reductions for the inner products. Output parameters are $\theta+\delta_j$. The nonnegative inner-product constraint prevents a first-order decrease **on this sampled contrast with $b^+$ fixed**. At a differentiable point the linearization residual is $o(\|\delta_j\|)$, and a locally Lipschitz gradient gives a quadratic remainder. Native score kinks, parameter motion of finite size, different anchors and changes to $b$ can affect the exact result. Keep the native state-update rule and audit score drift due to state separately. For one fixed probe bank with mean contrast $M$, let $b^-$ be the native readout state before the main forward. The following identity separates parameter movement from changes outside gradient control:

$$
M(\theta+\delta_j;b^+)-M(\theta;b^-)
=[M(\theta+\delta_j;b^+)-M(\theta;b^+)]
+[M(\theta;b^+)-M(\theta;b^-)].
$$

At steps 1, 500, 1,000, 1,500 and 2,000, record the terms on every complete nonoverlapping SOURCE_VALIDATION window, on a grid anchored at that role's start. Score in bounded batches. Retain the starting panel at step 0. Diagnostic checkpoints never select a model.

The mechanism allows native fitting to proceed when the proposed direction respects the contrast. It removes only the conflicting component otherwise. This is how adaptation can accommodate ordinary target variation without taking the tested route toward lower source discrimination. The hypothesis requires a source interaction that is stable across the normal source record, informative for at least some target anomalies, and distinguishable from real benign changes. It also requires useful target fitting directions to remain under the constraint. The learner has no labels or calibration feedback to enforce these conditions. New normal regimes that alter the same interaction, anomalies outside the probed subspace and failure of support transfer can defeat the method.

### 3.4 Inference

Score the unchanged input with $q_{\theta_{2000},b_{2000}}$ on the native 192-point full-window grid. Score $q_t$ becomes available at the end of its input window. Probe construction and optimizer constraints have no inference-time path. The method adds zero learned parameters, inference passes or score weights. Apply each frozen model's own q99/higher over the fixed 512 legal CAL points and alarm when $q_t>\tau$. Three-consecutive-point confirmation continues without backfilling. These are offline retrospective results. The archived calibration sets contain 38 attack points out of 512 in this SWaT task and 512 out of 512 in PSM. The PSM thresholds describe an empirical operating point.

### 3.5 Implementation boundary

```text
SOURCE_FIT source pair selection -> fixed two-channel probe specification
Original paired plan -> T, S -> LEFT.forward(train) -> native loss, updated state b+
                          S[:8] -> four source probes -> LEFT native score(eval, grad)
                                                       -> mixed difference M -> g
native loss.backward() -> Adam proposal p -> half-space projection(p, g) -> theta_next
theta_2000, b_2000 + untouched CAL/QUERY -> native LEFT.infer -> native point scores
```

Use [`left_domain_replay.py`](../../src/hybrid_tsad/left_domain_replay.py) for role loading, original batches, optimizer and endpoint/readout. Add one project-side probe and projection module, one versioned config and two optional modes in the shared driver. The diagnostic readout reuses the state/RNG guard in [`left_source_preservation.py`](../../src/hybrid_tsad/left_source_preservation.py), then computes `_ms_anomaly_score` using native `model._forward` outputs 1, 0, 2, 3, 4, 5, 9 and 10 in that order (zero-based), then averages over channels as in the archived readout. Native `infer` is decorated with `no_grad` and remains the inference entry. Preserve upstream LEFT files, existing modes, configs and result snapshots. After the native backward, compute the probe gradient using `autograd.grad` so it does not enter Adam's moments. Snapshot trainable parameters, make the native Adam step and apply the correction to the actual increment with `no_grad`. Drop the native graph before building the auxiliary graph. Record extra training time, peak memory and correction sizes. The analytical workload adds 32 eval windows and one score-gradient pass per 256-window step. This is a 12.5% increment in the number of windows presented in forward passes with a separate backward path. It is not a measured latency estimate.

Necessary checks are energy/sign cancellation with a channel-separable test scorer, same native score along `infer` and the differentiable path, finite-difference check of the projected direction on a legal source batch, exact no-op when the constraint is inactive, and restoration of heterogeneous modes, buffers, clock and RNG. Disabling the hook must recover the native training update. Source probe indexes must never reference U-FIT, CAL or QUERY. Use a joint quadratic score with a nonzero interaction to check that the four-way difference retains the joint response.

## 4. Experiment presentation and prespecified decisions

### 4.1 Data and matched evaluation

| Public task | Native source fit / validation | Mixed target fit | CAL role | Complete score axis and events |
|---|---|---|---|---|
| SWaT, 31 channels | `[0,396000)` / `[396000,495000)` | `[495000,585000)` | `[585544,607479)` | `[608535,944856)`, 336,321 points, 22 complete events |
| PSM, 25 channels | `[0,105984)` / `[105984,132481)` | `[0,17568)` | `[18112,21960)` | `[22656,87744)`, 65,088 points, 48 complete events |

Coordinates, partial boundary events, fixed CAL indexes and event definitions follow the [existing reproducibility record](../source_replay_mlj_20261121/REPRODUCIBILITY.md). PSM uses released-test relative coordinates for target roles. SWaT uses its combined timeline. Both are exposed public development tasks.

**Main public benchmark table.** List the complete proposed method, native LEFT, CANDI, plain MLP, TimeRadar and History Mean where cached results are legally compatible. Show source and target training access, labels used in learning, scaler fit range, context, actual score availability, scoring axis and whether adaptation stops before CAL/QUERY. A row on a different axis or context/permission track stays separately labeled. Trim an archived row to a common axis only where identity and comparable metrics can be established. Keep official paper numbers, externally pretrained TimeRadar tracks, and local results separate. Keep unavailable compatible cells as NA. [CANDI local comparison](../../docs/experiments/candi_native_a_20261118/RESULTS.md) · [MLP adaptation comparison](../../docs/experiments/mlp_replay_20261119/RESULTS.md).

| Main-table row | SWaT AP / ROC-AUC | PSM AP / ROC-AUC | Scoring permissions |
|---|---:|---:|---|
| Proposed complete method | Pending | Pending | Native LEFT 192-point retrospective window. Target-fit updates, then frozen. |
| Native LEFT, frozen source checkpoint | 0.607560 / 0.842267 | 0.583688 / 0.771345 | 192-point retrospective window. Source-validation-selected initial state. |
| Plain MLP, frozen source checkpoint | 0.783949 / 0.856207 | 0.604606 / 0.810453 | 10-point endpoint reconstruction score. Own source-fit-only scaler. |
| Complete CANDI, local fit-then-freeze | 0.783885 / 0.856961 | 0.604606 / 0.810451 | 10-point endpoint score. Normal source validation for mining references. Target-fit updates, then frozen. |

The displayed rows are the archived fixed-axis values. The external operating panel must show the alarm burden alongside ranking. In SWaT, the archived plain MLP and CANDI rows exceed 96% normal FPR after three-point confirmation at their fixed operating points. The TimeRadar PSM cache uses 64,728 points. Preserve the values on that track and report common-axis values only if a legal pointwise intersection is evaluated consistently for all included rows. History Mean is subject to the same identity check. [Full local comparison](../../docs/experiments/mlp_replay_20261119/comparison.csv) · [TimeRadar benchmark](../../docs/experiments/timeradar_native_v_replay_20261120/RESULTS.md).

**Matched mixed adaptation table.** Use the names *Frozen source checkpoint* (`noU`), *Target-only adaptation* (`TT`), *Source-only continuation* (`SS`), *Source replay* (`R`), and *Score-contrast constrained adaptation*. Primary comparison is the proposed method versus ordinary Source replay. Evaluate each dataset and seed individually with paired differences, mean and sample SD with $n=3$. A frozen checkpoint is one reference per task. Treat seeds as repetitions of adaptation from the same initial state. Measure cost for the proposed method and leave archived models in their original runtime accounting.

**Ablation table.** Source replay already disables the constraint completely. Include one separate **step-length control**, which uses the same R data and Adam rule and changes only the proposal length by the candidate's presaved per-step ratio (r_j=\|\delta_j\|/\|p_j\|\), keeping its own Adam direction. Define $r_j=1$ for zero proposals. Generate this trace with source and U-FIT access only, before any query scoring, then lock it. This matches relative attenuation, while actual norms can differ once trajectories diverge. It tests whether the proposed direction matters beyond generic shrinkage. Reuse all compatible noU/TT/SS/R results. Add the candidate and step-length control for the three existing seeds in each domain. No new backbone, weighting sweep or contamination matrix is part of this design.

**Mechanism analysis.** Report $d(X)$ and its four constituent scores on unchanged/constructed source validation windows, separation by four chronological source blocks, fraction of conflicting updates and projection magnitudes. Report the unmodified original-input score alongside common and opposite phase probes. Track finite and state-mediated drift. Query labels can subsequently describe performance on normal points, all complete events and long-event interiors. A score contrast retained only on the probes supports the algebra and implementation, even if downstream discrimination fails. Prior controlled abnormal exposure and its $B-C$ and replay-interaction results belong in supplementary evidence. Those interventions address different windows and cannot by themselves validate this mechanism.

**Claims and metrics.** Primary is unadjusted point AP on each fixed query axis, with the proposed method compared to both Source replay and the step-length control. Seek directionally repeated AP benefits across the three paired seeds on at least one prespecified task and disclose the full other task. A primarily positive result restricted to one task supports a task-bounded claim. ROC-AUC and VUS-PR/ROC are secondary rank metrics. At the fixed operating points report strict and three-point false alarms and recall, one-to-one events and any-hit separately, long-event macro and pooled coverage, longest miss and runtime. Use SWaT's observed normal false alarms and matched/long-event coverage to qualify any AP benefit. A gain with materially heavier alarm burden or collapsed coverage supports, at most, a rank-only claim. PSM's all-attack CAL prevents claims at a validated normal operating budget. No query threshold sweep sets the deployment threshold.

**Selection and stopping.** Lock the base checkpoints, archived stages/scalers, 192-point windows, 2,000 Adam updates at (10^{-4}), final checkpoint, seeds 2025/2026/2027, fixed probe geometry, pair rule and exact projection before scoring. Source FIT determines the pair. Source VALIDATION is used for the already archived initialization and a nonselective read-only check of the probe hypothesis, with quarter boundaries defined by chronological window count. No search over probe variables, angles, frequencies, step count or checkpoints follows that check. Without reproducible positive initialization contrast across source blocks, there is insufficient evidence to describe this as preserving useful source evidence. If the contrast is supported but the candidate cannot beat ordinary replay or matches the step-length control, there is no evidence for the proposed mechanism as a paper contribution. An independent public confirmation task is required for a broader claim and is not yet part of this design.
