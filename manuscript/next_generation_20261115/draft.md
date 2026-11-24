---
title: "Source-Supervised Learning of Native Reconstruction Disagreement for Public Time-Series Anomaly Detection"
author: []
date: "Working method and experiment draft, 24 November 2026"
lang: en
bibliography: references.bib
---

> The current project candidate has task-dependent source-supervised ranking benefits. The fixed target U and single-event F extensions are completed without an established net advantage. This is a public TSAD method-research draft. It does not establish mixed-unlabeled robustness, independent FRFT novelty, weak-anomaly sensitivity or directed lag detection.

# Introduction

  Industrial monitoring requires more than recognizing abrupt, high-amplitude departures from a nominal range. A weak deviation can be significant when it violates a normal response relationship, whereas a large change can be acceptable when several variables respond coherently to an operating transition. An anomaly may also persist after its initial transition: an effective monitor should discover the event and retain useful evidence during its abnormal steady state. These requirements are particularly demanding for multivariate streams in which responses are delayed, operating conditions change, and target-domain training records are unlabeled rather than independently certified as healthy.

TimeRadar supplies an adaptive fractional time–frequency reconstruction detector with a native score based on disagreement among masked reconstructions [@timeradar2026]. LEFT learns interacting temporal, spectral and multiscale representations [@left2026]. Both motivate learning informative anomaly representations, but they also raise a training question. Reconstruction quality on nominal windows and the discrimination of real anomalous events are different objectives. A model can reduce average reconstruction error without improving the ordering of anomalous and normal observations. A separately supervised classification head can also learn source labels without transferring as well as the original detection criterion.

We study source supervision that acts directly on the original reconstruction-disagreement score. The current Hybrid candidate, termed source-supervised native disagreement here, retains TimeRadar's complete pretrained representation, mask construction and inference score, while updating its reconstruction decoder with nominal reconstruction and a real-public-source-event pair loss. The pair loss compares native endpoint variance at abnormal and normal windows. Sampling strata cover each source event's legal endpoint range, and their weights reflect the numbers of available positive endpoints. They are temporal sampling groups, rather than physical fault phases. The construction aims to connect source supervision to the criterion actually deployed. Whether that connection transfers is tested on public target records.

Three comparisons frame the analysis. First, does supervised learning of native disagreement improve on the frozen official detector and reconstruction-only continuation? Second, how does it compare with supervision of mean reconstruction error or a standard MLP on the frozen representation under the same source-label pool? Third, how competitive is the complete detector against target-normal-trained external methods when evaluated on legal common points? The comparisons distinguish detector-level performance from attribution to an individual formula. Their permissions differ and are reported explicitly.

The existing three-seed evidence supports average AP improvements on PSM, MSL and SMAP, alongside NASA absolute-score limitations and SMD ranking losses. Matched source supervision of reconstruction error and a standard classifier provides further information about the learning objective and readout. Fixed U and F extensions test separate uses of the same detector; neither has demonstrated a reliable net advantage. The present method study therefore concerns source-supervised detection and its transfer boundaries. Mixed-unlabeled target learning remains part of the research objective, with its negative extension results retained. All data are public. First-paper laboratory runs, weak labels and enterprise records are excluded.

# Related Work

## Time–frequency representations and reconstruction disagreement

TimeRadar's adaptive fractional transform, reconstruction chain, pretraining and native scoring are reused components [@timeradar2026]. LEFT's full tri-view model and analysis–synthesis scoring remain a named external comparator [@left2026]. CATCH uses frequency patching and channel-aware modeling [@catch2025]. Our source decoder objective changes learning within an existing complete detector. It does not introduce FRFT or establish that the transform itself explains the measured gain. The E-supervised and MLP controls retain the same TimeRadar chain or representation, limiting causal attribution of their performance differences.

## Supervision, target adaptation and information permissions

Real source events can provide discrimination signals that nominal reconstruction alone lacks. In this study source labels define whole abnormal intervals and label-normal comparison windows, not per-sensor fault identities or exact physical causes. The target QUERY labels never enter source training or source checkpoint selection. Comparison with reconstruction continuation includes additional paired-window exposure and a new objective; its gain cannot all be assigned to point weighting.

TSAD-C addresses contaminated multivariate anomaly learning through its complete released framework [@tsadc2025]. CANDI selects adaptation windows and updates input/output adapters around a pretrained reconstruction backbone [@candi2026]. These methods address different learning permissions from our fixed source-supervised detector. Source replay is a mature reference strategy in the archived mixed-adaptation studies. No target-adaptation or contamination guarantee follows from the present source pair loss. An offline complete-window method can be evaluated on the same timestamps as an endpoint-causal method while still using different observations. Our tables retain that distinction.

# Problem Formulation and Data Permissions

A multivariate record is $x_0,\ldots,x_{N-1}$, with $x_t\in\mathbb R^C$. A causal endpoint detector consumes $X_t=x_{t-99:t+1}$ after observation $x_t$ arrives and produces a scalar $s_t$. Its available index is $t$. Each input remains inside one admitted record and role. No missing initial score is backfilled.

SOURCE FIT contains nominal reconstruction windows from SWaT and the 28 SMD normal training records, together with five labeled SWaT FIT events and their normal comparisons. SOURCE SELECT uses disjoint nominal validation blocks and four different labeled SWaT events. The labeled SWaT source interval is [495000,585000), split at 502578. Every positive window's full 100-point footprint lies in one abnormal event; every paired normal footprint lies in its own continuous label-normal source interval. Windows are never formed by concatenating an abnormal block and a normal block.

PSM, MSL and SMAP are excluded from this project's source gradient updates. PSM is an exposed development task; NASA is a fixed transfer evaluation on admitted entity lists. SMD's nominal training records participated in source reconstruction, so its test is a seen-normal-source held-out comparison. The official checkpoint has its authors' upstream pretraining permissions, disclosed separately from project source learning. Target FIT statistics and empirical CAL scores are additional target information, even in the absence of target gradient updates.

The Z setting makes no target parameter updates. SOURCE SELECT chooses its checkpoint. U and F are separate completed extensions. U consumes isolated mixed-unlabeled target windows and source replay. F uses one fixed public support event and 512 labeled-normal points from the target's U-FIT role. Those SUPPORT labels are distinct from QUERY. CAL scores determine the frozen empirical threshold; retrospective CAL label composition explains its limitations. QUERY labels are used only to calculate the fixed test outcomes.

# Method

## Native endpoint evidence and updated parameters

Let $\phi$ denote the frozen official TimeRadar representation, fractional-order network and non-decoder components, and $\theta$ its original reconstruction decoder. For fixed complementary masks $m=1,\ldots,M$, $M=10$, the complete chain returns $\hat X_t^{(m)}\in\mathbb R^{100\times C}$. Every native forward has batch one, uses the fixed isolated mask RNG and runs in evaluation mode. Fractional order is computed independently for each input window with the official within-window variable-sharing rule. Project training updates only $\theta$; official non-decoder parameters and buffers stay fixed.

The mean reconstruction at the endpoint and native disagreement score are

$$\bar x_{t,c}=\frac1M\sum_m\hat X^{(m)}_{t,99,c},\qquad
V_\theta(X_t)=\frac1C\sum_c\frac1{M-1}\sum_m(\hat X^{(m)}_{t,99,c}-\bar x_{t,c})^2.$$

The $M-1$ denominator preserves the original sample-variance convention. The deployed score is $s_t=V_\theta(X_t)$, without a learned post-hoc head, error mixture or temporal accumulation.

## Nominal reconstruction and source-event discrimination

The nominal term retains the original per-copy full-window squared loss,

$$L_{copy}=\frac1{|\mathcal B_N|}\sum_{X\in\mathcal B_N}\frac1{M\,100\,C_X}\sum_{m,j,c}(\hat X^{(m)}_{j,c}-X_{j,c})^2.$$

The batch contains 16 SWaT and 16 SMD windows. SMD entities are sampled uniformly, then a legal nominal window is sampled within that entity. Each window uses its own channel normalization in the loss, allowing records with different channel counts to contribute to the same decoder update.

Split the legal positive endpoints of each FIT event $e$ into three consecutive, nearly equal groups. Let $c_{e,k}$ be the number of legal endpoints in group $k$. Sample $p_{e,k}$ and an independent legal label-normal comparison $n_{e,k}$ from the same source release and role. The scale

$$a=\max(10^{-8},\operatorname{median}_{n\in\mathcal N_{FIT}}V_{\theta_0}(X_n)),\quad
z_\theta(X)=\log(1+V_\theta(X)/a)$$

is fixed at official initialization on the existing 4,632 nominal source endpoints. It is not estimated from a target record. The source pair loss and complete training objective are

$$L_{pair}=\sum_{e,k}\frac{c_{e,k}}{\sum_{i,j}c_{i,j}}\operatorname{softplus}\{1-z_\theta(X_{p_{e,k}})+z_\theta(X_{n_{e,k}})\},\qquad
L=L_{copy}+L_{pair}.$$

Both margin and pair coefficient equal one. Each update uses five FIT events and 15 positive/15 normal pair windows. Point weighting describes the sampling-group masses. It does not claim equal event influence or give point-accurate physical fault onset. Gradients act through the native reconstructions into the decoder. They encourage larger endpoint disagreement on the sampled source positives relative to normal comparisons. Such a source ordering is a learning objective, not a transfer guarantee.

## Training, selection and inference

All source models share the official initialization, seed-specific normal sampling plans and masks. AdamW uses lr1e-4, betas (0.9,0.95), eps1e-8, weight decay0.1 except LayerNorm/bias, and a 24-update warmup. The cap is 240 updates. Checkpoints at steps 0,40,…,240 are evaluated on the source-event SELECT panel. Selection minimizes the mean pair risk with events and their three strata weighted equally, preferring the earliest minimum. The reconstruction-gradient control uses the same labeled SELECT criterion and is disclosed accordingly. It is not a completely unlabeled training-and-selection pipeline.

At deployment, the selected decoder returns ten native reconstructions of the received 100-point window. The score is the unnormalized endpoint $V$. Each task retains its own admitted input scaler and legal scoring axis. Empirical binary decisions use that detector's CAL scores, q99 with the higher quantile convention, and strict exceedance. Three-confirmation begins on the third consecutive crossing without backfill. Ranking and operating outcomes answer separate questions.

```text
Start with official frozen representation and native decoder.
For each source update:
    draw 16 SWaT and 16 SMD nominal windows;
    draw one abnormal and one normal comparison per event/time stratum;
    evaluate every window independently using the fixed ten native masks;
    accumulate the nominal full-window reconstruction gradient;
    add the point-count-weighted native-V pair gradient;
    advance the decoder optimizer once.
Select the earliest minimum source-event validation checkpoint.
For each target endpoint, return its native channel-mean sample variance.
```

## Fixed-weight complete-window offline readout

The independent offline evaluation keeps the selected checkpoint and all input/scoring constants. Entity block $X^{[b]}=x_{100b:100b+100}$ supplies ten reconstructions. For every position $j$, $V^{[b]}_j$ is the channel mean of sample variance across copies, and $E^{[b]}_j$ is the channel mean squared discrepancy of the copy-mean reconstruction. Their actual timestamp is $100b+j$ and available index is $100b+99$. The final incomplete block has no scores. All blocks use one-window native order computation, fixed complementary masks, frozen parameters/buffers and isolated RNG.

For the same input block, position99 is identical to the existing endpoint function. Earlier positions have later observations inside their input block and are offline retrospective outputs. Source pair learning still supervised endpoint99 only. Comparing the new score with rolling causal endpoint scores therefore tests the effect of input context and output position at fixed weights. It does not provide another source-learning gain or causal delay advantage. One reconstruction supplies both V and E without an extra model pass. The MLP supplies only a window-level logit and remains in its existing causal comparison. No window logit is broadcast over its input.

# Experimental Setup

## Public records, axes and aggregation

PSM uses the admitted 25-variable release and complete causal axis [99,87841), 87,742 endpoints. NASA uses all 27 MSL and 53 admitted SMAP entities, with sensor zero only and per-entity [99,N) axes. They contribute 71,056 and 422,370 scored points respectively. The existing SMAP admission rule excluded P-2 before these comparisons. SMD includes all 28 machines and 38 variables, totaling 705,648 scored causal endpoints. Metrics are calculated separately per entity and then averaged equally. Concatenated-point metrics are not substituted for entity macro averages.

The complete-window LEFT comparison intersects each entity's causal axis with its actual scored windows. Its available index is each full window's endpoint. LEFT therefore remains an offline comparator even on common scored timestamps. Full-axis and common-axis tables are separate. No tail scores are padded and no absent scores are assigned zero.

## Complete detectors and direct controls

The full detector comparison includes official TimeRadar V, E and fixed unit-coefficient V+E, source reconstruction continuation, source-supervised V/V, same-label E/E, the standard supervised MLP, and complete LEFT. The fixed V+E row follows the paper formula under the project's ten-copy causal endpoint reduction; the official implementation's frozen L=1 row uses V. CATCH, PCA and historical means enter only compatible common-axis panels. Training labels, target updates, window size and offline context are disclosed alongside the scores.

LEFT uses official revision 3fadb4811797075e233076b9faceb0d8ec59b697. Its NASA models train once per spacecraft domain, seed2025, on entity-local normal windows. Each entity's normal training record is split chronologically 80/20. Domain-level scaler statistics use all official normal training arrays, including the validation part, and no test values. MSL uses window96 and SMAP window192. Training stride is one; validation and test use non-overlapping complete windows. Shared models train for at most 20 epochs, batch128, Adam1e-4, official scheduler and patience3. Entity boundaries and role boundaries are preserved. The MSL configuration's alpha_casd fields are mapped explicitly to the alpha_ms fields consumed by the model using the original values. This is a project compatibility implementation. When normal training, validation or standardization overlaps project CAL information, LEFT's independent-CAL operating row remains NA.

The standard MLP freezes all official components and maps each complex copy/channel representation through 10240→64→1 with GELU, then averages logits. The same source labels and nominal panels train its window-level BCE. Rates1e-4 and1e-3 repeat the sealed source plan at most ten times. Full-range and ≤240-step source selection both chose lr1e-3 step40. Neither choice reads target metrics. The complete search and the selected checkpoint have distinct exposure and compute costs.

## Metrics, repeated training and operation

AP is average precision, rather than trapezoidal PR area. ROC-AUC and the two VUS metrics retain the archived implementations. VUS uses the pinned LEFT evaluator with windowSize100 and thre250 across all methods. The three source-training seeds share one official initial checkpoint; their paired means and sample SD use n=3 and ddof=1. Entity variation is reported separately from training-seed variation. LEFT, E/E and MLP have one project seed and no invented SD.

Strict and three-confirmation operation includes normal false-alarm point rate and fragments, abnormal recall, one-to-one event matches, any-hit, hit-only delay, complete-event coverage, and long steady coverage after the first 512 event points. Long-event macro and pooled coverages are both reported. MSL and SMAP have only four and ten eligible long events. PSM operation uses its original 64,728-point suffix, with all 512 CAL points labeled abnormal. NASA CAL is small and q99/higher equals its maximum. These thresholds do not certify normal 1% FPR. The SMD nominal-normal CAL and its actual normal test behavior are disclosed separately.

# Results

## Fixed-weight complete-window offline comparison

The following panel uses the eight previously selected checkpoints with W100 non-overlapping input blocks. Every position retains native V; original E and the pre-specified unit-coefficient combination share the same reconstruction. Neither weights nor thresholds are selected from this comparison. The MLP remains a window-level causal control and is not broadcast.


All methods in this table use identical scored points within each entity: the intersection of W100 complete-window outputs, archived causal endpoints and native LEFT coverage. PSM begins at99; SMD and NASA also begin at99 in each independent record. The last point is min(100 floor(N/100), W_LEFT floor(N/W_LEFT))−1. LEFT W192 applies to PSM, SMD and SMAP; MSL uses W96. Every TimeRadar full-window score is available at the containing W100 endpoint. Complete-window output is a project readout of fixed checkpoints. The three-seed mean and sample SD concern project source training at a common official initialization (n=3, ddof=1). Single frozen/E/LEFT models have no invented SD.

| Domain | Method | Project seeds | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | Official TimeRadar V | 1 | 0.44824 | 0.67842 | 0.45090 | 0.68182 |
| PSM | Official E | 1 | 0.43590 | 0.60779 | 0.43846 | 0.61526 |
| PSM | Official fixed V+E | 1 | 0.44836 | 0.61891 | 0.44821 | 0.62409 |
| PSM | Reconstruction continuation / V | 3 | 0.45283 ± 0.00291 | 0.68729 ± 0.00337 | 0.44955 ± 0.00120 | 0.67675 ± 0.00157 |
| PSM | Source-supervised native V | 3 | 0.44922 ± 0.00150 | 0.69227 ± 0.00085 | 0.45659 ± 0.00067 | 0.68701 ± 0.00057 |
| PSM | Source-supervised E / E | 1 | 0.38952 | 0.57682 | 0.40739 | 0.60085 |
| PSM | LEFT | 1 | 0.57690 | 0.77618 | 0.57413 | 0.78579 |
| SMD | Official TimeRadar V | 1 | 0.42716 | 0.82717 | 0.45194 | 0.88232 |
| SMD | Official E | 1 | 0.42720 | 0.81995 | 0.45557 | 0.86840 |
| SMD | Official fixed V+E | 1 | 0.43725 | 0.82679 | 0.46495 | 0.87375 |
| SMD | Reconstruction continuation / V | 3 | 0.44041 ± 0.00487 | 0.83733 ± 0.00368 | 0.46505 ± 0.00622 | 0.88696 ± 0.00200 |
| SMD | Source-supervised native V | 3 | 0.43614 ± 0.00139 | 0.84836 ± 0.00068 | 0.46682 ± 0.00071 | 0.89499 ± 0.00049 |
| SMD | Source-supervised E / E | 1 | 0.42158 | 0.83382 | 0.47466 | 0.88102 |
| SMD | LEFT | 1 | 0.49809 | 0.87855 | 0.54184 | 0.91853 |
| MSL | Official TimeRadar V | 1 | 0.34850 | 0.66935 | 0.38184 | 0.71798 |
| MSL | Official E | 1 | 0.39988 | 0.71093 | 0.43881 | 0.76245 |
| MSL | Official fixed V+E | 1 | 0.40407 | 0.71697 | 0.44381 | 0.76558 |
| MSL | Reconstruction continuation / V | 3 | 0.36811 ± 0.00294 | 0.67549 ± 0.00174 | 0.40369 ± 0.00295 | 0.72209 ± 0.00163 |
| MSL | Source-supervised native V | 3 | 0.35562 ± 0.00159 | 0.68606 ± 0.00074 | 0.39242 ± 0.00114 | 0.73144 ± 0.00022 |
| MSL | Source-supervised E / E | 1 | 0.39125 | 0.72008 | 0.43678 | 0.77492 |
| MSL | LEFT | 1 | 0.50834 | 0.76334 | 0.55308 | 0.79416 |
| SMAP | Official TimeRadar V | 1 | 0.30165 | 0.66240 | 0.29931 | 0.70426 |
| SMAP | Official E | 1 | 0.38124 | 0.71881 | 0.38102 | 0.76553 |
| SMAP | Official fixed V+E | 1 | 0.38662 | 0.72699 | 0.38210 | 0.77097 |
| SMAP | Reconstruction continuation / V | 3 | 0.32903 ± 0.01050 | 0.67517 ± 0.00245 | 0.32412 ± 0.00975 | 0.71470 ± 0.00187 |
| SMAP | Source-supervised native V | 3 | 0.33422 ± 0.00185 | 0.67061 ± 0.00081 | 0.33332 ± 0.00294 | 0.71166 ± 0.00071 |
| SMAP | Source-supervised E / E | 1 | 0.38984 | 0.71764 | 0.39124 | 0.76577 |
| SMAP | LEFT | 1 | 0.49315 | 0.76287 | 0.52172 | 0.78494 |

Official TimeRadar uses no project parameter updates. Reconstruction continuation is selected by real source-event validation but receives only nominal reconstruction gradients. The native-V/E supervised decoders use nominal source reconstruction plus real SWaT source pairs. LEFT uses the full target-normal records with its archived scaler/validation permissions and offline context. Current TimeRadar input scalers and mask identities remain fixed. NASA uses sensor zero only. SMD normal training records were among the project sources. MLP remains in its prior causal panel; no window logit was broadcast. No CAL, thresholds or online latency are evaluated in this readout experiment.

[Seed-specific macro values](macro.csv) · [complete per-entity rankings](entity_metrics.csv) · [same-axis repetitions](repeats_common.csv).


## Matched-weight endpoint and complete-window comparison


Every difference uses the identical per-entity common axis defined in Table1. Full-window minus endpoint measures a readout/context change at fixed parameters. It is separate from differences between training objectives. No result is selected by target performance.

## Same weights, different readout

| Domain | Weights / score | Endpoint AP | Complete-window AP | Δ AP | Δ ROC_AUC | Δ VUS_PR | Δ VUS_ROC |
|---|---|---|---|---|---|---|---|
| PSM | Reconstruction continuation | 0.53884 ± 0.00209 | 0.45283 ± 0.00291 | -0.08601 ± 0.00171 | -0.04444 ± 0.00245 | -0.04059 ± 0.00536 | 0.01032 ± 0.00324 |
| PSM | Source-supervised native V | 0.57306 ± 0.00469 | 0.44922 ± 0.00150 | -0.12384 ± 0.00380 | -0.07172 ± 0.00253 | -0.05177 ± 0.00460 | 0.01130 ± 0.00406 |
| PSM | Official V | 0.53528 | 0.44824 | -0.08704 | -0.05257 | -0.03269 | 0.01705 |
| PSM | Official E | 0.38734 | 0.43590 | 0.04856 | 0.03812 | 0.03999 | 0.03391 |
| PSM | Official V+E | 0.41947 | 0.44836 | 0.02889 | 0.01549 | 0.02454 | 0.01673 |
| PSM | E-supervised E | 0.52232 | 0.38952 | -0.13281 | -0.09783 | -0.09392 | -0.06940 |
| SMD | Reconstruction continuation | 0.44968 ± 0.00260 | 0.44041 ± 0.00487 | -0.00927 ± 0.00236 | 0.01859 ± 0.00043 | -0.00983 ± 0.00215 | 0.01287 ± 0.00077 |
| SMD | Source-supervised native V | 0.44023 ± 0.00255 | 0.43614 ± 0.00139 | -0.00409 ± 0.00344 | 0.02228 ± 0.00077 | 0.03680 ± 0.00485 | 0.03516 ± 0.00124 |
| SMD | Official V | 0.44480 | 0.42716 | -0.01764 | 0.01396 | -0.01526 | 0.01236 |
| SMD | Official E | 0.42914 | 0.42720 | -0.00194 | 0.01463 | 0.02194 | 0.01966 |
| SMD | Official V+E | 0.43885 | 0.43725 | -0.00159 | 0.01473 | 0.02219 | 0.01853 |
| SMD | E-supervised E | 0.42899 | 0.42158 | -0.00741 | 0.01989 | 0.03219 | 0.02015 |
| MSL | Reconstruction continuation | 0.34945 ± 0.00189 | 0.36811 ± 0.00294 | 0.01866 ± 0.00180 | 0.03923 ± 0.00149 | -0.01920 ± 0.00167 | 0.01454 ± 0.00086 |
| MSL | Source-supervised native V | 0.36592 ± 0.00188 | 0.35562 ± 0.00159 | -0.01030 ± 0.00260 | 0.01827 ± 0.00298 | -0.04205 ± 0.00390 | -0.00418 ± 0.00496 |
| MSL | Official V | 0.34256 | 0.34850 | 0.00594 | 0.03664 | -0.03285 | 0.01317 |
| MSL | Official E | 0.39044 | 0.39988 | 0.00943 | 0.01777 | -0.03125 | -0.00465 |
| MSL | Official V+E | 0.39986 | 0.40407 | 0.00421 | 0.01322 | -0.03671 | -0.00828 |
| MSL | E-supervised E | 0.38119 | 0.39125 | 0.01005 | 0.03478 | -0.01586 | 0.02332 |
| SMAP | Reconstruction continuation | 0.36633 ± 0.00216 | 0.32903 ± 0.01050 | -0.03730 ± 0.00878 | 0.03928 ± 0.00178 | -0.07356 ± 0.01182 | 0.03569 ± 0.00021 |
| SMAP | Source-supervised native V | 0.38265 ± 0.00218 | 0.33422 ± 0.00185 | -0.04843 ± 0.00109 | 0.03891 ± 0.00409 | -0.09670 ± 0.00138 | 0.02978 ± 0.00391 |
| SMAP | Official V | 0.36555 | 0.30165 | -0.06390 | 0.01971 | -0.10338 | 0.01401 |
| SMAP | Official E | 0.42536 | 0.38124 | -0.04412 | 0.03832 | -0.08762 | 0.02690 |
| SMAP | Official V+E | 0.41481 | 0.38662 | -0.02820 | 0.01900 | -0.07365 | 0.01155 |
| SMAP | E-supervised E | 0.39631 | 0.38984 | -0.00647 | 0.01737 | -0.04439 | 0.01217 |

## Source supervision within each fixed readout

Differences are candidate minus official V or same-seed reconstruction continuation. Each value is the mean ± sample SD of three paired project-seed differences. Entity macro averaging precedes seed aggregation.
| Domain | Candidate readout | Reference | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | full_window | official_V | 0.00098 ± 0.00150 | 0.01386 ± 0.00085 | 0.00569 ± 0.00067 | 0.00519 ± 0.00057 |
| PSM | full_window | copy | -0.00361 ± 0.00182 | 0.00499 ± 0.00265 | 0.00704 ± 0.00122 | 0.01026 ± 0.00158 |
| PSM | full_window | LEFT | -0.12768 ± 0.00150 | -0.08391 ± 0.00085 | -0.11754 ± 0.00067 | -0.09877 ± 0.00057 |
| PSM | causal_endpoint | official_V | 0.03779 ± 0.00469 | 0.03301 ± 0.00277 | 0.02477 ± 0.00488 | 0.01093 ± 0.00388 |
| PSM | causal_endpoint | copy | 0.03422 ± 0.00345 | 0.03227 ± 0.00201 | 0.01822 ± 0.00407 | 0.00928 ± 0.00332 |
| PSM | causal_endpoint | LEFT | -0.00383 ± 0.00469 | -0.01219 ± 0.00277 | -0.06576 ± 0.00488 | -0.11008 ± 0.00388 |
| SMD | full_window | official_V | 0.00898 ± 0.00139 | 0.02119 ± 0.00068 | 0.01488 ± 0.00071 | 0.01267 ± 0.00049 |
| SMD | full_window | copy | -0.00426 ± 0.00349 | 0.01102 ± 0.00306 | 0.00177 ± 0.00613 | 0.00803 ± 0.00173 |
| SMD | full_window | LEFT | -0.06194 ± 0.00139 | -0.03019 ± 0.00068 | -0.07502 ± 0.00071 | -0.02354 ± 0.00049 |
| SMD | causal_endpoint | official_V | -0.00457 ± 0.00255 | 0.01287 ± 0.00109 | -0.03718 ± 0.00539 | -0.01014 ± 0.00147 |
| SMD | causal_endpoint | copy | -0.00945 ± 0.00399 | 0.00734 ± 0.00323 | -0.04486 ± 0.00777 | -0.01426 ± 0.00323 |
| SMD | causal_endpoint | LEFT | -0.05785 ± 0.00255 | -0.05246 ± 0.00109 | -0.11182 ± 0.00539 | -0.05870 ± 0.00147 |
| MSL | full_window | official_V | 0.00712 ± 0.00159 | 0.01671 ± 0.00074 | 0.01058 ± 0.00114 | 0.01346 ± 0.00022 |
| MSL | full_window | copy | -0.01249 ± 0.00393 | 0.01057 ± 0.00232 | -0.01128 ± 0.00408 | 0.00935 ± 0.00141 |
| MSL | full_window | LEFT | -0.15272 ± 0.00159 | -0.07728 ± 0.00074 | -0.16067 ± 0.00114 | -0.06272 ± 0.00022 |
| MSL | causal_endpoint | official_V | 0.02336 ± 0.00188 | 0.03507 ± 0.00299 | 0.01979 ± 0.00284 | 0.03082 ± 0.00502 |
| MSL | causal_endpoint | copy | 0.01647 ± 0.00178 | 0.03152 ± 0.00373 | 0.01158 ± 0.00273 | 0.02807 ± 0.00562 |
| MSL | causal_endpoint | LEFT | -0.14242 ± 0.00188 | -0.09556 ± 0.00299 | -0.11861 ± 0.00284 | -0.05854 ± 0.00502 |
| SMAP | full_window | official_V | 0.03257 ± 0.00185 | 0.00822 ± 0.00081 | 0.03402 ± 0.00294 | 0.00741 ± 0.00071 |
| SMAP | full_window | copy | 0.00518 ± 0.00965 | -0.00456 ± 0.00307 | 0.00920 ± 0.00783 | -0.00304 ± 0.00231 |
| SMAP | full_window | LEFT | -0.15894 ± 0.00185 | -0.09226 ± 0.00081 | -0.18839 ± 0.00294 | -0.07328 ± 0.00071 |
| SMAP | causal_endpoint | official_V | 0.01709 ± 0.00218 | -0.01099 ± 0.00473 | 0.02734 ± 0.00331 | -0.00836 ± 0.00456 |
| SMAP | causal_endpoint | copy | 0.01632 ± 0.00221 | -0.00418 ± 0.00715 | 0.03234 ± 0.00568 | 0.00286 ± 0.00509 |
| SMAP | causal_endpoint | LEFT | -0.11051 ± 0.00218 | -0.13117 ± 0.00473 | -0.09170 ± 0.00331 | -0.10306 ± 0.00456 |

The LEFT comparison always retains LEFT’s native offline readout. Seed-wise and per-entity differences, counts of wins/ties/losses and medians are in [paired_distribution.csv](paired_distribution.csv). The underlying differences are in [paired_entity.csv](paired_entity.csv).

## Entity distribution of the full-window candidate-minus-LEFT AP

Counts compare each fixed training seed with LEFT on the exact per-entity joint axis; entities are not extra training repetitions.
| Domain | Candidate seed | Macro AP delta | Median AP delta | Wins/ties/losses |
|---|---|---|---|---|
| PSM | 2025_point_weighted | -0.12757 | -0.12757 | 0/0/1 |
| PSM | 2026_point_weighted | -0.12923 | -0.12923 | 0/0/1 |
| PSM | 2027_point_weighted | -0.12623 | -0.12623 | 0/0/1 |
| SMD | 2025_point_weighted | -0.06182 | -0.04550 | 7/0/21 |
| SMD | 2026_point_weighted | -0.06339 | -0.04796 | 6/0/22 |
| SMD | 2027_point_weighted | -0.06062 | -0.04471 | 6/0/22 |
| MSL | 2025_point_weighted | -0.15218 | -0.07096 | 6/0/21 |
| MSL | 2026_point_weighted | -0.15147 | -0.07325 | 6/0/21 |
| MSL | 2027_point_weighted | -0.15451 | -0.07181 | 6/0/21 |
| SMAP | 2025_point_weighted | -0.16012 | -0.02454 | 19/0/34 |
| SMAP | 2026_point_weighted | -0.15989 | -0.02665 | 19/0/34 |
| SMAP | 2027_point_weighted | -0.15680 | -0.01103 | 20/0/33 |


The candidate's full-minus-endpoint AP differences are −0.12384±0.00380 on PSM, −0.00409±0.00344 on SMD, −0.01030±0.00260 on MSL and −0.04843±0.00109 on SMAP. All are paired training-seed means and sample SD on the new identical common axes. The SMD readout change increases ROC and both VUS metrics; PSM increases VUS-ROC despite lower AP, ROC and VUS-PR. MSL and SMAP have additional ROC increases with other metric costs. Added future context therefore gives no general ranking improvement under the fixed native variance readout.

Under full-window readout, candidate AP exceeds the frozen official V mean in each domain, but falls below reconstruction continuation on PSM, SMD and MSL. SMAP's candidate-minus-continuation AP is +0.00518±0.00965, with a substantial training-seed spread. The corresponding paired source-learning effects remain distinct from the same-weight readout effects. LEFT has higher values for all four domain-level metrics than the full-window candidate. These comparisons retain the official TimeRadar representation, source labels and fixed masks, and concern project fixed-weight offline readout rather than an exact reproduction of the author's original evaluation.

Only endpoint99 was directly supervised by the source pair term. At identical input windows, that score agrees with the existing native endpoint function. Earlier block positions use both different output positions and a different input context from their rolling causal counterparts. The comparison cannot separate these effects. It provides no justification for a new threshold, score mixture or automatically selected readout. Existing causal results remain archived separately.

Compatible CATCH, PCA, HistoryMean and TimeRCD caches enter [separate common-axis external panels](../../docs/experiments/native_full_window_20261124/TABLE_EXTERNAL.md). TimeRCD retains W5000 offline context and its A whole-input-statistics versus B FIT-only permissions. NASA lacks corresponding TimeRCD scores. The [native complete-axis panel](../../docs/experiments/native_full_window_20261124/TABLE_NATIVE_AXES.md) retains all complete positions and unscored tails; those axes are not used for cross-axis differences.

## Training objective and deployed criterion


V/V and E/E change the supervised decoder objective and deployed evidence; cross-readouts are predetermined descriptive controls. The MLP changes objective, updated component and inference together. Source labels and original FIT/SELECT windows are fixed, but the selected exposure and two-rate development budget differ. All rows retain their original full causal axes. This panel is separate from the LEFT intersections.

## AP

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Reconstruction continuation / V | 0.53903 | 0.45021 | 0.34152 | 0.36830 |
| Source-supervised V / V | 0.57717 | 0.44206 | 0.35412 | 0.38272 |
| Source-supervised E / E | 0.52350 | 0.42858 | 0.36975 | 0.39527 |
| V-trained / E (cross-readout) | 0.55301 | 0.39016 | 0.37085 | 0.42851 |
| E-trained / V (cross-readout) | 0.45800 | 0.44134 | 0.32687 | 0.35378 |
| Supervised MLP | 0.50494 | 0.37652 | 0.28168 | 0.28896 |

## ROC_AUC

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Reconstruction continuation / V | 0.73152 | 0.82025 | 0.63225 | 0.63893 |
| Source-supervised V / V | 0.76580 | 0.82705 | 0.66573 | 0.62680 |
| Source-supervised E / E | 0.67552 | 0.81369 | 0.68323 | 0.69797 |
| V-trained / E (cross-readout) | 0.68619 | 0.81022 | 0.67817 | 0.70164 |
| E-trained / V (cross-readout) | 0.63872 | 0.83302 | 0.61482 | 0.60904 |
| Supervised MLP | 0.72168 | 0.75892 | 0.55090 | 0.53892 |

## VUS_PR

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Reconstruction continuation / V | 0.49180 | 0.47609 | 0.41388 | 0.39807 |
| Source-supervised V / V | 0.51378 | 0.43352 | 0.42298 | 0.42949 |
| Source-supervised E / E | 0.50290 | 0.44149 | 0.44146 | 0.43453 |
| V-trained / E (cross-readout) | 0.51334 | 0.39734 | 0.43768 | 0.46907 |
| E-trained / V (cross-readout) | 0.44784 | 0.49259 | 0.39115 | 0.39606 |
| Supervised MLP | 0.45263 | 0.31839 | 0.31698 | 0.31272 |

## VUS_ROC

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Reconstruction continuation / V | 0.66665 | 0.87510 | 0.70468 | 0.67898 |
| Source-supervised V / V | 0.67986 | 0.86098 | 0.73565 | 0.67618 |
| Source-supervised E / E | 0.67104 | 0.86057 | 0.74956 | 0.75068 |
| V-trained / E (cross-readout) | 0.66157 | 0.85187 | 0.74420 | 0.75565 |
| E-trained / V (cross-readout) | 0.62058 | 0.88260 | 0.68144 | 0.65864 |
| Supervised MLP | 0.60520 | 0.76763 | 0.62785 | 0.59110 |


Source-supervised V improves AP relative to official V and reconstruction continuation in three fixed project-training seeds on PSM, MSL and SMAP. The paired AP differences against official V are +0.03786±0.00467, +0.01707±0.00162 and +0.01727±0.00213 respectively. These are the original full causal axes. SMD instead has −0.00460±0.00254. Its ROC rises while AP, VUS and event discovery show costs. Common-axis LEFT results are reported separately from these full-axis repeat statistics.

The E control uses its own fixed normal-FIT median scale and source-event validation risk, with the same labels, exposures and decoder update range. Cross-readouts ask how each training objective changes the other evidence without selecting a better score on QUERY. V/V exceeds E/E on PSM and SMD macro AP, while E/E has higher absolute AP on MSL and SMAP. Official E itself exceeds E/E on both spacecraft. The MLP learned source discrimination but gives lower AP, ROC and both VUS scores than V/V on all four tasks. These findings support a task-dependent benefit for the complete variance-supervised detector. They do not establish universal variance superiority or identify an independent FRFT effect.

## Causal operating costs

PSM's V-supervised checkpoint has lower false-alarm point rate than its frozen reference but lower persistent coverage. MLP reduces PSM false alarms further at lower recall and fewer matched events. On SMAP the MLP has higher long coverage but more false fragments and fewer matched events. On SMD it has fewer false fragments and higher event F1, with lower recall and long coverage. These metric tradeoffs prevent treating one favorable ranking or event column as complete operational superiority.

# Z/U/F Extensions

The completed U study starts from the selected source-V checkpoint and compares replay reconstruction, pair-preserving mixed adaptation and source-only continuation. Pair preservation recovers PSM AP against replay reconstruction but remains below both its frozen initialization and source-only continuation. Neither SMD subset gives a consistent net advantage. The fixed one-event F comparison gives lower post-F AP for the candidate than both matched initialization controls on PSM and SMD machine-3-2. PSM F coverage and recall increase with large false-alarm cost; SMD F improves ROC/VUS but discovers no query events at its fixed CAL. These studies remain separate extensions, with no established U/F benefit or contamination-robustness claim for the recipe.

The fixed-weight offline study recorded 469.24 occupied A40-seconds (0.130344 GPU-hours) for scoring 109 entities across eight selected checkpoints. Process startup and the minimum normal-input engineering check were not separately timed. Main CPU metrics took about 1,089 seconds and the parallel external-cache queue about 557 seconds; these overlap and are not GPU hours. An independent CPU review checked four fixed entities, full/common AP and ROC, index/footprint/availability, and all-table macro, three-seed and paired-distribution aggregation. It did not rerun VUS scoring. The same-window native V endpoint check is exact. Archived PSM E recovery from NumPy terminal copies differs from current CUDA reduction by up to 3.81e-6 (4.50e-6 for V+E); both original and new scores remain unchanged. [Complete record and review](../../docs/experiments/native_full_window_20261124/RESULTS.md).

# Discussion and Limitations

The central tested intervention learns through the native deployed disagreement rather than installing a free scoring head. Its evidence concerns complete supervised source learning, which includes additional labeled pairs and their exposure. The point-count formula alone is not isolated by comparison with reconstruction-only continuation. Event-balanced training remains an archived direct ablation with mixed outcomes and is not substituted for the selected point-weighted method.

Training labels are real public SWaT source events; this method is not purely self-supervised. PSM influenced recipe selection. NASA results use one sensor per entity and cannot support a multivariate interaction claim. SMD contributed nominal source training windows and is not an unseen-domain transfer test. Upstream pretraining exposure remains an author disclosure. LEFT and the candidate use different training, scaler and context permissions, even after scored-point alignment. Domain-mean gains can coexist with weak or negative entity medians and greater false-alarm fragmentation.

Native sample variance, the fractional representation, replay and mature classification losses retain their established attribution. Explicit weak-anomaly, long-lag, Self/Cross-lag and contamination-specific claims require additional evidence beyond these average public scores. Closed historical candidates and negative results remain available in the supplement and existing experiment records. The current complete detector has measurable source-supervised ranking gains and unresolved adaptation limitations; it is not yet a demonstrated complete solution to mixed-unlabeled learning.

# Conclusion

Direct source-event supervision of native reconstruction disagreement produced reproducible task-dependent ranking gains within a fixed pretrained detector. Matched decoder, E and standard MLP comparisons help delimit this effect, while complete LEFT benchmarks place its absolute detection performance in context. The results retain negative SMD transfer and false-alarm/coverage costs. The fixed offline readout reduces mean candidate AP in all four tasks, and does not close its gap to complete LEFT; source learning and readout effects are therefore reported separately. Current mixed-U and single-event-F recipes do not establish reliable target adaptation. The supported contribution is therefore a bounded source-supervised detection candidate and its evaluated behavior, with broader mixed-unlabeled method claims unresolved.

# Appendix A. Reproducible assets and extended comparisons

The formal method is specified in [method_definition.md](method_definition.md). Source draws, source selection, masks, scalers, trained checkpoint SHA, public score axes and complete events are linked in the existing experiment records. The [source repetitions](../../docs/experiments/native_event_rank_repeats_20261122/RESULTS.md), [SMD28 repeats](../../docs/experiments/native_event_smd28_20261123/RESULTS.md), [V/E controls](../../docs/experiments/native_e_vs_v_20261123/RESULTS.md), [standard classifier](../../docs/experiments/native_supervised_mlp_20261123/RESULTS.md), [U extension](../../docs/experiments/native_event_stage2_u_20261123/RESULTS.md) and [F extension](../../docs/experiments/native_event_stage2_f_20261123/RESULTS.md) retain their exact axes and permissions. The new NASA LEFT comparison is reported in its dedicated result record. Historical mixtures and replacement experiments do not contribute extra evidence to the present source-supervised detector.

# Appendix B. Retained historical research text

The following text is retained from earlier research snapshots, including user edits. It records historical hypotheses and completed explorations. It does not define the current method, contributions or active experiment schedule. The formal sections above and the current method definition take precedence for this draft.



  Industrial monitoring requires more than recognizing abrupt, high-amplitude departures from a nominal range. A weak deviation can be significant when it violates a normal response relationship, whereas a large change can be acceptable when several variables respond coherently to an operating transition. An anomaly may also persist after its initial transition: an effective monitor should discover the event and retain useful evidence during its abnormal steady state. These requirements are particularly demanding for multivariate streams in which responses are delayed, operating conditions change, and target-domain training records are unlabeled rather than independently certified as healthy.

Time–frequency representations offer a useful way to describe both local variation and temporal organization. TimeRadar learns a fractional time–frequency representation and combines reconstruction with contextual deviation learning [@timeradar2026]. LEFT learns interacting temporal, spectral, and multiscale views, followed by joint representation learning and reconstruction [@left2026]. These methods motivate a richer representation than a single instantaneous vector. However, a representation with good reconstruction or prediction accuracy does not automatically provide a useful anomaly ranking. A flexible reconstruction model may reconstruct a sustained abnormal state, and a predictor may improve normal accuracy without increasing the separation between abnormal responses and difficult normal transitions. These are empirical risks, rather than universal properties of either modeling family.

The completed external baseline and its cache attribution ask whether complete official long-history multivariate forecasting supplies localized anomaly evidence beyond native TimeRadar reconstruction disagreement. Each forecast uses strictly past512 observations jointly across variables and scores the current arrived observation using a single predefined median-residual criterion. This is distinct from the earlier random-initialized lightweight TimesFM modules and from stopped target-mean, conditional-quantile and energy formulations. Complete external pretrained models remain external methods, and their scores cannot be relabeled as a Hybrid contribution.

Several boundaries matter. First, an exact density is not a semantic definition of normality: a high-likelihood input can still be undesirable, and a legitimate new operating state can have low likelihood. Second, a causal endpoint detector may use the current observation after it arrives, but it must not use later observations. This is different from predicting the current observation from strictly earlier inputs. Third, statistical cross-variable dependence is not evidence of physical causation or a verified directed lag mechanism. Finally, robustness to unlabeled training contamination requires a separate experiment after the detector has demonstrated useful baseline discrimination. Performance that remains unchanged near a detection floor does not establish robustness.

The fixed native-V study uses PSM for multivariate development comparison and SWaT for event and long-duration analyses. Earlier run-separated Tennessee Eastman Process (TEP) experiments supply historical conditional-response mechanism evidence; the native-V candidate is not evaluated on TEP. These public settings have already influenced the research program and are explicitly treated as development or mechanism evidence. They cannot establish a new blind-confirmation claim. Strong simple controls, including historical means and history/conditional Ridge models where applicable, remain necessary alongside LEFT and TimeRadar.

A defensible contribution must go beyond applying mature source replay: it requires an attributable detection increment with useful ranking, coverage and false-alarm trade-offs under matched permissions. The locked LEFT paired experiment separates abnormal replacement from normal replacement of the same target slots. Neither the stopped conditional-geometry study nor the subsequent normal-source V-conditioning comparison repeats that exposure experiment or inherits its protection effects. None establishes a new robust learning mechanism. No stopped architecture or isolated favorable metric is promoted to an original contribution.

### Related Work

#### Time–frequency and multiview anomaly representations

TimeRadar uses a learnable fractional time–frequency representation within its foundation-model anomaly-detection framework [@timeradar2026]. Its fractional transform spans different domain rotations; it should not be described as a frequency-only encoder. LEFT combines temporal, spectral, and multiscale tokens, selective cross-view interaction, and an analysis–synthesis reconstruction chain [@left2026]. CATCH instead examines frequency patching with channel-aware modeling [@catch2025]. These works show different ways of exploiting temporal and spectral structure. Our locked adaptation studies retain LEFT's complete native learning and scoring chain as an external reference. A stopped Hybrid study tested frozen official TimeRadar response geometry inside a conditional energy criterion; a subsequent stopped study conditioned native V on strictly historical inputs. The current full pretrained forecasting baseline does not change either stopped study. Neither study claims the transform, multiview fusion, causal attention or replay itself as an invention.

#### Temporal dependencies and relationship anomalies

Forecasting and conditional-response models can expose departures that are not apparent from individual variables. Long-context forecasting also motivates temporal encoders, but a forecasting foundation model is not automatically an anomaly detector. The original TimesFM is a forecasting model [@timesfm2024]; the historical lightweight modules came from the pinned TimesFM3 release [@timesfm3release2026], without its pretrained weights. The new external baseline instead loads all330,710,976 parameters of the official TimesFM3Forecaster from its separately pinned complete checkpoint. These are distinct evidence sources. Historical public TEP experiments retain conditional Ridge as a strong relationship-modeling control. Self-lag denotes dependence on earlier values of the same variable, whereas cross-lag denotes dependence on earlier values of another variable. Neither a nonzero routing weight nor an active gradient proves that a detector identifies changes in those relationships. Mechanism claims require known relation-preserving and relation-breaking cases, including marginally similar controls and normal synchronous changes; the present two-arm geometry comparison makes no directed-lag claim.

CAROTS uses causality-aware augmentations and contrastive learning to improve multivariate anomaly detection [@carots2025]. Its published method includes a complete augmentation, representation-learning, and detection procedure, rather than an arbitrary pairwise binary classifier. Accordingly, relationship-preserving augmentation and contrastive discrimination cannot be presented as novel merely by attaching them to a new backbone. The present conditional-geometry study does not resume stopped relation-routing or isolated contrastive-loss variants.

#### Mixed-unlabeled adaptation and evaluation permissions

Mixed-unlabeled adaptation raises a different question from normal-reference learning. A target adaptation pool can contain unusual but legitimate states, anomalous episodes, and distributional changes relative to the source data. TSAD-C studies contaminated multivariate anomaly detection using a complete graph-conditional diffusion framework [@tsadc2025]. Such work motivates complete mixed-unlabeled method comparisons with matched data permissions. Contamination-specific harm need not be established before such comparisons, and method effectiveness must be judged using absolute detection performance as well as differential responses. We separate original-background adaptation, anomalous replacement, and matched normal replacement, and distinguish adaptation effects from calibration effects. A full contamination-robustness claim requires both detectable anomalies and a reproducible benefit under matched training exposure.

CANDI curates adaptation windows using reconstruction scores and similarity to normal reference representations, then updates input/output adapters around a frozen reconstruction backbone [@candi2026]. Our external comparison retains its complete MLP, FPM filtering and SANA adapters at the pinned released implementation. Its adaptation is restricted to the authorized U-FIT stream and then frozen for CAL and QUERY; this differs from the author's continuously adapting test stream. The normal reference is the source validation segment alone. CANDI's internal source-reference threshold is distinct from the final mixed-CAL alarm threshold. Each real sliding-window MSE is assigned to its actual endpoint and evaluated against project point labels, rather than broadcasting a window label or padding a score sequence. Differences from LEFT therefore include backbone, context, source training, standardization and readout permissions; cross-model differences cannot be attributed solely to filtering. CANDI remains an external method, not a Hybrid component or contribution.

Information permissions also limit comparisons. A pretrained source detector, a target-trained detector, and an offline window detector answer different questions. Restricting an offline score to a common set of timestamps does not remove the future observations used to construct that score. We therefore report native offline and causal endpoint results separately, disclose target-training and standardization permissions, and use a shared evaluator only where the raw axes and labels are compatible. Published scores under different permissions are background references rather than interchangeable rows of a matched experiment.

### Problem Formulation

#### Observations, causal windows, and anomaly events

Let $x_t\in\mathbb R^C$ denote the multivariate observation received at time $t$, and let $\mathcal F_t=\sigma(x_1,\ldots,x_t)$ denote the available observation history. An endpoint detector produces $s_t$ measurable with respect to $\mathcal F_t$. For a context length $H$, its input is

$$X_t=[x_{t-H+1},\ldots,x_t]\in\mathbb R^{H\times C}.$$

The endpoint permission permits the current observation $x_t$ after arrival and excludes $x_{t+1:}$. A predictor of $x_t$ has the stricter input permission $\mathcal F_{t-1}$; the two tasks must not be conflated. These definitions distinguish permissions: the completed conditional generator had causal inputs, the present frozen forecast consumes only strictly past observations, and the external native LEFT full-window readout is retrospective. Window normalization and every model statistic must respect the declared permission.

Ground-truth anomaly events are maximal contiguous positive intervals in the fixed official label sequence, subject to the preregistered boundary and censoring rules. QUERY labels are used only for evaluation after model and calibration choices have been frozen. The historical controlled exposure experiment separately permits FIT labels for donor construction, while keeping learner updates label-free. A weak anomaly refers to a small change relative to an explicitly defined training scale, not an uncertain label or a short event. A sustained anomaly is assessed through its event-level discovery and within-event coverage, including a predefined steady portion when one exists. Public fault labels alone do not identify which variable, lag, or physical relationship is abnormal.

### Completed complete-pretrained TimesFM3 frozen forecast benchmark

The official TimesFM3Forecaster source is fixed at918df02c9d366b2145bc5876135fa2131d3ed2a6, with the full Google timesfm-3.0-pytorch checkpoint revision43046b85ec22d584a13f8098c2ed39c889e129c2. All330,710,976 parameters remain frozen. This is distinct from the earlier lightweight, randomly initialized official-module reuse. No training or adaptation is performed. For each origin, one sample is a joint C-by512 history; variables are not independent batch samples. Define source-FIT-standardized observations z, the official sorted median forecast m, and the sole project anomaly criterion by

$$m_t=F_{\rm official}(z_{t-512:t-1};H=1),\qquad s_t=\frac1C\sum_{c=1}^C(m_{t,c}-z_{t,c})^2.$$

The current target enters only after the forecast returns. Internal stitching, variate attention, detrending and CPM/RevIN are retained, with fixed API flags: no symmetric averaging, positivity, extra z-normalization or input padding; return sorted nine quantiles. Internal output-patch rounding to64 does not reuse later predictions at other origins. SavedFP32 median and arrived target values are promoted toFP64 before square/channel mean. The score is not an official anomaly head. Inputs, nine quantiles, per-channel errors, exact indices and strict-past footprints are retained privately with published content hashes.

| Public task | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| SWaT | Full pretrained TimesFM3 residual | 0.093973 | 0.288375 | 0.096544 | 0.298381 |
| SWaT | Official TimeRadar native V | 0.795725 | 0.857852 | 0.576600 | 0.708990 |
| SWaT | Historical source-trained lightweight raw predictor D | 0.762338 | 0.838454 | 0.546680 | 0.696022 |
| SWaT | Cached HistoryMean512, ranking only | 0.087661 | 0.227761 | 0.090893 | 0.232695 |
| PSM | Full pretrained TimesFM3 residual | 0.344512 | 0.566754 | 0.376176 | 0.608554 |
| PSM | Official TimeRadar native V | 0.550952 | 0.735048 | 0.484578 | 0.648161 |
| PSM | Historical source-trained lightweight raw predictor D | 0.567902 | 0.743845 | 0.575549 | 0.764875 |
| PSM | Cached HistoryMean512 | 0.303673 | 0.525589 | 0.332386 | 0.567946 |

Every row uses the same336321/64728 QUERY points within its task. External LEFT, MLP and CANDI comparisons, complete strict/C3 operations, fixed low-FPR summaries and event-level records are in the accompanying full table. These complete methods differ in training, standardization and contextual permissions; the full model versus D is not an isolated weights-only ablation. Pretraining is official but is not independently certified free of benchmark exposure. CAL is the existing512-point mixed role, with34 SWaT attack points and512 PSM attack points; ownq99/higher/strict> thresholds are not a certified normal1%FPR budget.

TimesFM discovers additional short-lived event evidence: after three-point confirmation it has two SWaT-only any-hit events and eight PSM-only events relative to native V at their independent original operations. The SWaT-only long event receives one confirmed point in601 and zero steady coverage; the other event receives six of469 points. All eight PSM-only C3 events are shorter than398 points. Long steady macro/pooled coverage is only0.148020%/0.043849% on SWaT and0.238479%/0.113821% on PSM. C3 normal FPR0.031191%/0.071868% accompanies anomaly recall0.064885%/0.826684%. In PSM, independent matching is14/48 versus V6/48, while any-hit is14/48 versus V17/48; matching and any-hit are distinct. No combined OR detector is fitted or evaluated as a new method.

At descriptive1%normal-FPR caps, strict recall is0.374657%/3.840634%, below V68.861586%/16.619783%; therefore the weak discrimination is not solely a CAL-threshold artifact. No QUERY-derived curve cutoff changes a deployed threshold. The additional triggers do not establish persistent detection or synergy. This negative applies to the fixed512-history/horizon1 median-residual baseline; it does not isolate architecture versus pretraining causes or disprove all temporal forecasting. Actual new cost is5.487239 A40 GPU hours for one source gate and22 readout shards, all23 OS0/cleared, with no training, adaptation or repeated old model execution. The benchmark ends without automatic fusion, adaptation or a new Hybrid recipe.

### Cached event and quantile attribution: evidence and architectural limits

This read-only analysis uses all prior public SWaT/PSM predictions, all nine native marginal quantiles, per-variate squared errors and native TimeRadar V. It does not run a model, fit a score, recalibrate a threshold or create an ensemble. Events retain the original strict/C3 any-hit grouping and independently matched metrics. All complete events and shared misses are retained; descriptive phases are fixed by the existing32-point patch and512-history definitions, not selected using favorable outcomes. Context phases stop at adjacent original events, with normal labels used only for posthoc description. PSM has6,865 positive QUERY points outside the48 complete-event interiors; whole-axis statistics and complete-event summaries use different declared denominators.

The two frozen criteria respond differently. On SWaT, native V's normal/attack medians are0.066592/0.859943, whereas prediction-MSE medians are0.000226/0.000078. In long steady portions, pooled MSE/V means are0.000750/0.739075; event-macro means are0.006610/0.147217. One event supplies91.28% of the pooled steady points, so the pooled pattern is not a uniform nine-event effect. In paired local pre-event versus steady summaries, six of nine events reduce prediction MSE and six increase V, but mean per-event MSE rises0.005419 to0.006610. The per-event negative and contrary rows are retained. Low conditional forecast error can coexist with high reconstruction disagreement after the512-point history lies within an attack-labeled interval; neither phenomenon proves physical fault observability or causation.

Marginal native[q10,q90] coverage, averaged over observation-variate pairs, is87.9951%/92.4801% for SWaT normal/attack labels and79.3665%/79.0702% for PSM. All nine less-than/equality frequencies and exact zero widths are retained; these are not joint coverage or new anomaly scores. PSM's marginal quantile positions are close across labels. SWaT attack-labeled points have more exact ties and zero-width responses. Interval admission therefore cannot be treated as semantic normality, and no alternative quantile score is selected from these data.

At the unchanged strict operation, SWaT TimesFM-only events contain84 alarm points: mean largest-variate error share92.9827%, with85.7143% of points having one variate explain at least80% of squared error. The same latter rate is88.9225% in normal false-positive points. After C3, the two extra events contain only seven points with broader error contributions. PSM's16 strict extra events have139 alarm points, requiring on average3.0647 variates to explain80% of error; the eight C3 extra events have28 points and a corresponding mean2.8571. These are statistical error contributions, not physically identified fault channels or evidence for variate-attention benefit. Normal false positives can also have multi-variate deviations.

The additional events differ by task: SWaT strict extras have10/54/20 alarms in the first32, subsequent480 and post512 portions; PSM strict extras have103/36/0, and C3 extras21/7/0. Extra triggers are not universally onset-only or sustained. Strict TimesFM normal false alarms form3,117/905 fragments on SWaT/PSM, mostly single-output-point fragments, compared with native V122/185 fragments. C3 and its fragment statistics remain separate, with no backfill; neither normal annotations nor these statistics identify trusted operating transitions. All prior CAL restrictions, including PSM's all-attack role, remain.

The resulting decision is to retain a falsifiable time-frequency/temporal complementarity hypothesis, not to promote the frozen residual or an untested fusion as Hybrid. The complete forecast baseline is a measurement of one fixed predictor/criterion. Actual jointC-by512 inputs,32-point input patches and causal permissions do not establish utilization of distant history or variable interactions. No same-weight history/variate counterfactual, hidden-state attribution or attention ablation exists in these caches. Native V is likewise evidence for the complete TimeRadar pipeline, not an isolated proof of adaptive FRFT. The original Hybrid time-frequency/long-history/multivariate research direction and mixed-unlabeled objective remain; this cache analysis establishes neither a new learning mechanism nor robustness after adaptation. No successor is launched.

The next controlled-complementarity protocol has only CPU input-geometry checks, with no model execution or measured detection effect. It fixes four non-overlapping public-normal background blocks per task, local pulses, tapered oscillations, smooth relative delays and pointwise-energy-matched amplitude controls. Eleven of sixteen relative-delay cases satisfy the preset marginal/autocorrelation gates; the others remainNA without replacement. The source data's nominal-normal status and reused CAL constants do not establish historical online deployability. No controlled mechanism, joint gain or architectural contribution is claimed before the separately approved frozen-inference stage.

### Completed native-V and strictly historical conditioning study

Let $h_t=x_{t-512:t-1}$ and $X_t=x_{t-99:t}$, with source-FIT-only normalization. For each actual observed window, frozen official TimeRadar produces ten complementary-mask reconstructions $r_{m,t,c}$. Every call has B=1, internal norm=0, native within-window order sharing, fixed isolated mask seed84551, and the unmodified criterion=None/L=1 variance:

$$V_t=\frac1C\sum_c\frac1{9}\sum_{m=1}^{10}(r_{m,t,c}-\bar r_{t,c})^2.$$

The ten masks measure reconstruction disagreement; they are not independent calibrated posterior samples. A uses $V_t$. B (MLP32/32) and C (official TimesFM3 ResidualBlock and mixing transformer, width128/two layers/four heads/patch16, causal temporal and contemporaneous variate attention) predict sorted $(q_{.05},q_{.50},q_{.95})$ for $z_t=V_t/s_V$, where $s_V=\max(Q^{FIT}_{.95}(V),10^{-12})$. The head reads ordered last-patch channel tokens; there are no official TimesFM weights or explicit directed-lag module. The equally weighted pinball objective and untruncated criterion are

$$\mathcal L_Q=\frac1{3N}\sum_{t,\tau}\max\{\tau(z_t-q_\tau(h_t)),(\tau-1)(z_t-q_\tau(h_t))\},\qquad S_t=\frac{V_t/s_V-q_{.95}(h_t)}{\max(q_{.95}(h_t)-q_{.05}(h_t),10^{-3})}.$$

The source global-quantile initialization uses zero final weights and FP32 biases. Its score is a positive affine transform of A; both full QUERY axes have identical step0 AP/ROC and strict alarms, with no new ties or reversed order. Stable sort permits ties, negative scores remain, and final score arithmetic is FP64. The current target is absent from the history predictor; it enters only the native V at observation arrival t. The99-point overlap between teacher and predictor histories is legal and disclosed.

D shares C's actual encoder initial tensors, predicts current source-normalized raw variables, minimizes their MSE and scores the mean per-channel squared residual. D uses neither V nor the quantile criterion, so it is a task reference rather than an equal-head or same-weight ablation. New trainable parameter counts are SWaT B509091/C587011/D698143 and PSM B410787/C584707/D655129, in addition to the separately frozen1,636,043-parameter TimeRadar needed by A/B/C.

#### Source permissions, fixed selection and evaluation

The source/target roles and mixed CAL are unchanged from the preceding public task. B/C/D share complete20-epoch no-replacement FIT permutations, batch128 including the final partial batch, Adam1e-4 and FP32. SWaT completes3880 updates and494360 window exposures per model; PSM1040 and131840. Full normal SELECT is evaluated at0 and every epoch; the earliest strictly minimum own-task loss selects SWaT epochs B14/C20/D20 and PSM B16/C7/D19. All six selected models freeze before CAL/QUERY value readout, and all16 score banks seal before label evaluation. No R, pollution, additional seed, heterogeneous source pretraining or private data is used.

The fixed QUERY axes contain336321 SWaT and64728 PSM points; each model uses its own score at the same512 mixed CAL points for q99/higher/strict>. PSM CAL remains all attack-labeled. These empirical thresholds do not certify normal1%FPR. The temporal information footprint is causal endpoint, but the evaluation is retrospective, not a deployed trial. Native LEFT192-point whole-window scores, MLP/CANDI10-point scores and historical TimeRadar retain their differing source statistics, pretraining, target adaptation and input availability. Ranking-only caches receive no invented CAL threshold. [Complete same-axis tables, curves, artifacts and permissions](../../docs/experiments/timeradar_native_v_conditioning_20261119/RESULTS.md) remain the authoritative exports.

#### Fixed results and scientific scope

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

### Native TimeRadar head probing on natural mixed target data

The conditional-V B/C recipe and its prospective R adaptation remain stopped. A separate fixed same-backbone experiment now tests native TimeRadar reconstruction-head probing with data replay. The official model's fractional-order network, embeddings, encoder and buffers are frozen; only the existing complex reconstruction decoder is updated. The native ten-copy terminal V criterion, SOURCE_FIT scaler, B=1 per-window order, norm0 and isolated84551 complementary masks are unchanged from A/noU. This is an established update scope [@timeradar2026] combined with mature replay, not a claimed original Hybrid mechanism or a restoration of former channel adapters.

For each logical batch of64 actual windows, the unchanged reconstruction objective is

$$\mathcal L_{\mathcal B}(\theta_{dec})=\frac1{64}\sum_{w\in\mathcal B}\frac1{10\cdot100\cdot C}\sum_{m,j,c}(\hat x^{(m)}_{w,j,c}-x_{w,j,c})^2.$$

Every native forward has B=1; gradients of64 scaled per-window losses are accumulated before one AdamW update. Frozen modules remain in eval, including backbone dropout. No independent windows are put into a native batch to average their fractional order. TT=[T,Tprime], SS=[Sprime,S] and R=[T,S] have32 windows per half, paired T and S positions, independently sampled Tprime/Sprime, and the same official initial state/fresh optimizer. Role-local nonoverlapping complete100-window pools follow the prior head sampler, with3960/900 source/target windows in SWaT and1059/175 in PSM. Repeated draws do not create independent events.

The fixed prior PSM probing configuration is logical batch64,2000 updates, AdamW1e-4/betas(.9,.95)/eps1e-8, weight decay.1 under the original name/LayerNorm grouping, and linear warmup200. Literal names containing 'bias' and LayerNorm parameters are exempt; complex rb*/ib* names are not. Use endpoint2000 without source SELECT, CAL or QUERY selection. Train only SOURCE_FIT and the existing natural U-FIT. This does not reproduce the LEFT256-batch training exposure or create a new normal-source initialization; noU remains the official pretrained TimeRadar plus fixed SOURCE_FIT normalization.

The primary learning contrasts are R−TT, with R−noU, R−SS and TT−noU retained. All methods share the same QUERY points,512 CAL observations and V criterion. Each uses its own q99/higher/strict> threshold; SWaT's selected mixed CAL has34 attack-labeled points and PSM512, as already recorded in the previous metrics. They do not certify a normal false-alarm budget. All new endpoints and scores must freeze before new label evaluation. Natural-mixture gains or losses cannot, without a separate intervention, be attributed specifically to anomaly contamination. Mature TimeRadar/replay retain their named external attribution; no innovation claim follows from training alone. All six endpoints and twelve new score banks were frozen before evaluation. The complete fixed comparison is now available below.


#### Fixed native-V adaptation results

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


### Fixed target-mean / source-copy reconstruction comparison

This completed single-factor experiment retains the preceding native TimeRadar head-only R initialization, paired T/S plan, effective batch64,2000 endpoint updates, optimizer and V inference. It changes only the target-window reconstruction constraint. No conditional-V B/C, old channel adapter, new backbone, score fusion or supervised target-label training is restored. Both public tasks, SWaT31 and PSM25, and seed2025 are fixed before new results.

For ten reconstructions $o_{w,m,j,c}$ of a single actual window, define

$$\bar o_{w,j,c}=\frac1{10}\sum_{m=1}^{10}o_{w,m,j,c},\qquad
E_w=\frac1{100C}\sum_{j,c}(\bar o_{w,j,c}-x_{w,j,c})^2,$$

$$V_w=\frac1{100C}\sum_{j,c}\frac1{9}\sum_{m=1}^{10}(o_{w,m,j,c}-\bar o_{w,j,c})^2.$$

Here $V_w$ is the full-window mean correction1 copy variance, not the terminal native detection score. The classical squared-error decomposition gives $L_{copy,w}=E_w+0.9V_w$. With32 target and32 source windows in the same logical batch, the new objective is

$$L_{new}=\frac1{64}\left[\sum_{w\in T}E_w+\sum_{w\in S}(E_w+0.9V_w)\right].$$

At identical parameters, windows and masks, the original R objective satisfies $L_R-L_{new}=0.45\,\operatorname{mean}_{w\in T}V_w$, including equality of parameter gradients. Training directly computes the mean-reconstruction MSE for target windows, rather than subtracting two large quantities. The mean remains differentiable, uses all100 positions and channels, and introduces no extra half weighting. Source windows retain the original ten-copy MSE. Each native forward remains B=1; each window contributes its loss divided by64 before a single optimizer update. $E_T,V_T,E_S,V_S$ are recorded from the same forwards and do not introduce another training pass.

The inference criterion remains the unchanged channel-mean, correction1 copy variance at position99. An increase in training variance does not establish improved anomaly discrimination. The hypothesis tested is whether removing direct target-copy agreement pressure, while retaining source-copy reconstruction, provides an actual detection increment over matched R. Classical decomposition, official TimeRadar and mature replay retain their original attribution; a new contribution would require the paired detection evidence, not merely this identity.

The primary contrast is new minus R; noU, TT and SS are retained unchanged. Common QUERY axes,512 mixed-CAL observations and each model's q99/higher/strict> threshold remain fixed. SWaT has34 attack-labeled selected CAL points and PSM512; no normal1% false-alarm guarantee follows. Point AP/ROC, fixed VUS, strict/C3 operating outcomes, one-to-one versus any-hit events, long-event macro/pooled coverage and longest misses will all be reported. QUERY curves are descriptive only. The fixed results are reported below; a bounded ranking increment is distinguished from operating cost and does not establish contamination robustness.


#### Fixed target-mean results and decision

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

Against matchedR, PSM AP/ROC improve0.864/0.774pp. C3 long-event macro/pooled coverage rises42.016/54.195% to53.380/72.179%, longest miss falls1228 to879, matched events11 to12 and any-hit11 to14. However normal FPR rises3.589% to8.924%, event F1 falls.081181 to.076433, and VUS-PR/ROC fall. Against noU, ranking improves and false alarms fall but sustained coverage remains lower; against SS, ranking improves and false alarms fall but coverage is lower. This is a real limited ranking/coverage positive with cost, not demonstrated low-FPR preservation.

On SWaT, AP drops0.209pp while ROC gains0.342pp; both VUS fall againstR. C3 FPR1.330% to1.247% is accompanied by recall67.039% to66.461%, long macro11.347% to11.266%, pooled79.974% to79.401% and longest miss4951 to5172; matched/any-hit stays10. Relative to noU, both new VUS retain a positive but AP/ROC and coverage remain lower. The source/target constraint has not established a stable two-task advantage.

All actual training functions, source roles, scalers, plans, masks, budget and nativeV inference remained fixed;4000 per-update order hashes matcholdR. E_T/V_T andE_S/V_S are archived from the same forwards. Full-window variance decreases in both new training traces; neither that trend nor training loss is a detection criterion, and the old/new loss decomposition is evaluated at identical weights, not between diverged trajectories.

Two trainings and10 native readouts plus gates consumed2.747930A40GPUh with14 successful cleaned jobs. One bounded independent same-family review checked45048 fields (max2.22e-16); the two new VUS states share pinned official metric code and oldVUS is reused. Complete method/operating tables, all events, single-factor contrasts, source-component traces and fullPR/low-FPRROC curves are in `docs/experiments/timeradar_target_mean_20261120/`. Exact mixed-CAL34/512 attack composition is unchanged. Mature TimeRadar, replay and error–disagreement decomposition retain their attribution. The candidate is recorded with its limited PSM positives and substantial costs; no coefficient, TT-mean, teacher, backbone, seed or data extension is automatically launched.


#### Fixed adaptation repetition and low-false-alarm description

The fixed repetition is complete: eight new adaptations and sixteen native score banks add seeds2026/2027 to both public tasks, with all2025 states and scores reused. Each job restores the same official pretrained tensors, SOURCE_FIT scaler and fresh optimizer. Within each seed, R and target-mean/source-copy share the same T/S plan, effective batch64,2000-step endpoint and fixed84551 masks/evaluation randomness. The actual CPU/CUDA seeds and all per-window order hashes were checked. These are adaptation-sampling repetitions at one pretrained state, not independent pretrained models or datasets. noU is a single frozen reference and TT/SS remain2025 historical controls; no QUERY-based checkpoint selection is used.

For a false-positive cap $a$, the descriptive partial area is $\int_0^a\operatorname{TPR}(u)\,du$ on the original piecewise-linear empirical ROC, with boundary interpolation only for integration. We also report area divided by$a$, rather than McClish standardization or a convex hull. The attainable recall under caps1%,3%,5% uses whole tied-score groups, without interpolating an attainable decision; equal-recall points prefer the smallest FPR and then highest cutoff. These QUERY-derived points are not used to set deployment thresholds or to change CAL.

Three-consecutive confirmation has a separate curve: at a valid three-observation history, its threshold score is the minimum of the current and previous two native scores. The first two points in each continuous QUERY block never alarm, gaps reset confirmation, and all QUERY positives/negatives remain in the rate denominators. Event and sustained coverage at descriptive cap points use the actual confirmed alarm bits once, not confirmation applied twice. Original q99/higher/strict> CAL operations remain separate and unchanged, including PSM's all-attack CAL limitation.

PSM point-ranking gains over R repeat in all three seeds: AP differences are+0.864,+0.400,+1.408 percentage points (mean+0.891, sampleSD0.505pp); ROC-AUC improves+0.758+/-0.125pp. Relative to the single fixed noU reference, AP improves+1.029+/-0.122pp and ROC-AUC+0.851+/-0.150pp. This does not establish uniform superiority: VUS-PR relative to R changes-0.534+/-0.700pp and VUS-ROC remains lower. At the unchanged CAL operation, C3 recall improves+5.393+/-2.816pp but normal FPR rises+2.256+/-2.678pp. Long-event steady macro coverage improves+5.230+/-5.329pp and pooled coverage+13.491+/-4.417pp; relative to noU they remain lower by38.016 and23.572pp respectively. C3 one-to-one matched events are12/10/11 versus R11/12/12; any-hit14/11/12 versus R11/12/12 is separately tabulated. Thus additional point/steady coverage is not a reproducible new-event advantage.

On SWaT, AP declines in all three pairs (mean-0.427+/-0.241pp), while ROC-AUC rises+0.338+/-0.024pp. VUS-PR declines-2.510+/-0.270pp relative to R. C3 FPR decreases0.091+/-0.007pp but recall decreases1.276+/-0.867pp; long steady macro coverage declines3.007+/-5.043pp and pooled coverage0.885+/-0.357pp. These tradeoffs provide no corresponding sustained-detection advantage. All reported standard deviations are sampleSD(ddof=1), n=3; event counts and seed-specific costs remain in the complete tables.

In the full repetition, PSM strict normalized partial area increases+0.265+/-0.099pp through1% FPR and+1.476+/-0.799pp through5%; C3 changes-0.267+/-0.303pp and+1.547+/-1.047pp respectively. Yet attainable recall at the1%cap changes-0.444+/-0.227pp(strict) and-0.335+/-0.731pp(C3). At3% and5% caps, C3 recall increases+2.025+/-1.597pp and+1.904+/-1.696pp, but at3% one-to-one matched events decrease by one in each seed. Thus the favorable budget region is limited, and improved aggregate coverage is not extra-event detection. SWaT strict and C3 normalized areas decline in every pair at both bounds. All such cap points are descriptive only.

The cached2025 low-FPR supplement remains immutable as a source: PSM C3 area/a improves at5% but not1%; its1%cap point recall decreases despite slightly higher complete-event coverage. All three-seed strict/C3 areas, actual cap recalls and event-wise sustained measures are reported together in the single supplement, without converting descriptive QUERY cutoffs into CAL thresholds. The strategy is retained as a reproducible but task-limited PSM ranking/coverage effect, with SWaT negatives, false-alarm costs and noU coverage gaps fully preserved. This is not evidence for contamination causality, independent pretrained-model robustness, or a completed original Hybrid contribution. No further recipe is automatically scheduled.

The new computation consumed10.828306A40GPUh including startup and cleanup:52 successful jobs (four engineering gates, eight adaptations, forty readout parts). One additional guard rejection occurred before process launch and used zero GPU seconds; only the ten unstarted readout items were recovered after the original thirty successful items closed. Original error/cleanup records are retained. Native weights, scalers, optimizer states and numerical banks remain archived outside publicGit; code, configurations, complete metrics and content hashes are published. This resource interruption did not change the fixed experiment matrix or any score.

#### Fixed SMD subset scope check

The preassigned, historically exposed SMD machine-3-2 and machine-3-3 extend the frozen target-mean hypothesis at seed2025. They are two entities of one dataset, not two independent domains, blind confirmation or the full28-machine benchmark. Only four adaptations are new: R and R-MeanT per entity, each from identical official pretrained weights and a fresh optimizer. SOURCE_FIT is each native train record's[0,14221), and natural U-FIT is the previously isolated test value prefix[0,7901); no donor/target labels enter learning. Separate train CAL roles preserve the one-point entity boundary difference. They have no independent point labels, so calibration attack counts are NA and no normal1%FPR certification is claimed. The fixed scored QUERY remains[8957,23703),14746 points per entity; the oldnorm1/batch32 caches were incompatible with the current source-FIT standardized norm0/B1 nativeV and were not substituted.

| Entity | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| machine-3-2 | noU | 0.036035 | 0.503761 | 0.068493 | 0.728434 |
| machine-3-2 | R | 0.037968 | 0.527994 | 0.136598 | 0.761325 |
| machine-3-2 | R-MeanT | 0.035829 | 0.517616 | 0.084654 | 0.737006 |
| machine-3-3 | noU | 0.043771 | 0.532765 | 0.066217 | 0.664504 |
| machine-3-3 | R | 0.092484 | 0.653673 | 0.132917 | 0.782451 |
| machine-3-3 | R-MeanT | 0.087649 | 0.626640 | 0.119191 | 0.753943 |

R-MeanT falls below R on all four ranking metrics in both entities. AP differences are-0.214pp on3-2 and-0.484pp on3-3. On3-3 it still exceeds noU by4.388pp AP, but R is stronger, so this does not establish the independent benefit of the mean constraint. At original CAL,3-3 C3 false positives75->60 and matched events9->10/16 are retained as an operating positive; recall remains only1.724->1.897% and longest miss is368 in both. Strict recall instead falls8.448->7.931% and longest miss96->134. On3-2 C3 hits, recall and longest miss do not change. Both QUERY sets contain zero qualified duration>512 complete events: long steady macro/pooled coverage and their changes are NA, not zero and not evidence against long-fault observability.

Low-FPR results do not rescue a broad claim. On3-3 the strict normalized area through1% improves0.589pp relative to R, but through5% declines0.757pp; its C3 cap1/3/5 recall differences are-0.517/-1.379/-1.034pp. All caps are posthoc descriptions under unchanged formulas, not learned deployment thresholds. The complete8/16-event sets, ineligible-event NA rows and bothMeanT-R/noU comparisons are preserved in the fixed-subset tables.

Across the entire accepted evidence, PSM retains repeatable AP/ROC and some wider-budget positives; the2025 longest-miss improvement1228->879 did not repeat in2026/2027 (both arms1228), and event-discovery gains did not repeat. SWaT AP negatives repeated. The additional SMD evidence therefore supports retaining a PSM-local learning result plus the disclosed3-3 operating tradeoff, rather than promoting the current constraint into a broadly established Hybrid method contribution. TimeRadar, replay and the classical error/disagreement identity retain their provenance. No further entity, seed, threshold or recipe is selected from these outcomes. The target-mean branch is now formally closed after its three-seed PSM/SWaT repetition and fixed SMD subset evaluation. It is retained as historical evidence, not an unfinished core method.

### Historical stopped conditional-energy candidate and fixed geometry comparison

Given strictly past observations $h_t=x_{t-512:t-1}$, a lightweight conditional generator produces $v_k=G_\theta(h_t,\epsilon_k)$, $k=1,\ldots,4$. It uses the official TimesFM3 residual and mixing blocks at width128, two layers, four heads and patch16, including variate attention. Shared8 and per-variate8 independent Gaussian noise allow joint current-value draws. No current target value enters this predictor. The generator has612626 trainable parameters; the separate official TimeRadar response operator retains1,636,043 frozen parameters. This is not the official330M TimesFM3 model, an official quantile forecast, or a restoration of the old Hybrid V1.

For each candidate v and the observed value, independently evaluate the native TimeRadar function on $[x_{t-99:t-1},v]$. Each B=1 call estimates its own fractional order, with native within-window variable sharing, and returns ten complementary-mask reconstructions. Masks use isolated seed84551; eval mode alone does not fix the random forward masks. Source-FIT mean/scale are fixed, and the native internal normalization is disabled. Defining the response as $r_h(v)$, arm A uses

$$g_h(v)=\left[v/\sqrt C,\;\operatorname{vec}r_h(v)/(s_r\sqrt{1000C})\right],$$

where $s_r$ is the aggregate RMS of complete SOURCE_FIT responses. Arm B uses only $v/\sqrt C$. Both minimize and report

$$s_t=\frac14\sum_{k=1}^4\|g_h(v_k)-g_h(x_t)\|_2-\frac1{24}\sum_{k\ne l}\|g_h(v_k)-g_h(v_l)\|_2.$$

This is the conventional energy score, equal to half the authors' implementation convention [@generative_scoring2024]. It is not a sum of independently calibrated reconstruction and forecast scores. Raw coordinates make the fixed response mapping injective; under finite first moments the population score is strictly proper for the corresponding conditional distribution. This mathematical property neither certifies anomaly semantics nor establishes contamination robustness. The frozen response parameters receive no gradients, while generated-input gradients pass through the actual fractional transform and reconstruction chain. No healthy counterfactual, memory retrieval or student-output preservation term is used.

We retain normal-source selected noU states and fixed2000-step R adaptations for both arms. Source training also has a2000-update cap, normal SELECT every100 including step0, logical batch16 and Adam1e-4. A/B share initial generator tensors, actual source/target draws and noise plans; R uses eight target and eight normal-source examples per optimizer update. It is a shared mature learning strategy, not an originality claim. These experiments are same-domain normal initialization followed by natural mixed adaptation, without a new heterogeneous multi-source pretraining stage or a new contamination-dose matrix.

#### Historical conditional-energy evaluation setup

Only public SWaT31 and PSM25 enter the experiment. No first-paper laboratory, weak-label, experimental-run or enterprise records are used. The original roles are unchanged. Because512 past points plus the observed target require513 legal points, the shared QUERY axes are SWaT `[608535,944856)` and PSM `[23016,87744)`, containing336321 and64728 points. PSM's additional360-point crop is a footprint requirement, not a score-dependent choice. Candidate scores are available at t; cached LEFT whole-window scores retain their later window-end availability and are marked retrospective.

Each model receives its own q99/higher/strict-greater-than threshold using512 fixed points in the same legal mixed-CAL range: SWaT `[586056,607320)` and PSM `[18624,21888)`. PSM's all-attack CAL is retained rather than cleaned using labels. Its threshold is an empirical operation point, not a normal1% false-alarm guarantee. Compatible historical caches are recropped/recalibrated on these points without overwriting old tables. A cached method with no compatible new CAL bank contributes ranking only; its old threshold is not substituted. Query labels remain sealed until the entire fixed set of new models and scores is complete.

The primary contrasts are A−B separately before and after R, and each arm's R−noU. AP, ROC-AUC, fixed-implementation VUS-PR/VUS-ROC, strict point false alarms/recall, one-to-one events, any-hit, three-point coverage and long-event macro/pooled coverage remain separate outcomes. Normal SELECT improvement does not replace anomaly evaluation. Native LEFT+R, MLP, CANDI, complete TimeRadar and simple controls retain their actual training, statistics, context and compute differences. A single seed on two exposed development tasks does not establish replication, unseen-domain generalization or a directed-lag contribution.

### Historical fixed-geometry results and technical decision

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

### Locked external LEFT basis and paired additional-exposure experiment

This completed external comparison uses complete native LEFT with source replay R, its original objective, state updates, inference and per-model calibration. TT is the matched target-only control. Here A denotes each arm's unchanged natural target-FIT draw plan: $R_A=[T,S]$ and $TT_A=[T,T']$; these historical A/B/C exposure conditions differ from the current candidate's A/B geometry arms. Each plan has 2,000 updates with halves of 128 windows, giving 256,000 shared T slots and 512,000 total window exposures. Source replay is an established control, not the claimed invention of Hybrid.

The new $\{TT,R\}\times\{B,C\}$ experiment changes only the shared T half. Before learning, an evaluation-side constructor uses labels within the current authorized U-FIT segment to identify full 192-point abnormal donor windows for B and full 192-point normal donor windows for C. Both conditions replace the same 25,600 T slots: 10% of T exposures and 5% of total exposures. R and TT receive identical replacement locations and corresponding donor draws within each condition. The independent T′ and source S halves remain unchanged. The learner receives numerical windows and the native loss, without labels or a label-dependent update rule. This is label-assisted controlled data construction, not an operational method for discovering unknown contamination.

Direct checks of the frozen U-FIT label arrays find 3,468 all-abnormal and 82,903 all-normal stride-one windows in SWaT, and 649 all-abnormal and 12,852 all-normal windows in PSM. The abnormal windows originate from nine and five qualifying event intervals respectively, not that many independent window events. Donor reuse and achieved exposure fractions are recorded separately. In seed2025, SWaT used3,466 distinct B windows(maximum19 presentations) and21,982 C windows(maximum4); PSM used649 B(maximum63) and11,080 C(maximum10). Repeated seeds use their own fixed draws from the same pools; their complete reuse/exposure counts are retained separately. These repeated presentations do not increase the number of underlying event intervals. A full-window donor criterion does not make the natural background clean: original T may already contain anomalies, and unchanged T′ remains mixed. Equal additional abnormal donor exposure in the shared slots does not imply equal total contamination between TT and R. The registered replacement fraction counts window-exposure slots, not unique windows, independent events or the total anomalous-point fraction.

For each separately reported metric M, define $M_{a,k}$ for arm $a\in\{R,TT\}$ and condition $k\in\{A,B,C\}$. The paired interaction is

$$I_M=(M_{R,B}-M_{R,C})-(M_{TT,B}-M_{TT,C}).$$

We report the absolute outcomes and within-arm B−C, B−A and C−A alongside $I_M$. B−C contrasts abnormal with normal replacement under the same slot plan; C−A reveals effects of replacement and normal-donor composition relative to the natural draw. A positive interaction is favorable for a higher-is-better metric only in the context of the absolute outcomes; false-positive counts and delays have the opposite direction. No rule forces B to degrade, and a favorable interaction alone does not establish good detection, difficult-normal protection or general robustness. Native readout, CAL locations and threshold rules remain fixed; the existing all-attack PSM CAL limitation applies.

#### Completed three-seed controlled-exposure evidence

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

### Native source replay: completed evidence (18 November 2026)

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

### Complete external adaptation references and matched plain-MLP replay

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

#### What the matched backbone comparison establishes

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

### Historical stopped P/V source-output preservation experiment

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

### Appendix A: stopped density candidate and original evaluation setup

This appendix retains the earlier mathematical definitions and experimental setup for historical traceability. They do not define the current LEFT-plus-replay baseline or its paired exposure experiment. References to the candidate in this appendix refer only to that stopped design.

#### Normality learning, invertible models, and anomaly criteria

Reconstruction, prediction, prototype distance, and likelihood make different assumptions about normality. Their training losses can emphasize different features from those that control event detection. Real NVP introduced tractable density estimation through invertible affine transformations with exact log-determinants [@realnvp2017]. Wavelet Flow already combines wavelets with normalizing flows and models signal scales explicitly [@waveletflow2020]. Hybrid reuses these established principles; a Haar transform inside a flow is not its novelty. GANF also uses joint density estimation for time-series anomaly detection through a graph-factorized flow [@ganf2022], and MTGFlow studies multivariate detection with dynamic graphs and entity-aware normalizing flows [@mtgflow2023]. Thus neither window likelihood nor graph/dependency modeling is unprecedented.

The stopped candidate tested a particular alternating-coordinate joint normality model and its sustained-anomaly behavior under fixed information permissions. An invertible representation does not discard dimensions through an encoder bottleneck, but this property alone does not guarantee that the learned density orders anomalies correctly. The identity-transform Student-t control and the time-only and local-context ablations test whether learned dependencies and multiscale organization are useful, rather than attributing improvement to a generic heavy-tailed score or a change in score units. A publication-level distinction from existing flow methods would require such empirical evidence and an appropriate direct comparison; architectural resemblance alone is not treated as a contribution.

#### Historical formulation: stopped joint-density candidate

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

#### Training regimes and future adaptation questions

We distinguish source training (Stage 1), source-calibrated zero-shot detection (Z-S), target-calibrated detection without parameter updates (Z-T), unlabeled target adaptation (U), and few-event supervision (F). Target CAL may define statistics and thresholds only within its authorized role; it never supplies gradients. Target FIT supplies U updates without anomaly labels. F supervision is counted in complete independent support events, with support, validation, and query events separated. A change from Z-S to target-trained learning changes the scientific question and is reported as such.

This first evaluation establishes only the candidate's basic detection capability under its stated training permissions. It does not establish multisource transfer, a contamination-robust U mechanism, or the value of few-event labels. Those questions remain in the research framework and require separate matched tests once the basic criterion is supported.

#### Historical experimental setup

#### Historical joint-density candidate: fixed public-data settings

The data roles are inherited from the existing frozen assets; they are not reselected using candidate scores. Every input window must be contained in a legal role or its explicitly authorized causal context. No physical record is split into overlapping training and query windows, and no discontinuous records are concatenated. Existing guards are retained where they exceed the candidate's actual footprint. The candidate uses $H=192$ for PSM and SWaT and $H=100$ for TEP. These settings are fixed before candidate QUERY results are opened.

| Setting | Fixed training and selection assets | Fixed evaluation scope and interpretation |
|:--|:--|:--|
| PSM | FIT $[0,79488)$, SELECT $[80032,105984)$, CAL $[106528,132481)$; 25-channel FIT-only standardization | Existing complete test axis $[512,87841)$ and 71 complete events; external comparisons additionally use their exact common scored intersection. Repeatedly exposed development data. |
| SWaT | Source/normal-reference FIT $[0,296020)$, separate historical U-FIT $[296564,395237)$, SELECT $[395781,445117)$, CAL $[445661,495000)$; 31 retained official channels | Existing scored QUERY $[608535,944919)$ with 22 complete attacks, including nine longer than 512 samples. The reserved donor interval is not merged into normal-reference FIT. One operating record, not 22 independent runs. |
| TEP Rieth2017 V1 | Normal simulation runs 1–60 FIT, 61–80 SELECT, 81–100 CAL; run boundaries preserved | Fault runs 201–220, all 20 fault types; the predefined 4/11/14 subset is reported alongside all types. Existing target axis is samples 101–960, with fault onset at sample 161. A separate normal-test pool is retained. |

TEP's fixed response task uses XMEAS9 as target and the established XMV10/XMEAS21 conditional information. The density candidate scores their complete three-channel joint window; this differs from the target-XMEAS9 residual scored by the historical regression task. Their data axes and legal inputs can be matched, but their detection objects must remain explicit. A gain in joint-window detection alone cannot establish improved conditional-XMEAS9 modeling. XMV10 is a closed-loop manipulated variable and is not assumed to be a strictly exogenous cause. The three-minute sample interval permits conversion of index delay into elapsed time in this task. Shared normal prefixes across faults from the same simulation run are not counted repeatedly as independent false-alarm evidence. Fault-wise prevalence on the scored axis is high, so AP is interpreted together with ROC-AUC, normal false alarms, delay, and coverage rather than alone.

SWaT attack labels identify system-level attack intervals; they are not precise per-channel abnormality labels. Long-event analyses retain the nine predefined events and separately describe onset, sustained state, and recovery where the fixed boundaries permit. A low score during an event does not by itself prove that a sustained observable deviation exists in the retained channels. PSM, SWaT, and these TEP assets have historical development exposure. No result from this evaluation is described as a blind confirmation, and no favorable entity or fault is substituted after evaluation.

#### Comparisons, information permissions, and ablations

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

#### Calibration, events, and sustained detection

Each method obtains a threshold from its own score units using the same authorized CAL locations and the frozen quantile rule. The existing rule is the 0.99 higher empirical quantile with a strict score-greater-than-threshold decision. A score-definition change therefore requires its own CAL evaluation; a historical numerical threshold is not transferable across units. CAL is not a QUERY-derived false-alarm guarantee, and its finite size and domain shift are disclosed. No QUERY labels choose a threshold, checkpoint, seed, horizon, aggregation weight, or entity.

TEP uses all 8,000 authorized normal CAL endpoints. PSM and SWaT use the same 512 pre-fixed mature CAL endpoints as their cached native controls. The new SWaT model can legally score earlier CAL endpoints with its shorter history, but the comparison retains the native control's original mature axis beginning at 446,173; the extra predictions are archived rather than substituted into calibration.

Pointwise Average Precision (AP) and ROC-AUC are computed on the exact common observed axis. AP is not renamed trapezoidal PR-AUC. Strict pointwise precision, recall, and F1 use the frozen calibrated threshold, without point adjustment. Three-consecutive-point confirmation is causal: confirmation occurs at the third scored positive point, without retrospectively marking the first two as detected. Both confirmed event discovery and confirmed anomalous-point coverage are reported.

Event evaluation uses the original event boundaries and a fixed one-to-one matching rule between predicted alarm fragments and true events. Precision, recall, and F1 under that matching are distinct from any-hit recall, which is reported separately. We report the first detection delay, undetected-event censoring, normal false-positive points, and false-alarm fragments per 1,000 observed normal points. Within-event coverage, predefined long-event steady-state macro and pooled coverage, and the longest consecutive miss quantify persistence rather than onset alone. Physical-time delay is supplied only when the sample interval is reliable. Unscored points are unknown, not automatically normal or missed. Discontinuous query segments are evaluated with their boundaries intact.

#### Reproducibility, uncertainty, and decision criteria

The initial complete candidate and matched controls use the fixed first seed 2025. Configurations record source and data versions, role manifests, actual training updates, selection decisions, parameter counts, runtime, peak memory, and failure costs. Complete continuous scores, original timestamps or indices, labels for evaluation, and calibration inputs are retained. Numerical checks verify inverse reconstruction, the analytic log-determinant on a minimal input, finite gradients, causal endpoint visibility, and equality between the implemented training criterion and evaluation score. These are validity checks, not evidence of detection improvement.

TEP and PSM reuse the frozen normal training draw plans and maximum update counts of the cached target-trained comparison (3,760 and 12,400 updates). SWaT uses the same maximum of 20 epochs and normal-SELECT patience of three; actual stopping and selection can differ between objectives. The new SWaT implementation halves its learning rate after each epoch. The historical native implementation's type-1 schedule may have a one-epoch phase difference, so this row is not described as an identical per-update optimizer trajectory. The three new learned variants share their actual schedule and selection rule exactly.

A single seed on exposed development data supports screening, not a statistically confirmed generalization claim. Event-level paired summaries retain every event, while TEP uncertainty uses simulation run as a clustering unit; overlapping points are not independent replications. If the candidate demonstrates useful detection, subsequent confirmation freezes the implementation before new independent entities or runs are opened and includes repeated seeds. No significance claim is made from an event count that overstates the number of independent operating runs.

The candidate is supported only if an improvement relevant to weak or sustained anomaly detection can be attributed to the learning criterion or its temporal/multiscale structure and is not purchased through a severe increase in normal false alarms. It need not dominate every dataset or every metric. Conversely, a small AP change alone, an overall likelihood rescaling, improved training loss, or a threshold-only gain does not establish a methodological contribution. If the fixed complete design lacks a useful effect relative to the mature methods and simple controls, this candidate is stopped without a sequence of post hoc score, threshold, or architectural adjustments.



### Appendix B: closed shallow TimeRadar adaptation branch

#### Completed adaptation-branch evidence (17 November 2026)

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

### Appendix E: initial seed2025 controlled-exposure result, retained

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


### Completed control: standard supervised TimeRadar readout

The source-supervised detector was compared with a conventional MLP readout on the frozen official TimeRadar representation. For each of ten copies and each variable, the real and imaginary 20-patch features were concatenated and flattened to 10,240 coordinates. A shared 64-unit GELU MLP produced a logit before copy/channel averaging. The loss operated on the resulting window score, with positive, paired-normal and nominal-normal softplus terms weighted 0.5, 0.25 and 0.25. The source event labels supplied no per-variable fault annotation. The head reused the seed-2025 source sampling plan, with five FIT and four SELECT events, and looped that plan at most ten times. Learning rates 1e-4 and 1e-3 used identical initialization. Source-only balanced BCE selection across all 2,400 updates chose lr1e-3, step40; a selection restricted through step240 chose identical weights. The complete search and selected-window exposures are disclosed separately. The baseline changes the learning objective, trainable component and inference criterion together.

The selected head learned the source task: SELECT loss improved from 0.697998 to 0.458485 before later overfitting. Its full-axis AP was 0.50494 on PSM, 0.28168 on MSL27, 0.28896 on SMAP53 and 0.37652 on SMD28. Existing supervised-V AP on those same axes was 0.57717, 0.35412, 0.38272 and 0.44206. The four ROC and VUS comparisons also favored the existing V candidate. These single-seed results show that this standard supervised head did not reproduce the candidate's cross-task ranking. They do not isolate variance learning from the choice of updated parameters or establish a general limitation of discriminative readouts.

Operation points still showed costs and benefits. PSM MLP three-confirmation normal FPR was 2.42%, compared with 5.92% for the candidate, while recall fell from 37.33% to 24.27% and one-to-one events from ten to seven of 48. PSM's attack-only CAL cannot certify a normal false-alarm budget. SMD MLP had fewer false fragments and higher event F1, but lower recall and steady coverage. SMAP MLP had greater steady coverage with more false fragments and fewer matched events. NASA used sensor zero and per-entity small CAL; SMD normal training was part of the source pool. Entity-level and complete-event results, fixed-score controls, external common-axis comparisons, source curves and compute costs are retained in the [completed supervised-readout study](../../docs/experiments/native_supervised_mlp_20261123/RESULTS.md). The current candidate remains source-supervised with task-dependent ranking benefits. The closed U/F results and the earlier NASA absolute-performance and SMD limits remain part of its evidence boundary.


# Appendix C. Prior causal common-axis panel

This panel retains the previous causal comparison and its exact axes. It is distinct from the new intersection of rolling endpoints, W100 complete-window output and external coverage. Its values are not substituted into new paired differences.

#### Complete public detection comparison


All entries are local measurements. PSM uses [512,87744), 87,232 points; SMD uses [99,192 floor(N/192)) within each machine; NASA uses [99,W floor(N/W)) within each entity, W96 for MSL and W192 for SMAP. Entity means are unweighted within NASA and SMD. These common points align outcomes while preserving distinct source supervision, target-normal training and offline context permissions. Each row below uses seed2025; separate repetition estimates follow.

#### AP

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.53627 | 0.44479 | 0.34078 | 0.36502 |
| Official TimeRadar E | 0.38751 | 0.42912 | 0.38602 | 0.42477 |
| Official fixed V+E | 0.41957 | 0.43882 | 0.39507 | 0.41422 |
| Reconstruction continuation / V | 0.54007 | 0.45079 | 0.34792 | 0.36788 |
| Source-supervised native V (candidate) | 0.57763 | 0.44266 | 0.36355 | 0.38185 |
| Source-supervised E / E | 0.52266 | 0.42896 | 0.37895 | 0.39566 |
| Supervised MLP | 0.50667 | 0.37710 | 0.29162 | 0.28740 |
| LEFT | 0.57733 | 0.49807 | 0.50800 | 0.49312 |

#### ROC_AUC

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.73120 | 0.81320 | 0.63182 | 0.64252 |
| Official TimeRadar E | 0.56764 | 0.80530 | 0.69213 | 0.68028 |
| Official fixed V+E | 0.60120 | 0.81205 | 0.70234 | 0.70776 |
| Reconstruction continuation / V | 0.73203 | 0.82049 | 0.63514 | 0.63963 |
| Source-supervised native V (candidate) | 0.76547 | 0.82731 | 0.66980 | 0.62720 |
| Source-supervised E / E | 0.67362 | 0.81391 | 0.68451 | 0.70006 |
| Supervised MLP | 0.72365 | 0.75929 | 0.55546 | 0.53829 |
| LEFT | 0.77558 | 0.87853 | 0.76252 | 0.76284 |

#### VUS_PR

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.48507 | 0.46718 | 0.40974 | 0.40221 |
| Official TimeRadar E | 0.39864 | 0.43357 | 0.46363 | 0.46807 |
| Official fixed V+E | 0.42372 | 0.44273 | 0.47360 | 0.45518 |
| Reconstruction continuation / V | 0.49301 | 0.47729 | 0.41830 | 0.39790 |
| Source-supervised native V (candidate) | 0.51410 | 0.43472 | 0.42916 | 0.42903 |
| Source-supervised E / E | 0.50163 | 0.44245 | 0.44674 | 0.43497 |
| Supervised MLP | 0.45480 | 0.31949 | 0.32752 | 0.31115 |
| LEFT | 0.57451 | 0.54180 | 0.55285 | 0.52170 |

#### VUS_ROC

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.66546 | 0.86997 | 0.70357 | 0.69006 |
| Official TimeRadar E | 0.57913 | 0.84872 | 0.76590 | 0.73844 |
| Official fixed V+E | 0.60494 | 0.85519 | 0.77255 | 0.75922 |
| Reconstruction continuation / V | 0.66751 | 0.87535 | 0.70670 | 0.68044 |
| Source-supervised native V (candidate) | 0.67959 | 0.86130 | 0.73865 | 0.67737 |
| Source-supervised E / E | 0.66896 | 0.86086 | 0.75027 | 0.75341 |
| Supervised MLP | 0.60747 | 0.76793 | 0.63193 | 0.58941 |
| LEFT | 0.78509 | 0.91851 | 0.79337 | 0.78490 |

The official, reconstruction-continuation and source-supervised candidates use W100 endpoint observations. The project-trained heads and decoders use real SWaT source labels for selection; the supervised arms also use them in training. Official scores have no project updates. The MLP freezes the representation; the V/E arms update its native decoder. LEFT uses domain/entity normal target training and full-window offline scoring. The project candidate has no target-gradient updates in this panel. NASA has sensor zero only. LEFT independent-CAL operation is NA.


On the NASA LEFT common axes, the source-supervised candidate has seed-2025 AP 0.36355 on MSL and 0.38185 on SMAP, compared with LEFT 0.50800 and 0.49312. LEFT also has higher ROC-AUC and both VUS metrics. Candidate-minus-LEFT AP has a median of −0.06491 across MSL entities and −0.07516 across SMAP entities, with 8/27 and 16/53 wins. Equal-entity average gains over frozen V therefore coexist with a substantial gap to this target-normal-trained offline comparator. PSM AP is nearly tied at 0.57763 versus native LEFT 0.57733; this 0.00030 difference alone is not a substantive superiority claim, and LEFT has higher ROC and VUS. On SMD, candidate AP 0.44266 is below LEFT 0.49807 and below reconstruction continuation 0.45079 on the same points. Training, standardization and context differences prevent attributing these detector-level gaps solely to architecture.

NASA common-axis candidate AP across the three project-training seeds is 0.36176±0.00174 on MSL and 0.38202±0.00219 on SMAP. Reconstruction continuation gives 0.34757±0.00183 and 0.36578±0.00215. These figures were recomputed after intersecting each original score bank with LEFT's scored points. LEFT is a single-seed result without an estimated training SD. The complete [same-axis uncertainty and original-axis repetition tables](../../docs/experiments/left_nasa_author_20261123/REPEATS.md) retain all four ranking metrics and the original SMD repeats.

The shared LEFT models ran 12 epochs on MSL (selected epoch9, 4,140 updates) and 16 on SMAP (selected epoch13, 12,256 updates). MSL's T-9 and SMAP's A-5, A-6, A-8, A-9 and D-12 have no complete entity-local validation window, but remain in test evaluation. The official entry restores the selected state_dict while preserving the final Python running step. Selected/final steps are 3,105/4,140 on MSL and 9,958/12,256 on SMAP; both selected and final states are beyond the original warmup/ramp durations. This runtime behavior follows the pinned entry, with no test-selected alternative. Total new training and native scoring consumed 626.93 occupied A40-seconds, 0.17415 GPU-hours. Its independent-CAL operation remains NA because domain-level normal training and standardization include the historical project CAL regions.
