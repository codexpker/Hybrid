# Current candidate: conditional energy in frozen TimeRadar response geometry

The current implemented research candidate combines a strictly historical
conditional generator with the frozen official TimeRadar response function.
Only public SWaT31 and PSM25 are used. BAEF-AD laboratory records, weak labels,
experimental runs and enterprise data are excluded. The former LEFT+R results
remain locked external learning controls; they are not replaced or renamed.
No new detection result is asserted while the fixed A/B experiment is running.

Let $h_t=x_{t-512:t-1}$ and let $v_k=G_\theta(h_t,\epsilon_k)$ be four independent
conditional current-value draws. $G_\theta$ uses the official TimesFM3 residual
and mixing blocks at a lightweight configuration, including variate attention;
it does not load official TimesFM3 pretrained weights or its quantile head.
The current observation is excluded from the generator input. A separate
frozen TimeRadar invocation evaluates each $[x_{t-99:t-1},v]$ window with its
own order and ten fixed complementary masks, preserving the native complex
encoding, decoding, iFRFT and magnitude projection [@timeradar2026].

The full geometry is

$$g_{h_t}(v)=\left[v/\sqrt C,\;\operatorname{vec}r_{h_t}(v)/(s_r\sqrt{1000C})\right],$$

where $s_r$ is fixed using only normal SOURCE_FIT responses. The paired raw-only
model uses $g_{h_t}(v)=v/\sqrt C$. Both minimize and report the same energy score

$$s_t=\frac14\sum_{k=1}^4\|g_{h_t}(v_k)-g_{h_t}(x_t)\|_2
-\frac1{24}\sum_{k\ne l}\|g_{h_t}(v_k)-g_{h_t}(v_l)\|_2.$$

This is half the convention in the authors' generative scoring-rule code
[@generative_scoring2024]. Conditional generative energy training is an
established method; the hypothesis here is that fixed adaptive time-frequency
response geometry supplies useful detection evidence beyond the matched raw
geometry. The raw coordinates preserve injectivity, supporting strict
properness of the population score under finite first moments. This does not
guarantee abnormal observations have high score, calibrated normal false-alarm
control, or robustness when mixed adaptation learns abnormal states.

Only the generator is optimized. Generated-input gradients pass through the
frozen response computation; no_grad is restricted to observed responses.
Separate B=1 calls avoid sharing the fractional order across independent
windows or draws. eval() does not fix native random masks: isolated seed84551
is explicitly restored per response call, while generation noise has its own
paired streams. The two arms share source initialization and plans,2000 source
updates with normal SELECT selection including step0, then2000 fixed-endpoint
1:1 source-replay updates. They retain separate noU and R states. These are
same-domain normal initializations and natural mixed adaptations, not a new
heterogeneous pretraining experiment. Replay itself is not an originality
claim. The candidate uses logicalB16; historical LEFT+R usesB256, so external
cost/exposure comparisons are not equal-budget objective ablations.

Strict history plus current requires513 valid points. SWaT uses the existing
336321-point common axis; PSM excludes360 additional initial points solely for
this footprint, leaving64728 points. Cached methods are recropped without
changing their original scores. Their availability indices, source statistics
and offline permissions remain explicit. New mixed-CAL points are legal for
the actual footprint and thresholds use q99/higher/strict greater-than in each
model's own units. PSM's all-attack CAL remains a limitation. No favorable
results from old replay or controlled-contamination experiments are inherited
by this untested candidate. The running implementation and fixed definitions
are recorded in [the implementation note](../../docs/experiments/tf_conditional_energy_20261119/IMPLEMENTATION.md).

# Locked external learning basis: native LEFT with source replay R

The locked LEFT study addresses adaptation and detection retention of time–frequency multiview detectors under mixed-unlabeled target data. Its empirical basis is complete native LEFT plus established source replay R, not a new backbone or an original replay mechanism. The native objective, prototype/curriculum state updates, retrospective full-window inference and per-model CAL remain intact. P/V is stopped and retained below solely as historical evidence; no preservation term or teacher entered that completed replay experiment.

The natural-mixture and additional-exposure experiments are locked after two tasks and three seeds. A completed seed2025 external CANDI comparison retains its official MLP, full FPM and input/output SANA, with source training and source-SELECT references, chronological U-FIT adaptation and frozen CAL/QUERY. It does not modify this R rule or become a Hybrid module. Its own noU comparison establishes no useful net adaptation increment in the fixed task; cross-backbone ranking advantages and coverage costs are reported separately. CANDI's10-point endpoint MSE, FIT-only scaler and update schedule differ from native LEFT's192-point retrospective outputs and original scaler. See the [full fixed comparison](../../docs/experiments/candi_native_a_20261118/RESULTS.md); no follow-up method, seed or B/C matrix is automatically opened.

## Completed cross-backbone control, 19 November 2026

The fixed plain-MLP comparison adds only TT/R at seed2025 in SWaT and PSM, initialized from the accepted source MLP without SANA/FPM. It reuses each original LEFT192-point exposure endpoint with a10-point suffix, freshAdam1e-4/defaults/B256/2000 endpoints, native MSE and unchanged mixed-CAL rules. R−TT ranking gains in both backbones support general replay, not a frequency-specific invention. MLP is cheaper and strong in ranking but retains extreme SWaT false alarms and low PSM sustained coverage; LEFT has useful full-configuration coverage/operating trade-offs with higher cost. Different context, scalers and source losses preclude a pure representation attribution. [The complete fixed comparison](../../docs/experiments/mlp_replay_20261119/RESULTS.md) closes this task without adding methods or changing the R rule.

## Matched additional-exposure construction

For each domain, A is the unchanged natural U-FIT plan: $R_A=[T,S]$, $TT_A=[T,T']$. R and TT share T position by position; T′ is an independent target draw and S is actual source FIT. With 2,000 updates and 128 windows per half, T has 256,000 exposure slots and each full plan has 512,000. These counts measure repeated training exposures, not independent observations.

The current factorial comparison is $\{TT,R\}\times\{B,C\}$. A constructor may inspect labels only within current U-FIT to qualify full 192-point abnormal donors (B) and full 192-point normal donors (C). It replaces the same 25,600 shared-T slots in both conditions and both arms: 10% of T and 5% of all exposures. Within each condition, donor draws are paired across R/TT. T′ and S do not change. Current-U-FIT eligibility is confirmed: SWaT has3,468 all-abnormal /82,903 all-normal windows with nine qualifying abnormal event intervals; PSM has649 /12,852 windows with five qualifying intervals. Repeated overlapping windows are not independent events. Achieved exposure and donor reuse are audited separately from training.

The learner remains label-free: it receives windows, runs the original native forward/loss and updates without donor labels, anomaly weights or label-based skipping. Construction is nevertheless label-assisted and must be disclosed. The natural background may already contain anomalies. Consequently, matching added abnormal exposure does not equate total contamination between R and TT; neither B−A nor the slot fraction is the total anomalous-point proportion. Donors must be role-contained and valid at the complete-window level. There is no QUERY-based donor selection.

## Paired estimand and reporting

For a metric $M$, arm $a\in\{R,TT\}$ and condition $k\in\{A,B,C\}$, let $M_{a,k}$ denote the evaluated outcome. Report absolute values and all within-arm B−C, B−A and C−A differences, followed by

$$I_M=(M_{R,B}-M_{R,C})-(M_{TT,B}-M_{TT,C}).$$

B−C estimates the controlled abnormal-versus-normal replacement contrast under the shared slot plan; C−A checks normal replacement/composition effects against the original plan. Interpret $I_M$ in each metric's direction and together with absolute ranking, event discovery, sustained coverage and normal false alarms. The experiment does not require B to harm either arm and cannot turn an unchanged detection floor or a favorable difference of differences into proof of robustness. It neither identifies unknown contamination in deployment nor certifies difficult-normal protection.

Initialization, native training schedule, original role permissions, update budget, shared draws, inference and CAL rules are held to the matched contract. Each resulting model is calibrated on the same authorized CAL locations in its own score units; CAL is not a learning input. PSM's previously verified all-attack CAL remains a limitation, not a normal 1% false-alarm certification. The fixed replication is now complete at seeds2025/2026/2027:16 additional B/C jobs, with all old assets reused. Each seed uses its own accepted A plan and the same isolated replacement rule, giving joint adaptation/intervention-sampling replication at a fixed native initial model. Pairwise effects are computed within seed, then mean and sample SD(ddof1,n3); noU remains n1/SD=NA.

SWaT has positive I_AP and I_C3 complete-event coverage in all three seeds, but effect magnitudes vary and no additional event is found. Direct R-B has fewer false-positive points than TT-B despite the positive mean FP interaction. Long macro coverage remains low and the pooled denominator is dominated by one long event. PSM B-C AP is positive for both arms in every seed, so there is no corresponding ranking-specific contamination penalty; positive coverage interaction does not reverse its worse absolute event/coverage trade-off. See [all fixed results](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md).

The learning rule remains one full native LEFT forward on concat(T_k,H_a), with H_R=S and H_TT=T-prime; global/native state terms are not replaced by an average of separate forwards. No preservation term, teacher or inference change is introduced. The source-replay control and native LEFT retain their attribution. The experiment is locked without automatic additional seed/rate/loss. Earlier natural-mixture evidence and stopped definitions remain below and in the draft; these findings do not establish general cross-domain/field robustness or an original Hybrid mechanism.

# Historical stopped P/V source-output preservation experiment

This historical section defines the completed P/V experiment, evaluated at seed2025 on SWaT and PSM. It added a training constraint to native LEFT source-replay R and is now stopped. Its formulas and results are retained, but neither term defines the current learning rule. It is separate from the stopped density design archived below. Neither functional preservation nor replay is claimed as an invention.

## Frozen teacher, source permission, and outputs

For each domain, let $\theta_0$ denote its fixed native pretrained LEFT checkpoint and $\theta$ the adapting student initialized from it. A frozen teacher with parameters $\theta_0$ reads only the authorized source FIT windows $S$; it does not read target FIT $T$, SELECT, CAL, QUERY or labels. R trains with the native objective on concatenated target/source batches $[T,S]$. Each preservation term is evaluated on the same source half, without replacement by newly selected examples.

For a source batch of $B$ windows of length $T_w=192$ with $C$ channels, native `_forward` returns three reconstructions with identical shape $B\times192\times C$ and the same input-coordinate units:

$$g_1(\theta;S)=\widehat X,\qquad g_2(\theta;S)=\widehat X_{\leftarrow f},\qquad g_3(\theta;S)=\widehat X_{\mathrm{ms}}.$$

These are respectively `X_hat`, `X_hat_from_S` and `X_hat_ms`. They are outputs of interacting, shared native pathways, not three independent views or three independently trained teachers. Define $\Delta_v=g_v(\theta;S)-\operatorname{stopgrad}g_v(\theta_0;S)$, $\overline\Delta=\frac13\sum_{v=1}^3\Delta_v$, and $D=BT_wC$. The squared norm below sums all batch, time and channel entries. With all outputs evaluated in the same source coordinate system,

$$P=\frac{\|\overline\Delta\|_F^2}{D},\qquad V=\frac{1}{3D}\sum_{v=1}^3\|\Delta_v\|_F^2,$$

$$\mathcal L_P=\mathcal L_{\mathrm{native},R}+\lambda P,\qquad \mathcal L_V=\mathcal L_{\mathrm{native},R}+\lambda V,\qquad\lambda=1.$$

Here $\mathcal L_{\mathrm{native},R}$ is the complete native LEFT loss on $[T,S]$, including its original weighted components and state updates. The added terms constrain source outputs, not their agreement with ground truth or their deployed anomaly scores. In particular, native reconstruction heads need not be accurate merely because their teacher responses are preserved.

The finite-dimensional variance identity gives

$$V-P=\frac{1}{3D}\sum_{v=1}^3\|\Delta_v-\overline\Delta\|_F^2\geq0.$$

To verify it, expand $\Delta_v=(\Delta_v-\overline\Delta)+\overline\Delta$; the summed cross term vanishes because $\sum_v(\Delta_v-\overline\Delta)=0$. P permits cancellation between pathway drifts; V also penalizes their dispersion. Equality holds exactly when all three drift tensors coincide. At the same $\lambda$, V is therefore pointwise at least as strong as P; this is not an equal-strength comparison or a clean isolation of independent-view information. A V advantage alone would not identify which of stronger preservation and reduced cancellation caused it. The identity imposes no ordering on downstream AP, false alarms or event coverage.

## State-safe differentiable preservation pass

The replay training pass retains native training behavior. The additional student preservation pass uses eval behavior while retaining autograd through its outputs; the teacher uses eval behavior and `no_grad`. Around these auxiliary forwards, execution snapshots and restores mutable buffers, the native running `step`, every module's individual train/eval flag, and CPU/CUDA RNG states. Restoring only the root mode would destroy heterogeneous child modes. Eval alone is also insufficient because native forward code can update non-parameter state. Restoration must preserve the student's autograd graph; a successful backward and frozen-state check are execution requirements, not detection evidence.

The auxiliary forwards must not advance the replay RNG trajectory or permanently update prototypes/curriculum state. Native R and P/V share initialization, replay draws and update schedule; teacher and auxiliary student forward/backward costs are reported separately. Matching optimizer steps does not imply matching total computation. The teacher and auxiliary terms are absent at inference: the native score, native readout permission, CAL locations and threshold rule remain unchanged, with each model calibrated in its own native score units.

## Prior work and falsifiable scope

Replay and teacher-response preservation are mature ideas. [Buzzega et al., NeurIPS 2020, §3/Eq5](https://proceedings.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) uses stored historical logits for Dark Experience Replay. This experiment instead fixes one domain-specific native reconstruction teacher and compares two aggregations of source-output drift. It does not reproduce DER, its reservoir, classification objective or reported benchmarks. The native interacting reconstructions belong to LEFT, not to a new backbone.

The hypothesis is that preserving source reconstruction functions can improve the replay adaptation trade-off without changing inference. Its evidence must come from V−R, V−P and P−R under fixed training and evaluation. Ranking, coverage and false-alarm effects are reported jointly; an aggregate-only improvement belongs to P, while a V advantage must be interpreted with its stronger preservation constraint. All four runs and bounded key-result review are complete. V has not established incremental benefit over R or P. P retains a SWaT ranking positive and PSM event-coverage trade-off, with PSM ranking loss and higher false positives. See the completed tables and costs in [the result report](../../docs/experiments/left_source_preservation_20261118/RESULTS.md). No automatic variant follows. Contamination identification and general robustness require their own evidence.

# Historical stopped design: alternating-domain joint normality density

**Historical implementation snapshot:** `d77252e17509d263d6d87a64c98b79516a935fd3`, `src/hybrid_tsad/normality_density.py`. The following mathematical account is retained for traceability of a stopped candidate, not the current LEFT-plus-replay definition or an established efficacy claim.

## Input and invertible computation

The observed window at endpoint $t$ is $X_t=x_{t-H+1:t}\in\mathbb R^{H\times C}$. FIT-only channelwise standardization preserves level relative to FIT. There is no windowwise centering or QUERY-fitted scale. The input layer is a learned invertible channelwise location–gain transform,

$$v_0=(\widetilde X_t-o)\odot e^g,\quad g=2\tanh(\gamma/2),\quad \log|\det J_0|=H\sum_c g_c.$$

Each of four coupling layers has a fixed checkerboard mask $m_{h,c}=(h+c+k)\bmod 2$, with parity $k$ alternating across layers. A conditioner receives $m\odot v$ and the mask; it does not receive transformed active coordinates. Its width-64 temporal network uses pointwise channel mixing and four residual kernel-three convolutions with dilations $(1,4,16,64)$. Let its outputs be raw log-scale $a$ and translation $b$. Then

$$\bar a=(1-m)\odot2\tanh(a/2),\quad \bar b=(1-m)\odot b,$$

$$u=m\odot v+(1-m)\odot(v\odot e^{\bar a}+\bar b),\qquad \log|\det J_k|=\sum_{h,c}\bar a_{h,c}.$$

The inverse is explicit. The conditioner depends only on the unchanged masked partition, which makes the determinant expression exact even though it uses temporal convolutions and channel mixing. Zero initialization of each conditioner's last projection starts its coupling at identity; this does not imply that the full time–wavelet model has the same initial density as a time-only identity transform.

The full forward sequence is affine input transform, coupling 0, Haar analysis, coupling 1, Haar synthesis, coupling 2, Haar analysis, coupling 3. Two-level orthogonal Haar transforms act on time, leave channel count unchanged, and preserve both dimension and total energy. Adjacent pairs $(a,b)$ map to $((a+b)/\sqrt2,(a-b)/\sqrt2)$ recursively on the low-frequency block. Because $H$ is divisible by four, no padding is needed and $|\det W|=1$. No hidden projection or discarded scale requires an approximate Jacobian.

The historical parameter counts are 200,990 (TEP, three channels), 223,738 (PSM, 25 channels), and 229,942 (SWaT, 31 channels). Full, time-only, and local variants have equal counts within a dataset. Time-only omits the fixed Haar changes of coordinates; local retains them but uses conditioner dilations $(1,1,1,1)$. The latter has local dependence per conditioner; it should not be claimed that composing coupling layers produces a perfectly independent nine-sample model.

## Learning and detection

With $D=HC$, $z=f_\theta(\widetilde X_t)$ and independent standard Student-t($\nu=5$) latent coordinates, the implemented coordinate prior loss is

$$r(z_i)=\log\Gamma(5/2)-\log\Gamma(3)+\tfrac12\log(5\pi)+3\log(1+z_i^2/5).$$

The exact standardized-window negative log density and endpoint score are

$$\ell_\theta(\widetilde X_t)=\sum_i r(z_i)-\log|\det J_{f_\theta}|,\qquad s_t=\ell_\theta(\widetilde X_t)/D.$$

FIT minimizes the mean of this same normalized quantity, and SELECT chooses by normal-reference NLL. The score is assigned to the window endpoint when $x_t$ arrives. No future sample, target-label selection, prediction residual, nearest-reference lookup, moving average, or score mixture is used. A channelwise intermediate returned by `score()` must be averaged across **all channels** to recover this joint criterion. After Haar and cross-channel mixing, intermediate allocations are not per-variable marginal probabilities. Scoring only XMEAS9 would no longer be the stated joint density; the TEP joint three-channel detector is therefore disclosed separately from target-XMEAS9 Ridge residuals.

The temporal conditioner uses left padding, but this is a computational property rather than proof of directed physical causal structure. Haar coordinates combine already observed timestamps within the window. The complete observed-window likelihood is endpoint causal because its latest raw observation is $t$; it is not a prediction of $x_t$ from $\mathcal F_{t-1}$. The compatibility input of length $H+1$ is sliced to its last $H$ positions, which requires raw endpoint alignment to be verified at the caller.

## Established components, new hypothesis, and limitations

Affine-coupling density estimation is mature (Real NVP); wavelets inside flows and explicit scale modeling are mature (Wavelet Flow); density-based multivariate anomaly detection is mature (GANF and MTGFlow). This candidate does not claim those ideas, heavy-tailed priors, channel mixing, or dilation as inventions. The empirical hypothesis is narrower: **alternating temporal and multiresolution transformations with one training-and-detection likelihood may preserve useful sustained and asynchronous anomaly evidence better than time-only/local alternatives and mature reference methods.** A parameter-matched ablation and real detection result are required, not merely improved likelihood or active gradients.

The design differs from stopped fixed-reference/nearest-memory models because it learns an invertible distribution; from blind-point models because no unobserved replacement target defines its readout; and from shared predictive reconstruction because the same likelihood is both the training objective and deployed score. These differences establish a new hypothesis, not an expected positive result. High likelihood is not semantic normality, known operating changes can cause false alarms, a whole-window score can dilute weak local changes or delay discovery, and likelihood gains may reflect scale rather than anomaly separation. Normal-state controls, coverage, delay, false alarms and strong simple baselines remain necessary.
