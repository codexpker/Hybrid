# History-Conditioned Time–Frequency Reconstruction for Multivariate Time-Series Anomaly Detection

*Active method research draft. The previous LEFT empirical manuscript is preserved in [its byte-preserving snapshot](draft_left_empirical_before_history_development.md); its relative evidence links retain their original base directory. This draft concerns the complete TimeRadar history-context model. Method efficacy remains subject to the paired results below.*

## Abstract

Multivariate reconstruction detectors can use the current window without explicitly modeling the preceding operating history. We investigate whether trainable, past-only multivariate context can improve the native disagreement statistic of a complete time–frequency detector. The model retains the official TimeRadar forward and ten complementary reconstruction copies. A shared-channel encoder represents preceding observations in 32-point patches at three temporal scales, mixes variables at historical patch endpoints, and conditions the complex current representation through cross-attention before the original decoder. Project source learning uses only normal public TEP and SMD records. A matched native continuation receives the same current windows, input-footprint eligibility, normal validation panel and decoder update budget. Target use includes frozen deployment Z and ordinary mixed-target source replay U, with the same native endpoint variance score. The first development comparison uses PSM and SWaT, one project seed, two shared source fits and four independent U fits. Public and isolated QUERY axes are evaluated separately. Final ranking findings are reported from sealed score banks; normal reconstruction improvements alone do not establish detection improvements. Few-label F support and the executable learner are prepared, but full F outcomes are not claimed in this development run.

## 1. Introduction

An abnormal observation may be unusual relative to a preceding process state even when it resembles observations seen elsewhere. Conversely, a change in operating level can be normal given its history. A current-window detector can represent local structure while having no explicit input for longer preceding dynamics. This motivates testing historical context inside a complete detector rather than attaching an unrelated predictive residual to its output.

TimeRadar provides the detector in this study: its official order network, fractional transform, complex representation, complementary masks, decoder and inverse transform remain intact. Our additional path takes strictly earlier observations from the same record and role. It preserves variable-specific temporal patches and source-standardized level and spread, constructs multiscale historical memories and conditions each reconstruction copy's complex representation. It does not load TimesFM weights. TimesFM motivates historical patch and variable modeling, and LEFT motivates multiscale representation; those inherited ideas are not separate originality claims.

The scientific question is whether this complete historical conditioning improves actual anomaly ranking beyond independently trained native decoder continuation. A second question is whether ordinary mixed-target replay preserves or improves that ability. A third, narrower diagnostic asks whether the trained path depends on the window's corresponding history. These require distinct comparisons. Lower normal reconstruction loss, nonzero attention or a score change under bypass does not independently answer them.

The present work defines one implementable detector, a matched source/target-learning comparison and a fixed diagnostic panel. It does not yet claim established weak-anomaly, explicit lag, physical causal or contamination-robust capabilities. Public targets have previous project development exposure; the present comparison is developmental rather than a new blind confirmation.

## 2. Related work and attribution

TimeRadar supplies the complete pretrained time–frequency detector and its native reconstruction-disagreement statistic. Its official code and checkpoint are bound by the existing [initializer](../../src/hybrid_tsad/native_v_conditioning.py). LEFT supplies an independent complete detector and historical multiscale/interaction experience. Its source-frozen and target-normal-trained score banks remain background comparisons with offline W192 permission. TimesFM's historical patch and multivariate attention motivate representation choices, while the implemented historical encoder is small and trainable. Cross-attention, residual conditioning, source replay and two-stage use are established components; the research claim must concern the complete learned detector and the evidence for its historical contribution.

The archived V1 predictor/fusion, frozen TimesFM conditioning, held-out reconstruction and confirmed-delay programs retain their own results. They do not become parts of this model. This model jointly learns the historical conditioner with the deployed reconstruction path, preserves external level information, uses one native score and has an independently trained no-history control. The [method definition](method_definition.md) and [fixed development contract](../../docs/experiments/timeradar_history_development/PROTOCOL.md) provide the direct implementation binding.

## 3. Problem and information permissions

For a current window starting at s, let X contain the 100 observations [s,s+100) and H contain [s-h,s), with h=256 for a TEP run and 512 for longer records. Both tensors preserve the same ordered variables and belong to one original continuous record and role. A scored endpoint t=s+99 has access only to observations at or before t. The current window is processed in full, and its sole published point score is the last point. No current-window value enters the history encoder.

Project source FIT and SELECT use normal TEP FaultFree Training and the normal SMD training records. PSM and SWaT are excluded from that project source stage. This exclusion does not establish that the upstream official pretraining never encountered those datasets. Source scalers are fixed from their original FIT permissions. Target scalers reuse the archived normal-prefix statistics, including normal validation, and provide target statistics permission to all TimeRadar rows. Target parameter zero-update Z therefore does not mean absence of target statistics.

U exposes only the preassigned mixed U-FIT values to optimization. CAL and QUERY do not enter updates or endpoint selection. Complete historical footprints stay within U-FIT during learning and within QUERY during isolated scoring. Support labels used later for F are an independent information permission. They are not silently introduced into Z or U.

## 4. Complete method

### 4.1 Causal multiscale history

Each variable's history is divided into nonoverlapping 32-point patches and embedded in dimension 64. Fixed chronological position encoding and an affine projection of cumulative historical mean and standard deviation are added. The cumulative statistics stop at their patch endpoint. There is no window RevIN that removes historical level.

For scales r in {1,2,4}, average each r adjacent base patches and apply a shared two-layer, four-head causal temporal encoder. At each historical patch endpoint, a shared variable-attention block mixes the variable tokens. Concatenating the three scale memories yields Z with variable, scale and historical-time coordinates. Let u_c be variable c's mean final token across scales. Variable attention describes statistical interaction; it does not identify physical causation or an explicit time delay.

### 4.2 Complex reconstruction conditioning

Let F_mcp be the native complex encoder output for copy m, variable c and native patch p. Its native feature width is 256. The new path computes

$$
Q_{mcp}=W_Q[\operatorname{Re}F_{mcp};\operatorname{Im}F_{mcp}]+u_c,
\qquad A=\operatorname{MHA}(Q,\operatorname{vec}Z,\operatorname{vec}Z),
$$
$$
[\delta_R;\delta_I]=W_OA+b_O,\qquad F'_{mcp}=F_{mcp}+\delta_R+\mathrm{i}\delta_I.
$$

The output projection starts at zero. The original complex decoder, inverse transform and absolute-value output produce ten reconstructed current windows. All native nondecoder parameters remain frozen in this first development version. The new history/interaction parameters and original decoder are trainable. Registered parameters total 1,804,811; the added path has 168,768 parameters and the candidate trains 962,378 parameters. The native control trains its original decoder only. Capacity and actual cost are reported rather than assumed equal.

![Complete history-conditioned TimeRadar](figures/timeradar_history_context.svg)

### 4.3 Learning and deployment statistic

For ten copies, project learning uses

$$
\mathcal L_{\mathrm{copy}}=\frac{1}{10\cdot100\cdot C}
\sum_{m,j,c}(\widehat X^{(m)}_{j,c}-X_{j,c})^2.
$$

Writing E_full for squared error of the copy mean and V_full for sample variance with correction=1 gives L_copy=E_full+0.9 V_full. This identity explains why a reduced normal copy loss must be decomposed: it can change mean reconstruction and disagreement differently. It supplies no anomaly-retention or transfer guarantee.

The unique deployment score remains

$$
s(s+99)=\frac1C\sum_c\frac1{9}\sum_{m=1}^{10}
(\widehat X^{(m)}_{99,c}-\overline{\widehat X}_{99,c})^2.
$$

No forecast error, relation score, bypass-selected output or weighted score fusion is added. Each native forward uses B=1, ten fixed complementary masks, norm=0, eval mode, isolated RNG and the original inverse transform. At inference the actual trained historical path remains active.

```text
Construct one legal record-and-role history/current footprint.
Encode the past once, retaining variable and scale coordinates.
Run the official current encoder with its fixed ten-copy mask plan.
Condition each complex copy representation using the shared historical memory.
Run the original decoder and inverse transform.
Training: accumulate copy MSE over the logical batch, then one AdamW update.
Detection: publish only sample variance averaged over variables at position 99.
```

### 4.4 Two-stage use

The shared source stage presents 16 TEP and 16 SMD windows per update for 240 updates. Fresh AdamW uses learning rate 1e-4, betas (0.9,0.95), eps 1e-8, native decay groups and 24 warmup steps. Source normal SELECT is the same 32-window panel, evaluated at 0/40/80/120/160/200/240. The earliest minimum copy MSE selects the checkpoint. Both target tasks share source endpoints because their complete source tensors, variable mapping, scalers and plans coincide and the model is not tied to one channel count.

Z freezes the source endpoint. U separately restores it and performs 2,000 ordinary replay updates with 16 mixed target, eight normal TEP and eight normal SMD windows per logical batch. Fresh AdamW and 200 warmup steps are matched between candidate and native control. All 2,000 steps and the final endpoint are retained regardless of intermediate Z performance. No target labels enter fitting or selection. Ordinary replay is an established training tool, not the claimed representation contribution.

F is prepared as a direct source-checkpoint use with an isolated public support event and normal block, native normal copy learning and a fixed native-V pair loss. Its separate label and numeric-history budgets are registered before F evaluation. It has not established an advantage in this new model. A future no-project-source ablation must reset both the complete native initialization and the initial historical path.

## 5. Experiments

### 5.1 Matched design and axes

The first development seed is 2025. A source fit presents 7,680 windows; a complete U fit presents 64,000 windows. Logical batch 32 is implemented by accumulating independent B=1 gradients. Both source arms share the official parent, current windows, complete-footprint eligibility and SELECT rule. They differ in the trained historical path and its capacity. A trained candidate's bypass is diagnostic and cannot replace the independently trained native control.

Public Z axes are PSM test-relative [611,87744) and SWaT absolute [495611,944856). Isolated Z/U QUERY axes are [23115,87744) and [608634,944856), respectively. The source Z bank is rescored/sliced on that exact QUERY grid for Z-to-U differences. Missing warmup points are excluded, never zero-filled. PSM has 25 channels and SWaT retains the project's established 31-channel mapping.

AP is primary. ROC-AUC, VUS-PR and VUS-ROC retain their fixed implementation and parameters. Independent tie-aware rank calculations check AP/ROC. Existing LEFT banks are intersected at the same ticks; their full W192 score is only available at the corresponding window end. Identical ticks do not equalize history, source supervision, target training, standardization or availability. No new CAL or threshold is selected. Historical PSM all-attack and mixed-CAL limitations remain; no normal-FPR certification is claimed.

### 5.2 Complete public Z comparison

The full-precision result source is [the development experiment directory](../../docs/experiments/timeradar_history_development/). Tables distinguish official TimeRadar, native continuation, the complete history model, frozen source LEFT and target-normal LEFT. All new weights receive fresh score banks with input footprints and availability.

<!-- history-development-public-results -->

The public Z table will be inserted from sealed metrics. No performance claim is inferred from the completed source losses.

### 5.3 Ordinary mixed-target replay

<!-- history-development-query-results -->

The QUERY table and Z-to-U differences will be inserted from the same sealed indices. Each of the four planned U endpoints remains in the comparison, including negative results.

### 5.4 Source learning and history diagnostics

Both source arms selected step 240 on normal SOURCE SELECT. Candidate copy MSE was 0.9195197355 versus 0.9331550044 for matched native continuation; the identical official initial value was 1.0013698074. Candidate final E_full was 0.8559245733 and V_full 0.0706613292, compared with native 0.8683406464 and 0.0720159198. These are normal modeling statistics, not target detection evidence. The lowest loss at the budget boundary and continued decrease should be retained as a finite-development-budget limitation.

The first real GPU logical update records loss, gradients, peak allocation and terminal V. Zero historical gradients at the first step follow the zero output projection; subsequent steps have actual historical gradients. Sealed source and U models are inspected on the fixed source SELECT panel under correct history, an earlier legal nonoverlapping history where available, and exact bypass. Short TEP runs cannot supply that nonoverlapping-history intervention and remain NA. These interventions establish path participation or correspondence sensitivity only.

### 5.5 F preparation and computational cost

The first legal PSM support event is [3248,3439); SWaT uses [496754,497694). Their complete-event label budgets are 191 and 940 points, alongside 512 labeled normal points. Strict 612-point footprints leave 92/841 legal positive endpoints and only one legal normal endpoint per chosen earliest normal block. Numeric past may include unlabeled anomalies. This is an important future F support limitation, not additional normal evidence. The learner receives fixed numeric footprints and endpoint plans, and disposable in-memory update checks do not count as complete F training.

Source training took 165.69 seconds for the native control and 319.34 seconds for the history model on A40. Further training/scoring costs and exact forward counts are reported with final receipts. Parallel processes can share a GPU; process-seconds must not be mislabeled as exclusive GPU allocation. Scoring cost is measured for this actual B=1 TimeRadar path rather than extrapolated from LEFT throughput.

## 6. Discussion and limitations

The primary test concerns complete detection gains beyond same-budget native continuation. Lower source risk and a nonzero history path are separate engineering/modeling observations. An improvement confined to Z would not establish adaptation retention; a better U result must be compared with native U and with its own Z on identical QUERY indices. Bypass and earlier-history substitutions are post-training interventions, not independently optimized alternatives.

This initial run has one project training seed, historically exposed targets and a short source-learning budget. Target normal-prefix statistics are available. The official pretraining's upstream data permissions remain those of the original method. PSM/SWaT findings do not establish complete-SMD generality, physical lag identification, weak-anomaly specificity, contamination robustness or effective F. The planned SMD source fold must exclude all SMD source records; it cannot silently reuse this TEP+SMD source model as an unseen-SMD model. NASA's existing single-sensor assets remain supplementary rather than cross-variable mechanism evidence.

## 7. Conclusion

This study implements one complete TimeRadar detector with trainable, strictly past multivariate multiscale context and the unchanged native endpoint variance readout. Its first paired source and mixed-target comparison separates normal modeling, complete anomaly ranking and history-path diagnostics. The final empirical judgment is determined by the sealed public and QUERY tables, with all negative effects retained. It does not replace the method-research objective with an empirical LEFT manuscript, or treat architecture completion as a proven scientific contribution.
