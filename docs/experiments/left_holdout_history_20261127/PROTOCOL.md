# Historical information in contiguous holdout LEFT reconstruction

The project tests whether the official frozen TimesFM 3.0 history provides additional detection evidence when all current encoders lose access to the observations being scored. This is a single-seed development comparison on public data. PSM and SWaT have prior project exposure.

## Information and score

For a current window starting at \(s\), the unchanged TimesFM adapter reads \(X[s-h:s]\) from the same record and role. \(h=512\) except for TEP and the archived controlled records, which have 256 valid observations in the existing masked 512-slot structure. Channel ordering, scaler identity, padding, official checkpoint and random encoder are inherited from the [previous evaluation](../left_timesfm_hidden_20261126/RESULTS.md).

Split the 192 current points into eight consecutive 24-point patches. For mask \(r\in\{0,1,2,3\}\), hide patches \(r\) and \(r+4\) across every channel, or 48 points per window. Replace hidden entries with a fixed zero in the previously fitted standardized coordinates. This operation occurs before temporal and frequency encoders, multiscale analysis, fusion, routing, prototype association and all reconstruction paths. A fixed deterministic cycle \(r=k\bmod4\) supplies one mask per optimization batch. Evaluation runs all four complementary masks on each current window.

The normal parameter objective uses the existing upstream coefficients. Let \(D_T,D_F,D_M\) denote time, inverse-frequency and fine multiscale outputs from the masked forward, \(J\) its hidden indices, \(H\) default SmoothL1, and \(\mathcal T\) the native time–frequency transform. The loss is

\[
\begin{aligned}
C&=H(\mathcal T(\operatorname{splice}(X,D_T;J)),\mathcal T(X))
     +H(D_F[J],X[J]),\\
M&=H(D_M[J],X[J])+\alpha_{\rm band}\mathcal R_{\rm band},\\
K&=H(D_M[J],\operatorname{stopgrad}(D_F[J])),\\
L&=\alpha_{\rm cycle,l}C+\alpha_{\rm ms,l}M+\alpha_{\rm cons,l}K.
\end{aligned}
\]

The splice copies visible observations and predicts only the hidden positions. The spectral target, fine target and inverse-frequency target enter the loss after prediction. Supervision on coarse analysis products is omitted because a coarse target can mix visible and hidden positions. The original multiscale encoder and upsampling decoder remain. The prototype confidence, JS criteria and EMA stay inside the native forward; the reconstruction reliability term in EMA therefore uses the **masked** input, with one native curriculum increment and one opportunity to update EMA per training batch. SELECT and scoring run in evaluation mode. Native loss values computed internally on masked inputs are discarded.

For point \(j\), read the unique mask that hides it. The score is the channel mean of
\[
\alpha_{\rm cycle,s}\big[\alpha_{\rm time}|D_T[j]-X[j]|
          +\alpha_{\rm freq}|D_F[j]-X[j]|\big]
     +\alpha_{\rm ms,s}\,H(D_M[j],X[j]).
\]
The three weights come from the corresponding LEFT domain configuration. Neighboring mask runs, prototype entropy, standalone consistency scores and smoothing do not contribute to this readout. Scores become available at the end of the current 192-point window. The method uses visible current observations on both sides of a target patch and is a retrospective window detector.

## Matched training and evaluation

| Method | History supplied to the existing gated cross-attention |
|---|---|
| Holdout LEFT (A) | None |
| Holdout LEFT + pretrained history (B) | Official frozen TimesFM 3.0 hidden sequence |
| Holdout LEFT + random history (C) | Frozen randomly initialized TimesFM of identical architecture |

The two conditioned arms share the initial conditioner as well as the original LEFT parameters. All three restore the initial multisource LEFT checkpoint (25/31 variables), use exactly the archived e59 current windows and channel draws, and continue with 2,000 Adam updates, batch32, learning rate \(10^{-4}\), and fixed source endpoints. The normal SELECT panel uses the archived 1,024 windows and evaluates every mask. The consumed history embeddings are float32; their hashes are checked against the archived forward receipts. The SMD fold, if triggered, uses the matching 38-variable source model and the original TEP+HAI roles. SMD is excluded from project source learning in that fold.

The N comparison begins independently from each sealed source endpoint, uses the archived target normal-prefix scaler and the established chronological FIT/SELECT split, and restricts every 512+192 footprint to its role. Each role uses a W192 grid beginning after its initial 512-point warmup. Match a fixed 20-epoch schedule, batch32, fresh Adam with constant learning rate \(10^{-4}\), all legal FIT windows shuffled within each epoch using seed2025, and complete normal SELECT. Earliest minimum SELECT risk at epochs 1–20 selects each arm. The three holdout models use the all-mask mean of the objective above. A necessary matched-budget native reference starts from the archived e59 native source continuation and uses the same normal windows, optimizer, epoch budget and earliest-minimum selection procedure with the native unmasked objective. Earlier normal-trained LEFT references remain separate because those jobs used stride-one training and native early stopping.

Before test-label access, the complete TEST axes inherit the e59 512-history protocol (PSM \([576,87744)\) and SWaT \([495576,944856)\)). External intersections reuse e59 permissions, including the SMD tail restriction. Preserve original CAL eligibility and operation NA. TimeRadar, DADA, TimeRCD, CATCH, PCA, the zero-training normal-level deviation and native LEFT retain their documented training, statistics and context privileges. Compare AP, ROC-AUC, VUS-PR and VUS-ROC, keeping Z and N separate. For optional SMD, compute the metrics within entity before the equal-weight macro aggregation. Entity variation never substitutes for a training seed.

The fixed controlled bank retains the clean, weak change, smooth control, relative delay, amplitude control and synchronous change streams with the original core, boundary and full regions. Use the same history for each edited/clean pair. The long-record test replaces B's 512-point history by the history from exactly three W192 starts earlier, whose last point precedes the original history. Compare with the same B weight on the eligible subset, excluding the first three starts. Short TEP runs and the single-history controlled records are ineligible for this comparison.

Development expansion requires B to exceed both A and C in AP in both Z and N within at least one prespecified domain, with the smaller advantage at least 0.005 AP in at least one of its stages. This is a resource decision made before target scores and labels are evaluated, with one project seed. A normal reconstruction improvement by itself does not trigger expansion. If eligible, the same pipeline covers all SMD28. F, mixed U and controlled contamination are outside this run.
