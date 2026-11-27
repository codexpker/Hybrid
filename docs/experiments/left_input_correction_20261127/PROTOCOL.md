# Bounded training-input correction for multi-source LEFT

This development matrix begins at `6bea2ec2c15446142347d0a7290fabf85897345b`. It tests a shared repair of mixed adaptation inputs before native LEFT builds any view. The two development folds use the frozen 25- and 31-channel source checkpoints. Their names identify width and native configuration. **No PSM or SWaT target samples, labels, scores, or metrics enter this development matrix.** The 38-channel SMD target fold retains its TEP/HAI source split and is reserved for the separate formal matrix. [Source identities and native replay](../left_multisource_20261126/RESULTS.md) · [SMD and window weighting](../left_smd_reliability_20261126/SMD_RESULTS.md)

## Arms and update contract

| ID | Domain-shared operation on the target half of each training batch |
| --- | --- |
| R0 | Identity, ordinary native replay |
| R1 | Residual shrinkage, rho .25 and q .95 |
| R2 | Residual shrinkage, rho .50 and q .95 |
| R3 | Residual shrinkage, rho .25 and q .99 |
| R4 | Residual shrinkage, rho .50 and q .99 |
| R5 | Replace the input with the frozen teacher's full-rate multi-scale output |
| R6 | Clamp marginal values to pooled source reference quantiles .005 and .995 |
| R7 | Existing whole-window reliability weighting, with original input |

For each fold, set `r = X − R0(X)`, where `R0(X)` denotes the frozen model's `X_hat_ms` at the finest scale. The arm R0 in the table is ordinary replay and uses no teacher forward. Finite repair is `Y = X − rho * sign(r) * max(abs(r) − tau_q, 0)`. Threshold `tau_q` is the scalar empirical q quantile over all positions and channel slots from 512 normal reference windows, using linear interpolation. Random channel dimensions do not represent shared sensor identities, so the threshold is not coordinate-specific. R6 uses the same reference windows. R7 uses midranks of teacher point scores over all repeated proxy-support exposures, pointwise `1.5 − rank`, mean-one target normalization within each batch and window-average weights. Source weights are one and the native independent loss terms and coefficients retain their definitions. [Existing weighting implementation](../../../src/hybrid_tsad/left_reliability.py)

All arms receive 16 proxy-target and 16 source windows per step, balanced separately within each half between TEP and SMD. Every job runs 2,000 native updates with a fresh Adam at 1e−4, the fixed source checkpoint, W192, unchanged dropout and curriculum, and one native EMA/Adam update per batch. No endpoint is selected by risk or proxy AP. The finite-repair arm supplies the same corrected input to all views, reconstruction targets, and native EMA selection. The teacher remains in eval mode with parameters frozen, no gradient path and no optimizer membership. Source replay is raw. Detection uses the student native score on original raw standardized inputs and the original time grid. [Implementation](../../../src/hybrid_tsad/left_input_correction.py)

## Roles and synthetic supervision

The archived TEP normal FIT runs 1–4 provide reference, runs 5–12 provide proxy support, and runs 13–20 provide the raw replay half. Runs 61–65 supply QUERY. In lexicographically sorted SMD machine order, FIT records 1–6 provide reference, 7–17 provide support, and 18–28 provide replay. Every machine's archived later SELECT segment is eligible for QUERY with its legal FIT scaler. For each role, the exact record identifiers and time bounds are saved as `roles.json`. The compact `plan.npz` saves the selected record index, local start and ordered columns for every window. The complete footprint is the W192 interval starting there. No role overlap occurs, including the same-window shift donors, and neither manufactured proxy label nor location enters native update.

The sampling seed uses SeedSequence([2025, 20261127, 427, stream, index]). Reference and query draw 512 and 256 windows respectively, half per source. Support and replay each draw 2,000 × 16 windows. Four of 16 support windows per step receive a proxy edit, one amplitude and one shift in each source half, assigned to distinct random support slots. In the QUERY panel each of the 256 clean windows has two independent versions, one per family. The inherited editor acts on ceil(C/4) channels for 20 consecutive interior points with a squared-sine envelope. Magnitude is signed Uniform(1,2) in source standardized units. Relative shift is uniform over ±3/±6 with donor indices inside the window. Clean and edited query instances use matching edit positions. These labels describe only the researcher-defined operations. [Existing editor and source loader](../../../src/hybrid_tsad/intervention_reference.py)

## Evaluation, selection and resources

Compare clean and edited native point scores at the same 20 positions for each family. Compute AP and ROC-AUC per source/family and average AP equally across the four cells in each width fold, then equally across folds. Record clean SOURCE SELECT native eval loss separately per source plus correction RMS, changed clean fraction, fraction of actual edited coordinates changed, residual versus clean after correction, and collateral change outside the actual edit. A small frozen reconstruction residual at an edited coordinate is a diagnostic for cases where teacher agreement suppresses the correction trigger. It does not identify a causal reason for agreement.

The selection is among R0–R4 only. A nonzero arm is eligible if its raw clean QUERY loss is no more than 105% of R0 in every fold/source cell, mean proxy AP improves by at least .005, and neither width fold drops by more than .005 AP. Select the highest mean AP among eligible arms, breaking exact ties by smaller rho then higher q. If none is eligible, retain R0. R5–R7 never enter this selection. Report the selected arm against all three and assess whether any improvement exceeds those controls before arguing for a specific repair effect. No target QUERY result changes this rule. Proxy support editing is one fixed source development task. It is not a real contamination rate result.

The development matrix is 16 adaptation jobs and fixed readouts, with four A40 lanes and a total successful GPU cap of 8 A40·hours. Separate teacher calibration and engineering/failed time. The operational cap is 6,900 seconds per lane, with bounded exit grace. Experiments beyond this source matrix await the scheduled December 3 freeze. No target labels or formal query metrics are needed for this stage.
