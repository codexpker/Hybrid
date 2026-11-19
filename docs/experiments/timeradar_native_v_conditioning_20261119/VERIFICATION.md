# Final bounded verification

- Fixed protocol `1be5e0001ae80b787302c2fc62f58860e101cd11`; implementation first pushed `f53992b52d6554b1dd1912fccdf149c3112502bf`. Actual cache/training runtime bytes are independently bound in ARTIFACT_MANIFEST, not inferred from Git line endings.
- Affected model/evaluation tests: **13 PASS**. This is not a full-repository test claim. Two true B128 source GPU gates verified gradients, C/D initial encoder equality, no current input, fixed native state/RNG, and the zero-head reference.
- Six normal source trainings completed their full20epoch budget. Twenty-one source SELECT observations/model include0; earliest strict minima are SWaT14/20/20, PSM16/7/19. Six selected/endpoint/optimizer identities, plans and updates were checked.
- Shared caches:8 domain/role banks, terminal10copies/nativeV/actual order/maskbits/observations/scaler/full record identity and footprints retained. Independent NumPy variance recomputation matches the native FP32 score at rtol1e-5/atol1e-7, maximum absolute difference6.30095e-7; **this is tolerance agreement, not bit equality**. B1/no cross-sample order and all-role mask identity verified.
- Sixteen final banks: exact archived scalar score formula, shared V/actual and per-point q/width/floor or D residual. Step0 full-axis A equivalence: zero inversions/new ties/split ties/strict-alarm differences, identical AP/ROC.
- Independent role-separated reviewer recomputed25,227 fields, maximum metric/formula difference1.11e-16, no scientific differences. AP/ROC, CAL order statistic, strict/C3, any-hit/one-to-one, long macro/pooled/miss/pairs/per-event results checked. New8 VUS reused the same pinned official implementation; old external VUS was not rerun. This is same-model-family responsibility separation, **not external double-blind review**.
- Public export:25 ranking rows/23 operating rows/10C3 contrasts/8 manuscript rows correctly rounded; public CSV/curves checked against original exports;150 training-cost fields verified. Missing event-pair `end` export corrected to original `end_exclusive`; the former CSV remains archived. No training/score/threshold/event-metric change. Event P/R/F1 are fractions, percent-marked columns are percentages.
- All34 owned GPU jobs OS0/cleaned: **2.290003054230474 A40GPUh**. No failure or repeated training/inference. Main metric CPU wall78.232773900s, curve wall19.512337600s. Independent metadata+NumPy wall7.8610341s is included in metadata+new8 VUS wall75.6164734s; final comparison CLI6.5665918s and export0.0234999s are separate. Static reads/edits not fully instrumented; these are not an exact total CPU-consumption claim.

Original final audit SHA256:
`505a91bc8eeb0397861566c2e139cf678d2fc98bd9d32a25209bb45d7fa16452`.

METRICS SHA256:
`a3c31f2325174a02fb76bd76486013064c71e83e5909ed90368f9b46e3ce9ccb`.

GLOBAL seal SHA256:
`120dbf7a74280e6a3b717ea57d800dff6ff2c07d5608fb8f8a51b8dd13a3cdd1`.

Original audit/metrics/raw banks stay unchanged. The manuscript's final editorial change replaces “export verification is being finalized” with “including public export verification, is complete”; no scientific value changed. The user's pre-existing two leading spaces in the first Introduction paragraph stay only in their working copy, excluded from this publication. Git may normalize public text line endings; raw byte hashes bind the archived originals, whereas the six binary selected model files are byte-identical exports.

Six new project-trained normal-source B/C/D models are published under the explicit current delivery authorization. Upstream TimeRadar PT, raw data/labels/predictions, full curve arrays and private assets are not uploaded; their hashes and local archive identities are supplied. No official TimesFM PT is loaded or redistributed. All inference is frozen and all results are public development evidence, not new multi-source training, directed-lag proof or mixed-contamination robustness. No follow-on GPU job is active.
