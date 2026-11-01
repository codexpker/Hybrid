# TEP partial-observation validation

This package preserves the official LEFT backbone and inference mathematics.
TimeRadar's earlier eight-dataset reproduction is a separate frozen asset; this
TEP experiment does not use TimeRadar weights or claim its official TEP result.

The fixed observation columns are `xmeas_9`, `xmeas_21`, and `xmv_10`, in this
order. The manipulated variable is a feedback-control signal, not a strictly
causal exogenous instrument. All source values remain float64; only the model
input uses the explicitly frozen training preprocessing. FIT, SELECT and CAL
are separate normal runs. No model or threshold is selected on fault labels.

Shared interfaces:

- `tep_faulty_inputs`: published-file identity checks and one RData extraction;
  writes 400 opaque numeric inputs and a separate workflow identity catalog.
- `tep_ridge_readout`: inference with four saved normal Ridge models and ZMAX;
  independent normal-CAL distributions and ten fixed thresholds, with no fit.
- `normal_b1_readout`: the same native B1/H100 scorer for normal calibration and
  explicitly released public QUERY, preserving raw scores and original indices.
- `tep_ridge_query`: deterministic saved-parameter inference on the same QUERY
  endpoint grid, with no label, refit or threshold-selection path.

For 960-point query cases, every complete H100 window is scored at its endpoint
99..959. `last` and `mean32` are both available at that endpoint. The first
Ridge mean32 uses residual times 68..99, not 64..95. Reconstruction at a current
endpoint is not advance fault prediction. All query cases remain separate.

As of 2026-11-02, three normal LEFT trainings and calibrations, the four Ridge
normal fits, normal Ridge/ZMAX calibration, and the actual public-file extraction
have passed bounded artifact review. Fixed QUERY readout has completed; its
native spot checks and final acceptance precede label evaluation. No fault
detection gain is established. Gaussian mechanism controls retained earlier
negative results. The prototype confidence audit concerns frozen checkpoints,
not every training step or a new model improvement.

Use the research workspace's contracts, exact configuration/release hashes,
`CURRENT_RESEARCH_STATE.md` and `results/START_HERE.md` for actual run status.
No universal run command is supplied here: production jobs require those exact
bound artifacts. Preparation refusal and pure synthetic tests must work before
permission to access actual arrays, weights or label evaluation.

The file was parsed historically, so new run selectors do not make this a
strictly untouched dataset or new-device confirmation. Nominal whole-process
fault labels are not proof of changed lag or physical impact in these three
channels. High AP alone, especially with about 92.9% positives, cannot establish
the proposed weak-relation mechanism. Raw inputs, weights and upstream snapshots
remain outside this Git repository.

Evaluation is separated into `engine.tep_fixed_evaluation` (pure event/ranking/
run-cluster statistics) and `engine.tep_evaluation_runner` (hash-bound artifact
loader and outputs). The runner requires all three accepted LEFT score banks
plus the deterministic controls before opening identities or compact scores.
An external root-provided release hash is a workflow trust anchor, not a claim
of cryptographic signatures. Synthetic complete-flow checks are not real
benchmark results.

A segment confirmed before fault injection remains one normal false alarm even
if it continues past injection. It is also reported as a boundary continuation
and never counted as a newly detected fault. Segments only confirmed after the
boundary, with a pre-injection start, are separately reported. Timely recall
(confirmation by sample181) is distinct from full-event precision. Missing
precision and censored delays remain missing, while a positive case with no
alarm has F1 zero. Paired uncertainty uses twenty simulation-run clusters with
the same predeclared resampling matrix, not overlapping points as replicates.
