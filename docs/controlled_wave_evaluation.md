# Controlled wave evaluation sidecar

Independent module `hybrid_tsad.hybrid_v1.controlled_wave_evaluation` is a candidate
CPU-only fixed-bank caller, separate from the immutable readout implementation.
It does not extend the model or change metric formulas. Real wave evaluation has
not been executed by the engineering tests.

## Runtime contract

Default invocation with `--config` prints preview without opening the config,
banks or truth. Execution additionally requires `--execute --release FILE
--trusted-release-sha256 SHA`. Operation is
`HYBRID_V1_CONTROLLED_WAVE1_COMPLETE_FIXED_EVALUATION`.
The root config/release must bind identical permissions
`evaluation/labels=true`, `training/GPU/model/fit/resume=false`, resources
`wall_seconds=900, cleanup_seconds=5, threads=1, GPUh=0`, and the exact
`required_closure()` absolute path/hash map. A separately reviewed outer resource
supervisor must enforce these time/thread limits; this CLI checks authorization
metadata, not operating-system limits.

Config also binds the e56cc3e9 protocol, the complete ordered 70 bank descriptors
and root raw seal using the existing `evaluation_metadata` contract, original
source calibrations, whole `TRUTH_LABELS.npz`, official affiliation identity and
fresh output path. The release binds truth/raw/affiliation path+hash explicitly.
All neural and baseline bank metadata, ranges, raw records, source scorer
valid/reason/score/alarm and tail coverage must pass before the only label
deserialization. `--preflight-only` stops after this label-free gate.

## Delegated definitions

- `controlled_wave.evaluation_metadata/expected_pairs`: all 48 neural and 22
  baseline banks, 81 prespecified comparisons, fixed phases and aliases.
- `scoring.apply_calibration`: original own source recipe and strict threshold;
  TF-off inactive NaNs remain NaNs. Invalid points retain reasons and cannot be
  treated as normal predictions.
- `evaluation.evaluate_arrays/save_evaluation`: full original axis, warmup,
  real gaps, confirmation, events, coverage, delays, normal false alarms and
  official Affiliation; unchanged calculation-input/provenance serializer.
- `named_evaluate_cli.point_pair/paired_events`: same-case fixed comparisons;
  no comparisons between different-truth semantic aliases, no CI/bootstrap.

No phase metric restarts occur. Phase output only counts the existing full-axis
alarm stream. All-normal cases retain original NA/zero rules. S0 is a reference
to its original bank, while alternate all-normal truth results reuse the same
scores and are explicitly non-independent. No additional forward or input
generation takes place.

Formal execution requires existing sklearn/scipy and exact official six-file
Affiliation source (loader-enforced identity/license). No dependency downloads
are performed. Before truth access the caller actually imports sklearn.metrics
and requires the original four metric functions to be callable without invoking
them; finding the top-level package alone is insufficient. Explicit dependency
failure NA returned by the old backend aborts the caller before it saves that
result or claims completion. Legitimate single-class and zero-denominator NA
remain unchanged. Missing dependencies block formal evaluation. Local small-array
tests may expose the accepted evaluator's explicit missing-sklearn NA behavior;
that is not evidence that AP/ROC/Aff were computed in the formal environment.

## Remaining activation steps

Root and independent reviewer must inspect this new candidate, freeze the
complete runtime package bytes (including parent package initializers), bind
actual 70 completed bank identities, original truth and affiliation metadata,
then grant a separate resource/label release. No root release or real truth was
created for this engineering task. The new module has not been exercised on
real banks, real labels, official Aff callback or the final 81 comparisons.
