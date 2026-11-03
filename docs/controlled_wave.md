# Controlled wave glue

`hybrid_tsad.hybrid_v1.controlled_wave` adds no network, loss, score or metric.
The original004b sidecar and frozen13/23/24 files remain unchanged. The four
formal operations are input derivation, source-only model bank, deterministic
baseline banks and fixed-evaluation **metadata**. Default CLI only previews.
Every execute/preflight requires exact root status/action/config/source/resource
approval; an engineering release cannot authorize a scientific operation.

The protocol is pinned to e56cc3e9 (11cases/run1, N2560/H512/P32). `seed_role`
names latent/audit seeds, while actual Run role is QUERY. One pair is generated
per family/seed-role, not per model or output. The LEVEL view is exactly the old
pair's A; pulse views add .25 or2 to the same float32 N; BREAK replaces only the
u contribution at t-d0 (d0=12) by one independent shared +/- driver. Independent
allowed-normal roles have zero-label ledgers. No alias gets its own inference.
Values/observed/index NPZ and original load_manifest schema have no truth path;
truth/latent bundles and explicit TRUTH_LABELS.npz are separate audit artifacts.

`AnchorModel` delegates to the real original model. It intercepts RETURNED
forecast/reconstruct only at first/middle/last of each nonempty fixed phase.
It never invokes an additional pass, changes masks, detaches the model itself,
or substitutes an attention operator. It checks export RNG invariance and saves
pre-forecast/post-reconstruct CPU/CUDA/NumPy/Python RNG plus legal lag axes,
flags, masks, endpoint tensors and footprint. Invalid candidate -Inf/NaN remain.
The original iterator owns eval/no_grad, arrived windows and residual computation.
The sink is synchronous, so all saved observations belong to this exact pass.
No multi-threaded use of one model/proxy is supported.

`consume_records` uses the old iterator output, old cli.score_records and old
ChunkedBank. It requires finite nonnegative ACTIVE residuals only; TFoff inactive
r_tf/r_var NaN are never filled or rejected. Each model retains its original
sourceCAL weights/scales/threshold. Existing output refuses; interrupted prefix
remains incomplete, and there is no automatic resume or retry. 2048 records give
32 chunks of64; last31 means arrived horizon-incomplete points, not a short chunk.

Source gates pin both frozen23 and the actual local adapter imports scoring.py
and chunked_bank.py to those same old SHA values. No source glob is used. All8
official files and installedFRFT byte identities precede PT loading. External
owned supervisor must additionally enforce fresh UUID/idle, wall/AS/RSS policy,
post-run source/asset hashes and resource cleanup. This module does not replace
that supervisor or itself verify a budget ledger.

Evaluation metadata requires the exact whole70-bank descriptor list (48 model
case banks +22 deterministic banks), all complete identities/2048 rows and own
source scalar replay BEFORE truth I/O. It defines81 comparisons, phase metadata
and aliases without computing a metric. `dispatch_evaluation` passes full2560
labels and warmup512 unchanged to the accepted evaluator; no phase slicing,
confirm reset or NA normalization. Allnormal ranking/Recall/Aff remain NA; FP
Precision/F1 are zero, not NA; zero-GT/zero-alarm Precision/F1 remain NA. Same-bank
different-truth aliases must never enter paired_events across truth identities.
Self post-event alarms remain primary normal FP; pulse steady is NA. The CLI
currently supports only evaluation metadata preflight: final evaluator-runtime,
truth assets and output serializer need an exact later binding, not a fake
completed evaluation. No actual metrics have been run by this engineering step.

Separate TECHNICAL_NORMAL operation supports future exact CPU parity and GPU
throughput releases. Its independently seeded N640 standard-normal inputs have
no mechanism truth and are never formal N2560 cases. CPU parity is three new
singleton passes (fullC1/TFoffC1/fullC3) plus three exact oldV4 references; the
singleton index plan avoids preceding windows. CPU/CUDA state, raw metadata,
float tolerances and reference post-RNG are archived and main-final RNG restored.
GPU cost is full2025 only,64 C1+64 C3 endpoints. Neither operation is currently
released. Their PT/calibration assets stay original and resulting banks are
technical-only, never substituted for formal wave banks.

Engineering tests use only N96 scaled fixtures, original NumPy loader/scorer/
bank, and explicitly fake non-Torch objects for pass/RNG tests. They do NOT prove
official neural parity, CUDA throughput, scientific utility or old collector
root cause. Source/tests are frozen before the <=60s CPU-only test execution.
