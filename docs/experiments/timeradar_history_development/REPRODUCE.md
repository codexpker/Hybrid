# Reproduction entry points

The initial architecture is frozen at `6f6cf181daac386d9371048c6d92f690caf5469a`.
The source and U implementation is `hybrid_tsad.timeradar_history_experiment`.
Do not substitute the older LEFT, proxy-ranking, or supervised-V source endpoints.

## Existing environment and assets

The executed project environment uses Python 3.12.13, PyTorch 2.12.0+cu130
and CUDA 13 on NVIDIA A40 GPUs.
The source initializer uses the existing official TimeRadar checkpoint with SHA256
`528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c`.
`native_v_conditioning.load_native` enforces the original code/weight binding.
`intervention_reference.source_records` supplies the existing standardized TEP
FaultFree Training and SMD normal record roles. `left_multisource.target_inputs`
enforces the numerical target normal-prefix/scaler identity.

Raw datasets, upstream weights, derived numeric inputs, new weights and score
banks are author-local assets. They are not uploaded by these commits. Recorded
hashes establish identity; they do not grant outside access to a dataset or weight.
TEP and SMD provenance/access remain as documented in the existing asset records.
The numeric record identities in `contract.json` and `plans.npz` hash in the
receipts are necessary to reproduce this exact draw sequence.

## Source and U

Use the same workspace asset layout as the existing project. Set `PYTHONPATH` to
this checkout's `src`, use an unused output directory and choose physical GPUs
according to availability. Commands below do not implicitly queue new work.

```bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py="$PWD/envs/timeradar/bin/python"
out=/path/to/new_timeradar_history_run
$py -m hybrid_tsad.timeradar_history_experiment prepare "$PWD" "$out"
CUDA_VISIBLE_DEVICES=0 $py -m hybrid_tsad.timeradar_history_experiment source "$PWD" "$out" --arm native
CUDA_VISIBLE_DEVICES=1 $py -m hybrid_tsad.timeradar_history_experiment source "$PWD" "$out" --arm history
CUDA_VISIBLE_DEVICES=0 $py -m hybrid_tsad.timeradar_history_experiment U "$PWD" "$out" --arm native --domain PSM
# Repeat U for history/PSM and native,history/SWAT using the same prepared plans.
```

`selected.pt` stores the model and selected step; `resume.pt` stores model,
optimizer, next step and accumulated traces every 40 updates. Resume uses the
same command and immutable plan. Never reuse the output directory with changed
data or code semantics. Completed receipts are verified and skipped.

## Scoring and metrics

```bash
CUDA_VISIBLE_DEVICES=0 $py -m hybrid_tsad.timeradar_history_evaluation worker "$PWD" "$out"
# Up to four workers may share the same chunk queue, one per physical GPU.
# Run after all four U receipts and all eight score banks are sealed:
CUDA_VISIBLE_DEVICES=0 $py -m hybrid_tsad.timeradar_history_diagnostics "$PWD" "$out"
CUDA_VISIBLE_DEVICES='' $py -m hybrid_tsad.timeradar_history_evaluation evaluate "$PWD" "$out"
python scripts/build_timeradar_history_report.py "$out" docs/experiments/timeradar_history_development
```

Every score chunk stores index, native V, availability, history start and current
start, with a SHA256 receipt. The official reference is generated alongside
native source-Z scoring by reusing its identical frozen encoder and the separate
official decoder. The initial independent full-forward check is bitwise exact.
The score bank assembler checks all expected indices and chunk hashes. Scoring
uses one current window per native forward; scheduling chunks across GPUs does
not batch native order estimation. A `.claim` file is a live worker's chunk
claim. After an interrupted process, only release that claim after verifying the
recorded process is gone; never run duplicate writers for that chunk.

The running scorer was subsequently accelerated by paired frozen-encoder reuse.
`paired_encoder_reuse.json` enables this optional scheduling path. Both native
nondecoder states must be identical. The native forward produces the unchanged
encoder tensor once; the candidate runs its own full history/native forward with
that same tensor substituted at the frozen encoder output. Its conditioner,
decoder and inverse transform remain active. The original independent route is
the default without the flag. Full-copy, full-position bitwise checks are saved
before activation and on the first needed candidate chunk of each worker/task.
Completed independent chunks are reused. The paired ledger records measured
joint chunk time; per-column time apportionment is explicitly labeled and must
not be interpreted as separately measured method latency or exclusive GPU hours.
Scheduling preemption did not interrupt training or alter model endpoints.

AP/ROC use both the fixed sklearn-based evaluator and an independent tie-aware
rank/sorted-recall calculation. VUS uses the existing pinned implementation and
parameters; no separate VUS reimplementation is claimed. Table/figure rebuilding
from `metrics.csv` is a lighter reproduction level than rerunning models.

## Prepared F (not an automatic job)

```bash
$py -m hybrid_tsad.timeradar_history_f prepare "$PWD" "$out"
CUDA_VISIBLE_DEVICES=0 $py -m hybrid_tsad.timeradar_history_f fit "$PWD" "$out" --domain PSM --arm history
# The paired native arm uses the identical frozen support plan.
# A later two-stage ablation can use --initial official: this resets both the
# complete native initial state and the new history path, not just one branch.
```

The offline qualifier alone receives U-FIT support annotations. The learner sees
only fixed numeric footprints and endpoint plans. The recorded earliest legal
normal support block provides only **one** independent eligible normal endpoint
in each domain; repeated draws do not increase that support. Full F endpoints
must not be represented as completed by the disposable entry check.
