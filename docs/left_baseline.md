# External official LEFT baseline adapter

This adapter is independent of the frozen Hybrid model. It does not redistribute
LEFT source, data, or weights. The official external revision is pinned to
`3fadb4811797075e233076b9faceb0d8ec59b697`; the source's software license remains
unverified. Neither packaging nor a successful synthetic check grants a license.

One entry point supports `profile`, `train`, and separately approved R-only
`score`; C and label evaluation are refused. Preview does not import Torch or
load arrays (score inspects only metadata and the query NPY header):

```bash
PYTHONPATH=src python -m hybrid_tsad.left_baseline --config /external/profile.json
```

Execution additionally requires `--execute` and a root-issued approval matching
the exact config SHA and adapter SHA. A template or PENDING approval is not a
release. The caller must retain the whole process OS exit/timing/log and enforce
the approved Linux timeout; the adapter also uses SIGALRM after approval.

The B128 profile uses the unchanged author model/PSM config, seed2025, Adam1e-4,
three synthetic updates, and no checkpoint. It does not rerun B2 or estimate
whether B128 should be skipped from B2's cold timing. A 20GiB allocator limit and
peak reset are set before model construction. Actual module paths/hashes, GPU
UUID, cold/subsequent times, losses, finite gradients and full-call peaks are
recorded. It does not test 1000-step warmup, contamination robustness, or AD value.

The train adapter inherits the official Exp's train/validation/optimizer/LR/early
stopping methods. Only data access, test refusal and observational logging are
adapted. It calls `train()` directly rather than `run.py`'s automatic `test()`.
The original `_basic()` is retained: thop is needed, and its RNG/modes/step/state
effects are measured, not silently removed. No dependency is installed by this
adapter. The optimizer logging hook uses the currently audited Torch2.8 API; it
does not clip, reweight or alter gradients.

R preprocessing accepts only a SHA-pinned float64 raw prefix `[132481,25]`, fits
StandardScaler on that entire prefix, splits 80/20, and casts each sliced window
to float32. The C-order cache is first copied to the author's dict-dataframe
Fortran layout before scaler fitting; a high-offset synthetic counterexample
shows that leaving C layout can change float32 windows. This includes validation
values in scaler fitting and is disclosed as
official behavior. Train is stride1/shuffled, validation stride192/unshuffled;
drop_last=False. Validation retains the author's unweighted mean of batch means
(138 windows split128+10). Dummy labels have shape[192,1] and are never used by
the official train/validation loss. The train operation has no query loading branch.

`prepare_prefix()` creates a separate raw derivative and lineage manifest; it
does not fit a scaler or read a test/label file. Finite float64 cache and provenance
checks do not prove that this is the author's exact CSV version. Raw means
unstandardized, not necessarily unprocessed: historical source missing-value
filling remains a separate provenance question. No implicit imputation is
performed by this adapter. C, real query readout and label/threshold operations
need separate execution contracts. A training-origin
calibration holdout is not automatically verified healthy, and an empirical q99
does not guarantee1% real-world FPR.

The R-score candidate loads the root-frozen epoch20 checkpoint and saved R
scaler. It never fits on query. Fresh-model strict=False load accepts exactly the
86 declared thop statistic keys and zero missing scientific keys; every other
unexpected key fails. Eval/no_grad and full state/step/mode hashes guard the four
B128/128/128/73 inference calls. Original scores remain per-channel float32;
canonical mean follows the author's CPU NumPy concatenate, reshape, np.array,
mean(axis=-1) sequence, with no GPU-mean substitution or thresholding.

Raw scores and absolute time/window/channel indices retain457 windows/87744
scored points, plus a separate97-point unscored tail mask. Window input_end and
batch_end specify the earliest available observation index: retrospective
localization is not early warning. Profile/train forbid query permission, score
requires it explicitly, and every operation forbids label access. The entry has
no AP/F1/evaluation branch. The candidate's existence is not a real-query release.

CPU tests use synthetic arrays and a mocked synthetic dataframe for the external
official loader, checking exact scaler/window equivalence, role footprints,
float32 output, identities and refusal paths. No model is constructed by tests:

```bash
CUDA_VISIBLE_DEVICES='' LEFT_OFFICIAL_ROOT=/external/fixed_left PYTHONPATH=src \
python -m unittest discover -s tests -p test_left_baseline.py -v
```
