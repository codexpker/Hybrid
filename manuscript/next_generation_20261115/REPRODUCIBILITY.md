# Reproducing the controlled LEFT study

The [active manuscript](draft.md) distinguishes the full public axis, external overlap and mixed QUERY suffix. The [frozen evidence](../../docs/experiments/left_evidence_freeze_20261128/README.md) binds the original metrics to their source rows. The [repeat protocol](../../docs/experiments/left_fixed_exposure_repeats_20261128/PROTOCOL.md) binds the new runs to the archived parents and plans.

| Level | Starting material | Verification in this study |
| --- | --- | --- |
| 1. Rebuild tables and figures | Versioned full-precision aggregate and entity-level tables, plus repeat metrics | Clean-directory rebuild of the frozen tables, paired tables and all three figures. This checks assembly, not ranking metrics from raw scores. |
| 2. Score saved checkpoints | Author-held checkpoints, canonical arrays, original LEFT code/configuration and train-only statistics | All 24 new checkpoints scored on the native W192 QUERY grid. Two AP/ROC-AUC implementations agree on each new bank. VUS uses the frozen implementation only. Historical scores were reused. |
| 3. Refit from data and plans | Canonical arrays and fixed roles, frozen source parents, target plans, teacher coefficients and original model code | All 24 matched U fits executed with archived inputs. An external cold-start acquisition, preprocessing and retraining of every model was outside the verified scope. |

## Level 1 · Archived metrics to camera-ready figures

From the repository root in a clean checkout, install Matplotlib for figure generation. The table builders use Python's standard library.

```sh
python scripts/build_left_evidence_freeze.py .
python scripts/summarize_left_exposure_repeats.py tables . docs/experiments/left_fixed_exposure_repeats_20261128
python scripts/build_left_manuscript_tables.py .
python scripts/build_left_submission_figures.py . manuscript/next_generation_20261115/figures
python scripts/render_left_manuscript.py . /absolute/path/to/chromium manuscript/next_generation_20261115/manuscript.pdf
```

Formatting uses Pandoc and Chromium with MathML. Figures are vector SVG files generated from the same unrounded CSVs used for the manuscript. The table and figure commands need no data, model weights or accelerator.

## Dataset release, preparation and roles

| Dataset | Release and acquisition | Use and input mapping |
| --- | --- | --- |
| PSM | Author release from [eBay RANSynCoders](https://github.com/eBay/RANSynCoders), with train, test and point labels. | The archived version uses 25 numeric columns in original order, excluding timestamp and label. Canonical `data/common_psm_raw_20261029_v1/{prefix,test}.npy` and `data/common_psm_20261028_v1/test_label.npy` are checked by SHA-256. |
| SWaT | SWaT A1 December 2015, available on application from [iTrust, SUTD](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/). Access conditions are set by the provider. | The archived 31-channel numeric version is a selected-channel view, with the absolute time axis retained. Use the original [channel and data binding](../../docs/experiments/left_multisource_20261126/RESULTS.md) and the canonical `preparations/left_multievent_mixed_u_20261111_v1/data/ALL_VALUES.npy` plus its label and scaler bindings. |
| SMD | All 28 machines from the author [OmniAnomaly release](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset). | All 38 variables, per-machine records and labels. SMD is a normal source for the PSM/SWaT folds. The SMD target fold excludes SMD entirely from source fitting and source selection. Save per-machine scores before computing the equal-weight macro. |
| TEP | [Rieth et al., Harvard Dataverse V1](https://doi.org/10.7910/DVN/6C3JR1), file `TEP_FaultFree_Training.RData`. | Normal source only. Export 52 process variables in original order, grouped and ordered within each run. Exclude identity fields from features. Runs 1–20 are FIT and 61–65 SELECT. The checked source tensor is `results/native_tep_support_20261125_v1/fault_free_training.npy` with shape `[1,25,500,52]`. |
| HAI | [Official HAI 21.03](https://github.com/icsdataset/hai/tree/2a814cebc9a66b06c9e5cd545e2d72e65d383737), [CSET 2021 data paper](https://doi.org/10.1145/3474718.3474719). | SMD-fold source only. Use normal train1/train3 for FIT and train2 for SELECT. Preserve file boundaries and discard timestamp and labels. [HAI role, file hashes and statistics](../../docs/experiments/left_smd_reliability_20261126/source_inputs.json) |

SWaT retains the following zero-based positions after removing the time and label columns, in original channel order: `0, 1, 3, 5, 6, 7, 8, 12, 14, 16, 17, 18, 23, 24, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 49`. This mapping is stored as `OFFICIAL_31_CHANNEL_MAPPING.json` alongside the author-held SWaT preparation files.

The role coordinates and canonical input digests for PSM/SWaT are enforced in [trajectory_experiment.py](../../src/hybrid_tsad/trajectory_experiment.py). Training targets and statistics are retrieved by the [native model binding](../../src/hybrid_tsad/left_multisource.py). PSM/SWaT fit target scalers on their full archived normal prefixes, including the normal SELECT segment. SMD uses separate public-normal and mixed FIT-only protocols. The SMD mixed CAL is held out from both fitting and statistics. See the [SMD record binding](../../src/hybrid_tsad/left_smd28_author.py) and [role definitions](../../src/hybrid_tsad/left_smd_targets.py). The processed arrays and released raw downloads are distinct artifacts. Matching release names alone does not certify the project preprocessing.

## Level 2 · Score a saved endpoint and verify ranking

The repeat runner uses LEFT revision `3fadb4811797075e233076b9faceb0d8ec59b697` and the project's matched implementation, including the prototype pre-EMA snapshot. It validates upstream source hashes through `SOURCE_BINDING.json`. The workspace needs the original LEFT checkout at `preparations/left_baseline_build_20261101_v1/upstream`, the validated SWaT configuration under `preparations/left_swat_author_20261118_v1/configs`, and the native scaler and `CONFIG.json` under `results/left_decoder_context_20261124_v1/{PSM,SWAT}/native`.

Use the evaluation environment recorded in [evaluation_versions.txt](../../docs/experiments/left_fixed_exposure_repeats_20261128/evaluation_versions.txt). It additionally requires `threadpoolctl` and the upstream LEFT profiling/import dependencies (including `thop`). The archived VUS implementation is `preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py`, SHA-256 `808a07460a84d31c87edce1d803e1df231111ec53ffa702ee32eb54a79897dfc`. This program uses the native evaluation recipe of window size 100 and 250 thresholds. AP is unadjusted step-wise average precision. ROC-AUC follows the tie-aware rank statistic. For each new bank both reference scalar values were independently checked using grouped sorted scores and rank sums.

```sh
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m hybrid_tsad.left_fixed_exposure_repeat "$WORKSPACE" "$RUN_DIR" score --domain PSM --method R --condition A --seed 2026
CUDA_VISIBLE_DEVICES= python "$REPO/scripts/summarize_left_exposure_repeats.py" evaluate "$WORKSPACE" "$RUN_DIR"
```

The first command expects `$RUN_DIR/PSM/R_A_2026/selected.pt` and its `receipt.json`. Re-score into a separate directory with the checkpoint and receipt copied and no score banks present. The second command requires all 24 sealed endpoints and banks, joins the digest-checked target labels only after identity checks, and emits `new_metrics.csv`. Exact input grids and score availability are checked against the native archived QUERY bank. There are no new CAL scores in the repeated study.

## Level 3 · Repeat fitting with the fixed source parents

The complete experiment uses 4,000-step, width-specific source models (25, 31 and 38 channels) and 2,000 U updates with fresh Adam at `1e-4`, batch 32, source/target split 16/16 and eight samples from each normal source. Every native endpoint keeps prototype and course state. Randomness for sampling and teacher preparation remains at the archived state. Only training RNG changes between 2025, 2026 and 2027.

In a ready author workspace the repeat endpoints require these existing files. Match them by the [original manifest](../../docs/experiments/left_evidence_freeze_20261128/source_manifest.csv) and [new identity receipt](../../docs/experiments/left_fixed_exposure_repeats_20261128/identity.json).

- `results/left_multisource_20261126_v1/{PSM,SWAT}/source/{selected.pt,receipt.json}` and the original replay QUERY banks.
- `results/left_memory_exposure_20261126_v1/{PSM,SWAT}/plan/{A.npz,B.npz,C.npz,audit_slots.npy,audit.json}`. Plan indices preserve target slot order and donors. The original source window generator reconstructs the fixed source draws.
- `results/left_local_reliability_20261126_v1/{PSM,SWAT}/weights/{teacher.npz,A.npy,B.npy,C.npy,receipt.json}`. These are the teacher score bank and condition-specific coefficient tensors, held fixed across the new seeds.

```sh
CUDA_VISIBLE_DEVICES= python -m hybrid_tsad.left_fixed_exposure_repeat "$WORKSPACE" "$RUN_DIR" verify --archive "$REPO/docs/experiments"
CUDA_VISIBLE_DEVICES=0 python -m hybrid_tsad.left_fixed_exposure_repeat "$WORKSPACE" "$RUN_DIR" probe --domain PSM
CUDA_VISIBLE_DEVICES=2 python -m hybrid_tsad.left_fixed_exposure_repeat "$WORKSPACE" "$RUN_DIR" probe --domain SWAT
bash "$REPO/scripts/run_left_fixed_exposure_repeat.sh" "$WORKSPACE" "$REPO/src" "$RUN_DIR"
```

The runner expects the recorded interpreter at `$WORKSPACE/envs/timeradar/bin/python` and four idle GPU lanes. A single endpoint can also be fitted with the module's `adapt` action and the corresponding method, condition, seed and domain flags, then scored with `score`.

To reconstruct historical dependencies in a separate workspace after acquiring the exact arrays and author configuration, the two-domain source entry is `python -m hybrid_tsad.left_multisource "$WORKSPACE" PSM source` (also `SWAT`). Existing fixed exposure plans were generated by `python -m hybrid_tsad.left_memory_exposure "$WORKSPACE" PSM` and `SWAT`, using U-FIT labels only to qualify donors offline. When the archived plan receipts are unavailable, the historical binding requires the matching `restricted_A`, `restricted_B` and `restricted_C` receipts from `python -m hybrid_tsad.left_memory_update "$WORKSPACE" <domain> <arm>`. The teacher coefficient entry is `python -m hybrid_tsad.left_reliability "$WORKSPACE" PSM prepare` and `SWAT`. The SMD source is prepared and fitted with `python -m hybrid_tsad.left_smd_source "$WORKSPACE" prepare` and then `train`. For each machine the [SMD target entry](../../src/hybrid_tsad/left_smd_targets.py) accepts `"$WORKSPACE" <machine> <arm>`. The historical fits, plans and coefficients were reused. Regenerating those assets was outside the execution scope of this report.

## Access

Source and target weights, score banks, canonical arrays and full exposure plans reside in author storage. The released repository contains executable entry points, protocol metadata, results and hashes. Access to author archives and protected SWaT originals requires author and provider permission as applicable. Hashes allow validation after retrieval. The public repository alone supports Level 1.
