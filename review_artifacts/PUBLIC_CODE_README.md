# Public experiment code snapshots

This is a bounded **review archive**, not a standalone reproduction release or a new training authorization. It contains 89 self-authored scientific workers/fixtures/evaluation scripts and effective numeric configurations across 17 exported public-data experiment groups; 18 groups were inspected, with one legacy adapter already present exactly in the package. Historical candidate execution status is `stopped_archive_do_not_autorun`. The three November 13 candidate groups retain their limited NO-GO decisions; the TEP relation check retains its limited mechanism GO, which is not a Hybrid innovation claim.

`PUBLIC_CODE_MANIFEST.json` maps each original research-workspace-relative path to its exported path, original SHA256, exported SHA256 and recorded path/EOL changes. Original experiment files are not modified. Exported bytes must not be called bit-identical originals when those hashes differ. Official effective configurations are not claimed as newly authored model designs.

## Paths and safe restoration

Legacy workers often derive the research root with `R=P.parents[1]`. Moving such a file below `review_artifacts/code` does **not** preserve that assumption. Do not run the exported file in place or reinterpret its local directory as the original experiment workspace.

Set `HYBRID_RESEARCH_ROOT` to a separate external research workspace. `restore_public_code.py` checks the manifest and, only with `--write`, materializes the text at its original relative paths below that root. It substitutes the explicit `__RESEARCH_ROOT__` placeholder, parses all restored Python/JSON before writing and refuses to overwrite differing existing files. It does not execute experiments. Path binding and UTF-8/LF normalization are engineering transformations, not algorithm changes; historical trusted-source hash guards can still require the owner's original immutable runtime snapshots.

Example, from the repository root:

```text
python review_artifacts/restore_public_code.py --root /external/research_workspace
python review_artifacts/restore_public_code.py --root /external/research_workspace --write
```

Keep the working package available as `<HYBRID_RESEARCH_ROOT>/hybrid_ad` or install it separately. The archive deliberately omits data arrays, fitted statistics files, random-plan arrays, checkpoints, raw score banks, logs, signed runtime permissions, local transport and job controllers. Obtain public datasets, fit-only statistics and regenerated plans under the documented original permissions; do not fit them using QUERY labels. Configurations may reference absent assets and historical source hashes. This archive alone does not make those assets or permissions available.

## Dependencies and coverage

- `hybrid_tsad` scientific kernels are in the package. Exact copies already in the package are not re-vendored here, including `hybrid_v1/target_fullshot.py`; channel-blindspot and predictive-cycle workers use the package kernels and their original runtime import layout.
- Complete official LEFT code is an **external** dependency at the historical upstream path, pinned by the existing project provenance. It is not copied into this review archive. Modules such as `models.Left.LEFT`, `exp` and their native utilities must come from the verified upstream version and license.
- Scientific dependencies include NumPy, PyTorch, SciPy, scikit-learn and pandas; some TEP input extraction code additionally requires `pyreadr`. Use the original experiment environment records for exact versions, not a claim that arbitrary current versions reproduce historical outputs.
- Stage1 plan generation is archived; its shared implementation remains in the package. H5 is a **source-training contamination** experiment, distinct from true target Stage2-U. Stage2-U examples include PSM scope and SWaT real-event controlled replacement. F's archived fixture is a synthetic/mock interface test, **not** a completed F detection result; actual `event_adaptation`, `event_data`, `event_query` and `event_readout` kernels are in `src/hybrid_tsad/hybrid_v1`.
- Z-S's original native consumer is archived; public-asset hashes, source checkpoints and calibration banks remain external. An old zero-shot record does not imply that target-trained LEFT candidates used zero-shot permissions.
- Latest aux-native confirmation uses native-only SELECT/readout; the earlier whole-channel blindspot experiment has a different stopped readout. Neither is silently substituted for the other.

`PUBLIC_CODE_DEPENDENCIES.json` lists static import and Python-asset references, including unresolved runtime helpers. It is an inventory, not proof of standalone runnability. Deployment-bearing evaluation binding and remote-generation wrappers were excluded rather than edited into purported scientific workers. No private motor/lab/field experiments, third-party full repository, papers, weights or server credentials are included.

Specific unresolved runtime requirements remain: historical workers import standalone aliases such as `left_predictive_cycle`, `left_native_history_lag` and `native_score_numpy`, or load `runtime/left_native_u.py`, `runtime/left_real_attack_u.py`, `runtime/left_loss_observer.py` and `runtime/relationship_residual.py`. Corresponding package modules exist, but the original runtime layout and expected revision/hash must be restored explicitly; current package files are not silently declared byte-identical to every historical runtime. Some entries also require omitted `DATA_GATE`, learner-plan qualification, checkpoint/receipt and source-identity files. Replacing those with dummy permission or hash records is not a valid reproduction. All role/plan `.npz` files, scaler arrays, source checkpoints and raw-score banks stay external. `pyreadr` extraction plus the Unix-only stdlib `resource` module require the original suitable platform. The local F subprocess fixture uses only an isolated local Python interpreter; it is not a deployment or SSH script.
