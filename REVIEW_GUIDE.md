# Hybrid global code-review handoff — 2026-11-14

This branch is a frozen engineering/research-review snapshot. It does not promote a stopped candidate into a validated final method. No training, new model search, checkpoint selection, threshold fitting, or historical metric recalculation was performed for this handoff. CPU tests use synthetic fixtures only.

## Start here

1. [Public experiment index](review_artifacts/EXPERIMENT_INDEX.md): 31 historical stages, information permissions, bounded decisions and source hashes; includes positive and negative evidence.
2. [Complete V1 architecture](docs/hybrid_v1/architecture.md), [module/provenance mapping](docs/hybrid_v1/paper_module_map.md) and [Stage1](docs/hybrid_v1/stage1_source.md)/[Z-U-F](docs/hybrid_v1/stage2_adaptation.md).
3. `src/hybrid_tsad/hybrid_v1/`: V1 model, FRFT/time/self-lag/cross-lag paths, training, source/target readers, calibration, full-query and event evaluation. `stage2_u_*`, `target_fullshot.py`, `reference_center*`, `p32_readout.py`, `score_aligned_discrimination.py` and `t_anchor_completion.py` retain later experiments rather than replacing historical definitions.
4. `src/hybrid_tsad/left_*.py`, `relationship_residual.py`, `normal_reference.py`, `roca_*.py`, `alarm_complementarity.py`, `fit_context_reliability.py`: LEFT-native adapters and later bounded research candidates/controls.
5. `review_artifacts/code/`: exported public experiment drivers and configuration snapshots that were outside the installable package. Read [snapshot provenance and restoration](review_artifacts/PUBLIC_CODE_README.md) before execution. This is not a claim that all historical runners have been refactored into one standalone CLI.
6. [Public numeric summaries](review_artifacts/results/README.md): fixed per-method/per-seed aggregates, original field names, source/export hashes. Any-hit, one-to-one matching, strict points and three-point confirmation are different metrics.

## Version and scientific boundaries

Hybrid is the proposed research method. LEFT, TimeRadar, CATCH, ARX/Ridge and simple controls remain external comparisons or mature components; reusing them is not itself an original contribution. The early source prototype, complete V1, partial joint-cycle replacement and later native-LEFT candidates are separate implementations with separate permissions. Do not combine their best numbers into one leaderboard.

Stage1 source training, Z-S/Z-T, target mixed-U, event-supervised F and same-target normal/full-shot training must remain distinct. The final SMD auxiliary-training confirmation is same-target normal initialization plus paired continuation, not multisource pretraining or target-free zero-shot. Three training seeds on one entity are not three independent test domains.

The two most recent channel-masking candidates are stopped: (a) blindspot auxiliary learning **and blindspot inference** failed to produce useful low-false-alarm persistent detection; (b) auxiliary learning with **native inference only** did not reproduce a stable learning increment in its frozen SMD setting. Seed2026 and PSM/SWaT development positives are retained, not retroactively promoted to confirmation. No final architecture or paper type is selected by this handoff.

## Installation and verification

Python >=3.10 is declared; the local handoff check used Python3.12. Install the project with an appropriate existing scientific/PyTorch environment:

```text
python -m pip install -e .
python -m hybrid_tsad --help
python -m compileall -q src scripts tests review_artifacts/code
python -m unittest discover -s tests -v
```

Optional groups: `.[data,evaluation]`, `.[model]`, `.[test]`, `.[tep]` and `.[profile]`. They are dependency declarations, not an instruction to download models or start training. Install the correct PyTorch CPU/CUDA build for the host separately. The upstream benchmark framework, official Affiliation implementation, TimeRadar/LEFT/CATCH/RoCA repositories, datasets and checkpoints remain external and require their own licenses and pinned versions; see [third-party boundaries](THIRD_PARTY_NOTICES.md).

The full test collection is **not green in this standalone Windows environment**. Preserve this fact. [Validation notes](review_artifacts/VALIDATION.md) distinguish missing external fixtures, platform assumptions, test-order isolation and frozen-oracle mismatches. They are engineering issues for review, not newly run scientific results. Do not disable integrity checks or alter pinned hashes to make the suite pass. Some metadata-only tests expect a fresh Python process without Torch previously imported; run those modules in separate processes.

## External workspace and reproducibility

Author sources, raw data, physical FIT/SELECT/CAL/QUERY files, checkpoints and full score banks are not distributed. Example and snapshot configuration paths therefore need binding to an authorized external workspace. A `.PENDING.json` or example is historical/configuration documentation, not a live execution release. Default preview/permission gates are intentionally retained. The reviewer can trace code, configurations, seeds, aggregation definitions and bounded conclusions without assuming the unavailable artifacts have been reproduced by cloning this branch.

Manifest hashes identify retained source bytes; an export may have machine-path redaction. Source and exported hashes are separate. Related current package paths are navigation aids, not proof that every old run used the latest module bytes. Original local experiments remain immutable outside this repository.

Some existing compatibility parsers, frozen test fixtures and provenance patches retain canonical historical workspace-prefix strings. They do not distribute the files at those locations or provide server access. New exported snapshots and examples use placeholders; real hardware UUIDs, transport credentials and deployment bindings are excluded. Frozen scientific kernels are not rewritten merely to remove a legacy string or make an integrity test green.

## Deliberate exclusions

- All enterprise, laboratory and field raw/derived material, device identities and private application results.
- Credentials, SSH/deployment/mirroring helpers, private infrastructure configuration and machine identifiers.
- All model/optimizer weights, original observations/labels, per-point score banks, large binary plans and run logs.
- Full third-party repositories, PDFs, restricted datasets and resources without verified redistribution permission.
- Build products, caches and the parent business application.

Public aggregate excerpts and hashes do not grant rights to external data, weights or upstream code. This snapshot does not impose a new permissive license on the entire research project.
