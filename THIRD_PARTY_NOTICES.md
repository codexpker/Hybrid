# Third-party and redistribution boundaries

This public review snapshot is not a redistribution of the complete upstream research environments. No repository-wide commercial or unrestricted MIT license is asserted here.

- **TimeRadar, LEFT, CATCH, RoCA, Autoformer and TimesFM:** pinned official provenance and actual usage are documented under `docs/provenance/`, `docs/hybrid_v1/paper_module_map.md` and the external adapters. Complete upstream repositories, checkpoints and training data are not included. Reviewers must obtain authorized versions separately and check code, weights and data licenses independently. A dynamic import or configuration URL is not a redistribution permission.
- **CAROTS SOC control:** the limited vendored `src/hybrid_tsad/experimental_controls/carots_soc/upstream_loss.py` retains the original file, `LICENSE`, `ORIGIN.json` and local scope README. Its license is **Modified MIT Non-Commercial with Permission**, not unrestricted MIT. Preserve its notices and restrictions, including for the adapter. This is a published-mechanism control, not a full CAROTS reproduction or claimed Hybrid innovation.
- **Public benchmark summaries:** only aggregate excerpts from local public-benchmark experiments are supplied, with source/export hashes and protocol limitations. Raw datasets, annotations, full prediction banks and model weights are not included or relicensed.
- **Independent project code:** reference/inspired/reused boundaries are given in the provenance documents. Public visibility alone does not grant an unrestricted license. Material that could not safely be redistributed was left outside this review snapshot.

Historical candidates and negative findings remain labeled by their original scope. No third-party capability is relabeled as a new empirical contribution of Hybrid.

## Explicit project-checkpoint export, 19 November 2026

The prior review-snapshot exclusion remains applicable to all upstream weights,
raw data, labels and prediction banks. Under the user's explicit current model
delivery authorization, the six small **project-initialized and project-trained**
B/C/D source predictors under `docs/experiments/timeradar_native_v_conditioning_20261119/models/`
are a bounded exception. They contain no official pretrained tensors or raw data.
C/D reuse the Apache2-licensed TimesFM3 code architecture without official
pretrained weights; the code license and actual source commit are included.
Their model card and byte hashes govern this export. They do not relicense the
public training data, upstream software, or the rest of this repository.
