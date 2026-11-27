# Public benchmark supplement and evidence index

The active manuscript is [Source transfer and mixed-unlabeled adaptation with LEFT](draft.md). The [frozen evidence index](../../docs/experiments/left_evidence_freeze_20261128/README.md) defines the evaluated methods, claims and table identities. The [method index](method_definition.md) and `draft.tex` retain historical material. The current [typeset manuscript](manuscript.pdf), [reproducibility guide](REPRODUCIBILITY.md) and [fixed-exposure repeat evidence](../../docs/experiments/left_fixed_exposure_repeats_20261128/README.md) accompany the active draft. The supplementary panels below are separate archived evaluations with their own protocols.

## C. Matched causal suffix, scores available at the current observation

PSM uses [23016,87744) and SWaT [608535,944856). The PSM LEFT endpoint uses a target-normal-trained checkpoint and scaler including SELECT. SWaT uses the restricted FIT-only LEFT regime. TimeRadar evaluates one W100 window per point with target FIT statistics. TimesFM 3.0 uses the complete official weights and joint 512-step history, followed by current-observation median forecast error in FIT units. History Mean uses the archive's residual scaling. All four rows use current-or-past data. These suffixes have different prevalence from the full tests. [Axes and source receipts](../../docs/experiments/common_native_benchmarks_20261122/PSM_SWAT_COMMON_AXIS.json)

**AP**

| Training group | Method | PSM | SWaT |
| --- | ---: | ---: | ---: |
| Target normal | LEFT W192 endpoint | 0.5743 | 0.1514 |
| Pretrained | TimeRadar terminal V | 0.5510 | 0.7957 |
| Pretrained | TimesFM 3.0 official forecast | 0.3445 | 0.0940 |
| Statistical | HistoryMean512 | 0.3037 | 0.0877 |

**ROC-AUC**

| Training group | Method | PSM | SWaT |
| --- | ---: | ---: | ---: |
| Target normal | LEFT W192 endpoint | 0.7629 | 0.5467 |
| Pretrained | TimeRadar terminal V | 0.7350 | 0.8579 |
| Pretrained | TimesFM 3.0 official forecast | 0.5668 | 0.2884 |
| Statistical | HistoryMean512 | 0.5256 | 0.2278 |

**VUS-PR**

| Training group | Method | PSM | SWaT |
| --- | ---: | ---: | ---: |
| Target normal | LEFT W192 endpoint | 0.5796 | 0.1591 |
| Pretrained | TimeRadar terminal V | 0.4846 | 0.5766 |
| Pretrained | TimesFM 3.0 official forecast | 0.3762 | 0.0965 |
| Statistical | HistoryMean512 | 0.3324 | 0.0909 |

**VUS-ROC**

| Training group | Method | PSM | SWaT |
| --- | ---: | ---: | ---: |
| Target normal | LEFT W192 endpoint | 0.7789 | 0.5402 |
| Pretrained | TimeRadar terminal V | 0.6482 | 0.7090 |
| Pretrained | TimesFM 3.0 official forecast | 0.6086 | 0.2984 |
| Statistical | HistoryMean512 | 0.5679 | 0.2327 |

## Per-training-seed values underlying the reported dispersions

| Panel | Dataset | Method | Training seed | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | PSM | LEFT | 2025 | 0.576639 | 0.776095 | 0.572611 | 0.785067 |
| A | PSM | LEFT | 2026 | 0.570185 | 0.771785 | 0.562230 | 0.778429 |
| A | PSM | CATCH | 0 | 0.445990 | 0.653418 | 0.536671 | 0.703657 |
| A | PSM | CATCH | 1 | 0.430264 | 0.641399 | 0.519273 | 0.690691 |
| A | PSM | CATCH | 2 | 0.433044 | 0.647248 | 0.526447 | 0.698733 |
| A | SWaT | LEFT | 2025 | 0.543866 | 0.820853 | 0.534601 | 0.809255 |
| A | SWaT | LEFT | 2026 | 0.586210 | 0.814548 | 0.522656 | 0.750588 |
| B | PSM | CATCH | 0 | 0.446458 | 0.653765 | 0.536065 | 0.702828 |
| B | PSM | CATCH | 1 | 0.430683 | 0.641721 | 0.518706 | 0.689799 |
| B | PSM | CATCH | 2 | 0.433369 | 0.647474 | 0.526733 | 0.698310 |

Each domain/seed metric was already aggregated across entities where relevant. Dispersion is across seed-level results with denominator n−1. The single selected LEFT reference in B has no compatible second-seed row on that axis. Entity heterogeneity is documented in the linked CSVs and is distinct from training repetition.

## Archived evidence

- [External endpoint comparison including whole-QUERY TimeRCD and history-only intersections](../../docs/experiments/native_endpoint_external_20261125/RESULTS.md).
- [Public window comparisons, dataset permissions and full entity distributions](../../docs/experiments/left_decoder_context_20261124/RESULTS.md).
- [Locked replay exposure results](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md).
- [Frozen endpoint manuscript and evidence](../native_endpoint_v_research_20261126/manuscript.md).
- [Previous working draft](https://github.com/codexpker/Hybrid/blob/87f6aaab085d58277aae8987aa390ec206803ee3/manuscript/next_generation_20261115/draft.md) and [previous full benchmark index](https://github.com/codexpker/Hybrid/blob/87f6aaab085d58277aae8987aa390ec206803ee3/manuscript/next_generation_20261115/README.md) preserve earlier working text, protocols and literature tables. The historical LaTeX export in this directory predates this working revision.
