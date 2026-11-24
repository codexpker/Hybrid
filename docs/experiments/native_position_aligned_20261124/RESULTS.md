# Position-aligned source supervision for full-window offline detection

[Complete four-domain method table and paired changes](TABLES.md) · [entity results](entity_metrics.csv) · [entity-level differences](paired_entity.csv) · [wins/ties/losses and medians](paired_distribution.csv)

## Completed training and source-only selection

The three arms start from the same official TimeRadar checkpoint, seed2025 and fixed nominal-source plan. Only the native decoder updates. Encoder/FRFT parameters and buffers, B1 order computation, ten complementary masks, eval mode, isolated RNG and scalers remain fixed. Every arm completes240 original AdamW updates with24-step warmup. The nominal reconstruction term is unchanged. The two supervised arms share all15 positive and15 paired-normal windows per update, the same five FIT events and original normalized window-stratum masses, unit margin, pair coefficient1 and fixed scale a=0.02649214304983616.

For each pair, the candidate computes softplus(1−log1p(Vpositive[j]/a)+log1p(Vnormal[j]/a)) separately at each of100 positions, then averages those losses before original window-stratum weighting. It does not average V before softplus. All windows remain fully inside their own labeled abnormal/normal interval. A100-position window does not constitute100 events or uniform raw-timepoint supervision.

All three use the same full-position SOURCE SELECT risk on the original four held-out events, with event/time groups averaged equally. Nominal reconstruction MSE is separate. Seven fixed checkpoints including initialization are eligible, with the earliest minimum selected. Reconstruction continuation uses labels for selection, not its training gradient. The table's risk and MSE are source diagnostics, not target detection outcomes.

| Arm | Executed updates | Selected step | Full-position SELECT risk | Nominal SELECT copy MSE |
|---|---|---|---|---|
| copy | 240 | 120 | 0.893636504 | 0.613082 |
| terminal | 240 | 200 | 0.888699353 | 0.628327 |
| all_positions | 240 | 200 | 0.770134859 | 0.603891 |

Step0 risk is0.954938557 for all arms. Selected supervised checkpoints both occur at200, so their selected target-comparison states also share the same nominal and paired-window exposure counts. Reconstruction selection instead occurs at120. Each supervised trajectory uses7,680 nominal windows plus3,600 positive/3,600 pair-normal exposures through240; its selected200-state has6,400/3,000/3,000. Reconstruction's selected state has3,840 nominal exposures. There remain five FIT and four SELECT events. Distinct and repeated window counts are retained in [EXPOSURE.json](EXPOSURE.json).

The old trajectories contained only selected.pt, so they could not support the new selection rule. This round rebuilt the required controls and retained all21 full checkpoints and21 per-position validation banks. Rebuilt reconstruction step80 exactly matches its historical selected state, and rebuilt terminal step240 exactly matches the older V-supervised selected state (all110 tensors). Their original files are untouched. The new full-position validation selects different checkpoints; old selected scores do not substitute for the new controls. [Asset scan](ASSET_REUSE.json) · [trajectory identity checks](TRAJECTORY_IDENTITY.json) · [source selection curves](SOURCE_SELECT_CURVES.csv) · [all source model identities](ALL_MODELS_FROZEN.json)

## Actual detection results

The single-seed candidate's common-axis AP is0.46225 on PSM,0.45628 on SMD,0.36696 on MSL and0.33681 on SMAP. All109 entities remain. Ranking uses the same complete W100/stride100 grid and the latest intersection with historical causal coverage and native LEFT outputs. There are87,645/702,996/69,495/417,265 common points respectively. Native complete-axis results are separately preserved in macro.csv. New scores become available at the containing window end. No rolling endpoint bank, CAL, threshold or operating metric was generated.

The all-position candidate minus the matched terminal control has AP differences +0.01326/+0.02238/+0.01176/−0.00018 across PSM/SMD/MSL/SMAP. This is the direct supervision-range comparison under the same source validation rule. PSM AP and ROC increase while both VUS measures fall. SMD AP,ROC and VUS-PR rise, with VUS-ROC lower. MSL gains AP and VUS-PR but loses ROC and VUS-ROC. SMAP is essentially unchanged in AP, lower in ROC and higher in VUS.

Against nominal reconstruction, AP differences are +0.00697/+0.00559/−0.00370/−0.00053. Thus MSL's benefit over terminal supervision is not a net AP benefit over the normal-learning control. SMAP has VUS-PR improvement with approximately unchanged/slightly lower AP. SMD has17/28 AP wins over reconstruction, median+0.00249, and21/28 over terminal supervision, median+0.01402. MSL has11/27 AP wins over reconstruction; SMAP26/53. All per-entity differences and negative outcomes remain in the paired files.

The candidate exceeds frozen official V in each domain's mean AP, but complete LEFT has higher AP,ROC and both VUS measures in all four tasks. Official E and the fixed V+E also exceed the candidate AP on MSL and SMAP. LEFT's target-normal training and context differ from the source-trained shared decoder. The existing official E/combination and other mature external scores are reused on their original compatible axes. Earlier E-supervised and supervised-MLP comparisons, three-seed causal endpoint results and fixed-weight readout effects are retained separately; this new single-seed method does not inherit their repetition estimates. The previous full-window external panels remain [available with their information permissions](../native_full_window_20261124/TABLE_EXTERNAL.md).

## Scientific decision and updated formal text

Position-aligned supervision has a limited positive development signal: at seed2025, full-window AP improves over both matched controls on PSM and SMD, including a positive median machine-level effect in SMD. The NASA reconstruction controls and the VUS tradeoffs limit generality. Source SELECT improvement alone does not establish detection success, and no result identifies endpoint99 as the unique historical failure cause. The intervention changes the supervised position range; the common full-position selection rule is matched within this trial. Cross-round differences from older endpoint-selected models include a different selection criterion and must not be called a pure position-loss effect.

This remains an offline source-supervised candidate rather than an established complete mixed-unlabeled solution. It does not demonstrate broad superiority to mature methods, causal persistent detection, weak/lag anomaly detection or contamination robustness. The historical endpoint candidate and U/F results remain unchanged. No new position subset, weighting, scale, window, seed or follow-on loss is started.

The existing [method definition](../../../manuscript/next_generation_20261115/method_definition.md) and [draft](../../../manuscript/next_generation_20261115/draft.md) now define the three objectives, common selection rule and offline inference, with actual primary/paired tables. Earlier fixed-weight readout results and endpoint-method text are retained in explicitly historical appendices. The first paper's laboratory data, weak labels and enterprise records are excluded.

## Runtime, saved artifacts and independent check

Three source trajectories consumed767.496 occupied A40-seconds (0.213193 GPU-hours). The109 new score banks consumed268.542 A40-seconds. The combined recorded total is **0.287788 A40-hours**. Minimal checks/process startup were not separately timed. No failed training or score runs occurred. Model states differed from old compatible banks, so only necessary new scoring was performed. CPU metrics took480.945seconds with six workers and no GPU. [Training ledger](COST.json) · [combined ledger](TOTAL_COST.json) · [CPU timing](METRIC_COST.json)

All seven checkpoints and SOURCE validation arrays per arm are stored on the server and at D:/code/bst/seal_detection_project/research_workspace/results/native_position_aligned_20261124_v1/train/. The new readout banks and receipts are in its readout/ directory. Local score-bank SHA verification passed for109 objects. Weights and raw banks are not uploaded to Git; their identities, source/target indices and per-entity results are committed. [Source delivery](LOCAL_SOURCE_DELIVERY.json) · [score receipts](banks.csv)

One independent CPU review recomputed all21 SOURCE SELECT risks from saved position arrays, verified21 checkpoint identities and selected steps, common plan/scale and initial states. The maximum source risk discrepancy was2.22e-16. Target review used PSM,SMD machine-1-1,MSL C-1 and SMAP A-1 for full/common AP/ROC, with maximum discrepancy8.33e-17. Whole-table macro and paired distribution aggregation matched within2.99e-16. VUS was only independently aggregated from production entity metrics; its formula was not rerun. No independent model forward or GPU was used. [Review scope](review.json)
