# Shared LEFT NASA benchmarks and formal public-method comparison

This round completed two target-normal LEFT shared-model fits at seed2025 and all 80 admitted NASA entities, retaining every test entity. Prior PSM/SMD LEFT results and all candidate, E and MLP weights and scores were reused. No candidate training, adaptation, contamination or score modification was performed.

## Actual public results

[Four-domain common-axis AP, ROC-AUC and both VUS tables](PUBLIC_TABLES.md) · [fixed objective/readout ablation](ABLATIONS.md) · [full precision common method values](common_methods.csv) · [training-repeat mean and sample SD](REPEATS.md)

| Domain | LEFT full native AP / ROC / VUS-PR / VUS-ROC | LEFT common AP / ROC / VUS-PR / VUS-ROC | Common points |
|---|---|---|---|
| MSL27 | .50394 / .76518 / .54839 / .79574 | .50800 / .76252 / .55285 / .79337 | 69,711 |
| SMAP53 | .49231 / .76343 / .52053 / .78546 | .49312 / .76284 / .52170 / .78490 | 417,921 |

Each entity's complete native LEFT axis is [0,W floor(N/W)), totaling 72,384 MSL and 423,168 SMAP points. The joint axis begins at endpoint99; W is96 and192 respectively. The candidate and all existing readouts are cut to the same ending points separately for each entity. Scores remain available only at their containing window end for LEFT, compared with the scored endpoint for native V. No scores are padded or broadcast. VUS uses the existing author implementation, windowSize100 and thre250. Metrics are computed by entity before equal-weight aggregation.

Candidate seed2025 common-axis AP is .36355 on MSL and .38185 on SMAP; official V is .34078/.36502 and official E is .38602/.42477. Candidate-minus-LEFT AP wins/ties/losses are 8/0/19 and 16/0/37; the paired medians are −.06491 and −.07516. The source-supervised candidate retains a gain over official V but is below the full target-normal-trained LEFT detector on all four macro ranking measures. Its common-axis three-seed AP is .36176±.00174 and .38202±.00219. These are three project-training seeds with a shared official initialization, not three independent pretrained detectors. LEFT has one seed and no SD.

The PSM primary common panel uses [512,87744), 87,232 points. Candidate AP .57763 is close to native offline LEFT .57733, with higher LEFT ROC and VUS. That .00030 AP difference is insufficient by itself to establish a substantive benefit. On SMD's existing 703,212-point joint axis, candidate AP .44266 is below LEFT .49807 and reconstruction continuation .45079. Seen-normal-source permissions and the original causal/offline distinction remain. Earlier PSM endpoint LEFT/HistoryMean512, CATCH and PCA, and SMAP compatible simple methods remain in the [external panels](../native_supervised_mlp_20261123/EXTERNAL.md). Different axis panels must not be subtracted.

[Full and common per-entity scores](entity_metrics.csv) · [macro values](macro.csv) · [per-method paired distribution against LEFT](paired.csv) · [same-axis three-seed values](repeats_common.csv) · [archived complete causal ablation](full_axis_ablation.csv)

## Training and time permissions

The source revision is 3fadb4811797075e233076b9faceb0d8ec59b697, verified against official model, tools, experiment and configuration files. MSL maps alpha_casd_l/s to model-read alpha_ms_l/s explicitly with the original values0.5/0.3. Original files remain alongside the derived mapping. This is a project-compatible complete-model implementation, rather than an untouched numeric reproduction claim. Source/model/loss and infer are the pinned official paths. [Identity checks](OFFICIAL_IDENTITY.json) · [additional upstream identities](ADDITIONAL_OFFICIAL_IDENTITY.json) · [executed worker](left_nasa_author.executed.py)

Each entity normal array is independently split first80% train/last20% validation. Shared-domain scaling uses all official normal training values, including normal validation, with test excluded. The dataset never forms a window across entities or roles. MSL T-9 and SMAP A-5/A-6/A-8/A-9/D-12 have no complete validation window; they remain in the test, scored by their domain shared model. Every entity has legitimate training windows.

| Domain | Actual / selected epochs | Updates | Selected / final running step | Warmup / ramp steps | Training seconds | Scoring seconds |
|---|---|---|---|---|---|---|
| MSL | 12 / 9 | 4,140 | 3,105 / 4,140 | 500 / 1,000 | 158.919 | .562 |
| SMAP | 16 / 13 | 12,256 | 9,958 / 12,256 | 1,000 / 2,000 | 463.934 | 3.474 |

Both stopped via normal-validation patience3 under the20-epoch cap and official type1 learning-rate schedule. Warmup and ramp were complete at both selected and final steps. The official entry restores the selected model state_dict, including prototype buffers, but retains final Python model.step. That behavior was retained and recorded. No QUERY-selected running-state adjustment was made. The original TRAIN field target_values_or_labels_in_training=false means test values and labels are excluded; the shared model intentionally uses target-domain normal training arrays and normal validation. The [permissions note](README.md) clarifies this raw receipt wording without replacing the original.

Domain scaling and normal validation include historical project CAL regions. LEFT's independent-CAL operating measures are therefore NA. Existing V/E/MLP operation remains available on its original axes and CAL permissions, including PSM attack-only and NASA small-sample CAL. No separate normal interval was selected to fill LEFT alarm columns.

## Formal manuscript and evidence scope

The existing [draft](../../../manuscript/next_generation_20261115/draft.md) now contains a formal problem formulation, native-score definition, nominal reconstruction and point-weighted source pair objective, decoder learning procedure and inference pseudocode. It embeds actual four-domain tables and objective/readout ablations, then analyzes task boundaries, detection costs and separate U/F extensions. The [method definition](../../../manuscript/next_generation_20261115/method_definition.md) is synchronized. Prior draft paragraphs and user edits remain under an explicitly historical appendix; original versions are retained in Git.

The supported result is repeatable task-dependent source-supervised ranking improvement within native V, plus a measured advantage over this fixed standard MLP and, for particular task/metric combinations, matched E supervision. Strong NASA E scores, stronger target-trained LEFT scores, SMD negative source adaptation and false-alarm/coverage costs bound the claim. The three common axes do not equalize target-normal training or offline context. The present evidence does not establish overall dominance over mature TSAD methods, a complete mixed-unlabeled solution, independent FRFT innovation or explicit weak/lag anomaly mechanisms. Existing U/F recipes remain completed without demonstrated stable net value. No new candidate is launched by this comparison.

## Resource ledger and independent check

Two formal fits and all native score readouts used 626.927 occupied A40-seconds, **0.174146 A40-hours**, across devices0/1. No failed training or inference runs occurred. Candidate/common-axis metrics reused sealed caches on CPU. [Cost](COST.json) · [training and score identities](MSL/TRAIN.json) · [SMAP](SMAP/TRAIN.json) · [metric bindings](METRIC_BINDINGS.json)

A single independent CPU review checked original normal-array identities, exact domain scaler, entity-local split/window counts, normal validation checkpoint selection, official source identities and the MSL mapping. It independently recalculated full/common AP and ROC for the first two fixed entities in each spacecraft, maximum difference1.11e-16. It recomputed macro aggregation, same-axis three-seed means and sample SD and pair win/tie/loss counts and medians from the full per-entity CSV, all below2e-16 discrepancy. VUS values were independently aggregated from production entity metrics, not re-evaluated from raw scores. No independent GPU or model forward was used. [Review scope](review.json)
