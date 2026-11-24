# Fixed target-normal training of source-supervised native V initializations

[Complete public method table and initialization/training ablations](TABLES.md) | [entity metrics](entity_metrics.csv) | [paired effects](paired_macro.csv) | [entity differences](paired_entity.csv) | [wins/ties/losses and medians](paired_distribution.csv)

## Completed fixed program

Six new target-normal trajectories were completed, one project seed2025: PSM25 variables and MSL27 sensor0 with one shared domain model per initialization. The official checkpoint, archived source-copy step120 and archived source-all-position step200 are fixed; no source trajectory is rerun or reselected. The copy initialization previously used source-event labels for checkpoint selection. Source-supervised initialization also used the public SWaT FIT event pairs. No source pair/replay/preservation loss is applied during this target-normal program.

Each entity's raw official normal train is split at floor(0.8N). W100/stride100 windows remain inside entity and role. StandardScaler is fitted to complete official normal training arrays including normal validation, never test. PSM uses per-channel statistics; MSL pools sensor0. All three methods share each task's exact scaler, window permutations, masks and optimizer schedule. Old source-transfer scaled inputs differ in every evaluated object, so all three zero-update references receive necessary new readout. The new zero-update references are not a target-information-free Z claim.

Each trajectory completes20 epochs with B1 native ten-copy forward, original per-copy full-window MSE, decoder-only AdamW1e-4 and logical batch32. Last batches divide by actual3 PSM/6 MSL windows. Total updates680/300 give warmup68/30, then constant rate. Epoch0 and20 completed epochs supply full normal-validation banks; earliest minimum MSE selects the model. No target anomaly labels, test values or test labels enter training/selection. Global selection is sealed before readout. This is a unified project training procedure, not an unchanged author numerical reproduction.

| Task | Initialization | Executed / selected epoch | Updates | Window exposures | Selected normal validation MSE |
|---|---|---|---|---|---|
| PSM | official | 20 / 20 | 680 | 21180 | 0.245262 |
| PSM | copy | 20 / 20 | 680 | 21180 | 0.236597 |
| PSM | all_positions | 20 / 20 | 680 | 21180 | 0.234622 |
| MSL | official | 20 / 20 | 300 | 9080 | 0.506822 |
| MSL | copy | 20 / 20 | 300 | 9080 | 0.503767 |
| MSL | all_positions | 20 / 20 | 300 | 9080 | 0.506030 |

All six happen to select epoch20 by the fixed normal rule; that outcome was not forced. The126 complete epoch states remain on the server, and all126 per-window validation banks and six selected states are locally archived and SHA-verified. [Curves](source/curves.csv) | [training and exposure ledger](source/training.csv) | [source/target role and scaler configuration](source/) | [selection audit](SOURCE_REVIEW.json)

## Actual result and scope

The common offline axes remain87,645 PSM points and69,495 MSL points. Native full axes separately retain87,800/72,600 points. All27 MSL entities are evaluated first and equally macro-averaged. Full-window V has window-end availability, with no tail padding. All new rows share the new preprocessing. Mature external caches retain their own documented training/scaler/window permissions. The three CATCH archived seeds are displayed separately rather than silently averaged. No independent-CAL operation row is filled: old PSM all-attack CAL does not certify normal FPR, and NASA normal roles overlap new training/validation/statistics.

Under the same new preprocessing and target training, source-supervised initialization exceeds source reconstruction initialization in all four domain-macro metrics in both tasks. The paired AP differences are +0.001568 on PSM and +0.004057 on MSL. This is a single-seed complete-initialization comparison, not evidence that each entity benefits or that the effect is reproducible after target training. MSL has13 AP wins and14 losses against the reconstruction initialization, with median difference -0.001032. Against the official initialization after training, MSL has18 wins and9 losses, median +0.003263.

The net effects differ sharply by task. PSM source-supervised AP declines from0.438880 to0.421540 (-0.017340), with ROC and both VUS values also decreasing. Its advantage over source reconstruction shrinks from0.004576 before target training to0.001568 afterwards. The trained source-supervised model is0.002133 below the official-initialization trained model in AP, although its ROC and VUS values are slightly higher. Official-initialization AP also falls slightly under this fixed normal-learning program. Lower normal-validation reconstruction MSE therefore does not establish improved detection ranking.

MSL source-supervised AP rises from0.282944 to0.371977 (+0.089033). All three initializations improve all four macro metrics after target-normal learning. Before learning, the supervised initialization is0.024296 below source reconstruction in AP; afterwards it is0.004057 above it. This is not simply preservation of a pre-existing MSL AP lead. The domain-mean reversal coexists with the negative entity median against source reconstruction. MSL uses only sensor0 and cannot establish cross-variable modeling benefits.

The complete detector remains below LEFT in AP, ROC and both VUS metrics on both tasks. PSM source-supervised target-trained AP is also below each of the three cached CATCH runs and the PCA residual baseline; its ROC is higher than CATCH but lower than PCA. These are complete-method comparisons with different native windows, training granularity and optimization permissions, not isolated architecture effects. LEFT trains with stride1 and native W192 (PSM)/W96 (MSL), whereas the new TimeRadar decoder program trains W100/stride100. No claim attributes the remaining gap to any one of these differences.

The supported result is a small, bounded initialization advantage over source reconstruction after ordinary normal-target learning, plus a substantial MSL benefit from normal-target learning itself. The fixed program does not establish a net PSM detection benefit or close mature-method competitiveness. It neither validates mixed-unlabeled adaptation nor identifies contamination as a cause. No new loss, coefficient, seed or follow-on recipe is started.

## Costs, fixed artifacts and independent recomputation

Six target training jobs consumed1,930.743509 recorded A40-seconds, including21 normal validations and checkpoint saves per job. Successful readout consumed46.086053 A40-seconds. The recorded total is **0.549119 A40 GPU-hours**. Normal-input checks and readout process import/setup were not independently GPU-metered. One shell CRLF startup failed before Python/model/GPU execution, and was corrected without changing scientific settings. CPU metric workers used540.126 summed wall-seconds; driver elapsed987.909 seconds includes waiting for training and readout, not GPU time. [Combined cost](TOTAL_COST.json) | [readout cost](READOUT_COST.json)

Six selected models,126 validation banks, scalers/plans and28 new score banks are stored outside Git at D:/code/bst/seal_detection_project/research_workspace/results/native_target_normal_20261124_v1. Full epoch states also remain under the same result root on the server. Scores retain160,400 complete-axis positions in total, six V columns, true indices, input footprints, availability and weight/scaler/raw-data identities. All local delivery SHA checks passed. Models, public raw arrays and numeric banks are not redistributed in Git. [Training delivery](source/LOCAL_DELIVERY.json) | [score delivery](READOUT_DELIVERY.json) | [score seal](SCORES_FROZEN.json)

The CPU independent check verified raw-array identity,80/20 boundaries, complete scaler statistics,20 permutations and actual update/exposure/warmup counts, all126 epoch/validation SHA identities, mean validation losses, earliest minima, initial model identities and frozen non-decoder states. It also verified all28 bank hashes, six-column/scaler bindings and time footprints, all domain/paired/entity-distribution summaries. Raw AP/ROC recomputation covers PSM and fixed MSL C-1, six methods and two axes:24 rows, plus three CATCH caches and PCA on the common PSM axis. VUS was checked for aggregation/differences/entity distributions only, not independent integration. [Independent report](review.json)

The initial derived table merged three CATCH display names. Their already-computed metric rows were relabeled seed0/1/2 and summaries regenerated without recomputing metrics or changing scores; pre-correction derived files remain in the server label_fix_original folder. The audit also records harmless JSON-to-CSV float parsing differences (at most1.11e-16 for LEFT). No frozen score integrity check was disabled. [Label correction](EXTERNAL_LABEL_FIX.json)

PSM normal statistics use the same raw data and permissions as LEFT, with float summation-layout differences around1e-14; they are not claimed bit-exact to LEFT. MSL statistics are bit-exact to LEFT's archived shared normal scaler. All three new methods are exact matches to one another within task. Source-only three-seed results and the previous E/MLP/U/F panels remain unchanged under their own preprocessing and information roles. This batch uses public data only, does not resume U/F or contamination, and ends without an additional variant.
