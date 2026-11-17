# Fixed decoder-freeze paired comparison

Complete:8 adapter-only trainings of2000 updates and8 scoring jobs. Original native16 and auxiliary72 rows reused. No identical completed adapter-only control found in bounded history/code/config search.

| Task | Update policy / arm | Official V AP | ROC | C3 normal FPR (%) | C3 anomaly coverage (%) |
|---|---|---:|---:|---:|---:|
| PSM | zero | 0.431166 | 0.670150 | 0.0000 | 0.0988 |
| PSM | self | 0.426359 | 0.678390 | 0.4602 | 1.9472 |
| PSM | cross | 0.425339 | 0.680241 | 1.2418 | 4.2331 |
| PSM | self_adapter_only | 0.429517 | 0.686881 | 0.0032 | 0.2022 |
| PSM | cross_adapter_only | 0.428966 | 0.666631 | 0.0000 | 0.1223 |
| machine-1-1 | zero | 0.609689 | 0.902007 | 2.7542 | 44.9889 |
| machine-1-1 | self | 0.603667 | 0.905258 | 3.1004 | 46.3994 |
| machine-1-1 | cross | 0.616477 | 0.905044 | 3.2522 | 49.7773 |
| machine-1-1 | self_adapter_only | 0.595749 | 0.898500 | 2.9293 | 43.3556 |
| machine-1-1 | cross_adapter_only | 0.601071 | 0.899159 | 2.7853 | 43.3927 |
| machine-1-2 | zero | 0.282421 | 0.921863 | 0.0477 | 15.1292 |
| machine-1-2 | self | 0.298940 | 0.922488 | 0.0997 | 16.4207 |
| machine-1-2 | cross | 0.303594 | 0.922703 | 0.0911 | 17.1587 |
| machine-1-2 | self_adapter_only | 0.293192 | 0.922718 | 0.0694 | 14.5756 |
| machine-1-2 | cross_adapter_only | 0.283690 | 0.921145 | 0.0607 | 15.4982 |
| machine-1-3 | zero | 0.357354 | 0.864324 | 0.0219 | 5.5080 |
| machine-1-3 | self | 0.383640 | 0.887578 | 0.0000 | 5.3856 |
| machine-1-3 | cross | 0.385731 | 0.889472 | 0.0000 | 5.5080 |
| machine-1-3 | self_adapter_only | 0.357892 | 0.862952 | 0.0393 | 5.5080 |
| machine-1-3 | cross_adapter_only | 0.361542 | 0.868264 | 0.0393 | 5.5080 |

Decision: complete native freezing does not reliably preserve or improve native detection. PSM ranking recovers part of the joint-update loss, but remains below zero-update AP; lower false alarms coincide with substantially weaker sustained detection. All three SMD Cross-only AP values are below the previous decoder+Cross values; machine1-1 is also below zero-update. Cross-only versus Self-only AP differences(pp): PSM -0.0551, SMD1-1 +0.5322, SMD1-2 -0.9503, SMD1-3 +0.3650. Thus Cross has no stable independent value under the frozen-decoder policy. Stop this fixed experiment without regularization, distillation or new variants. Decoder updates cannot be identified as the sole cause of earlier failures.

The entire native model, including decoder and buffers, is frozen/eval. Decoder operations remain on the autograd path; FIT tests verify nonzero gradients reach the adapter. Target trainables16912 versus810522 for the old joint adapter runs. Initial weights/FIT draws/masks/budget/normal SELECT/own CAL are unchanged. Primary checks recomputed all8 raw AP/ROC and checked whole-native freeze receipts; these are primary verification, not an independent audit. Original formal scores and stopped-candidate conclusions remain unchanged.

[Complete paired table including official V, strict/C3 events, errors, coverage and auxiliary E/E+0.9V](paired_comparison.csv). [Auxiliary distributions and rankings](auxiliary_decomposition_96rows.csv) preserve the old72 rows and add24; they are not used to choose scoring or thresholds. The auxiliary NumPy variance reduction is not substituted for canonical native torch scores. Missing/boundary event details and offline timing remain in the48 hashed references in the raw archive. All native results are offline retrospective.

Actual cost2.0051954378684362 GPUh including FIT tests, training and scoring; failures0. CPU evaluation4.844 seconds, no GPU evaluation. Full raw scores/weights/traces/proofs remain outside Git. No extra module, seed, training or scoring formula follows this result.
