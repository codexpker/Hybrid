# Fixed same-data-permission LEFT external comparison

Status: completed and independently checked for this four-case subset. The fixed full-SMD28 extension is now completed; see RESULTS.md for its separate full-axis and common-axis reports.

| Entity | Method | AP | ROC-AUC | Strict normal FPR (%) | C3 any-hit events | C3 matched event recall (%) | C3 anomaly coverage (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| PSM | LEFT native | 0.571098 | 0.764667 | 0.0000 | 8 | 11.4286 | 1.6182 |
| PSM | TimeRadar zero | 0.432502 | 0.670965 | 0.0000 | 2 | 2.8571 | 0.0988 |
| PSM | Head | 0.427837 | 0.678159 | 1.2852 | 10 | 14.2857 | 1.8911 |
| PSM | Self | 0.427206 | 0.678538 | 1.3091 | 10 | 14.2857 | 1.9475 |
| PSM | Hybrid Cross | 0.426164 | 0.680347 | 2.6626 | 11 | 15.7143 | 4.2337 |
| PSM | History Mean512 | 0.316848 | 0.530450 | 0.5465 | 12 | 17.1429 | 1.0067 |
| machine-1-1 | LEFT native | 0.388016 | 0.837270 | 13.2532 | 7 | 87.5000 | 45.9169 |
| machine-1-1 | TimeRadar zero | 0.612285 | 0.901876 | 3.8978 | 8 | 100.0000 | 44.9889 |
| machine-1-1 | Head | 0.605361 | 0.904564 | 4.2391 | 8 | 100.0000 | 46.1767 |
| machine-1-1 | Self | 0.606455 | 0.905165 | 4.2669 | 8 | 100.0000 | 46.3994 |
| machine-1-1 | Hybrid Cross | 0.619073 | 0.904915 | 4.6162 | 8 | 100.0000 | 49.7773 |
| machine-1-1 | History Mean512 | 0.206087 | 0.722219 | 1.3376 | 7 | 87.5000 | 2.4128 |
| machine-1-2 | LEFT native | 0.398419 | 0.938496 | 0.1685 | 5 | 50.0000 | 10.8856 |
| machine-1-2 | TimeRadar zero | 0.286288 | 0.921543 | 0.4923 | 5 | 50.0000 | 15.1292 |
| machine-1-2 | Head | 0.303242 | 0.922140 | 0.4701 | 5 | 50.0000 | 17.1587 |
| machine-1-2 | Self | 0.302738 | 0.922145 | 0.4657 | 5 | 50.0000 | 16.4207 |
| machine-1-2 | Hybrid Cross | 0.307326 | 0.922357 | 0.4568 | 5 | 50.0000 | 17.1587 |
| machine-1-2 | History Mean512 | 0.105610 | 0.841062 | 0.7052 | 3 | 30.0000 | 0.5535 |
| machine-1-3 | LEFT native | 0.344046 | 0.883564 | 0.0179 | 1 | 9.0909 | 5.6747 |
| machine-1-3 | TimeRadar zero | 0.351515 | 0.862250 | 0.0583 | 1 | 9.0909 | 5.6747 |
| machine-1-3 | Head | 0.375789 | 0.886172 | 0.0314 | 1 | 9.0909 | 5.5485 |
| machine-1-3 | Self | 0.376888 | 0.886283 | 0.0314 | 1 | 9.0909 | 5.5485 |
| machine-1-3 | Hybrid Cross | 0.378949 | 0.888234 | 0.0314 | 1 | 9.0909 | 5.6747 |
| machine-1-3 | History Mean512 | 0.154823 | 0.776147 | 0.3944 | 3 | 27.2727 | 3.6570 |

The CSV retains strict/C3 precision, recall, F1, false-positive points/fragments, delays, complete/long-event coverage and longest misses. NA remains NA. C3 any-hit and one-to-one event metrics are distinct.

## Scope and permissions

- Same target FIT/SELECT, FIT-only standardization, physically separate CAL/QUERY. Each method calibrates its own score on its complete legitimate CAL at quantile 0.99 (higher), strict `>`; QUERY never fits a threshold.
- Shared QUERY indices: PSM [512,87744); SMD 1-1 [512,28400); 1-2 [512,23600); 1-3 [512,23616). Events are eligible only when fully observed on this axis. Counts therefore differ from full native-axis reports.
- All LEFT W192 and TimeRadar W100 results remain offline retrospective, including same-batch dependence for TimeRadar. Cropping does not make scores online. Delay columns are offline score-axis offsets, not causal online detection guarantees.
- LEFT uses official 3fadb4811797075e233076b9faceb0d8ec59b697 native model/loss/20-epoch maximum/patience3, random initialization. All four selected epoch20: PSM12400 updates, SMD1-1 2640, SMD1-2/1-3 2200 each. Native warmup1000/ramp2000 is unchanged: SMD does not complete the full curriculum. No training extension was selected from QUERY.
- TimeRadar/Hybrid start from official source-pretrained weights. Head/Self/Cross have 2000 target reconstruction updates with frozen backbone; this matches data permissions, not initialization/optimization/compute budget with LEFT. H512 is untrained.
- LEFT actual new compute: 0.4025681857616615 GPU hours across four training and four scoring jobs; all exited0. Previously accepted TimeRadar/Hybrid and H512 scores were reused, with only necessary common-axis CPU evaluation.
- LEFT does better on PSM and SMD1-2 AP; Cross does better on SMD1-1 and SMD1-3 AP. Neither establishes universal ranking, controlled false-alarm, or sustained-event superiority. No module/threshold/seed changes follow these development results.
- PSM and these SMD entities have historical exposure; this is development comparison, not blind or independent-domain confirmation. Different-permission historical LEFT-R/CATCH rows remain separate and are not substituted here.

## Provenance

- Raw matched CSV SHA256: `a9b42297e16b196399a370db42bec2933b2896abc717feed475a1da21dd6289b`.
- LEFT raw-score global seal SHA256: `bb419cf0a8301b96768aba06f6e3763183508f0107dcb9f332ecb7bfee16709a`.
- Same-family independent responsibility audit LEFT subset: PASS; receipt SHA256 `723960464a0a8aa33601eec240aa92f72d1a98f0594a0d2ee73c96593579fbca`. This is not an external blinded review. The final full-SMD audit is also complete (SHA db7bb22156d84c9cae047fb3170c6fafbaf01f062f7d81ea0e79a9613646b4d9).
- Baseline implementation commit: `d3fd9750bb8a28333742578d209eb2868cd96756`; fixed scientific recipe for TimeRadar adapters remains `7480d5f215292ced9d0a9328eed274aff805584e`.
- No raw observations, checkpoints, private identifiers, credentials, or upstream third-party source files are published here.
