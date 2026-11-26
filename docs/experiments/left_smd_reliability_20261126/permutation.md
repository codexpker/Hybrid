# Window reliability and its shuffled assignment

Each of six runs used the matching source checkpoint, replay batches and frozen coefficient bank. A separate NumPy generator shuffled the 16 window coefficients inside each batch. The same permutation sequence was used for both datasets and all three conditions. Ordinary random permutations retained 2001 of the 32,000 assignments in their original slot per run. Each batch retained the exact coefficient multiset. The 16 source coefficients remained one.

## Mixed-target detection

A is natural exposure. B and C replace the same 3,200 common target slots with fully anomalous and fully nominal 192-point windows. All methods use native retrospective LEFT scores; PSM uses [22656,87744) and SWaT uses [608535,944856).

| condition | method | PSM AP | PSM ROC_AUC | PSM VUS_PR | PSM VUS_ROC | SWAT AP | SWAT ROC_AUC | SWAT VUS_PR | SWAT VUS_ROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | permuted | 0.5894 | 0.7546 | 0.5904 | 0.7709 | 0.7863 | 0.8419 | 0.6343 | 0.7403 |
| A | replay | 0.5919 | 0.7556 | 0.5920 | 0.7717 | 0.7861 | 0.8420 | 0.6338 | 0.7403 |
| A | window | 0.5949 | 0.7625 | 0.5940 | 0.7770 | 0.7872 | 0.8420 | 0.6318 | 0.7384 |
| B | permuted | 0.5778 | 0.7319 | 0.5777 | 0.7486 | 0.7810 | 0.8420 | 0.6328 | 0.7444 |
| B | replay | 0.5754 | 0.7328 | 0.5766 | 0.7498 | 0.7806 | 0.8418 | 0.6327 | 0.7444 |
| B | window | 0.5957 | 0.7573 | 0.5946 | 0.7728 | 0.7845 | 0.8426 | 0.6338 | 0.7432 |
| C | permuted | 0.5977 | 0.7568 | 0.5976 | 0.7729 | 0.7870 | 0.8419 | 0.6323 | 0.7391 |
| C | replay | 0.5960 | 0.7570 | 0.5958 | 0.7730 | 0.7861 | 0.8418 | 0.6326 | 0.7394 |
| C | window | 0.6002 | 0.7653 | 0.5988 | 0.7797 | 0.7872 | 0.8421 | 0.6319 | 0.7378 |

## Correct assignment minus shuffled assignment

| domain | condition | AP | ROC_AUC | VUS_PR | VUS_ROC |
| --- | --- | --- | --- | --- | --- |
| PSM | A | 0.0055 | 0.0078 | 0.0036 | 0.0061 |
| PSM | B | 0.0180 | 0.0254 | 0.0169 | 0.0242 |
| PSM | C | 0.0025 | 0.0085 | 0.0012 | 0.0068 |
| PSM | B-C | 0.0155 | 0.0169 | 0.0157 | 0.0174 |
| SWAT | A | 0.0009 | 0.0001 | -0.0024 | -0.0020 |
| SWAT | B | 0.0034 | 0.0006 | 0.0010 | -0.0012 |
| SWAT | C | 0.0002 | 0.0002 | -0.0005 | -0.0013 |
| SWAT | B-C | 0.0032 | 0.0004 | 0.0015 | 0.0002 |

Correctly assigned coefficients improve all four PSM ranking measures over shuffled coefficients in A, B and C. The largest AP difference (+0.0180) occurs under additional anomalous exposure. The original mixed CAL gives fewer confirmed false alarms alongside lower recall and fewer one-to-one event matches. On SWaT the AP advantage is smaller (+0.0009 in A, +0.0034 in B); VUS-ROC is lower in all conditions. In B, correct assignment gains 0.210 in long-event macro coverage but adds 0.054 in confirmed false-positive rate and matches two fewer events. These comparisons support a task-dependent contribution from matching weights to windows. They do not establish a general advantage at the original operating points.

B−C in this table is the difference of the two methods’ within-method exposure contrasts. Positive ranking differences favor correct assignment. One development seed is reported.

Alarm evaluation used each endpoint’s own fixed CAL q99/higher threshold and strict exceedance. The 512 selected CAL points contain 512 attack-labeled points on PSM and 38 on SWaT. [Operating results](permutation_operations.csv) · [paired operating differences](permutation_operating_contrasts.csv) · [events](permutation_events.csv).

Six new training runs used 467.24 A40 seconds in total; twelve score banks used 12.88 seconds. Independent CPU checks reproduced the assignments, per-batch coefficient multisets, model and bank hashes, AP, ROC-AUC, CAL thresholds and strict/three-confirmation counts. VUS and event matching used the existing evaluator. [Checks and costs](permutation_review.csv).
