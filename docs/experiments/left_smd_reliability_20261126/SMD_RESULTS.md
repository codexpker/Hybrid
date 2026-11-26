# Multisource LEFT on the SMD28 held-out target fold

Each SMD cell gives an equal-weight mean of eligible machines. The complete SMD dataset was excluded from both source fitting and source selection in this fold. PSM and SWaT use the earlier TEP–SMD source pair. All three tasks have historical project exposure. The fixed training seed is 2025.

## Public detection

The primary panel uses every complete W192 test block. All LEFT methods use normal-prefix standardization including target normal validation in this public panel. Z has no target parameter updates. N and the scratch detector use the same target-normal training contract. Interior LEFT scores become available at the window endpoint. PSM uses [0,87744), SWaT uses [495000,944856), and each SMD machine uses [0,192⌊N/192⌋). CATCH uses normal target training with archived FIT statistics (mean of three seeds); PCA uses a fitted linear residual; level deviation uses the normal-prefix scaler and current observation. TimeRadar's native block-grid comparator uses the original blocked inference cache.

### Complete LEFT axes

**AP**

| method                                   | PSM               | SWaT    | SMD28   |
|:-----------------------------------------|:------------------|:--------|:--------|
| CATCH                                    | 0.43643 ± 0.00839 | NA      | NA      |
| Frozen multisource LEFT                  | 0.49639           | 0.72392 | 0.49690 |
| Multisource LEFT + normal training       | 0.55438           | 0.54250 | 0.51765 |
| PCA residual                             | 0.53440           | NA      | NA      |
| Pointwise level deviation                | 0.43827           | 0.74480 | 0.42651 |
| Pretrained TimeRadar (native block grid) | 0.43327           | 0.63930 | NA      |
| Target-normal LEFT from scratch          | 0.57664           | 0.54387 | 0.49751 |

**ROC_AUC**

| method                                   | PSM               | SWaT    | SMD28   |
|:-----------------------------------------|:------------------|:--------|:--------|
| CATCH                                    | 0.64736 ± 0.00601 | NA      | NA      |
| Frozen multisource LEFT                  | 0.70108           | 0.81151 | 0.86792 |
| Multisource LEFT + normal training       | 0.75671           | 0.85558 | 0.88301 |
| PCA residual                             | 0.79332           | NA      | NA      |
| Pointwise level deviation                | 0.65423           | 0.83182 | 0.80337 |
| Pretrained TimeRadar (native block grid) | 0.67067           | 0.82197 | NA      |
| Target-normal LEFT from scratch          | 0.77610           | 0.82085 | 0.87868 |

**VUS_PR**

| method                                   | PSM               | SWaT    | SMD28   |
|:-----------------------------------------|:------------------|:--------|:--------|
| CATCH                                    | 0.52746 ± 0.00874 | NA      | NA      |
| Frozen multisource LEFT                  | 0.50096           | 0.49756 | 0.54191 |
| Multisource LEFT + normal training       | 0.55201           | 0.53098 | 0.56463 |
| PCA residual                             | 0.52899           | NA      | NA      |
| Pointwise level deviation                | 0.44602           | 0.46454 | 0.42547 |
| Pretrained TimeRadar (native block grid) | 0.44126           | 0.51406 | NA      |
| Target-normal LEFT from scratch          | 0.57261           | 0.53460 | 0.54151 |

**VUS_ROC**

| method                                   | PSM               | SWaT    | SMD28   |
|:-----------------------------------------|:------------------|:--------|:--------|
| CATCH                                    | 0.69769 ± 0.00655 | NA      | NA      |
| Frozen multisource LEFT                  | 0.70649           | 0.66106 | 0.91292 |
| Multisource LEFT + normal training       | 0.76627           | 0.84311 | 0.92087 |
| PCA residual                             | 0.76669           | NA      | NA      |
| Pointwise level deviation                | 0.65429           | 0.64428 | 0.86664 |
| Pretrained TimeRadar (native block grid) | 0.67837           | 0.76683 | NA      |
| Target-normal LEFT from scratch          | 0.78507           | 0.80926 | 0.91862 |

### Shared external axes

Each entity uses [99,min(100⌊N/100⌋,192⌊N/192⌋)). PSM and SMD use the verified archive input identities. DADA uses a frozen W100 block model and full normal-prefix scaler. TimeRCD uses original logits, FIT-only statistics and W5000 offline context. The frozen TimeRadar V endpoint comparator reads the trailing W100. SWaT has no equivalent archived DADA comparison in this panel. Source training, target normal fitting and lookahead permissions vary by row.

**AP**

| method                                   | PSM               | SMD28   |
|:-----------------------------------------|:------------------|:--------|
| CATCH                                    | 0.43684 ± 0.00844 | NA      |
| Frozen multisource LEFT                  | 0.49671           | 0.49748 |
| Multisource LEFT + normal training       | 0.55457           | 0.51818 |
| PCA residual                             | 0.53444           | NA      |
| Pretrained DADA                          | 0.39565           | 0.41461 |
| Pretrained TimeRCD (FIT-only)            | 0.30193           | 0.47295 |
| Pretrained TimeRadar V (causal endpoint) | 0.53528           | 0.44480 |
| Target-normal LEFT from scratch          | 0.57689           | 0.49808 |

**ROC_AUC**

| method                                   | PSM               | SMD28   |
|:-----------------------------------------|:------------------|:--------|
| CATCH                                    | 0.64765 ± 0.00602 | NA      |
| Frozen multisource LEFT                  | 0.70127           | 0.86779 |
| Multisource LEFT + normal training       | 0.75672           | 0.88294 |
| PCA residual                             | 0.79312           | NA      |
| Pretrained DADA                          | 0.60496           | 0.85136 |
| Pretrained TimeRCD (FIT-only)            | 0.47619           | 0.85942 |
| Pretrained TimeRadar V (causal endpoint) | 0.73098           | 0.81321 |
| Target-normal LEFT from scratch          | 0.77618           | 0.87855 |

**VUS_PR**

| method                                   | PSM               | SMD28   |
|:-----------------------------------------|:------------------|:--------|
| CATCH                                    | 0.52717 ± 0.00869 | NA      |
| Frozen multisource LEFT                  | 0.50229           | 0.54215 |
| Multisource LEFT + normal training       | 0.55329           | 0.56510 |
| PCA residual                             | 0.52873           | NA      |
| Pretrained DADA                          | 0.43945           | 0.47465 |
| Pretrained TimeRCD (FIT-only)            | 0.34850           | 0.51514 |
| Pretrained TimeRadar V (causal endpoint) | 0.48359           | 0.46719 |
| Target-normal LEFT from scratch          | 0.57412           | 0.54184 |

**VUS_ROC**

| method                                   | PSM               | SMD28   |
|:-----------------------------------------|:------------------|:--------|
| CATCH                                    | 0.69698 ± 0.00662 | NA      |
| Frozen multisource LEFT                  | 0.70738           | 0.91282 |
| Multisource LEFT + normal training       | 0.76690           | 0.92082 |
| PCA residual                             | 0.76699           | NA      |
| Pretrained DADA                          | 0.65197           | 0.89970 |
| Pretrained TimeRCD (FIT-only)            | 0.53197           | 0.90975 |
| Pretrained TimeRadar V (causal endpoint) | 0.66478           | 0.86996 |
| Target-normal LEFT from scratch          | 0.78578           | 0.91853 |

## Independent mixed adaptation

| arm                           |       AP |   ROC_AUC |   VUS_PR |   VUS_ROC |
|:------------------------------|---------:|----------:|---------:|----------:|
| Multisource target U          | 0.555152 |  0.888150 | 0.608026 |  0.926151 |
| Random-init target U          | 0.545181 |  0.888176 | 0.597083 |  0.924072 |
| Multisource source replay     | 0.559843 |  0.895781 | 0.613119 |  0.929473 |
| Frozen multisource LEFT       | 0.540538 |  0.871383 | 0.590123 |  0.916426 |
| Source continuation           | 0.540520 |  0.875003 | 0.588346 |  0.916472 |
| Window-weighted source replay | 0.558367 |  0.895324 | 0.611582 |  0.929960 |

CAL is taken from the reserved nominal-normal training record with FIT-only standardization. Each checkpoint sets its own strict-greater q99/higher cutoff on 512 covered CAL points. The operating table gives entity-macro summaries and eligible denominators for each measure, and the event file preserves one-to-one matching, any-hit, retrospective offsets and censored intervals. Per-entity measurement precedes aggregation. A machine without both QUERY classes has undefined ranking metrics. Coverage restricted to long events requires complete events longer than 512 points.

### Paired effects across machines

| panel   | target   | reference   | metric   |   n |      mean |    median |   win |   tie |   loss |
|:--------|:---------|:------------|:---------|----:|----------:|----------:|------:|------:|-------:|
| public  | N        | source      | AP       |  28 |  0.020757 |  0.005955 |    18 |     0 |     10 |
| public  | N        | source      | ROC_AUC  |  28 |  0.015095 |  0.008931 |    21 |     0 |      7 |
| public  | N        | source      | VUS_PR   |  28 |  0.022720 |  0.010174 |    15 |     0 |     13 |
| public  | N        | source      | VUS_ROC  |  28 |  0.007944 |  0.003974 |    16 |     0 |     12 |
| public  | N        | scratch_N   | AP       |  28 |  0.020139 |  0.007523 |    21 |     0 |      7 |
| public  | N        | scratch_N   | ROC_AUC  |  28 |  0.004331 |  0.003801 |    20 |     0 |      8 |
| public  | N        | scratch_N   | VUS_PR   |  28 |  0.023120 |  0.010399 |    21 |     0 |      7 |
| public  | N        | scratch_N   | VUS_ROC  |  28 |  0.002252 |  0.000845 |    18 |     0 |     10 |
| mixed   | window   | replay      | AP       |  28 | -0.001477 | -0.000344 |    12 |     0 |     16 |
| mixed   | window   | replay      | ROC_AUC  |  28 | -0.000457 |  0.000031 |    15 |     0 |     13 |
| mixed   | window   | replay      | VUS_PR   |  28 | -0.001537 | -0.000095 |    12 |     0 |     16 |
| mixed   | window   | replay      | VUS_ROC  |  28 |  0.000488 |  0.000000 |    14 |     0 |     14 |
| mixed   | window   | source      | AP       |  28 |  0.017829 |  0.013956 |    21 |     0 |      7 |
| mixed   | window   | source      | ROC_AUC  |  28 |  0.023941 |  0.013665 |    22 |     0 |      6 |
| mixed   | window   | source      | VUS_PR   |  28 |  0.021459 |  0.013632 |    22 |     0 |      6 |
| mixed   | window   | source      | VUS_ROC  |  28 |  0.013535 |  0.001353 |    22 |     0 |      6 |
| mixed   | replay   | ordinary    | AP       |  28 |  0.004692 |  0.005065 |    19 |     0 |      9 |
| mixed   | replay   | ordinary    | ROC_AUC  |  28 |  0.007631 |  0.001263 |    18 |     0 |     10 |
| mixed   | replay   | ordinary    | VUS_PR   |  28 |  0.005094 |  0.003869 |    18 |     0 |     10 |
| mixed   | replay   | ordinary    | VUS_ROC  |  28 |  0.003321 |  0.000263 |    17 |     0 |     11 |
| mixed   | window   | source_only | AP       |  28 |  0.017847 |  0.015360 |    23 |     0 |      5 |
| mixed   | window   | source_only | ROC_AUC  |  28 |  0.020321 |  0.011670 |    23 |     0 |      5 |
| mixed   | window   | source_only | VUS_PR   |  28 |  0.023237 |  0.011744 |    22 |     0 |      6 |
| mixed   | window   | source_only | VUS_ROC  |  28 |  0.013488 |  0.003881 |    25 |     0 |      3 |
| mixed   | ordinary | random_u    | AP       |  28 |  0.009971 |  0.004204 |    16 |     0 |     12 |
| mixed   | ordinary | random_u    | ROC_AUC  |  28 | -0.000026 |  0.000912 |    14 |     0 |     14 |
| mixed   | ordinary | random_u    | VUS_PR   |  28 |  0.010942 |  0.006515 |    21 |     0 |      7 |
| mixed   | ordinary | random_u    | VUS_ROC  |  28 |  0.002079 |  0.000545 |    21 |     0 |      7 |

Win, tie and loss counts use full-precision signed differences. Machine counts describe the distribution over this fixed SMD panel with one training seed.

[Operating metrics and denominators](smd_operation_macro.csv) · [Per-machine ranking](smd_entity_metrics.csv) · [Per-machine differences](smd_entity_deltas.csv) · [Per-machine operations](smd_entity_operations.csv) · [Event detail](smd_events.csv) · [Counts and pooled coverage across complete events](smd_event_totals.csv).

### Normal calibration and event costs

**strict**

| arm                           |     FPR |   Point recall |   Event F1 |   Any-hit recall |   FP fragments/machine |   Long macro coverage |   Long pooled coverage per machine |
|:------------------------------|--------:|---------------:|-----------:|-----------------:|-----------------------:|----------------------:|-----------------------------------:|
| Multisource target U          | 0.24499 |        0.66273 |    0.14742 |          0.87577 |              232.64286 |               0.75927 |                            0.75923 |
| Random-init target U          | 0.24106 |        0.65989 |    0.15035 |          0.88125 |              252.10714 |               0.75083 |                            0.75079 |
| Multisource source replay     | 0.23492 |        0.64642 |    0.15654 |          0.87902 |              216.42857 |               0.75516 |                            0.75513 |
| Frozen multisource LEFT       | 0.22150 |        0.61353 |    0.17725 |          0.87530 |              162.60714 |               0.75806 |                            0.75802 |
| Source continuation           | 0.22436 |        0.61248 |    0.17039 |          0.87976 |              157.25000 |               0.75705 |                            0.75705 |
| Window-weighted source replay | 0.23822 |        0.64835 |    0.15635 |          0.87902 |              227.85714 |               0.75925 |                            0.75923 |

**confirmed3**

| arm                           |     FPR |   Point recall |   Event F1 |   Any-hit recall |   FP fragments/machine |   Long macro coverage |   Long pooled coverage per machine |
|:------------------------------|--------:|---------------:|-----------:|-----------------:|-----------------------:|----------------------:|-----------------------------------:|
| Multisource target U          | 0.22341 |        0.62954 |    0.20448 |          0.71311 |               75.14286 |               0.74278 |                            0.74278 |
| Random-init target U          | 0.21745 |        0.62860 |    0.21356 |          0.72175 |               72.82143 |               0.73458 |                            0.73458 |
| Multisource source replay     | 0.21475 |        0.61471 |    0.22747 |          0.72860 |               70.75000 |               0.73891 |                            0.73892 |
| Frozen multisource LEFT       | 0.20726 |        0.58459 |    0.20953 |          0.66147 |               48.35714 |               0.75097 |                            0.75097 |
| Source continuation           | 0.21061 |        0.58258 |    0.20606 |          0.66351 |               48.92857 |               0.75121 |                            0.75121 |
| Window-weighted source replay | 0.21701 |        0.61665 |    0.21413 |          0.72146 |               71.39286 |               0.74301 |                            0.74302 |

Ranking supports both classes on 28/28 mixed suffixes. 28/28 have an eligible complete event; 4/28 have an eligible long event. Timing is measured in sample points with a known window-end availability time. The independent CAL has nominal-normal permissions and provides an empirical operating point.

## Source identity and compute

HAI 21.03 train1 and train3 contributed 216,001 and 478,801 consecutive one-second FIT observations, respectively. Independent train2 SELECT contained 226,801 observations. All three files passed the pinned Git blob checks. Each has 79 process columns, zero missing values, and four all-zero attack indicators. The timestamps and all indicators were excluded from the learning arrays. Each FIT file used its own scaler, while train2 used point-pooled FIT statistics. The two source domains each supplied 64,000 training windows. The checkpoint at step 4,000 had the lowest fixed source SELECT loss (0.523464); 1,001 of the 4,000 updates reached the maximum memory mixing weight. [File and scaler identities](source_inputs.json) · [Source training receipt](source_training.json).

The 38-channel source training took 152.54 A40 seconds. The 140 per-machine training runs and one shared source-only run took 12,116.83 A40 seconds. The 392 new SMD score banks took 202.10 seconds. Six pre-update setup attempts stopped at scaler or teacher checks and were recovered; the retry intervals, which include idle time, are listed separately. The six PSM/SWaT permutation trainings took a further 467.24 seconds and their twelve score banks took 12.88 seconds. Successful training and scoring totaled 3.60 A40 hours. CPU evaluation, file transfer and failed setup intervals are excluded from this total. [Training and recovery receipts](training_review.json) · [Metric review](metric_review.json) · [Permutation results and costs](permutation.md).
