# Fixed single-event F comparison

| Task | Initialization | Phase | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---|---|---|---|---|
| PSM | Official initialization | Z | 0.55095 | 0.73505 | 0.48458 | 0.64816 |
| PSM | Official initialization | F | 0.54716 | 0.75300 | 0.47475 | 0.66318 |
| PSM | Reconstruction continuation | Z | 0.55607 | 0.73569 | 0.49478 | 0.65082 |
| PSM | Reconstruction continuation | F | 0.54371 | 0.75339 | 0.47335 | 0.66151 |
| PSM | Source V supervision | Z | 0.61342 | 0.78640 | 0.53270 | 0.67210 |
| PSM | Source V supervision | F | 0.53416 | 0.75208 | 0.47179 | 0.66017 |
| machine-3-2 | Official initialization | Z | 0.03604 | 0.50376 | 0.06849 | 0.72843 |
| machine-3-2 | Official initialization | F | 0.02720 | 0.70007 | 0.10216 | 0.76195 |
| machine-3-2 | Reconstruction continuation | Z | 0.03484 | 0.50602 | 0.06940 | 0.73055 |
| machine-3-2 | Reconstruction continuation | F | 0.02694 | 0.68994 | 0.10132 | 0.75903 |
| machine-3-2 | Source V supervision | Z | 0.03174 | 0.51506 | 0.06593 | 0.70779 |
| machine-3-2 | Source V supervision | F | 0.02669 | 0.69176 | 0.10016 | 0.75217 |

## Paired changes

| Task | Contrast | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | official F minus official Z | -0.00380 | 0.01796 | -0.00983 | 0.01502 |
| PSM | copy F minus copy Z | -0.01237 | 0.01770 | -0.02144 | 0.01068 |
| PSM | point_weighted F minus point_weighted Z | -0.07926 | -0.03432 | -0.06092 | -0.01193 |
| PSM | point_weighted F minus copy F | -0.00955 | -0.00131 | -0.00156 | -0.00133 |
| PSM | point_weighted F minus official F | -0.01300 | -0.00092 | -0.00297 | -0.00301 |
| machine-3-2 | official F minus official Z | -0.00883 | 0.19631 | 0.03367 | 0.03352 |
| machine-3-2 | copy F minus copy Z | -0.00791 | 0.18392 | 0.03192 | 0.02848 |
| machine-3-2 | point_weighted F minus point_weighted Z | -0.00505 | 0.17670 | 0.03423 | 0.04438 |
| machine-3-2 | point_weighted F minus copy F | -0.00025 | 0.00182 | -0.00116 | -0.00686 |
| machine-3-2 | point_weighted F minus official F | -0.00051 | -0.00831 | -0.00200 | -0.00979 |

## Existing CAL operating points

| Task | Initialization | Phase | Stream | Normal FPR | Recall | False fragments | Matched/eligible | Any-hit | Event F1 | Long macro | Long pooled | Longest miss | Hit-only delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSM | Official initialization | Z | strict | 0.29669 | 0.50870 | 185 | 18/48 | 28 | 0.14458 | 0.86955 | 0.92407 | 486.00000 | 0.00000 |
| PSM | Official initialization | Z | confirmed3 | 0.29096 | 0.48878 | 88 | 6/48 | 17 | 0.09023 | 0.86453 | 0.92114 | 694.00000 | 0.00000 |
| PSM | Official initialization | F | strict | 0.37777 | 0.67300 | 126 | 18/48 | 33 | 0.19355 | 1.00000 | 1.00000 | 399.00000 | 0.00000 |
| PSM | Official initialization | F | confirmed3 | 0.37433 | 0.64832 | 69 | 7/48 | 22 | 0.12613 | 1.00000 | 1.00000 | 399.00000 | 0.00000 |
| PSM | Reconstruction continuation | Z | strict | 0.29381 | 0.50416 | 176 | 18/48 | 28 | 0.14938 | 0.85028 | 0.91285 | 693.00000 | 0.00000 |
| PSM | Reconstruction continuation | Z | confirmed3 | 0.28855 | 0.48436 | 77 | 6/48 | 17 | 0.10000 | 0.84525 | 0.90992 | 718.00000 | 0.00000 |
| PSM | Reconstruction continuation | F | strict | 0.37604 | 0.67047 | 129 | 19/48 | 34 | 0.19588 | 1.00000 | 1.00000 | 397.00000 | 0.00000 |
| PSM | Reconstruction continuation | F | confirmed3 | 0.37253 | 0.64499 | 73 | 7/48 | 22 | 0.11864 | 1.00000 | 1.00000 | 399.00000 | 0.00000 |
| PSM | Source V supervision | Z | strict | 0.06724 | 0.40060 | 251 | 19/48 | 20 | 0.09291 | 0.55067 | 0.74797 | 874.00000 | 0.50000 |
| PSM | Source V supervision | Z | confirmed3 | 0.05919 | 0.37333 | 93 | 10/48 | 11 | 0.10638 | 0.53069 | 0.73561 | 874.00000 | 3.00000 |
| PSM | Source V supervision | F | strict | 0.36934 | 0.65520 | 121 | 18/48 | 33 | 0.21176 | 1.00000 | 1.00000 | 399.00000 | 0.00000 |
| PSM | Source V supervision | F | confirmed3 | 0.36606 | 0.63115 | 75 | 7/48 | 22 | 0.12844 | 1.00000 | 1.00000 | 399.00000 | 0.00000 |
| machine-3-2 | Official initialization | Z | strict | 0.00034 | 0.02033 | 4 | 1/8 | 1 | 0.16667 | NA | NA | 123.00000 | 0.00000 |
| machine-3-2 | Official initialization | Z | confirmed3 | 0.00007 | 0.01220 | 1 | 1/8 | 1 | 0.22222 | NA | NA | 123.00000 | 2.00000 |
| machine-3-2 | Official initialization | F | strict | 0.00041 | 0.00000 | 2 | 0/8 | 0 | 0.00000 | NA | NA | 123.00000 | NA |
| machine-3-2 | Official initialization | F | confirmed3 | 0.00014 | 0.00000 | 2 | 0/8 | 0 | 0.00000 | NA | NA | 123.00000 | NA |
| machine-3-2 | Reconstruction continuation | Z | strict | 0.00041 | 0.02033 | 3 | 1/8 | 1 | 0.18182 | NA | NA | 123.00000 | 0.00000 |
| machine-3-2 | Reconstruction continuation | Z | confirmed3 | 0.00021 | 0.01220 | 1 | 1/8 | 1 | 0.22222 | NA | NA | 123.00000 | 2.00000 |
| machine-3-2 | Reconstruction continuation | F | strict | 0.00014 | 0.00000 | 1 | 0/8 | 0 | 0.00000 | NA | NA | 123.00000 | NA |
| machine-3-2 | Reconstruction continuation | F | confirmed3 | 0.00000 | 0.00000 | 0 | 0/8 | 0 | 0.00000 | NA | NA | 123.00000 | NA |
| machine-3-2 | Source V supervision | Z | strict | 0.00034 | 0.01626 | 2 | 1/8 | 1 | 0.18182 | NA | NA | 123.00000 | 0.00000 |
| machine-3-2 | Source V supervision | Z | confirmed3 | 0.00014 | 0.00813 | 1 | 1/8 | 1 | 0.20000 | NA | NA | 123.00000 | 2.00000 |
| machine-3-2 | Source V supervision | F | strict | 0.00021 | 0.00000 | 1 | 0/8 | 0 | 0.00000 | NA | NA | 123.00000 | NA |
| machine-3-2 | Source V supervision | F | confirmed3 | 0.00007 | 0.00000 | 1 | 0/8 | 0 | 0.00000 | NA | NA | 123.00000 | NA |
