# Complete SMD28 frozen Z results

This is the completed Z panel of the fixed SMD28 study. The U matrix has a separate endpoint gate.
Both TimeRadar source fits exclude SMD and use the normal TEP and HAI plans. All 28 SMD machines contribute all four scores.
Every device's Z scores use the same 612-point QUERY footprint that will be used by U. Values are equal-device mean and sample SD across devices.
There is one project training seed.

| Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | ---: | ---: | ---: | ---: |
| native Z | 0.508680 ± 0.279915 | 0.823137 ± 0.143833 | 0.522723 ± 0.271470 | 0.876205 ± 0.120870 |
| history Z | 0.510073 ± 0.281870 | 0.824876 ± 0.144118 | 0.524997 ± 0.275229 | 0.877945 ± 0.118983 |

| Contrast | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | ---: | ---: | ---: | ---: |
| History Z − Native Z | +0.001393 ± 0.010417 (15/0/13) | +0.001738 ± 0.013308 (18/0/10) | +0.002274 ± 0.014778 (18/0/10) | +0.001740 ± 0.009115 (16/0/12) |

Parentheses show wins, ties and losses across all devices.
The AP increment is small, with substantial device-level heterogeneity.
These Z data do not settle the U update-scope question.

Z scoring required 3.2262 A40 process-hours, with 134.59 MiB peak allocated.
Independent AP/ROC maximum discrepancy: 4.44e-16.
[Per-device metrics](Z_metrics.csv) · [Bank and footprint hashes](Z_bank_manifest.csv) · [Source and query contract](PROTOCOL.md)
