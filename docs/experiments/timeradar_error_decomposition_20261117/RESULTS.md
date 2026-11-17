# Fixed cache-only error decomposition

V=mean_c archived sample variance (native ddof1); E=mean_c((mean reconstruction - input)^2); fixed combined=E+0.9V. All calculations are pointwise float32 on each task's identical full native axis and original FIT-standardized units (including native inverse normalization). No batch-scalar MSE, threshold fitting, training or forward pass. All six caches and inputs exist; identities, indices, shapes and finite values checked, reconstructed V matches archived native score within float32 reduction tolerance. No identical analysis found in current TimeRadar experiment records.

| Task | Arm | V AP/ROC | E AP/ROC | E+0.9V AP/ROC |
|---|---|---|---|---|
| PSM | zero | 0.431166/0.670150 | 0.454323/0.620565 | 0.463446/0.630006 |
| PSM | decoder_head | 0.426975/0.677992 | 0.353784/0.542891 | 0.374031/0.564435 |
| PSM | self | 0.426359/0.678390 | 0.353439/0.542471 | 0.373293/0.563831 |
| PSM | cross | 0.425339/0.680241 | 0.355426/0.543065 | 0.375622/0.563854 |
| PSM | selective | 0.423699/0.680241 | 0.354618/0.542498 | 0.375114/0.563899 |
| PSM | capacity_self | 0.426409/0.678637 | 0.353990/0.542973 | 0.373576/0.564088 |
| machine-1-1 | zero | 0.609689/0.902007 | 0.672197/0.942031 | 0.687574/0.950290 |
| machine-1-1 | decoder_head | 0.602559/0.904654 | 0.664183/0.936604 | 0.681657/0.947436 |
| machine-1-1 | self | 0.603667/0.905258 | 0.664156/0.936596 | 0.681493/0.947371 |
| machine-1-1 | cross | 0.616477/0.905044 | 0.666731/0.937055 | 0.683329/0.946913 |
| machine-1-1 | selective | 0.619314/0.905594 | 0.666158/0.937012 | 0.682904/0.946878 |
| machine-1-1 | capacity_self | 0.603850/0.905252 | 0.664260/0.936657 | 0.681559/0.947386 |
| machine-1-2 | zero | 0.282421/0.921863 | 0.217149/0.902662 | 0.237036/0.920532 |
| machine-1-2 | decoder_head | 0.299507/0.922483 | 0.197140/0.889647 | 0.223758/0.916820 |
| machine-1-2 | self | 0.298940/0.922488 | 0.197367/0.889712 | 0.224124/0.917197 |
| machine-1-2 | cross | 0.303594/0.922703 | 0.199779/0.891493 | 0.225248/0.917329 |
| machine-1-2 | selective | 0.299965/0.921148 | 0.199072/0.891411 | 0.225560/0.917679 |
| machine-1-2 | capacity_self | 0.299699/0.923056 | 0.197378/0.889803 | 0.223525/0.916602 |
| machine-1-3 | zero | 0.357354/0.864324 | 0.427002/0.916514 | 0.435533/0.923318 |
| machine-1-3 | decoder_head | 0.382512/0.887474 | 0.425138/0.920606 | 0.434616/0.925364 |
| machine-1-3 | self | 0.383640/0.887578 | 0.425868/0.920378 | 0.434902/0.925123 |
| machine-1-3 | cross | 0.385730/0.889472 | 0.426368/0.921945 | 0.435069/0.926015 |
| machine-1-3 | selective | 0.384068/0.889327 | 0.425180/0.921409 | 0.434105/0.925430 |
| machine-1-3 | capacity_self | 0.383687/0.887712 | 0.425961/0.920361 | 0.434834/0.925142 |

Full normal/anomaly counts, means, standard deviations, q05/median/q95/q99 for all72 combinations are in comparison.csv. These are descriptive QUERY-label diagnostics, not training or model-selection inputs.

Technical decision: the stronger explanation is that adaptation has not produced broadly additional anomaly information under these fixed criteria. There is also a local readout omission: E/combined expose better ranking on SMD1-1 and1-3, but that advantage already exists in the zero-update model. Adapted E AP is below zero-update E in all four tasks. Combined AP is also below zero-update combined in all four tasks. On PSM and SMD1-2, adding E worsens adapted ranking relative to V. Thus the caches do not support rescuing selective adaptation by changing readout. No best column is promoted; original scores and NO-GO remain unchanged. This bounded analysis does not prove no information exists in unexamined representations.

Zero GPU usage. The first analysis completed comparison.csv but provenance writing failed on a string/Path conversion; the small helper was fixed and the existing table retained, without rerunning inference or rewriting original artifacts. No new module or experiment follows.
