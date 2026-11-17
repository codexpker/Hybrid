# Fixed cached probability/logit ranking check

Completed cache-only check of A/B on all29 entities. QUERY comparisons use each bank's identical original full QUERY index and valid mask, not the cropped cross-model common axis. CAL reports only numerical relations/saturation; CAL AP/ROC are NA. No model, inference, threshold or primary table changed. No identical prior ranking check was found; the prior check only verified head-logit subtraction identity.

Probability is the archived native float32 softmax class1; logit is archived abnormal-class minus normal-class head output, not an inverse transform of probability. Both columns are reported for every entity, without selecting the better one. Full116 role rows, including normal numerical rounding and saturation collisions, are in probability_logit_comparison.csv.

| Full QUERY setting | Probability AP | Logit AP | Probability ROC | Logit ROC |
|---|---:|---:|---:|---:|
| A SMD28 macro | .486238 | .486366 | .838635 | .838635 |
| B SMD28 macro | .410389 | .472115 | .856963 | .859568 |
| A PSM | .358774 | .358774 | .550462 | .550462 |
| B PSM | .302273 | .302273 | .475716 | .475716 |

QUERY pooled across points for saturation descriptions only: A p=0:0%, p=1:.072715%; B p=0:.003391%, p=1:2.045812%. Points in probability bins merging distinct logits: A .116545%, B2.793431%. Largest B p=1 proportion is22.1181% on machine3-5. These pooled descriptive proportions are not pooled performance metrics. CAL p=0 is0% in both tracks; p=1 A.019965%, B1.554852%; points in merged bins A.027225%, B1.580867%. CAL has no ranking metrics.

Major B examples (probability AP -> logit AP): machine2-8 .146188 -> .935809; machine3-11 .057553 -> .537509; machine3-4 .279437 -> .507293. A maximum absolute AP difference is.004044, ROC difference.000005013. All probability-bin ordering checks found zero inversions; loss is chiefly tie compression/saturation, not a reversal of the numerical monotone relation. Refining ties does not guarantee an improvement: for example B machine3-9 AP decreases .763883 -> .745096. The recorded direct float64 sigmoid relationship calculation emitted benign exp-overflow warnings at very negative logits (limiting sigmoid0); no source/model result or ranking metric depended on that diagnostic expression.

Decision: probability expression causes substantial ranking-information loss for B on several SMD machines, but not broad loss for A or either PSM track. After uniformly reporting logits, A SMD AP remains above B (.486366 vs .472115), with a much smaller gap than probability (.486238 vs .410389); B SMD ROC remains higher than A. PSM A/B ranking difference remains unchanged. Existing probability-based CAL/alarms/false-alarm/coverage findings remain the original findings: this check does not retune thresholds or establish what logit-based alarms would do. Neither numerical saturation nor its correction is a methodological innovation. No follow-on task is started.

Raw diagnostic artifacts remain results/timercd_probability_logit_20261117_v1; original b19301f method comparison is unchanged. GPU use0; no training/forward/threshold fit.
