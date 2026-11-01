# Fixed normal Ridge baselines

`tep_normal_ridge` implements four simple, deterministic controls for the public
TEP three-variable task. It introduces no neural network. Native scikit-learn
Ridge uses float64, an unpenalized intercept and the Cholesky solver.

Variables are ordered `xmeas_9`, `xmeas_21`, `xmv_10`. The accepted FIT-only
scaler is fixed. Each run contributes target positions64–499; histories never
cross run boundaries and the current target is excluded from its own features.

| Family | Features per target |
| --- | --- |
| SELF | Own lags1–64 |
| SYNC | SELF plus both other channels at lag0 |
| PEAK | SELF plus each other's FIT-only maximum absolute Pearson lag1–64 |
| MULTI | SELF plus both others at all lags0–64 |

An undefined/nonfinite Pearson denominator raises an explicit error rather than
being replaced with zero. Real zero correlation remains a legal value. Exact
absolute-correlation ties choose the smallest lag. The earlier candidate and its
two counterexamples remain archived outside this package.

For each family, alpha is chosen from .1,1,10,100 using the mean squared error
over all three targets and independent normal SELECT runs. Exact ties choose the
larger alpha. Every candidate's coefficients, intercepts, ordered features and
normal predictions are retained. No anomaly labels choose a parameter.

The formal normal run completed48 fits and passed separate audit: all saved
FIT/SELECT predictions replayed exactly from coefficients; all16 losses and384
Pearson values were independently checked. All four families selected alpha100,
the upper edge of the frozen grid. The grid was not expanded after seeing this
result. Normal prediction improvements do not establish anomaly-detection or
causal gains; current other-channel readings may contain feedback information.

The target grid and training exposure differ from LEFT's window reconstruction.
Do not use their normal training losses as a direct model comparison. Normal CAL,
fault scoring and label evaluation have separate contracts and releases. Current
status, original continuous artifacts and primary receipts are linked from the
workspace's results/START_HERE.md.
