# Normal-only relation qualification

This is a NumPy-only exploratory task, not an anomaly detector or a new TimeRadar/LEFT module. Pure functions live in `hybrid_tsad.normal_relation_screen`; each run uses a configuration instead of a new scientific driver.

## Frozen profile

The audited CLI profile uses the existing public SMD machine-1-1 train file, SHA256 `ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753`. Only FIT [0,13669) and SELECT [13861,18226) are converted to numeric values. The other roles remain sealed. Raw data, personal paths, releases and result arrays belong outside this repository.

FIT-only scaling, constant/affine-duplicate checks and AR8 innovation correlations select one directed pair and lag 1..32. Fixed AR8, AR32, synchronous, selected-lag and multiple-lag Ridge models are then compared on one common SELECT axis. Ridge minimizes mean squared error plus 0.001 L2 with an unpenalized intercept. The synchronous model observes another channel at the current time, never the current prediction target. This is current-state association, not advance forecasting or causal identification.

SELECT numerical correlation degeneracy uses a frozen float64 precision guard. A mathematically constant sequence must produce NA, rather than a spurious correlation from floating-point centering. This diagnostic guard does not change predictions or the selection rule.

## Preview and execution

Start from `configs/normal-relation-screen.example.json`. Preview reads only the configuration:

```text
python -m hybrid_tsad.normal_relation_screen --config /external/config.json
```

Real execution additionally requires the exact contract, bound config, source digest and external root release digest. Invoke the same module with `--execute --release /external/PRIMARY_APPROVAL.json --trusted-release-sha256 DIGEST`. Set OMP/MKL/OpenBLAS threads to 1 and CUDA_VISIBLE_DEVICES to the empty string. An independent parent must enforce the 120-second wall/CPU budget and sample the 4 GiB RSS cap; callbacks inside the worker do not enforce address-space limits.

No environment installation is required for pure tests. The local audited invocation used existing embedded Python 3.8 and NumPy 1.24.4 with an explicit source-path/runpy bootstrap because that interpreter is isolated and ignores PYTHONPATH. That is a temporary pure-module execution path, not an assertion that the whole package's Python >=3.10 installation contract was tested or changed. A normal interpreter with this package available can use the module command above. Keep machine-specific launch paths outside Git.

## Saved evidence and tests

The worker saves FIT preprocessing, all candidate correlations, selected pair, coefficients, original point indices, all five predictions, standardized/original-scale residuals, 25 model/block metric rows, 40 fixed comparison rows, descriptive correlation denominators/NA flags, execution metadata and a hash manifest. Four contiguous SELECT blocks are correlated descriptions, not four independent runs.

`tests/test_normal_relation_screen.py` contains ten synthetic tests. They cover the mean-loss Ridge objective, known lag, no current-target/future inputs, role isolation, constant/duplicate handling, sealed-role parsing, fixed sorting and numerical NA boundaries. Official models and weights are not imported.

## Recorded result and limits

The fixed 2026-11-02 profile selected source 5 -> target 6, lag 1 (zero-based anonymous channels, sample-index units). MULTI increased SELECT MSE by 88.49% relative to AR32 and improved zero of four blocks. It failed the prospective pilot continuation gate; the negative result is retained. Full raw evidence, independent audit and root acceptance are indexed from the parent research workspace's `results/START_HERE.md`.

There are no anomaly labels in this task: AP, Affiliation, Range-AUC, VUS, event recall and detection delay are NA. Neither a lower prediction loss nor this failed normal-only candidate establishes anomaly-detection performance. Do not replace the pair or tune against SELECT to force a positive result, claim causal lag, call the machine unseen, or infer that the entire lag research route failed.
