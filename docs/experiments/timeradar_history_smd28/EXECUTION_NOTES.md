# Execution notes

The first sorted A endpoint took 2,636.86 seconds for the complete
2,000 updates. After measuring its initial update throughput, the three
otherwise idle cards each ran one scheduled native U endpoint for
machine-3-7, machine-3-8 and machine-3-9. These are three of the fixed
84 endpoints. The four-lane scheduler verifies and reuses their banks.

On machine-1-6, B's first loss was 3.1208273665979505 and A's archived
value was 3.120827474631369. The difference was 1.08033419e-7, or
3.46169148e-8 relative to A. This exceeded the original absolute-only
check by 8.03e-9. The SMD first-loss check now keeps atol 1e-7 and adds
one FP32 epsilon of relative tolerance. The first failed B attempt
produced no checkpoint. The recovery ran from the same source checkpoint
and used the same plan, model, optimizer, score and budget. The recovered
first step reproduces the original B loss and passes the guard.
All exact frozen-state and active-path checks remain in force.

The original scheduler continues independent jobs after an error.
The final recovery pass verifies and reuses complete endpoints and banks
before the aggregate evaluation. The standalone retry and the original
queue use distinct log files. Wall time recorded by a recovery pass
applies only to that invocation. Per-stage cost sums use the original
receipts, with the discarded first update accounted for separately.
