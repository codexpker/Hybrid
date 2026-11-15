# G36 bounded conditional-response mechanism verification

Status: fixed design, computation in progress. No outcome-based revision is permitted.

This experiment follows the user's explicit authorization to use **simulation** after the real paired-data premise could not be closed. It is not independent real-world confirmation, a contamination experiment, or evidence that the final Hybrid contribution has been established.

Data: Figshare article 29297999 version 3, archive file 57268553, `05_G36-5wk`. Only the four original CSVs were retrieved by bounded range reads. Original documentation specifies 5-minute samples and identical external conditions for a healthy baseline and three continuously injected faults: cooling valve stuck closed, outdoor damper stuck open, supply duct leakage 20%. Weather arrays match exactly. Occupancy samples differ at 13–14 transition points; each run therefore uses **its own past observed mode**, never the simultaneously simulated healthy counterpart. Fault-truth columns and closed-loop reset setpoints are excluded.

Response variables: supply temperature, mixed-air temperature, supply mass flow, duct static pressure. Context: outdoor dry/wet bulb temperatures, recorded occupancy mode, and time-of-day sine/cosine. These are statistical conditioning signals, not a claim of physical causal identification or authenticated actuator commands.

Normal baseline roles (zero-based half-open raw indices): FIT `[0,1152)`, SELECT `[1440,1728)`, CAL `[1728,2016)`. QUERY is `[2016,10021)` in all four files. The unused fifth day is a temporal buffer. Every model's score starts 51 samples into each role; all raw footprints stay inside their role. The entire injected interval `[2100,10020]` is scored. Post-injection recovery is excluded rather than automatically labeled healthy. Baseline future observations are evaluation-only. Scaling uses FIT only.

One conditional model is tested. For endpoint t, context is x[t−51:t−3] (48 strictly earlier observations); response is the four-dimensional block y[t−3:t+1]. The TCN's actual receptive field is 31 context points (155 minutes), not all 48 supplied points. Four affine coupling transforms alternate time and orthonormal two-level Haar coordinates, conditioned on the past TCN state. With Student-t(5) base density,

`score(t) = [-log p_base(f(y_block; context)) - log|det df/dy_block|] / 16`.

The normal FIT objective is exactly the mean of this per-example score. This is a conditional response distribution, not the stopped unconditional whole-window density/HMM, fixed reference retrieval, EMA, supervised F, or shared reconstruction/prediction decoder. Standard flow/Haar primitives are reused, not claimed as inventions. Legitimate evolution may receive high scores if normal FIT lacks its support; conditional density does not solve observational non-identifiability.

Fixed comparisons:

1. Conditional flow, the limited Hybrid mechanism candidate.
2. Matched history-only flow: identical tensors/initialization/updates, five context channels zeroed. It is unconditional with respect to operating context, not devoid of response history.
3. Official LEFT native training/inference, H48/C9, scoring four task-response channels. Official training retains all nine channels and its original loss; this task difference is disclosed.
4. Conditional Ridge, fixed alpha=1, 48 past points to the same four-point response block.
5. History-only Ridge, same settings excluding context channels.

All neural methods: seed2025, batch64, 3000 updates maximum, Adam1e−4, gradient clip1; identical FIT endpoint draws. SELECT every300 updates, including initialization, chooses each method's own normal training objective. No QUERY labels are used. LEFT's original warmup500/ramp1000 and architecture settings remain. CPU Ridge does not consume a neural training budget.

Each score uses its own normal CAL 99th percentile (`higher`, strict `>`), with **237 correlated calibration samples from one simulated day**. Nominal 1% is not certified future FPR. Report actual normal QUERY FPR, switching ±6 samples and other normal intervals. C3 starts only on the third consecutive hit; no retrospective filling. Per-scenario AP/ROC compares the entire fault run to the same normal query; pooled metrics count normal baseline only once. One-to-one event precision counts alarm fragmentation, separately from any-hit. Sustained coverage includes full injection and the interior after one hour; occupied interior is a descriptive subgroup. Three scenarios share weather and simulator, and daily operating periods are not independent faults.

No new model/threshold/seed search follows this run. A useful mechanism signal requires attributable conditional benefit against history-only and simple Ridge, sustained fault evidence, and tolerable measured normal switching errors. A simulation signal alone cannot authorize claims of real industrial confirmation or mixed-unlabeled robustness.
