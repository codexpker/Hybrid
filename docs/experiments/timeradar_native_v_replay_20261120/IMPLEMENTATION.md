# Native TimeRadar head probing with matched natural-mixture replay

Public SWaT31/PSM25, seed2025. This experiment keeps the previous A/noU native V
and changes only the data composition during reconstruction-head updates.
It uses the official source/PT identities in the fixed config. Native FRFT,
encoder, fractional-order network, all buffers and module modes stay frozen/eval;
only the existing decoder's793610 parameters are trainable. There is no adapter,
CDL change, auxiliary loss, quantile predictor or score fusion.

Each training micro-forward uses B=1,100 observed points, norm0, ten complementary
masks reset with isolated CPU/CUDA seed84551. No inter-window order averaging is
allowed. The original reconstruction objective averages squared error over all
copies,100 positions and channels. Divide each of64 window losses by64 and
accumulate before one optimizer step; do not put64 windows into native forward.
TT=[T,Tprime], SS=[Sprime,S], R=[T,S], with32 windows per half. T and S are paired
in the same positions; the other streams are independently sampled, not copied.

Reuse the old PSM head-probing optimization settings: AdamW1e-4, betas(.9,.95),
eps1e-8, original LayerNorm/name-bias decay grouping (otherwise wd.1), linear
warmup200,2000 updates. Each arm restores the official initial model and fresh
optimizer. Keep endpoint2000; no SELECT/CAL/QUERY checkpoint choice. Frozen dropout
remains disabled; decoder has no training/eval-dependent layers. Each model has
128000 actual B1 window exposures, which are repeated windows, not new events.
Role-local full100-point nonoverlapping pools are anchored at each role start,
following the old head sampler. SWaT source/target pools3960/900; PSM1059/175.
Do not describe this as the LEFT256-batch replay protocol or equal model costs.

Two real64-window gradient gates pass: original noU V bitexact, nonzero decoder
updates through iFRFT, frozen states unchanged, B1 neighbor independence and RNG
restoration. Logical update1.3494s/1.4023s projects roughly4.6GPUh for six trainings;
12new CAL/QUERY readouts add roughly4.4GPUh. These are estimates, not budgets or
final occupancy. Four A40,20GiB training/30GiB scoring, owned timeouts/cleanup.
Actual runtime bytes are frozen separately from Git's text normalization.

The previous A's four CAL/QUERY banks and metrics are reused; no duplicate noU
inference. Six endpoints freeze before new readout, then all12 new banks seal
before label evaluation. Each new bank keeps ten terminal reconstructions,V,
actual order,observations,scaler and current100-window footprints; native V is
used unchanged. Bind actual endpoint SHA separately from official initialization.
QUERY partitioning only changes work allocation; merge the full fixed axis.

Keep SWaT336321/PSM64728 QUERY points and the same512 CAL observations; own
q99/higher/strict> thresholds. PSM CAL is all attack-labeled, not a normal FPR
budget. Natural-mixture results cannot isolate abnormal contamination. Mature
TimeRadar and replay retain their attribution; this is not an original Hybrid
mechanism claim. Main contrasts R−TT, R−noU, R−SS and TT−noU retain all operational
and sustained-coverage costs. No subsequent recipe is automatically queued.

The prior native-V RESULTS prose incorrectly said38 SWaT CAL attack points;
its archived methods.csv already says34. Only that sentence is corrected in
this branch. Scores,thresholds,metrics,weights and the historical audit snapshot
are unchanged; the original report remains available at commitf1f593a.
