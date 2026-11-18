# Frozen seven-model SWaT B transfer evaluation

This run performs inference only: the untouched official multivariate TimeRCD checkpoint and the six previously selected batch/reference checkpoints from seeds2025/2026/2027. No weight selection, training, score changes or fusion. All original results remain immutable.

The exact existing31-channel raw order is preserved, including the original leading space in ` AIT201`. FIT is [0,296020), CAL input [445661,495000), QUERY input [608023,944919). Each role is independently inferred on the original5000-point grid anchored at its input start, batch1. Padding is not a real observation. The established QUERY comparison axis is [608535,944919), retaining all22 eligible events and the original9 long events (duration>512, steady starts at event start+512).

B uses raw FIT population mean/std(ddof0; zero→1e-8) exactly once, shared by all seven models. Thresholds are independently determined from each model's512 previously frozen CAL positions `np.linspace(446173,494999,512,dtype=np.int64)` using q99/higher and strict greater-than. Full CAL predictions are saved; no QUERY label or statistic chooses a threshold. These are offline retrospective development evaluations, not causal online or blind confirmation claims.

TASK_BINDING_PUBLIC.json contains safe channel/interval/checkpoint identities, not raw arrays or weights. Original historical LEFT/simple comparisons are reused only when their existing axes and permissions are compatible. No automatic follow-on experiment is authorized by this evaluation.
