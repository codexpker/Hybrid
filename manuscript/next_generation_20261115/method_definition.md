# Current method: point-weighted source supervision of native endpoint variance

**Scope and motivation.** Keep the complete official TimeRadar detector, its ten complementary masks, and its endpoint anomaly score. Change source learning inside the native decoder. In the preceding fixed PSM development fold, the separately fitted synthetic scoring head had AP .261274 in Z even before target adaptation. The unchanged native score gave .550952 on the same points. Source synthetic SELECT loss fell from .500028 to .193758. This establishes that the fitted proxy and the actual detector objective diverged in this fold. Target contamination and binary calibration cannot explain a point-AP gap already present in Z. The symmetric-U versus Logistic-U gain of .00814 remains a narrow result. The source-positive distribution assumption still needs evidence. [Preceding development fold](../../docs/experiments/native_pair_rank_20261122/PSM_FOLD_METRICS.json) · [Symmetric losses](https://proceedings.mlr.press/v97/charoenphakdee19a.html).

**Proposed learning object.** Start at the official checkpoint. Freeze the encoder, fractional order and all existing non-decoder weights. Update only the shared decoder of the ten native reconstructions. All forward calls remain in eval mode, with batch one, the original 100-point right-aligned input window and fixed mask seed 84551. Let $V_\theta(x)$ be the original channel-mean sample variance (`correction=1`) at the window endpoint. Use the original per-copy reconstruction MSE $L_{\mathrm{copy}}$ on nominal source windows. For the added objective, use annotated public source incidents. Require the complete 100-point positive footprint to lie within one labeled event and the matched normal footprint to lie within a contiguous label-normal interval of the same release and source role. Split each qualifying event's legal endpoints into three consecutive, nearly equal groups. Use these three groups as sampling strata and draw one positive per stratum and one independently sampled normal from the same source role.

For positive and normal endpoints $p$ and $n$, let $z_\theta(x)=\log(1+V_\theta(x)/a)$ with $a=\max(10^{-8},\operatorname{median}_{n\in\text{source rank FIT normals}} V_{\theta_0}(n))$ from the frozen initialization. Write $c_{e,k}$ for the number of legal positive endpoints in stratum $k$ of source FIT event $e$, and $n_{e,k}$ for its sampled nominal comparison window. The fixed Stage 1 decoder objective is

$$
L_{\mathrm{copy}}+\sum_{e=1}^E\sum_{k=1}^3\frac{c_{e,k}}{\sum_{i,j}c_{i,j}}
\operatorname{softplus}\big(1-z_\theta(p_{e,k})+z_\theta(n_{e,k})\big).
$$

The unit margin and coefficient one are fixed. The pair term acts directly on the deployed score. It encourages each sampled event stratum to clear a margin over normal windows. The groups stratify the draws in time. The archived event-balanced ablation used the same samples and replaced $c_{e,k}/\sum_{i,j}c_{i,j}$ by $1/(3E)$. The groups do not identify physical phases. The first 99 positions of each source event have no full positive footprint in this objective. Onset and delay are separate evaluation quantities. Source labels specify event intervals. They give no fine-grained physical fault onset, weak-amplitude or lag truth. Decoder updates can help or hurt the original score on target data. The copy loss and source validation control what is trained and selected. Target-score preservation remains a hypothesis. This fold is a source-supervised, zero-target-label transfer validation. Real public source events supervise learning and checkpoint selection. First-paper laboratory data and weak labels are excluded. This candidate does not replace the project's multi-source self-supervision and mixed-unlabeled research. [TimeRadar, original reconstruction, CDL and readout](https://arxiv.org/html/2602.19068v1) · [Score bias under limited labeled anomalies](https://www.ijcai.org/proceedings/2021/456).

**Source and target roles.** The first fold again holds out the entire PSM dataset from the project's Stage 1. Normal copy pools are the already admitted SWaT normal train prefix and train-side SMD28 FIT roles, using their independent nominal SELECT blocks and fixed original FIT scalers. The real-event ranking source is SWaT's earlier exposure-audit interval `[495000,585000)`, now explicitly a labeled **source** for this PSM fold, not a source of any target labels. The published event segments stay intact. The nine eligible events are split in time by a cut at 502578 into five FIT events and four SELECT events. Only these two disjoint roles see their own label-normal comparison points. The original SWaT QUERY and PSM annotations remain out of project Stage 1. The earlier SWaT U experiment retains its original unlabeled protocol. Any SWaT results produced by this trained checkpoint would be seen-source results. Five-domain unseen-target claims require separate leave-one-dataset-out source manifests, with a separately eligible labeled-event source for each fold and upstream exposure disclosed. As the original paper declares, official weight pretraining used Monash+ with synthetic anomalies and excluded its downstream evaluation sets. This project's exclusion is auditable separately. [TimeRadar, Appendix A](https://arxiv.org/html/2602.19068v1#A1.SS1).

**Direct fixed comparison.** First evaluate Z under the common PSM causal endpoint and archived scaler/CAL. This is the historically exposed development domain held out of project Stage 1. The frozen official detector is the direct reference. The decoder-only copy continuation uses the same 16 SWaT plus 16 SMD normal windows and AdamW steps, with the native copy loss only. Sample SMD entities uniformly within its half of the copy batch. The point-weighted arm additionally uses 15 positive endpoints and 15 matched normal endpoints per step. The historical event-balanced ablation changes only their weights. Comparison with reconstruction continuation measures the full effect of the additional labeled exposure and objective. All decoder arms start from the same official checkpoint, use seed 2025, one 240-update cap, the existing decoder AdamW settings with 24 warmup steps, and the same fixed draws. All retain the native readout. All three decoder arms use the earliest minimum on the same disjoint event-macro source SELECT criterion, including the arm with reconstruction gradients only and source-event validation selection. Check selection at initialization and every 40 steps, with independent nominal-source SELECT MSE reported separately. Transfer requires improvement on real endpoint comparisons after source lock. Report AP/ROC and VUS point ranking plus complete-event and sustained coverage and the existing strict/C3 work points. Keep the PSM attack-only CAL limitation. The first development run begins with Z and matched source learning only. Target annotations play no role in estimating the scale or selecting a model.

**Same method in Z/U/F.** Z applies the selected native variance detector without target parameter updates. For a later U comparison retain the point-count-weighted source pair term above with its original source FIT counts and scale while applying the same per-copy reconstruction term to a fixed, matched mixture of nominal source and unlabeled target FIT windows. Both U arms start from the same selected Z checkpoint and have identical target inputs. Ordinary U sets the pair term to zero. This comparison tests the incremental effect of the known source score objective during mixed adaptation. Robustness to a specified contamination intervention remains empirical. F, when separately admissible, uses the same decoder, score and pair loss with up to two complete public support events isolated from the target query and from selection. F's label cost is reported separately. No test labels, first-paper data or first-paper weak labels enter any setting.

**Public benchmark selection.** PSM development selected the point-weighted recipe. Source-event SELECT fixed each repeat checkpoint before target scoring. For seeds 2025, 2026 and 2027 it selected steps 240/240/240 for point weighting and 80/40/80 for reconstruction continuation. The official frozen variance supplies the common zero-update reference. The full PSM ranking axis extends the exposed development task. MSL and SMAP are held out of this project's source training. SMD28 and SWaT contributed nominal source windows, and SWaT also contributed labeled FIT and SELECT events.

**Implementation and first-fold result.** The native endpoint scoring/gradient wrapper, chronological source event-role packer and matched pair losses are implemented in [`native_event_rank.py`](../../src/hybrid_tsad/native_event_rank.py). The matched [training](../../src/hybrid_tsad/native_event_rank_training.py) and [causal readout](../../src/hybrid_tsad/native_event_rank_readout.py) completed under the fixed protocol. Source SELECT chose steps 80, 240 and 240 for reconstruction-gradient continuation, point weighting and event balancing. Their PSM AP values were .556073, .613415 and .588312, with frozen native variance at .550952. Real public source supervision improved ranking in this fold. Event balancing gave lower ranking metrics than point weighting and higher long-event coverage at a higher observed false-alarm rate. In the [three-seed evaluation](../../docs/experiments/native_event_rank_repeats_20261122/RESULTS.md), AP and VUS-PR improved against both controls on PSM, MSL and SMAP. SMAP had additional false-alarm fragments and PSM lost sustained coverage. On the [SMD28 multivariate held-out test](../../docs/experiments/native_event_smd28_20261123/RESULTS.md), the same three supervised decoders lowered macro AP relative to frozen V by 0.00460 ± 0.00254 and to reconstruction continuation by 0.00946 ± 0.00397 (paired mean ± sample SD). ROC-AUC increased in every repeat, while both VUS metrics and three-confirmation event F1 fell against both controls. These machines contributed nominal source FIT windows. The public comparisons therefore establish task-dependent ranking gains under source supervision. U/F remain untested for this recipe. [Fixed-fold results and direct ablation](../../docs/experiments/native_event_rank_20261122/RESULTS.md).

The source packer was checked using only the archived 90,000-point source-eligible SWaT label slice, SHA-256 `2a9fb52f192067dd258f73be4691678b1c42e09c3dc87d259fea616640818039`. FIT contains 1,857 legal positive endpoints and 4,632 legal normal endpoints. SELECT contains 2,439 and 79,092 respectively. Every input footprint stayed in its source role. A CPU engineering fixture using the archived official checkpoint matched the original endpoint variance and copy MSE exactly. The pair objective gave nonzero decoder gradient norm 0.46395, with no frozen-parameter gradient and unchanged weights, buffers, and external RNG. The two focused tests passed. The independent review verified the matched draw plan, source-only selection, non-decoder state identity, cached native variance and PSM metrics. [Run and review provenance](../../docs/experiments/native_event_rank_20261122/PROVENANCE.json).

# Archived development experiment: symmetric ranking of time–frequency and causal patch evidence

This was a small synthetic-supervision scoring-head development experiment on frozen TimeRadar features. The fixed SWaT/SMD → PSM fold and all positive and adverse differences remain archived. This head is closed to additional seeds, F, contamination sweeps and new score variants.

The second paper studies public multivariate time-series anomaly detection under mixed-unlabeled adaptation. Hybrid is the proposed detector under validation. We use the full, frozen [official TimeRadar](https://github.com/mala-lab/TimeRadar) reconstruction chain with its official weights, FRFT/iFRFT, adaptive order and ten complementary masks. The release paper describes an external Monash+ corpus with synthetic anomalies and exclusion of the downstream datasets. We record that as the authors’ disclosure. This project did not independently recover the entire upstream training corpus. The official per-channel variance is a direct comparator. A shared variable-count-independent head can transfer between SWaT, SMD and PSM without fitting a separate encoder to each channel layout. The current LEFT assets are strong external references with task-dependent channel counts and window readouts. [TimeRadar paper, Appendix A](https://arxiv.org/html/2602.19068v1#A1.SS1).

**Endpoint representation.** At time $t$, evaluate one native window $x_{t-99:t}$, with independent batch size one and archived mask seed 84551. For channel $c$, retain the sample variance $v_{t,c}$ of its ten terminal reconstructions, correction one. The temporal path uses $x_{t-96:t-1,c}$ split in chronological order into three disjoint 32-point patches. For each patch use its within-block standard deviation and root-mean-square first differences. Form seven features per channel by applying $\log(1+z)$ to the native variance and the six history statistics. A shared 7→24→16 GELU MLP and mean/max channel reductions produce a 32-dimensional vector. A 32→16→1 GELU head emits one logit $s_\theta(x)_t$, higher being more anomalous. The head adds 1137 learned parameters. This construction uses only observed inputs at $t$. It uses three short past patches and supplies no claim of learned long-lag or Cross-lag structure. Shared nonlinear features provide a test of joint evidence, with a symmetric-risk control that drops the six temporal features.

**Stage 1.** The inherited official encoder is frozen. Pretrain the shared scoring head on disjoint, released nominal-normal SWaT and SMD train/validation roles. Use all 28 SMD entities with equal per-entity pool sizes and sample both source datasets equally at each update. Realized per-machine counts may differ because draws within the SMD pool use replacement. Exclude all PSM values and labels from the project head's source pretraining and selection. Each base window supplies a matched counterfactual with a fixed signed one-channel 0.5 standardized-unit raised-cosine shift over its last 32 points. It rises over 24 points and then holds. This is **synthetic supervision**, with no real anomaly labels, and no project reconstruction or teacher loss. Train on the paired objective $\ell(s_\theta(x^+)-s_\theta(x))$ and select the earliest minimum held-out source-pair loss from fixed checkpoints including initialization. Source normality is nominal. Detection of synthetic perturbations provides no independent evidence of weak real-anomaly detection.

**Stage 2.** Z scores without any target parameter updates. It uses the already fixed PSM nominal-source-prefix scaler and the original 512-point mixed CAL. For U, draw synthetic positive source anchors $A$ and unlabeled PSM FIT windows $U$ independently and rank $s_\theta(A)$ above $s_\theta(U)$ with $\ell(d)=\sigma(-d)$. No target labels, anomaly ratio or inferred clean flags enter updates. The test below pairs windows and update counts with an ordinary ranking arm using $0.5\,\mathrm{softplus}(-d)$, whose slope at a tie also equals $-1/4$. This comparison tests the chosen loss as a whole, including its tail gradients. F begins from the frozen U head and reuses the feature and scoring path and ranking form, replacing synthetic anchors with sampled positive windows from at most two eligible, complete, labeled public support events and matching negatives from a separated annotated normal support role. Give the included events equal weight and keep their complete event inputs and labels disjoint from query. F requires identified target support roles under separate event admission. It is outside this initial PSM fold.

**Why symmetry targets mixed adaptation.** If the score distribution of the positive anchors matches that of the target positives and draws are independent, $\ell(d)+\ell(-d)=1$ gives $\mathbb E_{P,P}\ell(s(P_1)-s(P_2))=1/2$. For target mixture $U=(1-\pi)N+\pi P$, the population ranking risk equals $(1-\pi)R_{P,N}+\pi/2$. The contaminated term then contributes no directional signal at the population level. The identity may fail for synthetic-to-real distribution mismatch, dependent windows or an unrepresentative source pair. Even under matching it does not guarantee AP, event coverage or calibration. We credit the symmetric ranking principle to prior work. The proposed application tests whether the shared time–frequency/temporal scoring path with public synthetic anchors makes it effective under actual mixed TSAD adaptation. Ordinary replay, MeanT, score projection and old score fusion are separate archived experiments. [Symmetric loss under label noise](https://proceedings.mlr.press/v97/charoenphakdee19a.html).

**Frozen development comparison.** One seed 2025 and one held-out-dataset development fold. Stage 1 draws 112 FIT and 56 SELECT windows per source dataset, with four/two per SMD machine, and uses 240 updates, 16 pairs per source per step, Adam 0.001 and source selection every 40 steps. U uses 224 legal PSM windows in `[0,17568)` and 120 endpoint updates from the already selected Z initialization, with 16 positive anchors per source and 32 target comparison windows. Compare official native V, Z, U with symmetric ranking, U with ordinary logistic ranking, and U with no past-patch features. All use the native-V PSM QUERY axis `[23016,87744)` and 512 fixed legal CAL points. Match the exposure plan, avoid query selection, and report AP, ROC, VUS, strict/C3 normal FPR and event/coverage/delay using the archived endpoint conventions. PSM CAL contains only attack-labeled points. There is no normal-FPR certification. No F or new contamination matrix is run in this development step. A gain over the ordinary U control identifies only an initial fixed-fold increment. For an original method claim, the full score also needs credible evidence beyond the native reference and a justified and testable contamination mechanism.

Implementation is in [`native_pair_rank.py`](../../src/hybrid_tsad/native_pair_rank.py), with [`native_pair_rank_data.py`](../../src/hybrid_tsad/native_pair_rank_data.py) for isolated feature roles and [`native_pair_rank_run.py`](../../src/hybrid_tsad/native_pair_rank_run.py) for sealed training and evaluation. Readouts on existing legal native caches reuse the archived replicas. No data or checkpoint is redistributed here.

### First development fold: results and scope

Source pretraining and U adaptation completed at seed 2025. Both source regressions chose step 240 by the frozen source SELECT rule. The full head's held-out synthetic pair loss moved from 0.500028 to 0.193758. The head without past patches moved from 0.500001 to 0.499990. Those losses measure the source proxy. Final heads, input plans and both sets of target scores were sealed at their designated stages. The independent evaluator verified input identities, all 58 source-role banks, the regenerated U plan and 155 point, event and calibration checks. The maximum absolute difference in those metric checks was $5.55\times10^{-17}$. [Full-precision measures and artifact hashes](../../docs/experiments/native_pair_rank_20261122/PSM_FOLD_METRICS.json).

All rows use the same 64,728 PSM QUERY timestamps, 512 attack-labeled calibration points and the fixed q99 / higher / strict-greater rule. C3 means three consecutive point alarms with no backfill. Delays are median samples among hit events. Long steady coverage is macro-averaged, with distinct pooled coverage available in the full result.

| Fixed method | AP | ROC-AUC | VUS-PR / ROC | Strict / C3 normal FPR % | Strict / C3 point recall % | C3 matched / any-hit (48 events) | C3 complete-event point cover % | C3 long steady macro % | C3 longest miss / first-hit median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official TimeRadar terminal V | .550952 | .735048 | .484578 / .648161 | 29.669 / 29.096 | 50.870 / 48.878 | 6 / 17 | 73.195 | 86.453 | 694 / 0 |
| Hybrid Z, multisource head frozen | .261274 | .414047 | .303908 / .454379 | 2.067 / 1.885 | 4.484 / 4.024 | 22 / 23 | 5.410 | .263 | 3389 / 9 |
| Hybrid U, symmetric ranking | .302144 | .465191 | .351311 / .499329 | 2.184 / 2.017 | 4.478 / 4.093 | 20 / 21 | 5.183 | .226 | 3660 / 10 |
| Matched U, ordinary logistic ranking | .294008 | .454991 | .341183 / .487707 | 2.190 / 2.029 | 4.294 / 3.887 | 20 / 21 | 4.927 | .221 | 3661 / 10 |
| Symmetric U without past features | .182856 | .282692 | .218115 / .273811 | 61.360 / 61.045 | 20.713 / 20.259 | 17 / 25 | 18.514 | 0 | 4852 / 0 |

Symmetric U improves AP by 0.00814 over the exposure-matched logistic arm and by 0.04087 over Z on this fold. It leaves matched C3 event counts unchanged versus logistic U and raises C3 point coverage by 0.256 percentage points. Against native V it loses 0.24881 AP and most long-event coverage. The higher C3 matched-event count coexists with more than five times the longest unalarmed event interval. The lower observed FPR is at a different model-specific operating point from the same attack-only CAL. These results do not establish performance at equal normal false-alarm budgets, generalized contamination robustness or a validated Hybrid improvement. The exact mixture identity assumes compatible source-proxy and target-positive score distributions, which this fold cannot establish. The study records this fixed-fold result and leaves the full public Hybrid row pending. F, corruption stress tests, repeat seed and wider confirmation remain unexecuted.

Frozen feature calculation used 728 new native windows with 168 cached clean source windows. Cost was 12.86 A40 seconds (about 0.00357 GPU-hours) and 4.30 CPU seconds for the score-head fits. The 512 CAL and 64,728 QUERY feature windows reused the original terminal replicas with zero new native calls and 3.51 CPU seconds for cache readout. An initial missing module in the small deployment and a missing byte-exact VUS source at its previous local path were resolved before the respective stages. Logs remain archived. No successful training or official inference was repeated.

# Archived candidate: conditional energy in frozen TimeRadar response geometry

The current implemented research candidate combines a strictly historical
conditional generator with the frozen official TimeRadar response function.
Only public SWaT31 and PSM25 are used. BAEF-AD laboratory records, weak labels,
experimental runs and enterprise data are excluded. The former LEFT+R results
remain locked external learning controls; they are not replaced or renamed.
The fixed A/B experiment is complete. It does not establish a response-geometry
ranking increment that survives mixed adaptation. The bounded SWaT noU
false-alarm/coverage trade-off is retained; this candidate is not expanded.

Let $h_t=x_{t-512:t-1}$ and let $v_k=G_\theta(h_t,\epsilon_k)$ be four independent
conditional current-value draws. $G_\theta$ uses the official TimesFM3 residual
and mixing blocks at a lightweight configuration, including variate attention;
it does not load official TimesFM3 pretrained weights or its quantile head.
The current observation is excluded from the generator input. A separate
frozen TimeRadar invocation evaluates each $[x_{t-99:t-1},v]$ window with its
own order and ten fixed complementary masks, preserving the native complex
encoding, decoding, iFRFT and magnitude projection [@timeradar2026].

The full geometry is

$$g_{h_t}(v)=\left[v/\sqrt C,\;\operatorname{vec}r_{h_t}(v)/(s_r\sqrt{1000C})\right],$$

where $s_r$ is fixed using only normal SOURCE_FIT responses. The paired raw-only
model uses $g_{h_t}(v)=v/\sqrt C$. Both minimize and report the same energy score

$$s_t=\frac14\sum_{k=1}^4\|g_{h_t}(v_k)-g_{h_t}(x_t)\|_2
-\frac1{24}\sum_{k\ne l}\|g_{h_t}(v_k)-g_{h_t}(v_l)\|_2.$$

This is half the convention in the authors' generative scoring-rule code
[@generative_scoring2024]. Conditional generative energy training is an
established method; the hypothesis here is that fixed adaptive time-frequency
response geometry supplies useful detection evidence beyond the matched raw
geometry. The raw coordinates preserve injectivity, supporting strict
properness of the population score under finite first moments. This does not
guarantee abnormal observations have high score, calibrated normal false-alarm
control, or robustness when mixed adaptation learns abnormal states.

Only the generator is optimized. Generated-input gradients pass through the
frozen response computation; no_grad is restricted to observed responses.
Separate B=1 calls avoid sharing the fractional order across independent
windows or draws. eval() does not fix native random masks: isolated seed84551
is explicitly restored per response call, while generation noise has its own
paired streams. The two arms share source initialization and plans,2000 source
updates with normal SELECT selection including step0, then2000 fixed-endpoint
1:1 source-replay updates. They retain separate noU and R states. These are
same-domain normal initializations and natural mixed adaptations, not a new
heterogeneous pretraining experiment. Replay itself is not an originality
claim. The candidate uses logicalB16; historical LEFT+R usesB256, so external
cost/exposure comparisons are not equal-budget objective ablations.

Strict history plus current requires513 valid points. SWaT uses the existing
336321-point common axis; PSM excludes360 additional initial points solely for
this footprint, leaving64728 points. Cached methods are recropped without
changing their original scores. Their availability indices, source statistics
and offline permissions remain explicit. New mixed-CAL points are legal for
the actual footprint and thresholds use q99/higher/strict greater-than in each
model's own units. PSM's all-attack CAL remains a limitation. No favorable
results from old replay or controlled-contamination experiments are inherited
by this candidate. The implementation and fixed definitions are recorded in
[the implementation note](../../docs/experiments/tf_conditional_energy_20261119/IMPLEMENTATION.md).
The completed [full method and paired tables](../../docs/experiments/tf_conditional_energy_20261119/RESULTS.md)
retain all positive and negative results. SWaT A−B AP is−1.0594pp/noU and
−7.8658pp/R; PSM is+0.0165pp and−0.5707pp. SWaT A_noU C3 normal FPR is.8917%
versus B16.8674%, while complete-event coverage is63.8676% versus74.7933%.
After R, A's SWaT coverage falls to1.0486% and long steady macro/pooled
coverage to.1809%/.1238%. This is not a persistent-detection or pollution
robustness contribution. Threshold-free ranking losses cannot be explained
by CAL alone; this experiment does not uniquely locate the learning cause.
No new candidate or local repair is started automatically.

# Locked external learning basis: native LEFT with source replay R

The locked LEFT study addresses adaptation and detection retention of time–frequency multiview detectors under mixed-unlabeled target data. Its empirical basis is complete native LEFT plus established source replay R, not a new backbone or an original replay mechanism. The native objective, prototype/curriculum state updates, retrospective full-window inference and per-model CAL remain intact. P/V is stopped and retained below solely as historical evidence; no preservation term or teacher entered that completed replay experiment.

The natural-mixture and additional-exposure experiments are locked after two tasks and three seeds. A completed seed2025 external CANDI comparison retains its official MLP, full FPM and input/output SANA, with source training and source-SELECT references, chronological U-FIT adaptation and frozen CAL/QUERY. It does not modify this R rule or become a Hybrid module. Its own noU comparison establishes no useful net adaptation increment in the fixed task; cross-backbone ranking advantages and coverage costs are reported separately. CANDI's10-point endpoint MSE, FIT-only scaler and update schedule differ from native LEFT's192-point retrospective outputs and original scaler. See the [full fixed comparison](../../docs/experiments/candi_native_a_20261118/RESULTS.md); no follow-up method, seed or B/C matrix is automatically opened.

## Completed cross-backbone control, 19 November 2026

The fixed plain-MLP comparison adds only TT/R at seed2025 in SWaT and PSM, initialized from the accepted source MLP without SANA/FPM. It reuses each original LEFT192-point exposure endpoint with a10-point suffix, freshAdam1e-4/defaults/B256/2000 endpoints, native MSE and unchanged mixed-CAL rules. R−TT ranking gains in both backbones support general replay, not a frequency-specific invention. MLP is cheaper and strong in ranking but retains extreme SWaT false alarms and low PSM sustained coverage; LEFT has useful full-configuration coverage/operating trade-offs with higher cost. Different context, scalers and source losses preclude a pure representation attribution. [The complete fixed comparison](../../docs/experiments/mlp_replay_20261119/RESULTS.md) closes this task without adding methods or changing the R rule.

## Matched additional-exposure construction

For each domain, A is the unchanged natural U-FIT plan: $R_A=[T,S]$, $TT_A=[T,T']$. R and TT share T position by position; T′ is an independent target draw and S is actual source FIT. With 2,000 updates and 128 windows per half, T has 256,000 exposure slots and each full plan has 512,000. These counts measure repeated training exposures, not independent observations.

The current factorial comparison is $\{TT,R\}\times\{B,C\}$. A constructor may inspect labels only within current U-FIT to qualify full 192-point abnormal donors (B) and full 192-point normal donors (C). It replaces the same 25,600 shared-T slots in both conditions and both arms: 10% of T and 5% of all exposures. Within each condition, donor draws are paired across R/TT. T′ and S do not change. Current-U-FIT eligibility is confirmed: SWaT has3,468 all-abnormal /82,903 all-normal windows with nine qualifying abnormal event intervals; PSM has649 /12,852 windows with five qualifying intervals. Repeated overlapping windows are not independent events. Achieved exposure and donor reuse are audited separately from training.

The learner remains label-free: it receives windows, runs the original native forward/loss and updates without donor labels, anomaly weights or label-based skipping. Construction is nevertheless label-assisted and must be disclosed. The natural background may already contain anomalies. Consequently, matching added abnormal exposure does not equate total contamination between R and TT; neither B−A nor the slot fraction is the total anomalous-point proportion. Donors must be role-contained and valid at the complete-window level. There is no QUERY-based donor selection.

## Paired estimand and reporting

For a metric $M$, arm $a\in\{R,TT\}$ and condition $k\in\{A,B,C\}$, let $M_{a,k}$ denote the evaluated outcome. Report absolute values and all within-arm B−C, B−A and C−A differences, followed by

$$I_M=(M_{R,B}-M_{R,C})-(M_{TT,B}-M_{TT,C}).$$

B−C estimates the controlled abnormal-versus-normal replacement contrast under the shared slot plan; C−A checks normal replacement/composition effects against the original plan. Interpret $I_M$ in each metric's direction and together with absolute ranking, event discovery, sustained coverage and normal false alarms. The experiment does not require B to harm either arm and cannot turn an unchanged detection floor or a favorable difference of differences into proof of robustness. It neither identifies unknown contamination in deployment nor certifies difficult-normal protection.

Initialization, native training schedule, original role permissions, update budget, shared draws, inference and CAL rules are held to the matched contract. Each resulting model is calibrated on the same authorized CAL locations in its own score units; CAL is not a learning input. PSM's previously verified all-attack CAL remains a limitation, not a normal 1% false-alarm certification. The fixed replication is now complete at seeds2025/2026/2027:16 additional B/C jobs, with all old assets reused. Each seed uses its own accepted A plan and the same isolated replacement rule, giving joint adaptation/intervention-sampling replication at a fixed native initial model. Pairwise effects are computed within seed, then mean and sample SD(ddof1,n3); noU remains n1/SD=NA.

SWaT has positive I_AP and I_C3 complete-event coverage in all three seeds, but effect magnitudes vary and no additional event is found. Direct R-B has fewer false-positive points than TT-B despite the positive mean FP interaction. Long macro coverage remains low and the pooled denominator is dominated by one long event. PSM B-C AP is positive for both arms in every seed, so there is no corresponding ranking-specific contamination penalty; positive coverage interaction does not reverse its worse absolute event/coverage trade-off. See [all fixed results](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md).

The learning rule remains one full native LEFT forward on concat(T_k,H_a), with H_R=S and H_TT=T-prime; global/native state terms are not replaced by an average of separate forwards. No preservation term, teacher or inference change is introduced. The source-replay control and native LEFT retain their attribution. The experiment is locked without automatic additional seed/rate/loss. Earlier natural-mixture evidence and stopped definitions remain below and in the draft; these findings do not establish general cross-domain/field robustness or an original Hybrid mechanism.

# Historical stopped P/V source-output preservation experiment

This historical section defines the completed P/V experiment, evaluated at seed2025 on SWaT and PSM. It added a training constraint to native LEFT source-replay R and is now stopped. Its formulas and results are retained, but neither term defines the current learning rule. It is separate from the stopped density design archived below. Neither functional preservation nor replay is claimed as an invention.

## Frozen teacher, source permission, and outputs

For each domain, let $\theta_0$ denote its fixed native pretrained LEFT checkpoint and $\theta$ the adapting student initialized from it. A frozen teacher with parameters $\theta_0$ reads only the authorized source FIT windows $S$; it does not read target FIT $T$, SELECT, CAL, QUERY or labels. R trains with the native objective on concatenated target/source batches $[T,S]$. Each preservation term is evaluated on the same source half, without replacement by newly selected examples.

For a source batch of $B$ windows of length $T_w=192$ with $C$ channels, native `_forward` returns three reconstructions with identical shape $B\times192\times C$ and the same input-coordinate units:

$$g_1(\theta;S)=\widehat X,\qquad g_2(\theta;S)=\widehat X_{\leftarrow f},\qquad g_3(\theta;S)=\widehat X_{\mathrm{ms}}.$$

These are respectively `X_hat`, `X_hat_from_S` and `X_hat_ms`. They are outputs of interacting, shared native pathways, not three independent views or three independently trained teachers. Define $\Delta_v=g_v(\theta;S)-\operatorname{stopgrad}g_v(\theta_0;S)$, $\overline\Delta=\frac13\sum_{v=1}^3\Delta_v$, and $D=BT_wC$. The squared norm below sums all batch, time and channel entries. With all outputs evaluated in the same source coordinate system,

$$P=\frac{\|\overline\Delta\|_F^2}{D},\qquad V=\frac{1}{3D}\sum_{v=1}^3\|\Delta_v\|_F^2,$$

$$\mathcal L_P=\mathcal L_{\mathrm{native},R}+\lambda P,\qquad \mathcal L_V=\mathcal L_{\mathrm{native},R}+\lambda V,\qquad\lambda=1.$$

Here $\mathcal L_{\mathrm{native},R}$ is the complete native LEFT loss on $[T,S]$, including its original weighted components and state updates. The added terms constrain source outputs, not their agreement with ground truth or their deployed anomaly scores. In particular, native reconstruction heads need not be accurate merely because their teacher responses are preserved.

The finite-dimensional variance identity gives

$$V-P=\frac{1}{3D}\sum_{v=1}^3\|\Delta_v-\overline\Delta\|_F^2\geq0.$$

To verify it, expand $\Delta_v=(\Delta_v-\overline\Delta)+\overline\Delta$; the summed cross term vanishes because $\sum_v(\Delta_v-\overline\Delta)=0$. P permits cancellation between pathway drifts; V also penalizes their dispersion. Equality holds exactly when all three drift tensors coincide. At the same $\lambda$, V is therefore pointwise at least as strong as P; this is not an equal-strength comparison or a clean isolation of independent-view information. A V advantage alone would not identify which of stronger preservation and reduced cancellation caused it. The identity imposes no ordering on downstream AP, false alarms or event coverage.

## State-safe differentiable preservation pass

The replay training pass retains native training behavior. The additional student preservation pass uses eval behavior while retaining autograd through its outputs; the teacher uses eval behavior and `no_grad`. Around these auxiliary forwards, execution snapshots and restores mutable buffers, the native running `step`, every module's individual train/eval flag, and CPU/CUDA RNG states. Restoring only the root mode would destroy heterogeneous child modes. Eval alone is also insufficient because native forward code can update non-parameter state. Restoration must preserve the student's autograd graph; a successful backward and frozen-state check are execution requirements, not detection evidence.

The auxiliary forwards must not advance the replay RNG trajectory or permanently update prototypes/curriculum state. Native R and P/V share initialization, replay draws and update schedule; teacher and auxiliary student forward/backward costs are reported separately. Matching optimizer steps does not imply matching total computation. The teacher and auxiliary terms are absent at inference: the native score, native readout permission, CAL locations and threshold rule remain unchanged, with each model calibrated in its own native score units.

## Prior work and falsifiable scope

Replay and teacher-response preservation are mature ideas. [Buzzega et al., NeurIPS 2020, §3/Eq5](https://proceedings.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) uses stored historical logits for Dark Experience Replay. This experiment instead fixes one domain-specific native reconstruction teacher and compares two aggregations of source-output drift. It does not reproduce DER, its reservoir, classification objective or reported benchmarks. The native interacting reconstructions belong to LEFT, not to a new backbone.

The hypothesis is that preserving source reconstruction functions can improve the replay adaptation trade-off without changing inference. Its evidence must come from V−R, V−P and P−R under fixed training and evaluation. Ranking, coverage and false-alarm effects are reported jointly; an aggregate-only improvement belongs to P, while a V advantage must be interpreted with its stronger preservation constraint. All four runs and bounded key-result review are complete. V has not established incremental benefit over R or P. P retains a SWaT ranking positive and PSM event-coverage trade-off, with PSM ranking loss and higher false positives. See the completed tables and costs in [the result report](../../docs/experiments/left_source_preservation_20261118/RESULTS.md). No automatic variant follows. Contamination identification and general robustness require their own evidence.

# Historical stopped design: alternating-domain joint normality density

**Historical implementation snapshot:** `d77252e17509d263d6d87a64c98b79516a935fd3`, `src/hybrid_tsad/normality_density.py`. The following mathematical account is retained for traceability of a stopped candidate, not the current LEFT-plus-replay definition or an established efficacy claim.

## Input and invertible computation

The observed window at endpoint $t$ is $X_t=x_{t-H+1:t}\in\mathbb R^{H\times C}$. FIT-only channelwise standardization preserves level relative to FIT. There is no windowwise centering or QUERY-fitted scale. The input layer is a learned invertible channelwise location–gain transform,

$$v_0=(\widetilde X_t-o)\odot e^g,\quad g=2\tanh(\gamma/2),\quad \log|\det J_0|=H\sum_c g_c.$$

Each of four coupling layers has a fixed checkerboard mask $m_{h,c}=(h+c+k)\bmod 2$, with parity $k$ alternating across layers. A conditioner receives $m\odot v$ and the mask; it does not receive transformed active coordinates. Its width-64 temporal network uses pointwise channel mixing and four residual kernel-three convolutions with dilations $(1,4,16,64)$. Let its outputs be raw log-scale $a$ and translation $b$. Then

$$\bar a=(1-m)\odot2\tanh(a/2),\quad \bar b=(1-m)\odot b,$$

$$u=m\odot v+(1-m)\odot(v\odot e^{\bar a}+\bar b),\qquad \log|\det J_k|=\sum_{h,c}\bar a_{h,c}.$$

The inverse is explicit. The conditioner depends only on the unchanged masked partition, which makes the determinant expression exact even though it uses temporal convolutions and channel mixing. Zero initialization of each conditioner's last projection starts its coupling at identity; this does not imply that the full time–wavelet model has the same initial density as a time-only identity transform.

The full forward sequence is affine input transform, coupling 0, Haar analysis, coupling 1, Haar synthesis, coupling 2, Haar analysis, coupling 3. Two-level orthogonal Haar transforms act on time, leave channel count unchanged, and preserve both dimension and total energy. Adjacent pairs $(a,b)$ map to $((a+b)/\sqrt2,(a-b)/\sqrt2)$ recursively on the low-frequency block. Because $H$ is divisible by four, no padding is needed and $|\det W|=1$. No hidden projection or discarded scale requires an approximate Jacobian.

The historical parameter counts are 200,990 (TEP, three channels), 223,738 (PSM, 25 channels), and 229,942 (SWaT, 31 channels). Full, time-only, and local variants have equal counts within a dataset. Time-only omits the fixed Haar changes of coordinates; local retains them but uses conditioner dilations $(1,1,1,1)$. The latter has local dependence per conditioner; it should not be claimed that composing coupling layers produces a perfectly independent nine-sample model.

## Learning and detection

With $D=HC$, $z=f_\theta(\widetilde X_t)$ and independent standard Student-t($\nu=5$) latent coordinates, the implemented coordinate prior loss is

$$r(z_i)=\log\Gamma(5/2)-\log\Gamma(3)+\tfrac12\log(5\pi)+3\log(1+z_i^2/5).$$

The exact standardized-window negative log density and endpoint score are

$$\ell_\theta(\widetilde X_t)=\sum_i r(z_i)-\log|\det J_{f_\theta}|,\qquad s_t=\ell_\theta(\widetilde X_t)/D.$$

FIT minimizes the mean of this same normalized quantity, and SELECT chooses by normal-reference NLL. The score is assigned to the window endpoint when $x_t$ arrives. No future sample, target-label selection, prediction residual, nearest-reference lookup, moving average, or score mixture is used. A channelwise intermediate returned by `score()` must be averaged across **all channels** to recover this joint criterion. After Haar and cross-channel mixing, intermediate allocations are not per-variable marginal probabilities. Scoring only XMEAS9 would no longer be the stated joint density; the TEP joint three-channel detector is therefore disclosed separately from target-XMEAS9 Ridge residuals.

The temporal conditioner uses left padding, but this is a computational property rather than proof of directed physical causal structure. Haar coordinates combine already observed timestamps within the window. The complete observed-window likelihood is endpoint causal because its latest raw observation is $t$; it is not a prediction of $x_t$ from $\mathcal F_{t-1}$. The compatibility input of length $H+1$ is sliced to its last $H$ positions, which requires raw endpoint alignment to be verified at the caller.

## Established components, new hypothesis, and limitations

Affine-coupling density estimation is mature (Real NVP); wavelets inside flows and explicit scale modeling are mature (Wavelet Flow); density-based multivariate anomaly detection is mature (GANF and MTGFlow). This candidate does not claim those ideas, heavy-tailed priors, channel mixing, or dilation as inventions. The empirical hypothesis is narrower: **alternating temporal and multiresolution transformations with one training-and-detection likelihood may preserve useful sustained and asynchronous anomaly evidence better than time-only/local alternatives and mature reference methods.** A parameter-matched ablation and real detection result are required, not merely improved likelihood or active gradients.

The design differs from stopped fixed-reference/nearest-memory models because it learns an invertible distribution; from blind-point models because no unobserved replacement target defines its readout; and from shared predictive reconstruction because the same likelihood is both the training objective and deployed score. These differences establish a new hypothesis, not an expected positive result. High likelihood is not semantic normality, known operating changes can cause false alarms, a whole-window score can dilute weak local changes or delay discovery, and likelihood gains may reflect scale rather than anomaly separation. Normal-state controls, coverage, delay, false alarms and strong simple baselines remain necessary.
