---
title: "Hybrid: Learning Multiscale Normal Evolution for Sustained Time-Series Anomaly Detection"
author: []
date: "Working draft, 15 November 2026"
lang: en
bibliography: references.bib
---

> Working status: the fixed experiment and bounded independent recomputation are complete. The full recipe receives a scope-limited NO-GO: TEP contains genuine detection gains, but the ablations do not establish the proposed evolution mechanism's independent sustained benefit; PSM and SWaT violate the normal false-alarm objective. The following sections document the tested hypothesis, not an established contribution. See [accepted results](../../docs/experiments/normal_evolution_20261115/RESULTS.md). No contamination robustness or independent confirmation is asserted, and earlier experimental records remain unchanged.

# Introduction

Industrial anomaly detection must distinguish unusual observations from unusual behavior. Several measured variables may change together during a normal transition, and a detector that treats all departures from one historical level as faults can generate persistent false alarms. Conversely, an abnormal state may become locally predictable after its onset. Detecting its first change is then insufficient: the monitor should retain evidence during the abnormal interval while remaining quiet on comparable normal variation. These requirements are difficult when operating modes are not authenticated, responses are delayed, and available training records are unlabeled or only believed to be predominantly normal.

Time–frequency methods offer complementary descriptions of levels, local fluctuations, and multiscale structure. TimeRadar develops domain-rotatable time–frequency modeling and contextual deviation learning [@timeradar2026]; LEFT learns interacting temporal, spectral, and multiscale representations and uses an analysis–synthesis reconstruction chain [@left2026]. These are mature external methods, not components whose reuse establishes a new contribution. More generally, lower reconstruction or prediction loss need not imply improved anomaly discrimination. A model can fit a sustained abnormal state, lose absolute-level information through local normalization, or produce scores whose normal distribution changes between calibration and deployment. These failure mechanisms require different remedies and should not be inferred from average precision alone.

Our question is whether observable normal evolution can provide a more useful detection criterion. We study a model that jointly learns an invertible blockwise time–frequency representation, a finite set of latent statistical regimes, and stable regime-dependent dynamics. The regime distribution for the current block is predicted using only earlier blocks. The current block is then evaluated against this predictive mixture, rather than becoming a retrieved normal reference or being judged only by its own reconstruction. A learned transition can accommodate a recurring normal change; an observation that disagrees with every plausible normal evolution can remain unusual after its onset.

This formulation has a necessary qualification. The learned regimes are statistical summaries, not verified operating commands, physical states, or causal explanations. Normal transitions absent from training may remain anomalous to the model. A fault that produces the same complete observable law as normal operation is not identifiable from these measurements alone. Stability also does not guarantee sustained detection: it constrains unrestricted copying of arbitrary levels, but the representation, regime coverage, and predictive uncertainty can still absorb a fault. The empirical task is to test these conditions, not to claim that they have been removed by model complexity.

The current Hybrid experiment tests one complete learning–criterion design. Likelihood learning links the time–frequency map to regime-dependent evolution; a fixed predictive-mixture transformation provides the anomaly evidence. The study compares the complete model with removal of time–frequency mixing and removal of temporal evolution, and with strong external and simple methods under disclosed information permissions. PSM, SWaT, and the fixed TEP task are development and mechanism settings because they have already influenced this project. Their results can justify or reject further development, but cannot serve as blind confirmation.

The intended contribution is a reproducible gain in sustained anomaly discrimination at an acceptable normal false-alarm cost, attributable to this complete design. Invertible transforms, regime switching, stable autoregression, and probability-integral transforms are established tools. Their combination is not by itself a publication-level claim. Only accepted comparisons can establish whether Hybrid contributes beyond these ingredients. Multisource training and mixed-unlabeled target adaptation remain the broader objective; neither robustness nor few-event supervision is used to rescue the basic detector in this experiment.

# Related Work

## Time–frequency representations and normality learning

LEFT, TimeRadar, and CATCH provide different ways to represent multivariate temporal and spectral structure [@left2026; @timeradar2026; @catch2025]. Their native training and scoring procedures must be distinguished from a causal endpoint adaptation. Cropping an offline score to a shared timestamp axis does not remove future observations used when constructing that score. We therefore retain separate information-permission tracks for external comparisons. Hybrid uses an orthogonal multiresolution transform inside an invertible block map, preserving both coarse and detail coordinates. This is related to established wavelet-flow construction [@waveletflow2020], rather than a claim to invent wavelet density modeling.

Normalizing flows provide tractable likelihood through invertible mappings and Jacobian terms [@realnvp2017]. GANF and MTGFlow already apply dependency-aware flow models to multivariate anomaly detection [@ganf2022; @mtgflow2023]. Thus neither a joint density nor a temporal conditioner is new. The present experiment instead examines whether a normal-regime evolution model and its predictive compliance criterion help preserve event evidence when a normal distribution contains multiple recurring states. The ablations must distinguish this effect from the expressive observation map alone.

## Switching dynamics and latent anomaly criteria

Balsells-Rodas et al. analyze switching dynamical systems and give identification results under explicit distributional and mapping assumptions [@balsells2024]. Their model separates continuous dynamics from discrete switches; the official implementation includes log-space forward inference and autoregressive Gaussian emissions. These are technical foundations for marginalizing state paths. Their identification results do not certify that inferred states in industrial data represent authorized modes, nor do they establish anomaly-detection accuracy for Hybrid.

Baumgartner et al. jointly train conditional flows and prescribed latent dynamics, then assess whitened latent compliance using a multivariate goodness-of-fit test [@baumgartner2026]. This UAI 2026 work already establishes the relevance of latent dynamics and distinguishes compliance from observation likelihood. Hybrid must not claim that general idea as new. Our fixed experiment uses a switching predictive mixture, stable block dynamics, and an exact mixture Rosenblatt transformation followed by a coordinate-tail score. It does not reproduce their single latent-dynamics model, MV-KS test, or hyperparameter search. The tail score is also narrower than a full goodness-of-fit test: concentration near central probability values may escape detection.

## Probability transforms, adaptation, and evaluation

The Rosenblatt transform converts a correctly specified continuous multivariate distribution into independent uniform coordinates through successive conditional cumulative probabilities [@rosenblatt1952]. Using it here provides a defined relationship between the fitted predictive mixture and the score. It does not yield automatic false-alarm certification for an estimated model on dependent industrial windows. We keep normal calibration separate and report actual downstream normal errors.

Mixed-unlabeled adaptation remains distinct from normal-reference learning. TSAD-C explicitly studies contamination with a complete conditional diffusion method [@tsadc2025]. In our setting, normal distribution change, anomalous replacement during training, and calibration change must be separated before claiming contamination-specific harm or robustness. This experiment uses the existing normal-reference roles without anomaly labels; it makes no mixed-unlabeled robustness claim. Stage1, Z, U, and F remain different training permissions rather than interchangeable labels for a single table.

# Problem Formulation and Fixed Learning Mechanism

## Observable task and information permissions

Let \(x_t\in\mathbb R^C\) be the observation received at time \(t\). A causal endpoint detector produces \(s_t\) from \(x_{\le t}\), with no observation after \(t\). Normal-reference FIT data estimate parameters and the fixed affine standardizer. SELECT chooses the checkpoint using the same normal likelihood objective. CAL supplies only the fixed empirical threshold rule. QUERY labels are withheld from all these operations and are used only to evaluate the frozen detector. Windows must remain inside legal records and roles; no state is carried across run or role boundaries.

The event annotations define contiguous evaluation intervals. They are not assumed to certify a physical fault at every point. Event discovery, one-to-one matching, any-hit, and sustained point coverage are therefore separate outcomes. Likewise, a long event's interior is an index-based evaluation region, not independently verified physical steady state.

## Invertible time–frequency blocks

For each legal trailing window ending at \(t\), use \(H=192\) for PSM/SWaT and \(H=100\) for TEP, divided into \(J=H/B\) contiguous blocks of \(B=4\) observations. Blocks are anchored at the scored endpoint. Let \(u_b\in\mathbb R^D\), \(D=BC\), be the flattened FIT-standardized block. A shared bijection produces

\[
z_b=g_\theta(u_b),\qquad d_b=\log\left|\det Dg_\theta(u_b)\right|.
\]

The fixed implementation uses four affine coupling layers, width 64, interleaved with an orthonormal two-level Haar change of coordinates. All coordinates are retained, including the coarse component; no current-window mean or variance is discarded. The Haar transform has unit absolute determinant. The coupling Jacobians remain in likelihood learning. This block map is learned afresh and does not reuse stopped whole-window density weights.

## Normal evolution and training

Let \(q_b\in\{1,\ldots,K\}\), with fixed \(K=4\), denote a latent statistical regime. Learn initial probabilities \(\pi\), row-stochastic transitions \(T\), regime means \(\mu_k\), positive diagonal noise \(\sigma_k\), and stable maps \(A_k\). The model is

\[
p(q_1=k)=\pi_k,\qquad p(q_b=k\mid q_{b-1}=i)=T_{ik},
\]
\[
p(z_1\mid q_1=k)=\mathcal N(z_1;\mu_k,\operatorname{diag}\sigma_k^2),
\]
\[
p(z_b\mid z_{b-1},q_b=k)=\mathcal N\!\left(z_b;\mu_k+A_k(z_{b-1}-\mu_k),\operatorname{diag}\sigma_k^2\right),\quad b>1.
\]

Each \(A_k\) has a diagonal plus rank-8 term and a fixed spectral-norm upper bound of 0.9. This is a modeling constraint, not proof that physical dynamics obey a contraction. It prevents an unconstrained identity predictor from automatically reproducing every constant latent displacement; it cannot prevent all alternative regimes or observation-map effects from explaining that displacement.

Regime paths are marginalized by a log-space forward recursion. With filtered probabilities \(r_{b-1,i}=p(q_{b-1}=i\mid z_{1:b-1})\), the current prior is \(w_{b,k}=\sum_i r_{b-1,i}T_{ik}\). Define

\[
c_b=\sum_k w_{b,k}\,p(z_b\mid z_{b-1},q_b=k),\qquad
r_{b,k}=\frac{w_{b,k}\,p(z_b\mid z_{b-1},q_b=k)}{c_b}.
\]

For the first block, use \(w_1=\pi\) and the initial emission. The per-window training objective is the exact normalized trajectory negative log-likelihood

\[
\mathcal L(\theta,\phi)=-\frac{1}{JD}\sum_{b=1}^J(\log c_b+d_b).
\]

All trainable parameters are updated only on FIT. SELECT uses this same objective. The current block is incorporated into \(r_b\) only after its predictive distribution has been formed. At QUERY, parameters are frozen and the finite-history filter is recomputed inside each legal window; there is no online reference update, pseudo-label update, or cross-role state persistence.

## Predictive-mixture compliance score

The final block has a diagonal-Gaussian mixture conditional on preceding blocks. Its coordinates are ordered by fixed flattened time/channel order. Let \(m_{k,j}\) be the corresponding predictive mean. For coordinate \(j\), update mixture probabilities using only the earlier coordinates of this same arrived block:

\[
\omega_{k,j}\propto w_{J,k}\prod_{i<j}\mathcal N(z_{J,i};m_{k,i},\sigma_{k,i}^2),
\qquad
v_j=\sum_k\omega_{k,j}\Phi\!\left(\frac{z_{J,j}-m_{k,j}}{\sigma_{k,j}}\right).
\]

This is the exact Rosenblatt transform of the fitted predictive mixture, not separate marginal CDFs with unchanged mixture weights. The fixed score is

\[
s_t=\frac{1}{D}\sum_{j=1}^D-\log\!\left(2\min\{\bar v_j,1-\bar v_j\}\right),\qquad
\bar v_j=\operatorname{clip}(v_j,10^{-7},1-10^{-7}).
\]

The prediction is made before the terminal block is observed, but the score is produced only when that entire block has arrived at \(t\). It is therefore a four-observation block score assigned to the endpoint, not a forecast scored at all four earlier times. The Jacobian belongs to likelihood training; it is not added to this latent compliance score. This deliberately differs from the stopped observation-NLL criterion. It also differs from reconstruction/residual fusion: the score is derived from one learned predictive distribution, not a weighted sum of external detector outputs.

With a correctly specified continuous conditional model, the unclipped \(v_j\) are independent uniform coordinates. That population statement does not certify a learned industrial model or overlapping windows. The score emphasizes tail departures; it does not test all violations of uniformity and can miss anomalous concentration near the median. The empirical threshold is the unchanged legal CAL 0.99 quantile with the fixed `higher` convention and strict exceedance. The existing three-consecutive-point rule remains unchanged.

## Scope of identifiability and attribution

The model can protect recurring normal changes only when their observation distributions and transitions have adequate FIT support and are estimated sufficiently well. No trusted external command is assumed. An unseen benign mode can cause false alarms; an anomaly lying inside the learned normal predictive distribution can be missed. If normal and anomalous processes have the same complete observable distribution, no detector based on these observations alone can separate them. A low false-alarm rate and a useful long-event score must therefore be demonstrated together, not asserted from stability or likelihood.

The fixed matched ablations remove Haar mixing (`time_only`) or remove evolution and history filtering (`no_evolution`, iid regime mixture, \(A=0\)). The latter has fewer active dynamics parameters, so it is not described as an equal-effective-capacity control. Shared active initialization, normal data roles, draws, seed 2025, training limits, checkpoint selection, and CAL permissions are retained. Neither a successful flow inversion nor nonzero gradients establish a scientific contribution. The complete public-data comparison must show an actual detection benefit beyond the ablations and mature baselines without unacceptable normal false alarms.

No U/F or contamination experiment is part of this round. A favorable outcome would justify freezing this method before independent confirmation and later adaptation research. An unfavorable outcome closes this specific complete normal-evolution hypothesis without launching another local variant.

# Experimental Setup: Evidence Boundaries

PSM, SWaT, and TEP reuse the existing fixed FIT/SELECT/CAL/QUERY roles, FIT-only standardizers, full common axes, and preserved event boundaries. The three settings are historically exposed development assets, not independent confirmation. All 20 TEP fault classes remain in evaluation; faults 4/11/14 are the predetermined mechanism subset, not a subset selected by new performance. Existing LEFT, TimeRadar, historical-mean, and task-appropriate Ridge banks are reused only when axes and input permissions are compatible. Their target-training, source-training, and offline-versus-causal permissions accompany the results. No external model is retrained to improve a comparison row.

The fixed evaluation reports AP, ROC-AUC, strict-point recall, any-hit and one-to-one event metrics, three-consecutive-point confirmation, normal false-alarm points and fragments, first-detection delay, long-event interior coverage, and longest missed segments. Missed-event delays are censored rather than assigned zero. For a label-free description of observable normal changes, define \(d_t=C^{-1}\sum_c|\tilde x_{t,c}-\tilde x_{t-4,c}|\), where the standardizer is fitted only on FIT. Its FIT 0.90 quantile fixes high-change versus other bins. QUERY normal labels are used solely to report false alarms within these frozen bins; they do not define a model, reference, threshold, or authenticated operating state. This analysis measures observable changes, not whether a controller authorized them.

Normal false-alarm cost is evaluated on the actual downstream normal points. A CAL quantile is not taken as proof that the same rate transfers. An improvement in abnormal coverage that accompanies excessive normal alarms does not establish a useful detector. Strong simple models remain necessary to distinguish a genuinely learned detection contribution from a generic level or scale effect.

The completed experiment uses seed 2025 and nine matched domain/variant trainings, with selected checkpoints determined by normal SELECT loss. Actual execution consumed 1.178158 GPU hours; bounded result review adds at most 0.000980 GPU hours including startup and cleanup. The full model's AP is 0.444863 on PSM, 0.798098 on SWaT, and 0.971480 on the fixed all-fault TEP axis. Corresponding strict normal false-alarm rates are 10.4891%, 99.9837%, and 1.6337%, against a nominal 1% CAL tail. TEP fault 11 long-interior coverage improves to 96.6625% from native LEFT's 69.95%, but the no-evolution ablation achieves 96.5625%. Thus this positive observation does not establish an independent evolution benefit. SWaT's any-hit count of 22 must not be confused with its four one-to-one matches. Full event, delay, coverage, normal-cost, external-permission and ablation tables are retained in the linked accepted results. The recipe is stopped without further threshold, score, checkpoint or architectural adjustment; these development results do not justify U/F expansion or a validated method-contribution claim.
