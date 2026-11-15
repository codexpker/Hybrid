# Implemented method definition: alternating-domain joint normality density

**Implementation snapshot:** `d77252e17509d263d6d87a64c98b79516a935fd3`, `src/hybrid_tsad/normality_density.py`. This is a mathematical account of implemented behavior, not an experiment plan or an established efficacy claim. It is a new complete Hybrid candidate; it does not rename the native LEFT model.

## Input and invertible computation

The observed window at endpoint $t$ is $X_t=x_{t-H+1:t}\in\mathbb R^{H\times C}$. FIT-only channelwise standardization preserves level relative to FIT. There is no windowwise centering or QUERY-fitted scale. The input layer is a learned invertible channelwise location–gain transform,

$$v_0=(\widetilde X_t-o)\odot e^g,\quad g=2\tanh(\gamma/2),\quad \log|\det J_0|=H\sum_c g_c.$$

Each of four coupling layers has a fixed checkerboard mask $m_{h,c}=(h+c+k)\bmod 2$, with parity $k$ alternating across layers. A conditioner receives $m\odot v$ and the mask; it does not receive transformed active coordinates. Its width-64 temporal network uses pointwise channel mixing and four residual kernel-three convolutions with dilations $(1,4,16,64)$. Let its outputs be raw log-scale $a$ and translation $b$. Then

$$\bar a=(1-m)\odot2\tanh(a/2),\quad \bar b=(1-m)\odot b,$$

$$u=m\odot v+(1-m)\odot(v\odot e^{\bar a}+\bar b),\qquad \log|\det J_k|=\sum_{h,c}\bar a_{h,c}.$$

The inverse is explicit. The conditioner depends only on the unchanged masked partition, which makes the determinant expression exact even though it uses temporal convolutions and channel mixing. Zero initialization of each conditioner's last projection starts its coupling at identity; this does not imply that the full time–wavelet model has the same initial density as a time-only identity transform.

The full forward sequence is affine input transform, coupling 0, Haar analysis, coupling 1, Haar synthesis, coupling 2, Haar analysis, coupling 3. Two-level orthogonal Haar transforms act on time, leave channel count unchanged, and preserve both dimension and total energy. Adjacent pairs $(a,b)$ map to $((a+b)/\sqrt2,(a-b)/\sqrt2)$ recursively on the low-frequency block. Because $H$ is divisible by four, no padding is needed and $|\det W|=1$. No hidden projection or discarded scale requires an approximate Jacobian.

The current parameter counts are 200,990 (TEP, three channels), 223,738 (PSM, 25 channels), and 229,942 (SWaT, 31 channels). Full, time-only, and local variants have equal counts within a dataset. Time-only omits the fixed Haar changes of coordinates; local retains them but uses conditioner dilations $(1,1,1,1)$. The latter has local dependence per conditioner; it should not be claimed that composing coupling layers produces a perfectly independent nine-sample model.

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
