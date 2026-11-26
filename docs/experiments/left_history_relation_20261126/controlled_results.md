# Controlled weak amplitude and relative delay

The positive class is an edited record. Scores are the mean over the two current 32-point blocks at positions 96–159. For every lag up to 64 the observations used in these blocks fall inside the full-strength edit. A separate transition comparison uses 0–31 and 160–191. Full-window records include all six blocks. Native offline LEFT may use the entire current window.

For each target width, 64 independently seeded unit-variance AR(1) records have AR coefficient 0.85. In the first 2 floor(C/4) columns, each receiver equals 0.8 times its driver at lag 4 plus 0.6 times independent AR noise. Remaining channels retain independent AR trajectories. In a separate transfer check, 16 evenly spaced nonoverlapping 448-point windows come from the released target normal SELECT. They share one underlying time series per domain. The archived target normal-prefix scaler includes SELECT. No target test scores or annotations entered this construction. The pairing and channel choices are identical on real and simulated inputs. A known lag of four applies only to the simulation.

The weak shift adds ±0.25 standardized units to each receiver under a smooth 32-point rise and fall. Its same-support bipolar control uses a cosine profile and matches the total edit energy per receiver. The delay edit gradually replaces each receiver by its own value 12 steps earlier. In the steady synthetic core this moves the known relative delay from 4 to 16 and preserves each channel's stationary marginal distribution in expectation. A synchronous sham shifts both partners, with its own energy. A smooth same-support amplitude control matches the receiver-specific total delay-edit energy. All controls retain the original history and both outer endpoint samples. The four-condition comparison tests block-level discrimination in these fixtures. It does not establish physical fault timing or point-level localization.

## Core record discrimination

Cells report ROC-AUC on equal counts of edited and reference records. Pairing enters the separate mean and median differences and sign counts; AP here has balanced artificial prevalence.

### coupled_AR

|                                                   |   lag vs clean |   lag vs lag_amplitude |   lag vs synchronous |   weak vs clean |   weak vs weak_control |
|:--------------------------------------------------|---------------:|-----------------------:|---------------------:|----------------:|-----------------------:|
| ('PSM', 'Continuation + historical correlation')  |         0.5510 |                 0.2756 |               0.5220 |          0.5200 |                 0.5000 |
| ('PSM', 'Frozen source LEFT')                     |         0.5078 |                 0.0415 |               0.4866 |          0.5376 |                 0.5076 |
| ('PSM', 'Historical correlation reference')       |         0.5554 |                 0.5554 |               0.5398 |          0.5004 |                 0.4958 |
| ('PSM', 'Joint LEFT + learned relation')          |         0.5286 |                 0.2957 |               0.5017 |          0.5139 |                 0.5017 |
| ('PSM', 'Learned relation component')             |         0.5293 |                 0.5293 |               0.5059 |          0.5007 |                 0.4980 |
| ('PSM', 'Matched continuation LEFT')              |         0.5076 |                 0.0464 |               0.4844 |          0.5381 |                 0.5056 |
| ('SWAT', 'Continuation + historical correlation') |         0.5256 |                 0.4944 |               0.5173 |          0.5032 |                 0.4949 |
| ('SWAT', 'Frozen source LEFT')                    |         0.5022 |                 0.4453 |               0.5188 |          0.5049 |                 0.5015 |
| ('SWAT', 'Historical correlation reference')      |         0.5330 |                 0.5330 |               0.5095 |          0.4998 |                 0.4944 |
| ('SWAT', 'Joint LEFT + learned relation')         |         0.5156 |                 0.4802 |               0.4993 |          0.5044 |                 0.4944 |
| ('SWAT', 'Learned relation component')            |         0.5186 |                 0.5186 |               0.4871 |          0.5006 |                 0.4917 |
| ('SWAT', 'Matched continuation LEFT')             |         0.5017 |                 0.4509 |               0.5183 |          0.5051 |                 0.5012 |

### normal_SELECT

|                                                   |   lag vs clean |   lag vs lag_amplitude |   lag vs synchronous |   weak vs clean |   weak vs weak_control |
|:--------------------------------------------------|---------------:|-----------------------:|---------------------:|----------------:|-----------------------:|
| ('PSM', 'Continuation + historical correlation')  |         0.5977 |                 0.5820 |               0.4648 |          0.5312 |                 0.2969 |
| ('PSM', 'Frozen source LEFT')                     |         0.4961 |                 0.4766 |               0.5039 |          0.5234 |                 0.5039 |
| ('PSM', 'Historical correlation reference')       |         0.5859 |                 0.5859 |               0.4844 |          0.5039 |                 0.2852 |
| ('PSM', 'Joint LEFT + learned relation')          |         0.4961 |                 0.4922 |               0.5000 |          0.5195 |                 0.2617 |
| ('PSM', 'Learned relation component')             |         0.4922 |                 0.4922 |               0.5039 |          0.5000 |                 0.1992 |
| ('PSM', 'Matched continuation LEFT')              |         0.5156 |                 0.4688 |               0.5117 |          0.5273 |                 0.5117 |
| ('SWAT', 'Continuation + historical correlation') |         0.4766 |                 0.4766 |               0.4961 |          0.5312 |                 0.1367 |
| ('SWAT', 'Frozen source LEFT')                    |         0.5234 |                 0.5000 |               0.4961 |          0.7305 |                 0.0664 |
| ('SWAT', 'Historical correlation reference')      |         0.4766 |                 0.4766 |               0.4922 |          0.5039 |                 0.1406 |
| ('SWAT', 'Joint LEFT + learned relation')         |         0.5117 |                 0.5117 |               0.4922 |          0.5312 |                 0.0000 |
| ('SWAT', 'Learned relation component')            |         0.5117 |                 0.5117 |               0.4922 |          0.5000 |                 0.0000 |
| ('SWAT', 'Matched continuation LEFT')             |         0.5273 |                 0.5000 |               0.5039 |          0.7266 |                 0.0586 |

The lag edit produced less core separation from its equal-energy amplitude control for the learned relation component than for simple historical correlation on both synthetic widths. Adding native scores reduced synthetic discrimination further. On the real nominal blocks, joint core ROC-AUC was near 0.5 on both widths. Weak shifts did not consistently separate from the matched bipolar control. These findings limit claims of either lag specificity or a weak-amplitude advantage.

## Manipulation checks

|                                            |    energy |   correlation_lag4 |   correlation_lag16 |   receiver_core_abs_mean_shift |   receiver_core_abs_sd_shift |
|:-------------------------------------------|----------:|-------------------:|--------------------:|-------------------------------:|-----------------------------:|
| ('coupled_AR', 'PSM', 'clean')             |    0.0000 |             0.7881 |             -0.0217 |                         0.0000 |                       0.0000 |
| ('coupled_AR', 'PSM', 'lag')               | 1554.9288 |            -0.0307 |              0.7848 |                         0.1509 |                       0.0772 |
| ('coupled_AR', 'PSM', 'lag_amplitude')     | 1554.9288 |             0.7881 |             -0.0217 |                         1.2930 |                       0.0000 |
| ('coupled_AR', 'PSM', 'synchronous')       | 3092.0281 |             0.7848 |             -0.0133 |                         0.1509 |                       0.0772 |
| ('coupled_AR', 'PSM', 'weak')              |   57.0937 |             0.7881 |             -0.0217 |                         0.2500 |                       0.0000 |
| ('coupled_AR', 'PSM', 'weak_control')      |   57.0937 |             0.7838 |             -0.0226 |                         0.1935 |                       0.0330 |
| ('coupled_AR', 'SWAT', 'clean')            |    0.0000 |             0.7938 |             -0.0228 |                         0.0000 |                       0.0000 |
| ('coupled_AR', 'SWAT', 'lag')              | 1839.8251 |            -0.0260 |              0.7956 |                         0.1585 |                       0.0845 |
| ('coupled_AR', 'SWAT', 'lag_amplitude')    | 1839.8251 |             0.7938 |             -0.0228 |                         1.3023 |                       0.0000 |
| ('coupled_AR', 'SWAT', 'synchronous')      | 3709.6551 |             0.7956 |             -0.0254 |                         0.1585 |                       0.0845 |
| ('coupled_AR', 'SWAT', 'weak')             |   66.6094 |             0.7938 |             -0.0228 |                         0.2500 |                       0.0000 |
| ('coupled_AR', 'SWAT', 'weak_control')     |   66.6094 |             0.7879 |             -0.0235 |                         0.1935 |                       0.0331 |
| ('normal_SELECT', 'PSM', 'clean')          |    0.0000 |             0.1288 |             -0.0183 |                         0.0000 |                       0.0000 |
| ('normal_SELECT', 'PSM', 'lag')            |   65.3670 |             0.0629 |              0.1359 |                         0.0351 |                       0.0199 |
| ('normal_SELECT', 'PSM', 'lag_amplitude')  |   65.3670 |             0.1288 |             -0.0183 |                         0.2173 |                       0.0000 |
| ('normal_SELECT', 'PSM', 'synchronous')    |  130.6922 |             0.1359 |             -0.0022 |                         0.0351 |                       0.0199 |
| ('normal_SELECT', 'PSM', 'weak')           |   57.0938 |             0.1288 |             -0.0183 |                         0.2500 |                       0.0000 |
| ('normal_SELECT', 'PSM', 'weak_control')   |   57.0937 |             0.1170 |             -0.0023 |                         0.1935 |                       0.0543 |
| ('normal_SELECT', 'SWAT', 'clean')         |    0.0000 |             0.0257 |              0.0472 |                         0.0000 |                       0.0000 |
| ('normal_SELECT', 'SWAT', 'lag')           |   24.0841 |             0.0109 |              0.0804 |                         0.0095 |                       0.0047 |
| ('normal_SELECT', 'SWAT', 'lag_amplitude') |   24.0841 |             0.0257 |              0.0472 |                         0.0465 |                       0.0000 |
| ('normal_SELECT', 'SWAT', 'synchronous')   |   57.0799 |             0.0804 |              0.0274 |                         0.0095 |                       0.0047 |
| ('normal_SELECT', 'SWAT', 'weak')          |   66.6094 |             0.0257 |              0.0472 |                         0.2500 |                       0.0000 |
| ('normal_SELECT', 'SWAT', 'weak_control')  |   66.6094 |            -0.0737 |             -0.0563 |                         0.1935 |                       0.0976 |

Maximum absolute differences in total stored float32 edit energy were 5.9e-06 (weak) and 1.85e-05 (delay). Each channel pair was also checked against float32 roundoff bounds. Finite-window SD and mean shifts quantify residual marginal differences. Constant pairs were excluded from the input correlation summary.

An independent pair-count ROC calculation reproduced four joint lag-versus-amplitude core comparisons with maximum absolute discrepancy 0. Batched bank generation for all twelve score banks took 5.768 A40 seconds, including CPU transfers and extraction; weights were frozen.

[All core, transition and full-window comparisons](controlled_pairs.csv) · [Record-level means](controlled_records.csv.gz) · [Input diagnostics](controlled_inputs.csv) · [Input identities](controlled_identities.json) · [Cost and bank hashes](controlled_cost.csv).
