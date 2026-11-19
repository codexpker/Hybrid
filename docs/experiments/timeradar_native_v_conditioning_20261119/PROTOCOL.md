# TimeRadar原生V与严格历史条件动态：有限验证协议

**本轮裁决：GO，值得进入一次有限正常源训练与固定评价；尚未获主研究决策执行确认，因此未启动正式训练。** 本轮仅完成源码/角色核查及允许的六窗正常工程检查。未构建B/C/D候选、未补旧Energy生成样本、未读QUERY标签、未改旧权重/分数/阈值。只用公开SWaT31、PSM25；不引入第一篇数据或标签。SMD、R、污染矩阵、F、Self/Cross-lag均不在本次实施范围。

## 1．原生V协议核查：公式保留，运行权限单列

固定官方源码提交[4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2](https://github.com/mala-lab/TimeRadar/tree/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2)。`models/TimeRadar.py` SHA256=`881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317`，固定公开预训练权重SHA256=`528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c`，1,636,043参数，全部冻结。本次重新取得四个固定提交源码文件并核Git blob；外部源码/权重不随公开工程再分发。

**FTFRecon真实前向：**SOURCE_FIT固定标准化后的100×C输入→按B*C组织→`net(input.mean(dim=0))`产生窗级阶次→FRFT→实/虚patch投影→五组mask及其互补，共十副本→FractionalEncoder/复数decoder→iFRFT→abs→`[10,1,100,C]`重建。参数冻结不等于阶次固定：不同输入窗仍可输出不同阶次。[固定官方模型](https://github.com/mala-lab/TimeRadar/blob/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2/models/TimeRadar.py)

**CDL真实路径与限制：**官方训练入口根据patch标签计算normal/abnormal affinity→`_global_mean(x.detach())`→no_grad EMA更新`tau_n/tau_a`→margin加入重构loss。这个固定版本的EMA margin没有沿该路径回传到表示；该事实已在2026-11-02旧源码阅读记录披露，本次复核一致，不修补。推理`Model.forward/cal_anomaly_score`不执行CDL loss或更新tau；保留权重中的三个tau bookkeeping buffers并验证不变。不能仅凭权重含tau就证明其训练时获得了可微CDL收益，也不由当前代码反推该公开权重历史训练的全部细节。[固定官方训练入口](https://github.com/mala-lab/TimeRadar/blob/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2/exp/exp_anomaly_detection_timeradar.py)

令十副本重建为r[m,j,c]，j=0…99，先计算：

$$\bar r_{j,c}=\frac1{10}\sum_m r_{m,j,c},\qquad
V_j=\frac1C\sum_c\frac1{9}\sum_m(r_{m,j,c}-\bar r_{j,c})^2.$$

本任务的**原生V_t就是V_99**。实际调用官方`cal_anomaly_score(batch_x,out,anomaly_criterion=None,L=1)[0,99]`，不取整窗均值，不广播末点，也不替换为Energy或预测残差。`torch.var(dim=0)`默认correction=1，必须保留分母9；互补副本并非独立样本，不能仅凭该分母声称它是独立抽样不确定性的无偏估计。

作为区别，逐点重构误差可写为`E_j=mean_c((rbar[j,c]-x[j,c])²)`，它与V不同，**不进入新评分**。官方FTFRecon训练的`criterion(outputs,batch_x)`将原输入广播到十副本，对每副本重构误差整体取平均；逐点恒等式是`mean_m,c((r_m-x)²)=E_j+0.9*V_j`，不是仅拟合平均重构。官方评分API有criterion时计算`L*V+(1-L)*recon_term`；固定官方test传入默认归约的`nn.MSELoss()`，该调用路径的recon_term是整batch标量再广播，不能把一般L混合项都描述为逐点E。L=1时该项为零；本任务直接criterion=None，避免无关归约语义。[SWaT AUC脚本](https://github.com/mala-lab/TimeRadar/blob/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2/scripts/anomaly_detection/SWAT.sh)、[PSM AUC脚本](https://github.com/mala-lab/TimeRadar/blob/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2/scripts/anomaly_detection/PSM.sh)均用norm=0、L=1；本协议固定这一预设，不混用PSM affiliation脚本的norm=1。

**批次与掩码规则：**官方阶次的B*C均值使原生B>1受到其他窗口影响。必须逐窗B=1；外部可排队/并行，但不能把独立窗口合成一次原生forward。每进程串行调用，跨卡用独立进程，不能在共享全局RNG的多个线程中并发fork/reseed。保留同一窗口内部跨变量共享阶次。每窗使用`mask_mode=c,copies=10`，在隔离CPU/CUDA RNG上下文内重置seed84551并恢复状态；native.eval冻结dropout，不能替代固定mask。掩码按变量/patch组织，不因当前点、角色、方法或邻窗改变。固定框架、GPU后端、源码/PT/通道顺序及mask生成版本；CPU同seed不能冒称与CUDA同mask。A/B/C共享**同一份V缓存**。

因此这是**官方公式/权重的B1因果末点协议适配**，不是作者batch128/64、stride100全窗离线基准的原样复现。历史TimeRadar离线分数不能替代新A，也不覆盖历史主表。

## 2．时间、目标及三个方法

使用半开记法：`h_t=x[t-512:t]`，即512个严格过去点；`X_t=x[t-99:t+1]`，即当前100点窗。二者交集恰为`[t-99,t)`的99点。重叠合法：预测器看不到x_t，V_t是x_t到达后才计算的监督目标；推理时也先预测分位数，再用实际到达的V_t评分。每点`window_start=t-99`、`window_end_exclusive=t+1`、`history_start=t-512`、`history_end_exclusive=t`、`available_index=t`。只取100窗末点；不将早于末点的位置标成当时可用。

使用原任务SOURCE_FIT-only均值/总体标准差，零尺度处理沿用原规则，同一scaler用于h与TimeRadar输入，只应用一次；不使用旧Energy的响应RMS。native内部norm=0。下面所有训练只用正常SOURCE_FIT，验证只用正常SOURCE_SELECT；冻结TimeRadar只提供自动生成的V目标，不接收梯度。这里no_grad不会截断B/C：梯度由pinball预测值直接回到历史预测器。

定义固定分位集合`T={.05,.50,.95}`。仅用训练端点的V计算`s_V=max(Q95(V_FIT),1e-12)`，令`z=V/s_V`，B/C学习三个z的条件分位数。该源常数只作数值缩放，不用CAL/QUERY拟合。

$$\mathcal L_Q(\theta)=\frac1{3N}\sum_{t\in FIT}\sum_{\tau\in T}
\rho_\tau\{z_t-\hat q_\tau(h_t)\},\quad
\rho_\tau(u)=u\{\tau-\mathbf1[u<0]\}.$$

预测头三个实数输出经**稳定升序排序**，允许相等，不强制严格正间距，不添加crossing损失。最后线性层权重初始化为零，bias为同一组FIT全局z分位数，允许零/并列；这解决softplus严格正间隔不能精确表示step0常数的问题。有限样本预测可以偏离非负支持域，不按QUERY另行截断。实现时先完成官方lazy层维度设置，再创建优化器。零末层使首步encoder梯度为零是预期；应在非零末层的合成扰动下验证编码路径可导，不因此更改训练配方。

| 方法 | 学习与结构 | 唯一正式分数 |
| --- | --- | --- |
| A：TimeRadar原生V | 官方PT完全冻结；无新增学习器 | `S_A=V_t` |
| B：V＋简单历史模型 | flatten同一512×C历史，MLP hidden32→32、GELU、线性3输出；原生V分位pinball | `S_B=(z-q95_B(h))/max(q95_B(h)-q05_B(h),1e-3)` |
| C：V＋TimesFM轻量因果模块 | 官方ResidualBlock patch16、宽128、2层、4heads、因果temporal与variate attention；最后patch变量token按固定通道顺序flatten→线性3输出；同一pinball | 与B完全相同，只替换分位预测器 |

C为拟研究的Hybrid臂，A/B/D为对照，当前尚无该候选的实现或训练结果。C只复用已核[TimesFM3官方模块](https://github.com/google-research/timesfm/tree/e51928e27119cb17bebc005be2696b75e0a9e688/src/timesfm3/torch)，**不加载官方预训练权重或官方分位头**，不恢复旧V1或条件生成器。MLP设计参数约SWaT.509M/PSM.411M，C约.587M/.585M；二者不是等参数、等计算量消融。固定顺序读出保留变量身份：当前官方模块配置没有变量位置编码，而固定mask按变量索引分配，不能用完全置换不变的均值池化悄悄丢掉这个区别。所有角色通道顺序固定，不作随机通道置换。变量attention不等于显式有向Cross-lag，512历史可用也不等于长历史贡献已经成立。

分数只看V的条件上尾；保留负值，不截零/取绝对值/做sigmoid，不把A分数另行相加。物理V单位下同式为`(V-s_V*q95)/max(s_V*(q95-q05),1e-3*s_V)`，禁止混用单位。固定1e-3分母floor只避免区间趋零，不根据QUERY调整。原生V仍保留原float32计算结果；分位模型FP32，最终减法/除法将已保存V和分位数提升float64计算，不从概率反求分数。step0是V的严格递增仿射变换，理论排序及各自CAL操作应与A一致；实施时要另核浮点等价。三个分位数不等于已学出完整概率密度或已校准条件分布。

**必要原始变量残差参照D，不是第四个新候选：**使用与C相同配置及相同初始化的历史encoder，末patch同样按固定变量顺序flatten，线性输出C个当前均值（约SWaT.698M/PSM.655M参数）；正常SOURCE_FIT上最小化`mean_c((yhat_c-x_t,c)²)`，正式分数为同一原值标准化空间的通道平均平方预测残差。无Energy、TimeRadar响应距离或评分融合，D读出不需要TimeRadar。D按正常SELECT MSE选权，其余数据/端点/预算与B/C一致。由于目标、损失、头、教师资产依赖、推理成本和SELECT依据不同，它是强任务参照，不能称同权纯读出消融。

## 3．数据角色及固定匹配规则

全部索引半开。SWaT为现有31通道任务的总轴，PSM源为发布训练段、目标为测试段相对轴；不引入私有材料，不重选事件。两域均已历史曝光，属于开发/机制验证，不称盲确认。这是同域正常源训练设置，不是新完成的异构多源预训练或官方TimesFM零样本能力评价。

缓存身份使用完整`domain/entity/原始record(or run)/time`及scaler/PT/mask版本，角色另作权限校验。PSM训练段与测试段都从0编号，不能仅凭`PSM+t`合并或复用；增加role也不能掩盖同一原始记录的跨角色重叠。六个正常fixture只能复用到相同原始记录的SOURCE_FIT点。

| 角色 | SWaT31 | PSM25 | 用途 |
| --- | --- | --- | --- |
| SOURCE_FIT | [0,396000) | 训练段[0,105984) | scaler、V源尺度/step0分位数、B/C/D训练 |
| SOURCE_SELECT | [396000,495000) | 训练段[105984,132481) | 正常选权；不回流FIT或scaler |
| U-FIT | [495000,585000) | 测试段[0,17568) | 保留但本轮不使用，不启动R |
| CAL角色 | [585544,607479) | 测试段[18112,21960) | 只用预定512点各自定阈值 |
| 512 CAL点覆盖 | 首586056、末607319 | 首18624、末21887 | 逐项复用原TASK cal_points，不重新等距采样 |
| QUERY共同轴 | [608535,944856)，336321点 | 测试段[23016,87744)，64728点 | 固定评价，不用于任何选择 |

SOURCE_FIT及SELECT都从各角色起点+512开始，每16点一个目标。FIT分别24718/6592点，SELECT6156/1625点；角色内历史和100窗均完整。源标签均由固定V函数生成，不是异常标签。训练种子2025，B/C/D同一端点及逐epoch置换；无放回完整遍历，batch128，末批保留。拟固定Adam1e-4、默认betas/eps、wd0、FP32，无新增mask/dropout/辅助项；最多20epochs或4000updates先到者，正常SELECT每epoch且含step0，按全体SELECT点的平均任务损失选最早最小值，不按QUERY选权或加训。对应SWaT每法最多3880更新，PSM1040更新；每域三法相同曝光，跨域不强求同更新数。

实施顺序仅在主决策确认后执行：先生成正常FIT/SELECT的原生V与源尺度→完成B/C/D及正常选权并封存→补CAL/QUERY的V与预测→最后开放标签统一评价。正常SELECT改善不替代检测增量，也不用极短试训是否赢正常MSE否决科学问题。A不需要新训练；新B/C/D不能拿旧Energy、旧V1或旧重放权重替代。

统一报告同轴AP、ROC-AUC、固定实现VUS、严格点与C3误报/召回、any-hit和一对一事件、延迟、长事件宏/合并覆盖与最长漏检。各方法使用自身512 CAL分数的q99/higher/严格>，不复制A阈值；PSM全攻击CAL原样保留，不能宣称正常1%误报控制。无额外参数/公式选择。

现有LEFT+R、MLP、CANDI和TimeRadar离线结果只按实际统计、训练/适配、上下文和available_index分列；共同点不等于同信息权限。A/B/C的available_index=t是信息可用时刻，不含实际计算/批次等待延迟；这次仍是有因果输入足迹的公开数据回顾评价，不是现场在线试验。

为避免再次无法归因：新缓存至少保存逐点V、十副本**末点**重构[10,C]、阶次/掩码版本、原值/scaler绑定及时间足迹；B/C保存三分位数、分母floor激活标志、未截断分数；D保存原值预测与残差。不必存每点完整[10,100,C]。十副本末点在全部441164个预定点约529MB原始float32，加其它数据仍可控制在约1GB以内，留在研究工作区，不推原数据或权重到GitHub。

## 4．已完成的最小工程检查与成本

仅取两域SOURCE_FIT的t=512/1024/2048，共六个真实正常窗；其余调用是这六窗的固定重复、顺序/邻窗核验及计时，合计60次原生B1 forward。没有条件模型、优化器、反向更新、Energy、adapter、QUERY或标签。

| 检查项 | 实际结果 |
| --- | --- |
| 原生V vs NumPy ddof1→均通道→末点99 | 6/6一致；末点最大绝对差8.7211e-8 |
| 主复核完整100位置公式 | 复用正常缓存，最大绝对差2.55693e-7，rtol1e-5/atol1e-7通过；仅末点用于任务 |
| 固定窗重复、B1循环更换邻窗、顺序置换 | 全副本与V均bitexact |
| CPU/CUDA RNG、所有参数/缓冲、eval/grad | RNG与张量哈希不变、全eval/冻结/无grad |
| SWaT单点计时 | 预热后16次同步中位11.3913ms，峰102.602MiB |
| PSM单点计时 | 同规则中位10.4372ms，峰86.044MiB |
| 本轮实际作业 | 一张A40、OS0、完整启动至退出4.990261771秒=.001386184GPUh，清理无残留 |

原件`results/timeradar_native_v_protocol_20261119_v1/NATIVE_V_NORMAL_GATE.json`，SHA256=`00d27d9d9d5ce1c6043d3534256a361afcc74a1b1444685c8f40e4660aeafff3`；正常数组、完整计时及真实设备/PID/环境/退出记录保存在原目录。主复核只读六个已存正常数组，未重复GPU前向。本检查是工程性质，不是异常检测效果证明，六个点不能估计源分位数或替代完整正常验证。

**未来有限矩阵成本估算，不是本轮已执行：**A/B/C/D共享一份冻结V银行；SWaT需367707个预定点、PSM73457个，共441164点。六个兼容正常检查点可以复用。按实测纯原生forward中位耗时，理想合计约1.3765 A40GPUh；含读入、RNG隔离、输出/缓存与启动，估计1.5–2.2GPUh。不是给每个方法各跑一份V，也不再做K次生成。

B/C/D共六次正常源训练，源FIT最大曝光每法SWaT494360/PSM131840。条件模型尚未实现和计时，不能报精确训练成本。参考既有相近TimesFM编码器的B1训练速度，C/D若不利用批量加速会接近10GPUh；批量128预计更低，当前只能给宽估计约1–10GPUh（含两份廉价MLP）。已有相近历史模型B1读出约5.82/6.97ms每点，C/D单点循环读出保守约1.34GPUh，批处理预计更低。完整有限矩阵暂估**约3–14GPUh**；这不是硬额度或保证，用实施后的无更新测速记录修正耗时估计，不能据QUERY调整训练上限。最多四独立作业、训练20GiB/评价30GiB、具名单作业超时/清理及实际耗时账本继续保留。无需恢复旧逐任务预算审批，但**本轮只交协议，不自动启动这六次训练**。

## 5．明确GO意见与不能先承诺的结果

**GO进入一次上述有限源训练验证。**现在原生V公式、批独立、掩码、末点/历史权限、区间稳定性定义和共同轴可以闭合；一个简单MLP条件对照足以使TimesFM模块接受真实竞争，不必先恢复旧配方或扩大数据搜索。B/C/D尚未实现，主决策确认后的实现仍需检查分位形状、单位、排序梯度、encoder梯度和step0数值等价；不能把本次原生V检查说成条件模型已通过测试。此意见不是已成立的Hybrid贡献或正式开跑许可，等待主研究决策。

唯一假设是：在正常历史支持域内，条件分位模型能解释正常V变化，而异常仍表现为超出条件上尾的分歧。如果给定历史后的正常/异常V分布相同，这一判据无法区分，复杂历史网络也不能凭空恢复V已丢失的当前异常信息。只看条件上尾，不能检测所有低V异常；异常若已进入历史，条件模型也可能解释掉持续异常，未知正常工况可能导致区间过窄/误报。十个互补mask的V不是已校准不确定性，三个预测分位数也不自动有覆盖保证。必须据实际异常排序、覆盖和误报判断，不能用正常pinball下降替代。

C相对A/B的检测增量才支持结构化历史条件化的有限价值；C与D用于检查是否只是普通原值预测已有证据，不是纯损失消融。若结果仅等于step0常数、只改善源预测、不优于简单条件模型，或靠严重误报换覆盖，不扩展为TimesFM/时频协同创新。首轮无长短历史或Cross-lag独立消融，不声称已证明长历史长度或有向滞后贡献。只有结果出来并由主研究决策判断后，才讨论R，不能继承LEFT+R的污染保护结论。
