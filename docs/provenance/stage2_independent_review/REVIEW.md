# U_NORMAL_HEAD共享工程与L0资源壳独立审查

**PASS_FINITE_ENGINEERING / ACCEPTABLE_FOR_NEXT_STAGE。** 允许作为主智能体单独签署有界、丢弃权重GPU L0的工程依据；本报告不启动GPU，不批准正式1000步训练、真实校准或query。实际设备上的更新/重放/资源尚INCOMPLETE，收益NONE、创新UNSUPPORTED、科研HOLD。职责独立但同模型家族/共享工作区，非外部或双盲。

最终受审Stage2 SHA`9579090dc13ee368c58c2cb4b96745ff7667bf68105c2028a00719ed1b9dd6da`；stage2_identity SHA`8f403510601aed2ec964003864f5a78c786b31f41883aec1f8603e46b481afed`、stage2_data SHA`10d473dc159dff207288c695b98b5e1c59304c1e54844affb640c496735a999a`、CLI SHA`4e71b228e6fb0ff488967df9670930d156881b46ad8aa8f2a46662ae5efc85cb`保持原冻结版本。最终资源壳SHA`bd7ad22a2332953e74e1b66cce85ddad8d51b8ab92b40f24e37f2d5577136398`。原Stage2 `1f5688…`和原壳`dbef7808…`及其失败证据均保留，没有覆盖为PASS。

## 字节、来源与权限

原FREEZE/验证报告SHA349212a3…及patch35f3e3ec…逐SHA核对并保存raw-byte快照；canonical修订记录SHA2fbb3aa4…和patch476c02d0…再次锁新快照。独立AST比较确认仅calibration_tracks与calibrate两个函数变化；三core、Stage1训练、source_gate、原Z控制、LR/head step/L0/data/identity逻辑没有改变。没有用git archive的换行副本代替受审字节。

schema5显式stage/operation绑定全部固定输入与job/device，排除approval自指；正式entry gate另需主接受的双L0和独立review，CPU fixture不能冒充GPU证据。旧Z拒绝U child的stage/target身份，query/F/labels不在Stage2输入集合。数据接受及三plan原审计复用，没有重算260万正常值，也没有打开真实正常/源cal值、parent权重或query。本轮模型/1000步/校准均只用合成fixture，真实core使用随机初始化合成CPU输入。

## 实际core与共享步进

`audit_kernel.py`在不变实际temporal/tf/hybrid三mode上，各作五个B64×H100×C1更新。与独立直接forward、原objective、手工LR、clip1和AdamW路径按同RNG逐步对照，全部参数、head梯度产生路径、AdamW状态和RNG逐位一致。实际optimizer仅645标量；全部non-head参数及buffer哈希不变，非head梯度None，只有head保持train，其余module eval。LR核1/5/99/100/101/999/1000及越界拒绝，原100-warmup/1000 schedule没有误用Stage1的500/10k。

TF辅助真实为NO_GRAD_CONNECTION（不是伪造有效零梯度张量），恢复同mask时total与future-only的head梯度相等。未来扰动只改loss、不改历史表示或预测；select/cal恢复Python、NumPy、Torch CPU RNG与flags。真实CUDA RNG路径在静态fork_rng代码中明确，尚未在设备上实测。本代理最初手写LR结合顺序造成末位差断言失败，改为与正式前缀相同浮点结合后对照通过；原日志`kernel_initial_LR_association_assumption.log`保留，该失败不是开发kernel故障。

## 共享L0、1000步及回归

`audit_l0_entry.py`独立调用共享run_stage2的两初始化合成CPU L0，拦截torch.save并计数真实AdamW.step：各恰5次，诊断额外0次；head更新非零，frozen/flags/RNG不变。没有checkpoint/optimizer或可复用cal产物。select first64一次，source first64仅PSM且不冒称覆盖SMD；target两窗原序run95/origin465、100原样保留，未排序或重抽。临时source单位和target float32原公式独核相同，均L0_ONLY。random路径拦torch.load仍成功，未借parent head。原CPU-L0 summary不能自升正式gate，CPU fixture即使重批device也在GPU探测前拒绝。

原冻结9个Stage2及51项相关回归由本代理实际重跑，60distinct、0fail/0error/0skip、30.788秒，含两init完整toy1000→source/target-cal、best补充/固定1000主权重、旧Z及事件/Aff/native/source守卫。不能把这些开发测试原样改名独立原创测试；它们只是回归，与上述独立实际core oracle、独立L0和原失败重放分列。

## canonical阻断与窄修闭合

旧1f5688…的独立正确toy1000→cal先通过，再固定source scale/checkpoint/runtime/op/target trace及target_global_rescale=false，仅把target canonical数组×2并重签NPZ SHA和q99，旧validator仍接受。原完整raw fixture在`canonical_failure/`，`canonical_audit.json`记录此真失败。

新957909…的schema2必须引用raw target_cal_bank、正常manifest/plan、seed及fitted checkpoint metadata；按原score_block和source scale逐窗重算、对原run/origin顺序逐项核验。`audit_canonical_revision.py`独立验证旧两个schema1原件都因缺证据被拒绝，不自动修补；新schema2完整正确toy1000→cal通过，两轨引用同一score对象。再施加同原式×2＋重签q99，准确以bound physical bank不符拒绝；改plan SHA、改fitted seed、颠倒原bank顺序也拒绝。原数值kernel与评分政策未改，没有重跑无关51回归。新schema2独立正确流程为pretrained路径；初始化无关的两函数没有新增random分支，原两初始化正确路径及L0已有前述证据。

## 资源壳阻断与v2闭合

旧dbef…有两项已复现缺口：空lifecycle目录可走公开child绕过supervisor timeout/OS链；identity可不列stage2.py，入口字节变更仍通过。原件及`lifecycle_shell/REVIEW_PARTIAL.md`保持FAIL历史。

`audit_shell_v2.py`独立重放两原语义反例：先提供完整正确identity以到达child检查，空目录准确以缺真实launch拒绝；漏列并改entry准确以全.py覆盖不符拒绝。真实direct-child CLI返回OS2、torch_imported=false。CPU插桩核SIGALRM120在child preflight之前启动，handler执行硬exit124；import路径和entry SHA均比对已批准identity。supervisor新nonce、实时PPID/父进程start tick/完整命令及release/config/approval/digest绑定从源码与冻结diff核对。未实际启动GPU成功路径或等120秒；真实OS wait、UUID、版本、allocated/reserved峰与超时失败回执仍须主签L0实测，不能把CPU插桩称为真实设备生命周期通过。

## 准入范围与交付

代码来源/工程实现/合成正确与拒绝路径在此有限范围PASS，三项原阻断已关闭。真实GPU L0与正式科学训练/校准/指标仍INCOMPLETE。部署必须使用最终raw bytes及完整新runtime identity，批准先于PID，精确hybrid seed0 pretrained/random pair、物理0/1→logical0、120秒/20GiB、pair≤.1GPUh，并纳入首pair1h及Stage2总8h；L0权重/optimizer丢弃，正式训练另签且从原parent/随机seed重启。不能拿CPU fixture或旧L0/runtime identity释放新权重。

原始JSON：`kernel_audit.json`、`l0_entry_audit.json`、`regressions.json`、`canonical_audit.json`、`canonical_revision_audit.json`、`lifecycle_shell/audit.json`、`lifecycle_shell_v2/independent_audit.json`。独立脚本和两代快照/全部合成raw留本目录，AUDIT_MANIFEST给出SHA；关键原件已原字节同步本地。没有训练/校准真实目标、读取query或标签、GPU或新增网络；本阶段CPU执行不虚计GPU费用，也不提供吞吐推定。原Z负结果、float32敏感性和历史遥测边界不变，不能由工程PASS宣称HY-H3收益。
