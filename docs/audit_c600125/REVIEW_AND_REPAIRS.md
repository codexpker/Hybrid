# c600125 有界审计复核与工程修复

基准：`c60012524d0a5c39337bdbcf258b1d581f35506b`；独立修复分支：`fix/audit-c600125`。
该提交、`review/hybrid-20261114`、`main` 和外置历史运行原件不改写。
本轮不训练科研模型、不重评真实 QUERY、不拟合阈值、不改变论文方向。

## 复核与处置

|编号|裁决与实际路径|工程处置/边界|对历史结论的影响|
|---|---|---|---|
|F01|成立，限已停止 `left_predictive_cycle`。`_terminal` 只监督末点，`_prediction_score` 使用整窗原生评分；实际 TEP/PSM 配置均单尺度 `[4]`、`smooth_k=3`。最小反例末点损失不变、末点评分增加约1.65。|保留源码、评分及检查点；新增实际封装梯度/依赖回归。夹具不是官方 LEFT 或真实分数重放。|记录学习—评分语义缺口，不能断言是历史低性能根因；不影响后来的 aux-native 路径，不恢复任何候选。|
|F02|成立，属于比较协议限制。native `forward` 仅优化原生任务，`selection_loss` 和归档 worker 的 `selection/train` 仍按 joint SELECT 选权。|保留原规则，新增 native forward/selection 调用和排序可逆反例。该行应称 joint-SELECT 下匹配续训，不能冒称作者原生选权协议。|未量化归档各检查点的反事实排序；不能据此称旧检查点选错或重新选择。|
|F03|成立，旧 `hybrid_v1.training.validation` 仅 CPU fork、root 模式恢复且无 finally；`fractional.encode` 原 seeding 波及所有可见卡。|当前接口接入 `runtime_state`：每模块模式 finally 恢复、参数及 buffer 所用 CUDA 设备 RNG 隔离；FRFT 只种被保护的生成器。新版 matched-U helper 原字节保留。|是未来复用风险修复，旧运行继续绑定旧 commit/runtime；不估计或改写旧指标。|
|F04|成立，旧 manifest/event overlap 和 F lookup 省 domain；无历史真实碰撞证据。|统一记录身份 `(domain,entity,run)`；role 仅决定用途/打开权限，不加入记录重叠键。先核全角色元数据，再加载允许角色；F 只解析合法唯一 SUPPORT 区间；旧 CLI 输出隔离同名记录。|不修改历史 manifest；新版要求显式 domain，缺失不猜测。不能宣称历史训练泄漏或所有多源结果无效。|
|F05|成立，归档 `fixed_metrics.evaluate` 的主汇总已排除不完整事件，明细却可将 UNKNOWN 显示成数值漏检。|原脚本/CSV/主表不动；新增显式选择的 `event_details` v2，仅为新明细导出 NA 与已观测统计，写入使用独占创建。|历史主汇总没有因此被证明污染。旧明细必须联读 main_eligible；本轮不重导出真实历史表。|
|F06|成立但须分型，测试非全绿不等于同数量算法缺陷。|增加分组/隔离运行与具体资产要求；修复 POSIX import 时机和 Windows 测试 memmap 清理。旧 frozen runtime/oracle 断言不改。|原 VALIDATION 文件不改；新运行另存。当前版本和历史 runtime 身份不等须保留，不能重新签 hash 冒充旧实验。|

## 原始检查与新增回归不是同一件事

附件三个文件均按 `AUDIT_PACKAGE_MANIFEST.json` 原 SHA256 验证。原样运行
`run_audit_checks.py --repo <base checkout> --output <new path>`，得到14项有界见证。
见 [输入身份](BASELINE_INPUTS.json) 和 [本机基线输出](baseline_checks.json)。附件自己的结果未覆盖。
脚本检查的两个 Git blob 与基准一致；其预期值没有修改。
其中 legacy 模式恢复/重叠检查等部分是表达式见证，不会随当前实现修复自动变成行为验证。
修复行为使用新增测试；没有把“旧快照检查通过”当作修复正确的依据。

F01/F02 回归位于 `tests/test_audit_predictive_semantics.py`；F03 在
`tests/test_audit_validation_state.py`；F04 在 `tests/test_audit_f04_identity.py`；
F05 在 `tests/test_audit_event_details.py`（也直接调用旧固定评价器验证明细与主汇总差别）。
F06 环境/生命周期回归及完整分型运行方法见 `docs/testing.md`。
实际新结果另存本目录，不替换 `review_artifacts/VALIDATION.*`。

实际新增定点回归 **34/34 PASS**，包括 opt-in 本机 CUDA RNG 检查；
另外 CPU profile 的32项审计用例为31 PASS/1显式 CUDA opt-in skip。
详见 [回归记录](regression_results.json)。编译和 CLI help 通过。
递归发现首轮是129模块/863项，2失败、11错误、16跳过、73项明确集成未运行，
另1模块180秒超时；不是对旧622项成绩的覆盖，也不称全套通过。
两个 DLL 导入隔离问题的22+9项定点复算全过，首轮失败保留。
剩余外部绑定、当前代码与历史身份不兼容及未定位超时逐条见
[F06实际状态](F06_TEST_RESULTS.json)。
本机原32文件 runtime 确实存在且 COPY 相符；五个冻结核心只有
`engine/evaluation.py` 与当前包不等，其余四个相同，见
[原值对照](F06_frozen_kernel_check.json)。本轮没有修改该核心或其预期 SHA。

完整修改文件见 [CHANGED_FILES.md](CHANGED_FILES.md)；
归档树、预测候选、新版 matched-U、冻结协议等不变范围见
[PRESERVATION_CHECK.json](PRESERVATION_CHECK.json)。

## 新接口的可复用边界

### validation/RNG

`training.validation` 仍返回相同 SSL 定义；只修复调用前后状态隔离。
`runtime_state.evaluation_state` 恢复异质模块 training 标志、CPU 与所用 CUDA 生成器；
自定义 eval forward 若擅自修改 buffer 不由此 helper 回滚，不能把它当任意模型状态事务。
helper 不使用 Python/NumPy RNG，亦不为外部回调承诺恢复它们。
`fractional.encode` 在已有固定种子下保留当前 CPU/当前卡的随机序列语义，
去除对其它可见设备的副作用；`mask_seed=None` 仍按原设计消费调用者 RNG。
matched-U 已审 helper 不替换、不重签身份。

CPU 测试覆盖正常、异常、无合法 SELECT 和坏 plan 退出、嵌套 guard、异质子模块模式、
FRFT 相同种子掩码/输出及旧单设备随机序列一致。
另在本机 RTX 3080 Ti 执行最小合成 CUDA RNG 正常/异常退出检查；
只有一张本机可见 GPU，未宣称真实多卡位级验证。
不同 CUDA 设备的种子权限另有 API spy 测试。无科研训练或服务器作业。

### 记录身份与 role

新版原始记录键不含 role，同一 `(domain,entity,run)` 的 FIT 与 QUERY 时间重叠必须拒绝。
不同 domain 的同名 entity/run 合法；plan 查找另带 role/index，SUPPORT 另核用途与完整区间。
角色过滤只控制打开值数组，不绕过全 manifest 的重叠元数据检查。
历史缺 domain 的输入应在独立新配置中显式补齐，不能修改历史原件或默认为某个来源。
旧 CLI 标签配置也必须显式给 domain，输出文件名以完整身份摘要区分，并保存身份字段。

### 新事件明细 schema

调用 `event_details(full_labels, original_index, frozen_alarm, ...)`，再用
`write_new_details(new_csv, rows)` 独占写新文件。输入为已确定报警规则的布尔报警，
不接受阈值搜索或拟合。只增加明细说明，不重新定义一对一匹配、any-hit 或 Aff-F1。

- `coverage`、`longest_miss_points`、`steady_coverage`：仅完整且边界合格事件有值，否则 NA。
- `observed_only_coverage`、`observed_alarm_points`、`longest_observed_miss_points`：明确仅描述已评分点；未知间隔打断已观测漏段，不跨缺口拼接。
- `coverage_ratio` 是评分覆盖率，不是检测召回；同时提供缺失点数和原轴缺失区间。
- 完全未观测事件的已观测统计也是 NA，不能记作零命中或全段漏检。
- 原标签事件边界不改；触及原运行边界的事件仍单独标记 censored。

它是新版本 opt-in 明细接口；归档 fixed_metrics 仍按原身份复现，未暗中重接旧科研评价。

## 仅登记、不执行的方法变更

若将预测评分改成纯受监督末点，或对所有进入平滑分数的右移输出增加监督，属于新方法定义。
若 native 改用 native SELECT、固定末次检查点或其他规则，属于新比较协议。
本轮均未实施，也未读取 QUERY 成绩作任何此类选择。
F01/F02/F05 原归档代码保留；两个遮蔽候选和共享预测候选的限定 NO-GO 全部不变。
没有证据把本轮工程问题解释成所有负结果的统一根因，也不承诺检测性能提高。
