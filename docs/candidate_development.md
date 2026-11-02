# 完整候选的共享开发入口

`python -m hybrid_tsad.candidate_experiment --config <config>` 默认只预览，不打开数据、权重或标签。`configs/candidate-development.example.json` 是未放行样例；填路径本身不会启动任何动作。原 `--config <L0-config> --execute` 固定合成 GPU profile 接口及函数保留；历史冻结目录不变。

正式动作使用 `--action prepare|train|profile_infer|infer|evaluate`；train/profile_infer/infer 另加 `--mode tf|lag|concat|gated`。release 必须包含 release_id、approved_actions、modes、protocol_sha256、source_sha256（candidate_experiment.py/candidate_tf_lag.py/candidate_autoformer_lag.py），且逐项匹配。source_sha256 可附加 engine 依赖相对路径，所有条目均检查；前阶段源码身份不一致则拒绝静默迁移。protocol_sha256 是除 release 外全部配置的 canonical JSON SHA；preview 输出它。公开数据逐文件 SHA、shape 也必须匹配。每阶段独立目录存在即拒绝，不覆盖失败原件。依赖 Python≥3.10、NumPy、Torch、torch-frft；评价再需 scikit-learn，Aff 是可选已核官方外置依赖。

2026-11-02 实际进度：PSM 四模式各200步训练及固定128窗口成本测试已完成；全量 CAL/TEST infer 已签放行，evaluate 和标签读取尚未获批。当前使用科学模型 v2、共享 runner v4，源码身份和结果路径见 [candidate_usage.md](candidate_usage.md)。

v4 只扩展资源执行和旧产物来源兼容，训练/推理/评价计算及科学模型不变。当前发布核验十文件源码闭包：三个 candidate 模块、包 `__init__.py`、`config.py`、`engine/__init__.py`、`engine/evaluation.py`、`engine/events.py`、`engine/statistics.py`、`engine/official_affiliation.py`。每个条目均核 SHA，不以同 commit 替代实际文件身份。

原科学配置的 `limits.max_seconds=1800` 不变；root 通过 `release.action_limits.infer` 明确指定 `approved_by: root`、具名 `approval_id`、`max_seconds: 4800`。只有 CLI 已授权的当前 infer action 可使用该 override，CLI 日志记录 action/release/base limits/override；直接调用函数或未获批 action 不自动放宽。override 不能增加显存字段，显存仍不超过原 limit 与20GiB的较小值。**4800秒是单个 infer 作业上限；4GPUh 是本开发阶段累计上限**，包含已消耗约0.11345GPUh，四卡并行也须按各进程 GPU 时间求和。阶段总量由 root 与作业记录管理，不是每作业4GPUh，也不重置总300GPUh计划。

复用旧 prepare/train 的唯一兼容条目是 `release.parent_source_compatibility`：包含 `approved_by: root`、具名 `approval_id`、原十文件完整 `parent_source_sha256` 及精确 `allowed_old_stages: [prepare, train]`。原/当前十文件集合必须相同，唯一差异只能是 `candidate_experiment.py` 从已固定 v3 SHA `c38123ef…` 变成当前 v4 SHA `6dd4b59e…`，另外九份不变；仍核相同科学 protocol 和原 receipt 中每个产物 SHA。infer/profile/evaluate 的旧来源不能借此放行。新产物记录当前 v4 身份，不改旧 receipt、原批准配置或历史冻结件，也不因此重训。

prepare 只打开 train：CSV 明确剔除 timestamp，保留 NaN，以 FIT 每列中位数填补；不后向填充、不增加旧 scaler，FIT 全缺失列和 infinity 报错。窗口 input[t-H:t]、target[t] 完全落在各角色。H100/P1，FIT[0,79488)、SELECT[79588,105984)、CAL[106084,132481)，两处 guard100。TEST 使用 t100..87840。模型自身按历史归一化，评分为归一化逐通道平方误差及通道均值。

prepare 保存 gated master state bank、四模式 step0 状态、共同张量逐键 SHA、容量及独立 NumPy RNG 的完整采样表。四模式所有同名同 shape 公共张量复制一致；concat 专有层独立固定 seed，记录实际 SHA，不声称同函数。Torch mask seed 与验证 seed 独立保存；不同模式随机掩码轨迹不假称完全相同。首轮 from_scratch，不读取预训练权重；模型原有严格 SHA/未知 tau 拒绝接口留在模型包，当前配方不开放其加载。

train 只读 train：batch4、seed20261102、AdamW lr1e-3/wd.01、200 updates、warmup0、clip1。每50步（及最后步）对固定 SELECT 前1024窗口做 MSE，最小者选 checkpoint，同分保留最早；验证固定 RNG 且退出后恢复训练 RNG。保存每步 loss/lr/裁剪前梯度、loss 曲线 NPZ、验证轨迹、实际张量最大变化、选中与最终状态、最终 optimizer/RNG、时间、峰值显存、硬件和 Torch 版本。步骤 JSONL 在失败前逐步 flush。目录 receipt 绑定全部产物、配置/协议/源码 SHA；数据 SHA 在绑定配置内。wall/memory 限制是进程内检查，实际作业另需外部 timeout。

infer 读取选中 checkpoint、FIT 中位数、全部 CAL 及 TEST，绝不打开标签。保存原 float32 连续均值分数、逐通道误差、预测、填补后的目标、原目标索引、input 起止与 available_index；目标到达后才有残差，历史不回填。CAL 全合法窗口的 q99（NumPy higher）固定阈值，strict >，并列等于阈值时不报警。不存在 query 标准化或 query 择优。

evaluate 要求所有四模式 CAL/TEST npz 已落盘且其 SHA 在 release.approved_raw_sha256 中逐件批准，然后才懒加载 test_label.csv。复用 engine.evaluation 的 raw AP、梯形 PR-AUC、ROC-AUC 和固定报警评价，两个流明确分开：strict_point 是原 CAL 阈值逐点 P/R/F1，confirmed3_current 从连续第三个超阈点开始、前两点不回填，缺口重置。每个流分报 point 指标、事件召回/一对一 precision、命中延迟、未命中删失、正常误报点/段及固定正常暴露点数；事件采用完整原标签及覆盖范围，部分未覆盖事件不冒充完整漏检。延迟单位是原采样点，不虚构物理时间。confirmed 流的 point 指标是辅助列，不替代 strict_point 主列。

Aff 仅调用 engine.official_affiliation 已固定身份的真实外置实现；报告明确归属各报警流，缺依赖则不评价。Range/VUS 未冻结真实依赖及原点 buffer 时显式 NA，不私自实现同名指标。单 seed 小预算只作工程开发，不支持新颖性、泛化或泄漏物理机制结论。首轮 query 全量前应由 root 明确批准无标签固定块成本测试和总预算；不得用标签选择测试范围。

`profile_infer` 仅测 TEST 前128个合法窗口，另存固定原索引分数、实测计算时间/峰显存和 CAL+TEST 线性成本估计。估计含首 forward 开销、排除读取，不是保证上界。它不加载标签、不选择范围；全量 infer 必须另行明确批准。

合成测试：`python -m unittest discover -s tests -p test_candidate_experiment.py -v`（包需安装或 src 在 sys.path）。测试包含几何/NaN/校准/三连边界/无标签权限及 Tiny 四模式全部 prepare/train/infer；Tiny 是明示测试 fixture，不是生产模型替代。无 Torch/sklearn 环境明确 skip，远端已有依赖应要求全量无 skip。
