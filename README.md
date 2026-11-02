# Hybrid TSAD 独立研究工程

这是公开时间序列异常检测研究的独立工程，保留既有 Hybrid 核心、参考方法适配和可复用实验入口。当前按2026-11-02正式纠偏，构建微弱异常与时滞关联异常的完整 Hybrid V1；工程可运行不代表检测收益或创新已成立。

## 当前入口：完整 Hybrid V1

先读[完整架构](docs/hybrid_v1/architecture.md)、[正式论文与实际代码映射](docs/hybrid_v1/paper_module_map.md)、[实现和系统消融计划](docs/hybrid_v1/implementation_plan.md)。共享实现位于 `src/hybrid_tsad/hybrid_v1/`，使用真实外置 TimeRadar FRFT、TimesFM 3.0 小型因果块、LEFT 多尺度与交互子组件，并明确区分项目适配的有限 Self-lag 和 Cross-lag。TimesFM 3.0 是官方技术发布，不冒称已核实的正式顶会论文。旧实现、权重和全部正负结果保留。

统一运行入口见[训练与评分用法](docs/hybrid_v1/tools_usage.md)：

首轮公开多源的角色、三seed、8000次抽样、源验证/校准和权重存档见[Stage1说明](docs/hybrid_v1/stage1_source.md)；原始数据与运行产物保持外置。

```text
python -m hybrid_tsad.hybrid_v1.cli --config configs/hybrid-v1-synthetic.example.json --phase synthetic_smoke --output-dir /new/output
```

命令默认只预览；需绑定本机已有官方源路径及相应运行配置。`--execute` 才实际执行，已有配置示例不代表正式大规模训练已获放行。CPU合成闭环、A40兼容、公开数据科学实验分别验收；最新真实进展在上一级 `CURRENT_RESEARCH_STATE.md`。模型采用源自监督与独立Z/U/F接口，尚不能据代码存在声称污染鲁棒或检测收益成立。

## 保留基线：LEFT官方实现

参见[LEFT基线使用说明](docs/left_baseline.md)，统一入口为 `python -m hybrid_tsad.left_baseline --config ...`。作者代码、公开数据和检查点放在工程之外，本包只提供自有适配与配置；原模型保持固定版本。PSM seed2025完整训练、固定读出和点级/VUS评价均已有限验收（AP 0.576644、VUS-PR 0.572602），没有新方法收益结论。原TimeRadar八基准复评保留，不重新启动。

独立normal-run适配使用 `python -m hybrid_tsad.left_normal_runs --config ...`，默认只预览；正式运行绑定独立协议、来源及批准。当前C2只作为原LEFT的机制能力诊断：LIN/WEAK各seed2025完整正常训练、80固定分数银行及评价均已独审/root接受，没有弱关系稳定收益结论；不改原模型、不接续PSM25权重。透明weak-lag CPU控制保留全部负结果，不能当新的真实公开数据集或已证明的创新。结果导航见上一级 `results/START_HERE.md`，真实终态以 `CURRENT_RESEARCH_STATE.md` 为准。

下方Hybrid工程阶段记录为历史资产，不触发旧训练/适配作业。最新状态在上一级 `CURRENT_RESEARCH_STATE.md`；第二篇继续只使用公开数据。

已有频域强对照使用[CATCH外置官方实现](docs/catch_baseline.md)：共享normal-run训练、固定原生分项捕获与独立CPU评价入口。C2正常训练及80raw已完成并有限复核，完整固定评价也已实际运行；当前没有稳定超过强简单lag对照的证据，不新增模型。两方法训练预算、原生随机mask和回溯时间权限的差异保留，不能把本工程代码入口当论文创新。

先读[研究范围与尚缺部分](docs/research_scope.md)和[官方源码对应表](docs/provenance/references.md)。FRFT本身已含时域信息，新增时域分支的必要性仍须检验；当前多源正常数据预训练不等于已经解决异常污染的无标签学习。

## 历史工程阶段（2026-11-01，旧路线已冻结）

已增加[已知正常数据的预测头适配入口](docs/stage2_u_normal_head.md)：固定现有645参数头，主干、原核心和评分数学不变。三种操作共用`stage2 --config`：1000步正式训练、源/目标阈值校准，以及独立五步丢弃L0；默认都只预览。

[有限独立工程审查](docs/provenance/stage2_independent_review/REVIEW.md)已通过；包括真实三模式CPU步进对照、两初始化合成L0和三个已修复反例。职责独立但同模型家族，不称外部审稿。目标分数[校准修订](docs/provenance/stage2_canonical_revision/README.md)要求从绑定原始预测银行逐点复算，两个阈值轨共用同一canonical分数对象。

该阶段初始记录为“实际GPU L0、正式目标适配与新query尚未执行、服务器SSH超时”；这不是当前运行状态，后续完成与失败记录以上一级状态和原run为准。该控制使用已知正常数据，不是混合异常污染鲁棒或完整U/F成果。旧作业保持冻结，原始TimeRadar复评及已接受的负结果保留。

## 安装与入口

Python ≥3.10。管理 CLI 仅使用标准库，可使用已有 Python：

```powershell
cd research_workspace/hybrid_ad
python -m pip install --no-deps --no-build-isolation -e .
python -m hybrid_tsad --help
hybrid-tsad check-env
python -m hybrid_tsad verify-core --source-root ..
python -m hybrid_tsad config validate configs/core.example.json
python -m hybrid_tsad inspect-run ../results/hybrid_10k_wave3_20261031_v1
python -m unittest discover -s tests -v
```

安装命令只安装本包，不下载模型依赖；需要已有 setuptools≥68。也可临时设置 `PYTHONPATH=src` 使用模块入口。数据接口需要已有 NumPy/Pandas；模型需要已有 PyTorch、torch-frft和外部 TimeRadar 文件自身的依赖。model extra已声明torch-frft（远端已有版本0.8.1）；extra不是完整官方环境锁文件。`check-env --require-model` 在缺少核心依赖时返回1，不导入torch、不探测CUDA。检查到模块存在不等于其二进制/上游依赖可运行。本轮未为现有环境安装任何新依赖。

## 配置与外部资产

`configs/core.example.json` 是可验证的工程库存配置，不是训练合同或新实验计划。复制为 `configs/core.local.json` 后按实际位置填写；路径相对配置文件解析，支持 `${ENV_NAME}` 和用户目录，不硬编码个人服务器目录。默认示例的外部 TimeRadar 文件尚未供应；`config validate ... --check-paths` 会要求两个manifest、合同、官方文件存在，并核验官方SHA。常规validate仅校验结构与模型参数/目标排除，不读取数据。示例种子只为API配置，不代替原wave冻结配置；实际历史运行仍以原run内config、inputs、plan及合同为准。

原数据、manifest、TimeRadar源码、权重、连续分数与运行目录均外置，不能放入本仓库。上游代码/权重/数据许可分别待确认，仓库未附带上游源码、权重或再分发许可；本项目也尚未指定对外开源许可证，不应将可见性视为授权。SHA只验证文件身份，不证明许可或科研正确性。历史manifest内部路径保持原样；迁移磁盘不能仅改库存配置就宣称历史manifest可移植。

## 真正的预测 API

```python
import torch
from hybrid_tsad.api import predictor_from_config, predict

torch.manual_seed(41)
model = predictor_from_config("configs/core.local.json", checkpoint="/external/step10000.pt", device="cpu")
# past_values为已有传感器历史，形状[batch,400,channels]；不传入未来target。
torch.manual_seed(53)
output = predict(model, past_values)
forecast = output["prediction"]  # [batch,5,channels]
```

也可调用 `create_predictor(official_file, mode=..., history=..., copies=..., temporal_depth=..., checkpoint=...)`。API是原 `HybridPredictor` 的薄封装；checkpoint必须是可由 `torch.load(weights_only=True)` 读取的原生state_dict，严格匹配全部键/张量形状。未传checkpoint时为随机初始化，不能当训练后模型。返回原模型输出dict，不增加分数、平滑或阈值策略。TF掩码随机性由调用者管理；固定模型仍需固定RNG才能精确重放。原数据、协议模块可分别通过 `hybrid_tsad.data`、`hybrid_tsad.protocol` 直接导入；保留原接口和限制。

## 已审工程范围与尚缺部分

已审A已推送至`7931ffe`。以下为限定工程状态，不代表真实目标评价完成、检测收益或创新成立：

|范围|状态与入口|说明|
|---|---|---|
|原生源训练|已通过限定独审；`python -m hybrid_tsad train --config configs/train.example.json`|schema3调用包内固定预算训练/源验证循环，100/400历史与plan、源cal、checkpoint metadata绑定；见[原生训练](docs/native_training.md)|
|源L0诊断|已通过限定独审；`python -m hybrid_tsad.engine.source_gate --config /external/source_l0.local.json`|使用外部已批准诊断配置，5个固定batch及mask案例后丢弃模型；入口见[source_gate](src/hybrid_tsad/engine/source_gate.py)|
|多实体固定Z读出A|已通过限定独审；`python -m hybrid_tsad readout --config configs/readout_infer.example.json`|schema4共享source controls、按白名单infer、全组seal和all-bank标签门禁；见[多实体读出](docs/multi_readout.md)|
|多实体评价B|有限合成CPU独审通过；`python -m hybrid_tsad readout --config configs/readout_evaluate.example.json`|逐实体官方Aff外置适配、AP/ROC与固定run聚类统计；见[B评价说明](docs/multi_entity_evaluation.md)及[限定验收](docs/provenance/multi_evaluation_b_acceptance.md)|
|U正常头适配|已有独立Stage2入口；`python -m hybrid_tsad stage2-train --config configs/stage2-train.example.json`|仅固定645参数正常头适配；见[Stage2范围](docs/stage2_u_normal_head.md)，不代表混合污染鲁棒或F完成|
|U读出schema6|v9有限CPU独审通过；`python -m hybrid_tsad stage2-readout --config configs/stage2-readout.infer.example.json`|18U独立身份、9Z正常阈值、单canonical双阈值及全组门禁；见[读出说明](docs/stage2_readout.md)|
|U评价schema7|v2及资源v3有限CPU独审通过；`python -m hybrid_tsad stage2-evaluate --config configs/stage2-evaluation.example.json`|原rank/事件/Aff核与固定run统计，合成300秒/未来TEP900秒上限；见[评价说明](docs/stage2_evaluation.md)|
|F事件少样本适配|尚未完成|不由正常头适配或Z/U读出验收替代|

上述命令默认只预览。示例路径/SHA是占位，不是可执行科学合同；`source_l0.local.json`须由外部已批准资产提供。真实执行需要具体合同、规范operation manifest绑定的主批准、固定输入SHA和`--execute`。旧评价入口的`--allow-labels`要求以其命令为准；schema7的`stage2-evaluate`通过独立主label release授权，不接受该旧参数。入口存在不构成执行授权。

A的其余配置为[源controls](configs/readout_controls.example.json)和[全组seal](configs/readout_seal.example.json)，共用`readout --config ...`入口。TEP100完整组要求9权重×400实体的3600个物理预测/RNG银行、6方法×3seed×400实体的7200个固定评分，逐键核验并绑定源校准/权重、原始数组/目标、索引/足迹和分数单位。L0、部分分片、重复或损坏银行均不能获得标签评价资格；门禁已实现，不再列为未来缺项。B有限合成CPU独审已通过；真实TEP评价仍须完整封存和主label release，工程验收不代替真实query放行。

核心model/data/protocol保持原字节。较早共享引擎的单组`infer/calibrate/evaluate`仍保留，见[共享引擎](docs/shared_engine.md)；Z多实体合同使用schema4的`readout`入口，U使用显式独立stage路由。尚未迁入完整prepare/实体准入、作业级精确恢复、通用数据重定位器和F适配，不能据此宣称两阶段科研流程全部完成。

schema6原读出32文件与schema7资源v3评价34文件分别以原字节外置冻结，后者独立目录名为`runtime_evaluation_31141c00_v1`；新增评价模块不回填旧读出运行时。仓库主源码与两个执行快照的身份不同，不能拿当前动态identity替代旧gate。交付身份及候选清单见[交付索引](docs/provenance/stage2_delivery/README.md)，[原字节审查回执](docs/provenance/stage2_review_receipts/COPY_MANIFEST.json)注明同家族、共享工作区、职责独立且非外部双盲。CPU验收不授予真实标签许可：真实读出、完整seal、原冻结runtime实际gate的独审/主接受和独立label release仍按各自实际原件推进，本文不充当实时实验状态或性能报告。

## 历史runner兼容桥

legacy显式桥用于追溯外部冻结runner，不是当前原生train或多实体A的主入口，默认仅输出命令：

```powershell
python -m hybrid_tsad legacy source-wave --script ../scripts/hybrid_10k_wave.py -- --help
python -m hybrid_tsad legacy readout --script ../scripts/run_hybrid_10k_psm_inference.py -- --help
```

readout桥锁定已恢复的implementation_v2版本（ce37f1…），拒绝已知job键冲突的旧v1（3cd868…）。桥接核外部脚本固定SHA，不复制/修改原脚本、run或snapshot；`--`后参数逐字传给原runner，用同一Python启动。执行需显式`--execute`及既有科学合同/预算/主批准，桥本身不替代这些门禁。readout仍依赖原相邻`hybrid_readout_guard.py`，source-wave使用run内implementation快照，不会自动使用本包新代码。旧结果和失败证据保持原位。

## 目录与分阶段验证

`src/hybrid_tsad`为安装包；`configs`为库存/操作配置；`tests`为基础检查与有条件的合成CPU回归；`docs/provenance`为来源及迁移记录。[历史索引](docs/provenance/history_index.md)保留外置资产索引，三核心来源SHA见[core_provenance.json](src/hybrid_tsad/core_provenance.json)。

基础管理CLI不导入torch。可选模型测试需要已有torch和外部官方文件；旧实现/API对照还需`HYBRID_OFFICIAL_FILE`、`HYBRID_LEGACY_MODEL`（同工作区可自动识别旧实现）。条件不足明确SKIP，不算数值验证通过。完整测试集合包含可选合成优化/源验证回归，不是全部无torch或零优化步；工程fixture不计科学训练证据。`verify-core`只证明三核心字节身份；`inspect-run`仅读限定元数据，不加载数组/权重/标签，也不把status当独审验收。

历史测试数字对应各自冻结阶段，不能相加或当作当前全套测试数：

- 初始包阶段：本机11项通过、1项条件跳过；另有三模式旧实现/API、strict state_dict及checkpoint加载CPU逐元素一致，wheel/独立安装检查通过。见[初始迁移](docs/provenance/migration.md)与[初始验证](docs/provenance/validation_results.json)。
- 共享引擎阶段：独立CPU25项通过；对应当时主本机37项中29通过、8跳过。多块Aff整体NA、coverage/连续事件边界和源cal/score-space批准绑定的修复见[共享修订](docs/provenance/shared_engine_revision.md)及[事件边界修订](docs/provenance/adjacent_event_revision.md)。
- 原生train/source_gate/A：均已通过各自限定独审，范围见上表。原循环来源、合成等价/seed报告修复及A原始反例与修订保留在[原生来源记录](docs/provenance/native_training_validation.json)、[seed报告修订](docs/provenance/native_seed_report_revision.md)、[A绑定修订](docs/provenance/multi_readout_a_revision.md)和[A冷启动修订](docs/provenance/readout_cold_cuda_revision.md)；其中早期文件的待审状态是历史记录，不回写成当时已验收。
- B：有限合成CPU独审通过，四份独立复核报告的范围/SHA及旧dist边界见[B限定验收](docs/provenance/multi_evaluation_b_acceptance.md)。[开发验证记录](docs/provenance/multi_evaluation_b_validation.json)保留当时待审状态，不回写历史；真实TEP评价仍须完整封存和主label release，不据合成指标声称检测增益。

物理尺度核验已改为[显式生产后端与执行证据绑定](docs/physical_validation_backend.md)，见[有限主验收](docs/provenance/physical_backend_acceptance.md)。真实TEP seal/evaluate需配置固定CUDA参考；原模型/银行/分数不变，全部400实体仍须具体运行验收，不能用CPU跨后端重算冒作CUDA身份基准。
