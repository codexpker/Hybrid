# Hybrid TSAD 独立研究工程

这是现有 Hybrid 核心的可安装工程，研究主线是异构公开时序的时频—时域协同预训练与两阶段异常检测。本次工程整理不增加网络、修改评分或重跑实验；工程可用性不代表检测收益或创新已成立。

先读[研究范围与尚缺部分](docs/research_scope.md)和[官方源码对应表](docs/provenance/references.md)。FRFT本身已含时域信息，新增时域分支的必要性仍须检验；当前多源正常数据预训练不等于已经解决异常污染的无标签学习。

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
|U/F适配|尚未完成|无标签目标适配和事件级少样本适配不由Z读出或源训练验收替代|

上述命令默认只预览。示例路径/SHA是占位，不是可执行科学合同；`source_l0.local.json`须由外部已批准资产提供。真实执行需要具体合同、规范operation manifest绑定的主批准、固定输入SHA和`--execute`；评价还需`--allow-labels`。入口存在不构成执行授权。

A的其余配置为[源controls](configs/readout_controls.example.json)和[全组seal](configs/readout_seal.example.json)，共用`readout --config ...`入口。TEP100完整组要求9权重×400实体的3600个物理预测/RNG银行、6方法×3seed×400实体的7200个固定评分，逐键核验并绑定源校准/权重、原始数组/目标、索引/足迹和分数单位。L0、部分分片、重复或损坏银行均不能获得标签评价资格；门禁已实现，不再列为未来缺项。B有限合成CPU独审已通过；真实TEP评价仍须完整封存和主label release，工程验收不代替真实query放行。

核心model/data/protocol保持原字节。较早共享引擎的单组`infer/calibrate/evaluate`仍保留，见[共享引擎](docs/shared_engine.md)；本轮多实体合同使用schema4的`readout`入口。尚未迁入完整prepare/实体准入、作业级精确恢复、通用数据重定位器和U/F适配，不能据此宣称两阶段科研流程全部完成。

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
