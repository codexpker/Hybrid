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

## 训练与完整读出的迁移范围

核心model/data/protocol已逐字节复制，预测API可调用。训练/采样计划/源验证/校准/推理银行/标签开锁/统计流程仍由外置冻结runner承担，没有宣称全流程已迁入新包。统一CLI可显式桥接已有两个runner，默认仅输出命令：

```powershell
python -m hybrid_tsad legacy source-wave --script ../scripts/hybrid_10k_wave.py -- --help
python -m hybrid_tsad legacy readout --script ../scripts/run_hybrid_10k_psm_inference.py -- --help
```

readout桥锁定已恢复的implementation_v2版本（ce37f1…），拒绝已知job键冲突的旧v1（3cd868…）；已只读核对远端两个版本SHA。桥接会校验指定外部脚本的固定SHA；不会复制/修改原脚本、run或snapshot。需要执行时，显式加入 `--execute`，`--` 后参数逐字传给原runner，用同一Python解释器启动。例中的`--help`仅打印原runner帮助。source-wave支持原prepare/gate/train，readout使用原run/job参数。任何真实训练/读出仍须已有合同、主审批、预算、冻结run完整输入及相邻依赖；桥接不检查或替代这些科研门禁。readout还需要原相邻`hybrid_readout_guard.py`，source-wave使用run内implementation快照，**不会自动改为本包新代码**。本轮未执行任何训练或读出桥接，不修复旧runner，也不兼容SHA不符的新版本。

下一工程阶段应先迁入原runner的通用职责，保持计划、RNG、step、state_dict、源校准与输出身份，再在合成/冻结银行上验证数值等价；未完成前继续原已批准流程。少样本适配、跨域训练一键流水线与通用数据重定位器尚未迁入，不能据本工程宣称完成两阶段训练。

## 目录与验证

`src/hybrid_tsad` 为安装包；`configs` 为库存配置；`tests` 为标准库测试和可选CPU模型兼容测试；`docs/provenance` 为来源及迁移记录。历史索引见 `docs/provenance/history_index.md`，来源SHA见包内 `core_provenance.json`。

标准测试不需要torch，不运行训练。可选模型测试仅在已有torch且设置 `HYBRID_OFFICIAL_FILE`、`HYBRID_LEGACY_MODEL` 指向现存旧实现时运行（同工作区可自动识别旧实现）：比对原模型全部state_dict、strict加载与相同CPU/RNG输入下三模式预测。没有这些条件明确SKIP，不声称数值验证通过。`verify-core` 仅证明三个核心文件字节相同；`inspect-run` 仅读目录和限定JSON元数据，不加载数组、检查点、标签或复算指标，且不把元数据status解释为审查通过。

本次实际验收：本机11项通过、1项模型条件跳过；另经授权在已有远端环境仅用CPU完成三模式旧实现/API、strict state_dict与checkpoint加载兼容检查，同输入同RNG预测逐元素相同。wheel构建及独立target安装验证通过。详见 `docs/provenance/migration.md` 与 `validation_results.json`；这不是全量训练/读出迁移验收。
