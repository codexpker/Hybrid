# Hybrid V1：正式论文—实际源码—项目模块映射

核验日期：2026-11-02。不是只读摘要；阅读范围与缓存见下表及 notes/hybrid_v1_20261102 的两份原始审读。论文出版状态、代码许可、预训练权重许可分别记录。

|来源|正式版本与阅读范围|固定官方代码与实际位置|V1 采用方式与不采用部分|
|---|---|---|---|
|TimeRadar: A Domain-Rotatable Foundation Model for Time Series Anomaly Detection|Hui He et al.; KDD2026 V.2,1602–1613; DOI10.1145/3770855.3818062。已读正式全文§3.3–3.5、方法公式与实验；作者机构[出版记录](https://ink.library.smu.edu.sg/sis_research/11305/)|[mala-lab/TimeRadar](https://github.com/mala-lab/TimeRadar/tree/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2)，models/TimeRadar.py::Model.forward、FractionalEncoder、MLP_Decoder、cal_anomaly_score；exp/exp_anomaly_detection_timeradar.py::_build_model、_ema_update、train|复用FRFT/order/复数编解码类与既有权重接口；native轴保留、逐原窗口隔离。新双pass、融合后原生解码与signed-real重建是项目适配，保留author_abs控制。CDL不伪装成无标签污染鲁棒；原score保留为控制。代码/权重许可UNVERIFIED，外置。|
|LEFT: Learnable Fusion of Tri-view Tokens for Unsupervised Time Series Anomaly Detection|Dezheng Wang et al.; KDD2026 V.2,4800–4811; DOI10.1145/3770855.3818044。已读全文§3.1–3.7及实验、Eq1–35；[出版记录](https://ink.library.smu.edu.sg/sis_research/11302/)。本地可读稿为含正式DOI的arXiv2602.08638v2，不将两种PDF字节说成同一版本|[DezhengWang/Left](https://github.com/DezhengWang/Left/tree/3fadb4811797075e233076b9faceb0d8ec59b697)，models/Left/Tools.py::MS_Utils、CrossAttnFFNBlock、MultiViewFusion；models/Left/LEFT.py::_forward、anomaly_score|复用真实T←F、F←T、M←T结构、多尺度Nyquist算子；将固定C投影改为共享单变量组织是项目适配。不默认搬入完整prototype bank、STFT cycle/全部评分gate。repo当前简介称LIFT，与论文名分列，不从简介推正式题名。代码许可未核，外置。|
|TimesFM 3.0 官方技术发布|Ayush Jain、Rajat Sen；Google Research，2026-08-31，[官方技术说明](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)。已读architecture/inference/training相关说明和真实源码；3.0正式会议/期刊论文未核实|[google-research/timesfm](https://github.com/google-research/timesfm/tree/e51928e27119cb17bebc005be2696b75e0a9e688)，src/timesfm3/torch/transformer.py::MixingTransformer、StackedMixingTransformer、make_attn_mask；dense.py::ResidualBlock；configs.py / 模型配置；model.py::_preprocess仅用于核语义|复用小D/少层、关闭variate attention的因果核心，加项目训练入口；不加载330M权重、不采用inference-only整包装。官方同步variate attention不是显式Cross-lag。源码Apache2；3.0权重另有非商业限制，本轮不使用。|
|A decoder-only foundation model for time-series forecasting|Abhimanyu Das、Weihao Kong、Rajat Sen、Yichen Zhou；ICML2024/PMLR235,10148–10167，[正式论文](https://proceedings.mlr.press/v235/das24c.html)。实际读§4 Eq2–5、§5预训练、§6实验|同官方TimesFM项目，初版模型与3.0分开标版本|为patch化、因果decoder、next-patch预测和异构预训练提供正式基础，不把初版正式发表移接成3.0发表，也不证明异常污染鲁棒。|
|Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting|Haixu Wu et al.; NeurIPS2021；已读完整§3.2 Eq5–8、§4和AppendixG算法。[论文](https://arxiv.org/html/2106.13008v5)|[thuml/Autoformer](https://github.com/thuml/Autoformer/tree/51c7d416ae120b805fd5beef2f4ccf7de496a6ff)，layers/AutoCorrelation.py::AutoCorrelationLayer、AutoCorrelation.forward、time_delay_agg_training/inference|成熟lag相关/聚合基础与旧原算子对照；V1非循环正lag/有效重叠/有向变量对是项目新适配，不称作者官方Cross-lag模块。MIT已核。|
|TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting|ICML2025/PMLR267:24893–24911，[正式论文](https://proceedings.mlr.press/v267/hu25ac.html)；方法§4.1–4.4、实验§5已读|[官方仓库](https://github.com/TROUBADOUR000/TimeFilter)，固定dffde87e4fff0fdeeebbacde03dc1e432e15b3a1；GraphLearner/mask_moe，详TEMPORAL_LAG_PAPER_CODE_EVIDENCE.md|作为近期空间—时间关系近邻；含MoE的完整过滤器不接入本轮。不把它描述为显式物理lag估计器。|
|AAAD: Asynchronous Inter-Variable Relationship-Aware Anomaly Detection for Multivariate Time Series|ICME2025，DOI10.1109/ICME59968.2025.11210144；全文仍受访问限制|固定44080a3cca1476d56a2842e49072fd477ce9aa77，model/encoder.py::GraphBlock.forward 已读，真实N×N pair和循环shift|只记录代码邻近事实，不按摘要设计完整模块；negative circular shift/wrap不直接当有限正历史lag。代码许可未核。|

## 直接复用、思想借鉴、新实现的分界

污染损失第一基线直接采用 [PyTorch 2.8 HuberLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.HuberLoss.html) 的逐元素公式，delta=1；与MSE同曝光比较。此处已读实际公式/API，不将Huber描述为2025–2026新模块或已经解决污染。多源、Z/U/F和事件隔离是本项目训练协议，不能伪装成TimesFM或LEFT原文已经提供的完整污染异常检测方案。

直接复用：外置TimeRadar算子类、LEFT交互/滤波子组件、TimesFM3因果块和mask实现；保留作者身份/许可文件。思想借鉴：异构预训练、长历史、next-patch和多尺度协同。新实现：原窗口隔离、可变C适配、Self/Cross有限lag拓扑、两pass/可用性索引、融合后解码接线、混合训练损失与Z/U/F数据合同。不能把后者的效果或新颖性借作者声誉预先认证。

## 本地原件入口

- TimeRadar正式全文：literature/novelty_review/local_pdf_reading_20261031/H242KUSZ_56fc808edf.txt；模型源码SHA881e8912…；原复现与权重兼容结果不重跑。
- LEFT全文：同目录MQEIVD5X_3ba7aff9ed.txt；固定源码literature/novelty_review/left_source_20261031/。
- TimesFM3：literature/reference_source_20261031/timesfm3/，固定commit及逐源manifest已存。2026-11-02审读 notes/hybrid_v1_20261102/TEMPORAL_LAG_PAPER_CODE_EVIDENCE.md。
- 工程者源码核对：notes/hybrid_v1_20261102/EXECUTOR_REUSE_AND_IMPLEMENTATION_PLAN.md、EXECUTOR_SOURCE_BINDINGS.json。

若网页元数据与可读稿/代码版本有差异，分列证据，不用一次search摘要替代正式全文。未认证的软件许可不通过复制到Hybrid仓库自动获得授权；公开仓库保留项目实现与来源记录，外部源码、数据和权重继续外置。
