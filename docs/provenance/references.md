# 官方源码与当前实现的对应关系

核验日期：2026-10-31。此表区分源码阅读、代码复用与自写候选，不是效果或创新认证。

|参考|固定官方来源|已读位置|当前工程关系|
|---|---|---|---|
|TimeRadar|[mala-lab/TimeRadar](https://github.com/mala-lab/TimeRadar/tree/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2)|models/TimeRadar.py完整Model调用链及exp/exp_anomaly_detection_timeradar.py训练、微调、CDL、test入口|外部只读导入Model类，未复制其源码或发布权重；并未导入完整CDL/官方评分训练流程|
|TimesFM3|[google-research/timesfm](https://github.com/google-research/timesfm/tree/e51928e27119cb17bebc005be2696b75e0a9e688)|src/timesfm3/torch/model.py、transformer.py、configs.py及包装输出/协变量mask接口|仅设计参考；本项目4层128维CausalBlock独立编写，不称官方模块移植或330M权重迁移|
|DS-MAE|[正式论文](https://doi.org/10.1109/ACCESS.2026.3716860)|方法、两阶段训练、实验及局限|官方作者代码未确认，CODE_NOT_ACQUIRED；仅借鉴训练范式，不能称代码复现|
|LEFT|[DezhengWang/Left](https://github.com/DezhengWang/Left/tree/3fadb4811797075e233076b9faceb0d8ec59b697)|models/Left/Tools.py、LEFT.py、数据/评价入口|近邻审查；没有复制其交互或评分模块；选择性有向交互不是本项目自动成立的原创贡献|

TimeRadar官方Model源码SHA256固定为`881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317`。实际使用FRFT、实虚Patch嵌入、互补掩码、chirp卷积、decoder及逆变换；阶数网络从batch/channel平均输入推断共享阶数。该分支不是“完全没有时间结构的纯频率网络”，不能据新增时域分支先验声称获得新信息。

固定官方评分需核cal_anomaly_score及真实调用的L和criterion；不能将论文、预印本与默认代码评分混成一个唯一公式。固定CDL执行入口包含detach及无梯度EMA路径，不能仅因日志有margin就称有效训练信号；本包不声称解决该上游实现差异。

TimesFM3所读源码为推理实现，默认20层、1280宽、16头，输入32点Patch、输出64点Patch与9个分位数，含因果时间注意力/位置编码和变量注意力；未取得同配方完整预训练流程。代码Apache-2.0与3.0权重单独使用许可必须分开，本包不附带权重或源代码。

后续修改参考模块前，先登记commit、文件、函数、许可、实际阅读范围、计划差异和验证合同；仅思想启发就标independently implemented/inspired，没有源码则标未取得，不凭空补作“官方实现”。
