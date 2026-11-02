# 完整V1：首波公开多源预训练

2026-11-02。完整模型定义见[架构](architecture.md)，不是旧四模式预测候选的续跑。来源、计划、权限、实际状态统一外置到研究工作区；不向Git加入原始数据或权重。

## 当前来源

- SMD：machine-1-1/1-2/1-3的原train文件FIT，完整38通道；machine-2-1的不同时间区间SELECT和CAL，中间800点隔离。
- Monash Sunspot：原生单变量T1，73924点，按原时间顺序划分FIT/SELECT/CAL，两道544点隔离。3240个自然缺失位置已按官方with-missing版本恢复，旧LOCF缓存保留，填补值不作教师GT。
- MSL：当前缓存缺原始流边界，标记PENDING，不把拼接表冒充连续run。PSM不进本次源训练，保留为后续探索目标域。

Sunspot来源和归属政策见[Zenodo](https://zenodo.org/records/4654773)及[SILSO](https://www.sidc.be/SILSO/citations)，本轮仅非商业研究。无异常标签不妨碍SSL；不宣称自然异常污染为0，后续污染曲线仍需独立实验。

## 固定训练规则

完整V1三seed2025/2026/2027，分别单卡并行。每seed8000次事前固定的源抽样attempt，取原10k计划的同一前缀，不重采样、不按loss或query AP选步。因2倍保守成本规划，10k超首波4GPUh；8000预测3.8773GPUh，仍不是耗时保证。

H512/W100/D128/P32/B1；AdamW lr3e-4/wd.01/clip1，float32、无AMP、无warmup；TF/MS/预测Huber损失加0.1一致性。每100attempt固定64窗SELECT（每域32），只以源SSL选权重，平手取较早。实际attempt/update/skip、来源和通道曝光分列，不能预写8000次成功更新。

源CAL另动作执行，每域512个固定原始索引，所有seed共用；等域加权经验CDF校准分项中位尺度及q99，首次累计权重达到q，严格score>阈值。不能将它误写为NumPy所有分位数统一higher，也不保证未见目标域的1%误报。

角色过滤只将相应切片交模型/统计，但共同TXT/NPZ为校验与解析可能整体读入；不声称CAL原始字节从未读过。成本探针已作少量CAL前向，权重、阈值与数值不用于正式初始化或选择。

## 统一入口与存档

共享 `hybrid_tsad.hybrid_v1.cli`：

```text
python -m hybrid_tsad.hybrid_v1.cli --config <bound-config> --phase source_train --output-dir <fresh-path>
python -m hybrid_tsad.hybrid_v1.cli --config <checkpoint-bound-config> --phase calibrate_source --output-dir <fresh-path>
```

默认预览；真实执行另需具名release、其SHA和`--execute`。训练许可不自动放行CAL，更不放行query或Z/U/F适配。

每run保留initial.pt、source.pt（selected）、final_training_state.pt、训练trace及selected_attempt；每SELECT原子更新recovery_latest.pt，仅为恢复证据存档，**尚未实现resume入口**。初始、selected和final不能混称；checkpoint身份以文件SHA为准。

本首波的source_action刻意限制具名SMD/Sunspot角色，避免误开target；核心模型和通用data/API不依赖固定C。其他源组合需新配置/权限和协议，不以本签件外推。

## 实际结果位置和边界

研究工作区 `results/hybrid_v1_stage1_full_N8000_seed{2025,2026,2027}_20261102_v1/source_train/`；各生命周期在 `preparations/hybrid_v1_stage1_source_build_20261102_v1/N8000_seed*_TRAIN_LAUNCH_v1/`。当前真实状态看工作区CURRENT_RESEARCH_STATE，不凭本文断言训练已结束。

训练每卡20GiB、4400秒TERM+5秒KILL；未来源CAL每卡30GiB、300+5秒。六动作硬预留加已耗成本不超4GPUh，沿原300GPUh账，不重置或重复累加预留。超限或失败保留原件，不自动换seed、延时或重跑。

完整V1与两阶段接口可运行不证明检测、污染鲁棒或创新已成立。后续先固定完整模型读出，再做原模块、长时序、Self/Cross-lag、多视图融合、单证据与预训练/污染/ZUF系统消融；不以短诊断的涨跌否定统一路线。
