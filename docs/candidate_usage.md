# 完整候选使用与结果导航

当前链路是历史100点 → TimeRadar 时频摘要和 Autoformer self/cross 时延端点 → 下一点预测 → 连续归一化残差。科学实现 v2 按原窗口 B1 调用官方算子；共享 runner v4 管理数据角色、训练、推理、校准和独立评价。

|模式|实际组成|
|---|---|
|tf|仅时频摘要和共同预测头|
|lag|仅 self/cross 时延适配和共同预测头|
|concat|两路拼接、线性融合和预测头|
|gated|两路 sigmoid 门控融合和预测头|

共享入口：`python -m hybrid_tsad.candidate_experiment --config <config>` 默认 preview；执行使用 `--action prepare|train|profile_infer|infer|evaluate`，其中 train/profile_infer/infer 另需 `--mode <mode>`。示例见 [candidate-development.example.json](../configs/candidate-development.example.json)，实际作业使用 root 已绑定来源/数据/动作的配置。已有目录和结果保持原样，不通过重复执行命令覆盖。

2026-11-02：PSM 四模式各200步真实训练、四组固定128窗口无标签成本测试均 OS0。normal SELECT 选中 step 为 tf200、lag150、concat150、gated150；这是选模记录，不是异常检测排名。prepare、训练和成本测试合计实际约0.11345GPUh。全量 CAL/TEST 推理已签放行（单作业4800秒/20GiB、阶段累计4GPUh），此状态不代表分数或评价已完成；当前尚无 evaluate/标签读取许可。

查看顺序：

1. [当前研究状态](../../CURRENT_RESEARCH_STATE.md)顶部确认最新执行终态；正文保留的早期状态只作历史。
2. [训练与成本摘要](../../results/candidate_psm_development_20261102_v1/TRAIN_PROFILE_CORE_SUMMARY.json)查看各组选中步数、更新次数、实际参数变化、峰显存和实测128窗口成本。
3. [结果目录](../../results/candidate_psm_development_20261102_v1/)中 `prepare/` 保存共同初始化/采样索引/填补中位数；`train_<mode>/` 保存 steps.jsonl、validation.json、parameter_changes.json、loss_curve.npz、selected.pt、final.pt 和 receipt；`profile_infer_<mode>/` 保存固定块实测记录。
4. [全量推理签件](../../preparations/candidate_psm_infer_bound_20261102_v1/PRIMARY_INFER_APPROVAL.json)和 [绑定配置](../../preparations/candidate_psm_infer_bound_20261102_v1/INFER.APPROVED.json)明确授权边界。全量成功后 `infer_<mode>/CAL.npz`、`TEST.npz`、calibration.json、receipt.json 才是分数与阈值原件；以实际存在且核验通过的文件为准。

评价需四模式 CAL/TEST 分数全部落盘并核 SHA 后另行批准，再打开官方标签。主点级指标用 CAL q99 higher 的 strict >；连续第三点确认的事件流单独报告，不回填前两点。AP、梯形 PR、ROC 和固定阈值 P/R/F1、事件、延迟、正常误报与暴露一起看。Aff 依赖真实官方外置实现；Range/VUS 未冻结依赖和参数时显式 NA。

本轮 from_scratch、单seed、每组200步是完整链工程开发：没有加载官方预训练权重，不是官方 TimeRadar/CDL 复现或零样本结果，也不是最终多源/Z/U/F 实验。预测 MSE、门控结构或成功运行均不自动支持检测收益、污染鲁棒、跨域泛化或物理密封故障结论。

模块与来源见 [candidate_tf_lag.md](candidate_tf_lag.md)；角色、初始化、评分和 v4 来源兼容细则见 [candidate_development.md](candidate_development.md)。
