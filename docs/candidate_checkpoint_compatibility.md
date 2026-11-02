# 官方 TimeRadar 权重的显式 EMA metadata 兼容

本修复只处理加载格式，不增网络、不训练、不打开 query，也没有核查或加载真实 `528aea6e…` checkpoint。原科学 v2 快照、历史权重和实训参数保留；新加载版不得冒用旧十文件闭包的批准身份。

已读固定作者 `exp/exp_anomaly_detection_timeradar.py:Exp_Anomaly_Detection_TimeRadar._build_model`（commit `4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2`，本地原件 SHA `ab7208da3f5c1e201e76167b6d8003d12e0fd357bef721a053598b5d9c222ab0`）：该包装在 `Model` 之外注册 `tau_n`、`tau_a` 浮点 scalar 和 `tau_inited` bool scalar。候选没有导入该包装或 CDL 目标。

`load_tf_checkpoint(..., allow_official_ema_buffers=False)` 默认仍拒绝任何 extra key。显式设 True 时仅允许完整三个已知键，scalar shape `[]`，tau_n/a 浮点且有限，tau_inited 严格 bool。部分 triple、其他 extra、错误 shape/dtype、非有限值、模型缺键/shape/dtype 错误均在实际 load 前拒绝。模型原键仍 `strict=True` 加载，绝不使用 `strict=False`，不向候选注册 EMA buffer。

剥离 metadata 的原值、shape、dtype、每个 tensor 原始 CPU 连续字节 SHA、checkpoint SHA 和作者来源写入返回 provenance 及 `model.tf_checkpoint_provenance`；后者是普通属性，不进入 state_dict 或预测计算。调用方应将这份 provenance 保存到后续独立运行记录。

配置中的 `tf_checkpoint` 可显式增加 `allow_official_ema_buffers: true`；未给出时默认 False。tf/concat/gated 会核整个权重 SHA 并严格加载；lag-only 即使共用含该字段的配置，也记录 `NOT_APPLICABLE_LAG_ONLY / checkpoint_opened=false`，不解析该权重路径、不打开文件、不调用 TF loader。无权重配置的四模式构造、默认参数和 state_dict 不变。

新增 `tests/test_candidate_checkpoint_ema.py` 只生成 synthetic checkpoint fixture，覆盖严格拒绝、完整 triple、元数据原值追溯、加载前原子拒绝及四模式 factory 路由。本机没有 Torch，语法可检查但这些 Torch 测试应明确 skip，等待既有远端 CPU 环境核验；不得据 fixture 声称真实官方权重已载入、已预训或检测有效。当前开发 runner 仍明确拒绝预训练配置，本修复没有修改或放宽它。
