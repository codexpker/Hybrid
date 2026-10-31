# Stage2 canonical 消费者闭合修订

原候选 `stage2.py` SHA `1f5688dcf0dcf03b17e84397985fe5f01752e6ff9a0324d7916d7944f34d115a`
及原测试保存在 `before/`；上一阶段 FREEZE、验证报告和独审失败原件均保留。
原失败：仅把 target canonical 分数整体乘2，更新分数文件 SHA 与 q99，
保留 source 尺度、checkpoint/trace 身份和 `target_global_rescale=false`，旧消费者仍接受。

本修订仅修改 `calibration_tracks`、`calibrate` 的输出身份装配及其回归断言。
target threshold 使用 schema2，必须引用物理 `target_cal_bank`、目标 manifest/plan、seed、
fitted checkpoint metadata；旧 schema1 因缺少绑定而明确拒绝，不作自动补齐。
消费者复用冻结 manifest/trace 验证，核对 checkpoint 的目标/trace/seed/源码身份，
核对银行和分数对象的 run/origin 原顺序，再以原 `score_block` 对银行的
prediction/target/history_scale 及既定 source 尺度逐窗复算，要求 float32 逐值完全相同。
之后才核原 q99；不增加目标尺度或另一份合法排序分数。

验证不调用模型，不改银行或原评分公式。固定数据资产的 SHA 仍需可读；
只做正常训练缓存的身份核验，不解码标签或访问 query。
正确新格式完整共享校准路径、×2/q99 原式反例、银行重排、trace SHA 和 fitted seed 篡改均纳入 CPU 回归。
原式反例在新产物上重放，原审查 schema1 产物也需明确拒绝；原失败本身不被覆盖或改写。

本修订仍只属于合成 CPU 工程验证，没有真实目标训练/cal、GPU、query 或标签执行。
最终字节、原审查证据引用、相对原候选 diff 和回执见 `revision.json`。
