# Native源训练seed报告装配修订

2026-10-31。主及独审锁定原training.py `4be47ac3…`后，才进行本次有限修复。原`HYBRID-NATIVE-TEP100/seed_summary_failure.json`及failure_snapshot保留，不把修复自检称为独审通过。

原问题：末尾使用`dict(..., **spec, ..., seed=seed, ...)`，合法spec包含per-job seed时重复关键字抛TypeError；第一次测试只覆盖不含seed的job，遗漏了真实seed0/1/2路径。原冻结runner使用dict literal，未产生这一constructor冲突。

现仅在报告阶段复制spec，核其seed与已经用于初始化/更新的derived seed一致后移出该键，再装summary；不静默覆盖冲突，也不把合法per-job覆盖全局seed误判为矛盾。优化、source evaluate、LR、RNG、原checkpoint与所有分数代码不改。

同一输出装配修复按主授权补充checkpoint metadata：seed、source_domains、excluded_target、source_manifest_sha256、shared_plan_sha256、training_config_sha256、operation_manifest_sha256，均直接来自已校验cfg/spec/paths。weight_sha256与现有checkpoint_sha256同值。它们绑定summary/provenance供追溯，不单独证明训练历史。

回归直接调用实际完整输出流程，仅合成fixture把旧/新10000循环缩为500：不含job seed及含seed0/1/2均与冻结原循环的参数、AdamW、曲线、select/cal、score-bank和后续RNG精确一致，真实生成summary及sidecar并核所有字段。另在fixture校准完成时刻意使spec.seed与实际训练seed冲突，实际函数拒绝，且没有正常summary/sidecar。不是只测一个dict helper。

远端另建`projects/hybrid_ad_package_validation_20261031_native_seed_v1`，CUDA不可见、线程2、60秒上限，8项native测试全通过8.425秒；包括history100/400旧回归。本机3通过、5因torch缺失跳过，1.072秒；compileall通过。所有优化仅为合成CPU工程验证，无科学训练、GPU、真实query或旧run修改。源码其他冻结模块与三核心未编辑；executor拥有的source_gate文件未编辑。

精确diff与新旧hash、测试、原失败引用见`native_seed_report_revision.patch`和`native_seed_report_revision.json`。待主和独立代理回审，实际TEP训练及L0放行仍需另行审批。
