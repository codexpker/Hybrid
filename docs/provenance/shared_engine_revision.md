# 独立审查后的三项修订

2026-10-31。针对`HYBRID-SHARED-ENGINE/REVIEW.md`的REQUIRES_REVISION；本记录是执行者修复与自检交付，等待独立回审，不自行改写为审查通过。

原审查明确复现：整体Aff误取最后块、假完整/缺失coverage可通过、固定审批下阈值可改。原REVIEW、adversarial_results、provenance_results保持原位且未改。其SHA及修复diff身份见`shared_engine_revision_audit_references.json`；`shared_engine_revision.patch`与独立审查远端只读package_snapshot中的工程wrapper比较，不使用模型重跑结果覆盖失败。

## 修订内容

1. Aff每个真实连续有效块保留结果，无GT块明确`NO_GROUND_TRUTH_EVENTS`且不调用外部callback。仅单合法真实有效块可填整体Aff；多块整体NA，原因`MULTIPLE_REAL_BLOCKS_NO_GLOBAL_AGGREGATION`，没有取末块/自造平均。
2. 统计前验证coverage必须有唯一非空ID、整数合法有序不重叠区间、真实total/scored计数、bool资格、完整区间逐点存在且全部正标签、所有评分正点恰被映射。矛盾直接拒绝。合法不完整事件保持原标签与全部点指标，排出完整事件指标，不将缺点改为正常、不修边界。
3. engine操作schema升级2；v1拒绝执行。规范manifest绑定operation、合同、输入路径/SHA、全部parameters、score_space、输出路径；排序UTF8 JSON后SHA256。approval整体排除，避免自指；审批必须包含`operation_manifest_sha256`。预览打印manifest和digest，批准后只更新配置中approval文件SHA。固定合同与输入下改阈值/scale/单位/输出会导致批准失配。

infer/evaluate必须读取固定SHA源cal JSON；显式scale/threshold严格等于产物。产物有schema、source_cal角色、源bank SHA、来源operation SHA及native score-space。infer银行metadata绑定源cal SHA、score-field及实际bank SHA；evaluate另核metadata、银行身份和实际`standardized_score == raw_score / fixed_source_scale`后才加载标签。既有legacy canonical等异单位没有隐式映射；没有经单独批准的适配器就拒绝，不猜测换算。

这些检查不宣称从字符串角色自动证明源/目标隔离；外置源cal产物、源bank和原操作manifest仍需上层科学合同与独立审查追溯。不会给旧run/审批补schema2字段，也不会重跑或覆写旧银行。

## 回归与边界

新增`tests/test_revision_guards.py`共12项（带多项子用例）：双块Aff[.25,.75]整体NA、单块可用、无GT块不调用callback；跨缺口假完整/遗漏正点/重复ID/重叠/非法边界/计数/错误资格均拒绝；合法不完整仍保留AP；固定审批改阈值/scale拒绝，即使新审批但显式数值不符固定cal也拒绝；改bank unit、raw field、cal身份、bank SHA及把raw冒充标准化均拒绝。新fixture批准只用于合成测试，不签发实际研究批准。

本机Python3.12：34项，27 PASS、7因既有torch/sklearn缺失SKIP，0.921秒；compileall PASS。远端另建`projects/hybrid_ad_package_validation_20261031_engine_revision_v1`，既有环境、CUDA不可见、torch线程2、60秒上限：12修订回归＋9共享引擎测试＋1真合成infer测试，共22/22 PASS，0.970秒。原source计划/源cal/forecast数组与RNG等价测试继续通过。未训练、无GPU、没有真实query或新标签选择，没有依赖安装、旧run及三核心变更。

尚未迁移清单保持：原生优化循环、全部bank齐备/恢复放行/标签开锁门禁、官方Aff callback接入与完整多seed装配、U/F适配、通用路径迁移。此次只修工程资格/审批/单位边界，不能追加检测收益或科研验收结论。下一步由未参与修复的独立代理复核新diff、反例与正确路径。
