# 连续正事件紧邻切分：单项修订

2026-10-31。收到`REVIEW_REVISION.md`冻结通知后才修改工程。独审已确认原三项阻断关闭，本次仅修其新增的二值连续事件切分反例，仍待独立回审，不自行验收。

`validate_coverage`为每个评分点记录coverage owner；对原始时间索引相邻、标签同时为1的两个评分点，拒绝owner切换。不会改标签、边界、分数、事件定义或把用户给定事件自动合并。原反例同index/label/score下，合法`[1,3)`仍是1事件，拆为`[1,2)+[2,3)`直接拒绝，不能据切分变更分母/命中率。由正常点分隔的两个事件仍合法。

这只验证评分子集可见的一致性。缺失时间索引不被拼接，评分区域外的原始事件边界仍须由上层冻结完整原标签、原始最大连续正段清单及其SHA来证明；当前函数不声称局部样本已认证全局原边界。新测试特意保留原索引缺口场景，不推测缺口里的标签。

新增`tests/test_adjacent_event_revision.py`三项：同一可见正段拆ID拒绝；正常分隔与真实时间缺口保持；完全相同分数/标签下合法合并原事件命中率1，错误拆分拒绝，而真正两个正常分隔事件分母2/命中率.5照常计算。原三项审批/单位/Aff/coverage修订及原source/cal/forecast数组/RNG对照继续通过。

本机37项：29 PASS、8既有依赖SKIP，0.756秒；compileall PASS。另建远端`projects/hybrid_ad_package_validation_20261031_adjacent_v1`，既有环境、CUDA不可见、torch线程2、60秒上限：新3项＋原12修订项＋9共享项＋1真合成infer，共25/25 PASS，0.986秒。无训练、无GPU、未读真实query标签、未装依赖、未改旧run或Git。

`adjacent_event_revision.patch`保存相对于本次冻结前evaluation及新增测试的精确diff。`adjacent_event_revision_files.json`锁定新代码、原回审及反例SHA、结果和其他已审源码未变检查；旧REVIEW/REVIEW_REVISION、失败JSON及上轮diff/manifest均保留。三核心原SHA不变。未迁移清单与上一阶段一致，没有扩大范围。
