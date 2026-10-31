# 多实体评价B：有限工程验收

日期：2026-11-01。主已读取以下四份独审JSON并接受B有限合成CPU工程PASS；这是独立职责复核，不是外部或双盲审查。仅核合成输入，未读真实TEP目标values/labels、未使用GPU、未优化训练模型，不构成检测收益或创新证据。

真实TEP评价仍须3600个物理银行与7200个固定评分完整封存、全部身份/数值/索引门禁通过，以及独立的主label release。`--allow-labels`本身不替代科学批准；整体跨实体Aff仍NA。U/F未完成。

## 独审原件身份

外置原件保留在研究工作区`independent_audit/HYBRID-TEP100-READOUT-B/`；不将包外路径冒作远程仓库内链接。本记录固定原报告字节SHA，范围以原JSON为准。

|报告|SHA256|已核范围|
|---|---|---|
|`local_identity.json`|`31855450acf596a4a49ede1086664ddabfffb83357881deeb266df3eabfb754d`|冻结B源码/测试/config身份、A/core/training未变；专用B wheel原字节一致，不含官方源码；状态`PASS_FROZEN_B_AND_WHEEL_IDENTITIES`|
|`pipeline_audit.json`|`6190ea417a4edf6cc2ac4f896ecbaee72993fc5f7beeb01c3101e7ea2b17473f`|独立合成配置正路径、72指标/坐标/逐实体Aff、86文件封存；AP/ROC/事件/延迟、标签锁和7类拒绝反例；状态`PASS_INDEPENDENT_SYNTHETIC_B_PIPELINE`|
|`statistics_audit.json`|`92e1f8a2004f86a62144ee8c59e4aa7e1e1d0f83c54b2e4d8484c2fd001eeff0`|7200合成单元，3seed先均值再20fault×20run宏平均；2000共同run抽样/99174、闭式差与CI、共同NA且不重抽；状态`PASS_INDEPENDENT_CLOSED_FORM_STATISTICS`|
|`provenance_audit.json`|`395ac6d4ddb3a67bbadd3596b87bb40b6b85a8aa81339ae46affc85c0d1404c8`|项目matching函数AST同源；外置官方Aff六文件SHA与原冻结一致，未复制源码/不新增许可主张；状态`PASS_PROJECT_HELPER_AND_OFFICIAL_PROVENANCE`|

## 已核包与历史产物边界

|文件|固定源码SHA256|
|---|---|
|`src/hybrid_tsad/engine/multi_evaluation.py`|`8a7ab0780554e3a83164a388367c5d83915c637bd498fd05b6bb4aba505b3b73`|
|`src/hybrid_tsad/engine/official_affiliation.py`|`b422b71d0b7d6de655b398b3bbd1d4b6f32c22e0270358a3e1502c0cbb8c2412`|
|`src/hybrid_tsad/engine/run_statistics.py`|`4f26305938f237052b30313b562cf96f8d87e731e6d4cc26c1844faa547a80af`|
|`tests/test_run_statistics.py`|`9530c45adf08cbae3cbfda912e308e281e8552d25ceb6e0f2448794f2ca4cdb8`|

独审核的是`build/readout-b-wheel/hybrid_tsad_research-0.1.0-py3-none-any.whl`，SHA256为`444f10df4e5cb1c2705279bd88b436ed14073f57af9021dc9459adec8856b971`，三个B模块与已审源码原字节一致。默认`dist`的旧native产物SHA`77628fc31cd85911d64f0894cb69a906fd286f0af21c93b4d771fd7b64df5cc0`不是B，不可拿它部署或声称包含本次修复。wheel/合成银行留在外置或忽略的build资产，不附带上游源码。

[开发验证记录](multi_evaluation_b_validation.json)保留`DEVELOPER_STAGE_COMPLETED_PENDING_INDEPENDENT_REVIEW`及当时文档SHA，属于历史证据，不回写。此次仅更新README/评价说明并新增本验收记录，三B源码/测试/config、A/core/train/source_gate及运行快照不变；不构建、不执行评价、不操作Git。

外部官方Aff许可继续按`UNVERIFIED_EXISTING_LOCAL_DEPENDENCY`披露，不声称已取得再分发/商业授权；职责分离的同家族Agent复核不表述为外部双盲。原始失败和反例不覆盖。
