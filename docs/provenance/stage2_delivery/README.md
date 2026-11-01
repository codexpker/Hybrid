# Stage2 已审工程交付索引

本次仅交付原字节快照、工程候选清单与审查回执，不执行模型、gate、评价或标签读取；不 stage/commit/push。真实工作流由各实际独审、主接受和具名 release 决定，本文不是实时运行状态。

## 独立运行时

已将资源 v3 实际 **34 Python + core_provenance.json** 原字节 COPY 到本地 `research_workspace/preparations/hybrid_tep100_u_readout_20261101_v1/runtime_evaluation_31141c00_v1` 和服务器相同 preparations 相对路径。两端集合和逐文件 SHA 全核，无 package import。

- identity SHA `31141c0059fed5af83a50bb47b596f6b62e60df3f692e08c5ab086ce9597ca50`，见[原冻结身份](../stage2_evaluation_resource_v3/EVALUATION_RUNTIME_IDENTITY.json)。
- [COPY manifest](runtime_evaluation_COPY_MANIFEST.json) SHA `9d7da9fbada9d5ead7a84b6e9bfeb8cd8eb654a06ec8899831b1ffbe0ffa3fe7`；[远端校验](runtime_evaluation_REMOTE_COPY_VERIFICATION.json)记录未执行。
- 传输 raw ZIP SHA `59bd56d299c9286c80ac81e9348647c92eeef4962d405f651cd6579e8dc3c4b0`。ZIP及执行源码快照外置，不纳入Git候选。
- 原读出 `runtime_stage2_readout_a712825a_v1` 为32文件，原957训练运行时及v2均未覆盖；评价新模块不可插入原读出快照。Git archive/换行过滤后的字节不能冒作该原字节身份。

## 候选源码和范围

仅四个已审新增模块：`stage2_readout_identity`（schema6权限/身份）、`stage2_readout`（18U+9Z正常校准/推理/全组封存装配）、`stage2_evaluation`（独立schema7门禁消费/单rank双阈值/逐实体核）、`stage2_run_statistics`（stage/init维度与固定run簇统计）；CLI只添加明确路由。两个对应tests、示例、说明和来源/正负证据一并列出。core/source训练/score/原Z守卫不改，不纳入任何真实数据、权重、分数、外置Aff或官方源码。

[CANDIDATE_FILES.json](CANDIDATE_FILES.json) 列确切文件与原字节SHA供主逐项选择；[CANDIDATE_DIFF.patch](CANDIDATE_DIFF.patch) 对当前HEAD/新文件生成，未操作index；[DELIVERY_CHECK.json](DELIVERY_CHECK.json)记录只读Git diff/check、attributes过滤身份、文档链接与原字节核查。patch上下文空格是必要证据，禁止清理或转行；新增科学文件和证据使用`-text`。

## 限定审查和未完成工作

[六份原字节REVIEW/verdict](../stage2_review_receipts/COPY_MANIFEST.json)来自v9/v2/v3独立职责复核，保留各自当时结论及局限：同家族/共享工作区，非外部或双盲；工程有限PASS不等于检测收益。v2 7项、v3 8项是各阶段合成CPU证据，不与独审测试数累加。原失败/历史候选待审措辞保留，不回写历史。

未来真实评价必须消费已具名原冻结gate实际执行、完整seal、OS/resource、独审/主接受及独立label release；当前交付不产生它们。合成300秒、TEP100预定900秒/2GiB不代表全400实测吞吐；不选择阈值/方法/分数，不宣称检测增益、创新或F完成。
