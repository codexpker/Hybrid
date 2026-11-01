# 有限 CPU 评价装配来源

- 契约：`HYBRID_TEP100_U_EVALUATION_CPU_20261101.md`（71fde43d…），32 文件计数勘误（9c247113…）；未改原件。
- `stage2_evaluation.threshold_only` 直接提取包内 `engine/evaluation.py:evaluate_fixed_scores`。AST 校验仅删除局部 sklearn rank import、将 AP/ROC 调用换成已缓存值，并增加缓存 positive_fraction 核对；其余函数体逐语句 AST 相同。来源文件 SHA、行号与断言见 `VALIDATION.json`。不是新评分/事件算法。
- source 路径直接调用未改的 `evaluate_fixed_scores`，依赖其 `validate_coverage` 与 `engine/events.py` 的真实块、二值事件和匹配。外置官方 Aff 仍走未改的 `official_affiliation.load_official_affiliation` 六文件 SHA；不复制上游源码。`multi_evaluation.write_csv` 仅复用表格写出。
- 新 `stage2_run_statistics` 是对未改 `engine/run_statistics.py:run_cluster_statistics` 的 stage/init/threshold 维度装配：同实体均 seed、等权实体 macro、同 run 跨 fault 配对、共同 default_rng99174 的 2000 draws。未冒称整函数 AST 等价；新增固定六比较、S/T 同对象检查、重复前缀和 miss 分母。小格手算与共同抽样矩阵逐值回归是本轮工程证据。旧 PSM time-block helper 未被调用。
- `runtime_stage2_readout_a712825a_v1` 在新增文件之前复制原 32 Python 字节，完整身份与 core_provenance 一并核验。此复制没有执行原 gate。新运行时有 34 文件；旧 31 非 CLI 源文件原样，CLI 唯一新增 `stage2-evaluate` 路由。
- v1 原 ZIP 与失败回执保留；错误是测试 oracle 的 ndarray JSON 序列化，非原核计算失败。v2 使用相同源码字节并修测试序列化、补 registry 语义反例，7/7 合成测试通过。`build` ZIP 不作为正式软件分发；其 SHA 固定于 `VALIDATION.json`。

全程只生成合成 CPU 评价产物；未打开真实 query 特征/标签/分数、未运行模型/GPU/训练/cal、未改旧运行快照。合成 gate 只能说明消费接口，不能充当独审或主真实释放。新科学代码尚待独审；本记录不是性能证据或外部双盲审查。
