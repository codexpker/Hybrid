# A读出封存三项独审修订及必要资源字段

2026-10-31。独审锁定原readout_controls `dccd0126…`、multi_readout `fa002175…`及三份synthetic失败后，按主授权一次修订；原REVIEW和unbound_simple_rows/physical_target_failure/control_units_failure保留。B草稿暂停，未将其算作完成验收。

1. **源绑定**：rows逐条等于原source manifest；实际raw source/manifest必须有原source_inputs哈希并核验；cal trace固定256、dtype/role/channel/origin/100→5足迹/64窗域序。通过原source_batch重取目标与finite mask，逐值匹配全部native cal银行。simple与native同实际source计划，不再只相信文件声明。
2. **目标物理绑定**：seal及labelgrant都核原数组SHA/shape/float32/有限值、完整原index/forecast_origin/历史足迹、bank.target逐值等于原array[index]。history scale调用未改HybridPredictor.preprocess原公式在CPU复算，事前固定rtol/atol各2e-6，仅容纳float32 CPU/GPU归约误差；不挑评分版本。bank内部残差/分数算式仍照旧核验，未改collect或原数据。
3. **18项源单位绑定**：catalog保存固定source definition和source assets；每次load（infer/seal/labelgrant之前）重新核原native cal JSON、数组、预测bank及源计划；simple bank逐值等于同源计划下原canonical公式输出，scale/99从固定bank复算；late严格同索引native .5/.5组合99。改裸scale/threshold不能靠有限正数通过；不隐式换PSM legacy单位。

必要资源补充只在A wrapper：每entity模型/输入I/O/计算至写bank前的实际seconds和明确scope；原collect primary forward times的P50/P95及样本数；CUDA在加载模型前reset并记录实际allocated peak，CPU为null。descriptor另存包括NPZ/RNG/metadata写出的whole_entity_seconds，worker总elapsed仍由执行回执记录。两块L0分位明确不代表全组预算；测量不调用随机数。

新增/扩展回归覆盖：原恶意换CSV且native/manifest/plan不变、raw source漏hash、native cal target/mask错配、cal足迹/域序；物理target+7且内部算式/SHA全重算、origin错位、history scale×2，在seal和labelgrant均拒绝；simple scale×11、simple threshold×3、late threshold×3、native scale×2在infer/seal/grant前均拒绝；正常18 controls、36银行/72分数完整组继续通过。计时替换为确定counter后，全部数组和逐块RNG仍逐值相等。

远端新目录`projects/hybrid_ad_package_validation_20261031_readout_a_revision_v1`，既有环境、CUDA不可见、线程2、60秒上限：8/8 PASS，5.717秒。本机1 PASS、7因torch缺失SKIP，0.040秒，compileall PASS。都是synthetic工程验证，无真实TEP值/标签、GPU或科学训练；生产source run、三核心、training/source_gate不改。精确diff/hash、原失败SHA引用见`multi_readout_a_revision.json/patch`，仍待独立回审，不自行写PASS。
