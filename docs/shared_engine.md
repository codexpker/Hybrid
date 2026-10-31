# 最小共享引擎：第二工程阶段

本阶段将已有研究脚本中的通用计算归入`hybrid_tsad.engine`。三个核心model/data/protocol不改；没有训练、真实query读出或新标签调参。工程合成验证不算新增科研证据。

## 统一操作配置

`core.example.json`仍是模型库存配置；操作配置calibrate/evaluate使用schema2，native train使用schema3，infer schema3显式绑定history和checkpoint metadata（schema2只保留原400兼容；旧v1执行拒绝），由相同的`config validate`入口识别。四个操作有各自示例；默认只解析配置和路径，不打开数据、checkpoint或标签：

```powershell
python -m hybrid_tsad config validate configs/infer.example.json
python -m hybrid_tsad train --config configs/train.example.json
python -m hybrid_tsad infer --config configs/infer.example.json
python -m hybrid_tsad calibrate --config configs/calibrate.example.json
python -m hybrid_tsad evaluate --config configs/evaluate.example.json
```

示例中的`0…0` SHA、scale/threshold=1及示例job均为占位，不是实验配置。实际配置使用外置合同与批准文件、每个输入文件的SHA、真实源校准尺度/阈值及一个全新输出目录；路径相对配置文件，可用`${NAME}`引用环境变量。不要把配置示例当新训练许可。`config validate ... --check-paths`额外核全部输入身份和批准的合同SHA，但不执行计算。

只有显式`--execute`才执行；evaluate还必须显式`--allow-labels`。批准文件须有`status: APPROVED`、匹配合同的`contract_sha256`及匹配规范操作清单的`operation_manifest_sha256`。预览输出的清单绑定operation、合同、所有输入路径/SHA、全部参数、score_space和输出路径，排除approval自身以避免自指；更改参数/输入/单位/输出需重新批准。批准写入后只更新配置中approval文件的SHA，操作清单digest不变。这项工程检查只核批准文件身份，不能理解科学合同、证明标签使用权限或替代独立审查。当前evaluate处理一份已经固定/归档的bank；完整研究仍须在上层保证全部比较模型已完成并封存后才开标签。

|操作|实际执行实现|保留边界|
|---|---|---|
|train|schema3调用包内共享固定预算优化/source evaluate循环|10000×64预算不变；仅完成合成CPU对照，真实训练未执行；legacy另保留|
|infer|包原生读取固定checkpoint和数值array，调用原模型，逐400历史→5未来块写出预测/分数/RNG银行|默认CPU；保持顺序、评分、首两块RNG重放；不读取标签|
|calibrate|包原生对明确标记source_cal的预测银行应用原源cal公式|输入须由上层证实是合法源校准，标记本身不是泄漏审计|
|evaluate|包原生对固定分数和源阈值计算点/事件结果，保持真实时间间断|不选阈值、不做point-adjust；CLI未接Affiliation外部实现，明确NOT_EVALUATED|

## 输入/输出接口

infer输入：官方模型文件、原生state_dict、float32二维数值`.npy`、query记录JSON。记录包含`history:400`、`horizon:5`、有序不重叠`origins`及完整`expected_time_index`。每块只向模型传过去400点，预测完成后取未来5点计算原`protocol.score_block`。scale来自源cal；模型初始化seed先于load，mask_seed在逐块前设置，首两块重放后恢复RNG，保存每块前状态。CPU路径只用于工程检查，不宣称其结果等同所有CUDA算子。

银行输出字段与原脚本相同：`prediction`、`target`、`history_scale_by_block`、`normalized_channel_residual`、`raw_score`、`standardized_score`、`time_index`、`forecast_origin`、`valid`；另保存`rng_states.npz`和v2`inference.json`。训练job与readout_job分开。时间/显存缺项为null，不编造性能。metadata另绑定实际银行SHA、source_calibration_sha256、score_space和score_field；evaluate核银行值等于raw_score/source_scale之后才加载标签。

calibrate输入包含`prediction/target/history_scale/valid_target`的源cal `.npz`；输出原`cal_scores.npz`字段和`normal_error_scale/threshold99`，并标记calibration_schema_version/role/score_space/source_bank_sha256/operation_manifest_sha256。infer/evaluate都必须输入此类固定SHA源cal产物，显式scale/threshold须严格等于产物；当前只支持native score-space，legacy映射未定义时拒绝。无target选择或新拟合。

evaluate输入包括已归档bank、完整原索引标签`.npy`及coverage JSON列表。coverage每项包含`event_id/start/end_exclusive/eligible_complete/scored_points/total_points`。沿bank的原time_index取标签；不把间断时间拼接为连续事件；不完整阳性事件仍参与点级AP/ROC，统计前严格验证ID唯一、区间有序不重叠、total/scored计数、实际索引完整资格及全部正点映射；矛盾直接拒绝，不修标签。合法不完整事件保留点指标并排除完整事件分母。完整事件子集另算命中、sample-point延迟与一对一贪心匹配。结果输出JSON及评价坐标；原undefined NaN在JSON中编码为null，Affiliation状态明确未评。原legacy评价CSV及其NA约定不被覆盖。

## 共享Python接口与来源

```python
from hybrid_tsad.engine.source import draw_source_plan, calibrate_source
from hybrid_tsad.engine.bank import collect_forecasts, write_bank, recovered_metadata
from hybrid_tsad.engine.evaluation import evaluate_fixed_scores, paired_block_differences
```

源计划保持原400/5/64、1000次质量拒绝上限、seed偏移0/50000/60000及uint16/uint32/uint8 trace。正式默认曝光仍640000/256/256。`role_sizes`仅让合成测试缩短样本，未被CLI暴露为训练预算参数。helper尚未替换原冻结prepare，不能据抽取存在就宣称所有训练已调用新引擎。

`events.py`三个函数和`statistics.py`两个函数从项目自有脚本原样抽取；函数AST及源文件SHA映射在`docs/provenance/engine_extractions.json`。改签名/拆解的source/cal/forecast参考摘录及原文件SHA保存在`tests/fixtures/legacy_engine_reference.json`；对照测试直接执行冻结摘录，不把新实现自我比较当等价证据。测试只将原正式样本数缩为合成规模、把CUDA设备/RNG/synchronize接口替为CPU对应操作，不改采样、残差、排序或时间足迹。

|共享职责|原项目代码位置（均在外置scripts）|
|---|---|
|source.draw_source_plan|hybrid_10k_wave.py:42起prepare角色循环|
|source.calibrate_source|hybrid_10k_wave.py:120起train源cal语句|
|bank.collect_forecasts|run_hybrid_10k_psm_inference.py:40–78，v2逐块循环/银行组装|
|bank.inference_metadata|run_hybrid_10k_psm_inference.py:79起readout_job修复版|
|evaluation.evaluate_fixed_scores|evaluate_hybrid_10k_psm_readout.py:101起固定阈值点/事件计算|
|evaluation.paired_block_differences|evaluate_hybrid_10k_psm_readout.py:153起固定block抽样/加权AP|

## 实测与未迁移

来源及结果记录见`docs/provenance/shared_engine_validation.json`。本阶段新增测试包括源计划/RNG/拒绝计数/dtype精确匹配、源cal数组和尺度精确匹配、原forecast循环全部银行数组/逐块RNG精确匹配、事件/加权AP helper AST身份、成对block重采样与sklearn显式重复样本AP核对，以及真正配置→小合成模型推理→银行和配置→源cal输出。全部是工程自检。

尚未迁移：包原生优化/源验证/学习率循环、通用prepare/实体准入和计划快照归档、11bank齐备及恢复放行/标签开锁的科学门禁、完整多seed多方法汇总/CSV与历史负对照合并、外部Affiliation实现与31块正式合同装配、U/F适配。共享`paired_block_differences`是可复用计算接口，未自动运行正式31块/2000次研究统计。`recovered_metadata`只生成缺项显式null的元数据，不能自动认证银行或恢复作业。下一工程阶段应围绕这些具体剩余职责评审，不重跑既有模型。

独立审查修订：多真实块Aff只保留分块值，整体为NA并说明MULTIPLE_REAL_BLOCKS_NO_GLOBAL_AGGREGATION，禁止取最后块/自造平均；无GT块不调用官方callback并记录NO_GROUND_TRUTH_EVENTS。仅合法单真实有效块可使用其整体Aff值。原三项失败证据、修复diff与新增拒绝测试见`docs/provenance/shared_engine_revision.md`。旧run及其批准文件不补改为schema2；train新绑定用于未来合法新run，不能为迁移重写历史审批。

二值连续事件单项回审修订：原索引相邻且均为阳性的评分点不得跨coverage owner；错误拆ID拒绝，不自动合并或改标签。真实正常分隔仍通过，索引缺口不推测。评分域外原边界仍须上层冻结完整标签/原事件清单SHA；局部validator不构成全局认证。新diff和回归见`docs/provenance/adjacent_event_revision.md`，待独立回审。

后续native阶段：原优化/source evaluate循环现已迁入候选，见`docs/native_training.md`；上面的“尚未迁移”保留作为上一工程阶段历史范围。当前仍缺完整prepare/作业恢复/科研门禁，不将native候选写成独审PASS。
