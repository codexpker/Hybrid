# I2 共享训练与评分工具

## 正式入口索引（2026-11-03）

|用途|共享模块|边界|
|---|---|---|
|源训练/独立源校准|`hybrid_tsad.hybrid_v1.cli` 的具名 source phase|固定角色与计划，各phase单独权重/配置/原件；不是重新启动官方八基准|
|目标无标签训练/目标校准|`hybrid_tsad.hybrid_v1.stage2_adaptation`|target_train只FIT/SELECT；target_calibrate只CAL；U与零更新Z-Tcal分列|
|完整模型或具名源控制/U只读推理|`hybrid_tsad.hybrid_v1.readout`|原完整模型严格路由与新的SOURCE_CONTROL/U_TCAL路由分开；具名实际权重/CAL/receipt身份，不改原模型|
|原3seed＋2基线评价|`hybrid_tsad.hybrid_v1.evaluate_cli`|保留原固定五银行协议，不把不同控制当seed|
|同seed不同控制评价|`hybrid_tsad.hybrid_v1.named_evaluate_cli`|首六方法source profile；原指标/事件/Aff实现复用，全部银行先过验再读标签；generic CI仍NA，target使用下方独立入口|
|原始银行重新校准|`hybrid_tsad.hybrid_v1.bank_recalibration`|新标量覆盖层引用原producer与位置；不改原预测/残差/RNG、不跑模型、不拟合阈值|
|区分仅校准与权重适配|`hybrid_tsad.hybrid_v1.target_evaluate_cli`|固定Z-S/Z-Tcal/U-Tcal单seed三方法，全部原银行与CAL核验后才读取标签；不造三seed均值|
|事件监督训练/独立logit校准|`hybrid_tsad.hybrid_v1.event_adaptation`|独立F operation；显式支持事件/匹配正常、32SELECT及512CAL计划；不借U或PSM只读路由冒F query|

入口默认预览，真实执行需具名配置及许可；历史快照与产物外置、冻结不覆盖。`configs/hybrid_v1/evaluate_named_source.PENDING.json` 是尚未绑定真实银行的模板，不能拿null身份运行正式评价。具名评价的28项合成/回归测试已实际通过并独立复核；这不是当前四个真实query银行已经完成或已计算指标的声明。最新状态和精确运行命令以研究工作区的CURRENT_RESEARCH_STATE及各run配置为准。

目标重新校准的10项新测试及上述28项回归共38项真实通过；Z-Tcal已实际完成87329点标量覆盖层并验收，不代表U标签评价已运行。F独立入口先通过9项NumPy/mock测试，再通过真实Torch小C3/H128/P4集成：K1两次、K3六次更新、SELECT2/CAL3，保存权重/优化器/RNG/原分数并独立复核。它不等于真实C38/H512、32/512计划或F检测收益已验收。F入口不带`--execute`时做元数据预检查，可能读取文件身份和SHA，但不加载数组或构造模型；正式执行还需要外层资源壳的具名签件和构造前显存上限。资源mock测试不冒实际GPU验证。

统一入口 `python -m hybrid_tsad.hybrid_v1.cli --config <json> --phase pretrain|adapt|infer|evaluate|synthetic_smoke --output-dir <新目录>` 默认仅预览，显式 `--execute` 才执行。示例 `configs/hybrid-v1-synthetic.example.json` 需绑定已有官方源路径；本次只准 CPU 合成运行，公开数据/GPU训练另放行。

公开数据通过 `manifest.runs` 列表：每条显式 domain/entity/run/role、path、format(npy/npz/csv)、bounds半开原行区间；npz指定key，csv指定columns，original_start可记原坐标起点。自然missing由finite值掩码保留；不把占位零作为GT。支持 FIT/SELECT/CAL/QUERY/SUPPORT/NORMAL_SELECT/NORMAL_CAL，同run角色不重叠。训练完整H+P足迹必须在role内；推理采用独立H+1已到达窗口，覆盖最后P−1个可评分点，不读取未来占位值。SSL manifest不接受labels/label_path，只有独立evaluate入口读标签。

Stage1按domain、run、合法窗口三级均衡采样，保留完整变量维C。half_mse与Huberδ1均逐元素算；重建GT必须observed且人工hidden，预测GT必须observed，返回每窗loss/分项/有效计数与自然missing/人工hidden计数。MS各尺度独立GT误差并等尺度加权。评分保存原预测、r_tf/r_ms/r_var/r_pred、每通道及签发/可用索引；末点只取其own-hidden组。

源CAL每domain总权重相等，各分项中位尺度下限1e-6，log1p后固定TF.25/MS.25/pred.5，阈值取加权经验CDF首次≥.99的值、strict>。某必要分项缺失时主score缺失，不补0或临时重权。Z精确复用源校准且零更新/零目标fit；U只更新fusion/tf_fusion_delta/ms_decoder/forecast_head，独立无标签SELECT及混合CAL；F另更新event_head，真实事件BCE+.1匹配正常SSL、独立正常SELECT/CAL原logit q99，K0走Z不更新。支持event_id、匹配正常长度、全部K预算与留出footprint审计存入日志。

`active_components`是事前固定配方：完整r_tf/r_ms/r_pred=.25/.25/.5；去TF为MS/pred=.5/.5，去MS为TF/pred=.5/.5，rec-only两路各.5，pred-only=1。禁用路不进入loss有效窗口判断，运行时缺少任何静态必需路仍NA，不能逐点重权。该静态配方随校准保存，并与core禁用开关核对。源某domain全无有效支持明确报错，不静默丢域。F残差在减法/除法/平方前掩掉自然缺失目标，端点特征按有效channel池化，避免NaN反传。

pretrain保存source.pt/训练trace/源calibration；adapt读同源checkpoint独立深拷贝，不串联U/F状态，保存各路径权重、参数变化、分项loss、原索引及score JSON。配置需给manifest/checkpoint/source_calibration、route及F support.events/normal/forbidden/K。infer读取固定checkpoint/calibration；evaluate复用既有评价模块、逐run标签原坐标映射，不重写指标库。所有缺失数值落盘为null，保留valid字段。

有限修复v4：F仅在全部K个异常事件和全部K个匹配正常片段都获得有效监督更新后，才允许进入正常SELECT的checkpoint比较；记录selected_attempt和所选状态的累计曝光集合。所有支持及额外normalSSL完整足迹都检查forbidden，F日志保留BCE、normalSSL分项和足迹。推理各入口显式传配置seed，每点保存actual_seed、分组/副本mask（bitpack）、order、visible统计、当前GT/observed、TF及逐尺度MS的组/副本末点预测、输入数组身份和时间足迹。源及目标CAL原始records一起保存，允许独立复算尺度、阈值和分数。正式大序列需要评估分块数组归档成本，小合成JSON并非无成本的全量生产格式。

synthetic_smoke在3个domain、C=1/2/3、独立run/role及1个显式synthetic异常事件上运行同一pretrain→保存/加载→独立Z/U/F→分项与原预测落盘链。STATUS明确synthetic non-scientific-results，不能用其分数作为公开数据收益、鲁棒或泛化证据。真实核心/官方外置源的CPU集成由Executor在独立快照运行，600秒、RSS4GiB、CUDA_VISIBLE_DEVICES空；本机无Torch时不冒称通过。

默认完整配置的GPU资源诊断另有 `--phase synthetic_gpu_l0`，只包装同一核心/损失，需要精确配置与root release，不能用它启动pretrain/adapt等其他阶段。2026-11-02单卡A40的C3两次更新、C55一次更新和两组推理已完成并有限独审：3,458,561参数、整进程5.626秒、最大allocator allocated约1.087GiB/reserved约1.227GiB。原始输入、预测和损失银行外置归档；未保存可用权重，不能将该短诊断当公开数据训练、收敛或异常检测收益。分阶段报告以上一级CURRENT_RESEARCH_STATE及primary acceptance为准。
