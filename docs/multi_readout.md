# 多实体固定Z读出：A/B工程交付

A已交候选独审；B接续官方Aff与跨run统计。未执行TEP数值/标签读出、训练或GPU计算。例子中的路径/SHA是占位，synthetic批准不是科学release。

后续验收：A三类源/物理目标/单位绑定反例及cold CUDA初始化修复均通过独立有限工程回审（9项最终CPU回归、三模式真实core正确路径），主已接受A范围。实际GPU冷启动和TEP两块读出仍须具体合同与主release；B评价/统计尚未验收，不把A通过当作检测结果或整套流程完成。

统一入口：`python -m hybrid_tsad readout --config configs/readout_infer.example.json`，默认只预览schema4规范operation manifest及SHA。执行必须`--execute`、合同/批准/全部输入SHA与operation manifest一致，新输出目录不可覆盖。infer/seal/controls的输入字段明确禁止label文件；evaluate还需显式`--allow-labels`，完整组缺项仍拒绝。所有原始run/implementation不改。

阶段操作：readout-controls从固定source artifacts构建catalog；readout-infer按明确seed/mode/entity/origins白名单产独立shard；readout-seal先核完整native组，再归档全部固定native/late/simple评分；readout-evaluate只有全部封存后才有读label资格。源cal的scale/99分位与checkpoint history100、mode/seed、TEP排除和源op身份关联。native仍调用原collect_forecasts，B1/C52/100→5、每实体84551重置、首两块重放、原float32/原索引不改。

生产profile固定TEP100：seed0/1/2×temporal/tf/hybrid×fault1..20×run31..50，共3600物理预测/RNG银行；6方法×3seed×400实体共7200连续评分。完整期望笛卡尔集合逐键比较，不只数文件。L0仅seed0首实体前2块，或正式分片白名单，状态明确不完整；L0/缺bank/重复分片/损坏输出均不能开标签。没有全组group_seal.json时，部分写出不是全组成功。可用新的分片输出目录交接已完成银行，原银行不覆盖。

Source definition需含official固定SHA和每个native模型的checkpoint、checkpoint_metadata、source cal预测/分数/参数record，以及每seed的source_inputs/manifest/rows/plan。完整9权重catalog生成18项source controls：late固定.5两个native标准分，组合99分位来自同索引source cal；simple沿既有readout_hybrid_controls.forward的torch float32 mean/var(unbiased=False)+1e-5/末值公式，以同source cal索引重新校准，query亦用同CPU canonical后端。旧PSM legacy排序/阈值不复用。simple遇非有限history明确阻断，不删样本、不暗换插补。

Synthetic profile只用于明确标SYNTHETIC的测试元数据，不能把真实TEP manifest声明成缩小测试组；仍保留3seed×3mode和全部6方法。A回归用2fault×2run、52通道：36银行与72评分全封存但不读标签，缺1项/重复shard/L0/改输出hash均拒绝；source控制catalog实际生成并复算native cal数组。远端既有环境CUDA不可见、线程2，5项通过1.355秒，非独立科学验收。A代码hash见provenance/multi_readout_a_validation.json。

Aff依赖和统计由B新增共享模块接续；A本身不提供虚假的全组检测结论或整体Aff。不能将通过此门禁说成证明source角色、训练历史或故障标签真实性；它们仍需上层冻结原件/合同/独审。当前未迁入作业级自动恢复，失败银行可保留供独立取证，不默跑补偿推理。
