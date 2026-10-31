# 原生源训练与受控历史合同

本阶段基线为主记录的`f7858a4`。新的`train`入口调用包内`engine.training.run_native_training`，不再转交一个外置实验driver。legacy入口继续用于历史追溯；旧run、脚本、权重、审批及既有已审证据不改。本阶段只有合成CPU工程更新，没有新科学训练、真实query、GPU或新增模型。

## 配置与执行范围

```powershell
python -m hybrid_tsad config validate configs/train.example.json
python -m hybrid_tsad train --config configs/train.example.json
python -m hybrid_tsad infer --config configs/infer.example.json
```

默认仅预览。native train使用engine schema3，要求外部合同/批准、固定SHA的wave_config、source_inputs、source_manifest、compact plan、rows、官方文件、entry_gate及全新输出目录。参数只有job和device，不能从CLI缩减步数或改LR；所有输入、参数、score-space及输出仍绑定规范operation manifest。示例仍是占位，不是合法新实验合同。真实执行需独立验收、具体科学合同与资源批准，存在入口不代表授权开训。

wave_config必须维持10000步、batch64、lr3e-4、weight_decay.01、warmup500、每500步source select及query_evaluation=false。原step25曲线、step500选模、patience warning、step10000 primary不改。源域只接当前core数据访问器支持的PSM/SMD_SUBSET/TEP，target从source排除；不借此宣称MSL源实体已准入。

执行前核原source_inputs全部hash并要求覆盖实际raw source/TEP cache；rows必须逐条等于所选source在原manifest的顺序及内容；fit/select/cal exposure=640000/256/256，trace三列dtype、角色、channel、完整历史/未来足迹及64窗源域轮转都核验。entry_gate和源/目标角色的科学有效性仍由上层冻结证据与审查负责，PASS字符串不是独立科研认证。

## 100/400历史与权重身份

仅把源计划、batch、source evaluate和forecast银行中的历史长度提升为合同字段，允许100或400，horizon仍5。400默认路径与原循环比较；100是原模型本已支持的输入历史，不修改core。source rows、wave_config、query、源cal及checkpoint metadata必须对齐，不能拿400模型/校准换名当100。

新infer schema3要求parameters.history及固定SHA checkpoint_metadata，metadata包括checkpoint_sha256/history/horizon/mode/copies/temporal_depth；新native训练按实际history产生它。源cal v2也显式带history。旧infer schema2仅保留原400兼容，旧cal v1只能400；不会把旧产物补标为100。API加载非400 checkpoint同样要求metadata；`predictor_from_config`/`create_predictor`接受checkpoint_metadata参数。

权重张量键/形状本身无法判别100/400训练来源；验证的是已批准元数据与其固定权重SHA一致，而非对权重进行训练历史取证。上层仍须从冻结训练配置/源码/原run证明metadata真实。当前只交接口及合成资格，不自动为TEP生成科学划分、budget或新fold配置。

## 数值与产物

训练仍随机初始化原HybridPredictor，不warm-start；仅requires_grad参数进入AdamW，LR/Huber与辅助重建/clip1不变。source evaluate用原mask_seed，在保存/恢复CPU及所用CUDA RNG的作用域内执行；验证不能消耗后续训练随机流。

原产物保留：best.pt、step10000.pt、optimizer_step10000.pt（仍仅optimizer+step）、curve、select_stepN、select/cal_primary、cal_scores、summary。另附checkpoint_metadata、带history的源cal身份和本包源码只读快照/代码hash；外部官方源码不复制。时间/内存只记录实际测量，CPU无GPU peak写null；源码变化会拒绝发布正常summary。

旧optimizer checkpoint缺少模型/RNG，不能凭它声称支持中途精确续训。`capture_step_state/restore_step_state`为共享kernel的独立工程sidecar API，携带模型、AdamW、CPU/设备/NumPy RNG及history/mode；合成测试验证恢复后下一步精确相同。正式CLI的作业级resume/历史best与曲线恢复尚未迁入，不能把kernel roundtrip说成完整作业恢复。

## 对照、证据与未迁移

来源是项目自有冻结`hybrid_10k_wave.py`，SHA00ed75a…；原batch/evaluate/train函数完整摘录在`tests/fixtures/native_train_reference.json`，不是新实现的自我比较。对照直接执行旧函数与新真实函数，仅工程fixture缩短10000循环为500及将旧CUDA设备/RNG作用域替为CPU。500覆盖原第一次scheduled select/best/checkpoint，而不是把验证间隔改短。fixture保留原文件名，但临时目录内的step10000名称/原summary常量不代表完成正式10k训练，也不进入研究结果表。

同初始化/trace下，500步loss/梯度裁剪/LR、曲线（除实际时间）、active/inactive参数、AdamW状态、best/primary checkpoint、select/cal与cal_scores、后续CPU/NumPy RNG逐值一致；另核step1/499/500/501/10000 LR边界、batch内容及sidecar恢复。history100用原core做3次合成更新（含一次恢复重放），验证形状、输入足迹、未来target扰动不进history、银行索引和RNG恢复。未做CUDA数值等价、完整10k工程执行或真实域效果验证。

具体结果与源码身份见`docs/provenance/native_training_validation.json`。先交独审，不能以开发测试PASS代替独立验收。未迁入：完整prepare/实体准入、作业级恢复、all-bank科学门禁、U/F适配、多域正式实验装配。TEP实际合同字段未由此工程自行决定；history100接口可用不等于该fold已正式准入。

正式输出在初始化前保存冻结输入JSON/合同/批准与本包源码快照，并记录全部外置输入路径/SHA；结束前复核这些输入、原raw source hash和代码身份。原数据/官方源码仍外置。最后IO记录完善后额外6项native CPU回归通过3.279秒；全套43项CPU回归通过5.100秒，三核心与原400算法未改，结果仍待独审。
