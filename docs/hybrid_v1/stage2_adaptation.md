# 完整V1 Stage2 U/F与污染：最小协议及资产边界

## 最新执行进展（2026-11-03）

下文保留的是先于运行制定的协议和当时资产状态，不能以其中“尚未准备”重启或阻断已完成阶段。完整 V1/source2025 在独立 PSM 原 train 上的 U200 已实际完成：200更新、0skip、4次固定32点验证，最早最小值在attempt199。独审及root验收包括268项state、207冻结参数/缓冲区逐字节不变、61个允许训练参数和四块真正更新，原source不改。U只更新fusion、tf_fusion_delta、ms_decoder、forecast_head；并非重新训练全部骨干。

零更新的Z-Tcal与U-Tcal已在相同512个独立目标CAL点、相同掩码上实际完成并独审/root接受；其q99分别为2.294127729771862和2.453259601732828。校准包括分项尺度和最终阈值，可能改变融合排序；这两个数字不是检测性能。短组三phase合计实际0.041973522161GPUh，不重复按0.35预留计耗。U的全长PSM只读评分已于UTC19:29启动，模型/评分不再拟合，真实结束以外置OS为准；新的query结果尚未评价，不能宣称U收益或污染鲁棒。

F固定SMD2-1前缀已完成原标签/观察有效性审查，K1/K3可行、K5不足。正常匹配、SELECT/CAL还使用额外前缀标签，因此不是“总共K标签”；需完整annotation ledger。F真实资产生成、训练和新query仍未完成。所有污染率实验也仍待执行；公开原train的无标签适配不替代污染验证。主接受及最新资源见上级研究工作区CURRENT_RESEARCH_STATE与notes/hybrid_v1_20261102。

2026-11-03；PREPARATION_ONLY。只核现有代码、split/draw/manifest、旧事件元数据与修复报告；未读当前PSM分数/测试标签、任何新数据数组或标签体，未训练/GPU/下载，未改代码或旧split。机读证据与10项资产状态见`notes/hybrid_v1_20261102/STAGE2_ASSET_AUDIT.json`。本协议不把已有CPU合成Z/U/F接口通过，写成真实Stage2或污染鲁棒已经完成。

## 1. 当前唯一优先：PSM原train上的小预算U

先以固定source seed2025做200attempt的U，回答“目标正常训练用途数据上的小头/融合适配，是否超过只调目标尺度/阈值”，不把它当真实混合污染实验。seed2025不是根据当前AP选取；源checkpoint/模型结构/所有其余超参在label评价之前固定。full当前Z1及全部旧正负结果继续保留。初步只有一个适配seed，不声称三seed稳健。

原始`/home/a312/tsad_research/data/psm_original/train.csv`已有132481×25 metadata与SHA816e0e2a5356b3b09d356925cd9a0fe1d327e3b350e9336d26ee3e650c72e112。当前不能凭“未提供异常标签”推定其真实污染为0，更不能用正常训练用途的结果证明混合健康状态鲁棒。后续准入才重核列/observed/原轴/字节身份，不在此打开body。

拟新版本train-only角色（不是覆盖旧H100/H400划分）：FIT[0,79488)、SELECT[80032,105984)、CAL[106528,132481)，相邻guard544；所有H512/P32训练足迹完整属于本角色，query仍是已定原test[0,87841)，不从train借历史。此切分是已开发公开域上的新探索协议，非历史盲留出。预先固定32个合法唯一SELECT位置、512个合法唯一CAL位置；位置只按合法原轴/observed条件与固定seed选，不按loss、残差、AP或标签选。FIT采样仍可用现有确定性BalancedSampler(seed2025)并封实际轨迹，不能静默重抽补skip；若要预生成精确200 FIT计划，须修现接口限制后单独验收。

固定U配方：从accepted source fresh deepcopy、200attempt、lr1e-4/wd.01/Huber/clip1、validate_every50，4次×32固定SELECT；原mask/观察权限、float32/noAMP、H512/P32和三分项不变。只更新现有`fusion/tf_fusion_delta/ms_decoder/forecast_head`；不新建网络，不让source模型被原地修改。SELECT只看SSL，不看query；保存selected/final/初始/实际更新与所有冻结参数和buffer的差分。正常SELECT损失不是检测成功gate。

必须有三个分列控制：Z-S（复用当前source-only原bank）；Z-Tcal（同一冻结Z模型只在上述目标CAL512点重新定分项尺度/阈值，零参数更新）；U-Tcal（适配模型＋同位置目标CAL）。Z-Tcal可在原Z raw分项bank封存且准入后score-only重算，不重跑Z模型。每模型自己的CAL raw独立算，禁止把未适配source阈值误称适配后正常1%FPR。U-Tcal沿原`fit_calibration(source=False, active_components=...)`、固定recipe与strict>；目标CAL无标签、可能混合，q99仍不保证真实正常1%FPR。Tcal本身是目标接触，不能把Z-Tcal写成纯source校准Z。

## 2. 资产可用性：PSM F与SMD B目前真实阻断

|资产|可核事实|当前可做/不能做|
|---|---|---|
|PSM originaltrain|132481×25缓存身份已有；曾作旧开发FIT|U候选；不是“真实混合污染率已知”的数据；实际H512新manifest/observed计划尚未生成|
|PSM originaltest|当前Z1整个[0,87841)已定QUERY，含暖机输入，并非只有有score的512..87840|不得把首512点或旧support事件转回训练/污染donor|
|旧PSM事件split|旧H400 metadata的support[0,35136)、validation[35136,52704)、query[52704,87841)都在该87841点测试序列；旧34支持/11验证不是今天独立新支持池|旧报告/IDs保留；与当前全test query冲突，F及真实PSM污染不能复用这些正例|
|SMD source-side旧A|machine1-1/1-2/1-3旧support分别8/10/12事件；旧同来源正常匹配已审|可作为新F-A/源污染**候选**，但H100/H400 eligibility、正常窗口和表示bank不是H512/P32现成资产；先重审完整事件/guard/missing|
|SMD target3-1/2/3|旧4/10/26事件全是query，独立目标正例支持为0；现缓存清单无已核这三机器原train|B目标设备F/U仍阻断，不能挪query正例、用源异常＋目标正常冒称B|
|其他SMD机器|本任务已有filename/manifest只证明4个源train、7个旧test|未做V1训练不等于已有可用/盲数据；无额外缓存/边界证据，不下载补齐|
|SMD2-1新时间切分备选|23694×38 test旧整段是validation；其normal train又进入V1 SELECT/CAL|可另立新探索协议，不称未见设备或盲确认。仅建议、未切数据：support[0,9477)、normal-select候选[10021,13031)、normal-cal候选[13575,16585)、new-query候选[17129,23694)，guard544。旧事件metadata在support内仅E0/E1/E2三条满足H/P几何足迹上界；normal纯净、observed和完整guard尚未认证。K5在此固定候选不可用，不改切口凑数；旧split原件不动|
|MSL processed CSV|只有现存126933151-byte缓存，训练stream边界仍PENDING|不得假拼成连续run或直接进严格U/F/源污染|
|Sunspot|原生C1、3240 missing已恢复|missing不是异常真值，不能提供“真实污染标签”|
|TEP旧H100资产|旧正常run500点，旧fault从160起|500<544不能直用本V1训练；H512后的旧faulty query会丢正常前缀。不代表所有新TEP原件不可用，但需另审完整run，本最小波不扩入|
|旧CICIDS/SWAN r0/r20|有限清单只证其pool filename/size|未核H512连续记录、事件隔离和五rate定义，不能冒充当前0/1/5/10/20完整污染资产|

以上“可用”均只指metadata；本任务没有读取这些数组、重切任何旧run或新标签。原TimeRadar八基准不重跑，旧M3/HOLD作为历史边界保留，不据此否定完整V1。

## 3. F的两种问题必须分开

F-A是源设备有K个完整异常事件、目标设备不参与拟合的有标签迁移，可从旧source-side SMD A候选池重建；F-B是目标设备少样本适配，需要目标独立support与query。当前PSM与SMD3目标B池不可用，不能把A改名B。可选SMD2-1新时间协议必须另登记并审核，不能称既有B已就绪。

固定预算K∈{1,3,5}，K0严格回到原Z（无target更新/target CAL），不凑K10。每个合法新pool按固定原event-ID排序再用独立default_rng99175预生成嵌套抽样；抽样和候选过滤不看query分数。实际pool少于K则该K=NA，保存原因与重复组合数，不复制同一事件冒充K个独立事件。旧draw算法/ID追溯可复用，旧draw结果不能跳过H512新eligibility审查。

事件必须与完整原label目录的[start,end)精确相等；所有被标注/暴露的正事件纳入K台账，非只计训练窗口。对每个事件保存至少[start−512,end+31)完整可见足迹；它不得触SELECT/CAL/query或未预算正例。角色之间可用544点统一guard，但最终以每个实际H+P足迹判定，不能只看中心点。正常负例逐事件同来源匹配、等长；首选同dataset/entity/原record的全正常片段，若引用同entity另一个正常train原record须明列来源层级，不能声称时间/工况已匹配。正常SSL的完整[start−512,end+31)也必须不含任一已知异常、不触heldout角色。正常label筛查、NORMAL_SELECT/NORMAL_CAL标签成本和事件区间标注成本额外记，K不是全部标注费用。

从同source checkpoint独立deepcopy后固定300attempt/seed2025，round-robin K个正例及其匹配正常例、BCE＋.1正常SSL；只有所有K正/负都已真实有效更新，checkpoint才有资格按独立NORMAL_SELECT SSL选择。缺测令某事件没有有效监督时保留失败，不静默删事件或换K。F头是既有`event_head`，score为raw event logit，阈值在独立NORMAL_CAL的normal logits上用现有`logit_threshold` q99 higher；不能混称原三残差recipe或继承U分项阈值。边界/label/sample-clock评价沿已接受CPU evaluator，事件K1/3/5和所有失败分列。

## 4. 当前代码支持及小工程缺口（不需新增网络）

- `update_policy/adapt`真实存在：Z全冻结，U上述四prefix，F再加event_head；每条路线deepcopy，F-K0转Z。已有合成集成证明接口可走，不等于本资产已可执行。requires_grad只控制梯度，不能代替对冻结buffer及训练模式状态的实测。
- `fit`真实支持U SSL、F round-robin、BCE+.1额外normal SSL、所有K更新资格、正常SELECT选模；`audit_events`检查K唯一、等长normal映射、区间与expanded footprint隔离。它**不自行证明完整原事件目录、同来源匹配、normal纯净或forbidden列表完整**。目前lookup/overlap只用entity/run，外层须验证domain和原record身份/避免跨域同名；这些是metadata校验补齐，不是新模型。
- U直接传预生成sampling_plan目前被`route != STAGE1`拒绝；CLI `options`没有把validation_plan、validate_every、weight_decay等目标配置传给adapt。最小路线可保持已有Sampler固定seed并封轨迹，显式传32点validation_plan与validate_every50；若要固定FIT计划，窄修允许route U及正确FIT角色，先合成测试，不私改旧source数学。
- 现`cli.execute(adapt)`会先`load_manifest`所有角色，又把adapt/CAL/query连续执行，**不能作为真实200步小适配的安全调试入口**。需共享phase装配分离target-train（只FIT/SELECT或SUPPORT/NORMAL_SELECT）、target-CAL、独立query stream和后置labels CPU，复用现有engine/资源壳，不新建每实验driver/网络。目标source权重、CAL和query签件分别绑定，U路径不能接异常标签文件。
- 真实污染率builder/标签隔离事件目录/曝光ledger目前没有可直接认证的新V1资产；BalancedSampler按domain/run均匀不是指定ρ。Huber/half_mse开关已有，污染验证尚未做，不能把实现Huber写成鲁棒已证实。

## 5. 污染轴：真实事件与合成严格分列

预定ρ∈{0,1,5,10,20%}，先只在**SMD域的固定FIT slots**研究added labeled-anomaly-window exposure：污染slot的target t位于隔离目录中的真实异常事件，原始H512/P32上下文保持同一native38C record；不把事件片段拼成假连续run，也不改label边界凑rate。0是“未额外加入已标事件”的原正常训练用途参考，不保证无内生未知污染。每档另报真实current/future-GT/input-context异常点比例、observed/skip、事件ID/重复窗口与domain/nativeC；名义window比例不冒称原始异常点比例。

两源原8000计划中SMD3982 slots、Sunspot4018，若只污染SMD，整个源的added-window比例约ρ×3982/8000，不能称全源20%。固定domain次序、Sunspot位置、总attempts、SELECT/CAL、seed、init与优化曝光。label只在离线builder证明供体和ρ，给SSL的manifest不带label字段，不能按label过滤困难训练loss。真实事件donor必须与所有query、SELECT/CAL及另作F支持的目录隔离；现有候选可按原record预定machine1-1_test供体、machine1-2/1-3_test F-A候选，**尚未冻结或改旧角色**，还需H/P/normal/observed实际准入。不得使用PSM test或SMD3旧query作donor，也不能把38C事件直接塞入PSM25C。

合成污染使用另manifest、固定操作/幅度/事件种子及原始时间轴，真实/合成表分开；不以circular shift或未来值拷贝制造伪因果样本。合成event真值仅表示指定干预，不证明真实工业/密封故障。对真实event donor不足的rate写NA，不靠合成替换后仍标real。

最小先验比较是0 vs10% added-rate × Huber/half_mse 2×2，full-Huber0只有base输入/原8000轨迹完全相同才复用；另外三格需fresh同初始训练。先单seed2025、仅Z读出，不同时乘U/F/K/all-seed矩阵。剩余1/5/20%仍预登记、状态PENDING；没有全谱就不能声称0–20%稳健。若进一步主张Huber优于MSE，各rate对照曝光与query必须一致，不能挑最好的污染档、只隐藏失败档。source loss不是异常鲁棒结论。

## 6. 最小波次与资源（只是规划）

1. **U-prep/CPU**：补phase权限/plan传递与冻结state对比的共享小工程；只合成CPU验证。目标原train实际载入和新manifest/observed/32SELECT/512CAL生成需root单独授权，不读test labels，不看AP。当前尚不具备真实执行包。
2. **首U波**：只seed2025 U200＋Z-Tcal，已定PSM query87329点完整覆盖；原Z-S及两确定性baseline原bank复用。新U全量query只能另签并封raw，不能取前缀代表完整收益。拟合计≤6.5 GPUh：train≤.25、Z/U两个targetCAL合≤.10、必要固定shape成本≤.05、一次U全query含bank/cleanup≤6.0，余.10；这是待实测修订的管理cap，不是完成保证，不启动或自动重试。无label前rank择优，不增加seed矩阵。
3. **F准入波**：先解决pool隔离、完整目录及正常匹配；PSM F与SMD3 B保持BLOCKED。F-A或新时间协议有合法池时，先seed2025 K1（source train≤.25h＋CAL≤.05h；query按事前完整新fold点数profile，拟总≤2.5h），工程验收后按预登记接K3/K5可用项。不能看K1 AP后决定保留哪个K，也不能通过修改旧query切口凑预算；K5不足保NA。
4. **污染准入波**：先纯metadata/CPU构造隔离ledger，再0/10% 2×2的三项新source训练（full-Huber0满足同输入时复用），按原source壳训练＋CAL拟≤4 GPUh；后续每个完整query读出另profile/另签，不把source-only完成说成鲁棒结果。五rates×三seed×Z/U/F/K的全矩阵不授权。

以上未来cap须核真实剩余账与现300/新增120框架，当前Z1预留、A1预留、source已耗不双计，也不因新文档重置余额。执行阻断、缺资产与NA都保留，不改六项主线，不承诺已经实现真实混合污染鲁棒或目标few-shot适配。


## 7. 分phase工程准备接续（2026-11-03，未执行）

共享新增`hybrid_tsad.hybrid_v1.stage2_adaptation`及10项合成CPU用例已准备，详见`notes/hybrid_v1_20261102/STAGE2_U_PHASE_CPU_PREPARATION.md`。仅target_train/target_calibrate，复用原Sampler/adapt/iter_infer/calibrate_route，未改fit数学。原资产JSON不改，真实manifest/observed/32SELECT/512CAL仍阻断；本段不是CPU通过、真实数据或GPU许可。当前source专用readout绑定不接受U receipt/targetCAL，后续需显式U身份窄接线与新freeze，本阶段不实现query。正式Z inference_seed为各2025/2026/2027。
