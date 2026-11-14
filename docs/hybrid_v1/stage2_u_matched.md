# 独立 matched-U 工程接口

`hybrid_tsad.hybrid_v1.stage2_u_matched` 复用原模型、two-pass/objective、UPDATE_U、残差和校准；不改变旧 training.py、stage2_adaptation.py 或入口。新策略是冻结子模块eval、许可四块train，每步重新施加；根model保train以保留显式历史mask。SELECT使用全部实际设备的fork RNG和逐模块模式恢复，只作诊断；主权重是200次attempt终点，不回载最佳SELECT。skip仍消耗一次plan与mask，不补抽。仅实现plain Huber；robust字符串显式拒绝，未添加加权公式。

角色manifest每entry精确字段：domain/entity/run/role/path/format/key/observed_key/sha256/expected_shape/bounds/original_start。NPZ只包含float32 values及bool observed，bounds=[0,n]，original_start保原记录轴。不同role不可同文件或hardlink；同domain/entity/run的跨role必须隔H+P，不可通过重新命名run伪造独立记录。load_roles只hash/打开请求role，其余文件仅作metadata/stat身份检查；不把整个共享NPZ先解码再slice。独立审计ledger不得随learner manifest/plan进入。

共同plan顶层history/horizon/seed与caller一致；FIT200/SELECT32/CAL512，entry只含domain/entity/run/role/index(global)/可选footprint(global)，不含label、eventID、slotkind或权重。FIT允许重复合法点；SELECT/CAL唯一。同计划接口为未来plain/单因素robust匹配而保留，当前没有robust实现。CAL只加载CAL，arrived H+1，保末P-1，不要求未来标签。当前新CLI只有train_endpoint/calibrate，无QUERY/评价/F入口。

CLI默认预览且需精确root config/release及显式source map，非工作树glob：

```text
python -B -m hybrid_tsad.hybrid_v1.stage2_u_matched --config CFG --release REL --trusted-release-sha256 SHA
```

`--execute`须后续具名科学与资源批准。本轮仅新11项增量CPU工程门，tiny H128/P2/C3、合成数组/随机模型、不加载公共权重，CUDA仅受控stub逻辑；不代表CUDA RNG实测、正式H512/200适配或检测收益。真实CUDA另需获批L0。输入供体是否合格及完整事件数由独立数据审计处理，不由此训练器选择。已关闭reference候选不进入此接口。

后到非数值补充：返回 `scientific_training_status` 明示零更新/部分更新/全部更新，零更新不能称有效U。CLI每attempt开始/完成逐行flush到training_trace.jsonl，异常留下FAILURE.json且无完成receipt；不自动恢复或补抽。该补充在原CPU10项通过+一个fixture恢复之后新增，仅需后到NumPy/mock回归，不称最终新版11项全跑。

独立SMD读出入口 `stage2_u_readout` 只复用原readout.write_model_bank/iter_infer/scoring/ChunkedBank。新typed rootseal绑定父source、own训练config/receipt/endpoint、own targetCAL config/receipt、manifest/plan/seed、显式source闭包；旧PSM reader不改。ZS_ZT只能一份同权重canonical raw，Z-T仅离线重算另一CAL视图，不重复forward；U用own adapted/CAL。QUERY独立文件、原run/轴不改、512热身及末31全保；无标签/指标API。当前只metadata/NumPy假体，不代表真实SMD读出已执行或资源已获批。
