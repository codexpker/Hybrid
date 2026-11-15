# G36既有线性动态权重：正常NLL与持续偏离LR固定评价

用户新明确授权，独立于此前full正常泛化失败的停止记录。本轮不训练、不重新选择权重、不重估FIT偏离先验，不修改此前full停止gate。

- 固定权重：`hybrid_g36_persistent_lr_20261116_v1/models/linear/train/selected.pth`，selected step1200，SHA256 `8ac0cb804d9640feaf2688d09af9010b6abb5879bd999eede20031e686f40665`。
- 固定FIT prior：同run `models/linear/assess/FIT_PRIOR.npz`，SHA256 `7405c1bd13ddbbf5d056f7edbc5648162833c60b1f2c741c2111da2e3b92d964`。只加载，不读取FIT重新估计。
- 固定原config/scaler/data manifest及原G36全轴，52点可见输入，响应4:52/条件0:48。identity表示、8维线性状态动态及共同GRU条件编码；不是完全线性ARX，也不是新Hybrid原创机制。
- 同一轮冻结前向同时归档H0/H1 logp、LR=`logp_H1-logp_H0`、正常NLL=`-logp_H0/192`及原时间轴/预测/innovation。identity映射Jac=0。不搜索窗口、先验倍数、方向、起点或读出。
- 两个方法各用同一合法正常CAL237点的自身q99/higher阈值，strict>；原C3第三点开始，不回填。不得使用QUERY选阈值或重新选checkpoint。
- 完整4 QUERY轴各7954点。原LEFT、条件/历史Ridge银行直接复用，无重新训练/推理。每故障与正常baseline的配对AP/ROC及三故障宏平均，合并指标正常baseline只计一次。
- 健康副本只在分数封存后用于故障独有/共同报警诊断；沿原52点模式历史一致的稳态mask，所有模式不一致区间仍保留单列，不删不利点。
- 报告正常误报点/片段、切换误报、事件命中和一对一P/R/F1、持续覆盖/故障独有覆盖、首次延迟及最长漏检。CAL一天237重叠点不能认证总体1%误报。

仅新增狭窄的frozen-linear读出授权路径，校验原PT/prior/receipt/config身份；旧full `NORMAL_GATE_FAILED_STOP`原件及其入口拒绝行为保留。模型数学、原训练、旧结果不变，原权重/先验在运行前后核SHA。

若该固定判据没有可接受误报下的实际增量，停止，不调先验/阈值或扩展变体。即便有增量，也只能算成熟动态模型/似然判据的有限机制证据，不能直接宣称Hybrid时频创新、真实独立确认或U/F与污染鲁棒贡献。
