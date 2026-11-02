# 候选 v2：原窗口分组适配

v1四模式GPU合成L0原件已封存；本修订不回写v1模型/测试/结果。官方TimeRadar和Autoformer文件不改，结构、head、预测目标、loss与score公式不变。

实际源码风险：TimeRadar `Model.forward` 的order由调用的B*C平均决定；Autoformer训练top-k由调用batch平均决定。若把不同时点/记录的窗口混进这些共享调用，一个窗口可能间接使用另一窗口的较晚观测。固定evalB1不足以关闭训练接口的该依赖。

本项目wrapper v2对每个原窗口分别调用作者TimeRadar（其全部C通道保持一起）；Autoformer self/cross也先按原窗口分组，再在窗口内部展开通道/有向对。窗口内作者共享order/top-k及train/eval原行为保留，窗口外不共享。最后stack回`[B,C,d]`，统一融合/预测头不变。它不是修改原作者相关公式或新增模型机制。

显式代价：顺序调用数随原B增长，跨batch吞吐可能下降；不能使用v1B2 GPU成本冒称v2已测成本。原始C²pair成本在单窗口内仍存在，不静默chunk原作者选择集合。多源训练只允许参数梯度按普通minibatch聚合，不允许其它窗口的观测进入当前forecast特征。

新增CPU合成测试针对四mode、train/eval及受保护的batch例0和1：固定同一RNG重放，任意改变另一个history的非仿射数值，要求受保护forecast逐位不变；这是窗口间输入独立性验证，不是固定不同mask后误判的数值差，也不是物理因果/检测性能认证。

旧`candidate_tf_lag.md`描述的是已封存v1来源和首轮实现；当前可执行v2的分组规则以本文及新源码/冻结快照为准。实际共享模块和源身份每阶段都单独绑定，不用同名动态文件代替历史运行版本。
