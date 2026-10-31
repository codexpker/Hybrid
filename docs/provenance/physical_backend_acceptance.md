# 物理校验后端：主验收

2026-11-01。主已读取冻结源码/diff、19项字节身份及独立CPU/GPU/标签前拒绝原始报告，接受有限工程PASS。原infer、model/data/protocol、银行、source cal、native/simple/late评分表达式和2e-6容差不变。

独立报告在研究工作区`independent_audit/HYBRID-TEP100-BACKEND-REVISION`。最终verdict SHA cbe271697dc1870b10d0dc01c7863f634812f9640df694f4b1d51acb27dd00ba；REVIEW SHA e35dd48de1c7b3cfd6ae8eb408ca61b0c0c4f98164bbf0bfdc63198e30e1442c。

已核：13个独立CPU拒绝反例、22项既有/新增回归、72行完整B合成路径与官方Aff/2000run抽样、5个实际评价入口的标签前拒绝；原故障实体172窗CUDA参考与9个原银行逐位一致，常值/zero/ramp控制、篡改拒绝及CPU/CUDA RNG不变。GPU完整子进程含启动导入退出2.617563秒，约0.000727101 GPU小时。没有网络构造/forward、训练、真实标签或生产seal。

本验收不代表全部400实体已经通过；全量验证/固定评分封存和标签评价仍分别需要主具体release。原CPU参考失败与历史资产保留。当前原float32在一个高基线常值实体的尺度相对float64偏差最高约1.185%，检测影响未知，不改normalizer或分数掩盖。历史job-time GPU UUID/model/Torch版本未录，不回填；新验证只证明当前固定A40后端对保存数值的可复核性。职责独立属于同模型家族共享工作区，不是外部双盲。

[修订来源与CPU记录](physical_backend_revision.json)及[验证后端约束](../physical_validation_backend.md)保留具体范围；开发报告中的待审字样为原时点记录，不倒填。科研HOLD，收益和创新尚未获本轮真实指标支持。