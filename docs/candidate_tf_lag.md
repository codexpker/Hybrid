# 时频与时滞端点预测候选（科学实现 v2）

新实现 `candidate_tf_lag.py` + `candidate_autoformer_lag.py`，与旧 `model.py` 和历史400/100/P5训练/结果分离。它是可训练的候选组合，不是已证明创新、完整TimeRadar/CDL复现或已证明微弱异常有效的模型。

2026-11-02 状态：共享 loader/训练/推理/评价入口已实现，PSM 四模式各200步训练及固定128窗口无标签成本测试已实际完成；全量 CAL/TEST 推理已签放行，尚不能据此声称全量分数或标签评价完成。使用与结果导航见 [candidate_usage.md](candidate_usage.md)，角色与执行规则见 [candidate_development.md](candidate_development.md)。

## 唯一实现与来源

|部分|直接复用/本项目适配|实际shape|
|---|---|---|
|TimeRadar时频|外置官方commit `4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2` 的 `models/TimeRadar.py:Model.forward`；SHA `881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317`|history `[B,H,C]` 按原窗口逐次调用，每次 `[1,H,C]` → complex repr `[copies*C,H/patch,Dtf]`；原decoder也被作者forward执行，但在预测目标中不使用，参数冻结并另计stored容量。|
|原生域端点摘要|本项目将repr按mask copies与FRFT-domain patches池化，拼实虚、线性投影|`[B,C,2Dtf]→[B,C,d]`；不是原时间token，不与原时间位置逐点相加。|
|Autoformer时延|外置作者commit `51c7d416ae120b805fd5beef2f4ccf7de496a6ff` 的 `layers/AutoCorrelation.py:AutoCorrelation/AutoCorrelationLayer`，MIT；SHA `dddbc3cc533f9640bbee0e624a49532efe4ee99735101f51a4b03e105772af01`。实际调用原FFT相关、top-k与循环roll/gather，不自行重写attention|按原窗口 B1 调用：每点scalar线性嵌入→`[1,C,H,d]`；self展开`C`，cross有向(i,j),i≠j展开`C*(C−1)`；最后拼回原B。|
|自身/跨通道端点|本项目pair枚举、source聚合和线性merge。self Q=K=V同通道；cross Q目标通道/K,V来源通道；两者均取末token，cross只对source维均值|两支各`[B,C,d]`→合并`[B,C,d]`；不能全时间均值，因为循环时延聚合的全时间均值不识别shift。|
|融合与统一头|本项目线性concat或sigmoid逐维gate，统一future预测头|endpoint `[B,C,d]→prediction[B,P,C]`；首候选H100/P1、d64。|

时延单位为原采样点（每点一个token，未patch lag分支）；native delay包含0与周期回绕，并非已认证的单向物理lag。科学 v2 将 Autoformer 调用隔离到同一个原始历史窗口：训练时 self 的 top-k 在该窗口 C 个通道之间共享，cross 的 top-k 在该窗口 C*(C−1) 个有向通道对之间共享；eval 仍按作者逻辑逐展开通道/通道对选择。不同原窗口之间不共享 top-k。它不是作者现成的跨变量异常检测器，适配贡献需之后实验验证。原mask_flag/attn_mask不执行逐token因果mask；整个prefix均≤首target−1，故预测目标仍在所有输入之外，不声称prefix内部历史点在线可得。

TimeRadar 也逐原窗口 B1 调用，其 FRFT order 只在该窗口 C 个通道之间平均共享，不跨原始 B 共享。这样避免另一训练窗口中时间更晚的观测间接进入当前窗口的表示选择；同窗跨变量历史关系仍保留。互补随机mask用偶数copies、H整除patch，避开原random分支未定义变量/非整除reshape路径。原件不改；本项目每窗归一化只拟合当次历史，target不进入mean/scale/编码。TF的表示pool/目标是新预测任务，不能把`tf`组叫官方zero-shot成绩。

## 四组共同合同与API

|mode|分支|融合|
|---|---|---|
|`tf`|仅官方时频summary|不额外融合|
|`lag`|作者算子的self/cross适配|不含TF|
|`concat`|TF+相同lag|concat→linear|
|`gated`|TF+相同lag|g·TF+(1−g)·lag|

lag内部另有self/cross开关，不在本次自行扩训练矩阵。四组必须同H/P、源与目标索引、归一化、曝光和损失；总参数和active参数分别报告，容量控制之后登记。TF mask RNG/Autoformer train/eval差异须冻结，不按query挑seed或mask。

```python
from hybrid_tsad.candidate_tf_lag import candidate_from_config, candidate_objective, residual_scores
model = candidate_from_config('configs/candidate-tf-lag.local.json', mode='gated')
output = model(history)               # history only, no target argument
loss = candidate_objective(output, future)  # MSE normalized by history-only scale
loss.backward()                       # caller supplies approved optimizer/recipe
score = residual_scores(output, future)['continuous']  # [B,P], no labels/thresholds
```

四组统一预测标准化MSE；分数为逐通道标准化future平方残差的均值，保留per_channel。`forecast_coordinates`明确forecast输入在target前截止，异常残差须等对应target实际到达才可用；不回填过去、不把预测误差下降当AD收益。官方重建/variance若对比须另轨原公式，不在新分数偷偷混权。

模型参数与channel名称无绑定，支持不同C的多源调用；cross总成本O(B·C²·H·d)，但每个作者算子调用仅处理一个原窗口的 C*(C−1) 对。`max_expanded_pairs`在每窗检查，超限拒绝，不静默切同窗pair改变共享top-k。异构来源仍要独立采样、实体/时间划分，不能把同列号当同物理变量。

`adaptation_parameters('Z')`返回空；U仅暴露可训练参数，不声称混合污染鲁棒；F明确NotImplemented。共享 `candidate_experiment` 已有 CSV loader、配置驱动训练、CAL q99 higher 校准、连续残差评分与独立评价动作。PSM 四组真实训练已完成，全量推理获批；示例配置本身仍不是执行批准。首轮 from_scratch/单seed/200步，不是官方预训练复现或完整 Z/U/F 实验。源代码保持外置，官方TimeRadar许可未认证，不能随本仓库再分发；Autoformer外置目录保留其MIT LICENSE。

## 已完成验证与实际训练

`tests/test_candidate_tf_lag.py`覆盖四组shape/真实官方模块梯度、统一score与target外置、错误SHA/非finite/shape拒绝、self/cross开关、pair预算、作者eval直接调用等价与时延末点保留；fixture lag只用于接口单测，实际模型factory必须加载固定作者模块，不会回退自写lag。CPU测试不等于GPU成本或检测收益。

科学 v2 已通过19项 CPU 合成测试，包含训练/eval时不同原窗口输入相互独立的检查；既有 A40 合成 L0 已完成四模式共12次 optimizer 更新。共享运行入口 v4 另通过22项 CPU 测试，无 skip。PSM 四模式各200步真实训练全部 OS0，按 normal SELECT 选中的 step 分别为 tf200、lag150、concat150、gated150；不能据验证 MSE 排异常检测优劣。当前全量 CAL/TEST infer 已由 root 放行，标签评价需全部分数落盘并另行批准。真实角色、阈值与禁止 query 调参规则继续生效。
