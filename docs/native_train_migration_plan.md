# 原生源训练迁移：有限工程阶段计划

2026-10-31。当前仅检查和计划；等待主智能体告知上一阶段基线commit后才改现有源码/测试/配置。此阶段不是科学训练合同，不授权新GPU作业、真实数据更新或重新计算既有结果。

## 已核来源与不可变项

来源是项目自有冻结`hybrid_10k_wave.py`，SHA256 `00ed75a4c1da76600d573be67dead4179b455131610607d7b1cea3bb0dc8e394`。已读其batch、evaluate、train完整调用链；三核心model/data/protocol沿用包内provenance原SHA，不编辑。官方TimeRadar文件保持外部固定SHA，训练类仍随机初始化，不引入上游源码副本、额外模型或预训练权重。

原数值合同：history400、horizon5、同域batch64、10000步/640000曝光、AdamW3e-4/weight_decay.01、warmup500后9500步cosine、gradient clip1、每25步曲线/每500步source select、patience仅warning、固定step10000为primary、best仅补充。三分支初始化与inactive参数冻结、原objective、源cal公式、计划trace顺序和mask_seed不变。source evaluate保存/恢复RNG，并在其内部重置mask_seed；不能让验证随机性改变后续训练。

## 拟迁入的共享职责

1. 在`engine`新增通用训练模块，迁入原batch构造、source evaluate、单步优化/固定LR、10000步循环和原产物写出。共享source计划/源cal继续用已验收helper；不另造一份采样或评分算法。
2. `train --config ...`从显式legacy桥改为调用原生共享模块，默认仍preview。新训练输入必须由schema2操作manifest和批准绑定，包括外部wave配置、inputs清单、compact plan/rows、source manifest、官方文件及entry gate；执行前核冻结预算、数据文件hash、target排除和job/source关系。输出必须全新目录；旧run、已有implementation和审批不改。legacy桥保留为历史追溯入口。
3. 保留原step10000/best/optimizer checkpoint、select/cal预测、cal_scores、curve和summary语义；新的身份/源cal元数据只能附加，不能替换旧数值。当前仅工程CPU验证，不据原生入口存在即启动正式训练。

## 等价验证与来源记录

直接从固定旧脚本抽取实际batch/evaluate/train步进与LR语句，以文件SHA、函数/摘录身份对应，拒绝用新实现自我比较。工程fixture允许减少执行次数、将device/CUDA RNG保存恢复替为CPU对应操作；正式CLI仍固定完整10000步预算，不把合成小步结果写成SOURCE_10K_COMPLETE。

小型合成CPU检查包括：同初始化/输入plan/seed下训练trace与batch逐值一致；LR在step1/499/500/501/10000边界一致；实际若干更新的loss、梯度裁剪、所有active/inactive参数、AdamW状态及后续RNG逐值一致；source select/cal预测、mask重置后重放和外部训练RNG恢复一致；模型state_dict与原optimizer checkpoint严格加载兼容。必要时用既有环境、CUDA不可见、线程2、单次≤60秒，无新依赖或真实query。

恢复边界必须单列：旧optimizer checkpoint只有optimizer和step，缺少完整随机状态，不能声称它支持中途中断的精确续训。若增加独立RNG sidecar，只作为额外可审计元数据/合成kernel恢复验证，不改变原checkpoint字段、步进或输出；完整作业恢复流程未实现就明确未迁移，不凭seed猜测过去RNG。

## 验收、停机与未迁移

交付：真实包原生train入口、统一配置/来源映射、CPU等价结果/命令/hash、负向拒绝测试、原有回归及三核心SHA检查、迁移范围与缺项。工程执行者发STAGE_COMPLETED交主和独审，不自签独立PASS，不Git提交。原三项与邻接事件拒绝包装保持。

若旧冻结源码身份不符、CPU对照不能复现参数/RNG、需改原模型/预算/数据协议才能完成，立即BLOCKED并附最小反例；不能以“重构必要”为由改变科学合同。科学收益/污染鲁棒/跨域贡献保持原状态。完整GPU数值等价、合法新科学训练、U/F适配、通用prepare/实体准入、all-bank标签开锁及原生作业恢复不自动纳入本阶段。

实施记录：主已告知基线f7858a4后解除源码等待。随后受控加入100/400历史接口，保持horizon5及原优化预算；具体实现、合成结果与余项见`docs/native_training.md`。TEP科学划分/预算不由本工程生成。
