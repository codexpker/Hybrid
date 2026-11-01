# Schema6 U读出候选有限CPU独审

**PASS_LIMITED_SCHEMA6_CPU_V9**。v7已知gold选择错误与v8独立发现的RNG/metadata语义缺口均已在有限修订中关闭。仅接受CPU合成工程范围；不批准真实catalog构建、9项Z正常cal、U query-L0/profile/full、seal或标签评价，不给科研GO。科学状态仍HOLD、收益/贡献证据INCOMPLETE。

## 来源与范围

读取12项合同56f4106f…、主CPU-only决定9ec890f8…；v7 FREEZE ea63754b…、v8 c530d52c…、v9 9487c5d3…。v9新readout源码为a712825a1eb757bafb1de5f4cec9def6ede1a47174cb65461eb70b5b7bae8b02，identity为045927f0…、CLI为f1771033…。29项旧源码字节不变；17处函数/类出处在同一远端Python3.12的文件SHA/AST全部相符，直接调用原core、collector、score_block和cal helper。原本地Python3.8 AST结构不兼容的结果保留、不作源码变化裁决。未使用旧dist包或开发者40PASS替代独审。

v7取证时开发已并发修改三项新文件，本审从原hash固定candidate.patch恢复并逐SHA匹配v7原字节，再制作独立ZIP；没有把当前v8源码当v7。v8→v9只改seal/test/doc，AST核得唯一变化源函数为seal，其他代码及核心完整保留。

## 正确路径与拒绝证据

- 复用开发fixture仅作合成产物布线，独立运行catalog→9 Z normalcal→18×2实体=36银行→seal→gate；两init键分离，S/T引用同一canonical对象，gate明确labels_permission=false。
- 独立×2重签U canonical、Z canonical+q99，缺银行、重复shard、L0缺init/错seed、缺U row、错U/Z stage、best替代、只有train接受而无cal接受、PENDING均拒绝。严格18row和逐波接受树在合成范围闭合；本次没有加载真实生产18项cal/预测/分数或制作production catalog。
- 用真实既有三mode架构、六个合成随机checkpoint做CPU差分：新infer与原collect_forecasts全部数组/RNG逐位相同。两种init名称只是身份键，不是真实预训练。CPU峰值字段为null；改变首块未来观测不改变首块prediction而改变其观测分数。无真实权重或新训练。
- 真实CLI runtime/preview/PENDING在Torch及np.load guard下分别退出0/0/2，没有触及模型或数组值。新评价/统计/标签适配未迁移，不在本CPU候选中冒称完成。

## 失败、归属及关闭

| 项目 | 原始反例 | 归属 | 当前裁决 |
|---|---|---|---|
| gold桥 | v7固定temporal会拒绝事前固定hybrid桥 | root先发现，本审独立复现 | v8固定严格整数seed0/hybrid，正路及错gold拒绝通过 |
| RNG语义 | v8允许(4,0)空状态或同宽全0的RNG封存complete | 本审独立合成反例 | v9原两NPZ攻击字节重用，均拒绝；额外单字节篡改也拒绝 |
| metadata矛盾 | v8允许seed999/mode=tf/init=random/training=true但保留正确job/checkpoint | 本审独立合成反例 | v9核JOBS、状态/key/点数/块数/batch/重放标志，原组合及逐字段攻击全部拒绝 |

这类反例隔离合成semantic校验，不声称伪造真实production OS/shard签名；执行receipt仍另行约束事后篡改。原v8失败和完整fixture未覆盖。v9窄回审不生成新forward参考：重用v8原numeric banks/RNG/Zcal bank与canonical，只在新合成目录重绑runtime/operation/metadata SHA，正确36银行seal/gate仍通过，原三攻击及单字节/逐字段共15个负例拒绝。模型/collector/Zcal-forward入口均设失败guard，确认未再调用。

## 同旧Z RNG闭合的数学条件

原collector在构造/加载模型后每entity重置84551；前2块重放后恢复after状态。HybridPredictor模式分支仅由mode控制，head Linear无RNG，temporal attention dropout=0；模型eval使原repr_dropout禁用。外置原TimeRadar源码SHA881e8912…中启用mask_mode=c的torch.rand尺寸仅由copies、B、C、patch_num决定，与权重/alpha/head值无关。H100、copies2、B1C52、原origin顺序及同后端时，改变checkpoint/head不改变随机消耗。因此v9与旧Z同seed/mode/entity的**实际已存RNG字节**比较成立，不猜CPU/CUDA格式、不调用新参考推理。它只证明随机序列身份，不能证明新预测值正确；物理target/scale、评分数学和真实执行身份仍独立验。

## 限制及后续必要条件

实际生产catalog/旧桥消费、Z正常cal与U查询尚未执行；真实GPU吞吐/峰值、操作审批时序、资源壳账本、完整7200银行与后续新维评价均INCOMPLETE，须各自另签及终审。本次不读取真实query值、预测、残差、分数、labels或AP，不改原score/阈值/权重/数据。保留高基线低方差float32敏感性、旧job时环境缺录UNVERIFIED、经验q99无FPR保证、TEP已见探索性及known-normal非混合污染鲁棒等边界。

CPU主体：v8语义14.6382秒、合成真实架构差分/CLI7.0630秒、v9窄回审4.7311秒，合计约26.43秒；峰RSS619,384,832B，小于600秒/2GiB上限。归档/静态校验另为少量CPU；新增GPU为0。同家族、共享工作区的职责独立审查，非外部或双盲。

原始证据：semantic_audit.json、transport_audit.json、provenance_remote.json、v9_revision_audit.json、三个固定snapshot ZIP及v8完整合成fixture ZIP；REVIEW_V7_V8.md保留原阻断。脚本audit_schema6.py/audit_transport.py/audit_revision_v9.py只写独立目录。本次实际远端命令均为现有Python运行相应脚本，CUDA_VISIBLE_DEVICES在脚本内置空；已有输出拒绝覆盖。原件与最终manifest同步本地，后续执行权限仅由主代理按具体冻结原件另签。
