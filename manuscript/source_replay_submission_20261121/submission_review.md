# 投稿前学术质量审查

2026-11-21。审查基准为 [原稿提交 e26286a](https://github.com/codexpker/Hybrid/commit/e26286a3004b92690ea65cf672a498999c81df65)。本次复核直接读取公开归档的 CSV、配置、资源回执和结果报告，不调用历史指标实现。主文表 2–5 的 212 个展示数值均在原始储存字段的对应显示精度内。数值溯源另有一名未读前次审查的同模型家族审核者，只读核对。结论为实证稿可以进入内部投稿准备，外部可复现材料和投稿表单仍需收尾。

## 1. 逐项事实与统计核对

| 项目 | 证据与裁定 |
|:--|:--|
| 数据范围、通道及评分位置 | SWaT 31 通道，PSM 25 通道。修订稿保留表 1 的 336,321 与 65,088 个评分点，在附录明确 raw QUERY、可评分窗口支撑与最终裁剪。SWaT 开头另裁去 63 个已有合法分数的点。PSM 两个边界截断事件只进入点指标。 [SWaT 配置](../../configs/left-exposure-swat-20261118.json) · [PSM 配置](../../configs/left-author-psm-source-replay-20261118.json) · [PSM 归档](../../docs/experiments/left_psm_replay_20261118/RESULTS.md) |
| noU／TT／SS／R 权限与预算 | 同任务共用初态、scaler、原型和运行状态。noU 单个冻结参照。三条更新臂使用全参数、Adam (10^{-4})、批量 256、2,000 次更新和终点。源 scaler 覆盖作者协议的源验证段，源更新采样限 SOURCE_FIT。R−TT 是固定更新数和窗口呈现数下将第二目标半批替换成源半批的效果，同时改变源曝光和目标曝光。 [SWaT 匹配记录](../../docs/experiments/left_swat_source_replay_20261118/RESULTS.md) · [PSM 配置](../../configs/left-author-psm-source-replay-20261118.json) |
| 自然 A 三种子 | 表 2 的均值来自归档汇总，表 3 的配对差来自任务各自的原始配对汇总。先在种子内作差，组间报告样本标准差（ddof=1）。两任务 R−TT 的 AP、ROC-AUC、VUS-PR 和 VUS-ROC 三对均为正，严格误报点数均减少。SWaT 三连覆盖平均增加，PSM 下降。noU 的常数参照不增加独立样本量。 [SWaT 配对 CSV](../../docs/experiments/left_swat_replay_repeat_20261118/paired_summary.csv) · [PSM 配对 CSV](../../docs/experiments/left_psm_replay_20261118/paired_summary.csv) |
| A/B/C 槽位与交互 | 每臂替换共享目标槽中的 25,600 个窗口呈现，占全批呈现 5%。同种子固定槽位，同条件跨臂固定供体。同条件内供体重复不增加独立事件。训练进程不接触供体标签。表 4 为绝对结果，表 5 分开报告 B−C、R-B−TT-B 及差分交互。两臂总异常曝光可不同，供体池的运行状态等内容差异未被匹配。 [曝光 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/exposure_by_seed.csv) · [交互 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/interaction.csv) · [配对汇总 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/paired_summary.csv) |
| AP／ROC／VUS | AP 为不做 point adjustment 的点级平均精度。VUS 调用已归档 LEFT 实现，`windowSize=100`、`thre=250`。表中的数值以原存档小数按稿件精度展示，未用正文舍入值反推。 [绝对结果 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/method_summary.csv) · [逐臂逐种子 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/comparison.csv) |
| 严格点、三连、事件与持续覆盖 | q99／higher／严格大于。三连从第三点确认，按连续评分位置，不回填。一对一为时间贪婪首次未配对重叠规则，any-hit 另报。主表为完整事件内的合并点覆盖，长事件宏平均和合并覆盖分别报告。离线位置和可用时间偏移不作实时延迟解读。 [逐臂逐种子 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/comparison.csv) · [逐长事件 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/all_seed_nine_long_events.csv) · [曲线操作点](../../docs/experiments/left_replay_exposure_repeat_20261118/curves/operating_points.csv) |
| CAL 天然构成 | SWaT 选中 38/512 攻击点，PSM 选中 512/512，且 CAL 角色全域 3,848/3,848 为攻击标签。两者均为事后披露。阈值是各自分数上的经验操作点。PSM 排序指标独立于阈值，操作比较的边界在摘要、协议、主表、外部对照、讨论与结论中明确。 [SWaT CAL 记录](../../docs/experiments/left_swat_natural_u_20261118/RESULTS.md) · [PSM CAL 记录](../../docs/experiments/left_psm_replay_20261118/RESULTS.md) |
| 外部比较 | LEFT 和 MLP/CANDI 可以在相同查询点对照，窗口、源 scaler、初始化及更新权限仍不同。CANDI 为 U-FIT 更新后冻结。TimeRadar 使用独立协议及 PSM 64,728 点轴，只报告该协议内差值。TSAD-C 本地 NA。 [MLP 与 CANDI 表](../../docs/experiments/mlp_replay_20261119/comparison.csv) · [CANDI 协议](../../docs/experiments/candi_native_a_20261118/RESULTS.md) · [TimeRadar 协议](../../docs/experiments/timeradar_native_v_replay_20261120/RESULTS.md) |
| GPU 成本 | 附录 B 直接列已存档的 A40 分阶段占用。PSM 的 0.628679 小时包括九次适配和一次冻结读出，不含复用初态原始训练。暴露复跑的 1.190616 小时为 16 个成功任务实际 GPU 占用，保守资源台账含无 CUDA 的启动守卫，二者分开。缺少完整统一口径的存量成本，不汇报整项目耗时或跨模型效率排名。 [重复曝光资源回执](../../docs/experiments/left_replay_exposure_repeat_20261118/resource_summary.json) · [逐作业 CSV](../../docs/experiments/left_replay_exposure_repeat_20261118/new_job_costs.csv) · [附录成本表](supplement.md#historical-computing-budget) |

## 2. 严格审稿问题清单

| 严重度 | 审稿问题 | 修订或剩余限制 |
|:--|:--|:--|
| 重大 | 仅两个已曝光任务、每任务一个初态。三种子只能衡量固定任务内重复性。 | 摘要、引言与讨论明确归纳范围。没有把窗口、点、事件或跨任务结果当作独立训练样本。泛化仍缺独立任务证据，当前投稿应依靠受控比较价值。 |
| 重大 | PSM 的正常操作点不可认证。跨方法上下文也不等价。 | 明示 CAL 构成与各方法权限，分开解释排序和经验阈值上的检测代价。当前操作指标无法建立等正常误报预算。 |
| 重大 | A/B/C 匹配替换槽，不匹配供体属性。R−TT 也无法独立识别源记忆保持和目标曝光减少。 | 改写摘要、方法和 Discussion。交互限定于该预设干预。去除可能被解读为普适抗污染或内部遗忘机制的表述。 |
| 重大 | 论文中的方法定义可以追溯，但公共仓库不足以从零重建全部原件。 | 补充数据区间、上游版本、冻结配置、随机流、评分约定与可追溯原件的位置。对外复现仍需确认 SWaT 31 通道导出映射、依赖环境和按许可提供的原件访问方式。 |
| 一般 | 旧附录把自然配对差链接到曝光对照 CSV，也未解释 SWaT 的额外裁剪。 | 已修订分阶段链接和三层评分区间。主表数字与归档一致。 |
| 一般 | 原摘要和引言没有首先突出等预算反事实。 | 摘要与引言重写为固定替换策略的两层对照，三个研究问题保留。结果明确保留事件与覆盖下降，结论陈述两域边界。 |
| 一般 | 原 Related Work 已覆盖相关工作，缺少近期直接处理困难正常与异常污染区分、以及直接分析 TSAD 复现问题的研究。 | 仅补 PLDA 和 Xu–Boström 两篇。CANDI、M2N2、RoCA、TSAD-C 继续分别说明适配时点、污染处理权限与本地评价限制。 [PLDA 出版信息](https://doi.org/10.1016/j.neunet.2025.108532) · [TSAD 复现研究](https://link.springer.com/article/10.1007/s10994-026-07183-y) |
| 语言 | *matched donor* 容易暗示供体特征匹配，*coverage* 容易混指事件宏平均。 | 摘要改为 *matched exposure slots*。定义 pooled complete-event point coverage，并解释事件计数与受警时长。 |
| 语言 | *fixed compute* 容易暗示完全相等的设备时间。 | 以相同更新次数和窗口呈现数表述控制项，设备占用另列。 |

## 3. 文献复核

[逐条引用核查表](citation_sources.md) 列有原 13 条作者、出版年份、期刊或会议、DOI 或正式会议条目，以及在正文中的证据用途。原 13 条保留。用于放置适配研究、复现研究、数据和指标的句段已逐次检查。PLDA 比单纯损失筛选更接近本文讨论的困难正常问题。Xu–Boström 直接支持角色、轴和基线权限的透明披露。两者均用于相关工作，没有增加未经执行的基准行。

## 4. 推荐期刊与投稿风险

查证日期为 2026-11-21。JIF 列为出版社展示的 **2025** 年数据。SCImago 是基于 Scopus 的外部分区，列出相应类别或最佳分区，仅用于参考。JCR 学科分区及中科院分区未采用第三方数值替代，若有学校硬要求，以机构可查版本为准。三刊的 SCIE 收录均见出版社页面。风险为对稿件匹配度的判断。

| 优先级 | 期刊与官方指标 | 分区参考 | 实证研究匹配与风险 |
|:--|:--|:--|:--|
| 1 | **Machine Learning**。2025 JIF **4.9**。 [官方首页与 SCIE 信息](https://link.springer.com/journal/10994) | SCImago 2025，Software **Q1**。 [SCImago](https://www.scimagojr.com/journalsearch.php?q=24775&tip=sid) | 官方范围明确包括研究方法学和可复现的实证比较，无新增算法的绝对要求。近期徐与 Boström 发表了 TSAD 评估中泄漏、数据变体和基线复现的实证研究。本稿的匹配干预和操作代价分析契合，风险是两任务及单初态限制提供的证据广度。 [官方范围](https://link.springer.com/journal/10994/aims-and-scope) · [2026 年近例](https://link.springer.com/article/10.1007/s10994-026-07183-y) |
| 2 | **Data Mining and Knowledge Discovery**。2025 JIF **5.5**。 [官方首页与 SCIE 信息](https://link.springer.com/journal/10618) | 2025 SJR 最佳 **Q1**，大学索引转引。 [大学 AVİD 数据](https://avid.selcuk.edu.tr/dergiler/13579) | 官方范围覆盖原创技术研究与实践，允许以评价方法和新发现形成贡献。该刊发表过聚焦时序异常指标的 *Navigating the metric maze*。技术深度要求高，本稿要依靠干预对照、分母区别和可复现包，算法新颖性较弱与样本范围有限会提高初筛风险。 [官方范围](https://link.springer.com/journal/10618/aims-and-scope) · [2024 卷期近例](https://link.springer.com/article/10.1007/s10618-023-00988-8) |
| 3 | **Knowledge and Information Systems**。2025 JIF **3.6**。 [官方首页与 SCIE 信息](https://link.springer.com/journal/10115) | SCImago 2025，Information Systems **Q1**、Artificial Intelligence **Q2**。 [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=15703&tip=sid) · [大学分类数据](https://avid.selcuk.edu.tr/dergiler?jcr_quartile=Q2&kategori=Information+Systems&quartile=Q1&sayfa=2&sirala=total_docs) | 官方范围包含学习与适配、数据挖掘和经验论文。近例包括可穿戴时间序列的无标签跨域适配，以及带校准的异常分析。两例提出了新算法，因此只能证明主题匹配。当前稿件应以适配决策、可审查的操作边界及材料完整性争取适用性。 [官方范围](https://link.springer.com/journal/10115/aims-and-scope) · [2026 适配近例](https://link.springer.com/article/10.1007/s10115-026-02918-8) · [2026 校准近例](https://link.springer.com/article/10.1007/s10115-026-02733-1) |

首选 **Machine Learning** 的常规研究稿，依据是其明确接受研究方法学型实证贡献和已有同领域近例。选择时无需将源重放重新写成算法创新。

## 5. 最终稿件分配及待办

主文保留摘要、八章、自然条件绝对结果及配对差、A/B/C 绝对结果及交互、代表性低误报曲线和外部权限表。补充材料保留精确角色与初始化、每种子完整指标、所有历史对照及事件详情、计算台账。停止的方案仅在公共历史索引中定位。作者页、资助和利益冲突声明、数据及代码可用性最终措辞、目标期刊模板与匿名稿规则仍需定稿。

[主文 PDF](manuscript.pdf) 11 页，[补充材料 PDF](supplement.pdf) 4 页。两份均由 Pandoc 3.8 生成 LaTeX、由 Tectonic 0.17.0 编译，检查了文献、图表、字符及分页。正文的引用由 Citeproc 从本目录的 `references.bib` 生成。本稿采用通用单栏排版，期刊模板待最终选刊后应用。
