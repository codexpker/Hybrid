# Machine Learning 最终投稿检查

官方指南查阅于 2026-11-21。实验与科学内容冻结在原 [投稿修订版 `47232b2`](https://github.com/codexpker/Hybrid/commit/47232b29c2fcf0b399f5d157400bbacc1f3863e1)。该包提供期刊排版、匿名审稿材料和归档证据溯源。

## 官方格式与文件分配

| 项目 | 核对结果 |
|:--|:--|
| LaTeX | 期刊建议使用 Springer Nature 原始 LaTeX 模板，并提供可编辑源文件、样式、图和 PDF。主文使用官方 `sn-jnl` v3.1（2024 年 12 月模板包），`pdflatex,sn-apa` 选项。模板类和 BST 按发行包原件提供，保留其 LPPL 许可声明。 [期刊指南](https://link.springer.com/journal/10994/submission-guidelines) · [官方模板](https://www.springernature.com/gp/authors/campaigns/latex-author-support) |
| 摘要、关键词和结构 | 203 词英文摘要，定义两数据集简称，5 个关键词。主文八章采用十进制层级，7 个顺序编号的表和 1 张图。引入第 2 表为更新臂定义，结果表随之顺延。 |
| 引文与参考文献 | 姓名—年份引用，15 项参考文献按首作者姓氏排序。APA 模板生成，已有 DOI 以可点击的全链接呈现。TSB-AD 的 DOI 与页码采用 NeurIPS 正式页面和 proceedings 记录。作者完整名单保留在 BibTeX，模板可对长作者名单缩写。 [参考文献要求](https://link.springer.com/journal/10994/submission-guidelines#Instructions%20for%20Authors_References) · [TSB-AD 记录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html) |
| 匿名性 | 公开期刊指南明确要求作者标题页，只在 Springer 通用条款中条件式提到双匿名，未明确普通稿的匿名审稿模式。已按用户要求准备独立匿名文件。主文、补充及说明的 PDF 作者字段为空，文本和链接不含本项目可识别仓库及本机路径。已有公开稿和实验仓库仍可能使人通过内容识别作者，匿名文件的范围仅限所交文件。最终上传类型以投稿界面要求为准。 [官方投稿流程](https://link.springer.com/journal/10994/submission-guidelines) |
| Contribution Information Sheet | Machine Learning 要求每次投稿附 1–2 页贡献说明。此处提供 1 页的 [英文说明](anonymous/contribution_info.pdf)。前 3 项完成，第 4 项的既往发表、公开稿与潜在重叠情况须由作者确认。 [期刊投稿要求](https://link.springer.com/journal/10994/submission-guidelines) |

审稿文件夹为 [anonymous/](anonymous/)。主文使用 [main.tex](anonymous/main.tex)、`sn-jnl.cls`、`sn-apacite.bst`、`references.bib` 和 `Fig1.pdf`，预览为 [main.pdf](anonymous/main.pdf)。原有 PR/ROC 图由 Matplotlib 3.10.8 绘制，曲线和操作点未改变。本次去掉图内总标题并转换为矢量 PDF，保留主文题注。 [补充材料](anonymous/supplement.pdf) 与 `tables/` 内的 11 份 CSV 随稿提供。主文含研究问题、主要对照、结果与局限。补充材料含精确数据轴、归档结果、事件细节、历史算力和外部基线权限。

[作者标题页](title_page.tex)、[待填写声明](SUBMISSION_DECLARATIONS.md) 和 [复现归档指南](REPRODUCIBILITY.md) 是作者侧材料，包含未填项目或可识别归档链接。按编辑端要求填写并选择可见性。投稿前核对期刊系统合成的 PDF。

## 科学与数字终检

| 检查项 | 结果 |
|:--|:--|
| 贡献措辞 | 摘要、引言、结果和结论使用同一固定替换策略解释。源重放归于已有学习策略。 |
| R−TT 与三种子 | 两臂等更新次数和窗口呈现数。R 将第二目标半批替换为源半批，目标曝光随之改变。三种子是两个任务各自冻结初态下的适配重复，按种子内差值、样本标准差汇报。 |
| 误报、覆盖和 CAL | 保留 SWaT 三连覆盖上升与 PSM 下降、严格误报、any-hit 和一对一匹配。摘要、协议、主表、外部对照、讨论与结论披露 PSM 全攻击 CAL。离线可用时间说明保留。 |
| A/B/C | 槽位匹配，同条件跨臂共享供体。无工况匹配声明。直接比较、绝对指标和交互分别解读。 |
| 数值与图 | 四张数字结果主表的 30 行与已逐项追溯归档 CSV 的冻结修订稿一致。匿名包中的 11 份衍生 CSV 与原件逐字节相同。补充材料的区间、阈值角色及计算台账保留原归档字段。未重算历史指标。 |
| 排版复核 | 本地用 Tectonic 0.17.0 与内置 BibTeX 编译。主文 **16 页**，补充材料 **3 页**，贡献说明 **1 页**。检查了每页图表与分页、7 个表号、15 条引文、DOI 链接和匿名字段。最终构建没有溢出版芯、缺字或未解析引用。 |

## 待作者确认

1. 作者姓名、单位、通讯作者及作者贡献，资助、利益冲突、伦理和数据使用、代码和数据可用性声明。
2. 贡献说明第 4 项的既往发表、公开草稿和重叠披露。作者须审阅最终 AI 写作辅助说明。
3. 锁定允许提供的归档资产、负责人与访问渠道。冻结版的 31/25 通道顺序已找回。原始 SWaT 提供方修订到 DADA 导出的完整链、公共 PSM 文件与冻结长表的逐值等价、每次运行的完整依赖锁均未证实。
4. Editorial Manager 所选稿件类型、文件可见性及服务器端 PDF 尚待真正上传时核对。当前没有发起投稿。
