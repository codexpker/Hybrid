# 第二篇论文英文可编辑初稿

2026-11-15。本目录为新一代完整 Hybrid 的写作分支材料，不覆盖历史论文稿或实验。正文已实际写出 Introduction、Related Work、Problem Formulation、Experimental Setup；不是只有提纲。

- `draft.md`：正文源文件，Pandoc 引文格式。
- `draft.tex`：由 Markdown 生成的通用 article LaTeX，可独立编辑；尚未选定投稿模板。
- `references.bib`：十条可追溯文献，MTGFlow extension 以已读预印本身份引用，不冒称正式期刊版本已核。
- `citation_sources.md`：正式来源及精读范围。
- `method_definition.md`：与已冻结实现 `d77252e1` 对应的简短结构、公式、权限和局限说明。

当前模型定义来自主 Agent 本轮确认的唯一完整设计：交替时间/正交 Haar 域的可逆联合密度学习，固定 Student-t(5) 潜变量先验，训练与原生评分同为窗口归一化 NLL。代码最终参数、调度器、最大更新及结果由实际运行配置补入，本文不虚构这些尚未终态的内容。原生 LEFT、TimeRadar 为外部比较方法，旧 Hybrid 是内部历史版本。

本文**没有**宣称新模型已经改善排序、持续检测、微弱异常、Self/Cross-lag 或污染鲁棒性。Stage1/Z/U/F 与混合无标签适配保留为研究框架，当前阶段只验证基础学习与检测机制。PSM/SWaT/TEP 明确标为已曝光开发/机制数据。故障子集、完整事件、信息权限与强简单基线均保留。

编译方式（有 Pandoc 和 LaTeX/BibTeX 环境时）：

```powershell
pandoc draft.md --standalone --natbib --top-level-division=section -V documentclass=article -V geometry:margin=1in -o draft.tex
pdflatex draft.tex
bibtex draft
pdflatex draft.tex
pdflatex draft.tex
```

本机可用 Pandoc，尚未找到 pdflatex，因此仅生成和检查可编辑 LaTeX，不声称已编译 PDF。原生公开数据来源的正式数据集引用尚需从对应已冻结 release manifest 的确切版本补入；当前未杜撰数据集论文作者/DOI。
