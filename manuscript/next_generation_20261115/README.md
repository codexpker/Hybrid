# 第二篇论文英文可编辑初稿

## 公开 TSAD 基准总表（2026-11-22）

第二篇以完整 Hybrid 的两阶段公开数据 TSAD 为当前主线。`Hybrid（待验证）` 保持缺项。下面复用固定归档和论文原表，数值范围为 0–1。论文报告的 AUC-PR（部分论文记作 AUC-P）、本地未调整点 AP 和 VUS-PR 是各自口径。只有在时间轴、聚合及权限相容的固定子轨内，才作相应直接比较。Published 结果未查到兼容本地分数的项目仍缺本地行。

### A. 本地公开原协议及固定全域基准

单元格为 **AP / ROC-AUC / VUS-PR / VUS-ROC**。`NA` 表示该格在对应轨没有已核结果，单元格中的 `—` 仅为该指标缺项。

| 方法与来源 | SMD | PSM | SWaT | MSL | SMAP |
|---|---|---|---|---|---|
| Hybrid 所提方法（待验证） | NA | NA | NA | NA | NA |
| TimeRadar 官方公开权重，公开脚本 block0（本地兼容执行） | .146927 / .727109 / — / — | .433413 / .670444 / — / — | .639247 / .821948 / — / — | .235469 / .752300 / — / — | .155959 / .586702 / — / — |
| LEFT 原生目标训练（本地复现） | NA | .576644 / .776096 / .572602 / .785067 | .543249 / .821511 / .533707 / .809348 | NA | NA |
| CATCH 固定目标训练（本地共同 PSM 轨，3 个训练种子） | NA | .436432 / .647355 / — / — | NA | NA | NA |
| PCA 残差（本地共同 PSM 轨，目标训练前缀拟合） | NA | .534404 / .793321 / — / — | NA | NA | NA |
| TimeRadar 官方公开权重，固定本地 SMD28／PSM 角色，无目标参数更新 | .416983 / .855398 / — / — | .432502 / .670965 / — / — | NA | NA | NA |
| LEFT 同角色（本地 FIT-only） | NA（仅 3 台） | .571098 / .764667 / — / — | NA | NA | NA |
| TimeRCD A，官方权重与角色内归一化（本地，同原时间轴的头部 logit 差） | .487280 / .838074 / — / — | .359946 / .552157 / — / — | NA | NA | NA |
| TimeRCD B，官方权重与 FIT-only 归一化（本地项目固定版本，同原时间轴的头部 logit 差） | .473616 / .859199 / — / — | .303012 / .476555 / — / — | NA | NA | NA |

- **公开脚本轨。** TimeRadar 官方提交 `4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2`，公开权重和兼容入口，源预训练后测试时不更新。block0 是各公开脚本的首配置，非按结果择选，SMD 为作者处理序列上的 pooled 708,400 个评分点，PSM 87,800 点，SWaT 449,900 点，MSL 73,700 点，SMAP 427,600 点。公开输入通道分别为 38、25、31、1、1。原作者处理元数据字节身份与论文主表逐行配置映射仍未闭合。`norm=1` 的 SMD/MSL/SMAP 使用整个目标 pretest 前缀（含 SELECT）统计，`norm=0` 的 PSM/SWaT 按其原入口执行，SMAP block0 的 `L=2`，其余此行 `L=1`。整窗输出及批共享阶数允许未来窗内/同批信息。PSM 另一原生配置 block1（`norm=1, L=1`）为 AP/ROC .362481/.566741，保留原档。[公开配置结果档](../../../results/official_block_completion_20261028/official_block_coverage_20261028_v1/RESULT_TABLE.md) · [统一环境补核](../../../results/timeradar_reproduction_refresh_20261029/REPORT.md) · [官方代码](https://github.com/mala-lab/TimeRadar/tree/4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2)。
- **独立实体及共享轴。** SMD28 使用原 OmniAnomaly 文件固定版本 `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4` 的全部 28 个 38 通道实体，按机器等权宏平均，TimeRadar SMD28 数字取原生全轴。TimeRCD SMD28 用其固定共同轴，仍按 28 台宏平均；与其同轴的 TimeRadar 和 History Mean512 专项完整逐实体表保持独立。固定目标 FIT-only 的 PSM 三行在 `[512,87744)` 有 87,232 共同评分点。匹配 LEFT 仅完成三台 SMD：AP/ROC 宏均为 .376827/.886443，覆盖不足记 NA。CATCH 在 SMD machine-1-7 固定局部任务有三种子 AP/ROC .251235/.684101，亦不填进 SMD28 格。低覆盖值和全部实体结果可从原档追溯。[SMD28 各轴及 LEFT 切分](../../docs/experiments/timeradar_smd_full_left_matched_20261116/RESULTS.md) · [LEFT 共同比较](../../docs/experiments/timeradar_smd_full_left_matched_20261116/LEFT_RESULTS.md) · [TimeRCD 对照](../../docs/experiments/timercd_fixed_20261117/logit/RESULTS.md) · [CATCH 局部 SMD 档](../../../results/catch_boundary_smd_20261029/REPORT.md)。
- **独立目标域训练版本。** LEFT `3fadb4811797075e233076b9faceb0d8ec59b697` 的 SWaT 原生结果用 31 通道、正常前缀 495,000 点（396,000 FIT + 99,000 SELECT），测试 `[495000,944856)` 共 449,856 有效点。PSM 为公开 132,481 点前缀及 87,841 点原测试、25 通道、前缀含 SELECT 的 scaler、归档 R 轨 seed2025/runtime step0，在 `[0,87744)` 保留 87,744 点；实际公开 CSV 身份限制仍保留。两者均为 192 点整窗离线读出。CATCH PSM 以 `3647c69be5eb56649b072596cf89098e689e20c3` 目标 FIT-only scaler 和自身验证选权，PSM 共同原轴 `[0,87744)`，与此行 PCA 相容，CATCH 为 seed0/1/2 均值。整个 PSM 原轴与其裁掉前 512 点的轨分开。[LEFT 原生 SWaT](../../docs/experiments/left_native_swat_20261118/RESULTS.md) · [LEFT PSM AP/ROC](../../../results/left_psm_R_point_evaluation_20261101_v1/REPORT.md) · [LEFT PSM VUS](../../../results/left_psm_R_vus_seed2025_20261101_v1/REPORT.md) · [CATCH/PCA 共同 PSM](../../../results/catch_common_repeats_20261029/REPORT.md)。

### B. 论文报告的外部参照

列顺序相同。前三行数值仅按源论文标记为 **AUC-PR / ROC-AUC**。后续行均为 **VUS-PR / VUS-ROC**。没有把 AUC-PR 写作本地 AP。文献同名数据的聚合、预处理和实现有差别，不把论文 SMD 行等同于本地 28 台宏均。TimeRadar v1 Table 1 的 TimesFM 行为 v2.0（Table 5 标注 500M、冻结零样本），另有论文与本地接口的异常读出差别，论文对预测量到异常分数的精确实现未核定。本地正式 TimesFM 3.0 行见 D。

| 方法，论文来源与评价轨 | SMD | PSM | SWaT | MSL | SMAP |
|---|---|---|---|---|---|
| TimeRadar 零样本，TimeRadar Table 1 | .1609 / .7454 | .4444 / .6772 | .7033 / .8331 | .2438 / .7602 | .1642 / .5985 |
| TimesFM 2.0 零样本，TimeRadar Table 1/5 | .0737 / .6280 | .2930 / .5104 | .6932 / .8098 | .2026 / .6969 | .1431 / .5757 |
| CrossAD 目标训练，CrossAD Table 3 | .1864 / .7823 | .4906 / .6810 | NA | NA | NA |
| LEFT 目标训练，LEFT Table 2 | .2389 / .8638 | .5701 / .7836 | .5290 / .7991 | .3342 / .8157 | .1726 / .6562 |
| CrossAD 目标训练，CrossAD Table 2 | .2344 / .8580 | .5596 / .7302 | .4767 / .7865 | .3144 / .8091 | .1443 / .5779 |
| CATCH 目标训练，CATCH Tables 6–7 | .159 / .797 | .436 / .639 | .241 / .462 | .256 / .735 | .155 / .543 |
| ModernTCN 目标训练，LEFT Table 2 所引统一基准 | .1596 / .7707 | .4668 / .6480 | .0941 / .2735 | .3010 / .7747 | .1395 / .5470 |
| Anomaly Transformer 目标训练，LEFT Table 2 所引统一基准 | .0796 / .5117 | .3309 / .5186 | .2679 / .5561 | .1041 / .3890 | .1239 / .4571 |
| Isolation Forest，LEFT Table 2 所引统一基准 | .1304 / .7224 | .3964 / .6009 | .1011 / .3677 | .1631 / .5638 | .1315 / .4960 |

TimeRadar Table 6 另报 SMD、PSM、MSL 的 VUS-PR/ROC，为 .2410/.8616、.4800/.7208、.3595/.8163，SWaT/SMAP 缺项。CrossAD 对应 Table 1 的 SMD/PSM/SWaT/MSL/SMAP 通道分别为 38/25/31/1/1，LEFT 采用相同名义通道，并保留原表预处理及汇总的不确定性。ModernTCN 原论文 Table 31 主要报告其阈值 F1，无法替代此处所引 VUS。CATCH 原论文表的模型与配置轨独立，LEFT 统一基准中同名 CATCH 使用另一评价轨。[TimeRadar Tables 1/4/5/6](https://arxiv.org/html/2602.19068v1) · [LEFT Table 2](https://arxiv.org/html/2602.08638v2#S4.T2) · [CrossAD 论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c9658e8c20879632cb1cfca91d80ceb7-Abstract-Conference.html) · [CrossAD Tables 2–3](https://arxiv.org/html/2510.12489) · [CATCH 附录](https://proceedings.iclr.cc/paper_files/paper/2025/file/2b25c39788e5cf11d3541de433ebf4c0-Paper-Conference.pdf) · [ModernTCN 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/86b1437c1e4c3b3c4debff98234a67e7-Paper-Conference.pdf)。

### C. 自然混合适配后缀（本地）

取同一固定评价点，PSM `[22656,87744)`（65,088 点），SWaT `[608535,944856)`（336,321 点）。数值为 **AP / ROC-AUC / VUS-PR / VUS-ROC**。均为各任务自身的正常初态。当前归档的 LEFT 以完整 192 点窗回顾读出，MLP/CANDI 以 10 点整窗 MSE 赋给该窗末点。两者初始训练、读取足迹和标准化权限有差别。CANDI 按目标 FIT 适配后冻结的本地接线执行，保留训练评估轨区别。

| 方法与适配权限 | SMD | PSM | SWaT | MSL | SMAP |
|---|---|---|---|---|---|
| LEFT 正常初态，零目标范围更新（单固定初态） | NA | .583688 / .771345 / .575447 / .775647 | .607560 / .842267 / .606345 / .829634 | NA | NA |
| LEFT 仅目标更新（TT，3 种子宏均） | NA | .571614 / .732122 / .561903 / .739760 | .602121 / .883489 / .608004 / .879869 | NA | NA |
| LEFT 仅源继续训练（SS，3 种子宏均） | NA | .575601 / .763598 / .571421 / .771357 | .608429 / .850662 / .608127 / .836893 | NA | NA |
| LEFT 源重放（R，3 种子宏均） | NA | .587530 / .747381 / .585520 / .761011 | .628215 / .889039 / .633021 / .884876 | NA | NA |
| Plain MLP 正常初态（单固定初态） | NA | .604606 / .810453 / .625606 / .819068 | .783949 / .856207 / .543552 / .696882 | NA | NA |
| CANDI 完整本地接线（单种子） | NA | .604606 / .810451 / .625607 / .819065 | .783885 / .856961 / .541710 / .695700 | NA | NA |

受控污染比较单列保留。后缀的 AP/ROC/VUS 按原点级和 `RangeAUC_volume_opt(windowSize=100, thre=250)` 定义。没有使用查询标签选阈值。PSM 512 个 CAL 点全部为攻击点，此轨阈值只能解释为固定经验工作点。三连事件命中、误报、持续覆盖、按事件稳态覆盖、最长漏检和窗口可用时间在已有报告及 CSV 中完整保留。举例，SWaT 的原生 LEFT 初态严格正常 FPR 为 1.758%，CANDI 为 97.383%，因此 AP 单项不足以判断预警工作点。[SWaT 三种子表](../../docs/experiments/left_swat_replay_repeat_20261118/RESULTS.md) · [PSM 三种子表](../../docs/experiments/left_psm_replay_20261118/RESULTS.md) · [CANDI 工作点](../../docs/experiments/candi_native_a_20261118/RESULTS.md) · [独立受控污染表](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md)。

### D. 当点可用评分／严格过去预测（本地）

数值同样为 **AP / ROC-AUC / VUS-PR / VUS-ROC**。PSM 和 SWaT 统一使用因果 V 对照轨，与 C 的 PSM 后缀不同。SMD 和 SMAP 简单参照的时间轴与汇总分别在行下注明。这个面板描述输入信息的可用时点，不主张历史执行为实测实时部署。

| 方法与权重 | SMD | PSM | SWaT | MSL | SMAP |
|---|---|---|---|---|---|
| TimeRadar 官方权重，因果 100 点末点原生 V | NA | .550952 / .735048 / .484578 / .648161 | .795725 / .857852 / .576600 / .708990 | NA | NA |
| TimesFM 3.0 官方完整权重冻结，H512→1 中位数 MSE | NA | .344512 / .566754 / .376176 / .608554 | .093973 / .288375 / .096544 / .298381 | NA | NA |
| TimesFM 3.0 轻量代码模块，正常前缀自训原值预测 MSE | NA | .567902 / .743845 / .575549 / .764875 | .762338 / .838454 / .546680 / .696022 | NA | NA |
| History Mean512（无神经训练） | .111306 / .661935 / — / —（28 台宏均） | .303673 / .525589 / .332386 / .567946 | .087661 / .227761 / .090893 / .232695 | NA | .261003 / .657611 / — / —（53 实体微统计） |

- PSM 25 通道、测试相对轴 `[23016,87744)` 64,728 点。SWaT 31 通道、组合轴 `[608535,944856)` 336,321 点。TimesFM 3.0 官方代码固定提交 `918df02c9d366b2145bc5876135fa2131d3ed2a6`，权重修订 `43046b85ec22d584a13f8098c2ed39c889e129c2`，未更新权重。官方联合预测仅用过去 512 点，使用固定源 FIT 标准化下的中位数单步预测 MSE 作为项目异常分数。轻量模块没有加载官方权重，自己在任务正常源前缀训练。两种预测器无法单独辨识权重的作用。TimeRadar B1 用以 t 为右端的 100 点及固定十副本方差，不等同 A 中的整窗与共享批次原生输入。PSM CAL 仍是全攻击。本面板的 SWaT History Mean512 没有兼容 CAL，只有排序。[冻结 TimesFM 3.0 与报警细目](../../docs/experiments/timesfm3_frozen_20261121/RESULTS.md) · [因果 V 读出与轻量模型](../../docs/experiments/timeradar_native_v_conditioning_20261119/RESULTS.md)。
- History Mean512 的 SMD 行对全部 28 个实体分别在 `[512, floor(testN/100)×100)` 计算 AP/ROC 后等权宏平均。SMAP 使用原生 53 条 sensor0 序列、保留时间边界，合计 400,481 个已评分点。表内是微 AP/ROC，另存实体宏 AP .318134。MSL/SMAP 的单通道裁法只能用于通用性，无法验证跨通道机制。[SMD28 共同轴归档](../../docs/experiments/timeradar_smd_full_left_matched_20261116/RESULTS.md) · [SMAP 原生实体归档](../../../results/hybrid_v1_public_native_zs_20261105_v1/PUBLIC_DOMAIN_PRIMARY_REPORT.md)。

**必要缺项。** 当前待验证 Hybrid 的全域主表、贯通 Z/U/F 和同协议机制消融是主要缺口。完整同角色 LEFT SMD28 和 CATCH SMD28 未完成，CrossAD、ModernTCN 和经典深度检测器暂无兼容本地全域表。外部行均复用论文原表或已归档指标。本轮新候选及其有限开发验证见[方法章节](method_definition.md)。

**所提方法的独立开发折。** SWaT＋全 SMD28 预训练、PSM 单 seed 的对称排序 U 已完成。相同 64,728 点上 AP 为 .302144，普通 Logistic U 为 .294008，Z 为 .261274，原生 TimeRadar V 为 .550952。持续覆盖显著低于原生 V。该初步对照的差值和代价写在[现有方法章节](method_definition.md#first-development-fold-results-and-scope)，不回填全域主表。

## 2026-11-15 初稿和归档

2026-11-15。本目录为新一代完整 Hybrid 的写作分支材料，不覆盖历史论文稿或实验。正文已实际写出 Introduction、Related Work、Problem Formulation、Experimental Setup；不是只有提纲。

- `draft.md`：正文源文件，Pandoc 引文格式。
- `draft.tex`：由 Markdown 生成的通用 article LaTeX，可独立编辑；尚未选定投稿模板。
- `references.bib`：十条可追溯文献，MTGFlow extension 以已读预印本身份引用，不冒称正式期刊版本已核。
- `citation_sources.md`：正式来源及精读范围。
- `method_definition.md`：当前唯一方法的结构、公式和权限，附历史候选定义。

2026-11-15 冻结的历史实现 `d77252e1` 使用交替时间／正交 Haar 域的可逆联合密度学习和固定 Student-t(5) 潜变量先验。九组实际训练与评价结果保留。[训练表](../../docs/experiments/normality_density_20261115/training_summary.csv) · [固定评价](../../docs/experiments/normality_density_20261115/all_metrics.csv)。

本文**没有**宣称论文贡献已经成立。实际结果中PSM排序有正项，但持续检测未改善；SWaT正常误报不可接受；TEP尚不能超出简单先验或归因于条件关系学习。不能把这些结果改写成微弱异常、Self/Cross-lag或污染鲁棒性已经获证。Stage1/Z/U/F 与混合无标签适配保留为研究框架，当前阶段只验证基础学习与检测机制。PSM/SWaT/TEP 明确标为已曝光开发/机制数据。故障子集、完整事件、信息权限与强简单基线均保留。

编译方式（有 Pandoc 和 LaTeX/BibTeX 环境时）：

```powershell
pandoc draft.md --standalone --natbib --top-level-division=section -V documentclass=article -V geometry:margin=1in -o draft.tex
pdflatex draft.tex
bibtex draft
pdflatex draft.tex
pdflatex draft.tex
```

本机可用 Pandoc，尚未找到 pdflatex，因此仅生成和检查可编辑 LaTeX，不声称已编译 PDF。原生公开数据来源的正式数据集引用尚需从对应已冻结 release manifest 的确切版本补入；当前未杜撰数据集论文作者/DOI。
