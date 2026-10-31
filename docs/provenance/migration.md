# 来源、迁移边界与工程验收

记录日期：2026-10-31。仅本次工程阶段，非科研审查结论。

## 字节身份

从研究工作区当前`src/hybrid_tsad`复制到本包同名模块；未格式化或编辑核心。原始路径仍在。包内`core_provenance.json`为可机器核验映射，安装后的`verify-core`可以不依赖原目录校验包内副本；加`--source-root`核原件。

|核心文件|SHA256|
|---|---|
|model.py|`03022d55353f502c18aec31b379a14dbd0fd7964be0ac198dee6bf644810da41`|
|data.py|`2c6b145166f4d5f900b26e7e05e8e823fee8ffdd33fc90ec0bfb8a1fcd4e8328`|
|protocol.py|`f663b74af0758e7d6b604daac54e24ca45a9724fa4878c6421a6975b7152d3ed`|

外部TimeRadar模型文件固定SHA：`881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317`。本工程不附带该文件，其导入仍由原模型执行。上游及本项目代码的再分发权尚未作为本次任务核定，不新增许可证。

外置runner身份：source-wave为`00ed75a4c1da76600d573be67dead4179b455131610607d7b1cea3bb0dc8e394`；readout为`ce37f1f025c5ad29e3ffe537d582f703fed39f5d7650bdd1b33141e5c67b3866`。桥接只在身份匹配时转交命令，不修改脚本。

已只读核对远端`implementation_v2/scripts`中readout的SHA就是上述ce37f1版本；已知失败v1为`3cd868ec922bbf276bea4e2bf255903697134ad89466eb8e4a0ed5718195a0ea`，桥接不会接受该版本。远端已有依赖metadata为`torch-frft==0.8.1`，已纳入model extra及依赖发现，无新安装。

## 已迁入

- src布局、可安装元数据、统一CLI、非训练配置验证和外置run元数据检查。
- 三个当前核心的原样副本。
- 真正的模型构造/配置构造/预测API，strict旧state_dict加载。
- 对原训练与读出runner的显式外置legacy桥；默认预览，不自动启动。
- 依赖发现、来源hash与可选CPU兼容测试。

## 尚未迁入

- 源训练采样trace、源select/cal、优化循环的包原生调度；目前仍从原run implementation加载。
- 完整推理银行与恢复守卫、标签开锁与科学指标统计的一体化命令。
- 通用manifest路径重定位、目标无标签/事件少样本适配的统一实现。
- 冻结10k权重与原银行逐点等价验证。此次用相同初始化的合成输入验证API兼容，不冒称全量科学结果复现。

后续迁移应分别冻结源采样与RNG、输入预处理、state_dict、原始预测、校准和时间索引，先比较再替换原入口；不能因包可以安装就删除历史脚本或改动运行中的run。

## 实际验证

本机已有Python3.12.14、setuptools/pip、NumPy2.3.5、Pandas3.0.1，无torch；无需新装依赖。标准库工程测试11项通过，模型兼容测试本机1项条件跳过。compileall通过。wheel成功构建，以`--no-deps --no-index`安装本包到独立临时target，安装包CLI的check-env、verify-core、config validate、inspect-run均通过；未读取旧分数/标签/权重。

经主授权额外远端CPU检查：新建`projects/hybrid_ad_package_validation_20261031`目录，仅传入本包源码/测试；使用已有`envs/timeradar_author_core_20261028_r2/bin/python`。`CUDA_VISIBLE_DEVICES=''`、torch线程2，未调用GPU、无训练/优化。直接从外置旧`projects/hybrid_v1/src/hybrid_tsad/model.py`导入旧类，与包API对比；旧model SHA等于上表。PyTorch2.8.0+cu128，三模式temporal/tf/hybrid分别166个state_dict键，strict加载与临时checkpoint权重加载通过；相同种子41初始化、53预测，同一`[1,400,1]`线性合成历史，预测`[1,5,1]`逐元素精确相同。单测试0.972秒；临时checkpoint由测试自动清理。结构化证据见`validation_results.json`。

可重放检查（在另一个新验证目录，路径相对外置research root；不要覆盖本次目录）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
HYBRID_LEGACY_MODEL="$RESEARCH_ROOT/projects/hybrid_v1/src/hybrid_tsad/model.py" \
HYBRID_OFFICIAL_FILE="$RESEARCH_ROOT/runs/few_shot_baseline_20261030_v1/official_source/models/TimeRadar.py" \
timeout 60s "$RESEARCH_ROOT/envs/timeradar_author_core_20261028_r2/bin/python" -c \
'import torch,unittest; torch.set_num_threads(2); suite=unittest.defaultTestLoader.discover("tests",pattern="test_model_compatibility.py"); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

此验收由工程开发者执行，属于工程自检，不称未参与实现的独立科研审查。
