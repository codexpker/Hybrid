# 预 label 资源 v3 窄修订

依据资源补充合同 SHA `4f57994daa517278ea6456c00a447643b61ca74c79025488604e5702b66b020b`。原合同及 v2 冻结（7f4c1032…）、34 Python 原字节 raw ZIP、独审与主接受保留不改。

唯一生产代码差分：`evaluation_limit_seconds(profile)` 仅按固定 name 返回 SYNTHETIC=300 / TEP100=900，其他 profile 拒绝；原超限保护从硬 300 改用该函数。2 GiB 不变，未来外层 timeout/终止宽限仍由主另签。模型、分数、阈值、事件、rank、统计、抽样矩阵、门禁、标签许可及数据集合均不变。900 只是预先固定的资源上限，不是全 400 实体吞吐实测，也不是实际标签评价 release。

CPU 8/8 回归：原 7 工程用例全部重跑，加 profile 300/900 选择、未知/大小写错误拒绝，以及真实合成 `run_evaluation` 路径模拟 301 秒后留下 failure 且无 summary。未运行真实 TEP 路径。使用现有环境和外部官方 Aff，不安装依赖；3.835 秒、RSS 150,994,944 bytes、地址空间 2 GiB、CUDA 空且未导入 Torch。

`VALIDATION.json` 逐 SHA 核 v2 数学/统计/旧核未变，并对 `run_evaluation` AST 证明仅替换限时 if。`candidate.patch` 相对 v2 原字节；`EVALUATION_RUNTIME_IDENTITY.json` 是新的完整 34 文件身份。v2 原件不被覆盖，不修改旧读出快照或运行中 runtime。
