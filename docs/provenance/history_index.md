# 外置历史资产索引

路径相对本工程根目录；不移动、复制或覆盖原件。不把历史负结果隐藏为“清理”。

|资产|原位置|用途|
|---|---|---|
|原三个核心|`../src/hybrid_tsad/`|本包逐字节复制来源|
|最小实现与L0|`../results/hybrid_preflight_20261031_v1/`|实现检查，不是检测收益|
|冻结query合同|`../results/hybrid_protocol_freeze_20261031_v1/`|输入/原索引/事件/覆盖|
|2k源开发|`../results/hybrid_stage1_development_20261031_v1/`|探索训练原件|
|PSM探索读出|`../results/hybrid_psm_readout_20261031_v1/`|保留正负检测结果|
|历史容量控制|`../results/hybrid_history_controls_20261031_v1/`|简单/随机对照与反例|
|10k波次|`../results/hybrid_10k_wave1_20261031_v1/`、`../results/hybrid_10k_wave2_20261031_v1/`、`../results/hybrid_10k_wave3_20261031_v1/`|source限定验收及manifest|
|新读出元数据|`../results/hybrid_10k_psm_readout_20261031_v1/`|按治理入口查恢复与状态，不在此重复裁决|
|原训练runner|`../scripts/hybrid_10k_wave.py`|外置legacy bridge，固定SHA|
|原读出runner|`../scripts/run_hybrid_10k_psm_inference.py`|外置legacy bridge，固定SHA|
|其余历史脚本|`../scripts/`|仅追溯原合同及实验，不纳入新包主入口|
|正式合同|`../experiments/contracts/`|批准边界，配置验证不代替审批|
|当前科研状态|`../CURRENT_RESEARCH_STATE.md`|状态唯一入口|
|治理与登记|`../RESEARCH_PROTOCOL.md`、`../RESEARCH_HYPOTHESES.md`、`../EXPERIMENT_REGISTRY.csv`|科研主张和任务等级|

远端权重/数组/完整run仍位于原外置存储；各结果目录的`REMOTE_RUN.txt`、`RUN_COMMANDS.json`及manifest记录原地址。历史实现快照中的核心版本可能与当前副本不同（例如后加temporal_depth）；新包只声明匹配2026-10-31当前三个核心，不把所有历史快照倒标为同一版本。
