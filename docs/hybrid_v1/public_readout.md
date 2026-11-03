# 公共处理域的 source-only 读出

共享入口 `python -m hybrid_tsad.hybrid_v1.public_readout --config <json>` 默认仅配置预览，不hash资产或打开数组，不构模。实际需 `--execute --release <root-json> --trusted-release-sha256 <external-sha>`。无resume参数，输出必须全新；操作分别为 PUBLIC_SOURCE_ONLY_MODEL_READOUT、PUBLIC_SOURCE_ONLY_BASELINE_READOUT、PUBLIC_SYNTHETIC_C1_COST（完整前缀见模块常量）。

它复用既有readout的 `named_binding_metadata` 六资产SOURCE_CONTROL链、`write_model_bank`、`write_baseline_bank`、`_state`/RNG，以及cli的 `load_model/save_runtime/score_records`；原PSM guard和22源字节不变。新冻结副本是旧22文件加这一adapter，不能把变动的工作树当冻结运行包。

当前真实profile只接受已root接受的MSL col0 NPZ/manifest、匿名单处理序列73729×1、original_start0、QUERY。H512/P32/chunk64、73217点[512,73729)，包含最后31点，mask=source_seed+test_index；处理全局轴不是Run.start。27旧候选边界不切分，不另出分段成绩；没有物理时钟/原生实体/Cross机制主张。

原完整三seed2025/26/27的checkpoint和ownsourceCAL逐SHA固定，不借H5或targetCAL。简单baseline persistence/history_mean512各一次，复用原源CAL，不伪训练seed或重新拟合。每个真实操作还要root精确预算/资产seal/release，工程通过不授权真实query。

合成成本是独立operation：仅C1/rng84551/N640/2025原权重/128到达点512..639/chunk64。先cold主writer128点；再恢复主起始RNG，以原V4 40e224训练文件的infer接口做固定首3点raw/RNG/state比较；最终恢复主完成RNG。总131个点级前向都计成本，不重复训练。保存两chunk、raw、点时钟、runtime、state/RNG、冷加载/score与bankIO/finalize、allocator/RSS/吞吐成本估算。175TERM+5清理硬总180秒=.05GPUh、30GiB，无retry；模型只eval、零更新，主阈值原样复用，不能走入真实MSL数组。GPU仍需另签。
