# 处理版公开数据的只读query转换

共享入口：`python -m hybrid_tsad.hybrid_v1.processed_query --config <json>`。默认只检查配置，不打开输入。实际转换必须同时提供 `--execute --release <root-json> --trusted-release-sha256 <external-root-sha>`，操作为 `HYBRID_V1_PROCESSED_PUBLIC_QUERY_INPUT_DERIVATION`；release状态必须 `PRIMARY_RELEASED_PUBLIC_QUERY_INPUT_DERIVATION`、approved_by=root、精确config/source/permissions。

这是通用feature-major CSV转换，不是MSL科学driver。配置显式绑定input path/bytes/SHA、header、block_rows、feature_names、primary_feature_indices、split、fresh output及metadata_assets。只将主列test部分转float64再cast float32；NaN保留且observedfalse，Inf/overflow拒绝。train和非主列只核date/cols/行数，不numeric解析；CSV迭代恰好停在最后特征行，绝不请求下一label记录。全文件SHA会读取嵌入标签字节，仅作身份；不能写成物理上完全未读取标签字节。

无scaler/impute/训练/模型或GPU。输入源不改，输出已存在拒绝，错误保留fresh目录下failure记录，不自动retry。NPZ有values/observed/processed_index/test_index；INPUT_MANIFEST明确QUERY/匿名处理序列、format=npz、key=values、observed_key=observed、bounds、expected_shape、NPZ SHA、original_start=0，直接支持旧data.load_manifest。处理全局轴另存，不能冒原生实体或物理时钟认证。

MSL当前根事前决定是官方factory所选col_0/C1、test局部0..73728、处理全局58317..132045；后续模型仍H512，因此候选评分[512,73729)73217点，mask=seed+test_index。27旧label-derived边界只透明审计，不另分段选成绩。该模块不运行任何窗口或评价，也不能借C1检验Cross-lag。

准备包先封源码/13项合成测试清单，测试只用临时小CSV和synthetic root授权JSON，没有真实数据/权重。真实127MB转换另由外层owned CPU壳限制180+5秒、AS4GiB、CORE0、线程1、CVD空，root精确签件前不执行。历史data/readout/model/scoring及旧快照不变。
