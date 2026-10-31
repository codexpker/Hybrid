# 物理scale验证后端修订

本修订基于已推送B基线`e004350`，仅纠正验证参考后端和原执行证据绑定。模型/数据/协议核心、training、infer_shard、collect_forecasts、source calibration、simple CPU canonical、原生score/阈值/RNG及3600银行均不改。原容差仍为`rtol=2e-6, atol=2e-6`，物理target/index/origin仍逐值exact。

## 原失败与数值边界

旧CPU参考不能作为CUDA银行的跨后端身份判定器。独审原`scale_diagnostic.json`在fault06_run031的172个B1×100×52历史窗确认：原GPU preprocess与九银行scale逐位一致；CPU参考116格不符。常值float32输入3000.10009765625的CPU/GPU归约产生不同mean及scale。GPU scale相对float64诊断最多偏约1.185%，检测影响未知；不能称其科学上无害，也不将float64换成新normalizer。

原始`RAW_SCALE_GATE_FIRST_COUNTEREXAMPLE.json`、`RAW_ARCHIVE_AUDIT_FAILURE_v1.json`和独审原件保持不变。本次验证修正不改变已产生分数，不重跑网络，不提高容差，不把工程PASS升级为检测GO。

## 显式合同与来源链

真实TEP的schema4 seal/evaluate必须显式包含：

```json
"physical_validation": {
  "backend": "original_cuda_a40_float32_v1",
  "device": "cuda:0"
}
```

参数固定进入现有operation manifest哈希，任何更改需新批准。该版本枚举固定当前验证设备为NVIDIA A40/compute8.6、torch2.8.0+cu128、CUDA12.8、无autocast、逐窗B1×100×52 float32。缺CUDA、设备/版本不符或未知backend明确拒绝，不CPU回退、不尝试多个后端选通过者。SYNTHETIC省略参数继续旧CPU路径；真实TEP不能使用该默认。

seal的`inputs.execution_evidence`是外置`PREPROCESS_BACKEND_PROVENANCE_CANDIDATE.json`的path/SHA record。它是已批准operation的一个固定输入，自身不是审批。共享`engine.physical_validation`从原件复核：原A代码identity及23源码字节；原infer配置/合同/approval的规范digest；pilot原approval兼release或remaining独立release；批准/放行早于launch；实际启动命令中的解释器、PYTHONPATH、CUDA_VISIBLE_DEVICES和逻辑device；全部shard与whitelist、终态stdout JSON、银行/RNG/metadata描述符的完整无重复覆盖。银行实际SHA与内容仍由原门禁另核。

原五job缺独立OS wait-exit receipt、job时GPU UUID/model及torch/CUDA版本遥测，不补造。以原终态JSON/log和冻结输出证明完成资产，不能宣称独立实测exit0；原先同解释器环境记录只作先前来源引用，不冒作每job测量。当前验证必须实测固定A40/版本；即使将来400实体参考全匹配，也不能补证历史硬件遥测。

CUDA seal记录同一`physical_validation`、`execution_evidence`及原`source_shards` record。evaluate的参数须与seal一致，且读取label前重新核同一来源链和每个物理银行。新参考仍逐窗调用原`HybridPredictor.preprocess`，不构造模型或调用forward；目标和simple评分继续CPU原公式。CPU合成seal不增加生产来源链约束，保留旧正反路径。

## 本轮证据与未执行项

已有远端环境、CUDA不可见、线程2，A9+新8+B5共22项合成CPU回归全部通过；B正路径保留72指标/逐实体Aff/原始坐标和2000共同抽样。新测试核CPU常值原公式和RNG、scale篡改、未许可/不可用backend、伪造原launch/op/log/source/bank/release及审批哈希绑定。metadata fixture显式mock原代码identity pin，仅验证拒绝逻辑，不能代替真实CUDA或原件证明。

另对真实五shard候选作纯标准库静态来源检查：3600银行描述符链通过，未导入torch、未打开NPZ/query数组/标签、CUDA调用0。这仅证明metadata/SHA链一致，不证明GPU scale已匹配。

首次回归v1的3处负测被新增CPU来源引用提前拦截，拒绝原因与原断言不符；修订为只对CUDA seal绑定该链，原A测试不改。v1失败、v2的17项正回归及最终v3的22项回归分别保留，详见[冻结修订记录](provenance/physical_backend_revision.json)。

仍待独审及主另批：新实现的真实A40合成常值/zero/ramp正反例、CUDA RNG不变和scale篡改拒绝；完整400实体共68800窗的当前后端物理参考及实际成本。不得把CPU测试中的固定GPU诊断字面量拒绝或mock runtime通过写成CUDA验证通过。真实seal和label评价仍HOLD；无真实query/label读取、GPU、训练或重新推理由本轮执行。
