# A冷启动CUDA资源初始化修订

2026-11-01。主明确解除等待后，仅修`infer_shard`的设备setup：CUDA分支先`set_device`选择签名配置中的设备，显式`init`，然后reset allocated peak，再加载模型。CPU分支不调用这三项。没有随机warmup、额外前向或collect/RNG/评分改动；metadata原计时范围和descriptor含输出I/O的whole秒数保持。

原`033e8d5a…`快照及cold设备错误证据保留；已审A三类失效与修订也保留。本次增加实际wrapper的合成mock顺序测试，确认select→init→reset→model，且CPU分支若触碰CUDA setup即失败。实际模型为CPU fixture，没有调用CUDA backend；不能把mock通过说成真实GPU冷启动已实测。

既有资源字段/替换clock后数组与RNG不变、源绑定、目标物理绑定、18控制单位、完整组及标签锁回归全部继续通过。远端新目录`projects/hybrid_ad_package_validation_20261031_readout_cuda_setup_v1`，CUDA不可见、线程2、60秒上限，9/9 PASS，5.932秒；本机1 PASS、8因torch缺失SKIP，0.040秒，compileall通过。无实际GPU、真实目标/标签、训练或生产run修改。

精确diff及源码/测试hash见`readout_cold_cuda_revision.json/patch`。B保持未验收草稿，当前只交该小修回审；三核心、training、collect、source_gate未编辑。
