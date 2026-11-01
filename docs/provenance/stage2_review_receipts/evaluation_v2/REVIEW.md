# CPU评价候选独立有限审查

2026-11-01。结论：**PASS_LIMITED_SYNTHETIC_CPU / ACCEPTABLE_FOR_NEXT_STAGE**。仅批准本候选作为待主接受的CPU评价工程；不释放真实query、推理、分数或标签，不代表真实完整gate通过。

审查者未参与本评价实现，独立编写断言并运行；与主代理属同模型家族、共享workspace，不是外部或双盲审查。复用开发Fixture仅构造合成布线；261组差分、完整20×20×3统计手算、门禁负例与明确跨起点构造的预期值均由本审查独立断言。未以开发7/7代替独审，未修改候选源码、模型或Git。

## 锁定对象

- FREEZE：`7f4c1032c753c4c68d09cf610739043e26fafae46ce2e2b0c5864046f395d233`。
- candidate.patch：`b1a08bdf2e32b728648144f932dd140a5bd958e8c8691194034032112603f9e6`。
- 完整raw ZIP：`9f5ba263b0e3831c0d1c05c407d2fefad19b1bef3bea56a04828003a1ff96900`，48成员中freeze列文件和全部34py逐SHA一致。此前开发测试source ZIP 68bfc329…是不同容器，未误认为源码变更。
- 新34py runtime identity：`852d5878c2dce9a72c2d4fff3c50b908c1c2a631e3c5d544f47b9386dfd4fa1c`。
- 原32py只读snapshot identity：`076d9bca571f8fb56d228e30dad7d4ef51db42f437c010f55aaadc1ea9d5f665`，逐文件仍一致；未运行其真实gate。
- 合同71fde43d…和计数勘误9c247113…，无模型/score核改动。完整SHA及文件表见source_lock.json。

## 独立结果

1. `threshold_only`对原`evaluate_fixed_scores`用实际外置官方Aff六文件作261组全结构差分，输出完全相等。覆盖严格`score > threshold`、相等阈值、无报警/合法undefined、全报警、预报警、跨起点、真实索引gap和不完整事件。保留原块级Aff与全局NA，未拼不连续索引，也未为缺评点补正常。
2. 独立构造完整400合成entity、3seed、全部stage/init/mode/track，共25200指标行。逐run手算六个AP差与2000×20共同draw的bootstrap均一致；不是pooled AP。缺格、重复、未知init均拒绝。bootstrap draw精确来自default_rng(99174)。此为统计内核合成测试，不是400真实实体全流程资源测试。
3. 20fault重复前缀分母由60小时重复量还原为单run3小时，与FP和片段数一致；score不同而binary相同允许正确去重；binary冲突则unique计数/时间均NA，没有代表、union或均值替代。
4. 三个eligible中2 detected/1 miss、另1 incomplete，detected-only median为4.5min，miss未填零，全eligible mean保持NA。另明确全程预报警跨故障起点，source delay=0同时pre_fault与cross_start为真；target阈值导致miss时delay仍None/NOT_DETECTED，两轨AP/ROC相同。
5. PENDING主release、PENDING gate acceptance、实际OS wait code=124且链重新绑定三类负例，都在`np.load`调用前拒绝。源码检查另见生产分支要求主接受状态、gate输入链/规范operation审批、原32py/runtime、summary/seal/OS/resource身份；合成release使用单独scope，不能被描述为真实gate。
6. 完整4entity合成授权路径只调用144次rank helper（108 native=27×4；36旧控制=9×4），共252两阈值/单阈值行。未用第二次rank帮助target；旧late/history_mean/persistence仅source。候选输入身份在远端运行前后逐34py SHA未变，未导入Torch。

本独立主体最后一次运行3.174007184978109秒、峰RSS179503104字节；RLIMIT_AS=2GiB及外层timeout=300秒，CUDA_VISIBLE_DEVICES为空。输出raw_tests.json、statistics_oracle.json、integration_result.json及可重运行independent_tests.py；本地lock_sources.py只读源码/元数据/归档锁身份。

## 范围与限制

|项目|结论|
|---|---|
|CPU合成实现/数学装配|PASS，有限范围|
|CPU合成数据协议与门禁负例|PASS，有限范围|
|真实完整seal、原snapshot实际gate、真实OS链及独审主接受|INCOMPLETE，未执行|
|真实标签release/实际AP、误报、事件/时延|INCOMPLETE，未授权、未读取|
|完整真实400资源/推理预算|INCOMPLETE，4entity布线与400统计合成不替代|
|检测收益可信度|NONE，当前没有真实检测证据|
|创新贡献|UNSUPPORTED，未在本轮评估|

外置Aff只使用已有六文件精确SHA；许可仍UNVERIFIED_EXISTING_LOCAL_DEPENDENCY，redistribute=false，未下载、复制重分发或宣称许可获核。生产gate真实性最终仍须由未来实际原snapshot运行、独审原件和主接受建立；本轮JSON合成输入不提供此事实。未来真实评价还须单独绑定完整seal、真实gate接受、evaluation runtime/config/operation及显式label release。不能把本CPU PASS转换为真实标签许可或科研GO。

审查没有发现本有限候选需要阻断的数学反例。候选可交主审作CPU工程接受；整体科学HOLD、创新UNSUPPORTED保持。
