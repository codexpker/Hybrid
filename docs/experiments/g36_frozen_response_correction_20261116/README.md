# G36冻结时频表示与正常响应修正：一次固定匹配验证

用户2026-11-16批准。沿用原G36 split/scaler、已训练LEFT selected2100、冻结条件Ridge及全部目标/故障；无神经重训、无新数据搜索，旧模型与停止结论不变。

## 固定预测与表示

输入52点，前48点9通道作为严格过去H，后4点的4响应作为Y。已有条件Ridge R(H)直接重放冻结coef/intercept。LEFT只输入H，全部参数/模块eval、原型与step冻结；forward hook在原`model.fusion`取输出Ttok_f/Ftok_f，分别时间平均后concat。该位置在prototype/memory混合之前，捕获后不能从当前Y补充特征。

**融合后Ttok_f自身含跨视图信息。** 原设计指定融合后提取，故本轮保持该位置，把消融准确称为`T-token-only`（F槽置零）对`T+F-token`；它检验附加F-token读出的价值，不等价于整个网络frequency-off，也不能独立证明纯频域贡献。不得因结果换成融合前层/其他池化。

FIT全1101合法端点用于一次冻结特征提取与修正头拟合；SELECT237端点用于固定正常预测比较。特征每维仅用FIT mean/std标准化（floor1e-6），两头共用标准化，T-only在标准化后F槽置0。无头截距。

`Yhat_std = frozen_Ridge(H_std) + phi_std @ W.T`。

只拟合W，目标为 `sum(error_std²)/(N*16) + lambda*||W||F²`，lambda=1。**对应误差总和形式的闭式矩阵为Phi.T Phi + N*16 I；N=1101时alpha=17616。** 不能误用软件默认sumloss的alpha=1。该尺度在任何结果开放前固定，两头相同；不搜索正则化、维数、特征层或截距。

原尺度误差通过原FIT响应scale反映射报告，检测分数为16个受监督标量的FIT标准化残差平方均值。没有当前窗口重构分数、LEFT native score融合或自由概率密度变换。

## 正常门槛与固定评价

先只访问FIT/SELECT，保存特征/原index/两个W/标准化参数/冻结Ridge预测/修正预测/原尺度与标准化误差。只有T+F修正在正常SELECT上逐标量MSE严格优于zero-correction Ridge和T-only，才放行本轮CAL与完整QUERY。否则停止，不看QUERY救援，不调lambda/特征/头。

通过后仍使用同一模型/同一两头，一次冻结读出CAL237及4 QUERY各7954点；零修正直接复用原Ridge bank。每方法用同一合法CAL的自身q99/higher，strict>，C3原第三点起生效、不回填。保留LEFT、历史Ridge原银行作为外部参照，不重推。健康副本仅后验诊断，全部3故障/完整轴/模式不一致均保留。

正常预测改善不是异常检测贡献。即使正常门槛通过，若未来正常误报过高、故障也被预测吸收或不能超简单对照，不能宣称Hybrid创新成立。成熟LEFT表示及Ridge/线性回归均不称原创；本轮不声称完整时频消融、多源预训或Stage2-U/F、污染鲁棒。

只执行上述一个版本。无增量即停止，不恢复其他已停止模型，不扩展变体。资源仅必要冻结推理和CPU闭式求解，记录实际成本与原资产SHA，先原始结果封存再评价。
