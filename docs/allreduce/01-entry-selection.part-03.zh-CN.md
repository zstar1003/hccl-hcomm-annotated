# AllReduce逐行对照：入口、引擎与算法选择

[返回阅读指南](../READING_GUIDE.zh-CN.md)。S为审读快照行号；L为带本次逐行注释的源码行号。每个L链接定位到固定源码提交；长语句按物理行分别说明。空行及原注释不重复注释。

审读快照：`f8af6a36831195a8440de6ec6183856bb72af907`；源码提交：`824a8a80731bd66ef6eb78891b9aec0acfca281c`。

[返回本阶段函数导航](01-entry-selection.zh-CN.md)。第3/3页。

## 41. AutoSelectorBase::CalcFrameNum

用 level0 实例规模的最大公约数推算框数。

完整范围：[auto_selector_base.cc:L228–L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L228-L272)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

功能与分支：

- [S131 / L234](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L234)：分支条件为 topoInfo->topoLevelNums 不超过 1 或 topoInfo->当前网络层编号Details.instSizeListOfLayer[0].empty(；成立进入本块，未成立继续后续分支。

- [S135 / L242](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L242)：在 CalcFrameNum 中遍历 (size_t i = 1; i 小于 topoInfo->当前网络层编号Details.instSizeListOfLayer[0].size(); ++i 指定的集合或索引区间；边界/迭代规则为 (size_t i = 1; i 小于 topoInfo->当前网络层编号Details.instSizeListOfLayer[0].size(); ++i。

- [S138 / L248](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L248)：在 (b 不等于 0 条件下继续重复当前处理。

- [S144 / L260](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L260)：分支条件为 gcd 等于 1；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S128 / L228](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L228) | <code>u32 AutoSelectorBase::CalcFrameNum(const TopoInfoWithNetLayerDetails* topoInfo)</code> | 定义 CalcFrameNum 入口：用 level0 实例规模的最大公约数推算框数。 |
| [S129 / L230](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L230) | <code>{</code> | 进入 CalcFrameNum 的实现作用域；用 level0 实例规模的最大公约数推算框数。 |
| [S130 / L232](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L232) | <code>    u32 frameNum = 0;</code> | 设置 frameNum 为 0；该值供下方当前分支使用。 |
| [S131 / L234](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L234) | <code>    if (topoInfo-&gt;topoLevelNums &lt;= 1 &#124;&#124; topoInfo-&gt;netLayerDetails.instSizeListOfLayer[0].empty()) {</code> | 分支条件为 topoInfo-&gt;topoLevelNums 不超过 1 或 topoInfo-&gt;当前网络层编号Details.instSizeListOfLayer[0].empty(；成立进入本块，未成立继续后续分支。 |
| [S132 / L236](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L236) | <code>        return frameNum;</code> | 直接返回 frameNum，调用者取得本分支结果。 |
| [S133 / L238](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L238) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S134 / L240](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L240) | <code>    u32 gcd = topoInfo-&gt;netLayerDetails.instSizeListOfLayer[0][0];</code> | 设置 gcd 为 topoInfo-&gt;当前网络层编号Details.instSizeListOfLayer[0][0]；该值供下方当前分支使用。 |
| [S135 / L242](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L242) | <code>    for (size_t i = 1; i &lt; topoInfo-&gt;netLayerDetails.instSizeListOfLayer[0].size(); ++i) {</code> | 在 CalcFrameNum 中遍历 (size_t i = 1; i 小于 topoInfo-&gt;当前网络层编号Details.instSizeListOfLayer[0].size(); ++i 指定的集合或索引区间；边界/迭代规则为 (size_t i = 1; i 小于 topoInfo-&gt;当前网络层编号Details.instSizeListOfLayer[0].size(); ++i。 |
| [S136 / L244](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L244) | <code>        u32 a = gcd;</code> | 设置 a 为 gcd；该值供下方当前分支使用。 |
| [S137 / L246](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L246) | <code>        u32 b = topoInfo-&gt;netLayerDetails.instSizeListOfLayer[0][i];</code> | 设置 b 为 topoInfo-&gt;当前网络层编号Details.instSizeListOfLayer[0][i]；该值供下方当前分支使用。 |
| [S138 / L248](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L248) | <code>        while (b != 0) {</code> | 在 (b 不等于 0 条件下继续重复当前处理。 |
| [S139 / L250](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L250) | <code>            u32 r = a % b;</code> | 设置 r 为 a % b；该值供下方当前分支使用。 |
| [S140 / L252](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L252) | <code>            a = b;</code> | 设置 a 为 b；该值供下方当前分支使用。 |
| [S141 / L254](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L254) | <code>            b = r;</code> | 设置 b 为 r；该值供下方当前分支使用。 |
| [S142 / L256](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L256) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S143 / L258](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L258) | <code>        gcd = a;</code> | 设置 gcd 为 a；该值供下方当前分支使用。 |
| [S144 / L260](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L260) | <code>        if (gcd == 1) {</code> | 分支条件为 gcd 等于 1；成立进入本块，未成立继续后续分支。 |
| [S145 / L262](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L262) | <code>            break;</code> | 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。 |
| [S146 / L264](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L264) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S147 / L266](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L266) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S148 / L268](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L268) | <code>    frameNum = (gcd &gt; 0) ? topoInfo-&gt;userRankSize / gcd : 0;</code> | 设置 frameNum 为 (gcd 大于 0) ? 通信域 Rank 总数 / gcd : 0；该值供下方当前分支使用。 |
| [S149 / L270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L270) | <code>    return frameNum;</code> | 直接返回 frameNum，调用者取得本分支结果。 |
| [S150 / L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L272) | <code>}</code> | 结束 CalcFrameNum 实现；其返回状态或已写回字段由调用者接收。 |


## 42. AutoSelectorBase::SelectCcuMsAlgo

基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。

完整范围：[auto_selector_base.cc:L275–L293](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L275-L293)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S152 / L275](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L275) | <code>SelectorStatus AutoSelectorBase::SelectCcuMsAlgo(</code> | 定义 SelectCcuMsAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S153 / L277](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L277) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,</code> | 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,；引用参数按声明的 const 限制读写。 |
| [S154 / L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L279) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName) const</code> | 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S155 / L281](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L281) | <code>{</code> | 进入 SelectCcuMsAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S156 / L283](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L283) | <code>    (void)opParam;</code> | 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S157 / L285](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L285) | <code>    (void)topoInfo;</code> | 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S158 / L287](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L287) | <code>    (void)configAlgMap;</code> | 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S159 / L289](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L289) | <code>    (void)selectAlgName;</code> | 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S160 / L291](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L291) | <code>    return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S161 / L293](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L293) | <code>}</code> | 结束 SelectCcuMsAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 43. AutoSelectorBase::SelectCcuScheduleAlgo

基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。

完整范围：[auto_selector_base.cc:L296–L314](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L296-L314)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S163 / L296](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L296) | <code>SelectorStatus AutoSelectorBase::SelectCcuScheduleAlgo(</code> | 定义 SelectCcuScheduleAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S164 / L298](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L298) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,</code> | 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,；引用参数按声明的 const 限制读写。 |
| [S165 / L300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L300) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName) const</code> | 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S166 / L302](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L302) | <code>{</code> | 进入 SelectCcuScheduleAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S167 / L304](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L304) | <code>    (void)opParam;</code> | 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S168 / L306](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L306) | <code>    (void)topoInfo;</code> | 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S169 / L308](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L308) | <code>    (void)configAlgMap;</code> | 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S170 / L310](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L310) | <code>    (void)selectAlgName;</code> | 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S171 / L312](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L312) | <code>    return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S172 / L314](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L314) | <code>}</code> | 结束 SelectCcuScheduleAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 44. AutoSelectorBase::SelectAicpuAlgo

基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。

完整范围：[auto_selector_base.cc:L317–L335](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L317-L335)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S174 / L317](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L317) | <code>SelectorStatus AutoSelectorBase::SelectAicpuAlgo(</code> | 定义 SelectAicpuAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S175 / L319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L319) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,</code> | 续接 SelectAicpuAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,；引用参数按声明的 const 限制读写。 |
| [S176 / L321](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L321) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName) const</code> | 续接 SelectAicpuAlgo 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S177 / L323](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L323) | <code>{</code> | 进入 SelectAicpuAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S178 / L325](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L325) | <code>    (void)opParam;</code> | 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S179 / L327](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L327) | <code>    (void)topoInfo;</code> | 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S180 / L329](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L329) | <code>    (void)configAlgMap;</code> | 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S181 / L331](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L331) | <code>    (void)selectAlgName;</code> | 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S182 / L333](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L333) | <code>    return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S183 / L335](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L335) | <code>}</code> | 结束 SelectAicpuAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 45. AutoSelectorBase::SelectAivAlgo

基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。

完整范围：[auto_selector_base.cc:L338–L356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L338-L356)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S185 / L338](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L338) | <code>SelectorStatus AutoSelectorBase::SelectAivAlgo(</code> | 定义 SelectAivAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S186 / L340](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L340) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,</code> | 续接 SelectAivAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,；引用参数按声明的 const 限制读写。 |
| [S187 / L342](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L342) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName) const</code> | 续接 SelectAivAlgo 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S188 / L344](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L344) | <code>{</code> | 进入 SelectAivAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S189 / L346](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L346) | <code>    (void)opParam;</code> | 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S190 / L348](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L348) | <code>    (void)topoInfo;</code> | 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S191 / L350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L350) | <code>    (void)configAlgMap;</code> | 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S192 / L352](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L352) | <code>    (void)selectAlgName;</code> | 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S193 / L354](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L354) | <code>    return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S194 / L356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L356) | <code>}</code> | 结束 SelectAivAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 46. AutoSelectorBase::SelectDPUAlgo

基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。

完整范围：[auto_selector_base.cc:L359–L377](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L359-L377)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S196 / L359](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L359) | <code>SelectorStatus AutoSelectorBase::SelectDPUAlgo(</code> | 定义 SelectDPUAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S197 / L361](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L361) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,</code> | 续接 SelectDPUAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,；引用参数按声明的 const 限制读写。 |
| [S198 / L363](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L363) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName) const</code> | 续接 SelectDPUAlgo 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S199 / L365](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L365) | <code>{</code> | 进入 SelectDPUAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。 |
| [S200 / L367](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L367) | <code>    (void)opParam;</code> | 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S201 / L369](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L369) | <code>    (void)topoInfo;</code> | 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S202 / L371](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L371) | <code>    (void)configAlgMap;</code> | 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S203 / L373](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L373) | <code>    (void)selectAlgName;</code> | 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S204 / L375](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L375) | <code>    return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S205 / L377](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L377) | <code>}</code> | 结束 SelectDPUAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 47. AutoSelectorBase::IsLayerAllConnetedWithTopo

检查指定层是否有一个给定拓扑实例覆盖本层全部本地 Rank；缺少层或拓扑记录返回 false。

完整范围：[auto_selector_base.cc:L380–L435](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L380-L435)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

功能与分支：

- [S210 / L386](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L386)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S218 / L401](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L401)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S226 / L416](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L416)：分支条件为 rankNumForTopoTypeItr 等于 topoInfo->topoInstDetailsOfLayer[当前网络层编号].rankNumForTopoType.end(；成立进入本块，未成立继续后续分支。

- [S230 / L423](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L423)：在 IsLayerAllConnetedWithTopo 中遍历 (auto topoRankNum : rankNumForTopoTypeItr->second 指定的集合或索引区间；边界/迭代规则为 (auto topoRankNum : rankNumForTopoTypeItr->second。

- [S231 / L425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L425)：分支条件为 topoRankNum 等于 localRankSize；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S207 / L380](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L380) | <code>bool AutoSelectorBase::IsLayerAllConnetedWithTopo(</code> | 定义 IsLayerAllConnetedWithTopo 入口：检查指定层是否有一个给定拓扑实例覆盖本层全部本地 Rank；缺少层或拓扑记录返回 false。 |
| [S208 / L382](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L382) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)</code> | 续接 IsLayerAllConnetedWithTopo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const u32 当前网络层编号, const CommTopo topoType)；引用参数按声明的 const 限制读写。 |
| [S209 / L384](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L384) | <code>{</code> | 进入 IsLayerAllConnetedWithTopo 的实现作用域；检查指定层是否有一个给定拓扑实例覆盖本层全部本地 Rank；缺少层或拓扑记录返回 false。 |
| [S210 / L386](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L386) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S211 / L388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L388) | <code>        topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer.size() &lt;= netLayer,</code> | 调用 size 完成当前参数所指的子步骤；本行实参为 topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer.size() 不超过 当前网络层编号,。 |
| [S212 / L390](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L390) | <code>        HCCL_WARNING(</code> | 开始 HCCL_WARNING 诊断输出，记录 IsLayerAllConnetedWithTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S213 / L392](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L392) | <code>            &quot;[BaseSelector][IsLayerAllConnetedWithTopo] localNetInsSizeOfLayer size[%u] &lt;= netLayer[%u]&quot;,</code> | 续接 IsLayerAllConnetedWithTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S214 / L394](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L394) | <code>            topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer.size(), netLayer),</code> | 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer.size(), 当前网络层编号，与前面的格式占位依次对应。 |
| [S215 / L396](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L396) | <code>        false);</code> | 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。 |
| [S216 / L398](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L398) | <code>    u32 localRankSize = topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer[netLayer];</code> | 设置 localRankSize 为 topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer[当前网络层编号]；该值供下方当前分支使用。 |
| [S218 / L401](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L401) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S219 / L403](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L403) | <code>        topoInfo-&gt;topoInstDetailsOfLayer.size() &lt;= netLayer,</code> | 调用 size 完成当前参数所指的子步骤；本行实参为 topoInfo-&gt;topoInstDetailsOfLayer.size() 不超过 当前网络层编号,。 |
| [S220 / L405](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L405) | <code>        HCCL_WARNING(</code> | 开始 HCCL_WARNING 诊断输出，记录 IsLayerAllConnetedWithTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S221 / L407](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L407) | <code>            &quot;[BaseSelector][IsLayerAllConnetedWithTopo] topoInstDetailsOfLayer size[%u] &lt;= netLayer[%u]&quot;,</code> | 续接 IsLayerAllConnetedWithTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S222 / L409](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L409) | <code>            topoInfo-&gt;topoInstDetailsOfLayer.size(), netLayer),</code> | 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：topoInfo-&gt;topoInstDetailsOfLayer.size(), 当前网络层编号，与前面的格式占位依次对应。 |
| [S223 / L411](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L411) | <code>        false);</code> | 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。 |
| [S225 / L414](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L414) | <code>    auto rankNumForTopoTypeItr = topoInfo-&gt;topoInstDetailsOfLayer[netLayer].rankNumForTopoType.find(topoType);</code> | 设置 rankNumForTopoTypeItr 为 topoInfo-&gt;topoInstDetailsOfLayer[当前网络层编号].rankNumForTopoType.find(topoType)；该值供下方当前分支使用。 |
| [S226 / L416](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L416) | <code>    if (rankNumForTopoTypeItr == topoInfo-&gt;topoInstDetailsOfLayer[netLayer].rankNumForTopoType.end()) {</code> | 分支条件为 rankNumForTopoTypeItr 等于 topoInfo-&gt;topoInstDetailsOfLayer[当前网络层编号].rankNumForTopoType.end(；成立进入本块，未成立继续后续分支。 |
| [S227 / L418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L418) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S228 / L420](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L420) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S230 / L423](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L423) | <code>    for (auto topoRankNum : rankNumForTopoTypeItr-&gt;second) {</code> | 在 IsLayerAllConnetedWithTopo 中遍历 (auto topoRankNum : rankNumForTopoTypeItr-&gt;second 指定的集合或索引区间；边界/迭代规则为 (auto topoRankNum : rankNumForTopoTypeItr-&gt;second。 |
| [S231 / L425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L425) | <code>        if (topoRankNum == localRankSize) {</code> | 分支条件为 topoRankNum 等于 localRankSize；成立进入本块，未成立继续后续分支。 |
| [S232 / L427](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L427) | <code>            return true;</code> | 当前能力/拓扑/匹配检查满足，返回 true。 |
| [S233 / L429](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L429) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S234 / L431](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L431) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S235 / L433](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L433) | <code>    return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S236 / L435](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L435) | <code>}</code> | 结束 IsLayerAllConnetedWithTopo 实现；其返回状态或已写回字段由调用者接收。 |


## 48. AutoSelectorBase::CheckMeshNumEqualToClosNum

检查第零层 Mesh/CLOS 实例规模并比较首个实例 Rank 数；缺少数据返回内部错误。

完整范围：[auto_selector_base.cc:L438–L477](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L438-L477)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

功能与分支：

- [S243 / L446](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L446)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S251 / L461](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L461)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S238 / L438](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L438) | <code>HcclResult AutoSelectorBase::CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails* topoInfo, bool&amp; isEqual)</code> | 定义 CheckMeshNumEqualToClosNum 入口：检查第零层 Mesh/CLOS 实例规模并比较首个实例 Rank 数；缺少数据返回内部错误。 |
| [S239 / L440](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L440) | <code>{</code> | 进入 CheckMeshNumEqualToClosNum 的实现作用域；检查第零层 Mesh/CLOS 实例规模并比较首个实例 Rank 数；缺少数据返回内部错误。 |
| [S240 / L442](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L442) | <code>    const auto&amp; topoInstDetails = topoInfo-&gt;topoInstDetailsOfLayer;</code> | 设置 &amp; topoInstDetails 为 topoInfo-&gt;topoInstDetailsOfLayer；该值供下方当前分支使用。 |
| [S243 / L446](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L446) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S244 / L448](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L448) | <code>        topoInstDetails.empty(),</code> | 调用 empty 完成当前参数所指的子步骤；本行实参为 topoInstDetails.empty(),。 |
| [S245 / L450](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L450) | <code>        HCCL_ERROR(&quot;[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero.&quot;),</code> | 开始 HCCL_ERROR 诊断输出，记录 CheckMeshNumEqualToClosNum 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S246 / L452](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L452) | <code>        HCCL_E_INTERNAL);</code> | 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。 |
| [S248 / L455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L455) | <code>    const auto&amp; rankNumMap = topoInstDetails[0].rankNumForTopoType;</code> | 设置 &amp; rankNumMap 为 topoInstDetails[0].rankNumForTopoType；该值供下方当前分支使用。 |
| [S249 / L457](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L457) | <code>    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);</code> | 设置 closItr 为 rankNumMap.find(COMM_TOPO_CLOS)；该值供下方当前分支使用。 |
| [S250 / L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L459) | <code>    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);</code> | 设置 meshItr 为 rankNumMap.find(COMM_TOPO_1DMESH)；该值供下方当前分支使用。 |
| [S251 / L461](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L461) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S252 / L463](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L463) | <code>        closItr == rankNumMap.end() &#124;&#124; closItr-&gt;second.empty() &#124;&#124; meshItr == rankNumMap.end()</code> | 调用 end 完成当前参数所指的子步骤；本行实参为 closItr 等于 rankNumMap.end() 或 closItr-&gt;second.empty() 或 meshItr 等于 rankNumMap.end()。 |
| [S253 / L465](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L465) | <code>            &#124;&#124; meshItr-&gt;second.empty(),</code> | 调用 empty 完成当前参数所指的子步骤；本行实参为 或 meshItr-&gt;second.empty(),。 |
| [S254 / L467](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L467) | <code>        HCCL_ERROR(&quot;[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero.&quot;),</code> | 开始 HCCL_ERROR 诊断输出，记录 CheckMeshNumEqualToClosNum 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S255 / L469](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L469) | <code>        HCCL_E_INTERNAL);</code> | 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。 |
| [S258 / L473](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L473) | <code>    isEqual = (closItr-&gt;second[0] == meshItr-&gt;second[0]);</code> | 续接 CheckMeshNumEqualToClosNum 当前语句的具体实参/字段：isEqual = (closItr-&gt;second[0] 等于 meshItr-&gt;second[0])；由其完整表达式完成参数组装、检查或结果写回。 |
| [S259 / L475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L475) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S260 / L477](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L477) | <code>}</code> | 结束 CheckMeshNumEqualToClosNum 实现；其返回状态或已写回字段由调用者接收。 |


## 49. AutoSelectorBase::CheckClosNumMultipleOfMeshNum

检查第零层 CLOS Rank 数是否严格大于且整除 Mesh Rank 数，Mesh 本身还需大于 1。

完整范围：[auto_selector_base.cc:L480–L526](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L480-L526)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

功能与分支：

- [S267 / L489](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L489)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S275 / L504](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L504)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S262 / L480](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L480) | <code>HcclResult</code> | 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。 |
| [S263 / L482](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L482) | <code>AutoSelectorBase::CheckClosNumMultipleOfMeshNum(const TopoInfoWithNetLayerDetails* topoInfo, bool&amp; isMultiple)</code> | 定义 CheckClosNumMultipleOfMeshNum 入口：检查第零层 CLOS Rank 数是否严格大于且整除 Mesh Rank 数，Mesh 本身还需大于 1。 |
| [S264 / L484](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L484) | <code>{</code> | 进入 CheckClosNumMultipleOfMeshNum 的实现作用域；检查第零层 CLOS Rank 数是否严格大于且整除 Mesh Rank 数，Mesh 本身还需大于 1。 |
| [S265 / L486](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L486) | <code>    const auto&amp; topoInstDetails = topoInfo-&gt;topoInstDetailsOfLayer;</code> | 设置 &amp; topoInstDetails 为 topoInfo-&gt;topoInstDetailsOfLayer；该值供下方当前分支使用。 |
| [S267 / L489](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L489) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S268 / L491](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L491) | <code>        topoInstDetails.empty(),</code> | 调用 empty 完成当前参数所指的子步骤；本行实参为 topoInstDetails.empty(),。 |
| [S269 / L493](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L493) | <code>        HCCL_ERROR(&quot;[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero.&quot;),</code> | 开始 HCCL_ERROR 诊断输出，记录 CheckClosNumMultipleOfMeshNum 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S270 / L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L495) | <code>        HCCL_E_INTERNAL);</code> | 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。 |
| [S272 / L498](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L498) | <code>    const auto&amp; rankNumMap = topoInstDetails[0].rankNumForTopoType;</code> | 设置 &amp; rankNumMap 为 topoInstDetails[0].rankNumForTopoType；该值供下方当前分支使用。 |
| [S273 / L500](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L500) | <code>    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);</code> | 设置 closItr 为 rankNumMap.find(COMM_TOPO_CLOS)；该值供下方当前分支使用。 |
| [S274 / L502](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L502) | <code>    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);</code> | 设置 meshItr 为 rankNumMap.find(COMM_TOPO_1DMESH)；该值供下方当前分支使用。 |
| [S275 / L504](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L504) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S276 / L506](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L506) | <code>        closItr == rankNumMap.end() &#124;&#124; closItr-&gt;second.empty() &#124;&#124; meshItr == rankNumMap.end()</code> | 调用 end 完成当前参数所指的子步骤；本行实参为 closItr 等于 rankNumMap.end() 或 closItr-&gt;second.empty() 或 meshItr 等于 rankNumMap.end()。 |
| [S277 / L508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L508) | <code>            &#124;&#124; meshItr-&gt;second.empty(),</code> | 调用 empty 完成当前参数所指的子步骤；本行实参为 或 meshItr-&gt;second.empty(),。 |
| [S278 / L510](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L510) | <code>        HCCL_ERROR(&quot;[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero.&quot;),</code> | 开始 HCCL_ERROR 诊断输出，记录 CheckClosNumMultipleOfMeshNum 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S279 / L512](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L512) | <code>        HCCL_E_INTERNAL);</code> | 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。 |
| [S282 / L516](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L516) | <code>    const auto closRankNums = closItr-&gt;second[0];</code> | 设置 closRankNums 为 closItr-&gt;second[0]；该值供下方当前分支使用。 |
| [S283 / L518](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L518) | <code>    const auto meshRankNums = meshItr-&gt;second[0];</code> | 设置 meshRankNums 为 meshItr-&gt;second[0]；该值供下方当前分支使用。 |
| [S286 / L522](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L522) | <code>    isMultiple = (meshRankNums &gt; 1) &amp;&amp; (closRankNums &gt; meshRankNums) &amp;&amp; (closRankNums % meshRankNums == 0);</code> | 续接 CheckClosNumMultipleOfMeshNum 当前语句的具体实参/字段：isMultiple = (meshRankNums 大于 1) 且 (closRankNums 大于 meshRankNums) 且 (closRankNums % meshRankNums 等于 0)；由其完整表达式完成参数组装、检查或结果写回。 |
| [S287 / L524](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L524) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S288 / L526](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L526) | <code>}</code> | 结束 CheckClosNumMultipleOfMeshNum 实现；其返回状态或已写回字段由调用者接收。 |


## 50. AutoSelectorBase::IsTwoLevelNetLayer

判定可用的二级网络：排除 HostDPUOnly，要求第二网络层含 CLOS 且第零层不止一个本地 Rank。

完整范围：[auto_selector_base.cc:L529–L604](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L529-L604)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

功能与分支：

- [S292 / L533](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L533)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S296 / L540](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L540)：分支条件为 (CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) 等于 成功状态) 且 hostDPUOnly；成立进入本块，未成立继续后续分支。

- [S300 / L548](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L548)：分支条件为 topoInfo->当前网络层编号Details.网络层数量 不超过 1；成立进入本块，未成立继续后续分支。

- [S310 / L568](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L568)：分支条件为 !hasLevel1Clos；成立进入本块，未成立继续后续分支。

- [S315 / L578](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L578)：分支条件为 topoInfo->当前网络层编号Details.localNetInsSizeOfLayer.size() 小于 1；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S290 / L529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L529) | <code>bool AutoSelectorBase::IsTwoLevelNetLayer(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam)</code> | 定义 IsTwoLevelNetLayer 入口：判定可用的二级网络：排除 HostDPUOnly，要求第二网络层含 CLOS 且第零层不止一个本地 Rank。 |
| [S291 / L531](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L531) | <code>{</code> | 进入 IsTwoLevelNetLayer 的实现作用域；判定可用的二级网络：排除 HostDPUOnly，要求第二网络层含 CLOS 且第零层不止一个本地 Rank。 |
| [S292 / L533](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L533) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S293 / L535](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L535) | <code>        topoInfo == nullptr, HCCL_WARNING(&quot;[AutoSelectorBase][IsTwoLevelNetLayer] topoInfo is nullptr.&quot;), false);</code> | 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo 等于 nullptr, HCCL_WARNING(&quot;[AutoSelectorBase][IsTwoLevelNetLayer] topoInfo is nullptr.&quot;), false，与前面的格式占位依次对应。 |
| [S295 / L538](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L538) | <code>    bool hostDPUOnly = false;</code> | 设置 hostDPUOnly 为 false；该值供下方当前分支使用。 |
| [S296 / L540](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L540) | <code>    if ((CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) == HCCL_SUCCESS) &amp;&amp; hostDPUOnly) {</code> | 分支条件为 (CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) 等于 成功状态) 且 hostDPUOnly；成立进入本块，未成立继续后续分支。 |
| [S297 / L542](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L542) | <code>        HCCL_INFO(&quot;[AutoSelectorBase][IsTwoLevelNetLayer] host DPU only, not two level net layer.&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S298 / L544](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L544) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S299 / L546](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L546) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S300 / L548](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L548) | <code>    if (topoInfo-&gt;netLayerDetails.netLayerNum &lt;= 1) {</code> | 分支条件为 topoInfo-&gt;当前网络层编号Details.网络层数量 不超过 1；成立进入本块，未成立继续后续分支。 |
| [S301 / L550](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L550) | <code>        HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S302 / L552](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L552) | <code>            &quot;[AutoSelectorBase][IsTwoLevelNetLayer] netLayerNum[%u] &lt;= 1, not two level net layer.&quot;,</code> | 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S303 / L554](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L554) | <code>            topoInfo-&gt;netLayerDetails.netLayerNum);</code> | 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo-&gt;当前网络层编号Details.网络层数量，与前面的格式占位依次对应。 |
| [S304 / L556](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L556) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S305 / L558](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L558) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S306 / L560](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L560) | <code>    u32 level1Idx = topoInfo-&gt;netLayerDetails.netLayers[1];</code> | 设置 level1Idx 为 topoInfo-&gt;当前网络层编号Details.当前网络层编号s[1]；该值供下方当前分支使用。 |
| [S307 / L562](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L562) | <code>    bool hasLevel1Clos = topoInfo-&gt;topoInstDetailsOfLayer.size() &gt; level1Idx</code> | 设置 hasLevel1Clos 为 topoInfo-&gt;topoInstDetailsOfLayer.size() 大于 level1Idx；该值供下方当前分支使用。 |
| [S308 / L564](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L564) | <code>                         &amp;&amp; topoInfo-&gt;topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS)</code> | 调用 find 完成当前参数所指的子步骤；本行实参为 且 topoInfo-&gt;topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS)。 |
| [S309 / L566](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L566) | <code>                                != topoInfo-&gt;topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end();</code> | 调用 end 完成当前参数所指的子步骤；本行实参为 不等于 topoInfo-&gt;topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end()。 |
| [S310 / L568](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L568) | <code>    if (!hasLevel1Clos) {</code> | 分支条件为 !hasLevel1Clos；成立进入本块，未成立继续后续分支。 |
| [S311 / L570](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L570) | <code>        HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S312 / L572](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L572) | <code>            &quot;[AutoSelectorBase][IsTwoLevelNetLayer] level1[%u] has no CLOS topo, not two level net layer.&quot;, level1Idx);</code> | 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S313 / L574](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L574) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S314 / L576](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L576) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S315 / L578](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L578) | <code>    if (topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer.size() &lt; 1</code> | 分支条件为 topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer.size() 小于 1；成立进入本块，未成立继续后续分支。 |
| [S316 / L580](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L580) | <code>        &#124;&#124; topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer[0] &lt;= 1) {</code> | 补充同一条件的 或者 子条件：topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer[0] 不超过 1。 |
| [S317 / L582](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L582) | <code>        HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S318 / L584](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L584) | <code>            &quot;[AutoSelectorBase][IsTwoLevelNetLayer] level0 localNetInsSizeOfLayer[%zu] &lt;= 1, not two level net layer.&quot;,</code> | 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S319 / L586](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L586) | <code>            topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer.size());</code> | 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer.size(，与前面的格式占位依次对应。 |
| [S320 / L588](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L588) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S321 / L590](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L590) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S322 / L592](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L592) | <code>    HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S323 / L594](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L594) | <code>        &quot;[AutoSelectorBase][IsTwoLevelNetLayer] topoLevelNums[%u], netLayerNum[%u], level0Topo[MESH_1D], &quot;</code> | 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S324 / L596](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L596) | <code>        &quot;level1Idx[%u] has CLOS, level0LocalNetInsSize[%u], is two level net layer.&quot;,</code> | 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S325 / L598](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L598) | <code>        topoInfo-&gt;topoLevelNums, topoInfo-&gt;netLayerDetails.netLayerNum, level1Idx,</code> | 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo-&gt;topoLevelNums, topoInfo-&gt;当前网络层编号Details.网络层数量, level1Idx，与前面的格式占位依次对应。 |
| [S326 / L600](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L600) | <code>        topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer[0]);</code> | 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer[0]，与前面的格式占位依次对应。 |
| [S327 / L602](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L602) | <code>    return true;</code> | 当前能力/拓扑/匹配检查满足，返回 true。 |
| [S328 / L604](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L604) | <code>}</code> | 结束 IsTwoLevelNetLayer 实现；其返回状态或已写回字段由调用者接收。 |


## 51. AutoSelectorBase::IsDevType960

读取当前设备类型并判断是否 DEV_TYPE_960；代码未检查此处设备查询返回码。

完整范围：[auto_selector_base.cc:L607–L617](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L607-L617)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S330 / L607](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L607) | <code>bool AutoSelectorBase::IsDevType960()</code> | 定义 IsDevType960 入口：读取当前设备类型并判断是否 DEV_TYPE_960；代码未检查此处设备查询返回码。 |
| [S331 / L609](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L609) | <code>{</code> | 进入 IsDevType960 的实现作用域；读取当前设备类型并判断是否 DEV_TYPE_960；代码未检查此处设备查询返回码。 |
| [S332 / L611](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L611) | <code>    HcclDevType deviceType;</code> | 声明本阶段局部变量 HcclDevType deviceType，实际值由后续查询/计算填写。 |
| [S333 / L613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L613) | <code>    HcclGetDeviceType(deviceType);</code> | 调用 HcclGetDeviceType 完成当前参数所指的子步骤；本行实参为 HcclGetDeviceType(deviceType)。 |
| [S334 / L615](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L615) | <code>    return deviceType == HcclDevType::DEV_TYPE_960;</code> | 直接返回 deviceType 等于 HcclDevType::DEV_TYPE_960，调用者取得本分支结果。 |
| [S335 / L617](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L617) | <code>}</code> | 结束 IsDevType960 实现；其返回状态或已写回字段由调用者接收。 |


## 52. AutoSelectorBase::IsInputOutputOverlap

按输入输出基址和字节容量计算闭区间，判断地址区间是否交叠；空地址/零大小判为不重叠。

完整范围：[auto_selector_base.cc:L620–L685](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L620-L685)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

功能与分支：

- [S339 / L624](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L624)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S347 / L638](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L638)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S363 / L666](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L666)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S337 / L620](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L620) | <code>bool AutoSelectorBase::IsInputOutputOverlap(const OpParam&amp; opParam) const</code> | 定义 IsInputOutputOverlap 入口：按输入输出基址和字节容量计算闭区间，判断地址区间是否交叠；空地址/零大小判为不重叠。 |
| [S338 / L622](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L622) | <code>{</code> | 进入 IsInputOutputOverlap 的实现作用域；按输入输出基址和字节容量计算闭区间，判断地址区间是否交叠；空地址/零大小判为不重叠。 |
| [S339 / L624](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L624) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S340 / L626](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L626) | <code>        opParam.inputPtr == nullptr &#124;&#124; opParam.outputPtr == nullptr,</code> | 续接本次错误检查/子调用实参：opParam.inputPtr 等于 nullptr 或 opParam.outputPtr 等于 nullptr；返回行为由所在完整宏决定。 |
| [S341 / L628](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L628) | <code>        HCCL_INFO(&quot;[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer is null. Not overlap.&quot;),</code> | 开始 HCCL_INFO 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S342 / L630](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L630) | <code>        false);</code> | 为 IsInputOutputOverlap 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。 |
| [S344 / L633](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L633) | <code>    u64 inputDataSize = opParam.inputSize;</code> | 设置 inputDataSize 为 opParam.inputSize；该值供下方当前分支使用。 |
| [S345 / L635](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L635) | <code>    u64 outputDataSize = opParam.outputSize;</code> | 设置 outputDataSize 为 opParam.outputSize；该值供下方当前分支使用。 |
| [S347 / L638](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L638) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S348 / L640](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L640) | <code>        inputDataSize == 0 &#124;&#124; outputDataSize == 0,</code> | 续接本次错误检查/子调用实参：inputDataSize 等于 0 或 outputDataSize 等于 0；返回行为由所在完整宏决定。 |
| [S350 / L643](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L643) | <code>        HCCL_INFO(&quot;[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer size is 0. Not overlap.&quot;),</code> | 开始 HCCL_INFO 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S351 / L645](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L645) | <code>        false);</code> | 为 IsInputOutputOverlap 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。 |
| [S353 / L648](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L648) | <code>    uintptr_t inputStart = reinterpret_cast&lt;uintptr_t&gt;(opParam.inputPtr);</code> | 设置 uintptr_t inputStart 为 reinterpret_cast&lt;uintptr_t&gt;(opParam.inputPtr)；该值供下方当前分支使用。 |
| [S354 / L650](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L650) | <code>    uintptr_t outputStart = reinterpret_cast&lt;uintptr_t&gt;(opParam.outputPtr);</code> | 设置 uintptr_t outputStart 为 reinterpret_cast&lt;uintptr_t&gt;(opParam.outputPtr)；该值供下方当前分支使用。 |
| [S355 / L652](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L652) | <code>    uintptr_t inputEnd = inputStart + inputDataSize - 1;</code> | 设置 uintptr_t inputEnd 为 inputStart + inputDataSize - 1；该值供下方当前分支使用。 |
| [S356 / L654](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L654) | <code>    uintptr_t outputEnd = outputStart + outputDataSize - 1;</code> | 设置 uintptr_t outputEnd 为 outputStart + outputDataSize - 1；该值供下方当前分支使用。 |
| [S358 / L657](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L657) | <code>    HCCL_DEBUG(</code> | 开始 HCCL_DEBUG 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S359 / L659](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L659) | <code>        &quot;[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], &quot;</code> | 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S360 / L661](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L661) | <code>        &quot;outputEnd[%llu].&quot;,</code> | 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S361 / L663](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L663) | <code>        inputStart, inputEnd, outputStart, outputEnd);</code> | 为 IsInputOutputOverlap 的诊断/错误宏提供实参：inputStart, inputEnd, outputStart, outputEnd，与前面的格式占位依次对应。 |
| [S363 / L666](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L666) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S364 / L668](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L668) | <code>        inputStart &lt;= outputEnd &amp;&amp; outputStart &lt;= inputEnd,</code> | 续接本次错误检查/子调用实参：inputStart 不超过 outputEnd 且 outputStart 不超过 inputEnd；返回行为由所在完整宏决定。 |
| [S365 / L670](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L670) | <code>        HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S366 / L672](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L672) | <code>            &quot;[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], &quot;</code> | 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S367 / L674](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L674) | <code>            &quot;outputEnd[%llu]. Overlap detected.&quot;,</code> | 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S368 / L676](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L676) | <code>            inputStart, inputEnd, outputStart, outputEnd),</code> | 为 IsInputOutputOverlap 的诊断/错误宏提供实参：inputStart, inputEnd, outputStart, outputEnd，与前面的格式占位依次对应。 |
| [S369 / L678](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L678) | <code>        true);</code> | 为 IsInputOutputOverlap 的诊断/错误宏提供实参：true，与前面的格式占位依次对应。 |
| [S371 / L681](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L681) | <code>    HCCL_DEBUG(&quot;[Algo][AutoSelectorBase][IsInputOutputOverlap]No overlap between input and output memory.&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S372 / L683](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L683) | <code>    return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S373 / L685](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L685) | <code>}</code> | 结束 IsInputOutputOverlap 实现；其返回状态或已写回字段由调用者接收。 |


## 53. AutoSelectorBase::ProcessAivConfig

处理 AIV/AIV_ONLY 配置，普通 AIV 不匹配可回到 CCU_FAIL，AIV_ONLY 保留不匹配结果供上层报错。

完整范围：[auto_selector_base.cc:L688–L739](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L688-L739)；文件：`hccl/src/ops/op_common/selector/auto_selector_base.cc`。

功能与分支：

- [S380 / L698](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L698)：分支条件为 当前执行配置 不等于 OpExecuteConfig::AIV 且 当前执行配置 不等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。

- [S384 / L705](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L705)：分支条件为 topoInfo->topLevelUboe；成立进入本块，未成立继续后续分支。

- [S390 / L716](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L716)：分支条件为 ret 等于 当前选择器不匹配；成立进入本块，未成立继续后续分支。

- [S391 / L718](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L718)：分支条件为 当前执行配置 等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S375 / L688](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L688) | <code>bool AutoSelectorBase::ProcessAivConfig(</code> | 定义 ProcessAivConfig 入口：处理 AIV/AIV_ONLY 配置，普通 AIV 不匹配可回到 CCU_FAIL，AIV_ONLY 保留不匹配结果供上层报错。 |
| [S376 / L690](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L690) | <code>    OpParam&amp; opParam, TopoInfoWithNetLayerDetails* topoInfo,</code> | 续接 ProcessAivConfig 的入口参数/基类初始化：OpParam&amp; opParam, TopoInfoWithNetLayerDetails* topoInfo,；引用参数按声明的 const 限制读写。 |
| [S377 / L692](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L692) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName,</code> | 续接 ProcessAivConfig 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数,；引用参数按声明的 const 限制读写。 |
| [S378 / L694](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L694) | <code>    SelectorStatus&amp; ret) const</code> | 续接 ProcessAivConfig 的入口参数/基类初始化：SelectorStatus&amp; ret) const；引用参数按声明的 const 限制读写。 |
| [S379 / L696](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L696) | <code>{</code> | 进入 ProcessAivConfig 的实现作用域；处理 AIV/AIV_ONLY 配置，普通 AIV 不匹配可回到 CCU_FAIL，AIV_ONLY 保留不匹配结果供上层报错。 |
| [S380 / L698](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L698) | <code>    if (opParam.opExecuteConfig != OpExecuteConfig::AIV &amp;&amp; opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY) {</code> | 分支条件为 当前执行配置 不等于 OpExecuteConfig::AIV 且 当前执行配置 不等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。 |
| [S381 / L700](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L700) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S382 / L702](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L702) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S384 / L705](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L705) | <code>    if (topoInfo-&gt;topLevelUboe) {</code> | 分支条件为 topoInfo-&gt;topLevelUboe；成立进入本块，未成立继续后续分支。 |
| [S385 / L707](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L707) | <code>        opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;</code> | 设置 当前执行配置 为 OpExecuteConfig::CCU_FAIL；该值供下方当前分支使用。 |
| [S386 / L709](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L709) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S387 / L711](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L711) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S389 / L714](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L714) | <code>    ret = SelectAivAlgo(topoInfo, opParam, configAlgMap, selectAlgName);</code> | 设置 ret 为 SelectAivAlgo(topoInfo, opParam, configAlgMap, 算法名输出参数)；该值供下方当前分支使用。 |
| [S390 / L716](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L716) | <code>    if (ret == SelectorStatus::NOT_MATCH) {</code> | 分支条件为 ret 等于 当前选择器不匹配；成立进入本块，未成立继续后续分支。 |
| [S391 / L718](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L718) | <code>        if (opParam.opExecuteConfig == OpExecuteConfig::AIV_ONLY) {</code> | 分支条件为 当前执行配置 等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。 |
| [S392 / L720](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L720) | <code>            HCCL_WARNING(</code> | 开始 HCCL_WARNING 诊断输出，记录 ProcessAivConfig 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S393 / L722](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L722) | <code>                &quot;[Algo][AutoSelectorBase] opType[%d] no aiv algorithm matched, current opExecuteConfig[%d].&quot;,</code> | 续接 ProcessAivConfig 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S394 / L724](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L724) | <code>                static_cast&lt;int&gt;(opParam.opType), static_cast&lt;int&gt;(opParam.opExecuteConfig));</code> | 为 ProcessAivConfig 的诊断/错误宏提供实参：static_cast&lt;int&gt;(opParam.opType), static_cast&lt;int&gt;(当前执行配置，与前面的格式占位依次对应。 |
| [S395 / L726](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L726) | <code>            return true;</code> | 当前能力/拓扑/匹配检查满足，返回 true。 |
| [S396 / L728](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L728) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S397 / L730](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L730) | <code>        opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;</code> | 设置 当前执行配置 为 OpExecuteConfig::CCU_FAIL；该值供下方当前分支使用。 |
| [S398 / L732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L732) | <code>        return false;</code> | 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。 |
| [S399 / L734](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L734) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S401 / L737](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L737) | <code>    return true;</code> | 当前能力/拓扑/匹配检查满足，返回 true。 |
| [S402 / L739](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/auto_selector_base.cc#L739) | <code>}</code> | 结束 ProcessAivConfig 实现；其返回状态或已写回字段由调用者接收。 |


## 54. ExecuteSelector::Run

读取算子注册选择器，MC2 专用优先级 18；普通调用按 std::map 优先级顺序返回第一个 MATCH。

完整范围：[execute_selector.cc:L20–L93](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L20-L93)；文件：`hccl/src/ops/op_common/selector/execute_selector.cc`。

功能与分支：

- [S25 / L31](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L31)：分支条件为 opParam.isMc2；成立进入本块，未成立继续后续分支。

- [S27 / L35](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L35)：分支条件为 iter 等于 selectors.end(；成立进入本块，未成立继续后续分支。

- [S31 / L43](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L43)：分支条件为 iter->second->Select(opParam, topoInfo, 算法名输出参数) 等于 当前选择器匹配；成立进入本块，未成立继续后续分支。

- [S43 / L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L66)：按旧选择器优先级顺序尝试已注册候选；边界/迭代规则为 (auto iter : selectors。

- [S45 / L70](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L70)：分支条件为 iter.second->Select(opParam, topoInfo, 算法名输出参数) 等于 当前选择器匹配；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S19 / L20](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L20) | <code>HcclResult</code> | 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。 |
| [S20 / L22](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L22) | <code>ExecuteSelector::Run(OpParam&amp; opParam, TopoInfoWithNetLayerDetails* topoInfo, std::string&amp; selectAlgName) const</code> | 定义 Run 入口：读取算子注册选择器，MC2 专用优先级 18；普通调用按 std::map 优先级顺序返回第一个 MATCH。 |
| [S21 / L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L24) | <code>{</code> | 进入 Run 的实现作用域；读取算子注册选择器，MC2 专用优先级 18；普通调用按 std::map 优先级顺序返回第一个 MATCH。 |
| [S22 / L26](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L26) | <code>    HCCL_DEBUG(&quot;[Algo][Selector] Run.&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S23 / L28](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L28) | <code>    std::map&lt;u32, AutoSelectorBase*&gt; selectors = SelectorRegistry::Global()-&gt;GetAllSelectors();</code> | 设置 std::map&lt;u32, AutoSelectorBase*&gt; selectors 为 SelectorRegistry::Global()-&gt;GetAllSelectors()；该值供下方当前分支使用。 |
| [S25 / L31](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L31) | <code>    if (opParam.isMc2) {</code> | 分支条件为 opParam.isMc2；成立进入本块，未成立继续后续分支。 |
| [S26 / L33](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L33) | <code>        auto iter = selectors.find(18);</code> | 设置 iter 为 selectors.find(18)；该值供下方当前分支使用。 |
| [S27 / L35](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L35) | <code>        if (iter == selectors.end()) {</code> | 分支条件为 iter 等于 selectors.end(；成立进入本块，未成立继续后续分支。 |
| [S28 / L37](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L37) | <code>            HCCL_ERROR(&quot;[Algo][Selector] CCU selector is not registered.&quot;);</code> | 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S29 / L39](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L39) | <code>            return HcclResult::HCCL_E_NOT_SUPPORT;</code> | 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S30 / L41](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L41) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S31 / L43](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L43) | <code>        if (iter-&gt;second-&gt;Select(opParam, topoInfo, selectAlgName) == SelectorStatus::MATCH) {</code> | 分支条件为 iter-&gt;second-&gt;Select(opParam, topoInfo, 算法名输出参数) 等于 当前选择器匹配；成立进入本块，未成立继续后续分支。 |
| [S32 / L45](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L45) | <code>            HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S33 / L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L47) | <code>                &quot;[Algo][Selector] The ccu selector[priority of %u] is matched, the selected algo type is %s&quot;,</code> | 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S34 / L49](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L49) | <code>                iter-&gt;first, selectAlgName.c_str());</code> | 为 Run 的诊断/错误宏提供实参：iter-&gt;first, 算法名输出参数.c_str(，与前面的格式占位依次对应。 |
| [S35 / L51](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L51) | <code>            return HcclResult::HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S36 / L53](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L53) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S37 / L55](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L55) | <code>        HCCL_ERROR(&quot;[Algo][Selector] CCU selector can not match for optype[%d].&quot;, opParam.opType);</code> | 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S38 / L57](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L57) | <code>        return HcclResult::HCCL_E_NOT_SUPPORT;</code> | 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S39 / L59](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L59) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S41 / L62](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L62) | <code>    selectors = SelectorRegistry::Global()-&gt;GetSelectorsByOpType(opParam.opType);</code> | 设置 selectors 为 SelectorRegistry::Global()-&gt;GetSelectorsByOpType(opParam.opType)；该值供下方当前分支使用。 |
| [S42 / L64](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L64) | <code>    HCCL_INFO(&quot;[Algo][Selector] The selector nums of optype[%d] is [%zu].&quot;, opParam.opType, selectors.size());</code> | 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S43 / L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L66) | <code>    for (auto iter : selectors) {</code> | 按旧选择器优先级顺序尝试已注册候选；边界/迭代规则为 (auto iter : selectors。 |
| [S44 / L68](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L68) | <code>        HCCL_DEBUG(&quot;[Algo][Selector] The selector[priority of %llu] is running.&quot;, iter.first);</code> | 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S45 / L70](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L70) | <code>        if (iter.second-&gt;Select(opParam, topoInfo, selectAlgName) == SelectorStatus::MATCH) {</code> | 分支条件为 iter.second-&gt;Select(opParam, topoInfo, 算法名输出参数) 等于 当前选择器匹配；成立进入本块，未成立继续后续分支。 |
| [S46 / L72](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L72) | <code>            HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S47 / L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L74) | <code>                &quot;[Algo][Selector] The selector[priority of %llu] is matched, the selected algo type is %s&quot;, iter.first,</code> | 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S48 / L76](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L76) | <code>                selectAlgName.c_str());</code> | 为 Run 的诊断/错误宏提供实参：算法名输出参数.c_str(，与前面的格式占位依次对应。 |
| [S49 / L78](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L78) | <code>            return HcclResult::HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S50 / L80](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L80) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S51 / L82](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L82) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S53 / L85](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L85) | <code>    HCCL_ERROR(</code> | 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S54 / L87](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L87) | <code>        &quot;[Algo][Selector] No supported algorithm for optype[%d], opExecuteConfig[%d].&quot;,</code> | 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S55 / L89](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L89) | <code>        static_cast&lt;int&gt;(opParam.opType), static_cast&lt;int&gt;(opParam.opExecuteConfig));</code> | 为 Run 的诊断/错误宏提供实参：static_cast&lt;int&gt;(opParam.opType), static_cast&lt;int&gt;(当前执行配置，与前面的格式占位依次对应。 |
| [S56 / L91](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L91) | <code>    return HcclResult::HCCL_E_NOT_SUPPORT;</code> | 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S57 / L93](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/execute_selector.cc#L93) | <code>}</code> | 结束 Run 实现；其返回状态或已写回字段由调用者接收。 |


## 55. SelectorEngine::Global

返回进程内 SelectorEngine 单例。

完整范围：[selector_engine.cc:L36–L44](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L36-L44)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S35 / L36](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L36) | <code>SelectorEngine* SelectorEngine::Global()</code> | 定义 Global 入口：返回进程内 SelectorEngine 单例。 |
| [S36 / L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L38) | <code>{</code> | 进入 Global 的实现作用域；返回进程内 SelectorEngine 单例。 |
| [S37 / L40](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L40) | <code>    static SelectorEngine* globalSelectorEngine = new SelectorEngine;</code> | 设置 static SelectorEngine* globalSelectorEngine 为 new SelectorEngine；该值供下方当前分支使用。 |
| [S38 / L42](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L42) | <code>    return globalSelectorEngine;</code> | 直接返回 globalSelectorEngine，调用者取得本分支结果。 |
| [S39 / L44](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L44) | <code>}</code> | 结束 Global 实现；其返回状态或已写回字段由调用者接收。 |


## 56. SelectorEngine::IsOpSupported

查询新成本选择器支持的算子集合；AllReduce 在支持列表中。

完整范围：[selector_engine.cc:L47–L68](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L47-L68)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S41 / L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L47) | <code>bool SelectorEngine::IsOpSupported(HcclCMDType opType)</code> | 定义 IsOpSupported 入口：查询新成本选择器支持的算子集合；AllReduce 在支持列表中。 |
| [S42 / L49](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L49) | <code>{</code> | 进入 IsOpSupported 的实现作用域；查询新成本选择器支持的算子集合；AllReduce 在支持列表中。 |
| [S44 / L52](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L52) | <code>    static const std::set&lt;HcclCMDType&gt; supportedOps = {</code> | 设置 static const std::set&lt;HcclCMDType&gt; supportedOps 为 {；该值供下方当前分支使用。 |
| [S45 / L54](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L54) | <code>        HcclCMDType::HCCL_CMD_ALLREDUCE,  HcclCMDType::HCCL_CMD_REDUCE_SCATTER,  HcclCMDType::HCCL_CMD_ALLGATHER,</code> | 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_ALLREDUCE, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLGATHER,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S46 / L56](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L56) | <code>        HcclCMDType::HCCL_CMD_REDUCE,     HcclCMDType::HCCL_CMD_SCATTER,         HcclCMDType::HCCL_CMD_BROADCAST,</code> | 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_REDUCE, HcclCMDType::HCCL_CMD_SCATTER, HcclCMDType::HCCL_CMD_BROADCAST,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S47 / L58](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L58) | <code>        HcclCMDType::HCCL_CMD_BARRIER,    HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, HcclCMDType::HCCL_CMD_ALLTOALLV,</code> | 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_BARRIER, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, HcclCMDType::HCCL_CMD_ALLTOALLV,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S48 / L60](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L60) | <code>        HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLGATHER_V,     HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V,</code> | 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S49 / L62](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L62) | <code>        HcclCMDType::HCCL_CMD_SEND,       HcclCMDType::HCCL_CMD_RECEIVE,         HcclCMDType::HCCL_CMD_ALLTOALL,</code> | 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_SEND, HcclCMDType::HCCL_CMD_RECEIVE, HcclCMDType::HCCL_CMD_ALLTOALL,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S50 / L64](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L64) | <code>    };</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S51 / L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L66) | <code>    return supportedOps.count(opType) &gt; 0;</code> | 直接返回 调用 count 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S52 / L68](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L68) | <code>}</code> | 结束 IsOpSupported 实现；其返回状态或已写回字段由调用者接收。 |


## 57. SelectorEngine::GetEngineByAlgName

按注册的 Pascal 前缀反推算法执行配置，未知前缀默认 AICPU_TS。

完整范围：[selector_engine.cc:L71–L93](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L71-L93)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S58 / L79](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L79)：在 GetEngineByAlgName 中遍历 (int i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (int i = 0; i 小于 count; ++i。

- [S60 / L83](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L83)：分支条件为 algName.size() 至少 len 且 algName.substr(0, len) 等于 entries[i].pascal；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S54 / L71](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L71) | <code>OpExecuteConfig SelectorEngine::GetEngineByAlgName(const std::string&amp; algName)</code> | 定义 GetEngineByAlgName 入口：按注册的 Pascal 前缀反推算法执行配置，未知前缀默认 AICPU_TS。 |
| [S55 / L73](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L73) | <code>{</code> | 进入 GetEngineByAlgName 的实现作用域；按注册的 Pascal 前缀反推算法执行配置，未知前缀默认 AICPU_TS。 |
| [S56 / L75](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L75) | <code>    int count = 0;</code> | 设置 count 为 0；该值供下方当前分支使用。 |
| [S57 / L77](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L77) | <code>    const EnginePrefixEntry* entries = GetEnginePrefixEntries(count);</code> | 设置 const EnginePrefixEntry* entries 为 GetEnginePrefixEntries(count)；该值供下方当前分支使用。 |
| [S58 / L79](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L79) | <code>    for (int i = 0; i &lt; count; ++i) {</code> | 在 GetEngineByAlgName 中遍历 (int i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (int i = 0; i 小于 count; ++i。 |
| [S59 / L81](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L81) | <code>        size_t len = strlen(entries[i].pascal);</code> | 设置 len 为 strlen(entries[i].pascal)；该值供下方当前分支使用。 |
| [S60 / L83](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L83) | <code>        if (algName.size() &gt;= len &amp;&amp; algName.substr(0, len) == entries[i].pascal) {</code> | 分支条件为 algName.size() 至少 len 且 algName.substr(0, len) 等于 entries[i].pascal；成立进入本块，未成立继续后续分支。 |
| [S61 / L85](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L85) | <code>            return entries[i].engine;</code> | 直接返回 entries[i].engine，调用者取得本分支结果。 |
| [S62 / L87](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L87) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S63 / L89](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L89) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S64 / L91](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L91) | <code>    return OpExecuteConfig::AICPU_TS;</code> | 直接返回 OpExecuteConfig::AICPU_TS，调用者取得本分支结果。 |
| [S65 / L93](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L93) | <code>}</code> | 结束 GetEngineByAlgName 实现；其返回状态或已写回字段由调用者接收。 |


## 58. SelectorEngine::CandidateEnginesToPrefixes

把可候选执行引擎转为算法名前缀，供 HCCL_ALGO 成本模型过滤。

完整范围：[selector_engine.cc:L96–L120](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L96-L120)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S73 / L108](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L108)：在 CandidateEnginesToPrefixes 中遍历 (int i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (int i = 0; i 小于 count; ++i。

- [S74 / L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L110)：分支条件为 engineSet.count(entries[i].engine) 不等于 0；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S67 / L96](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L96) | <code>std::vector&lt;std::string&gt; SelectorEngine::CandidateEnginesToPrefixes(const std::vector&lt;OpExecuteConfig&gt;&amp; engines)</code> | 定义 CandidateEnginesToPrefixes 入口：把可候选执行引擎转为算法名前缀，供 HCCL_ALGO 成本模型过滤。 |
| [S68 / L98](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L98) | <code>{</code> | 进入 CandidateEnginesToPrefixes 的实现作用域；把可候选执行引擎转为算法名前缀，供 HCCL_ALGO 成本模型过滤。 |
| [S69 / L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L100) | <code>    std::set&lt;OpExecuteConfig&gt; engineSet(engines.begin(), engines.end());</code> | 调用 engineSet 完成当前参数所指的子步骤；本行实参为 std::set&lt;OpExecuteConfig&gt; engineSet(engines.begin(), engines.end())。 |
| [S70 / L102](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L102) | <code>    std::vector&lt;std::string&gt; prefixes;</code> | 建立本阶段局部对象 std::vector&lt;std::string&gt; prefixes，供 CandidateEnginesToPrefixes 下方参数组装和子调用使用。 |
| [S71 / L104](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L104) | <code>    int count = 0;</code> | 设置 count 为 0；该值供下方当前分支使用。 |
| [S72 / L106](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L106) | <code>    const EnginePrefixEntry* entries = GetEnginePrefixEntries(count);</code> | 设置 const EnginePrefixEntry* entries 为 GetEnginePrefixEntries(count)；该值供下方当前分支使用。 |
| [S73 / L108](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L108) | <code>    for (int i = 0; i &lt; count; ++i) {</code> | 在 CandidateEnginesToPrefixes 中遍历 (int i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (int i = 0; i 小于 count; ++i。 |
| [S74 / L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L110) | <code>        if (engineSet.count(entries[i].engine) != 0) {</code> | 分支条件为 engineSet.count(entries[i].engine) 不等于 0；成立进入本块，未成立继续后续分支。 |
| [S75 / L112](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L112) | <code>            prefixes.emplace_back(entries[i].pascal);</code> | 对 prefixes 就地追加 entries[i].pascal，准备或更新本阶段列表。 |
| [S76 / L114](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L114) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S77 / L116](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L116) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S78 / L118](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L118) | <code>    return prefixes;</code> | 直接返回 prefixes，调用者取得本分支结果。 |
| [S79 / L120](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L120) | <code>}</code> | 结束 CandidateEnginesToPrefixes 实现；其返回状态或已写回字段由调用者接收。 |


## 59. SelectorEngine::GetEnginePriority

根据指定引擎生成可回退引擎候选列表，AIV_ONLY 列表仅 AIV。

完整范围：[selector_engine.cc:L123–L163](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L123-L163)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S84 / L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L129)：处理 OpExecuteConfig::CCU_MS 的专用实现，不同类型或运算在其它 case 分开处理。

- [S88 / L137](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L137)：处理 OpExecuteConfig::CCU_SCHED 的专用实现，不同类型或运算在其它 case 分开处理。

- [S90 / L141](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L141)：处理 OpExecuteConfig::AIV 的专用实现，不同类型或运算在其它 case 分开处理。

- [S92 / L145](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L145)：处理 OpExecuteConfig::AIV_ONLY 的专用实现，不同类型或运算在其它 case 分开处理。

- [S94 / L149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L149)：处理 OpExecuteConfig::AICPU_TS 的专用实现，不同类型或运算在其它 case 分开处理。

- [S96 / L153](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L153)：处理 OpExecuteConfig::HOSTCPU 的专用实现，不同类型或运算在其它 case 分开处理。

- [S98 / L157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L157)：未列出的类型或状态进入兜底分支；按下面返回码判为不支持或错误。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S81 / L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L123) | <code>std::vector&lt;OpExecuteConfig&gt; SelectorEngine::GetEnginePriority(OpExecuteConfig opExecuteConfig)</code> | 定义 GetEnginePriority 入口：根据指定引擎生成可回退引擎候选列表，AIV_ONLY 列表仅 AIV。 |
| [S82 / L125](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L125) | <code>{</code> | 进入 GetEnginePriority 的实现作用域；根据指定引擎生成可回退引擎候选列表，AIV_ONLY 列表仅 AIV。 |
| [S83 / L127](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L127) | <code>    switch (opExecuteConfig) {</code> | 续接 GetEnginePriority 当前语句的具体实参/字段：switch (opExecuteConfig) {；由其完整表达式完成参数组装、检查或结果写回。 |
| [S84 / L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L129) | <code>        case OpExecuteConfig::CCU_MS:</code> | 处理 OpExecuteConfig::CCU_MS 的专用实现，不同类型或运算在其它 case 分开处理。 |
| [S85 / L131](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L131) | <code>            return {</code> | 直接返回 {，调用者取得本分支结果。 |
| [S86 / L133](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L133) | <code>                OpExecuteConfig::CCU_MS, OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS,</code> | 续接 GetEnginePriority 当前语句的具体实参/字段：OpExecuteConfig::CCU_MS, OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S87 / L135](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L135) | <code>                OpExecuteConfig::HOSTCPU};</code> | 续接 GetEnginePriority 当前语句的具体实参/字段：OpExecuteConfig::HOSTCPU}；由其完整表达式完成参数组装、检查或结果写回。 |
| [S88 / L137](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L137) | <code>        case OpExecuteConfig::CCU_SCHED:</code> | 处理 OpExecuteConfig::CCU_SCHED 的专用实现，不同类型或运算在其它 case 分开处理。 |
| [S89 / L139](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L139) | <code>            return {OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};</code> | 直接返回 {OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。 |
| [S90 / L141](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L141) | <code>        case OpExecuteConfig::AIV:</code> | 处理 OpExecuteConfig::AIV 的专用实现，不同类型或运算在其它 case 分开处理。 |
| [S91 / L143](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L143) | <code>            return {OpExecuteConfig::AIV, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};</code> | 直接返回 {OpExecuteConfig::AIV, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。 |
| [S92 / L145](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L145) | <code>        case OpExecuteConfig::AIV_ONLY:</code> | 处理 OpExecuteConfig::AIV_ONLY 的专用实现，不同类型或运算在其它 case 分开处理。 |
| [S93 / L147](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L147) | <code>            return {OpExecuteConfig::AIV};</code> | 直接返回 {OpExecuteConfig::AIV}，调用者取得本分支结果。 |
| [S94 / L149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L149) | <code>        case OpExecuteConfig::AICPU_TS:</code> | 处理 OpExecuteConfig::AICPU_TS 的专用实现，不同类型或运算在其它 case 分开处理。 |
| [S95 / L151](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L151) | <code>            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};</code> | 直接返回 {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。 |
| [S96 / L153](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L153) | <code>        case OpExecuteConfig::HOSTCPU:</code> | 处理 OpExecuteConfig::HOSTCPU 的专用实现，不同类型或运算在其它 case 分开处理。 |
| [S97 / L155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L155) | <code>            return {OpExecuteConfig::HOSTCPU};</code> | 直接返回 {OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。 |
| [S98 / L157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L157) | <code>        default:</code> | 未列出的类型或状态进入兜底分支；按下面返回码判为不支持或错误。 |
| [S99 / L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L159) | <code>            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};</code> | 直接返回 {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。 |
| [S100 / L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L161) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S101 / L163](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L163) | <code>}</code> | 结束 GetEnginePriority 实现；其返回状态或已写回字段由调用者接收。 |


## 60. SelectorEngine::FilterCmByEngine

把不在候选引擎集合中的成本项 count 清零，保留可参加成本计算的算法。

完整范围：[selector_engine.cc:L166–L192](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L166-L192)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S106 / L172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L172)：遍历成本模型项，按候选引擎过滤算法条目；边界/迭代规则为 (int i = 0; i 小于 cm.count; i++。

- [S107 / L174](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L174)：分支条件为 cm.costAlgoParams[i].algName 等于 nullptr；成立进入本块，未成立继续后续分支。

- [S111 / L182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L182)：分支条件为 engineSet.count(engine) 等于 0；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S103 / L166](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L166) | <code>HcclResult SelectorEngine::FilterCmByEngine(CostModel&amp; cm, const std::vector&lt;OpExecuteConfig&gt;&amp; candidateEngines)</code> | 定义 FilterCmByEngine 入口：把不在候选引擎集合中的成本项 count 清零，保留可参加成本计算的算法。 |
| [S104 / L168](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L168) | <code>{</code> | 进入 FilterCmByEngine 的实现作用域；把不在候选引擎集合中的成本项 count 清零，保留可参加成本计算的算法。 |
| [S105 / L170](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L170) | <code>    std::set&lt;OpExecuteConfig&gt; engineSet(candidateEngines.begin(), candidateEngines.end());</code> | 调用 engineSet 完成当前参数所指的子步骤；本行实参为 std::set&lt;OpExecuteConfig&gt; engineSet(candidateEngines.begin(), candidateEngines.end())。 |
| [S106 / L172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L172) | <code>    for (int i = 0; i &lt; cm.count; i++) {</code> | 遍历成本模型项，按候选引擎过滤算法条目；边界/迭代规则为 (int i = 0; i 小于 cm.count; i++。 |
| [S107 / L174](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L174) | <code>        if (cm.costAlgoParams[i].algName == nullptr) {</code> | 分支条件为 cm.costAlgoParams[i].algName 等于 nullptr；成立进入本块，未成立继续后续分支。 |
| [S108 / L176](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L176) | <code>            continue;</code> | 跳过当前遍历项的剩余步骤，直接处理下一项。 |
| [S109 / L178](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L178) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S110 / L180](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L180) | <code>        OpExecuteConfig engine = GetEngineByAlgName(cm.costAlgoParams[i].algName);</code> | 设置 OpExecuteConfig engine 为 GetEngineByAlgName(cm.costAlgoParams[i].algName)；该值供下方当前分支使用。 |
| [S111 / L182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L182) | <code>        if (engineSet.count(engine) == 0) {</code> | 分支条件为 engineSet.count(engine) 等于 0；成立进入本块，未成立继续后续分支。 |
| [S112 / L184](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L184) | <code>            cm.costAlgoParams[i].count = 0;</code> | 设置 cm.costAlgoParams[i].count 为 0；该值供下方当前分支使用。 |
| [S113 / L186](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L186) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S114 / L188](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L188) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S115 / L190](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L190) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S116 / L192](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L192) | <code>}</code> | 结束 FilterCmByEngine 实现；其返回状态或已写回字段由调用者接收。 |


## 61. SelectorEngine::InitCostModel

建立每通信域每执行引擎独立成本模型，上下文深拷贝参数，再按引擎和 HCCL_ALGO 配置过滤。

完整范围：[selector_engine.cc:L240–L347](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L240-L347)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S188 / L283](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L283)：分支条件为 srcCm.count 大于 0 且 srcCm.costAlgoParams 不等于 nullptr；成立进入本块，未成立继续后续分支。

- [S195 / L295](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L295)：遍历成本模型项，按候选引擎过滤算法条目；边界/迭代规则为 (int i = 0; i 小于 storedCm->count; ++i。

- [S198 / L301](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L301)：分支条件为 n 大于 0 且 srcParam 不等于 nullptr；成立进入本块，未成立继续后续分支。

- [S200 / L305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L305)：分支条件为 owned 等于 nullptr；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S163 / L240](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L240) | <code>HcclResult</code> | 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。 |
| [S164 / L242](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L242) | <code>SelectorEngine::InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, OpParam&amp; param, CostModel*&amp; cm)</code> | 定义 InitCostModel 入口：建立每通信域每执行引擎独立成本模型，上下文深拷贝参数，再按引擎和 HCCL_ALGO 配置过滤。 |
| [S165 / L244](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L244) | <code>{</code> | 进入 InitCostModel 的实现作用域；建立每通信域每执行引擎独立成本模型，上下文深拷贝参数，再按引擎和 HCCL_ALGO 配置过滤。 |
| [S166 / L246](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L246) | <code>    HCCL_INFO(&quot;[SelectorEngine] Initializing costModel for comm, engine=%d.&quot;, static_cast&lt;int&gt;(param.opExecuteConfig));</code> | 开始 HCCL_INFO 诊断输出，记录 InitCostModel 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S169 / L250](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L250) | <code>    static std::once_flag algoMapperFlag;</code> | 续接 InitCostModel 当前语句的具体实参/字段：static std::once_flag algoMapperFlag；由其完整表达式完成参数组装、检查或结果写回。 |
| [S170 / L252](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L252) | <code>    std::call_once(algoMapperFlag, []() {</code> | 调用 std::call_once 完成当前参数所指的子步骤；本行实参为 std::call_once(algoMapperFlag, []() {。 |
| [S171 / L254](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L254) | <code>        AlgoNameMapper::Global()-&gt;Init(*GetAllAlgos());</code> | 调用 AlgoNameMapper::Global 完成当前参数所指的子步骤；本行实参为 AlgoNameMapper::Global()-&gt;Init(*GetAllAlgos())。 |
| [S172 / L256](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L256) | <code>    });</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S175 / L260](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L260) | <code>    CostModelManager* costModelMgr = CostModelManager::Global();</code> | 设置 CostModelManager* costModelMgr 为 CostModelManager::Global()；该值供下方当前分支使用。 |
| [S176 / L262](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L262) | <code>    CostModel srcCm{nullptr, 0};</code> | 续接 InitCostModel 当前语句的具体实参/字段：CostModel srcCm{nullptr, 0}；由其完整表达式完成参数组装、检查或结果写回。 |
| [S177 / L264](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L264) | <code>    CHK_RET(costModelMgr-&gt;InitCostModel(comm, topoInfo, srcCm, param));</code> | 首次建立此通信域此引擎的成本模型；返回值非成功时立即从当前函数返回该错误。 |
| [S180 / L268](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L268) | <code>    void* ctxPtr = nullptr;</code> | 设置 ctxPtr 为 nullptr；该值供下方当前分支使用。 |
| [S181 / L270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L270) | <code>    uint64_t flatSize = sizeof(CostModel) + static_cast&lt;uint64_t&gt;(srcCm.count) * sizeof(CostAlgoParams);</code> | 设置 uint64_t flatSize 为 sizeof(CostModel) + static_cast&lt;uint64_t&gt;(srcCm.count) * sizeof(CostAlgoParams)；该值供下方当前分支使用。 |
| [S182 / L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L272) | <code>    std::string costModelTag = std::string(COST_MODEL_TAG) + &quot;_&quot; + ENGINE_STR_MAP.at(param.opExecuteConfig);</code> | 设置 costModelTag 为 std::string(COST_MODEL_TAG) + &quot;_&quot; + ENGINE_STR_MAP.at(当前执行配置)；该值供下方当前分支使用。 |
| [S183 / L274](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L274) | <code>    CHK_RET(HcclEngineCtxCreate(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, flatSize, &amp;ctxPtr));</code> | 调用 HcclEngineCtxCreate 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S185 / L277](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L277) | <code>    CostModel* storedCm = static_cast&lt;CostModel*&gt;(ctxPtr);</code> | 设置 CostModel* storedCm 为 static_cast&lt;CostModel*&gt;(ctxPtr)；该值供下方当前分支使用。 |
| [S186 / L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L279) | <code>    storedCm-&gt;count = srcCm.count;</code> | 设置 storedCm-&gt;count 为 srcCm.count；该值供下方当前分支使用。 |
| [S187 / L281](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L281) | <code>    storedCm-&gt;costAlgoParams = reinterpret_cast&lt;CostAlgoParams*&gt;(storedCm + 1);</code> | 设置 storedCm-&gt;costAlgoParams 为 reinterpret_cast&lt;CostAlgoParams*&gt;(storedCm + 1)；该值供下方当前分支使用。 |
| [S188 / L283](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L283) | <code>    if (srcCm.count &gt; 0 &amp;&amp; srcCm.costAlgoParams != nullptr) {</code> | 分支条件为 srcCm.count 大于 0 且 srcCm.costAlgoParams 不等于 nullptr；成立进入本块，未成立继续后续分支。 |
| [S189 / L285](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L285) | <code>        CHK_SAFETY_FUNC_RET(memcpy_s(</code> | 调用 CHK_SAFETY_FUNC_RET 完成当前参数所指的子步骤；本行实参为 CHK_SAFETY_FUNC_RET(memcpy_s(。 |
| [S190 / L287](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L287) | <code>            storedCm-&gt;costAlgoParams, static_cast&lt;uint64_t&gt;(srcCm.count) * sizeof(CostAlgoParams), srcCm.costAlgoParams,</code> | 续接 InitCostModel 当前语句的具体实参/字段：storedCm-&gt;costAlgoParams, static_cast&lt;uint64_t&gt;(srcCm.count) * sizeof(CostAlgoParams), srcCm.costAlgoParams,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S191 / L289](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L289) | <code>            static_cast&lt;uint64_t&gt;(srcCm.count) * sizeof(CostAlgoParams)));</code> | 按目标类型转换 static_cast&lt;uint64_t&gt;(srcCm.count) * sizeof(CostAlgoParams)))，保持接口参数的数值或地址语义。 |
| [S192 / L291](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L291) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S195 / L295](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L295) | <code>    for (int i = 0; i &lt; storedCm-&gt;count; ++i) {</code> | 遍历成本模型项，按候选引擎过滤算法条目；边界/迭代规则为 (int i = 0; i 小于 storedCm-&gt;count; ++i。 |
| [S196 / L297](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L297) | <code>        int n = storedCm-&gt;costAlgoParams[i].count;</code> | 设置 n 为 storedCm-&gt;costAlgoParams[i].count；该值供下方当前分支使用。 |
| [S197 / L299](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L299) | <code>        const CostModelParam* srcParam = storedCm-&gt;costAlgoParams[i].param;</code> | 设置 const CostModelParam* srcParam 为 storedCm-&gt;costAlgoParams[i].param；该值供下方当前分支使用。 |
| [S198 / L301](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L301) | <code>        if (n &gt; 0 &amp;&amp; srcParam != nullptr) {</code> | 分支条件为 n 大于 0 且 srcParam 不等于 nullptr；成立进入本块，未成立继续后续分支。 |
| [S199 / L303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L303) | <code>            CostModelParam* owned = new (std::nothrow) CostModelParam[n];</code> | 设置 CostModelParam* owned 为 new (std::nothrow) CostModelParam[n]；该值供下方当前分支使用。 |
| [S200 / L305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L305) | <code>            if (owned == nullptr) {</code> | 分支条件为 owned 等于 nullptr；成立进入本块，未成立继续后续分支。 |
| [S201 / L307](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L307) | <code>                HCCL_ERROR(&quot;[SelectorEngine] alloc param failed, i=%d count=%d.&quot;, i, n);</code> | 开始 HCCL_ERROR 诊断输出，记录 InitCostModel 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S202 / L309](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L309) | <code>                return HcclResult::HCCL_E_PARA;</code> | 终止当前函数并向上返回 HcclResult::参数错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S203 / L311](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L311) | <code>            }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S204 / L313](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L313) | <code>            CHK_SAFETY_FUNC_RET(memcpy_s(</code> | 调用 CHK_SAFETY_FUNC_RET 完成当前参数所指的子步骤；本行实参为 CHK_SAFETY_FUNC_RET(memcpy_s(。 |
| [S205 / L315](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L315) | <code>                owned, static_cast&lt;uint64_t&gt;(n) * sizeof(CostModelParam), srcParam,</code> | 续接 InitCostModel 当前语句的具体实参/字段：owned, static_cast&lt;uint64_t&gt;(n) * sizeof(CostModelParam), srcParam,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S206 / L317](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L317) | <code>                static_cast&lt;uint64_t&gt;(n) * sizeof(CostModelParam)));</code> | 按目标类型转换 static_cast&lt;uint64_t&gt;(n) * sizeof(CostModelParam)))，保持接口参数的数值或地址语义。 |
| [S207 / L319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L319) | <code>            storedCm-&gt;costAlgoParams[i].param = owned;</code> | 设置 storedCm-&gt;costAlgoParams[i].param 为 owned；该值供下方当前分支使用。 |
| [S208 / L321](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L321) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S209 / L323](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L323) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S212 / L327](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L327) | <code>    CostModelManager::FreeCostModel(srcCm);</code> | 调用 CostModelManager::FreeCostModel 完成当前参数所指的子步骤；本行实参为 CostModelManager::FreeCostModel(srcCm)。 |
| [S215 / L331](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L331) | <code>    std::vector&lt;OpExecuteConfig&gt; candidateEngines = GetEnginePriority(param.opExecuteConfig);</code> | 设置 std::vector&lt;OpExecuteConfig&gt; candidateEngines 为 GetEnginePriority(当前执行配置)；该值供下方当前分支使用。 |
| [S216 / L333](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L333) | <code>    CHK_RET(FilterCmByEngine(*storedCm, candidateEngines));</code> | 调用 FilterCmByEngine 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S217 / L335](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L335) | <code>    std::vector&lt;std::string&gt; candidatePrefixes = CandidateEnginesToPrefixes(candidateEngines);</code> | 设置 std::vector&lt;std::string&gt; candidatePrefixes 为 CandidateEnginesToPrefixes(candidateEngines)；该值供下方当前分支使用。 |
| [S218 / L337](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L337) | <code>    CHK_RET(FilterCmByHcclAlgo(comm, *storedCm, candidatePrefixes));</code> | 调用 FilterCmByHcclAlgo 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S220 / L340](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L340) | <code>    cm = storedCm;</code> | 设置 cm 为 storedCm；该值供下方当前分支使用。 |
| [S222 / L343](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L343) | <code>    HCCL_INFO(&quot;[SelectorEngine] costModel initialized and stored in comm ctx, count=%d.&quot;, storedCm-&gt;count);</code> | 开始 HCCL_INFO 诊断输出，记录 InitCostModel 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S223 / L345](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L345) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S224 / L347](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L347) | <code>}</code> | 结束 InitCostModel 实现；其返回状态或已写回字段由调用者接收。 |


## 62. SelectorEngine::TunerEnrichCostTable

把成本模型转换为当前数据量成本表，并在 tuner 已加载且表非空时允许插件改成本。

完整范围：[selector_engine.cc:L350–L395](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L350-L395)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S231 / L359](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L359)：分支条件为 ct.count 大于 0 且 HcclTunerIsLoaded(；成立进入本块，未成立继续后续分支。

- [S237 / L369](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L369)：分支条件为 param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALL 或 param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALLV；成立进入本块，未成立继续后续分支。

- [S243 / L381](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L381)：分支条件为 tunerModified；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S226 / L350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L350) | <code>HcclResult SelectorEngine::TunerEnrichCostTable(</code> | 定义 TunerEnrichCostTable 入口：把成本模型转换为当前数据量成本表，并在 tuner 已加载且表非空时允许插件改成本。 |
| [S227 / L352](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L352) | <code>    HcclComm comm, CostModel* cm, CostTable&amp; ct, TopoInfoWithNetLayerDetails* topoInfo, OpParam&amp; param)</code> | 续接 TunerEnrichCostTable 的入口参数/基类初始化：HcclComm comm, CostModel* cm, CostTable&amp; ct, TopoInfoWithNetLayerDetails* topoInfo, OpParam&amp; param)；引用参数按声明的 const 限制读写。 |
| [S228 / L354](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L354) | <code>{</code> | 进入 TunerEnrichCostTable 的实现作用域；把成本模型转换为当前数据量成本表，并在 tuner 已加载且表非空时允许插件改成本。 |
| [S229 / L356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L356) | <code>    CHK_RET(CostTableManager::Global()-&gt;CostTableGen(*cm, ct, topoInfo, param));</code> | 调用 CostTableManager::Global 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S231 / L359](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L359) | <code>    if (ct.count &gt; 0 &amp;&amp; HcclTunerIsLoaded()) {</code> | 分支条件为 ct.count 大于 0 且 HcclTunerIsLoaded(；成立进入本块，未成立继续后续分支。 |
| [S233 / L362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L362) | <code>        AlgoNameMapper::Global()-&gt;Enrich(ct.costs, ct.count);</code> | 调用 AlgoNameMapper::Global 完成当前参数所指的子步骤；本行实参为 AlgoNameMapper::Global()-&gt;Enrich(ct.costs, ct.count)。 |
| [S234 / L364](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L364) | <code>        bool tunerModified = false;</code> | 设置 tunerModified 为 false；该值供下方当前分支使用。 |
| [S236 / L367](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L367) | <code>        HcclDataType tunerDataType = param.DataDes.dataType;</code> | 设置 HcclDataType tunerDataType 为 输入元素类型；该值供下方当前分支使用。 |
| [S237 / L369](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L369) | <code>        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL &#124;&#124; param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV</code> | 分支条件为 param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALL 或 param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALLV；成立进入本块，未成立继续后续分支。 |
| [S238 / L371](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L371) | <code>            &#124;&#124; param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {</code> | 补充同一条件的 或者 子条件：param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALLVC。 |
| [S239 / L373](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L373) | <code>            tunerDataType = param.all2AllVDataDes.sendType;</code> | 设置 tunerDataType 为 param.all2AllVDataDes.sendType；该值供下方当前分支使用。 |
| [S240 / L375](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L375) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S241 / L377](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L377) | <code>        CHK_RET(HcclTunerCallGetCollInfo(</code> | 调用 HcclTunerCallGetCollInfo 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S242 / L379](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L379) | <code>            comm, param.opType, param.inputSize, tunerDataType, ct.costs, ct.count, &amp;tunerModified));</code> | 续接本次错误检查/子调用实参：comm, param.opType, 输入字节容量, tunerDataType, ct.costs, ct.count, &amp;tunerModified；返回行为由所在完整宏决定。 |
| [S243 / L381](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L381) | <code>        if (tunerModified) {</code> | 分支条件为 tunerModified；成立进入本块，未成立继续后续分支。 |
| [S244 / L383](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L383) | <code>            HCCL_INFO(&quot;[SelectorEngine] tuner modified cost table.&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 TunerEnrichCostTable 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S245 / L385](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L385) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S246 / L387](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L387) | <code>            HCCL_INFO(&quot;[SelectorEngine] tuner did not modify cost table, using CostModel selection.&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 TunerEnrichCostTable 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S247 / L389](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L389) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S248 / L391](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L391) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S249 / L393](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L393) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S250 / L395](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L395) | <code>}</code> | 结束 TunerEnrichCostTable 实现；其返回状态或已写回字段由调用者接收。 |


## 63. SelectorEngine::Run

选择器执行入口；ExecuteSelector 按优先级首个 MATCH，SelectorEngine 按成本模型/成本表选择。

完整范围：[selector_engine.cc:L398–L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L398-L495)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S259 / L411](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L411)：分支条件为 当前执行配置 不等于 OpExecuteConfig::AIV_ONLY 且 AutoSelectorBase::IsRollBackAiv(param, topoInfo；成立进入本块，未成立继续后续分支。

- [S267 / L425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L425)：分支条件为 HcclEngineCtxGet(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, &tunerCtxPtr, &tunerCtxSize；成立进入本块，未成立继续后续分支。

- [S278 / L445](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L445)：分支条件为 HcclEngineCtxGet(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, &ctxPtr, &ctxSize) 等于 成功状态；成立进入本块，未成立继续后续分支。

- [S296 / L475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L475)：分支条件为 ret 不等于 成功状态；成立进入本块，未成立继续后续分支。

- [S298 / L479](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L479)：分支条件为 当前执行配置 等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S252 / L398](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L398) | <code>HcclResult</code> | 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。 |
| [S253 / L400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L400) | <code>SelectorEngine::Run(HcclComm comm, OpParam&amp; param, TopoInfoWithNetLayerDetails* topoInfo, std::string&amp; algName)</code> | 定义 Run 入口：选择器执行入口；ExecuteSelector 按优先级首个 MATCH，SelectorEngine 按成本模型/成本表选择。 |
| [S254 / L402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L402) | <code>{</code> | 进入 Run 的实现作用域；选择器执行入口；ExecuteSelector 按优先级首个 MATCH，SelectorEngine 按成本模型/成本表选择。 |
| [S255 / L404](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L404) | <code>    HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S256 / L406](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L406) | <code>        &quot;[SelectorEngine] Run start, opType=%d, opExecuteConfig=%d.&quot;, static_cast&lt;int&gt;(param.opType),</code> | 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S257 / L408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L408) | <code>        static_cast&lt;int&gt;(param.opExecuteConfig));</code> | 为 Run 的诊断/错误宏提供实参：static_cast&lt;int&gt;(当前执行配置，与前面的格式占位依次对应。 |
| [S259 / L411](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L411) | <code>    if (param.opExecuteConfig != OpExecuteConfig::AIV_ONLY &amp;&amp; AutoSelectorBase::IsRollBackAiv(param, topoInfo)) {</code> | 分支条件为 当前执行配置 不等于 OpExecuteConfig::AIV_ONLY 且 AutoSelectorBase::IsRollBackAiv(param, topoInfo；成立进入本块，未成立继续后续分支。 |
| [S260 / L413](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L413) | <code>        HCCL_DEBUG(&quot;[SelectorEngine] Need to roll back AIV algo&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S261 / L415](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L415) | <code>        param.opExecuteConfig = OpExecuteConfig::AIV_ONLY;</code> | 设置 当前执行配置 为 OpExecuteConfig::AIV_ONLY；该值供下方当前分支使用。 |
| [S262 / L417](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L417) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S265 / L421](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L421) | <code>    void* tunerCtxPtr = nullptr;</code> | 设置 tunerCtxPtr 为 nullptr；该值供下方当前分支使用。 |
| [S266 / L423](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L423) | <code>    uint64_t tunerCtxSize = 0;</code> | 设置 uint64_t tunerCtxSize 为 0；该值供下方当前分支使用。 |
| [S267 / L425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L425) | <code>    if (HcclEngineCtxGet(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, &amp;tunerCtxPtr, &amp;tunerCtxSize)</code> | 分支条件为 HcclEngineCtxGet(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, &amp;tunerCtxPtr, &amp;tunerCtxSize；成立进入本块，未成立继续后续分支。 |
| [S268 / L427](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L427) | <code>        != HCCL_SUCCESS) {</code> | 续接 Run 当前语句的具体实参/字段：不等于 成功状态) {；由其完整表达式完成参数组装、检查或结果写回。 |
| [S269 / L429](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L429) | <code>        CHK_RET(HcclTunerInit(comm, topoInfo));</code> | 调用 HcclTunerInit 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S270 / L431](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L431) | <code>        CHK_RET(HcclEngineCtxCreate(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, 1, &amp;tunerCtxPtr));</code> | 调用 HcclEngineCtxCreate 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S271 / L433](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L433) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S274 / L437](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L437) | <code>    CostModel* cm = nullptr;</code> | 设置 CostModel* cm 为 nullptr；该值供下方当前分支使用。 |
| [S275 / L439](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L439) | <code>    void* ctxPtr = nullptr;</code> | 设置 ctxPtr 为 nullptr；该值供下方当前分支使用。 |
| [S276 / L441](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L441) | <code>    uint64_t ctxSize = 0;</code> | 设置 uint64_t ctxSize 为 0；该值供下方当前分支使用。 |
| [S277 / L443](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L443) | <code>    std::string costModelTag = std::string(COST_MODEL_TAG) + &quot;_&quot; + ENGINE_STR_MAP.at(param.opExecuteConfig);</code> | 设置 costModelTag 为 std::string(COST_MODEL_TAG) + &quot;_&quot; + ENGINE_STR_MAP.at(当前执行配置)；该值供下方当前分支使用。 |
| [S278 / L445](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L445) | <code>    if (HcclEngineCtxGet(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, &amp;ctxPtr, &amp;ctxSize) == HCCL_SUCCESS) {</code> | 分支条件为 HcclEngineCtxGet(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, &amp;ctxPtr, &amp;ctxSize) 等于 成功状态；成立进入本块，未成立继续后续分支。 |
| [S279 / L447](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L447) | <code>        cm = static_cast&lt;CostModel*&gt;(ctxPtr);</code> | 设置 cm 为 static_cast&lt;CostModel*&gt;(ctxPtr)；该值供下方当前分支使用。 |
| [S280 / L449](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L449) | <code>        HCCL_DEBUG(&quot;[SelectorEngine] costModel found in comm ctx, count=%d.&quot;, cm-&gt;count);</code> | 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S281 / L451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L451) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S282 / L453](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L453) | <code>        CHK_RET(InitCostModel(comm, topoInfo, param, cm));</code> | 首次建立此通信域此引擎的成本模型；返回值非成功时立即从当前函数返回该错误。 |
| [S283 / L455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L455) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S286 / L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L459) | <code>    CostTable ct{nullptr, 0};</code> | 续接 Run 当前语句的具体实参/字段：CostTable ct{nullptr, 0}；由其完整表达式完成参数组装、检查或结果写回。 |
| [S287 / L461](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L461) | <code>    CHK_RET(TunerEnrichCostTable(comm, cm, ct, topoInfo, param));</code> | 建立本次数据量成本表并允许 tuner 改写；返回值非成功时立即从当前函数返回该错误。 |
| [S290 / L465](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L465) | <code>    HcclResult ret = SelectMinCost(ct, param, algName);</code> | 设置 ret 为 SelectMinCost(ct, param, algName)；该值供下方当前分支使用。 |
| [S292 / L468](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L468) | <code>    delete[] ct.costs;</code> | 续接 Run 当前语句的具体实参/字段：delete[] ct.costs；由其完整表达式完成参数组装、检查或结果写回。 |
| [S293 / L470](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L470) | <code>    ct.costs = nullptr;</code> | 设置 ct.costs 为 nullptr；该值供下方当前分支使用。 |
| [S294 / L472](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L472) | <code>    ct.count = 0;</code> | 设置 ct.count 为 0；该值供下方当前分支使用。 |
| [S296 / L475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L475) | <code>    if (ret != HCCL_SUCCESS) {</code> | 分支条件为 ret 不等于 成功状态；成立进入本块，未成立继续后续分支。 |
| [S297 / L477](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L477) | <code>        HCCL_ERROR(&quot;[SelectorEngine] Run failed, no algorithm selected.&quot;);</code> | 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S298 / L479](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L479) | <code>        if (param.opExecuteConfig == OpExecuteConfig::AIV_ONLY) {</code> | 分支条件为 当前执行配置 等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。 |
| [S299 / L481](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L481) | <code>            LogAivOnlyNotMatch(param, topoInfo);</code> | 调用 LogAivOnlyNotMatch 完成当前参数所指的子步骤；本行实参为 LogAivOnlyNotMatch(param, topoInfo)。 |
| [S300 / L483](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L483) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S301 / L485](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L485) | <code>        return ret;</code> | 直接返回 ret，调用者取得本分支结果。 |
| [S302 / L487](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L487) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S304 / L490](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L490) | <code>    LogSelectedAlgo(param, topoInfo, algName);</code> | 调用 LogSelectedAlgo 完成当前参数所指的子步骤；本行实参为 LogSelectedAlgo(param, topoInfo, algName)。 |
| [S306 / L493](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L493) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S307 / L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L495) | <code>}</code> | 结束 Run 实现；其返回状态或已写回字段由调用者接收。 |


## 64. SelectorEngine::SelectMinCost

忽略空算法名和负成本，保留首次最小项；同值项仅日志提示而不改首个胜者，写回算法名及执行配置。

完整范围：[selector_engine.cc:L558–L702](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L558-L702)；文件：`hccl/src/ops/op_common/selector/selector_engine.cc`。

功能与分支：

- [S371 / L562](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L562)：分支条件为 ct.count 不超过 0；成立进入本块，未成立继续后续分支。

- [S390 / L598](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L598)：逐项比较当前数据量成本表中的算法成本；边界/迭代规则为 (int i = 0; i 小于 ct.count; ++i。

- [S397 / L612](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L612)：分支条件为 costRet 小于 0；成立进入本块，未成立继续后续分支。

- [S403 / L623](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L623)：分支条件为 name 等于 nullptr 或 cost 小于 0.0f；成立进入本块，未成立继续后续分支。

- [S406 / L629](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L629)：分支条件为 当前最低成本项索引 等于 -1 或 cost 小于 当前最低候选成本；成立进入本块，未成立继续后续分支。

- [S411 / L639](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L639)：分支条件为 cost 等于 当前最低候选成本；成立进入本块，未成立继续后续分支。

- [S418 / L652](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L652)：分支条件为 当前最低成本项索引 小于 0；成立进入本块，未成立继续后续分支。

- [S425 / L665](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L665)：分支条件为 tiedAlgos.size() 大于 1；成立进入本块，未成立继续后续分支。

- [S427 / L669](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L669)：逐项比较当前数据量成本表中的算法成本；边界/迭代规则为 (size_t i = 0; i 小于 tiedAlgos.size(); ++i。

- [S428 / L671](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L671)：分支条件为 i 大于 0；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S369 / L558](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L558) | <code>HcclResult SelectorEngine::SelectMinCost(const CostTable&amp; ct, OpParam&amp; param, std::string&amp; algName)</code> | 定义 SelectMinCost 入口：忽略空算法名和负成本，保留首次最小项；同值项仅日志提示而不改首个胜者，写回算法名及执行配置。 |
| [S370 / L560](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L560) | <code>{</code> | 进入 SelectMinCost 的实现作用域；忽略空算法名和负成本，保留首次最小项；同值项仅日志提示而不改首个胜者，写回算法名及执行配置。 |
| [S371 / L562](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L562) | <code>    if (ct.count &lt;= 0) {</code> | 分支条件为 ct.count 不超过 0；成立进入本块，未成立继续后续分支。 |
| [S372 / L564](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L564) | <code>        HCCL_ERROR(</code> | 开始 HCCL_ERROR 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S373 / L566](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L566) | <code>            &quot;[SelectorEngine] SelectMinCost: costTable is empty, opType=%d, dataSize=%llu.&quot;,</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S374 / L568](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L568) | <code>            static_cast&lt;int&gt;(param.opType), param.inputSize);</code> | 为 SelectMinCost 的诊断/错误宏提供实参：static_cast&lt;int&gt;(param.opType), 输入字节容量，与前面的格式占位依次对应。 |
| [S375 / L570](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L570) | <code>        return HCCL_E_NOT_SUPPORT;</code> | 终止当前函数并向上返回 不支持错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S376 / L572](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L572) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S379 / L576](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L576) | <code>    HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S380 / L578](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L578) | <code>        &quot;[SelectorEngine] SelectMinCost: costTable count=%d, opType=%d, dataSize=%llu.&quot;, ct.count,</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S381 / L580](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L580) | <code>        static_cast&lt;int&gt;(param.opType), param.inputSize);</code> | 为 SelectMinCost 的诊断/错误宏提供实参：static_cast&lt;int&gt;(param.opType), 输入字节容量，与前面的格式占位依次对应。 |
| [S382 / L582](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L582) | <code>    HCCL_INFO(&quot;[SelectorEngine] &quot;</code> | 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S383 / L584](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L584) | <code>              &quot;+-----+--------------------------------------------------+--------------+&quot;);</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S384 / L586](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L586) | <code>    HCCL_INFO(&quot;[SelectorEngine] &#124; idx &#124; algName                                          &#124; cost         &#124;&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S385 / L588](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L588) | <code>    HCCL_INFO(&quot;[SelectorEngine] &quot;</code> | 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S386 / L590](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L590) | <code>              &quot;+-----+--------------------------------------------------+--------------+&quot;);</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S387 / L592](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L592) | <code>    int minIdx = -1;</code> | 设置 当前最低成本项索引 为 -1；该值供下方当前分支使用。 |
| [S388 / L594](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L594) | <code>    float minCost = 0.0f;</code> | 设置 float 当前最低候选成本 为 0.0f；该值供下方当前分支使用。 |
| [S389 / L596](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L596) | <code>    std::vector&lt;std::string&gt; tiedAlgos;</code> | 建立本阶段局部对象 std::vector&lt;std::string&gt; tiedAlgos，供 SelectMinCost 下方参数组装和子调用使用。 |
| [S390 / L598](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L598) | <code>    for (int i = 0; i &lt; ct.count; ++i) {</code> | 逐项比较当前数据量成本表中的算法成本；边界/迭代规则为 (int i = 0; i 小于 ct.count; ++i。 |
| [S391 / L600](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L600) | <code>        const char* name = ct.costs[i].algName;</code> | 设置 const char* name 为 ct.costs[i].algName；该值供下方当前分支使用。 |
| [S392 / L602](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L602) | <code>        float cost = ct.costs[i].cost;</code> | 设置 float cost 为 ct.costs[i].cost；该值供下方当前分支使用。 |
| [S393 / L604](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L604) | <code>        std::string nameStr = name != nullptr ? name : &quot;-&quot;;</code> | 建立本阶段局部对象 std::string nameStr = name 不等于 nullptr ? name : &quot;-&quot;，供 SelectMinCost 下方参数组装和子调用使用。 |
| [S394 / L606](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L606) | <code>        char costBuf[32];</code> | 续接 SelectMinCost 当前语句的具体实参/字段：char costBuf[32]；由其完整表达式完成参数组装、检查或结果写回。 |
| [S395 / L608](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L608) | <code>        const char* fmt = (cost &gt;= 0.0f &amp;&amp; cost &lt; 1.0f) ? &quot;%.6f&quot; : &quot;%.2f&quot;;</code> | 建立本阶段局部对象 const char* fmt = (cost 至少 0.0f 且 cost 小于 1.0f) ? &quot;%.6f&quot; : &quot;%.2f&quot;，供 SelectMinCost 下方参数组装和子调用使用。 |
| [S396 / L610](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L610) | <code>        int costRet = sprintf_s(costBuf, sizeof(costBuf), fmt, cost);</code> | 设置 costRet 为 sprintf_s(costBuf, sizeof(costBuf), fmt, cost)；该值供下方当前分支使用。 |
| [S397 / L612](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L612) | <code>        if (costRet &lt; 0) {</code> | 分支条件为 costRet 小于 0；成立进入本块，未成立继续后续分支。 |
| [S398 / L614](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L614) | <code>            HCCL_ERROR(&quot;[SelectorEngine] SelectMinCost: sprintf_s failed.&quot;);</code> | 开始 HCCL_ERROR 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S399 / L616](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L616) | <code>            return HCCL_E_INTERNAL;</code> | 终止当前函数并向上返回 内部错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S400 / L618](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L618) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S401 / L620](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L620) | <code>        HCCL_INFO(&quot;[SelectorEngine] &#124; %3d &#124; %-48s &#124; %12s &#124;&quot;, i, nameStr.substr(0, 48).c_str(), costBuf);</code> | 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S403 / L623](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L623) | <code>        if (name == nullptr &#124;&#124; cost &lt; 0.0f) {</code> | 分支条件为 name 等于 nullptr 或 cost 小于 0.0f；成立进入本块，未成立继续后续分支。 |
| [S404 / L625](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L625) | <code>            continue;</code> | 跳过当前遍历项的剩余步骤，直接处理下一项。 |
| [S405 / L627](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L627) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S406 / L629](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L629) | <code>        if (minIdx == -1 &#124;&#124; cost &lt; minCost) {</code> | 分支条件为 当前最低成本项索引 等于 -1 或 cost 小于 当前最低候选成本；成立进入本块，未成立继续后续分支。 |
| [S407 / L631](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L631) | <code>            minIdx = i;</code> | 设置 当前最低成本项索引 为 i；该值供下方当前分支使用。 |
| [S408 / L633](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L633) | <code>            minCost = cost;</code> | 设置 当前最低候选成本 为 cost；该值供下方当前分支使用。 |
| [S409 / L635](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L635) | <code>            tiedAlgos.clear();</code> | 对 tiedAlgos 清空 ，准备或更新本阶段列表。 |
| [S410 / L637](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L637) | <code>            tiedAlgos.emplace_back(name);</code> | 对 tiedAlgos 就地追加 name，准备或更新本阶段列表。 |
| [S411 / L639](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L639) | <code>        } else if (cost == minCost) {</code> | 分支条件为 cost 等于 当前最低候选成本；成立进入本块，未成立继续后续分支。 |
| [S412 / L641](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L641) | <code>            tiedAlgos.emplace_back(name);</code> | 对 tiedAlgos 就地追加 name，准备或更新本阶段列表。 |
| [S413 / L643](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L643) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S414 / L645](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L645) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S415 / L647](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L647) | <code>    HCCL_INFO(&quot;[SelectorEngine] &quot;</code> | 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S416 / L649](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L649) | <code>              &quot;+-----+--------------------------------------------------+----------+--------------+----------+&quot;);</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S418 / L652](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L652) | <code>    if (minIdx &lt; 0) {</code> | 分支条件为 当前最低成本项索引 小于 0；成立进入本块，未成立继续后续分支。 |
| [S419 / L654](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L654) | <code>        HCCL_ERROR(</code> | 开始 HCCL_ERROR 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S420 / L656](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L656) | <code>            &quot;[SelectorEngine] SelectMinCost: no valid algorithm found, expansionMode=%d, costTable count=%d.&quot;,</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S421 / L658](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L658) | <code>            static_cast&lt;int&gt;(param.commOpExpansionMode), ct.count);</code> | 为 SelectMinCost 的诊断/错误宏提供实参：static_cast&lt;int&gt;(param.commOpExpansionMode), ct.count，与前面的格式占位依次对应。 |
| [S422 / L660](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L660) | <code>        return HCCL_E_NOT_SUPPORT;</code> | 终止当前函数并向上返回 不支持错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S423 / L662](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L662) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S425 / L665](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L665) | <code>    if (tiedAlgos.size() &gt; 1) {</code> | 分支条件为 tiedAlgos.size() 大于 1；成立进入本块，未成立继续后续分支。 |
| [S426 / L667](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L667) | <code>        std::string algoNames;</code> | 建立本阶段局部对象 std::string algoNames，供 SelectMinCost 下方参数组装和子调用使用。 |
| [S427 / L669](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L669) | <code>        for (size_t i = 0; i &lt; tiedAlgos.size(); ++i) {</code> | 逐项比较当前数据量成本表中的算法成本；边界/迭代规则为 (size_t i = 0; i 小于 tiedAlgos.size(); ++i。 |
| [S428 / L671](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L671) | <code>            if (i &gt; 0) {</code> | 分支条件为 i 大于 0；成立进入本块，未成立继续后续分支。 |
| [S429 / L673](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L673) | <code>                algoNames += &quot;, &quot;;</code> | 设置 algoNames + 为 &quot;, &quot;；该值供下方当前分支使用。 |
| [S430 / L675](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L675) | <code>            }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S431 / L677](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L677) | <code>            algoNames += tiedAlgos[i];</code> | 设置 algoNames + 为 tiedAlgos[i]；该值供下方当前分支使用。 |
| [S432 / L679](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L679) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S433 / L681](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L681) | <code>        HCCL_WARNING(</code> | 开始 HCCL_WARNING 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S434 / L683](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L683) | <code>            &quot;[SelectorEngine] multiple algos with same cost=%f: [%s], selecting %s.&quot;, minCost, algoNames.c_str(),</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S435 / L685](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L685) | <code>            tiedAlgos[0].c_str());</code> | 为 SelectMinCost 的诊断/错误宏提供实参：tiedAlgos[0].c_str(，与前面的格式占位依次对应。 |
| [S436 / L687](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L687) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S438 / L690](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L690) | <code>    algName = ct.costs[minIdx].algName;</code> | 设置 algName 为 ct.costs[当前最低成本项索引].algName；该值供下方当前分支使用。 |
| [S439 / L692](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L692) | <code>    param.opExecuteConfig = GetEngineByAlgName(algName);</code> | 设置 当前执行配置 为 GetEngineByAlgName(algName)；该值供下方当前分支使用。 |
| [S440 / L694](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L694) | <code>    HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S441 / L696](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L696) | <code>        &quot;[SelectorEngine] SelectMinCost: selected algName=%s, engine=%s, cost=%f.&quot;, algName.c_str(),</code> | 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S442 / L698](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L698) | <code>        ENGINE_STR_MAP.at(param.opExecuteConfig), minCost);</code> | 为 SelectMinCost 的诊断/错误宏提供实参：ENGINE_STR_MAP.at(当前执行配置), 当前最低候选成本，与前面的格式占位依次对应。 |
| [S443 / L700](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L700) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S444 / L702](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/selector/selector_engine.cc#L702) | <code>}</code> | 结束 SelectMinCost 实现；其返回状态或已写回字段由调用者接收。 |
