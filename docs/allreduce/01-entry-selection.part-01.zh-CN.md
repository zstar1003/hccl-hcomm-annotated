# AllReduce逐行对照：入口、引擎与算法选择

[返回阅读指南](../READING_GUIDE.zh-CN.md)。S为审读快照行号；L为带本次逐行注释的源码行号。每个L链接定位到固定源码提交；长语句按物理行分别说明。空行及原注释不重复注释。

审读快照：`f8af6a36831195a8440de6ec6183856bb72af907`；源码提交：`824a8a80731bd66ef6eb78891b9aec0acfca281c`。

[返回本阶段函数导航](01-entry-selection.zh-CN.md)。第1/3页。

## 1. ParseNewSelector

解析规则/成本选择器开关，AllReduce主例使用规则选择器。

完整范围：[alg_env_config.cc:L818–L858](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L818-L858)；文件：`hccl/src/common/alg_env_config.cc`。

功能与分支：

- 未配置保持默认0；仅字符串0/1合法；非法值报错。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S817 / L818](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L818) | <code>HcclResult ParseNewSelector()</code> | 声明选择器开关解析入口，返回配置解析结果。 |
| [S818 / L820](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L820) | <code>{</code> | 进入开关解析函数体。 |
| [S819 / L822](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L822) | <code>    std::string useNewSelectorEnv = GetEnv(&quot;HCCL_USE_NEW_SELECTOR&quot;);</code> | 读取HCCL_USE_NEW_SELECTOR原始环境字符串，尚未选择算法。 |
| [S820 / L824](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L824) | <code>    if (useNewSelectorEnv == &quot;EmptyString&quot;) {</code> | 检测GetEnv用EmptyString表示的未配置情况。 |
| [S821 / L826](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L826) | <code>        HCCL_INFO(&quot;HCCL_USE_NEW_SELECTOR set by default to [0]&quot;);</code> | 记录当前快照未配置时默认使用规则选择器0。 |
| [S822 / L828](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L828) | <code>        return HCCL_SUCCESS;</code> | 未配置时成功返回，不覆盖现有配置字段。 |
| [S823 / L830](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L830) | <code>    }</code> | 结束未配置的提前返回分支。 |
| [S824 / L832](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L832) | <code>    if (useNewSelectorEnv != &quot;0&quot; &amp;&amp; useNewSelectorEnv != &quot;1&quot;) {</code> | 拒绝除字符串0和1以外的显式值。 |
| [S825 / L834](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L834) | <code>        HCCL_ERROR(</code> | 开始记录非法选择器开关错误。 |
| [S826 / L836](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L836) | <code>            &quot;[Parser][NewSelector]environmental variable HCCL_USE_NEW_SELECTOR [%s] is invalid, set by &quot;</code> | 提供错误文本，标明出错环境变量和值占位。 |
| [S827 / L838](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L838) | <code>            &quot;default to [0]&quot;,</code> | 续接日志中的默认值提示；实际随后返回参数错误。 |
| [S828 / L840](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L840) | <code>            useNewSelectorEnv.c_str());</code> | 把非法环境字符串填入错误日志。 |
| [S829 / L842](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L842) | <code>        return HCCL_E_PARA;</code> | 中止配置解析并返回HCCL_E_PARA。 |
| [S830 / L844](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L844) | <code>    }</code> | 结束非法值分支。 |
| [S831 / L846](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L846) | <code>    g_algEnvConfig.useNewSelector = false;</code> | 合法值先设useNewSelector=false，对应规则选择器。 |
| [S832 / L848](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L848) | <code>    if (useNewSelectorEnv == &quot;1&quot;) {</code> | 检查合法值是否为1。 |
| [S833 / L850](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L850) | <code>        g_algEnvConfig.useNewSelector = true;</code> | 值为1时启用成本模型SelectorEngine。 |
| [S834 / L852](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L852) | <code>    }</code> | 结束启用新选择器分支。 |
| [S835 / L854](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L854) | <code>    HCCL_INFO(&quot;HCCL_USE_NEW_SELECTOR set by environment to [%u]&quot;, g_algEnvConfig.useNewSelector);</code> | 记录最终useNewSelector布尔值供定位选算法路径。 |
| [S836 / L856](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L856) | <code>    return HCCL_SUCCESS;</code> | 合法配置解析成功。 |
| [S837 / L858](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/alg_env_config.cc#L858) | <code>}</code> | 结束ParseNewSelector。 |


## 2. HcclHcommBatchTransferOnThread

把HCCL批传输描述转交dlsym取得的HCOMM入口。

完整范围：[hcomm_primitives_dl.cc:L88–L106](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L88-L106)；文件：`hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc`。

功能与分支：

- 函数指针为空返回-1；存在则转发所有参数。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S87 / L88](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L88) | <code>extern &quot;C&quot; int32_t HcclHcommBatchTransferOnThread(</code> | 声明HCCL批传输动态适配入口，保持C链接符号。 |
| [S88 / L90](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L90) | <code>    ThreadHandle thread, ChannelHandle channel, const HcclHcommBatchTransferDesc* transferDescs,</code> | 接收执行Thread、Channel和只读批描述数组。 |
| [S89 / L92](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L92) | <code>    uint32_t transferDescNum)</code> | 接收描述数，不是数据字节数。 |
| [S90 / L94](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L94) | <code>{</code> | 进入动态桥函数体。 |
| [S91 / L96](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L96) | <code>    if (g_HcommBatchTransferOnThread == nullptr) {</code> | 先检查HCOMM批传输符号是否加载成功。 |
| [S92 / L98](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L98) | <code>        HCCL_COMPAT_ERROR(&quot;[HcclWrapper] HcommBatchTransferOnThread not supported&quot;);</code> | 缺少符号时记录兼容层错误。 |
| [S93 / L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L100) | <code>        return -1;</code> | 缺少批接口返回-1，未执行传输。 |
| [S94 / L102](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L102) | <code>    }</code> | 结束符号缺失分支。 |
| [S95 / L104](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L104) | <code>    return g_HcommBatchTransferOnThread(thread, channel, transferDescs, transferDescNum);</code> | 调用加载的HCOMM函数，转发Thread、Channel、描述数组和数量，并返回其结果。 |
| [S96 / L106](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L106) | <code>}</code> | 结束批传输桥。 |


## 3. IsHcommDefaultTimeoutSupported

同时检查默认超时设置与Thread默认等待能力。

完整范围：[hcomm_primitives_dl.cc:L158–L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L158-L164)；文件：`hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S147 / L158](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L158) | <code>bool IsHcommDefaultTimeoutSupported()</code> | 声明默认超时能力联合探测函数。 |
| [S148 / L160](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L160) | <code>{</code> | 进入能力检查函数体。 |
| [S149 / L162](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L162) | <code>    return HcommIsSupportHcommSetNotifyWaitTimeOut() &amp;&amp; HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout();</code> | 只有设置默认等待时间和默认Thread Wait两种符号均支持才返回true。 |
| [S150 / L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L164) | <code>}</code> | 结束能力联合检查。 |


## 4. HcclSetNotifyWaitTimeOut

兼容不同HCOMM版本的通知等待时间设置。

完整范围：[hcomm_primitives_dl.cc:L167–L187](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L167-L187)；文件：`hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc`。

功能与分支：

- 能力缺失返回NOT_SUPPORT；按编译宏选择float或整数超时参数。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S152 / L167](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L167) | <code>HcclResult HcclSetNotifyWaitTimeOut(uint32_t timeout)</code> | 声明通知等待时间的HCCL适配接口，timeout沿用下层单位。 |
| [S153 / L169](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L169) | <code>{</code> | 进入通知等待时间设置函数体。 |
| [S154 / L171](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L171) | <code>    if (!HcommIsSupportHcommSetNotifyWaitTimeOut()) {</code> | 先探测动态加载的HcommSetNotifyWaitTimeOut是否支持。 |
| [S155 / L173](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L173) | <code>        return HCCL_E_NOT_SUPPORT;</code> | 符号缺失时返回NOT_SUPPORT，不调用空入口。 |
| [S156 / L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L175) | <code>    }</code> | 结束接口缺失分支。 |
| [S157 / L177](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L177) | <code>#ifdef HCOMM_TIMEOUT_FLOAT_TYPE</code> | 编译期选择HCOMM使用float超时参数的ABI。 |
| [S158 / L179](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L179) | <code>    return static_cast&lt;HcclResult&gt;(HcommSetNotifyWaitTimeOut(static_cast&lt;float&gt;(timeout)));</code> | 将timeout转为float调用HcommSetNotifyWaitTimeOut，并转换返回码类型。 |
| [S159 / L181](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L181) | <code>#else</code> | 编译期切换到整数参数ABI。 |
| [S160 / L183](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L183) | <code>    return static_cast&lt;HcclResult&gt;(HcommSetNotifyWaitTimeOut(timeout));</code> | 保持timeout整数类型调用HcommSetNotifyWaitTimeOut，并返回其状态。 |
| [S161 / L185](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L185) | <code>#endif</code> | 结束超时ABI条件编译。 |
| [S162 / L187](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L187) | <code>}</code> | 结束HcclSetNotifyWaitTimeOut。 |


## 5. HcclThreadResAcquireTimeOut

兼容不同HCOMM版本的执行流资源申请等待时间设置。

完整范围：[hcomm_primitives_dl.cc:L190–L210](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L190-L210)；文件：`hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc`。

功能与分支：

- 能力缺失返回NOT_SUPPORT；按编译宏选择float或整数超时参数。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S164 / L190](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L190) | <code>HcclResult HcclThreadResAcquireTimeOut(uint32_t timeout)</code> | 声明执行流资源申请等待时间的HCCL适配接口，timeout沿用下层单位。 |
| [S165 / L192](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L192) | <code>{</code> | 进入执行流资源申请等待时间设置函数体。 |
| [S166 / L194](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L194) | <code>    if (!HcommIsSupportHcommThreadResAcquireTimeOut()) {</code> | 先探测动态加载的HcommThreadResAcquireTimeOut是否支持。 |
| [S167 / L196](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L196) | <code>        return HCCL_E_NOT_SUPPORT;</code> | 符号缺失时返回NOT_SUPPORT，不调用空入口。 |
| [S168 / L198](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L198) | <code>    }</code> | 结束接口缺失分支。 |
| [S169 / L200](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L200) | <code>#ifdef HCOMM_TIMEOUT_FLOAT_TYPE</code> | 编译期选择HCOMM使用float超时参数的ABI。 |
| [S170 / L202](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L202) | <code>    return static_cast&lt;HcclResult&gt;(HcommThreadResAcquireTimeOut(static_cast&lt;float&gt;(timeout)));</code> | 将timeout转为float调用HcommThreadResAcquireTimeOut，并转换返回码类型。 |
| [S171 / L204](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L204) | <code>#else</code> | 编译期切换到整数参数ABI。 |
| [S172 / L206](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L206) | <code>    return static_cast&lt;HcclResult&gt;(HcommThreadResAcquireTimeOut(timeout));</code> | 保持timeout整数类型调用HcommThreadResAcquireTimeOut，并返回其状态。 |
| [S173 / L208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L208) | <code>#endif</code> | 结束超时ABI条件编译。 |
| [S174 / L210](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L210) | <code>}</code> | 结束HcclThreadResAcquireTimeOut。 |


## 6. HcclThreadNotifyWaitOnThreadDefault

为Thread通知等待选择默认超时或显式超时入口。

完整范围：[hcomm_primitives_dl.cc:L213–L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L213-L225)；文件：`hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc`。

功能与分支：

- 默认超时设置与默认Wait同时支持才用默认接口，否则用fallbackTimeout。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S176 / L213](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L213) | <code>HcclResult HcclThreadNotifyWaitOnThreadDefault(ThreadHandle thread, uint32_t notifyIdx, uint32_t fallbackTimeout)</code> | 声明Thread等待适配入口，接收Thread、通知槽和兼容超时。 |
| [S177 / L215](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L215) | <code>{</code> | 进入Thread等待适配函数体。 |
| [S178 / L217](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L217) | <code>    if (HcommIsSupportHcommSetNotifyWaitTimeOut() &amp;&amp; HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout()) {</code> | 要求设置默认超时和默认Thread Wait两项动态能力都支持。 |
| [S179 / L219](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L219) | <code>        return static_cast&lt;HcclResult&gt;(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, notifyIdx));</code> | 走默认超时Thread Wait；通知槽与Record配对，返回排队结果。 |
| [S180 / L221](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L221) | <code>    }</code> | 结束默认超时路径。 |
| [S181 / L223](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L223) | <code>    return static_cast&lt;HcclResult&gt;(HcommThreadNotifyWaitOnThread(thread, notifyIdx, fallbackTimeout));</code> | 能力不足时使用显式fallbackTimeout等待，并返回下层状态。 |
| [S182 / L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L225) | <code>}</code> | 结束Thread默认超时适配。 |


## 7. HcclChannelNotifyWaitOnThreadDefault

为Channel通知等待选择默认超时或显式超时入口。

完整范围：[hcomm_primitives_dl.cc:L228–L244](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L228-L244)；文件：`hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc`。

功能与分支：

- 默认设置与默认Channel Wait同时支持才用默认接口。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S184 / L228](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L228) | <code>HcclResult HcclChannelNotifyWaitOnThreadDefault(</code> | 声明在Thread上等待Channel通知的HCCL兼容接口。 |
| [S185 / L230](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L230) | <code>    ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t fallbackTimeout)</code> | 接收Thread、Channel、本地通知槽以及兼容超时。 |
| [S186 / L232](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L232) | <code>{</code> | 进入Channel等待适配函数体。 |
| [S187 / L234](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L234) | <code>    if (HcommIsSupportHcommSetNotifyWaitTimeOut() &amp;&amp; HcommIsSupportHcommChannelNotifyWaitOnThreadWithDefaultTimeout()) {</code> | 同时探测默认超时设置与Channel默认Wait能力。 |
| [S188 / L236](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L236) | <code>        return static_cast&lt;HcclResult&gt;(</code> | 把默认Wait返回码转换为HCCL返回码。 |
| [S189 / L238](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L238) | <code>            HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, localNotifyIdx));</code> | 在给定Thread等待Channel的localNotifyIdx，超时使用先前默认配置。 |
| [S190 / L240](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L240) | <code>    }</code> | 结束默认Channel等待路径。 |
| [S191 / L242](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L242) | <code>    return static_cast&lt;HcclResult&gt;(HcommChannelNotifyWaitOnThread(thread, channel, localNotifyIdx, fallbackTimeout));</code> | 能力不齐时将fallbackTimeout显式传给普通Channel Wait并返回状态。 |
| [S192 / L244](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L244) | <code>}</code> | 结束Channel默认超时适配。 |


## 8. HcclAllReduce

单算子公开入口：版本/设备兼容分流，非零参数校验，入口日志和 OPBASE 公共调度。

完整范围：[all_reduce.cc:L27–L82](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L27-L82)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

功能与分支：

- [S32 / L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L38)：分支条件为 GetHcommVersion() 小于 CANN_VERSION(9, 0, 0；成立进入本块，未成立继续后续分支。

- [S39 / L50](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L50)：分支条件为 !isOutPlace；成立进入本块，未成立继续后续分支。

- [S43 / L57](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L57)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S26 / L27](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L27) | <code>HcclResult HcclAllReduce(</code> | 定义 HcclAllReduce 入口：单算子公开入口：版本/设备兼容分流，非零参数校验，入口日志和 OPBASE 公共调度。 |
| [S27 / L29](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L29) | <code>    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,</code> | 续接 HcclAllReduce 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。 |
| [S28 / L31](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L31) | <code>    aclrtStream stream)</code> | 续接 HcclAllReduce 的入口参数/基类初始化：aclrtStream stream)；引用参数按声明的 const 限制读写。 |
| [S29 / L33](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L33) | <code>{</code> | 进入 HcclAllReduce 的实现作用域；单算子公开入口：版本/设备兼容分流，非零参数校验，入口日志和 OPBASE 公共调度。 |
| [S30 / L35](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L35) | <code>    HCCL_INFO(&quot;Start to run execute HcclAllReduce&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 HcclAllReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S32 / L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L38) | <code>    if (GetHcommVersion() &lt; CANN_VERSION(9, 0, 0)) { // compat handle</code> | 分支条件为 GetHcommVersion() 小于 CANN_VERSION(9, 0, 0；成立进入本块，未成立继续后续分支。 |
| [S33 / L40](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L40) | <code>        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);</code> | 直接返回 调用 HcclAllReduceInner 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S34 / L42](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L42) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S37 / L46](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L46) | <code>    bool isOutPlace = false;</code> | 设置 isOutPlace 为 false；该值供下方当前分支使用。 |
| [S38 / L48](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L48) | <code>    CHK_RET(IsOutPlaceDevice(isOutPlace));</code> | 调用 IsOutPlaceDevice 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S39 / L50](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L50) | <code>    if (!isOutPlace) {</code> | 分支条件为 !isOutPlace；成立进入本块，未成立继续后续分支。 |
| [S40 / L52](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L52) | <code>        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);</code> | 直接返回 调用 HcclAllReduceInner 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S41 / L54](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L54) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S43 / L57](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L57) | <code>    CHK_PRT_RET(count == 0, HCCL_WARNING(&quot;input count is 0, return all reduce success&quot;), HCCL_SUCCESS);</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S45 / L60](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L60) | <code>    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内</code> | 设置 HcclUs startut 为 TIME_NOW()；该值供下方当前分支使用。 |
| [S46 / L62](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L62) | <code>    OpParam param;</code> | 声明本阶段局部变量 OpParam param，实际值由后续查询/计算填写。 |
| [S48 / L65](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L65) | <code>    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, count, dataType, op, stream, param));</code> | 完成 AllReduce 入口环境与参数验证；返回值非成功时立即从当前函数返回该错误。 |
| [S51 / L69](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L69) | <code>    CHK_RET(AllReduceEntryLog(sendBuf, recvBuf, count, dataType, op, stream, param.tag, &quot;HcclAllReduce&quot;));</code> | 记录 AllReduce 本次调用信息；返回值非成功时立即从当前函数返回该错误。 |
| [S55 / L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L74) | <code>    CHK_RET_AND_PRINT_IDE(AllReduceOutPlace(sendBuf, recvBuf, count, dataType, op, comm, stream, param), param.tag);</code> | 调用 CHK_RET_AND_PRINT_IDE 完成当前参数所指的子步骤；本行实参为 CHK_RET_AND_PRINT_IDE(AllReduceOutPlace(sendBuf, recvBuf, count, dataType, op, comm, stream, param), param.tag)。 |
| [S57 / L77](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L77) | <code>    CHK_RET(LogHcclExit(&quot;HcclAllReduce&quot;, param.tag, startut));</code> | 调用 LogHcclExit 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S59 / L80](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L80) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S60 / L82](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L82) | <code>}</code> | 结束 HcclAllReduce 实现；其返回状态或已写回字段由调用者接收。 |


## 9. HcclAllReduceGraphMode

通过 group 定位通信域并收集图模式外部从流、scratch 与 tag，进入 OFFLOAD 公共调度。

完整范围：[all_reduce.cc:L85–L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L85-L175)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

功能与分支：

- [S74 / L107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L107)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S84 / L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L123)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S90 / L131](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L131)：分支条件为 strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) 不等于 0；成立进入本块，未成立继续后续分支。

- [S95 / L140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L140)：分支条件为 streams 不等于 nullptr 且 streamCount 大于 0；成立进入本块，未成立继续后续分支。

- [S96 / L142](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L142)：在 HcclAllReduceGraphMode 中遍历 (size_t i = 0; i 小于 streamCount; i++ 指定的集合或索引区间；边界/迭代规则为 (size_t i = 0; i 小于 streamCount; i++。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S62 / L85](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L85) | <code>HcclResult HcclAllReduceGraphMode(</code> | 定义 HcclAllReduceGraphMode 入口：通过 group 定位通信域并收集图模式外部从流、scratch 与 tag，进入 OFFLOAD 公共调度。 |
| [S63 / L87](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L87) | <code>    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,</code> | 续接 HcclAllReduceGraphMode 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,；引用参数按声明的 const 限制读写。 |
| [S64 / L89](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L89) | <code>    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,</code> | 续接 HcclAllReduceGraphMode 的入口参数/基类初始化：aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,；引用参数按声明的 const 限制读写。 |
| [S65 / L91](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L91) | <code>    uint64_t scratchMemSize)</code> | 续接 HcclAllReduceGraphMode 的入口参数/基类初始化：uint64_t scratchMemSize)；引用参数按声明的 const 限制读写。 |
| [S66 / L93](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L93) | <code>{</code> | 进入 HcclAllReduceGraphMode 的实现作用域；通过 group 定位通信域并收集图模式外部从流、scratch 与 tag，进入 OFFLOAD 公共调度。 |
| [S67 / L95](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L95) | <code>    HCCL_INFO(&quot;Start to run execute HcclAllReduceGraphMode&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 HcclAllReduceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S70 / L99](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L99) | <code>    CHK_PTR_NULL(group);</code> | 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(group)。 |
| [S71 / L101](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L101) | <code>    HcclComm comm = nullptr;</code> | 设置 HcclComm comm 为 nullptr；该值供下方当前分支使用。 |
| [S72 / L103](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L103) | <code>    HCCL_INFO(&quot;[HcclAllReduceGraphMode] get group name: %s&quot;, group);</code> | 开始 HCCL_INFO 诊断输出，记录 HcclAllReduceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S73 / L105](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L105) | <code>    CHK_RET(HcomGetCommHandleByGroup(group, &amp;comm));</code> | 调用 HcomGetCommHandleByGroup 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S74 / L107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L107) | <code>    CHK_PRT_RET(sendCount == 0, HCCL_WARNING(&quot;input sendCount is 0, return all reduce success&quot;), HCCL_SUCCESS);</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S76 / L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L110) | <code>    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内</code> | 设置 HcclUs startut 为 TIME_NOW()；该值供下方当前分支使用。 |
| [S77 / L112](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L112) | <code>    OpParam param;</code> | 声明本阶段局部变量 OpParam param，实际值由后续查询/计算填写。 |
| [S78 / L114](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L114) | <code>    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, op, stream, param));</code> | 完成 AllReduce 入口环境与参数验证；返回值非成功时立即从当前函数返回该错误。 |
| [S82 / L119](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L119) | <code>    CHK_RET(HcclCheckTag(tag));</code> | 调用 HcclCheckTag 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S83 / L121](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L121) | <code>    int ret = sprintf_s(param.tag, sizeof(param.tag), &quot;%s&quot;, tag);</code> | 设置 ret 为 sprintf_s(param.tag, sizeof(param.tag), &quot;%s&quot;, tag)；该值供下方当前分支使用。 |
| [S84 / L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L123) | <code>    CHK_PRT_RET((ret &lt;= 0), HCCL_ERROR(&quot;failed to fill param.tag&quot;), HCCL_E_INTERNAL);</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S88 / L128](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L128) | <code>    ResPackGraphMode resPack;</code> | 声明本阶段局部变量 ResPackGraphMode resPack，实际值由后续查询/计算填写。 |
| [S90 / L131](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L131) | <code>    if (strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) != 0) {</code> | 分支条件为 strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) 不等于 0；成立进入本块，未成立继续后续分支。 |
| [S91 / L133](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L133) | <code>        HCCL_ERROR(&quot;failed to fill resPack.tag&quot;);</code> | 开始 HCCL_ERROR 诊断输出，记录 HcclAllReduceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S92 / L135](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L135) | <code>        return HCCL_E_INTERNAL;</code> | 终止当前函数并向上返回 内部错误；调用者 CHK_RET 决定是否继续向上传播。 |
| [S93 / L137](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L137) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S95 / L140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L140) | <code>    if (streams != nullptr &amp;&amp; streamCount &gt; 0) {</code> | 分支条件为 streams 不等于 nullptr 且 streamCount 大于 0；成立进入本块，未成立继续后续分支。 |
| [S96 / L142](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L142) | <code>        for (size_t i = 0; i &lt; streamCount; i++) {</code> | 在 HcclAllReduceGraphMode 中遍历 (size_t i = 0; i 小于 streamCount; i++ 指定的集合或索引区间；边界/迭代规则为 (size_t i = 0; i 小于 streamCount; i++。 |
| [S97 / L144](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L144) | <code>            resPack.streams.push_back(static_cast&lt;aclrtStream&gt;(streams[i]));</code> | 对 resPack 追加 static_cast&lt;aclrtStream&gt;(streams[i])，准备或更新本阶段列表。 |
| [S98 / L146](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L146) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S99 / L148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L148) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S101 / L151](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L151) | <code>    resPack.scratchMemAddr = scratchMemAddr;</code> | 设置 resPack.scratchMemAddr 为 scratchMemAddr；该值供下方当前分支使用。 |
| [S102 / L153](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L153) | <code>    resPack.scratchMemSize = scratchMemSize;</code> | 设置 resPack.scratchMemSize 为 scratchMemSize；该值供下方当前分支使用。 |
| [S103 / L155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L155) | <code>    std::string tagStr = tag;</code> | 设置 tagStr 为 tag；该值供下方当前分支使用。 |
| [S106 / L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L159) | <code>    CHK_RET(AllReduceEntryLog(</code> | 记录 AllReduce 本次调用信息；返回值非成功时立即从当前函数返回该错误。 |
| [S107 / L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L161) | <code>        sendBuf, recvBuf, sendCount, dataType, op, stream, param.tag, &quot;HcclAllReduceGraphMode&quot;, true));</code> | 续接本次错误检查/子调用实参：sendBuf, recvBuf, sendCount, dataType, op, stream, param.tag, &quot;HcclAllReduceGraphMode&quot;, true；返回行为由所在完整宏决定。 |
| [S109 / L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L164) | <code>    CHK_RET_AND_PRINT_IDE(</code> | 调用 CHK_RET_AND_PRINT_IDE 完成当前参数所指的子步骤；本行实参为 CHK_RET_AND_PRINT_IDE(。 |
| [S110 / L166](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L166) | <code>        AllReduceOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, op, comm, stream, resPack, param),</code> | 调用 AllReduceOutPlaceGraphMode 完成当前参数所指的子步骤；本行实参为 AllReduceOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, op, comm, stream, resPack, param),。 |
| [S111 / L168](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L168) | <code>        tagStr.c_str());</code> | 调用 c_str 完成当前参数所指的子步骤；本行实参为 tagStr.c_str())。 |
| [S112 / L170](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L170) | <code>    CHK_RET(LogHcclExit(&quot;HcclAllReduceGraphMode&quot;, param.tag, startut, true));</code> | 调用 LogHcclExit 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S114 / L173](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L173) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S115 / L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L175) | <code>}</code> | 结束 HcclAllReduceGraphMode 实现；其返回状态或已写回字段由调用者接收。 |


## 10. AllReduceInitAndCheck

解析环境，检查四个关键句柄，创建通信域关联 tag，验证 Rank、元素数量、可归约类型与运算组合。

完整范围：[all_reduce.cc:L179–L227](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L179-L227)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

功能与分支：

- [S137 / L210](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L210)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S118 / L179](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L179) | <code>HcclResult AllReduceInitAndCheck(</code> | 定义 AllReduceInitAndCheck 入口：解析环境，检查四个关键句柄，创建通信域关联 tag，验证 Rank、元素数量、可归约类型与运算组合。 |
| [S119 / L181](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L181) | <code>    HcclComm comm, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,</code> | 续接 AllReduceInitAndCheck 的入口参数/基类初始化：HcclComm comm, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,；引用参数按声明的 const 限制读写。 |
| [S120 / L183](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L183) | <code>    const aclrtStream stream, OpParam&amp; param)</code> | 续接 AllReduceInitAndCheck 的入口参数/基类初始化：const aclrtStream stream, OpParam&amp; param)；引用参数按声明的 const 限制读写。 |
| [S121 / L185](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L185) | <code>{</code> | 进入 AllReduceInitAndCheck 的实现作用域；解析环境，检查四个关键句柄，创建通信域关联 tag，验证 Rank、元素数量、可归约类型与运算组合。 |
| [S124 / L189](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L189) | <code>    CHK_RET(InitEnvConfig());</code> | 解析环境配置供模式与算法选择使用；返回值非成功时立即从当前函数返回该错误。 |
| [S127 / L193](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L193) | <code>    CHK_RET(CheckAllReduceInputPara(comm, sendBuf, recvBuf, stream));</code> | 拒绝空输入、输出、通信域或流句柄；返回值非成功时立即从当前函数返回该错误。 |
| [S128 / L195](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L195) | <code>    u32 rankSize = INVALID_VALUE_RANKSIZE;</code> | 设置 rankSize 为 INVALID_VALUE_RANKSIZE；该值供下方当前分支使用。 |
| [S129 / L197](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L197) | <code>    CHK_RET(HcclGetRankSize(comm, &amp;rankSize));</code> | 调用 HcclGetRankSize 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S130 / L199](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L199) | <code>    u32 userRank = INVALID_VALUE_RANKID;</code> | 设置 userRank 为 INVALID_VALUE_RANKID；该值供下方当前分支使用。 |
| [S131 / L201](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L201) | <code>    CHK_RET(HcclGetRankId(comm, &amp;userRank));</code> | 调用 HcclGetRankId 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S132 / L203](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L203) | <code>    CHK_RET(HcclGetCommName(comm, param.commName));</code> | 调用 HcclGetCommName 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S136 / L208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L208) | <code>    int ret = sprintf_s(param.tag, sizeof(param.tag), &quot;AllReduce_%s&quot;, param.commName);</code> | 设置 ret 为 sprintf_s(param.tag, sizeof(param.tag), &quot;AllReduce_%s&quot;, param.commName)；该值供下方当前分支使用。 |
| [S137 / L210](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L210) | <code>    CHK_PRT_RET((ret &lt;= 0), &quot;failed to fill param.tag&quot;, HCCL_E_INTERNAL);</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S139 / L213](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L213) | <code>    CHK_RET(HcclCheckTag(param.tag));</code> | 调用 HcclCheckTag 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S140 / L215](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L215) | <code>    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), param.tag);</code> | 调用 CHK_RET_AND_PRINT_IDE 完成当前参数所指的子步骤；本行实参为 CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), param.tag)。 |
| [S142 / L218](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L218) | <code>    CHK_RET(CheckCount(count));</code> | 调用 CheckCount 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S143 / L220](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L220) | <code>    CHK_RET(CheckDataType(dataType, true));</code> | 调用 CheckDataType 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S144 / L222](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L222) | <code>    CHK_RET(CheckReduceOp(dataType, op));</code> | 调用 CheckReduceOp 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S146 / L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L225) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S147 / L227](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L227) | <code>}</code> | 结束 AllReduceInitAndCheck 实现；其返回状态或已写回字段由调用者接收。 |


## 11. CheckAllReduceInputPara

分别报告并拒绝空 stream、comm、sendBuf、recvBuf；不在此计算网络资源。

完整范围：[all_reduce.cc:L230–L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L230-L272)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S149 / L230](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L230) | <code>HcclResult</code> | 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。 |
| [S150 / L232](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L232) | <code>CheckAllReduceInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)</code> | 定义 CheckAllReduceInputPara 入口：分别报告并拒绝空 stream、comm、sendBuf、recvBuf；不在此计算网络资源。 |
| [S151 / L234](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L234) | <code>{</code> | 进入 CheckAllReduceInputPara 的实现作用域；分别报告并拒绝空 stream、comm、sendBuf、recvBuf；不在此计算网络资源。 |
| [S153 / L237](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L237) | <code>    RPT_INPUT_ERR(</code> | 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。 |
| [S154 / L239](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L239) | <code>        stream == nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),</code> | 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：stream 等于 nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),；由其完整表达式完成参数组装、检查或结果写回。 |
| [S155 / L241](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L241) | <code>        std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;stream&quot;, &quot;non-null pointer&quot;}));</code> | 建立本阶段局部对象 std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;stream&quot;, &quot;non-null pointer&quot;}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。 |
| [S156 / L243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L243) | <code>    CHK_PTR_NULL(stream);</code> | 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(stream)。 |
| [S157 / L245](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L245) | <code>    RPT_INPUT_ERR(</code> | 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。 |
| [S158 / L247](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L247) | <code>        comm == nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),</code> | 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：comm 等于 nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),；由其完整表达式完成参数组装、检查或结果写回。 |
| [S159 / L249](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L249) | <code>        std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;comm&quot;, &quot;non-null pointer&quot;}));</code> | 建立本阶段局部对象 std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;comm&quot;, &quot;non-null pointer&quot;}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。 |
| [S160 / L251](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L251) | <code>    CHK_PTR_NULL(comm);</code> | 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(comm)。 |
| [S161 / L253](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L253) | <code>    RPT_INPUT_ERR(</code> | 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。 |
| [S162 / L255](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L255) | <code>        sendBuf == nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),</code> | 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：sendBuf 等于 nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),；由其完整表达式完成参数组装、检查或结果写回。 |
| [S163 / L257](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L257) | <code>        std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;sendBuf&quot;, &quot;non-null pointer&quot;}));</code> | 建立本阶段局部对象 std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;sendBuf&quot;, &quot;non-null pointer&quot;}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。 |
| [S164 / L259](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L259) | <code>    CHK_PTR_NULL(sendBuf);</code> | 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(sendBuf)。 |
| [S165 / L261](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L261) | <code>    RPT_INPUT_ERR(</code> | 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。 |
| [S166 / L263](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L263) | <code>        recvBuf == nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),</code> | 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：recvBuf 等于 nullptr, &quot;EI0003&quot;, std::vector&lt;std::string&gt;({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;}),；由其完整表达式完成参数组装、检查或结果写回。 |
| [S167 / L265](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L265) | <code>        std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;recvBuf&quot;, &quot;non-null pointer&quot;}));</code> | 建立本阶段局部对象 std::vector&lt;std::string&gt;({&quot;HcclAllReduce&quot;, &quot;nullptr&quot;, &quot;recvBuf&quot;, &quot;non-null pointer&quot;}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。 |
| [S168 / L267](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L267) | <code>    CHK_PTR_NULL(recvBuf);</code> | 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(recvBuf)。 |
| [S170 / L270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L270) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S171 / L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L272) | <code>}</code> | 结束 CheckAllReduceInputPara 实现；其返回状态或已写回字段由调用者接收。 |


## 12. FillAllReduceOpParam

把用户地址、元素 count、类型、归约 op、模式与设备身份组装成统一 OpParam；输入输出字节容量相等。

完整范围：[all_reduce.cc:L275–L332](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L275-L332)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S173 / L275](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L275) | <code>HcclResult FillAllReduceOpParam(</code> | 定义 FillAllReduceOpParam 入口：把用户地址、元素 count、类型、归约 op、模式与设备身份组装成统一 OpParam；输入输出字节容量相等。 |
| [S174 / L277](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L277) | <code>    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, const HcclComm comm,</code> | 续接 FillAllReduceOpParam 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, const HcclComm comm,；引用参数按声明的 const 限制读写。 |
| [S175 / L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L279) | <code>    aclrtStream stream, OpMode opMode, OpParam&amp; param)</code> | 续接 FillAllReduceOpParam 的入口参数/基类初始化：aclrtStream stream, OpMode opMode, OpParam&amp; param)；引用参数按声明的 const 限制读写。 |
| [S176 / L281](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L281) | <code>{</code> | 进入 FillAllReduceOpParam 的实现作用域；把用户地址、元素 count、类型、归约 op、模式与设备身份组装成统一 OpParam；输入输出字节容量相等。 |
| [S178 / L284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L284) | <code>    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];</code> | 设置 单元素字节数 为 类型到元素字节数的查找表[dataType]；该值供下方当前分支使用。 |
| [S179 / L286](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L286) | <code>    u64 outputSize = count * perDataSize;</code> | 设置 outputSize 为 count * 单元素字节数；该值供下方当前分支使用。 |
| [S180 / L288](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L288) | <code>    u64 inputSize = outputSize;</code> | 设置 inputSize 为 outputSize；该值供下方当前分支使用。 |
| [S183 / L292](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L292) | <code>    param.hcclComm = comm;</code> | 设置 param.hcclComm 为 comm；该值供下方当前分支使用。 |
| [S184 / L294](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L294) | <code>    param.stream = stream;</code> | 设置 param.stream 为 stream；该值供下方当前分支使用。 |
| [S185 / L296](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L296) | <code>    param.reduceType = op;</code> | 设置 归约运算 为 op；该值供下方当前分支使用。 |
| [S186 / L298](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L298) | <code>    param.opMode = opMode;</code> | 设置 调用模式 为 opMode；该值供下方当前分支使用。 |
| [S187 / L300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L300) | <code>    param.supportSymmetricMemory = false;</code> | 设置 对称内存标志 为 false；该值供下方当前分支使用。 |
| [S189 / L303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L303) | <code>    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;</code> | 设置 HcclDevType deviceType 为 HcclDevType::DEV_TYPE_COUNT；该值供下方当前分支使用。 |
| [S190 / L305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L305) | <code>    CHK_RET(HcclGetDeviceType(deviceType));</code> | 调用 HcclGetDeviceType 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S194 / L310](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L310) | <code>    param.inputPtr = sendBuf;</code> | 设置 用户输入基址 为 sendBuf；该值供下方当前分支使用。 |
| [S195 / L312](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L312) | <code>    param.inputSize = inputSize;</code> | 设置 输入字节容量 为 inputSize；该值供下方当前分支使用。 |
| [S196 / L314](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L314) | <code>    param.outputPtr = recvBuf;</code> | 设置 用户输出基址 为 recvBuf；该值供下方当前分支使用。 |
| [S197 / L316](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L316) | <code>    param.outputSize = outputSize;</code> | 设置 输出字节容量 为 outputSize；该值供下方当前分支使用。 |
| [S198 / L318](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L318) | <code>    param.DataDes.count = count;</code> | 设置 本 Rank 输入元素数 为 count；该值供下方当前分支使用。 |
| [S199 / L320](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L320) | <code>    param.DataDes.dataType = dataType;</code> | 设置 输入元素类型 为 dataType；该值供下方当前分支使用。 |
| [S200 / L322](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L322) | <code>    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;</code> | 设置 param.opType 为 HcclCMDType::HCCL_CMD_ALLREDUCE；该值供下方当前分支使用。 |
| [S201 / L324](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L324) | <code>    param.enableDetour = false;</code> | 设置 param.enableDetour 为 false；该值供下方当前分支使用。 |
| [S202 / L326](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L326) | <code>    param.deviceType = deviceType;</code> | 设置 param.deviceType 为 deviceType；该值供下方当前分支使用。 |
| [S203 / L328](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L328) | <code>    param.reduceType = op;</code> | 设置 归约运算 为 op；该值供下方当前分支使用。 |
| [S204 / L330](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L330) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S205 / L332](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L332) | <code>}</code> | 结束 FillAllReduceOpParam 实现；其返回状态或已写回字段由调用者接收。 |


## 13. AllReduceOutPlaceCommon

新流程分发：兼容 CCU、快速发射、AIV 回放、单 Rank、对称内存筛选，最后选算法并进入 HcclExecOp。

完整范围：[all_reduce.cc:L335–L467](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L335-L467)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

功能与分支：

- [S220 / L355](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L355)：分支条件为 opMode 等于 单算子 OPBASE 且 GetHcommVersion() 等于 CANN_VERSION(9, 0, 0；成立进入本块，未成立继续后续分支。

- [S227 / L367](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L367)：分支条件为 (opMode 等于 单算子 OPBASE) 且 ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx；成立进入本块，未成立继续后续分支。

- [S232 / L375](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L375)：分支条件为 实际通信引擎 等于 CommEngine::AIV 引擎；成立进入本块，未成立继续后续分支。

- [S235 / L381](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L381)：分支条件为 aivCacheHit；成立进入本块，未成立继续后续分支。

- [S244 / L396](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L396)：分支条件为 userRankSize 等于 1；成立进入本块，未成立继续后续分支。

- [S251 / L408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L408)：分支条件为 GetHcommVersion() 至少 CANN_VERSION(9, 1, 0) 且 调用模式 等于 单算子 OPBASE；成立进入本块，未成立继续后续分支。

- [S261 / L424](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L424)：分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑 且 !第零层 PCIe 混合标志；成立进入本块，未成立继续后续分支。

- [S264 / L430](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L430)：分支条件为 !isTwoLevelMeshNhrOmni；成立进入本块，未成立继续后续分支。

- [S275 / L451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L451)：分支条件为 !supportSymmetricMemory；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S207 / L335](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L335) | <code>HcclResult AllReduceOutPlaceCommon(</code> | 定义 AllReduceOutPlaceCommon 入口：新流程分发：兼容 CCU、快速发射、AIV 回放、单 Rank、对称内存筛选，最后选算法并进入 HcclExecOp。 |
| [S208 / L337](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L337) | <code>    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,</code> | 续接 AllReduceOutPlaceCommon 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。 |
| [S209 / L339](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L339) | <code>    aclrtStream stream, OpMode opMode, const ResPackGraphMode&amp; resPack, OpParam&amp; param)</code> | 续接 AllReduceOutPlaceCommon 的入口参数/基类初始化：aclrtStream stream, OpMode opMode, const ResPackGraphMode&amp; resPack, OpParam&amp; param)；引用参数按声明的 const 限制读写。 |
| [S210 / L341](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L341) | <code>{</code> | 进入 AllReduceOutPlaceCommon 的实现作用域；新流程分发：兼容 CCU、快速发射、AIV 回放、单 Rank、对称内存筛选，最后选算法并进入 HcclExecOp。 |
| [S211 / L343](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L343) | <code>    HCCL_INFO(&quot;Start to execute AllReduceOutPlace&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceCommon 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S214 / L347](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L347) | <code>    CHK_RET(FillAllReduceOpParam(sendBuf, recvBuf, count, dataType, op, comm, stream, opMode, param));</code> | 建立统一算子描述；返回值非成功时立即从当前函数返回该错误。 |
| [S216 / L350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L350) | <code>    CHK_RET(HcclGetOpExpansionMode(comm, param));</code> | 调用 HcclGetOpExpansionMode 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S220 / L355](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L355) | <code>    if (opMode == OpMode::OPBASE &amp;&amp; GetHcommVersion() == CANN_VERSION(9, 0, 0)</code> | 分支条件为 opMode 等于 单算子 OPBASE 且 GetHcommVersion() 等于 CANN_VERSION(9, 0, 0；成立进入本块，未成立继续后续分支。 |
| [S221 / L357](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L357) | <code>        &amp;&amp; param.engine == CommEngine::COMM_ENGINE_CCU) {</code> | 补充同一条件的 并且 子条件：实际通信引擎 等于 CommEngine::CCU 引擎。 |
| [S222 / L359](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L359) | <code>        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);</code> | 直接返回 调用 HcclAllReduceInner 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S223 / L361](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L361) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S226 / L365](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L365) | <code>    CcuFastLaunchCtx* ccuFastLaunchCtx = nullptr;</code> | 设置 CcuFastLaunchCtx* ccuFastLaunchCtx 为 nullptr；该值供下方当前分支使用。 |
| [S227 / L367](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L367) | <code>    if ((opMode == OpMode::OPBASE) &amp;&amp; ShouldGoCcuFastLaunch(comm, param, &amp;ccuFastLaunchCtx)) {</code> | 分支条件为 (opMode 等于 单算子 OPBASE) 且 ShouldGoCcuFastLaunch(comm, param, &amp;ccuFastLaunchCtx；成立进入本块，未成立继续后续分支。 |
| [S228 / L369](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L369) | <code>        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);</code> | 直接返回 调用 HcclExecOpCcuFastLaunch 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S229 / L371](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L371) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S232 / L375](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L375) | <code>    if (param.engine == CommEngine::COMM_ENGINE_AIV) {</code> | 分支条件为 实际通信引擎 等于 CommEngine::AIV 引擎；成立进入本块，未成立继续后续分支。 |
| [S233 / L377](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L377) | <code>        bool aivCacheHit = false;</code> | 设置 aivCacheHit 为 false；该值供下方当前分支使用。 |
| [S234 / L379](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L379) | <code>        CHK_RET(HcclAivCacheCheckAndReplay(comm, param, aivCacheHit));</code> | 调用 HcclAivCacheCheckAndReplay 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S235 / L381](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L381) | <code>        if (aivCacheHit) {</code> | 分支条件为 aivCacheHit；成立进入本块，未成立继续后续分支。 |
| [S236 / L383](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L383) | <code>            return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S237 / L385](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L385) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S238 / L387](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L387) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S242 / L392](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L392) | <code>    u32 userRankSize;</code> | 声明本阶段局部变量 u32 userRankSize，实际值由后续查询/计算填写。 |
| [S243 / L394](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L394) | <code>    CHK_RET(HcclGetRankSize(comm, &amp;userRankSize));</code> | 调用 HcclGetRankSize 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S244 / L396](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L396) | <code>    if (userRankSize == 1) {</code> | 分支条件为 userRankSize 等于 1；成立进入本块，未成立继续后续分支。 |
| [S245 / L398](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L398) | <code>        HCCL_WARNING(&quot;[%s] ranksize == 1, enter SingleRankProc&quot;, __func__);</code> | 开始 HCCL_WARNING 诊断输出，记录 AllReduceOutPlaceCommon 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S246 / L400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L400) | <code>        CHK_RET(SingleRankProc(comm, param));</code> | 调用 SingleRankProc 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。 |
| [S247 / L402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L402) | <code>        return HcclResult::HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S248 / L404](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L404) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S251 / L408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L408) | <code>    if (GetHcommVersion() &gt;= CANN_VERSION(9, 1, 0) &amp;&amp; param.opMode == OpMode::OPBASE) {</code> | 分支条件为 GetHcommVersion() 至少 CANN_VERSION(9, 1, 0) 且 调用模式 等于 单算子 OPBASE；成立进入本块，未成立继续后续分支。 |
| [S252 / L410](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L410) | <code>        CheckAndSetSymmetricMemory(param);</code> | 调用 CheckAndSetSymmetricMemory 完成当前参数所指的子步骤；本行实参为 CheckAndSetSymmetricMemory(param)。 |
| [S253 / L412](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L412) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S256 / L416](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L416) | <code>    std::string algName;</code> | 建立本阶段局部对象 std::string algName，供 AllReduceOutPlaceCommon 下方参数组装和子调用使用。 |
| [S257 / L418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L418) | <code>    std::unique_ptr&lt;TopoInfoWithNetLayerDetails&gt; topoInfo = std::make_unique&lt;TopoInfoWithNetLayerDetails&gt;();</code> | 设置 std::unique_ptr&lt;TopoInfoWithNetLayerDetails&gt; topoInfo 为 std::make_unique&lt;TopoInfoWithNetLayerDetails&gt;()；该值供下方当前分支使用。 |
| [S258 / L420](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L420) | <code>    CHK_RET(Selector(comm, param, topoInfo, algName));</code> | 根据拓扑和配置选算法；返回值非成功时立即从当前函数返回该错误。 |
| [S261 / L424](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L424) | <code>    if (topoInfo-&gt;level0Topo == Level0Shape::MESH_1D_CLOS &amp;&amp; !topoInfo-&gt;level0PcieMix) {</code> | 分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑 且 !第零层 PCIe 混合标志；成立进入本块，未成立继续后续分支。 |
| [S262 / L426](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L426) | <code>        const bool isTwoLevelMeshNhrOmni</code> | 建立本阶段局部对象 const bool isTwoLevelMeshNhrOmni，供 AllReduceOutPlaceCommon 下方参数组装和子调用使用。 |
| [S263 / L428](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L428) | <code>            = algName == &quot;AicpuAllReducePipeLineMeshNHR&quot; &amp;&amp; topoInfo-&gt;topoLevelNums == TOPO_LEVEL_NUM_1;</code> | 续接 AllReduceOutPlaceCommon 当前语句的具体实参/字段：= algName 等于 &quot;AicpuAllReducePipeLineMeshNHR&quot; 且 topoInfo-&gt;topoLevelNums 等于 TOPO_LEVEL_NUM_1；由其完整表达式完成参数组装、检查或结果写回。 |
| [S264 / L430](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L430) | <code>        if (!isTwoLevelMeshNhrOmni) {</code> | 分支条件为 !isTwoLevelMeshNhrOmni；成立进入本块，未成立继续后续分支。 |
| [S265 / L432](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L432) | <code>            param.supportSymmetricMemory = false;</code> | 设置 对称内存标志 为 false；该值供下方当前分支使用。 |
| [S266 / L434](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L434) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S267 / L436](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L436) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S269 / L439](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L439) | <code>        const bool supportSymmetricMemory</code> | 建立本阶段局部对象 const bool supportSymmetricMemory，供 AllReduceOutPlaceCommon 下方参数组装和子调用使用。 |
| [S270 / L441](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L441) | <code>            = (param.engine == CommEngine::COMM_ENGINE_AICPU_TS &amp;&amp; topoInfo-&gt;level0Topo == Level0Shape::MESH_1D</code> | 续接 AllReduceOutPlaceCommon 当前语句的具体实参/字段：= (实际通信引擎 等于 CommEngine::AICPU_TS 引擎 且 第零层拓扑形状 等于 单层 Mesh1D；由其完整表达式完成参数组装、检查或结果写回。 |
| [S271 / L443](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L443) | <code>               &amp;&amp; param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_INT64</code> | 补充同一条件的 并且 子条件：输入元素类型 不等于 HcclDataType::有符号 64 位整数。 |
| [S272 / L445](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L445) | <code>               &amp;&amp; param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_UINT64</code> | 补充同一条件的 并且 子条件：输入元素类型 不等于 HcclDataType::无符号 64 位整数。 |
| [S273 / L447](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L447) | <code>               &amp;&amp; param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_FP64</code> | 补充同一条件的 并且 子条件：输入元素类型 不等于 HcclDataType::FP64 类型。 |
| [S274 / L449](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L449) | <code>               &amp;&amp; param.reduceType != HcclReduceOp::HCCL_REDUCE_PROD);</code> | 补充同一条件的 并且 子条件：归约运算 不等于 HcclReduceOp::乘积归约。 |
| [S275 / L451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L451) | <code>        if (!supportSymmetricMemory) {</code> | 分支条件为 !supportSymmetricMemory；成立进入本块，未成立继续后续分支。 |
| [S276 / L453](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L453) | <code>            param.supportSymmetricMemory = false;</code> | 设置 对称内存标志 为 false；该值供下方当前分支使用。 |
| [S277 / L455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L455) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S278 / L457](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L457) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S281 / L461](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L461) | <code>    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));</code> | 进入资源与执行器公共调度；返回值非成功时立即从当前函数返回该错误。 |
| [S282 / L463](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L463) | <code>    HCCL_INFO(&quot;Execute AllReduceOutPlace success.&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceCommon 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S283 / L465](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L465) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S284 / L467](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L467) | <code>}</code> | 结束 AllReduceOutPlaceCommon 实现；其返回状态或已写回字段由调用者接收。 |


## 14. AllReduceEntryLog

可配置或强制打印设备、流与输入输出地址等入口信息；日志组装失败仅警告继续。

完整范围：[all_reduce.cc:L470–L511](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L470-L511)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

功能与分支：

- [S290 / L478](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L478)：分支条件为 forceLog 或 GetExternalInputHcclEnableEntryLog(；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S286 / L470](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L470) | <code>HcclResult AllReduceEntryLog(</code> | 定义 AllReduceEntryLog 入口：可配置或强制打印设备、流与输入输出地址等入口信息；日志组装失败仅警告继续。 |
| [S287 / L472](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L472) | <code>    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, aclrtStream stream,</code> | 续接 AllReduceEntryLog 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, aclrtStream stream,；引用参数按声明的 const 限制读写。 |
| [S288 / L474](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L474) | <code>    const char* tag, const std::string&amp; opName, bool forceLog)</code> | 续接 AllReduceEntryLog 的入口参数/基类初始化：const char* tag, const std::string&amp; opName, bool forceLog)；引用参数按声明的 const 限制读写。 |
| [S289 / L476](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L476) | <code>{</code> | 进入 AllReduceEntryLog 的实现作用域；可配置或强制打印设备、流与输入输出地址等入口信息；日志组装失败仅警告继续。 |
| [S290 / L478](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L478) | <code>    if (forceLog &#124;&#124; GetExternalInputHcclEnableEntryLog()) {</code> | 分支条件为 forceLog 或 GetExternalInputHcclEnableEntryLog(；成立进入本块，未成立继续后续分支。 |
| [S291 / L480](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L480) | <code>        s32 deviceId = 0;</code> | 设置 s32 deviceId 为 0；该值供下方当前分支使用。 |
| [S292 / L482](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L482) | <code>        ACLCHECK(aclrtGetDevice(&amp;deviceId));</code> | 调用 ACLCHECK 完成当前参数所指的子步骤；本行实参为 ACLCHECK(aclrtGetDevice(&amp;deviceId))。 |
| [S293 / L484](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L484) | <code>        s32 streamId = 0;</code> | 设置 s32 streamId 为 0；该值供下方当前分支使用。 |
| [S294 / L486](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L486) | <code>        ACLCHECK(aclrtStreamGetId(stream, &amp;streamId));</code> | 调用 ACLCHECK 完成当前参数所指的子步骤；本行实参为 ACLCHECK(aclrtStreamGetId(stream, &amp;streamId))。 |
| [S295 / L488](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L488) | <code>        char stackLogBuffer[LOG_TMPBUF_SIZE];</code> | 续接 AllReduceEntryLog 当前语句的具体实参/字段：char stackLogBuffer[LOG_TMPBUF_SIZE]；由其完整表达式完成参数组装、检查或结果写回。 |
| [S296 / L490](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L490) | <code>        s32 ret = snprintf_s(</code> | 设置 s32 ret 为 snprintf_s(；该值供下方当前分支使用。 |
| [S297 / L492](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L492) | <code>            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,</code> | 续接 AllReduceEntryLog 当前语句的具体实参/字段：stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,；由其完整表达式完成参数组装、检查或结果写回。 |
| [S298 / L494](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L494) | <code>            &quot;tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceId[%d]&quot;,</code> | 续接本次调用的字符串常量 &quot;tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceId[%d]&quot;,，由所在注册/日志/条件语句整体使用。 |
| [S299 / L496](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L496) | <code>            tag, sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),</code> | 调用 GetDataTypeEnumStr 完成当前参数所指的子步骤；本行实参为 tag, sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),。 |
| [S300 / L498](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L498) | <code>            streamId, deviceId);</code> | 续接 AllReduceEntryLog 当前语句的具体实参/字段：streamId, deviceId)；由其完整表达式完成参数组装、检查或结果写回。 |
| [S302 / L501](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L501) | <code>        CHK_PRT_CONT(ret == -1, HCCL_WARNING(&quot;Failed to build log info, tag[%s].&quot;, tag));</code> | 为 AllReduceEntryLog 的诊断/错误宏提供实参：CHK_PRT_CONT(ret 等于 -1, HCCL_WARNING(&quot;Failed to build log info, tag[%s].&quot;, tag，与前面的格式占位依次对应。 |
| [S303 / L503](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L503) | <code>        std::string logInfo = &quot;Entry-&quot; + opName + &quot;:&quot; + std::string(stackLogBuffer);</code> | 设置 logInfo 为 &quot;Entry-&quot; + opName + &quot;:&quot; + std::string(stackLogBuffer)；该值供下方当前分支使用。 |
| [S304 / L505](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L505) | <code>        HCCL_RUN_INFO(&quot;%s&quot;, logInfo.c_str());</code> | 调用 HCCL_RUN_INFO 完成当前参数所指的子步骤；本行实参为 HCCL_RUN_INFO(&quot;%s&quot;, logInfo.c_str())。 |
| [S305 / L507](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L507) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S306 / L509](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L509) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S307 / L511](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L511) | <code>}</code> | 结束 AllReduceEntryLog 实现；其返回状态或已写回字段由调用者接收。 |


## 15. AllReduceOutPlaceGraphMode

用 OFFLOAD 和外部图资源包转调公共 AllReduce 实现。

完整范围：[all_reduce.cc:L514–L532](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L514-L532)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S309 / L514](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L514) | <code>HcclResult AllReduceOutPlaceGraphMode(</code> | 定义 AllReduceOutPlaceGraphMode 入口：用 OFFLOAD 和外部图资源包转调公共 AllReduce 实现。 |
| [S310 / L516](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L516) | <code>    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,</code> | 续接 AllReduceOutPlaceGraphMode 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。 |
| [S311 / L518](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L518) | <code>    aclrtStream stream, const ResPackGraphMode&amp; resPack, OpParam&amp; param)</code> | 续接 AllReduceOutPlaceGraphMode 的入口参数/基类初始化：aclrtStream stream, const ResPackGraphMode&amp; resPack, OpParam&amp; param)；引用参数按声明的 const 限制读写。 |
| [S312 / L520](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L520) | <code>{</code> | 进入 AllReduceOutPlaceGraphMode 的实现作用域；用 OFFLOAD 和外部图资源包转调公共 AllReduce 实现。 |
| [S313 / L522](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L522) | <code>    HCCL_INFO(&quot;Start to execute AllReduceOutPlaceGraphMode&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S314 / L524](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L524) | <code>    CHK_RET(</code> | 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。 |
| [S315 / L526](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L526) | <code>        AllReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OFFLOAD, resPack, param));</code> | 进入共享 AllReduce 模式分发；本行实参为 AllReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, comm, stream, 图模式 OFFLOAD, resPack, param))。 |
| [S316 / L528](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L528) | <code>    HCCL_INFO(&quot;Execute AllReduceOutPlaceGraphMode success.&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S317 / L530](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L530) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S318 / L532](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L532) | <code>}</code> | 结束 AllReduceOutPlaceGraphMode 实现；其返回状态或已写回字段由调用者接收。 |


## 16. AllReduceOutPlace

用 OPBASE 和默认资源包转调公共 AllReduce 实现。

完整范围：[all_reduce.cc:L535–L553](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L535-L553)；文件：`hccl/src/ops/all_reduce/all_reduce.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S320 / L535](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L535) | <code>HcclResult AllReduceOutPlace(</code> | 定义 AllReduceOutPlace 入口：用 OPBASE 和默认资源包转调公共 AllReduce 实现。 |
| [S321 / L537](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L537) | <code>    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,</code> | 续接 AllReduceOutPlace 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。 |
| [S322 / L539](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L539) | <code>    aclrtStream stream, OpParam&amp; param)</code> | 续接 AllReduceOutPlace 的入口参数/基类初始化：aclrtStream stream, OpParam&amp; param)；引用参数按声明的 const 限制读写。 |
| [S323 / L541](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L541) | <code>{</code> | 进入 AllReduceOutPlace 的实现作用域；用 OPBASE 和默认资源包转调公共 AllReduce 实现。 |
| [S324 / L543](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L543) | <code>    HCCL_INFO(&quot;Start to execute AllReduceOutPlace&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlace 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S325 / L545](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L545) | <code>    CHK_RET(AllReduceOutPlaceCommon(</code> | 进入共享 AllReduce 模式分发；返回值非成功时立即从当前函数返回该错误。 |
| [S326 / L547](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L547) | <code>        sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OPBASE, ResPackGraphMode(), param));</code> | 调用 ResPackGraphMode 完成当前参数所指的子步骤；本行实参为 sendBuf, recvBuf, count, dataType, op, comm, stream, 单算子 OPBASE, ResPackGraphMode(), param))。 |
| [S327 / L549](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L549) | <code>    HCCL_INFO(&quot;Execute AllReduceOutPlace success.&quot;);</code> | 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlace 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S328 / L551](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L551) | <code>    return HCCL_SUCCESS;</code> | 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。 |
| [S329 / L553](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/all_reduce.cc#L553) | <code>}</code> | 结束 AllReduceOutPlace 实现；其返回状态或已写回字段由调用者接收。 |


## 17. AllReduceAutoSelector::SelectCcuMsAlgo

旧自动选择器的 CCU_MS 能力门槛：拒绝 strict、多拓扑层、INT8、PROD 和 64 位类型，匹配 Mesh 候选。

完整范围：[all_reduce_auto_selector.cc:L41–L106](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L41-L106)；文件：`hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc`。

功能与分支：

- [S48 / L55](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L55)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S53 / L64](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L64)：分支条件为 topoInfo->topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。

- [S59 / L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L74)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S66 / L86](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L86)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S71 / L95](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L95)：分支条件为 Is64BitDataType(输入元素类型；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S40 / L41](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L41) | <code>SelectorStatus AllReduceAutoSelector::SelectCcuMsAlgo(</code> | 定义 SelectCcuMsAlgo 入口：旧自动选择器的 CCU_MS 能力门槛：拒绝 strict、多拓扑层、INT8、PROD 和 64 位类型，匹配 Mesh 候选。 |
| [S41 / L43](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L43) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,</code> | 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,；引用参数按声明的 const 限制读写。 |
| [S42 / L45](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L45) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName) const</code> | 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S43 / L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L47) | <code>{</code> | 进入 SelectCcuMsAlgo 的实现作用域；旧自动选择器的 CCU_MS 能力门槛：拒绝 strict、多拓扑层、INT8、PROD 和 64 位类型，匹配 Mesh 候选。 |
| [S44 / L49](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L49) | <code>    (void)configAlgMap;</code> | 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S45 / L51](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L51) | <code>    HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]&quot;, __func__, topoInfo-&gt;topoLevelNums);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S48 / L55](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L55) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S49 / L57](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L57) | <code>        IsNeedStrictModeForOrderPreserved(opParam, topoInfo-&gt;userRankSize),</code> | 调用 IsNeedStrictModeForOrderPreserved 完成当前参数所指的子步骤；本行实参为 IsNeedStrictModeForOrderPreserved(opParam, 通信域 Rank 总数),。 |
| [S50 / L59](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L59) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] DETERMINISTIC_STRICT mode not supported for CCU_MS, fallback to AICPU.&quot;),</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S51 / L61](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L61) | <code>        SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuMsAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S53 / L64](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L64) | <code>    if (topoInfo-&gt;topoLevelNums &gt; 1) {</code> | 分支条件为 topoInfo-&gt;topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。 |
| [S54 / L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L66) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] levelNum &gt; 1 is not supported yet for ccu_ms mode.&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S55 / L68](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L68) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S56 / L70](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L70) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S59 / L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L74) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S60 / L76](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L76) | <code>        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,</code> | 续接本次错误检查/子调用实参：输入元素类型 等于 HcclDataType::INT8 类型；返回行为由所在完整宏决定。 |
| [S61 / L78](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L78) | <code>        HCCL_DEBUG(</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S62 / L80](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L80) | <code>            &quot;[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.&quot;, opParam.DataDes.dataType),</code> | 续接 SelectCcuMsAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S63 / L82](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L82) | <code>        SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuMsAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S66 / L86](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L86) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S67 / L88](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L88) | <code>        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,</code> | 续接本次错误检查/子调用实参：归约运算 等于 HcclReduceOp::乘积归约；返回行为由所在完整宏决定。 |
| [S68 / L90](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L90) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.&quot;, opParam.reduceType),</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S69 / L92](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L92) | <code>        SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuMsAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S71 / L95](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L95) | <code>    if (Is64BitDataType(opParam.DataDes.dataType)) {</code> | 分支条件为 Is64BitDataType(输入元素类型；成立进入本块，未成立继续后续分支。 |
| [S72 / L97](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L97) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S73 / L99](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L99) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S74 / L101](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L101) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S76 / L104](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L104) | <code>    return SelectMeshAlgo(topoInfo, opParam, selectAlgName);</code> | 直接返回 调用 SelectMeshAlgo 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S77 / L106](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L106) | <code>}</code> | 结束 SelectCcuMsAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 18. AllReduceAutoSelector::SelectMeshUBXAlgo

CCU_MS 下根据 Mesh/CLOS 实例关系、Rank 数和数据阈值选择 UBX 并发、流水线或单 Mesh。

完整范围：[all_reduce_auto_selector.cc:L109–L179](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L109-L179)；文件：`hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc`。

功能与分支：

- [S85 / L120](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L120)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S88 / L126](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L126)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S91 / L132](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L132)：分支条件为 isMeshNumEqualToClosNum 且 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_CONCURRENT_ALGO；成立进入本块，未成立继续后续分支。

- [S93 / L135](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L135)：分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。

- [S100 / L147](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L147)：分支条件为 isClosNumMultipleOfMeshNum 且 !IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。

- [S101 / L149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L149)：分支条件为 本 Rank 数据字节数 小于 OMNI_UBX_AR_MS_DATA_SIZE；成立进入本块，未成立继续后续分支。

- [S107 / L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L161)：分支条件为 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S79 / L109](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L109) | <code>SelectorStatus AllReduceAutoSelector::SelectMeshUBXAlgo(</code> | 定义 SelectMeshUBXAlgo 入口：CCU_MS 下根据 Mesh/CLOS 实例关系、Rank 数和数据阈值选择 UBX 并发、流水线或单 Mesh。 |
| [S80 / L111](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L111) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, std::string&amp; selectAlgName, u64 dataSize) const</code> | 续接 SelectMeshUBXAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, std::string&amp; 算法名输出参数, u64 本 Rank 数据字节数) const；引用参数按声明的 const 限制读写。 |
| [S81 / L113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L113) | <code>{</code> | 进入 SelectMeshUBXAlgo 的实现作用域；CCU_MS 下根据 Mesh/CLOS 实例关系、Rank 数和数据阈值选择 UBX 并发、流水线或单 Mesh。 |
| [S83 / L116](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L116) | <code>    bool isMeshNumEqualToClosNum = false;</code> | 设置 isMeshNumEqualToClosNum 为 false；该值供下方当前分支使用。 |
| [S84 / L118](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L118) | <code>    bool isClosNumMultipleOfMeshNum = false;</code> | 设置 isClosNumMultipleOfMeshNum 为 false；该值供下方当前分支使用。 |
| [S85 / L120](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L120) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S86 / L122](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L122) | <code>        CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,</code> | 调用 CheckMeshNumEqualToClosNum 完成当前参数所指的子步骤；本行实参为 CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) 不等于 成功状态,。 |
| [S87 / L124](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L124) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed.&quot;), SelectorStatus::NOT_MATCH);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S88 / L126](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L126) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S89 / L128](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L128) | <code>        CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,</code> | 调用 CheckClosNumMultipleOfMeshNum 完成当前参数所指的子步骤；本行实参为 CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) 不等于 成功状态,。 |
| [S90 / L130](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L130) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed.&quot;), SelectorStatus::NOT_MATCH);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S91 / L132](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L132) | <code>    if (isMeshNumEqualToClosNum &amp;&amp; topoInfo-&gt;userRankSize &lt;= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {</code> | 分支条件为 isMeshNumEqualToClosNum 且 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_CONCURRENT_ALGO；成立进入本块，未成立继续后续分支。 |
| [S93 / L135](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L135) | <code>        if (IsSmallData(dataSize)) {</code> | 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。 |
| [S95 / L138](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L138) | <code>            selectAlgName = &quot;CcuMSAllReduceSoleMeshOneShot&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceSoleMeshOneShot&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S96 / L140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L140) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S98 / L143](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L143) | <code>            selectAlgName = &quot;CcuMSAllReduceConcurMeshNHRMultiLink&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceConcurMeshNHRMultiLink&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S99 / L145](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L145) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S100 / L147](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L147) | <code>    } else if (isClosNumMultipleOfMeshNum &amp;&amp; !IsSmallData(dataSize)) {</code> | 分支条件为 isClosNumMultipleOfMeshNum 且 !IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。 |
| [S101 / L149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L149) | <code>        if (dataSize &lt; OMNI_UBX_AR_MS_DATA_SIZE) {</code> | 分支条件为 本 Rank 数据字节数 小于 OMNI_UBX_AR_MS_DATA_SIZE；成立进入本块，未成立继续后续分支。 |
| [S102 / L151](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L151) | <code>            HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] MESH_1D_CLOS not match.&quot;, __func__);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S103 / L153](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L153) | <code>            return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S104 / L155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L155) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S105 / L157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L157) | <code>            selectAlgName = &quot;CcuMSAllReducePipeLineMeshNHR&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReducePipeLineMeshNHR&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S106 / L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L159) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S107 / L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L161) | <code>    } else if (topoInfo-&gt;userRankSize &lt;= MAX_RANK_NUM_FOR_REDUCE_MS_ALGO) {</code> | 分支条件为 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO；成立进入本块，未成立继续后续分支。 |
| [S109 / L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L164) | <code>        selectAlgName = &quot;CcuMSAllReduceSoleMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceSoleMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S110 / L166](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L166) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S111 / L168](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L168) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] level0Topo[%u] is not supported mesh yet.&quot;, topoInfo-&gt;level0Topo);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S112 / L170](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L170) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S113 / L172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L172) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S115 / L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L175) | <code>    HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] Algo match [%s]&quot;, __func__, selectAlgName.c_str());</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S116 / L177](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L177) | <code>    return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S117 / L179](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L179) | <code>}</code> | 结束 SelectMeshUBXAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 19. AllReduceAutoSelector::SelectMeshAlgo

CCU_MS 下处理 Mesh1D/UBX；拒绝原地重叠、非规则双 Die 和需要 AICPU 的大数据 2P 绕路。

完整范围：[all_reduce_auto_selector.cc:L182–L274](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L182-L274)；文件：`hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc`。

功能与分支：

- [S125 / L193](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L193)：分支条件为 IsTwoLevelNetLayer(topoInfo, opParam) 且 通信域 Rank 总数 等于 2 且 本 Rank 数据字节数 至少 8MiB 双 Rank 绕路门槛；成立进入本块，未成立继续后续分支。

- [S132 / L207](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L207)：分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。

- [S133 / L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L209)：分支条件为 IsInputOutputOverlap(opParam) 等于 true；成立进入本块，未成立继续后续分支。

- [S136 / L215](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L215)：分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_REGULAR；成立进入本块，未成立继续后续分支。

- [S137 / L217](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L217)：分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。

- [S142 / L227](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L227)：分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_NOT_REGULAR；成立进入本块，未成立继续后续分支。

- [S145 / L233](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L233)：分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。

- [S148 / L239](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L239)：分支条件为 IsDevType960() 且 本 Rank 数据字节数 大于 SMALL_COUNT_16M 且 IsTwoLevelNetLayer(topoInfo, opParam；成立进入本块，未成立继续后续分支。

- [S154 / L251](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L251)：分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑；成立进入本块，未成立继续后续分支。

- [S155 / L253](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L253)：分支条件为 IsInputOutputOverlap(opParam) 等于 true；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S119 / L182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L182) | <code>SelectorStatus AllReduceAutoSelector::SelectMeshAlgo(</code> | 定义 SelectMeshAlgo 入口：CCU_MS 下处理 Mesh1D/UBX；拒绝原地重叠、非规则双 Die 和需要 AICPU 的大数据 2P 绕路。 |
| [S120 / L184](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L184) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam, std::string&amp; selectAlgName) const</code> | 续接 SelectMeshAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S121 / L186](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L186) | <code>{</code> | 进入 SelectMeshAlgo 的实现作用域；CCU_MS 下处理 Mesh1D/UBX；拒绝原地重叠、非规则双 Die 和需要 AICPU 的大数据 2P 绕路。 |
| [S122 / L188](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L188) | <code>    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];</code> | 设置 单元素字节数 为 类型到元素字节数的查找表[输入元素类型]；该值供下方当前分支使用。 |
| [S123 / L190](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L190) | <code>    u64 dataSize = opParam.DataDes.count * perDataSize;</code> | 设置 本 Rank 数据字节数 为 本 Rank 输入元素数 * 单元素字节数；该值供下方当前分支使用。 |
| [S125 / L193](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L193) | <code>    if (IsTwoLevelNetLayer(topoInfo, opParam) &amp;&amp; topoInfo-&gt;userRankSize == 2 &amp;&amp; dataSize &gt;= AR_2P_DETOUR_DATA_SIZE) {</code> | 分支条件为 IsTwoLevelNetLayer(topoInfo, opParam) 且 通信域 Rank 总数 等于 2 且 本 Rank 数据字节数 至少 8MiB 双 Rank 绕路门槛；成立进入本块，未成立继续后续分支。 |
| [S126 / L195](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L195) | <code>        HCCL_DEBUG(</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S127 / L197](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L197) | <code>            &quot;[AllReduceAutoSelector] 2P scenario with data size[%llu], &quot;</code> | 续接 SelectMeshAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S128 / L199](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L199) | <code>            &quot;fallback to AICPU for better performance.&quot;,</code> | 续接 SelectMeshAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S129 / L201](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L201) | <code>            dataSize);</code> | 为 SelectMeshAlgo 的诊断/错误宏提供实参：本 Rank 数据字节数，与前面的格式占位依次对应。 |
| [S130 / L203](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L203) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S131 / L205](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L205) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S132 / L207](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L207) | <code>    if (topoInfo-&gt;level0Topo == Level0Shape::MESH_1D) {</code> | 分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。 |
| [S133 / L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L209) | <code>        if (IsInputOutputOverlap(opParam) == true) { // 不支持 inplace 场景</code> | 分支条件为 IsInputOutputOverlap(opParam) 等于 true；成立进入本块，未成立继续后续分支。 |
| [S134 / L211](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L211) | <code>            return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S135 / L213](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L213) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S136 / L215](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L215) | <code>        if (topoInfo-&gt;level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {</code> | 分支条件为 topoInfo-&gt;level0MeshType 等于 Level0MeshType::TWO_DIE_REGULAR；成立进入本块，未成立继续后续分支。 |
| [S137 / L217](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L217) | <code>            if (IsSmallData(dataSize)) {</code> | 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。 |
| [S138 / L219](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L219) | <code>                selectAlgName = &quot;CcuMSAllReduceSoleMesh2Die&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceSoleMesh2Die&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S139 / L221](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L221) | <code>            } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S140 / L223](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L223) | <code>                selectAlgName = &quot;CcuMSAllReduceSequenceMesh2Die&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceSequenceMesh2Die&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S141 / L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L225) | <code>            }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S142 / L227](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L227) | <code>        } else if (topoInfo-&gt;level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {</code> | 分支条件为 topoInfo-&gt;level0MeshType 等于 Level0MeshType::TWO_DIE_NOT_REGULAR；成立进入本块，未成立继续后续分支。 |
| [S143 / L229](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L229) | <code>            HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] TWO_DIE_NOT_REGULAR not match&quot;, __func__);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S144 / L231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L231) | <code>            return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S145 / L233](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L233) | <code>        } else if (IsSmallData(dataSize)) {</code> | 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。 |
| [S146 / L235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L235) | <code>            selectAlgName = &quot;CcuMSAllReduceSoleMeshOneShot&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceSoleMeshOneShot&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S147 / L237](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L237) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S148 / L239](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L239) | <code>            if (IsDevType960() &amp;&amp; dataSize &gt; SMALL_COUNT_16M &amp;&amp; IsTwoLevelNetLayer(topoInfo, opParam)) {</code> | 分支条件为 IsDevType960() 且 本 Rank 数据字节数 大于 SMALL_COUNT_16M 且 IsTwoLevelNetLayer(topoInfo, opParam；成立进入本块，未成立继续后续分支。 |
| [S149 / L241](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L241) | <code>                selectAlgName = &quot;CcuMSAllReduceSoleMeshConcur&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceSoleMeshConcur&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S150 / L243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L243) | <code>            } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S151 / L245](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L245) | <code>                selectAlgName = &quot;CcuMSAllReduceSoleMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuMSAllReduceSoleMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S152 / L247](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L247) | <code>            }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S153 / L249](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L249) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S154 / L251](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L251) | <code>    } else if (topoInfo-&gt;level0Topo == Level0Shape::MESH_1D_CLOS) {</code> | 分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑；成立进入本块，未成立继续后续分支。 |
| [S155 / L253](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L253) | <code>        if (IsInputOutputOverlap(opParam) == true) {</code> | 分支条件为 IsInputOutputOverlap(opParam) 等于 true；成立进入本块，未成立继续后续分支。 |
| [S157 / L256](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L256) | <code>            return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S158 / L258](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L258) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S159 / L260](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L260) | <code>        return SelectMeshUBXAlgo(topoInfo, selectAlgName, dataSize);</code> | 直接返回 调用 SelectMeshUBXAlgo 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S160 / L262](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L262) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S161 / L264](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L264) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] level0Topo[%u] is not supported yet.&quot;, topoInfo-&gt;level0Topo);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S162 / L266](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L266) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S163 / L268](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L268) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S164 / L270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L270) | <code>    HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] Algo match [%s]&quot;, __func__, selectAlgName.c_str());</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S165 / L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L272) | <code>    return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S166 / L274](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L274) | <code>}</code> | 结束 SelectMeshAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 20. AllReduceAutoSelector::SelectCcuScheduleAlgo

旧 CCU 调度算法能力检查：拒绝 UB_RTP、三层、strict、PROD、64 位类型，再处理多层或单层拓扑。

完整范围：[all_reduce_auto_selector.cc:L277–L481](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L277-L481)；文件：`hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc`。

功能与分支：

- [S176 / L292](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L292)：分支条件为 topoInfo->level2UbRtp；成立进入本块，未成立继续后续分支。

- [S182 / L303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L303)：分支条件为 topoInfo->topoLevelNums 至少 TOPO_LEVEL_NUM_3；成立进入本块，未成立继续后续分支。

- [S191 / L319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L319)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S197 / L329](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L329)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S203 / L340](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L340)：分支条件为 Is64BitDataType(输入元素类型；成立进入本块，未成立继续后续分支。

- [S210 / L353](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L353)：分支条件为 topoInfo->topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。

- [S211 / L355](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L355)：分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。

- [S212 / L357](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L357)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S217 / L366](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L366)：分支条件为 topoInfo->Level1Nhr；成立进入本块，未成立继续后续分支。

- [S221 / L374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L374)：分支条件为 topoInfo->当前网络层编号Details.localNetInsSizeOfLayer[0] 等于 1；成立进入本块，未成立继续后续分支。

- [S223 / L378](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L378)：分支条件为 topoInfo->is2DieFullMesh；成立进入本块，未成立继续后续分支。

- [S226 / L384](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L384)：分支条件为 本 Rank 数据字节数 不超过 AR_MORE_64P_SEQ_MAX_DATA_SIZE 且 通信域 Rank 总数 大于 ccuSize；成立进入本块，未成立继续后续分支。

- [S228 / L388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L388)：分支条件为 ；成立进入本块，未成立继续后续分支。

- [S233 / L398](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L398)：分支条件为 ；成立进入本块，未成立继续后续分支。

- [S238 / L408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L408)：分支条件为 ；成立进入本块，未成立继续后续分支。

- [S243 / L418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L418)：分支条件为 IsSmallDataCCU(本 Rank 数据字节数, 通信域 Rank 总数；成立进入本块，未成立继续后续分支。

- [S245 / L421](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L421)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S257 / L445](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L445)：分支条件为 第零层拓扑形状 等于 CLOS 拓扑 且 (!IsInputOutputOverlap(opParam；成立进入本块，未成立继续后续分支。

- [S258 / L447](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L447)：分支条件为 本 Rank 数据字节数 小于 AR_CCU_CLOS_1D_SMALL_DATA_SIZE；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S168 / L277](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L277) | <code>SelectorStatus AllReduceAutoSelector::SelectCcuScheduleAlgo(</code> | 定义 SelectCcuScheduleAlgo 入口：旧 CCU 调度算法能力检查：拒绝 UB_RTP、三层、strict、PROD、64 位类型，再处理多层或单层拓扑。 |
| [S169 / L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L279) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,</code> | 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam,；引用参数按声明的 const 限制读写。 |
| [S170 / L281](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L281) | <code>    const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; selectAlgName) const</code> | 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const std::map&lt;HcclCMDType, std::vector&lt;HcclAlgoType&gt;&gt;&amp; configAlgMap, std::string&amp; 算法名输出参数) const；引用参数按声明的 const 限制读写。 |
| [S171 / L283](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L283) | <code>{</code> | 进入 SelectCcuScheduleAlgo 的实现作用域；旧 CCU 调度算法能力检查：拒绝 UB_RTP、三层、strict、PROD、64 位类型，再处理多层或单层拓扑。 |
| [S172 / L285](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L285) | <code>    (void)configAlgMap;</code> | 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。 |
| [S173 / L287](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L287) | <code>    u32 ccuSize = AR_CCU_MAX_RANK_SIZE;</code> | 设置 ccuSize 为 AR_CCU_MAX_RANK_SIZE；该值供下方当前分支使用。 |
| [S174 / L289](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L289) | <code>    HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]&quot;, __func__, topoInfo-&gt;topoLevelNums);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S176 / L292](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L292) | <code>    if (topoInfo-&gt;level2UbRtp) {</code> | 分支条件为 topoInfo-&gt;level2UbRtp；成立进入本块，未成立继续后续分支。 |
| [S177 / L294](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L294) | <code>        HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S178 / L296](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L296) | <code>            &quot;[AllReduceAutoSelector][%s] ccu schedule is not supported with level2UbRtp, reset to default.&quot;, __func__);</code> | 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S179 / L298](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L298) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S180 / L300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L300) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S182 / L303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L303) | <code>    if (topoInfo-&gt;topoLevelNums &gt;= TOPO_LEVEL_NUM_3) {</code> | 分支条件为 topoInfo-&gt;topoLevelNums 至少 TOPO_LEVEL_NUM_3；成立进入本块，未成立继续后续分支。 |
| [S183 / L305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L305) | <code>        HCCL_INFO(</code> | 开始 HCCL_INFO 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S184 / L307](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L307) | <code>            &quot;[AllReduceAutoSelector][%s] ccu schedule is not supported when topoLevelNums &gt;= 3(levelNum[%u]), reset to &quot;</code> | 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S185 / L309](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L309) | <code>            &quot;default.&quot;,</code> | 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S186 / L311](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L311) | <code>            __func__, topoInfo-&gt;topoLevelNums);</code> | 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：__func__, topoInfo-&gt;topoLevelNums，与前面的格式占位依次对应。 |
| [S187 / L313](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L313) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S188 / L315](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L315) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S191 / L319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L319) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S192 / L321](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L321) | <code>        IsNeedStrictModeForOrderPreserved(opParam, topoInfo-&gt;userRankSize),</code> | 调用 IsNeedStrictModeForOrderPreserved 完成当前参数所指的子步骤；本行实参为 IsNeedStrictModeForOrderPreserved(opParam, 通信域 Rank 总数),。 |
| [S193 / L323](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L323) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] DETERMINISTIC_STRICT mode not supported for CCU_SCHED, fallback to AICPU.&quot;),</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S194 / L325](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L325) | <code>        SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S197 / L329](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L329) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S198 / L331](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L331) | <code>        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,</code> | 续接本次错误检查/子调用实参：归约运算 等于 HcclReduceOp::乘积归约；返回行为由所在完整宏决定。 |
| [S199 / L333](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L333) | <code>        HCCL_DEBUG(</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S200 / L335](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L335) | <code>            &quot;[AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.&quot;, opParam.reduceType),</code> | 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S201 / L337](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L337) | <code>        SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S203 / L340](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L340) | <code>    if (Is64BitDataType(opParam.DataDes.dataType)) {</code> | 分支条件为 Is64BitDataType(输入元素类型；成立进入本块，未成立继续后续分支。 |
| [S204 / L342](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L342) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S205 / L344](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L344) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S206 / L346](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L346) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S207 / L348](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L348) | <code>    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];</code> | 设置 单元素字节数 为 类型到元素字节数的查找表[输入元素类型]；该值供下方当前分支使用。 |
| [S208 / L350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L350) | <code>    u64 dataSize = opParam.DataDes.count * perDataSize;</code> | 设置 本 Rank 数据字节数 为 本 Rank 输入元素数 * 单元素字节数；该值供下方当前分支使用。 |
| [S210 / L353](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L353) | <code>    if (topoInfo-&gt;topoLevelNums &gt; 1) {</code> | 分支条件为 topoInfo-&gt;topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。 |
| [S211 / L355](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L355) | <code>        if (topoInfo-&gt;level0Topo == Level0Shape::MESH_1D) {</code> | 分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。 |
| [S212 / L357](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L357) | <code>            CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S213 / L359](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L359) | <code>                IsInputOutputOverlap(opParam) == true,</code> | 调用 IsInputOutputOverlap 完成当前参数所指的子步骤；本行实参为 IsInputOutputOverlap(opParam) 等于 true,。 |
| [S214 / L361](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L361) | <code>                HCCL_WARNING(&quot;[Algo][AllReduceAutoSelector] ccu_sched does not support inplace allreduce.&quot;),</code> | 开始 HCCL_WARNING 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S215 / L363](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L363) | <code>                SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S217 / L366](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L366) | <code>            if (topoInfo-&gt;Level1Nhr) {</code> | 分支条件为 topoInfo-&gt;Level1Nhr；成立进入本块，未成立继续后续分支。 |
| [S218 / L368](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L368) | <code>                selectAlgName = &quot;CcuSchedAllReduceSoleNHR&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleNHR&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S219 / L370](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L370) | <code>                HCCL_INFO(&quot;[AllReduceAutoSelector] Level1Nhr=true, select [%s]&quot;, selectAlgName.c_str());</code> | 开始 HCCL_INFO 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S220 / L372](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L372) | <code>                return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S221 / L374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L374) | <code>            } else if (topoInfo-&gt;netLayerDetails.localNetInsSizeOfLayer[0] == 1) {</code> | 分支条件为 topoInfo-&gt;当前网络层编号Details.localNetInsSizeOfLayer[0] 等于 1；成立进入本块，未成立继续后续分支。 |
| [S222 / L376](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L376) | <code>                selectAlgName = &quot;CcuSchedAllReduceSoleNHR&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleNHR&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S223 / L378](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L378) | <code>            } else if (topoInfo-&gt;is2DieFullMesh) {</code> | 分支条件为 topoInfo-&gt;is2DieFullMesh；成立进入本块，未成立继续后续分支。 |
| [S224 / L380](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L380) | <code>                HCCL_DEBUG(&quot;[AllReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S225 / L382](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L382) | <code>                return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S226 / L384](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L384) | <code>            } else if (dataSize &lt;= AR_MORE_64P_SEQ_MAX_DATA_SIZE &amp;&amp; topoInfo-&gt;userRankSize &gt; ccuSize) {</code> | 分支条件为 本 Rank 数据字节数 不超过 AR_MORE_64P_SEQ_MAX_DATA_SIZE 且 通信域 Rank 总数 大于 ccuSize；成立进入本块，未成立继续后续分支。 |
| [S227 / L386](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L386) | <code>                selectAlgName = &quot;CcuSchedAllReduceSequenceMeshMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSequenceMeshMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S228 / L388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L388) | <code>            } else if (</code> | 分支条件为 ；成立进入本块，未成立继续后续分支。 |
| [S229 / L390](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L390) | <code>                dataSize &lt;= RS_MAX_DATA_SIZE &amp;&amp; topoInfo-&gt;userRankSize &gt;= ccuSize</code> | 续接 SelectCcuScheduleAlgo 当前语句的具体实参/字段：本 Rank 数据字节数 不超过 RS_MAX_DATA_SIZE 且 通信域 Rank 总数 至少 ccuSize；由其完整表达式完成参数组装、检查或结果写回。 |
| [S230 / L392](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L392) | <code>                &amp;&amp; !Is8BitDataType(opParam.DataDes.dataType)) {</code> | 调用 Is8BitDataType 完成当前参数所指的子步骤；本行实参为 且 !Is8BitDataType(输入元素类型)) {。 |
| [S231 / L394](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L394) | <code>                selectAlgName = &quot;CcuSchedAllReduceSequenceMeshMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSequenceMeshMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S232 / L396](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L396) | <code>                return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S233 / L398](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L398) | <code>            } else if (</code> | 分支条件为 ；成立进入本块，未成立继续后续分支。 |
| [S234 / L400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L400) | <code>                dataSize &lt;= AR_FLATTEN_MAX_DATA_SIZE &amp;&amp; topoInfo-&gt;userRankSize &lt;= ccuSize</code> | 续接 SelectCcuScheduleAlgo 当前语句的具体实参/字段：本 Rank 数据字节数 不超过 AR_FLATTEN_MAX_DATA_SIZE 且 通信域 Rank 总数 不超过 ccuSize；由其完整表达式完成参数组装、检查或结果写回。 |
| [S235 / L402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L402) | <code>                &amp;&amp; (!IsInputOutputOverlap(opParam)) &amp;&amp; !Is8BitDataType(opParam.DataDes.dataType)) {</code> | 调用 IsInputOutputOverlap 完成当前参数所指的子步骤；本行实参为 且 (!IsInputOutputOverlap(opParam)) 且 !Is8BitDataType(输入元素类型)) {。 |
| [S236 / L404](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L404) | <code>                selectAlgName = &quot;CcuSchedAllReduceSoleMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S237 / L406](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L406) | <code>                return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S238 / L408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L408) | <code>            } else if (</code> | 分支条件为 ；成立进入本块，未成立继续后续分支。 |
| [S239 / L410](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L410) | <code>                dataSize &lt;= AR_CCU_SEQ_MAX_DATA_SIZE &amp;&amp; topoInfo-&gt;userRankSize &lt; ccuSize</code> | 续接 SelectCcuScheduleAlgo 当前语句的具体实参/字段：本 Rank 数据字节数 不超过 AR_CCU_SEQ_MAX_DATA_SIZE 且 通信域 Rank 总数 小于 ccuSize；由其完整表达式完成参数组装、检查或结果写回。 |
| [S240 / L412](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L412) | <code>                &amp;&amp; !Is8BitDataType(opParam.DataDes.dataType)) {</code> | 调用 Is8BitDataType 完成当前参数所指的子步骤；本行实参为 且 !Is8BitDataType(输入元素类型)) {。 |
| [S241 / L414](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L414) | <code>                selectAlgName = &quot;CcuSchedAllReduceSequenceMeshMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSequenceMeshMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S242 / L416](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L416) | <code>                return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S243 / L418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L418) | <code>            } else if (IsSmallDataCCU(dataSize, topoInfo-&gt;userRankSize)) { // 64M以下跑ccu</code> | 分支条件为 IsSmallDataCCU(本 Rank 数据字节数, 通信域 Rank 总数；成立进入本块，未成立继续后续分支。 |
| [S245 / L421](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L421) | <code>                CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S246 / L423](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L423) | <code>                    opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,</code> | 续接本次错误检查/子调用实参：输入元素类型 等于 HcclDataType::INT8 类型；返回行为由所在完整宏决定。 |
| [S247 / L425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L425) | <code>                    HCCL_DEBUG(</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S248 / L427](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L427) | <code>                        &quot;[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode with ms &quot;</code> | 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S249 / L429](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L429) | <code>                        &quot;reduce. levelNum[%u]&quot;,</code> | 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S250 / L431](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L431) | <code>                        opParam.DataDes.dataType, topoInfo-&gt;topoLevelNums),</code> | 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：输入元素类型, topoInfo-&gt;topoLevelNums，与前面的格式占位依次对应。 |
| [S251 / L433](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L433) | <code>                    SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S252 / L435](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L435) | <code>                selectAlgName = &quot;CcuSchedAllReduceParallelMeshNHR&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceParallelMeshNHR&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S253 / L437](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L437) | <code>                return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S254 / L439](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L439) | <code>            } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S255 / L441](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L441) | <code>                return SelectorStatus::NOT_MATCH; // 64M以上切为aicpu</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S256 / L443](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L443) | <code>            }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S257 / L445](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L445) | <code>        } else if (topoInfo-&gt;level0Topo == Level0Shape::CLOS &amp;&amp; (!IsInputOutputOverlap(opParam))) {</code> | 分支条件为 第零层拓扑形状 等于 CLOS 拓扑 且 (!IsInputOutputOverlap(opParam；成立进入本块，未成立继续后续分支。 |
| [S258 / L447](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L447) | <code>            if (dataSize &lt; AR_CCU_CLOS_1D_SMALL_DATA_SIZE) {</code> | 分支条件为 本 Rank 数据字节数 小于 AR_CCU_CLOS_1D_SMALL_DATA_SIZE；成立进入本块，未成立继续后续分支。 |
| [S259 / L449](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L449) | <code>                selectAlgName = &quot;CcuSchedAllReduceSoleNHR&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleNHR&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S260 / L451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L451) | <code>                return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S261 / L453](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L453) | <code>            } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S262 / L455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L455) | <code>                return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S263 / L457](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L457) | <code>            }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S264 / L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L459) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S265 / L461](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L461) | <code>            HCCL_DEBUG(</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S266 / L463](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L463) | <code>                &quot;[AllReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.&quot;,</code> | 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S267 / L465](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L465) | <code>                topoInfo-&gt;level0Topo);</code> | 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：第零层拓扑形状，与前面的格式占位依次对应。 |
| [S268 / L467](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L467) | <code>            return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S269 / L469](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L469) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S270 / L471](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L471) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S271 / L473](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L473) | <code>        return SelectCcuScheduleLevel0Algo(topoInfo, opParam, selectAlgName, dataSize);</code> | 直接返回 调用 SelectCcuScheduleLevel0Algo 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。 |
| [S272 / L475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L475) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S273 / L477](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L477) | <code>    HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] Algo match [%s]&quot;, __func__, selectAlgName.c_str());</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S274 / L479](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L479) | <code>    return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S275 / L481](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L481) | <code>}</code> | 结束 SelectCcuScheduleAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 21. AllReduceAutoSelector::SelectCcuScheduleLevel0UBXAlgo

单层 UBX 中按 Mesh/CLOS 关系与大小选择并发、多 Jetty、流水线或 NHR。

完整范围：[all_reduce_auto_selector.cc:L484–L547](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L484-L547)；文件：`hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc`。

功能与分支：

- [S283 / L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L495)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S286 / L501](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L501)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S289 / L507](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L507)：分支条件为 isMeshNumEqualToClosNum 且 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_CONCURRENT_ALGO；成立进入本块，未成立继续后续分支。

- [S291 / L510](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L510)：分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。

- [S298 / L522](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L522)：分支条件为 isClosNumMultipleOfMeshNum 且 !IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。

- [S300 / L525](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L525)：分支条件为 本 Rank 数据字节数 小于 OMNI_UBX_AR_SCHED_DATA_SIZE；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S277 / L484](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L484) | <code>SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0UBXAlgo(</code> | 定义 SelectCcuScheduleLevel0UBXAlgo 入口：单层 UBX 中按 Mesh/CLOS 关系与大小选择并发、多 Jetty、流水线或 NHR。 |
| [S278 / L486](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L486) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, std::string&amp; selectAlgName, const u64 dataSize) const</code> | 续接 SelectCcuScheduleLevel0UBXAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, std::string&amp; 算法名输出参数, const u64 本 Rank 数据字节数) const；引用参数按声明的 const 限制读写。 |
| [S279 / L488](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L488) | <code>{</code> | 进入 SelectCcuScheduleLevel0UBXAlgo 的实现作用域；单层 UBX 中按 Mesh/CLOS 关系与大小选择并发、多 Jetty、流水线或 NHR。 |
| [S281 / L491](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L491) | <code>    bool isMeshNumEqualToClosNum = false;</code> | 设置 isMeshNumEqualToClosNum 为 false；该值供下方当前分支使用。 |
| [S282 / L493](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L493) | <code>    bool isClosNumMultipleOfMeshNum = false;</code> | 设置 isClosNumMultipleOfMeshNum 为 false；该值供下方当前分支使用。 |
| [S283 / L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L495) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S284 / L497](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L497) | <code>        CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,</code> | 调用 CheckMeshNumEqualToClosNum 完成当前参数所指的子步骤；本行实参为 CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) 不等于 成功状态,。 |
| [S285 / L499](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L499) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed.&quot;), SelectorStatus::NOT_MATCH);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0UBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S286 / L501](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L501) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S287 / L503](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L503) | <code>        CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,</code> | 调用 CheckClosNumMultipleOfMeshNum 完成当前参数所指的子步骤；本行实参为 CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) 不等于 成功状态,。 |
| [S288 / L505](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L505) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed.&quot;), SelectorStatus::NOT_MATCH);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0UBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S289 / L507](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L507) | <code>    if (isMeshNumEqualToClosNum &amp;&amp; topoInfo-&gt;userRankSize &lt;= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {</code> | 分支条件为 isMeshNumEqualToClosNum 且 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_CONCURRENT_ALGO；成立进入本块，未成立继续后续分支。 |
| [S291 / L510](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L510) | <code>        if (IsSmallData(dataSize)) {</code> | 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。 |
| [S293 / L513](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L513) | <code>            selectAlgName = &quot;CcuSchedAllReduceSoleMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S294 / L515](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L515) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S296 / L518](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L518) | <code>            selectAlgName = &quot;CcuSchedAllReduceConcurMeshNHRMultiLink&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceConcurMeshNHRMultiLink&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S297 / L520](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L520) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S298 / L522](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L522) | <code>    } else if (isClosNumMultipleOfMeshNum &amp;&amp; !IsSmallData(dataSize)) {</code> | 分支条件为 isClosNumMultipleOfMeshNum 且 !IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。 |
| [S300 / L525](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L525) | <code>        if (dataSize &lt; OMNI_UBX_AR_SCHED_DATA_SIZE) {</code> | 分支条件为 本 Rank 数据字节数 小于 OMNI_UBX_AR_SCHED_DATA_SIZE；成立进入本块，未成立继续后续分支。 |
| [S301 / L527](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L527) | <code>            selectAlgName = &quot;CcuSchedAllReduceParallelMeshNHRMultiJetty&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceParallelMeshNHRMultiJetty&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S302 / L529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L529) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S303 / L531](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L531) | <code>            selectAlgName = &quot;CcuSchedAllReducePipeLineMeshNHR&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReducePipeLineMeshNHR&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S304 / L533](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L533) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S305 / L535](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L535) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S307 / L538](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L538) | <code>        selectAlgName = &quot;CcuSchedAllReduceSoleNHRMultiLink&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleNHRMultiLink&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S308 / L540](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L540) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S310 / L543](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L543) | <code>    HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] Algo match [%s]&quot;, __func__, selectAlgName.c_str());</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0UBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S311 / L545](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L545) | <code>    return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S312 / L547](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L547) | <code>}</code> | 结束 SelectCcuScheduleLevel0UBXAlgo 实现；其返回状态或已写回字段由调用者接收。 |


## 22. AllReduceAutoSelector::SelectCcuScheduleLevel0AlgoMesh1D

单层 CCU Mesh 按 INT8 禁限、Rank 缩放数据阈值及双 Die 规则性选择算法。

完整范围：[all_reduce_auto_selector.cc:L550–L621](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L550-L621)；文件：`hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc`。

功能与分支：

- [S319 / L559](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L559)：开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。

- [S327 / L575](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L575)：分支条件为 通信域 Rank 总数 等于 0；成立进入本块，未成立继续后续分支。

- [S333 / L587](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L587)：分支条件为 本 Rank 数据字节数 * Rank 数平方缩放因子 大于 AR_M2M_1D_MAX_DATA_SIZE；成立进入本块，未成立继续后续分支。

- [S336 / L593](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L593)：分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_REGULAR；成立进入本块，未成立继续后续分支。

- [S337 / L595](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L595)：分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。

- [S342 / L605](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L605)：分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_NOT_REGULAR；成立进入本块，未成立继续后续分支。



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S314 / L550](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L550) | <code>SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0AlgoMesh1D(</code> | 定义 SelectCcuScheduleLevel0AlgoMesh1D 入口：单层 CCU Mesh 按 INT8 禁限、Rank 缩放数据阈值及双 Die 规则性选择算法。 |
| [S315 / L552](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L552) | <code>    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam, std::string&amp; selectAlgName,</code> | 续接 SelectCcuScheduleLevel0AlgoMesh1D 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam&amp; opParam, std::string&amp; 算法名输出参数,；引用参数按声明的 const 限制读写。 |
| [S316 / L554](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L554) | <code>    const u64 dataSize) const</code> | 续接 SelectCcuScheduleLevel0AlgoMesh1D 的入口参数/基类初始化：const u64 本 Rank 数据字节数) const；引用参数按声明的 const 限制读写。 |
| [S317 / L556](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L556) | <code>{</code> | 进入 SelectCcuScheduleLevel0AlgoMesh1D 的实现作用域；单层 CCU Mesh 按 INT8 禁限、Rank 缩放数据阈值及双 Die 规则性选择算法。 |
| [S319 / L559](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L559) | <code>    CHK_PRT_RET(</code> | 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。 |
| [S320 / L561](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L561) | <code>        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,</code> | 续接本次错误检查/子调用实参：输入元素类型 等于 HcclDataType::INT8 类型；返回行为由所在完整宏决定。 |
| [S321 / L563](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L563) | <code>        HCCL_DEBUG(</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S322 / L565](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L565) | <code>            &quot;[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode &quot;</code> | 续接 SelectCcuScheduleLevel0AlgoMesh1D 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S323 / L567](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L567) | <code>            &quot;with ms reduce.&quot;,</code> | 续接 SelectCcuScheduleLevel0AlgoMesh1D 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。 |
| [S324 / L569](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L569) | <code>            opParam.DataDes.dataType),</code> | 为 SelectCcuScheduleLevel0AlgoMesh1D 的诊断/错误宏提供实参：输入元素类型，与前面的格式占位依次对应。 |
| [S325 / L571](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L571) | <code>        SelectorStatus::NOT_MATCH);</code> | 为 SelectCcuScheduleLevel0AlgoMesh1D 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。 |
| [S326 / L573](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L573) | <code>    double ratio;</code> | 声明本阶段局部变量 double Rank 数平方缩放因子，实际值由后续查询/计算填写。 |
| [S327 / L575](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L575) | <code>    if (topoInfo-&gt;userRankSize == 0) {</code> | 分支条件为 通信域 Rank 总数 等于 0；成立进入本块，未成立继续后续分支。 |
| [S328 / L577](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L577) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector] the selector userRankSize not set&quot;);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S329 / L579](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L579) | <code>        ratio = 1;</code> | 设置 Rank 数平方缩放因子 为 1；该值供下方当前分支使用。 |
| [S330 / L581](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L581) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S331 / L583](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L583) | <code>        ratio = DEFAULT_RANK_SIZE / topoInfo-&gt;userRankSize / topoInfo-&gt;userRankSize;</code> | 设置 Rank 数平方缩放因子 为 默认参考 Rank 数 / 通信域 Rank 总数 / 通信域 Rank 总数；该值供下方当前分支使用。 |
| [S332 / L585](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L585) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S333 / L587](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L587) | <code>    if (dataSize * ratio &gt; AR_M2M_1D_MAX_DATA_SIZE) {</code> | 分支条件为 本 Rank 数据字节数 * Rank 数平方缩放因子 大于 AR_M2M_1D_MAX_DATA_SIZE；成立进入本块，未成立继续后续分支。 |
| [S334 / L589](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L589) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S335 / L591](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L591) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S336 / L593](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L593) | <code>    if (topoInfo-&gt;level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {</code> | 分支条件为 topoInfo-&gt;level0MeshType 等于 Level0MeshType::TWO_DIE_REGULAR；成立进入本块，未成立继续后续分支。 |
| [S337 / L595](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L595) | <code>        if (IsSmallData(dataSize)) {</code> | 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。 |
| [S338 / L597](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L597) | <code>            selectAlgName = &quot;CcuSchedAllReduceSoleMesh2Die&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleMesh2Die&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S339 / L599](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L599) | <code>        } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S340 / L601](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L601) | <code>            selectAlgName = &quot;CcuSchedAllReduceSequenceMesh2Die&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSequenceMesh2Die&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S341 / L603](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L603) | <code>        }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S342 / L605](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L605) | <code>    } else if (topoInfo-&gt;level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {</code> | 分支条件为 topoInfo-&gt;level0MeshType 等于 Level0MeshType::TWO_DIE_NOT_REGULAR；成立进入本块，未成立继续后续分支。 |
| [S343 / L607](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L607) | <code>        HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] TWO_DIE_NOT_REGULAR not match&quot;, __func__);</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S344 / L609](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L609) | <code>        return SelectorStatus::NOT_MATCH;</code> | 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。 |
| [S345 / L611](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L611) | <code>    } else {</code> | 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。 |
| [S346 / L613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L613) | <code>        selectAlgName = &quot;CcuSchedAllReduceSoleMesh&quot;;</code> | 写回候选算法名：算法名输出参数 = &quot;CcuSchedAllReduceSoleMesh&quot;；此处只选择注册名，执行资源在后续流程申请。 |
| [S347 / L615](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L615) | <code>    }</code> | 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。 |
| [S348 / L617](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L617) | <code>    HCCL_DEBUG(&quot;[AllReduceAutoSelector][%s] Algo match [%s]&quot;, __func__, selectAlgName.c_str());</code> | 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。 |
| [S349 / L619](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L619) | <code>    return SelectorStatus::MATCH;</code> | 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。 |
| [S350 / L621](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L621) | <code>}</code> | 结束 SelectCcuScheduleLevel0AlgoMesh1D 实现；其返回状态或已写回字段由调用者接收。 |
