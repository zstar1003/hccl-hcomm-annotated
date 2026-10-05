# AllReduce逐行对照：公共调度、资源上下文与设备入口

[返回阅读指南](../READING_GUIDE.zh-CN.md)。S为审读快照行号；L为带本次逐行注释的源码行号。每个L链接定位到固定源码提交；长语句按物理行分别说明。空行及原注释不重复注释。

审读快照：`f8af6a36831195a8440de6ec6183856bb72af907`；源码提交：`824a8a80731bd66ef6eb78891b9aec0acfca281c`。

[返回本阶段函数导航](02-dispatch-device.zh-CN.md)。第3/3页。

## 51. SaveUnfoldThreadInfo

按通信域名字保存Host展开Thread

完整范围：[op_common.cc:L2870–L2900](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2870-L2900)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S2004 / L2870](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2870) | <code>HcclResult SaveUnfoldThreadInfo(HcclComm comm, const OpParam&amp; param, ThreadHandle unfoldThread)</code> | 声明SaveUnfoldThreadInfo接口：按通信域名字保存Host展开Thread。 |
| [S2005 / L2872](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2872) | <code>{</code> | 开始SaveUnfoldThreadInfo的函数体。 |
| [S2006 / L2874](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2874) | <code>    uint64_t size = sizeof(ThreadHandle);</code> | 展开线程缓存只存放一个ThreadHandle。 |
| [S2007 / L2876](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2876) | <code>    void* ctx = nullptr;</code> | 初始化展开线程Host缓存地址。 |
| [S2009 / L2879](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2879) | <code>    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};</code> | 创建保存通信域展开线程缓存tag的固定长度字符数组。 |
| [S2010 / L2881](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2881) | <code>    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, &quot;%s_unfold&quot;, param.commName);</code> | 用通信域名加_unfold构造展开线程缓存tag。 |
| [S2011 / L2883](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2883) | <code>    CHK_PRT_RET(ret &lt;= 0, HCCL_ERROR(&quot;[%s] failed to fill unfoldAlgTag&quot;, __func__), HCCL_E_INTERNAL);</code> | 展开tag构造失败返回内部错误。 |
| [S2012 / L2885](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2885) | <code>    CHK_RET(HcclEngineCtxCreate(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, size, &amp;ctx));</code> | 按通信域展开tag创建CPU_TS Host上下文。 |
| [S2014 / L2888](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2888) | <code>    ThreadHandle* threadPtr = reinterpret_cast&lt;ThreadHandle*&gt;(ctx);</code> | 将Host上下文首地址解释为ThreadHandle存储位置。 |
| [S2015 / L2890](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2890) | <code>    *threadPtr = unfoldThread;</code> | 保存Host展开线程句柄。 |
| [S2016 / L2892](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2892) | <code>    HCCL_INFO(</code> | 输出运行日志，记录SaveUnfoldThreadInfo当前阶段和相关参数。 |
| [S2017 / L2894](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2894) | <code>        &quot;[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]&quot;, unfoldAlgTag, threadPtr,</code> | 补充日志格式：[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]&quot;, unfoldAlgTag, threadPtr。 |
| [S2018 / L2896](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2896) | <code>        unfoldThread);</code> | 提供上述日志的实参，涉及Host展开线程句柄。 |
| [S2019 / L2898](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2898) | <code>    return HCCL_SUCCESS;</code> | 按通信域名字保存Host展开Thread处理完成，返回成功。 |
| [S2020 / L2900](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2900) | <code>}</code> | 结束SaveUnfoldThreadInfo函数体。 |


## 52. GetUnfoldThreadInfo

读取通信域的Host展开Thread

完整范围：[op_common.cc:L2903–L2938](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2903-L2938)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S2022 / L2903](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2903) | <code>HcclResult GetUnfoldThreadInfo(HcclComm comm, const OpParam&amp; param, ThreadHandle&amp; unfoldThread)</code> | 声明GetUnfoldThreadInfo接口：读取通信域的Host展开Thread。 |
| [S2023 / L2905](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2905) | <code>{</code> | 开始GetUnfoldThreadInfo的函数体。 |
| [S2024 / L2907](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2907) | <code>    uint64_t size = sizeof(ThreadHandle);</code> | 设置展开线程缓存查询预期长度为sizeof(ThreadHandle)。 |
| [S2025 / L2909](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2909) | <code>    void* ctx = nullptr;</code> | 初始化Host上下文地址输出。 |
| [S2026 / L2911](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2911) | <code>    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};</code> | 创建展开线程缓存tag数组，初始化为空。 |
| [S2027 / L2913](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2913) | <code>    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, &quot;%s_unfold&quot;, param.commName);</code> | 按通信域名字构造_unfold tag。 |
| [S2028 / L2915](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2915) | <code>    CHK_PRT_RET(ret &lt;= 0, HCCL_ERROR(&quot;[%s] failed to fill unfoldAlgTag&quot;, __func__), HCCL_E_INTERNAL);</code> | tag字符串格式化失败返回内部错误。 |
| [S2029 / L2917](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2917) | <code>    HcclResult getRet = HcclEngineCtxGet(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, &amp;ctx, &amp;size);</code> | 查询CPU_TS Host展开线程上下文。 |
| [S2030 / L2919](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2919) | <code>    if (getRet != HCCL_SUCCESS) {</code> | 查询失败时直接返回，调用方可决定是否新建展开线程。 |
| [S2031 / L2921](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2921) | <code>        return getRet;</code> | 返回展开线程上下文查询错误。 |
| [S2032 / L2923](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2923) | <code>    }</code> | 结束条件if (getRet != HCCL_SUCCESS)。 |
| [S2034 / L2926](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2926) | <code>    ThreadHandle* threadPtr = reinterpret_cast&lt;ThreadHandle*&gt;(ctx);</code> | 把Host上下文解释为ThreadHandle存储位置。 |
| [S2035 / L2928](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2928) | <code>    unfoldThread = *threadPtr;</code> | 读取Host展开线程句柄到输出参数。 |
| [S2036 / L2930](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2930) | <code>    HCCL_INFO(</code> | 输出运行日志，记录GetUnfoldThreadInfo当前阶段和相关参数。 |
| [S2037 / L2932](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2932) | <code>        &quot;[GetUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]&quot;, unfoldAlgTag, threadPtr,</code> | 补充日志格式：[GetUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]&quot;, unfoldAlgTag, threadPtr。 |
| [S2038 / L2934](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2934) | <code>        unfoldThread);</code> | 提供上述日志的实参，涉及Host展开线程句柄。 |
| [S2039 / L2936](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2936) | <code>    return HCCL_SUCCESS;</code> | 读取通信域的Host展开Thread处理完成，返回成功。 |
| [S2040 / L2938](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2938) | <code>}</code> | 结束GetUnfoldThreadInfo函数体。 |


## 53. GetMainThreadInfo

读取算法主Thread及通知容量

完整范围：[op_common.cc:L2941–L2974](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2941-L2974)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S2042 / L2941](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2941) | <code>HcclResult GetMainThreadInfo(HcclComm comm, const OpParam&amp; param, ThreadHandle&amp; thread, u32&amp; notifyNum)</code> | 声明GetMainThreadInfo接口：读取算法主Thread及通知容量。 |
| [S2043 / L2943](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2943) | <code>{</code> | 开始GetMainThreadInfo的函数体。 |
| [S2044 / L2945](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2945) | <code>    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);</code> | 主线程信息的预期长度为ThreadHandle加u32通知数。 |
| [S2045 / L2947](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2947) | <code>    void* ctx = nullptr;</code> | 初始化主线程信息Host上下文查询地址。 |
| [S2046 / L2949](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2949) | <code>    CHK_RET(HcclEngineCtxGet(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, &amp;ctx, &amp;size));</code> | 按本算法algTag查询CPU_TS Host主线程信息上下文。 |
| [S2049 / L2953](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2953) | <code>    ThreadHandle* threadPtr = reinterpret_cast&lt;ThreadHandle*&gt;(ctx);</code> | 将缓存首地址解释为ThreadHandle存储位置。 |
| [S2050 / L2955](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2955) | <code>    thread = *threadPtr;</code> | 读取算法Device主线程句柄。 |
| [S2052 / L2958](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2958) | <code>    char* curPtr = reinterpret_cast&lt;char*&gt;(ctx);</code> | 将上下文地址转换为字节指针。 |
| [S2053 / L2960](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2960) | <code>    curPtr += sizeof(ThreadHandle);</code> | 跳过ThreadHandle定位通知数存储段。 |
| [S2054 / L2962](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2962) | <code>    u32* notifyNumPtr = reinterpret_cast&lt;u32*&gt;(curPtr);</code> | 将通知数所在字节解释为u32指针。 |
| [S2055 / L2964](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2964) | <code>    notifyNum = *notifyNumPtr;</code> | 读取主线程总通知槽数到输出参数。 |
| [S2056 / L2966](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2966) | <code>    HCCL_INFO(</code> | 输出运行日志，记录GetMainThreadInfo当前阶段和相关参数。 |
| [S2057 / L2968](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2968) | <code>        &quot;[GetMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]&quot;, threadPtr, thread,</code> | 补充日志格式：[GetMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]&quot;, threadPtr, thread。 |
| [S2058 / L2970](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2970) | <code>        notifyNumPtr, notifyNum);</code> | 提供上述日志的实参，涉及主线程通知容量。 |
| [S2059 / L2972](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2972) | <code>    return HCCL_SUCCESS;</code> | 读取算法主Thread及通知容量处理完成，返回成功。 |
| [S2060 / L2974](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2974) | <code>}</code> | 结束GetMainThreadInfo函数体。 |


## 54. HcclGetChannel

按算法层和端点所在位置分组申请通道，OFFLOAD额外注册用户区

完整范围：[op_common.cc:L2979–L3036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2979-L3036)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S2064 / L2979](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2979) | <code>HcclResult HcclGetChannel(</code> | 声明HcclGetChannel接口：按算法层和端点所在位置分组申请通道，OFFLOAD额外注册用户区。 |
| [S2065 / L2981](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2981) | <code>    HcclComm comm, const OpParam&amp; param, AlgResourceRequest&amp; resRequest, AlgResourceCtxSerializable* resCtxHost)</code> | 函数参数包含通信域句柄、算子参数、Host资源描述对象、线程/通道/通知资源需求，本行延续接口声明。 |
| [S2066 / L2983](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2983) | <code>{</code> | 开始HcclGetChannel的函数体。 |
| [S2067 / L2985](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2985) | <code>    MemRegInfo memRegInfo;</code> | 创建图模式用户内存注册信息；OPBASE本例不注册图用户区。 |
| [S2069 / L2988](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2988) | <code>    if (param.opMode == OpMode::OFFLOAD) {</code> | OFFLOAD图模式才注册本次用户输入输出区。 |
| [S2070 / L2990](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2990) | <code>        HCCL_INFO(&quot;[HcclGetChannelImpl] start to RegGraphModeBuffers&quot;);</code> | 输出运行日志，记录HcclGetChannel当前阶段和相关参数。 |
| [S2071 / L2992](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2992) | <code>        CHK_RET(</code> | 开始图模式用户区注册调用。 |
| [S2072 / L2994](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2994) | <code>            RegGraphModeBuffers(comm, param, memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, memRegInfo.memHandles));</code> | 注册用户输入/输出内存并取得tag及内存注册句柄。 |
| [S2073 / L2996](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2996) | <code>    }</code> | 结束条件if (param.opMode == OpMode::OFFLOAD)。 |
| [S2074 / L2998](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2998) | <code>    resCtxHost-&gt;channels.resize(resRequest.channels.size());</code> | 将资源对象的通道层数量扩展为算法请求的层数。 |
| [S2076 / L3001](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3001) | <code>    for (u32 level = 0; level &lt; resRequest.channels.size(); level++) {</code> | 逐一处理算法层级的通道请求。 |
| [S2078 / L3004](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3004) | <code>        std::vector&lt;HcclChannelDesc&gt;&amp; levelNChannelRequest = resRequest.channels[level];</code> | 取当前算法层的建链请求数组引用。 |
| [S2079 / L3006](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3006) | <code>        std::vector&lt;HcclChannelDesc&gt; deviceChannelRequest;</code> | 创建当前层Device端点通道请求组。 |
| [S2080 / L3008](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3008) | <code>        std::vector&lt;HcclChannelDesc&gt; hostChannelRequest;</code> | 创建当前层Host端点通道请求组。 |
| [S2081 / L3010](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3010) | <code>        for (auto&amp; channelRequest : levelNChannelRequest) {</code> | 遍历当前层各个通道请求。 |
| [S2082 / L3012](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3012) | <code>            if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {</code> | 本地端点位于Device时归入Device建链组。 |
| [S2083 / L3014](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3014) | <code>                deviceChannelRequest.emplace_back(channelRequest);</code> | 复制该通道描述到Device建链请求组。 |
| [S2084 / L3016](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3016) | <code>            } else if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {</code> | 本地端点位于Host时归入Host建链组。 |
| [S2085 / L3018](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3018) | <code>                hostChannelRequest.emplace_back(channelRequest);</code> | 复制该通道描述到Host建链请求组。 |
| [S2086 / L3020](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3020) | <code>            }</code> | 结束条件} else if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST)。 |
| [S2087 / L3022](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3022) | <code>        }</code> | 结束循环for (auto&amp; channelRequest : levelNChannelRequest)。 |
| [S2089 / L3025](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3025) | <code>        CHK_RET(</code> | 开始取得当前层Device组通道。 |
| [S2090 / L3027](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3027) | <code>            HcclGetChannelImpl(level, comm, param, deviceChannelRequest, COMM_ENGINE_AICPU_TS, resCtxHost, memRegInfo));</code> | Device端点使用AICPU_TS执行域申请通道。 |
| [S2092 / L3030](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3030) | <code>        CHK_RET(HcclGetChannelImpl(level, comm, param, hostChannelRequest, COMM_ENGINE_CPU, resCtxHost, memRegInfo));</code> | Host端点使用CPU执行域申请通道，空组直接返回成功。 |
| [S2093 / L3032](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3032) | <code>    }</code> | 结束循环for (u32 level = 0; level &lt; resRequest.channels.size(); level++)。 |
| [S2094 / L3034](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3034) | <code>    return HCCL_SUCCESS;</code> | 按算法层和端点所在位置分组申请通道，OFFLOAD额外注册用户区处理完成，返回成功。 |
| [S2095 / L3036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3036) | <code>}</code> | 结束HcclGetChannel函数体。 |


## 55. BuildChannelInfo

从句柄构造模板使用的通道描述，查询端口属性与远端CCL地址

完整范围：[op_common.cc:L3041–L3140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3041-L3140)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S2099 / L3041](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3041) | <code>static HcclResult BuildChannelInfo(</code> | 声明BuildChannelInfo接口：从句柄构造模板使用的通道描述，查询端口属性与远端CCL地址。 |
| [S2100 / L3043](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3043) | <code>    HcclComm comm, const OpParam&amp; param, const HcclChannelDesc&amp; channelDesc, ChannelHandle channelHandle, u32 userRank,</code> | 函数参数包含通信域句柄、算子参数、HCOMM通道句柄、通道请求描述、本端rank，本行延续接口声明。 |
| [S2101 / L3045](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3045) | <code>    MemRegInfo&amp; memRegInfo, ChannelInfo&amp; channel)</code> | 函数参数包含模板通道信息、图模式内存注册信息，本行延续接口声明。 |
| [S2102 / L3047](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3047) | <code>{</code> | 开始BuildChannelInfo的函数体。 |
| [S2105 / L3051](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3051) | <code>    channel.isValid = true;</code> | 初始化模板通道有效标志为true。 |
| [S2106 / L3053](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3053) | <code>    channel.remoteRank = channelDesc.remoteRank;</code> | 保存本通道远端用户rank。 |
| [S2107 / L3055](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3055) | <code>    channel.protocol = channelDesc.channelProtocol;</code> | 保存通道采用的传输协议。 |
| [S2108 / L3057](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3057) | <code>    channel.locationType = channelDesc.remoteEndpoint.loc.locType;</code> | 保存远端端点位于Host或Device的位置类型。 |
| [S2109 / L3059](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3059) | <code>    channel.notifyNum = channelDesc.notifyNum;</code> | 保存本通道通知容量。 |
| [S2110 / L3061](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3061) | <code>    channel.handle = channelHandle;</code> | 保存HCOMM通道句柄供数据原语使用。 |
| [S2111 / L3063](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3063) | <code>#ifndef AICPU_COMPILE</code> | 仅未定义AICPU_COMPILE时编译以下Host端查询逻辑。 |
| [S2112 / L3065](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3065) | <code>    EndpointDesc localEndpoint = channelDesc.localEndpoint;</code> | 拷贝本地端点描述用于查询物理端口属性。 |
| [S2113 / L3067](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3067) | <code>    using portSizeType = uint32_t;</code> | 定义端口带宽系数查询结果类型为32位无符号整数。 |
| [S2114 / L3069](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3069) | <code>    const uint32_t portSizeTypeSize = sizeof(portSizeType);</code> | 保存该属性查询结果字节长度。 |
| [S2115 / L3071](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3071) | <code>    portSizeType portSize = 0;</code> | 将端口带宽系数结果初始化为0。 |
| [S2117 / L3074](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3074) | <code>    CHK_RET(HcclRankGraphGetEndpointInfo(</code> | 开始查询本地端点的带宽系数。 |
| [S2118 / L3076](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3076) | <code>        comm, userRank, &amp;localEndpoint, ENDPOINT_ATTR_BW_COEFF, portSizeTypeSize, static_cast&lt;void*&gt;(&amp;portSize)));</code> | 按本端rank和Endpoint查询ENDPOINT_ATTR_BW_COEFF属性。 |
| [S2119 / L3078](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3078) | <code>    channel.portGroupSize = portSize;</code> | 保存端口分组带宽系数到ChannelInfo。 |
| [S2120 / L3080](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3080) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S2121 / L3082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3082) | <code>        portSize == 0, HCCL_ERROR(&quot;[HcclGetChannelImpl] userRank [%d], portSize [%u] is 0.&quot;, userRank, portSize),</code> | 带宽系数为0时视为内部错误，不能继续建立有效通道描述。 |
| [S2122 / L3084](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3084) | <code>        HcclResult::HCCL_E_INTERNAL);</code> | 带宽系数非法时返回内部错误。 |
| [S2123 / L3086](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3086) | <code>    EndpointAttrDieId dieId = INVALID_VALUE_RANKID;</code> | 初始化端点dieId为无效值。 |
| [S2124 / L3088](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3088) | <code>    const uint32_t dieIdSize = sizeof(EndpointAttrDieId);</code> | 保存dieId属性查询的字节长度。 |
| [S2125 / L3090](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3090) | <code>    HcclResult dieIdRet = HcclRankGraphGetEndpointInfo(</code> | 声明端点dieId查询结果返回码。 |
| [S2126 / L3092](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3092) | <code>        comm, userRank, &amp;localEndpoint, ENDPOINT_ATTR_DIE_ID, dieIdSize, static_cast&lt;void*&gt;(&amp;dieId));</code> | 查询本地端点所属die编号。 |
| [S2127 / L3094](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3094) | <code>    if (dieIdRet == HCCL_SUCCESS) {</code> | 查询成功时保存die编号。 |
| [S2128 / L3096](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3096) | <code>        channel.dieId = dieId;</code> | 写入查询所得dieId。 |
| [S2129 / L3098](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3098) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S2130 / L3100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3100) | <code>        HCCL_WARNING(</code> | 输出警告日志，记录BuildChannelInfo当前阶段和相关参数。 |
| [S2131 / L3102](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3102) | <code>            &quot;[HcclGetChannelImpl] failed to get dieId for userRank[%u], remoteRank[%u], &quot;</code> | 补充日志格式：[HcclGetChannelImpl] failed to get dieId for userRank[%u], remoteRank[%u]。 |
| [S2132 / L3104](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3104) | <code>            &quot;ret[0x%016llx]. POD convergence adjustment will not be used for this channel.&quot;,</code> | 补充日志格式：ret[0x%016llx]. POD convergence adjustment will not be used for this channel.。 |
| [S2133 / L3106](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3106) | <code>            userRank, channel.remoteRank, HCCL_ERROR_CODE(dieIdRet));</code> | 提供上述日志的实参，涉及模板通道信息、本端rank。 |
| [S2134 / L3108](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3108) | <code>    }</code> | 结束条件} else。 |
| [S2135 / L3110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3110) | <code>#endif</code> | 非Host编译不查询上述物理端口属性。 |
| [S2137 / L3113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3113) | <code>    void* remoteCclBufferAddr = nullptr;</code> | 初始化远端CCL缓冲区地址输出为空。 |
| [S2138 / L3115](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3115) | <code>    uint64_t remoteCclBufferSize = 0;</code> | 初始化远端CCL容量输出为零。 |
| [S2139 / L3117](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3117) | <code>    CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandle, &amp;remoteCclBufferAddr, &amp;remoteCclBufferSize));</code> | 根据通道句柄查询对端CCL中转区地址和大小。 |
| [S2140 / L3119](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3119) | <code>    channel.remoteCclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteCclBufferAddr, remoteCclBufferSize};</code> | 以Device内存类型保存远端CCL区描述，Host不能直接解引用该远端地址。 |
| [S2141 / L3121](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3121) | <code>    HCCL_INFO(</code> | 输出运行日志，记录BuildChannelInfo当前阶段和相关参数。 |
| [S2142 / L3123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3123) | <code>        &quot;[%s]remoteRank[%u] protocol[%u] portGroupSize[%u] dieId[%u] &quot;</code> | 补充日志格式：[%s]remoteRank[%u] protocol[%u] portGroupSize[%u] dieId[%u]。 |
| [S2143 / L3125](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3125) | <code>        &quot;remoteCclBufferAddr[0x%llx] remoteCclBufferSize[%u]&quot;,</code> | 补充日志格式：remoteCclBufferAddr[0x%llx] remoteCclBufferSize[%u]。 |
| [S2144 / L3127](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3127) | <code>        __func__, channelDesc.remoteRank, channelDesc.channelProtocol, channel.portGroupSize, channel.dieId,</code> | 提供上述日志的实参，涉及通道请求描述、模板通道信息。 |
| [S2145 / L3129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3129) | <code>        remoteCclBufferAddr, remoteCclBufferSize);</code> | 提供上述日志的实参：remoteCclBufferAddr, remoteCclBufferSize。 |
| [S2147 / L3132](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3132) | <code>    if (param.opMode == OpMode::OFFLOAD) {</code> | OFFLOAD模式还需查询对端已注册的用户输入输出区。 |
| [S2148 / L3134](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3134) | <code>        CHK_RET(GetGraphModeBuffers(comm, channelHandle, memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, channel));</code> | 通过注册tag取得该通道对端的图模式用户区描述。 |
| [S2149 / L3136](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3136) | <code>    }</code> | 结束条件if (param.opMode == OpMode::OFFLOAD)。 |
| [S2150 / L3138](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3138) | <code>    return HCCL_SUCCESS;</code> | 从句柄构造模板使用的通道描述，查询端口属性与远端CCL地址处理完成，返回成功。 |
| [S2151 / L3140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3140) | <code>}</code> | 结束BuildChannelInfo函数体。 |


## 56. HcclGetChannelImpl

登记交换信息并申请通道，逐条构建ChannelInfo存入对应层

完整范围：[op_common.cc:L3145–L3208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3145-L3208)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S2161 / L3155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3155)：空请求；直接成功返回

- [S2175 / L3182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3182)：有请求；AddExchangeInfo -> HcclChannelAcquire -> BuildChannelInfo



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S2155 / L3145](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3145) | <code>HcclResult HcclGetChannelImpl(</code> | 声明HcclGetChannelImpl接口：登记交换信息并申请通道，逐条构建ChannelInfo存入对应层。 |
| [S2156 / L3147](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3147) | <code>    const u32 level, HcclComm comm, const OpParam&amp; param, std::vector&lt;HcclChannelDesc&gt;&amp; channelRequest,</code> | 函数参数包含通信域句柄、算子参数、建链描述请求数组、算法通信层序号，本行延续接口声明。 |
| [S2157 / L3149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3149) | <code>    const CommEngine commEngine, AlgResourceCtxSerializable* resCtxHost, MemRegInfo&amp; memRegInfo)</code> | 函数参数包含Host资源描述对象、图模式内存注册信息、通道执行域，本行延续接口声明。 |
| [S2158 / L3151](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3151) | <code>{</code> | 开始HcclGetChannelImpl的函数体。 |
| [S2161 / L3155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3155) | <code>    if (channelRequest.empty()) {</code> | 请求组为空时无需登记交换信息或创建通道。 |
| [S2162 / L3157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3157) | <code>        HCCL_INFO(&quot;[HcclGetChannelImpl] channelRequest is empty&quot;);</code> | 输出运行日志，记录HcclGetChannelImpl当前阶段和相关参数。 |
| [S2163 / L3159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3159) | <code>        return HCCL_SUCCESS;</code> | 空组直接成功返回。 |
| [S2164 / L3161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3161) | <code>    }</code> | 结束条件if (channelRequest.empty())。 |
| [S2165 / L3163](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3163) | <code>    u32 channelNum = channelRequest.size();</code> | 保存本组申请通道数量。 |
| [S2166 / L3165](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3165) | <code>    std::vector&lt;ChannelHandle&gt; levelNChannels;</code> | 创建通道句柄返回数组。 |
| [S2167 / L3167](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3167) | <code>    levelNChannels.resize(channelNum);</code> | 将句柄数组长度扩展为本组请求数量。 |
| [S2169 / L3170](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3170) | <code>    if (param.opMode == OpMode::OFFLOAD) {</code> | OFFLOAD模式才附加此前注册的用户内存句柄。 |
| [S2170 / L3172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3172) | <code>        for (auto&amp; channelDesc : channelRequest) {</code> | 逐一设置本组图模式通道描述。 |
| [S2171 / L3174](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3174) | <code>            channelDesc.memHandles = memRegInfo.memHandles.data();</code> | 把图模式内存注册句柄数组地址挂到通道描述。 |
| [S2172 / L3176](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3176) | <code>            channelDesc.memHandleNum = memRegInfo.memHandles.size();</code> | 设置图模式内存注册句柄数量。 |
| [S2173 / L3178](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3178) | <code>        }</code> | 结束循环for (auto&amp; channelDesc : channelRequest)。 |
| [S2174 / L3180](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3180) | <code>    }</code> | 结束条件if (param.opMode == OpMode::OFFLOAD)。 |
| [S2175 / L3182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3182) | <code>    if (channelNum &gt; 0) {</code> | 本组有请求时登记算子交换信息并取得通道。 |
| [S2178 / L3186](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3186) | <code>        CHK_RET(AddExchangeInfo(comm, param));</code> | 按一致性校验标志登记供本次建链读取的算子元信息。 |
| [S2179 / L3188](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3188) | <code>        CHK_RET(HcclChannelAcquire(comm, commEngine, channelRequest.data(), channelNum, levelNChannels.data()));</code> | 通过HCOMM通信域API取得整组通道；内部可能复用或新建。 |
| [S2180 / L3190](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3190) | <code>    }</code> | 结束条件if (channelNum &gt; 0)。 |
| [S2183 / L3194](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3194) | <code>    for (u32 idx = 0; idx &lt; channelNum; idx++) {</code> | 遍历本组取得的通道句柄。 |
| [S2184 / L3196](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3196) | <code>        ChannelInfo channel;</code> | 创建单个模板通道描述对象。 |
| [S2185 / L3198](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3198) | <code>        CHK_RET(BuildChannelInfo(</code> | 开始填充模板ChannelInfo。 |
| [S2186 / L3200](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3200) | <code>            comm, param, channelRequest[idx], levelNChannels[idx], resCtxHost-&gt;topoInfo.userRank, memRegInfo, channel));</code> | 以通道请求和实际句柄查询并填入端口/远端CCL属性。 |
| [S2187 / L3202](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3202) | <code>        resCtxHost-&gt;channels[level].push_back(channel);</code> | 将填好的通道描述追加到本算法层的资源通道列表。 |
| [S2188 / L3204](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3204) | <code>    }</code> | 结束循环for (u32 idx = 0; idx &lt; channelNum; idx++)。 |
| [S2189 / L3206](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3206) | <code>    return HCCL_SUCCESS;</code> | 登记交换信息并申请通道，逐条构建ChannelInfo存入对应层处理完成，返回成功。 |
| [S2190 / L3208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3208) | <code>}</code> | 结束HcclGetChannelImpl函数体。 |


## 57. CheckCount

限制输入元素count不超过系统支持的SYS_MAX_COUNT

完整范围：[op_common.cc:L4282–L4300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4282-L4300)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S3263 / L4282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4282) | <code>HcclResult CheckCount(const u64 count)</code> | 声明CheckCount接口：限制输入元素count不超过系统支持的SYS_MAX_COUNT。 |
| [S3264 / L4284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4284) | <code>{</code> | 开始CheckCount的函数体。 |
| [S3265 / L4286](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4286) | <code>    if (UNLIKELY(count &gt; SYS_MAX_COUNT)) {</code> | 超过SYS_MAX_COUNT的元素数量返回参数错误。 |
| [S3266 / L4288](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4288) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录CheckCount当前阶段和相关参数。 |
| [S3267 / L4290](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4290) | <code>            &quot;[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])&quot;,</code> | 补充日志格式：[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])。 |
| [S3268 / L4292](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4292) | <code>            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);</code> | 输出错误日志，记录CheckCount当前阶段和相关参数。 |
| [S3269 / L4294](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4294) | <code>        return HCCL_E_PARA;</code> | 元素count超出系统范围时返回HCCL_E_PARA。 |
| [S3270 / L4296](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4296) | <code>    }</code> | 结束条件if (UNLIKELY(count &gt; SYS_MAX_COUNT))。 |
| [S3271 / L4298](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4298) | <code>    return HCCL_SUCCESS;</code> | 限制输入元素count不超过系统支持的SYS_MAX_COUNT处理完成，返回成功。 |
| [S3272 / L4300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4300) | <code>}</code> | 结束CheckCount函数体。 |


## 58. CheckDataType

检查枚举值合法性并按归约/非归约场景排除不支持dtype

完整范围：[op_common.cc:L4303–L4374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4303-L4374)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S3274 / L4303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4303) | <code>HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)</code> | 声明CheckDataType接口：检查枚举值合法性并按归约/非归约场景排除不支持dtype。 |
| [S3275 / L4305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4305) | <code>{</code> | 开始CheckDataType的函数体。 |
| [S3276 / L4307](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4307) | <code>    const std::vector&lt;std::string&gt; infoTitle({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;});</code> | 创建错误上报字段名字列表，用于后面的dtype诊断。 |
| [S3278 / L4310](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4310) | <code>    bool notValid = VALID_HCCL_DATA_TYPES.find(dataType) == VALID_HCCL_DATA_TYPES.end();</code> | 通过VALID_HCCL_DATA_TYPES集合判断dataType是否是有效枚举。 |
| [S3279 / L4312](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4312) | <code>    if (needReduce) {</code> | needReduce为true走归约dtype限制，AllReduce满足该条件。 |
| [S3281 / L4315](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4315) | <code>        static const std::set&lt;HcclDataType&gt; REDUCE_UNSUPPORTED</code> | 定义归约不支持的数据类型集合。 |
| [S3282 / L4317](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4317) | <code>            = {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,  HCCL_DATA_TYPE_UINT32,  HCCL_DATA_TYPE_INT128,</code> | 归约排除UINT8/16/32及INT128。 |
| [S3283 / L4319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4319) | <code>               HCCL_DATA_TYPE_HIF8,  HCCL_DATA_TYPE_FP8E4M3, HCCL_DATA_TYPE_FP8E5M2, HCCL_DATA_TYPE_FP8E8M0};</code> | 归约同时排除HIF8和各FP8格式。 |
| [S3284 / L4321](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4321) | <code>        if (notValid &#124;&#124; REDUCE_UNSUPPORTED.find(dataType) != REDUCE_UNSUPPORTED.end()) {</code> | 枚举不合法或处于归约不支持列表时诊断并返回错误。 |
| [S3285 / L4323](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4323) | <code>            RPT_INPUT_ERR(</code> | 开始归约数据类型错误上报。 |
| [S3286 / L4325](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4325) | <code>                true, &quot;EI0003&quot;, infoTitle,</code> | 指定EI0003错误码和上报字段列表。 |
| [S3287 / L4327](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4327) | <code>                std::vector&lt;std::string&gt;(</code> | 构造归约dtype错误上报的字符串参数数组。 |
| [S3288 / L4329](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4329) | <code>                    {&quot;CheckDataType&quot;, GetDataTypeEnumStr(dataType), &quot;dataType&quot;, GetSupportDataType(needReduce)}));</code> | 上报当前dtype名字及按needReduce生成的支持类型字符串。 |
| [S3289 / L4331](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4331) | <code>            HCCL_ERROR(</code> | 输出错误日志，记录CheckDataType当前阶段和相关参数。 |
| [S3290 / L4333](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4333) | <code>                &quot;[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]&quot;,</code> | 补充日志格式：[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]。 |
| [S3291 / L4335](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4335) | <code>                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),</code> | 输出错误日志，记录CheckDataType当前阶段和相关参数。 |
| [S3292 / L4337](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4337) | <code>                GetSupportDataType(needReduce).c_str());</code> | 提供上述日志的实参：GetSupportDataType(needReduce).c_str(。 |
| [S3293 / L4339](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4339) | <code>            return HCCL_E_NOT_SUPPORT;</code> | 返回归约数据类型不支持错误。 |
| [S3294 / L4341](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4341) | <code>        }</code> | 结束条件if (notValid &#124;&#124; REDUCE_UNSUPPORTED.find(dataType) != REDUCE_UNSUPPORTED.end())。 |
| [S3295 / L4343](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4343) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S3297 / L4346](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4346) | <code>        if (notValid &#124;&#124; dataType == HCCL_DATA_TYPE_INT128) {</code> | 非归约场景排除非法枚举及INT128。 |
| [S3298 / L4348](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4348) | <code>            RPT_INPUT_ERR(</code> | 开始非归约数据类型错误上报。 |
| [S3299 / L4350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4350) | <code>                true, &quot;EI0003&quot;, infoTitle,</code> | 指定EI0003错误码和上报字段列表。 |
| [S3300 / L4352](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4352) | <code>                std::vector&lt;std::string&gt;(</code> | 创建非归约dtype诊断参数数组。 |
| [S3301 / L4354](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4354) | <code>                    {&quot;CheckDataType&quot;, GetDataTypeEnumStr(dataType), &quot;dataType&quot;,</code> | 上报当前数据类型名字及dataType参数名。 |
| [S3302 / L4356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4356) | <code>                     GetSupportDataType(needReduce).c_str()}));</code> | 补充非归约支持的数据类型列表字符串并结束上报。 |
| [S3303 / L4358](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4358) | <code>            HCCL_ERROR(</code> | 输出错误日志，记录CheckDataType当前阶段和相关参数。 |
| [S3304 / L4360](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4360) | <code>                &quot;[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]&quot;,</code> | 补充日志格式：[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]。 |
| [S3305 / L4362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4362) | <code>                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),</code> | 输出错误日志，记录CheckDataType当前阶段和相关参数。 |
| [S3306 / L4364](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4364) | <code>                GetSupportDataType(needReduce).c_str());</code> | 提供上述日志的实参：GetSupportDataType(needReduce).c_str(。 |
| [S3307 / L4366](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4366) | <code>            return HCCL_E_NOT_SUPPORT;</code> | 返回非归约数据类型不支持错误。 |
| [S3308 / L4368](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4368) | <code>        }</code> | 结束条件if (notValid &#124;&#124; dataType == HCCL_DATA_TYPE_INT128)。 |
| [S3309 / L4370](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4370) | <code>    }</code> | 结束条件} else。 |
| [S3310 / L4372](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4372) | <code>    return HCCL_SUCCESS;</code> | 检查枚举值合法性并按归约/非归约场景排除不支持dtype处理完成，返回成功。 |
| [S3311 / L4374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4374) | <code>}</code> | 结束CheckDataType函数体。 |


## 59. CheckReduceOp

PROD归约额外检查dtype支持列表，其它reduce类型本函数不做附加检查

完整范围：[op_common.cc:L4402–L4444](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4402-L4444)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S3338 / L4402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4402) | <code>HcclResult CheckReduceOp(const HcclDataType dataType, const HcclReduceOp op)</code> | 声明CheckReduceOp接口：PROD归约额外检查dtype支持列表，其它reduce类型本函数不做附加检查。 |
| [S3339 / L4404](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4404) | <code>{</code> | 开始CheckReduceOp的函数体。 |
| [S3340 / L4406](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4406) | <code>    std::vector&lt;HcclDataType&gt; prodSupportList</code> | 创建PROD归约专用支持数据类型列表。 |
| [S3341 / L4408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4408) | <code>        = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,</code> | PROD支持INT8/INT32/INT64/UINT64等整数类型。 |
| [S3342 / L4410](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4410) | <code>           HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_FP64};</code> | PROD同时支持FP16/FP32/FP64，列表中没有INT16和BFP16。 |
| [S3343 / L4412](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4412) | <code>    const std::vector&lt;std::string&gt; infoTitle({&quot;ccl_op&quot;, &quot;value&quot;, &quot;parameter&quot;, &quot;expect&quot;});</code> | 创建归约数据类型错误上报字段列表。 |
| [S3344 / L4414](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4414) | <code>    if (op == HcclReduceOp::HCCL_REDUCE_PROD) {</code> | 仅PROD归约进入本函数附加dtype支持检查；SUM等跳过。 |
| [S3345 / L4416](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4416) | <code>        if (std::find(prodSupportList.begin(), prodSupportList.end(), dataType) == prodSupportList.end()) {</code> | 在PROD支持列表中找不到当前dtype则返回不支持。 |
| [S3346 / L4418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4418) | <code>            RPT_INPUT_ERR(</code> | 开始PROD数据类型不支持的错误上报。 |
| [S3347 / L4420](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4420) | <code>                true, &quot;EI0003&quot;, infoTitle,</code> | 指定EI0003错误码和上报字段名字。 |
| [S3348 / L4422](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4422) | <code>                std::vector&lt;std::string&gt;(</code> | 构造PROD dtype诊断参数数组。 |
| [S3349 / L4424](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4424) | <code>                    {&quot;CheckReduceDataType&quot;, GetDataTypeEnumStr(dataType), &quot;dataType&quot;, GetReduceProdSupportDataType()}));</code> | 上报当前dtype及PROD允许的数据类型字符串。 |
| [S3350 / L4426](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4426) | <code>            HCCL_ERROR(</code> | 输出错误日志，记录CheckReduceOp当前阶段和相关参数。 |
| [S3351 / L4428](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4428) | <code>                &quot;[Check][ReduceOp][DataType]errNo[0x%016llx] reduceop is [%s] data type[%s] not supported, support &quot;</code> | 补充日志格式：[Check][ReduceOp][DataType]errNo[0x%016llx] reduceop is [%s] data type[%s] not supported, support。 |
| [S3352 / L4430](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4430) | <code>                &quot;range=[%s]&quot;,</code> | 补充日志格式：range=[%s]。 |
| [S3353 / L4432](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4432) | <code>                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetReduceOpEnumStr(op).c_str(),</code> | 输出错误日志，记录CheckReduceOp当前阶段和相关参数。 |
| [S3354 / L4434](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4434) | <code>                GetDataTypeEnumStr(dataType).c_str(), GetReduceProdSupportDataType().c_str());</code> | 提供上述日志的实参：GetDataTypeEnumStr(dataType).c_str(), GetReduceProdSupportDataType().c_str(。 |
| [S3355 / L4436](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4436) | <code>            return HCCL_E_NOT_SUPPORT;</code> | 返回PROD数据类型不支持错误。 |
| [S3356 / L4438](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4438) | <code>        }</code> | 结束条件if (std::find(prodSupportList.begin(), prodSupportList.end(), dataType) == prodSupportList.end())。 |
| [S3357 / L4440](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4440) | <code>    }</code> | 结束条件if (op == HcclReduceOp::HCCL_REDUCE_PROD)。 |
| [S3358 / L4442](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4442) | <code>    return HCCL_SUCCESS;</code> | PROD归约额外检查dtype支持列表，其它reduce类型本函数不做附加检查处理完成，返回成功。 |
| [S3359 / L4444](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4444) | <code>}</code> | 结束CheckReduceOp函数体。 |


## 60. SetCommEngine

将最终opExecuteConfig转换为执行engine

完整范围：[op_common.cc:L4463–L4508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4463-L4508)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S3377 / L4463](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4463) | <code>HcclResult SetCommEngine(OpParam&amp; param)</code> | 声明SetCommEngine接口：将最终opExecuteConfig转换为执行engine。 |
| [S3378 / L4465](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4465) | <code>{</code> | 开始SetCommEngine的函数体。 |
| [S3380 / L4468](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4468) | <code>    static const std::unordered_map&lt;OpExecuteConfig, CommEngine&gt; ConfigToEngineMap = {</code> | 建立执行配置到通信引擎的静态映射表。 |
| [S3381 / L4470](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4470) | <code>        {OpExecuteConfig::HOSTCPU_TS, COMM_ENGINE_CPU_TS},</code> | HOSTCPU_TS配置对应CPU_TS线程执行域。 |
| [S3382 / L4472](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4472) | <code>        {OpExecuteConfig::AICPU_TS, COMM_ENGINE_AICPU_TS},</code> | AICPU_TS配置对应AICPU_TS执行域，本例在此映射。 |
| [S3383 / L4474](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4474) | <code>        {OpExecuteConfig::AIV, COMM_ENGINE_AIV},</code> | AIV配置对应AIV执行域。 |
| [S3384 / L4476](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4476) | <code>        {OpExecuteConfig::AIV_ONLY, COMM_ENGINE_AIV}, // AIV_ONLY 和 AIV 映射到同一引擎</code> | AIV_ONLY配置也映射为AIV执行域，强制模式约束在Selector中另查。 |
| [S3385 / L4478](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4478) | <code>        {OpExecuteConfig::CCU_MS, COMM_ENGINE_CCU},</code> | CCU_MS配置对应CCU执行域。 |
| [S3386 / L4480](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4480) | <code>        {OpExecuteConfig::CCU_SCHED, COMM_ENGINE_CCU},</code> | CCU_SCHED配置也对应CCU执行域。 |
| [S3387 / L4482](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4482) | <code>        {OpExecuteConfig::AICPU, COMM_ENGINE_AICPU},</code> | 裸AICPU配置对应AICPU执行域。 |
| [S3388 / L4484](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4484) | <code>        {OpExecuteConfig::HOSTCPU, COMM_ENGINE_CPU},</code> | HOSTCPU配置对应CPU执行域。 |
| [S3389 / L4486](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4486) | <code>    };</code> | 结束配置到引擎的映射表初始化。 |
| [S3391 / L4489](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4489) | <code>    auto it = ConfigToEngineMap.find(param.opExecuteConfig);</code> | 按当前param.opExecuteConfig查询映射表。 |
| [S3392 / L4491](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4491) | <code>    if (it != ConfigToEngineMap.end()) {</code> | 配置存在于映射表时写回引擎。 |
| [S3393 / L4493](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4493) | <code>        param.engine = it-&gt;second;</code> | 保存该配置对应的通信执行域到param.engine。 |
| [S3394 / L4495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4495) | <code>        return HCCL_SUCCESS;</code> | 执行配置转换成功返回。 |
| [S3395 / L4497](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4497) | <code>    }</code> | 结束条件if (it != ConfigToEngineMap.end())。 |
| [S3397 / L4500](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4500) | <code>    HCCL_ERROR(</code> | 输出错误日志，记录SetCommEngine当前阶段和相关参数。 |
| [S3398 / L4502](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4502) | <code>        &quot;[op_common][SetCommEngine] Unsupported or unknown opExecuteConfig: {%d}&quot;,</code> | 补充日志格式：[op_common][SetCommEngine] Unsupported or unknown opExecuteConfig: {%d}。 |
| [S3399 / L4504](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4504) | <code>        static_cast&lt;int&gt;(param.opExecuteConfig));</code> | 提供上述日志的实参，涉及算子参数。 |
| [S3400 / L4506](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4506) | <code>    return HCCL_E_NOT_SUPPORT;</code> | 未知执行配置返回不支持错误。 |
| [S3401 / L4508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4508) | <code>}</code> | 结束代码块。 |


## 61. SingleRankProc

单rank算子旁支只需本地复制输入到输出，不执行多rankAllReduce算法

完整范围：[op_common.cc:L4511–L4623](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4511-L4623)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S3403 / L4511](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4511) | <code>HcclResult SingleRankProc(HcclComm comm, OpParam&amp; param)</code> | 声明SingleRankProc接口：单rank算子旁支只需本地复制输入到输出，不执行多rankAllReduce算法。 |
| [S3404 / L4513](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4513) | <code>{</code> | 开始SingleRankProc的函数体。 |
| [S3405 / L4515](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4515) | <code>    uint64_t beginTime = HcommGetProfilingSysCycleTime();</code> | 记录单rank处理profiling开始时间。 |
| [S3406 / L4517](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4517) | <code>    HCCL_INFO(&quot;[SingleRankProc]Start to execute HcclExecOp. HcommGetProfilingSysCycleTime[%llu us]&quot;, beginTime);</code> | 输出运行日志，记录SingleRankProc当前阶段和相关参数。 |
| [S3407 / L4519](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4519) | <code>    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY) {</code> | AIV_ONLY模式不能使用此单rank处理旁支。 |
| [S3408 / L4521](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4521) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录SingleRankProc当前阶段和相关参数。 |
| [S3409 / L4523](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4523) | <code>            &quot;[SingleRankProc] opType[%s] currently do not select aiv mode, aiv only not support, &quot;</code> | 补充日志格式：[SingleRankProc] opType[%s] currently do not select aiv mode, aiv only not support。 |
| [S3410 / L4525](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4525) | <code>            &quot;please ensure rankNum is greater than one&quot;,</code> | 补充日志格式：please ensure rankNum is greater than one。 |
| [S3411 / L4527](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4527) | <code>            GetHcclCMDTypeStr(param.opType));</code> | 提供上述日志的实参，涉及算子参数。 |
| [S3412 / L4529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4529) | <code>        return HCCL_E_NOT_SUPPORT;</code> | 单rank但强制AIV_ONLY时返回不支持。 |
| [S3413 / L4531](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4531) | <code>    }</code> | 结束条件if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY)。 |
| [S3414 / L4533](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4533) | <code>    if (param.opType == HcclCMDType::HCCL_CMD_SEND &#124;&#124; param.opType == HcclCMDType::HCCL_CMD_RECEIVE) {</code> | 单rank的Send/Receive按源码成功返回，不执行数据复制。 |
| [S3415 / L4535](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4535) | <code>        HCCL_WARNING(&quot;[%s] ranksize == 1 is not support BATCHSENDRECV SEND RECV&quot;, __func__);</code> | 输出警告日志，记录SingleRankProc当前阶段和相关参数。 |
| [S3416 / L4537](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4537) | <code>        return HcclResult::HCCL_SUCCESS;</code> | 返回单rankP2P分支成功状态。 |
| [S3417 / L4539](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4539) | <code>    }</code> | 结束条件if (param.opType == HcclCMDType::HCCL_CMD_SEND &#124;&#124; param.opType == HcclCMDType::HCCL_CMD_RECEIVE)。 |
| [S3418 / L4541](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4541) | <code>    if (param.inputPtr == param.outputPtr) {</code> | 输入输出地址相同时无需本地复制。 |
| [S3419 / L4543](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4543) | <code>        HCCL_WARNING(&quot;[%s] sendBuf == recvBuf, return success&quot;, __func__);</code> | 输出警告日志，记录SingleRankProc当前阶段和相关参数。 |
| [S3420 / L4545](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4545) | <code>        return HcclResult::HCCL_SUCCESS;</code> | 原地单rank场景直接成功返回。 |
| [S3421 / L4547](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4547) | <code>    }</code> | 结束条件if (param.inputPtr == param.outputPtr)。 |
| [S3422 / L4549](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4549) | <code>    u64 len{0};</code> | 初始化需要本地复制的字节数量为零。 |
| [S3423 / L4551](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4551) | <code>    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL &#124;&#124; param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV</code> | AllToAll和AllToAllV使用其发送dtype/count布局。 |
| [S3424 / L4553](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4553) | <code>        &#124;&#124; param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {</code> | AllToAllVC同样使用all2AllVDataDes布局。 |
| [S3425 / L4555](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4555) | <code>        len = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType]</code> | 读取AllToAll发送数据类型对应的元素字节大小。 |
| [S3426 / L4557](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4557) | <code>              * *(static_cast&lt;const u64*&gt;(param.all2AllVDataDes.sendCounts));</code> | 单Rank的AllToAll长度续行：取首个sendCounts元素，乘前行的数据类型字节数得到拷贝长度；AllReduce不走此分支。 |
| [S3427 / L4559](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4559) | <code>    } else if (param.opType == HCCL_CMD_ALLGATHER_V &#124;&#124; param.opType == HCCL_CMD_REDUCE_SCATTER_V) {</code> | AllGatherV或ReduceScatterV使用vDataDes布局。 |
| [S3428 / L4561](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4561) | <code>        len = DATATYPE_SIZE_TABLE[param.vDataDes.dataType] * *(static_cast&lt;const u64*&gt;(param.vDataDes.counts));</code> | vDataDes数据类型大小乘counts第一项得到有效复制长度。 |
| [S3429 / L4563](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4563) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S3430 / L4565](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4565) | <code>        len = DATATYPE_SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;</code> | 普通AllReduce按DataDes.count乘dtype字节数计算复制长度。 |
| [S3431 / L4567](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4567) | <code>    }</code> | 结束条件} else。 |
| [S3432 / L4569](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4569) | <code>    HCCL_INFO(&quot;[%s] sendBuf[%p], recvBuf[%p], len[%llu]&quot;, __func__, param.inputPtr, param.outputPtr, len);</code> | 输出运行日志，记录SingleRankProc当前阶段和相关参数。 |
| [S3433 / L4571](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4571) | <code>    if (len &gt; 0) {</code> | 有非零数据字节时才创建流线程并安排复制。 |
| [S3434 / L4573](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4573) | <code>        ThreadHandle cpuTsThread{0};</code> | 初始化本次用户流CPU_TS线程句柄。 |
| [S3435 / L4575](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4575) | <code>        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &amp;cpuTsThread));</code> | 将本次ACL用户流包装为CPU_TS线程，通知容量为1。 |
| [S3436 / L4577](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4577) | <code>        HcclDfxOpInfoCompat hcclDfxOpInfo{}; // Op注册</code> | 创建单rank算子维测信息结构。 |
| [S3437 / L4579](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4579) | <code>        hcclDfxOpInfo.opMode = static_cast&lt;u32&gt;(param.opMode);</code> | 保存单算子或图模式到维测信息。 |
| [S3438 / L4581](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4581) | <code>        hcclDfxOpInfo.opType = static_cast&lt;u32&gt;(param.opType);</code> | 保存当前命令类型到维测信息。 |
| [S3439 / L4583](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4583) | <code>        hcclDfxOpInfo.reduceOp = static_cast&lt;u32&gt;(param.reduceType);</code> | 保存当前SUM等归约类型到维测信息。 |
| [S3440 / L4585](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4585) | <code>        CHK_RET(GetHcclDfxOpInfoDataType(param, hcclDfxOpInfo.dataType));</code> | 按算子描述布局读取数据类型写进维测结构。 |
| [S3441 / L4587](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4587) | <code>        u32 userRankSize{0}; // rankSize获取指定算子的dataCount</code> | 初始化通信域rank数量查询结果。 |
| [S3442 / L4589](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4589) | <code>        CHK_RET(HcclGetRankSize(comm, &amp;userRankSize));</code> | 查询本次通信域rank数量。 |
| [S3443 / L4591](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4591) | <code>        CHK_RET(GetHcclDfxOpInfoDataCount(param, userRankSize, hcclDfxOpInfo.dataCount));</code> | 按算子类型及rank数计算维测记录的元素总数。 |
| [S3444 / L4593](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4593) | <code>        hcclDfxOpInfo.root = param.root;</code> | 保存通用root字段到维测结构。 |
| [S3445 / L4595](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4595) | <code>        hcclDfxOpInfo.engine = param.engine;</code> | 保存本次执行引擎到维测结构。 |
| [S3446 / L4597](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4597) | <code>        hcclDfxOpInfo.cpuTsThread = cpuTsThread;</code> | 将本次用户流对应CPU_TS线程放进维测结构。 |
| [S3447 / L4599](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4599) | <code>        hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;</code> | 设置维测结构Host结果通知槽字段。 |
| [S3448 / L4601](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4601) | <code>        CHK_RET(SetOpParamAlgTag(param, &quot;SingleRankProc&quot;));</code> | 用SingleRankProc名字构造本地复制旁支的算法tag。 |
| [S3449 / L4603](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4603) | <code>        s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, param.algTag, ALG_TAG_LENGTH);</code> | 将生成的算法tag复制到维测固定字符串区。 |
| [S3450 / L4605](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4605) | <code>        CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S3451 / L4607](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4607) | <code>            sRet != EOK,</code> | tag安全复制失败时触发日志和错误返回。 |
| [S3452 / L4609](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4609) | <code>            HCCL_ERROR(&quot;%s call strncpy_s failed, param.algTag %s, return %d.&quot;, __func__, param.algTag, sRet),</code> | 输出错误日志，记录SingleRankProc当前阶段和相关参数。 |
| [S3453 / L4611](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4611) | <code>            HCCL_E_MEMORY);</code> | 算法tag复制失败返回内存错误。 |
| [S3454 / L4613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4613) | <code>        CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast&lt;void*&gt;(&amp;hcclDfxOpInfo)));</code> | 向通信域登记单rank算子的维测信息。 |
| [S3455 / L4615](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4615) | <code>        CHK_RET(static_cast&lt;HcclResult&gt;(HcommLocalCopyOnThread(cpuTsThread, param.outputPtr, param.inputPtr, len)));</code> | 在用户CPU_TS线程安排input到output的len字节本地复制。 |
| [S3456 / L4617](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4617) | <code>    }</code> | 结束条件if (len &gt; 0)。 |
| [S3457 / L4619](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4619) | <code>    CHK_RET(HcclProfilingReportOp(comm, beginTime));</code> | 上报单rank算子profiling时间。 |
| [S3458 / L4621](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4621) | <code>    return HcclResult::HCCL_SUCCESS;</code> | 单rank复制任务安排成功返回，实际完成仍由用户流执行。 |
| [S3459 / L4623](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4623) | <code>}</code> | 结束SingleRankProc函数体。 |


## 62. SetOpParamAlgTag

构造算法资源关联tag，CCU额外添加dtype/reduce字段

完整范围：[op_common.cc:L4665–L4738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4665-L4738)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S3500 / L4665](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4665) | <code>HcclResult SetOpParamAlgTag(OpParam&amp; param, const std::string&amp; algName)</code> | 声明SetOpParamAlgTag接口：构造算法资源关联tag，CCU额外添加dtype/reduce字段。 |
| [S3501 / L4667](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4667) | <code>{</code> | 开始SetOpParamAlgTag的函数体。 |
| [S3502 / L4669](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4669) | <code>    std::string temp = algName; // 创建algName的副本</code> | 拷贝算法名字符串用于组装算法tag。 |
| [S3504 / L4672](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4672) | <code>    const char* launchMode</code> | 声明tag的执行位置后缀字符串。 |
| [S3505 / L4674](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4674) | <code>        = (((param.engine == CommEngine::COMM_ENGINE_AICPU) &#124;&#124; (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ?</code> | AICPU或AICPU_TS引擎的tag后缀为device。 |
| [S3506 / L4676](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4676) | <code>               &quot;device&quot; :</code> | Device展开模式使用device字符串后缀。 |
| [S3507 / L4678](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4678) | <code>               &quot;host&quot;);</code> | 其它引擎使用host字符串后缀。 |
| [S3508 / L4680](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4680) | <code>    int len;</code> | 声明tag格式化返回的实际字符数。 |
| [S3510 / L4683](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4683) | <code>    if (param.opMode == OpMode::OFFLOAD &amp;&amp; param.engine == CommEngine::COMM_ENGINE_CCU) {</code> | CCU OFFLOAD图模式不带用户tag前缀，便于复用同算法资源。 |
| [S3511 / L4685](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4685) | <code>        len = snprintf_s(</code> | 开始格式化CCU图模式关联tag。 |
| [S3512 / L4687](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4687) | <code>            param.algTag, sizeof(param.algTag), sizeof(param.algTag), &quot;Graph_%s_%s&quot;, temp.c_str(), launchMode);</code> | 构造Graph_算法名_执行位置的固定数组字符串。 |
| [S3513 / L4689](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4689) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S3514 / L4691](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4691) | <code>        len = snprintf_s(</code> | 普通场景开始格式化算法关联tag。 |
| [S3515 / L4693](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4693) | <code>            param.algTag, sizeof(param.algTag), sizeof(param.algTag), &quot;%s_%s_%s&quot;, param.tag, temp.c_str(), launchMode);</code> | 构造算子tag_算法名_执行位置；本例算法名字为AicpuAllReduceSoleMeshOneShot。 |
| [S3516 / L4695](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4695) | <code>    }</code> | 结束条件} else。 |
| [S3517 / L4697](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4697) | <code>    if (len &lt; 0 &#124;&#124; len &gt;= sizeof(param.algTag)) {</code> | 检查snprintf返回负数或超出固定tag容量。 |
| [S3518 / L4699](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4699) | <code>        HCCL_ERROR(&quot;failed to fill param.algTag&quot;);</code> | 输出错误日志，记录SetOpParamAlgTag当前阶段和相关参数。 |
| [S3519 / L4701](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4701) | <code>        return HcclResult::HCCL_E_INTERNAL;</code> | 算法tag构造失败返回内部错误。 |
| [S3520 / L4703](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4703) | <code>    }</code> | 结束条件if (len &lt; 0 &#124;&#124; len &gt;= sizeof(param.algTag))。 |
| [S3523 / L4707](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4707) | <code>    if (param.engine == CommEngine::COMM_ENGINE_CCU) {</code> | CCU算法tag还需增加数据类型及归约类型等专属字段。 |
| [S3524 / L4709](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4709) | <code>        try {</code> | 用try包围CCU枚举字符串查询，捕获越界异常。 |
| [S3525 / L4711](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4711) | <code>            std::string ccuExtraTag;</code> | 创建存放CCU附加字段的临时字符串。 |
| [S3526 / L4713](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4713) | <code>            CHK_RET(BuildCcuExtraTag(param, ccuExtraTag));</code> | 调用BuildCcuExtraTag生成CCU额外资源关联字段，非本AICPU例。 |
| [S3527 / L4715](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4715) | <code>            size_t remainBytes = sizeof(param.algTag) - len;</code> | 计算基础tag之后固定字符数组还剩的字节容量。 |
| [S3529 / L4718](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4718) | <code>            int len_ccu = snprintf_s(param.algTag + len, remainBytes, remainBytes, &quot;%s&quot;, ccuExtraTag.c_str());</code> | 从algTag+len处追加CCU附加字符串。 |
| [S3530 / L4720](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4720) | <code>            CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S3531 / L4722](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4722) | <code>                (len_ccu &lt; 0 &#124;&#124; len_ccu &gt;= sizeof(param.algTag) - len),</code> | 检查追加字符串失败或超过剩余tag容量。 |
| [S3532 / L4724](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4724) | <code>                HCCL_ERROR(&quot;failed to fill alg tag with ccu dataType&quot;), HCCL_E_INTERNAL);</code> | CCU附加tag写入失败记录日志并返回内部错误。 |
| [S3533 / L4726](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4726) | <code>        } catch (const std::out_of_range&amp; e) {</code> | 捕获CCU数据类型/归约枚举映射查询的out_of_range异常。 |
| [S3534 / L4728](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4728) | <code>            HCCL_ERROR(&quot;[SetOpParamAlgTag] dataType or reduceType out of range: %s&quot;, e.what());</code> | 输出错误日志，记录SetOpParamAlgTag当前阶段和相关参数。 |
| [S3535 / L4730](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4730) | <code>            return HCCL_E_PARA;</code> | 枚举映射越界时返回参数错误。 |
| [S3536 / L4732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4732) | <code>        }</code> | 结束代码块。 |
| [S3537 / L4734](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4734) | <code>    }</code> | 结束条件if (param.engine == CommEngine::COMM_ENGINE_CCU)。 |
| [S3538 / L4736](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4736) | <code>    return HcclResult::HCCL_SUCCESS;</code> | 基础tag及必要的CCU附加字段均构造成功，返回HCCL_SUCCESS。 |
| [S3539 / L4738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4738) | <code>}</code> | 结束SetOpParamAlgTag函数体。 |


## 63. OpLaunchGetUnfoldStream

以可选接口查询Host展开线程关联ACL stream

完整范围：[order_launch.cc:L19–L63](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L19-L63)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S18 / L19](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L19) | <code>static HcclResult OpLaunchGetUnfoldStream(HcclComm comm, ThreadHandle unfoldThread, aclrtStream&amp; resolvedStream)</code> | 声明OpLaunchGetUnfoldStream接口：以可选接口查询Host展开线程关联ACL stream。 |
| [S19 / L21](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L21) | <code>{</code> | 开始OpLaunchGetUnfoldStream的函数体。 |
| [S20 / L23](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L23) | <code>    void* unfoldStream = nullptr;</code> | 初始化待查询的Host展开流地址。 |
| [S21 / L25](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L25) | <code>    auto&amp; HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();</code> | 取得可选HCOMM动态函数表。 |
| [S22 / L27](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L27) | <code>    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo) {</code> | 缺少HcclThreadResGetInfo函数时无法查询展开流。 |
| [S23 / L29](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L29) | <code>        resolvedStream = nullptr;</code> | 没有查询能力时将流输出置空。 |
| [S24 / L31](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L31) | <code>        HCCL_WARNING(&quot;HcclThreadResGetInfoFunc dlHcclThreadResGetInfo is invalid.&quot;);</code> | 输出警告日志，记录OpLaunchGetUnfoldStream当前阶段和相关参数。 |
| [S25 / L33](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L33) | <code>        return HCCL_SUCCESS;</code> | 此辅助函数以成功和空流表示查询不可用，后续调用者再检查。 |
| [S26 / L35](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L35) | <code>    }</code> | 结束条件if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo)。 |
| [S27 / L37](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L37) | <code>    HcclResult ret</code> | 声明展开流查询返回码。 |
| [S28 / L39](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L39) | <code>        = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &amp;unfoldStream);</code> | 查询unfoldThread的ACL stream关联资源。 |
| [S29 / L41](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L41) | <code>    if (ret == HCCL_E_NOT_SUPPORT) {</code> | 明确NOT_SUPPORT时使用空流返回。 |
| [S30 / L43](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L43) | <code>        resolvedStream = nullptr;</code> | 将不支持查询的展开流结果设为空。 |
| [S31 / L45](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L45) | <code>        HCCL_WARNING(&quot;HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not support.&quot;);</code> | 输出警告日志，记录OpLaunchGetUnfoldStream当前阶段和相关参数。 |
| [S32 / L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L47) | <code>        return HCCL_SUCCESS;</code> | 返回成功，由上层OpLaunchGetOrderStreams检查空流。 |
| [S33 / L49](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L49) | <code>    } else if (ret != HCCL_SUCCESS) {</code> | 其它查询错误也返回空流，不向上传递原ret。 |
| [S34 / L51](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L51) | <code>        resolvedStream = nullptr;</code> | 将查询失败输出流设为空。 |
| [S35 / L53](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L53) | <code>        HCCL_WARNING(&quot;HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not success.&quot;);</code> | 输出警告日志，记录OpLaunchGetUnfoldStream当前阶段和相关参数。 |
| [S36 / L55](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L55) | <code>        return HCCL_SUCCESS;</code> | 返回成功，由使用者处理空流。 |
| [S37 / L57](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L57) | <code>    }</code> | 结束条件} else if (ret != HCCL_SUCCESS)。 |
| [S38 / L59](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L59) | <code>    resolvedStream = unfoldStream;</code> | 查询成功时输出实际展开流。 |
| [S39 / L61](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L61) | <code>    return HCCL_SUCCESS;</code> | 以可选接口查询Host展开线程关联ACL stream处理完成，返回成功。 |
| [S40 / L63](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L63) | <code>}</code> | 结束OpLaunchGetUnfoldStream函数体。 |


## 64. OpLaunchGetHostOrderStream

以可选接口查询Host保序线程关联ACL stream

完整范围：[order_launch.cc:L66–L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L66-L110)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S42 / L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L66) | <code>static HcclResult OpLaunchGetHostOrderStream(ThreadHandle hostOrderThread, aclrtStream&amp; resolvedStream)</code> | 声明OpLaunchGetHostOrderStream接口：以可选接口查询Host保序线程关联ACL stream。 |
| [S43 / L68](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L68) | <code>{</code> | 开始OpLaunchGetHostOrderStream的函数体。 |
| [S44 / L70](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L70) | <code>    void* hostOrderStream = nullptr;</code> | 初始化Host保序流输出地址。 |
| [S45 / L72](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L72) | <code>    auto&amp; HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();</code> | 取得可选HCOMM动态函数表。 |
| [S46 / L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L74) | <code>    if (!HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo) {</code> | 缺少无通信域版本HcommThreadResGetInfo函数时无法查询保序流。 |
| [S47 / L76](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L76) | <code>        resolvedStream = nullptr;</code> | 没有查询能力时流输出置空。 |
| [S48 / L78](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L78) | <code>        HCCL_WARNING(&quot;HcclThreadResGetInfoFunc dlHcommThreadResGetInfo is invalid.&quot;);</code> | 输出警告日志，记录OpLaunchGetHostOrderStream当前阶段和相关参数。 |
| [S49 / L80](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L80) | <code>        return HCCL_SUCCESS;</code> | 返回成功和空流供上层检查。 |
| [S50 / L82](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L82) | <code>    }</code> | 结束条件if (!HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo)。 |
| [S51 / L84](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L84) | <code>    HcclResult ret</code> | 声明保序流查询返回码。 |
| [S52 / L86](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L86) | <code>        = HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo(hostOrderThread, 0, sizeof(void*), &amp;hostOrderStream);</code> | 查询Host专用保序线程对应ACL stream。 |
| [S53 / L88](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L88) | <code>    if (ret == HCCL_E_NOT_SUPPORT) {</code> | 明确NOT_SUPPORT时使用空流结果。 |
| [S54 / L90](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L90) | <code>        resolvedStream = nullptr;</code> | 将不支持查询的保序流输出置空。 |
| [S55 / L92](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L92) | <code>        HCCL_WARNING(&quot;HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not support.&quot;);</code> | 输出警告日志，记录OpLaunchGetHostOrderStream当前阶段和相关参数。 |
| [S56 / L94](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L94) | <code>        return HCCL_SUCCESS;</code> | 返回成功，由上层统一处理空流。 |
| [S57 / L96](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L96) | <code>    } else if (ret != HCCL_SUCCESS) {</code> | 其它查询错误同样转换为空流。 |
| [S58 / L98](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L98) | <code>        resolvedStream = nullptr;</code> | 将保序流查询失败输出置空。 |
| [S59 / L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L100) | <code>        HCCL_WARNING(&quot;HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not success.&quot;);</code> | 输出警告日志，记录OpLaunchGetHostOrderStream当前阶段和相关参数。 |
| [S60 / L102](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L102) | <code>        return HCCL_SUCCESS;</code> | 返回成功，未传播原始ret。 |
| [S61 / L104](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L104) | <code>    }</code> | 结束条件} else if (ret != HCCL_SUCCESS)。 |
| [S62 / L106](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L106) | <code>    resolvedStream = hostOrderStream;</code> | 查询成功时输出真实Host保序流。 |
| [S63 / L108](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L108) | <code>    return HCCL_SUCCESS;</code> | 以可选接口查询Host保序线程关联ACL stream处理完成，返回成功。 |
| [S64 / L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L110) | <code>}</code> | 结束OpLaunchGetHostOrderStream函数体。 |


## 65. GetOrderLaunchModeName

将保序模式转换为日志名字

完整范围：[order_launch.cc:L113–L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L113-L129)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S66 / L113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L113) | <code>static const char* GetOrderLaunchModeName(OrderLaunchMode mode)</code> | 声明GetOrderLaunchModeName接口：将保序模式转换为日志名字。 |
| [S67 / L115](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L115) | <code>{</code> | 开始GetOrderLaunchModeName的函数体。 |
| [S68 / L117](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L117) | <code>    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {</code> | ACL图捕获使用Aclgraph模式日志名字。 |
| [S69 / L119](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L119) | <code>        return &quot;Aclgraph&quot;;</code> | 返回Aclgraph字符串。 |
| [S70 / L121](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L121) | <code>    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {</code> | GE图执行使用GE模式日志名字。 |
| [S71 / L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L123) | <code>        return &quot;GE&quot;;</code> | 返回GE字符串。 |
| [S72 / L125](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L125) | <code>    }</code> | 结束条件} else if (mode == OrderLaunchMode::ORDER_LAUNCH_GE)。 |
| [S73 / L127](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L127) | <code>    return &quot;Opbase&quot;;</code> | 其余模式使用Opbase名字，本例在此返回。 |
| [S74 / L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L129) | <code>}</code> | 结束GetOrderLaunchModeName函数体。 |


## 66. GetOrderLaunchHostThreadType

按OPBASE/GE/ACLGRAPH选择Host专用保序线程类型

完整范围：[order_launch.cc:L132–L148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L132-L148)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S76 / L132](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L132) | <code>static HcclDedicatedThreadType GetOrderLaunchHostThreadType(OrderLaunchMode mode)</code> | 声明GetOrderLaunchHostThreadType接口：按OPBASE/GE/ACLGRAPH选择Host专用保序线程类型。 |
| [S77 / L134](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L134) | <code>{</code> | 开始GetOrderLaunchHostThreadType的函数体。 |
| [S78 / L136](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L136) | <code>    if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {</code> | GE模式选择GE专用Host保序线程。 |
| [S79 / L138](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L138) | <code>        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE;</code> | 返回AICPU_ORDER_LAUNCH_GE线程类型。 |
| [S80 / L140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L140) | <code>    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {</code> | ACLGRAPH模式选择ACL图专用Host保序线程。 |
| [S81 / L142](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L142) | <code>        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH;</code> | 返回AICPU_ORDER_LAUNCH_ACLGRAPH线程类型。 |
| [S82 / L144](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L144) | <code>    }</code> | 结束条件} else if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。 |
| [S83 / L146](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L146) | <code>    return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE;</code> | 其余模式选择OPBASE专用Host保序线程。 |
| [S84 / L148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L148) | <code>}</code> | 结束GetOrderLaunchHostThreadType函数体。 |


## 67. OpLaunchGetOrderStreams

取得保序和展开两个ACL stream并检查非空

完整范围：[order_launch.cc:L151–L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L151-L175)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S86 / L151](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L151) | <code>static HcclResult OpLaunchGetOrderStreams(</code> | 声明OpLaunchGetOrderStreams接口：取得保序和展开两个ACL stream并检查非空。 |
| [S87 / L153](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L153) | <code>    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, aclrtStream&amp; hostOrderStream,</code> | 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。 |
| [S88 / L155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L155) | <code>    aclrtStream&amp; unfoldStream)</code> | 函数参数包含aclrtStream&amp; unfoldStream，本行延续接口声明。 |
| [S89 / L157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L157) | <code>{</code> | 开始OpLaunchGetOrderStreams的函数体。 |
| [S90 / L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L159) | <code>    CHK_RET(OpLaunchGetHostOrderStream(hostOrderThread, hostOrderStream));</code> | 查询Host保序线程对应ACL stream。 |
| [S91 / L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L161) | <code>    CHK_RET(OpLaunchGetUnfoldStream(comm, unfoldThread, unfoldStream));</code> | 查询Host展开线程对应ACL stream。 |
| [S92 / L163](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L163) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S93 / L165](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L165) | <code>        hostOrderStream == nullptr &#124;&#124; unfoldStream == nullptr,</code> | 任一查询流为空时返回运行时错误。 |
| [S94 / L167](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L167) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录OpLaunchGetOrderStreams当前阶段和相关参数。 |
| [S95 / L169](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L169) | <code>            &quot;[%s] failed to get hostOrderStream[%p] or unfoldStream[%p]&quot;, __func__, hostOrderStream, unfoldStream),</code> | 日志输出查询到的两条流地址，方便定位空流错误。 |
| [S96 / L171](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L171) | <code>        HCCL_E_RUNTIME);</code> | 提供上述日志的实参：HCCL_E_RUNTIME。 |
| [S97 / L173](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L173) | <code>    return HCCL_SUCCESS;</code> | 取得保序和展开两个ACL stream并检查非空处理完成，返回成功。 |
| [S98 / L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L175) | <code>}</code> | 结束OpLaunchGetOrderStreams函数体。 |


## 68. AclgraphOrderLaunchEventToOrderStream

ACL图第一阶段：展开流RecordEvent，Host保序流等待

完整范围：[order_launch.cc:L178–L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L178-L209)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S100 / L178](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L178) | <code>static HcclResult AclgraphOrderLaunchEventToOrderStream(</code> | 声明AclgraphOrderLaunchEventToOrderStream接口：ACL图第一阶段：展开流RecordEvent，Host保序流等待。 |
| [S101 / L180](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L180) | <code>    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, HcclRtEvent event)</code> | 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。 |
| [S102 / L182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L182) | <code>{</code> | 开始AclgraphOrderLaunchEventToOrderStream的函数体。 |
| [S103 / L184](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L184) | <code>    aclrtStream hostOrderStream = nullptr;</code> | 初始化ACL图第一阶段Host保序流地址。 |
| [S104 / L186](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L186) | <code>    aclrtStream unfoldStream = nullptr;</code> | 初始化ACL图第一阶段展开流地址。 |
| [S105 / L188](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L188) | <code>    CHK_RET(OpLaunchGetOrderStreams(comm, hostOrderThread, unfoldThread, hostOrderStream, unfoldStream));</code> | 同时获取Host保序流和展开流，任一为空则停止。 |
| [S107 / L191](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L191) | <code>    aclError retEvent = aclrtRecordEvent(event, unfoldStream);</code> | 在展开流记录第一阶段event。 |
| [S108 / L193](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L193) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S109 / L195](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L195) | <code>        retEvent != ACL_SUCCESS, HCCL_ERROR(&quot;[%s]aclrtRecordEvent failed, ret[%d]&quot;, __func__, retEvent),</code> | RecordEvent失败时打印ACL返回码并触发错误返回。 |
| [S110 / L197](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L197) | <code>        HCCL_E_RUNTIME);</code> | 补足错误检查宏返回值：HCCL_E_RUNTIME。 |
| [S111 / L199](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L199) | <code>    retEvent = aclrtStreamWaitEvent(hostOrderStream, event);</code> | 在Host保序流排入event等待，确保展开流前序步骤先完成。 |
| [S112 / L201](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L201) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S113 / L203](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L203) | <code>        retEvent != ACL_SUCCESS, HCCL_ERROR(&quot;[%s]aclrtStreamWaitEvent failed, ret[%d]&quot;, __func__, retEvent),</code> | WaitEvent失败时打印ACL返回码并触发错误返回。 |
| [S114 / L205](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L205) | <code>        HCCL_E_RUNTIME);</code> | 补足错误检查宏返回值：HCCL_E_RUNTIME。 |
| [S115 / L207](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L207) | <code>    return HCCL_SUCCESS;</code> | ACL图第一阶段：展开流RecordEvent，Host保序流等待处理完成，返回成功。 |
| [S116 / L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L209) | <code>}</code> | 结束AclgraphOrderLaunchEventToOrderStream函数体。 |


## 69. AclgraphOrderLaunchEventToKernelStream

ACL图第二阶段：Host保序流RecordEvent，展开流等待

完整范围：[order_launch.cc:L212–L243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L212-L243)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S118 / L212](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L212) | <code>static HcclResult AclgraphOrderLaunchEventToKernelStream(</code> | 声明AclgraphOrderLaunchEventToKernelStream接口：ACL图第二阶段：Host保序流RecordEvent，展开流等待。 |
| [S119 / L214](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L214) | <code>    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, HcclRtEvent event)</code> | 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。 |
| [S120 / L216](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L216) | <code>{</code> | 开始AclgraphOrderLaunchEventToKernelStream的函数体。 |
| [S121 / L218](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L218) | <code>    aclrtStream hostOrderStream = nullptr;</code> | 初始化ACL图第二阶段Host保序流地址。 |
| [S122 / L220](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L220) | <code>    aclrtStream unfoldStream = nullptr;</code> | 初始化ACL图第二阶段展开流地址。 |
| [S123 / L222](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L222) | <code>    CHK_RET(OpLaunchGetOrderStreams(comm, hostOrderThread, unfoldThread, hostOrderStream, unfoldStream));</code> | 同时获取Host保序流和展开流，任一为空则停止。 |
| [S125 / L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L225) | <code>    aclError retEvent = aclrtRecordEvent(event, hostOrderStream);</code> | 在Host保序流记录第二阶段event。 |
| [S126 / L227](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L227) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S127 / L229](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L229) | <code>        retEvent != ACL_SUCCESS, HCCL_ERROR(&quot;[%s]aclrtRecordEvent failed, ret[%d]&quot;, __func__, retEvent),</code> | RecordEvent失败时打印ACL返回码并触发错误返回。 |
| [S128 / L231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L231) | <code>        HCCL_E_RUNTIME);</code> | 补足错误检查宏返回值：HCCL_E_RUNTIME。 |
| [S129 / L233](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L233) | <code>    retEvent = aclrtStreamWaitEvent(unfoldStream, event);</code> | 在展开流排入event等待，连接Host保序流顺序。 |
| [S130 / L235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L235) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S131 / L237](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L237) | <code>        retEvent != ACL_SUCCESS, HCCL_ERROR(&quot;[%s]aclrtStreamWaitEvent failed, ret[%d]&quot;, __func__, retEvent),</code> | WaitEvent失败时打印ACL返回码并触发错误返回。 |
| [S132 / L239](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L239) | <code>        HCCL_E_RUNTIME);</code> | 补足错误检查宏返回值：HCCL_E_RUNTIME。 |
| [S133 / L241](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L241) | <code>    return HCCL_SUCCESS;</code> | ACL图第二阶段：Host保序流RecordEvent，展开流等待处理完成，返回成功。 |
| [S134 / L243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L243) | <code>}</code> | 结束AclgraphOrderLaunchEventToKernelStream函数体。 |


## 70. HcclOrderLaunchToOrderStream

第一阶段：取得Host/Device保序线程，Host保序流通知展开流

完整范围：[order_launch.cc:L262–L374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L262-L374)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S152 / L262](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L262) | <code>HcclResult HcclOrderLaunchToOrderStream(</code> | 声明HcclOrderLaunchToOrderStream接口：第一阶段：取得Host/Device保序线程，Host保序流通知展开流。 |
| [S153 / L264](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L264) | <code>    HcclComm comm, OpParam&amp; param, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, OrderLaunchMode mode,</code> | 函数参数包含通信域句柄、算子参数、Host展开线程句柄，本行延续接口声明。 |
| [S154 / L266](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L266) | <code>    HcclRtEvent event)</code> | 函数参数包含HcclRtEvent event，本行延续接口声明。 |
| [S155 / L268](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L268) | <code>{</code> | 开始HcclOrderLaunchToOrderStream的函数体。 |
| [S156 / L270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L270) | <code>    const char* modeName = GetOrderLaunchModeName(mode);</code> | 取得当前OPBASE/GE/ACLGRAPH的日志名字。 |
| [S157 / L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L272) | <code>    HcclDedicatedThreadType hostThreadType = GetOrderLaunchHostThreadType(mode);</code> | 选择该模式的专用Host保序线程类型。 |
| [S160 / L276](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L276) | <code>    ThreadHandle hostOrderThread;</code> | 声明Host保序线程句柄。 |
| [S161 / L278](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L278) | <code>    ThreadHandle exportHostOrderThread;</code> | 声明Host保序线程导出到Device的句柄。 |
| [S162 / L280](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L280) | <code>    if (!HcommIsSupportHcclDedicatedThreadAcquire()) {</code> | 运行时没有专用线程申请能力时跳过保序。 |
| [S163 / L282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L282) | <code>        param.exportHostOrderThread = 0;</code> | 把参数中的Host保序导出句柄清零，Device不发保序通知。 |
| [S164 / L284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L284) | <code>        param.deviceOrderThread = 0;</code> | 把Device保序线程句柄清零，Device入口跳过保序通知。 |
| [S165 / L286](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L286) | <code>        HCCL_WARNING(&quot;[%s]. HcclDedicatedThreadAcquire not supported, %s OrderLaunch is skipped.&quot;, __func__, modeName);</code> | 输出警告日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。 |
| [S166 / L288](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L288) | <code>        return HCCL_SUCCESS;</code> | 专用线程能力缺失时当前保序阶段成功跳过。 |
| [S167 / L290](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L290) | <code>    }</code> | 结束条件if (!HcommIsSupportHcclDedicatedThreadAcquire())。 |
| [S168 / L292](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L292) | <code>    CHK_RET(HcclDedicatedThreadAcquire(comm, hostThreadType, HOST_ORDER_THREAD_NOTIFY_NUM, &amp;hostOrderThread));</code> | 获取当前模式的专用Host保序线程及其通知资源。 |
| [S169 / L294](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L294) | <code>    HCCL_INFO(</code> | 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。 |
| [S170 / L296](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L296) | <code>        &quot;[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]&quot;, __func__, modeName, hostOrderThread);</code> | 补充日志格式：[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]&quot;, __func__, modeName, hostOrderThread)。 |
| [S171 / L298](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L298) | <code>    if (hostOrderThread == 0) {</code> | 专用线程返回0表示无需该保序链，按源码规则成功跳过。 |
| [S172 / L300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L300) | <code>        param.exportHostOrderThread = 0;</code> | 清除Host保序线程导出句柄。 |
| [S173 / L302](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L302) | <code>        param.deviceOrderThread = 0;</code> | 清除Device保序线程句柄。 |
| [S174 / L304](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L304) | <code>        HCCL_INFO(</code> | 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。 |
| [S175 / L306](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L306) | <code>            &quot;[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.&quot;, __func__,</code> | 补充日志格式：[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.&quot;, __func__。 |
| [S176 / L308](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L308) | <code>            modeName);</code> | 提供上述日志的实参：modeName。 |
| [S177 / L310](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L310) | <code>        return HCCL_SUCCESS;</code> | 无需保序时成功返回，后面不建立通知链。 |
| [S178 / L312](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L312) | <code>    }</code> | 结束条件if (hostOrderThread == 0)。 |
| [S181 / L316](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L316) | <code>    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &amp;hostOrderThread, COMM_ENGINE_AICPU_TS, &amp;exportHostOrderThread));</code> | 把Host保序线程导出到AICPU_TS，使Device入口可发通知。 |
| [S182 / L318](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L318) | <code>    param.exportHostOrderThread = exportHostOrderThread;</code> | 把导出后的句柄放入OpParam供Device kernel读取。 |
| [S183 / L320](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L320) | <code>    HCCL_INFO(</code> | 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。 |
| [S184 / L322](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L322) | <code>        &quot;[%s]. %s After HcclThreadExportToCommEngine hostOrderThread [0x%llx], exportHostOrderThread[0x%llx]&quot;, __func__,</code> | 补充日志格式：[%s]. %s After HcclThreadExportToCommEngine hostOrderThread [0x%llx], exportHostOrderThread[0x%llx]&quot;, __func__。 |
| [S185 / L324](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L324) | <code>        modeName, hostOrderThread, exportHostOrderThread);</code> | 提供上述日志的实参：modeName, hostOrderThread, exportHostOrderThread。 |
| [S188 / L328](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L328) | <code>    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {</code> | 仅ACLGRAPH模式用第一阶段event建立额外流依赖。 |
| [S189 / L330](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L330) | <code>        CHK_RET(AclgraphOrderLaunchEventToOrderStream(comm, hostOrderThread, unfoldThread, event));</code> | 展开流RecordEvent，Host保序流等待同一event。 |
| [S190 / L332](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L332) | <code>    }</code> | 结束条件if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。 |
| [S193 / L336](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L336) | <code>    ThreadHandle deviceOrderThread;</code> | 声明Device专用保序线程句柄。 |
| [S194 / L338](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L338) | <code>    CHK_RET(HcclDedicatedThreadAcquire(</code> | 开始取得Device专用保序线程。 |
| [S195 / L340](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L340) | <code>        comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, DEVICE_ORDER_THREAD_NOTIFY_NUM, &amp;deviceOrderThread));</code> | 请求AICPU_ORDER_LAUNCH_DEVICE线程及其通知容量。 |
| [S196 / L342](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L342) | <code>    if (deviceOrderThread == 0) {</code> | Device专用保序线程句柄为0时跳过后续保序。 |
| [S197 / L344](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L344) | <code>        param.exportHostOrderThread = 0;</code> | 清除已经写入的Host导出保序句柄。 |
| [S198 / L346](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L346) | <code>        param.deviceOrderThread = 0;</code> | 清除Device保序句柄。 |
| [S199 / L348](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L348) | <code>        HCCL_INFO(</code> | 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。 |
| [S200 / L350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L350) | <code>            &quot;[%s]. HcclDedicatedThreadAcquire unable to obtain deviceOrderThread, %s OrderLaunch is not Required.&quot;,</code> | 补充日志格式：[%s]. HcclDedicatedThreadAcquire unable to obtain deviceOrderThread, %s OrderLaunch is not Required.。 |
| [S201 / L352](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L352) | <code>            __func__, modeName);</code> | 提供上述日志的实参：__func__, modeName。 |
| [S202 / L354](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L354) | <code>        return HCCL_SUCCESS;</code> | 无Device保序线程时成功跳过当前保序阶段。 |
| [S203 / L356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L356) | <code>    }</code> | 结束条件if (deviceOrderThread == 0)。 |
| [S204 / L358](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L358) | <code>    param.deviceOrderThread = deviceOrderThread;</code> | 保存Device专用保序线程句柄到OpParam。 |
| [S205 / L360](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L360) | <code>    HCCL_INFO(</code> | 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。 |
| [S206 / L362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L362) | <code>        &quot;[%s]. %s After HcclDedicatedThreadAcquire deviceOrderThread [0x%llx]&quot;, __func__, modeName, deviceOrderThread);</code> | 补充日志格式：[%s]. %s After HcclDedicatedThreadAcquire deviceOrderThread [0x%llx]&quot;, __func__, modeName, deviceOrderThread)。 |
| [S209 / L366](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L366) | <code>    CHK_RET(static_cast&lt;HcclResult&gt;(HcommThreadNotifyRecordOnThread(hostOrderThread, unfoldThread, notifyIdx)));</code> | 在Host保序线程向Host展开线程记录第一阶段通知。 |
| [S210 / L368](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L368) | <code>    CHK_RET(static_cast&lt;HcclResult&gt;(HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, timeout)));</code> | Host展开线程等待第一阶段通知，保护后续kernel发射顺序。 |
| [S211 / L370](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L370) | <code>    HCCL_INFO(&quot;[%s]. %s OrderLaunch Phase1 Success, timeout[%u].&quot;, __func__, modeName, timeout);</code> | 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。 |
| [S212 / L372](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L372) | <code>    return HCCL_SUCCESS;</code> | 第一阶段：取得Host/Device保序线程，Host保序流通知展开流处理完成，返回成功。 |
| [S213 / L374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L374) | <code>}</code> | 结束HcclOrderLaunchToOrderStream函数体。 |


## 71. HcclOrderLaunchToKernelStream

第二阶段：Host保序流等待Device入口已进入展开的通知

完整范围：[order_launch.cc:L391–L450](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L391-L450)；文件：`hccl/src/ops/op_common/order_launch.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S229 / L391](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L391) | <code>HcclResult HcclOrderLaunchToKernelStream(</code> | 声明HcclOrderLaunchToKernelStream接口：第二阶段：Host保序流等待Device入口已进入展开的通知。 |
| [S230 / L393](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L393) | <code>    HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, OrderLaunchMode mode, HcclRtEvent event)</code> | 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。 |
| [S231 / L395](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L395) | <code>{</code> | 开始HcclOrderLaunchToKernelStream的函数体。 |
| [S232 / L397](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L397) | <code>    const char* modeName = GetOrderLaunchModeName(mode);</code> | 取得本次模式日志名字。 |
| [S233 / L399](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L399) | <code>    HcclDedicatedThreadType hostThreadType = GetOrderLaunchHostThreadType(mode);</code> | 选择OPBASE/GE/ACLGRAPH对应Host保序线程类型。 |
| [S234 / L401](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L401) | <code>    HCCL_INFO(&quot;%s OrderLaunch Phase2 Start, Comm[%p], timeout[%u].&quot;, modeName, comm, timeout);</code> | 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。 |
| [S237 / L405](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L405) | <code>    ThreadHandle hostOrderThread;</code> | 声明第二阶段Host保序线程句柄。 |
| [S238 / L407](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L407) | <code>    if (!HcommIsSupportHcclDedicatedThreadAcquire()) {</code> | 没有专用线程申请能力时第二阶段成功跳过。 |
| [S239 / L409](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L409) | <code>        HCCL_WARNING(&quot;[%s]. HcclDedicatedThreadAcquire not supported, %s OrderLaunch is skipped.&quot;, __func__, modeName);</code> | 输出警告日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。 |
| [S240 / L411](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L411) | <code>        return HCCL_SUCCESS;</code> | 缺少能力时返回成功，不执行后续保序通知等待。 |
| [S241 / L413](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L413) | <code>    }</code> | 结束条件if (!HcommIsSupportHcclDedicatedThreadAcquire())。 |
| [S242 / L415](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L415) | <code>    CHK_RET(HcclDedicatedThreadAcquire(comm, hostThreadType, HOST_ORDER_THREAD_NOTIFY_NUM, &amp;hostOrderThread));</code> | 取得第一阶段使用的同类型专用Host保序线程。 |
| [S243 / L417](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L417) | <code>    HCCL_INFO(</code> | 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。 |
| [S244 / L419](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L419) | <code>        &quot;[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]&quot;, __func__, modeName, hostOrderThread);</code> | 补充日志格式：[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]&quot;, __func__, modeName, hostOrderThread)。 |
| [S245 / L421](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L421) | <code>    if (hostOrderThread == 0) {</code> | 取得的Host保序线程句柄为0表示无需保序。 |
| [S246 / L423](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L423) | <code>        HCCL_INFO(</code> | 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。 |
| [S247 / L425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L425) | <code>            &quot;[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.&quot;, __func__,</code> | 补充日志格式：[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.&quot;, __func__。 |
| [S248 / L427](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L427) | <code>            modeName);</code> | 提供上述日志的实参：modeName。 |
| [S249 / L429](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L429) | <code>        return HCCL_SUCCESS;</code> | 无需保序时成功返回。 |
| [S250 / L431](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L431) | <code>    }</code> | 结束条件if (hostOrderThread == 0)。 |
| [S253 / L435](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L435) | <code>    CHK_RET(static_cast&lt;HcclResult&gt;(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, timeout)));</code> | Host保序线程等待Device入口发来的第二阶段通知。 |
| [S256 / L439](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L439) | <code>    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {</code> | 仅ACLGRAPH模式额外建立Host保序流到展开流的event依赖。 |
| [S257 / L441](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L441) | <code>        CHK_RET(AclgraphOrderLaunchEventToKernelStream(comm, hostOrderThread, unfoldThread, event));</code> | Host保序流RecordEvent，展开流等待该event。 |
| [S258 / L443](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L443) | <code>    }</code> | 结束条件if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。 |
| [S260 / L446](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L446) | <code>    HCCL_INFO(&quot;[%s]. %s OrderLaunch Phase2 Success.timeout[%u].&quot;, __func__, modeName, timeout);</code> | 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。 |
| [S261 / L448](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L448) | <code>    return HCCL_SUCCESS;</code> | 第二阶段：Host保序流等待Device入口已进入展开的通知处理完成，返回成功。 |
| [S262 / L450](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L450) | <code>}</code> | 结束HcclOrderLaunchToKernelStream函数体。 |
