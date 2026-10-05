# AllReduce逐行对照：公共调度、资源上下文与设备入口

[返回阅读指南](../READING_GUIDE.zh-CN.md)。S为审读快照行号；L为带本次逐行注释的源码行号。每个L链接定位到固定源码提交；长语句按物理行分别说明。空行及原注释不重复注释。

审读快照：`f8af6a36831195a8440de6ec6183856bb72af907`；源码提交：`824a8a80731bd66ef6eb78891b9aec0acfca281c`。

[返回本阶段函数导航](02-dispatch-device.zh-CN.md)。第2/3页。

## 29. HcclExecOp

执行分发：复用/创建算法资源，关联Host与Device线程，按引擎发射或Host编排

完整范围：[op_common.cc:L848–L1154](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L848-L1154)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S798 / L870](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L870)：编译插件且pluginSelected；ExecutePluginAlgorithm

- [S810 / L891](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L891)：历史fallback上下文命中；HcclExecOp递归调用缓存回退算法

- [S866 / L988](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L988)：资源返回HCCL_E_UNAVAIL；FallbackOp

- [S902 / L1047](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1047)：AICPU_TS/CPU；HcclAicpuKernelEntranceLaunch

- [S914 / L1067](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1067)：AIV；HcclAivKernelEntranceLaunch -> ExecuteAivCacheLogic

- [S922 / L1082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1082)：CCU；executor::Orchestrate

- [S949 / L1130](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1130)：其它engine；executor::Orchestrate



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S783 / L848](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L848) | <code>HcclResult HcclExecOp(</code> | 声明HcclExecOp接口：执行分发：复用/创建算法资源，关联Host与Device线程，按引擎发射或Host编排。 |
| [S784 / L850](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L850) | <code>    HcclComm comm, OpParam&amp; param, std::unique_ptr&lt;TopoInfoWithNetLayerDetails&gt;&amp; topoInfo, std::string&amp; algName,</code> | 函数参数包含通信域句柄、算子参数、物理拓扑对象、算法名字，本行延续接口声明。 |
| [S785 / L852](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L852) | <code>    const ResPackGraphMode&amp; resPack)</code> | 函数参数包含图模式资源包，本行延续接口声明。 |
| [S786 / L854](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L854) | <code>{</code> | 开始HcclExecOp的函数体。 |
| [S787 / L856](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L856) | <code>    uint64_t beginTime = HcommGetProfilingSysCycleTime();</code> | 记录本次Host执行入口的profiling起始时间。 |
| [S788 / L858](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L858) | <code>    HCCL_INFO(&quot;[HcclExecOp]Start to execute HcclExecOp. HcommGetProfilingSysCycleTime[%llu us]&quot;, beginTime);</code> | 输出运行日志，记录HcclExecOp当前阶段和相关参数。 |
| [S797 / L868](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L868) | <code>#ifdef HCCL_ALGO_PLUGIN_ENABLE</code> | 此段仅在HCCL_ALGO_PLUGIN_ENABLE启用的构建中编译。 |
| [S798 / L870](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L870) | <code>    if (param.pluginSelected) {</code> | 已选中插件算法时走插件执行分支。 |
| [S799 / L872](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L872) | <code>        CHK_RET(ExecutePluginAlgorithm(comm, param, topoInfo.get(), algName));</code> | 调用插件编排函数，失败直接返回而不继续内置算法。 |
| [S800 / L874](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L874) | <code>        CHK_RET(ReportOpProfilingInfo(comm, param.opType, beginTime));</code> | 记录插件算子的profiling统计。 |
| [S801 / L876](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L876) | <code>        return HCCL_SUCCESS;</code> | 插件执行分支成功后直接返回。 |
| [S802 / L878](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L878) | <code>    }</code> | 结束条件if (param.pluginSelected)。 |
| [S803 / L880](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L880) | <code>#endif</code> | 结束上述编译期条件控制的源码范围。 |
| [S807 / L885](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L885) | <code>    void* fallbackCtx = nullptr;</code> | 初始化历史回退上下文地址为空。 |
| [S808 / L887](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L887) | <code>    uint64_t fallbackCtxSize = 0;</code> | 初始化历史回退上下文长度为零。 |
| [S809 / L889](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L889) | <code>    CHK_RET(SetOpParamFallbackTag(param, algName));</code> | 根据原算法名构造历史回退缓存tag。 |
| [S810 / L891](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L891) | <code>    if (HcclEngineCtxGet(comm, param.fallbackTag, param.engine, &amp;fallbackCtx, &amp;fallbackCtxSize) == HCCL_SUCCESS) {</code> | 历史回退上下文命中时直接恢复上次协商后的算法配置。 |
| [S811 / L893](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L893) | <code>        HCCL_INFO(&quot;[HcclExecOp] Engine ctx exists, try to fallback.&quot;);</code> | 输出运行日志，记录HcclExecOp当前阶段和相关参数。 |
| [S812 / L895](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L895) | <code>        auto* ctxData = static_cast&lt;FallbackCtxData*&gt;(fallbackCtx);</code> | 把上下文地址解释为FallbackCtxData对象。 |
| [S813 / L897](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L897) | <code>        std::string newAlgName = ctxData-&gt;algName;</code> | 从缓存取出已经回退到的算法名字。 |
| [S814 / L899](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L899) | <code>        HCCL_INFO(</code> | 输出运行日志，记录HcclExecOp当前阶段和相关参数。 |
| [S815 / L901](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L901) | <code>            &quot;[HcclExecOp] Cached algo[%s], config[%u].&quot;, newAlgName.c_str(),</code> | 补充日志格式：[HcclExecOp] Cached algo[%s], config[%u].&quot;, newAlgName.c_str()。 |
| [S816 / L903](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L903) | <code>            static_cast&lt;uint32_t&gt;(ctxData-&gt;opExecuteConfig));</code> | 提供上述日志的实参：static_cast&lt;uint32_t&gt;(ctxData-&gt;opExecuteConfig。 |
| [S817 / L905](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L905) | <code>        param.opExecuteConfig = ctxData-&gt;opExecuteConfig;</code> | 恢复历史回退选择的执行配置。 |
| [S818 / L907](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L907) | <code>        CHK_RET(SetCommEngine(param));</code> | 按恢复的执行配置重新设置引擎。 |
| [S819 / L909](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L909) | <code>        CHK_RET(SetOpParamAlgTag(param, newAlgName));</code> | 按回退算法名字重新构造算法tag。 |
| [S820 / L911](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L911) | <code>        CHK_RET(HcclExecOp(comm, param, topoInfo, newAlgName, resPack));</code> | 递归进入HcclExecOp执行缓存的回退算法。 |
| [S821 / L913](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L913) | <code>        return HCCL_SUCCESS;</code> | 回退算法执行成功后结束当前调用层。 |
| [S822 / L915](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L915) | <code>    }</code> | 结束条件if (HcclEngineCtxGet(comm, param.fallbackTag, param.engine, &amp;fallbackCtx, &amp;fallbackCtxSize) == HCCL_SUCCESS)。 |
| [S825 / L919](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L919) | <code>    int result = sprintf_s(param.algName, sizeof(param.algName), &quot;%s&quot;, algName.c_str());</code> | 把选择器返回的算法名字复制到设备参数固定字符串区。 |
| [S826 / L921](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L921) | <code>    if (result &lt;= 0) {</code> | sprintf_s未成功写入名字时终止执行。 |
| [S827 / L923](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L923) | <code>        HCCL_ERROR(&quot;failed to fill param.algName&quot;);</code> | 输出错误日志，记录HcclExecOp当前阶段和相关参数。 |
| [S828 / L925](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L925) | <code>        return HCCL_E_INTERNAL;</code> | 算法名字写入失败返回内部错误。 |
| [S829 / L927](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L927) | <code>    }</code> | 结束条件if (result &lt;= 0)。 |
| [S831 / L930](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L930) | <code>    param.hcclComm = comm;</code> | 保存后续资源调用和Device展开使用的通信域句柄。 |
| [S832 / L932](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L932) | <code>    bool isOpBase = param.opMode == OpMode::OPBASE;</code> | 判断当前是否是单算子OPBASE模式。 |
| [S833 / L934](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L934) | <code>    const char* opModeStr = isOpBase ? &quot;_opbase&quot; : &quot;_offload&quot;;</code> | 根据执行模式选择通信域资源关联后缀。 |
| [S834 / L936](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L936) | <code>    auto ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), &quot;%s_%s&quot;, param.commName, opModeStr);</code> | 用通信域名和模式后缀构造commModeTag。 |
| [S835 / L938](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L938) | <code>    if (ret &lt;= 0) {</code> | 检查commModeTag字符串格式化是否成功。 |
| [S836 / L940](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L940) | <code>        HCCL_ERROR(&quot;[%s] failed to fill param.commModeTag&quot;, __func__);</code> | 输出错误日志，记录HcclExecOp当前阶段和相关参数。 |
| [S837 / L942](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L942) | <code>        return HCCL_E_INTERNAL;</code> | 关联tag生成失败返回内部错误。 |
| [S838 / L944](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L944) | <code>    }</code> | 结束条件if (ret &lt;= 0)。 |
| [S841 / L948](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L948) | <code>    std::unique_ptr&lt;InsCollAlgBase&gt; executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);</code> | 按ALLREDUCE和AicpuAllReduceSoleMeshOneShot等名字查工厂，新建具体executor。 |
| [S842 / L950](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L950) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S843 / L952](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L952) | <code>        executor.get() == nullptr, HCCL_ERROR(&quot;Fail to find executor for algName[%s]&quot;, algName.c_str()), HCCL_E_PARA);</code> | 找不到工厂创建出的executor时记录错误并返回参数错误。 |
| [S846 / L956](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L956) | <code>    std::unique_ptr&lt;AlgResourceCtxSerializable&gt; resCtxHost = std::make_unique&lt;AlgResourceCtxSerializable&gt;();</code> | 创建Host资源描述对象，稍后记录拓扑、CCL、线程及通道。 |
| [S848 / L959](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L959) | <code>    resCtxHost-&gt;isHcommBatchTransferOnThreadSupported = HcommIsSupportHcommBatchTransferOnThread();</code> | 保存批传输接口是否可用，Device wrapper据此选择批传输或逐片原语。 |
| [S850 / L962](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L962) | <code>    void* resCtxSequence = nullptr;</code> | 初始化Device资源上下文返回地址为空。 |
| [S851 / L964](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L964) | <code>    bool isResourceReused = false;</code> | 初始化本次资源尚未复用标志为false。 |
| [S855 / L969](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L969) | <code>    ThreadHandle cpuTsThread{0};</code> | 初始化用户流CPU_TS线程句柄。 |
| [S856 / L971](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L971) | <code>    ThreadHandle exportedAicpuTsThread{0};</code> | 初始化用户流导出到AICPU_TS的线程句柄。 |
| [S857 / L973](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L973) | <code>    if ((param.engine == COMM_ENGINE_AICPU_TS) &#124;&#124; (param.engine == COMM_ENGINE_CPU)) {</code> | AICPU_TS/CPU路径需要建立用户流与Device主线程的通知关系。 |
| [S858 / L975](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L975) | <code>        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, CPU_TS_NOTIFY_NUM, &amp;cpuTsThread));</code> | 将用户ACL stream包装为CPU_TS线程，配置Host侧通知容量。 |
| [S860 / L978](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L978) | <code>        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &amp;cpuTsThread, COMM_ENGINE_AICPU_TS, &amp;exportedAicpuTsThread));</code> | 把用户流线程导出为Device可引用的AICPU_TS句柄。 |
| [S861 / L980](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L980) | <code>    }</code> | 结束条件if ((param.engine == COMM_ENGINE_AICPU_TS) &#124;&#124; (param.engine == COMM_ENGINE_CPU))。 |
| [S864 / L984](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L984) | <code>    auto resRet</code> | 声明资源准备返回码，下一行调用会填入结果。 |
| [S865 / L986](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L986) | <code>        = HcclGetAlgRes(comm, param, executor, topoInfo.get(), resCtxHost, &amp;resCtxSequence, isResourceReused, resPack);</code> | 复用或创建算法资源，返回Device资源序列化地址及复用标志。 |
| [S866 / L988](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L988) | <code>    if (resRet == HCCL_E_UNAVAIL) {</code> | 仅资源不可用返回码触发算法回退。 |
| [S867 / L990](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L990) | <code>        HCCL_WARNING(&quot;[HcclGetAlgRes] resource unavailable, try to fallback.&quot;);</code> | 输出警告日志，记录HcclExecOp当前阶段和相关参数。 |
| [S868 / L992](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L992) | <code>        CHK_RET(FallbackOp(comm, param, topoInfo, algName, resPack));</code> | 重新选择并执行降级算法；该函数内部负责保存回退关系。 |
| [S869 / L994](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L994) | <code>        return HCCL_SUCCESS;</code> | 回退执行成功后当前分支直接返回。 |
| [S870 / L996](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L996) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S871 / L998](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L998) | <code>        CHK_RET(resRet);</code> | 其它资源错误通过CHK_RET直接向上返回。 |
| [S872 / L1000](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1000) | <code>    }</code> | 结束条件} else。 |
| [S874 / L1003](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1003) | <code>    param.cacheValid = isResourceReused;</code> | 把资源复用结果写入Device参数，作为缓存有效性判断输入。 |
| [S876 / L1006](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1006) | <code>    HCCL_CONFIG_INFO(</code> | 输出运行日志，记录HcclExecOp当前阶段和相关参数。 |
| [S877 / L1008](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1008) | <code>        HCCL_ALG, &quot;resourceReuse[%d] algName[%s] engine[%s] algTag[%s]&quot;, static_cast&lt;int&gt;(param.cacheValid),</code> | 提供上述日志的实参，涉及算子参数、算法名字、算法关联tag。 |
| [S878 / L1010](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1010) | <code>        param.algName, ENGINE_STR_MAP.at(param.opExecuteConfig), param.algTag);</code> | 提供上述日志的实参，涉及算子参数、算法名字、算法关联tag。 |
| [S882 / L1015](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1015) | <code>    HcclDfxOpInfoCompat hcclDfxOpInfo{};</code> | 创建本次算子的维测元信息结构。 |
| [S883 / L1017](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1017) | <code>    CHK_RET(ConstructHcclDfxOpInfo(param, param.algTag, ALG_TAG_LENGTH, hcclDfxOpInfo, cpuTsThread));</code> | 把算子参数转换为维测信息，并关联用户流线程。 |
| [S884 / L1019](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1019) | <code>    param.dataCount = hcclDfxOpInfo.dataCount;</code> | 把维测结构统计出的总数据量写回param。 |
| [S885 / L1021](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1021) | <code>    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast&lt;void*&gt;(&amp;hcclDfxOpInfo)));</code> | 向通信域登记算子维测信息。 |
| [S886 / L1023](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1023) | <code>    ThreadHandle exportedCpuTsThread;</code> | 声明Device主线程导出到CPU_TS后的句柄。 |
| [S887 / L1025](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1025) | <code>    ThreadHandle mainThread;</code> | 声明Device算法主线程句柄。 |
| [S888 / L1027](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1027) | <code>    u32 notifyNumOnMainThread;</code> | 声明该Device主线程的通知槽总容量。 |
| [S889 / L1029](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1029) | <code>    if ((param.engine == COMM_ENGINE_AICPU_TS) &#124;&#124; (param.engine == COMM_ENGINE_CPU)) {</code> | AICPU_TS/CPU路径查询并导出算法主线程。 |
| [S892 / L1033](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1033) | <code>        CHK_RET(GetMainThreadInfo(comm, param, mainThread, notifyNumOnMainThread));</code> | 从Host上下文读取算法主线程及通知容量。 |
| [S894 / L1036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1036) | <code>        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &amp;mainThread, COMM_ENGINE_CPU_TS, &amp;exportedCpuTsThread));</code> | 把算法Device主线程导出到CPU_TS，供Host用户流发送通知。 |
| [S896 / L1039](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1039) | <code>        param.opThread = exportedAicpuTsThread;</code> | 把用户流导出后的Device侧句柄写进入口参数。 |
| [S897 / L1041](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1041) | <code>    }</code> | 结束条件if ((param.engine == COMM_ENGINE_AICPU_TS) &#124;&#124; (param.engine == COMM_ENGINE_CPU))。 |
| [S902 / L1047](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1047) | <code>    if ((param.engine == COMM_ENGINE_AICPU_TS) &#124;&#124; (param.engine == COMM_ENGINE_CPU)) {</code> | AICPU_TS或CPU引擎在Device入口展开算法任务。 |
| [S904 / L1050](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1050) | <code>        ThreadHandle unfoldThread;</code> | 声明用于提前展开和保序的Host展开线程句柄。 |
| [S905 / L1052](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1052) | <code>        CHK_RET(GetUnfoldThreadInfo(comm, param, unfoldThread));</code> | 读取通信域对应的Host展开线程。 |
| [S907 / L1055](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1055) | <code>        CHK_RET(CaptureSlaveStreams(comm, param.stream, {mainThread, unfoldThread}, param.isCapture));</code> | 按用户流捕获状态关联算法主线程和展开线程。 |
| [S910 / L1059](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1059) | <code>        param.aicpuCacheEnable = GetExternalInputHcclAicpuCacheEnable();</code> | 读取环境AICPU task cache开关写入入口参数。 |
| [S911 / L1061](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1061) | <code>        CHK_RET(HcclAicpuKernelEntranceLaunch(</code> | 开始调用AICPU入口发射包装器，后两行补充线程/资源参数。 |
| [S912 / L1063](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1063) | <code>            comm, param, cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread, resCtxSequence, algName,</code> | 传入Host用户线程、Device主线程导出句柄、通知容量和资源地址。 |
| [S913 / L1065](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1065) | <code>            unfoldThread));</code> | 补充展开线程句柄并结束发射调用，错误由CHK_RET返回。 |
| [S914 / L1067](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1067) | <code>    } else if (param.engine == COMM_ENGINE_AIV) {</code> | AIV执行路径先确定核数，再使用AIV缓存/发射逻辑。 |
| [S916 / L1070](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1070) | <code>        uint64_t aivBeginTime = HcommGetProfilingSysCycleTime();</code> | 记录AIV分支开始时间。 |
| [S917 / L1072](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1072) | <code>        param.resCtx = resCtxSequence;</code> | AIV资源地址写入参数，供AIV编排读取。 |
| [S918 / L1074](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1074) | <code>        AlgResourceCtxSerializable&amp; aivResCtxHost = *static_cast&lt;AlgResourceCtxSerializable*&gt;(resCtxSequence);</code> | 将AIV资源地址解释为Host资源对象引用。 |
| [S919 / L1076](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1076) | <code>        CHK_RET(HcclAivKernelEntranceLaunch(comm, param, topoInfo, aivResCtxHost));</code> | 确定AIV核数等入口设置。 |
| [S920 / L1078](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1078) | <code>        CHK_RET(ExecuteAivCacheLogic(comm, param, algName, executor, aivResCtxHost));</code> | 执行AIV缓存逻辑或实际AIV编排发射。 |
| [S921 / L1080](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1080) | <code>        CHK_RET(HcclReportAivKernel(comm, aivBeginTime));</code> | 上报AIV kernel profiling。 |
| [S922 / L1082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1082) | <code>    } else if (param.engine == COMM_ENGINE_CCU) {</code> | CCU执行路径在Host恢复资源并直接调用执行器编排。 |
| [S923 / L1084](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1084) | <code>        if (isResourceReused) {</code> | CCU资源复用时需要从序列化上下文恢复Host对象。 |
| [S926 / L1088](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1088) | <code>            char* ctx = static_cast&lt;char*&gt;(resCtxSequence);</code> | 将CCU缓存资源地址解释为字节指针。 |
| [S927 / L1090](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1090) | <code>            std::vector&lt;char&gt; seq(ctx, ctx + param.ctxSize);</code> | 按param.ctxSize构造序列化字节序列。 |
| [S928 / L1092](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1092) | <code>            resCtxHost-&gt;DeSerialize(seq);</code> | 恢复CCU资源对象，其中含线程及kernel信息。 |
| [S930 / L1095](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1095) | <code>            ThreadHandle thread;</code> | 声明当前用户流所对应的CCU主线程句柄。 |
| [S931 / L1097](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1097) | <code>            CHK_RET(HcclThreadAcquireWithStream(</code> | 从本次用户流获取CCU线程，下一行补充通知数和输出句柄。 |
| [S932 / L1099](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1099) | <code>                comm, param.engine, param.stream, resCtxHost-&gt;notifyNumOnMainThread, &amp;thread));</code> | 使用恢复对象的主线程通知数，输出本次流的线程句柄。 |
| [S933 / L1101](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1101) | <code>            if (resCtxHost-&gt;threads.empty()) {</code> | 防止复用对象没有线程时覆盖不存在的第一个元素。 |
| [S934 / L1103](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1103) | <code>                HCCL_ERROR(&quot;[%s] reused threads is empty after DeSerialize, cannot overwrite main thread.&quot;, __func__);</code> | 输出错误日志，记录HcclExecOp当前阶段和相关参数。 |
| [S935 / L1105](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1105) | <code>                return HCCL_E_UNAVAIL;</code> | 缺少复用主线程返回资源不可用。 |
| [S936 / L1107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1107) | <code>            }</code> | 结束条件if (resCtxHost-&gt;threads.empty())。 |
| [S937 / L1109](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1109) | <code>            resCtxHost-&gt;threads[0] = thread;</code> | 用本次用户流线程替换缓存对象原主线程。 |
| [S940 / L1113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1113) | <code>            if (param.opMode != OpMode::OPBASE) {</code> | OFFLOAD图模式需要按本次GE资源包覆盖复用资源。 |
| [S941 / L1115](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1115) | <code>                CHK_RET(GeReuseResource(comm, param, executor, resCtxHost, topoInfo.get(), resPack));</code> | 使用本次GE从流/临时内存更新CCU复用对象。 |
| [S942 / L1117](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1117) | <code>            }</code> | 结束条件if (param.opMode != OpMode::OPBASE)。 |
| [S943 / L1119](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1119) | <code>        }</code> | 结束条件if (isResourceReused)。 |
| [S945 / L1122](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1122) | <code>        if (resCtxHost-&gt;slaveThreadNum &gt; 0) {</code> | 有从线程的CCU算法需登记流捕获关系。 |
| [S946 / L1124](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1124) | <code>            CHK_RET(CaptureSlaveStreams(comm, param.stream, resCtxHost-&gt;threads, param.isCapture));</code> | 把所有CCU线程纳入当前用户流的捕获关系。 |
| [S947 / L1126](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1126) | <code>        }</code> | 结束条件if (resCtxHost-&gt;slaveThreadNum &gt; 0)。 |
| [S948 / L1128](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1128) | <code>        CHK_RET(executor-&gt;Orchestrate(param, *resCtxHost));</code> | 在Host调用CCU执行器编排。 |
| [S949 / L1130](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1130) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S950 / L1132](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1132) | <code>        if (isResourceReused) {</code> | 其它引擎复用时同样从序列化数据恢复Host资源对象。 |
| [S952 / L1135](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1135) | <code>            char* ctx = static_cast&lt;char*&gt;(resCtxSequence);</code> | 获取缓存上下文的字节起点。 |
| [S953 / L1137](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1137) | <code>            std::vector&lt;char&gt; seq(ctx, ctx + param.ctxSize);</code> | 按ctxSize构造缓存字节序列。 |
| [S954 / L1139](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1139) | <code>            resCtxHost-&gt;DeSerialize(seq);</code> | 反序列化为Host资源对象。 |
| [S955 / L1141](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1141) | <code>        }</code> | 结束条件if (isResourceReused)。 |
| [S956 / L1143](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1143) | <code>        CHK_RET(executor-&gt;Orchestrate(param, *resCtxHost));</code> | 其它引擎在Host调用执行器编排。 |
| [S957 / L1145](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1145) | <code>    }</code> | 结束条件} else。 |
| [S959 / L1148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1148) | <code>    CHK_RET(ReportOpProfilingInfo(comm, param.opType, beginTime));</code> | 上报整个算子Host执行阶段profiling。 |
| [S960 / L1150](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1150) | <code>    HCCL_INFO(&quot;Execute HcclExecOp success.&quot;);</code> | 输出运行日志，记录HcclExecOp当前阶段和相关参数。 |
| [S961 / L1152](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1152) | <code>    return HCCL_SUCCESS;</code> | 执行分发：复用/创建算法资源，关联Host与Device线程，按引擎发射或Host编排处理完成，返回成功。 |
| [S962 / L1154](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1154) | <code>}</code> | 结束HcclExecOp函数体。 |


## 30. HcclAicpuKernelEntranceLaunch

排入Host输入通知、执行两阶段保序、发射AICPU入口，最后排入Host等待Device结果

完整范围：[op_common.cc:L1221–L1451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1221-L1451)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1045 / L1249](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1249)：专用接口支持且Send/Recv；HcclAicpuKernelLaunch；AllReduce不走此分支

- [S1125 / L1382](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1382)：ACL图保序；HcclRtEventGuard::Create

- [S1136 / L1400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1400)：普通集合通信；AicpuKernelLaunch



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1028 / L1221](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1221) | <code>HcclResult HcclAicpuKernelEntranceLaunch(</code> | 声明HcclAicpuKernelEntranceLaunch接口：排入Host输入通知、执行两阶段保序、发射AICPU入口，最后排入Host等待Device结果。 |
| [S1029 / L1223](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1223) | <code>    HcclComm comm, OpParam&amp; param, ThreadHandle cpuTsThread, ThreadHandle exportedCpuTsThread,</code> | 函数参数包含通信域句柄、算子参数、用户流CPU_TS线程、Device主线程导出到Host的句柄，本行延续接口声明。 |
| [S1030 / L1225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1225) | <code>    u32 notifyNumOnMainThread, void* resCtxSequence, std::string&amp; algName, ThreadHandle unfoldThread)</code> | 函数参数包含算法名字、Device资源序列化地址输出、Host展开线程句柄、Device主线程通知容量，本行延续接口声明。 |
| [S1031 / L1227](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1227) | <code>{</code> | 开始HcclAicpuKernelEntranceLaunch的函数体。 |
| [S1032 / L1229](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1229) | <code>    HCCL_DEBUG(&quot;[HcclAicpuKernelEntranceLaunch]start to run aicpu kernel&quot;);</code> | 输出调试日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。 |
| [S1033 / L1231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1231) | <code>    (void)algName;</code> | algName参数在本包装器中未直接使用，显式标记避免未使用告警。 |
| [S1036 / L1235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1235) | <code>    param.resCtx = resCtxSequence;</code> | 将Device可读资源地址写入单一kernel参数。 |
| [S1037 / L1237](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1237) | <code>    param.aicpuRecordCpuIdx = HOST_WAIT_AICPU_NOTIFYIDX;</code> | 指定用户流等待Device完成的通知槽。 |
| [S1039 / L1240](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1240) | <code>    if (param.engine == COMM_ENGINE_CPU) {</code> | CPU/DPU引擎需先注册Host DPU kernel回调。 |
| [S1041 / L1243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1243) | <code>        CHK_RET(static_cast&lt;HcclResult&gt;(HcclTaskRegister(comm, param.algTag, HcclLaunchDPUKernel)));</code> | 将HcclLaunchDPUKernel登记到当前通信域/算法tag。 |
| [S1042 / L1245](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1245) | <code>    }</code> | 结束条件if (param.engine == COMM_ENGINE_CPU)。 |
| [S1045 / L1249](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1249) | <code>    if (HcommIsSupportHcclAicpuKernelLaunch()</code> | 专用AICPU发射能力存在时，检查是否是点对点Send/Recv。 |
| [S1046 / L1251](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1251) | <code>        &amp;&amp; (param.opType == HcclCMDType::HCCL_CMD_SEND &#124;&#124; param.opType == HcclCMDType::HCCL_CMD_RECEIVE)) {</code> | 仅Send或Receive满足此专用分支；AllReduce走后面的通用kernel入口。 |
| [S1047 / L1253](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1253) | <code>        HCCL_INFO(</code> | 输出运行日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。 |
| [S1048 / L1255](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1255) | <code>            &quot;[HcclAicpuKernelEntranceLaunch] P2P opType[%d], use HcclAicpuKernelLaunch&quot;,</code> | 补充日志格式：[HcclAicpuKernelEntranceLaunch] P2P opType[%d], use HcclAicpuKernelLaunch。 |
| [S1049 / L1257](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1257) | <code>            static_cast&lt;int&gt;(param.opType));</code> | 提供上述日志的实参，涉及算子参数。 |
| [S1052 / L1261](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1261) | <code>        HcclOpDesc opInfo;</code> | 声明点对点算子描述，普通AllReduce不使用它。 |
| [S1054 / L1264](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1264) | <code>        CHK_SAFETY_FUNC_RET(memset_s(&amp;opInfo, sizeof(HcclOpDesc), 0, sizeof(HcclOpDesc)));</code> | 清零P2P描述，安全函数失败由宏返回。 |
| [S1055 / L1266](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1266) | <code>        opInfo.opDescType = 1; // 1: P2P</code> | 设置描述类型1代表P2P。 |
| [S1057 / L1269](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1269) | <code>        std::string opNameStr = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? &quot;HcclSend&quot; : &quot;HcclRecv&quot;;</code> | 根据发送或接收命令选择P2P算子名字。 |
| [S1058 / L1271](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1271) | <code>        CHK_SAFETY_FUNC_RET(</code> | 开始安全复制P2P算子名字。 |
| [S1059 / L1273](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1273) | <code>            strncpy_s(opInfo.opName, HCCL_OP_DESC_OP_NAME_MAX_LEN, opNameStr.c_str(), opNameStr.size()));</code> | 把P2P名字写进固定长度opName区。 |
| [S1061 / L1276](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1276) | <code>        opInfo.p2p.buffer = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? param.inputPtr : param.outputPtr;</code> | Send使用输入地址，Receive使用输出地址作为P2P数据缓冲区。 |
| [S1062 / L1278](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1278) | <code>        opInfo.p2p.cmdType = param.opType;</code> | 设置P2P命令类型。 |
| [S1063 / L1280](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1280) | <code>        opInfo.p2p.dataType = param.DataDes.dataType;</code> | 设置P2P数据类型。 |
| [S1064 / L1282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1282) | <code>        opInfo.p2p.count = param.DataDes.count;</code> | 设置P2P元素数量。 |
| [S1065 / L1284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1284) | <code>        opInfo.p2p.remoteRank = param.sendRecvRemoteRank;</code> | 设置P2P对端rank。 |
| [S1066 / L1286](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1286) | <code>        aclrtStream resolvedStream;</code> | 声明解析展开线程后得到的ACL流句柄。 |
| [S1067 / L1288](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1288) | <code>        (void)GetUnfoldStream(comm, param, unfoldThread, resolvedStream);</code> | 尝试查展开流，但本行显式忽略返回值。 |
| [S1068 / L1290](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1290) | <code>        HCCL_INFO(&quot;unfoldThread[%llu]&quot;, unfoldThread);</code> | 输出运行日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。 |
| [S1070 / L1293](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1293) | <code>        opInfo.p2p.unfoldStream = resolvedStream;</code> | 将解析的展开流放进P2P描述。 |
| [S1072 / L1296](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1296) | <code>        HcclKernelFuncInfo funcInfo;</code> | 声明P2P kernel函数信息描述。 |
| [S1073 / L1298](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1298) | <code>        CHK_SAFETY_FUNC_RET(memset_s(&amp;funcInfo, sizeof(HcclKernelFuncInfo), 0, sizeof(HcclKernelFuncInfo)));</code> | 安全清零P2P kernel信息。 |
| [S1075 / L1301](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1301) | <code>        int soRet = sprintf_s(funcInfo.kernelSoName, sizeof(funcInfo.kernelSoName), &quot;libscatter_aicpu_kernel.so&quot;);</code> | 指定AICPU kernel所在动态库名字。 |
| [S1076 / L1303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1303) | <code>        CHK_PRT_RET(soRet &lt;= 0, HCCL_ERROR(&quot;[%s] failed to fill kernelSoName&quot;, __func__), HCCL_E_INTERNAL);</code> | 库名写入失败返回内部错误。 |
| [S1078 / L1306](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1306) | <code>        int funcRet = sprintf_s(funcInfo.kernelFuncName, sizeof(funcInfo.kernelFuncName), &quot;HcclLaunchP2pAicpuKernel&quot;);</code> | 指定P2P专用kernel入口名字。 |
| [S1079 / L1308](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1308) | <code>        CHK_PRT_RET(funcRet &lt;= 0, HCCL_ERROR(&quot;[%s] failed to fill kernelFuncName&quot;, __func__), HCCL_E_INTERNAL);</code> | 入口名字写入失败返回内部错误。 |
| [S1082 / L1312](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1312) | <code>        ThreadHandle aicpuThreadHandle;</code> | 声明P2P发射需要的Device主Thread句柄。 |
| [S1083 / L1314](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1314) | <code>        u32 mainNotifyNum;</code> | 声明P2P主线程通知数。 |
| [S1084 / L1316](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1316) | <code>        CHK_RET(GetMainThreadInfo(comm, param, aicpuThreadHandle, mainNotifyNum));</code> | 查询P2P算法主Thread及通知容量。 |
| [S1087 / L1320](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1320) | <code>        void* args = &amp;param;</code> | 固定入口参数地址指向本次OpParam。 |
| [S1088 / L1322](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1322) | <code>        uint32_t argSize = sizeof(OpParam) + param.varMemSize;</code> | kernel参数大小包含OpParam及尾部变长描述字节。 |
| [S1090 / L1325](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1325) | <code>        funcInfo.args = args;</code> | 写入P2P kernel参数地址。 |
| [S1091 / L1327](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1327) | <code>        funcInfo.argSize = argSize;</code> | 写入P2P kernel参数总长度。 |
| [S1093 / L1330](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1330) | <code>        HcclKernelLaunchCfg kernelLaunchCfg;</code> | 声明P2P kernel launch配置。 |
| [S1094 / L1332](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1332) | <code>        AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);</code> | 从算子执行超时派生AICPU超时。 |
| [S1095 / L1334](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1334) | <code>        u16 kernelLaunchTimeout</code> | 声明P2P kernel启动超时变量。 |
| [S1096 / L1336](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1336) | <code>            = IsHcommDefaultTimeoutSupported() ?</code> | 若运行时支持默认超时机制，采用统一派生值。 |
| [S1097 / L1338](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1338) | <code>                  timeout.kernelLaunchTimeout :</code> | 默认超时能力成立时取派生kernelLaunchTimeout。 |
| [S1098 / L1340](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1340) | <code>                  ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));</code> | 不支持默认超时时按execTimeout加启动偏移并转换为u16。 |
| [S1099 / L1342](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1342) | <code>        kernelLaunchCfg.timeOut = kernelLaunchTimeout;</code> | 把启动超时写入P2P kernel发射配置。 |
| [S1101 / L1345](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1345) | <code>        CHK_RET(HcclAicpuKernelLaunch(comm, &amp;opInfo, &amp;funcInfo, aicpuThreadHandle, param.stream, &amp;kernelLaunchCfg));</code> | 调用HCOMM专用P2P AICPU发射API。 |
| [S1103 / L1348](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1348) | <code>        HCCL_INFO(&quot;[HcclAicpuKernelEntranceLaunch] P2P launch success, algTag[%s]&quot;, param.algTag);</code> | 输出运行日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。 |
| [S1104 / L1350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1350) | <code>        return HCCL_SUCCESS;</code> | P2P成功后直接结束，不执行后面的集合通信通用路径。 |
| [S1105 / L1352](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1352) | <code>    }</code> | 结束代码块。 |
| [S1109 / L1357](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1357) | <code>    CHK_RET(static_cast&lt;HcclResult&gt;(</code> | 开始记录Host输入就绪通知，下一行补齐目标线程和槽位。 |
| [S1110 / L1359](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1359) | <code>        HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread - 1)));</code> | 在用户流线程记录通知到Device主线程的最后一个通知槽。 |
| [S1114 / L1364](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1364) | <code>    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();</code> | 读取保序操作所需执行超时。 |
| [S1117 / L1368](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1368) | <code>    OrderLaunchMode launchMode = param.isCapture ?</code> | 依据用户流是否图捕获选择ACLGRAPH保序。 |
| [S1118 / L1370](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1370) | <code>                                     OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH :</code> | 图捕获时使用ACLGRAPH事件链。 |
| [S1119 / L1372](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1372) | <code>                                     (param.opMode == OpMode::OFFLOAD ? OrderLaunchMode::ORDER_LAUNCH_GE :</code> | 非捕获时根据OFFLOAD决定GE保序或OPBASE保序。 |
| [S1120 / L1374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1374) | <code>                                                                        OrderLaunchMode::ORDER_LAUNCH_OPBASE);</code> | 普通单算子例使用ORDER_LAUNCH_OPBASE。 |
| [S1123 / L1378](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1378) | <code>    HcclRtEventGuard event0Guard;</code> | 声明第一个保序事件的RAII guard。 |
| [S1124 / L1380](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1380) | <code>    HcclRtEventGuard event1Guard;</code> | 声明第二个保序事件的RAII guard。 |
| [S1125 / L1382](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1382) | <code>    if (launchMode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {</code> | ACLGRAPH模式才创建两阶段保序事件。 |
| [S1126 / L1384](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1384) | <code>        CHK_RET(event0Guard.Create());</code> | 创建第一阶段保序事件。 |
| [S1127 / L1386](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1386) | <code>        CHK_RET(event1Guard.Create());</code> | 创建第二阶段保序事件。 |
| [S1128 / L1388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1388) | <code>    }</code> | 结束条件if (launchMode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。 |
| [S1130 / L1391](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1391) | <code>    CHK_RET(HcclOrderLaunchToOrderStream(</code> | 开始第一阶段Host保序调用。 |
| [S1131 / L1393](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1393) | <code>        comm, param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX, execTimeout, launchMode, event0Guard.Get()));</code> | 把本次展开线程纳入OrderStream顺序，传入Notify槽/事件及超时。 |
| [S1134 / L1397](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1397) | <code>    uint64_t beginTime = HcommGetProfilingSysCycleTime();</code> | 记录AICPU发射开始时间用于profiling。 |
| [S1136 / L1400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1400) | <code>    CHK_RET(AicpuKernelLaunch(comm, param, unfoldThread));</code> | 发射已加载二进制中的HcclLaunchAicpuKernel入口。 |
| [S1137 / L1402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1402) | <code>    CHK_PTR_NULL(comm);</code> | 检查通信域句柄非空。 |
| [S1140 / L1406](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1406) | <code>    CHK_RET(HcclOrderLaunchToKernelStream(</code> | 开始第二阶段kernel保序调用。 |
| [S1141 / L1408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1408) | <code>        comm, unfoldThread, HOST_ORDER_THREAD_NOTIFY_IDX, execTimeout, launchMode, event1Guard.Get()));</code> | 通过Host保序通知/事件建立本次kernel与后续算子的顺序关系。 |
| [S1143 / L1411](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1411) | <code>    std::string kernelName = &quot;HcclLaunchAicpuKernel&quot;;</code> | 记录实际发射的kernel入口名字。 |
| [S1144 / L1413](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1413) | <code>    char* kernelNameCStr = const_cast&lt;char*&gt;(kernelName.c_str());</code> | 为profiling接口取得可写类型的入口名字指针。 |
| [S1145 / L1415](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1415) | <code>    HcclResult ret = HcclReportAicpuKernel(comm, beginTime, kernelNameCStr);</code> | 上报本次AICPU kernel发射profiling信息。 |
| [S1146 / L1417](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1417) | <code>    if (ret != HCCL_SUCCESS) {</code> | profiling上报失败时直接返回其错误。 |
| [S1147 / L1419](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1419) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。 |
| [S1148 / L1421](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1421) | <code>            &quot;[HcclAicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d &quot;,</code> | 补充日志格式：[HcclAicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d。 |
| [S1149 / L1423](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1423) | <code>            beginTime, kernelNameCStr, ret);</code> | 提供上述日志的实参：beginTime, kernelNameCStr, ret。 |
| [S1150 / L1425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1425) | <code>        return ret;</code> | profiling上报失败时向上返回该错误码。 |
| [S1151 / L1427](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1427) | <code>    }</code> | 结束条件if (ret != HCCL_SUCCESS)。 |
| [S1154 / L1431](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1431) | <code>    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);</code> | 从本次执行超时派生Host通知等待超时。 |
| [S1156 / L1434](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1434) | <code>    u32 hostNotifyWaitTime = IsHcommDefaultTimeoutSupported() ?</code> | 根据运行时默认超时能力选择Host等待时长。 |
| [S1157 / L1436](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1436) | <code>                                 timeout.hostNotifyTimeout :</code> | 默认超时机制使用派生的hostNotifyTimeout。 |
| [S1158 / L1438](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1438) | <code>                                 AddAicpuTimeoutOffset(param.opConfig.execTimeout, HOST_NOTIFY_TIMEOUT_OFFSET);</code> | 不支持默认超时时使用执行超时加Host通知偏移。 |
| [S1159 / L1440](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1440) | <code>    if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {</code> | 有设置通知默认超时能力时才更新该配置。 |
| [S1160 / L1442](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1442) | <code>        CHK_RET(HcclSetNotifyWaitTimeOut(hostNotifyWaitTime));</code> | 设置本次用户流等待Device结果的默认通知超时。 |
| [S1161 / L1444](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1444) | <code>    }</code> | 结束条件if (HcommIsSupportHcommSetNotifyWaitTimeOut())。 |
| [S1162 / L1446](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1446) | <code>    CHK_RET(HcclThreadNotifyWaitOnThreadDefault(cpuTsThread, param.aicpuRecordCpuIdx, hostNotifyWaitTime));</code> | 在用户CPU_TS流中排入对Device完成通知的等待。 |
| [S1164 / L1449](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1449) | <code>    return HCCL_SUCCESS;</code> | 发射包装及流依赖建立成功后返回；返回不意味着硬件已完成通信。 |
| [S1165 / L1451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1451) | <code>}</code> | 结束HcclAicpuKernelEntranceLaunch函数体。 |


## 31. AicpuKernelLaunch

从已加载二进制取得入口，复制OpParam参数并通过ACL发射到展开流或用户流

完整范围：[op_common.cc:L1454–L1613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1454-L1613)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1229 / L1571](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1571)：无ThreadResGetInfo能力或OFFLOAD；aclrtLaunchKernelWithConfig(param.stream)

- [S1234 / L1581](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1581)：ThreadResGetInfo返回NOT_SUPPORT；aclrtLaunchKernelWithConfig(param.stream)

- [S1238 / L1589](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1589)：ThreadResGetInfo成功；aclrtLaunchKernelWithConfig(unfoldStream)



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1167 / L1454](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1454) | <code>HcclResult AicpuKernelLaunch(HcclComm comm, OpParam&amp; param, ThreadHandle unfoldThread)</code> | 声明AicpuKernelLaunch接口：从已加载二进制取得入口，复制OpParam参数并通过ACL发射到展开流或用户流。 |
| [S1168 / L1456](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1456) | <code>{</code> | 开始AicpuKernelLaunch的函数体。 |
| [S1169 / L1458](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1458) | <code>    std::string kernelName = &quot;HcclLaunchAicpuKernel&quot;;</code> | 指定本次二进制kernel入口为HcclLaunchAicpuKernel。 |
| [S1170 / L1460](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1460) | <code>    aclrtFuncHandle funcHandle;</code> | 声明ACL函数句柄输出变量。 |
| [S1171 / L1462](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1462) | <code>    aclrtArgsHandle argsHandle;</code> | 声明ACL kernel参数句柄输出变量。 |
| [S1175 / L1467](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1467) | <code>    aclError ret</code> | 声明ACL运行时调用返回码。 |
| [S1176 / L1469](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1469) | <code>        = aclrtBinaryGetFunction(g_binKernelHandle.load(std::memory_order_acquire), kernelName.c_str(), &amp;funcHandle);</code> | 原子读取已加载的二进制句柄，从中查找AICPU入口函数。 |
| [S1177 / L1471](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1471) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S1178 / L1473](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1473) | <code>        ret != ACL_SUCCESS,</code> | ACL运行时返回码非成功时触发下面的日志和错误返回。 |
| [S1179 / L1475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1475) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。 |
| [S1180 / L1477](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1477) | <code>            &quot;[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, &quot;</code> | 补充日志格式：[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed。 |
| [S1181 / L1479](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1479) | <code>            &quot;kernelName:%s&quot;,</code> | 补充日志格式：kernelName:%s。 |
| [S1182 / L1481](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1481) | <code>            ret, kernelName.c_str()),</code> | 提供上述日志的实参：ret, kernelName.c_str()),。 |
| [S1183 / L1483](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1483) | <code>        HCCL_E_RUNTIME);</code> | 提供上述日志的实参：HCCL_E_RUNTIME。 |
| [S1184 / L1485](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1485) | <code>    ret = aclrtKernelArgsInit(funcHandle, &amp;argsHandle);</code> | 为该入口函数初始化kernel参数句柄。 |
| [S1185 / L1487](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1487) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S1186 / L1489](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1489) | <code>        ret != ACL_SUCCESS,</code> | ACL运行时返回码非成功时触发下面的日志和错误返回。 |
| [S1187 / L1491](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1491) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。 |
| [S1188 / L1493](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1493) | <code>            &quot;[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, &quot;</code> | 补充日志格式：[aclrtKernelArgsInit]errNo[0x%016llx] args init failed。 |
| [S1189 / L1495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1495) | <code>            &quot;kernelName:%s&quot;,</code> | 补充日志格式：kernelName:%s。 |
| [S1190 / L1497](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1497) | <code>            ret, kernelName.c_str()),</code> | 提供上述日志的实参：ret, kernelName.c_str()),。 |
| [S1191 / L1499](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1499) | <code>        HCCL_E_RUNTIME);</code> | 提供上述日志的实参：HCCL_E_RUNTIME。 |
| [S1192 / L1501](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1501) | <code>    aclrtParamHandle paraHandle;</code> | 声明本次追加的单个参数句柄。 |
| [S1194 / L1504](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1504) | <code>    size_t paramSize = sizeof(OpParam) + param.varMemSize;</code> | 计算固定OpParam加尾部变长区域的总参数大小。 |
| [S1195 / L1506](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1506) | <code>    ret = aclrtKernelArgsAppend(argsHandle, &amp;param, paramSize, &amp;paraHandle);</code> | 把本次OpParam及相邻尾部描述追加到ACL参数对象。 |
| [S1196 / L1508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1508) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S1197 / L1510](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1510) | <code>        ret != ACL_SUCCESS,</code> | ACL运行时返回码非成功时触发下面的日志和错误返回。 |
| [S1198 / L1512](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1512) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。 |
| [S1199 / L1514](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1514) | <code>            &quot;[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append &quot;</code> | 补充日志格式：[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append。 |
| [S1200 / L1516](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1516) | <code>            &quot;size %u, kernelName:%s&quot;,</code> | 补充日志格式：size %u, kernelName:%s。 |
| [S1201 / L1518](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1518) | <code>            ret, paramSize, kernelName.c_str()),</code> | 提供上述日志的实参：ret, paramSize, kernelName.c_str()),。 |
| [S1202 / L1520](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1520) | <code>        HCCL_E_RUNTIME);</code> | 提供上述日志的实参：HCCL_E_RUNTIME。 |
| [S1203 / L1522](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1522) | <code>    ret = aclrtKernelArgsFinalize(argsHandle);</code> | 完成kernel参数构造，交运行时校验/固化参数。 |
| [S1204 / L1524](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1524) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S1205 / L1526](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1526) | <code>        ret != ACL_SUCCESS,</code> | ACL运行时返回码非成功时触发下面的日志和错误返回。 |
| [S1206 / L1528](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1528) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。 |
| [S1207 / L1530](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1530) | <code>            &quot;[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, &quot;</code> | 补充日志格式：[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed。 |
| [S1208 / L1532](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1532) | <code>            &quot;kernelName:%s&quot;,</code> | 补充日志格式：kernelName:%s。 |
| [S1209 / L1534](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1534) | <code>            ret, kernelName.c_str()),</code> | 提供上述日志的实参：ret, kernelName.c_str()),。 |
| [S1210 / L1536](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1536) | <code>        HCCL_E_RUNTIME);</code> | 提供上述日志的实参：HCCL_E_RUNTIME。 |
| [S1213 / L1540](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1540) | <code>    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);</code> | 从执行配置派生kernel启动超时。 |
| [S1214 / L1542](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1542) | <code>    u16 kernelLaunchTimeout</code> | 声明最终16位kernel启动超时。 |
| [S1215 / L1544](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1544) | <code>        = IsHcommDefaultTimeoutSupported() ?</code> | 根据HCOMM默认超时机制是否支持选择计算路径。 |
| [S1216 / L1546](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1546) | <code>              timeout.kernelLaunchTimeout :</code> | 支持时直接使用统一派生的启动超时。 |
| [S1217 / L1548](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1548) | <code>              ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));</code> | 旧机制使用执行超时加启动偏移，再限制为launch超时类型。 |
| [S1218 / L1550](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1550) | <code>    aclrtLaunchKernelCfg cfg;</code> | 声明kernel启动配置结构。 |
| [S1219 / L1552](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1552) | <code>    aclrtLaunchKernelAttr attr;</code> | 声明单个kernel启动属性结构。 |
| [S1220 / L1554](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1554) | <code>    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;</code> | 指定该属性设置kernel启动超时。 |
| [S1221 / L1556](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1556) | <code>    attr.value.timeout = kernelLaunchTimeout;</code> | 设置启动超时的实际数值。 |
| [S1222 / L1558](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1558) | <code>    cfg.numAttrs = 1;</code> | 本次启动只设置一个属性。 |
| [S1223 / L1560](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1560) | <code>    cfg.attrs = &amp;attr;</code> | 将配置属性数组指向上面构造的超时属性。 |
| [S1224 / L1562](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1562) | <code>    constexpr u32 numBlocks = 1;</code> | 本次AICPU入口按一个block发射。 |
| [S1225 / L1564](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1564) | <code>    HCCL_INFO(&quot;[AicpuKernelLaunch] unfoldThread [%lu]&quot;, unfoldThread); // 通过Thread获取展开流stream</code> | 输出运行日志，记录AicpuKernelLaunch当前阶段和相关参数。 |
| [S1226 / L1566](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1566) | <code>    void* unfoldStream = nullptr;</code> | 初始化Host展开流返回地址为空。 |
| [S1227 / L1568](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1568) | <code>    auto&amp; HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();</code> | 取得动态HCOMM函数表，用于检查展开线程查询函数。 |
| [S1229 / L1571](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1571) | <code>    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo &#124;&#124; param.opMode == OpMode::OFFLOAD) { // 不走提前展开</code> | 无资源查询函数或OFFLOAD模式时直接在用户流发射。 |
| [S1230 / L1573](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1573) | <code>        ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &amp;cfg, argsHandle, nullptr);</code> | 使用param.stream发射单block AICPU入口。 |
| [S1231 / L1575](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1575) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1232 / L1577](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1577) | <code>        HcclResult ret1</code> | 声明展开线程查询返回码。 |
| [S1233 / L1579](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1579) | <code>            = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &amp;unfoldStream);</code> | 通过可选HCOMM函数查询unfoldThread关联的ACL stream。 |
| [S1234 / L1581](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1581) | <code>        if (ret1 == HCCL_E_NOT_SUPPORT) {</code> | 查询明确返回NOT_SUPPORT时退回用户流。 |
| [S1235 / L1583](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1583) | <code>            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &amp;cfg, argsHandle, nullptr);</code> | 在用户流上发射AICPU入口作为不支持提前展开的兼容路径。 |
| [S1236 / L1585](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1585) | <code>        } else if (ret1 != HCCL_SUCCESS) {</code> | 查询其它错误直接返回。 |
| [S1237 / L1587](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1587) | <code>            return ret1;</code> | 返回展开线程查询错误，未继续发射。 |
| [S1238 / L1589](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1589) | <code>        } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1239 / L1591](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1591) | <code>            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &amp;cfg, argsHandle, nullptr);</code> | 查询成功时在unfoldStream提前展开流上发射入口。 |
| [S1240 / L1593](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1593) | <code>        }</code> | 结束条件} else。 |
| [S1241 / L1595](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1595) | <code>    }</code> | 结束条件} else。 |
| [S1242 / L1597](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1597) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S1243 / L1599](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1599) | <code>        ret != ACL_SUCCESS,</code> | 检查最终ACL发射结果。 |
| [S1244 / L1601](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1601) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。 |
| [S1245 / L1603](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1603) | <code>            &quot;[LoadCustomKernel][aclrtLaunchKernelWithConfig]&quot;</code> | 补充日志格式：[LoadCustomKernel][aclrtLaunchKernelWithConfig]。 |
| [S1246 / L1605](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1605) | <code>            &quot;errNo[0x%016llx] launch kernel failed&quot;,</code> | 补充日志格式：errNo[0x%016llx] launch kernel failed。 |
| [S1247 / L1607](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1607) | <code>            ret),</code> | 提供上述日志的实参：ret),。 |
| [S1248 / L1609](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1609) | <code>        HCCL_E_OPEN_FILE_FAILURE);</code> | ACL发射失败时返回源码指定的OPEN_FILE_FAILURE错误码。 |
| [S1249 / L1611](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1611) | <code>    return HCCL_SUCCESS;</code> | 从已加载二进制取得入口，复制OpParam参数并通过ACL发射到展开流或用户流处理完成，返回成功。 |
| [S1250 / L1613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1613) | <code>}</code> | 结束AicpuKernelLaunch函数体。 |


## 32. HcclCalcTopoInfo

读取或新建基于算子tag的Host拓扑上下文

完整范围：[op_common.cc:L1689–L1738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1689-L1738)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1333 / L1703](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1703)：NOT_FOUND/PARA；InitRankInfo -> Serialize -> EngineCtxCreate

- [S1346 / L1724](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1724)：其它返回结果落入当前恢复路径；DeSerialize



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1325 / L1689](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1689) | <code>HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam&amp; param, std::unique_ptr&lt;TopoInfoWithNetLayerDetails&gt;&amp; topoInfo)</code> | 声明HcclCalcTopoInfo接口：读取或新建基于算子tag的Host拓扑上下文。 |
| [S1326 / L1691](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1691) | <code>{</code> | 开始HcclCalcTopoInfo的函数体。 |
| [S1327 / L1693](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1693) | <code>    HCCL_INFO(&quot;[%s] HcclCalcTopoInfo start.&quot;, __func__);</code> | 输出运行日志，记录HcclCalcTopoInfo当前阶段和相关参数。 |
| [S1328 / L1695](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1695) | <code>    uint64_t size = 0;</code> | 初始化拓扑上下文查询的字节长度输出。 |
| [S1329 / L1697](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1697) | <code>    void* ctx = nullptr;</code> | 初始化拓扑上下文地址输出为空。 |
| [S1332 / L1701](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1701) | <code>    HcclResult ret = HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &amp;ctx, &amp;size);</code> | 按算子tag和CPU_TS存储引擎查询已有Host拓扑上下文。 |
| [S1333 / L1703](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1703) | <code>    if (ret == HCCL_E_NOT_FOUND &#124;&#124; ret == HCCL_E_PARA) {</code> | NOT_FOUND或PARA按未缓存拓扑处理。 |
| [S1336 / L1707](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1707) | <code>        CHK_RET(InitRankInfo(comm, topoInfo.get()));</code> | 从HCOMM rank graph初始化基础拓扑信息。 |
| [S1338 / L1710](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1710) | <code>        std::vector&lt;char&gt; seq = topoInfo-&gt;Serialize();</code> | 将初始化的拓扑对象序列化为字节数组。 |
| [S1339 / L1712](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1712) | <code>        size = seq.size();</code> | 保存拓扑序列化数据的大小。 |
| [S1341 / L1715](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1715) | <code>        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &amp;ctx));</code> | 创建CPU_TS Host上下文存储拓扑缓存。 |
| [S1342 / L1717](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1717) | <code>        CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));</code> | 安全复制拓扑序列化字节到上下文。 |
| [S1343 / L1719](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1719) | <code>        return HCCL_SUCCESS;</code> | 新建拓扑缓存成功后直接返回。 |
| [S1344 / L1721](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1721) | <code>    }</code> | 结束条件if (ret == HCCL_E_NOT_FOUND &#124;&#124; ret == HCCL_E_PARA)。 |
| [S1346 / L1724](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1724) | <code>    char* ctxTemp = reinterpret_cast&lt;char*&gt;(ctx);</code> | 将查询返回的地址解释成序列化字节起点；此处未统一检查其它ret值。 |
| [S1347 / L1726](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1726) | <code>    std::vector&lt;char&gt; seq(ctxTemp, ctxTemp + size);</code> | 按返回长度构造拓扑序列化数据视图副本。 |
| [S1348 / L1728](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1728) | <code>    TopoInfoWithNetLayerDetails topoInfoTemp;</code> | 创建临时拓扑对象接收反序列化结果。 |
| [S1349 / L1730](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1730) | <code>    topoInfoTemp.DeSerialize(seq);</code> | 恢复拓扑对象的字段。 |
| [S1350 / L1732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1732) | <code>    topoInfo = std::make_unique&lt;TopoInfoWithNetLayerDetails&gt;(std::move(topoInfoTemp));</code> | 移动构造新的topoInfo对象，供选择器继续使用。 |
| [S1351 / L1734](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1734) | <code>    HCCL_INFO(&quot;[%s] HcclCalcTopoInfo end.&quot;, __func__);</code> | 输出运行日志，记录HcclCalcTopoInfo当前阶段和相关参数。 |
| [S1352 / L1736](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1736) | <code>    return HCCL_SUCCESS;</code> | 读取或新建基于算子tag的Host拓扑上下文处理完成，返回成功。 |
| [S1353 / L1738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1738) | <code>}</code> | 结束HcclCalcTopoInfo函数体。 |


## 33. CompReqChannelWithExistChannel

增量请求只保留尚未存在的远端rank，AllReduce普通主例不进入此辅助分支

完整范围：[op_common.cc:L1741–L1773](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1741-L1773)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1355 / L1741](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1741) | <code>void CompReqChannelWithExistChannel(</code> | 声明CompReqChannelWithExistChannel接口：增量请求只保留尚未存在的远端rank，AllReduce普通主例不进入此辅助分支。 |
| [S1356 / L1743](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1743) | <code>    const std::vector&lt;std::vector&lt;ChannelInfo&gt;&gt;&amp; existChannels, AlgResourceRequest&amp; resRequest)</code> | 函数参数包含线程/通道/通知资源需求，本行延续接口声明。 |
| [S1357 / L1745](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1745) | <code>{</code> | 开始CompReqChannelWithExistChannel的函数体。 |
| [S1358 / L1747](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1747) | <code>    std::set&lt;u32&gt; existRemoteRankSet = {};</code> | 初始化已存在远端rank集合。 |
| [S1359 / L1749](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1749) | <code>    std::vector&lt;HcclChannelDesc&gt; needAllocChannelDesc;</code> | 创建缺失通道请求列表。 |
| [S1361 / L1752](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1752) | <code>    for (const ChannelInfo&amp; channel : existChannels[0]) {</code> | 遍历现有第0层通道。 |
| [S1362 / L1754](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1754) | <code>        existRemoteRankSet.insert(channel.remoteRank);</code> | 将已有通道对端rank加入集合。 |
| [S1363 / L1756](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1756) | <code>    }</code> | 结束循环for (const ChannelInfo&amp; channel : existChannels[0])。 |
| [S1365 / L1759](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1759) | <code>    for (const HcclChannelDesc&amp; channelDesc : resRequest.channels[0]) {</code> | 遍历本次第0层新请求。 |
| [S1366 / L1761](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1761) | <code>        if (existRemoteRankSet.find(channelDesc.remoteRank) == existRemoteRankSet.end()) {</code> | 对端rank不在已有集合中才保留此请求。 |
| [S1367 / L1763](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1763) | <code>            needAllocChannelDesc.push_back(channelDesc);</code> | 把缺失通道请求加入待分配列表。 |
| [S1368 / L1765](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1765) | <code>        }</code> | 结束条件if (existRemoteRankSet.find(channelDesc.remoteRank) == existRemoteRankSet.end())。 |
| [S1369 / L1767](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1767) | <code>    }</code> | 结束循环for (const HcclChannelDesc&amp; channelDesc : resRequest.channels[0])。 |
| [S1370 / L1769](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1769) | <code>    resRequest.channels = {needAllocChannelDesc};</code> | 用缺失请求替换原资源通道请求。 |
| [S1371 / L1771](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1771) | <code>    return;</code> | 增量请求筛选结束。 |
| [S1372 / L1773](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1773) | <code>}</code> | 结束CompReqChannelWithExistChannel函数体。 |


## 34. TryReuseResource

按算法tag及实际上下文存储engine查询已存在的资源

完整范围：[op_common.cc:L1779–L1838](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1779-L1838)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1383 / L1789](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1789)：BatchSendRecv OPBASE；标记增量并返回NOT_FOUND

- [S1388 / L1798](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1798)：非OPBASE且非CCU；NOT_FOUND

- [S1402 / L1822](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1822)：EngineCtxGet成功；返回已有上下文，设置cacheValid相关数据



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1377 / L1779](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1779) | <code>static HcclResult TryReuseResource(</code> | 声明TryReuseResource接口：按算法tag及实际上下文存储engine查询已存在的资源。 |
| [S1378 / L1781](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1781) | <code>    HcclComm comm, OpParam&amp; param, bool&amp; increCreateChannelFlag, void** resCtxSequence, uint64_t&amp; size,</code> | 函数参数包含通信域句柄、算子参数、Device资源序列化地址输出、上下文字节长度、增量建链标志，本行延续接口声明。 |
| [S1379 / L1783](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1783) | <code>    bool&amp; isResourceReused)</code> | 函数参数包含资源复用标志，本行延续接口声明。 |
| [S1380 / L1785](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1785) | <code>{</code> | 开始TryReuseResource的函数体。 |
| [S1383 / L1789](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1789) | <code>    if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV &amp;&amp; param.opMode == OpMode::OPBASE) {</code> | BatchSendRecv OPBASE需要增量通道，不走普通整套资源复用。 |
| [S1384 / L1791](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1791) | <code>        increCreateChannelFlag = true;</code> | 标记本次资源准备为增量建链。 |
| [S1385 / L1793](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1793) | <code>        return HCCL_E_NOT_FOUND;</code> | 用NOT_FOUND指示普通资源复用未命中。 |
| [S1386 / L1795](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1795) | <code>    }</code> | 结束条件if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV &amp;&amp; param.opMode == OpMode::OPBASE)。 |
| [S1388 / L1798](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1798) | <code>    if (param.opMode != OpMode::OPBASE &amp;&amp; param.engine != CommEngine::COMM_ENGINE_CCU) {</code> | 非OPBASE且非CCU时不允许这条普通资源复用路径。 |
| [S1389 / L1800](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1800) | <code>        return HCCL_E_NOT_FOUND;</code> | 返回未找到，交后续重新准备资源。 |
| [S1390 / L1802](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1802) | <code>    }</code> | 结束条件if (param.opMode != OpMode::OPBASE &amp;&amp; param.engine != CommEngine::COMM_ENGINE_CCU)。 |
| [S1391 / L1804](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1804) | <code>    void* ctx = nullptr;</code> | 初始化资源上下文返回地址。 |
| [S1393 / L1807](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1807) | <code>    CommEngine ctxEngine = param.engine;</code> | 默认资源存储engine等于执行engine。 |
| [S1394 / L1809](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1809) | <code>    if (param.engine == CommEngine::COMM_ENGINE_AIV) {</code> | AIV执行资源对象实际存放在Host CPU_TS上下文。 |
| [S1396 / L1812](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1812) | <code>        ctxEngine = COMM_ENGINE_CPU_TS;</code> | AIV查询转为CPU_TS存储engine。 |
| [S1397 / L1814](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1814) | <code>    } else if (param.engine == COMM_ENGINE_CPU) {</code> | CPU/DPU展开资源对象实际使用Device AICPU_TS上下文。 |
| [S1399 / L1817](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1817) | <code>        ctxEngine = COMM_ENGINE_AICPU_TS;</code> | CPU查询转为AICPU_TS存储engine。 |
| [S1400 / L1819](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1819) | <code>    }</code> | 结束条件} else if (param.engine == COMM_ENGINE_CPU)。 |
| [S1402 / L1822](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1822) | <code>    if (HcclEngineCtxGet(comm, param.algTag, ctxEngine, &amp;ctx, &amp;size) == HCCL_SUCCESS) {</code> | 按algTag和实际存储engine查询资源上下文。 |
| [S1403 / L1824](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1824) | <code>        HCCL_DEBUG(&quot;Already have context, skip create, ctxSize is %llu&quot;, size);</code> | 输出调试日志，记录TryReuseResource当前阶段和相关参数。 |
| [S1404 / L1826](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1826) | <code>        isResourceReused = true;</code> | 标记本次算法资源来自复用。 |
| [S1405 / L1828](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1828) | <code>        *resCtxSequence = ctx;</code> | 返回已有序列化资源地址给调用方。 |
| [S1406 / L1830](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1830) | <code>        param.ctxSize = size;</code> | 保存资源上下文字节长度到Device参数。 |
| [S1407 / L1832](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1832) | <code>        return HCCL_SUCCESS;</code> | 命中资源上下文后直接成功返回。 |
| [S1408 / L1834](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1834) | <code>    }</code> | 结束条件if (HcclEngineCtxGet(comm, param.algTag, ctxEngine, &amp;ctx, &amp;size) == HCCL_SUCCESS)。 |
| [S1409 / L1836](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1836) | <code>    return HCCL_E_NOT_FOUND;</code> | 资源未命中返回NOT_FOUND，由HcclGetAlgRes继续资源计算。 |
| [S1410 / L1838](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1838) | <code>}</code> | 结束TryReuseResource函数体。 |


## 35. HcclGetAlgRes

优先复用，未命中时计算拓扑层次和资源请求，分配资源并按条件核对一致性

完整范围：[op_common.cc:L1844–L1957](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1844-L1957)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1432 / L1873](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1873)：TryReuseResource成功；立即返回

- [S1452 / L1905](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1905)：资源UNAVAIL；向上传递UNAVAIL

- [S1474 / L1944](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1944)：needInconsistentCheck；FillOpExchangeInfo -> CompareOpExchangeInfos



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1415 / L1844](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1844) | <code>HcclResult HcclGetAlgRes(</code> | 声明HcclGetAlgRes接口：优先复用，未命中时计算拓扑层次和资源请求，分配资源并按条件核对一致性。 |
| [S1416 / L1846](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1846) | <code>    HcclComm comm, OpParam&amp; param, std::unique_ptr&lt;InsCollAlgBase&gt;&amp; executor, TopoInfoWithNetLayerDetails* topoInfo,</code> | 函数参数包含通信域句柄、算子参数、物理拓扑对象、具体算法执行器，本行延续接口声明。 |
| [S1417 / L1848](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1848) | <code>    std::unique_ptr&lt;AlgResourceCtxSerializable&gt;&amp; resCtxHost, void** resCtxSequence, bool&amp; isResourceReused,</code> | 函数参数包含Host资源描述对象、Device资源序列化地址输出、资源复用标志，本行延续接口声明。 |
| [S1418 / L1850](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1850) | <code>    const ResPackGraphMode&amp; resPack)</code> | 函数参数包含图模式资源包，本行延续接口声明。 |
| [S1419 / L1852](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1852) | <code>{</code> | 开始HcclGetAlgRes的函数体。 |
| [S1420 / L1854](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1854) | <code>    HCCL_INFO(&quot;[HcclGetAlgRes] Start to execute HcclGetAlgRes.&quot;);</code> | 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。 |
| [S1424 / L1859](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1859) | <code>    AlgAttrs algoMeta = executor-&gt;GetAlgoMeta(std::string(param.algName));</code> | 从具体executor获取选中算法的属性元数据。 |
| [S1425 / L1861](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1861) | <code>    HCCL_INFO(</code> | 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。 |
| [S1426 / L1863](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1863) | <code>        &quot;[HcclGetAlgRes] algName=%s, engine=%d, opType=%d, algoTypes.size=%zu.&quot;, param.algName,</code> | 补充日志格式：[HcclGetAlgRes] algName=%s, engine=%d, opType=%d, algoTypes.size=%zu.&quot;, param.algName。 |
| [S1427 / L1865](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1865) | <code>        static_cast&lt;int&gt;(algoMeta.engine), static_cast&lt;int&gt;(algoMeta.opType), algoMeta.algoTypes.size());</code> | 提供上述日志的实参，涉及上下文字节长度。 |
| [S1430 / L1869](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1869) | <code>    bool increCreateChannelFlag = false;</code> | 默认不进行增量通道创建。 |
| [S1431 / L1871](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1871) | <code>    uint64_t size = 0;</code> | 初始化资源上下文长度变量。 |
| [S1432 / L1873](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1873) | <code>    if (TryReuseResource(comm, param, increCreateChannelFlag, resCtxSequence, size, isResourceReused) == HCCL_SUCCESS) {</code> | 尝试按算法tag复用已存在资源。 |
| [S1433 / L1875](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1875) | <code>        return HCCL_SUCCESS;</code> | 命中时跳过拓扑分层和资源申请。 |
| [S1434 / L1877](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1877) | <code>    }</code> | 结束条件if (TryReuseResource(comm, param, increCreateChannelFlag, resCtxSequence, size, isResourceReused) == HCCL_SUCCESS)。 |
| [S1438 / L1882](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1882) | <code>    needInconsistentCheck = NeedInconsistentCheck(comm, param);</code> | 未命中资源时决定是否需要跨rank算子参数一致性校验。 |
| [S1442 / L1887](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1887) | <code>    AlgHierarchyInfoForAllLevel algHierarchyInfo; // 分级通信域信息{localRankId, localRankSize}</code> | 创建算法层级通信域描述对象。 |
| [S1443 / L1889](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1889) | <code>    CHK_RET(executor-&gt;CalcAlgHierarchyInfoV2(topoInfo, algHierarchyInfo, algoMeta));</code> | 调用具体executor把物理拓扑匹配为算法层级通信域。 |
| [S1445 / L1892](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1892) | <code>    HCCL_INFO(&quot;[HcclGetAlgRes] executor-&gt;CalcRes.&quot;);</code> | 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。 |
| [S1446 / L1894](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1894) | <code>    AlgResourceRequest resRequest;</code> | 创建本算法线程、通道、通知等资源需求对象。 |
| [S1447 / L1896](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1896) | <code>    CHK_RET(executor-&gt;CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));</code> | 调用具体executor::CalcRes计算资源需求，尚未物理创建通道。 |
| [S1449 / L1899](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1899) | <code>    auto ret = GetAlgResWithEngine(</code> | 开始按引擎准备实际资源。 |
| [S1450 / L1901](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1901) | <code>        comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size, increCreateChannelFlag,</code> | 传入需求、Host对象、层次信息和Device资源地址等输出。 |
| [S1451 / L1903](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1903) | <code>        resPack);</code> | 补充GE资源包并结束资源准备调用。 |
| [S1452 / L1905](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1905) | <code>    if (ret == HCCL_E_UNAVAIL) {</code> | 资源不可用时保留UNAVAIL供HcclExecOp回退。 |
| [S1453 / L1907](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1907) | <code>        return HCCL_E_UNAVAIL;</code> | 向上返回资源不可用。 |
| [S1454 / L1909](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1909) | <code>    }</code> | 结束条件if (ret == HCCL_E_UNAVAIL)。 |
| [S1455 / L1911](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1911) | <code>    CHK_RET(ret);</code> | 其它非成功资源错误直接传递。 |
| [S1457 / L1914](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1914) | <code>    if (resCtxHost != nullptr) {</code> | Host资源对象存在时整理资源数量日志。 |
| [S1459 / L1917](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1917) | <code>        std::string channelNumInfo;</code> | 初始化按层通道数量的日志字符串。 |
| [S1460 / L1919](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1919) | <code>        for (size_t i = 0; i &lt; resCtxHost-&gt;channels.size(); i++) {</code> | 遍历Host资源对象的算法通信层。 |
| [S1461 / L1921](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1921) | <code>            if (i &gt; 0)</code> | 第二层及以后在数量信息之间添加分隔符。 |
| [S1462 / L1923](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1923) | <code>                channelNumInfo += &quot;, &quot;;</code> | 追加通道层统计的逗号分隔符。 |
| [S1463 / L1925](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1925) | <code>            channelNumInfo += &quot;level&quot; + std::to_string(i) + &quot;[&quot; + std::to_string(resCtxHost-&gt;channels[i].size()) + &quot;]&quot;;</code> | 把当前层序号和该层通道数量追加到日志信息。 |
| [S1464 / L1927](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1927) | <code>        }</code> | 结束循环for (size_t i = 0; i &lt; resCtxHost-&gt;channels.size(); i++)。 |
| [S1465 / L1929](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1929) | <code>        HCCL_RUN_INFO(</code> | 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。 |
| [S1466 / L1931](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1931) | <code>            &quot;[HcclGetAlgRes] engine[%s], algTag[%s], resource allocated: thread num[%u], &quot;</code> | 补充日志格式：[HcclGetAlgRes] engine[%s], algTag[%s], resource allocated: thread num[%u]。 |
| [S1467 / L1933](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1933) | <code>            &quot;channel num per level[%s], ccu kernel num[%u].&quot;,</code> | 补充日志格式：channel num per level[%s], ccu kernel num[%u].。 |
| [S1468 / L1935](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1935) | <code>            GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str(), param.algTag,</code> | 提供上述日志的实参，涉及算子参数、算法关联tag。 |
| [S1469 / L1937](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1937) | <code>            resCtxHost-&gt;threads.size(), channelNumInfo.c_str(), resCtxHost-&gt;ccuKernels.size());</code> | 提供上述日志的实参，涉及Host资源描述对象、上下文字节长度、主从线程列表。 |
| [S1470 / L1939](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1939) | <code>    }</code> | 结束条件if (resCtxHost != nullptr)。 |
| [S1474 / L1944](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1944) | <code>    if (needInconsistentCheck) {</code> | 仅启用一致性校验时比较各rank的交换参数。 |
| [S1475 / L1946](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1946) | <code>        OpExchangeInfo exchangeInfo{};</code> | 创建本端OpExchangeInfo结构。 |
| [S1476 / L1948](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1948) | <code>        CHK_RET(FillOpExchangeInfo(comm, param, exchangeInfo));</code> | 构造本端算子元信息，包括count、类型、归约类型等。 |
| [S1477 / L1950](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1950) | <code>        CHK_RET(CompareOpExchangeInfos(comm, param, resRequest, exchangeInfo));</code> | 与本次资源请求涉及的对端交换信息比较。 |
| [S1478 / L1952](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1952) | <code>    }</code> | 结束条件if (needInconsistentCheck)。 |
| [S1480 / L1955](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1955) | <code>    return HCCL_SUCCESS;</code> | 优先复用，未命中时计算拓扑层次和资源请求，分配资源并按条件核对一致性处理完成，返回成功。 |
| [S1481 / L1957](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1957) | <code>}</code> | 结束HcclGetAlgRes函数体。 |


## 36. FillOpExchangeInfo

构造建链时交换的算子元信息

完整范围：[op_common.cc:L1960–L2031](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1960-L2031)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1483 / L1960](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1960) | <code>HcclResult FillOpExchangeInfo(HcclComm comm, const OpParam&amp; param, OpExchangeInfo&amp; exchangeInfo)</code> | 声明FillOpExchangeInfo接口：构造建链时交换的算子元信息。 |
| [S1484 / L1962](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1962) | <code>{</code> | 开始FillOpExchangeInfo的函数体。 |
| [S1485 / L1964](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1964) | <code>    CHK_PTR_NULL(comm);</code> | 检查通信域句柄非空。 |
| [S1486 / L1966](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1966) | <code>    void* cclBufferAddr = nullptr; // 不使用，仅为调用HcclGetHcclBuffer获取cclBufferSize</code> | 创建不使用的CCL地址变量，查询仅需返回缓冲区大小。 |
| [S1487 / L1968](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1968) | <code>    CHK_RET(HcclGetHcclBuffer(comm, &amp;cclBufferAddr, &amp;exchangeInfo.cclBufferSize));</code> | 取得本域CCL容量，写进一致性交换信息。 |
| [S1488 / L1970](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1970) | <code>    exchangeInfo.root = param.root;</code> | 保存本次root字段，AllReduce虽不依赖root也使用通用结构。 |
| [S1489 / L1972](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1972) | <code>    exchangeInfo.opType = param.opType;</code> | 保存本次命令ALLREDUCE。 |
| [S1490 / L1974](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1974) | <code>    exchangeInfo.opExecuteConfig = param.opExecuteConfig;</code> | 保存最终执行配置，供不同rank比较。 |
| [S1491 / L1976](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1976) | <code>    exchangeInfo.reduceType = param.reduceType;</code> | 保存本次SUM等归约类型。 |
| [S1492 / L1978](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1978) | <code>    CHK_RET(FillOpExchangeInfoWithDataDes(param, exchangeInfo));</code> | 按算子数据描述布局填入dtype和count。 |
| [S1494 / L1981](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1981) | <code>    u32 numBlocksLimit = 0;</code> | 初始化AIV核数限制查询结果为零。 |
| [S1495 / L1983](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1983) | <code>    AivParamStorage* aivParam = nullptr;</code> | 初始化通信域AIV参数对象地址。 |
| [S1496 / L1985](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1985) | <code>    HcclResult ret = GetAivParamStorageByComm(comm, &amp;aivParam, false);</code> | 尝试查询已有通信域AIV参数，但不创建新对象。 |
| [S1497 / L1987](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1987) | <code>    if (ret == HCCL_SUCCESS &amp;&amp; aivParam != nullptr) {</code> | 已有AIV参数对象时读取核数限制。 |
| [S1498 / L1989](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1989) | <code>        numBlocksLimit = aivParam-&gt;aivCoreLimit;</code> | 从AIV参数对象取核数限制。 |
| [S1499 / L1991](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1991) | <code>        exchangeInfo.aivCoreLimit = numBlocksLimit;</code> | 将核数限制写到算子交换结构。 |
| [S1500 / L1993](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1993) | <code>    }</code> | 结束条件if (ret == HCCL_SUCCESS &amp;&amp; aivParam != nullptr)。 |
| [S1501 / L1995](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1995) | <code>    if (numBlocksLimit == 0 &amp;&amp; param.opMode == OpMode::OPBASE) {</code> | 无既有核数限制且OPBASE时查询当前运行时VectorCore数量。 |
| [S1502 / L1997](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1997) | <code>        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &amp;numBlocksLimit));</code> | 通过ACL读取当前线程绑定资源的VectorCore数量。 |
| [S1503 / L1999](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1999) | <code>        exchangeInfo.aivCoreLimit = numBlocksLimit;</code> | 保存运行时核数限制到交换结构。 |
| [S1504 / L2001](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2001) | <code>    }</code> | 结束条件if (numBlocksLimit == 0 &amp;&amp; param.opMode == OpMode::OPBASE)。 |
| [S1506 / L2004](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2004) | <code>    CHK_RET(HcclGetCommName(comm, exchangeInfo.group));</code> | 查询通信域名字写入交换结构。 |
| [S1507 / L2006](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2006) | <code>    exchangeInfo.group[MAX_LENGTH - 1] = &#x27;\0&#x27;;</code> | 强制域名字数组末端为字符串终止符。 |
| [S1508 / L2008](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2008) | <code>    s32 sRet = strncpy_s(exchangeInfo.tag, TAG_LENGTH, param.tag, TAG_LENGTH);</code> | 把本次算子tag复制到交换结构。 |
| [S1509 / L2010](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2010) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S1510 / L2012](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2012) | <code>        sRet != EOK, HCCL_ERROR(&quot;[%s] call strncpy_s failed, param.tag[%s], return[%d].&quot;, __func__, param.tag, sRet),</code> | 检查tag安全复制是否成功，失败时打印tag和返回码。 |
| [S1511 / L2014](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2014) | <code>        HCCL_E_MEMORY);</code> | 字符串复制失败返回内存错误。 |
| [S1513 / L2017](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2017) | <code>    HCCL_INFO(</code> | 输出运行日志，记录FillOpExchangeInfo当前阶段和相关参数。 |
| [S1514 / L2019](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2019) | <code>        &quot;[%s] success. exchangeInfo dump: cclBufferSize[%llu], root[%u], opType[%u], opExecuteConfig[%u], &quot;</code> | 补充日志格式：[%s] success. exchangeInfo dump: cclBufferSize[%llu], root[%u], opType[%u], opExecuteConfig[%u]。 |
| [S1515 / L2021](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2021) | <code>        &quot;reduceType[%u], dataType[%u], count[%llu], aivCoreLimit[%u], group[%s], tag[%s]&quot;,</code> | 补充日志格式：reduceType[%u], dataType[%u], count[%llu], aivCoreLimit[%u], group[%s], tag[%s]。 |
| [S1516 / L2023](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2023) | <code>        __func__, exchangeInfo.cclBufferSize, exchangeInfo.root, exchangeInfo.opType, exchangeInfo.opExecuteConfig,</code> | 提供上述日志的实参：__func__, exchangeInfo.cclBufferSize, exchangeInfo.root, exchangeInfo.opType, exchangeInfo.opExecuteConfig,。 |
| [S1517 / L2025](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2025) | <code>        exchangeInfo.reduceType, exchangeInfo.dataType, exchangeInfo.count, exchangeInfo.aivCoreLimit,</code> | 提供上述日志的实参：exchangeInfo.reduceType, exchangeInfo.dataType, exchangeInfo.count, exchangeInfo.aivCoreLimit,。 |
| [S1518 / L2027](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2027) | <code>        exchangeInfo.group, exchangeInfo.tag);</code> | 提供上述日志的实参，涉及算法名字键。 |
| [S1519 / L2029](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2029) | <code>    return HCCL_SUCCESS;</code> | 构造建链时交换的算子元信息处理完成，返回成功。 |
| [S1520 / L2031](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2031) | <code>}</code> | 结束FillOpExchangeInfo函数体。 |


## 37. FillOpExchangeInfoWithDataDes

按算子参数布局填充数据类型和count，AllReduce走default

完整范围：[op_common.cc:L2034–L2082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2034-L2082)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1522 / L2034](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2034) | <code>HcclResult FillOpExchangeInfoWithDataDes(const OpParam&amp; param, OpExchangeInfo&amp; exchangeInfo)</code> | 声明FillOpExchangeInfoWithDataDes接口：按算子参数布局填充数据类型和count，AllReduce走default。 |
| [S1523 / L2036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2036) | <code>{</code> | 开始FillOpExchangeInfoWithDataDes的函数体。 |
| [S1524 / L2038](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2038) | <code>    switch (param.opType) {</code> | 按算子类型选择数据描述字段布局。 |
| [S1525 / L2040](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2040) | <code>        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:</code> | BatchSendRecv布局没有统一count，在该分支不填dtype/count。 |
| [S1526 / L2042](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2042) | <code>            break;</code> | 结束当前算子类型分支，退出switch。 |
| [S1527 / L2044](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2044) | <code>        case HcclCMDType::HCCL_CMD_ALLTOALL:</code> | 固定AllToAll从all2AllVDataDes布局读取数据类型。 |
| [S1528 / L2046](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2046) | <code>            exchangeInfo.dataType = param.all2AllVDataDes.sendType;</code> | 保存AllToAll发送数据类型。 |
| [S1529 / L2048](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2048) | <code>            CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);</code> | 检查AllToAll的sendCounts指针非空。 |
| [S1530 / L2050](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2050) | <code>            exchangeInfo.count = static_cast&lt;u64*&gt;(param.all2AllVDataDes.sendCounts)[0];</code> | 从AllToAll第一个sendCounts元素取得统一count。 |
| [S1531 / L2052](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2052) | <code>            break;</code> | 结束当前算子类型分支，退出switch。 |
| [S1532 / L2054](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2054) | <code>        case HcclCMDType::HCCL_CMD_ALLTOALLV:</code> | 变长AllToAllV使用all2AllVDataDes发送数据类型。 |
| [S1533 / L2056](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2056) | <code>        case HcclCMDType::HCCL_CMD_ALLTOALLVC:</code> | AllToAllVC与AllToAllV共享此分支。 |
| [S1534 / L2058](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2058) | <code>            exchangeInfo.dataType = param.all2AllVDataDes.sendType;</code> | 保存AllToAllV/VC的发送数据类型，不在这里填统一count。 |
| [S1535 / L2060](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2060) | <code>            break;</code> | 结束当前算子类型分支，退出switch。 |
| [S1536 / L2062](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2062) | <code>        case HcclCMDType::HCCL_CMD_ALLGATHER_V:</code> | 变长AllGatherV使用vDataDes布局。 |
| [S1537 / L2064](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2064) | <code>        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:</code> | 变长ReduceScatterV与AllGatherV共享数据类型读取逻辑。 |
| [S1538 / L2066](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2066) | <code>            exchangeInfo.dataType = param.vDataDes.dataType;</code> | 保存变长集合通信的vDataDes数据类型。 |
| [S1539 / L2068](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2068) | <code>            break;</code> | 结束当前算子类型分支，退出switch。 |
| [S1540 / L2070](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2070) | <code>        default:</code> | 其它算子含AllReduce进入统一DataDes路径。 |
| [S1541 / L2072](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2072) | <code>            exchangeInfo.dataType = param.DataDes.dataType;</code> | AllReduce将DataDes.dataType写入交换结构。 |
| [S1542 / L2074](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2074) | <code>            exchangeInfo.count = param.DataDes.count;</code> | AllReduce将DataDes.count写入交换结构。 |
| [S1543 / L2076](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2076) | <code>            break;</code> | 结束当前算子类型分支，退出switch。 |
| [S1544 / L2078](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2078) | <code>    }</code> | 结束代码块。 |
| [S1545 / L2080](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2080) | <code>    return HCCL_SUCCESS;</code> | 按算子参数布局填充数据类型和count，AllReduce走default处理完成，返回成功。 |
| [S1546 / L2082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2082) | <code>}</code> | 结束FillOpExchangeInfoWithDataDes函数体。 |


## 38. AddExchangeInfo

启用参数一致性检查时登记下一次建链交换的OpExchangeInfo

完整范围：[op_common.cc:L2087–L2107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2087-L2107)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1550 / L2087](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2087) | <code>HcclResult AddExchangeInfo(HcclComm comm, const OpParam&amp; param)</code> | 声明AddExchangeInfo接口：启用参数一致性检查时登记下一次建链交换的OpExchangeInfo。 |
| [S1551 / L2089](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2089) | <code>{</code> | 开始AddExchangeInfo的函数体。 |
| [S1552 / L2091](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2091) | <code>    CHK_PTR_NULL(comm);</code> | 检查登记交换信息的通信域非空。 |
| [S1553 / L2093](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2093) | <code>    if (needInconsistentCheck) {</code> | 仅needInconsistentCheck为true时登记元信息。 |
| [S1554 / L2095](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2095) | <code>        OpExchangeInfo exchangeInfo{};</code> | 创建本端算子交换结构。 |
| [S1555 / L2097](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2097) | <code>        CHK_RET(FillOpExchangeInfo(comm, param, exchangeInfo));</code> | 填充本端count、dtype、reduceType、配置等交换字段。 |
| [S1556 / L2099](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2099) | <code>        CHK_RET(HcclCommAddExchangeInfo(comm, &amp;exchangeInfo, sizeof(exchangeInfo)));</code> | 将结构字节登记到HCOMM通信域，供随后的Acquire建链读取。 |
| [S1557 / L2101](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2101) | <code>        HCCL_INFO(&quot;[%s] success.&quot;, __func__);</code> | 输出运行日志，记录AddExchangeInfo当前阶段和相关参数。 |
| [S1558 / L2103](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2103) | <code>    }</code> | 结束条件if (needInconsistentCheck)。 |
| [S1559 / L2105](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2105) | <code>    return HCCL_SUCCESS;</code> | 启用参数一致性检查时登记下一次建链交换的OpExchangeInfo处理完成，返回成功。 |
| [S1560 / L2107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2107) | <code>}</code> | 结束AddExchangeInfo函数体。 |


## 39. GetAlgResWithEngine

按CPU/AICPU_TS/AIV/CCU引擎分发资源准备，CCU可跨rank协商回退

完整范围：[op_common.cc:L2130–L2235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2130-L2235)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1591 / L2146](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2146)：CPU；GetAlgResDPU

- [S1599 / L2160](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2160)：AICPU_TS；GetAlgResAICPU

- [S1603 / L2168](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2168)：AIV；GetAlgResAiv

- [S1605 / L2172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2172)：CCU；GetAlgResCcu -> 9.2及以上CheckCcuResNegotiation

- [S1631 / L2219](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2219)：其它未识别engine；PARA



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1582 / L2130](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2130) | <code>HcclResult GetAlgResWithEngine(</code> | 声明GetAlgResWithEngine接口：按CPU/AICPU_TS/AIV/CCU引擎分发资源准备，CCU可跨rank协商回退。 |
| [S1583 / L2132](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2132) | <code>    HcclComm comm, OpParam&amp; param, AlgResourceRequest&amp; resRequest,</code> | 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。 |
| [S1584 / L2134](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2134) | <code>    std::unique_ptr&lt;AlgResourceCtxSerializable&gt;&amp; resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,</code> | 函数参数包含物理拓扑对象、Host资源描述对象，本行延续接口声明。 |
| [S1585 / L2136](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2136) | <code>    AlgHierarchyInfoForAllLevel&amp; algHierarchyInfo, void** resCtxSequence, uint64_t&amp; size, bool increCreateChannelFlag,</code> | 函数参数包含Device资源序列化地址输出、算法分层通信域、上下文字节长度、增量建链标志，本行延续接口声明。 |
| [S1586 / L2138](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2138) | <code>    const ResPackGraphMode&amp; resPack)</code> | 函数参数包含图模式资源包，本行延续接口声明。 |
| [S1587 / L2140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2140) | <code>{</code> | 开始GetAlgResWithEngine的函数体。 |
| [S1589 / L2143](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2143) | <code>    if (param.engine == COMM_ENGINE_RESERVED) {</code> | RESERVED引擎分支目前没有资源准备动作。 |
| [S1591 / L2146](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2146) | <code>    } else if (param.engine == COMM_ENGINE_CPU) {</code> | CPU引擎走Host DPU资源申请。 |
| [S1592 / L2148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2148) | <code>        CHK_RET(GetAlgResDPU(</code> | 开始调用GetAlgResDPU。 |
| [S1593 / L2150](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2150) | <code>            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size,</code> | 传入DPU资源请求、拓扑、资源对象和序列化输出。 |
| [S1594 / L2152](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2152) | <code>            increCreateChannelFlag, resPack));</code> | 补充增量标志和GE资源包，结束DPU资源调用。 |
| [S1595 / L2154](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2154) | <code>    } else if (param.engine == COMM_ENGINE_CPU_TS) {</code> | CPU_TS分支当前没有资源准备实现。 |
| [S1597 / L2157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2157) | <code>    } else if (param.engine == COMM_ENGINE_AICPU) {</code> | 裸AICPU分支当前没有资源准备实现。 |
| [S1599 / L2160](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2160) | <code>    } else if (param.engine == COMM_ENGINE_AICPU_TS) {</code> | AICPU_TS调用AICPU线程/通道资源准备，是本例主分支。 |
| [S1600 / L2162](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2162) | <code>        CHK_RET(GetAlgResAICPU(</code> | 开始调用GetAlgResAICPU。 |
| [S1601 / L2164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2164) | <code>            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size,</code> | 传入本次资源需求、Host对象、算法层次及Device地址输出。 |
| [S1602 / L2166](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2166) | <code>            increCreateChannelFlag, resPack));</code> | 补充增量标志和GE资源包，结束AICPU资源准备调用。 |
| [S1603 / L2168](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2168) | <code>    } else if (param.engine == COMM_ENGINE_AIV) {</code> | AIV引擎申请其专属资源。 |
| [S1604 / L2170](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2170) | <code>        CHK_RET(GetAlgResAiv(comm, param, resRequest, topoInfo, algHierarchyInfo, resCtxSequence));</code> | 调用AIV资源准备，返回AIV上下文地址。 |
| [S1605 / L2172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2172) | <code>    } else if (param.engine == COMM_ENGINE_CCU) {</code> | CCU引擎申请其专属资源。 |
| [S1607 / L2175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2175) | <code>        auto ret = GetAlgResCcu(</code> | 开始获取CCU资源并保留返回码用于跨rank协商。 |
| [S1608 / L2177](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2177) | <code>            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size, resPack);</code> | 传入CCU需求、上下文、拓扑层次和GE资源包。 |
| [S1610 / L2180](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2180) | <code>#if CANN_VERSION_NUM &gt;= CANN_VERSION(9, 2, 0)</code> | 按CANN_VERSION_NUM &gt;= CANN_VERSION(9, 2, 0)这个编译期版本条件决定是否包含以下分支。 |
| [S1611 / L2182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2182) | <code>        if (ret == HCCL_E_UNAVAIL &#124;&#124; ret == HCCL_SUCCESS) {</code> | CANN 9.2及以上，本端成功或资源不足都需跨rank协商。 |
| [S1613 / L2185](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2185) | <code>            bool localResAvailable = (ret == HCCL_SUCCESS);</code> | 以本端CCU资源是否成功作为协商输入。 |
| [S1614 / L2187](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2187) | <code>            auto negRet = CheckCcuResNegotiation(comm, param, localResAvailable);</code> | 协商各rank是否均有CCU资源可执行。 |
| [S1615 / L2189](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2189) | <code>            if (negRet == HCCL_E_UNAVAIL) {</code> | 协商结果UNAVAIL表示全域应回退。 |
| [S1618 / L2193](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2193) | <code>                ReleaseCcuAcquiredChannels(comm, resRequest);</code> | 请求释放本端已取得的CCU通道。 |
| [S1619 / L2195](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2195) | <code>                return HCCL_E_UNAVAIL;</code> | 将跨rank资源不可用结果传回HcclExecOp。 |
| [S1620 / L2197](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2197) | <code>            }</code> | 结束条件if (negRet == HCCL_E_UNAVAIL)。 |
| [S1621 / L2199](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2199) | <code>            CHK_RET(negRet);</code> | 其它协商错误直接向上传递。 |
| [S1622 / L2201](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2201) | <code>        } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1623 / L2203](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2203) | <code>            CHK_RET(ret);</code> | CCU申请的非UNAVAIL错误直接传递。 |
| [S1624 / L2205](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2205) | <code>        }</code> | 结束条件} else。 |
| [S1625 / L2207](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2207) | <code>#else</code> | 切换到上述编译期条件不成立的兼容实现。 |
| [S1626 / L2209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2209) | <code>        if (ret == HCCL_E_UNAVAIL) {</code> | 较旧CANN分支只有本端CCU资源不可用判断。 |
| [S1627 / L2211](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2211) | <code>            return HCCL_E_UNAVAIL;</code> | 将CCU资源不可用向上传递。 |
| [S1628 / L2213](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2213) | <code>        }</code> | 结束条件if (ret == HCCL_E_UNAVAIL)。 |
| [S1629 / L2215](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2215) | <code>        CHK_RET(ret);</code> | 其它CCU申请错误直接传递。 |
| [S1630 / L2217](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2217) | <code>#endif</code> | 结束上述编译期条件控制的源码范围。 |
| [S1631 / L2219](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2219) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1632 / L2221](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2221) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录GetAlgResWithEngine当前阶段和相关参数。 |
| [S1633 / L2223](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2223) | <code>            &quot;fail to get engine, invalid engine type[%s].&quot;,</code> | 补充日志格式：fail to get engine, invalid engine type[%s].。 |
| [S1634 / L2225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2225) | <code>            GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str());</code> | 提供上述日志的实参，涉及算子参数。 |
| [S1635 / L2227](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2227) | <code>        return HCCL_E_PARA;</code> | 无法识别的引擎返回参数错误。 |
| [S1636 / L2229](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2229) | <code>    }</code> | 结束条件} else。 |
| [S1637 / L2231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2231) | <code>    param.ctxSize = size;</code> | 保存资源序列化长度到Device参数。 |
| [S1638 / L2233](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2233) | <code>    return HCCL_SUCCESS;</code> | 按CPU/AICPU_TS/AIV/CCU引擎分发资源准备，CCU可跨rank协商回退处理完成，返回成功。 |
| [S1639 / L2235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2235) | <code>}</code> | 结束GetAlgResWithEngine函数体。 |


## 40. CacheHostCtxToEngine

增量建链首次保存Host资源副本，失败按源码执行Device/Host上下文回滚

完整范围：[op_common.cc:L2238–L2282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2238-L2282)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1641 / L2238](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2238) | <code>HcclResult CacheHostCtxToEngine(</code> | 声明CacheHostCtxToEngine接口：增量建链首次保存Host资源副本，失败按源码执行Device/Host上下文回滚。 |
| [S1642 / L2240](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2240) | <code>    HcclComm comm, const char* algTag, const std::string&amp; hostCacheTag, const std::vector&lt;char&gt;&amp; hostCtxSeq)</code> | 函数参数包含通信域句柄、算法关联tag，本行延续接口声明。 |
| [S1643 / L2242](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2242) | <code>{</code> | 开始CacheHostCtxToEngine的函数体。 |
| [S1644 / L2244](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2244) | <code>    void* hostCtxPtr = nullptr;</code> | 初始化增量Host缓存上下文地址为空。 |
| [S1645 / L2246](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2246) | <code>    HcclResult createRet = HcclEngineCtxCreate(</code> | 开始创建CPU_TS Host缓存上下文。 |
| [S1646 / L2248](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2248) | <code>        comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, hostCtxSeq.size(), &amp;hostCtxPtr);</code> | 按hostCacheTag为Host序列化副本分配存储，大小为hostCtxSeq长度。 |
| [S1647 / L2250](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2250) | <code>    if (createRet != HCCL_SUCCESS) {</code> | Host缓存创建失败时回滚此前创建的Device资源上下文。 |
| [S1648 / L2252](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2252) | <code>        HCCL_ERROR(&quot;failed to create host EngineCtx for caching, ret[%d].&quot;, createRet);</code> | 输出错误日志，记录CacheHostCtxToEngine当前阶段和相关参数。 |
| [S1649 / L2254](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2254) | <code>        HcclResult destroyRet = HcclEngineCtxDestroy(comm, algTag, COMM_ENGINE_AICPU_TS);</code> | 尝试销毁AICPU_TS算法上下文，记录销毁返回码。 |
| [S1650 / L2256](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2256) | <code>        if (destroyRet != HCCL_SUCCESS) {</code> | Device上下文销毁也失败时只打印错误，仍返回原createRet。 |
| [S1651 / L2258](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2258) | <code>            HCCL_ERROR(&quot;failed to destroy device ctx on host ctx create failure rollback, ret[%d].&quot;, destroyRet);</code> | 输出错误日志，记录CacheHostCtxToEngine当前阶段和相关参数。 |
| [S1652 / L2260](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2260) | <code>        }</code> | 结束条件if (destroyRet != HCCL_SUCCESS)。 |
| [S1653 / L2262](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2262) | <code>        return createRet;</code> | 返回原Host缓存创建失败的错误码。 |
| [S1654 / L2264](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2264) | <code>    }</code> | 结束条件if (createRet != HCCL_SUCCESS)。 |
| [S1655 / L2266](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2266) | <code>    errno_t memcpyRet = memcpy_s(hostCtxPtr, hostCtxSeq.size(), hostCtxSeq.data(), hostCtxSeq.size());</code> | 安全复制Host资源序列化字节到新缓存地址。 |
| [S1656 / L2268](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2268) | <code>    if (memcpyRet != EOK) {</code> | Host缓存字节复制失败时销毁两侧缓存上下文。 |
| [S1657 / L2270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2270) | <code>        HCCL_ERROR(&quot;memcpy_s failed writing to host EngineCtx cache, ret=%d.&quot;, memcpyRet);</code> | 输出错误日志，记录CacheHostCtxToEngine当前阶段和相关参数。 |
| [S1658 / L2272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2272) | <code>        HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);</code> | 请求销毁CPU_TS Host缓存，本行未检查返回码。 |
| [S1659 / L2274](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2274) | <code>        HcclEngineCtxDestroy(comm, algTag, COMM_ENGINE_AICPU_TS);</code> | 请求销毁AICPU_TS Device上下文，本行未检查返回码。 |
| [S1660 / L2276](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2276) | <code>        return HCCL_E_INTERNAL;</code> | 缓存复制失败返回内部错误。 |
| [S1661 / L2278](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2278) | <code>    }</code> | 结束条件if (memcpyRet != EOK)。 |
| [S1662 / L2280](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2280) | <code>    return HCCL_SUCCESS;</code> | 增量建链首次保存Host资源副本，失败按源码执行Device/Host上下文回滚处理完成，返回成功。 |
| [S1663 / L2282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2282) | <code>}</code> | 结束CacheHostCtxToEngine函数体。 |


## 41. ReuseCachedDeviceCtx

增量请求没有新Peer时直接查已有Device序列化上下文

完整范围：[op_common.cc:L2285–L2319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2285-L2319)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1665 / L2285](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2285) | <code>HcclResult ReuseCachedDeviceCtx(HcclComm comm, const OpParam&amp; param, void** resCtxSequence, uint64_t&amp; ctxSize)</code> | 声明ReuseCachedDeviceCtx接口：增量请求没有新Peer时直接查已有Device序列化上下文。 |
| [S1666 / L2287](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2287) | <code>{</code> | 开始ReuseCachedDeviceCtx的函数体。 |
| [S1667 / L2289](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2289) | <code>    void* ctx = nullptr;</code> | 初始化已有Device上下文查询地址。 |
| [S1668 / L2291](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2291) | <code>    uint64_t size = 0;</code> | 初始化已有Device上下文查询长度。 |
| [S1669 / L2293](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2293) | <code>    HcclResult ret;</code> | 声明Device上下文查询返回码。 |
| [S1670 / L2295](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2295) | <code>    if (param.engine == COMM_ENGINE_CPU) {</code> | CPU/DPU资源实际存储在AICPU_TS，所以查询需转换engine。 |
| [S1671 / L2297](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2297) | <code>        ret = HcclEngineCtxGet(comm, param.algTag, COMM_ENGINE_AICPU_TS, &amp;ctx, &amp;size);</code> | 按AICPU_TS存储engine查CPU执行资源。 |
| [S1672 / L2299](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2299) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1673 / L2301](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2301) | <code>        ret = HcclEngineCtxGet(comm, param.algTag, param.engine, &amp;ctx, &amp;size);</code> | 其它引擎按param.engine查询资源上下文。 |
| [S1674 / L2303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2303) | <code>    }</code> | 结束条件} else。 |
| [S1675 / L2305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2305) | <code>    if (ret == HCCL_SUCCESS) {</code> | 查询成功时把已有上下文地址与大小返回调用方。 |
| [S1676 / L2307](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2307) | <code>        *resCtxSequence = ctx;</code> | 返回已有Device资源序列化地址。 |
| [S1677 / L2309](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2309) | <code>        ctxSize = size;</code> | 返回已有Device资源序列化长度。 |
| [S1678 / L2311](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2311) | <code>        return HCCL_SUCCESS;</code> | 增量无新增Peer时成功复用Device上下文。 |
| [S1679 / L2313](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2313) | <code>    }</code> | 结束条件if (ret == HCCL_SUCCESS)。 |
| [S1680 / L2315](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2315) | <code>    HCCL_ERROR(&quot;failed to get device ctx.&quot;);</code> | 输出错误日志，记录ReuseCachedDeviceCtx当前阶段和相关参数。 |
| [S1681 / L2317](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2317) | <code>    return ret;</code> | 未能查询到已有Device上下文时返回原始错误码。 |
| [S1682 / L2319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2319) | <code>}</code> | 结束ReuseCachedDeviceCtx函数体。 |


## 42. IncrementalCreateChannel

创建缺失通道并更新Device序列化上下文及Host缓存，AllReduce普通主例不进入

完整范围：[op_common.cc:L2322–L2418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2322-L2418)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1684 / L2322](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2322) | <code>HcclResult IncrementalCreateChannel(</code> | 声明IncrementalCreateChannel接口：创建缺失通道并更新Device序列化上下文及Host缓存，AllReduce普通主例不进入。 |
| [S1685 / L2324](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2324) | <code>    HcclComm comm, const OpParam&amp; param, AlgResourceRequest&amp; resRequest, AlgResourceCtxSerializable&amp; hostCtxObj,</code> | 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。 |
| [S1686 / L2326](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2326) | <code>    const std::string&amp; hostCacheTag, void** resCtxSequence, uint64_t&amp; ctxSize)</code> | 函数参数包含Device资源序列化地址输出、资源上下文字节长度，本行延续接口声明。 |
| [S1687 / L2328](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2328) | <code>{</code> | 开始IncrementalCreateChannel的函数体。 |
| [S1688 / L2330](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2330) | <code>    HcclResult ret = HcclGetChannel(comm, param, resRequest, &amp;hostCtxObj);</code> | 为筛选后的新增Peer取得通道并追加到Host资源对象。 |
| [S1689 / L2332](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2332) | <code>    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR(&quot;failed to incrementally create channel.&quot;), ret);</code> | 新增通道获取失败直接返回该错误。 |
| [S1690 / L2334](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2334) | <code>    if (param.engine == COMM_ENGINE_CPU) {</code> | CPU/DPU的旧资源上下文存于AICPU_TS。 |
| [S1691 / L2336](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2336) | <code>        ret = HcclEngineCtxDestroy(comm, param.algTag, COMM_ENGINE_AICPU_TS);</code> | 请求销毁旧AICPU_TS序列化资源上下文。 |
| [S1692 / L2338](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2338) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1693 / L2340](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2340) | <code>        ret = HcclEngineCtxDestroy(comm, param.algTag, param.engine);</code> | 其它引擎按param.engine销毁旧Device资源上下文。 |
| [S1694 / L2342](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2342) | <code>    }</code> | 结束条件} else。 |
| [S1695 / L2344](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2344) | <code>    if (ret != HCCL_SUCCESS) {</code> | 旧Device上下文销毁失败时打印错误，但仍继续下面的重建流程。 |
| [S1696 / L2346](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2346) | <code>        HCCL_ERROR(&quot;failed to destroy device Ctx, ret[%d].&quot;, ret);</code> | 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1697 / L2348](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2348) | <code>    }</code> | 结束条件if (ret != HCCL_SUCCESS)。 |
| [S1698 / L2350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2350) | <code>    std::vector&lt;char&gt; newSeq = hostCtxObj.Serialize();</code> | 将增量更新后的Host资源对象序列化。 |
| [S1699 / L2352](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2352) | <code>    ret = HcclMemcpyCtxHostToDevice(comm, param, newSeq, resCtxSequence, ctxSize);</code> | 创建新的Device上下文并复制更新后的资源字节。 |
| [S1700 / L2354](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2354) | <code>    if (ret != HCCL_SUCCESS) {</code> | 重建Device上下文失败时尝试清理已有Host缓存。 |
| [S1701 / L2356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2356) | <code>        HCCL_ERROR(&quot;failed to memcpy hostCtx to device after incremental channel creation, ret[%d].&quot;, ret);</code> | 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1702 / L2358](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2358) | <code>        HcclResult destroyRet = HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);</code> | 请求销毁CPU_TS Host缓存并保存清理返回码。 |
| [S1703 / L2360](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2360) | <code>        if (destroyRet != HCCL_SUCCESS) {</code> | Host缓存清理失败时只打印错误，保留原Device重建错误。 |
| [S1704 / L2362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2362) | <code>            HCCL_ERROR(&quot;failed to destroy host ctx on incremental path failure rollback, ret[%d].&quot;, destroyRet);</code> | 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1705 / L2364](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2364) | <code>        }</code> | 结束条件if (destroyRet != HCCL_SUCCESS)。 |
| [S1706 / L2366](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2366) | <code>        return ret;</code> | 返回原Device资源复制失败错误。 |
| [S1707 / L2368](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2368) | <code>    }</code> | 结束条件if (ret != HCCL_SUCCESS)。 |
| [S1708 / L2370](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2370) | <code>    HcclResult destroyRet = HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);</code> | 更新Device成功后销毁旧CPU_TS Host缓存。 |
| [S1709 / L2372](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2372) | <code>    if (destroyRet != HCCL_SUCCESS) {</code> | 旧Host缓存销毁失败时打印错误，继续创建更新缓存。 |
| [S1710 / L2374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2374) | <code>        HCCL_ERROR(&quot;failed to destroy old host EngineCtx for cache update, ret[%d].&quot;, destroyRet);</code> | 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1711 / L2376](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2376) | <code>    }</code> | 结束条件if (destroyRet != HCCL_SUCCESS)。 |
| [S1712 / L2378](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2378) | <code>    void* newHostCtxPtr = nullptr;</code> | 初始化更新后的Host缓存存储地址。 |
| [S1713 / L2380](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2380) | <code>    HcclResult cacheRet = HcclEngineCtxCreate(</code> | 开始创建更新CPU_TS Host缓存。 |
| [S1714 / L2382](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2382) | <code>        comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, newSeq.size(), &amp;newHostCtxPtr);</code> | 按更新序列化长度创建Host资源缓存上下文。 |
| [S1715 / L2384](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2384) | <code>    if (cacheRet != HCCL_SUCCESS) {</code> | 更新Host缓存创建失败时回滚新Device上下文。 |
| [S1716 / L2386](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2386) | <code>        HCCL_ERROR(&quot;failed to create host EngineCtx for cache update, ret[%d].&quot;, cacheRet);</code> | 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1717 / L2388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2388) | <code>        HcclResult devDestroyRet = HcclEngineCtxDestroy(comm, param.algTag, param.engine);</code> | 请求销毁param.engine算法Device资源上下文，保存销毁错误。 |
| [S1718 / L2390](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2390) | <code>        if (devDestroyRet != HCCL_SUCCESS) {</code> | 回滚Device上下文销毁失败时只打印错误。 |
| [S1719 / L2392](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2392) | <code>            HCCL_ERROR(&quot;failed to destroy device ctx on host cache update failure rollback, ret[%d].&quot;, devDestroyRet);</code> | 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1720 / L2394](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2394) | <code>        }</code> | 结束条件if (devDestroyRet != HCCL_SUCCESS)。 |
| [S1721 / L2396](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2396) | <code>        return cacheRet;</code> | 返回原Host缓存创建错误。 |
| [S1722 / L2398](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2398) | <code>    }</code> | 结束条件if (cacheRet != HCCL_SUCCESS)。 |
| [S1723 / L2400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2400) | <code>    errno_t memcpyRet = memcpy_s(newHostCtxPtr, newSeq.size(), newSeq.data(), newSeq.size());</code> | 安全复制更新资源字节到新Host缓存。 |
| [S1724 / L2402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2402) | <code>    if (memcpyRet != EOK) {</code> | 更新Host缓存字节复制失败时请求销毁两侧上下文。 |
| [S1725 / L2404](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2404) | <code>        HCCL_ERROR(&quot;memcpy_s failed writing to updated host EngineCtx cache, ret=%d.&quot;, memcpyRet);</code> | 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1726 / L2406](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2406) | <code>        HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);</code> | 请求销毁更新后的Host缓存，本行未检查返回码。 |
| [S1727 / L2408](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2408) | <code>        HcclEngineCtxDestroy(comm, param.algTag, param.engine);</code> | 请求销毁新的Device资源上下文，本行未检查返回码。 |
| [S1728 / L2410](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2410) | <code>        return HCCL_E_INTERNAL;</code> | 更新Host缓存复制失败返回内部错误。 |
| [S1729 / L2412](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2412) | <code>    }</code> | 结束条件if (memcpyRet != EOK)。 |
| [S1730 / L2414](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2414) | <code>    HCCL_INFO(&quot;Incrementally add channel success&quot;);</code> | 输出运行日志，记录IncrementalCreateChannel当前阶段和相关参数。 |
| [S1731 / L2416](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2416) | <code>    return HCCL_SUCCESS;</code> | 创建缺失通道并更新Device序列化上下文及Host缓存，AllReduce普通主例不进入处理完成，返回成功。 |
| [S1732 / L2418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2418) | <code>}</code> | 结束IncrementalCreateChannel函数体。 |


## 43. GetAlgResAICPU

首次请求构造Host资源对象并复制Device，增量请求可筛除已存在通道

完整范围：[op_common.cc:L2423–L2499](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2423-L2499)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1749 / L2447](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2447)：普通申请或Host缓存未命中；HcclAllocAlgResourceAICPU -> Serialize -> HcclMemcpyCtxHostToDevice

- [S1762 / L2472](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2472)：增量申请且Host缓存命中；DeSerialize -> CompReqChannelWithExistChannel -> ReuseCachedDeviceCtx或IncrementalCreateChannel；AllReduce普通主例不进入增量分支



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1736 / L2423](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2423) | <code>HcclResult GetAlgResAICPU(</code> | 声明GetAlgResAICPU接口：首次请求构造Host资源对象并复制Device，增量请求可筛除已存在通道。 |
| [S1737 / L2425](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2425) | <code>    HcclComm comm, const OpParam&amp; param, AlgResourceRequest&amp; resRequest,</code> | 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。 |
| [S1738 / L2427](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2427) | <code>    std::unique_ptr&lt;AlgResourceCtxSerializable&gt;&amp; resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,</code> | 函数参数包含物理拓扑对象、Host资源描述对象，本行延续接口声明。 |
| [S1739 / L2429](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2429) | <code>    AlgHierarchyInfoForAllLevel&amp; algHierarchyInfo, void** resCtxSequence, uint64_t&amp; ctxSize,</code> | 函数参数包含Device资源序列化地址输出、算法分层通信域、资源上下文字节长度，本行延续接口声明。 |
| [S1740 / L2431](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2431) | <code>    bool increCreateChannelFlag, const ResPackGraphMode&amp; resPack)</code> | 函数参数包含图模式资源包、增量建链标志，本行延续接口声明。 |
| [S1741 / L2433](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2433) | <code>{</code> | 开始GetAlgResAICPU的函数体。 |
| [S1743 / L2436](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2436) | <code>    std::string hostCacheTag = std::string(param.algTag) + &quot;_hostCache&quot;;</code> | 构造增量通道资源的Host缓存tag。 |
| [S1744 / L2438](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2438) | <code>    void* hostCtxPtr = nullptr;</code> | 初始化Host缓存资源地址为空。 |
| [S1745 / L2440](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2440) | <code>    uint64_t hostCtxSize = 0;</code> | 初始化Host缓存字节长度输出。 |
| [S1746 / L2442](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2442) | <code>    HcclResult hostCtxRet</code> | 声明Host缓存查询结果。 |
| [S1747 / L2444](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2444) | <code>        = HcclEngineCtxGet(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &amp;hostCtxPtr, &amp;hostCtxSize);</code> | 查CPU_TS Host序列化资源缓存；普通AllReduce即便命中也走首次/普通资源准备分支。 |
| [S1749 / L2447](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2447) | <code>    if (!increCreateChannelFlag &#124;&#124; hostCtxRet != HCCL_SUCCESS) {</code> | 非增量请求或无Host缓存时构造完整资源对象。 |
| [S1750 / L2449](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2449) | <code>        resCtxHost-&gt;commInfoPtr = static_cast&lt;void*&gt;(comm);</code> | 保存Host通信域地址，用于Device资源缓存是否陈旧的判断。 |
| [S1751 / L2451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2451) | <code>        resCtxHost-&gt;topoInfo = *topoInfo;</code> | 把已选择时得到的拓扑复制到资源对象。 |
| [S1752 / L2453](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2453) | <code>        resCtxHost-&gt;algHierarchyInfo = algHierarchyInfo;</code> | 保存算法匹配出的分层通信域信息。 |
| [S1753 / L2455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2455) | <code>        HcclResult ret = HcclAllocAlgResourceAICPU(comm, param, resRequest, resCtxHost, resPack);</code> | 准备CCL中转缓冲区、执行线程及通信通道。 |
| [S1754 / L2457](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2457) | <code>        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR(&quot;failed to alloc alg resource.&quot;), ret);</code> | 实际资源准备失败直接返回该错误。 |
| [S1756 / L2460](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2460) | <code>        std::vector&lt;char&gt; hostCtxSeq = resCtxHost-&gt;Serialize();</code> | 将资源对象序列化为字节序列。 |
| [S1757 / L2462](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2462) | <code>        ret = HcclMemcpyCtxHostToDevice(comm, param, hostCtxSeq, resCtxSequence, ctxSize);</code> | 创建Device上下文并拷贝资源序列化数据。 |
| [S1758 / L2464](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2464) | <code>        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR(&quot;failed to memcpy hostCtx to device.&quot;), ret);</code> | Device资源复制失败直接返回该错误。 |
| [S1759 / L2466](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2466) | <code>        if (increCreateChannelFlag) {</code> | 仅增量建链模式额外保存Host序列化副本。 |
| [S1760 / L2468](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2468) | <code>            CHK_RET(CacheHostCtxToEngine(comm, param.algTag, hostCacheTag, hostCtxSeq));</code> | 把Host资源字节存入CPU_TS上下文缓存。 |
| [S1761 / L2470](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2470) | <code>        }</code> | 结束条件if (increCreateChannelFlag)。 |
| [S1762 / L2472](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2472) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1764 / L2475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2475) | <code>        std::vector&lt;char&gt; cachedData(static_cast&lt;char*&gt;(hostCtxPtr), static_cast&lt;char*&gt;(hostCtxPtr) + hostCtxSize);</code> | 增量缓存命中时从Host缓存地址构造字节序列。 |
| [S1765 / L2477](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2477) | <code>        AlgResourceCtxSerializable hostCtxObj;</code> | 创建用于恢复增量资源的临时Host对象。 |
| [S1766 / L2479](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2479) | <code>        hostCtxObj.DeSerialize(cachedData);</code> | 反序列化现有Host资源副本。 |
| [S1767 / L2481](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2481) | <code>        CompReqChannelWithExistChannel(hostCtxObj.channels, resRequest);</code> | 从请求中剔除已有对端通道。 |
| [S1769 / L2484](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2484) | <code>        if (resRequest.channels[0].size() == 0) {</code> | 没有新增通道时直接复用Device上下文。 |
| [S1770 / L2486](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2486) | <code>            return ReuseCachedDeviceCtx(comm, param, resCtxSequence, ctxSize);</code> | 读取已有Device资源上下文并返回地址和大小。 |
| [S1771 / L2488](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2488) | <code>        }</code> | 结束条件if (resRequest.channels[0].size() == 0)。 |
| [S1772 / L2490](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2490) | <code>        CHK_RET(IncrementalCreateChannel(comm, param, resRequest, hostCtxObj, hostCacheTag, resCtxSequence, ctxSize));</code> | 创建缺失通道后更新Host及Device资源副本。 |
| [S1773 / L2492](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2492) | <code>    }</code> | 结束条件} else。 |
| [S1775 / L2495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2495) | <code>    HCCL_INFO(&quot;Execute GetAlgResAICPU success.&quot;);</code> | 输出运行日志，记录GetAlgResAICPU当前阶段和相关参数。 |
| [S1776 / L2497](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2497) | <code>    return HCCL_SUCCESS;</code> | 首次请求构造Host资源对象并复制Device，增量请求可筛除已存在通道处理完成，返回成功。 |
| [S1777 / L2499](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2499) | <code>}</code> | 结束GetAlgResAICPU函数体。 |


## 44. HcclMemcpyCtxHostToDevice

创建AICPU资源上下文存储并复制序列化字节

完整范围：[op_common.cc:L2504–L2529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2504-L2529)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1781 / L2504](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2504) | <code>HcclResult HcclMemcpyCtxHostToDevice(</code> | 声明HcclMemcpyCtxHostToDevice接口：创建AICPU资源上下文存储并复制序列化字节。 |
| [S1782 / L2506](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2506) | <code>    HcclComm comm, const OpParam&amp; param, const std::vector&lt;char&gt;&amp; seq, void** resCtxSequence, uint64_t&amp; ctxSize)</code> | 函数参数包含通信域句柄、算子参数、Device资源序列化地址输出、资源上下文字节长度，本行延续接口声明。 |
| [S1783 / L2508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2508) | <code>{</code> | 开始HcclMemcpyCtxHostToDevice的函数体。 |
| [S1784 / L2510](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2510) | <code>    uint64_t size = seq.size();</code> | 读取待复制序列化字节长度。 |
| [S1785 / L2512](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2512) | <code>    void* ctx = nullptr;</code> | 初始化Device上下文地址输出。 |
| [S1787 / L2515](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2515) | <code>    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, COMM_ENGINE_AICPU_TS, size, &amp;ctx));</code> | 为algTag创建AICPU_TS Device上下文存储区。 |
| [S1789 / L2518](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2518) | <code>    CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_AICPU_TS, param.algTag, seq.data(), size, 0));</code> | 从Host复制资源描述字节到Device上下文，偏移为0。 |
| [S1791 / L2521](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2521) | <code>    *resCtxSequence = ctx;</code> | 返回Device上下文地址供kernel参数引用。 |
| [S1792 / L2523](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2523) | <code>    ctxSize = size;</code> | 返回上下文字节长度供反序列化。 |
| [S1793 / L2525](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2525) | <code>    HCCL_INFO(&quot;Memcpy hostCtx to device success.&quot;);</code> | 输出运行日志，记录HcclMemcpyCtxHostToDevice当前阶段和相关参数。 |
| [S1794 / L2527](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2527) | <code>    return HCCL_SUCCESS;</code> | 创建AICPU资源上下文存储并复制序列化字节处理完成，返回成功。 |
| [S1795 / L2529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2529) | <code>}</code> | 结束HcclMemcpyCtxHostToDevice函数体。 |


## 45. HcclAllocAlgResourceAICPU

获得CCL缓冲区、线程和通道，写入可序列化资源对象

完整范围：[op_common.cc:L2534–L2570](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2534-L2570)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1799 / L2534](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2534) | <code>HcclResult HcclAllocAlgResourceAICPU(</code> | 声明HcclAllocAlgResourceAICPU接口：获得CCL缓冲区、线程和通道，写入可序列化资源对象。 |
| [S1800 / L2536](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2536) | <code>    HcclComm comm, const OpParam&amp; param, AlgResourceRequest&amp; resRequest,</code> | 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。 |
| [S1801 / L2538](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2538) | <code>    std::unique_ptr&lt;AlgResourceCtxSerializable&gt;&amp; resCtxHost, const ResPackGraphMode&amp; resPack)</code> | 函数参数包含图模式资源包、Host资源描述对象，本行延续接口声明。 |
| [S1802 / L2540](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2540) | <code>{</code> | 开始HcclAllocAlgResourceAICPU的函数体。 |
| [S1803 / L2542](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2542) | <code>    HCCL_INFO(&quot;Start to execute AllocAlgResource.&quot;);</code> | 输出运行日志，记录HcclAllocAlgResourceAICPU当前阶段和相关参数。 |
| [S1804 / L2544](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2544) | <code>    void* cclBufferAddr;</code> | 声明本域CCL缓冲区地址输出变量。 |
| [S1805 / L2546](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2546) | <code>    uint64_t cclBufferSize;</code> | 声明本域CCL缓冲区容量输出变量。 |
| [S1807 / L2549](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2549) | <code>    CHK_RET(HcclGetHcclBuffer(comm, &amp;cclBufferAddr, &amp;cclBufferSize));</code> | 查询通信域持有的CCL中转缓冲区。 |
| [S1810 / L2553](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2553) | <code>    resCtxHost-&gt;cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};</code> | 以Device内存类型保存CCL中转区地址和大小。 |
| [S1811 / L2555](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2555) | <code>    resCtxHost-&gt;notifyNumOnMainThread = resRequest.notifyNumOnMainThread;</code> | 保存算法主线程内部通知需求数。 |
| [S1812 / L2557](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2557) | <code>    resCtxHost-&gt;slaveThreadNum = resRequest.slaveThreadNum;</code> | 保存算法从线程数量。 |
| [S1813 / L2559](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2559) | <code>    UpdateAicpuTimeoutCtx(param, *resCtxHost);</code> | 根据本次算子超时更新资源对象的通知/队列等待超时。 |
| [S1814 / L2561](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2561) | <code>    resCtxHost-&gt;notifyNumPerThread = resRequest.notifyNumPerThread;</code> | 保存各从线程通知需求数。 |
| [S1816 / L2564](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2564) | <code>    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost, resPack));</code> | 申请或复用本次算法使用的执行线程/展开线程。 |
| [S1817 / L2566](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2566) | <code>    CHK_RET(HcclGetChannel(comm, param, resRequest, resCtxHost.get()));</code> | 按Peer和通信层取得通道及远端内存属性。 |
| [S1818 / L2568](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2568) | <code>    return HCCL_SUCCESS;</code> | 获得CCL缓冲区、线程和通道，写入可序列化资源对象处理完成，返回成功。 |
| [S1819 / L2570](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2570) | <code>}</code> | 结束HcclAllocAlgResourceAICPU函数体。 |


## 46. HcclGetThreadWithConfig

使用逐Thread配置申请设备主从线程，必要时申请Host展开线程

完整范围：[op_common.cc:L2573–L2634](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2573-L2634)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1821 / L2573](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2573) | <code>static HcclResult HcclGetThreadWithConfig(</code> | 声明HcclGetThreadWithConfig接口：使用逐Thread配置申请设备主从线程，必要时申请Host展开线程。 |
| [S1822 / L2575](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2575) | <code>    HcclComm comm, const OpParam&amp; param, AlgResourceRequest&amp; resRequest, u32 threadNum,</code> | 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求、主从线程总数，本行延续接口声明。 |
| [S1823 / L2577](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2577) | <code>    std::vector&lt;ThreadHandle&gt;&amp; threads, std::unique_ptr&lt;AlgResourceCtxSerializable&gt;&amp; resCtxHost, bool unfoldReady)</code> | 函数参数包含Host资源描述对象、主从线程列表、展开线程是否已存在，本行延续接口声明。 |
| [S1824 / L2579](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2579) | <code>{</code> | 开始HcclGetThreadWithConfig的函数体。 |
| [S1825 / L2581](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2581) | <code>    std::vector&lt;ThreadConfig&gt; threadConfigs(threadNum);</code> | 创建每个主从线程对应的ThreadConfig对象数组。 |
| [S1826 / L2583](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2583) | <code>    CHK_RET(static_cast&lt;HcclResult&gt;(ThreadConfigInit(threadConfigs.data(), threadNum)));</code> | 通过HCOMM接口初始化ThreadConfig ABI字段。 |
| [S1828 / L2586](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2586) | <code>    threadConfigs[0].notifyNumPerThread = resRequest.notifyNumOnMainThread + 1; // 主流上多一个用于host-device同步</code> | 主线程通知需求额外加1，最后一个槽用于Host输入就绪通知。 |
| [S1829 / L2588](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2588) | <code>    HCCL_DEBUG(&quot;[HcclGetThread] AICPU thread[0] notify num[%u].&quot;, threadConfigs[0].notifyNumPerThread);</code> | 输出调试日志，记录HcclGetThreadWithConfig当前阶段和相关参数。 |
| [S1830 / L2590](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2590) | <code>    CHK_PRT_RET(</code> | 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。 |
| [S1831 / L2592](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2592) | <code>        resRequest.notifyNumPerThread.size() &lt; threadNum - 1,</code> | 检查各从线程的通知需求数组足够长。 |
| [S1832 / L2594](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2594) | <code>        HCCL_ERROR(</code> | 输出错误日志，记录HcclGetThreadWithConfig当前阶段和相关参数。 |
| [S1833 / L2596](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2596) | <code>            &quot;[HcclGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].&quot;,</code> | 补充日志格式：[HcclGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].。 |
| [S1834 / L2598](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2598) | <code>            resRequest.notifyNumPerThread.size(), threadNum - 1),</code> | 错误日志中展示实际需求数组长度和从线程数。 |
| [S1835 / L2600](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2600) | <code>        HCCL_E_INTERNAL);</code> | 需求数组不足返回内部错误。 |
| [S1836 / L2602](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2602) | <code>    for (u32 i = 1; i &lt; threadNum; i++) {</code> | 从第1个线程开始逐一配置从线程通知容量。 |
| [S1837 / L2604](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2604) | <code>        threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i - 1];</code> | 第i个从线程取需求数组的第i-1个元素。 |
| [S1838 / L2606](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2606) | <code>        HCCL_DEBUG(&quot;[HcclGetThread] AICPU thread[%u] notify num[%u].&quot;, i, threadConfigs[i].notifyNumPerThread);</code> | 输出调试日志，记录HcclGetThreadWithConfig当前阶段和相关参数。 |
| [S1839 / L2608](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2608) | <code>    }</code> | 结束循环for (u32 i = 1; i &lt; threadNum; i++)。 |
| [S1840 / L2610](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2610) | <code>    CHK_RET(HcclThreadAcquireWithConfig(</code> | 开始以逐Thread配置接口申请线程。 |
| [S1841 / L2612](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2612) | <code>        comm, COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS, threadConfigs.data(), threads.data()));</code> | 使用AICPU执行域及TS线程类型申请主从线程。 |
| [S1843 / L2615](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2615) | <code>    if (!unfoldReady) {</code> | 尚无展开线程时申请Host CPU展开线程。 |
| [S1844 / L2617](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2617) | <code>        ThreadConfig unfoldThreadConfig;</code> | 创建Host展开线程配置结构。 |
| [S1845 / L2619](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2619) | <code>        CHK_RET(static_cast&lt;HcclResult&gt;(ThreadConfigInit(&amp;unfoldThreadConfig, 1)));</code> | 初始化该展开线程ThreadConfig ABI字段。 |
| [S1847 / L2622](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2622) | <code>        unfoldThreadConfig.notifyNumPerThread = ORDER_UNFOLD_THREAD_NOTIFY_NUM;</code> | 展开线程保序只需要ORDER_UNFOLD_THREAD_NOTIFY_NUM个通知槽。 |
| [S1848 / L2624](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2624) | <code>        CHK_RET(HcclThreadAcquireWithConfig(</code> | 开始申请Host展开线程。 |
| [S1849 / L2626](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2626) | <code>            comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &amp;unfoldThreadConfig, &amp;resCtxHost-&gt;unfoldThread));</code> | 使用COMM_ENGINE_CPU及TS线程类型输出unfoldThread句柄。 |
| [S1850 / L2628](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2628) | <code>    }</code> | 结束条件if (!unfoldReady)。 |
| [S1851 / L2630](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2630) | <code>    CHK_RET(SaveMainThreadInfo(comm, param, threads[0], resRequest.notifyNumOnMainThread + 1));</code> | 保存Device算法主线程句柄和包含Host同步槽的总通知数。 |
| [S1852 / L2632](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2632) | <code>    return HCCL_SUCCESS;</code> | 使用逐Thread配置申请设备主从线程，必要时申请Host展开线程处理完成，返回成功。 |
| [S1853 / L2634](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2634) | <code>}</code> | 结束HcclGetThreadWithConfig函数体。 |


## 47. GetMaxNotifyNum

从主从线程通知需求求最大值，用于旧接口的统一通知配置

完整范围：[op_common.cc:L2637–L2655](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2637-L2655)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1855 / L2637](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2637) | <code>static u32 GetMaxNotifyNum(const std::vector&lt;u32&gt;&amp; notifyNumPerThread, u32 initNotifyNum)</code> | 声明GetMaxNotifyNum接口：从主从线程通知需求求最大值，用于旧接口的统一通知配置。 |
| [S1856 / L2639](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2639) | <code>{</code> | 开始GetMaxNotifyNum的函数体。 |
| [S1857 / L2641](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2641) | <code>    u32 maxNotifyNum = initNotifyNum;</code> | 把初始主线程通知需求作为最大值起点。 |
| [S1858 / L2643](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2643) | <code>    for (u32 notifyNum : notifyNumPerThread) {</code> | 逐个读取从线程通知需求。 |
| [S1859 / L2645](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2645) | <code>        if (notifyNum &gt; maxNotifyNum) {</code> | 从线程需求大于当前最大值时更新。 |
| [S1860 / L2647](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2647) | <code>            maxNotifyNum = notifyNum;</code> | 保存较大的通知需求。 |
| [S1861 / L2649](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2649) | <code>        }</code> | 结束条件if (notifyNum &gt; maxNotifyNum)。 |
| [S1862 / L2651](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2651) | <code>    }</code> | 结束循环for (u32 notifyNum : notifyNumPerThread)。 |
| [S1863 / L2653](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2653) | <code>    return maxNotifyNum;</code> | 返回主从线程通知需求最大值。 |
| [S1864 / L2655](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2655) | <code>}</code> | 结束GetMaxNotifyNum函数体。 |


## 48. HcclGetAicpuThread

申请算法执行线程并复用/创建Host展开线程

完整范围：[op_common.cc:L2661–L2732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2661-L2732)；文件：`hccl/src/ops/op_common/op_common.cc`。

功能与分支：

- [S1878 / L2678](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2678)：已有unfoldThread；复用展开线程

- [S1884 / L2689](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2689)：WithConfig支持；HcclGetThreadWithConfig

- [S1886 / L2693](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2693)：旧接口；HcclThreadAcquire并使用最大notify数



| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1869 / L2661](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2661) | <code>static HcclResult HcclGetAicpuThread(</code> | 声明HcclGetAicpuThread接口：申请算法执行线程并复用/创建Host展开线程。 |
| [S1870 / L2663](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2663) | <code>    HcclComm comm, const OpParam&amp; param, AlgResourceRequest&amp; resRequest,</code> | 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。 |
| [S1871 / L2665](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2665) | <code>    std::unique_ptr&lt;AlgResourceCtxSerializable&gt;&amp; resCtxHost)</code> | 函数参数包含Host资源描述对象，本行延续接口声明。 |
| [S1872 / L2667](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2667) | <code>{</code> | 开始HcclGetAicpuThread的函数体。 |
| [S1873 / L2669](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2669) | <code>    u32 threadNum = resRequest.slaveThreadNum + 1;</code> | 本算法线程总数等于从线程数量加一个主线程。 |
| [S1874 / L2671](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2671) | <code>    std::vector&lt;ThreadHandle&gt; threads(threadNum);</code> | 创建对应数量的线程句柄输出数组。 |
| [S1875 / L2673](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2673) | <code>    bool unfoldReady = false;</code> | 默认展开线程尚未准备好。 |
| [S1876 / L2675](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2675) | <code>    ThreadHandle existingUnfoldThread = 0;</code> | 初始化可复用的展开线程句柄变量。 |
| [S1878 / L2678](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2678) | <code>    if (GetUnfoldThreadInfo(comm, param, existingUnfoldThread) == HCCL_SUCCESS) {</code> | 成功读到通信域已有展开线程时复用。 |
| [S1879 / L2680](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2680) | <code>        resCtxHost-&gt;unfoldThread = existingUnfoldThread;</code> | 将旧展开线程句柄写入本次资源对象。 |
| [S1880 / L2682](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2682) | <code>        unfoldReady = true;</code> | 标记无需重新申请展开线程。 |
| [S1881 / L2684](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2684) | <code>        HCCL_INFO(&quot;[HcclGetThread] reuse unfoldThread [%lu]&quot;, resCtxHost-&gt;unfoldThread);</code> | 输出运行日志，记录HcclGetAicpuThread当前阶段和相关参数。 |
| [S1882 / L2686](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2686) | <code>    }</code> | 结束条件if (GetUnfoldThreadInfo(comm, param, existingUnfoldThread) == HCCL_SUCCESS)。 |
| [S1884 / L2689](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2689) | <code>    if (HcommIsSupportHcclThreadAcquireWithConfig()) {</code> | 具备逐Thread配置接口时调用新接口申请主从线程。 |
| [S1885 / L2691](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2691) | <code>        CHK_RET(HcclGetThreadWithConfig(comm, param, resRequest, threadNum, threads, resCtxHost, unfoldReady));</code> | 以通知需求配置申请AICPU执行线程，必要时创建展开线程。 |
| [S1886 / L2693](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2693) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1887 / L2695](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2695) | <code>        u32 maxNotifyNum = GetMaxNotifyNum(resRequest.notifyNumPerThread, resRequest.notifyNumOnMainThread);</code> | 旧接口按所有线程需求最大值统一申请通知容量。 |
| [S1888 / L2697](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2697) | <code>        HCCL_DEBUG(&quot;[HcclGetThread] require maxNotifyNum[%u] for all AICPU threads.&quot;, maxNotifyNum);</code> | 输出调试日志，记录HcclGetAicpuThread当前阶段和相关参数。 |
| [S1889 / L2699](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2699) | <code>        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, threadNum, maxNotifyNum + 1, threads.data()));</code> | 统一通知容量加Host同步槽，申请AICPU_TS主从线程。 |
| [S1890 / L2701](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2701) | <code>        if (!unfoldReady) {</code> | 尚无展开线程时调用旧接口另行申请Host展开线程。 |
| [S1892 / L2704](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2704) | <code>            CHK_RET(</code> | 开始旧接口展开线程申请调用。 |
| [S1893 / L2706](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2706) | <code>                HcclThreadAcquire(comm, COMM_ENGINE_CPU, 1, ORDER_UNFOLD_THREAD_NOTIFY_NUM, &amp;resCtxHost-&gt;unfoldThread));</code> | 使用CPU执行域和固定保序通知数申请展开线程。 |
| [S1894 / L2708](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2708) | <code>        }</code> | 结束条件if (!unfoldReady)。 |
| [S1895 / L2710](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2710) | <code>        CHK_RET(SaveMainThreadInfo(comm, param, threads[0], maxNotifyNum + 1));</code> | 保存主线程句柄和最大通知容量加Host同步槽。 |
| [S1896 / L2712](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2712) | <code>    }</code> | 结束条件} else。 |
| [S1897 / L2714](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2714) | <code>    if (!unfoldReady) {</code> | 本次新建展开线程才需要写入Host缓存。 |
| [S1898 / L2716](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2716) | <code>        CHK_RET(SaveUnfoldThreadInfo(comm, param, resCtxHost-&gt;unfoldThread));</code> | 按通信域名保存新展开线程句柄。 |
| [S1899 / L2718](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2718) | <code>    }</code> | 结束条件if (!unfoldReady)。 |
| [S1900 / L2720](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2720) | <code>    HCCL_INFO(&quot;[HcclGetThread] unfoldThread [%lu]&quot;, resCtxHost-&gt;unfoldThread);</code> | 输出运行日志，记录HcclGetAicpuThread当前阶段和相关参数。 |
| [S1901 / L2722](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2722) | <code>    HCCL_DEBUG(&quot;threads ptr is %p\n&quot;, threads.data());</code> | 输出调试日志，记录HcclGetAicpuThread当前阶段和相关参数。 |
| [S1902 / L2724](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2724) | <code>    for (u32 i = 0; i &lt; threadNum; i++) {</code> | 遍历本次申请的所有算法线程。 |
| [S1903 / L2726](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2726) | <code>        resCtxHost-&gt;threads.push_back(threads[i]);</code> | 将每个主从线程句柄写入可序列化资源列表。 |
| [S1904 / L2728](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2728) | <code>    }</code> | 结束循环for (u32 i = 0; i &lt; threadNum; i++)。 |
| [S1905 / L2730](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2730) | <code>    return HCCL_SUCCESS;</code> | 申请算法执行线程并复用/创建Host展开线程处理完成，返回成功。 |
| [S1906 / L2732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2732) | <code>}</code> | 结束HcclGetAicpuThread函数体。 |


## 49. HcclGetThread

按engine选择AICPU线程资源或Host流包装/图模式从流

完整范围：[op_common.cc:L2735–L2782](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2735-L2782)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1908 / L2735](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2735) | <code>HcclResult HcclGetThread(</code> | 声明HcclGetThread接口：按engine选择AICPU线程资源或Host流包装/图模式从流。 |
| [S1909 / L2737](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2737) | <code>    HcclComm comm, const OpParam&amp; param, AlgResourceRequest&amp; resRequest,</code> | 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。 |
| [S1910 / L2739](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2739) | <code>    std::unique_ptr&lt;AlgResourceCtxSerializable&gt;&amp; resCtxHost, const ResPackGraphMode&amp; resPack)</code> | 函数参数包含图模式资源包、Host资源描述对象，本行延续接口声明。 |
| [S1911 / L2741](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2741) | <code>{</code> | 开始HcclGetThread的函数体。 |
| [S1912 / L2743](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2743) | <code>    resCtxHost-&gt;isHcclThreadAcquireWithConfigSupported = HcommIsSupportHcclThreadAcquireWithConfig();</code> | 把ThreadAcquireWithConfig能力标志写到资源对象，Device据此定位Host同步槽。 |
| [S1913 / L2745](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2745) | <code>    if ((param.engine == COMM_ENGINE_AICPU_TS) &#124;&#124; (param.engine == COMM_ENGINE_CPU)) {</code> | AICPU_TS或CPU引擎走AICPU线程资源获取。 |
| [S1914 / L2747](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2747) | <code>        CHK_RET(HcclGetAicpuThread(comm, param, resRequest, resCtxHost));</code> | 调用AICPU主从线程及Host展开线程准备函数。 |
| [S1915 / L2749](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2749) | <code>    } else {</code> | 上述条件不成立时进入替代分支。 |
| [S1917 / L2752](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2752) | <code>        ThreadHandle thread;</code> | 声明其它引擎主Thread句柄。 |
| [S1918 / L2754](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2754) | <code>        CHK_RET(</code> | 开始按用户ACL stream包装其它引擎主线程。 |
| [S1919 / L2756](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2756) | <code>            HcclThreadAcquireWithStream(comm, param.engine, param.stream, resRequest.notifyNumOnMainThread, &amp;thread));</code> | 用算法主线程通知需求配置流包装，并输出Thread句柄。 |
| [S1920 / L2758](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2758) | <code>        resCtxHost-&gt;threads.push_back(thread);</code> | 将该主线程加入资源对象。 |
| [S1922 / L2761](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2761) | <code>        u32 maxNotifyNum = GetMaxNotifyNum(resRequest.notifyNumPerThread, 0);</code> | 计算其它引擎从线程的最大通知需求。 |
| [S1923 / L2763](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2763) | <code>        CHK_RET(GeGetThread(comm, param, resRequest, resCtxHost, resPack, maxNotifyNum));</code> | 从GE资源包或普通接口准备其它引擎的从线程。 |
| [S1924 / L2765](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2765) | <code>    }</code> | 结束条件} else。 |
| [S1926 / L2768](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2768) | <code>    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {</code> | 仅DEBUG日志级别开启时遍历打印线程信息。 |
| [S1927 / L2770](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2770) | <code>        HCCL_DEBUG(&quot;[HcclGetThread] slaveThreadNum[%u]&quot;, resRequest.slaveThreadNum);</code> | 输出调试日志，记录HcclGetThread当前阶段和相关参数。 |
| [S1928 / L2772](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2772) | <code>        for (u32 i = 0; i &lt; resRequest.slaveThreadNum + 1; i++) {</code> | 遍历所有主从线程句柄进行调试输出。 |
| [S1929 / L2774](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2774) | <code>            HCCL_DEBUG(&quot;[HcclGetThread] threads[%u]=[%llu]&quot;, i, resCtxHost-&gt;threads[i]);</code> | 输出调试日志，记录HcclGetThread当前阶段和相关参数。 |
| [S1930 / L2776](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2776) | <code>        }</code> | 结束循环for (u32 i = 0; i &lt; resRequest.slaveThreadNum + 1; i++)。 |
| [S1931 / L2778](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2778) | <code>    }</code> | 结束条件if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG)))。 |
| [S1932 / L2780](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2780) | <code>    return HCCL_SUCCESS;</code> | 按engine选择AICPU线程资源或Host流包装/图模式从流处理完成，返回成功。 |
| [S1933 / L2782](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2782) | <code>}</code> | 结束HcclGetThread函数体。 |


## 50. SaveMainThreadInfo

按算法tag保存主Thread和通知容量

完整范围：[op_common.cc:L2834–L2867](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2834-L2867)；文件：`hccl/src/ops/op_common/op_common.cc`。

| 源码定位 | 原代码 | 这一行的功能 |
|---|---|---|
| [S1984 / L2834](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2834) | <code>HcclResult SaveMainThreadInfo(HcclComm comm, const OpParam&amp; param, ThreadHandle thread, u32 notifyNum)</code> | 声明SaveMainThreadInfo接口：按算法tag保存主Thread和通知容量。 |
| [S1985 / L2836](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2836) | <code>{</code> | 开始SaveMainThreadInfo的函数体。 |
| [S1986 / L2838](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2838) | <code>    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);</code> | 主线程缓存长度为一个ThreadHandle加一个u32通知数。 |
| [S1987 / L2840](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2840) | <code>    void* ctx = nullptr;</code> | 初始化Host主线程信息上下文地址。 |
| [S1989 / L2843](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2843) | <code>    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, &amp;ctx));</code> | 按算法tag创建CPU_TS Host存储区。 |
| [S1991 / L2846](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2846) | <code>    ThreadHandle* threadPtr = reinterpret_cast&lt;ThreadHandle*&gt;(ctx);</code> | 将Host上下文起始地址解释为ThreadHandle存储位置。 |
| [S1992 / L2848](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2848) | <code>    *threadPtr = thread;</code> | 把算法Device主线程句柄写入上下文首段。 |
| [S1994 / L2851](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2851) | <code>    char* curPtr = reinterpret_cast&lt;char*&gt;(ctx);</code> | 将上下文地址转成字节指针便于移动偏移。 |
| [S1995 / L2853](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2853) | <code>    curPtr += sizeof(ThreadHandle);</code> | 跳过ThreadHandle长度，定位通知数存储段。 |
| [S1996 / L2855](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2855) | <code>    u32* notifyNumPtr = reinterpret_cast&lt;u32*&gt;(curPtr);</code> | 将通知数存储段解释为u32指针。 |
| [S1997 / L2857](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2857) | <code>    *notifyNumPtr = notifyNum;</code> | 保存主线程总通知槽数。 |
| [S1998 / L2859](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2859) | <code>    HCCL_INFO(</code> | 输出运行日志，记录SaveMainThreadInfo当前阶段和相关参数。 |
| [S1999 / L2861](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2861) | <code>        &quot;[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]&quot;, threadPtr, thread,</code> | 补充日志格式：[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]&quot;, threadPtr, thread。 |
| [S2000 / L2863](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2863) | <code>        notifyNumPtr, notifyNum);</code> | 提供上述日志的实参，涉及主线程通知容量。 |
| [S2001 / L2865](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2865) | <code>    return HCCL_SUCCESS;</code> | 按算法tag保存主Thread和通知容量处理完成，返回成功。 |
| [S2002 / L2867](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2867) | <code>}</code> | 结束SaveMainThreadInfo函数体。 |
