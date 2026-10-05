# AllReduce分阶段调用关系与分支

[返回阅读指南](../READING_GUIDE.zh-CN.md)。以下关系由固定快照逐函数审读；路径定位已转换为带逐行注释的固定提交行号。Host→AICPU为runtime发射关系，虚调用按本例注册的executor/template/UB传输类型展开。树内简写行号对应审读快照S行；树后定位表和正文链接对应新增注释后的L行。

[返回调用关系导航](CALL_RELATIONS.zh-CN.md)。公共调度与设备入口，第1/1页。

# AllReduce dispatch / launch / resource 调用关系实证

未链接的简写行号取固定审读快照 `f8af6a3` 的S行；链接已转换为带本次逐行注释源码的精确L行。这里以 Ascend 950、OPBASE、AICPU_TS、Mesh1D、`AicpuAllReduceSoleMeshOneShot` 为主例。代码的其它引擎/图模式/缓存分支均明确列出，但没有把它们伪装成同一次调用的连续步骤。

以下缩写：

- O = `hccl/src/ops/op_common/op_common.cc`
- K = `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc`
- OL = `hccl/src/ops/op_common/order_launch.cc`
- IC = `hccl/src/common/inconsistent_check.cc`
- R = `hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc`
- RH = `hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h`
- PDL = `hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc`

## Host 算法选择、执行及资源树

```text
AllReduceOutPlaceCommon -> Selector [op_common.cc:S158–S223]
  ├─163 有CommGetStatus能力：HcclCommGetStatus
  │   └─165 非READY：返回HCCL_E_SUSPENDING [167]
  ├─173 HcclCalcTopoInfo [op_common.cc:S1325–S1353]
  │   ├─1332 HcclEngineCtxGet(tag, CPU_TS)
  │   ├─1333 NOT_FOUND/PARA
  │   │   └─1336 InitRankInfo -> 1338 topoInfo.Serialize
  │   │       -> 1341 HcclEngineCtxCreate(CPU_TS) -> 1342 memcpy_s
  │   └─1346 其它返回结果走恢复逻辑：1349 DeSerialize
  │       -> 1350 make_unique(移动构造恢复出的拓扑对象)
  ├─174 topLevelUboe：opExecuteConfig=AICPU_TS [175]
  ├─179 编译插件时 HCCL_ALGO_PLUGIN_SELECTOR_FAST_PATH（插件旁支）
  ├─183 新selector启用且算子受支持
  │   ├─是：SelectorEngine::Global()->Run [184]
  │   └─否：ExecuteSelector::Run [187]（传统规则选择，本例算法细分见AllReduce selector图）
  ├─189 checkValidAlgName -> 190 SetCommEngine
  ├─192 AIV_ONLY却最终非AIV：返回NOT_SUPPORT [197]
  ├─200 AICPU_TS/CPU：LoadAICPUKernel [202]（保证二进制入口已加载）
  ├─207 AIV：RegisterKernel [209]
  ├─212 SetOpParamAlgTag -> 214 SetExecTimeout -> 216 SetMultipleDimensionSplitRatio
  └─219 CANN≥9.2：CheckCcuParamAndFallback（CCU配置协商旁支）

HcclExecOp [op_common.cc:S783–S962]
  ├─798 pluginSelected且编译插件
  │   └─799 ExecutePluginAlgorithm -> 800 ReportOpProfilingInfo -> return [801]
  ├─809 SetOpParamFallbackTag
  ├─810 HcclEngineCtxGet(fallbackTag,engine)命中
  │   └─812–817 恢复FallbackCtxData中的algName/opExecuteConfig
  │       -> 818 SetCommEngine -> 819 SetOpParamAlgTag
  │       -> 820 HcclExecOp(缓存回退算法)递归 -> return [821]
  ├─825 sprintf_s(param.algName) -> 834 sprintf_s(param.commModeTag)
  ├─841 CollAlgExecRegistryV2::Instance [coll_alg_v2_exec_registry.cc:S15–S19] -> GetAlgExec [coll_alg_v2_exec_registry.cc:S33–S41]
  │   ├─35 算子type或算法tag无工厂：return nullptr [37]
  │   └─40 execCreators_[type][tag]() -> DefaultExecCreatorV2<P> [coll_alg_v2_exec_registry.h:S26–S32]
  │       └─31 new(nothrow) P()；P由REGISTER_EXEC_V2注册绑定（下文）
  ├─846 创建Host资源对象 -> 848 保存HCOMM批传输能力
  ├─857 AICPU_TS/CPU
  │   └─858 HcclThreadAcquireWithStream(CPU_TS,param.stream)
  │       -> 860 HcclThreadExportToCommEngine(AICPU_TS)：用户流可在Device被引用
  ├─865 HcclGetAlgRes [op_common.cc:S1415–S1481]
  │   ├─1424 executor.GetAlgoMeta
  │   ├─1432 TryReuseResource [op_common.cc:S1377–S1410]
  │   │   ├─1383 BatchSendRecv OPBASE：标记增量并返回NOT_FOUND [1384–1385]
  │   │   ├─1388 非OPBASE且非CCU：返回NOT_FOUND [1389]
  │   │   ├─1394 AIV：上下文查询engine转换为CPU_TS [1396]
  │   │   ├─1397 CPU：上下文查询engine转换为AICPU_TS [1399]
  │   │   ├─1402 EngineCtxGet成功
  │   │   │   └─1404–1406 返回地址和ctxSize，标记isResourceReused -> return [1407]
  │   │   └─1409 未命中：NOT_FOUND；HcclGetAlgRes继续分配
  │   ├─1433 资源复用命中立即return，跳过后续CalcRes/申请
  │   ├─1438 NeedInconsistentCheck [inconsistent_check.cc:S18–S35]
  │   ├─1443 executor.CalcAlgHierarchyInfoV2：物理拓扑→算法通信域
  │   ├─1447 executor.CalcRes：需求计算，不是实际HCOMM建链
  │   ├─1449 GetAlgResWithEngine [op_common.cc:S1582–S1639]
  │   │   ├─1589 RESERVED /1595 CPU_TS /1597 AICPU：本函数分支为空
  │   │   ├─1591 CPU：GetAlgResDPU [1592]
  │   │   ├─1599 AICPU_TS：GetAlgResAICPU [1600]（本例）
  │   │   ├─1603 AIV：GetAlgResAiv [1604]
  │   │   ├─1605 CCU：GetAlgResCcu [1607]
  │   │   │   ├─CANN≥9.2且ret为UNAVAIL/SUCCESS [1611]
  │   │   │   │   └─1614 CheckCcuResNegotiation
  │   │   │   │       ├─1615 UNAVAIL：1618 ReleaseCcuAcquiredChannels ->return UNAVAIL
  │   │   │   │       └─1621 其它协商错误由CHK_RET返回
  │   │   │   ├─1623 其它CCU资源错误直接返回
  │   │   │   └─旧CANN：1626 UNAVAIL直接向上返回，其它CHK_RET
  │   │   └─1631 未识别engine：返回PARA [1635]
  │   ├─1452 分配UNAVAIL：保留专用资源不足码向上传递
  │   ├─1455 其它错误直接返回
  │   └─1474 启用一致性校验
  │       └─1476 FillOpExchangeInfo ->1477 CompareOpExchangeInfos [inconsistent_check.cc:S52–S69]
  ├─866 资源UNAVAIL：868 FallbackOp ->return [869]
  ├─871 其它资源错误直接返回
  ├─874 cacheValid=isResourceReused（资源对象cache策略会用此值）
  ├─883 ConstructHcclDfxOpInfo ->885 HcclDfxRegOpInfoByCommId（维测边界）
  ├─889 AICPU_TS/CPU
  │   └─892 GetMainThreadInfo [op_common.cc:S2042–S2060]
  │       ->894 HcclThreadExportToCommEngine(CPU_TS)：Host可给Device主线程发通知
  │       ->896 param.opThread=用户流导出到Device的句柄
  ├─902 AICPU_TS/CPU（本例）
  │   └─905 GetUnfoldThreadInfo [op_common.cc:S2022–S2040]
  │       ->907 CaptureSlaveStreams（流捕获辅助边界）
  │       ->910 读取AICPU task缓存开关
  │       ->911 HcclAicpuKernelEntranceLaunch [op_common.cc:S1028–S1165]
  ├─914 AIV：919 HcclAivKernelEntranceLaunch
  │   ->920 ExecuteAivCacheLogic ->921 HcclReportAivKernel
  ├─922 CCU
  │   ├─923 若资源复用：928 DeSerialize ->931 HcclThreadAcquireWithStream
  │   │   ->937 替换本次主Thread ->941 OFFLOAD时GeReuseResource
  │   ├─945 有从线程：946 CaptureSlaveStreams
  │   └─948 executor.Orchestrate（Host编排）
  ├─949 其它engine：复用时DeSerialize [954] ->956 executor.Orchestrate
  └─959 ReportOpProfilingInfo ->return [961]
```

## AICPU资源准备深入到 HCOMM 公共ABI

```text
GetAlgResAICPU [op_common.cc:S1736–S1777]
  ├─1747 查询algTag+"_hostCache"的CPU_TS资源副本
  ├─1749 非增量 或 无Host副本（本例非增量，一定走此分支）
  │   ├─1750–1752 保存commInfoPtr、topoInfo、algHierarchyInfo
  │   ├─1753 HcclAllocAlgResourceAICPU [op_common.cc:S1799–S1819]
  │   │   ├─1807 HcclGetHcclBuffer ->1810 cclMem
  │   │   ├─1811–1814 通知需求、从线程数、超时
  │   │   ├─1816 HcclGetThread [op_common.cc:S1908–S1933]
  │   │   │   ├─1912 保存WithConfig能力
  │   │   │   ├─1913 AICPU_TS/CPU：HcclGetAicpuThread [op_common.cc:S1869–S1906]
  │   │   │   │   ├─1873 threadNum=slaveThreadNum+1
  │   │   │   │   ├─1878 GetUnfoldThreadInfo成功：复用Host展开线程
  │   │   │   │   ├─1884 WithConfig能力存在
  │   │   │   │   │   └─1885 HcclGetThreadWithConfig [op_common.cc:S1821–S1853]
  │   │   │   │   │       ├─1826 ThreadConfigInit
  │   │   │   │   │       ├─1828 主线程notify=算法需求+1（Host输入同步槽）
  │   │   │   │   │       ├─1831 检查从线程notify数组足够
  │   │   │   │   │       ├─1836–1837 各从线程按各自需求填config
  │   │   │   │   │       ├─1840 HcclThreadAcquireWithConfig(AICPU,TS)
  │   │   │   │   │       ├─1843 无展开线程：1845 ThreadConfigInit
  │   │   │   │   │       │   ->1848 HcclThreadAcquireWithConfig(CPU,TS)
  │   │   │   │   │       └─1851 SaveMainThreadInfo [op_common.cc:S1984–S2002]
  │   │   │   │   ├─1886 旧接口：1887 GetMaxNotifyNum [op_common.cc:S1855–S1864]
  │   │   │   │   │   ->1889 HcclThreadAcquire(AICPU_TS,maxNotify+1)
  │   │   │   │   │   ->1890 无展开线程时1893 HcclThreadAcquire(CPU)
  │   │   │   │   │   ->1895 SaveMainThreadInfo
  │   │   │   │   ├─1897 新建展开线程：1898 SaveUnfoldThreadInfo [op_common.cc:S2004–S2020]
  │   │   │   │   └─1902–1903 主从Thread句柄写入资源对象
  │   │   │   └─1915 其它engine：1919 AcquireWithStream
  │   │   │       ->1922 GetMaxNotifyNum ->1923 GeGetThread（非本例）
  │   │   └─1817 HcclGetChannel [op_common.cc:S2064–S2095]
  │   │       ├─2069 OFFLOAD：2071 RegGraphModeBuffers（OPBASE不走）
  │   │       ├─2076 遍历算法层
  │   │       │   └─2082/2084 按本地端点位置分Device/Host请求组
  │   │       ├─2089 HcclGetChannelImpl(Device,AICPU_TS) [op_common.cc:S2155–S2190]
  │   │       └─2092 HcclGetChannelImpl(Host,CPU)
  │   │           ├─2161 请求为空：2163 成功跳过（本例Host组常为空）
  │   │           ├─2169 OFFLOAD：挂已注册用户内存句柄
  │   │           ├─2178 AddExchangeInfo [op_common.cc:S1550–S1560]
  │   │           │   └─1553 若校验：1555 FillOpExchangeInfo
  │   │           │       ->1556 HcclCommAddExchangeInfo（HCOMM域元信息）
  │   │           ├─2179 HcclChannelAcquire（HCOMM域资源入口，另见跨仓链）
  │   │           └─2183 每个句柄：BuildChannelInfo [op_common.cc:S2099–S2151]
  │   │               ├─2105–2110 有效标志、Peer、协议、位置、notify、句柄
  │   │               ├─Host编译：2117 RankGraphGetEndpointInfo(BW_COEFF)
  │   │               │   ->2121 系数为0时INTERNAL
  │   │               │   ->2125 RankGraphGetEndpointInfo(DIE_ID)
  │   │               │   ->2127 成功保存dieId，否则warning并继续
  │   │               ├─2139 HcclChannelGetHcclBuffer（HCOMM远端注册内存描述）
  │   │               ├─2140 保存remoteCclMem
  │   │               └─2147 OFFLOAD：GetGraphModeBuffers（非本例）
  │   ├─1756 Serialize ->1757 HcclMemcpyCtxHostToDevice [op_common.cc:S1781–S1795]
  │   │   └─1787 HcclEngineCtxCreate(AICPU_TS,algTag,size)
  │   │       ->1789 HcclEngineCtxCopy(Host字节→DeviceCtx,offset=0)
  │   │       ->1791–1792 返回Device地址及长度
  │   └─1759 若是增量申请：CacheHostCtxToEngine [op_common.cc:S1641–S1663]
  │       └─1645 CPU_TS创建Host副本 ->1655 memcpy_s
  │           ->失败时1649/1658/1659请求清理对应Device/Host上下文
  └─1762 增量申请且Host副本存在（BatchSendRecv旁支，本AllReduce例不走）
      ├─1766 DeSerialize
      ├─1767 CompReqChannelWithExistChannel [op_common.cc:S1355–S1372]：按已有远端rank剔除请求
      ├─1769 无新增Peer：ReuseCachedDeviceCtx [op_common.cc:S1665–S1682]
      │   └─1670 CPU改查AICPU_TS，其它按param.engine查上下文
      └─1772 有新增Peer：IncrementalCreateChannel [op_common.cc:S1684–S1732]
          └─1688 HcclGetChannel ->1691/1693 Destroy旧DeviceCtx
              ->1698 Serialize ->1699 HcclMemcpyCtxHostToDevice
              ->1708 Destroy旧HostCache ->1713 Create新HostCache ->1723 memcpy_s
              （各失败清理以源码为准，不能声称所有回滚都保证成功）
```

## 输入/输出依赖与保序发射

```text
HcclAicpuKernelEntranceLaunch [op_common.cc:S1028–S1165]
  ├─1036 param.resCtx=Device序列化资源地址
  ├─1037 用户流结果等待槽=HOST_WAIT_AICPU_NOTIFYIDX
  ├─1039 CPU：1041 HcclTaskRegister(HcclLaunchDPUKernel)（非本例）
  ├─1045/1046 可选P2P专用发射接口+Send/Receive（AllReduce不走）
  │   └─1052–1099 填HcclOpDesc、HcclKernelFuncInfo及timeout
  │       ->1101 HcclAicpuKernelLaunch ->return [1104]
  ├─1109 HcommThreadNotifyRecordOnThread(cpuTsThread,exportedDeviceMain,mainNotifyNum-1)
  │   └─与K:305 Device主线程输入等待配对
  ├─1117–1120 capture?ACLGRAPH:OFFLOAD?GE:OPBASE
  ├─1125 ACLGRAPH：创建event0/event1 guard
  ├─1130 HcclOrderLaunchToOrderStream [order_launch.cc:S152–S213]
  │   ├─156 GetOrderLaunchModeName ->157 GetOrderLaunchHostThreadType
  │   ├─162 无DedicatedThreadAcquire能力：句柄置0，成功跳过
  │   ├─168 HcclDedicatedThreadAcquire(Host保序线程)
  │   ├─171 Host线程为0：句柄置0，成功跳过
  │   ├─181 HcclThreadExportToCommEngine(AICPU_TS)
  │   ├─188 ACLGRAPH：AclgraphOrderLaunchEventToOrderStream [order_launch.cc:S100–S116]
  │   │   └─105 OpLaunchGetOrderStreams [order_launch.cc:S86–S98]
  │   │       ->查询Host保序流 [order_launch.cc:S42–S64]/展开流 [order_launch.cc:S18–S40]
  │   │       ->107 RecordEvent(展开流) ->111 WaitEvent(Host保序流)
  │   ├─194 HcclDedicatedThreadAcquire(Device保序线程)
  │   ├─196 Device线程为0：句柄清0，成功跳过
  │   └─209 Host保序线程Record到展开线程
  │       ->210 展开线程Wait（第一阶段）
  ├─1136 AicpuKernelLaunch [op_common.cc:S1167–S1250]
  │   ├─1176 aclrtBinaryGetFunction(g_binKernelHandle,"HcclLaunchAicpuKernel")
  │   ├─1184 aclrtKernelArgsInit
  │   ├─1195 aclrtKernelArgsAppend(OpParam+varMemSize)
  │   ├─1203 aclrtKernelArgsFinalize
  │   ├─1213–1223 派生启动timeout，构造ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT
  │   ├─1224 numBlocks=1
  │   ├─1229 无ThreadResGetInfo或OFFLOAD
  │   │   └─1230 aclrtLaunchKernelWithConfig(用户流)
  │   └─1233 ThreadResGetInfo查unfoldStream
  │       ├─1234 NOT_SUPPORT：1235 在用户流发射
  │       ├─1236 其它失败：1237 直接返回ret1
  │       └─1239 成功：在Host展开流发射
  │           -> CANN runtime 调用Device二进制入口HcclLaunchAicpuKernel [kernel_launch.cc:S353]
  ├─1140 HcclOrderLaunchToKernelStream [order_launch.cc:S229–S262]
  │   ├─238 无专用线程能力或245 Host保序线程为0：成功跳过
  │   ├─242 取得Host专用保序线程
  │   ├─253 等待Device入口的HcclOrderLaunchNotifyRecord [kernel_launch.cc:S331–S348]
  │   │   └─kernel_launch.cc:S339 Host/Device保序线程均有效才Record [341]
  │   └─256 ACLGRAPH：AclgraphOrderLaunchEventToKernelStream [order_launch.cc:S118–S134]
  │       └─125 RecordEvent(Host保序流) ->129 WaitEvent(展开流)
  ├─1145 HcclReportAicpuKernel（profiling辅助边界）
  ├─1154–1160 派生并设置Host结果等待超时
  └─1162 HcclThreadNotifyWaitOnThreadDefault(cpuTsThread,resultSlot,timeout)
      └─hcomm_primitives_dl.cc:S176–S182 有默认timeout机制→HcommThreadNotifyWaitOnThreadWithDefaultTimeout
          否则→HcommThreadNotifyWaitOnThread(...,fallbackTimeout)
```

## Device入口与执行器编排/缓存树

```text
HcclLaunchAicpuKernel [kernel_launch.cc:S353–S769]
  ├─359 sched_setscheduler(SCHED_OTHER,priority=0)失败：return1
  ├─364 参数null：return1
  ├─369 HcommAcquireComm(commName)：获取设备域占用标记
  ├─376 HcclOrderLaunchNotifyRecord [kernel_launch.cc:S331–S348]（配对Host第二阶段保序）
  ├─380 非OpsV2：旧Scatter元信息/异常回调登记（950不走）
  ├─404 IsOpsV2 [kernel_launch.cc:S243–S259]（opv2_前缀或out-place芯片）
  │   ├─408 CommStatus能力存在：409 HcclCommGetStatus
  │   │   ├─414 SUSPENDING：415 HcommReleaseComm ->return301 [421]
  │   │   └─423 非READY：return1 [425]
  │   ├─434 非BatchSendRecv：437 g_cacheManager.Get [kernel_launch.cc:S116–S134]
  │   │   └─118 ExtractCommName [kernel_launch.cc:S196–S207]
  │   │       ->123 GetOrCreateComm [kernel_launch.cc:S211–S220]
  │   │       ->126 CommDomainCache.Get [kernel_launch.cc:S68–S73]
  │   ├─438 缓存对象存在且IsResCtxCacheReusable [kernel_launch.h:26–29]
  │   │   └─cacheValid && cached.commInfoPtr==param.hcclComm
  │   │       ->452 shared_ptr持有资源对象，免反序列化
  │   ├─453 未命中或陈旧
  │   │   └─457 DeserializeResCtx [kernel_launch.cc:S229–S236]
  │   │       ->234 AlgResourceCtxSerializable.DeSerialize
  │   │       ->458 g_cacheManager.Put [kernel_launch.cc:S137–S147]
  │   │       ->459 使用新恢复的资源对象
  │   ├─468 BatchSendRecv：469 每次DeserializeResCtx（非本例）
  │   ├─477–486 变长描述恢复按命令布局选择；AllReduce没有此恢复分支
  │   ├─494 thread=resCtx.threads[0]
  │   ├─495 HcommBatchModeStart(algTag)
  │   ├─503 ConvertToHcclDfxOpInfo ->507 HcclDfxRegOpInfoByCommId
  │   ├─513 ProfilingReportKernelStartTask
  │   ├─525 有TaskCacheLookup能力：526 AicpuTaskCachePolicy.IsAicpuTaskCacheEnable
  │   ├─541 允许task缓存
  │   │   ├─548 本次input/output地址数组 ->551 GetInputOutputInfoForCache有效容量
  │   │   ├─559 GetAicpuTaskCacheTag ->563 HcommAicpuTsTaskCacheLookup
  │   │   ├─568 miss
  │   │   │   └─574 TaskCacheStart(当前地址/长度)
  │   │   │       ->582 OpOrchestrate [kernel_launch.cc:S277–S329]
  │   │   │       ->592 EnforceLaunchTask [kernel_launch.cc:S262–S273]
  │   │   │       │   └─264 BatchModeEnd实际提交 ->268 BatchModeStart恢复批模式
  │   │   │       ->600 TaskCacheEnd
  │   │   │       ├─608 失败：611 TaskCacheClear(若有能力) ->return原cacheRet [613]
  │   │   │       └─618 AddCommTagMap(域→缓存tag绑定)
  │   │   └─619 hit：624 TaskCacheExecute(刷新当前用户地址与长度，回放缓存SQE)
  │   │       └─本次没有OpOrchestrate/模板KernelRun调用
  │   └─627 未启用task缓存：630 OpOrchestrate [kernel_launch.cc:S277–S329]
  │       ├─282 支持队列超时接口：283 HcclThreadResAcquireTimeOut(fullTimeout)
  │       ├─287 支持通知超时接口：288 HcclSetNotifyWaitTimeOut(waitTimeout)
  │       ├─293 Host输入通知槽初值=notifyNumOnMainThread
  │       ├─294 旧申请接口：295–297 求各线程最大通知数
  │       ├─305 HcclThreadNotifyWaitOnThreadDefault(DeviceMain,inputSlot)
  │       │   └─与O:1109用户流输入Record匹配
  │       ├─308 SetExecTimeout ->312 InitHcommBatchTransferOnThreadSupported
  │       ├─316 CollAlgExecRegistryV2.GetAlgExec(ALLREDUCE,algName)
  │       └─323 executor.Orchestrate ->InsV2AllReduceSoleExecutor::Orchestrate
  ├─633–640 新流程向用户流导出线程槽0排入完成通知
  ├─643 ProfilingReportKernelEndTask ->650 ProfilingReportDeviceOp
  ├─656 BatchModeEnd提交剩余任务
  ├─660 非OpsV2旧流程（950不走）
  │   └─661 旧CollAlgExecRegistry.GetAlgExec
  │       ->666–672 解释旧AlgResourceCtx及结构尾部ThreadHandle数组
  │       ->673 BatchModeStart
  │       ├─678 已导出用户流：679 ProfilingInit ->685首任务report
  │       │   ->695 ThreadNotifyWaitOnThread(DeviceMain)
  │       └─697 未导出用户流：AclrtNotifyWaitOnThread
  │       ->704 旧ExecutorBase.Orchestrate
  │       ├─709 已导出用户流：725旧op report ->733记录结果通知
  │       │   ->736末任务report ->741 BatchModeEnd ->746 ProfilingEnd
  │       └─751 未导出用户流：AclrtNotifyRecordOnThread ->756 BatchModeEnd
  └─763 HcommReleaseComm(commName) ->return0 [768]
```

重要的读码边界：`HcommAcquireComm` 的占用保护是运行时状态标记，并非这里自动建立一个覆盖所有提前返回的RAII清理器；`HcclLaunchAicpuKernel` 有很多错误分支在尾部ReleaseComm以前返回。此导读按源码说清实际分支，不保证所有错误都清理资源。普通Host API成功、入口展开成功、BatchModeEnd提交成功都不能等价为硬件已完成AllReduce；输入/结果通知与用户流上的执行顺序承担数据依赖。

## 一致性核对

`HcclGetAlgRes:1438 -> NeedInconsistentCheck [[inconsistent_check.cc:L19–L50](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L19-L50)]`：有交换能力且非空域才考虑检查；显式off(-1)不查；默认/first(0)、OPBASE且CheckCtxStatus已存在时，普通算子不查；BatchSendRecv增量请求不按“已存在”跳过。资源快复用在1433已提前返回，这里主要在实际准备资源路径发生。

`AddExchangeInfo [[op_common.cc:L2087–L2107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2087-L2107)] -> FillOpExchangeInfo [[op_common.cc:L1960–L2031](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1960-L2031)] -> FillOpExchangeInfoWithDataDes [[op_common.cc:L2034–L2082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2034-L2082)]`：本AllReduce进入default，dtype在1541、count在1542；交换的是元信息，不是用户张量。

`CompareOpExchangeInfos [[inconsistent_check.cc:L82–L116](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L82-L116)]`：非CCU按各算法层通道，CCU按kernelInfo通道，进入`InconsistentCheckParams [[inconsistent_check.cc:L119–L269](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L119-L269)]`；空层跳过、远端长度0跳过、非0长度与sizeof不符报PARA；后续检查字段与源码对应：CCL容量92、root97、opType101（辅助`InconsistentCheckOpType:148–174`）、执行配置102、归约108、dtype113、count118、AIV核数123（只warning）、group129、tag133（Group模式可跳过tag检查）。不一致上报工具`ReportOpExchangeInfoCheckFailed:176–206`和枚举名字映射`GetOpTypeName:208`是诊断边界，未递归注释所有日志/错误上报实现。

## 注册关系与HCOMM链接边界

`ins_v2_all_reduce_sole_executor.cc:378` 的 `REGISTER_EXEC_V2(ALLREDUCE,AicpuAllReduceSoleMeshOneShot,InsV2AllReduceSoleExecutor,TopoMatchOneLevel,InsTempAllReduceMesh1DOneShot)` 将算法名字映射为具体模板执行器。宏核心 [coll_alg_v2_exec_registry.h:L92–L97](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h#L92-L97) 中 `Register(type, #name, DefaultExecCreatorV2<insCollAlgBase<AlgTopoMatch,InsAlgTemplate>>)` 保存creator；[coll_alg_v2_exec_registry.cc:L64](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L64) 实际调用该creator。同一名字Host用于CalcRes，Device用于Orchestrate，因此两个新创建的executor实例依赖的是传递的资源对象，不应理解为Host把executor对象本身传到了Device。

常规HCOMM公共ABI如`HcclChannelAcquire`、`HcclEngineCtxCreate`、`HcclThreadAcquireWithStream`直接继续追踪到HCOMM的同名导出函数（跨仓图提供实际定义）；不要错误画成所有API都通过`DlHcommFunction`表跳转。可选展开流查询才通过`DlHcommFunction`：`dlhcomm_function.cc:16` GetInstance ->48 Init ->55 Dlopen libhcomm.so ->58 InterInit；InterInit的39/44使用Dlsym填`HcclThreadResGetInfo`/`HcommThreadResGetInfo`指针，[op_common.cc:L1579](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1579)/[order_launch.cc:L39](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L39)/[order_launch.cc:L86](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L86)真正通过函数指针调用。

能力检测的另一条初始化链：`hcomm_dlsym.cc:63 HcommDlInit ->68 HcclDlopen(libhcomm.so) ->76 HcclResDlInit ->78 HcommPrimitivesDlInit`，后者按符号是否存在设置HcommIsSupport标记；弱函数桩与能力检测的含义不等于所有实际同名强符号调用经过一张函数指针表。

未无限递归展开：日志宏、安全内存/字符串函数、std容器/智能指针、序列化字段通用工具、profiling/异常报告、CANN ACL运行时及驱动硬件边界、非主例算法实现、AICPU task缓存策略/key的通用实现。核心资源控制、发射保序、cache命中/未命中调度、Device执行器展开已提供逐行notes。

## 公共检查与单rank/算法tag旁支

- `CheckCount` 定义 [op_common.cc:L4282–L4300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4282-L4300)：3265 比较 SYS_MAX_COUNT，超限3269返回PARA；本函数不计算字节数。
- `CheckDataType` 定义 [op_common.cc:L4303–L4374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4303-L4374)：3278查合法枚举集合；3279 needReduce=true进入归约限制（AllReduce），3281–3283禁用UINT8/16/32、INT128、HIF8、FP8；不支持3293返回NOT_SUPPORT；非归约旁支3297只排除非法枚举和INT128，3307返回NOT_SUPPORT。错误消息的支持列表工具是诊断边界，没有额外展开。
- `CheckReduceOp` 定义 [op_common.cc:L4402–L4444](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4402-L4444)：3344只对PROD检查dtype，3345用列表查支持性，不支持3355返回NOT_SUPPORT；SUM等不进此附加检查。本函数不能被描述成对所有reduce枚举都做合法性校验。
- `SetCommEngine` 定义 [op_common.cc:L4463–L4508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4463-L4508)：3380–3389静态配置映射；AICPU_TS→COMM_ENGINE_AICPU_TS在3382，AIV/AIV_ONLY同引擎在3383/3384，CCU_MS/CCU_SCHED同引擎在3385/3386；3391查表、3393写入engine，未知配置3400返回NOT_SUPPORT。
- `SingleRankProc` 定义 [op_common.cc:L4511–L4623](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4511-L4623)：3407 AIV_ONLY返回NOT_SUPPORT；3414单rankSend/Recv成功返回；3418输入输出同地址成功返回；3423–3428变长/AllToAll布局分别算len，AllReduce走3430 `dtype字节数 * count`；非零len先3435 AcquireWithStream，再3455 HcommLocalCopyOnThread；没有多rank模板调用或通信通道创建。
- `SetOpParamAlgTag` 定义 [op_common.cc:L4665–L4738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4665-L4738)：3505 AICPU/AICPU_TS用device后缀，其余host；3510 CCU OFFLOAD图模式组`Graph_算法名_执行位置`避免用户tag不同影响复用，其余3515组`算子tag_算法名_执行位置`；3517验证长度；3523 CCU追加BuildCcuExtraTag（3526，工具定义3473–3498非主例边界），3529追加时再次检查余量；3533捕获枚举字符串查表out_of_range返回PARA。本例主tag包含AicpuAllReduceSoleMeshOneShot及device。

## 逐行notes覆盖的函数范围

| 函数 | 固定 f8af6a3 源码范围 | 功能 |
|---|---|---|
| `UpdateAicpuTimeoutCtx` | [op_common.cc:L80–L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L80-L100) | 把算子执行超时派生为资源上下文中的通知等待及队列资源等待超时 |
| `Selector` | [op_common.cc:L170–L287](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L170-L287) | 选择引擎和算法，恢复拓扑、加载对应kernel并建立算法tag |
| `HcclExecOp` | [op_common.cc:L848–L1154](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L848-L1154) | 执行分发：复用/创建算法资源，关联Host与Device线程，按引擎发射或Host编排 |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1221–L1451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1221-L1451) | 排入Host输入通知、执行两阶段保序、发射AICPU入口，最后排入Host等待Device结果 |
| `AicpuKernelLaunch` | [op_common.cc:L1454–L1613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1454-L1613) | 从已加载二进制取得入口，复制OpParam参数并通过ACL发射到展开流或用户流 |
| `HcclCalcTopoInfo` | [op_common.cc:L1689–L1738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1689-L1738) | 读取或新建基于算子tag的Host拓扑上下文 |
| `CompReqChannelWithExistChannel` | [op_common.cc:L1741–L1773](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1741-L1773) | 增量请求只保留尚未存在的远端rank，AllReduce普通主例不进入此辅助分支 |
| `TryReuseResource` | [op_common.cc:L1779–L1838](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1779-L1838) | 按算法tag及实际上下文存储engine查询已存在的资源 |
| `HcclGetAlgRes` | [op_common.cc:L1844–L1957](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1844-L1957) | 优先复用，未命中时计算拓扑层次和资源请求，分配资源并按条件核对一致性 |
| `FillOpExchangeInfo` | [op_common.cc:L1960–L2031](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1960-L2031) | 构造建链时交换的算子元信息 |
| `FillOpExchangeInfoWithDataDes` | [op_common.cc:L2034–L2082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2034-L2082) | 按算子参数布局填充数据类型和count，AllReduce走default |
| `AddExchangeInfo` | [op_common.cc:L2087–L2107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2087-L2107) | 启用参数一致性检查时登记下一次建链交换的OpExchangeInfo |
| `GetAlgResWithEngine` | [op_common.cc:L2130–L2235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2130-L2235) | 按CPU/AICPU_TS/AIV/CCU引擎分发资源准备，CCU可跨rank协商回退 |
| `GetAlgResAICPU` | [op_common.cc:L2423–L2499](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2423-L2499) | 首次请求构造Host资源对象并复制Device，增量请求可筛除已存在通道 |
| `HcclMemcpyCtxHostToDevice` | [op_common.cc:L2504–L2529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2504-L2529) | 创建AICPU资源上下文存储并复制序列化字节 |
| `HcclAllocAlgResourceAICPU` | [op_common.cc:L2534–L2570](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2534-L2570) | 获得CCL缓冲区、线程和通道，写入可序列化资源对象 |
| `HcclGetThreadWithConfig` | [op_common.cc:L2573–L2634](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2573-L2634) | 使用逐Thread配置申请设备主从线程，必要时申请Host展开线程 |
| `GetMaxNotifyNum` | [op_common.cc:L2637–L2655](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2637-L2655) | 从主从线程通知需求求最大值，用于旧接口的统一通知配置 |
| `HcclGetAicpuThread` | [op_common.cc:L2661–L2732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2661-L2732) | 申请算法执行线程并复用/创建Host展开线程 |
| `HcclGetThread` | [op_common.cc:L2735–L2782](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2735-L2782) | 按engine选择AICPU线程资源或Host流包装/图模式从流 |
| `SaveMainThreadInfo` | [op_common.cc:L2834–L2867](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2834-L2867) | 按算法tag保存主Thread和通知容量 |
| `SaveUnfoldThreadInfo` | [op_common.cc:L2870–L2900](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2870-L2900) | 按通信域名字保存Host展开Thread |
| `GetUnfoldThreadInfo` | [op_common.cc:L2903–L2938](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2903-L2938) | 读取通信域的Host展开Thread |
| `GetMainThreadInfo` | [op_common.cc:L2941–L2974](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2941-L2974) | 读取算法主Thread及通知容量 |
| `HcclGetChannel` | [op_common.cc:L2979–L3036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2979-L3036) | 按算法层和端点所在位置分组申请通道，OFFLOAD额外注册用户区 |
| `BuildChannelInfo` | [op_common.cc:L3041–L3140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3041-L3140) | 从句柄构造模板使用的通道描述，查询端口属性与远端CCL地址 |
| `HcclGetChannelImpl` | [op_common.cc:L3145–L3208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3145-L3208) | 登记交换信息并申请通道，逐条构建ChannelInfo存入对应层 |
| `CommDomainCache::Get` | [kernel_launch.cc:L74–L84](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L74-L84) | 以algTag查设备资源对象缓存，返回shared_ptr保持对象存活 |
| `CommDomainCache::Put` | [kernel_launch.cc:L88–L96](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L88-L96) | 以algTag保存资源对象的独立拷贝 |
| `CommDomainCache::GetCacheSize` | [kernel_launch.cc:L118–L126](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L118-L126) | 加锁读取一个通信域的资源缓存条数 |
| `CommDomainCacheManager::Get` | [kernel_launch.cc:L140–L174](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L140-L174) | 从algTag解析通信域后查询资源对象缓存并统计命中率 |
| `CommDomainCacheManager::Put` | [kernel_launch.cc:L178–L197](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L178-L197) | 选择通信域缓存并存入反序列化资源对象 |
| `CommDomainCacheManager::GetCommStats` | [kernel_launch.cc:L208–L230](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L208-L230) | 读取域的缓存统计值和缓存条数 |
| `CommDomainCacheManager::ExtractCommName` | [kernel_launch.cc:L259–L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L259-L279) | 提取algTag第一和第二个下划线之间的通信域名称，失败返回空串 |
| `CommDomainCacheManager::GetOrCreateComm` | [kernel_launch.cc:L284–L302](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L284-L302) | 加锁查找或创建通信域资源缓存 |
| `DeserializeResCtx` | [kernel_launch.cc:L312–L326](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L312-L326) | 按param.resCtx/ctxSize恢复Host序列化资源对象 |
| `IsOpsV2` | [kernel_launch.cc:L334–L362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L334-L362) | opv2_前缀或支持out-place芯片走新执行器流程 |
| `EnforceLaunchTask` | [kernel_launch.cc:L366–L388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L366-L388) | 结束批模式强制提交当前任务，再恢复批模式 |
| `OpOrchestrate` | [kernel_launch.cc:L393–L478](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L393-L478) | 配置超时/批传输能力，排入输入等待并按注册器调用executor::Orchestrate |
| `HcclOrderLaunchNotifyRecord` | [kernel_launch.cc:L481–L513](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L481-L513) | 有效的Host/Device保序线程存在时记录入口已展开通知 |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L519–L1251](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L519-L1251) | Device入口：获取通信域占用标记、恢复资源、回放/展开任务、排入完成通知并清除占用标记 |
| `CollAlgExecRegistryV2::Instance` | [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) | 取得全局执行器工厂注册表 |
| `CollAlgExecRegistryV2::GetAlgExec` | [coll_alg_v2_exec_registry.cc:L50–L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L50-L66) | 按算子类型和算法名调用已注册creator新建执行器 |
| `OpLaunchGetUnfoldStream` | [order_launch.cc:L19–L63](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L19-L63) | 以可选接口查询Host展开线程关联ACL stream |
| `OpLaunchGetHostOrderStream` | [order_launch.cc:L66–L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L66-L110) | 以可选接口查询Host保序线程关联ACL stream |
| `GetOrderLaunchModeName` | [order_launch.cc:L113–L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L113-L129) | 将保序模式转换为日志名字 |
| `GetOrderLaunchHostThreadType` | [order_launch.cc:L132–L148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L132-L148) | 按OPBASE/GE/ACLGRAPH选择Host专用保序线程类型 |
| `OpLaunchGetOrderStreams` | [order_launch.cc:L151–L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L151-L175) | 取得保序和展开两个ACL stream并检查非空 |
| `AclgraphOrderLaunchEventToOrderStream` | [order_launch.cc:L178–L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L178-L209) | ACL图第一阶段：展开流RecordEvent，Host保序流等待 |
| `AclgraphOrderLaunchEventToKernelStream` | [order_launch.cc:L212–L243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L212-L243) | ACL图第二阶段：Host保序流RecordEvent，展开流等待 |
| `HcclOrderLaunchToOrderStream` | [order_launch.cc:L262–L374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L262-L374) | 第一阶段：取得Host/Device保序线程，Host保序流通知展开流 |
| `HcclOrderLaunchToKernelStream` | [order_launch.cc:L391–L450](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L391-L450) | 第二阶段：Host保序流等待Device入口已进入展开的通知 |
| `NeedInconsistentCheck` | [inconsistent_check.cc:L19–L50](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L19-L50) | 由能力、配置开关、上下文存在性决定是否登记比较算子参数 |
| `CheckCtxStatus` | [inconsistent_check.cc:L53–L79](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L53-L79) | 查资源上下文是否已存在，注意CPU/AIV使用不同的存储engine |
| `CompareOpExchangeInfos` | [inconsistent_check.cc:L82–L116](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L82-L116) | 按算法层或CCU kernel通道比较远端算子交换信息 |
| `InconsistentCheckParams` | [inconsistent_check.cc:L119–L269](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L119-L269) | 逐Peer查询建链交换的元信息，核对CCL大小、root、命令、引擎、归约、dtype、count、group及tag |
| `InconsistentCheckOpType` | [inconsistent_check.cc:L272–L323](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L272-L323) | 集合通信命令应一致，P2P需Send/Recv配对或已启用Group |
| `IsResCtxCacheReusable` | [kernel_launch.h:L27–L33](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.h#L27-L33) | 资源cacheValid成立且缓存记录的通信域地址等于本次域地址才可复用 |
| `IsHcommDefaultTimeoutSupported` | [hcomm_primitives_dl.cc:L158–L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L158-L164) | 默认超时能力同时要求设置等待超时和无显式timeout等待接口 |
| `HcclSetNotifyWaitTimeOut` | [hcomm_primitives_dl.cc:L167–L187](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L167-L187) | 按运行时能力和ABI超时类型调用HCOMM设置通知等待超时 |
| `HcclThreadResAcquireTimeOut` | [hcomm_primitives_dl.cc:L190–L210](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L190-L210) | 按运行时能力和ABI类型调用HCOMM设置队列资源等待超时 |
| `HcclThreadNotifyWaitOnThreadDefault` | [hcomm_primitives_dl.cc:L213–L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L213-L225) | 有默认等待能力时省略timeout参数，否则使用显式fallbackTimeout |
| `CacheStats::hitRate` | [kernel_launch.cc:L48–L56](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L48-L56) | 由原子命中/未命中计数计算资源对象缓存命中比例 |
| `CommDomainCache::GetStats` | [kernel_launch.cc:L113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L113) | 返回可修改的缓存命中计数结构引用 |
| `CommDomainCache::GetStats const` | [kernel_launch.cc:L115](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L115) | 返回只读缓存命中计数结构引用 |
| `CacheHostCtxToEngine` | [op_common.cc:L2238–L2282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2238-L2282) | 增量建链首次保存Host资源副本，失败按源码执行Device/Host上下文回滚 |
| `ReuseCachedDeviceCtx` | [op_common.cc:L2285–L2319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2285-L2319) | 增量请求没有新Peer时直接查已有Device序列化上下文 |
| `IncrementalCreateChannel` | [op_common.cc:L2322–L2418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2322-L2418) | 创建缺失通道并更新Device序列化上下文及Host缓存，AllReduce普通主例不进入 |
| `CheckCount` | [op_common.cc:L4282–L4300](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4282-L4300) | 限制输入元素count不超过系统支持的SYS_MAX_COUNT |
| `CheckDataType` | [op_common.cc:L4303–L4374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4303-L4374) | 检查枚举值合法性并按归约/非归约场景排除不支持dtype |
| `CheckReduceOp` | [op_common.cc:L4402–L4444](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4402-L4444) | PROD归约额外检查dtype支持列表，其它reduce类型本函数不做附加检查 |
| `SetCommEngine` | [op_common.cc:L4463–L4508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4463-L4508) | 将最终opExecuteConfig转换为执行engine |
| `SingleRankProc` | [op_common.cc:L4511–L4623](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4511-L4623) | 单rank算子旁支只需本地复制输入到输出，不执行多rankAllReduce算法 |
| `SetOpParamAlgTag` | [op_common.cc:L4665–L4738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4665-L4738) | 构造算法资源关联tag，CCU额外添加dtype/reduce字段 |
| `DefaultExecCreatorV2` | [coll_alg_v2_exec_registry.h:L27–L39](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h#L27-L39) | 按注册的具体模板类型P创建执行器对象，返回InsCollAlgBase指针 |
| `CollAlgExecRegistryV2::Register` | [coll_alg_v2_exec_registry.cc:L27–L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L27-L47) | 以算子类型和算法tag保存creator，禁止同一个键重复注册 |

## 选中范围内部的直接调用点→定义（按源码实证）

下表只补唯一解析的直接函数名，虚调用执行器/模板与跨仓ABI关系使用上文手工核实的树；构造函数、日志、容器等不在这个附表里冒充算法调用。

| caller | 调用点 | callee定义 |
|---|---|---|
| `UpdateAicpuTimeoutCtx` | [op_common.cc:L98](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L98) | `IsHcommDefaultTimeoutSupported` — [hcomm_primitives_dl.cc:L158–L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L158-L164) |
| `Selector` | [op_common.cc:L198](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L198) | `HcclCalcTopoInfo` — [op_common.cc:L1689–L1738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1689-L1738) |
| `Selector` | [op_common.cc:L229](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L229) | `SetCommEngine` — [op_common.cc:L4463–L4508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4463-L4508) |
| `Selector` | [op_common.cc:L268](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L268) | `SetOpParamAlgTag` — [op_common.cc:L4665–L4738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4665-L4738) |
| `HcclExecOp` | [op_common.cc:L907](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L907) | `SetCommEngine` — [op_common.cc:L4463–L4508](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4463-L4508) |
| `HcclExecOp` | [op_common.cc:L909](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L909) | `SetOpParamAlgTag` — [op_common.cc:L4665–L4738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4665-L4738) |
| `HcclExecOp` | [op_common.cc:L948](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L948) | `CollAlgExecRegistryV2::Instance` — [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) |
| `HcclExecOp` | [op_common.cc:L948](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L948) | `CollAlgExecRegistryV2::GetAlgExec` — [coll_alg_v2_exec_registry.cc:L50–L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L50-L66) |
| `HcclExecOp` | [op_common.cc:L986](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L986) | `HcclGetAlgRes` — [op_common.cc:L1844–L1957](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1844-L1957) |
| `HcclExecOp` | [op_common.cc:L1033](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1033) | `GetMainThreadInfo` — [op_common.cc:L2941–L2974](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2941-L2974) |
| `HcclExecOp` | [op_common.cc:L1052](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1052) | `GetUnfoldThreadInfo` — [op_common.cc:L2903–L2938](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2903-L2938) |
| `HcclExecOp` | [op_common.cc:L1061](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1061) | `HcclAicpuKernelEntranceLaunch` — [op_common.cc:L1221–L1451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1221-L1451) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1316](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1316) | `GetMainThreadInfo` — [op_common.cc:L2941–L2974](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2941-L2974) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1336](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1336) | `IsHcommDefaultTimeoutSupported` — [hcomm_primitives_dl.cc:L158–L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L158-L164) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1364](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1364) | `CollAlgExecRegistryV2::Instance` — [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1391](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1391) | `HcclOrderLaunchToOrderStream` — [order_launch.cc:L262–L374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L262-L374) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1400) | `AicpuKernelLaunch` — [op_common.cc:L1454–L1613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1454-L1613) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1406](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1406) | `HcclOrderLaunchToKernelStream` — [order_launch.cc:L391–L450](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L391-L450) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1434](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1434) | `IsHcommDefaultTimeoutSupported` — [hcomm_primitives_dl.cc:L158–L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L158-L164) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1442](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1442) | `HcclSetNotifyWaitTimeOut` — [hcomm_primitives_dl.cc:L167–L187](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L167-L187) |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:L1446](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1446) | `HcclThreadNotifyWaitOnThreadDefault` — [hcomm_primitives_dl.cc:L213–L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L213-L225) |
| `AicpuKernelLaunch` | [op_common.cc:L1544](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1544) | `IsHcommDefaultTimeoutSupported` — [hcomm_primitives_dl.cc:L158–L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L158-L164) |
| `HcclGetAlgRes` | [op_common.cc:L1873](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1873) | `TryReuseResource` — [op_common.cc:L1779–L1838](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1779-L1838) |
| `HcclGetAlgRes` | [op_common.cc:L1882](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1882) | `NeedInconsistentCheck` — [inconsistent_check.cc:L19–L50](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L19-L50) |
| `HcclGetAlgRes` | [op_common.cc:L1899](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1899) | `GetAlgResWithEngine` — [op_common.cc:L2130–L2235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2130-L2235) |
| `HcclGetAlgRes` | [op_common.cc:L1948](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1948) | `FillOpExchangeInfo` — [op_common.cc:L1960–L2031](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1960-L2031) |
| `HcclGetAlgRes` | [op_common.cc:L1950](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1950) | `CompareOpExchangeInfos` — [inconsistent_check.cc:L82–L116](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L82-L116) |
| `FillOpExchangeInfo` | [op_common.cc:L1978](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1978) | `FillOpExchangeInfoWithDataDes` — [op_common.cc:L2034–L2082](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2034-L2082) |
| `AddExchangeInfo` | [op_common.cc:L2097](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2097) | `FillOpExchangeInfo` — [op_common.cc:L1960–L2031](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1960-L2031) |
| `GetAlgResWithEngine` | [op_common.cc:L2162](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2162) | `GetAlgResAICPU` — [op_common.cc:L2423–L2499](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2423-L2499) |
| `GetAlgResAICPU` | [op_common.cc:L2455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2455) | `HcclAllocAlgResourceAICPU` — [op_common.cc:L2534–L2570](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2534-L2570) |
| `GetAlgResAICPU` | [op_common.cc:L2462](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2462) | `HcclMemcpyCtxHostToDevice` — [op_common.cc:L2504–L2529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2504-L2529) |
| `GetAlgResAICPU` | [op_common.cc:L2468](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2468) | `CacheHostCtxToEngine` — [op_common.cc:L2238–L2282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2238-L2282) |
| `GetAlgResAICPU` | [op_common.cc:L2481](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2481) | `CompReqChannelWithExistChannel` — [op_common.cc:L1741–L1773](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1741-L1773) |
| `GetAlgResAICPU` | [op_common.cc:L2486](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2486) | `ReuseCachedDeviceCtx` — [op_common.cc:L2285–L2319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2285-L2319) |
| `GetAlgResAICPU` | [op_common.cc:L2490](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2490) | `IncrementalCreateChannel` — [op_common.cc:L2322–L2418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2322-L2418) |
| `HcclAllocAlgResourceAICPU` | [op_common.cc:L2559](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2559) | `UpdateAicpuTimeoutCtx` — [op_common.cc:L80–L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L80-L100) |
| `HcclAllocAlgResourceAICPU` | [op_common.cc:L2564](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2564) | `HcclGetThread` — [op_common.cc:L2735–L2782](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2735-L2782) |
| `HcclAllocAlgResourceAICPU` | [op_common.cc:L2566](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2566) | `HcclGetChannel` — [op_common.cc:L2979–L3036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2979-L3036) |
| `HcclGetThreadWithConfig` | [op_common.cc:L2630](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2630) | `SaveMainThreadInfo` — [op_common.cc:L2834–L2867](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2834-L2867) |
| `HcclGetAicpuThread` | [op_common.cc:L2678](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2678) | `GetUnfoldThreadInfo` — [op_common.cc:L2903–L2938](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2903-L2938) |
| `HcclGetAicpuThread` | [op_common.cc:L2691](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2691) | `HcclGetThreadWithConfig` — [op_common.cc:L2573–L2634](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2573-L2634) |
| `HcclGetAicpuThread` | [op_common.cc:L2695](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2695) | `GetMaxNotifyNum` — [op_common.cc:L2637–L2655](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2637-L2655) |
| `HcclGetAicpuThread` | [op_common.cc:L2710](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2710) | `SaveMainThreadInfo` — [op_common.cc:L2834–L2867](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2834-L2867) |
| `HcclGetAicpuThread` | [op_common.cc:L2716](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2716) | `SaveUnfoldThreadInfo` — [op_common.cc:L2870–L2900](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2870-L2900) |
| `HcclGetThread` | [op_common.cc:L2747](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2747) | `HcclGetAicpuThread` — [op_common.cc:L2661–L2732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2661-L2732) |
| `HcclGetThread` | [op_common.cc:L2761](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2761) | `GetMaxNotifyNum` — [op_common.cc:L2637–L2655](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2637-L2655) |
| `HcclGetChannel` | [op_common.cc:L3027](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3027) | `HcclGetChannelImpl` — [op_common.cc:L3145–L3208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3145-L3208) |
| `HcclGetChannel` | [op_common.cc:L3030](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3030) | `HcclGetChannelImpl` — [op_common.cc:L3145–L3208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3145-L3208) |
| `HcclGetChannelImpl` | [op_common.cc:L3186](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3186) | `AddExchangeInfo` — [op_common.cc:L2087–L2107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2087-L2107) |
| `HcclGetChannelImpl` | [op_common.cc:L3198](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3198) | `BuildChannelInfo` — [op_common.cc:L3041–L3140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3041-L3140) |
| `CommDomainCacheManager::Get` | [kernel_launch.cc:L144](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L144) | `CommDomainCacheManager::ExtractCommName` — [kernel_launch.cc:L259–L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L259-L279) |
| `CommDomainCacheManager::Get` | [kernel_launch.cc:L152](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L152) | `CommDomainCacheManager::GetOrCreateComm` — [kernel_launch.cc:L284–L302](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L284-L302) |
| `CommDomainCacheManager::Get` | [kernel_launch.cc:L156](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L156) | `CommDomainCache::GetStats` — [kernel_launch.cc:L113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L113) |
| `CommDomainCacheManager::Put` | [kernel_launch.cc:L182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L182) | `CommDomainCacheManager::ExtractCommName` — [kernel_launch.cc:L259–L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L259-L279) |
| `CommDomainCacheManager::Put` | [kernel_launch.cc:L189](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L189) | `CommDomainCacheManager::GetOrCreateComm` — [kernel_launch.cc:L284–L302](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L284-L302) |
| `CommDomainCacheManager::GetCommStats` | [kernel_launch.cc:L218](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L218) | `CommDomainCache::GetStats` — [kernel_launch.cc:L113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L113) |
| `CommDomainCacheManager::GetCommStats` | [kernel_launch.cc:L220](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L220) | `CommDomainCache::GetStats` — [kernel_launch.cc:L113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L113) |
| `CommDomainCacheManager::GetCommStats` | [kernel_launch.cc:L222](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L222) | `CommDomainCache::GetCacheSize` — [kernel_launch.cc:L118–L126](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L118-L126) |
| `OpOrchestrate` | [kernel_launch.cc:L403](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L403) | `HcclThreadResAcquireTimeOut` — [hcomm_primitives_dl.cc:L190–L210](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L190-L210) |
| `OpOrchestrate` | [kernel_launch.cc:L411](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L411) | `HcclSetNotifyWaitTimeOut` — [hcomm_primitives_dl.cc:L167–L187](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L167-L187) |
| `OpOrchestrate` | [kernel_launch.cc:L441](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L441) | `HcclThreadNotifyWaitOnThreadDefault` — [hcomm_primitives_dl.cc:L213–L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L213-L225) |
| `OpOrchestrate` | [kernel_launch.cc:L445](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L445) | `CollAlgExecRegistryV2::Instance` — [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) |
| `OpOrchestrate` | [kernel_launch.cc:L455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L455) | `CollAlgExecRegistryV2::Instance` — [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) |
| `OpOrchestrate` | [kernel_launch.cc:L455](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L455) | `CollAlgExecRegistryV2::GetAlgExec` — [coll_alg_v2_exec_registry.cc:L50–L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L50-L66) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L559](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L559) | `HcclOrderLaunchNotifyRecord` — [kernel_launch.cc:L481–L513](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L481-L513) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L565](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L565) | `IsOpsV2` — [kernel_launch.cc:L334–L362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L334-L362) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L610](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L610) | `IsOpsV2` — [kernel_launch.cc:L334–L362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L334-L362) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L672](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L672) | `IsResCtxCacheReusable` — [kernel_launch.h:L27–L33](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.h#L27-L33) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L676](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L676) | `CommDomainCacheManager::ExtractCommName` — [kernel_launch.cc:L259–L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L259-L279) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L687](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L687) | `CommDomainCacheManager::GetCommStats` — [kernel_launch.cc:L208–L230](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L208-L230) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L693](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L693) | `CacheStats::hitRate` — [kernel_launch.cc:L48–L56](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L48-L56) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L706](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L706) | `DeserializeResCtx` — [kernel_launch.cc:L312–L326](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L312-L326) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L730](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L730) | `DeserializeResCtx` — [kernel_launch.cc:L312–L326](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L312-L326) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L921](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L921) | `OpOrchestrate` — [kernel_launch.cc:L393–L478](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L393-L478) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L935](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L935) | `EnforceLaunchTask` — [kernel_launch.cc:L366–L388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L366-L388) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L979](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L979) | `CollAlgExecRegistryV2::Instance` — [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L999](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L999) | `OpOrchestrate` — [kernel_launch.cc:L393–L478](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L393-L478) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L1053](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L1053) | `CollAlgExecRegistryV2::Instance` — [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) |
| `HcclLaunchAicpuKernel` | [kernel_launch.cc:L1053](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L1053) | `CollAlgExecRegistryV2::GetAlgExec` — [coll_alg_v2_exec_registry.cc:L50–L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L50-L66) |
| `OpLaunchGetOrderStreams` | [order_launch.cc:L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L159) | `OpLaunchGetHostOrderStream` — [order_launch.cc:L66–L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L66-L110) |
| `OpLaunchGetOrderStreams` | [order_launch.cc:L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L161) | `OpLaunchGetUnfoldStream` — [order_launch.cc:L19–L63](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L19-L63) |
| `AclgraphOrderLaunchEventToOrderStream` | [order_launch.cc:L188](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L188) | `OpLaunchGetOrderStreams` — [order_launch.cc:L151–L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L151-L175) |
| `AclgraphOrderLaunchEventToKernelStream` | [order_launch.cc:L222](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L222) | `OpLaunchGetOrderStreams` — [order_launch.cc:L151–L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L151-L175) |
| `HcclOrderLaunchToOrderStream` | [order_launch.cc:L270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L270) | `GetOrderLaunchModeName` — [order_launch.cc:L113–L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L113-L129) |
| `HcclOrderLaunchToOrderStream` | [order_launch.cc:L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L272) | `GetOrderLaunchHostThreadType` — [order_launch.cc:L132–L148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L132-L148) |
| `HcclOrderLaunchToOrderStream` | [order_launch.cc:L330](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L330) | `AclgraphOrderLaunchEventToOrderStream` — [order_launch.cc:L178–L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L178-L209) |
| `HcclOrderLaunchToKernelStream` | [order_launch.cc:L397](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L397) | `GetOrderLaunchModeName` — [order_launch.cc:L113–L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L113-L129) |
| `HcclOrderLaunchToKernelStream` | [order_launch.cc:L399](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L399) | `GetOrderLaunchHostThreadType` — [order_launch.cc:L132–L148](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L132-L148) |
| `HcclOrderLaunchToKernelStream` | [order_launch.cc:L441](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L441) | `AclgraphOrderLaunchEventToKernelStream` — [order_launch.cc:L212–L243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L212-L243) |
| `NeedInconsistentCheck` | [inconsistent_check.cc:L30](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L30) | `CheckCtxStatus` — [inconsistent_check.cc:L53–L79](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L53-L79) |
| `CompareOpExchangeInfos` | [inconsistent_check.cc:L96](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L96) | `InconsistentCheckParams` — [inconsistent_check.cc:L119–L269](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L119-L269) |
| `CompareOpExchangeInfos` | [inconsistent_check.cc:L104](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L104) | `InconsistentCheckParams` — [inconsistent_check.cc:L119–L269](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L119-L269) |
| `InconsistentCheckParams` | [inconsistent_check.cc:L179](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L179) | `InconsistentCheckOpType` — [inconsistent_check.cc:L272–L323](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L272-L323) |
| `IncrementalCreateChannel` | [op_common.cc:L2330](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2330) | `HcclGetChannel` — [op_common.cc:L2979–L3036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2979-L3036) |
| `IncrementalCreateChannel` | [op_common.cc:L2352](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2352) | `HcclMemcpyCtxHostToDevice` — [op_common.cc:L2504–L2529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2504-L2529) |
| `SingleRankProc` | [op_common.cc:L4601](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4601) | `SetOpParamAlgTag` — [op_common.cc:L4665–L4738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L4665-L4738) |


### 调用树中的精确源码定位

| 审读快照位置 | 带逐行注释的固定源码 |
|---|---|
| `hccl/src/ops/op_common/op_common.cc:S158–S223` | [op_common.cc:L170–L287](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L170-L287) |
| `hccl/src/ops/op_common/op_common.cc:S1325–S1353` | [op_common.cc:L1689–L1738](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1689-L1738) |
| `hccl/src/ops/op_common/op_common.cc:S783–S962` | [op_common.cc:L848–L1154](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L848-L1154) |
| `hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc:S15–S19` | [coll_alg_v2_exec_registry.cc:L16–L24](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L16-L24) |
| `hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc:S33–S41` | [coll_alg_v2_exec_registry.cc:L50–L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc#L50-L66) |
| `hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h:S26–S32` | [coll_alg_v2_exec_registry.h:L27–L39](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h#L27-L39) |
| `hccl/src/ops/op_common/op_common.cc:S1415–S1481` | [op_common.cc:L1844–L1957](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1844-L1957) |
| `hccl/src/ops/op_common/op_common.cc:S1377–S1410` | [op_common.cc:L1779–L1838](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1779-L1838) |
| `hccl/src/common/inconsistent_check.cc:S18–S35` | [inconsistent_check.cc:L19–L50](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L19-L50) |
| `hccl/src/ops/op_common/op_common.cc:S1582–S1639` | [op_common.cc:L2130–L2235](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2130-L2235) |
| `hccl/src/common/inconsistent_check.cc:S52–S69` | [inconsistent_check.cc:L82–L116](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/inconsistent_check.cc#L82-L116) |
| `hccl/src/ops/op_common/op_common.cc:S2042–S2060` | [op_common.cc:L2941–L2974](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2941-L2974) |
| `hccl/src/ops/op_common/op_common.cc:S2022–S2040` | [op_common.cc:L2903–L2938](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2903-L2938) |
| `hccl/src/ops/op_common/op_common.cc:S1028–S1165` | [op_common.cc:L1221–L1451](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1221-L1451) |
| `hccl/src/ops/op_common/op_common.cc:S1736–S1777` | [op_common.cc:L2423–L2499](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2423-L2499) |
| `hccl/src/ops/op_common/op_common.cc:S1799–S1819` | [op_common.cc:L2534–L2570](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2534-L2570) |
| `hccl/src/ops/op_common/op_common.cc:S1908–S1933` | [op_common.cc:L2735–L2782](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2735-L2782) |
| `hccl/src/ops/op_common/op_common.cc:S1869–S1906` | [op_common.cc:L2661–L2732](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2661-L2732) |
| `hccl/src/ops/op_common/op_common.cc:S1821–S1853` | [op_common.cc:L2573–L2634](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2573-L2634) |
| `hccl/src/ops/op_common/op_common.cc:S1984–S2002` | [op_common.cc:L2834–L2867](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2834-L2867) |
| `hccl/src/ops/op_common/op_common.cc:S1855–S1864` | [op_common.cc:L2637–L2655](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2637-L2655) |
| `hccl/src/ops/op_common/op_common.cc:S2004–S2020` | [op_common.cc:L2870–L2900](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2870-L2900) |
| `hccl/src/ops/op_common/op_common.cc:S2064–S2095` | [op_common.cc:L2979–L3036](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2979-L3036) |
| `hccl/src/ops/op_common/op_common.cc:S2155–S2190` | [op_common.cc:L3145–L3208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3145-L3208) |
| `hccl/src/ops/op_common/op_common.cc:S1550–S1560` | [op_common.cc:L2087–L2107](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2087-L2107) |
| `hccl/src/ops/op_common/op_common.cc:S2099–S2151` | [op_common.cc:L3041–L3140](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L3041-L3140) |
| `hccl/src/ops/op_common/op_common.cc:S1781–S1795` | [op_common.cc:L2504–L2529](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2504-L2529) |
| `hccl/src/ops/op_common/op_common.cc:S1641–S1663` | [op_common.cc:L2238–L2282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2238-L2282) |
| `hccl/src/ops/op_common/op_common.cc:S1355–S1372` | [op_common.cc:L1741–L1773](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1741-L1773) |
| `hccl/src/ops/op_common/op_common.cc:S1665–S1682` | [op_common.cc:L2285–L2319](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2285-L2319) |
| `hccl/src/ops/op_common/op_common.cc:S1684–S1732` | [op_common.cc:L2322–L2418](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L2322-L2418) |
| `hccl/src/ops/op_common/order_launch.cc:S152–S213` | [order_launch.cc:L262–L374](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L262-L374) |
| `hccl/src/ops/op_common/order_launch.cc:S100–S116` | [order_launch.cc:L178–L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L178-L209) |
| `hccl/src/ops/op_common/order_launch.cc:S86–S98` | [order_launch.cc:L151–L175](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L151-L175) |
| `hccl/src/ops/op_common/order_launch.cc:S42–S64` | [order_launch.cc:L66–L110](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L66-L110) |
| `hccl/src/ops/op_common/order_launch.cc:S18–S40` | [order_launch.cc:L19–L63](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L19-L63) |
| `hccl/src/ops/op_common/op_common.cc:S1167–S1250` | [op_common.cc:L1454–L1613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/op_common.cc#L1454-L1613) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S353` | [kernel_launch.cc:L519](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L519) |
| `hccl/src/ops/op_common/order_launch.cc:S229–S262` | [order_launch.cc:L391–L450](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L391-L450) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S331–S348` | [kernel_launch.cc:L481–L513](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L481-L513) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S339` | [kernel_launch.cc:L496](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L496) |
| `hccl/src/ops/op_common/order_launch.cc:S118–S134` | [order_launch.cc:L212–L243](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/order_launch.cc#L212-L243) |
| `hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc:S176–S182` | [hcomm_primitives_dl.cc:L213–L225](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L213-L225) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S353–S769` | [kernel_launch.cc:L519–L1251](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L519-L1251) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S243–S259` | [kernel_launch.cc:L334–L362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L334-L362) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S116–S134` | [kernel_launch.cc:L140–L174](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L140-L174) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S196–S207` | [kernel_launch.cc:L259–L279](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L259-L279) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S211–S220` | [kernel_launch.cc:L284–L302](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L284-L302) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S68–S73` | [kernel_launch.cc:L74–L84](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L74-L84) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S229–S236` | [kernel_launch.cc:L312–L326](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L312-L326) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S137–S147` | [kernel_launch.cc:L178–L197](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L178-L197) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S277–S329` | [kernel_launch.cc:L393–L478](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L393-L478) |
| `hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:S262–S273` | [kernel_launch.cc:L366–L388](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L366-L388) |
