# AllReduce：从公开入口追到HCOMM任务提交

这份指南以一个实际注册的AllReduce算法为主线，按“调用者→被调用者→分支→数据和资源的变化”读代码。先看下面的调用树，再点击函数或代码行，与源码里的`[AllReduce逐行 S…]`说明对照。

本次逐行说明共{{noteCount}}条，覆盖{{functionCount}}个选定函数/代码范围。完整列表和范围在[逐行覆盖清单](allreduce/COVERAGE.zh-CN.md)，原先的AllToAll路线保留在[AllToAll阅读指南](ALLTOALL_READING_GUIDE.zh-CN.md)。

源码链接固定到提交`{{sourceRevision}}`。S表示审读快照`f8af6a36831195a8440de6ec6183856bb72af907`的物理行；L表示新增逐行注释后的源码物理行。主文档、附录和源码注释由[生成器](../scripts/build-allreduce-guide.py)绑定，避免插入注释以后行号失效。

## 1. 先固定一个能沿着往下读的例子

| 项目 | 本例条件 | 为什么要固定 |
|---|---|---|
| 公开入口 | `HcclAllReduce(sendBuf, recvBuf, count, FP32, SUM, comm, stream)` | `count`为元素数，FP32每元素4字节 |
| 设备与通信域 | Ascend 950；通信域与用户stream已准备，Rank>1 | 域创建不属于一次算子调用的内部步骤 |
| 执行模式 | OPBASE；有效展开配置为AICPU_TS | 950优先读取域配置，不能只凭Host环境变量判断Engine |
| 算法选择 | `HCCL_USE_NEW_SELECTOR=0`；无插件接管、无strict保序要求 | 明确走规则选择器；新成本选择器不能仅靠8MiB阈值推出算法 |
| 拓扑与数据量 | 单有效算法层，Mesh1D；每Rank输入≤8MiB | 规则选择器可以选到`AicpuAllReduceSoleMeshOneShot` |
| 内存与链路 | 普通非对称用户内存；RankGraph有符合条件的UB_CTP链路 | 走Write到远端CCL，再本地归约；对称内存改走ReadReduce |
| 缓存 | 无历史回退；本次算法资源首次申请，task缓存未命中/关闭 | 能观察完整控制面和模板编排；资源缓存与task缓存分开讨论 |

这些条件是静态源码路线的阅读条件，不是一次上板trace。若实际日志中的`algName`、`engine`或拓扑不同，应转到对应分支，不能继续套用本例模板。

选择器开关解析在{{src|env|817|837|ParseNewSelector}}；空值默认保持0，非法值返回参数错误。展开模式的决定与应用在{{src|common|3541|3560|HcclGetOpExpansionMode}}、{{src|common|3586|3627|DecideHcclOpExpansionMode}}、{{src|common|3649|3690|ApplyOpExpansionMode}}。

## 2. 整条主链的调用关系树

缩进表示内部调用，不表示父函数完成后再执行子函数。`[Host]`和`[AICPU]`之间由runtime发射衔接；它们不是同一个CPU调用栈。跨仓接口既有公共ABI同名入口，也有按能力探测选择的动态符号适配，不能把所有`→HCOMM`都画成同一个函数指针。

```text
[Host] HcclAllReduce
├─ AllReduceInitAndCheck
│  ├─ InitEnvConfig / CheckAllReduceInputPara
│  ├─ HcclGetRankSize / HcclGetRankId / HcclGetCommName →HCOMM域查询
│  └─ HcclCheckTag / HcomCheckUserRank / CheckCount / CheckDataType / CheckReduceOp
├─ AllReduceEntryLog
└─ AllReduceOutPlace
   └─ AllReduceOutPlaceCommon
      ├─ FillAllReduceOpParam
      ├─ HcclGetOpExpansionMode
      ├─ 兼容、CCU fast launch、AIV replay、单Rank分流
      ├─ Selector
      │  ├─ HcclCalcTopoInfo
      │  ├─ ExecuteSelector::Run                         [本例]
      │  │  └─ AutoSelectorBase::Select
      │  │     └─ AllReduceAutoSelector::SelectAicpuAlgo
      │  │        └─ SelectMeshAlgoAicpu → AicpuAllReduceSoleMeshOneShot
      │  └─ SelectorEngine::Run →成本模型/tuner→SelectMinCost [开关为1]
      └─ HcclExecOp
         ├─ CollAlgExecRegistryV2::GetAlgExec →SoleExecutor
         ├─ 包装用户stream / 导出Thread句柄 →HCOMM
         ├─ HcclGetAlgRes
         │  ├─ TryReuseResource                         [命中提前返回]
         │  ├─ SoleExecutor::CalcAlgHierarchyInfoV2 →TopoMatchOneLevel::MatchTopo
         │  ├─ SoleExecutor::CalcRes →OneShot::CalcRes
         │  │  └─ CalcChannelRequestMesh1D →RankGraph查询→Channel请求
         │  └─ GetAlgResWithEngine →GetAlgResAICPU
         │     ├─ HcclAllocAlgResourceAICPU
         │     │  ├─ HcclGetHcclBuffer →HCOMM本地CCL
         │     │  ├─ HcclGetThread →HCOMM Thread申请
         │     │  └─ HcclGetChannel →HcclGetChannelImpl
         │     │     ├─ AddExchangeInfo →HCOMM登记算子元信息
         │     │     ├─ HcclChannelAcquire →HCOMM MyRank::CreateChannels
         │     │     │  ├─ Socket准备
         │     │     │  ├─ BatchCreateChannels
         │     │     │  │  ├─ EndpointMgr::Get / RegisterMemory
         │     │     │  │  └─ EndpointPair::CreateChannel
         │     │     │  │     └─ HcommCollectiveChannelCreate →ChannelProcess
         │     │     │  ├─ 等待建链 / 一致性信息交换
         │     │     │  └─ 整理返回的Channel句柄
         │     │     └─ BuildChannelInfo →HCOMM查询远端CCL
         │     └─ 序列化资源 →HcclMemcpyCtxHostToDevice →设备EngineCtx
         └─ HcclAicpuKernelEntranceLaunch
            ├─ 用户流Record输入就绪 / 建立展开保序
            ├─ AicpuKernelLaunch →aclrtLaunchKernelWithConfig
            └─ 用户流Wait设备完成通知

[AICPU] runtime调用 HcclLaunchAicpuKernel
├─ HcommAcquireComm / 状态检查 / 恢复设备资源上下文
├─ HcommBatchModeStart
├─ task cache命中 →更新地址、回放任务                 [跳过模板]
└─ task cache未命中/关闭 →OpOrchestrate
   ├─ 主Thread Wait输入就绪
   ├─ CollAlgExecRegistryV2::GetAlgExec
   └─ SoleExecutor::Orchestrate →OrchestrateLoop
      └─ 每个数据块 OneShot::KernelRun
         ├─ CalcSlice →每个源Rank一个CCL槽
         ├─ PreSyncInterThreads →放行Peer从Thread
         ├─ RunAllReduce
         │  ├─ LocalCopy(input →本Rankoutput)
         │  └─ 每Peer一个从Thread
         │     ├─ 普通内存 SendRecvBatchWrite           [本例]
         │     │  ├─ ACK通知/等待
         │     │  ├─ 批接口支持 →RunBatchTransfer→HcommBatchTransferOnThread
         │     │  └─ 批接口不支持 →SendRecvWrite→HcommWriteOnThread
         │     └─ 对称内存 SymmetricReadReduce→SendRecvBatchReadReduce
         ├─ PostSyncInterThreads →主Thread收齐Peer完成
         └─ 普通内存 PostLocalReduce
            ├─ 特殊类型/PROD →提交已编排任务并ThreadJoin→AicpuReduce
            └─ FP32 SUM →LocalReduce→HcommLocalReduceOnThread

设备入口尾部：主Thread Record用户流完成 / HcommBatchModeEnd / HcommReleaseComm
HCOMM数据面：原语适配 →TransportLite / StreamLite →SQE/WQE构造与提交 →外部runtime/driver
```

下面各节逐级展开该树。[分阶段调用与分支详表](allreduce/CALL_RELATIONS.zh-CN.md)另列调用点、被调用函数和各条件的后续目标；函数内每条物理代码行见四份对照页：

1. [入口、引擎与算法选择](allreduce/01-entry-selection.zh-CN.md)。
2. [公共调度、资源上下文与设备入口](allreduce/02-dispatch-device.zh-CN.md)。
3. [HCOMM通信资源与建链](allreduce/03-hcomm-resources.zh-CN.md)。
4. [OneShot、传输包装与底层任务提交](allreduce/04-data-plane.zh-CN.md)。

## 3. HcclAllReduce：先判断是否进入这条新路径

读{{src|entry|26|60|HcclAllReduce完整入口}}。它本身不执行循环归约，负责分流与调用：

| 判断/调用 | 成立时 | 否则 |
|---|---|---|
| HCOMM版本<9.0 | 直接`HcclAllReduceInner`，进入兼容实现 | 判断设备 |
| `IsOutPlaceDevice`失败 | `CHK_RET`传播错误 | 检查返回的支持标志 |
| 设备不支持新流程 | `HcclAllReduceInner` | 检查count |
| `count==0` | 返回成功，此时还未做完整非空参数检查 | 初始化和检查参数 |
| `AllReduceInitAndCheck` | 解析配置、检查指针、取Rank/域名、构造tag、检查数量/类型/PROD组合 | 任一被检查调用失败，返回该错误 |
| `AllReduceEntryLog` | 配置使能时写入口日志 | 关闭时跳过日志主体 |
| `AllReduceOutPlace` | 以OPBASE和空图资源包调用公共入口 | 调用错误由宏传播 |

检查函数逐行见{{src|entry|118|171|AllReduceInitAndCheck与CheckAllReduceInputPara}}；参数结构填充见{{src|entry|173|205|FillAllReduceOpParam}}。`inputSize/outputSize=count×类型字节数`；用户地址只存入描述，此处没有跨Rank搬运。

`CheckReduceOp`在当前快照主要检查PROD与数据类型的组合；不要仅因名字就理解为此函数独自验证所有reduceOp枚举。日志、检查宏和安全字符串函数的公共实现按工具边界处理，相关调用行仍有逐行说明。

图模式是平行公开入口{{src|entry|62|115|HcclAllReduceGraphMode}}：先由group取得comm，检查外部tag，装入外部streams与scratchMem，再经{{src|entry|309|318|AllReduceOutPlaceGraphMode}}传OFFLOAD。它不是OPBASE入口内部的下一个步骤。

## 4. AllReduceOutPlaceCommon：所有提前离开的路口

读{{src|entry|207|284|AllReduceOutPlaceCommon}}，顺序很重要：

1. `FillAllReduceOpParam`记录本次输入、输出、数量、类型和reduceOp；`HcclGetOpExpansionMode`先决定候选引擎。
2. OPBASE且HCOMM恰为9.0且Engine为CCU：走兼容`HcclAllReduceInner`。
3. OPBASE且`ShouldGoCcuFastLaunch`命中：走`HcclExecOpCcuFastLaunch`，跳过下面的常规选择/资源准备。
4. 候选Engine为AIV：先检查AIV缓存，成功回放则返回；未命中才继续。
5. Rank数为1：调用`SingleRankProc`；同地址直接成功，不同地址排入本地Copy，既没有Peer Channel也没有OneShot跨Rank归约。
6. HCOMM≥9.1且OPBASE：探测对称内存；这还只是候选标志。
7. `Selector`根据配置与拓扑选出最终算法。
8. 用选中的拓扑、引擎、算法和运算再收紧对称内存标志，最后`HcclExecOp`。

对称内存过滤的两个分支也要读：Mesh1D_CLOS且非PCIe时，仅保留这里指定的`AicpuAllReducePipeLineMeshNHR`和层数条件；其他情况要求AICPU_TS＋Mesh1D，且不是INT64、UINT64、FP64或PROD。条件不符只清除支持标志，不把用户buffer换成另一个地址。

## 5. Selector：规则选择与成本选择是两条路

读{{src|common|158|223|Selector}}：先检查域状态（若接口支持），获取或构建拓扑；topLevelUboe可以强制AICPU_TS。编译启用的算法插件可以接管选择。然后判断新选择器开关。

### 5.1 本例的规则选择调用

{{src|rules|19|57|ExecuteSelector::Run}}按算子类型取得已注册selector，再按优先级遍历，首个MATCH即返回；全部NOT_MATCH则返回不支持。MC2是单独CCU selector分支，本例不走。

{{src|auto|17|68|AutoSelectorBase::Select}}先处理HostDPU，再依次处理CCU_MS→CCU_SCHED→CCU_FAIL、AIV配置和STARS类状态；选择失败可能改配置而继续下一个引擎。STARS分支中还有PCIe/ATU限制引起的AIV回退。本例AICPU_TS且无上述回退，调用{{src|arselect|403|480|AllReduceAutoSelector::SelectAicpuAlgo}}。

| SelectAicpuAlgo分支 | 结果/后续 |
|---|---|
| strict保序 | 按Rank阈值选StrictOrderedGroupMesh或StrictOrderedMesh，提前MATCH |
| 多拓扑层＋INT64/UINT64/FP64/PROD | SoleNHRAicpuReduce |
| 多层＋level0/1对称＋每module8设备＋topLevelUboe | 两层PipelineMeshNHR；其他层数PipelineMeshNHRNHR |
| 多层＋level0/1对称＋三层＋topLevelUboe | ParallelMeshNHR |
| 多层＋Level1Nhr或level0局部Rank数为1 | SoleNHR |
| 多层＋Mesh1D | 三层：对称用SequenceMeshConcurNHRNHR，否则SoleNHR；两层：按跨层小数据与Sequence阈值选SequenceMeshConcurNHR/ParallelMeshNHR/SoleNHR；其他用SoleNHR |
| 多层＋CLOS | SoleNHRMultiLink |
| 多层＋其他形状 | NOT_MATCH |
| 单层 | 继续`SelectMeshAlgoAicpu` |

{{src|arselect|523|595|SelectMeshAlgoAicpu}}的Mesh1D分支（`dataSize=count×类型字节数`）：

| 优先判断 | 算法 |
|---|---|
| 特殊类型/PROD | ≤8MiB OneShot；否则TwoShot |
| 普通类型，两层网络标志＋2Rank＋≥detour阈值 | SoleMeshConcur |
| 普通类型，≤8MiB | **SoleMeshOneShot，本例** |
| `dataSize×ratio>32MiB` | 两层标志且超Sequence阈值：SoleMeshConcur；否则SoleMeshChunkTwoShot |
| 其余 | SoleMeshTwoShot |

`ratio`按源码的`DEFAULT_RANK_SIZE / rankSize / rankSize`计算；Rank数为0则兜底1。不能把32MiB写成所有Rank规模共用的未缩放判断。

这个函数的其余形状分支：CLOS选SoleNHRAicpuReduce或SoleNHRMultiLink；Mesh1D_CLOS＋PCIe且全Mesh连通，按特殊类型/8MiB/缩放32MiB分OneShot、TwoShot和Chunk；PCIe不全连通，特殊类型走SequenceMeshNHRAicpuReduce，普通类型按OMNI_PCIE阈值分Parallel/Pipeline；非PCIe调用{{src|arselect|482|521|SelectMeshAlgoAicpuUBX}}，再根据Mesh/Clos等宽、Rank≤4、特殊类型、矩形和大数据、对称内存选OneShot/TwoShot/ConcurMeshTwoShotNHR/SoleNHRAicpuReduce/PipelineMeshNHR/ParallelMeshNHRMultiJetty/SoleNHR。未知形状NOT_MATCH。

CCU_MS、CCU_SCHED、AIV、DPU selector的每条代码分支也列入[选择逐行页](allreduce/01-entry-selection.zh-CN.md)，但这些其他算法的设备模板不纳入本例完整展开。

### 5.2 开关为1时的成本路径

{{src|cost|252|307|SelectorEngine::Run}}：按需要把配置回退至AIV_ONLY；域级tuner未初始化则初始化；按Engine tag复用或初始化CostModel；`TunerEnrichCostTable`生成成本表并允许tuner修改；{{src|cost|369|444|SelectMinCost}}跳过空算法名/负成本，选择最小有效成本，同成本保留遍历中先选项；空表或全无效返回不支持。最后释放成本表并记录结果。

这条路必须以选出的`algName`为准。即使输入≤8MiB，也不能把规则选择器的阈值结论搬过来。

两条选择路最终都回到`Selector`：校验算法名、设置CommEngine、限制AIV_ONLY回退、加载对应Kernel、设置algTag与超时/切分比例；新版本还有CCU参数跨Rank协商。这里的警告日志不等于运行时真的进入了三层拓扑分支。

## 6. 注册器怎样把字符串变成executor和template

`AicpuAllReduceSoleMeshOneShot`并不是一个同名普通函数。注册代码{{src|sole|378|393|OneShot的REGISTER_EXEC_V2与属性}}把它与三部分绑定：

| 部分 | 具体类型 | 职责 |
|---|---|---|
| executor | `InsV2AllReduceSoleExecutor<…>` | 资源计算和数据分块循环 |
| topology matcher | `TopoMatchOneLevel` | 选择覆盖所有用户Rank的有效物理层，生成算法层次 |
| template | `InsTempAllReduceMesh1DOneShot` | 当前块的Peer通信、同步和本地归约 |

`HcclExecOp`和设备端`OpOrchestrate`分别查询注册器取得executor，这是Host算资源与Device排任务的两次获取。`CalcRes`只生成需求；不能看见template的`CalcRes`就认为已经建好物理通道。

## 7. HcclExecOp与资源申请：区分命中和首次创建

读{{src|common|783|962|HcclExecOp}}，各分支按先后排列：

| 分支 | 实际执行 |
|---|---|
| 插件选择标志 | 插件ExecuteAlg；失败直接报错，不再进入本例资源/executor路径 |
| 历史fallback上下文命中 | 恢复协商过的配置与算法，重设tag，再调用HcclExecOp |
| 注册器未找到executor | 参数错误 |
| AICPU_TS或CPU | 包装用户stream为CPU_TS Thread并导出到AICPU_TS |
| HcclGetAlgRes返回UNAVAIL | FallbackOp协商/重新选择后执行；其他错误直接传播 |
| AICPU_TS或CPU的正常资源路径 | 取主Thread、导出CPU_TS句柄、取展开Thread、设置task缓存选项，进入Host发射 |
| AIV | 专属资源、核数上限、ExecuteAivCacheLogic与发射 |
| CCU | 必要时反序列化缓存，覆盖本次用户流/图资源，Host直接Orchestrate |
| 其他Engine | 必要时恢复资源，直接Orchestrate |

{{src|common|1415|1481|HcclGetAlgRes}}先`GetAlgoMeta`，再`TryReuseResource`；命中直接成功，跳过完整创建。未命中时判断首次一致性检查、匹配拓扑、调用executor的CalcRes，再按Engine申请。最后（若需要）比较交换的算子描述。

本例需求由{{src|shot|60|75|OneShot::CalcRes}}计算：Thread数为`max(templateRankSize,1)`，从Thread数少1；每从Thread需1个通知槽，主Thread需Peer数个槽；`CalcChannelRequestMesh1D`为每个Peer按RankGraph选择匹配Engine的协议链路。

本地资源次序见{{src|common|1736|1820|GetAlgResAICPU与HcclAllocAlgResourceAICPU}}：本地CCL→Thread→Channel；序列化后写入设备EngineCtx。增量建链和已有DeviceCtx复用是各自的条件分支，并非首次申请后必然重复申请。

## 8. Channel请求如何进入HCOMM

{{src|common|2064|2095|HcclGetChannel}}按算法层收集请求，再按Endpoint位置拆Device/Host；图模式先注册用户输入输出。{{src|common|2155|2190|HcclGetChannelImpl}}空组直接返回；有请求则登记一致性元信息，再调用HCOMM的`HcclChannelAcquire`。之后{{src|common|2099|2151|BuildChannelInfo}}取端口/Die属性和远端CCL地址，补全算法ChannelInfo。

详细HCOMM函数和精确代码行在[资源逐行页](allreduce/03-hcomm-resources.zh-CN.md)。按以下次序观察传入和返回的对象：

| 层/调用 | 输入 | 返回/下一层 |
|---|---|---|
| HcclChannelAcquire | comm、Engine、Peer/Endpoint/协议、notify数量、内存句柄 | 选择新域路径或兼容分支，交MyRank |
| MyRank::CreateChannels | 一组Channel请求 | 先准备Socket；批量建通道；按条件等连接；交换一致性描述；整理句柄 |
| BatchCreateChannels | 各Peer请求与连接信息 | 获取Endpoint，注册域内内存，取得EndpointPair与复用槽 |
| EndpointMgr::Get | EndpointDesc | 已有Endpoint复用；无则创建缓存 |
| EndpointMgr::RegisterMemory | Endpoint、域内注册内存、版本 | 按版本补注册；仅全部成功才提交新版本，失败可能留下已完成前缀 |
| EndpointPair::CreateChannel | Engine、reuseIdx、内存等 | 缓存槽存在则复用/更新；否则创建新基础Channel |
| HcommCollectiveChannelCreate / ChannelProcess | 基础通道描述 | 选择实际Channel/Transport，发起建链 |
| HcclChannelGetHcclBuffer | 已建立的Channel句柄 | 远端CCL地址及长度，回填ChannelInfo |

共享队列、是否需要等待、已有通道复用和Host→Device句柄准备均有条件分支，逐行页保留它们。不能把`HcclChannelQuery`返回成功误读为必然已有非零句柄，也不能把Host对象指针一律当Device Channel。

这里交换的是Endpoint/注册内存访问描述及算子一致性元信息，尚未交换用户FP32张量。公开`HcommChannelCreate`不是集合通信`HcommCollectiveChannelCreate`之后的下一次必经调用。

## 9. Host发射与AICPU设备入口：调用栈在这里跨边界

{{src|common|1028|1165|HcclAicpuKernelEntranceLaunch}}先把`resCtx`地址装入OpParam，把CPU_TS输入就绪Record排入用户流；设备主Thread的Wait与其配对。按ACL捕获/OFFLOAD/OPBASE选保序方案，然后调用{{src|common|1167|1250|AicpuKernelLaunch}}。

`AicpuKernelLaunch`取得二进制入口、建立/追加/封口参数，最后发射`aclrtLaunchKernelWithConfig`。支持展开Thread资源查询时使用展开stream；OFFLOAD、接口缺失或返回NOT_SUPPORT时用用户stream；其他查询错误返回。这里是runtime边界，不能画成普通C++直接调用`HcclLaunchAicpuKernel`。

Host随后加入第二阶段保序和用户流对设备完成的Wait。多数Record/Wait是队列依赖，并非Host线程当场等待所有设备搬运结束。

设备端{{src|kernel|353|769|HcclLaunchAicpuKernel}}的新路径：

1. Acquire已有设备通信域的占用保护并检查状态；不是重新创建comm。
2. 查设备资源对象缓存；未命中才从Host写入的描述反序列化。AllToAll特有变长参数恢复分支不会用于AllReduce。
3. BatchModeStart进入批编排，处理超时/DFX/Profiling。
4. task cache关闭或不可用：调用OpOrchestrate。启用且未命中：尝试记录任务后编排；采集或提交失败时，若接口支持则清理缓存并返回错误。task cache命中：若执行接口支持则更新地址并执行缓存，跳过executor/template。缓存未命中不等于本次不会建立缓存。
5. 正常尾部排入完成Record，BatchModeEnd提交，并ReleaseComm清除占用标记。函数中存在提前错误返回，不能把尾部Release当作全部失败路径的清理保证。

{{src|kernel|277|329|OpOrchestrate}}设置超时和批接口能力、等待输入、从注册器取得executor并调用Orchestrate；返回后由设备入口尾部排入完成通知。资源缓存命中不等于task cache命中：前者省申请，后者可能省模板生成任务。

## 10. executor先分块，template才处理Peer

{{src|sole|154|192|SoleExecutor::Orchestrate}}从resCtx恢复CCL、Threads和通道Map，读count/type并检查乘法溢出，然后进入{{src|sole|194|303|OrchestrateLoop}}。

单轮上限取通信传输上限与CCL暂存容量约束的较小者。OneShot的scratch倍数为Rank数，意味着CCL要能容纳每个源Rank的一整块。暂存上限按`HCCL_MIN_SLICE_ALIGN`对齐，再除类型大小得到本轮最大元素数。最大元素数为0报错；循环次数向上取整。

每轮记录：`currDataCount`，`inBuffBaseOff=outBuffBaseOff=processedDataCount×typeSize`，`hcclBuffBaseOff=0`，`sliceSize=currDataCount×typeSize`。最后一块用剩余元素数；每轮复用同一CCL起点。对称内存分支强制一轮；CCU单轮OPBASE还有FastLaunch上下文保存，本例不走。

读{{src|shot|104|141|OneShot::KernelRun}}：

| 次序 | 代码/作用 | 条件分支 |
|---|---|---|
| 1 | 取Thread数、当前块count/字节数、type、对称内存标志 | Thread数与模板Rank数不符报错 |
| 2 | CalcSlice | 每个源Rank一块等长CCL槽，检查累计大小 |
| 3 | PreSyncInterThreads | Thread>1时主Thread放行全部从Thread |
| 4 | RunAllReduce | 本地Copy＋Peer传输；子通信域只有自己时提早结束Peer循环 |
| 5 | PostSyncInterThreads | Thread>1时收齐全部从Thread完成 |
| 6 | PostLocalReduce | 普通内存才做；对称ReadReduce已经写入输出 |

注意第6步调用用的是`CHK_PRT`：当前宏记录失败但不会直接return，随后KernelRun仍可能返回成功。这是当前源码行为，逐行说明没有改成“所有归约错误都被上抛”。

## 11. 用4个Rank看清OneShot的地址和归约

设每Rank本轮处理两个FP32元素，`sliceSize=8`字节，本例把用户输入记为：Rank0 `[1,2]`，Rank1 `[10,20]`，Rank2 `[100,200]`，Rank3 `[1000,2000]`。理想SUM结果每个Rank都是`[1111,2222]`；这个数值示例用于解释索引，不是硬件测试结果。

{{src|shot|85|102|CalcSlice}}构造的CCL布局在每个Rank相同：

| 源Rank槽 | CCL相对偏移 | 长度 | 来源 |
|---|---|---|---|
| 0 | 0 | 8字节 | Rank0用户输入 |
| 1 | 8 | 8字节 | Rank1用户输入 |
| 2 | 16 | 8字节 | Rank2用户输入 |
| 3 | 24 | 8字节 | Rank3用户输入 |

读{{src|shot|143|205|RunAllReduce}}，以本端Rank1为例：

1. 主Thread先把本端输入`[10,20]`复制到本端输出。
2. 从Thread按`nextRank=(myRank_+queIdx)%templateRankSize_`找Peer，再从`subCommRanks_[0]`映射到通道Map键。本例Rank连续且顺序一致，依次是Rank2、3、0；真实映射不能省略。
3. 给每个Peer发送本端整块输入，目标是**对端CCL中的Rank1槽**：`remoteCcl.addr + sliceInfoVec[myRank_][0].offset_ + hcclBuffBaseOff`，本例为对端CCL＋8。
4. 各Peer也把自己的输入写入本端CCL的对应源Rank槽；本端输出此时只含自己的贡献。
5. PostSync把所有Peer从Thread汇合到主Thread；{{src|shot|232|267|PostLocalReduce}}遍历源Rank，跳过自身Rank，从本端CCL的其他槽逐个LocalReduce到本端output。

自Rank槽不要求靠网络填好，因为自身输入已经Copy到output，归约循环会跳过它。CCL槽偏移与用户buffer偏移是两个坐标：CCL按源Rank排槽；用户buffer按executor当前块累计字节偏移。

对称内存改走{{src|shot|207|230|SymmetricReadReduce}}：描述远端用户输入，ReadReduce直接合入本端输出，不走上述CCL收齐后PostLocalReduce。特殊类型/PROD的普通路径则先BatchModeEnd提交已有任务、重新Start，并ThreadJoin等待，再用CPU软件归约处理，避免CPU过早读取CCL。

## 12. 同步、批接口和底层提交到底如何连接

这部分的全部函数逐行定位在[数据面逐行页](allreduce/04-data-plane.zh-CN.md)。普通Write与对称ReadReduce的双向协议不同：

| 环节 | 普通SendRecvBatchWrite | 对称SendRecvBatchReadReduce |
|---|---|---|
| 开始 | 向接收通道发ACK；等待发送通道ACK | 向发送通道发ACK；等待接收通道ACK |
| 数据动作 | 本地src→远端dst的Write | 远端src→本地dst的ReadReduce |
| 结束 | 向发送通道发DATA_SIGNAL；等接收通道DATA_SIGNAL | 向接收通道发DATA_SIGNAL；等发送通道DATA_SIGNAL |
| 批接口支持 | 整理有效片描述，调用HcclHcommBatchTransferOnThread动态桥→HCOMM | 同类批传输，描述类型READ_REDUCE |
| 批接口不支持 | fallback SendRecvWrite，逐片HcommWriteOnThread | fallback SendRecvReadReduce，逐片HcommReadReduceOnThread |
| 零长片 | 不提交数据搬运，握手仍走 | 同理；归约片还检查count×typeSize与字节长度匹配 |

本例每Peer发送和接收通道可能是同一Peer的一条通道，但协议包装分别保留tx/rx角色。不能把“等待发送通道ACK”理解成接收数据已经完成。

主从Thread同步独立于Channel通知：`PreSyncInterThreads`对每从Thread的槽0发Record，随后从Thread Wait；`PostSyncInterThreads`由各从Thread向主Thread不同槽Record，主Thread逐槽Wait。跨Rank的ACK/DATA_SIGNAL在Channel自己的通知槽中使用。

HCCL通过动态加载与公共ABI使用HCOMM能力。`HcclChannelAcquire`、`HcclEngineCtxCreate`在此路径直接跳到HCOMM同名公开实现；可选的`HcclThreadResGetInfo`还需检查动态函数表。尤其wrapper中局部同名Wait会经过{{src|bridge|176|182|HcclThreadNotifyWaitOnThreadDefault}}与{{src|bridge|184|192|HcclChannelNotifyWaitOnThreadDefault}}，选择默认超时接口或显式fallbackTimeout接口；“看到了同名函数”不等于已经到HCOMM实现。

HCOMM的A5原语把Thread解释为设备通信执行流资源、Channel解释为轻量传输对象，继续经过TransportLite与StreamLite处理内存方向、同步资源、WQE/SQE和队列提交。批接口仍按各传输描述的类型派发；BatchModeEnd提交当前批次，并不等于整个设备所有流完成。

以普通Write和本地SUM再往下展开：

| 调用/步骤 | 继续调用 | 数据或任务怎样变化 |
|---|---|---|
| {{src|primitive|519|572|HcommWriteOnThread}} | A5分支→Transport的Write | 校验句柄/地址，登记Thread，组织本地和远端内存访问描述；其他设备走兼容适配 |
| {{src|ubtransport|664|705|UbTransportLiteImpl::Write}} | UB连接Write，再BuildUbDbSendTask | 获得本地/远端注册片、构造WQE，必要时记录cache；随后排入doorbell发送任务 |
| {{src|ubconn|367|386|UbConnLite::Write}} | ProcessSlices→FillOneSqeWrite→ProcessOneWqe | 按最大Write长度分片，填地址、Token、opcode和首尾片属性，维护队列生产索引 |
| {{src|ubconn|118|169|UbConnLite::ProcessSlices}} | 每片回调 | 整片和尾片分别构造；FIRST/MIDDLE/LAST/ONLY决定分片标志，地址相加前检查溢出 |
| {{src|primitive|163|207|HcommLocalReduceOnThread}} | A5 Thread的LocalReduce | 元素count先换算字节len，再传类型与归约运算 |
| {{src|aicputhread|350|376|AicpuTsThread::LocalReduce}} | IAicpuTsThread的SdmaReduce | 把软件参数转换为本地SDMA归约任务；不是用Host直接循环FP32数据 |
| {{src|rtsq|424|443|RtsqA5::SdmaReduce}} | 填SDMA SQE | 从src/目标dst、字节数、类型/运算生成任务，放入当前流待提交缓冲 |
| {{src|rtsq|203|255|RtsqA5::LaunchTask}} | CopySqeBufToSq→ConfigSqTail | 无待提交任务直接返回；有任务则确保空间、复制SQE、更新tail触发执行、维护缓存/计数 |
| {{src|rtsqbase|128|132|RtsqBase::ConfigSqTail}} | driver队列配置接口 | 把新的SQ tail交给外部驱动；驱动内部超出本仓源码边界 |

这里有两种不同队列描述：UB连接构造的网络WQE和执行流RTSQ的SQE（包括doorbell、SDMA、通知等）。Write的网络数据并不是把张量字节直接塞进所有RTSQ条目；连接与流各自记录描述，提交后由相应硬件资源执行。

外部runtime/driver接口是源码边界：这两个快照无法逐行解释未包含的CANN驱动实现。当前主线实际调用的legacy/ascend950中UB/RTSQ实现已纳入逐行页；legacy目录名不能用来判断本次入口是否走了旧回退。P2P/RoCE替代协议、CCU/AIV设备模板、域初始化、通用序列化/日志/配置基础设施的递归内部实现不展开；具体停止位置和已覆盖函数以[覆盖清单](allreduce/COVERAGE.zh-CN.md)为准。

## 13. 对着源码阅读时的顺序与验证

先沿主树看入口、Selector和注册类型；再看Host的CalcRes与HCOMM建链；跨过runtime发射边界后看Device的Orchestrate/KernelRun；最后用4Rank示例对照切片、Channel ACK/DATA和LocalReduce。每一步观察的是不同对象：OpParam→资源需求→资源句柄/远端地址→序列化描述→数据任务→输出。

可以直接检索逐行标记：

```bash
rg -n 'AllReduce逐行' hccl/src hcomm/src
python3 scripts/build-allreduce-guide.py --check
node --test scripts/verify-annotations.test.mjs
node scripts/verify-annotations.mjs
python3 scripts/build-directory-guides.py --check
```

生成器验证每条说明对应固定快照原文、每个列明范围中的代码行均有说明、当前源码与固定发布提交一致、所有生成页可复现。上游文本保留校验另确保移除新增中文导读后恢复最初基线字节。当前macOS环境未运行CANN编译、UT/ST或Ascend硬件执行；文档是静态调用和数据逻辑导读。
