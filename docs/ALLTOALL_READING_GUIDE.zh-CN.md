# 从算子下发读到数据传输

阅读前提：固定源码快照，主要讨论Ascend 950、新通信流程、AICPU_TS、单算子模式。通信域和用户stream已由调用者准备。以下是静态源码阅读路线，不是上板抓取的trace，也不表示每个分支每次必然执行。

## 1. 上层传入了什么

从[all_to_all_v.cc](../hccl/src/ops/all_to_all_v/all_to_all_v.cc)的`HcclAlltoAll`开始：输入输出地址、每Peer数量、数据类型、通信域、用户stream。新流程检查参数及Rank信息，把等长描述转换成AllToAllV所使用的counts/displacements，再进入`AlltoAllVOutPlace`和`AlltoAllVOutPlaceCommon`。

`AlltoAllVConstructOpParam`组织统一算子参数。复制进变长区的是数量/位移数组，而不是用户张量。`AlltoAllVExecDispatch`检查兼容回退、CCU快速路径、AIV缓存和单Rank分支；普通多Rank路径才继续选算法并进入`HcclExecOp`。

等长AllToAll、AllGather、AllReduce的顶层语义不同；它们使用公共执行/资源框架，并不意味着Peer布局、模板、归约能力和底层原语序列相同。

## 2. 公共调度先决定“需要什么”，再决定“如何获得”

阅读[op_common.cc](../hccl/src/ops/op_common/op_common.cc)：

| 函数/阶段 | 处理内容 | 不能误解为 |
|---|---|---|
| `HcclExecOp` | 取得executor，获得资源，按Engine执行 | 一次底层网络传输 |
| `HcclCalcTopoInfo` | 缓存查询或构建拓扑描述 | 每个算子都重新发现物理拓扑 |
| `TryReuseResource` | 依据tag/Engine查算法上下文 | 每次都分配所有资源 |
| `HcclGetAlgRes` / `executor->CalcRes` | 未命中时计算层级、Thread、Notify、Channel需求 | CalcRes已创建物理通道 |
| `GetAlgResWithEngine` | 进入AICPU、CCU或AIV等对应资源分支 | 顺序执行所有Engine分支 |
| `HcclAllocAlgResourceAICPU` | 本地CCL→Thread→Channel | 用户张量已经通信完成 |
| `HcclMemcpyCtxHostToDevice` | 序列化描述写入设备EngineCtx | 张量数据传到其他Rank |

Thread是通信任务执行资源抽象，不能简单理解成新建一个Linux业务线程。Channel表达本端到Peer的通信资源与协议关系；一个Peer可以使用多条Channel。

## 3. 一次通道申请怎样进入HCOMM

从`HcclGetChannel`按层级、Endpoint位置收集请求，进入`HcclGetChannelImpl`。空请求可直接返回；有请求时调用`AddExchangeInfo`和`HcclChannelAcquire`，随后用`BuildChannelInfo`查询通道属性及远端CCL地址。

`HcclChannelAcquire`虽然以Hccl开头，**实现在HCOMM**。阅读[coll_comm_res_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc)及[my_rank.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc)。新通信域的关键处理为：

| 次序 | 所在层/函数 | 输入与产物 |
|---|---|---|
| 1 | 域管理：`HcclChannelAcquire` | 规范化Peer/Endpoint/协议/内存请求，检查Engine，交给MyRank |
| 2 | 域管理：`MyRank::CreateChannels` | 准备Socket、创建或复用通道、按条件等待连接、交换域一致性信息、准备返回句柄 |
| 3 | 域资源：`BatchCreateChannels` | 按请求取得Endpoint、注册内存、选择EndpointPair及复用槽位 |
| 4 | 域资源：[EndpointMgr::Get/RegisterMemory](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc) | 缓存命中复用Endpoint；未命中创建；把域内内存注册到Endpoint |
| 5 | 基础资源：[EndpointPair::CreateChannel](../hcomm/src/base_comm/resources/endpoint_pairs/endpoint_pair.cc) | 按Engine和reuseIdx复用；需要新槽位才向下创建 |
| 6 | 基础接口：[HcommCollectiveChannelCreate](../hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc) | 规范化底层Channel描述，交`ChannelProcess::CreateChannelsLoop` |
| 7 | 返回上层：[HcclChannelGetHcclBuffer](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc) | 根据Channel取得远端CCL地址/长度，补全算法ChannelInfo |

表中第3–6项是第2项的内部展开，**不是第2项完成后再执行一遍**。某些Engine的句柄准备发生在其他适配步骤，不能一律将返回句柄强转为Host对象。公开的`HcommChannelCreate`与集合通信使用的`HcommCollectiveChannelCreate`共享部分底层逻辑，但不是连续必经调用。

### “交换资源”至少要分清两类对象

- **通道/内存访问描述**：连接协议使用的Endpoint及已注册内存访问资源。具体过程因协议和Engine而异；这让对端能识别可访问内存，不是在交换用户张量。
- **一致性描述**：`HcclCommAddExchangeInfo`登记算子元信息，建链流程交换它，上层再通过`HcclCommGetExchangeInfo`读取并比较参数。HCOMM自身配置的检查与上层算子参数检查也不是一回事。

这些描述都不同于后续`HcommWriteOnThread`/`HcommReadOnThread`实际访问的用户数据。`GetExchangeInfo`还具有读后消费语义，并不是任意大小缓冲区都可重复查询的普通getter，详见[exchange_info_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc)。

## 4. Host如何把执行交给AICPU

`HcclAicpuKernelEntranceLaunch`在用户流与设备主Thread之间组织输入就绪通知，处理展开保序，通过`AicpuKernelLaunch`下发入口，然后在用户流上排入完成等待。这里多数Record/Wait表达的是队列依赖，而不是Host线程逐条阻塞到设备完成。

在[kernel_launch.cc](../hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc)的`HcclLaunchAicpuKernel`中：

1. `HcommAcquireComm`取得已有设备通信域的占用保护，检查通信域状态。
2. 从缓存取出或反序列化算法资源上下文，恢复AllToAll变长参数。
3. `HcommBatchModeStart`进入批量任务模式，并进行DFX/Profiling处理。
4. 未走task缓存回放时，`OpOrchestrate`排入输入等待、取得executor并调用`Orchestrate`；缓存命中时可更新地址并直接回放任务。
5. 编排后排入完成Record，结束批量模式；正常路径最后`HcommReleaseComm`清除占用标记。错误路径有提前返回，不能把正常路径末尾当成覆盖所有失败路径的清理保证。

资源上下文缓存、AICPU task缓存是两件事。观测到“没有进入算法模板”可能是任务回放，不应直接判断算子没有执行。

## 5. 模板怎样产生真正的数据原语

本批以[InsTempAlltoAllVMesh1D](../hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc)为例；不是所有AllToAll算法都走此模板。

- `CalcRes`描述Peer通道、Thread和Notify需求。
- `KernelRun`取得当前块及资源，检测到PCIe通道时选择Read模式，否则选择Write模式。
- `RunALLtoALL`组织模板内部Peer轮次、自Rank复制与主从同步；外层数据分块还在executor中。
- `RunSendRecvByLoop`与`RunSendRecvByChannel`按Peer/Channel分片，计算CCL槽位，构造源/目标切片。
- `RunSendRecv`根据模式与零/非零收发长度选择双向或单向协议。

然后读[alg_data_trans_wrapper.cc](../hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc)：

| 协议 | 数据发起方 | 同步与搬运 |
|---|---|---|
| `SendWrite` + `RecvWrite` | 发送方写 | 接收方发ACK；发送方等ACK、逐片Write、发DATA_SIGNAL；接收方等DATA_SIGNAL |
| `SendRead` + `RecvRead` | 接收方读 | 提供方发ACK；接收方等ACK、逐片Read、发DATA_SIGNAL；提供方等待读完 |
| `SendRecvWrite` | 本Rank同时收发 | 向接收通道发ACK、等发送通道ACK、写数据、发写完成、等收到对端完成 |
| `PreSyncInterThreads` / `PostSyncInterThreads` | 本地主从Thread | 开始放行、结束汇合；与跨Rank的Channel通知分开 |
| `LocalCopy` | 本地Thread | 自Rank数据或CCL/用户区之间的复制；无远端Channel |
| `LocalReduce` | 本地Thread或回退实现 | 元素归约；不等于完整AllReduce，也不是AllToAll必需步骤 |

普通Write中转路径的数据先到对端CCL，再复制到对端用户输出；普通Read中转路径先准备本地CCL供Peer读取。直接访问远端用户内存的分支不走同样的中转复制。

## 6. HCOMM原语最后在哪里实现

阅读[aicpu_ts_primitives_c_adpt.cc](../hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc)：

| 类别 | 本批解释的接口 | 关键语义 |
|---|---|---|
| 数据搬运 | `HcommLocalCopyOnThread` | 本地src→dst，len为字节 |
| 数据搬运 | `HcommWriteOnThread` | 本地src→远端dst，经已有Channel |
| 数据搬运 | `HcommReadOnThread` | 远端src→本地dst，经已有Channel |
| 数据计算 | `HcommLocalReduceOnThread` | count为元素数，另传dataType/reduceOp |
| 同步控制 | `HcommThreadNotifyRecordOnThread` / `WaitOnThread` | 本地Thread的通知槽；Record目标Thread和执行Thread可不同 |
| 同步控制 | `HcommChannelNotifyRecordOnThread` / `WaitOnThread` | 跨Rank通知；发送remoteNotifyIdx与接收localNotifyIdx配对 |
| 执行控制 | `HcommBatchModeStart` / `End` | 设置任务提交模式，不等于全设备完成同步 |
| 通信域使用生命周期 | `HcommAcquireComm` / `ReleaseComm` | 取得/释放已有域的使用保护（950/960使用isUsed标记），不是创建/销毁整个域 |

A5数据分支把Channel解释为`BaseTransportLiteImpl`并配合`StreamLite`生成传输任务；其他设备有旧适配路径。HCCL经[src/common/hcomm_dlsym](../hccl/src/common/hcomm_dlsym)动态加载HCOMM接口；本文直接跳到HCOMM实现便于阅读，不意味着新增了跨仓私有头文件依赖。Wait包装还可能选择`WithDefaultTimeout`接口。

## 7. CCU / AIV不要套用AICPU的整条原语链

在公共资源框架中分别阅读`GetAlgResCcu`/`HcclAllocAlgResourceCcu`和`GetAlgResAiv`/`HcclAllocAlgResourceAiv`：CCU涉及专用通道及Kernel资源，受版本与资源可用性限制；AIV涉及通信信息、标记区与远端地址表。AIV的资源描述存储在CPU_TS上下文不代表计算改在CPU执行。

特别注意：当前`HcclAivKernelEntranceLaunch`只确定核数上限，真正执行还需跟踪`ExecuteAivCacheLogic`。CCU指令生成、AIV核内同步和完整协议实现**尚未逐函数注释**，请查[覆盖边界](ANNOTATION_COVERAGE.md)，不要据此宣称Engine之间全接口通用。
