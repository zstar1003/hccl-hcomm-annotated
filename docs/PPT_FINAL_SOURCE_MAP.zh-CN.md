# 最终版 PPT → 源码逐页对照

这份文件用于**一边看最终版 PPT，一边定位实现和设计测试**。P1–P14 对照 AllToAll 执行流程，P15–P23 对照按 Engine 分类的接口总表；不要求先读完整仓库。

## 对应版本与阅读约定

| 项目 | 本文锁定的版本 |
|---|---|
| 最终版 PPT | [FINAL_HCOMM_流程与Engine_API测试总表_r2.pptx](https://github.com/zstar1003/artifact-template-dense-tech-brief/blob/fec8fbc5011634be04fd2266717435c71aac5d24/presentations/hccl-hcomm/FINAL_HCOMM_%E6%B5%81%E7%A8%8B%E4%B8%8EEngine_API%E6%B5%8B%E8%AF%95%E6%80%BB%E8%A1%A8_r2.pptx)，共 23 页 |
| PPT SHA-256 | `3dcd029bbdd1f697366adbe44631e5c2f57c19b41061a5b7398d6933dc27caaf` |
| 本文源码链接 | 本注释仓库提交 `ece79923c7b7ba17f07a7b17ae80390fe3b12450` |
| HCCL 上游快照 | `170ddeec539b4d693028ce6e0cf5c58933e4d46d` |
| HCOMM 上游快照 | `87ce550f8f7c584e0ed89b0ec56699553d9c332d` |

使用方法：先找 PPT 页码，再在表内找到框中的函数名，点击“源码定位”。**“调用”链接指使用点或包含使用点的函数，“实现”链接指函数体；标明“头文件包装”的不是底层硬件实现。** 点击后 GitHub 显示完整目录与已定位的行；本地也可在对应文件搜索同名函数。

需要一直保留的边界：

- P1–P14 固定为 Ascend 950、新流程、AICPU_TS、OPBASE、多 Rank、非零数据、普通用户内存的 Mesh Write 示例；资源未命中、Task Cache 关闭、无算法插件。通信域与用户 stream 已存在，不从通信域初始化讲起。
- 算法示例是 `AicpuAllToAllSoleMeshSingleChannel`，不是所有 AllToAll 的唯一实现。PCIe/Read、对称内存、多通道、大数据、图模式、缓存及回退会改变路径。
- HCCL 负责算子、选算法和任务编排；HCOMM 提供域资源及通信原语。**不能按 Hccl/Hcomm 前缀判定仓库归属**，例如 `HcclChannelAcquire` 实现在 HCOMM。
- 表的上下次序是源码阅读顺序，嵌套展开不是再执行一次；记录通知、排入等待和任务提交不等于 Host 阻塞，也不等于所有 Rank 的数据已经完成。
- 这里是固定版本的静态分析，未上板采集调用 trace；P15–P23 的“共用”是路径族存在相同接口使用点，不表示每个算子每次必调。资源缓存与 Task Cache 是两种不同机制。
- PPT 原备注使用上游行号；本表已按**新增中文注释后的行号重新定位**。本文不修改 PPT，不收录参考照片。

## 页码导航

| 流程页 | 内容 | 接口总表页 | 内容 |
|---|---|---|---|
| [P1](#p1) | 跨层总览 | [P15](#p15) | 共用：域、通道、交换 |
| [P2](#p2) | 参数接入 | [P16](#p16) | 共用：Rank、拓扑、链路 |
| [P3](#p3) | 展开模式与选算法 | [P17](#p17) | 部分共用：Thread、远端缓冲 |
| [P4](#p4) | 用户流绑定与资源复用 | [P18](#p18) | AICPU_TS：上下文与执行 |
| [P5](#p5) | CCL、Thread、Channel 申请 | [P19](#p19) | AICPU_TS：搬运、归约、通知 |
| [P6](#p6) | 建链内部展开 | [P20](#p20) | CCU：资源与 Kernel 注册 |
| [P7](#p7) | 资源描述交给设备 | [P21](#p21) | CCU：发射与 C++ 数据面 |
| [P8](#p8) | Host 发射与等待 | [P22](#p22) | AIV：远端地址与向量核 |
| [P9](#p9) | AICPU 入口与恢复 | [P23](#p23) | 特殊、兼容、清理分支 |
| [P10](#p10) | 分块、轮次、Peer、Channel |  |  |
| [P11](#p11) | 单 Peer 握手与写入 |  |  |
| [P12](#p12) | 字节路径与 Write 实现 |  |  |
| [P13](#p13) | 汇合、完成通知、用户流继续 |  |  |
| [P14](#p14) | 两种资源/信息交换 |  |  |

<a id="p1"></a>

## P1｜AllToAll 跨层执行总览

汇报主线：**接请求 → 决定算法及资源需求 → 取得资源并建链 → 交付执行上下文 → 下发 AICPU → 编排通知和搬运 → 汇合并放行用户流。** 资源元数据与用户数据沿不同路径流转。

| PPT 阶段/主要 API | 这一阶段实际做什么 | 首先打开的源码 | 展开页 |
|---|---|---|---|
| `HcclAlltoAll` | 接收输入/输出地址、每 Peer 数量、类型、通信域及 stream | [HCCL/all_to_all_v.cc:26][src-entry-26] | P2 |
| `HcclConfigGetInfo`、`HcclRankGraphGetLinks` | 获取展开配置、构造拓扑并辅助选择算法；不是先搬数据 | [HCCL/op_common.cc:3468][src-op-3468]；[HCCL/op_common.cc:1301][src-op-1301] | P3 |
| `HcclGetAlgRes` | 查缓存；未命中才计算 Thread/Notify/Channel 需求并申请 | [HCCL/op_common.cc:1386][src-op-1386] | P4 |
| `HcclGetHcclBuffer`、`HcclThreadAcquireWithConfig`、`HcclChannelAcquire` | 取得 CCL、执行资源、到 Peer 的通道 | [HCCL/op_common.cc:1756][src-op-1756] | P5–P6 |
| `HcclEngineCtxCreate` / `HcclEngineCtxCopy` | 把序列化后的资源描述交给设备 | [HCCL/op_common.cc:1738][src-op-1738] | P7 |
| `HcclAicpuKernelEntranceLaunch` / `aclrtLaunchKernelWithConfig` | 组织输入依赖、展开保序和入口发射 | [HCCL/op_common.cc:1015][src-op-1015]；[HCCL/op_common.cc:1147][src-op-1147] | P8 |
| `HcclLaunchAicpuKernel` / `OpOrchestrate` | 恢复资源/参数，交设备 executor 展开算法 | [HCCL/kernel_launch.cc:348][src-kernel-348]；[HCCL/kernel_launch.cc:278][src-kernel-278] | P9–P10 |
| `HcommWriteOnThread` / `HcommLocalCopyOnThread` | 普通 Write 路径：本端输入 → 对端 CCL → 对端输出；自 Rank 本地复制 | [HCCL/alg_data_trans_wrapper.cc:435][src-wrap-435]；[HCCL/ins_temp_all_to_all_v_mesh_1D.cc:579][src-mesh-579] | P11–P12 |
| `PostSyncInterThreads` / `HcommThreadNotifyRecordOnThread` | 主从汇合后排入最终通知，使用户流上的完成等待得到满足 | [HCCL/alg_data_trans_wrapper.cc:1061][src-wrap-1061]；[HCCL/kernel_launch.cc:612][src-kernel-612] | P13 |
| 建链时“交换资源” | 交换远端可访问资源描述；另一路交换算子一致性信息 | [HCOMM/my_rank.cc:1236][src-myrank-1236]；[HCCL/inconsistent_check.cc:52][src-check-52] | P14 |

<a id="p2"></a>

## P2｜请求接入与 AllToAll 参数整理

| 步骤 / 函数 | 作用、输入与产物 | 源码定位 |
|---|---|---|
| `HcclAlltoAll` | 检查版本和设备分支，接收等长 AllToAll 请求。`sendCount/recvCount` 是每 Peer 元素数，不是字节数 | [HCCL/all_to_all_v.cc:26][src-entry-26] |
| `CheckAlltoAllInputPara` | 校验通信域、stream、地址、数量和类型等输入约束 | 调用在入口；实现 [HCCL/all_to_all_v.cc:481][src-entry-481] |
| `HcclGetRankSize` | 取域内 Rank 数，决定 counts/displacements 数组长度 | 调用 [HCCL/all_to_all_v.cc:50][src-entry-50]；实现 [HCOMM/coll_comm_rank_graph_a_adpt.cc:349][src-graph-349] |
| `HcclGetCommName` | 取域标识，用于 tag、上下文和日志关联 | 调用 [HCCL/all_to_all_v.cc:52][src-entry-52]；实现 [HCOMM/op_base.cc:1514][src-legacy-1514] |
| `HcclGetRankId` | 取本 Rank 编号，后续决定本地和 Peer 数据布局 | 调用 [HCCL/all_to_all_v.cc:54][src-entry-54]；实现 [HCOMM/coll_comm_rank_graph_a_adpt.cc:369][src-graph-369] |
| counts 初始化 + `ConvertAlltoAllParam` | 入口创建等长 `sendCounts/recvCounts`；转换函数填写 `sdispls/rdispls`。这里整理描述数组，不搬张量 | 数组 [HCCL/all_to_all_v.cc:66][src-entry-66]；位移 [HCCL/all_to_all_v.cc:418][src-entry-418] |
| `AlltoAllVOutPlace` → `AlltoAllVOutPlaceCommon` | 进入共享 AllToAllV 处理框架，单算子路径使用 OPBASE | [HCCL/all_to_all_v.cc:856][src-entry-856]；[HCCL/all_to_all_v.cc:799][src-entry-799] |
| `AlltoAllVConstructOpParam` / `ConstructVarData` | 组织统一 `OpParam`，保存地址、类型、数量和位移，变长区保存描述数组 | [HCCL/all_to_all_v.cc:654][src-entry-654]；[HCCL/all_to_all_v.cc:624][src-entry-624] |

**对 PPT 的精确补充：**`ConvertAlltoAllParam` 不是创建 counts 的地方；counts 在入口构造。数量/位移以元素计，后续构造数据切片时再换算为字节。

<a id="p3"></a>

## P3｜展开模式、拓扑与算法选择

| 步骤 / 函数 | 作用和分支条件 | 源码定位 |
|---|---|---|
| `HcclGetOpExpansionMode` → `HcclConfigGetInfo` | 读取/决定展开模式，设置本次执行配置与 Engine | HCCL [HCCL/op_common.cc:3468][src-op-3468]、[HCCL/op_common.cc:3513][src-op-3513]；HCOMM [HCOMM/op_base.cc:4152][src-legacy-4152] |
| `AlltoAllVExecDispatch` | 处理兼容、快速路径、缓存及单 Rank 等分支；普通多 Rank 才走常规 selector/executor | [HCCL/all_to_all_v.cc:746][src-entry-746] |
| `HcclCommGetStatus` | 进入 `Selector` 后，能力支持时检查通信域运行状态；不是创建通信域 | 调用 [HCCL/op_common.cc:164][src-op-164]；实现 [HCOMM/coll_comm_c_adpt.cc:19][src-domain-19] |
| `HcclCalcTopoInfo` → `HcclEngineCtxGet` | 先按 tag 取拓扑缓存；已有描述可反序列化复用 | [HCCL/op_common.cc:1301][src-op-1301]；[HCOMM/engine_ctx_c_adpt.cc:86][src-ctx-86] |
| `InitRankInfo` → RankGraph 查询 | 缓存缺失时组织 Rank、层、拓扑实例及链路信息；详细 API 逐项见 P16 | [HCCL/topo_host.cc:68][src-topo-68]；实例/端点查询 [HCCL/topo_host.cc:885][src-topo-885] |
| `HcclEngineCtxCreate` | 缓存序列化拓扑，以后相同条件可复用；并非创建物理拓扑 | 调用 [HCCL/op_common.cc:1301][src-op-1301]；实现 [HCOMM/engine_ctx_c_adpt.cc:25][src-ctx-25] |
| `Selector` / `AlltoAllAutoSelector::SelectAicpuAlgo` | 结合 Engine、拓扑、数据量、配置选择具体算法名 | Selector [HCCL/op_common.cc:159][src-op-159]；自动选择 [HCCL/alltoall_auto_selector.cc:131][src-auto-131] |
| `AicpuAllToAllSoleMeshSingleChannel` → `HcclExecOp` | 名字通过注册表对应 executor 与 Mesh 模板，进入公共执行框架 | 注册 [HCCL/ins_v2_all_to_all_v_sole_executor.cc:574][src-exec-574]；公共执行 [HCCL/op_common.cc:783][src-op-783] |

这里的拓扑 API 是**按需要查询的集合**，不是每个算子都按相同顺序调用全部查询。选择器还有大数据、多通道等分支；不能把本 PPT 的示例算法名写成 AllToAll 固定实现。

<a id="p4"></a>

## P4｜用户流绑定与算法资源复用

| 步骤 / 函数 | 作用和产物 | 源码定位 |
|---|---|---|
| Host `GetAlgExec` | 按算子类型及算法名取得 Host executor，供计算资源需求 | 使用点 [HCCL/op_common.cc:838][src-op-838] |
| `HcclThreadAcquireWithStream` | 将已有用户 stream 包装为 CPU_TS Thread；不是再创建一条相同用户流 | 调用 [HCCL/op_common.cc:854][src-op-854]；实现 [HCOMM/thread_c_adpt.cc:318][src-thread-318] |
| `HcclThreadExportToCommEngine` | 把该用户 Thread 导出给 AICPU_TS，设备最终可以通知用户流 | 调用 [HCCL/op_common.cc:856][src-op-856]；实现 [HCOMM/thread_c_adpt.cc:498][src-thread-498] |
| `HcclGetAlgRes` / `GetAlgoMeta` | 读取算法属性，组织后续的资源复用或分配 | [HCCL/op_common.cc:1386][src-op-1386] |
| `TryReuseResource` → `HcclEngineCtxGet` | 按算法 tag/Engine 查已存在上下文；满足条件则复用 | [HCCL/op_common.cc:1350][src-op-1350]；[HCOMM/engine_ctx_c_adpt.cc:86][src-ctx-86] |
| `CalcAlgHierarchyInfoV2` / `CalcRes` | 未命中时计算层级、主从 Thread、Notify、Peer/Channel 需求，产物是 `AlgResourceRequest` | 调用 [HCCL/op_common.cc:1386][src-op-1386]；executor [HCCL/ins_v2_all_to_all_v_sole_executor.cc:68][src-exec-68]；模板 [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:64][src-mesh-64] |
| `GetAlgResWithEngine` → `GetAlgResAICPU` | 根据 Engine 分流申请；这里才进入 AICPU 资源获取 | [HCCL/op_common.cc:1547][src-op-1547]；[HCCL/op_common.cc:1698][src-op-1698] |

**复用分支跳过的是部分资源创建，不是跳过本次用户数据通信。** 上述资源缓存也不能等同于 AICPU Task Cache 的任务回放。

<a id="p5"></a>

## P5｜CCL、设备 Thread 与 Peer 通道申请

| 步骤 / 函数 | 做什么 / 得到什么 | 源码定位 |
|---|---|---|
| `GetAlgResAICPU` → `HcclAllocAlgResourceAICPU` | 首次申请时依次准备 CCL、Thread、Channel；之后序列化 | [HCCL/op_common.cc:1698][src-op-1698]；[HCCL/op_common.cc:1756][src-op-1756] |
| `HcclGetHcclBuffer` | 取得通信域持有的本地中转区地址和容量，不等于每次重新分配 | 调用 [HCCL/op_common.cc:1756][src-op-1756]；实现 [HCOMM/comm_mem_c_adpt.cc:152][src-mem-152] |
| `HcclGetThread` → `HcclGetAicpuThread` | 主 Thread + 从 Thread 用于通信任务；另有 unfoldThread 用于入口展开 | [HCCL/op_common.cc:1860][src-op-1860]；[HCCL/op_common.cc:1823][src-op-1823] |
| `HcclGetThreadWithConfig` / `ThreadConfigInit` | 填各 Thread 的通知槽数；主 Thread 额外留 Host/Device 同步槽 | [HCCL/op_common.cc:1776][src-op-1776] |
| `HcclThreadAcquireWithConfig` | 接口能力存在时按逐 Thread 配置申请 | 调用 [HCCL/op_common.cc:1776][src-op-1776]；实现 [HCOMM/thread_c_adpt.cc:151][src-thread-151] |
| `HcclThreadAcquire` | 旧能力分支按统一最大通知数申请；不是 WithConfig 调用失败后必然重试 | 调用 [HCCL/op_common.cc:1823][src-op-1823]；实现 [HCOMM/thread_c_adpt.cc:218][src-thread-218] |
| `SaveMainThreadInfo` / `SaveUnfoldThreadInfo` | 保存主 Thread 和展开 Thread，后续发射、同步时取出 | [HCCL/op_common.cc:1936][src-op-1936]；[HCCL/op_common.cc:1956][src-op-1956] |
| `HcclGetChannel` → `HcclGetChannelImpl` | 按层级整理请求；非空请求登记交换信息并批量申请通道 | [HCCL/op_common.cc:2016][src-op-2016]；[HCCL/op_common.cc:2102][src-op-2102] |
| `HcclChannelAcquire` → `BuildChannelInfo` | HCOMM 获得通道，HCCL 再补齐通道属性和远端 CCL，供模板使用 | 实现 [HCOMM/coll_comm_res_c_adpt.cc:759][src-res-759]；返回后 [HCCL/op_common.cc:2049][src-op-2049] |

<a id="p6"></a>

## P6｜HcclChannelAcquire 建链内部展开

以下是**新通信域 / MyRank 分支**。缓存、Engine 和协议会使内部某些创建/等待步骤跳过；不把它推广成所有旧流程的唯一实现。

| 层级 / 步骤 | 函数与职责 | 源码定位 |
|---|---|---|
| HCOMM 域接口入口 | `HcclChannelAcquire`：校验通信域、Engine 和请求，转入域资源管理 | [HCOMM/coll_comm_res_c_adpt.cc:759][src-res-759] |
| 域内编排 | `MyRank::CreateChannels`：组织 Socket、创建/复用、连接检查、交换及句柄准备 | [HCOMM/my_rank.cc:1236][src-myrank-1236] |
| 建链辅助连接 | `BatchCreateSockets`：准备与 Peer 交换控制信息所需的 Socket | [HCOMM/my_rank.cc:567][src-myrank-567] |
| 批量通道准备 | `BatchCreateChannels`：逐个请求取得 Endpoint、准备内存、选择 RankPair/EndpointPair | [HCOMM/my_rank.cc:678][src-myrank-678] |
| Endpoint 创建 | `EndpointMgr::Get` → `HcommEndpointCreate`：缓存未命中才创建端点 | 域管理 [HCOMM/endpoint_mgr.cc:71][src-epmgr-71]；接口 [HCOMM/hcomm_endpoint_c_adpt.cc:307][src-bep-307] |
| Endpoint 监听 | `HcommEndpointStartListen`：按本次端点/协议参数启动监听 | 调用 [HCOMM/my_rank.cc:732][src-myrank-732]；实现 [HCOMM/hcomm_endpoint_c_adpt.cc:365][src-bep-365] |
| 内存准备 | `PrepareMemHandles` → `RegisterCommMemsToEndpoint` / `EndpointMgr::RegisterMemory`：将 CCL 及请求携带的内存描述准备为端点资源 | [HCOMM/my_rank.cc:238][src-myrank-238]；[HCOMM/my_rank.cc:227][src-myrank-227]；[HCOMM/endpoint_mgr.cc:125][src-epmgr-125] |
| 端点内存注册 | `HcommMemReg`：得到端点可使用的内存句柄 | 调用 [HCOMM/endpoint_mgr.cc:155][src-epmgr-155]；实现 [HCOMM/hcomm_mem_c_adpt.cc:35][src-bmem-35] |
| Peer/端点对通道 | `EndpointPair::CreateChannel`：根据 Engine、reuseIdx 复用槽位或新建 | [HCOMM/endpoint_pair.cc:202][src-epair-202] |
| 底层集合通信创建 | `HcommCollectiveChannelCreate` → `ChannelProcess::CreateChannelsLoop`：构造具体协议通道 | [HCOMM/hcomm_channel_c_adpt.cc:440][src-bchan-440]；[HCOMM/channel_process.cc:89][src-cprocess-89] |
| 推进并检查连接 | `BatchConnectChannels` → `ChannelProcess::ChannelGetStatus`：轮询/推进协议状态，处理就绪或失败 | [HCOMM/my_rank.cc:1050][src-myrank-1050]；[HCOMM/channel_process.cc:284][src-cprocess-284] |
| 交换域一致性信息 | `BatchExchangeAndCheckConsistency`：交换本轮登记的上层元信息，并处理 HCOMM 自身一致性信息 | [HCOMM/my_rank.cc:582][src-myrank-582] |
| 返回适合 Engine 的句柄 | `FinalizeChannelsByEngine`：按 Engine 进行最终句柄/资源准备 | [HCOMM/my_rank.cc:1196][src-myrank-1196] |

**两个容易看错的名字：**本路径用 `HcommCollectiveChannelCreate`，不是先 `HcommChannelCreate` 再调用它；内部 `ChannelProcess::ChannelGetStatus` 也不等于调用了公开的 `HcommChannelGetStatus`。

<a id="p7"></a>

## P7｜通道信息、上下文复制与主 Thread 导出

| 步骤 / 函数 | 作用和产物 | 源码定位 |
|---|---|---|
| `BuildChannelInfo` | 把返回句柄补成算法的 `ChannelInfo`：远端 Rank、协议、端点属性、远端 CCL 等 | [HCCL/op_common.cc:2049][src-op-2049] |
| `HcclRankGraphGetEndpointInfo` | 获取所选端点属性，不是再建一个端点 | 使用点在 `BuildChannelInfo`；实现 [HCOMM/coll_comm_rank_graph_a_adpt.cc:308][src-graph-308] |
| `HcclChannelGetHcclBuffer` | 查询该通道关联的远端 CCL 地址/大小 | [HCOMM/channel_c_adpt.cc:86][src-chan-86] |
| `HcommChannelGetRemoteMems` | 从底层通道取得已交换/解析的远端内存列表；不是当场再发送一份张量 | [HCOMM/hcomm_channel_c_adpt.cc:759][src-bchan-759] |
| `Serialize` | 将拓扑、Thread、Channel、缓冲描述组织为可交付设备的字节序列 | 使用点 [HCCL/op_common.cc:1715][src-op-1715] |
| `HcclMemcpyCtxHostToDevice` | 组织上下文的设备存储与复制，返回设备上下文地址 | [HCCL/op_common.cc:1738][src-op-1738] |
| `HcclEngineCtxCreate` | 申请/管理设备侧执行上下文保存空间 | 调用在上述函数；实现 [HCOMM/engine_ctx_c_adpt.cc:25][src-ctx-25] |
| `HcclEngineCtxCopy` | 将序列化资源描述复制进去，非用户张量传输 | 调用在上述函数；实现 [HCOMM/engine_ctx_c_adpt.cc:139][src-ctx-139] |
| `GetMainThreadInfo` → `HcclThreadExportToCommEngine` | 取设备主 Thread，导出为 CPU_TS 可引用的句柄，使 Host 能发送输入通知 | [HCCL/op_common.cc:1994][src-op-1994]；调用 [HCCL/op_common.cc:887][src-op-887]；实现 [HCOMM/thread_c_adpt.cc:498][src-thread-498] |

P4 导出的是**用户 Thread：CPU_TS → AICPU_TS**；本页导出的是**设备主 Thread：AICPU_TS → CPU_TS**。方向相反，服务于不同的通知关系。

<a id="p8"></a>

## P8｜Host 输入通知、Kernel 发射与完成等待

| 步骤 / 函数 | 做什么 | 源码定位 |
|---|---|---|
| `HcclAicpuKernelEntranceLaunch` | 组装本次上下文/通知参数，组织入口发射前后的依赖 | [HCCL/op_common.cc:1015][src-op-1015] |
| `HcommThreadNotifyRecordOnThread` | 用户 stream 执行到此通知时，设备主 Thread 才能越过输入等待 | 调用 [HCCL/op_common.cc:1095][src-op-1095]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:206][src-prim-206] |
| `HcclOrderLaunchToOrderStream` | 能力及模式满足时建立按序展开依赖；内部专用 Thread 接口见 P18 | [HCCL/order_launch.cc:152][src-order-152] |
| `AicpuKernelLaunch` | HCCL 内部函数：查设备入口、打包参数并调用 ACL | [HCCL/op_common.cc:1147][src-op-1147] |
| `aclrtBinaryGetFunction` | 从已加载 binary 获得 `HcclLaunchAicpuKernel` 的函数句柄 | HCCL 调用 [HCCL/op_common.cc:1155][src-op-1155] |
| `aclrtKernelArgsInit` | 创建 Kernel 参数包 | HCCL 调用 [HCCL/op_common.cc:1163][src-op-1163] |
| `aclrtKernelArgsAppend` | 加入 `OpParam` 和变长参数区 | HCCL 调用 [HCCL/op_common.cc:1173][src-op-1173] |
| `aclrtKernelArgsFinalize` | 完成参数封装 | HCCL 调用 [HCCL/op_common.cc:1181][src-op-1181] |
| `aclrtLaunchKernelWithConfig` | 按能力在展开流或用户流提交 AICPU Kernel；ACL 内部实现不在这两个仓库中 | 分支调用 [HCCL/op_common.cc:1206][src-op-1206] |
| `HcclOrderLaunchToKernelStream` | 组织发射后的按序依赖 | [HCCL/order_launch.cc:229][src-order-229] |
| 完成 `Wait` | 在用户流排入设备完成等待；会经 HCCL Default 包装选择实际 HCOMM Wait 入口 | 包含使用点 [HCCL/op_common.cc:1015][src-op-1015]；包装 [HCCL/hcomm_primitives_dl.cc:176][src-dl-176]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:243][src-prim-243] |

**命名边界：**这里是内部 `AicpuKernelLaunch` 调 ACL；`HcclAicpuKernelLaunch` 是另一个接口，本版本此入口对 Send/Recv 的条件分支才调用它。Host 继续提交与设备开始执行可以并发。

<a id="p9"></a>

## P9｜AICPU 入口、资源恢复与算法启动

| 步骤 / 函数 | 做什么 | 源码定位 |
|---|---|---|
| `HcclLaunchAicpuKernel` | 接收 ACL 交付的本次算子参数，进入设备侧通信编排入口 | [HCCL/kernel_launch.cc:348][src-kernel-348] |
| `HcommAcquireComm` | 取得已有设备通信域的使用引用/保护，非创建新通信域 | 调用 [HCCL/kernel_launch.cc:362][src-kernel-362]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:1138][src-prim-1138] |
| `HcclOrderLaunchNotifyRecord` | 开启按序能力时发送设备入口通知 | [HCCL/kernel_launch.cc:326][src-kernel-326] |
| `DeserializeResCtx` 或资源缓存对象 | 从传入序列化上下文恢复 Thread、Channel、缓冲描述；已有对象可复用 | [HCCL/kernel_launch.cc:229][src-kernel-229]；使用点 [HCCL/kernel_launch.cc:444][src-kernel-444] |
| `RestoreVarDataAlltoAllV` | 恢复本次 counts/displacements 指针；不是重新执行 Host 上的入口转换 | [HCCL/kernel_launch.cc:922][src-kernel-922] |
| `HcommBatchModeStart` | 进入批量组织/提交通信任务的模式 | 调用 [HCCL/kernel_launch.cc:480][src-kernel-480]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:1131][src-prim-1131] |
| `OpOrchestrate` | 未走任务回放时配置超时、排入输入等待、取得 executor 并编排 | [HCCL/kernel_launch.cc:278][src-kernel-278] |
| `HcclThreadNotifyWaitOnThreadDefault` → `HcommThreadNotifyWaitOnThreadWithDefaultTimeout` / 显式超时 Wait | 设备主 Thread 等 Host 输入通知；能力决定两个接口之一 | 使用点在 `OpOrchestrate`；包装 [HCCL/hcomm_primitives_dl.cc:176][src-dl-176]；默认超时实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:481][src-prim-481] |
| 设备 `GetAlgExec` → `executor->Orchestrate` | 按算法名找到设备 executor，进入 P10 的分块/模板处理 | 使用点 [HCCL/kernel_launch.cc:311][src-kernel-311]；executor [HCCL/ins_v2_all_to_all_v_sole_executor.cc:89][src-exec-89] |

“等待输入后执行”指**设备队列的依赖顺序**，不是 C++ 编排线程必须停在 Wait 函数直到张量可用；它可以继续生成依赖之后的任务。

<a id="p10"></a>

## P10｜设备算法按数据块、轮次、Peer 和 Channel 展开

需要分清四层：**外层数据块 → 模板内 Peer 轮次 → 当前轮 Peer → Peer 的 Channel/端口切片**，不能把它们当成同一个循环。

| 所在层 / 函数 | 做什么 | 源码定位 |
|---|---|---|
| executor `Orchestrate` | 取算子参数、层级和资源，准备本次执行 | [HCCL/ins_v2_all_to_all_v_sole_executor.cc:89][src-exec-89] |
| executor `OrchestrateLoop` | 依据 CCL 能容纳的数据组织多次数据块执行，更新当前块参数 | [HCCL/ins_v2_all_to_all_v_sole_executor.cc:202][src-exec-202] |
| 模板 `CalcScratchMultiple` | 计算当前缓冲模式所需的 scratch 倍数，辅助决定分块大小 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:106][src-mesh-106] |
| 模板 `KernelRun` | 接收当前块及 Thread/Channel 资源；依据链路等条件选择 Read/Write | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:172][src-mesh-172] |
| `RunALLtoALL` / `CalcCommLoops` | 计算 Peer 通信轮数并组织当前块的流程 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:227][src-mesh-227]；[HCCL/ins_temp_all_to_all_v_mesh_1D.cc:135][src-mesh-135] |
| `PreSyncInterThreads` | 主 Thread 向从 Thread 放行，保证从流数据任务依赖主流此前任务 | [HCCL/alg_data_trans_wrapper.cc:1024][src-wrap-1024] |
| `CalcCommRankSetForOneLoop` | 选择本轮参与通信的 Peer 集合 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:115][src-mesh-115] |
| `LocalCopyForMyRank` → `LocalCopy` | 第一轮处理给自己的那一段，输入直接复制到本地输出 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:205][src-mesh-205]；[HCCL/alg_data_trans_wrapper.cc:891][src-wrap-891] |
| `RunSendRecvByLoop` | 遍历本轮 Peer，取出对应通道列表，按端口分割发送/接收数据 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:283][src-mesh-283] |
| `RunSendRecvByChannel` | 对每条 Channel 组织切片、选 Thread、执行收发及后拷贝 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:342][src-mesh-342] |
| `CalcCclBuffIdx` / `BuildDataSlices` | 计算本地/远端 CCL 槽位及实际地址、偏移、大小；槽位不是简单等于 Rank 号 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:141][src-mesh-141]；[HCCL/ins_temp_all_to_all_v_mesh_1D.cc:443][src-mesh-443] |
| `RunSendRecv` → `SendRecvWrite` | 双向非零普通 Write 分支进入握手+写；仅单向非零时改用 SendWrite/RecvWrite | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:400][src-mesh-400]；[HCCL/alg_data_trans_wrapper.cc:435][src-wrap-435] |
| `PostCopy` / `PostSyncInterThreads` | 每 Channel 接收完成后复制到输出；当前块尾汇合，再执行下一块 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:579][src-mesh-579]；[HCCL/alg_data_trans_wrapper.cc:1061][src-wrap-1061] |

<a id="p11"></a>

## P11｜一个 Peer 的握手、远端写与接收后拷贝

以下按 `SendRecvWrite` 的双向非零分支排列。ACK 和 DATA_SIGNAL 是**不同用途的通知槽**，均不同于本地主从 Thread 通知。

| 次序 | API / 内部函数 | 含义 | 源码定位 |
|---|---|---|---|
| 1 | `HcommChannelNotifyRecordOnThread(...recvChannel, ACK)` | 告诉向本 Rank 发送数据的对端：本端接收资源可用 | 调用 [HCCL/alg_data_trans_wrapper.cc:444][src-wrap-444]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:1020][src-prim-1020] |
| 2 | `HcommChannelNotifyWaitOnThread(...sendChannel, ACK)` | 等要写入的对端确认其接收资源可用 | 调用 [HCCL/alg_data_trans_wrapper.cc:448][src-wrap-448]；实际能力适配 [HCCL/hcomm_primitives_dl.cc:184][src-dl-184] |
| 3 | `HcommWriteOnThread` | 对每个非零切片，将本端用户输入写入对端 CCL；长度为字节 | 调用 [HCCL/alg_data_trans_wrapper.cc:462][src-wrap-462]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:505][src-prim-505] |
| 4 | `HcommChannelNotifyRecordOnThread(...sendChannel, DATA_SIGNAL)` | 在写任务之后提交写完成通知 | 调用 [HCCL/alg_data_trans_wrapper.cc:466][src-wrap-466] |
| 5 | `HcommChannelNotifyWaitOnThread(...recvChannel, DATA_SIGNAL)` | 等对端写完本端接收区，使后续复制与之有序 | 调用 [HCCL/alg_data_trans_wrapper.cc:468][src-wrap-468] |
| 6 | `PostCopy` → `LocalCopy` → `HcommLocalCopyOnThread` | 返回 Mesh 模板后，把本端 CCL 中本 Peer 的片段复制到本端用户输出 | 模板 [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:579][src-mesh-579]；包装 [HCCL/alg_data_trans_wrapper.cc:891][src-wrap-891]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:129][src-prim-129] |

**`PostCopy` 不在 `SendRecvWrite` 函数体内。** 它在 `RunSendRecvByChannel` 的后续分支，满足普通 Write、不能直接访问远端用户内存且接收长度大于零才执行。

<a id="p12"></a>

## P12｜用户字节路径与远端写内部实现

普通中转 Write 的数据路径：`本端 input` → **HcommWriteOnThread** → `对端 CCL` → **对端 HcommLocalCopyOnThread** → `对端 output`。元数据序列化没有替代这两次数据移动。

| PPT 节点 / 源码步骤 | 作用 | 源码定位 |
|---|---|---|
| `BuildDataSlices` | 用 Peer、CCL 槽位、Channel 分片偏移算出真正参与搬运的地址和字节长度 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:443][src-mesh-443] |
| `HcommWriteOnThread` | 接收 Thread、Channel、远端 dst、本地 src、len，进入数据原语实现 | [HCOMM/aicpu_ts_primitives_c_adpt.cc:505][src-prim-505] |
| `UnwrapChannelHandle` | 处理 Channel 句柄包装，拿到当前实现需要的通道对象 | 调用 [HCOMM/aicpu_ts_primitives_c_adpt.cc:513][src-prim-513]；头文件实现 [HCOMM/aicpu_ts_channel_helper.h:74][src-unwrap-74] |
| `AddThread` | 把使用的 Thread 加入当前 launch context，关联任务提交模式 | [HCOMM/aicpu_ts_primitives_c_adpt.cc:50][src-prim-50] |
| `Thread::GetStreamLitePtr` | A5 分支从 Thread 取得 `StreamLite`，用于下发传输任务 | 调用 [HCOMM/aicpu_ts_primitives_c_adpt.cc:523][src-prim-523] |
| `BuildLocRmaBufferLite` | 将本地地址和长度构造成 RMA 访问描述；按传输类型使用具体实现 | 调用 [HCOMM/aicpu_ts_primitives_c_adpt.cc:527][src-prim-527]；UB 示例 [HCOMM/ub_transport_lite_impl.cc:307][src-ubtrans-307] |
| `BaseTransportLiteImpl::Write` 的虚调用 | 把本地 RMA 描述和远端地址/长度交具体传输实现 | 调用 [HCOMM/aicpu_ts_primitives_c_adpt.cc:537][src-prim-537]；UB 示例 [HCOMM/ub_transport_lite_impl.cc:664][src-ubtrans-664] |
| `HcommLocalCopyOnThread` → `Thread::LocalCopy` | 接收侧在同一依赖链上将 CCL 数据复制到用户输出；自 Rank 复制也用该原语 | [HCOMM/aicpu_ts_primitives_c_adpt.cc:129][src-prim-129]；AICPU_TS 实现 [HCOMM/aicpu_ts_thread.cc:326][src-aithread-326] |

UB 只是深入阅读的具体传输示例，不代表所有拓扑都选择它。检查数据正确性时，应同时检查 **Peer 对应关系、地址、偏移、字节数、通知槽和最终输出**，不能只看 Write 返回成功。

<a id="p13"></a>

## P13｜主从汇合、最终通知与用户流继续

| 步骤 / 函数 | 作用和时序边界 | 源码定位 |
|---|---|---|
| `PostSyncInterThreads` | 主 Thread 对每个从 Thread 排入 Wait，从 Thread 排入 Record；建立汇合依赖 | [HCCL/alg_data_trans_wrapper.cc:1061][src-wrap-1061] |
| `RunALLtoALL` 返回 → `OrchestrateLoop` | 当前块完成任务编排，若还有数据块继续；不是每个块都通知用户流最终完成 | [HCCL/ins_temp_all_to_all_v_mesh_1D.cc:227][src-mesh-227]；[HCCL/ins_v2_all_to_all_v_sole_executor.cc:202][src-exec-202] |
| `HcommThreadNotifyRecordOnThread`（最终通知） | 整个 executor 编排返回后，主 Thread 向已导出的用户 Thread 排入完成通知 | 调用 [HCCL/kernel_launch.cc:612][src-kernel-612]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:206][src-prim-206] |
| `HcommBatchModeEnd` | 结束批模式，恢复对应的任务提交模式；不是全设备完成屏障 | 调用 [HCCL/kernel_launch.cc:628][src-kernel-628]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:1134][src-prim-1134] |
| `HcommReleaseComm` | 正常路径归还设备通信域使用引用，非销毁通信域；错误返回要单独检查 | 调用 [HCCL/kernel_launch.cc:735][src-kernel-735]；实现 [HCOMM/aicpu_ts_primitives_c_adpt.cc:1200][src-prim-1200] |
| 用户流上的完成 Wait 被满足 | P8 已排入的等待得到设备通知，后续用户流任务才能越过该依赖 | 等待组织 [HCCL/op_common.cc:1015][src-op-1015]；能力包装 [HCCL/hcomm_primitives_dl.cc:176][src-dl-176] |

不要用 `Kernel` C++ 函数返回、`BatchModeEnd` 或 `ReleaseComm` 单独推断所有 Rank 的硬件通信已完成。应结合实际 stream 同步、任务依赖和结果校验。

<a id="p14"></a>

## P14｜建链交换：远端资源描述与算子参数是两条不同路径

### A. 通道资源描述：让对端“知道怎么访问”

| 步骤 / 函数 | 交换/处理什么 | 源码定位 |
|---|---|---|
| `HcclChannelAcquire` → `BatchCreateSockets` | 为本次 Peer 通道及控制信息交换准备连接 | [HCOMM/coll_comm_res_c_adpt.cc:759][src-res-759]；[HCOMM/my_rank.cc:567][src-myrank-567] |
| `HcommEndpointCreate` / `HcommMemReg` / `HcommCollectiveChannelCreate` | 创建端点、注册本地内存、构造协议通道；缓存命中时部分步骤复用 | [HCOMM/hcomm_endpoint_c_adpt.cc:307][src-bep-307]；[HCOMM/hcomm_mem_c_adpt.cc:35][src-bmem-35]；[HCOMM/hcomm_channel_c_adpt.cc:440][src-bchan-440] |
| `ChannelProcess::ChannelGetStatus` → `AicpuTsUbRtpChannel::GetStatus` | 以 UB_RTP 为例，状态查询会推进握手，而非只读取一个静态标志 | [HCOMM/channel_process.cc:284][src-cprocess-284]；[HCOMM/aicpu_ts_ub_rtp_channel.cc:182][src-ubchan-182] |
| `ProcessUbRtpState` / `ProcessUbRtpDataState` | 分阶段组织长度、描述数据收发与解析；`isRecvFirst_` 决定两端角色顺序 | [HCOMM/aicpu_ts_ub_rtp_channel.cc:108][src-ubchan-108]；[HCOMM/aicpu_ts_ub_rtp_channel.cc:154][src-ubchan-154] |
| `NotifyVecPack` / `BufferVecPack` / `ConnVecPack` | 序列化通知、内存缓冲和连接资源等描述，不是用户输入张量 | [HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:246][src-ubhelper-246]；[HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:261][src-ubhelper-261]；[HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:303][src-ubhelper-303] |
| `SendDataSize` / `RecvDataSize` | 先交换描述数据的长度，准备接收空间 | [HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:319][src-ubhelper-319]；[HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:343][src-ubhelper-343] |
| `SendExchangeData` / `RecvExchangeData` | 发送、接收已序列化的资源描述 | [HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:350][src-ubhelper-350]；[HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:356][src-ubhelper-356] |
| `RecvDataProcess` | 解析远端描述并构造本地可引用的远端资源视图 | [HCOMM/aicpu_ts_uboe_ub_rtp_channel_helper.cc:363][src-ubhelper-363] |
| `FinalizeChannelsByEngine` | 连接和交换后按 Engine 准备返回句柄 | [HCOMM/my_rank.cc:1196][src-myrank-1196] |
| `HcclChannelGetHcclBuffer` / `HcommChannelGetRemoteMems` → `BuildChannelInfo` | 查询已经建立的远端映射，交给 HCCL 算法使用 | [HCOMM/channel_c_adpt.cc:86][src-chan-86]；[HCOMM/hcomm_channel_c_adpt.cc:759][src-bchan-759]；[HCCL/op_common.cc:2049][src-op-2049] |
| `EndpointPair::CreateChannel` 复用 | 同一 Engine/复用槽位已有通道时复用；不能用 Acquire 次数推算新建通道数 | [HCOMM/endpoint_pair.cc:202][src-epair-202] |

### B. 算子一致性信息：检查双方“是否在做同一件事”

| 步骤 / 函数 | 做什么 | 源码定位 |
|---|---|---|
| `AddExchangeInfo` → `HcclCommAddExchangeInfo` | HCCL 填本次算子元信息，登记到 HCOMM 域；不是立即传输用户数据 | HCCL [HCCL/op_common.cc:1515][src-op-1515]；HCOMM [HCOMM/exchange_info_c_adpt.cc:19][src-exchange-19] |
| `BatchExchangeAndCheckConsistency` | 建链过程交换已登记信息，同时处理 HCOMM 自身配置一致性 | [HCOMM/my_rank.cc:582][src-myrank-582] |
| `CompareOpExchangeInfos` → `InconsistentCheckParams` → `HcclCommGetExchangeInfo` | HCCL 获取远端算子描述，再比较类型/参数等；上层比较不是都在 HCOMM 内完成 | [HCCL/inconsistent_check.cc:52][src-check-52]；[HCCL/inconsistent_check.cc:72][src-check-72]；[HCOMM/exchange_info_c_adpt.cc:37][src-exchange-37] |

`HcclCommGetExchangeInfo` 对非空已保存记录要求传入长度匹配，并有成功读取后消费记录的语义。测试应覆盖登记、交换、读取/消费及不一致场景；不要当作无状态、可无限重复的 getter。

<a id="p15"></a>

## P15｜共用接口：通信域、通道申请与资源交换

本页是 **AICPU_TS / CCU / AIV 资源路径族的共用项**。所谓“测试”是为调用条件设计覆盖，不是要求每次算子都看到这些接口各调用一次。

| API | 含义 / 应关注的条件 | HCCL 使用位置或上层入口 | HCOMM 实现 |
|---|---|---|---|
| `HcclGetCommName` | 取域标识；标识须与上下文、日志对应 | [HCCL/all_to_all_v.cc:26][src-entry-26] | [HCOMM/op_base.cc:1514][src-legacy-1514] |
| `HcclConfigGetInfo` | 取展开/算法配置；按版本与设置分支 | [HCCL/op_common.cc:3513][src-op-3513] | [HCOMM/op_base.cc:4152][src-legacy-4152] |
| `HcclCommGetStatus` | 检查域状态；能力支持时调用 | `Selector` 内 [HCCL/op_common.cc:164][src-op-164] | [HCOMM/coll_comm_c_adpt.cc:19][src-domain-19] |
| `HcclEngineCtxGet` | 查询 tag/Engine 的缓存；覆盖命中和不存在 | [HCCL/op_common.cc:1350][src-op-1350]；[HCCL/op_common.cc:1301][src-op-1301] | [HCOMM/engine_ctx_c_adpt.cc:86][src-ctx-86] |
| `HcclEngineCtxCreate` | 创建/管理域关联上下文；缺失时才申请 | [HCCL/op_common.cc:1738][src-op-1738]；[HCCL/op_common.cc:3045][src-op-3045] | [HCOMM/engine_ctx_c_adpt.cc:25][src-ctx-25] |
| `HcclGetHcclBuffer` | 取域 CCL 地址和容量；不是每次物理分配 | [HCCL/op_common.cc:1756][src-op-1756]；[HCCL/op_common.cc:2219][src-op-2219]；[HCCL/op_common.cc:3045][src-op-3045] | [HCOMM/comm_mem_c_adpt.cc:152][src-mem-152] |
| `HcclCommAddExchangeInfo` | 登记待交换的算子元信息；按一致性检查条件执行 | [HCCL/op_common.cc:1515][src-op-1515] | [HCOMM/exchange_info_c_adpt.cc:19][src-exchange-19] |
| `HcclChannelAcquire` | 按 Peer、Engine、端点、协议、内存申请/复用 Channel | [HCCL/op_common.cc:2102][src-op-2102]；[HCCL/op_common.cc:2266][src-op-2266]；[HCCL/op_common.cc:3045][src-op-3045] | [HCOMM/coll_comm_res_c_adpt.cc:759][src-res-759] |
| `HcommEndpointCreate` | 新端点创建；缓存命中可跳过 | HCOMM 上层 [HCOMM/endpoint_mgr.cc:71][src-epmgr-71] | [HCOMM/hcomm_endpoint_c_adpt.cc:307][src-bep-307] |
| `HcommEndpointStartListen` | 准备端点监听；按 Engine/协议实际行为验证 | HCOMM 上层 [HCOMM/my_rank.cc:678][src-myrank-678] | [HCOMM/hcomm_endpoint_c_adpt.cc:365][src-bep-365] |
| `HcommMemReg` | 内存注册到端点；检查类型、范围、标签及复用 | HCOMM 上层 [HCOMM/endpoint_mgr.cc:125][src-epmgr-125] | [HCOMM/hcomm_mem_c_adpt.cc:35][src-bmem-35] |
| `HcommCollectiveChannelCreate` | 创建集合通信底层通道；不与公开 ChannelCreate 混用统计 | HCOMM 上层 [HCOMM/endpoint_pair.cc:202][src-epair-202] | [HCOMM/hcomm_channel_c_adpt.cc:440][src-bchan-440] |
| `HcclCommGetExchangeInfo` | 获取远端上层描述并交 HCCL 比较；注意长度和消费语义 | [HCCL/inconsistent_check.cc:72][src-check-72] | [HCOMM/exchange_info_c_adpt.cc:37][src-exchange-37] |

<a id="p16"></a>

## P16｜共用接口：成员、拓扑与链路查询

下表给出**代表性使用点**；同一接口可能在不同拓扑分支多次出现。层、实例、Rank、Endpoint 和 Link 是不同对象，不能只验证返回指针非空。

| API | 查询对象 / 结果 | HCCL 使用位置 | HCOMM 实现 |
|---|---|---|---|
| `HcclGetRankSize` | 通信域 Rank 数 | [HCCL/all_to_all_v.cc:50][src-entry-50] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:349][src-graph-349] |
| `HcclGetRankId` | 本端在通信域内的 Rank | [HCCL/all_to_all_v.cc:54][src-entry-54] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:369][src-graph-369] |
| `HcclRankGraphGetLayers` | 网络层级列表 | [HCCL/topo_host.cc:960][src-topo-960] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:98][src-graph-98] |
| `HcclRankGraphGetRanksByLayer` | 某层包含的 Rank 集合 | [HCCL/topo_host.cc:991][src-topo-991] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:167][src-graph-167] |
| `HcclRankGraphGetInstSizeListByLayer` | 某层拓扑实例的规模列表 | [HCCL/topo_host.cc:976][src-topo-976] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:191][src-graph-191] |
| `HcclRankGraphGetTopoInstsByLayer` | 某层拓扑实例 ID 列表 | [HCCL/topo_host.cc:901][src-topo-901] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:215][src-graph-215] |
| `HcclRankGraphGetTopoType` | 指定层/实例的拓扑类型 | [HCCL/topo_host.cc:910][src-topo-910] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:235][src-graph-235] |
| `HcclRankGraphGetRanksByTopoInst` | 某个实例包含的 Rank | [HCCL/topo_host.cc:917][src-topo-917] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:252][src-graph-252] |
| `HcclRankGraphGetEndpointNum` | 某层/实例端点数量，供分配描述空间 | [HCCL/topo_host.cc:924][src-topo-924] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:271][src-graph-271] |
| `HcclRankGraphGetEndpointDesc` | 端点描述列表 | [HCCL/topo_host.cc:926][src-topo-926] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:288][src-graph-288] |
| `HcclRankGraphGetEndpointInfo` | 指定端点的具体属性 | [HCCL/topo_host.cc:1143][src-topo-1143]；[HCCL/op_common.cc:2049][src-op-2049] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:308][src-graph-308] |
| `HcclRankGraphGetLinks` | 两个 Rank 在指定层的可用链路 | [HCCL/topo_host.cc:1131][src-topo-1131] | [HCOMM/coll_comm_rank_graph_a_adpt.cc:63][src-graph-63] |

<a id="p17"></a>

## P17｜部分共用接口：Thread 同步与远端缓冲区

| API | 本 PPT 路径范围 | 作用 / 使用位置 | HCOMM 实现 |
|---|---|---|---|
| `HcclThreadAcquireWithStream` | AICPU_TS、CCU | 用户 stream 的 Thread 包装；[HCCL/op_common.cc:783][src-op-783]、[HCCL/op_common.cc:1860][src-op-1860] | [HCOMM/thread_c_adpt.cc:318][src-thread-318] |
| `HcclThreadAcquireWithConfig` | AICPU_TS、CCU | 支持逐 Thread 配置的申请；[HCCL/op_common.cc:1776][src-op-1776]、[HCCL/op_common.cc:1887][src-op-1887] | [HCOMM/thread_c_adpt.cc:151][src-thread-151] |
| `HcclThreadAcquire` | AICPU_TS、CCU | 统一通知数的兼容申请；[HCCL/op_common.cc:1823][src-op-1823]、[HCCL/op_common.cc:1887][src-op-1887] | [HCOMM/thread_c_adpt.cc:218][src-thread-218] |
| `HcommThreadNotifyRecordOnThread` | AICPU_TS、CCU | 本地主从/Host-Device Thread 依赖；[HCCL/alg_data_trans_wrapper.cc:1024][src-wrap-1024]、[HCCL/alg_data_trans_wrapper.cc:1061][src-wrap-1061] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:206][src-prim-206] |
| `HcommThreadNotifyWaitOnThread` | AICPU_TS、CCU | 显式超时的 Thread 通知等待；实际使用经 [HCCL/hcomm_primitives_dl.cc:176][src-dl-176] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:243][src-prim-243] |
| `HcommThreadNotifyWaitOnThreadWithDefaultTimeout` | AICPU_TS、CCU | 默认超时能力分支，与上项是替代关系；同上包装 | [HCOMM/aicpu_ts_primitives_c_adpt.cc:481][src-prim-481] |
| `HcclChannelGetHcclBuffer` | AICPU_TS、AIV | 取远端 CCL；[HCCL/op_common.cc:2049][src-op-2049]、[HCCL/op_common.cc:3045][src-op-3045] | [HCOMM/channel_c_adpt.cc:86][src-chan-86] |
| `HcommChannelGetRemoteMems` | AICPU_TS、AIV | 底层通道远端内存列表，供上层缓冲区/标签查询 | [HCOMM/hcomm_channel_c_adpt.cc:759][src-bchan-759] |

CCU 的上述 Thread 同步用于 Host 多流组织，例见 [HCCL/ins_v2_all_gather_parallel_executor.cc:660][src-agparallel-660] 附近的并行执行。它与 CCU Kernel 内部 `AscendC::ccu::Event* / Notify*` **不是同一套接口**，后者见 P21。

<a id="p18"></a>

## P18｜AICPU_TS 专用接口：设备上下文、保序与执行

“专用”仅指本 PPT 选定路径的分类，不是断言其他设备/Engine 永远不支持接口。

| API | 作用 / 触发条件 | 使用位置 | HCOMM 实现 |
|---|---|---|---|
| `HcclEngineCtxCopy` | 将 Host 序列化资源描述写入设备上下文 | [HCCL/op_common.cc:1738][src-op-1738] | [HCOMM/engine_ctx_c_adpt.cc:139][src-ctx-139] |
| `HcclThreadExportToCommEngine` | 用户 Thread 与设备主 Thread 的双向跨 Engine 句柄导出 | [HCCL/op_common.cc:783][src-op-783] | [HCOMM/thread_c_adpt.cc:498][src-thread-498] |
| `HcclDedicatedThreadAcquire` | 展开保序能力支持时获取专用 Thread | [HCCL/order_launch.cc:152][src-order-152] | [HCOMM/thread_c_adpt.cc:358][src-thread-358] |
| `HcommThreadAlloc` | 保序专用 Thread 的底层分配；不能据此说所有普通 Thread 都经过它 | HCOMM 调用 [HCOMM/thread_manager.cc:696][src-tmanager-696] | [HCOMM/hcomm_thread_c_adpt.cc:41][src-bthread-41] |
| `HcclThreadResGetInfo` | 从域管理的 unfoldThread 取实际 stream 等资源信息 | [HCCL/order_launch.cc:18][src-order-18]；[HCCL/op_common.cc:1147][src-op-1147] | [HCOMM/thread_c_adpt.cc:549][src-thread-549] |
| `HcommThreadResGetInfo` | 查询底层 Host order Thread 的资源 | [HCCL/order_launch.cc:42][src-order-42] | [HCOMM/hcomm_thread_c_adpt.cc:341][src-bthread-341] |
| `HcommAcquireComm` | 取得已有设备域的使用引用，执行前保护 | [HCCL/kernel_launch.cc:362][src-kernel-362] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:1138][src-prim-1138] |
| `HcommBatchModeStart` | 开始批量组织/提交任务 | [HCCL/kernel_launch.cc:480][src-kernel-480] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:1131][src-prim-1131] |
| `HcommBatchModeEnd` | 结束批模式；不等于所有设备任务完成 | [HCCL/kernel_launch.cc:628][src-kernel-628] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:1134][src-prim-1134] |
| `HcommReleaseComm` | 正常路径归还引用，非销毁域 | [HCCL/kernel_launch.cc:735][src-kernel-735] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:1200][src-prim-1200] |

<a id="p19"></a>

## P19｜AICPU_TS 专用接口：搬运、归约与通道通知

| API | 作用 / 用例条件 | HCCL 包装或使用位置 | HCOMM 实现 |
|---|---|---|---|
| `HcommLocalCopyOnThread` | 本地 src→dst，len 为字节；自 Rank、CCL 前/后拷贝 | [HCCL/alg_data_trans_wrapper.cc:891][src-wrap-891] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:129][src-prim-129] |
| `HcommWriteOnThread` | 本地 src→远端 dst，普通 Mesh Write 主线 | [HCCL/alg_data_trans_wrapper.cc:435][src-wrap-435] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:505][src-prim-505] |
| `HcommReadOnThread` | 远端 src→本地 dst；选择 Read 模板/模式才触发 | [HCCL/alg_data_trans_wrapper.cc:616][src-wrap-616]；[HCCL/alg_data_trans_wrapper.cc:661][src-wrap-661] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:791][src-prim-791] |
| `HcommBatchTransferOnThread` | 一次提交批量传输描述；模板使用批包装且能力支持时触发 | [HCCL/alg_data_trans_wrapper.cc:472][src-wrap-472]；[HCCL/alg_data_trans_wrapper.cc:645][src-wrap-645] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:922][src-prim-922] |
| `HcommLocalReduceOnThread` | 本地元素归约，count 为元素数，另传类型/算子；不是 AllToAll 必需项 | [HCCL/alg_data_trans_wrapper.cc:914][src-wrap-914] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:160][src-prim-160] |
| `HcommChannelNotifyRecordOnThread` | 向对端通知槽发送 ACK / 数据完成等同步信号 | [HCCL/alg_data_trans_wrapper.cc:435][src-wrap-435] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:1020][src-prim-1020] |
| `HcommChannelNotifyWaitOnThread` | 等本端通道通知槽；显式超时分支 | [HCCL/hcomm_primitives_dl.cc:184][src-dl-184] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:1068][src-prim-1068] |
| `HcommChannelNotifyWaitOnThreadWithDefaultTimeout` | 默认超时分支；与上一项二选一 | [HCCL/hcomm_primitives_dl.cc:184][src-dl-184] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:459][src-prim-459] |
| `HcommSetNotifyWaitTimeOut` | 设置设备编排期间 Notify Wait 超时；接口能力支持时生效 | [HCCL/kernel_launch.cc:278][src-kernel-278] → [HCCL/hcomm_primitives_dl.cc:152][src-dl-152] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:436][src-prim-436] |
| `HcommThreadResAcquireTimeOut` | 设置 Thread 资源等待超时；与 Notify 超时不是同一个设置 | [HCCL/kernel_launch.cc:278][src-kernel-278] → [HCCL/hcomm_primitives_dl.cc:164][src-dl-164] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:447][src-prim-447] |

HCCL `alg_data_trans_wrapper.cc` 中存在同名的局部 Wait 包装（[HCCL/alg_data_trans_wrapper.cc:26][src-wrap-26]）。查调用链必须继续读 `hcomm_primitives_dl.cc`，不能把局部包装当成 HCOMM 底层实现。

<a id="p20"></a>

## P20｜CCU 专用接口：指令资源申请与 Kernel 注册

主线是**统计指令资源需求 → 查询已绑定实例能否复用 → 必要时创建并绑定实例 → 注册 Kernel**。以下为动态资源能力支持的分支；接口不齐的兼容路径见 P23。

| API | 作用 | HCCL 使用位置 | HCOMM 实现 |
|---|---|---|---|
| `HcommCcuInsResDescCreate` | 创建某 die 的资源描述容器 | [HCCL/op_common.cc:2493][src-op-2493]；[HCCL/op_common.cc:2691][src-op-2691] | [HCOMM/ccu_res_c_adpt.cc:29][src-ccures-29] |
| `HcommCcuKernelQueryResReq` | 分析 Kernel 需要的 CCU 指令/变量等资源 | [HCCL/op_common.cc:2493][src-op-2493] | [HCOMM/ccu_res_c_adpt.cc:148][src-ccures-148] |
| `HcommCcuInsResDescQueryNum` | 读取某类资源的数量，用于汇总及容量比较 | [HCCL/op_common.cc:2493][src-op-2493]；[HCCL/op_common.cc:2533][src-op-2533] | [HCOMM/ccu_res_c_adpt.cc:66][src-ccures-66] |
| `HcommCcuInsResDescSetNum` | 把汇总需求或最低配额写入资源描述 | [HCCL/op_common.cc:2575][src-op-2575]；[HCCL/op_common.cc:2691][src-op-2691] | [HCOMM/ccu_res_c_adpt.cc:54][src-ccures-54] |
| `HcclCommQueryAssignedCcuIns` | 查询通信域已绑定的 CCU 实例 | [HCCL/op_common.cc:2948][src-op-2948] | [HCOMM/coll_comm_ccu_c_adpt.cc:109][src-ccudomain-109] |
| `HcommCcuInsQueryResDesc` | 查询已存在实例的容量，判断能否复用 | [HCCL/op_common.cc:2756][src-op-2756] | [HCOMM/ccu_res_c_adpt.cc:254][src-ccures-254] |
| `HcommCcuQueryRemainResDesc` | 查询剩余资源：用于探测 die 是否使能，以及申请资源不足后的诊断；不是每次创建前的统一容量检查 | [HCCL/op_common.cc:2659][src-op-2659]；[HCCL/op_common.cc:2846][src-op-2846] | [HCOMM/ccu_res_c_adpt.cc:90][src-ccures-90] |
| `HcommCcuInsCreate` | 按需求创建 CCU 指令实例 | [HCCL/op_common.cc:2889][src-op-2889] | [HCOMM/ccu_res_c_adpt.cc:209][src-ccures-209] |
| `HcclCommAssignCcuIns` | 将新实例绑定到通信域 | [HCCL/op_common.cc:2889][src-op-2889] | [HCOMM/coll_comm_ccu_c_adpt.cc:148][src-ccudomain-148] |
| `HcommCcuKernelRegisterStart` | 开始当前实例的 Kernel 注册批次 | [HCCL/op_common.cc:2603][src-op-2603] | [HCOMM/ccu_launch.cc:46][src-cculaunch-46] |
| `HcommCcuKernelRegister` | 注册具体 Kernel 及构建参数，返回 Kernel 句柄 | [HCCL/op_common.cc:2603][src-op-2603] | [HCOMM/ccu_launch.cc:76][src-cculaunch-76] |
| `HcommCcuKernelRegisterEnd` | 完成本轮注册，使 Kernel 可用于后续发射 | [HCCL/op_common.cc:2603][src-op-2603] | [HCOMM/ccu_launch.cc:119][src-cculaunch-119] |

本页测试重点：新建、复用、资源不足/回退、绑定失败的清理，以及版本/能力不满足的兼容分支；不能只测一次注册成功。申请配额的构造另见 `CreateFinalReqDescs`（[HCCL/op_common.cc:2813][src-op-2813]），它与查询硬件剩余资源不是同一步。

<a id="p21"></a>

## P21｜CCU 专用接口：Kernel 发射与 C++ 数据面

| API / C++ 操作 | 做什么 | HCCL 使用示例 | HCOMM 实现/包装 |
|---|---|---|---|
| `HcommCcuKernelLaunch` | 向指定 Thread 发射已注册 Kernel，携带本次任务参数 | AllGather 模板 [HCCL/ccu_temp_all_gather_mesh_1D.cc:170][src-ccuag-170] | C API 实现 [HCOMM/ccu_launch.cc:292][src-cculaunch-292] |
| `HcommCcuGetMemToken` | 获取本地地址范围的 CCU 访问 Token，用于算法运行参数 | [HCCL/ccu_alg_template_base.cc:247][src-ccubase-247] | C API 实现 [HCOMM/ccu_launch.cc:405][src-cculaunch-405] |
| `AscendC::ccu::LoadArg` | 把运行参数载入 CCU 变量 | AllToAll [HCCL/ccu_kernel_all_to_all_mesh1d.cc:61][src-ccua2a-61] | 头文件 inline 包装 [HCOMM/ccu_primitives.hpp:93][src-ccuheader-93] |
| `AscendC::ccu::WriteVariableWithNotify` | 向 Peer 写变量并带通知，用于地址/Token 等元信息协商 | [HCCL/ccu_kernel_all_to_all_mesh1d.cc:84][src-ccua2a-84] | [HCOMM/ccu_primitives.hpp:76][src-ccuheader-76] |
| `AscendC::ccu::NotifyRecord` | CCU 跨 Rank 通知 | [HCCL/ccu_kernel_all_to_all_mesh1d.cc:104][src-ccua2a-104] | [HCOMM/ccu_primitives.hpp:64][src-ccuheader-64] |
| `AscendC::ccu::NotifyWait` | 等待 CCU 跨 Rank 通知位 | [HCCL/ccu_kernel_all_to_all_mesh1d.cc:92][src-ccua2a-92] | [HCOMM/ccu_primitives.hpp:68][src-ccuheader-68] |
| `AscendC::ccu::EventRecord` | 记录本地事件/位掩码，推进本地依赖 | AllReduce [HCCL/ccu_kernel_all_reduce_mesh1d_mem2mem.cc:597][src-ccuar-597] | [HCOMM/ccu_primitives.hpp:57][src-ccuheader-57] |
| `AscendC::ccu::EventWait` | 等待本地数据操作或事件的完成位 | AllToAll [HCCL/ccu_kernel_all_to_all_mesh1d.cc:151][src-ccua2a-151] | [HCOMM/ccu_primitives.hpp:58][src-ccuheader-58] |
| `AscendC::ccu::Read` | CCU 远端读，所选 Kernel 需要时使用 | AllReduce [HCCL/ccu_kernel_all_reduce_mesh1d_mem2mem.cc:607][src-ccuar-607] | [HCOMM/ccu_primitives.hpp:166][src-ccuheader-166] |
| `AscendC::ccu::Write` | CCU 远端写 | AllToAll [HCCL/ccu_kernel_all_to_all_mesh1d.cc:146][src-ccua2a-146] | [HCOMM/ccu_primitives.hpp:188][src-ccuheader-188] |
| `AscendC::ccu::LocalCopy` | CCU 本地内存/缓冲间复制，存在多种重载 | [HCCL/ccu_kernel_all_to_all_mesh1d.cc:144][src-ccua2a-144] | [HCOMM/ccu_primitives.hpp:127][src-ccuheader-127] |
| `AscendC::ccu::LocalReduce` | CCU 本地数据归约，按类型及归约操作生成指令 | [HCCL/ccu_kernel_all_reduce_mesh1d_mem2mem.cc:91][src-ccuar-91] | [HCOMM/ccu_primitives.hpp:141][src-ccuheader-141] |

上表 C++ 操作经 `ccu_primitives.hpp` 转到 `Ccu*` 指令构建接口，发生在 Kernel 资源查询/注册阶段的构建逻辑中；**不是每搬一块数据都由 Host 调一次相应 C API**。测试应分开看构建/注册、任务发射和硬件执行结果。AllToAll 示例不使用归约，因此 Read/LocalReduce 示例取自 AllReduce，不能接到 P11 的 AICPU 链后面。

<a id="p22"></a>

## P22｜AIV 专用接口：远端地址准备与向量核执行

| 阶段 / API | 作用及所属组件 | 源码定位 |
|---|---|---|
| `HcclCommMemReg` | **HCOMM 控制面**：把 AIV tag/标志区注册到通信域，随后建链交换其描述 | 调用 [HCCL/op_common.cc:3045][src-op-3045]；实现 [HCOMM/comm_mem_c_adpt.cc:30][src-mem-30] |
| `HcclChannelGetRemoteMems` | **HCOMM 控制面**：取远端注册内存及标签，形成远端标志地址表 | 调用 [HCCL/op_common.cc:3045][src-op-3045]；实现 [HCOMM/channel_c_adpt.cc:137][src-chan-137] |
| `HcclCommRegCommStateCallback` | **HCOMM 域管理**：能力支持时注册 AIV 标志清理回调 | 调用 [HCCL/op_common.cc:3045][src-op-3045]；实现 [HCOMM/op_base.cc:5805][src-legacy-5805] |
| `ClearAivTagCb` | **HCCL 内部回调**：根据通信域状态阶段处理 AIV tag 区 | [HCCL/hccl_aiv_utils.cc:1053][src-aivutils-1053] |
| `HcclAivKernelEntranceLaunch` | **HCCL 内部**：当前版本只确定并校验 `numBlocksLimit`；函数名不能当成核已发射的证据 | [HCCL/op_common.cc:1230][src-op-1230] |
| `ExecuteAivCacheLogic` → executor `Orchestrate` | **HCCL 内部**：组织 AIV 编排及缓存记录逻辑，随后走具体模板/核发射 | [HCCL/op_common.cc:600][src-op-600] |
| `aclrtLaunchKernelWithHostArgs` | **ACL API**：真正提交 AIV Kernel，打包本次运行参数；这里仅能定位调用点 | [HCCL/hccl_aiv_utils.cc:1116][src-aivutils-1116] |
| `AivCommBase::CpGM2GM` | **HCCL 设备侧工具**：按块在 GM/UB 间搬运，含原子操作重载 | 普通 [HCCL/aiv_communication_base_v2.h:586][src-aivbase-586]；原子重载 [HCCL/aiv_communication_base_v2.h:614][src-aivbase-614] |
| `AscendC::DataCopy` / `DataCopyPad` | **Ascend C 能力**：在上述核内执行拷贝/非对齐处理；不是 HCOMM C 原语 | 调用 [HCCL/aiv_communication_base_v2.h:561][src-aivbase-561]；[HCCL/aiv_communication_base_v2.h:569][src-aivbase-569] |
| `SetSignalValue` / `WaitSignalValue` | **HCCL 设备侧同步工具**：写标志、轮询期望值，建立向量核通信依赖 | [HCCL/sync_interface.h:21][src-aivsync-21]；[HCCL/sync_interface.h:51][src-aivsync-51] |

P22 的 HCOMM 检查到“远端地址准备”为止并不足以证明 AIV 算子正确；还要检查 ACL 参数、核内复制/原子操作、标志同步及最终张量。

<a id="p23"></a>

## P23｜特殊与兼容接口：按条件补充，不列为公共必测

| API / 条件 | 为什么不是公共主线 | HCCL 使用位置 | HCOMM 实现 |
|---|---|---|---|
| `HcommReadReduceOnThread` | AICPU_TS 的读归约能力；需对应内存/算法分支，不能把普通 AllToAll Write 当覆盖 | [HCCL/alg_data_trans_wrapper.cc:725][src-wrap-725]；[HCCL/alg_data_trans_wrapper.cc:802][src-wrap-802] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:844][src-prim-844] |
| `HcommThreadJoin` | AllReduce 模板的 `needAicpuReduce_` 分支先等待 Thread 任务；随后 `LocalReduce` 对 INT64/UINT64/FP64 或 PROD 转入 `AicpuReduce`。Join 不在 LocalReduce/AicpuReduce 函数体内 | 模板调用 [HCCL/ins_temp_all_reduce_mesh_1D_one_shot.cc:243][src-arone-243]；类型分支 [HCCL/alg_data_trans_wrapper.cc:914][src-wrap-914]；CPU 归约 [HCCL/alg_data_trans_wrapper.cc:1275][src-wrap-1275] | [HCOMM/aicpu_ts_primitives_c_adpt.cc:1247][src-prim-1247] |
| `HcclChannelQuery` | CCU 资源准备时先查是否已有通道 | [HCCL/op_common.cc:2266][src-op-2266] | [HCOMM/coll_comm_res_c_adpt.cc:887][src-res-887] |
| `HcclChannelDestroy` | CCU 资源失败/回退清理本轮新增通道，不应销毁借用的已有通道 | [HCCL/op_common.cc:1528][src-op-1528] | [HCOMM/coll_comm_res_c_adpt.cc:940][src-res-940] |
| `HcclCommQueryCcuIns` | 动态 CCU 资源接口能力不足时查询预分配实例 | [HCCL/op_common.cc:3002][src-op-3002] | [HCOMM/coll_comm_ccu_c_adpt.cc:67][src-ccudomain-67] |
| `HcommCcuInsDestroy` | 新建 CCU 实例绑定失败等清理分支 | [HCCL/op_common.cc:2889][src-op-2889] | [HCOMM/ccu_res_c_adpt.cc:279][src-ccures-279] |
| `HcommCcuInsResDescDestroy` | 临时描述对象的生命周期结束/错误清理，不是销毁运行中的 Kernel | [HCCL/op_common.cc:2493][src-op-2493]；[HCCL/op_common.cc:2889][src-op-2889] | [HCOMM/ccu_res_c_adpt.cc:45][src-ccures-45] |
| `HcclCommSymWinGet`（补充探测） | 判断地址范围能否按对称内存窗口使用；存在调用不等于所有 Engine/算法均选择该路径 | [HCCL/op_common.cc:4004][src-op-4004]；[HCCL/all_to_all_v.cc:718][src-entry-718] | [HCOMM/op_base.cc:6285][src-legacy-6285] |

此页也用于排除误记：P6 内部状态推进不是公开 `HcommChannelGetStatus`；P8 的普通 AllToAll 发射不是导出 `HcclAicpuKernelLaunch`。

## 用这份对照表安排阅读和测试

建议第一遍只读 P2 → P4 → P5 → P6 → P7 → P8 → P9 → P11 → P13；再补 P3、P10、P12、P14 的算法/协议细节。P15–P23 用来反查某个接口的 Engine 范围和条件，不按表行机械拼成一条调用链。

| 想验证的问题 | 对应 PPT / 源码入口 | 应观察的结果 |
|---|---|---|
| 首次申请和后续复用是否不同 | P4 `TryReuseResource`；P6 `EndpointPair::CreateChannel` | 相同资源可复用；不以 Acquire 次数等同于物理创建次数 |
| 本地注册、远端交换是否匹配 | P6/P14 `HcommMemReg`、协议交换、远端内存查询 | Peer、标签、地址范围及通道一致；不能把资源描述当用户数据 |
| 输入未就绪会不会提前读取 | P8/P9 的 Thread Record/Wait | 设备数据任务受输入通知约束；同时检查实际队列依赖 |
| Peer 数据是否到正确位置 | P10/P11 `BuildDataSlices`、Write、PostCopy | 地址/偏移/长度/通知槽正确，最终按 Rank 校验输出 |
| 多块/多 Thread 是否漏完成 | P10/P13 分块与汇合 | 所有块、所有相关 Thread 的任务依赖完整，用户流后续任务不越过完成等待 |
| CCU/AIV 能否沿用 AICPU 的测试项 | P20–P22 | 控制面部分可复用；CCU 指令/发射、AIV 地址/核内同步必须分别验证 |
| 错误与兼容分支是否覆盖 | P23，及各能力检查/缓存判断 | 验证返回值、资源归属和回退/清理；不能只用一次正常 AllToAll 通过代表完整覆盖 |

接口共用可以减少重复的接口级测试，**不能取消不同算子的数据语义测试**：AllGather 的布局、AllReduce 的归约、AllToAll 的 Peer 切片仍需分别验证。当前仓库只补充源码注释和静态对照，不声称这些测试已经执行。

<!-- 源码链接固定到已核对的注释提交；请勿直接替换为 main。 -->
[src-entry-26]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L26
[src-op-3468]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L3468
[src-op-1301]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1301
[src-op-1386]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1386
[src-op-1756]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1756
[src-op-1738]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1738
[src-op-1015]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1015
[src-op-1147]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1147
[src-kernel-348]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L348
[src-kernel-278]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L278
[src-wrap-435]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L435
[src-mesh-579]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L579
[src-wrap-1061]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L1061
[src-kernel-612]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L612
[src-myrank-1236]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L1236
[src-check-52]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/common/inconsistent_check.cc#L52
[src-entry-481]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L481
[src-entry-50]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L50
[src-graph-349]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L349
[src-entry-52]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L52
[src-legacy-1514]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/legacy/ascend910/framework/op_base/src/op_base.cc#L1514
[src-entry-54]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L54
[src-graph-369]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L369
[src-entry-66]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L66
[src-entry-418]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L418
[src-entry-856]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L856
[src-entry-799]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L799
[src-entry-654]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L654
[src-entry-624]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L624
[src-op-3513]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L3513
[src-legacy-4152]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/legacy/ascend910/framework/op_base/src/op_base.cc#L4152
[src-entry-746]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L746
[src-domain-19]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_c_adpt.cc#L19
[src-ctx-86]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/engine_ctx_c_adpt.cc#L86
[src-topo-68]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L68
[src-topo-885]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L885
[src-ctx-25]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/engine_ctx_c_adpt.cc#L25
[src-auto-131]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/selector/alltoall_auto_selector.cc#L131
[src-exec-574]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/executor/ins_v2_all_to_all_v_sole_executor.cc#L574
[src-op-783]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L783
[src-op-838]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L838
[src-op-854]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L854
[src-thread-318]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L318
[src-op-856]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L856
[src-thread-498]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L498
[src-op-1350]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1350
[src-exec-68]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/executor/ins_v2_all_to_all_v_sole_executor.cc#L68
[src-mesh-64]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L64
[src-op-1547]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1547
[src-op-1698]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1698
[src-mem-152]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/comm_mem_c_adpt.cc#L152
[src-op-1860]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1860
[src-op-1823]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1823
[src-op-1776]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1776
[src-thread-151]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L151
[src-thread-218]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L218
[src-op-1936]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1936
[src-op-1956]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1956
[src-op-2016]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2016
[src-op-2102]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2102
[src-res-759]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc#L759
[src-op-2049]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2049
[src-myrank-567]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L567
[src-myrank-678]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L678
[src-epmgr-71]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc#L71
[src-bep-307]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/hcomm_endpoint_c_adpt.cc#L307
[src-myrank-732]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L732
[src-bep-365]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/hcomm_endpoint_c_adpt.cc#L365
[src-myrank-238]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L238
[src-myrank-227]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L227
[src-epmgr-125]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc#L125
[src-epmgr-155]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc#L155
[src-bmem-35]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/hcomm_mem_c_adpt.cc#L35
[src-epair-202]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/endpoint_pair.cc#L202
[src-bchan-440]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc#L440
[src-cprocess-89]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/channel_process.cc#L89
[src-myrank-1050]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L1050
[src-cprocess-284]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/channel_process.cc#L284
[src-myrank-582]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L582
[src-myrank-1196]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc#L1196
[src-graph-308]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L308
[src-chan-86]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc#L86
[src-bchan-759]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc#L759
[src-op-1715]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1715
[src-ctx-139]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/engine_ctx_c_adpt.cc#L139
[src-op-1994]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1994
[src-op-887]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L887
[src-op-1095]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1095
[src-prim-206]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L206
[src-order-152]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/order_launch.cc#L152
[src-op-1155]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1155
[src-op-1163]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1163
[src-op-1173]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1173
[src-op-1181]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1181
[src-op-1206]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1206
[src-order-229]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/order_launch.cc#L229
[src-dl-176]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L176
[src-prim-243]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L243
[src-kernel-362]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L362
[src-prim-1138]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L1138
[src-kernel-326]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L326
[src-kernel-229]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L229
[src-kernel-444]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L444
[src-kernel-922]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L922
[src-kernel-480]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L480
[src-prim-1131]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L1131
[src-prim-481]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L481
[src-kernel-311]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L311
[src-exec-89]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/executor/ins_v2_all_to_all_v_sole_executor.cc#L89
[src-exec-202]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/executor/ins_v2_all_to_all_v_sole_executor.cc#L202
[src-mesh-106]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L106
[src-mesh-172]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L172
[src-mesh-227]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L227
[src-mesh-135]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L135
[src-wrap-1024]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L1024
[src-mesh-115]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L115
[src-mesh-205]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L205
[src-wrap-891]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L891
[src-mesh-283]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L283
[src-mesh-342]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L342
[src-mesh-141]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L141
[src-mesh-443]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L443
[src-mesh-400]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc#L400
[src-wrap-444]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L444
[src-prim-1020]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L1020
[src-wrap-448]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L448
[src-dl-184]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L184
[src-wrap-462]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L462
[src-prim-505]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L505
[src-wrap-466]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L466
[src-wrap-468]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L468
[src-prim-129]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L129
[src-prim-513]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L513
[src-unwrap-74]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_channel_helper.h#L74
[src-prim-50]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L50
[src-prim-523]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L523
[src-prim-527]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L527
[src-ubtrans-307]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L307
[src-prim-537]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L537
[src-ubtrans-664]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L664
[src-aithread-326]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/comm_engine_res/threads/aicpu_ts_thread.cc#L326
[src-kernel-628]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L628
[src-prim-1134]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L1134
[src-kernel-735]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc#L735
[src-prim-1200]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L1200
[src-ubchan-182]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_ub_rtp_channel.cc#L182
[src-ubchan-108]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_ub_rtp_channel.cc#L108
[src-ubchan-154]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_ub_rtp_channel.cc#L154
[src-ubhelper-246]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L246
[src-ubhelper-261]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L261
[src-ubhelper-303]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L303
[src-ubhelper-319]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L319
[src-ubhelper-343]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L343
[src-ubhelper-350]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L350
[src-ubhelper-356]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L356
[src-ubhelper-363]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.cc#L363
[src-op-1515]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1515
[src-exchange-19]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc#L19
[src-check-72]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/common/inconsistent_check.cc#L72
[src-exchange-37]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc#L37
[src-op-3045]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L3045
[src-op-2219]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2219
[src-op-2266]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2266
[src-topo-960]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L960
[src-graph-98]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L98
[src-topo-991]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L991
[src-graph-167]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L167
[src-topo-976]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L976
[src-graph-191]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L191
[src-topo-901]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L901
[src-graph-215]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L215
[src-topo-910]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L910
[src-graph-235]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L235
[src-topo-917]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L917
[src-graph-252]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L252
[src-topo-924]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L924
[src-graph-271]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L271
[src-topo-926]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L926
[src-graph-288]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L288
[src-topo-1143]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L1143
[src-topo-1131]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/topo_info/topo_host.cc#L1131
[src-graph-63]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L63
[src-op-1887]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1887
[src-agparallel-660]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_parallel_executor.cc#L660
[src-thread-358]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L358
[src-tmanager-696]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads/thread_manager.cc#L696
[src-bthread-41]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/hcomm_thread_c_adpt.cc#L41
[src-order-18]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/order_launch.cc#L18
[src-thread-549]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L549
[src-order-42]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/order_launch.cc#L42
[src-bthread-341]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/hcomm_thread_c_adpt.cc#L341
[src-wrap-616]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L616
[src-wrap-661]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L661
[src-prim-791]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L791
[src-wrap-472]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L472
[src-wrap-645]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L645
[src-prim-922]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L922
[src-wrap-914]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L914
[src-prim-160]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L160
[src-prim-1068]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L1068
[src-prim-459]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L459
[src-dl-152]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L152
[src-prim-436]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L436
[src-dl-164]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc#L164
[src-prim-447]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L447
[src-wrap-26]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L26
[src-op-2493]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2493
[src-op-2691]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2691
[src-ccures-29]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L29
[src-ccures-148]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L148
[src-op-2533]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2533
[src-ccures-66]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L66
[src-op-2575]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2575
[src-ccures-54]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L54
[src-op-2948]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2948
[src-ccudomain-109]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_ccu_c_adpt.cc#L109
[src-op-2756]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2756
[src-ccures-254]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L254
[src-op-2813]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2813
[src-ccures-90]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L90
[src-op-2889]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2889
[src-ccures-209]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L209
[src-ccudomain-148]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_ccu_c_adpt.cc#L148
[src-op-2603]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2603
[src-cculaunch-46]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc#L46
[src-cculaunch-76]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc#L76
[src-cculaunch-119]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc#L119
[src-ccuag-170]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_gather/algorithm/template/ccu/ccu_temp_all_gather_mesh_1D.cc#L170
[src-cculaunch-292]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc#L292
[src-ccubase-247]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/ccu_alg_template_base.cc#L247
[src-cculaunch-405]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc#L405
[src-ccua2a-61]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L61
[src-ccuheader-93]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L93
[src-ccua2a-84]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L84
[src-ccuheader-76]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L76
[src-ccua2a-104]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L104
[src-ccuheader-64]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L64
[src-ccua2a-92]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L92
[src-ccuheader-68]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L68
[src-ccuar-597]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_reduce/algorithm/template/ccu/kernel/ccu_kernel_all_reduce_mesh1d_mem2mem.cc#L597
[src-ccuheader-57]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L57
[src-ccua2a-151]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L151
[src-ccuheader-58]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L58
[src-ccuar-607]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_reduce/algorithm/template/ccu/kernel/ccu_kernel_all_reduce_mesh1d_mem2mem.cc#L607
[src-ccuheader-166]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L166
[src-ccua2a-146]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L146
[src-ccuheader-188]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L188
[src-ccua2a-144]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L144
[src-ccuheader-127]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L127
[src-ccuar-91]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_reduce/algorithm/template/ccu/kernel/ccu_kernel_all_reduce_mesh1d_mem2mem.cc#L91
[src-ccuheader-141]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/include/ccu/ccu_primitives.hpp#L141
[src-mem-30]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/comm_mem_c_adpt.cc#L30
[src-chan-137]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc#L137
[src-legacy-5805]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/legacy/ascend910/framework/op_base/src/op_base.cc#L5805
[src-aivutils-1053]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/hccl_aiv_utils.cc#L1053
[src-op-1230]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1230
[src-op-600]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L600
[src-aivutils-1116]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/hccl_aiv_utils.cc#L1116
[src-aivbase-586]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/aiv_communication_base_v2.h#L586
[src-aivbase-614]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/aiv_communication_base_v2.h#L614
[src-aivbase-561]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/aiv_communication_base_v2.h#L561
[src-aivbase-569]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/aiv_communication_base_v2.h#L569
[src-aivsync-21]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/aiv_interface/sync_interface.h#L21
[src-aivsync-51]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/aiv/aiv_interface/sync_interface.h#L51
[src-wrap-725]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L725
[src-wrap-802]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L802
[src-prim-844]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L844
[src-wrap-1275]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L1275
[src-prim-1247]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc#L1247
[src-res-887]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc#L887
[src-op-1528]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L1528
[src-res-940]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc#L940
[src-op-3002]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L3002
[src-ccudomain-67]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_ccu_c_adpt.cc#L67
[src-ccures-279]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L279
[src-ccures-45]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/base_comm/primitives/api_c_adpt/ccu/ccu_res_c_adpt.cc#L45
[src-op-4004]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L4004
[src-entry-718]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_to_all_v/all_to_all_v.cc#L718
[src-legacy-6285]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hcomm/src/legacy/ascend910/framework/op_base/src/op_base.cc#L6285
[src-op-164]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L164
[src-op-159]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L159
[src-op-2659]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2659
[src-op-2846]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/op_common/op_common.cc#L2846
[src-arone-243]: https://github.com/zstar1003/hccl-hcomm-annotated/blob/ece79923c7b7ba17f07a7b17ae80390fe3b12450/hccl/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc#L243
