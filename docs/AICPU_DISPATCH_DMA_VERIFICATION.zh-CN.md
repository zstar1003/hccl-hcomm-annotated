# AICPU 任务下发与硬件搬运：图片核对说明

核对日期：2026-10-14。对应[更新后的 16:9 图片](images/aicpu-task-dispatch-and-dma.png)。

本轮保留原图的 Host、Device OS、PM/TSD/CP、AICPU/Taishan 核、Cache、RTSQ、STARS、SDMA、描述符小图、WQE 队列与通信硬件等元素，只调整画布比例、清理杂线、修正局部连线和文字。它是**机制示意图**：左侧核数、进程关系沿用原图，并非此次源码核对确认的所有型号通用硬件布局。

## 1. SDMA 是否已经很少使用？

**目前没有依据判断“SDMA 很少使用”。能够确认的是：SDMA 仍有明确使用路径，但其职责和占比取决于平台、引擎、算法及协议。**

- 官方 HCCL 背景文档区分 SDMA（HCCS/PCIe）与 RDMA（RoCE）传输；因此不能将 SDMA 从通用机制图直接删除。[官方背景知识](https://gitee.com/ascend/cann-hccl/blob/e048c5face419f24486bf26712a426ded81402e7/docs/hccl_customized_dev/%E8%83%8C%E6%99%AF%E7%9F%A5%E8%AF%86.md)
- CANN 9.1 的对称内存注册说明仍列出 Atlas A3 超节点内 HCCS 的 SDMA 通信；同一文档另外说明 Ascend 950 的 UB/URMA 场景。这说明必须区分平台，不能把 HCCS、RoCE、UB 看作同一种路径。此 API 的产品/算子限制不能外推为整个 HCCL 的能力限制。[HcclCommSymWinRegister](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/api_ref/comm_mgr_c/HcclCommSymWinRegister.md)
- 本仓 **Ascend 950 / AICPU_TS / 普通 MeshOneShot / FP32 SUM / UB Channel** 示例中，本地复制与受支持的本地归约使用 SDMA，而跨 Rank 写使用 UB WQE；两者可以参与同一个 AllReduce。见下方源码。
- SDMA 也不等于“只能搬运本地内存”：仓库 P2P 内存语义读路径可把远端映射地址交给 SdmaCopy。

判断“用得多不多”需要指定设备、CANN 版本和运行负载，再查看 Profiling 的任务类型、传输量及耗时。官方 MindStudio Insight 的通信分析提供链路类型与传输时间等字段；文档列出某条路径的存在，并不能证明其运行占比。[MindStudio Insight 带宽分析](https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0114.html)

## 2. SQE 与 WQE 的真实关系

图中两条线承载的是**任务/工作描述**，底部内存箭头才表示数据载荷。对已核对的普通 UB 路径，逻辑顺序为：

1. AICPU 侧 HCOMM 构造 WQE，写入 UB jetty 的传输 SQ，推进生产指针 PI。
2. 构造携带 jetty 标识及 PI 的 Doorbell SQE。SQE 先在本地任务缓存中形成，再由 LaunchTask 批量复制到设备 RTSQ、更新 tail。
3. STARS 调度 Doorbell 任务，通知通信引擎处理工作队列中的 WQE；引擎依据描述执行内存访问。
4. 同步依赖通过 Notify Wait 等机制控制后续任务，不能将“Doorbell 已提交”视为“传输已完成”。

另一条独立路径为：**SDMA SQE → RTSQ → STARS → SDMA**。SDMA SQE 已描述地址、长度及操作。SDMA 不是 WQE 到达通信硬件之前必须经过的一站。

原图两条线的交叉处现在使用**跨线弧**：横向 WQE 线跨过纵向 SDMA 线，交点不连接。完成依赖的虚线指向 STARS，表示调度中的同步约束；它不是中断线路图，也不表示每个 WQE 都产生 CQE 或都必须由 AICPU 轮询。

**UB 与 RoCE 不能混用同一 Doorbell 编码。** 此图保留“通信硬件 / 网卡”的抽象节点，具体队列实现在 UB 中是 jetty SQ，在 RoCE 中是 QP SQ。当前源码的 UB Doorbell 是 UBDMA 类型、Doorbell 模式；RoCE Doorbell 则使用 WRITE_VALUE SQE。图示不覆盖直接提交、任务缓存重放等全部优化路径。

## 3. 源码证据

下面均固定到核对时的注释仓库提交 `5ad7402546ea51efa7538ba30193274366bb65e5`，依据实际调用、结构体及类型字段，避免后续注释变化造成行号漂移。

| 核对点 | 实际行为 | 源码 |
|---|---|---|
| WQE 创建及入队 | 填地址、令牌和 SGE；LaunchOneWqe 将 WQE 复制至 UB SQ 环槽，推进 PI | [ub_conn_lite.cc:324–378](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L324-L378) |
| UB 提交先后 | Write 先调用 conn→Write，后生成 BuildUbDbSendTask | [ub_transport_lite_impl.cc:776–842](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L776-L842) |
| Doorbell SQE | 用 jetty/die/function 标识与 PI 构造 UBDMA Doorbell 模式 SQE | [sqe_build_a5.h:246–274](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L246-L274) |
| RTSQ 提交 | 复制本地 SQE 缓存到 RTSQ，再更新队尾 | [rtsq_a5.cc:288–356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L288-L356) |
| SDMA 拷贝/归约 | SdmaCopy 和 SdmaReduce 构造 SDMA SQE | [rtsq_a5.cc:558–628](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L558-L628) |
| 本地原语落点 | LocalCopy→SdmaCopy；受支持的 LocalReduce→SdmaReduce | [aicpu_ts_thread.cc:372–461](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/base_comm/resources/comm_engine_res/threads/aicpu_ts_thread.cc#L372-L461) |
| 远端映射内存 | P2P Read 使用远端地址作为源并调用 SdmaCopy | [p2p_transport_lite_impl.cc:222–255](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/p2p_transport_lite_impl.cc#L222-L255) |
| 同步等待 | WaitWithTimeout 向流的 RTSQ 生成 NotifyWait | [ub_transport_lite_impl.cc:526–534](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L526-L534) |
| RoCE 区别 | 先准备 WQE 再提交 Doorbell；Doorbell SQE 类型为 WRITE_VALUE | [roce_transport_lite_impl.cc:299–316](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/roce_transport_lite_impl.cc#L299-L316)；[sqe_build_a5.cc:175–194](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.cc#L175-L194) |
| MeshOneShot 实例 | 281 行复制本 Rank 输入；360 行批量写对端；452–476 行将其他 Rank 接收槽归约到输出 | [ins_temp_all_reduce_mesh_1D_one_shot.cc](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hccl/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc#L245-L476) |
| 归约边界 | INT64、UINT64、FP64 或 PROD 分支改走 AicpuReduce，不能把所有归约都归为 SDMA | [alg_data_trans_wrapper.cc:1271–1313](https://github.com/zstar1003/hccl-hcomm-annotated/blob/5ad7402546ea51efa7538ba30193274366bb65e5/hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc#L1271-L1313) |

源代码可以验证描述符构造、队列写入和驱动提交；图中硬件取任务/执行的箭头是机制抽象，并非对硬件内部逐周期实现的断言。官方术语表将 STARS 定义为系统任务与资源调度器，将 SDMA 解释为直接内存访问技术。[官方术语表](https://www.hiascend.com/document/detail/en/Glossary/gls/gls_0001.html)

## 4. 图片变更及范围

- 清除 CP 到 CPU 区域的多余细线，保留有意义的调度箭头；维持原图模块和配色。
- 画布调整为 16:9，去除左上标题和底部释义小字，保留图内模块与流程标签。
- Host 启动标为 Runtime；右上框标为 AICPU 侧 HCOMM 任务构造，区分启动与执行阶段。
- “搬运 SQE”改为“调度 SDMA 任务”，增加 SDMA 作用说明。
- WQE 路径标注“先填 WQE → Doorbell 通知 → 取 WQE”，并用跨线弧消除与 SDMA 的误连接。
- 完成依赖回到 STARS 一侧；队列、Doorbell 和 SDMA 的释义保留在本核对说明中。

图片通过内置 imagegen 进行局部编辑。本次以此前更新后的图片为蓝本，保留主要元素及左右布局，去除左上标题和底部释义小字，并调整为 16:9；此前清理杂线及修正 SQE/WQE 局部连线的内容继续保留。本轮没有改变通信代码，没有上板测量 SDMA 利用率，也没有将图中核/进程布局认定为所有型号的实现。
