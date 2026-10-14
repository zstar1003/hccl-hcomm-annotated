# 集合通信展开方式：四象限对比

[查看对比图](images/collective-expansion-four-quadrants.png)

本图比较 AIV、AICPU_TS、CCU MS、CCU Sched 的执行机制，沿用 AICPU 图的白底、蓝色边框、橙蓝功能模块与黑色箭头。四象限是分类排版，不是性能坐标轴。

**算子与展开方式并非一一对应。** 同一 AllReduce、AllGather 等算子可以由不同引擎实现；实际选择受产品型号、CANN 版本、数据规模、拓扑、数据类型、资源及模式配置共同限制。

| 模式 | 通信逻辑在哪里执行 | 数据搬运关系 | 主要取舍 |
|---|---|---|---|
| AIV | Vector Core 执行通信 Kernel | Vector Core 驱动搬运后端；MTE 是代表性路径，具体协议可不同 | 小数据低时延；占用 Vector 核，可能与计算任务竞争 |
| AICPU_TS | AICPU 展开算法，STARS 调度任务 | SQE 经 RTSQ 调度 SDMA 或通信任务；UB 另有 WQE 工作队列 | 适合高带宽通信，算子/路径覆盖较广；存在软件展开与任务提交开销 |
| CCU MS | CCU 执行预置指令流 | 通过片上 Memory Slice 中转，并可执行受支持的归约 | 复用片上数据、减少 HBM 重复访问；受 MS 和其他 CCU 资源限制 |
| CCU Sched | CCU 执行指令并调度 UB WQE 任务 | UB 引擎在两端 HBM 间搬运，不经 MS 中转 | 专用硬件调度；仍占用 CCU 资源，部分场景会回退 AICPU |

## 阅读边界

- 图中 HBM 指设备主存；MS 是 CCU 片上缓存；UB 引擎中的 UB 指总线/通信互联语义，不是 AIV 的 Unified Buffer。
- 箭头分别表示任务下发、硬件调度或数据访问，不能把数据载荷理解为经过 SQE/WQE 队列。
- CCU 的启动过程在图中简化为 Host 生成指令并交给 CCU 执行；实际还涉及指令部署、Kernel 任务提交和任务调度器启动。
- CCU Sched 象限只保留一个 UB 引擎，位于 CCU 边界之外；工作请求与 HBM 数据路径在引擎处相接。
- “不占 Vector 核”描述所画通信执行路径，不表示系统没有 Host 初始化工作，也不代表所有混合算法不使用其他引擎。
- “按需归约”仅针对包含归约的算子及受支持的数据类型/操作，AllGather、Broadcast 等并不因此变成归约操作。
- 图中“950PR 不支持 MS 模式”来自当前仓库锁定文档的产品约束；不能将 CCU_MS 的存在等同于所有 CANN 版本都开放该模式。旧版官方文档甚至注明该版本不支持配置，使用时应核对对应版本。
- 回退条件不是固定消息大小：CCU 资源不足，以及部分单机场景 AllReduce/ReduceScatter/Reduce 的数据规模条件都可能触发回退，阈值与模式、网络规模有关。图中不提供无实测依据的性能排名或统一阈值。

## 依据

核对基于当前仓库锁定源码及文档，日期 2026-10-14。

1. [HCOMM 通信引擎说明](../hcomm/docs/zh/comm_op_dev_guide/prog_models_concepts/comm_engine.md)：AIV、AICPU+TS、CCU 的任务执行流程、资源占用与适用倾向。
2. [HCOMM 架构说明](../hcomm/docs/zh/architecture/architecture-brief.md)：Thread/调度器/通信硬件的职责分工，AIV 搬运后端依协议变化。
3. [CCU 编程模型与概念](../hcomm/docs/zh/comm_op_dev_guide/prog_models_concepts/CCU_models_concepts.md)：MS 中转、片上归约、HBM 访问复用、指令流及同步资源。
4. [HcclCommConfig](../hcomm/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclCommConfig.md)：950 模式配置、CCU_MS 的产品限制、CCU_SCHED 的 UB WQE 调度及 HBM 直传。
5. [HCCL_OP_EXPANSION_MODE](../hccl/docs/zh/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md)：算子/模式边界与回退条件。该文档和通信域 API 文档的算子列表存在粒度差异，因此图中不将两份列表拼接成通用支持矩阵。
6. [官方 HCCL 集合通信库用户指南](https://cann-doc.obs.cn-north-4.myhuaweicloud.com/hccl/HCCL%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E5%BA%93%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97.pdf)：用于交叉核对 CCU MS 与调度模式的定义。
7. [CANN 9.0 模式说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclenvref_07_0009.html)：佐证 MS/调度模式机制，同时展示不同版本的可用性限制。

制图使用内置 imagegen。提示词核心约束为：16:9 四象限，同一工程图风格；各象限同时展示执行路径、资源占用、收益及边界；CCU MS 数据经过片上缓存，CCU Sched 的独立 UB 引擎直接访问两端 HBM；不添加未经证实的性能数值。
