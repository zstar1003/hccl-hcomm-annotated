# Aclgraph介绍

## 阅读路径

```text
§1  ACL Graph 是什么
        │
        ▼
§2  捕获的生命周期
        │
        ▼
§3  任务的更新
        │
        ▼
§4  已知失败模式
        │
        ▼
§5  进阶：多 stream 拓扑
        │
        ▼
附录 A  关键 API 速查表
附录 B  torch_npu 源码对接
```

> ### 简明捕获流程（先看这张图了解全貌）
>
>```mermaid
> sequenceDiagram
>     participant U as User
>     participant C as Runtime
>     participant G as Graph DAG
>     participant R as Device
>     U->>C: aclmdlRICaptureBegin
>     U->>G: 下发业务算子
>     G-->>C: 暂存 (不执行)
>     U->>C: aclmdlRICaptureEnd
>     C->>G: 校验捕获合法
>     Note over R: 重放期 (可多次)
>     U->>G: aclmdlRIExecuteAsync
>     G->>R: 一次性下发整段DAG
>     R-->>U: 同步完成
> ```
>
> **对照Eager模式**：Eager模式下"1. 捕获" = "1个算子"="1次dispatch开销"；ACL Graph把整段流程压缩为 **1次capture + N次1-syscall replay**。

## 1. ACL Graph是什么

### 1.1问题：Eager模式下的Host dispatch瓶颈

PyTorch框架默认采用Eager模式：单算子下发后立即执行。每个算子都要从Host侧Python API走到Host侧C++ 算子下发，再到Device侧算子kernel执行——在Device侧每次kernel执行之前都需要等待Host侧的下发逻辑完成。

因此当单个算子计算量过小或Host性能不佳时，很容易产生Device空闲时间：每个kernel执行完后都需要一段时间来等待下一个kernel下发完成。

### 1.2已有解法：CUDA Graph

为优化Host调度性能，CUDA提供了图模式方案，称为CUDA Graph——一种Device调度策略，**省略算子的Host调度过程**。PyTorch官方文档有详细描述。

CUDA Graph的核心思路是把"一次下发、一次执行"变成"一次下发、多次执行"——把多次重复的Host调度开销摊销到一次capture过程中。

### 1.3 ACL Graph：NPU侧的对应方案

ACL Graph是CANN Runtime提供的"**捕获—重放**"机制，对应CUDA Graph在NPU侧的等价物：在两段 `aclmdlRICaptureBegin` / `aclmdlRICaptureEnd` 调用之间，所有在指定stream上提交的任务不会被立即执行，而是被暂存为一个**模型运行实例**（`aclmdlRI`）；之后调用 `aclmdlRIExecuteAsync` 可以**多次**整体重放同一批任务。

本文档定位为ACL Graph本身的API视角——读者如需从PyTorch模型层直接启用，可走torch_npu / torchair等上层框架（本文不展开）。

### 1.4适用范围与限制

**适用**：

- 小算子密集推理（DDP/FSDP gradient bucketing、LLM decode阶段、推荐系统inference、动态batch推理）
- 通信密集训练（HCCL集合通信在stream上重复执行）
- 同一组任务在多轮推理中重复执行的场景

**不适用**：

- 任务拓扑每次都变化的场景
- CPU同步密集的业务
- 调试原型阶段（捕获不可见会增加调试难度）

## 2. 捕获的生命周期

### 2.1入门：4步API流程

ACL Graph的核心API（`aclmdlRICaptureBegin` / `aclmdlRICaptureEnd` / `aclmdlRIExecuteAsync` / `aclmdlRIDestroy`）、捕获模式（`ACL_MODEL_RI_CAPTURE_MODE_GLOBAL` / `RELAXED`）、扩展API（任务组 / 更新组 / 模式切换）的函数签名、参数说明与类型定义见cann/runtime仓头文件 [`include/external/acl/acl_rt.h`](https://gitcode.com/cann/runtime/blob/master/include/external/acl/acl_rt.h)。

> **完整示例**：cann/runtime仓 [`example/2_advanced_features/model_ri/0_simple_model/main.cpp`](https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/0_simple_model/main.cpp) 演示了完整的4步生命周期（CaptureBegin → 下发算子 → CaptureEnd → ExecuteAsync × N → Destroy），包括GLOBAL/RELAXED模式切换、异步memcpy入图、以及 `aclmdlRIDebugJsonPrint` 调试输出。

**下文仅讨论与HCCL集合通信协同相关的约束，ACL Graph基础用法不再重复。**

### 2.2准备：capture之前

两件事：

1. **派发HCCL watchdog排空**——`ProcessGroupHCCL` 后台watchdog线程会调 `aclrtEventQuery` 检查collective完成情况。capture期间若它调query，会把"事件查询"也写入捕获上下文。`NPUGraph::capture_begin` 函数（[NPUGraph.cpp L240-340](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUGraph.cpp#L240)）在调用 `AclmdlRICaptureBegin` 前完成mempool绑定与stream检查，确保capture上下文就绪。
2. **路由allocator到私有池**——capture期间分配的tensor生命周期绑到 `model_ri_` 上，销毁时连同mempool一起释放。

### 2.3捕获中约束（HCCL视角）

ACL Graph基础约束（同一stream捕获、跨流event/notify引入、捕获期同步API限制等）在cann/runtime示例 [`0_simple_model/main.cpp`](https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/0_simple_model/main.cpp) 中已有代码演示。以下仅补充HCCL协同场景下需特别注意的约束。

#### 跨流捕获必须"加入后回到主流"

要把其他stream上的任务也加入capture，必须用event/notify引入，并且**最终必须通过event/notify回到主流**。否则capture_end时校验报错。

![跨流捕获基本流程（主流 → Stream2/3 → 必须返回主流）](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/acldevg/figure/zh-cn_image_0000002531349376.png)

> 该流程图说明：在主流上调用 `Begin`，在Stream2/3上Record Event接入，在主流上Wait Event把任务拉回主流，最后在主流上调用 `End`。返回主流之后、结束捕获之前，**不能再在Stream2/3上下发任务**——否则会因"有未被关联的task"触发报错。

#### 同步原语的选择

捕获期需要在stream之间建立依赖时，根据语义选原语：

| 场景 | 选notify | 选event |
| ------ | ---------- | ---------- |
| 跨Device同步 | 推荐 | 也可以，但notify设计上更轻量 |
| 同一Device多stream同步 | 等价于event | 等价于notify |
| 需要一对多通知 | 不支持 | 支持 |
| 需要时间戳 | 不支持 | 支持 |
| Wait后状态 | **自动重置**（一次性） | **不自动重置**（可重用） |

> 设计含义：notify的"一次性自动重置"特性正好匹配HCCL内部"通知从stream → 等从stream → 通知依赖方"的有状态机；event的"多对多可重用"则适合业务/通信语义切换点。Event与Notify的创建、等待、查询等基础API详见cann/runtime仓 [Event 管理](https://gitcode.com/cann/runtime/blob/master/docs/zh/dev_guide/03-05_event_management.md)与 [Notify 管理](https://gitcode.com/cann/runtime/blob/master/docs/zh/dev_guide/03-06_notify_management.md)。

### 2.4 capture之后：replay与销毁

`capture_end` 内部完成 `aclmdlRIBuildModel`（即将DAG实例化为可执行图对象）。之后任意stream上调 `aclmdlRIExecuteAsync(model_ri, stream)` 即可整体重放——CPU端真正开销只剩一次syscall，与算子数量无关。

**销毁顺序约束**（与 `ProcessGroupHCCL` 配合时）：

1. `NPUGraph::reset()` 释放ACL mempool
2. `destroy_process_group()` 走HCCL watchdog join
3. `HcclCommDestroy` 销毁通信域

颠倒顺序会因HCCL端work元数据指向已释放buffer导致use-after-free。

## 3. 任务的更新

任务已被捕获、暂存到 `aclmdlRI` 后，若需要更新任务本身或参数信息，CANN 9.0.0提供两种方式。cann/runtime仓提供了完整的示例，以下仅给出选型建议。

### 3.1原地更新

适用场景：**少量任务需要更新**（如varlen attention每次forward的seq_len变化）。

> **完整示例**：cann/runtime仓 [`example/2_advanced_features/model_ri/1_model_update/main.cpp`](https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/1_model_update/main.cpp)。

![原地更新任务流程（重新捕获方式）](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/acldevg/figure/zh-cn_image_0000002562269291.png)

### 3.2重新捕获

适用场景：**大量任务需要更新**（如一个模型有多种不同Shape的input）。

> **完整示例**：cann/runtime仓 [`example/2_advanced_features/model_ri/2_model_switch/`](https://gitcode.com/cann/runtime/tree/master/example/2_advanced_features/model_ri/2_model_switch)（Stream绑定/跳转/切换）、[`3_cond_model/`](https://gitcode.com/cann/runtime/tree/master/example/2_advanced_features/model_ri/3_cond_model)（IF/WHILE/SWITCH条件操作）。

![原地更新任务流程（先更新再执行）](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/acldevg/figure/zh-cn_image_0000002562429279.png)

### 3.3选型原则

| 维度 | 原地更新 | 重新捕获 |
| ------ | --------- | ---------- |
| 任务数 | 少量 | 大量 |
| 接口复杂度 | 较复杂（需任务组/更新组） | 简单（重新调Begin/End） |
| 硬件资源 | 复用同一 `aclmdlRI` | 每种Shape维护一份 `aclmdlRI` |
| 主要风险 | 任务组边界不当时机错误 | `aclmdlRI` 数量超出硬件资源限制 |
| 典型场景 | varlen attention、动态batch | 多Shape模型训练 |

并发执行场景是原地更新的**特例**：当需要"先更新任务再执行capture模型"时，引入Update stream + Event Wait顺序保证。

![并发执行场景任务更新流程](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/acldevg/figure/zh-cn_image_0000002562429281.png)

## 4. 已知失败模式

按"用户感知到的影响"分类，每类给出触发条件与规避。

### 4.1启动期失败

| 失败模式 | 触发条件 | 规避 |
| --------- | --------- | ------ |
| `TASK_QUEUE_ENABLE=2` 报错 | 环境变量与ACL Graph互斥 | 设为 `0` 或 `1` |
| pin_memory_expandable_segments报错 | `NPUExpandableHostAllocator` 与图私有池冲突 | 改回 `NPUCachingAllocator` |
| 旧的CANN版本不识别 通信kernel下发stream | 通信kernel下发stream不进入 `model_ri_` | 升级CANN到8.5+ |

### 4.2捕获期失败

| 失败模式 | 触发条件 | 规避 |
| --------- | --------- | ------ |
| `aclrtEventQuery` 污染DAG | watchdog / 用户代码在capture期间调query | `NPUGraph::capture_begin` 入口检查 + mempool预绑定排空 |
| `aclrtMemcpy` 在GLOBAL模式被拒 | 业务有同步内存函数 | 调 `aclmdlRICaptureThreadExchangeMode` 降级到RELAXED |
| 跨流任务不返回主流 | 跨流捕获未在主流上 `End` | 严格按 §2.3设计捕获拓扑 |
| 默认Stream操作 | `with torch.npu.stream(s_default)` 嵌套在capture期间 | 显式切到非默认stream |

### 4.3释放期失败

| 失败模式 | 触发条件 | 规避 |
| --------- | --------- | ------ |
| `destroy_process_group` 挂起 | 销毁顺序颠倒（HCCL通信域先于 `aclmdlRI`） | 严格按 §2.4三步顺序 |
| UB overflow | graph一次性驻留所有intermediate tensor超出NPU UB大小 | 监控 `peak_mempool_bytes`；必要时graph partition + fallback（参见issue #102） |
| 显存超限 | 同 `aclmdlRI` 数量过多（重新捕获方式二） | 监控model数量；切到原地更新 |

## 5. 进阶：多stream拓扑

![多 stream 拓扑全景](./figures/multi_streams_in_aclgraph.png)

### 5.1 HCCL内部的多stream拆分

Runtime不感知通信语义——它只接收stream上的任务。HCCL作为ACL之上的"通信层"，按"控制面/数据面分离"和"并行流水"的硬件设计原则，把一次collective拆分到多条内部stream。

> ACLGraph的捕获原理与跨stream捕获用法详见cann/runtime仓 [ACL-Graph 开发指南](https://gitcode.com/cann/runtime/blob/master/docs/zh/dev_guide/04_ACL-Graph.md)与 [跨流捕获](https://gitcode.com/cann/runtime/blob/master/docs/zh/dev_guide/04-02_cross_stream_capture.md)。

| Stream | 角色 | 设备 | 设计目的 |
| ------ | ------ | ------ | --------- |
| **计算stream** `@pytorch` | 业务matmul / bias / relu | NPU device | 与通信流并行，隐藏通信latency |
| **通信主stream** `@pytorch` | `dist.all_gather(...)` 入口 | NPU device | 业务/通信语义切换点（user可见stream） |
| **通信kernel下发stream** `@HCCL_HOST` | HCCL AICPU kernel | **AICPU协处理器** | 把"通信描述符"从host卸载到协处理器，不阻塞host CPU |
| **通信执行stream** `@HCCL_DEVICE`（主） | 通信task（Ring/Tree/HalvingDoubling kernel） | NPU device | 真正搬数据的设备kernel |
| **通信从stream** `@HCCL_DEVICE` | 主从并行场景的另一段通信task | NPU device | 大集合通信的"主从"分段并行 |

### 5.2 `model_ri_` 实际只包含3层

通信执行stream / 通信从stream的"通信task实际搬数据"是**数据面**——若把数据面也固化到 `model_ri_`，会让图对象体积过大且失去"每次replay重新选算法"的灵活性。

ACL Graph在AICPU展开方式下采取的设计：

> `model_ri_` 实际只包含 计算stream、通信主stream、通信kernel下发stream（控制面）。通信执行stream / 通信从stream的通信task在AICPU kernel执行时被**实时展开**——replay时，Runtime调 `AclmdlRIExecuteAsync(model_ri, stream)` 触发通信kernel下发stream的AICPU kernel，AICPU再实时展开出 通信执行stream / 通信从stream的通信task。

带来的好处：

- 图对象体积大幅缩小（只capture控制流）
- 数据面算法可每次replay重新决定（Ring/Bruck切换、拓扑变化适应）
- 跨NPU部署时，replay阶段根据当前实际拓扑选最优算法

### 5.3 Runtime行为标注

Runtime与HCCL在捕获期有4个协同点：

| # | 行为 | 实现 |
| --- | ------ | ------ |
| 1 | 通信主stream加入 `model_ri_` | Runtime根据event关系自动纳入 |
| 2 | 通信kernel下发stream加入 `model_ri_` | HCCL主动拉入（通过 `currentStreamCaptureStatusMayInitCtx` 识别capture状态） |
| 3 | 通信执行stream / 通信从stream不加入 | AICPU端在replay时实时展开 |
| 4 | `capture_end` 移出所有stream | 一次性从 `model_ri_` 卸载，不会残留 |

> **ACL Graph源码实现**：cann/runtime仓 [`src/runtime/feature/aclgraph/`](https://gitcode.com/cann/runtime/tree/master/src/runtime/feature/aclgraph) 包含capture/stream_capture/model/event_capture等核心实现。

## 附录A关键API速查表

ACL Graph全部API的函数签名、参数说明与类型定义见cann/runtime仓头文件 [`include/external/acl/acl_rt.h`](https://gitcode.com/cann/runtime/blob/master/include/external/acl/acl_rt.h)。

完整示例见cann/runtime仓 [`example/2_advanced_features/model_ri/`](https://gitcode.com/cann/runtime/tree/master/example/2_advanced_features/model_ri) 目录（4个示例：基础捕获、任务更新、Stream切换、条件操作）。

## 附录B torch_npu源码对接

> 本附录面向内部开发同学，介绍torch_npu如何把ACL Graph包装成PyTorch高级API。以下源码均来自 [Ascend/pytorch](https://github.com/Ascend/pytorch) 公开仓。

### B.1 C++ 核心：`c10_npu::NPUGraph`

源码文件：[`torch_npu/csrc/core/npu/NPUGraph.cpp`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUGraph.cpp)。`capture_begin` 调用链与 `replay` 实现见源码L240-340（mempool绑定 → `AclmdlRICaptureBegin` → `AclmdlRICaptureGetInfo`）。

### B.2 Python端入口：`torch.npu.graph` / `torch.npu.NPUGraph`

源码文件：[`torch_npu/csrc/npu/Graph.cpp`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Graph.cpp)（PyBind11绑定）和 [`torch_npu/csrc/npu/Graph.h`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Graph.h)（声明）。

```python
g = torch.npu.NPUGraph()                              # 创建
with torch.npu.graph(g):                              # 进入捕获
    out = model(static_input)                          # 业务代码
g.replay()                                             # 多次执行
```

### B.3 OpHandler框架：算子级预处理

源码目录：[`torch_npu/npu/_npugraph_handlers/`](https://github.com/Ascend/pytorch/tree/master/torch_npu/npu/_npugraph_handlers)。

### B.4集合通信对接：`ProcessGroupHCCL`

源码文件：[`torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp)。关键协同点：

- 头部 `#include "torch_npu/csrc/core/npu/NPUGraph.h"`（[L43](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp#L43)，编译期绑定）
- `kWatchdogThreadSleepMillis = 1000`（[L441](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp#L441)，HCCL端watchdog轮询间隔）
