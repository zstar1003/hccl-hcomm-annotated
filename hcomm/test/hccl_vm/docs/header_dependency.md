# CheckerL2 三方头文件依赖清单

> 基于 `CheckerL2` 版本梳理，列出所有三方头文件的依赖关系。

## 说明

本文档列出项目中通过 `#include` 实际引用的所有三方头文件，标注其来源（CANN / HCCL / HCOMM）及在磁盘上的实际路径。

**路径约定**：

- CANN 根目录：`/home/teamserver/workspace/Ascend/cann-9.1.0/`（下文以 `CANN/` 缩写）
- HCOMM 根目录：`/home/teamserver/workspace/hcomm/`（下文以 `HCOMM/` 缩写）
- HCCL 根目录：`/home/teamserver/workspace/hccl/`（下文以 `HCCL/` 缩写）
- 项目根目录：`CheckerL2/`（下文以 `项目/` 缩写）

**解析原则**：CMake 中 `target_include_directories` 的声明顺序决定搜索优先级，先找到的优先使用。

---

## 一、来自 CANN 的头文件

### 1.1 ACL 运行时接口（`CANN/include/acl/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `acl/acl_base.h` | `CANN/include/acl/acl_base.h` | proxy 下所有 aclrt_*.cc、hccl_comm_stub.cc、hccp_stub.cc、hccp_ra_socket_stub.cc、ascend_hal_stub.cc、aclrt_exec_control.cc；include/runnerdb/db_sim_runner_common.h；test/proxy/ |
| `acl/acl_rt.h` | `CANN/include/acl/acl_rt.h` | proxy 下所有 aclrt_*.cc、hccl_op_stub.cc；test/proxy/ |

### 1.2 HCCL 公开接口（`CANN/include/hccl/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `hccl/hccl_types.h` | `CANN/include/hccl/hccl_types.h` | proxy/aclrt_kernel_stub.cc、hccl_op_stub.cc、hccl_proxy_common.h；plugin/runner/ccu_executor/ccu_fp16.h；common/sim_data_dump.cc；device_arm/proxy/device_sqe_parse_stub.h；test/proxy/ |
| `hccl/hccl.h` | `CANN/include/hccl/hccl.h` | **proxy/hccl_op_stub.cc**；test/proxy/hccl_comm_stub_test.cc |
| `hccl/hcom.h` | `CANN/include/hccl/hcom.h` | proxy/hccl_inner_stub.cc；test/proxy/hccl_inner_stub_test.cc |
| `hccl/base.h` | `CANN/include/hccl/base.h` | proxy/aclrt_context_stub.cc；plugin/runner/ccu_executor/ccu_fp16.h；plugin/checker/header/external/task_param.h；include/sim_ip_address.h |
| `hccl/dtype_common.h` | `CANN/include/hccl/dtype_common.h` | proxy/scatter_stub.cc；test/proxy/scatter_stub_test.cc |
| `hccl/hccl_common.h` | `CANN/x86_64-linux/asc/include/adv_api/hccl/hccl_common.h` | test/proxy/hccl_proxy_common_test.cc |

### 1.3 HCCL 内部接口（`CANN/pkg_inc/hccl/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"dtype_common.h"` | `CANN/pkg_inc/hccl/dtype_common.h` | proxy/aclrt_new_stub.cc、hccp_ccu_stub.cc、hccp_stub.cc、hccl_op_stub.cc、aclrt_device_stub.cc、hccl_proxy_common.cc；plugin/runner/runner_utils/storage_manager.h；plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.h；plugin/checker/src/framework/task_graph_generator/ccu_all_rank_param_recorder.h；plugin/checker/src/utils/storage_manager.h |

### 1.4 CANN Runtime 接口（`CANN/pkg_inc/runtime/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"runtime/base.h"` | `CANN/pkg_inc/runtime/runtime/base.h` | proxy/hccl_comm_stub.cc、aclrt_runtime_config.cc、aclrt_notify_stub.cc、hccp_stub.cc、aclrt_device_stub.cc、hccp_ra_socket_stub.cc、aclrt_stream_stub.cc、aclrt_stub.cc；test/proxy/ |
| `"runtime/event.h"` | `CANN/pkg_inc/runtime/runtime/event.h` | proxy/aclrt_runtime_config.cc、aclrt_notify_stub.cc |
| `"runtime/rt.h"` | `CANN/pkg_inc/runtime/runtime/rt.h` | device_arm/proxy/device_sqe_parse_stub.h |

### 1.5 CANN Profiling 接口（`CANN/pkg_inc/profiling/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"aprof_pub.h"` | `CANN/pkg_inc/profiling/aprof_pub.h` | proxy/aprofiling_stub.cc；device_arm/proxy/aprofiling_stub.cc；test/proxy/aprofiling_stub_test.cc |

### 1.6 CANN Trace 接口（`CANN/pkg_inc/trace/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"atrace_pub.h"` | `CANN/pkg_inc/trace/atrace_pub.h` | proxy/adapter_rts_stub.cc |
| `"atrace_types.h"` | `CANN/pkg_inc/trace/atrace_types.h` | proxy/aclrt_new_stub.cc、hccp_stub.cc |

### 1.7 CANN 驱动接口（`CANN/include/driver/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"ascend_hal.h"` | `CANN/include/driver/ascend_hal.h` | proxy/adapter_rts_stub.cc、aclrt_new_stub.cc、ascend_hal_stub.cc；device_arm/proxy/device_sqe_parse_stub.h、ascend_hal_stub.cc；test/device_arm/ |

### 1.8 CANN Base 接口（`CANN/pkg_inc/base/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"dlog_pub.h"` | `CANN/pkg_inc/base/dlog_pub.h` | plugin/checker/header/external/log.h（作为 fallback） |
| `"log_types.h"` | `CANN/pkg_inc/base/log_types.h` | plugin/checker/header/external/dlog_pub.h |

### 1.9 CANN CCU 微码接口（`CANN/pkg_inc/hcomm/ccu/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"ccu_microcode_v1.h"` | `CANN/pkg_inc/hcomm/ccu/ccu_microcode_v1.h` | proxy/hccp_ccu_stub.cc；plugin/runner/ccu_executor/ccu_resource_manager.cc、ccu_resource_common.h；plugin/checker/header/external/ccu_rep_base.h、ccu_instr_info.h；plugin/checker/header/internal/binary_data_type_pub.h；plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ 多个文件；plugin/checker/src/framework/task_graph_generator/ 多个文件；include/sim_binary_data_type_pub.h |
| `"ccu_common.h"` | `CANN/pkg_inc/hcomm/ccu/ccu_common.h` | proxy/hccp_ccu_stub.cc |

### 1.10 CANN RTS 设备接口（`CANN/pkg_inc/runtime/runtime/rts/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"rts_device.h"` | `CANN/pkg_inc/runtime/runtime/rts/rts_device.h` | proxy/hccp_stub.cc、adapter_rts_stub.cc、aclrt_new_stub.cc |

### 1.11 CANN Runtime 其他接口（`CANN/pkg_inc/runtime/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"mem.h"` | `CANN/pkg_inc/runtime/mem.h` | proxy/hccp_stub.cc、aclrt_new_stub.cc |
| `"rt_external_kernel.h"` | `CANN/pkg_inc/runtime/rt_external_kernel.h` | proxy/hccp_ccu_stub.cc |

---

## 二、来自 HCOMM 的头文件

### 2.1 HCCP 网络接口（`HCOMM/src/base_comm/resources/hccp/inc/network/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"hccp.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp.h` | proxy/adapter_rts_stub.cc |
| `"hccp_async.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_async.h` | proxy/adapter_rts_stub.cc |
| `"hccp_ping.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_ping.h` | proxy/adapter_rts_stub.cc |
| `"hccp_tlv.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_tlv.h` | proxy/adapter_rts_stub.cc、hccp_stub.cc、**hccp_ccu_stub.cc** |
| `"hccp_ctx.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_ctx.h` | proxy/hccp_ccu_stub.cc、hccp_stub.cc |
| `"hccp_common.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_common.h` | proxy/hccl_comm_stub.cc、aclrt_new_stub.cc、hccp_ccu_stub.cc、hccp_stub.cc、hccp_ra_socket_stub.cc |

### 2.2 HCCP CCU 外部依赖（`HCOMM/src/base_comm/resources/hccp/external_depends/ccu/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"ccu_u_comm.h"` | `HCOMM/src/base_comm/resources/hccp/external_depends/ccu/ccu_u_comm.h` | proxy/hccp_ccu_stub.cc |

---

## 三、来自 HCCL 的头文件（仅 ARM 交叉编译使用）

### 3.1 算子通用接口（`HCCL/src/ops/op_common/inc/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| `"alg_param.h"` | `HCCL/src/ops/op_common/inc/alg_param.h` | device_arm/hccl_kernel_executor.cc；test/device_arm/ |

### 3.2 HCCL 公共代码（`HCCL/src/common/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| （多个头文件通过 include 路径暴露） | `HCCL/src/common/*.h` | 仅 device_arm 构建模块通过 include 路径间接引用 |

### 3.3 动态符号加载（`HCCL/src/common/hcomm_dlsym/` 及 `ccu/`）

| #include | 实际文件路径 | 引用位置 |
|----------|------------|---------|
| （多个头文件通过 include 路径暴露） | `HCCL/src/common/hcomm_dlsym/*.h`、`HCCL/src/common/hcomm_dlsym/ccu/*.h`、`*.hpp` | 仅 device_arm 构建模块通过 include 路径间接引用 |

---

## 四、项目本地 External 头文件（Checker 专用，HCCL 接口定义副本）

路径：`项目/src/plugin/checker/header/external/`

当 CANN 包中不存在同名头文件时，使用这些本地副本作为 fallback。

| 文件名 | 说明 | 引用位置 |
|-------|------|---------|
| `hccl_types.h` | HcclResult 枚举（HCCL_SUCCESS 等 24 个错误码） | plugin/checker/ 下 50+ 文件通过 `#include "hccl_types.h"` 引用；include/sim_binary_data_type_pub.h、sim_loader.h、sim_ccu_jetty_ctx.h、sim_ccu_channel_ctx.h 等 |
| `base.h` | 基础类型定义 | plugin/checker/ 下 19 处引用（framework、semantics_check、checker 等）；header/external/log.h |
| `data_type.h` | 数据类型定义 | plugin/checker/src/framework/task_graph_generator_v3/ 下 3 处、task_graph_generator/ 下 4 处 |
| `ccu_instr_info.h` | CCU 指令信息 | plugin/checker/ 下 8 处引用（task_ccu.h、task_utils.cc、task_graph_generator 等） |
| `ccu_task_param.h` | CCU 任务参数 | plugin/checker/src/utils/task_ccu.h |
| `ccu_rep_base.h` | CCU Reporter 基础 | 文件存在，当前无直接 `#include` 引用 |
| `ccu_rep_type.h` | CCU Reporter 类型 | 文件存在，当前无直接 `#include` 引用 |
| `dlog_pub.h` | 日志公共接口（CANN 的 fallback） | plugin/checker/header/external/log.h |
| `enum_factory.h` | 枚举工厂 | plugin/checker/ 内部 5 处引用（sim_task.h、check_utils.h、data_slice.h、data_type.h、task_param.h） |
| `const_val.h` | 常量值定义 | plugin/checker/header/external/task_param.h |
| `log.h` | 日志接口 | plugin/checker/ 内部 |
| `log_types.h` | 日志类型（CANN 的 fallback） | plugin/checker/header/external/dlog_pub.h |
| `string_util.h` | 字符串工具 | plugin/checker/header/external/enum_factory.h；header/internal/task_def.h；src/utils/check_utils.h |
| `task_param.h` | 任务参数 | 文件存在，当前无直接 `#include` 引用（被其他头文件内部包含） |

---

## 五、AIV 依赖（AIV 算子仿真专用）

本章节说明工具中 AIV（AI Vector Core）算子仿真相关的依赖项。这些依赖源自 HCOMM 仓库和 AscendC 编程接口，工具为了解除对外部仓库的强依赖，部分关键定义已在项目内部做了备份。

### 5.1 AivOpArgs 数据结构（项目内备份）

**原始来源**：HCOMM 仓库的 `hccl_aiv_utils.h` 头文件

**备份位置**：`src/proxy/aclrt_kernel_stub.cc` 第 597-628 行

**说明**：`AivOpArgs` 是 AIV 集合通信算子的参数结构体，用于在 host 端与 device 端之间传递 AIV kernel 的执行参数。工具原本通过 `dlsym` 动态加载 HCOMM 库中的 `ops_hccl::ExecuteKernelLaunch()` 函数（符号名：`_ZN8ops_hccl19ExecuteKernelLaunchERKNS_9AivOpArgsE`），该函数接收此结构体作为参数。为了解除对 HCOMM 运行时库的依赖，工具在项目内部做了结构体的完整备份。

**结构体定义**：

```cpp
struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_MAX;   // 集合通信命令类型
    std::string comm = {};                              // 通信域标识
    HcclComm hcclComm = nullptr;                        // HCCL 通信器句柄
    uint32_t numBlocks = AIV_STUB_MAX_NUM_BLOCKS;       // AIV 核数
    void *stream = nullptr;                             // 执行流
    uint64_t beginTime = 0;                             // 开始时间戳
    OpCounterInfo counter = {};                         // 计数器信息
    void *buffersIn = nullptr;                          // 输入缓冲区地址
    uint64_t input = 0;                                 // 输入偏移
    uint64_t output = 0;                                // 输出偏移
    uint32_t rank = 0;                                  // 当前 rank
    uint32_t sendRecvRemoteRank = 0;                    // SendRecv 远端 rank
    uint32_t rankSize = 0;                              // rank 总数
    uint64_t xRankSize = 0, yRankSize = 0, zRankSize = 0;  // 拓扑维度
    uint64_t count = 0;                                 // 元素数量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;   // Reduce 操作类型
    uint32_t root = 0;                                  // Root rank
    uint32_t sliceId = 0;                               // Slice ID
    uint64_t inputSliceStride = 0;                      // 输入 slice 步长
    uint64_t outputSliceStride = 0;                     // 输出 slice 步长
    uint64_t repeatNum = 0;                             // 重复次数
    uint64_t inputRepeatStride = 0;                     // 输入重复步长
    uint64_t outputRepeatStride = 0;                    // 输出重复步长
    bool isOpBase = false;                              // 是否为基础算子
    ExtraArgs extraArgs = {};                           // 额外参数（AllToAllV 等）
    uint64_t topo_[AIV_STUB_TOPO_LEN] = {0};            // 拓扑信息
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;  // 参数类型
};
```

**依赖风险**：若 HCOMM 侧 `AivOpArgs` 结构体发生变化（如字段增删、顺序调整、类型变更），可能导致工具通过 `dlsym` 调用时参数错位，引发运行时错误。

### 5.2 aiv_communication_base_v2.h 头文件（项目内备份）

**原始来源**：HCOMM 仓库的同名头文件

**备份位置**：`src/proxy/aiv_kernel/hccl_op_stub/aiv_communication_base_v2.h`

**说明**：此头文件定义了 AIV 集合通信算子的基类 `AivCommBase`，以及相关的内核参数宏定义（`KERNEL_ARGS_DEF`、`KERNEL_ARGS_CALL` 等）。工具原本直接 include HCOMM 仓库中的该文件，为了解除依赖，在项目内部做了完整备份。

**文件包含的关键定义**：

| 定义 | 说明 |
|------|------|
| `AivSuperKernelArgs` | SuperKernel 参数结构体 |
| `ExtraArgs` | 额外参数结构体（sendCounts/recvCounts 等，用于 AllToAllV） |
| `KERNEL_ARGS_DEF` / `KERNEL_ARGS_CALL` | AIV kernel 参数宏定义（约 25 个参数） |
| `SUPERKERNEL_ARGS_DEF` / `SUPERKERNEL_ARGS_CALL` | SuperKernel 参数宏定义 |
| `AivCommBase` 类 | AIV 通信基类，封装了 Barrier、Record/WaitFlag、CpGM2GM 等核心方法 |
| `EXPORT_AIV_META_INFO` | AIV kernel 元信息导出宏 |
| `AIV_ATOMIC_DATA_TYPE_DEF` / `AIV_COPY_DATA_TYPE_DEF` | 支持的数据类型宏定义 |

**依赖风险**：若 HCOMM 侧 AIV 通信接口或 kernel 参数发生变化，需同步更新此备份文件。

### 5.3 AIV 算子 Op 头文件（HCOMM / HCCL 依赖）

工具中每种集合通信算子的 AIV kernel 实现文件（`src/proxy/aiv_kernel/hccl_op_stub/*/aiv_communication_v2.cc`）通过 `#include` 引用了对应算子的 op 头文件。这些头文件**不存在于项目内部**，分别来自 HCOMM 和 HCCL 仓库：

| #include | 来源 | 实际文件路径 | 引用位置 |
|----------|------|------------|---------|
| `"aiv_all_reduce_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_reduce/` | hccl_op_stub/all_reduce/aiv_communication_v2.cc |
| `"aiv_all_gather_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_gather/` | hccl_op_stub/all_gather/aiv_communication_v2.cc |
| `"aiv_reduce_scatter_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/reduce_scatter/` | hccl_op_stub/reduce_scatter/aiv_communication_v2.cc |
| `"aiv_broadcast_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/broadcast/` | hccl_op_stub/broadcast/aiv_communication_v2.cc |
| `"aiv_all_to_all_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/` | hccl_op_stub/all_to_all_v/aiv_communication_v2.cc |
| `"aiv_all_to_all_v_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/` | hccl_op_stub/all_to_all_v/aiv_communication_v2.cc |
| `"aiv_scatter_op.h"` | HCCL | `hccl/src/ops/scatter/algorithm/template/aiv/kernel/` | hccl_op_stub/scatter/aiv_communication_v2.cc |
| `"aiv_reduce_op.h"` | HCCL | `hccl/src/ops/reduce/algorithm/template/aiv/kernel/` | hccl_op_stub/reduce/aiv_communication_v2.cc |

**依赖风险**：若 HCOMM 或 HCCL 侧算子 op 头文件的接口签名、模板参数或成员函数发生变化，工具的 AIV kernel 编译将失败，或导致运行时行为不一致。

### 5.4 AscendC 接口依赖

工具依赖 CANN 包提供的 AscendC 编程接口来模拟 AIV kernel 的执行。项目中的 AscendC 接口全部通过 `src/proxy/aiv_kernel/ascendc_stub/` 目录下的本地 stub 文件实现，而非直接使用 CANN SDK 中的 AscendC 头文件。

**使用的 AscendC 接口**（通过 stub 实现）：

| Stub 头文件 | 封装的 AscendC 接口 | 引用位置 |
|------------|-------------------|---------|
| `ascendc_base_stub.h` | `GlobalTensor`、`LocalTensor`、`TPipe`、`TBuf`、`TQueBind`、`pipe_barrier`、`GetBlockIdx` 等基础类 | aiv_kernel/ 下几乎所有文件 |
| `ascendc_copy_stub.h` | `DataCopy`、`DataCopyPad` 等数据搬移接口 | ascendc_stub/kernel_operator.h |
| `ascendc_math_stub.h` | `Add`、`Sub`、`Mul`、`Reduce` 等数学运算接口 | ascendc_stub/kernel_operator.h |
| `ascendc_memory_stub.h` | `SetAtomicAdd`、`SetAtomicMax`、`SetAtomicMin` 等原子操作接口 | ascendc_stub/kernel_operator.h |
| `ascendc_sync_stub.h` | `send_flag`、`recv_flag`、`SyncAll` 等同步接口 | ascendc_stub/kernel_operator.h |
| `ascendc_utils_stub.h` | `PipeBarrier` 等工具函数 | ascendc_stub/kernel_operator.h |

**依赖风险**：

1. AscendC 接口 API 变化（如类成员函数签名变更、模板参数调整）可能导致 stub 实现与真实接口不匹配
2. AscendC 新增数据类型（如 `fp8_e4m3fn_t`、`fp8_e5m2_t`、`hifloat8_t`）需要同步更新 stub 支持
3. `AivCommBase` 类内部大量使用 AscendC 模板类（`GlobalTensor<T>`、`LocalTensor<T>`、`TQueBind` 等），这些类型的语义变化会影响 AIV 仿真的正确性

---

## 六、AICPU 依赖（AICPU 模式专用）

本章节说明工具在 AICPU 模式下模拟 host-device 交互的依赖情况。该模式通过 `src/device_arm/hccl_kernel_executor.cc` 中的 `ExecuteAicpuKernel` 函数实现，模拟 AICPU kernel 下发到 device 端的执行过程，涉及多个数据结构的序列化和地址转换。

### 6.1 AICPU 交互数据结构

工具在 `src/device_arm/aicpu_args_stub.h` 中对以下数据结构做了本地备份，以解除对 HCOMM 运行时库的强依赖：

| 数据结构 | 用途 | 对应 kernel 函数 |
|---------|------|-----------------|
| `CommAicpuParam` | 通信域初始化参数（hcomId、设备 ID、H2D/D2H 传输参数） | `RunAicpuCommInit` |
| `HDCommunicateParams` | H2D/D2H 控制传输参数（deviceAddr、readCacheAddr） | 被 `CommAicpuParam` 引用 |
| `ThreadMgrAicpuParam` | 线程管理参数（threadNum、序列化 threadParam 数组、deviceHandle） | `RunAicpuIndOpThreadInit`、`RunAicpuThreadSupplementNotify` |
| `AicpuTsThread` | AICPU 线程信息（streamType、notifyLoadType、devId） | 被 `ThreadMgrAicpuParam` 引用 |
| `InitTask` | 通道初始化任务（context 指针、isCustom 标志） | `RunAicpuIndOpChannelInitV2` |
| `HcclChannelUrmaRes` | 通道 URMA 资源描述（channelList、uniqueIdAddr、remoteRankList） | `RunAicpuIndOpChannelInitV2` |
| `UniqueIdV2Header` | UniqueId V2 头部（type、notifyNum、bufferNum、connNum） | 通道建链序列化数据解析 |
| `ConnUniqueBlock` | 连接唯一块（柔性数组，包含 `ConnUniqueIds`） | 通道建链序列化数据解析 |

### 6.2 OpParam 数据结构（外部依赖）

**来源**：HCCL 仓库的 `alg_param.h` 头文件（`HCCL/src/ops/op_common/inc/`），位于 `ops_hccl` 命名空间

**说明**：`OpParam` 是 AICPU 集合通信算子的通用参数结构体，用于 `HcclLaunchAicpuKernel` 函数。与上述本地备份的结构体不同，工具**直接使用**了 HCCL 源码中的 `OpParam` 定义，未做本地备份。

**依赖风险**：若 HCCL 侧 `OpParam` 结构体发生变化（字段增删、类型变更），可能导致工具解析参数错误或运行时崩溃。

### 6.3 AICPU Kernel 函数依赖

工具通过 `dlopen` + `dlsym` 动态加载以下 HCOMM 运行时库中的 kernel 函数，模拟 AICPU 侧的实际执行：

| dlsym 符号名 | 所在库 | 功能 |
|-------------|-------|------|
| `RunAicpuCommInit` | `libccl_kernel.so` | AICPU 通信域初始化 |
| `RunAicpuIndOpThreadInit` | `libccl_kernel.so` | AICPU 线程初始化 |
| `RunAicpuIndOpChannelInitV2` | `libccl_kernel.so` | AICPU 通道初始化 V2 |
| `RunAicpuDfxInitV2` | `libccl_kernel.so` | AICPU DFX 算子信息初始化 V2 |
| `RunAicpuThreadSupplementNotify` | `libccl_kernel.so` | AICPU 资源补充通知 |
| `HcclLaunchAicpuKernel` | `libscatter_aicpu_kernel.so` | AICPU 集合通信 kernel 启动 |

**依赖风险**：

1. 若 HCOMM 侧上述 kernel 函数的签名或行为发生变化，工具的 `dlsym` 调用可能失败或产生错误结果
2. 若 `libccl_kernel.so` 或 `libscatter_aicpu_kernel.so` 新增/移除 kernel 函数，工具需同步更新函数指针列表
3. `CommAicpuParam`、`ThreadMgrAicpuParam`、`HcclChannelUrmaRes` 等本地备份结构体若与 HCOMM 侧实际定义不一致，将导致地址转换错误和内存越界

### 6.4 Kernel SO 库名依赖

工具在 `InitKernelFuncHandle` 函数中硬编码了以下 SO 库名，用于加载 AICPU kernel 的执行入口：

| 硬编码 SO 名称 | 加载的 dlsym 符号 | 说明 |
|--------------|-----------------|------|
| `libccl_kernel.so` | `RunAicpuCommInit`、`RunAicpuIndOpThreadInit`、`RunAicpuIndOpChannelInitV2`、`RunAicpuDfxInitV2`、`RunAicpuThreadSupplementNotify` | HCOMM 通信框架核心库 |
| `libscatter_aicpu_kernel.so` | `HcclLaunchAicpuKernel` | AICPU 集合通信算子 kernel 库 |
| `libslog.so` | （日志库依赖） | CANN 安全日志库 |
| `libc_sec.so` | （安全库依赖） | CANN 安全函数库 |

**依赖风险**：

1. **SO 名称变更**：若 HCOMM 侧重命名或拆分上述 SO（如将 `libccl_kernel.so` 重命名），工具的 `dlopen` 将加载失败，导致所有 AICPU kernel 无法执行
2. **算子 kernel SO 名称歧义**：当前无论算子类型如何（AllReduce、AllGather、ReduceScatter 等），算子 kernel 的 SO 名称固定为 `libscatter_aicpu_kernel.so`。这一命名可能引起歧义——名称中的 "scatter" 暗示仅适用于 Scatter 算子，但实际上承载了所有集合通信算子的 kernel。若 HCOMM 侧后续按算子类型拆分 SO（如 `liballreduce_aicpu_kernel.so`），工具将无法适配
3. **用户自定义算子场景**：HCCL 业务层未对用户自定义算子的 SO 名称做严格规定，用户可能自行编译出私有 SO（如 `libcustom_alltoall_kernel.so`）。工具目前仅识别上述固定 SO 名称，对于用户自定义算子的私有 SO 将无法加载和仿真

---

## 七、依赖总览

| 来源 | 头文件数 | 使用模块 |
|------|---------|---------|
| **CANN** | 23 个不同头文件 | 几乎所有模块（proxy, checker, runner, common, store, topo, utils, device_arm） |
| **HCOMM** | 13 个不同头文件 | proxy 模块（hccp 网络接口 + AIV 算子 op） |
| **HCCL** | 3 个直接引用 + 间接引用 | device_arm 模块（ARM 交叉编译）+ proxy（AIV scatter/reduce op） |
| **项目本地 external** | 14 个头文件 | 仅 checker 模块（作为 CANN 缺失时的 fallback） |
| **AIV 备份** | 2 个关键定义 | 仅 proxy/aiv_kernel 模块（AIV 算子仿真） |
| **AIV 算子 Op** | 8 个头文件（6 个 HCOMM + 2 个 HCCL） | 仅 proxy/aiv_kernel 模块（AIV 算子仿真） |
| **AscendC 接口** | 6 个 stub 头文件 | 仅 proxy/aiv_kernel 模块（AIV 算子仿真） |
| **AICPU 备份** | 8 个本地结构体 + 6 个 kernel 函数 + 4 个 SO 库名 | 仅 device_arm 模块（AICPU 模式 host-device 交互） |
