# CheckerL2 Third-Party Header File Dependency List

> This list is based on the `CheckerL2` version and covers all third-party header file dependencies.

## Description

This document lists all third-party header files that the project references through `#include`. Each entry marks its origin (CANN / HCCL / HCOMM) and its actual path on disk.

**Path Conventions**:

- CANN root directory: `/home/teamserver/workspace/Ascend/cann-9.1.0/` (abbreviated as `CANN/` below)
- HCOMM root directory: `/home/teamserver/workspace/hcomm/` (abbreviated as `HCOMM/` below)
- HCCL root directory: `/home/teamserver/workspace/hccl/` (abbreviated as `HCCL/` below)
- Project root directory: `CheckerL2/` (abbreviated as `Project/` below)

**Resolution Principle**: The declaration order of `target_include_directories` in CMake determines the search priority. The compiler uses the first match found.

---

## 1. Header Files from CANN

### 1.1 ACL Runtime Interfaces (`CANN/include/acl/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `acl/acl_base.h` | `CANN/include/acl/acl_base.h` | all aclrt_*.cc under proxy, hccl_comm_stub.cc, hccp_stub.cc, hccp_ra_socket_stub.cc, ascend_hal_stub.cc, aclrt_exec_control.cc; include/runnerdb/db_sim_runner_common.h; test/proxy/ |
| `acl/acl_rt.h` | `CANN/include/acl/acl_rt.h` | all aclrt_*.cc under proxy, hccl_op_stub.cc; test/proxy/ |

### 1.2 HCCL Public Interfaces (`CANN/include/hccl/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `hccl/hccl_types.h` | `CANN/include/hccl/hccl_types.h` | proxy/aclrt_kernel_stub.cc, hccl_op_stub.cc, hccl_proxy_common.h; plugin/runner/ccu_executor/ccu_fp16.h; common/sim_data_dump.cc; device_arm/proxy/device_sqe_parse_stub.h; test/proxy/ |
| `hccl/hccl.h` | `CANN/include/hccl/hccl.h` | **proxy/hccl_op_stub.cc**; test/proxy/hccl_comm_stub_test.cc |
| `hccl/hcom.h` | `CANN/include/hccl/hcom.h` | proxy/hccl_inner_stub.cc; test/proxy/hccl_inner_stub_test.cc |
| `hccl/base.h` | `CANN/include/hccl/base.h` | proxy/aclrt_context_stub.cc; plugin/runner/ccu_executor/ccu_fp16.h; plugin/checker/header/external/task_param.h; include/sim_ip_address.h |
| `hccl/dtype_common.h` | `CANN/include/hccl/dtype_common.h` | proxy/scatter_stub.cc; test/proxy/scatter_stub_test.cc |
| `hccl/hccl_common.h` | `CANN/x86_64-linux/asc/include/adv_api/hccl/hccl_common.h` | test/proxy/hccl_proxy_common_test.cc |

### 1.3 HCCL Internal Interfaces (`CANN/pkg_inc/hccl/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"dtype_common.h"` | `CANN/pkg_inc/hccl/dtype_common.h` | proxy/aclrt_new_stub.cc, hccp_ccu_stub.cc, hccp_stub.cc, hccl_op_stub.cc, aclrt_device_stub.cc, hccl_proxy_common.cc; plugin/runner/runner_utils/storage_manager.h; plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.h; plugin/checker/src/framework/task_graph_generator/ccu_all_rank_param_recorder.h; plugin/checker/src/utils/storage_manager.h |

### 1.4 CANN Runtime Interfaces (`CANN/pkg_inc/runtime/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"runtime/base.h"` | `CANN/pkg_inc/runtime/runtime/base.h` | proxy/hccl_comm_stub.cc, aclrt_runtime_config.cc, aclrt_notify_stub.cc, hccp_stub.cc, aclrt_device_stub.cc, hccp_ra_socket_stub.cc, aclrt_stream_stub.cc, aclrt_stub.cc; test/proxy/ |
| `"runtime/event.h"` | `CANN/pkg_inc/runtime/runtime/event.h` | proxy/aclrt_runtime_config.cc, aclrt_notify_stub.cc |
| `"runtime/rt.h"` | `CANN/pkg_inc/runtime/runtime/rt.h` | device_arm/proxy/device_sqe_parse_stub.h |

### 1.5 CANN Profiling Interfaces (`CANN/pkg_inc/profiling/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"aprof_pub.h"` | `CANN/pkg_inc/profiling/aprof_pub.h` | proxy/aprofiling_stub.cc; device_arm/proxy/aprofiling_stub.cc; test/proxy/aprofiling_stub_test.cc |

### 1.6 CANN Trace Interfaces (`CANN/pkg_inc/trace/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"atrace_pub.h"` | `CANN/pkg_inc/trace/atrace_pub.h` | proxy/adapter_rts_stub.cc |
| `"atrace_types.h"` | `CANN/pkg_inc/trace/atrace_types.h` | proxy/aclrt_new_stub.cc, hccp_stub.cc |

### 1.7 CANN Driver Interfaces (`CANN/include/driver/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"ascend_hal.h"` | `CANN/include/driver/ascend_hal.h` | proxy/adapter_rts_stub.cc, aclrt_new_stub.cc, ascend_hal_stub.cc; device_arm/proxy/device_sqe_parse_stub.h, ascend_hal_stub.cc; test/device_arm/ |

### 1.8 CANN Base Interfaces (`CANN/pkg_inc/base/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"dlog_pub.h"` | `CANN/pkg_inc/base/dlog_pub.h` | plugin/checker/header/external/log.h (as fallback) |
| `"log_types.h"` | `CANN/pkg_inc/base/log_types.h` | plugin/checker/header/external/dlog_pub.h |

### 1.9 CANN CCU Microcode Interfaces (`CANN/pkg_inc/hcomm/ccu/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"ccu_microcode_v1.h"` | `CANN/pkg_inc/hcomm/ccu/ccu_microcode_v1.h` | proxy/hccp_ccu_stub.cc; plugin/runner/ccu_executor/ccu_resource_manager.cc, ccu_resource_common.h; plugin/checker/header/external/ccu_rep_base.h, ccu_instr_info.h; plugin/checker/header/internal/binary_data_type_pub.h; multiple files under plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/; multiple files under plugin/checker/src/framework/task_graph_generator/; include/sim_binary_data_type_pub.h |
| `"ccu_common.h"` | `CANN/pkg_inc/hcomm/ccu/ccu_common.h` | proxy/hccp_ccu_stub.cc |

### 1.10 CANN RTS Device Interfaces (`CANN/pkg_inc/runtime/runtime/rts/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"rts_device.h"` | `CANN/pkg_inc/runtime/runtime/rts/rts_device.h` | proxy/hccp_stub.cc, adapter_rts_stub.cc, aclrt_new_stub.cc |

### 1.11 CANN Runtime Other Interfaces (`CANN/pkg_inc/runtime/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"mem.h"` | `CANN/pkg_inc/runtime/mem.h` | proxy/hccp_stub.cc, aclrt_new_stub.cc |
| `"rt_external_kernel.h"` | `CANN/pkg_inc/runtime/rt_external_kernel.h` | proxy/hccp_ccu_stub.cc |

---

## 2. Header Files from HCOMM

### 2.1 HCCP Network Interfaces (`HCOMM/src/base_comm/resources/hccp/inc/network/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"hccp.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp.h` | proxy/adapter_rts_stub.cc |
| `"hccp_async.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_async.h` | proxy/adapter_rts_stub.cc |
| `"hccp_ping.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_ping.h` | proxy/adapter_rts_stub.cc |
| `"hccp_tlv.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_tlv.h` | proxy/adapter_rts_stub.cc, hccp_stub.cc, **hccp_ccu_stub.cc** |
| `"hccp_ctx.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_ctx.h` | proxy/hccp_ccu_stub.cc, hccp_stub.cc |
| `"hccp_common.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_common.h` | proxy/hccl_comm_stub.cc, aclrt_new_stub.cc, hccp_ccu_stub.cc, hccp_stub.cc, hccp_ra_socket_stub.cc |

### 2.2 HCCP CCU External Dependencies (`HCOMM/src/base_comm/resources/hccp/external_depends/ccu/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"ccu_u_comm.h"` | `HCOMM/src/base_comm/resources/hccp/external_depends/ccu/ccu_u_comm.h` | proxy/hccp_ccu_stub.cc |

---

## 3. Header Files from HCCL (ARM Cross-Compilation Only)

### 3.1 Operator Common Interfaces (`HCCL/src/ops/op_common/inc/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| `"alg_param.h"` | `HCCL/src/ops/op_common/inc/alg_param.h` | device_arm/hccl_kernel_executor.cc; test/device_arm/ |

### 3.2 HCCL Common Code (`HCCL/src/common/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| (multiple headers exposed through include paths) | `HCCL/src/common/*.h` | the device_arm build module references these indirectly through include paths |

### 3.3 Dynamic Symbol Loading (`HCCL/src/common/hcomm_dlsym/` and `ccu/`)

| #include | Actual File Path | Reference Locations |
|----------|------------|---------|
| (multiple headers exposed through include paths) | `HCCL/src/common/hcomm_dlsym/*.h`, `HCCL/src/common/hcomm_dlsym/ccu/*.h`, `*.hpp` | the device_arm build module references these indirectly through include paths |

---

## 4. Project Local External Header Files (Checker-Specific, HCCL Interface Definition Copies)

Path: `Project/src/plugin/checker/header/external/`

The project uses these local copies as a fallback when the CANN package does not contain a header file with the same name.

| File Name | Description | Reference Locations |
|-------|------|---------|
| `hccl_types.h` | HcclResult enumeration (24 error codes including HCCL_SUCCESS) | 50+ files under plugin/checker/ reference this through `#include "hccl_types.h"`; include/sim_binary_data_type_pub.h, sim_loader.h, sim_ccu_jetty_ctx.h, sim_ccu_channel_ctx.h, and so on |
| `base.h` | Basic type definitions | 19 references under plugin/checker/ (framework, semantics_check, checker, and so on); header/external/log.h |
| `data_type.h` | Data type definitions | 3 references under plugin/checker/src/framework/task_graph_generator_v3/, 4 references under task_graph_generator/ |
| `ccu_instr_info.h` | CCU instruction information | 8 references under plugin/checker/ (task_ccu.h, task_utils.cc, task_graph_generator, and so on) |
| `ccu_task_param.h` | CCU task parameters | plugin/checker/src/utils/task_ccu.h |
| `ccu_rep_base.h` | CCU Reporter base | the file exists; no direct `#include` reference currently |
| `ccu_rep_type.h` | CCU Reporter types | the file exists; no direct `#include` reference currently |
| `dlog_pub.h` | Logging public interface (CANN fallback) | plugin/checker/header/external/log.h |
| `enum_factory.h` | Enumeration factory | 5 internal references under plugin/checker/ (sim_task.h, check_utils.h, data_slice.h, data_type.h, task_param.h) |
| `const_val.h` | Constant value definitions | plugin/checker/header/external/task_param.h |
| `log.h` | Logging interface | internal to plugin/checker/ |
| `log_types.h` | Logging types (CANN fallback) | plugin/checker/header/external/dlog_pub.h |
| `string_util.h` | String utilities | plugin/checker/header/external/enum_factory.h; header/internal/task_def.h; src/utils/check_utils.h |
| `task_param.h` | Task parameters | the file exists; no direct `#include` reference currently (other header files include it internally) |

---

## 5. AIV Dependencies (AIV Operator Simulation Only)

This section describes the dependencies for AIV (AI Vector Core) operator simulation in the tool. These dependencies originate from the HCOMM repository and the AscendC programming interfaces. The tool maintains local backups of key definitions to remove strong dependencies on external repositories.

### 5.1 AivOpArgs Data Structure (Local Backup in Project)

**Original Source**: the `hccl_aiv_utils.h` header file from the HCOMM repository

**Backup Location**: `src/proxy/aclrt_kernel_stub.cc` lines 597-628

**Description**: `AivOpArgs` is the parameter structure for AIV collective communication operators. The structure passes AIV kernel execution parameters between the host side and the device side. The tool originally loaded the `ops_hccl::ExecuteKernelLaunch()` function from the HCOMM library through `dlsym` (symbol name: `_ZN8ops_hccl19ExecuteKernelLaunchERKNS_9AivOpArgsE`). That function accepts this structure as a parameter. The tool maintains a complete backup of the structure to remove the dependency on the HCOMM runtime library.

**Structure Definition**:

```cpp
struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_MAX;   // collective communication command type
    std::string comm = {};                              // communication domain identifier
    HcclComm hcclComm = nullptr;                        // HCCL communicator handle
    uint32_t numBlocks = AIV_STUB_MAX_NUM_BLOCKS;       // AIV core count
    void *stream = nullptr;                             // execution stream
    uint64_t beginTime = 0;                             // start timestamp
    OpCounterInfo counter = {};                         // counter information
    void *buffersIn = nullptr;                          // input buffer address
    uint64_t input = 0;                                 // input offset
    uint64_t output = 0;                                // output offset
    uint32_t rank = 0;                                  // current rank
    uint32_t sendRecvRemoteRank = 0;                    // SendRecv remote rank
    uint32_t rankSize = 0;                              // total rank count
    uint64_t xRankSize = 0, yRankSize = 0, zRankSize = 0;  // topology dimensions
    uint64_t count = 0;                                 // element count
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // data type
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;   // Reduce operation type
    uint32_t root = 0;                                  // Root rank
    uint32_t sliceId = 0;                               // Slice ID
    uint64_t inputSliceStride = 0;                      // input slice stride
    uint64_t outputSliceStride = 0;                     // output slice stride
    uint64_t repeatNum = 0;                             // repeat count
    uint64_t inputRepeatStride = 0;                     // input repeat stride
    uint64_t outputRepeatStride = 0;                    // output repeat stride
    bool isOpBase = false;                              // whether this is a base operator
    ExtraArgs extraArgs = {};                           // extra parameters (AllToAllV, and so on)
    uint64_t topo_[AIV_STUB_TOPO_LEN] = {0};            // topology information
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;  // parameter type
};
```

**Dependency Risk**: Changes to the `AivOpArgs` structure on the HCOMM side (such as field additions, field removals, order changes, or type changes) may cause parameter misalignment during `dlsym` calls. This leads to runtime errors.

### 5.2 aiv_communication_base_v2.h Header File (Local Backup in Project)

**Original Source**: the header file with the same name from the HCOMM repository

**Backup Location**: `src/proxy/aiv_kernel/hccl_op_stub/aiv_communication_base_v2.h`

**Description**: This header file defines the base class `AivCommBase` for AIV collective communication operators and the related kernel parameter macros (`KERNEL_ARGS_DEF`, `KERNEL_ARGS_CALL`, and so on). The tool originally included this file directly from the HCOMM repository. The tool maintains a complete backup to remove the dependency.

**Key Definitions in the File**:

| Definition | Description |
|------|------|
| `AivSuperKernelArgs` | SuperKernel parameter structure |
| `ExtraArgs` | Extra parameter structure (sendCounts/recvCounts, and so on, for AllToAllV) |
| `KERNEL_ARGS_DEF` / `KERNEL_ARGS_CALL` | AIV kernel parameter macros (approximately 25 parameters) |
| `SUPERKERNEL_ARGS_DEF` / `SUPERKERNEL_ARGS_CALL` | SuperKernel parameter macros |
| `AivCommBase` class | AIV communication base class that encapsulates core methods such as Barrier, Record/WaitFlag, and CpGM2GM |
| `EXPORT_AIV_META_INFO` | AIV kernel metadata export macro |
| `AIV_ATOMIC_DATA_TYPE_DEF` / `AIV_COPY_DATA_TYPE_DEF` | Supported data type macro definitions |

**Dependency Risk**: Changes to the AIV communication interfaces or kernel parameters on the HCOMM side require a synchronized update to this backup file.

### 5.3 AIV Operator Op Header Files (HCOMM / HCCL Dependencies)

Each AIV kernel implementation file for a collective communication operator in the tool (`src/proxy/aiv_kernel/hccl_op_stub/*/aiv_communication_v2.cc`) includes the corresponding operator op header file through `#include`. These header files **do not exist within the project**. They come from the HCOMM and HCCL repositories respectively:

| #include | Source | Actual File Path | Reference Locations |
|----------|------|------------|---------|
| `"aiv_all_reduce_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_reduce/` | hccl_op_stub/all_reduce/aiv_communication_v2.cc |
| `"aiv_all_gather_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_gather/` | hccl_op_stub/all_gather/aiv_communication_v2.cc |
| `"aiv_reduce_scatter_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/reduce_scatter/` | hccl_op_stub/reduce_scatter/aiv_communication_v2.cc |
| `"aiv_broadcast_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/broadcast/` | hccl_op_stub/broadcast/aiv_communication_v2.cc |
| `"aiv_all_to_all_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/` | hccl_op_stub/all_to_all_v/aiv_communication_v2.cc |
| `"aiv_all_to_all_v_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/` | hccl_op_stub/all_to_all_v/aiv_communication_v2.cc |
| `"aiv_scatter_op.h"` | HCCL | `hccl/src/ops/scatter/algorithm/template/aiv/kernel/` | hccl_op_stub/scatter/aiv_communication_v2.cc |
| `"aiv_reduce_op.h"` | HCCL | `hccl/src/ops/reduce/algorithm/template/aiv/kernel/` | hccl_op_stub/reduce/aiv_communication_v2.cc |

**Dependency Risk**: Changes to the interface signatures, template parameters, or member functions in the operator op header files on the HCOMM or HCCL side will cause AIV kernel compilation failures or lead to inconsistent runtime behavior.

### 5.4 AscendC Interface Dependencies

The tool depends on the AscendC programming interfaces provided by the CANN package to simulate AIV kernel execution. All AscendC interfaces in the project use local stub files under the `src/proxy/aiv_kernel/ascendc_stub/` directory. The tool does not use the AscendC header files from the CANN SDK directly.

**AscendC Interfaces in Use** (implemented through stubs):

| Stub Header File | Encapsulated AscendC Interfaces | Reference Locations |
|------------|-------------------|---------|
| `ascendc_base_stub.h` | base classes such as `GlobalTensor`, `LocalTensor`, `TPipe`, `TBuf`, `TQueBind`, `pipe_barrier`, and `GetBlockIdx` | nearly all files under aiv_kernel/ |
| `ascendc_copy_stub.h` | data copy interfaces such as `DataCopy` and `DataCopyPad` | ascendc_stub/kernel_operator.h |
| `ascendc_math_stub.h` | math operation interfaces such as `Add`, `Sub`, `Mul`, and `Reduce` | ascendc_stub/kernel_operator.h |
| `ascendc_memory_stub.h` | atomic operation interfaces such as `SetAtomicAdd`, `SetAtomicMax`, and `SetAtomicMin` | ascendc_stub/kernel_operator.h |
| `ascendc_sync_stub.h` | synchronization interfaces such as `send_flag`, `recv_flag`, and `SyncAll` | ascendc_stub/kernel_operator.h |
| `ascendc_utils_stub.h` | utility functions such as `PipeBarrier` | ascendc_stub/kernel_operator.h |

**Dependency Risks**:

1. AscendC API changes (such as class member function signature changes or template parameter adjustments) may cause mismatches between stub implementations and the real interfaces
2. New AscendC data types (such as `fp8_e4m3fn_t`, `fp8_e5m2_t`, and `hifloat8_t`) require synchronized updates to stub support
3. The `AivCommBase` class uses AscendC template classes extensively (`GlobalTensor<T>`, `LocalTensor<T>`, `TQueBind`, and so on). Semantic changes to these types affect AIV simulation correctness

---

## 6. AICPU Dependencies (AICPU Mode Only)

This section describes the dependencies for simulating host-device interaction in AICPU mode. The `ExecuteAicpuKernel` function in `src/device_arm/hccl_kernel_executor.cc` implements this mode. The function simulates the process of dispatching an AICPU kernel to the device side. This involves serialization and address translation of multiple data structures.

### 6.1 AICPU Interaction Data Structures

The tool maintains local backups of the following data structures in `src/device_arm/aicpu_args_stub.h` to remove the strong dependency on the HCOMM runtime library:

| Data Structure | Purpose | Corresponding Kernel Function |
|---------|------|-----------------|
| `CommAicpuParam` | Communication domain initialization parameters (hcomId, device ID, H2D/D2H transfer parameters) | `RunAicpuCommInit` |
| `HDCommunicateParams` | H2D/D2H control transfer parameters (deviceAddr, readCacheAddr) | referenced by `CommAicpuParam` |
| `ThreadMgrAicpuParam` | Thread management parameters (threadNum, serialized threadParam array, deviceHandle) | `RunAicpuIndOpThreadInit`, `RunAicpuThreadSupplementNotify` |
| `AicpuTsThread` | AICPU thread information (streamType, notifyLoadType, devId) | referenced by `ThreadMgrAicpuParam` |
| `InitTask` | Channel initialization task (context pointer, isCustom flag) | `RunAicpuIndOpChannelInitV2` |
| `HcclChannelUrmaRes` | Channel URMA resource description (channelList, uniqueIdAddr, remoteRankList) | `RunAicpuIndOpChannelInitV2` |
| `UniqueIdV2Header` | UniqueId V2 header (type, notifyNum, bufferNum, connNum) | channel link establishment serialized data parsing |
| `ConnUniqueBlock` | Connection unique block (flexible array that contains `ConnUniqueIds`) | channel link establishment serialized data parsing |

### 6.2 OpParam Data Structure (External Dependency)

**Source**: the `alg_param.h` header file from the HCCL repository (`HCCL/src/ops/op_common/inc/`), in the `ops_hccl` namespace

**Description**: `OpParam` is the general parameter structure for AICPU collective communication operators. The `HcclLaunchAicpuKernel` function uses this structure. Unlike the locally backed-up structures above, the tool **directly uses** the `OpParam` definition from the HCCL source code without a local backup.

**Dependency Risk**: Changes to the `OpParam` structure on the HCCL side (field additions, field removals, or type changes) may cause the tool to parse parameters incorrectly or crash at runtime.

### 6.3 AICPU Kernel Function Dependencies

The tool dynamically loads the following kernel functions from the HCOMM runtime library through `dlopen` + `dlsym` to simulate the actual execution on the AICPU side:

| dlsym Symbol Name | Library | Function |
|-------------|-------|------|
| `RunAicpuCommInit` | `libccl_kernel.so` | AICPU communication domain initialization |
| `RunAicpuIndOpThreadInit` | `libccl_kernel.so` | AICPU thread initialization |
| `RunAicpuIndOpChannelInitV2` | `libccl_kernel.so` | AICPU channel initialization V2 |
| `RunAicpuDfxInitV2` | `libccl_kernel.so` | AICPU DFX operator information initialization V2 |
| `RunAicpuThreadSupplementNotify` | `libccl_kernel.so` | AICPU resource supplement notification |
| `HcclLaunchAicpuKernel` | `libscatter_aicpu_kernel.so` | AICPU collective communication kernel launch |

**Dependency Risks**:

1. Changes to the signatures or behavior of the above kernel functions on the HCOMM side may cause `dlsym` calls to fail or produce incorrect results
2. If `libccl_kernel.so` or `libscatter_aicpu_kernel.so` adds or removes kernel functions, the tool must update its function pointer list accordingly
3. Inconsistencies between the locally backed-up structures (`CommAicpuParam`, `ThreadMgrAicpuParam`, `HcclChannelUrmaRes`, and so on) and the actual definitions on the HCOMM side will cause address translation errors and memory out-of-bounds access

### 6.4 Kernel SO Library Name Dependencies

The `InitKernelFuncHandle` function hardcodes the following SO library names to load the execution entry points for AICPU kernels:

| Hardcoded SO Name | Loaded dlsym Symbols | Description |
|--------------|-----------------|------|
| `libccl_kernel.so` | `RunAicpuCommInit`, `RunAicpuIndOpThreadInit`, `RunAicpuIndOpChannelInitV2`, `RunAicpuDfxInitV2`, `RunAicpuThreadSupplementNotify` | HCOMM communication framework core library |
| `libscatter_aicpu_kernel.so` | `HcclLaunchAicpuKernel` | AICPU collective communication operator kernel library |
| `libslog.so` | (logging library dependency) | CANN secure logging library |
| `libc_sec.so` | (security library dependency) | CANN security function library |

**Dependency Risks**:

1. **SO Name Changes**: If the HCOMM side renames or splits the above SOs (for example, renaming `libccl_kernel.so`), the `dlopen` call in the tool will fail. This causes all AICPU kernels to become non-executable
2. **Operator Kernel SO Name Ambiguity**: The operator kernel SO name is fixed as `libscatter_aicpu_kernel.so` regardless of the operator type (AllReduce, AllGather, ReduceScatter, and so on). The "scatter" in the name suggests applicability only to the Scatter operator, but the library actually hosts kernels for all collective communication operators. If the HCOMM side later splits SOs by operator type (for example, `liballreduce_aicpu_kernel.so`), the tool will fail to adapt
3. **User-Defined Operator Scenarios**: The HCCL business layer does not strictly define SO names for user-defined operators. Users may compile private SOs (for example, `libcustom_alltoall_kernel.so`). The tool currently recognizes only the fixed SO names listed above. The tool cannot load or simulate private SOs for user-defined operators

---

## 7. Dependency Overview

| Source | Header File Count | Modules in Use |
|------|---------|---------|
| **CANN** | 23 distinct header files | nearly all modules (proxy, checker, runner, common, store, topo, utils, device_arm) |
| **HCOMM** | 13 distinct header files | proxy module (hccp network interfaces + AIV operator ops) |
| **HCCL** | 3 direct references + indirect references | device_arm module (ARM cross-compilation) + proxy (AIV scatter/reduce ops) |
| **Project Local External** | 14 header files | checker module only (as a fallback when CANN headers are missing) |
| **AIV Backup** | 2 key definitions | proxy/aiv_kernel module only (AIV operator simulation) |
| **AIV Operator Ops** | 8 header files (6 from HCOMM + 2 from HCCL) | proxy/aiv_kernel module only (AIV operator simulation) |
| **AscendC Interfaces** | 6 stub header files | proxy/aiv_kernel module only (AIV operator simulation) |
| **AICPU Backup** | 8 local structures + 6 kernel functions + 4 SO library names | device_arm module only (AICPU mode host-device interaction) |
