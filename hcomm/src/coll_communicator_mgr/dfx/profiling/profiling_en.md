# Profiling Module Code Analysis

## Feature Description

The Profiling module is responsible for collecting and reporting performance data of HCCL collective communication tasks. It is a core component of the HCCL DFX (Diagnostics and Observability) system. The module is deployed on both the Host side and the Device side, providing unified profiling capabilities.

Core capabilities:

1. **Report communication tasks**: Report execution information of collective communication tasks (AllReduce, Broadcast, and so on) to the Profiling framework.
2. **Report operator information**: Report key metrics such as start and end times and operator types for communication operators (Host side only).
3. **Report MC2 communication information**: Report Stream, Rank, and other metadata of the MC2 communication domain (Host side only).
4. **Report Kernel**: Report AICPU or AIV Kernel execution timeline information. The Device side reports kernel start and end task events.
5. **Update Profiling status**: Update Profiling statistics based on task queue consumption progress.
6. **Manage Profiling switches**: Respond to subscription and unsubscription commands from the Profiling framework to control data collection start and stop.

### External Interfaces

| Header File | Interface | Side | Description |
|--------|------|------|----------|
| `hccl_diag.h` | `HcclDfxRegOpInfoByCommId` | Host + Device | Registers operator information with the communication domain, records `beginTime`, and stores it in `MirrorTaskManager`. |
| `hccl_diag.h` | `HcclProfilingReportOp` | Host | Reports operator execution events: first `ReportAllTasks`, then `ReportOp`. |
| `hccl_diag.h` | `HcclReportAicpuKernel` | Host | Reports AICPU Kernel execution events, records taskId and streamId, and adds task information. |
| `hccl_diag.h` | `HcclReportAivKernel` | Host | Reports AIV Kernel execution events, records taskId and streamId, and adds task information. |
| `hccl_diag.h` | `HcommGetProfilingSysCycleTime` | Host | Obtains the Profiling system cycle time. |
| `hcomm_diag.h` | `HcommProfilingReportDeviceOp` | Device | Reports Device operator execution events: first `ReportAllTasks`, then reports OP information through `ProfilingHandlerLite`. |
| `hcomm_diag.h` | `HcommProfilingReportKernelStartTask` | Device | Reports kernel start task events, reports HEAD type `FlagTaskInfo` through `ProfilingHandlerLite`. |
| `hcomm_diag.h` | `HcommProfilingReportKernelEndTask` | Device | Reports kernel end task events, reports TAIL type `FlagTaskInfo` through `ProfilingHandlerLite`. |

## Directory Description

```text
profiling/
├── CMakeLists.txt                          # Top-level build, includes aicpu and host subdirectories
├── host/
│   ├── CMakeLists.txt                      # Host-side build, compiles hcclCommProfiling.cc
│   ├── hcclCommProfiling.h                 # Host-side Profiling facade class definition
│   └── hcclCommProfiling.cc                # Host-side Profiling facade class implementation
└── aicpu/
    ├── CMakeLists.txt                      # AICPU-side build, compiles hcclCommProfilingLite.cc
    ├── hcclCommProfilingLite.h             # AICPU-side Profiling facade class definition
    ├── hcclCommProfilingLite.cc            # AICPU-side Profiling facade class implementation
    ├── aicpu_ts_urma_dfx_kernel.h          # [Deprecated] URMA DFX Kernel
    └── aicpu_ts_urma_dfx_kernel.cc         # [Deprecated] URMA DFX Kernel
```

### File Relationships

| File | Function | Dependencies |
|------|------|----------|
| `host/hcclCommProfiling.h` | Host-side Profiling facade class declaration | Depends on `MirrorTaskManager`, `ProfilingReporter` |
| `host/hcclCommProfiling.cc` | Host-side Profiling facade class implementation | Depends on `profiling_reporter.h`, `profiling_handler.h`, `dlprof_function.h` |
| `aicpu/hcclCommProfilingLite.h` | AICPU-side Profiling facade class declaration | Depends on `MirrorTaskManagerLite`, `ProfilingReporterLite` |
| `aicpu/hcclCommProfilingLite.cc` | AICPU-side Profiling facade class implementation | Depends on `profiling_reporter_lite.h`, `mirror_task_manager_lite.h` |

### Profiling File Interaction

```mermaid
graph TB
    subgraph External Callers
        HCCL[hccl]
    end

    subgraph Host Side
        HCD[HcclCommDfx]
        HCP[HcclCommProfiling]
        PR[ProfilingReporter]
        PH[ProfilingHandler]
        MTM[MirrorTaskManager]
    end

    subgraph Device Side
        AIP[AicpuIndopProcess]
        HCDL[HcclCommDfxLite]
        HCPL[HcclCommProfilingLite]
        PRL[ProfilingReporterLite]
        PHL[ProfilingHandlerLite]
        MTML[MirrorTaskManagerLite]
    end

    subgraph External Components
        Profapi[profapi component]
    end

    HCCL -->|hccl_diag.h| HCD
    HCCL -->|hcomm_diag.h| AIP
    HCD --> HCP
    HCP --> PR
    PR --> MTM
    PR --> PH
    PH -->|dlMsprofReportApi, etc.| Profapi
    AIP --> HCDL
    AIP --> PHL
    HCDL --> HCPL
    HCPL --> PRL
    PRL --> MTML
    PRL --> PHL
    PHL -->|MsprofReportAdditionalInfo| Profapi
    HCP -->|dlMsprofSysCycleTime/dlMsprofStr2Id| Profapi
```

## Flow Description

### Host Profiling Flow

#### Register Operator Information (HcclDfxRegOpInfoByCommId)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant HCD as HcclCommDfx
    participant PH as ProfilingHandler
    participant MTM as MirrorTaskManager
    participant Profapi as profapi component

    HCCL->>HCD: HcclDfxRegOpInfoByCommId
    Note right of HCD: Convert HcclDfxOpInfo to DfxOpInfo
    HCD->>Profapi: hrtMsprofSysCycleTime
    Note right of Profapi: Get beginTime
    HCD->>HCD: UpdateProfStat
    Note right of HCD: Update switch status
    HCD->>MTM: SetCurrDfxOpInfo
    Note right of MTM: Store current operator info for later reporting
    HCD->>PH: SetOpModeFlags
    Note right of PH: Set operator mode flag
```

#### Report Operator (HcclProfilingReportOp)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant HCD as HcclCommDfx
    participant HCP as HcclCommProfiling
    participant PR as ProfilingReporter
    participant PH as ProfilingHandler
    participant Profapi as profapi component

    HCCL->>HCD: HcclProfilingReportOp
    alt currDfxOpInfo is empty
        Note right of HCD: Skip reporting, return success
    else currDfxOpInfo is not empty
        HCD->>HCD: GetOpModeFlags
        HCD->>HCP: ReportAllTasks
        HCP->>PR: ReportAllTasks
        Note right of PR: Traverse the task queue in MirrorTaskManager
        PR->>PH: ReportHcclTaskDetails
        PH->>Profapi: dlMsprofReportAdditionalInfo
        Note right of Profapi: Report Task details
        HCD->>HCP: ReportOp
        HCP->>PR: ReportOp
        PR->>PH: ReportHostApi
        PH->>Profapi: dlMsprofReportApi
        Note right of Profapi: level=ACL_LEVEL, report Host API timestamps
    end
```

#### Report AICPU Kernel (HcclReportAicpuKernel)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant HCD as HcclCommDfx
    participant HCP as HcclCommProfiling
    participant PH as ProfilingHandler
    participant DPF as DlProfFunction
    participant Profapi as profapi component

    HCCL->>HCD: HcclReportAicpuKernel
    alt currDfxOpInfo is empty
        Note right of HCD: Skip reporting, return success
    else currDfxOpInfo is not empty
        HCD->>HCP: ReportKernel
        HCP->>DPF: dlMsprofSysCycleTime
        Note right of DPF: Get endTime
        HCP->>DPF: dlMsprofStr2Id
        Note right of DPF: Hash kernelName to cmdItemId
        HCP->>PH: ReportNodeApi
        PH->>Profapi: dlMsprofReportApi
        Note right of Profapi: level=NODE_LEVEL, type=NODE_LAUNCH_TYPE
        HCP->>PH: ReportNodeBasicInfo
        PH->>Profapi: dlMsprofReportCompactInfo
        Note right of Profapi: level=NODE_LEVEL, type=NODE_BASIC_INFO_TYPE
        DPF->>Profapi: dlMsprofSysCycleTime
        Note right of Profapi: Get endTime for taskParam
        HCD->>HCD: AddTaskInfoCallback
        Note right of HCD: Record AICPU Kernel task info
    end
```

#### Report AIV Kernel (HcclReportAivKernel)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant HCD as HcclCommDfx
    participant DPF as DlProfFunction
    participant Profapi as profapi component

    HCCL->>HCD: HcclReportAivKernel
    DPF->>Profapi: dlMsprofSysCycleTime
    Note right of Profapi: Get endTime
    HCD->>HCD: AddTaskInfoCallback
    Note right of HCD: taskType=TASK_AIV, isMaster=true
```

#### Report MC2 Communication Information

```mermaid
sequenceDiagram
    participant HCD as HcclCommDfx
    participant HCP as HcclCommProfiling
    participant PR as ProfilingReporter
    participant PH as ProfilingHandler
    participant Profapi as profapi component

    HCD->>HCP: ReportMc2CommInfo
    HCP->>PR: CallReportMc2CommInfo
    PR->>PH: ReportHcclMC2CommInfo
    Note right of PH: Assemble ProfilingDeviceCommResInfo, every 8 streamIds as a group
    PH->>Profapi: dlMsprofReportAdditionalInfo
    Note right of Profapi: level=NODE_LEVEL, type=MC2_COMMINFO
```

#### Manage Host-Side Profiling Switch

```mermaid
sequenceDiagram
    participant Profapi as profapi component
    participant PH as ProfilingHandler
    participant DPF as DlProfFunction

    Profapi->>PH: CommandHandleWrapper
    Note right of PH: profapi component notifies switch status changes through callback
    PH->>PH: CommandHandle
    Note right of PH: Parse the switch mask based on rtType, update enableHostApi_/enableHcclNode_/enableHcclL0_/enableHcclL1_
    PH->>DPF: dlMsprofRegTypeInfo
    Note right of DPF: Register type mapping with the profapi component
```

### Device Profiling Flow

#### Register Operator Information (HcclDfxRegOpInfoByCommId)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant AIP as AicpuIndopProcess
    participant HCDL as HcclCommDfxLite
    participant MTML as MirrorTaskManagerLite

    HCCL->>AIP: HcclDfxRegOpInfoByCommId
    AIP->>AIP: HcommThreadGetNotifyId
    Note right of AIP: Get cpuWaitAicpuNotifyId
    AIP->>AIP: AicpuDfxOpInfoInit
    Note right of AIP: Convert HcclDfxOpInfo to DfxOpInfo
    AIP->>HCDL: UpdateProfStat
    AIP->>MTML: SetCurrDfxOpInfo
    Note right of MTML: Store current operator info for later reporting
```

#### Report Device Operator (HcommProfilingReportDeviceOp)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant AIP as AicpuIndopProcess
    participant HCDL as HcclCommDfxLite
    participant HCPL as HcclCommProfilingLite
    participant PRL as ProfilingReporterLite
    participant PHL as ProfilingHandlerLite
    participant Profapi as profapi component

    HCCL->>AIP: HcommProfilingReportDeviceOp
    alt currDfxOpInfo is empty
        Note right of AIP: Skip reporting, return success
    else currDfxOpInfo is not empty
        AIP->>HCDL: ReportAllTasks
        HCDL->>HCPL: ReportAllTasks
        HCPL->>PRL: ReportAllTasks
        Note right of PRL: Traverse the task queue in MirrorTaskManagerLite
        PRL->>PHL: ReportHcclTaskDetails
        PHL->>Profapi: MsprofReportAdditionalInfo
        Note right of Profapi: Report Task details
        AIP->>PHL: ReportHcclOpInfo
        PHL->>Profapi: MsprofReportAdditionalInfo
        Note right of Profapi: Report HCCL OP info
    end
```

#### Report Kernel Start Task (HcommProfilingReportKernelStartTask)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant AIP as AicpuIndopProcess
    participant PHL as ProfilingHandlerLite
    participant Profapi as profapi component

    HCCL->>AIP: HcommProfilingReportKernelStartTask
    AIP->>AIP: UpdateTask
    Note right of AIP: Update Profiling status
    AIP->>PHL: ReportMainStreamTask
    Note right of PHL: type=HEAD, record the first task of the main stream
    PHL->>Profapi: MsprofReportAdditionalInfo
    Note right of Profapi: Report FlagTaskInfo
```

#### Report Kernel End Task (HcommProfilingReportKernelEndTask)

```mermaid
sequenceDiagram
    participant HCCL as hccl
    participant AIP as AicpuIndopProcess
    participant PHL as ProfilingHandlerLite
    participant Profapi as profapi component

    HCCL->>AIP: HcommProfilingReportKernelEndTask
    AIP->>PHL: ReportMainStreamTask
    Note right of PHL: type=TAIL, record the last task of the main stream
    PHL->>Profapi: MsprofReportAdditionalInfo
    Note right of Profapi: Report FlagTaskInfo
```

#### Manage Device-Side Profiling Switch

```mermaid
sequenceDiagram
    participant PHL as ProfilingHandlerLite
    participant Profapi as profapi component

    PHL->>PHL: UpdateProfSwitch
    Note right of PHL: Actively query the profapi component switch status
    PHL->>Profapi: AdprofCheckFeatureIsOn
    Note right of Profapi: Check whether L0/L1 switches are enabled
    PHL->>PHL: SetProL0On/SetProL1On
    Note right of PHL: Update enableHcclL0_/enableHcclL1_
```

### Report Level and Type Constants Summary

| Constant Name | Value | Description |
|--------|-----|------|
| `MSPROF_REPORT_ACL_LEVEL` | 20000 | ACL level, used for Host API reporting |
| `MSPROF_REPORT_NODE_LEVEL` | 10000 | Node level, used for Node BasicInfo, HCCL OP, and MC2 CommInfo reporting |
| `MSPROF_REPORT_HCCL_NODE_LEVEL` | 5500 | HCCL Node level, used for Task details and CCU information reporting |
| `MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE` | 0x070000 | ACL Host HCCL base type |
| `MSPROF_REPORT_NODE_LAUNCH_TYPE` | 5 | Node Launch type, used for `ReportNodeApi` |
| `MSPROF_REPORT_NODE_BASIC_INFO_TYPE` | 0 | Node basic information type, used for `ReportNodeBasicInfo` |
| `MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE` | 10 | Node HCCL OP information type |
| `MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE` | 12 | Node MC2 communication resource information type |
| `MSPROF_REPORT_HCCL_MASTER_TYPE` | 0x010001 | HCCL main stream type |
| `MSPROF_REPORT_HCCL_SLAVE_TYPE` | 0x010002 | HCCL slave stream type |
| `MSPROF_REPORT_CCU_TASK_INFO` | 14 | CCU Task information type |
| `MSPROF_REPORT_CCU_WAIT_SIGNAL_INFO` | 15 | CCU Wait Signal information type |
| `MSPROF_REPORT_CCU_GROUP_INFO` | 16 | CCU Group information type |

### Switch Control and Report Content Mapping

| Switch | Corresponding Mask | Controlled Report Content |
|------|----------|---------------|
| `enableHostApi_` | `PROF_ACL_API_MASK` = 0x1 | Host API timestamp reporting (`ReportAclApi`, `ReportNodeApi`, `ReportHcclOpApi`, `ReportHcclOpInfo`, `ReportMc2AdditionInfo`) |
| `enableHcclL0_` | `PROF_TASK_TIME_MASK` = 0x800 | HCCL operator granularity tracing (`ReportHcclOpApi`) |
| `enableHcclNode_` | `PROF_TASK_TIME_L1_MASK` = 0x2 | Task granularity tracing (`ReportHcclTaskApi`) |
| `enableHcclL1_` | `PROF_TASK_TIME_L1_MASK` = 0x2 | Task details reporting (`CallAddtionInfo`, `ReportNodeBasicInfo`) |
| `enableHcclL2_` | `PROF_TASK_TIME_L2_MASK` = 0x2000 | CCU details reporting (`ReportCcuInfo`) |

## Interface Description (Class Diagram)

```mermaid
classDiagram
    class HcclCommProfiling {
        -MirrorTaskManager* mirrorTaskManager_
        -unique_ptr~ProfilingReporter~ profilingReporter_
        +HcclCommProfiling(u32 deviceId, MirrorTaskManager* mirrorTaskManager)
        +ReportAllTasks(bool cachedReq) void
        +ReportOp(uint64_t beginTime, bool cachedReq, bool opbased) void
        +ReportMc2CommInfo(Mc2CommInfo& mc2CommInfo) void
        +UpdateProfStat() void
        +GetMirrorTaskManager() MirrorTaskManager*
        +ReportKernel(uint64_t beginTime, string& commTag, string& kernelName, uint32_t threadId, bool cachedReq) HcclResult
    }

    class HcclCommProfilingLite {
        -MirrorTaskManagerLite* mirrorTaskManagerLite_
        -unique_ptr~ProfilingReporterLite~ profilingReporterLite_
        +HcclCommProfilingLite(DevId deviceId, MirrorTaskManagerLite* mirrorTaskManagerLite)
        +ReportAllTasks() void
        +UpdateProfStat() void
        +GetMirrorTaskManagerLite() MirrorTaskManagerLite*
    }

    class ProfilingReporter {
        -MirrorTaskManager* mirrorTaskMgr_
        -bool enableHcclL1_
        -ProfilingHandler* profilingHandler_
        +ProfilingReporter(MirrorTaskManager*, ProfilingHandler*)
        +Init() void
        +ReportOp(uint64_t beginTime, bool cachedReq, bool opbased) void
        +ReportAllTasks(bool cachedReq) void
        +UpdateProfStat() void
        +CallReportMc2CommInfo(u32 kfcStreamId, vector~u32~ aicpuStreamsId, string id, RankId myRank, u32 rankSize, RankId rankInParentComm) void
    }

    class ProfilingReporterLite {
        -MirrorTaskManagerLite* mirrorTaskMgrLite_
        -ProfilingHandlerLite* profilingHandlerLite_
        -map lastPoses_
        +ProfilingReporterLite(MirrorTaskManagerLite*, ProfilingHandlerLite*, bool isIndop)
        +Init() void
        +ReportAllTasks() void
        +UpdateProfStat() void
    }

    class ProfilingHandler {
        -static ProfilingHandler instance_
        -bool enableHostApi_
        -bool enableHcclNode_
        -bool enableHcclL0_
        -bool enableHcclL1_
        +GetInstance() ProfilingHandler&$
        +ReportNodeApi(uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq) void
        +ReportNodeBasicInfo(uint64_t timeStamp, uint64_t cmdItemId, uint32_t threadId, bool cachedReq) void
        +ReportHostApi(OpType opType, uint64_t beginTime, uint64_t endTime, bool cachedReq, bool isAiCpu) void
        +ReportHcclOp(DfxOpInfo& opInfo, bool cachedReq) void
    }

    class ProfilingHandlerLite {
        -static ProfilingHandlerLite instance_
        -bool enableHcclL0_
        -bool enableHcclL1_
        +GetInstance() ProfilingHandlerLite&$
        +ReportHcclOpInfo(DfxOpInfo& opInfo) void
        +ReportHcclTaskDetails(vector~TaskInfo~& taskInfo) void
        +ReportMainStreamTask(FlagTaskInfo& flagTaskInfo) void
        +UpdateProfSwitch() void
    }

    class Mc2CommInfo {
        +u32 FreeStreamId
        +vector~u32~ streamsId
        +string groupname
        +u32 myRankId
        +u32 rankSize
        +u32 parentRankId
    }

    HcclCommProfiling o-- ProfilingReporter : holds
    HcclCommProfiling o-- MirrorTaskManager : holds pointer
    HcclCommProfiling ..> ProfilingHandler : accessed indirectly through Reporter
    HcclCommProfiling ..> profapi component : dlMsprofSysCycleTime/dlMsprofStr2Id
    HcclCommProfiling --> Mc2CommInfo : uses

    HcclCommProfilingLite o-- ProfilingReporterLite : holds
    HcclCommProfilingLite o-- MirrorTaskManagerLite : holds pointer
    HcclCommProfilingLite ..> ProfilingHandlerLite : accessed indirectly through Reporter

    ProfilingReporter o-- MirrorTaskManager : holds pointer
    ProfilingReporter ..> ProfilingHandler : calls reporting interface

    ProfilingReporterLite o-- MirrorTaskManagerLite : holds pointer
    ProfilingReporterLite ..> ProfilingHandlerLite : calls reporting interface

    ProfilingHandler ..> profapi component : dlMsprofReportApi/dlMsprofReportCompactInfo/dlMsprofReportAdditionalInfo
    ProfilingHandler ..> rts component : orion_adapter_rts.h
    ProfilingHandlerLite ..> profapi component : MsprofReportAdditionalInfo/AdprofReportAdditionalInfo
```

## Interface Description

### HcclCommProfiling (Host Side)

| Interface | Type | Parameters | Return Value | Description |
|--------|------|------|--------|----------|
| `HcclCommProfiling` | Public | `[in] u32 deviceId`, `[in] MirrorTaskManager* mirrorTaskManager` | - | Constructor, saves the task manager pointer, creates a `ProfilingReporter` instance (associated with `ProfilingHandler::GetInstance()`). |
| `ReportAllTasks` | Public | `[in] bool cachedReq = false` | void | Reports all communication tasks. When `cachedReq=true`, indicates cache request mode. Delegates to `ProfilingReporter::ReportAllTasks` to traverse the task queue and report through `ProfilingHandler`. |
| `ReportOp` | Public | `[in] uint64_t beginTime`, `[in] bool cachedReq`, `[in] bool opbased` | void | Reports operator information. Delegates to `ProfilingReporter::ReportOp`, which ultimately calls the profapi component `dlMsprofReportApi` through `ProfilingHandler::ReportHostApi`. |
| `ReportMc2CommInfo` | Public | `[in] const Mc2CommInfo& mc2CommInfo` | void | Reports MC2 communication domain information. Splits the `Mc2CommInfo` fields and calls `ProfilingReporter::CallReportMc2CommInfo`, which ultimately reports through `dlMsprofReportAdditionalInfo`. |
| `UpdateProfStat` | Public | - | void | Updates Profiling statistics. Delegates to `ProfilingReporter::UpdateProfStat` to update the switch status. |
| `GetMirrorTaskManager` | Public | - | `MirrorTaskManager*` | Returns the internally held `MirrorTaskManager` pointer. |
| `ReportKernel` | Public | `[in] uint64_t beginTime`, `[in] const string& commTag`, `[in] const string& kernelName`, `[in] uint32_t threadId`, `[in] bool cachedReq` | HcclResult | Reports CCU Kernel information. Calls the profapi component `dlMsprofSysCycleTime` to get endTime and `dlMsprofStr2Id` to get cmdItemId, then calls `ProfilingHandler` to report `ReportNodeApi` and `ReportNodeBasicInfo`. The `EXCEPTION_CATCH` macro catches exceptions and returns `HCCL_E_PTR` on failure. |

### HcclCommProfilingLite (Device Side)

| Interface | Type | Parameters | Return Value | Description |
|--------|------|------|--------|----------|
| `HcclCommProfilingLite` | Public | `[in] DevId deviceId`, `[in] MirrorTaskManagerLite* mirrorTaskManagerLite` | - | Constructor, saves the task manager pointer, creates a `ProfilingReporterLite` instance (`isIndop=true`, associated with `ProfilingHandlerLite::GetInstance()`). |
| `ReportAllTasks` | Public | - | void | Reports all communication tasks. Delegates to `ProfilingReporterLite::ReportAllTasks`, which ultimately calls the profapi component `MsprofReportAdditionalInfo` through a weak symbol. |
| `UpdateProfStat` | Public | - | void | Updates Profiling statistics. Delegates to `ProfilingReporterLite::UpdateProfStat` to update the switch status. |
| `GetMirrorTaskManagerLite` | Public | - | `MirrorTaskManagerLite*` | Returns the internally held `MirrorTaskManagerLite` pointer. |

## Usage Limitations

### Supported Scenarios

| Scenario | Host Side | Device Side | Description |
|------|---------|----------|------|
| Register operator information | Supported | Supported | Registered through `HcclDfxRegOpInfoByCommId`. |
| Report operator | Supported | Supported | Host reports through `HcclProfilingReportOp`; Device reports through `HcommProfilingReportDeviceOp`. |
| Report Kernel | Supported | Supported | Host supports AICPU and AIV Kernel reporting; Device supports kernel start and end task reporting. |
| Report MC2 communication info | Supported | Not supported | Only the Host side supports `ReportMc2CommInfo`. |
| Update Profiling status | Supported | Supported | Both sides support this. |
| Multi-device task management | Supported | Not supported | The Host-side `MirrorTaskManager` supports multi-device queue mapping. |
| Report CCU information | Supported | Not supported | Only the Host side supports CCU Task, WaitSignal, and Group information reporting. |
| Get system cycle time | Supported | Not supported | Only the Host side supports `HcommGetProfilingSysCycleTime`. |

### Constraint Specifications

1. **Maximum device count**: The Host-side `ProfilingReporter` static array `allLastPoses_` has a size of `REPORTER_MAX_MODULE_DEVICE_NUM` = 65, supporting profiling position records for up to 65 devices.
2. **ProfilingHandler singleton pattern**: Both Host-side and Device-side `ProfilingHandler` and `ProfilingHandlerLite` are singletons, globally unique. Copying and assignment are prohibited.
3. **DlProfFunction dynamic loading**: The Host side dynamically loads `libprofapi.so` through `DlProfFunction` using `dlopen`. If the SDK is unavailable, it falls back to stub functions (printing a WARNING log and skipping).
4. **Device-side weak symbol linking**: The Device side declares profapi component functions through `__attribute__((weak))` (such as `MsprofReportAdditionalInfo` and `AdprofReportAdditionalInfo`). At runtime, the selection priority is: `MsprofReportBatchAdditionalInfo` > `AdprofReportAdditionalInfo` > `MsprofReportAdditionalInfo`.
5. **Null pointer protection**: All reporting interfaces check for non-null pointers before calling the Reporter to avoid null pointer dereferences.
6. **EXCEPTION_CATCH macro**: `ReportKernel` uses the `EXCEPTION_CATCH` macro to catch exceptions during `ProfilingHandler` reporting, returning `HCCL_E_PTR` on failure.
7. **MC2 Stream group reporting**: `ReportMc2CommInfo` groups every 8 streamIds into a group and reports them through `ProfilingDeviceCommResInfo`. The `commStreamIds` array size is fixed at 8.
8. **Device-side device type restriction**: `HcommProfilingReportDeviceOp`, `HcommProfilingReportKernelStartTask`, and `HcommProfilingReportKernelEndTask` only execute on `DEV_TYPE_950` and `DEV_TYPE_960` devices. For other device types, they directly return success.

### Known Limitations

1. **`aicpu_ts_urma_dfx_kernel` is deprecated**: The Device-side `aicpu_ts_urma_dfx_kernel.h` and `.cc` files are deprecated and no longer maintained, but the build entry is still retained in `CMakeLists.txt`.
2. **`Mc2CommInfo` has no validation**: The `ReportMc2CommInfo` interface does not validate the length of the `streamsId` vector in `mc2CommInfo`. This is handled by the underlying `CallReportMc2CommInfo`.
3. **Thread safety**: The Host-side `ProfilingHandler` uses multiple mutexes (`cacheTaskInfosMutex_`, `cachedTaskApiInfoMutex_`, `cacheHcclOpInfoMutex_`, `cacheHcclAdditionInfoMutex_`) to protect cached data. The Device-side `ProfilingHandlerLite` does not use locks and relies on a single-threaded execution environment.
4. **Device-side switch query mode**: Unlike the Host side, which receives switch status passively through callbacks, the Device-side `ProfilingHandlerLite` must actively call `UpdateProfSwitch()` to query the switch status from the profapi component.
5. **Non-V2 communication domain handling**: On `DEV_TYPE_910B` devices, `HcclProfilingReportOp` directly returns success and skips reporting for non-`CommunicatorV2` communication domains. For other device types, `HCCL_E_NOT_SUPPORT` is returned for non-V2 domains.
