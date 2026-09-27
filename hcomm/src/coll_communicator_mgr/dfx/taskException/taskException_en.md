# taskException Module Code Analysis

## Feature Description

The taskException module is the **DFX (Design for eXcellence) exception diagnosis subsystem** in the HCCL collective communication library. It is responsible for exception capture, error information parsing, diagnostic log printing, and error reporting when collective communication tasks fail to execute. This module spans two runtime environments: the Host side and the AICPU side.

- **Host side**: Receives exception callbacks from the Runtime, distributes them to different exception handling flows based on task type (general task exception or CCU task exception), reads AICPU error information, prints UB DFX register information, and reports cluster monitoring errors.
- **AICPU side**: Runs as a daemon thread that periodically checks for CQE (Completion Queue Entry) exceptions on each communication domain stream. It parses exception CQEs, organizes error information, reports to the Host side through the HDC channel, and notifies TSFW through Mailbox.

Core capabilities include:

1. Registering and unregistering exception callback functions with the Runtime
2. Searching for abnormal TaskInfo based on GlobalMirrorTasks
3. Reading ErrorMessageReport from the AICPU side through the HDC channel
4. Parsing CCU (Cube Compute Unit) task exceptions and restoring the error instruction context based on the CcuRep instruction representation system
5. Printing UB DFX register information for hardware diagnostics
6. Reporting CQE error information to cluster monitoring
7. Printing the context of up to 50 preceding tasks before the exception task

---

## Directory Description

```text
taskException/
├── host/                                    # Host-side exception handling
│   ├── hcclCommTaskException.h              # TaskExceptionHost / TaskExceptionHostManager class declarations
│   ├── hcclCommTaskException.cc             # Host-side general exception handling implementation + callback registration management
│   ├── ccuTaskException.h                   # CcuTaskException class declaration
│   └── ccuTaskException.cc                  # CCU task exception handling implementation (instruction parsing, register reading, error information generation)
└── aicpu/                                   # AICPU-side exception handling
    ├── hcclCommTaskExceptionLite.h          # HcclCommTaskExceptionLite class declaration
    └── hcclCommTaskExceptionLite.cc         # AICPU-side exception detection, CQE parsing, and error reporting implementation
```

```text
ccu_dfx/ (located at base_comm/resources/ccu/ccu_dfx/)
├── ccu_error_info_v1.h                  # Ascend 950 error information data structure definitions (CcuErrorInfo, CcuLoopContext, CcuMissionContext, etc.)
├── ccu_error_info_v2.h                  # Ascend 960 error information data structure definitions (CcuLoopContextV2, CcuMissionContextV2)
├── ccu_dfx_schema.h                     # V1/V2 schema dispatch interface declarations (CcuVersionOps, GetCcuOps)
└── ccu_dfx_schema.cc                    # V1/V2 schema dispatch implementation (selects decode/print implementation by device type 950->V1 / 960->V2)
```

### File Relationships

| File | Function | Dependencies |
|------|------|----------|
| `hcclCommTaskException.h/.cc` | Host-side main entry point, registers Runtime exception callbacks, distributes to general or CCU exception handling | Depends on `ccuTaskException.h` for CCU type exception handling; depends on `global_mirror_tasks.h` for finding TaskInfo; obtains ErrorMessageReport from the AICPU side through callbacks |
| `ccuTaskException.h/.cc` | CCU task exception-specific handling, parses CcuRep instruction representation, reads hardware registers | Depends on data structures in `ccu_error_info_v1.h`; depends on `ccu_kernel_mgr.h` for CcuRepContext; depends on `ccu_urma_channel.h` for channel information |
| `ccu_error_info_v1.h` (located at `base_comm/resources/ccu/ccu_dfx/`) | Ascend 950 error information data structure definitions (CcuErrorInfo, CcuLoopContext, CcuMissionContext, etc.) | Referenced by `ccuTaskException.h/.cc`; depends on `ccu_rep_type_v1.h` for the CcuRepType enum definition |
| `ccu_error_info_v2.h` (located at `base_comm/resources/ccu/ccu_dfx/`) | Ascend 960 error information data structure definitions (CcuLoopContextV2, CcuMissionContextV2) | Referenced by `ccuTaskException.h`; zero external dependencies |
| `ccu_dfx_schema.h` (located at `base_comm/resources/ccu/ccu_dfx/`) | V1/V2 schema dispatch interface declarations, defines CcuMissionInfo/CcuLoopInfo semantic output structures and CcuVersionOps dispatch table entry, declares GetCcuOps entry | Depends on `hccl_types.h` for HcclResult; implemented by `ccu_dfx_schema.cc` and referenced by `ccuTaskException.cc` (GetCcuOps called by ProcessCcuException / PrintPanicLogInfo / DecodeRawMissionInfo / FetchAndDecodeLoopInfo) |
| `ccu_dfx_schema.cc` (located at `base_comm/resources/ccu/ccu_dfx/`) | V1/V2 schema dispatch implementation, selects corresponding Mission/Loop context decode and CCUM DFX register print implementation by device type (950->V1 / 960->V2) | Depends on `ccu_dfx_schema.h` interface; depends on `ccu_error_info_v1.h` for CcuMissionContext / CcuLoopContext; depends on `ccu_error_info_v2.h` for CcuMissionContextV2 / CcuLoopContextV2; depends on `adapter_rts_common.h` for hrtGetDeviceType to obtain device type |
| `hcclCommTaskExceptionLite.h/.cc` | AICPU-side daemon thread, detects CQE exceptions and reports to Host | Depends on `coll_comm_aicpu.h` for the AICPU communication domain; depends on `error_message_v2.h` for organizing ErrorMessageReport and reporting to the Host through HDC |

### TaskException File Interaction

```mermaid
graph TB
    subgraph Registration Entry
        HCCL[HCCL]
    end

    subgraph Exception Trigger Source
        Runtime[Runtime]
    end

    subgraph Host Side
        hcomm_c_adpt_H[hcomm_c_adpt]
        CollComm[CollComm]
        HcclCommDfx[HcclCommDfx]
        MirrorTaskManager[MirrorTaskManager]
        TaskExceptionHost[TaskExceptionHost]
        CcuTaskException[CcuTaskException]
        CcuKernelMgr[CcuKernelMgr]
    end

    subgraph AICPU Side
        hcomm_c_adpt_D[hcomm_c_adpt]
        AicpuIndopProcess[AicpuIndopProcess]
        CollCommAicpu[CollCommAicpu]
        HcclCommDfxLite[HcclCommDfxLite]
        MirrorTaskManagerLite[MirrorTaskManagerLite]
        BGThread[AICPU background thread]
        Lite[HcclCommTaskExceptionLite]
        StreamLite[StreamLite]
    end

    subgraph External Components
        hccp[hccp component]
        rts[rts component]
        error_manager[error_manager component]
        hal[hal component]
    end

    HCCL -->|HcclDfxRegOpInfoByCommId| hcomm_c_adpt_H
    HCCL -->|HcclDfxRegOpInfoByCommId| hcomm_c_adpt_D
    hcomm_c_adpt_H -->|RegisterAicpuTaskExceptionCallback| CollComm
    CollComm -->|Register| TaskExceptionHost
    hcomm_c_adpt_H -->|GetMirrorTaskManager| HcclCommDfx
    HcclCommDfx -->|SetCurrDfxOpInfo/AddTaskInfo| MirrorTaskManager
    hcomm_c_adpt_D -->|AicpuDfxOpInfoInit| AicpuIndopProcess
    AicpuIndopProcess -->|GetHcclCommDfxLite| CollCommAicpu
    CollCommAicpu -->|GetMirrorTaskManagerLite| HcclCommDfxLite
    HcclCommDfxLite -->|SetCurrDfxOpInfo/AddTaskInfo| MirrorTaskManagerLite
    Runtime -->|Process| TaskExceptionHost
    BGThread -->|Call| Lite
    TaskExceptionHost -->|ProcessCcuException| CcuTaskException
    CcuTaskException --> CcuKernelMgr
    Lite -->|GetThreadCqe| StreamLite
    Lite -->|HDC channel| TaskExceptionHost
    Lite -->|Mailbox| Runtime
    TaskExceptionHost -->|adapter_error_manager_pub| error_manager
    CcuTaskException -->|HccpRaCustomChannel| hccp
    TaskExceptionHost -->|orion_adapter_hccp| hccp
    TaskExceptionHost -->|orion_adapter_rts| rts
    Lite -->|dlHalEschedSubmitEvent| hal
```

---

## Flow Description

### Registration Flow

#### Aicpu Mode Host-Side Registration Flow

```mermaid
sequenceDiagram
    participant HCCL
    participant hcomm_c_adpt
    participant CollComm
    participant TaskExceptionHost
    participant HcclCommDfx
    participant MirrorTaskManager

    Note over HCCL,MirrorTaskManager: Operator registration phase

    HCCL->>hcomm_c_adpt: HcclDfxRegOpInfoByCommId
    Note right of hcomm_c_adpt: Register operator DFX info

    hcomm_c_adpt->>hcomm_c_adpt: ConvertToDfxOpInfo
    Note right of hcomm_c_adpt: Convert to internal DfxOpInfo, set isIndop_=true

    hcomm_c_adpt->>CollComm: RegisterAicpuTaskExceptionCallback
    Note right of CollComm: Register AICPU exception callback

    CollComm->>TaskExceptionHost: Register
    Note right of TaskExceptionHost: Write to CommRegisterMap_. The first comm registration calls aclrtExceptionInfoCallbackRegister.

    hcomm_c_adpt->>HcclCommDfx: GetMirrorTaskManager
    HcclCommDfx-->>hcomm_c_adpt: mirrorTaskManager
    hcomm_c_adpt->>MirrorTaskManager: SetCurrDfxOpInfo
    Note right of MirrorTaskManager: Set current operator info

    Note over HCCL,MirrorTaskManager: Task dispatch phase

    HcclCommDfx->>HcclCommDfx: AddTaskInfoCallback
    Note right of HcclCommDfx: Thread or Channel callback triggers TaskInfo creation associated with DfxOpInfo
    HcclCommDfx->>MirrorTaskManager: AddTaskInfo
    Note right of MirrorTaskManager: Register with GlobalMirrorTasks
```

#### Aicpu Mode Device-Side Registration Flow

```mermaid
sequenceDiagram
    participant HCCL
    participant hcomm_c_adpt
    participant AicpuIndopProcess
    participant CollCommAicpu
    participant HcclCommDfxLite
    participant MirrorTaskManagerLite

    Note over HCCL,MirrorTaskManagerLite: Operator registration phase

    HCCL->>hcomm_c_adpt: HcclDfxRegOpInfoByCommId
    Note right of hcomm_c_adpt: aicpu_ts_primitives_c_adpt, get cpuWaitAicpuNotifyId

    hcomm_c_adpt->>AicpuIndopProcess: AicpuDfxOpInfoInit
    Note right of AicpuIndopProcess: Initialize AICPU-side operator info

    AicpuIndopProcess->>AicpuIndopProcess: ConvertToDfxOpInfo
    Note right of AicpuIndopProcess: Set isIndop_=true, opIndex_ auto-increment

    AicpuIndopProcess->>CollCommAicpu: GetHcclCommDfxLite
    CollCommAicpu-->>AicpuIndopProcess: hcclCommDfxLite
    AicpuIndopProcess->>HcclCommDfxLite: GetMirrorTaskManagerLite
    HcclCommDfxLite-->>AicpuIndopProcess: mirrorTaskManagerLite
    AicpuIndopProcess->>MirrorTaskManagerLite: SetCurrDfxOpInfo
    Note right of MirrorTaskManagerLite: Set current operator info

    Note over HCCL,MirrorTaskManagerLite: Task dispatch phase

    HcclCommDfxLite->>HcclCommDfxLite: AddTaskInfoCallback
    Note right of HcclCommDfxLite: Thread callback triggers TaskInfo creation associated with DfxOpInfo
    HcclCommDfxLite->>MirrorTaskManagerLite: AddTaskInfo
    Note right of MirrorTaskManagerLite: Register with MirrorTaskManagerLite
```

#### CCU Mode Registration Flow

```mermaid
sequenceDiagram
    participant HCCL
    participant hcomm_c_adpt
    participant HcclCommDfx
    participant MirrorTaskManager

    Note over HCCL,MirrorTaskManager: CCU and Aicpu share the HcclDfxRegOpInfoByCommId entry

    HCCL->>hcomm_c_adpt: HcclDfxRegOpInfoByCommId
    Note right of hcomm_c_adpt: Skip RegAicpuTaskException when not AICPU_TS engine

    hcomm_c_adpt->>hcomm_c_adpt: ConvertToDfxOpInfo
    Note right of hcomm_c_adpt: Set isIndop_=true

    hcomm_c_adpt->>HcclCommDfx: GetMirrorTaskManager
    HcclCommDfx-->>hcomm_c_adpt: mirrorTaskManager
    hcomm_c_adpt->>MirrorTaskManager: SetCurrDfxOpInfo

    Note over HCCL,MirrorTaskManager: CCU task dispatch phase

    HcclCommDfx->>HcclCommDfx: AddTaskInfoCallback
    Note right of HcclCommDfx: Thread or Channel callback triggers. When taskType==TASK_CCU, traverse ccuDetailInfo to fill remoteRankId.
    HcclCommDfx->>HcclCommDfx: Create TaskInfo associated with DfxOpInfo
    HcclCommDfx->>MirrorTaskManager: AddTaskInfo
    Note right of MirrorTaskManager: Register with GlobalMirrorTasks
```

### Exception Handling Flow

#### Aicpu Mode Host-Side Exception Handling Flow

```mermaid
sequenceDiagram
    participant Runtime
    participant TaskExceptionHost
    participant MirrorTaskManager
    participant CollComm
    participant CcuTaskException
    participant Legacy
    participant hccp component
    participant error_manager component

    Runtime->>TaskExceptionHost: ProcessCallback
    Note right of TaskExceptionHost: Runtime exception callback, forwarded to Process

    alt IsMC2Exception
        TaskExceptionHost->>Legacy: TaskExceptionHandler::Process
        Note right of Legacy: Fall back to legacy flow
    else Non-MC2 exception
        TaskExceptionHost->>MirrorTaskManager: FindTaskInfo
        MirrorTaskManager-->>TaskExceptionHost: curTask

        alt Non-Indop task or commHandle not registered
            TaskExceptionHost->>Legacy: TaskExceptionHandler::Process
            Note right of Legacy: Fall back to legacy flow
        else Indop task and commHandle registered
            alt taskType==TASK_CCU
                TaskExceptionHost->>CcuTaskException: ProcessCcuException
                Note right of CcuTaskException: Forward to CCU exception handling flow
            else Non-CCU task
                TaskExceptionHost->>TaskExceptionHost: ProcessException
                Note right of TaskExceptionHost: Check hasAicpuReport_

                alt hasAicpuReport_ is false
                    TaskExceptionHost->>CollComm: GetAicpuTaskException
                    CollComm-->>TaskExceptionHost: ErrorMessageReport
                end

                alt AICPU has reported error (tag is not empty)
                    TaskExceptionHost->>TaskExceptionHost: HandleAicpuErrorReport
                    Note right of TaskExceptionHost: hasAicpuReport_=true
                    TaskExceptionHost->>TaskExceptionHost: Print error log
                    Note right of TaskExceptionHost: BaseInfo/ParaInfo/GroupInfo/OpDataInfo
                    TaskExceptionHost->>hccp component: PrintUbDfxInfo→PrintUbRegisters→RaGetAuxInfo
                    Note right of hccp component: RaCtxGetAuxInfo reads UB registers
                    TaskExceptionHost->>error_manager component: ReportErrorMsg→RPT_INPUT_ERR
                    Note right of error_manager component: TASK_NOTIFY_WAIT→EI0002, UB/Write→EI0018
                    TaskExceptionHost->>TaskExceptionHost: GetAicpuCqeErrInfo
                    Note right of TaskExceptionHost: Report to cluster monitoring, triggered when ubCqeStatus is non-zero
                else AICPU has no error reported
                    TaskExceptionHost->>TaskExceptionHost: HandleHostErrorReport
                    Note right of TaskExceptionHost: Host side prints error info independently, prints preceding tasks for TASK_NOTIFY_WAIT
                end
            end
        end
    end
```

#### Aicpu Mode Device-Side Exception Handling Flow

```mermaid
sequenceDiagram
    participant AICPU background thread
    participant HcclCommTaskExceptionLite
    participant CollCommAicpu
    participant StreamLite
    participant TaskExceptionHost
    participant hal component

    AICPU background thread->>HcclCommTaskExceptionLite: Call

    alt stopCall_ is true
        Note right of HcclCommTaskExceptionLite: Skip to prevent flooding
    else stopCall_ is false
        HcclCommTaskExceptionLite->>HcclCommTaskExceptionLite: HandleExceptionCqe
        Note right of HcclCommTaskExceptionLite: Detect CQE exceptions in all communication domains

        loop Traverse all communication domains
            HcclCommTaskExceptionLite->>CollCommAicpu: GetAllThread
            loop Traverse all threads
                HcclCommTaskExceptionLite->>StreamLite: GetThreadCqe
                Note right of StreamLite: Get CQE through CqReportRecv
                StreamLite-->>HcclCommTaskExceptionLite: cqeException, cqeStatus

                alt Exception CQE exists
                    HcclCommTaskExceptionLite->>HcclCommTaskExceptionLite: PrintTaskExceptionBySqeId
                    Note right of HcclCommTaskExceptionLite: Print Task/EID/Group/OpData
                    HcclCommTaskExceptionLite->>HcclCommTaskExceptionLite: ReportErrMsg

                    alt IsErrorReported is false
                        HcclCommTaskExceptionLite->>HcclCommTaskExceptionLite: GenerateErrorMessageReport
                        HcclCommTaskExceptionLite->>TaskExceptionHost: SendErrorMessageReportToHost
                        Note right of TaskExceptionHost: Report through HDC channel
                        HcclCommTaskExceptionLite->>hal component: SendTaskExceptionByMBox→dlHalEschedSubmitEvent
                        Note right of hal component: Mailbox reporting, <br/>See table below for error code conversion
                        HcclCommTaskExceptionLite->>CollCommAicpu: SetErrorReported(true)
                    end

                    alt NotifyWait timeout (sqeType==PLACE_HOLDER)
                        HcclCommTaskExceptionLite->>CollCommAicpu: PrintCommTaskException
                        Note right of CollCommAicpu: Print all streams of the current communication domain
                        loop Traverse other communication domains
                            HcclCommTaskExceptionLite->>HcclCommTaskExceptionLite: PrintAllCommTaskException
                        end
                    end
                end
            end
        end

        alt HandleExceptionCqe fails
            Note right of HcclCommTaskExceptionLite: stopCall_=true, stop subsequent calls
        end
    end
```

**SendTaskExceptionByMBox Error Code Conversion Table** (Source: `hcclCommTaskExceptionLite.cc:429-435`, constants defined in: `hcomm_task_scheduler_error.h`)

| CQE sqeType | CQE errorCode | TS Error Code | Value | Description |
|-------------|--------------|-----------|-----|------|
| UB (sqeType=9) | 0x02 | TS_ERROR_HCCL_OP_UB_DDRC_FAILED | 0x3ea | UB local side returns ERROR |
| UB (sqeType=9) | 0x03 | TS_ERROR_HCCL_OP_UB_POISON_FAILED | 0x3eb | UB remote side returns ERROR |
| UB (sqeType=9) | 0x05 | TS_ERROR_HCCL_OP_UB_LINK_FAILED | 0x3ec | UB network exception, taack timeout |
| UB (sqeType=9) | Other | TS_ERROR_HCCL_OTHER_ERROR | 0x223 | UB other error |
| SDMA (sqeType=11) | 0x09 | TS_ERROR_SDMA_LINK_ERROR | 0x222 | SDMA write copy timeout acknowledgment or address decoding error |
| SDMA (sqeType=11) | 0x0a | TS_ERROR_SDMA_POISON_ERROR | 0x221 | SDMA read copy timeout acknowledgment or read HBM returns ERROR |
| SDMA (sqeType=11) | 0x08 | TS_ERROR_SDMA_DDRC_ERROR | 0x220 | SDMA read HBM returns ERROR |
| SDMA (sqeType=11) | Other | TS_ERROR_HCCL_OTHER_ERROR | 0x223 | SDMA other error |
| Other sqeType | - | TS_ERROR_HCCL_OTHER_ERROR | 0x223 | Non-UB or SDMA error |

#### CCU Mode Exception Handling Flow

```mermaid
sequenceDiagram
    participant Runtime
    participant TaskExceptionHost
    participant MirrorTaskManager
    participant CcuTaskException
    participant CcuKernelMgr
    participant CcuComponent
    participant hccp component

    Runtime->>TaskExceptionHost: Process
    TaskExceptionHost->>MirrorTaskManager: FindTaskInfo
    MirrorTaskManager-->>TaskExceptionHost: curTask
    Note right of TaskExceptionHost: Dispatched when taskType==TASK_CCU

    TaskExceptionHost->>CcuTaskException: ProcessCcuException
    Note right of CcuTaskException: Print baseInfo/groupRank/opData

    CcuTaskException->>CcuKernelMgr: InitChannelMap
    Note right of CcuKernelMgr: GetInstance→GetKernel→GetChannels, build channelId→handle mapping

    loop Traverse ccuExDetailInfo.missionInfo
        CcuTaskException->>CcuTaskException: PrintCcuErrorInfo
        CcuTaskException->>CcuTaskException: PrintPanicLogInfo
        Note right of CcuTaskException: Print CCUM DFX registers
    end

    CcuTaskException->>CcuComponent: CleanTaskKillState
    CcuTaskException->>CcuComponent: CleanDieCkes

    Note over CcuTaskException,hccp component: PrintCcuErrorInfo internal flow

    CcuTaskException->>CcuTaskException: GetCcuErrorMsg
    Note right of CcuTaskException: Generate CcuErrorInfo list

    CcuTaskException->>hccp component: HccpRaCustomChannel
    Note right of hccp component: GetCcuMissionContext, read MissionContext registers
    hccp component-->>CcuTaskException: missionContext

    CcuTaskException->>CcuKernelMgr: GetKernel
    CcuKernelMgr-->>CcuTaskException: kernel

    CcuTaskException->>CcuKernelMgr: GetRepByInstrId
    Note right of CcuKernelMgr: kernel is cast to CcuRepContext, find the Rep corresponding to the exception instruction
    CcuKernelMgr-->>CcuTaskException: rep

    CcuTaskException->>CcuTaskException: GenStatusInfo
    Note right of CcuTaskException: Generate Mission status info, report EI0002

    alt LOOPGROUP
        CcuTaskException->>CcuTaskException: GenErrorInfoLoopGroup
        Note right of CcuTaskException: Parse LoopGroup→traverse Loop→GenErrorInfoLoop
    else LOC_WAIT_EVENT
        CcuTaskException->>CcuTaskException: GenErrorInfoByRepType
        CcuTaskException->>CcuTaskException: Backtrack dependency Rep by mask bit
        Note right of CcuTaskException: Check CKE expected value vs actual value<br/>backtrack dependency Rep by unset bit
    else Other Rep (incl. LOC_WAIT_NOTIFY)
        CcuTaskException->>CcuTaskException: GenErrorInfoByRepType
        Note right of CcuTaskException: HANDLER_MAP dispatches to Read/Write/Notify and other handlers
    end

    CcuTaskException->>CcuTaskException: PrintCcuErrorLog
    Note right of CcuTaskException: Print error info by CcuErrorType

    alt UB error (missionStatus 0x01~0x05)
        CcuTaskException->>hccp component: PrintCcuUbRegisters→RaBatchQueryJettyStatus
        Note right of hccp component: Query Jetty status
        CcuTaskException->>hccp component: PrintUbRegisters→RaGetAuxInfo
        Note right of hccp component: RaCtxGetAuxInfo reads UB registers
        alt repType is READ/WRITE/BUF_READ/BUF_WRITE
            CcuTaskException->>CcuTaskException: GetCcuCqeErrorInfo
            Note right of CcuTaskException: ClusterMoniterGetCcuCqeErrInfo, report to cluster monitoring
        end
    end
```

---

## Interface Description (Class Diagram)

```mermaid
classDiagram
    class TaskExceptionHost {
        -mutex taskExceptionMutex_
        -unordered_set~u64~ CommRegisterMap_
        -bool hasAicpuReport_
        +GetInstance(s32) TaskExceptionHost*$
        +Register(u64) HcclResult
        +UnRegister(u64) HcclResult
        +ProcessCallback(rtExceptionInfo_t*) void$
        -Process(rtExceptionInfo_t*) void
        -HandleAicpuErrorReport(rtExceptionInfo_t*, ErrorMessageReport, TaskInfo) void
        -HandleHostErrorReport(rtExceptionInfo_t*, TaskInfo) void
        -ReportErrorMsg(TaskInfo, string, ErrorMessageReport, rtExceptionInfo_t*) void
        -GetGroupRankInfo(TaskInfo) string
        -ProcessException(rtExceptionInfo_t*, TaskInfo) void
        -PrintTaskContextInfo(uint32_t, uint32_t, uint32_t) void
        -PrintUbDfxInfo(rtExceptionInfo_t*, ErrorMessageReport) void
        -PrintGroupErrorMessage(...) void
        -PrintOpDataErrorMessage(...) void
        -PrintUbRegisters(s32, RdmaHandle) HcclResult
        -GetAicpuCqeErrInfo(...) void
        -ClusterMoniterGetAicpuCqeErrInfo(...) void
        -GetAicpuCqeErrRemoteLocalIdByRankId(CollComm, uint32_t, u32&) void
        -GetAicpuCqeErrNetInstanceByRankId(CollComm, uint32_t, string&) void
    }

    class CcuTaskException {
        +ProcessCcuException(rtExceptionInfo_t*, TaskInfo) void$
        -InitChannelMap(s32, u64) HcclResult$
        -GetCcuErrorMsg(deviceId, missionStatus, ParaCcu, vector~CcuErrorInfo~&) HcclResult$
        -GenErrorInfoByRepType(ErrorInfoBase, CcuRepBase*, vector~CcuErrorInfo~&) void$
        -GenErrorInfoLoop(ErrorInfoBase, CcuRepContext&, vector~CcuErrorInfo~&) HcclResult$
        -GenErrorInfoLoopGroup(ErrorInfoBase, CcuRepBase*, CcuRepContext&, vector~CcuErrorInfo~&) HcclResult$
        -GenStatusInfo(ErrorInfoBase, vector~CcuErrorInfo~&) void$
        -PrintCcuErrorInfo(...) void$
        -PrintCcuErrorLog(...) void$
        -PrintCcuUbRegisters(...) HcclResult$
        -GetCcuJettys(...) HcclResult$
        -PrintUbRegisters(s32, RdmaHandle) HcclResult$
        -GetCcuErrorMsgByType(...) string$
        -GetCcuCqeErrorInfo(...) void$
        -GetRankIdByChannelId(...) RankId$
        -GetAddrPairByChannelId(...) pair$
    }

    class HcclCommTaskExceptionLite {
        -bool stopCall_
        -u32 devId_
        -MirrorTaskManager* mirrorTaskManager_
        -unordered_map threadsPrinted_
        +GetInstance() HcclCommTaskExceptionLite&$
        +Init(u32) void
        +Call() void
        +PrintAllCommTaskException() HcclResult
        +PrintCommTaskException(CollCommAicpu) HcclResult
        -HandleExceptionCqe() HcclResult
        -GetThreadCqe(thread, cqeException, cqeStatus) HcclResult
        -ProcessCqe(aicpuComm, exceptionInfo, cqeStatus, aicpuCommInfo) HcclResult
        -ReportErrMsg(aicpuComm, exceptionInfo) HcclResult
        -GenerateErrorMessageReport(...) HcclResult
        -GenerateTaskErrMsg(...) void
        -FillNotifyErrMsg(...) void
        -FillReduceErrMsg(...) void
        -FillDmaErrMsg(...) void
        -FillSdmaErrMsg(...) void
        -FillUbErrMsg(...) void
        -FillReduceInlineErrMsg(...) void
        -SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo) HcclResult
        -SwitchUBCqeErrCodeToTsErrCode(u32) uint16_t
        -SwitchSdmaCqeErrCodeToTsErrCode(u32) uint16_t
        -PrintTaskExceptionBySqeId(...) HcclResult
        -PrintTaskContextInfo(...) HcclResult
        -CollectTaskContext(...) HcclResult
        -PrintEid(TaskInfo) void
        -GetGroupInfo(CollCommAicpu) string
        -GetSqeId(uint16_t, uint16_t) u32
    }

    class CcuErrorInfo {
        +CcuErrorType type
        +CcuRepType repType
        +uint8_t dieId
        +uint8_t missionId
        +uint16_t instrId
        +union msg
        +SetBaseInfo(...) void
    }

    class ErrorInfoBase {
        +int32_t deviceId
        +uint8_t dieId
        +uint8_t missionId
        +uint16_t currentInsId
        +uint16_t status
    }

    class CcuLoopContext {
        +GetCurrentIns() uint16_t
        +GetCurrentCnt() uint16_t
        +GetAddrStride() uint32_t
    }

    class CcuMissionContext {
        +GetStatus() uint16_t
        +GetCurrentIns() uint16_t
        +GetStartIns() uint16_t
        +GetEndIns() uint16_t
    }

    class CcuComponent {
        +GetInstance(devLogicId) CcuComponent$
        +CleanTaskKillState() HcclResult
        +CleanDieCkes(dieId) HcclResult
    }

    TaskExceptionHost ..> CcuTaskException : dispatched when TASK_CCU
    HcclCommTaskExceptionLite ..> TaskExceptionHost : HDC channel reporting
    HcclCommTaskExceptionLite ..> hal component : dlHalEschedSubmitEvent
    TaskExceptionHost ..> hccp component : RaGetAuxInfo/HccpRaCustomChannel
    TaskExceptionHost ..> error_manager component : RPT_INPUT_ERR
    CcuTaskException ..> CcuComponent : cleanup status
    CcuTaskException ..> hccp component : HccpRaCustomChannel/RaGetAuxInfo
    CcuTaskException ..> CcuErrorInfo : generates
    CcuTaskException ..> ErrorInfoBase : uses
    CcuTaskException ..> CcuLoopContext : reads registers
    CcuTaskException ..> CcuMissionContext : reads registers
```

---

## Interface Description

### TaskExceptionHost

| Interface | Type | Parameters | Return Value | Description |
|------|------|------|--------|----------|
| `GetInstance(s32)` | Public static | [in] deviceLogicID | `TaskExceptionHost*` | Gets the exception handler for the specified device. Supports up to 65 devices. |
| `Register(u64)` | Public | [in] commHandle | `HcclResult` | Writes commHandle to CommRegisterMap_. The first comm registration calls `aclrtExceptionInfoCallbackRegister(ProcessCallback)` to register the exception callback with RTS. |
| `UnRegister(u64)` | Public | [in] commHandle | `HcclResult` | Removes commHandle from CommRegisterMap_. When the last comm is unregistered, calls `aclrtExceptionInfoCallbackUnregister(ProcessCallback)` to unregister the callback with RTS. |
| `ProcessCallback(rtExceptionInfo_t*)` | Public static | [in] exceptionInfo | void | Runtime exception callback entry point. Obtains the handler through GetInstance and forwards to Process. |
| `HandleAicpuErrorReport(rtExceptionInfo_t*, const ErrorMessageReport&, const TaskInfo&)` | Private | [in] exceptionInfo, [in] errorMessage, [in] taskInfo | void | Handles errors already reported by the AICPU side: prints BaseInfo/ParaInfo/GroupInfo/OpDataInfo, calls PrintUbDfxInfo, ReportErrorMsg. When ubCqeStatus is non-zero, calls GetAicpuCqeErrInfo. |
| `HandleHostErrorReport(rtExceptionInfo_t*, const TaskInfo&)` | Private | [in] exceptionInfo, [in] taskInfo | void | Host-side independent exception handling: prints preceding task context for TASK_NOTIFY_WAIT and reports EI0002, prints cluster monitoring error info. |
| `ReportErrorMsg(const TaskInfo&, const string&, const ErrorMessageReport&, rtExceptionInfo_t*)` | Private | [in] exceptionTaskInfo, [in] groupRankContent, [in] errorMessage, [in] exceptionInfo | void | Reports different error codes based on task type: TASK_NOTIFY_WAIT→EI0002, UB/Write type→EI0018. |
| `ProcessException(rtExceptionInfo_t*, const TaskInfo&)` | Private | [in] exceptionInfo, [in] taskInfo | void | General exception handling entry point: checks hasAicpuReport_, obtains ErrorMessageReport through CollComm::GetAicpuTaskException if not reported, dispatches to HandleAicpuErrorReport or HandleHostErrorReport. |
| `PrintTaskContextInfo(uint32_t, uint32_t, uint32_t)` | Private | [in] deviceId, [in] streamId, [in] taskId | void | Prints up to 50 preceding task context entries before the exception task. |
| `PrintUbDfxInfo(rtExceptionInfo_t*, const ErrorMessageReport&)` | Private | [in] exceptionInfo, [in] errorMessage | void | For UB type tasks (TASK_WRITE_WITH_NOTIFY/TASK_UB, and so on), prints UB CQE status and EID information, and reads UB DFX registers. |
| `PrintUbRegisters(s32, RdmaHandle)` | Private | [in] devLogicId, [in] rdmaHandle | `HcclResult` | Reads and prints UB DFX register information through RaGetAuxInfo. |
| `GetGroupRankInfo(const TaskInfo&)` | Private | [in] taskInfo | string | Gets group/rankSize/rankId information from TaskInfo. |
| `GetAicpuCqeErrRemoteLocalIdByRankId(CollComm*, uint32_t, u32&)` | Private | [in] collComm, [in] rankid, [out] remoteLocalId | void | Gets the LocalId corresponding to the remote rank through RankGraph. |
| `GetAicpuCqeErrNetInstanceByRankId(CollComm*, uint32_t, string&)` | Private | [in] collComm, [in] rankid, [out] netInstanceId | void | Gets the NetInstanceId corresponding to the remote rank through RankGraph. |

### CcuTaskException

| Interface | Type | Parameters | Return Value | Description |
|------|------|------|--------|----------|
| `ProcessCcuException(const rtExceptionInfo_t*, const TaskInfo&)` | Public static | [in] exceptionInfo, [in] taskInfo | void | Main entry point for CCU task exception handling: initializes channel mapping, traverses Missions, prints errors and registers, cleans up TaskKill status. |
| `InitChannelMap(s32, u64)` | Private static | [in] deviceId, [in] ccuKernelHandle | `HcclResult` | Initializes the global channelId→channelHandle mapping `g_channelIdToHandle`. |
| `GetCcuErrorMsg(int32_t, uint16_t, const ParaCcu&, vector<CcuErrorInfo>&)` | Private static | [in] deviceId, [in] missionStatus, [in] ccuTaskParam, [out] errorInfo | `HcclResult` | Core method for obtaining CCU error information: reads MissionContext, finds the exception Rep, dispatches to different GenErrorInfo methods. |
| `GenErrorInfoByRepType(const ErrorInfoBase&, shared_ptr<CcuRepBase>, vector<CcuErrorInfo>&)` | Private static | [in] baseInfo, [in] repBase, [out] errorInfo | void | Dispatches to the corresponding GenErrorInfo method based on CcuRepType (uses HANDLER_MAP function table). |
| `GenErrorInfoLoop(const ErrorInfoBase&, CcuRepContext&, vector<CcuErrorInfo>&)` | Private static | [in] baseInfo, [in] ctx, [out] errorInfo | `HcclResult` | Parses Loop type exceptions: reads LoopContext registers, recursively parses Reps within the Loop. |
| `GenErrorInfoLoopGroup(const ErrorInfoBase&, shared_ptr<CcuRepBase>, CcuRepContext&, vector<CcuErrorInfo>&)` | Private static | [in] baseInfo, [in] repBase, [in] ctx, [out] errorInfo | `HcclResult` | Parses LoopGroup type exceptions: expands all Loops and parses them individually. |
| `PrintCcuUbRegisters(const vector<CcuErrorInfo>&, s32, const TaskInfo&)` | Private static | [in] errorInfos, [in] devLogicId, [in] taskInfo | `HcclResult` | Gets CCU Jetty status and prints UB registers for error Jetties. |
| `GetCcuJettys(const CcuErrorInfo&, pair<CcuChannelInfo, vector<CcuJetty*>>&)` | Private static | [in] errorInfo, [out] ctx | `HcclResult` | Gets channelId from CcuErrorInfo and obtains Jetties through the channel→endpoint→ctxPool chain. |
| `GetCcuCqeErrorInfo(const CcuErrorInfo&, const TaskInfo&, u32, uint8_t)` | Private static | [in] ccuErrorInfo, [in] taskInfo, [in] locDeviceId, [in] missionStatus | void | Gets remoteRankId and NetInstanceId through channelId, calls ClusterMoniterGetCcuCqeErrInfo to report to cluster monitoring. |

### HcclCommTaskExceptionLite

| Interface | Type | Parameters | Return Value | Description |
|------|------|------|--------|----------|
| `GetInstance()` | Public static | None | `HcclCommTaskExceptionLite&` | Gets the singleton instance. |
| `Init(u32)` | Public | [in] devId | void | Initializes the device ID. |
| `Call()` | Public | None | void | Daemon thread callback entry point. Calls HandleExceptionCqe. Sets stopCall_ on failure to prevent flooding. |
| `HandleExceptionCqe()` | Private | None | `HcclResult` | Traverses all threads in all communication domains, detects and handles CQE exceptions. |
| `GetThreadCqe(Thread*, rtLogicCqReport_t&, CqeStatus&)` | Private | [in] thread, [out] cqeException, [out] cqeStatus | `HcclResult` | Gets CQE exception information for the specified thread through CqReportRecv. |
| `ProcessCqe(CollCommAicpu*, const rtLogicCqReport_t&, const CqeStatus&, const vector<pair<string,CollCommAicpuMgr*>>&)` | Private | [in] aicpuComm, [in] exceptionInfo, [in] cqeStatus, [in] aicpuCommInfo | `HcclResult` | Processes CQE exceptions: prints TaskException, reports to Host, prints all communication domain information on NotifyWait timeout. |
| `ReportErrMsg(CollCommAicpu*, const rtLogicCqReport_t&)` | Private | [in] aicpuComm, [in] exceptionInfo | `HcclResult` | Checks IsErrorReported. If not reported, generates ErrorMessageReport, reports to Host through HDC, notifies TSFW through Mailbox, and sets ErrorReported(true). |
| `GenerateErrorMessageReport(CollCommAicpu*, const TaskInfo&, const rtLogicCqReport_t&, ErrorMessageReport&)` | Private | [in] aicpuComm, [in] taskInfo, [in] exceptionInfo, [out] errMsgInfo | `HcclResult` | Fills common fields of ErrorMessageReport based on task information, then calls GenerateTaskErrMsg to fill task-type-specific fields. |
| `GenerateTaskErrMsg(const TaskInfo&, ErrorMessageReport&, const rtLogicCqReport_t&)` | Private | [in] taskInfo, [out] errMsgInfo, [in] exceptionInfo | void | Dispatches to FillNotifyErrMsg/FillReduceErrMsg/FillDmaErrMsg/FillUbErrMsg/FillSdmaErrMsg/FillReduceInlineErrMsg based on taskType. |
| `FillNotifyErrMsg(const TaskInfo&, ErrorMessageReport&)` | Private | [in] taskInfo, [out] errMsgInfo | void | Fills notifyId and notifyValue for NOTIFY_WAIT/NOTIFY_RECORD types. |
| `FillReduceErrMsg(const TaskInfo&, ErrorMessageReport&, const rtLogicCqReport_t&)` | Private | [in] taskInfo, [out] errMsgInfo, [in] exceptionInfo | void | Fills reduceOp, notifyId, locEid, rmtEid, ubCqeStatus, and so on for UB_REDUCE_INLINE/WRITE_REDUCE_WITH_NOTIFY types. |
| `FillDmaErrMsg(const TaskInfo&, ErrorMessageReport&, const rtLogicCqReport_t&)` | Private | [in] taskInfo, [out] errMsgInfo, [in] exceptionInfo | void | Fills for UB_INLINE_WRITE/WRITE_WITH_NOTIFY types. Internally calls FillUbErrMsg. |
| `FillUbErrMsg(const TaskInfo&, ErrorMessageReport&, const rtLogicCqReport_t&)` | Private | [in] taskInfo, [out] errMsgInfo, [in] exceptionInfo | void | Fills locEid, rmtEid, ubCqeStatus, linkType, size, and so on for UB types. |
| `FillSdmaErrMsg(const TaskInfo&, ErrorMessageReport&)` | Private | [in] taskInfo, [out] errMsgInfo | void | Fills linkType, size, srcAddr, dstAddr for SDMA types. |
| `FillReduceInlineErrMsg(const TaskInfo&, ErrorMessageReport&)` | Private | [in] taskInfo, [out] errMsgInfo | void | Fills reduceOp for REDUCE_INLINE types. |
| `SendTaskExceptionByMBox(u32, u32, const rtLogicCqReport_t&)` | Private | [in] notifyId, [in] tsId, [in] exceptionInfo | `HcclResult` | Reports task exception events to TSFW through Mailbox, including error code conversion (UB/SDMA error codes to TS error codes). |
| `SwitchUBCqeErrCodeToTsErrCode(u32)` | Private | [in] cqeErrCode | `uint16_t` | Converts UB CQE error codes to TS error codes. |
| `SwitchSdmaCqeErrCodeToTsErrCode(u32)` | Private | [in] cqeErrCode | `uint16_t` | Converts SDMA CQE error codes to TS error codes. |
| `PrintAllCommTaskException()` | Public | None | `HcclResult` | Prints all Task exception information for all communication domains. |
| `PrintCommTaskException(CollCommAicpu*)` | Public | [in] aicpuComm | `HcclResult` | Prints Task exception information for all threads in the specified communication domain. |
| `PrintTaskExceptionBySqeId(CollCommAicpu*, u32, u32)` | Private | [in] aicpuComm, [in] sqId, [in] sqeId | `HcclResult` | Prints Task exception information for the specified sqId/sqeId: BaseInfo/ParaInfo, EID, GroupInfo, OpData/TaskContext. |
| `PrintTaskContextInfo(CollCommAicpu*, u32, u32)` | Private | [in] aicpuComm, [in] sqId, [in] taskId | `HcclResult` | Prints up to 50 preceding task context entries before the exception task (printed in segments by opIndex). |
| `CollectTaskContext(CollCommAicpu*, u32, u32, vector<TaskInfo*>&)` | Private | [in] aicpuComm, [in] sqId, [in] taskId, [out] taskContext | `HcclResult` | Collects up to 50 task entries before the exception task from the MirrorTaskManagerLite queue. |
| `PrintEid(const TaskInfo&)` | Private | [in] taskInfo | void | Prints localEid and remoteEid for UB type tasks. |
| `GetGroupInfo(CollCommAicpu*)` | Private | [in] aicpuComm | string | Gets the group/rankSize/localRank information of the communication domain. |
| `GetSqeId(uint16_t, uint16_t)` | Private | [in] taskId, [in] streamId | `u32` | Combines taskId and streamId into sqeId. |

### Global Callback Registration Interfaces

| Interface | Parameters | Description |
|------|------|----------|
| `RegisterGetAicpuCqeErrInfoCallBackHcomm(callback)` | AICPU CQE error info callback | Registers the callback for AICPU CQE error information reporting to cluster monitoring. |
| `RegisterAicpuGetErrStatusVecCallBack(callback)` | AICPU error status vector callback | Registers the callback for getting the AICPU-side abnormal device status list. |
| `RegisterGetCcuCqeErrInfoCallBackHcomm(callback)` | CCU CQE error info callback | Registers the callback for CCU CQE error information reporting to cluster monitoring. |
| `RegisterCcuGetErrStatusVecCallBack(callback)` | CCU error status vector callback | Registers the callback for getting the CCU-side abnormal device status list. |

---

## Usage Limitations

### Supported Scenarios

| Chip | Mode | Host Side | AICPU Side | Description |
|------|------|---------|----------|------|
| Ascend 950PR/Ascend 950DT | AICPU | Supported | Supported | Indop tasks use the new flow; non-Indop tasks fall back to legacy TaskExceptionHandler. |
| Ascend 950PR/Ascend 950DT | CCU | Supported | Not applicable | TASK_CCU type is handled by CcuTaskException. |
| Ascend 950PR/Ascend 950DT | MC2 | Fall back to legacy | Not applicable | Falls back to TaskExceptionHandler under the legacy directory. |

### Constraint Specifications

1. **Device count limit**: Supports up to 65 devices (`MAX_MODULE_DEVICE_NUM_V2 = 65`). `TaskExceptionHostManager::GetHandler` returns nullptr when devId >= 65.
2. **CCU message length**: CCU transfer length must not exceed 256MB (`CCU_MSG_256MB_LEN`). A warning is printed when exceeded.
3. **AICPU duplicate report prevention**: The Host side uses the `hasAicpuReport_` flag to prevent the same TaskExceptionHost from repeatedly handling AICPU-reported errors. The AICPU side uses the `CollCommAicpu::IsErrorReported()` flag to prevent duplicate reporting from the same communication domain.
4. **Host-side comm registration management**: Uses `CommRegisterMap_` to manage commHandle registration. The Process callback validates whether the commHandle is registered. The RTS callback is cleared when the last comm is unregistered.
5. **AICPU-side flooding prevention**: `HcclCommTaskExceptionLite` sets `stopCall_=true` after `HandleExceptionCqe` fails, stopping subsequent calls.
6. **CQE error information retrieval**: The CCU side uses the `isGetCqeErrInfo` atomic flag to avoid redundant CQE error information retrieval.
7. **Cluster monitoring report limit**: The abnormal device information list is limited to up to 3 entries (`maxListSize = 3`).
8. **Task context print limit**: Prints up to 50 task entries before the exception task (`TASK_CONTEXT_SIZE = 50`). The single print length does not exceed `TASK_CONTEXT_INFO_SIZE`.
9. **CCU Mission count**: Currently `ccuMissionNum` is 1, only handling a single Mission exception.
10. **CCU instruction backtracking limit**: Prints up to 10 instructions before the error instruction (`loopUpInstrNum = 10`).
11. **CCU Loop expansion**: Instructions within a Loop are recursively expanded through `GenErrorInfoLoop`. LoopGroup expands all Loops through `GenErrorInfoLoopGroup`.
12. **Error code conversion**: AICPU-side UB and SDMA error codes must be converted to TS error codes before reporting through Mailbox. Unrecognized error codes are uniformly converted to `TS_ERROR_HCCL_OTHER_ERROR`.
13. **Thread safety**: `g_communicatorCallbackMapV2` and `g_channelIdToHandle` are both protected by mutexes for concurrent access.
