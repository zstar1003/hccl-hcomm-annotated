# HCCL 模拟运行器数据模型设计

## 1. 软硬件资源交互关系建模

数据流交互示意图。

```mermaid
sequenceDiagram
    participant HostThread as Host CPU (Runner)
    participant StreamQueue as Stream (Memory Queue)
    participant DeviceScheduler as Device Hardware (TS)
    participant EventMem as Event Status (Memory)

    Note over HostThread: 1. aclrtRecordEvent(evt1)
    HostThread->>StreamQueue: Push CMD: [Write Event1=Done]
  
    Note over HostThread: 2. aclrtStreamWaitEvent(evt1)
    HostThread->>StreamQueue: Push CMD: [Wait Event1==Done]
  
    Note over HostThread: 3. aclrtLaunchKernel(MatMul)
    HostThread->>StreamQueue: Push CMD: [Execute MatMul]
  
    Note over DeviceScheduler: 异步执行阶段 (Device侧)
  
    StreamQueue->>DeviceScheduler: Pop CMD: [Write Event1]
    DeviceScheduler->>EventMem: Update Status to DONE
  
    StreamQueue->>DeviceScheduler: Pop CMD: [Wait Event1]
    DeviceScheduler->>EventMem: Check Status?
    Note right of DeviceScheduler: 发现是 DONE，通过！<br/>(如果是 NotReady，硬件会在这里空转等待)
  
    StreamQueue->>DeviceScheduler: Pop CMD: [MatMul]
    DeviceScheduler->>DeviceScheduler: Start AI Core Computing...
```

只要是塞进 Stream 里的东西，都是由 Device 硬件执行的。

## 2. Device、Context、Stream 等硬件资源关系建模

Device、Context、Stream 与用户主机线程之间的关系。

```mermaid
graph TD
    subgraph Server1[Server 1]
        Host1
        Device
        Device2
    end

    subgraph Server2[Server 2]
        Host3
        Host2
        Device3
        Device4
      
    end

    subgraph Host1[Host1]
        Runner1[runner<br>用户线程1]
        Runner2[runner<br>用户线程2]
    end

    subgraph Host2[Host2]
        Runner3[Runner...]
    end

    subgraph Host3[Device CPU<br>边缘计算/嵌入式: atlas 500]
        embedRunner[Embed Runner...]
    end

    subgraph Device[Device 1]
        Context1
    end

    subgraph Context1[run-Context1]
        Stream1
    end

    subgraph Stream1[ctx-Stream1]
        TaskKernel
    end

    subgraph TaskKernel[Task/Kernel]

    end

    subgraph Device2[Device 2]
        Context2
    end

    subgraph Context2[run-Context2]
        Stream2
    end

    subgraph Stream2[ctx-Stream2]
        Kernel
    end

    subgraph Kernel[Kernel]
    end

    subgraph Device3[Device 3]
        ctxM[...]
    end

    subgraph Device4[Device 4]
        ctxN[...]
    end

    Runner1-->Device
    Runner2-->Device
    Runner2-->Device2
    Runner3-->Device3
    Runner3-.->|???sdid|Device
    embedRunner-->Device4
 
```

### 2.1 基础设备关系建模

#### 2.1.1 层次结构总览

```mermaid
erDiagram
    %% ==========================================
    %% 第一层：物理拓扑层 (Server -> Host/Device)
    %% ==========================================
    Server {
        typ server-id PK
        typ pod-id
        typ version
    }
    Host {
        typ host-id PK
        typ server-id FK
        typ ip
        typ arch
    }
    Device {
        typ device-id PK
        typ server-id FK
        typ logic-id "当前可用设备的序号"
        typ physical-id
        typ ccu-die-num "910D目前双die"
        typ overflow-mode
        typ status
        typ soc-version "A3"
        typ max-stream-cnt "1984"
    }
    Server ||--|{ Host : contains
    Server ||--o{ Device : contains

    %% ==========================================
    %% 第二层：进程与上下文层 (Runner -> Context -> Stream)
    %% ==========================================
    Runner {
        typ run-id PK
        typ host-id FK
        typ pid
        typ timeout-config-ms
        typ current-ctx-id FK
    }
    Context {
        typ ctx-id PK
        typ run-id FK
        typ thread-id
        typ device-id FK
        typ is-default
        typ float-overflow-addr
        typ capture-mode
    }
    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ sq-base-addr
        typ is-primary-default
        typ is-other-default
        typ priority
        typ schedule-strategy
        typ failure-mode
        typ user-tag
        typ overflow-switch
        typ activated
        typ capture-status
        typ task-complete-status
    }
    Host ||--o{ Runner : runs
    Runner ||--o{ Context : "creates/owns"
    Runner |o--o{ Context : "current activates"
    Context }o--|| Device : binds
    Context ||--|{ Stream : owns
    Device ||..|{ Stream : "硬件约束"

    %% ==========================================
    %% 第三层：设备内部资源层 (Port/EndPoint/Ccu)
    %% ==========================================
    Port {
        typ port-id PK
        typ device-id FK
        typ die-id "die Id"
        typ status "0:未使用/1:使用"
        typ name "0/0, 0/1"
    }
    Rank {
        typ id PK
        typ device-id FK
        typ rank-id
        typ commId
    }
    EndPoint {
        typ endpoint-id PK
        typ rank-id FK
        typ addr "IP地址/EID"
        typ type "IPV4/IPV6/EID"
    }
    Ccu {
        typ ccu-id PK
        typ device-id FK
        typ resource-addr
        type die-id
        typ status
    }
    DeviceConnection {
        typ connection-id PK
        typ src-dev-id FK
        typ dst-dev-id FK
        typ link-type
        typ access-by-remote
    }
    Device ||--o{ Port : "has"
    Device ||--o{ Rank : "has"
    Device ||--o{ EndPoint : "has"
    Device ||--o{ Ccu : "1:2双die"
    Device ||--o{ DeviceConnection : "peer access"
```

#### 2.1.2 网络通信资源

```mermaid
erDiagram
    %% ==========================================
    %% 网络拓扑与连接层
    %% ==========================================
    EndPoint {
        typ endpoint-id PK
        typ rank-id FK
        typ func_id "ccu 用到"
        typ addr "IP地址/EID"
        typ type "IPV4/IPV6/EID"
    }
    Port {
        typ port-id PK
        typ device-id FK
        typ die-id "die Id"
        typ name "0/0, 0/1"
    }
    EndPoint-Port-Mapping {
        typ mapping-id PK
        typ port-id FK
        typ endpoint-id FK
    }
    %% topo.json定义的物理连接
    Link {
        typ link-id PK
        typ local-endpoint-id FK
        typ remote-endpoint-id FK
        typ net-layer "网络层"
        typ type "连接类型"
    }
    Link-Protocol-Mapping {
        typ link-protocol-mapping-id PK
        typ link-id FK
        typ protocol "UB_CTP/UB_MEM/..."
    }
    %% 
    EndPoint-Pair {
        typ endpoint-pair-id PK
        typ local-endpoint-id FK
        typ remote-endpoint-id FK
    }
    CcuChannel {
        typ ccu-channel-id PK
        typ channel-id FK   "业务分配"
        typ local-endpoint-id FK
        typ remote-endpoint-id FK
        typ protocol "通信协议"
        typ src-jetty-start "jetty起始Id"
        typ jetty-num "jetty数量"
    }

    EndPoint-Port-Mapping }|--|| Port : maps
    EndPoint-Port-Mapping }|--|| EndPoint : maps
    Link ||--o{ EndPoint : connects
    Link ||--|{ Link-Protocol-Mapping : "supports"
    EndPoint-Pair ||--|{ EndPoint : mapping
    EndPoint-Pair }o..|| Link : "based-on"
    CcuChannel ||--|{ EndPoint : uses
    CcuChannel }o..|| Link : "based-on"
```

#### 2.1.3 关键关系说明

##### 第一层：物理拓扑层

| 关系 | 含义 | 说明/注意事项 |
| --- | --- | --- |
| Server → Host | 一个Server可能包含多个 Host 实例 | 比如多路 CPU 或虚拟化环境 |
| Server → Device | 一个Server包含多个 AI 设备 | 对应 /dev/davinci0, /dev/davinci1 等 |

##### 第二层：进程与上下文层

| 关系 | 含义 | 说明/注意事项 |
| --- | --- | --- |
| Host → Runner | 每个主机上有多个应用线程（Runner） | 每个 Runner 可以创建多个 Context |
| Runner → Context | 线程创建或切换到不同的 Context | 使用 aclrtCreateContext() 和 aclrtSetCurrentContext() |
| Context → Device | Context 绑定 Device | 一旦创建后不可跨设备 |
| Context → Stream | 每个 Context 可创建多个 Stream | 对应 aclrtCreateStream() |
| Runner ↔ Context (current) | 当前上下文激活状态 | aclrtGetCurrentContext()、aclrtSetCurrentContext() |
| Device .. Stream | 资源上限约束关系 | Stream 数量受硬件限制（max-stream-cnt） |

##### 第三层：设备内部资源层

| 关系 | 含义 | 说明/注意事项 |
| --- | --- | --- |
| Device → Port | 设备包含多个通信端口 | 用于网络拓扑连接，格式如 "0/0, 0/1" |
| Device → Rank | 设备关联通信Rank | Rank是通信域中的参与节点标识 |
| Device → EndPoint | 设备包含多个端点 | IP地址/EID寻址标识，用于网络通信 |
| Device → Ccu | 设备包含多个CCU单元 | 910D为双die架构，每个die一个CCU |
| Device → DeviceConnection | 设备间通信通道 | aclrtDeviceCanAccessPeer(), aclrtDeviceEnablePeerAccess() |

##### 网络通信资源层

| 关系 | 含义 | 说明/注意事项 |
| --- | --- | --- |
| EndPoint ↔ Port | 端点与端口映射 | 通过 EndPoint-Port-Mapping 实现多对多映射 |
| Link → EndPoint | 物理连接关联端点 | 由 topo.json 定义，描述物理拓扑 |
| Link → Link-Protocol-Mapping | 连接支持的协议 | 一个Link可支持多种协议（UB_CTP/UB_MEM等） |

#### 2.1.4 配套接口映射表

##### 基础与设备层 (Device / Server)

| 实体/属性                   | 关键API接口                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| Server/Host                 | `aclInit`, `aclFinalize`                                                                             |
| Runner.pid                  | `rtDeviceGetBareTgid`                                                                                |
| Device                      | `rtGetDeviceCount`                                                                                   |
| Device.logic-id             | `rtSetDevice`,`rtResetDevice`,`rtGetDevice`,`rtsGetLogicDevIdByPhyDevId`                             |
| Device.physical-id          | `rtGetPhyDevIdByLogicDevId`                                                                          |
| Device.soc-name             | `rtGetSocName`                                                                                       |
| Device.overflow-mode        | `rtSetDeviceSatMode`,`rtGetDeviceSatMode`                                                            |
| DeviceConnection            | `rtGetDevicesTopo`,`rtDeviceDisablePeerAccess`,`rtDeviceEnablePeerAccess`,`rtDevicePeerAccessStatus` |
| Context.context-id          | `rtCreateContext`,`rtDestroyContext`                                                                 |
| Context.is-default          | `rtSetCurrentContext`,`rtGetCurrentContext`                                                          |
| Context.float-overflow-addr | `rtCtxGetFloatOverflowAddr`                                                                          |
| Stream表                    | `aclrtGetStreamAvailableNum`                                                                         |
| Stream.stream-id            | `rtCreateStream`,`rtCreateStreamWithConfig`,`rtDestroyStream`,`rtDestroyStreamForce`                 |
| Stream.task-complete-status | `rtSynchronizeStream`, `rtSynchronizeStreamWithTimeout`                                              |
| Stream.activated            | `rtStreamStop`                                                                                       |
| Stream.failure-mode         | `rtSetStreamAttribute`,`rtGetStreamAttribute`                                                        |

### 2.2 基础内存管理关系建模

```mermaid
erDiagram
    PhyMemBlock ||--o{ VirtualPointerTable : "物理到虚拟映射"
    PhyMemBlock ||--o{ FdMemRecord : "文件描述符映射"
    PhyMemBlock {
        typ phy-mem-id PK "自增ID"
        typ device-id FK "0,1...或 -1(host)"
        typ size
        typ type 
        typ ref-count
    }

    VirtualPointerTable {
        typ start-ptr PK
        typ size
        typ context-id FK
        typ phy-mem-id FK
        typ owner-pid  "创建进程"
        typ source-type ""
        typ policy
    }

    VirtualPointerTable ||--o{ IpcMemRecord : "共享内存注册"
    IpcMemRecord {
        typ ipc-id PK
        typ vir_mem_id FK
        typ phy_mem_id FK
        typ name-or-key
        typ create-pid
    }

    IpcMemRecord ||--o{ IpcMemWhiteList : "进程白名单"
    IpcMemWhiteList {
        typ ipc-id FK
        typ pid
        typ create-pid
    }

    FdMemRecord {
        typ fd PK
        typ phy-mem-id FK
        typ name
        typ type
    }

    FdMemRecord ||--o{ FdMemWhiteList : "进程白名单"
    FdMemWhiteList {
        typ fd-id FK
        typ pid
        typ create-pid
    }

    %%VirtualPointerTable ||--o{ MemMapRecord : "映射关系"
    %%MemMapRecord {
    %%    typ ptr FK
    %%    typ phy-mem-id FK
    %%}
```

#### 2.2.1 关键关系说明（基础内存管理关系建模）

1. **物理内存核心地位**。
   `PhyMemBlock`作为基础实体，通过`phy_mem_id`与所有其他实体关联，体现华为昇腾"物理内存池化"的设计理念[3]。
2. **三层映射体系**：

   - 物理→虚拟（`VirtualPointerTable`）
   - 物理→IPC共享（`IpcMemRecord`）
   - 物理→文件描述符（`fdMemRecord`）
3. **安全控制**：
   `IpcMemWhiteList`通过进程PID白名单机制实现华为HCCS（Huawei Collective Communication Service）的安全共享[3]。
4. **特殊映射类型**：
   `MemMapRecord`记录双虚拟地址映射场景（如`aclrtMapMem`产生的映射），支持华为NPU的零拷贝数据传输[3]。

#### 2.1.2 配套接口映射表（基础内存管理关系建模）

| 实体                           | 关键管理接口                                                                                                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| PhyMemBlock                    | `rtMallocPhysical`, `rtFreePhysical`                                                                                                            |
| VirtualPointerTable            | `rtMallocWithCfg`,`rtMallocForTaskScheduler`,`rtMallocHostWithCfg`,`rtFree`,`rtReserveMemAddress`,`ReleaseMemAddress`, `rtMapMem`, `rtUnmapMem` |
| VirtualPointerTable.context-id | `rtPointerGetAttributes`                                                                                                                        |
| FdMemRecord.fd                 | `rtMemExportToShareableHandle`, `rtMemImportFromShareableHandle`                                                                                |
| FdMemWhiteList.pid             | `rtMemSetPidToShareableHandle`                                                                                                                  |
| IpcMemRecord.name-or-key       | `rtIpcMemGetExportKey`                                                                                                                          |
| IpcMemRecord.ipc-id            | `rtIpcMemImportByKey`,`IpcMemClose`                                                                                                             |
| IpcMemWhiteList.pid            | `rtIpcMemSetImportPid`                                                                                                                          |
| MemMapRecord                   | `rtHostRegister`, `rtHostUnRegister`                                                                                                            |

## 3. 在基础模型上扩展数据任务模型

### 3.1 数据 / 任务流建模

```mermaid
erDiagram
    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ task-id PK
        typ stream-id FK
        typ seq-number "stream内自增"
        typ type "Kernel/Memcpy/Callback"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    MemcpyTask {
        typ task-id FK
        typ src-addr
        typ dst-addr
        typ size
    }

    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ MemcpyTask : "is a"

    %% MemcpyTask 的地址应该在 VirtualPointerTable 可寻址
    MemcpyTask }o..|{ VirtualPointerTable : "Range Constraint"
    VirtualPointerTable {
        typ start-ptr PK
        typ ctx-id FK
    }

    VirtualPointerTable }o--|| Context : "belongs to"

```

### 3.2 ccu资源建模

一个NPU device包含2个CCU，分别为die0和die1。

```mermaid
graph RL
    subgraph DavidDevice0[David 0]
        direction RL
        Memory0[Memory]
        David0Die0[Die0_ccu]
        David0Die1[Die1_ccu]
    end

    subgraph David0Die0[Die0_ccu]

        CcuBuf00[CcuBuf]
        Variable00[Variable]
        Notify00[Notify]
        CompletedEvent00[CompletedEvent]
        Local/Rmt-Addr00[Local/Rmt-Addr]
    end

    subgraph David0Die1[Die1_ccu]

        CcuBuf01[CcuBuf]
        Variable01[Variable]
        Notify01[Notify]
        CompletedEvent01[CompletedEvent]
        Local/Rmt-Addr01[Local/Rmt-Addr]
    end

    David0Die0---Memory0
    David0Die1---Memory0

```

#### 3.2.1 配套接口映射表（ccu资源建模）

| 实体               | 关键管理接口                                             |
| ------------------ | -------------------------------------------------------- |
| CcuBuf             | `rtCcuBufAlloc`, `rtCcuBufFree`, `rtCcuBufGetAddr`       |
| Variable           | `rtVariableCreate`, `rtVariableDestroy`, `rtVariableSet` |
| Notify             | `rtCreateNotify`, `rtDestroyNotify`                      |
| CompletedEvent     | `rtCreateEvent`, `rtDestroyEvent`                        |
| Local/Rmt-Addr     | `rtGetDeviceLocalAddr`, `rtGetDeviceRemoteAddr`          |
| CCU资源查询        | `rtGetCcudieInfo`, `rtGetCcudieNum`                      |

#### 3.2.2 CCU资源生命周期说明

**CCU初始化流程**：

1. Device启动时，两个CCU（die0/die1）自动初始化。
2. 每个CCU分配独立的CcuBuf、Variable、Notify资源池。
3. CompletedEvent用于通知任务完成状态。

**资源约束**：

- 每个CCU的CcuBuf数量有限（与Device.soc-version相关）
- Variable用于存储通信过程中的共享变量。
- Notify用于跨CCU的同步通知机制。
- Local/Rmt-Addr用于跨die通信时的地址转换。

### 3.3 异步 / 同步执行建模

#### 3.3.1 [Notify资源管理](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/appdevg/acldevg/aclcppdevg_000524.html)

```mermaid
erDiagram
    Device ||..o{ Notify : "Hardware Limit"
    Device ||--o{ Context : "referred by"
    Device {
        typ device-id PK
        typ device-type "A3"
        typ max-notify-cnt "8192"
    }
    Context {
        typ ctx-id PK
        typ device-id FK
    }

    Notify ||--o| IpcNotify : "is a"
    Notify o|--|| Context : "record"
    Notify {
        typ notify-id PK
        typ create-ctx-id FK
        typ device-notify-seq "0~8191"
        typ value "notify读写寄存器"
    }

    IpcNotify {
        typ ipc-id PK
        typ notify-id FK
        typ name-or-key
        typ create-pid
    }

    IpcNotify ||--o{ IpcNotifyVistorList : "has"
    IpcNotifyVistorList {
        typ ipc-id FK
        typ vistor-pid
    }
```

##### 关键关系说明（[Notify资源管理](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/appdevg/acldevg/aclcppdevg_000524.html)）

**Notify与Device的硬件约束**：

- 每个Device的Notify数量上限为`max-notify-cnt`（如A3芯片为8192）
- `device-notify-seq`是Notify在Device内的物理序号（0~8191）
- Notify创建时必须指定所属Context，Context绑定到特定Device。

**Notify的IPC共享机制**：

- `IpcNotify`允许跨进程共享Notify实例。
- `name-or-key`是共享标识，通过`rtNotifyGetExportKey`获取。
- 其他进程通过`rtNotifyImportByKey`导入并使用。
- `IpcNotifyVistorList`记录有权访问该Notify的进程PID。

**Notify状态管理**：

- `value`字段映射到硬件寄存器，用于读写状态。
- `rtWaitAndResetNotify`等待Notify变为Ready状态并重置。
- Notify用于Stream间同步、跨进程同步场景。

##### 配套接口映射表（[Notify资源管理](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/appdevg/acldevg/aclcppdevg_000524.html)）

| 实体                       | 关键管理接口                                        |
| -------------------------- | --------------------------------------------------- |
| Notify.notify-id           | `rtCreateNotify`,`rtDestroyNotify`，`rtGetNotifyId` |
| Notify.state               | `lrtWaitAndResetNotify`, `rtWaitAndResetNotify`     |
| IpcNotify.name-or-key      | `rtNotifyGetExportKey`,`rtNotifyImportByKey`        |
| IpcNotifyVistorList.ipc-id | `rtNotifySetImportPid`                              |

#### 3.3.2 Notify同步控制

```mermaid
erDiagram
    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ task-id PK
        typ stream-id FK
        typ seq-number "stream内自增"
        typ type "Notify"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    NotifyRecordTask {
        typ notify-id  FK

    }

    NotifyWaitTask {
        typ notify-id FK
    }

    NotifyRecordTask }o..|| Notify : "use"
    NotifyWaitTask }o..|| Notify : "use"
    Notify {
        typ notify-id  PK
        typ value
    }
    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ NotifyRecordTask : "is a"
    Task ||--|{ NotifyWaitTask : "is a"
```

##### 关键关系说明（Notify同步控制）

**Notify任务类型**：

- `NotifyRecordTask`：将Notify状态设置为Ready，表示某个事件已完成。
- `NotifyWaitTask`：等待Notify状态变为Ready，实现Stream间的同步。

**任务执行顺序**：

- NotifyRecordTask在StreamA执行，设置Notify为Ready。
- NotifyWaitTask在StreamB执行，等待同一个Notify。
- 当Notify变为Ready后，StreamB后续任务才能继续执行。

**跨Stream同步示例**：

```text
StreamA: Task1 -> NotifyRecordTask(notify-id=1) -> Task2
StreamB: NotifyWaitTask(notify-id=1) -> Task3
// Task3必须等待Task1完成后才能执行
```

##### 配套接口映射表（Notify同步控制）

| 实体             | 关键管理接口            |
| ---------------- | ----------------------- |
| NotifyRecordTask | `rtRecordNotify`        |
| NotifyWaitTask   | `lrtWaitAndResetNotify` |

#### 3.3.3 Event资源管理

```mermaid
erDiagram
    Device ||..|{ Event : "Hardware Limit"
    Device ||--o{ Context : "refered by"
    Device {
        typ device-id PK
        typ device-type "A3"
        typ max-event-cnt "65536"
    }

    Event }o..|| Context : "created by"
    Event {
        typ event-id PK
        typ created-ctx-id FK
        typ event-flag
        typ device-res-seq "0~65535"
        typ created-time
        typ status
    }
```

##### 关键关系说明（Event资源管理）

**Event与Device的硬件约束**：

- 每个Device的Event数量上限为`max-event-cnt`（如A3芯片为65536）
- `device-res-seq`是Event在Device内的物理序号（0~65535）
- Event创建时必须指定所属Context，Context绑定到特定Device。

**Event与Context的关系**：

- `created-ctx-id`记录Event的创建Context。
- Event可在多个Stream间共享，但必须属于同一Context。
- 跨Context的Event共享需要通过IPC机制（类似Notify）

**Event状态管理**：

- `status`字段表示Event当前状态：NotRecorded/Recorded/Completed。
- `event-flag`用于控制Event的行为（如是否自动重置）
- `created-time`用于性能统计。

##### 配套接口映射表（Event资源管理）

| 实体           | 关键管理接口                                                             |
| -------------- | ------------------------------------------------------------------------ |
| Event.event-id | `rtCreateEvent`, `rtCreateEventWithFlag`,`rtDestroyEvent`,`rtGetEventId` |
| Event.status   | `rtRecordEvent`,`rtQueryEventStatus`                                     |

#### 3.3.4 Event 流程控制

```mermaid
erDiagram
    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ task-id PK "自增"
        typ stream-id FK
        typ seq-number "stream内自增"
        typ type "EVENT"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    EventTask {
        typ task-id FK
        typ event-id FK
        typ excute-time
        typ finish-time 
        typ first-capture-taskid  FK  
    }

    EventRICaptureTask {
        typ updated-time   
    }  

    EventSyncTask {
        typ event-id FK
        typ excute-time
        typ finish-time 
        typ op-timeout-s 
    }
    EventRecordTask {
        typ event-id FK
        typ excute-time
        typ finish-time 
    }
    EventWaitTask {
        typ event-id FK
        typ excute-time
        typ finish-time 
    }    
    EventTimeTask {
        typ event-id FK
        typ excute-time     
    }
    EventTraceTask {
        typ event-id FK
        typ start-task-id  FK  
    }      

    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ EventTask : "is a "
    EventTask ||--|{ EventRICaptureTask : "is a EXTERNAL"
    EventTask ||--|{ EventSyncTask : "is a EX"
    EventTask ||--|{ EventTimeTask : "is a EX"
    EventTask ||--|{ EventTraceTask : "is a EX"
    EventSyncTask ||--|{ EventRecordTask : "is a EX"
    EventSyncTask ||--|{ EventWaitTask : "is a EX"
```

##### 关键关系说明（Event 流程控制）

**Event任务类型分类**：

- `EventRecordTask`：将Event状态设置为Recorded/Completed。
- `EventWaitTask`：等待Event状态变为Completed。
- `EventSyncTask`：同步等待Event完成（阻塞调用）
- `EventRICaptureTask`：RI Capture模式下的特殊记录任务。
- `EventTimeTask`：记录时间戳相关任务。
- `EventTraceTask`：用于性能追踪的任务记录。

**Event任务继承关系**：

- `EventTask`是基类，包含task-id、event-id、execute-time、finish-time。
- `EventSyncTask`继承EventTask，增加op-timeout-s超时参数。
- `EventRecordTask`和`EventWaitTask`继承EventSyncTask。
- `EventRICaptureTask`、`EventTimeTask`、`EventTraceTask`直接继承EventTask。

**Event执行流程**：

```text
StreamA: KernelTask -> EventRecordTask(event-id=1)
StreamB: EventWaitTask(event-id=1) -> KernelTask2
// StreamB的KernelTask2必须等待StreamA的KernelTask完成
```

**任务追踪关系**：

- `first-capture-task-id`记录首次Capture的Task ID。
- `EventTraceTask.start-task-id`关联追踪的开始任务。
- EventRecordTask通过EventWaitTask映射实现跨Stream同步。

##### 配套接口映射表（Event 流程控制）

| 实体            | 关键管理接口                                         |
| --------------- | ---------------------------------------------------- |
| EventTask       | `rtRecordEvent`, `rtResetEvent`,`rtSynchronizeEvent` |
| EventRecordTask | `rtRecordEvent`, `rtResetEvent`                      |
| EventWaitTask   | `rtStreamWaitEvent`, `rtQueryEventWaitStatus`        |
| EventTimeTask   | `rtResetEvent`, `rtRecordEvent`                      |
| EventTraceTask  | `rtResetEvent`, `rtRecordEvent`                      |

## 4. 通信域建模

跨机通信涉及到多通信域混合任务编排。
通信域的本质是HCCL在框架层通过Rdma_Agent提供的建链能力，维护的一张多卡的网络拓扑，保存在host进程中。

### 4.1 通信域核心概念

通信域（Communicator）是HCCL集合通信的基本抽象，每个通信域定义了一组参与通信的Rank及其拓扑关系。

#### 4.1.1 通信域基础定义

```mermaid
erDiagram
    %% ==========================================
    %% 通信域定义 (HCCL Communicator)
    %% ==========================================
    Communicator {
        typ comm-id PK "通信域ID"
        typ run-id FK "所属Runner进程"
        typ world-size "总Rank数"
        typ my-rank "当前Rank"
        typ color "子通信域颜色标识"
        typ new-comm-id FK "派生的新通信域"
    }
    Rank {
        typ rank-id PK "Rank编号(0~world-size-1)"
        typ device-id FK "绑定的设备"
        typ comm-id FK "所属通信域"
    }
    Runner ||--o{ Communicator : "创建/持有"
    Communicator ||--|{ Rank : "包含"
    Rank }o--|| Device : "绑定"
    Communicator ||--o{ Communicator : "派生(MPI_Comm_split)"
```

#### 4.1.2 控制面：Socket通信

```mermaid
erDiagram
    %% ==========================================
    %% Socket通信 (RaSocket 系列接口)
    %% ==========================================
    Device ||--o{ RaSocket : creates
    RaSocket {
        typ socket-handle PK "Socket句柄"
        typ rdev-handle FK "所属RaDevice"
        typ state "LISTENING/CONNECTED/DISCONNECTED"
        typ role "SERVER/CLIENT"
        typ rank-id FK "关联的Rank"
        typ peer-rank-id "对端Rank ID"
    }
    RaSocketPair {
        typ pair-id PK "连接对ID"
        typ client-socket-handle FK "客户端Socket"
        typ server-socket-handle FK "服务端Socket"
        typ connect-time "连接建立时间"
        typ status "ACTIVE/CLOSED"
    }
    RaSocket ||--o| RaSocketPair : "参与连接"
    RaSocketPair ||--o{ VirtualPointerTable : "关联通信内存"
    
    %% Socket事件管理 (Epoll机制)
    RaSocketEvent {
        typ event-handle PK "事件句柄"
        typ max-events "最大事件数"
        typ timeout "超时时间(ms)"
    }
    RaEpoll {
        typ epoll-id PK "Epoll ID"
        typ event-handle FK "关联事件句柄"
        typ socket-handle FK "监控的Socket句柄"
        typ events "关注的事件类型"
    }
    RaSocketEvent ||--o{ RaEpoll : "管理"
    RaSocket ||--o{ RaEpoll : "被监控"
```

#### 4.1.3 数据面：RDMA通信

```mermaid
erDiagram
    %% ==========================================
    %% RDMA设备与资源 (RaRdev, RaQp, RaMr 接口)
    %% ==========================================
    Device ||--|{ RaDevice : "has virtual NIC"
    RaDevice {
        typ rdev-handle PK "RDMA设备句柄"
        typ device-id FK "关联的NPU Device"
        typ mac-addr "MAC地址"
        typ ip-addr "IP地址"
        typ state "UP/DOWN"
        typ port-num "物理端口编号"
        typ link-speed "链路速率"
        typ mtu "最大传输单元"
    }
    
    %% RDMA核心：QP (Queue Pair)
    RaQP {
        typ qp-handle PK "QP句柄"
        typ ra-dev-handle FK "所属RaDevice"
        typ qp-num "QPN(Queue Pair Number)"
        typ type "RC/UC/UD"
        typ state "RESET/INIT/RTR/RTS/SQD/SQE/Error"
        typ peer-qpn "对端QPN"
        typ send-cq-handle FK "发送完成队列"
        typ recv-cq-handle FK "接收完成队列"
        typ srq-handle FK "共享接收队列(可选)"
    }
    RaDevice ||--o{ RaQP : "拥有QP"
    RaQP ||--|| RaCQ : "send_cq"
    RaQP ||--|| RaCQ : "recv_cq"
    RaQP |o--o| RaQP : "逻辑链接"
    RaQP ||--o| RaSRQ : "使用共享RQ"
    
    %% 完成队列 CQ
    RaCQ {
        typ cq-handle PK "CQ句柄"
        typ ra-dev-handle FK "所属RaDevice"
        typ cqn "CQN"
        typ size "队列深度"
        typ policy "CQ完成策略"
    }
    RaCQe {
        typ cqe-id PK "CQE ID"
        typ cq-handle FK "所属CQ"
        typ wr-id "Work Request ID"
        typ status "SUCCESS/FLUSH_ERR/..."
        typ opcode "SEND/RECV/READ/WRITE"
        typ byte-len "传输字节数"
    }
    RaDevice ||--o{ RaCQ : "拥有CQ"
    RaCQ ||--o{ RaCQe : "包含"
    
    %% 内存注册 MR
    RaMR {
        typ mr-handle PK "MR句柄"
        typ ra-dev-handle FK "所属RaDevice"
        typ lkey "Local Key"
        typ rkey "Remote Key"
        typ addr "起始地址"
        typ length "内存长度(字节)"
        typ access "访问权限"
    }
    RaDevice ||--o{ RaMR : "注册内存"
    RaMR ||--|{ VirtualPointerTable : "映射到虚拟内存"
    
    %% 共享接收队列 SRQ
    RaSRQ {
        typ srq-handle PK "SRQ句柄"
        typ ra-dev-handle FK "所属RaDevice"
        typ srq-num "SRQN"
        typ max-wr "最大WR数"
        typ max-sge "最大SGE数"
    }
    RaDevice ||--o{ RaSRQ : "拥有SRQ"
    
    %% NDA直接访问
    RaNdaCQ {
        typ nda-cq-handle PK "NDA CQ句柄"
        typ rdma-handle FK "所属RDMA句柄"
        typ cqn "CQN"
        typ depth "队列深度"
    }
    RaNdaQP {
        typ nda-qp-handle PK "NDA QP句柄"
        typ rdma-handle FK "所属RDMA句柄"
        typ qp-num "QPN"
        typ nda-cq-handle FK "关联的NDA CQ"
    }
    RaDevice ||--o{ RaNdaCQ : "创建NDA CQ"
    RaDevice ||--o{ RaNdaQP : "创建NDA QP"
    RaNdaQP ||--|| RaNdaCQ : "使用"
```

#### 4.1.4 数据面：UB统一总线

```mermaid
erDiagram
    %% ==========================================
    %% UB上下文 (RaContext 系列接口)
    %% ==========================================
    Device ||--o{ RaContext : creates
    RaContext {
        typ ctx-handle PK "UB上下文句柄"
        typ device-id FK "关联设备ID"
        typ mode "模式:RDMA/UB/UB_PLUS"
        typ local-endpoint FK "本地端点,EID映射"
        typ max-jetty-num "最大Jetty数量"
        typ max-jfc-num "最大JFC数量"
    }
    
    EndPoint-Pair {
        typ endpoint-pair-id PK
        typ local-endpoint-id FK
        typ remote-endpoint-id FK
    }

    %% UB核心资源
    RaContext ||--o{ RaJetty : "创建Jetty"
    RaContext ||--o{ RaJfc : "创建JFC"
    RaContext ||--o{ RaLmem : "注册本地内存"
    RaContext ||--o{ RaRmem : "导入远端内存"
    RaContext ||--o{ RaTp : "管理传输路径"
    RaContext ||--o{ RaTokenId : "分配TokenID"
    RaContext ||--o{ RaChan : "创建通道"
    RaContext ||--o{ EndPoint-Pair : "关联EndPoint-Pair"

    %% Jetty (QP等价物)
    RaJetty {
        typ jetty-handle PK "Jetty句柄"
        typ ctx-handle FK "所属UB上下文"
        typ jetty-id "Jetty ID"
        typ mode "URMA_NORMAL/CACHE_LOCK_DWQE/CCU/..."
        typ sq-depth "发送队列深度"
        typ rq-depth "接收队列深度"
        typ state "RESET/READY/SUSPENDED/ERROR"
        typ peer-jetty-handle FK "对端Jetty"
    }
    RaJetty ||--o| RaJfc : "send_jfc"
    RaJetty ||--o| RaJfc : "recv_jfc"
    RaJetty |o--o| RaJetty : "逻辑绑定"
    
    %% JFC (CQ等价物)
    RaJfc {
        typ jfc-handle PK "JFC句柄"
        typ ctx-handle FK "所属UB上下文"
        typ jfc-id "JFC ID"
        typ depth "队列深度"
        typ mode "NORMAL/STARS_POLL/CCU_POLL"
        typ policy "完成策略"
    }
    RaCr {
        typ cr-id PK "完成请求ID"
        typ jfc-handle FK "所属JFC"
        typ status "SUCCESS/FLUSH_ERR/..."
        typ opcode "SEND/RECV/READ/WRITE"
        typ byte-len "传输字节数"
    }
    RaJfc ||--o{ RaCr : "包含"
    
    %% 本地内存注册
    RaLmem {
        typ lmem-handle PK "本地内存句柄"
        typ ctx-handle FK "所属UB上下文"
        typ addr "内存地址"
        typ size "内存大小(字节)"
        typ mem-key "内存密钥"
        typ token-id FK "关联TokenID"
    }
    RaLmem ||--|{ VirtualPointerTable : "映射"
    
    %% 远端内存导入
    RaRmem {
        typ rmem-handle PK "远端内存句柄"
        typ ctx-handle FK "所属UB上下文"
        typ mem-key "远端内存密钥"
        typ target-seg-handle FK "目标段句柄"
        typ remote-eid "远端EID"
    }
    
    %% 传输路径
    RaTp {
        typ tp-handle PK "传输路径句柄"
        typ ctx-handle FK "所属UB上下文"
        typ tp-type "RTP/CTP/UTP"
        typ tpn "传输路径号"
        typ speed "链路速率"
        typ status "UP/DOWN"
    }
    RaJetty ||--o{ RaTp : "使用"
    
    %% TokenID
    RaTokenId {
        typ token-handle PK "Token句柄"
        typ ctx-handle FK "所属UB上下文"
        typ token-id "Token ID"
        typ ref-count "引用计数"
    }
    
    %% 
    RaChan {
        typ chan-handle PK "通道句柄"
        typ ctx-handle FK "所属UB上下文"
        typ chan-id "通道ID"
        typ mode "通道模式"
    }
```

#### 4.1.5 异步请求管理

```mermaid
erDiagram
    AsyncRequest {
        typ req-handle PK "异步请求句柄"
        typ req-type "CONNECT/LISTEN/CLOSE/QP_CREATE/..."
        typ status "PENDING/COMPLETED/FAILED"
        typ submit-time "提交时间"
        typ complete-time "完成时间"
    }
```

#### 4.1.6 通信域架构总结

```text
┌─────────────────────────────────────────────────────────────────┐
│                    HCCL 通信域架构                               │
├─────────────────────────────────────────────────────────────────┤
│  应用层                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Communicator (通信域)                                     │   │
│  │  └── Rank[0..N] (参与节点，每个绑定一个Device)              │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  控制面 (建链/握手)                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  RaSocket (Socket通信)                                     │   │
│  │  ├── RaSocketPair (连接对)                                 │   │
│  │  └── RaEpoll (事件监控)                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  数据面 (数据传输)                                                │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │  RDMA (传统模式)      │    │  UB (统一总线)       │            │
│  │  ├── RaDevice        │    │  ├── RaContext      │            │
│  │  ├── RaQP (队列对)    │    │  ├── RaJetty (QP)   │            │
│  │  ├── RaCQ (完成队列)  │    │  ├── RaJfc (CQ)     │            │
│  │  ├── RaMR (内存注册)  │    │  ├── RaLmem/Rmem    │            │
│  │  └── RaSRQ (共享RQ)   │    │  └── RaTp (传输路径) │            │
│  └─────────────────────┘    └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.1.7 关键实体对照表

| 概念 | RDMA模式 | UB模式 | 说明 |
| --- | --- | --- | --- |
| 上下文 | RaDevice | RaContext | 设备/上下文句柄 |
| 队列对 | RaQP | RaJetty | 数据传输通道 |
| 完成队列 | RaCQ | RaJfc | 完成通知 |
| 完成元素 | RaCQe | RaCr | 完成状态 |
| 本地内存 | RaMR | RaLmem | 内存注册 |
| 远端内存 | - | RaRmem | 远端内存导入 |
| 传输路径 | - | RaTp | 物理路径管理 |
| 安全令牌 | - | RaTokenId | 访问控制 |

#### 4.1.8 HCCP接口分类

```text
HCCP Network API
├── 控制平面 (Socket通信)
|   ├── 初始化: RaSocketInit/RaSocketDeinit (Socket)
│   ├── 连接管理: RaSocketBatchConnect/Close/Abort
│   ├── 监听管理: RaSocketListenStart/Stop
│   ├── 数据收发: RaSocketSend/Recv
│   ├── 状态查询: RaGetSockets
│   └── 事件管理: RaEpollCtlAdd/Mod/Del
├── 数据平面 - RDMA
|   ├── 初始化: RaRdevInit/RaRdevDeinit (RDMA设备)
│   ├── QP管理: RaQpCreate/Destroy/ConnectAsync
│   ├── CQ管理: RaCqCreate/Destroy
│   ├── MR管理: RaMrReg/Dereg
│   ├── 工作请求: RaSendWr/RaRecvWrlist
│   └── 完成轮询: RaPollCq
├── 数据平面 - UB
|   ├── 初始化: RaCtxInit/RaCtxDeinit (统一上下文)
│   ├── Jetty管理: RaCtxQpCreate/Destroy/Import/Bind
│   ├── JFC管理: RaCtxCqCreate/Destroy
│   ├── 内存管理: RaCtxLmemRegister/RmemImport
│   ├── Token管理: RaCtxTokenIdAlloc/Free
│   └── 工作请求: RaBatchSendWr
├── 异步操作
│   ├── RaSocketBatchConnectAsync
│   ├── RaCtxQpCreateAsync/DestroyAsync
│   └── RaGetAsyncReqResult
├── 网络诊断
|   ├── RaPingInit/RaPingDeinit (Ping)
│   ├── RaPingTargetAdd/Del
│   ├── RaPingTaskStart/Stop
│   └── RaPingGetResults
└── TLV消息
|   ├── RaTlvInit/RaTlvDeinit (TLV)
    └── RaTlvRequest
```

##### Socket通信接口

| 接口                   | 功能           | 关键参数                                                                 |
| ---------------------- | -------------- | ------------------------------------------------------------------------ |
| `RaSocketInit`         | Socket初始化    | `mode`, `rdevInfo`, `socketHandle`                                      |
| `RaSocketDeinit`       | Socket去初始化 | `socketHandle`                                                           |
| `RaSocketBatchConnect` | 批量连接       | `SocketConnectInfoT[]`, `num`                                            |
| `RaSocketBatchClose`   | 批量关闭       | `SocketCloseInfoT[]`, `num`                                              |
| `RaSocketBatchAbort`   | 批量中止       | `SocketConnectInfoT[]`, `num`                                            |
| `RaSocketListenStart`  | 开始监听       | `SocketListenInfoT[]`, `num`                                             |
| `RaSocketListenStop`   | 停止监听       | `SocketListenInfoT[]`, `num`                                             |
| `RaGetSockets`         | 获取Socket状态 | `role`, `SocketInfoT[]`, `num`, `connectedNum`                           |
| `RaSocketSend`         | 发送数据       | `fdHandle`, `data`, `size`, `sentSize`                                   |
| `RaSocketRecv`         | 接收数据       | `fdHandle`, `data`, `size`, `receivedSize`                               |
| `RaEpollCtlAdd`        | 添加Epoll事件  | `fdHandle`, `event`                                                      |
| `RaEpollCtlMod`        | 修改Epoll事件  | `fdHandle`, `event`                                                      |
| `RaEpollCtlDel`        | 删除Epoll事件  | `fdHandle`                                                               |
| `RaCreateEventHandle`  | 创建事件句柄   | `eventHandle`                                                            |
| `RaWaitEventHandle`    | 等待事件       | `eventHandle`, `SocketEventInfoT[]`, `timeout`, `maxevents`, `eventsNum` |
| `RaDestroyEventHandle` | 销毁事件句柄   | `eventHandle`                                                            |
| `RaSocketWhiteListAdd` | 添加白名单     | `socketHandle`, `SocketWlistInfoT[]`, `num`                              |
| `RaSocketWhiteListDel` | 删除白名单     | `socketHandle`, `SocketWlistInfoT[]`, `num`                              |

###### RDMA操作接口

| 接口                  | 功能           | 关键参数                                                                |
| --------------------- | -------------- | ----------------------------------------------------------------------- |
| `RaRdevInit`           | RDMA设备初始化       | `mode`, `notifyType`, `rdevInfo`, `rdmaHandle`      |
| `RaRdevInitV2`         | RDMA设备初始化(扩展) | `RdevInitInfo`, `rdevInfo`, `rdmaHandle`            |
| `RaRdevInitWithBackup` | 带备份的初始化       | `initInfo`, `rdevInfo`, `backupRdevInfo`            |
| `RaRdevDeinit`         | RDMA设备去初始化     | `rdmaHandle`, `notifyType`                          |
| `RaQpCreate`          | 创建QP         | `rdevHandle`, `flag`, `qpMode`, `qpHandle`                              |
| `RaQpCreateWithAttrs` | 创建QP(带属性) | `rdevHandle`, `QpExtAttrs`, `qpHandle`                                  |
| `RaAiQpCreate`        | 创建AI QP      | `rdevHandle`, `QpExtAttrs`, `AiQpInfo`, `qpHandle`                      |
| `RaLoopbackQpCreate`  | 创建回环QP     | `rdevHandle`, `LoopbackQpPair`, `qpHandle`                              |
| `RaTypicalQpCreate`   | 创建典型QP     | `rdevHandle`, `flag`, `qpMode`, `TypicalQp`, `qpHandle`                 |
| `RaQpDestroy`         | 销毁QP         | `qpHandle`                                                              |
| `RaQpConnectAsync`    | 异步连接QP     | `qpHandle`, `fdHandle`                                                  |
| `RaGetQpStatus`       | 获取QP状态     | `qpHandle`, `status`                                                    |
| `RaTypicalQpModify`   | 修改典型QP     | `qpHandle`, `localQpInfo`, `remoteQpInfo`                               |
| `RaMrReg`             | 注册MR         | `qpHandle`, `MrInfoT`                                                   |
| `RaMrDereg`           | 注销MR         | `qpHandle`, `MrInfoT`                                                   |
| `RaRegisterMr`        | 独立注册MR     | `rdmaHandle`, `MrInfoT`, `mrHandle`                                     |
| `RaDeregisterMr`      | 独立注销MR     | `rdmaHandle`, `mrHandle`                                                |
| `RaRemapMr`           | 重映射MR       | `rdmaHandle`, `MemRemapInfo[]`, `num`                                   |
| `RaGetNotifyMrInfo`   | 获取通知MR信息 | `rdevHandle`, `MrInfoT`                                                 |
| `RaSendWr`            | 发送工作请求   | `qpHandle`, `SendWr`, `SendWrRsp`                                       |
| `RaSendWrV2`          | 发送工作请求V2 | `qpHandle`, `SendWrV2`, `SendWrRsp`                                     |
| `RaSendWrlist`        | 批量发送       | `qpHandle`, `SendWrlistData[]`, `SendWrRsp[]`, `sendNum`, `completeNum` |
| `RaRecvWrlist`        | 批量接收       | `qpHandle`, `RecvWrlistData`, `recvNum`, `completeNum`                  |
| `RaPollCq`            | 轮询CQ         | `qpHandle`, `isSendCq`, `numEntries`, `wc`                              |
| `RaCqCreate`          | 创建CQ         | `rdevHandle`, `CqAttr`                                                  |
| `RaCqDestroy`         | 销毁CQ         | `rdevHandle`, `CqAttr`                                                  |
| `RaCreateSrq`         | 创建SRQ        | `rdmaHandle`, `SrqAttr`                                                 |
| `RaDestroySrq`        | 销毁SRQ        | `rdmaHandle`, `SrqAttr`                                                 |
| `RaSetQpAttrQos`      | 设置QP QoS     | `qpHandle`, `QosAttr`                                                   |
| `RaSetQpAttrTimeout`  | 设置QP超时     | `qpHandle`, `timeout`                                                   |
| `RaSetQpAttrRetryCnt` | 设置QP重试次数 | `qpHandle`, `retryCnt`                                                  |
| `RaGetQpAttr`         | 获取QP属性     | `qpHandle`, `QpAttr`                                                    |
| `RaGetQpContext`      | 获取QP上下文   | `qpHandle`, `qp`, `sendCq`, `recvCq`                                    |

###### UB统一总线接口

| 接口                    | 功能             | 关键参数                                                         |
| ----------------------- | ---------------- | ---------------------------------------------------------------- |
| `RaCtxInit`            | 上下文初始化         | `CtxInitCfg`, `CtxInitAttr`, `ctxHandle`            |
| `RaCtxDeinit`          | 上下文去初始化       | `ctxHandle`                                         |
| `RaGetDevEidInfoNum`    | 获取EID数量      | `RaInfo`, `num`                                                  |
| `RaGetDevEidInfoList`   | 获取EID列表      | `RaInfo`, `HccpDevEidInfo[]`, `num`                              |
| `RaGetEidByIp`          | 通过IP获取EID    | `ctxHandle`, `IpInfo[]`, `HccpEid[]`, `num`                      |
| `RaGetDevBaseAttr`      | 获取设备属性     | `ctxHandle`, `DevBaseAttr`                                       |
| `RaCtxGetAsyncEvents`   | 获取异步事件     | `ctxHandle`, `AsyncEvent[]`, `num`                               |
| `RaCtxTokenIdAlloc`     | 分配TokenID      | `ctxHandle`, `HccpTokenId`, `tokenIdHandle`                      |
| `RaCtxTokenIdFree`      | 释放TokenID      | `ctxHandle`, `tokenIdHandle`                                     |
| `RaCtxLmemRegister`     | 注册本地内存     | `ctxHandle`, `MrRegInfoT`, `lmemHandle`                          |
| `RaCtxLmemUnregister`   | 注销本地内存     | `ctxHandle`, `lmemHandle`                                        |
| `RaCtxRmemImport`       | 导入远端内存     | `ctxHandle`, `MrImportInfoT`, `rmemHandle`                       |
| `RaCtxRmemUnimport`     | 取消导入远端内存 | `ctxHandle`, `rmemHandle`                                        |
| `RaCtxChanCreate`       | 创建通道         | `ctxHandle`, `ChanInfoT`, `chanHandle`                           |
| `RaCtxChanDestroy`      | 销毁通道         | `ctxHandle`, `chanHandle`                                        |
| `RaCtxCqCreate`         | 创建CQ           | `ctxHandle`, `CqInfoT`, `cqHandle`                               |
| `RaCtxCqDestroy`        | 销毁CQ           | `ctxHandle`, `cqHandle`                                          |
| `RaCtxQpCreate`         | 创建QP/Jetty     | `ctxHandle`, `QpCreateAttr`, `QpCreateInfo`, `qpHandle`          |
| `RaCtxQpQueryBatch`     | 批量查询QP       | `qpHandle[]`, `JettyAttr[]`, `num`                               |
| `RaCtxQpDestroy`        | 销毁QP/Jetty     | `qpHandle`                                                       |
| `RaCtxQpImport`         | 导入Jetty        | `ctxHandle`, `QpImportInfoT`, `remQpHandle`                      |
| `RaCtxQpUnimport`       | 取消导入Jetty    | `ctxHandle`, `remQpHandle`                                       |
| `RaCtxQpBind`           | 绑定Jetty        | `qpHandle`, `remQpHandle`                                        |
| `RaCtxQpUnbind`         | 解绑Jetty        | `qpHandle`                                                       |
| `RaBatchSendWr`         | 批量发送         | `qpHandle`, `SendWrData[]`, `SendWrResp[]`, `num`, `completeNum` |
| `RaCtxUpdateCi`         | 更新CI           | `qpHandle`, `ci`                                                 |
| `RaCtxGetAuxInfo`       | 获取辅助信息     | `ctxHandle`, `HccpAuxInfoIn`, `HccpAuxInfoOut`                   |
| `RaCtxGetCrErrInfoList` | 获取CR错误       | `ctxHandle`, `CrErrInfo[]`, `num`                                |

###### 异步操作接口

| 接口                        | 功能           | 关键参数                                                             |
| --------------------------- | -------------- | -------------------------------------------------------------------- |
| `RaGetAsyncReqResult`       | 获取异步结果   | `reqHandle`, `reqResult`                                             |
| `RaSocketBatchConnectAsync` | 异步批量连接   | `SocketConnectInfoT[]`, `num`, `reqHandle`                           |
| `RaSocketListenStartAsync`  | 异步开始监听   | `SocketListenInfoT[]`, `num`, `reqHandle`                            |
| `RaSocketListenStopAsync`   | 异步停止监听   | `SocketListenInfoT[]`, `num`, `reqHandle`                            |
| `RaSocketBatchCloseAsync`   | 异步批量关闭   | `SocketCloseInfoT[]`, `num`, `reqHandle`                             |
| `RaSocketSendAsync`         | 异步发送       | `fdHandle`, `data`, `size`, `sentSize`, `reqHandle`                  |
| `RaSocketRecvAsync`         | 异步接收       | `fdHandle`, `data`, `size`, `receivedSize`, `reqHandle`              |
| `RaCtxLmemRegisterAsync`    | 异步注册内存   | `ctxHandle`, `MrRegInfoT`, `lmemHandle`, `reqHandle`                 |
| `RaCtxLmemUnregisterAsync`  | 异步注销内存   | `ctxHandle`, `lmemHandle`, `reqHandle`                               |
| `RaCtxQpCreateAsync`        | 异步创建QP     | `ctxHandle`, `QpCreateAttr`, `QpCreateInfo`, `qpHandle`, `reqHandle` |
| `RaCtxQpDestroyAsync`       | 异步销毁QP     | `qpHandle`, `reqHandle`                                              |
| `RaCtxQpDestroyBatchAsync`  | 异步批量销毁   | `ctxHandle`, `qpHandle[]`, `num`, `reqHandle`                        |
| `RaCtxQpImportAsync`        | 异步导入Jetty  | `ctxHandle`, `QpImportInfoT`, `remQpHandle`, `reqHandle`             |
| `RaGetTpInfoListAsync`      | 异步获取TP信息 | `ctxHandle`, `GetTpCfg`, `HccpTpInfo[]`, `num`, `reqHandle`          |
| `RaGetEidByIpAsync`         | 异步获取EID    | `ctxHandle`, `IpInfo[]`, `HccpEid[]`, `num`, `reqHandle`             |
| `RaGetTpAttrAsync`          | 异步获取TP属性 | `ctxHandle`, `tpHandle`, `attrBitmap`, `TpAttr`, `reqHandle`         |
| `RaSetTpAttrAsync`          | 异步设置TP属性 | `ctxHandle`, `tpHandle`, `attrBitmap`, `TpAttr`, `reqHandle`         |

###### 网络诊断接口

| 接口               | 功能         | 关键参数                                     |
| ------------------ | ------------ | --------------------------------------------|
| `RaPingInit`       | Ping初始化    | `PingInitAttr`, `PingInitInfo`, `pingHandle`|
| `RaPingDeinit`     | Ping去初始化  | `pingHandle`                                |
| `RaPingTargetAdd`  | 添加Ping目标 | `pingHandle`, `PingTargetInfo[]`, `num`      |
| `RaPingTargetDel`  | 删除Ping目标 | `pingHandle`, `PingTargetCommInfo[]`, `num`  |
| `RaPingTaskStart`  | 启动Ping任务 | `pingHandle`, `PingTaskAttr`                 |
| `RaPingTaskStop`   | 停止Ping任务 | `pingHandle`                                 |
| `RaPingGetResults` | 获取Ping结果 | `pingHandle`, `PingTargetResult[]`, `num`    |

###### TLV消息接口

| 接口           | 功能        | 关键参数                                      |
| -------------- | ----------- | --------------------------------------------- |
| `RaTlvInit`    | TLV初始化   | `TlvInitInfo`, `bufferSize`, `tlvHandle`      |
| `RaTlvDeinit`  | TLV反初始化 | `tlvHandle`                                   |
| `RaTlvRequest` | TLV请求处理 | `tlvHandle`, `moduleType`, `TlvMsg`, `TlvMsg` |

###### NDA(Network Direct Acess)直接访问接口

| 接口                 | 功能             | 关键参数                                               |
| -------------------- | ---------------- | ------------------------------------------------------ |
| `RaNdaGetDirectFlag` | 获取直接访问标志 | `rdmaHandle`, `directFlag`                             |
| `RaNdaCqCreate`      | 创建NDA CQ       | `rdmaHandle`, `NdaCqInitAttr`, `NdaCqInfo`, `cqHandle` |
| `RaNdaCqDestroy`     | 销毁NDA CQ       | `rdmaHandle`, `cqHandle`                               |
| `RaNdaQpCreate`      | 创建NDA QP       | `rdmaHandle`, `NdaQpInitAttr`, `NdaQpInfo`, `qpHandle` |

###### 通用查询接口

| 接口                     | 功能           | 关键参数                                       |
| ------------------------ | -------------- | ----------------------------------------------|
| `RaGetIfnum`             | 获取接口数量   | `RaGetIfattr`, `num`                           |
| `RaGetIfaddrs`           | 获取接口地址   | `RaGetIfattr`, `InterfaceInfo[]`, `num`        |
| `RaSocketGetVnicIpInfos` | 获取虚拟网卡IP | `phyId`, `IdType`, `ids[]`, `num`, `IpInfo[]`  |
| `RaGetTlsEnable`         | 获取TLS状态    | `RaInfo`, `tlsEnable`                          |
| `RaGetHccnCfg`           | 获取HCCN配置   | `RaInfo`, `HccnCfgKey`, `value`, `valueLen`    |
| `RaGetInterfaceVersion`  | 获取接口版本   | `phyId`, `interfaceOpcode`, `interfaceVersion` |
| `RaRdevGetHandle`        | 获取Rdev句柄   | `phyId`, `rdmaHandle`                          |
| `RaRdevGetSupportLite`   | 获取Lite支持   | `rdmaHandle`, `supportLite`                    |
| `RaSaveSnapshot`         | 保存快照       | `RaInfo`, `SaveSnapshotAction`                 |
| `RaRestoreSnapshot`      | 恢复快照       | `RaInfo`                                       |
| `RaGetSecRandom`         | 获取安全随机数 | `RaInfo`, `value`                              |

##### 关键关系说明（HCCP接口分类）

**通信域层次结构**：

- `Communicator`是HCCL集合通信的核心抽象，定义了一组参与通信的Rank。
- `Rank`是通信域中的参与节点，每个Rank绑定到具体的Device。
- 父通信域派生子通信域（如`color`属性实现分组）

**控制平面与数据平面分离**：

- **控制平面(Socket)**：用于建链、交换QP信息、控制信令，基于TCP协议。
- **数据平面(RDMA/UB)**：用于高性能数据传输，基于RDMA Verbs或UB协议。

**RDMA资源层次**：

- `RaDevice`是虚拟网卡抽象，一个Device可创建多个RaDevice。
- `RaQP`是队列对，包含Send Queue和Receive Queue。
- `RaCQ`是完成队列，用于轮询WR完成状态。
- `RaMR`是内存注册，将虚拟内存映射为RDMA可访问的物理内存。
- `RaSRQ`是共享接收队列，多个QP可共享同一SRQ以提高资源利用率。

**UB资源层次**：

- `RaContext`是UB统一上下文，替代RaDevice的设备抽象。
- `RaJetty`是QP等价物，支持多种模式（URMA_NORMAL/CCU等）
- `RaJfc`是CQ等价物，用于完成请求管理。
- `RaLmem/RaRmem`是本地/远端内存管理，替代RaMR。
- `RaTp`是传输路径管理，支持RTP/CTP/UTP三种类型。
- `RaTokenId`是安全通信令牌，用于跨进程内存访问控制。

**实体关联要点**：

1. `RaSocketPair`需要两个`RaSocket`（client端和server端）才能建立连接。
2. `RaQP.state`必须经历RESET→INIT→RTR→RTS的状态转换才能正常通信。
3. `RaMR.lkey`用于本地访问，`RaMR.rkey`用于远端RDMA访问。
4. `RaJetty`通过`Bind`操作与对端Jetty建立逻辑连接。
5. `RaLmem`必须注册后，`RaRmem`才能从对端导入并访问。

**NDA(Network Direct Access)直接访问机制**：

- `RaNdaQP`和`RaNdaCQ`是网络直接访问的QP/CQ变体。
- NDA模式允许绕过部分协议栈，降低延迟。
- `RaNdaGetDirectFlag`检查设备是否支持NDA模式。

**异步请求管理**：

- `AsyncRequest`统一管理所有异步操作的请求句柄。
- 异步操作包括：连接、监听、QP创建/销毁、内存注册等。
- `RaGetAsyncReqResult`轮询异步操作结果。

**Socket事件机制**：

- `RaSocketEvent`是事件等待机制的句柄。
- `RaEpoll`实现类似Linux Epoll的事件监控机制。
- 支持添加、修改、删除监控的Socket事件。

**网络接口与配置**：

- `InterfaceInfo`描述网络接口的IP/MAC/MTU等属性。
- `HccnConfig`存储HCCN网络配置键值对。
- `Snapshot`支持设备状态的保存和恢复。

##### 控制平面 vs 数据平面

| 维度         | 控制平面                                                      | 数据平面                                   |
| ------------ | ------------------------------------------------------------- | ------------------------------------------ |
| **核心功能** | 建链、交换QP信息、控制信令                                    | 数据传输、RDMA操作                         |
| **关键实体** | RaSocket, SocketConnection, RaEpoll                           | RaQP, RaCQ, RaMR, RaJetty                  |
| **关键接口** | `RaSocketBatchConnect`, `RaSocketListenStart`, `RaGetSockets` | `RaSendWr`, `RaPollCq`, `RaQpConnectAsync` |
| **通信方式** | TCP Socket                                                    | RDMA Verbs / UB                            |

##### RDMA模式 vs UB模式

| 对比项       | RDMA模式         | UB模式                 |
| ------------ | ---------------- | ---------------------- |
| **设备抽象** | RaDevice         | RaContext              |
| **队列对**   | RaQP (QP)        | RaJetty (Jetty)        |
| **完成队列** | RaCQ (CQ)        | RaJfc (JFC)            |
| **内存注册** | RaMR (lkey/rkey) | RaLmem/RaRmem (MemKey) |
| **地址标识** | IP + GID         | EID (Endpoint ID)      |
| **传输路径** | QPN + GID        | RaTp (TPN)             |

##### 关键点说明

1. **RaContext/RaDevice**: 统一上下文实体，支持 RDMA 和 UB 双模式。
2. **RaJetty/RaJfc/RaQP/RaCQ**: UB 模式下的 QP/CQ 等价物。
3. **RaLmem/RaRmem/RaMR**:  UB 模式下的本地/远端内存管理。
4. **RaTp**: UB 传输路径管理。
5. **RaTokenId**: 安全通信令牌。

##### 关键属性补充

- **QP Mode**: `NOR`(普通), `GDR_TMPL`(模板), `OP`(操作), `GDR_ASYN`(异步GDR)
- **Transport Mode**: `RC`(可靠连接), `RM`(可靠消息，仅UB)
- **Jetty Mode**: `URMA_NORMAL`, `CACHE_LOCK_DWQE`, `CCU`, `USER_CTL_NORMAL`
- **JFC Mode**: `NORMAL`, `STARS_POLL`, `CCU_POLL`

##### 配套接口映射表（HCCP接口分类）

| 实体             | 初始化接口                   | 创建接口                                                 | 操作接口                                   | 销毁/清理接口                               |
| --------------- | ---------------------------- | -------------------------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| **Communicator** | -                            | `HcclCommInitRankInfo`, `HcclCommInitClusterInfo`        | `HcclGetRankId`, `HcclGetRankSize`         | `HcclCommDestroy`                          |
| **Rank**         | -                            | -                                          | -                   | -                        |
| **RaSocket**     | `RaSocketInit`               | `RaSocketBatchConnect`                                   | `RaSocketSend`, `RaSocketRecv`, `RaGetSockets` | `RaSocketDeinit`, `RaSocketBatchClose`, `RaSocketBatchAbort` |
| **RaSocketPair** | -                            | `RaSocketBatchConnect`                                   | `RaGetSockets`                             | `RaSocketBatchClose`                       |
| **RaSocketEvent** | `RaCreateEventHandle`       | -                                                        | `RaWaitEventHandle`                        | `RaDestroyEventHandle`                     |
| **RaEpoll**      | -                            | `RaEpollCtlAdd`                                          | `RaEpollCtlMod`                            | `RaEpollCtlDel`                            |
| **RaDevice**     | `RaRdevInit`, `RaRdevInitV2`, `RaRdevInitWithBackup` | -                                                        | `RaRdevGetHandle`, `RaRdevGetSupportLite`  | `RaRdevDeinit`                             |
| **RaQP**         | -                            | `RaQpCreate`, `RaQpCreateWithAttrs`, `RaTypicalQpCreate`, `RaAiQpCreate`, `RaLoopbackQpCreate` | `RaQpConnectAsync`, `RaSendWr`, `RaSendWrV2`, `RaSendWrlist`, `RaRecvWrlist`, `RaPollCq`, `RaGetQpStatus`, `RaTypicalQpModify` | `RaQpDestroy`                              |
| **RaCQ**         | -                            | `RaCqCreate`                                             | `RaPollCq`                                 | `RaCqDestroy`                              |
| **RaSRQ**        | -                            | `RaCreateSrq`                                            | `RaModifySrq`                              | `RaDestroySrq`                             |
| **RaMR**         | -                            | `RaMrReg`, `RaRegisterMr`                                | `RaRemapMr`, `RaGetNotifyMrInfo`           | `RaMrDereg`, `RaDeregisterMr`              |
| **RaCQe**        | -                            | -                                                        | `RaPollCq`                                 | -                                          |
| **RaQPAttr**     | -                            | -                                                        | `RaGetQpAttr`, `RaSetQpAttrQos`, `RaSetQpAttrTimeout`, `RaSetQpAttrRetryCnt`, `RaGetQpContext` | -                                          |
| **RaNdaQP**      | -                            | `RaNdaQpCreate`                                          | -                                          | -                                          |
| **RaNdaCQ**      | -                            | `RaNdaCqCreate`                                          | -                                          | `RaNdaCqDestroy`                           |
| **RaContext**    | `RaCtxInit`                  | -                                                        | `RaGetDevBaseAttr`, `RaGetDevEidInfoList`, `RaGetDevEidInfoNum`, `RaGetEidByIp`, `RaCtxGetAsyncEvents` | `RaCtxDeinit`                              |
| **RaJetty**      | -                            | `RaCtxQpCreate`                                          |  `RaBatchSendWr`, `RaCtxUpdateCi`, `RaCtxQpQueryBatch` | `RaCtxQpDestroy`        |
| **EndPointPair**      | -                            | -                                         | `RaCtxQpBind`, `RaCtxQpUnbind`, `RaCtxQpImport`, `RaCtxQpUnimport` | - |
| **RaJfc**        | -                            | `RaCtxCqCreate`                                          | `RaCtxGetAuxInfo`, `RaCtxGetCrErrInfoList` | `RaCtxCqDestroy`                           |
| **RaCr**         | -                            | -                                                        | `RaCtxGetAuxInfo`                          | -                                          |
| **RaLmem**       | -                            | `RaCtxLmemRegister`                                      | -                                          | `RaCtxLmemUnregister`                      |
| **RaRmem**       | -                            | `RaCtxRmemImport`                                        | -                                          | `RaCtxRmemUnimport`                        |
| **RaTp**         | -                            | `RaGetTpInfoListAsync`                                   | `RaGetTpAttrAsync`, `RaSetTpAttrAsync`     | -                                          |
| **RaTokenId**    | -                            | `RaCtxTokenIdAlloc`                                      | -                                          | `RaCtxTokenIdFree`                         |
| **RaChan**       | -                            | `RaCtxChanCreate`                                        | -                                          | `RaCtxChanDestroy`                         |
| **RaPing**       | `RaPingInit`                 | `RaPingTargetAdd`                                        | `RaPingTaskStart`, `RaPingGetResults`      | `RaPingDeinit`, `RaPingTargetDel`          |
| **RaTlv**        | `RaTlvInit`                  | -                                                        | `RaTlvRequest`                             | `RaTlvDeinit`                              |
| **AsyncRequest** | -                            | `RaSocketBatchConnectAsync`, `RaCtxQpCreateAsync`, `RaCtxQpDestroyAsync`, `RaCtxQpDestroyBatchAsync`, `RaCtxQpImportAsync`, `RaCtxLmemRegisterAsync`, `RaSocketListenStartAsync`, `RaSocketListenStopAsync`, `RaSocketBatchCloseAsync`, `RaSocketSendAsync`, `RaSocketRecvAsync`, `RaGetTpInfoListAsync`, `RaGetEidByIpAsync`, `RaGetTpAttrAsync`, `RaSetTpAttrAsync` | `RaGetAsyncReqResult` | - |
| **InterfaceInfo** | -                           | -                                                        | `RaGetIfnum`, `RaGetIfaddrs`, `RaSocketGetVnicIpInfos` | -                                          |
| **HccnConfig**   | -                            | -                                                        | `RaGetHccnCfg`, `RaGetTlsEnable`, `RaGetInterfaceVersion`, `RaGetSecRandom` | -                                          |
| **Snapshot**     | -                            | -                                                        | `RaSaveSnapshot`, `RaRestoreSnapshot`      | -                                          |

##### 实体属性与接口映射补充表

| 实体属性                    | 对应接口                                                                 |
| -------------------------- | ------------------------------------------------------------------------ |
| RaSocket.state             | `RaGetSockets` 返回状态                                                  |
| RaSocket.white-list        | `RaSocketWhiteListAdd`, `RaSocketWhiteListDel`                           |
| RaQP.peer-qpn/peer-lid     | `RaQpConnectAsync` 交换对端信息                                          |
| RaQP.attr                  | `RaGetQpAttr`, `RaSetQpAttrQos`, `RaSetQpAttrTimeout`, `RaSetQpAttrRetryCnt` |
| RaQP.context               | `RaGetQpContext` 返回QP的send_cq和recv_cq                                |
| RaCQe.wr-id                | `RaSendWr`, `RaSendWrV2`, `RaSendWrlist`, `RaRecvWrlist` 设置             |
| RaCQe.status               | `RaPollCq` 返回                                                          |
| RaJetty.state              | `RaCtxQpQueryBatch` 返回                                                 |
| RaJetty.peer-jetty-handle  | `RaCtxQpBind`, `RaCtxQpImport` 设置                                      |
| RaMR.access                | `RaMrReg` 参数设置                                                       |
| RaLmem.token-id            | `RaCtxTokenIdAlloc` 预分配                                               |
| RaTp.tp-type               | `RaGetTpInfoListAsync` 返回                                              |
| RaContext.local-eid        | `RaGetDevEidInfoList`, `RaGetEidByIp` 获取                               |
| RaContext.mode             | `RaGetDevBaseAttr` 返回                                                  |
| AsyncRequest.status        | `RaGetAsyncReqResult` 返回                                               |
| InterfaceInfo.*            | `RaGetIfnum`, `RaGetIfaddrs`, `RaSocketGetVnicIpInfos` 获取              |
| HccnConfig.value           | `RaGetHccnCfg` 获取                                                      |
| RaNdaQP.nda-direct-flag    | `RaNdaGetDirectFlag` 检查NDA支持                                         |
| Snapshot.data              | `RaSaveSnapshot` 保存，`RaRestoreSnapshot` 恢复                          |
| RaSocketEvent.events       | `RaEpollCtlAdd`, `RaEpollCtlMod`, `RaEpollCtlDel` 管理                   |
| RaDevice.direct-flag       | `RaNdaGetDirectFlag` 检查NDA支持                                         |

## 5. 回调与报告关系建模

```mermaid
erDiagram
    Runner ||--o{ Context: "creates"
    Runner {
        typ run-id PK
    }

    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        int stream-id PK
        int ctx-id FK
        string state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        int task-id PK
        int stream-id FK
        typ seq-number "stream内自增"
        string type "Kernel/Memcpy/Callback"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    CallbackTask {
        typ report-id FK
        typ callback-fn
        typ user-data
    }  

    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ CallbackTask : "is a"

    ReportChannel {
        typ report-id
        typ stream-id
        typ run-id
    }  

    CallbackTask }o ..|| ReportChannel : "push"  
    ReportChannel ||--o{ Runner : "trigger and called by"

```

### 5.1 关键关系说明（回调与报告关系建模）

==**设备**==执行到 CallbackTask 时，触发 Host Runner 线程执行回调。

#### 5.1.1 配套接口映射表（回调与报告关系建模）

| 实体          | 关键管理接口                                                          |
| ------------- | --------------------------------------------------------------------- |
| CallbackTask  | `rtSetExceptionInfoCallback`, `rtLaunchCallback`,`rtSynchronizeEvent` |
| ReportChannel | `rtSubscribeReport`, `rtUnSubscribeReport`                            |
| Runner        | `rtProcessReport`                                                     |

## 6. 基础`设备`模型细粒度底层扩展

```mermaid
erDiagram

    Device ||--|| DeviceStatus : "has a"
    Device ||--|{ TaskSchedulerDevice : "has "
    Device {
        typ device-id PK
    }
    DeviceStatus {
        typ device-id FK
        typ overflow-status
        typ synchronize-strategy
        typ synchronize-timeout
        typ capability-mask
        typ run-by-host
        typ ts-core
        typ online-status
    }

    TaskSchedulerDevice  ||--|| Scalar :"is a"
    TaskSchedulerDevice  ||--|| CCU :"is a"
    TaskSchedulerDevice  ||--|| CPU :"is a"
    TaskSchedulerDevice {
        typ ts-id PK
        typ device-id FK
        typ type "Scalar"
    }

   CPU {
   }  

   Scalar ||--|| ComputeDie :"schedule"
   Scalar {
   }

   CCU {
        typ ccu-id PK
        typ ts-id FK
        typ version "v1/v2"
        typ xnNum
        typ ckeNum
        typ msNum
        typ channelNum
    }

    ComputeDie  ||--|| Cube :"is a"
    ComputeDie  ||--|| Vector :"is a"
    ComputeDie  ||--|| HybridComputeDie :"is a  (vector+cube)"
    ComputeDie {
        typ compute-id PK
        typ ts-id FK
        typ type
    }  
    Cube {
    }
    Vector {
    }
```

### 6.1 关键关系说明（基础`设备`模型细粒度底层扩展）

**设备调度器层次结构**：

- `TaskSchedulerDevice`是设备调度器的抽象，一个Device可包含多个调度器。
- 调度器类型包括：`Scalar`（标量处理器）、`CCU`（集合通信单元）、`CPU`（AI CPU）
- 不同类型调度器负责不同计算任务类型。

**ComputeDie计算单元**：

- `ComputeDie`是计算单元的抽象，继承关系表示计算单元类型。
- `Vector`：向量计算单元，处理向量运算。
- `Cube`：立方计算单元，处理矩阵运算。
- `HybridComputeDie`：混合计算单元，同时支持Vector和Cube。

**CCU内部结构**：

- `xnNum`：XN节点数量（跨节点通信）
- `ckeNum`：CKE引擎数量（Checksum引擎）
- `msNum`：MS模块数量（Memory Scheduler）
- `channelNum`：通信通道数量。
- `version`：CCU版本（v1/v2，决定功能差异）

**DeviceStatus状态管理**：

- `overflow-status`：溢出状态。
- `synchronize-strategy`：同步策略配置。
- `synchronize-timeout`：同步超时设置。
- `capability-mask`：设备能力掩码。
- `run-by-host`：是否运行在Host模式。
- `ts-core`：调度器核心数量。
- `online-status`：设备在线状态。

**Scalar调度关系**：

- `Scalar`调度器管理`ComputeDie`的执行。
- 不同ComputeDie类型对应不同的计算负载。

#### 6.1.1 配套接口映射表（基础设备模型细粒度底层扩展）

| 实体                | 关键管理接口                       |
| ------------------- | ---------------------------------- |
| TaskSchedulerDevice | `rtGetDeviceInfo`, `rtSetTsDevice` |
| DeviceStatus        | `rtGetRunMode`                     |

## 7. Kernel 运行时关系建模

```mermaid
erDiagram
    KernelBinary {
        typ id PK
        typ file
        typ create-pid
    }

    KernelBinary  ||--|| KernelBinaryHandle :"loaded"
    KernelBinaryHandle  ||--o{ KernelFuncHandle :"contains"
    KernelBinaryHandle {
        typ handle-id PK
        typ kernel-id FK
    }

    KernelFuncHandle  ||--o{ KernelFuncArgsHandle :"has a"
    KernelFuncHandle  }o--|| Task :"Called "
    KernelFuncHandle  }o--|| KernelLaunchCfg :"launch config"
    KernelFuncHandle {
        typ handle-id PK
        typ binary-id FK
        typ func-name
        typ kernel-name
        typ aic-addr
        typ aiv-addr
    }

    KernelFuncArgsHandle  ||--o{ KernelFuncArgsParamHandle :"append"
    KernelFuncArgsHandle {
        typ args-handle-id PK
        typ func-id FK
        typ args-size
        typ type "device/host"
    }

    KernelFuncArgsParamHandle {
        typ args-param-id PK
        typ args-id FK
        typ param-size
        typ is-place-holder
    }
```

### 7.1 关键关系说明（Kernel 运行时关系建模）

**KernelBinary加载流程**：

1. `KernelBinary`存储.o/.so等二进制文件路径和创建进程PID。
2. `rtBinaryLoadFromFile`或`rtBinaryLoadFromData`将二进制加载到设备内存。
3. 加载后生成`KernelBinaryHandle`，包含handle-id和kernel-id。
4. `KernelFuncHandle`表示二进制中的具体函数，包含func-name、kernel-name、地址信息。

**Kernel函数调用关系**：

- `aic-addr`是AI Core函数地址。
- `aiv-addr`是AI Vector函数地址。
- `KernelFuncArgsHandle`存储函数参数信息。
- `KernelFuncArgsParamHandle`记录参数大小和是否占位符。
- `KernelLaunchCfg`配置Kernel启动参数（block/grid等）

**Kernel生命周期**：

```text
rtCreateBinary -> rtBinaryLoad -> rtBinaryGetFunction 
                -> rtLaunchKernel(funcHandle, argsHandle, cfg)
                -> rtBinaryUnLoad -> rtDestroyBinary
```

**参数管理机制**：

- args-handle支持device和host两种类型。
- is-place-holder标识参数是否为占位符（延迟绑定）
- param-size记录单个参数大小。

#### 7.1.1 配套接口映射表（Kernel 运行时关系建模）

| 实体               | 关键管理接口                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| KernelBinary       | `rtCreateBinary`, `rtDestroyBinary`                                                                             |
| KernelBinaryHandle | `rtBinaryLoad`, `rtBinaryUnLoad`,`rtBinaryLoadFromFile`,`rtBinaryLoadFromData`                                  |
| KernelFuncHandle   | `rtBinaryGetFunction`, `rtBinaryGetFunctionByEntry`,`rtGetFunctionAddr`,`rtGetFunctionName`,`rtRegisterCpuFunc` |

## 8. 模型加载关系建模
