# 总体流程

HCCL AI CPU通信算子的开发与执行分为Host侧准备和AICPU侧执行两个阶段。Host侧负责解析拓扑、选择算法、申请资源并下发AICPU Kernel；Kernel启动后，AICPU侧才根据资源上下文进行任务编排。因此，***Kernel下发在前，任务编排在后***。

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontFamily": "Microsoft YaHei, Arial, sans-serif",
    "primaryColor": "#EAF2FF",
    "primaryBorderColor": "#4F7CAC",
    "primaryTextColor": "#1F2937",
    "actorBorder": "#4F7CAC",
    "actorBkg": "#F8FBFF",
    "activationBkgColor": "#DCEBFA",
    "activationBorderColor": "#4F7CAC",
    "sequenceNumberColor": "#6B7280",
    "noteBkgColor": "#FFF7D6",
    "noteBorderColor": "#D6A700"
  }
}}%%
sequenceDiagram
    title HCCL AI CPU通信算子执行流程

    participant HCCL_H as 通信算子接口(Host)
    participant HCOMM_H as HCOMM(Host API)
    participant RTS as RTS
    participant HCCL_K as AICPU Kernel
    participant HCOMM_D as HCOMM(Device API)

    activate HCCL_H

    Note over HCCL_H: Host侧准备算子执行信息
    HCCL_H->>HCCL_H: 构造算子参数（输入输出地址、数据量、数据类型等）
    HCCL_H->>+HCOMM_H: 查询rank、拓扑层级和可用链路
    HCOMM_H-->>-HCCL_H: 返回通信域与拓扑信息

    opt 算子存在多种实现
        HCCL_H->>HCCL_H: 根据算子类型、执行引擎、拓扑和数据量选择算法
    end

    Note over HCCL_H,HCOMM_H: 资源以算子和算法Tag为粒度复用
    HCCL_H->>+HCOMM_H: HcclEngineCtxGet(tag, engine)
    alt 资源已存在
        HCOMM_H-->>HCCL_H: 返回Device Context
    else 首次执行
        HCOMM_H-->>HCCL_H: Context不存在
        HCCL_H->>HCOMM_H: 创建Context
        HCCL_H->>HCOMM_H: 申请并导出Host/AICPU控制Thread
        HCCL_H->>HCOMM_H: 申请算法Thread、Notify、Channel和通信内存
        HCCL_H->>HCOMM_H: 序列化资源上下文并拷贝到Device
        HCOMM_H-->>HCCL_H: 返回Device Context
    end
    deactivate HCOMM_H

    Note over HCCL_H,HCCL_K: Host先下发Kernel，任务编排在Kernel启动后执行
    HCCL_H->>+HCOMM_H: Host Thread通知AICPU控制Thread
    HCOMM_H-->>-HCCL_H: HcclResult
    HCCL_H->>+RTS: aclrtLaunchKernelWithConfig下发AICPU Kernel
    RTS-->>HCCL_H: aclResult
    RTS->>HCCL_K: 启动Kernel
    deactivate RTS
    HCCL_H->>+HCOMM_H: Host Thread等待AICPU完成通知

    activate HCCL_K
    HCCL_K->>HCCL_K: 反序列化资源Context
    HCCL_K->>+HCOMM_D: HcommBatchModeStart(tag)
    HCOMM_D-->>-HCCL_K: HcclResult
    HCCL_K->>+HCOMM_D: AICPU 控制Thread等待Host启动通知
    HCOMM_D-->>-HCCL_K: HcclResult

    Note over HCCL_K,HCOMM_D: Kernel内执行算法任务编排
    HCCL_K->>HCCL_K: ExecOp(param, resCtx)
    HCCL_K->>+HCOMM_D: 编排Thread同步、Channel同步和数据搬运任务
    HCOMM_D-->>-HCCL_K: HcclResult

    HCCL_K->>+HCOMM_D: AICPU控制Thread通知Host Thread
    HCOMM_D-->>-HCCL_K: HcclResult
    HCCL_K->>+HCOMM_D: HcommBatchModeEnd(tag)
    HCOMM_D-->>-HCCL_K: HcclResult
    deactivate HCCL_K

    HCOMM_H-->>-HCCL_H: AICPU执行完成
    deactivate HCCL_H
```

各阶段的主要职责如下：

1. ***定义算子接口***：明确输入输出、数据量、数据类型、通信域和执行流等信息。
2. ***查询拓扑信息***：获取rank数量、拓扑层级、层内连接关系和可用链路，为算法选择和资源计算提供依据。
3. ***算法选择***：根据算子类型、执行引擎、拓扑形态、数据量和数据类型等条件选择已注册的算法实现。只有一种固定实现的自定义算子可以省略该步骤。
4. ***创建资源***：计算并申请Thread、Notify、Channel、通信内存和资源Context，并将AICPU执行所需的上下文拷贝到Device。
5. ***下发Kernel***：Host与AICPU控制Thread建立启动同步关系，然后将AICPU Kernel 下发到执行流。
6. ***任务编排***：AICPU Kernel启动并取得资源上下文后，调用算法执行逻辑，将Thread同步、Channel同步、数据搬运等操作编排到对应Thread上。
7. ***完成同步***：AICPU侧完成编排后通知Host，Host侧等待该通知，保证通信任务与业务流的执行顺序。
