# CommProtocol

## 功能说明

定义通信协议类型枚举。

## 定义原型

```c
typedef enum {
    COMM_PROTOCOL_RESERVED = -1,  /* 保留协议类型 */
    COMM_PROTOCOL_HCCS = 0,       /* HCCS协议 */
    COMM_PROTOCOL_ROCE = 1,       /* RDMA over Converged Ethernet */
    COMM_PROTOCOL_PCIE = 2,       /* PCIE协议 */
    COMM_PROTOCOL_SIO = 3,        /* SIO协议 */
    COMM_PROTOCOL_UB_CTP = 4,     /* 华为统一总线UB_CTP */
    COMM_PROTOCOL_UBC_CTP = COMM_PROTOCOL_UB_CTP, /* 兼容旧协议名UBC_CTP */
    COMM_PROTOCOL_UBC_TP = 5,     /* 历史兼容协议UBC_TP，不建议新开发使用 */
    COMM_PROTOCOL_UB_MEM = 6,     /* UB_MEM */
    COMM_PROTOCOL_UBOE = 7,       /* UBoE */
    COMM_PROTOCOL_HCCS_ONLY = 8,  /* 一卡双DIE使用HCCS */
    COMM_PROTOCOL_UB_RTP = 9,     /* UB_RTP */
    COMM_PROTOCOL_UBG = COMM_PROTOCOL_UB_RTP, /* 兼容旧协议名UBG */
} CommProtocol;
```

## 产品支持情况

<!-- npu="950" id1 -->
针对Ascend 950PR/Ascend 950DT，各通信引擎支持的通信协议如下：

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UB_CTP
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_UB_RTP
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_CCU
    - COMM_PROTOCOL_UB_CTP
<!-- end id1 -->
    
<!-- npu="A3" id2 -->
针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，各通信引擎支持的通信协议如下：

  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY
<!-- end id2 -->
  
<!-- npu="910b" id3 -->
针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，各通信引擎支持的通信协议如下：

  - COMM_ENGINE_CPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY
<!-- end id3 -->
