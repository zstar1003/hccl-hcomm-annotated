# HcclP2pKernelParam

## 功能说明

用于描述P2P（点对点）通信任务的核函数参数，包含收发通信线程句柄和操作参数缓存。该结构体在AICPU核函数启动时用于传递P2P通信所需的上下文信息。

## 定义原型

```c
const uint32_t P2P_MAX_ARG_SIZE = 8192U;

typedef struct {
    ThreadHandle sendRecvThread;
    uint8_t opParams[P2P_MAX_ARG_SIZE];
} HcclP2pKernelParam;
```

## 参数说明

- **sendRecvThread**：收发通信线程句柄，用于标识执行点对点收发操作的通信线程，类型定义请参见[ThreadHandle](../../comm_opdev/datatype_definition/ThreadHandle.md)。
- **opParams**：操作参数缓存，用于存储核函数执行所需的参数数据，最大长度为`P2P_MAX_ARG_SIZE`（8192字节）。实际使用的参数长度由具体的通信操作决定。
