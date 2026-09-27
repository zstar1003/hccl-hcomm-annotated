# HcclOpP2pDesc

## 功能说明

用于描述Send和Receive通信任务的详细信息，包含数据缓冲区地址、通信命令类型、数据类型、数据个数、对端rank编号以及展开流等参数。该结构体作为[HcclOpDesc](./HcclOpDesc.md)的联合体成员使用，与`raws`共用76字节空间。

## 定义原型

```c
typedef struct {
    void *buffer;
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    void *unfoldStream;
} HcclOpP2pDesc;
```

## 参数说明

- **buffer**：数据缓冲区地址，用于发送或接收数据的内存地址。需确保内存已正确分配且可访问。
- **reserved**：预留字段，长度8字节，用于未来扩展。
- **cmdType**：通信命令类型，指定通信操作类型（如HCCL_CMD_SEND、HCCL_CMD_RECEIVE等）。HcclCMDType类型的定义可参见[HcclCMDType](./HcclCMDType.md)。
- **dataType**：数据类型，类型定义请参见[HcclDataType](./HcclDataType.md)。
- **count**：数据个数，指定需要传输的数据元素数量。需与dataType匹配buffer的实际数据大小和类型。
- **remoteRank**：对端rank编号，指定通信的对端节点编号。需在通信域的有效rank编号范围内。
- **unfoldStream**：展开流，用于AICPU通信任务的流控制。
