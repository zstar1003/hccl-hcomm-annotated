# HcclOpDesc

## 功能说明

用于描述AICPU运行核函数时的算子信息，包含算子类型、名称和通信相关参数。该结构体用于自定义通信算子场景，支持Send和Receive通信任务的描述。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union {
        uint8_t raws[76];
        HcclOpP2pDesc p2p;
    };
} HcclOpDesc;
```

## 参数说明

- **header**：ABI头部，包含版本等信息。类型定义请参见[CommAbiHeader](../../comm_opdev/datatype_definition/CommAbiHeader.md) 。
- **opDescType**：算子描述类型。
- **opName**：算子名称，最大长度为256字节（HCCL_OP_DESC_OP_NAME_MAX_LEN）。
- **raws**：原始数据，用于通用场景，长度76字节。
- **p2p**：Send和Receive通信任务的描述参数，类型定义请参见[HcclOpP2pDesc](./HcclOpP2pDesc.md)，作为联合体成员与raws共用76字节空间。

## 相关常量

```c
const uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;  // 算子名称最大长度
```
