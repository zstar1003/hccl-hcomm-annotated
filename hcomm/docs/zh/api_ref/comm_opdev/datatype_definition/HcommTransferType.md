# HcommTransferType

## 功能说明

定义批量传输操作中的传输类型枚举，用于[HcommBatchTransferDesc](HcommBatchTransferDesc.md)结构体中的transType字段，指定单个传输描述符所执行的操作类型。

## 定义原型

```c
typedef enum {
    HCOMM_TRANSFER_TYPE_INVALID = -1,                       /* 无效传输类型 */
    HCOMM_TRANSFER_TYPE_WRITE = 0,                          /* 单边写，对应transferInfo.write */
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE = 1,                   /* 单边写归约，对应transferInfo.reduce */
    HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,              /* 带通知的单边写，对应transferInfo.writeWithNotify */
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3,       /* 带通知的单边写归约，对应transferInfo.writeReduceWithNotify */
    HCOMM_TRANSFER_TYPE_READ = 4,                           /* 单边读，对应transferInfo.read */
    HCOMM_TRANSFER_TYPE_READ_REDUCE = 5,                    /* 单边读归约，对应transferInfo.reduce */
    HCOMM_TRANSFER_TYPE_NOTIFY_RECORD = 6                   /* 记录通知事件，对应transferInfo.notifyRecord */
} HcommTransferType;
```
