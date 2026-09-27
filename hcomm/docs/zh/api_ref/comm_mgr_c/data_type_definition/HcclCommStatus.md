# HcclCommStatus

## 功能说明

通信域状态信息。

## 定义原型

```c
typedef enum {
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDING = 1,
    HCCL_COMM_STATUS_INVALID = 254,
    HCCL_COMM_STATUS_RESERVED = 255
} HcclCommStatus;
```
