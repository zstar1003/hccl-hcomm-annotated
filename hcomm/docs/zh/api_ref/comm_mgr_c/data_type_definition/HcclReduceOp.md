# HcclReduceOp

## 功能说明

定义集合通信reduce操作的类型。

## 定义原型

```c
typedef enum {
    HCCL_REDUCE_SUM = 0,    /* sum */
    HCCL_REDUCE_PROD = 1,   /* prod */
    HCCL_REDUCE_MAX = 2,    /* max */
    HCCL_REDUCE_MIN = 3,    /* min */
    HCCL_REDUCE_RESERVED = 255 /* reserved */
} HcclReduceOp;
```
