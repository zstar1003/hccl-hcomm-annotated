# HcommReduceOp

## 功能说明

定义归约操作类型。

## 定义原型

```c
typedef enum {
    HCOMM_REDUCE_SUM = 0,    /* sum */
    HCOMM_REDUCE_PROD = 1,   /* prod */
    HCOMM_REDUCE_MAX = 2,    /* max */
    HCOMM_REDUCE_MIN = 3,    /* min */
    HCOMM_REDUCE_RESERVED = 255  /* reserved */
} HcommReduceOp;
```
