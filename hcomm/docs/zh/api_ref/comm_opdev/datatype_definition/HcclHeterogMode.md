# HcclHeterogMode

## 功能说明

定义异构组网模式。

## 定义原型

```c
typedef enum {
    HCCL_HETEROG_MODE_INVALID = -1,    /* 无效/未初始化 */
    HCCL_HETEROG_MODE_HOMOGENEOUS = 0, /* 同构组网：单一AI处理器类型 */
    HCCL_HETEROG_MODE_MIX_A2_A3,       /* 异构组网：Atlas A2系列产品和Atlas A3系列产品混合，当前暂不支持此模式组网 */
} HcclHeterogMode;
```
