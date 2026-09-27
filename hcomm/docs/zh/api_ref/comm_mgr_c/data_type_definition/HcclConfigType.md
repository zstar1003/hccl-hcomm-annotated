# HcclConfigType

## 功能说明

配置通信算子的展开模式、通信算法配置字符串及UB多channel数量。

## 定义原型

```c
typedef enum {
    HCCL_CONFIG_TYPE_INVALID               = -1,   /* 无效配置项类型 */
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE     = 0,    /* 算子展开模式，对应类型为hcclOpExpansionMode */
    HCCL_CONFIG_TYPE_HCCL_ALGO             = 1,    /* 通信算法配置字符串，对应类型为长度HCCL_COMM_ALGO_MAX_LENGTH的char数组, 新增枚举字段向后兼容，不影响旧版本的代码 */
    HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM  = 2,    /* UB多channel数量，对应类型为uint32_t */
} HcclConfigType;
```
