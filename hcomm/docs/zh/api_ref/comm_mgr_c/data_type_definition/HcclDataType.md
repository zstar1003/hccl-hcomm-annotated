# HcclDataType

## 功能说明

定义集合通信相关操作的数据类型。

## 定义原型

```c
typedef enum {
    HCCL_DATA_TYPE_INT8 = 0,     /* int8 */
    HCCL_DATA_TYPE_INT16 = 1,    /* int16 */
    HCCL_DATA_TYPE_INT32 = 2,    /* int32 */
    HCCL_DATA_TYPE_FP16 = 3,     /* fp16 */
    HCCL_DATA_TYPE_FP32 = 4,     /* fp32 */
    HCCL_DATA_TYPE_INT64 = 5,    /* int64 */
    HCCL_DATA_TYPE_UINT64 = 6,   /* uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,    /* uint8 */
    HCCL_DATA_TYPE_UINT16 = 8,   /* uint16 */
    HCCL_DATA_TYPE_UINT32 = 9,   /* uint32 */
    HCCL_DATA_TYPE_FP64 = 10,    /* fp64 */
    HCCL_DATA_TYPE_BFP16 = 11,   /* bfp16 */
    HCCL_DATA_TYPE_INT128 = 12,  /* int128 */
    HCCL_DATA_TYPE_HIF8 = 14,     /* hif8 */
    HCCL_DATA_TYPE_FP8E4M3 = 15,  /* fp8e4m3 */
    HCCL_DATA_TYPE_FP8E5M2 = 16,  /* fp8e5m2 */
    HCCL_DATA_TYPE_FP8E8M0 = 17,  /* fp8e8m0 */
    HCCL_DATA_TYPE_RESERVED =  255  /* reserved */
} HcclDataType;
```
