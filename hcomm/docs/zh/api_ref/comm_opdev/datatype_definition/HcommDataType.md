# HcommDataType

## 功能说明

定义数据类型。

## 定义原型

```c
typedef enum {
    HCOMM_DATA_TYPE_INT8 = 0,      /* int8 */
    HCOMM_DATA_TYPE_INT16 = 1,     /* int16 */
    HCOMM_DATA_TYPE_INT32 = 2,     /* int32 */
    HCOMM_DATA_TYPE_FP16 = 3,      /* fp16 */
    HCOMM_DATA_TYPE_FP32 = 4,      /* fp32 */
    HCOMM_DATA_TYPE_INT64 = 5,     /* int64 */
    HCOMM_DATA_TYPE_UINT64 = 6,    /* uint64 */
    HCOMM_DATA_TYPE_UINT8 = 7,     /* uint8 */
    HCOMM_DATA_TYPE_UINT16 = 8,    /* uint16 */
    HCOMM_DATA_TYPE_UINT32 = 9,    /* uint32 */
    HCOMM_DATA_TYPE_FP64 = 10,     /* fp64 */
    HCOMM_DATA_TYPE_BFP16 = 11,    /* bfp16 */
    HCOMM_DATA_TYPE_INT128 = 12,   /* int128 */
    HCOMM_DATA_TYPE_HIF8 = 14,     /* hif8 */
    HCOMM_DATA_TYPE_FP8E4M3 = 15,  /* fp8e4m3 */
    HCOMM_DATA_TYPE_FP8E5M2 = 16,  /* fp8e5m2 */
    HCOMM_DATA_TYPE_FP8E8M0 = 17,  /* fp8e8m0 */
    HCOMM_DATA_TYPE_RESERVED = 255 /* reserved */
} HcommDataType;
```
