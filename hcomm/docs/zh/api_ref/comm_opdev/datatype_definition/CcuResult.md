# CcuResult

## 功能说明

CCU接口的返回值类型，用于表示操作的执行结果。所有CCU控制面接口和数据面接口均以此类型返回操作状态。

## 定义原型

```c
typedef enum {
    CCU_SUCCESS = 0,
    CCU_E_PARA = 1,
    CCU_E_PTR = 2,
    CCU_E_INTERNAL = 4,
    CCU_E_NOT_SUPPORT = 5,
    CCU_E_NOT_FOUND = 6,
    CCU_E_UNAVAIL = 7,
    CCU_E_RUNTIME = 15,
    CCU_E_DRV_START = 4096,
    CCU_E_DRV_INIT_FAILED = 4097,
    CCU_E_DRV_BUSY = 4098,
    CCU_E_DRV_END = 4224,
    CCU_E_RESERVED = 9216
} CcuResult;
```

## 枚举值说明

| 枚举值 | 数值 | 说明 |
| --- | --- | --- |
| `CCU_SUCCESS` | 0 | 操作成功。 |
| `CCU_E_PARA` | 1 | 参数错误。 |
| `CCU_E_PTR` | 2 | 空指针错误。 |
| `CCU_E_INTERNAL` | 4 | 内部错误。 |
| `CCU_E_NOT_SUPPORT` | 5 | 不支持的特性。 |
| `CCU_E_NOT_FOUND` | 6 | 未找到指定资源。 |
| `CCU_E_UNAVAIL` | 7 | 资源不可用。 |
| `CCU_E_RUNTIME` | 15 | 运行时错误。 |
| `CCU_E_DRV_START` | 4096 | 驱动层错误码段起始标记（不作为实际错误码返回）。 |
| `CCU_E_DRV_INIT_FAILED` | 4097 | 驱动初始化失败。 |
| `CCU_E_DRV_BUSY` | 4098 | 驱动忙。 |
| `CCU_E_DRV_END` | 4224 | 驱动层错误码段结束标记（不作为实际错误码返回）。 |
| `CCU_E_RESERVED` | 9216 | 保留错误码段起始标记（不作为实际错误码返回，预留用于后续扩展）。 |
