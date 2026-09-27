# HcommResult

## 功能说明

HCOMM基础通信接口的返回值类型。

## 定义原型

```c
typedef int32_t HcommResult;
```

## 说明

HcommResult为int32_t类型，接口成功返回0，其他值表示失败。其错误码与HcclResult保持一致，具体错误码定义可参见[HcclResult](../../comm_mgr_c/data_type_definition/HcclResult.md)。
