# HcommCcuResType

## 功能说明

CCU资源类型枚举，用于[HcommCcuInsResDescSetNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescSetNum.md)和[HcommCcuInsResDescQueryNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescQueryNum.md)接口中指定资源描述符（[HcommCcuResDescHandle](HcommCcuResDescHandle.md)）内的资源类型。

## 定义原型

```c
typedef enum {
    HCOMM_CCU_RES_TYPE_INVALID = -1,
    HCOMM_CCU_RES_TYPE_LOOP = 0,
    HCOMM_CCU_RES_TYPE_CCU_BUF = 1,
    HCOMM_CCU_RES_TYPE_VARIABLE = 2,
    HCOMM_CCU_RES_TYPE_ADDRESS = 3,
    HCOMM_CCU_RES_TYPE_EVENT = 4,
    HCOMM_CCU_RES_TYPE_CCU_THREAD = 5,
    HCOMM_CCU_RES_TYPE_INSTRUCTION = 6
} HcommCcuResType;
```

## 字段说明

| 字段 | 值 | 说明 |
| --- | --- | --- |
| HCOMM_CCU_RES_TYPE_INVALID | -1 | 无效资源类型。 |
| HCOMM_CCU_RES_TYPE_LOOP | 0 | Loop资源。 |
| HCOMM_CCU_RES_TYPE_CCU_BUF | 1 | CCU Buffer资源，即Kernel内通过[CcuBuffer](../data_plane_api/ccu/resource_allocation_operation/CcuBuffer.md)创建的片上高速暂存区。 |
| HCOMM_CCU_RES_TYPE_VARIABLE | 2 | Variable资源，即Kernel内通过[Variable](../data_plane_api/ccu/resource_allocation_operation/Variable.md)创建的变量资源。 |
| HCOMM_CCU_RES_TYPE_ADDRESS | 3 | Address资源，即Kernel内通过[Address](../data_plane_api/ccu/resource_allocation_operation/Address.md)创建的地址资源。 |
| HCOMM_CCU_RES_TYPE_EVENT | 4 | Event资源，即Kernel内通过[Event](../data_plane_api/ccu/resource_allocation_operation/Event.md)创建的事件资源。 |
| HCOMM_CCU_RES_TYPE_CCU_THREAD | 5 | CCU Thread资源。 |
| HCOMM_CCU_RES_TYPE_INSTRUCTION | 6 | Instruction资源。 |
