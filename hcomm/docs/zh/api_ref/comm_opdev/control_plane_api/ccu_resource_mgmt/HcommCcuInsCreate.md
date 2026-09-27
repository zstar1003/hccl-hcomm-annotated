# HcommCcuInsCreate

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

根据一个或多个CCU资源描述符创建CCU实例，并返回实例句柄。每个资源描述符对应一个IO Die，用于指定该Die上各类CCU资源的申请数量。

底层按资源对齐粒度连续分配资源。当申请数量不是对齐粒度的整数倍时，实际分配数量可能向上取整。可通过[HcommCcuInsQueryResDesc](HcommCcuInsQueryResDesc.md)查询实例实际占用的资源数量。

本接口不读取资源描述符中`HCOMM_CCU_RES_TYPE_INSTRUCTION`对应的资源数量，也不申请Instruction资源。Instruction资源在注册CCU Kernel期间，根据Kernel实际指令数量申请。

`HCOMM_CCU_RES_TYPE_CCU_THREAD`对应的Mission资源按融合多Die模式申请：取各资源描述符中Mission申请数量的最大值，在当前Device的所有已使能IO Die上申请相同数量且ID范围相同的Mission资源；仅一个IO Die使能时只在该Die上申请。

## 函数原型

```c
CcuResult HcommCcuInsCreate(const HcommCcuResDescHandle *resDescs, uint32_t resDescNum, CcuInsHandle *ccuInsHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| resDescs | 输入 | 资源描述符句柄数组，不能为空指针。数组中的句柄须由[HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md)在当前Device上创建，每个句柄对应一个不同的IO Die。 |
| resDescNum | 输入 | `resDescs`中的句柄数量，取值范围为`(0, CCU_MAX_IODIE_NUM]`，当前最大值为2。 |
| ccuInsHandle | 输出 | 创建成功后返回的CCU实例句柄，不能为空指针。类型定义可参见[CcuInsHandle](../../datatype_definition/CcuInsHandle.md)。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 创建成功，`*ccuInsHandle`为有效的CCU实例句柄。 |
| `CCU_E_PTR` | `resDescs`或`ccuInsHandle`为空指针，或数组中的资源描述符句柄不属于当前Device。 |
| `CCU_E_PARA` | `resDescNum`超出合法范围，或多个资源描述符对应相同的IO Die。 |
| `CCU_E_UNAVAIL` | 当前Device上的连续资源不足，无法满足资源描述符中的申请数量。 |
| `CCU_E_DRV_BUSY` | 单卡多进程场景下，CCU驱动已被其他进程拉起。 |
| `CCU_E_INTERNAL` | CCU实例创建、资源申请或内部登记失败。 |
| 其他错误码 | 初始化CCU驱动或申请资源时产生的其他错误。 |

## 约束说明

- 当前仅支持集合通信场景，依赖通信域做资源管理。
- 调用本接口的线程须通过AscendCL接口`aclrtSetDevice(int32_t deviceId)`绑定到目标NPU Device。`resDescs`中的所有描述符须在同一Device上创建。
- `resDescs`中的IO Die编号不能重复。
- 创建过程中，调用方不得并发修改或销毁`resDescs`中的资源描述符。创建成功后，CCU实例独立持有已申请的资源，原资源描述符可继续使用或销毁。
- 创建成功后，实例所有权归调用方。调用方应通过[HcommCcuInsDestroy](HcommCcuInsDestroy.md)销毁实例；若通过[HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md)成功绑定到通信域，则所有权和销毁责任转移给通信域。
- 本接口只能在主机侧调用。

## 调用示例

```c
HcommCcuResDescHandle resDesc = 0;
CcuInsHandle insHandle = 0;

CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

ret = HcommCcuInsResDescSetNum(
    resDesc, HCOMM_CCU_RES_TYPE_LOOP, 8);
if (ret == CCU_SUCCESS) {
    ret = HcommCcuInsCreate(&resDesc, 1, &insHandle);
}

HcommCcuInsResDescDestroy(resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// 使用insHandle注册Kernel，或将其绑定到通信域。
// ...

return HcommCcuInsDestroy(insHandle);
```
