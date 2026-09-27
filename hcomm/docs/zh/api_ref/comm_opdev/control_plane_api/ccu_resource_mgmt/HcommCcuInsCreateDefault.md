# HcommCcuInsCreateDefault

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

使用当前Device上所有已使能IO Die的全部CCU资源创建CCU实例，并返回实例句柄。

本接口不申请Instruction资源。Instruction资源在注册CCU Kernel期间，根据Kernel实际指令数量申请。

当前版本不读取`dieIds`和`dieNum`，调用方应分别传入`NULL`和`0`。需要按需申请资源时，请使用[HcommCcuInsCreate](HcommCcuInsCreate.md)。

## 函数原型

```c
CcuResult HcommCcuInsCreateDefault(const uint32_t *dieIds, uint32_t dieNum, CcuInsHandle *ccuInsHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dieIds | 输入 | 保留参数，当前版本不读取，应传入`NULL`。 |
| dieNum | 输入 | 保留参数，当前版本不读取，应传入`0`。 |
| ccuInsHandle | 输出 | 创建成功后返回的CCU实例句柄，不能为空指针。类型定义可参见[CcuInsHandle](../../datatype_definition/CcuInsHandle.md)。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 创建成功，`*ccuInsHandle`为有效的CCU实例句柄。 |
| `CCU_E_PTR` | `ccuInsHandle`为空指针。 |
| `CCU_E_UNAVAIL` | 当前Device上的资源不足，无法申请全部资源。 |
| `CCU_E_DRV_BUSY` | 单卡多进程场景下，CCU驱动已被其他进程拉起。 |
| `CCU_E_INTERNAL` | CCU实例创建、资源查询、资源申请或内部登记失败。 |
| 其他错误码 | 初始化CCU驱动、查询IO Die状态或申请资源时产生的其他错误。 |

## 约束说明

- 当前仅支持集合通信场景，依赖通信域做资源管理。
- 调用本接口的线程须通过AscendCL接口`aclrtSetDevice(int32_t deviceId)`绑定到目标NPU Device。
- 本接口申请当前Device上所有已使能IO Die的全部CCU资源。若需与其他CCU实例共享资源，应改用[HcommCcuInsCreate](HcommCcuInsCreate.md)按需申请。
- 创建成功后，实例所有权归调用方。调用方应通过[HcommCcuInsDestroy](HcommCcuInsDestroy.md)销毁实例；若通过[HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md)成功绑定到通信域，则所有权和销毁责任转移给通信域。
- 本接口只能在主机侧调用。

## 调用示例

```c
CcuInsHandle insHandle = 0;
CcuResult ret = HcommCcuInsCreateDefault(NULL, 0, &insHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

HcclResult hcclRet = HcclCommAssignCcuIns(comm, insHandle);
if (hcclRet != HCCL_SUCCESS) {
    // 绑定失败，实例所有权仍归调用方。
    HcommCcuInsDestroy(insHandle);
    return CCU_E_INTERNAL;
}

// 绑定成功后由通信域管理实例生命周期。
return CCU_SUCCESS;
```
