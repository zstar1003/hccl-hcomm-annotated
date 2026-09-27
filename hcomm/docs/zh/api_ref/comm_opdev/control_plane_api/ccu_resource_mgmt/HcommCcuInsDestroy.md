# HcommCcuInsDestroy

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

销毁调用方持有的CCU实例，注销该实例关联的Kernel并释放实例占用的CCU资源。若实例上存在通过[HcommCcuVariableAlloc](HcommCcuVariableAlloc.md)/[HcommCcuEventAlloc](HcommCcuEventAlloc.md)预约的资源，销毁时预约句柄一并失效。销毁成功后，实例句柄失效。

## 函数原型

```c
CcuResult HcommCcuInsDestroy(CcuInsHandle ccuInsHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccuInsHandle | 输入 | 待销毁的CCU实例句柄，须由[HcommCcuInsCreate](HcommCcuInsCreate.md)或[HcommCcuInsCreateDefault](HcommCcuInsCreateDefault.md)在当前Device上创建。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 销毁成功，CCU实例句柄失效。 |
| `CCU_E_NOT_FOUND` | 当前Device上不存在`ccuInsHandle`对应的CCU实例，或该实例已被销毁。 |

## 约束说明

- 调用本接口的线程须绑定到创建该实例时使用的同一NPU Device。
- 仅当实例所有权仍归调用方时才可调用本接口。
- 通过[HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md)成功绑定到通信域后，实例所有权和销毁责任已转移给通信域，调用方不得再调用本接口。
- 销毁成功后不得继续使用`ccuInsHandle`。重复销毁返回`CCU_E_NOT_FOUND`。
- 调用方须保证没有其他线程正在使用该实例。
- 本接口只能在主机侧调用。

## 调用示例

```c
CcuInsHandle insHandle = 0;
CcuResult ret = HcommCcuInsCreateDefault(NULL, 0, &insHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// 使用insHandle注册和执行Kernel。
// ...

ret = HcommCcuInsDestroy(insHandle);
return ret;
```
