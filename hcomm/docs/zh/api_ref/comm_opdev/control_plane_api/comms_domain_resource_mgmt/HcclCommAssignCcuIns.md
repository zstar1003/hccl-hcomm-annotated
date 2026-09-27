# HcclCommAssignCcuIns

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

将调用方已创建的CCU实例绑定到指定HCCL通信域。绑定成功后，通信域保存该实例句柄，并接管CCU实例的所有权和销毁责任；后续可通过`HcclCommQueryAssignedCcuIns`查询该实例。

每个通信域只能绑定一个CCU实例。通信域已有绑定实例时，本接口不会覆盖原实例。

## 函数原型

```c
HcclResult HcclCommAssignCcuIns(HcclComm comm, CcuInsHandle insHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域句柄，不能为空指针。HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| insHandle | 输入 | 待绑定的CCU实例句柄，不能为`0`。实例须由`HcommCcuInsCreate`或`HcommCcuInsCreateDefault`在当前Device上创建。CcuInsHandle类型的定义可参见[CcuInsHandle](../../datatype_definition/CcuInsHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回`HCCL_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `HCCL_SUCCESS` | 绑定成功，`insHandle`的所有权和销毁责任转移给通信域。 |
| `HCCL_E_PTR` | `comm`为空指针，或通信域内部对象为空。 |
| `HCCL_E_PARA` | `insHandle`为`0`，或通信域已经绑定CCU实例。 |
| `HCCL_E_NOT_SUPPORT` | 通信域代际不支持CCU（早于Ascend 950PR/Ascend 950DT）。 |
| `HCCL_E_NOT_FOUND` | 当前Device上不存在`insHandle`对应的CCU实例。 |
| `HCCL_E_INTERNAL` | 发生其他内部错误。 |

## 约束说明

- 绑定成功后，调用方不得再调用`HcommCcuInsDestroy`销毁`insHandle`。通信域销毁时会销毁已绑定的CCU实例。
- 绑定失败时，`insHandle`的所有权仍归调用方，调用方负责继续使用或销毁该实例。
- 同一个`insHandle`只能绑定到一个通信域，不得重复绑定到其他通信域。
- 同一个通信域不支持重复绑定。无论新旧句柄是否相同，再次调用本接口均返回`HCCL_E_PARA`，原绑定关系保持不变。
- 本接口只保证多个`HcclCommAssignCcuIns`调用之间的并发安全。调用方须保证本接口不与`HcclCommQueryCcuIns`、`HcclCommQueryAssignedCcuIns`或`HcclCommDestroy`并发执行。
- `insHandle`须在`comm`所在的当前Device上创建。

## 调用示例

```c
CcuInsHandle insHandle = 0;
// dieIds和dieNum为保留参数，当前版本分别传入NULL和0。
CcuResult ccuRet = HcommCcuInsCreateDefault(NULL, 0, &insHandle);
if (ccuRet != CCU_SUCCESS) {
    return ccuRet;
}

HcclResult hcclRet = HcclCommAssignCcuIns(comm, insHandle);
if (hcclRet != HCCL_SUCCESS) {
    // 绑定失败，实例所有权仍归调用方。
    HcommCcuInsDestroy(insHandle);
    return hcclRet;
}

// 绑定成功，实例所有权已转移给通信域，调用方不再销毁insHandle。
CcuInsHandle queriedInsHandle = 0;
uint32_t insNum = 0;
hcclRet = HcclCommQueryAssignedCcuIns(comm, &queriedInsHandle, &insNum);
if (hcclRet != HCCL_SUCCESS || insNum != 1 ||
    queriedInsHandle != insHandle) {
    HcclCommDestroy(comm);
    return HCCL_E_INTERNAL;
}

// 销毁通信域时，由通信域销毁已绑定的CCU实例。
return HcclCommDestroy(comm);
```
