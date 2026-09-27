# HcclCommQueryAssignedCcuIns

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

查询指定HCCL通信域通过新方式绑定的CCU实例句柄，返回实例句柄用于后续CCU Kernel的注册与下发。

新方式指：调用方先通过`HcommCcuInsCreate`或`HcommCcuInsCreateDefault`创建CCU实例，再通过[HcclCommAssignCcuIns](HcclCommAssignCcuIns.md)将实例绑定到通信域。本接口查询的就是该绑定关系所对应的实例句柄（assignedCcuInsHandle）。查询成功时固定返回1个实例：insHandles[0]为已绑定的实例句柄，*insNum为1。

本接口不会创建CCU实例。若通信域尚未通过`HcclCommAssignCcuIns`绑定任何CCU实例，本接口返回HCCL_E_UNAVAIL。

> [!NOTE]注意
> - 本接口查询的是通过`HcclCommAssignCcuIns`绑定的CCU实例，与通信域自有的CCU实例（由[HcclCommQueryCcuIns](HcclCommQueryCcuIns.md)查询/创建）相互独立。若需查询通信域自有的实例，请使用`HcclCommQueryCcuIns`。
> - 查询结果仅供借用，不转移实例所有权。已绑定的CCU实例所有权归通信域，由通信域负责释放；调用者不可销毁该实例，否则会造成对同一实例的重复释放，破坏通信域的资源管理。

## 函数原型

```c
HcclResult HcclCommQueryAssignedCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄，不可为nullptr。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| insHandles | 输出 | CCU实例句柄数组，不可为nullptr。查询成功后insHandles[0]返回已绑定的CCU实例句柄。<br>调用方需分配至少1个CcuInsHandle元素的空间。<br>CcuInsHandle类型的定义如下：<br>typedef uint64_t CcuInsHandle; |
| insNum | 输出 | CCU实例数量，不可为nullptr。查询成功后返回值固定为1。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

- 当comm、insHandles或insNum为nullptr时，返回HCCL_E_PTR。
- 当通信域代际不支持CCU（早于Ascend 950PR/Ascend 950DT）时，返回HCCL_E_NOT_SUPPORT。
- 当通信域尚未通过`HcclCommAssignCcuIns`绑定CCU实例时，返回HCCL_E_UNAVAIL，本接口不会创建实例。

## 约束说明

1. CCU特性仅在Ascend 950PR/Ascend 950DT及以上代际支持，在更早代际的通信域上调用返回HCCL_E_NOT_SUPPORT。
2. 调用本接口前，须已通过`HcommCcuInsCreate`或`HcommCcuInsCreateDefault`创建CCU实例，并通过`HcclCommAssignCcuIns`绑定到通信域；否则返回HCCL_E_UNAVAIL。
3. 本接口查询的CCU实例与通信域自有的CCU实例（由`HcclCommQueryCcuIns`查询/创建）相互独立，二者不会相互覆盖或影响。
4. 返回的CCU实例句柄仅供借用，所有权仍归通信域所有，由通信域负责释放。调用者不可销毁该实例，否则会造成对同一实例的重复释放，破坏通信域的资源管理。
5. 本接口不保证与`HcclCommAssignCcuIns`或`HcclCommDestroy`并发执行的安全。调用方须保证本接口不与`HcclCommAssignCcuIns`或`HcclCommDestroy`并发执行。

## 调用示例

```c
// comm 需通过 HcclCommInitClusterInfo 等接口完成初始化，此处仅为示例占位
HcclComm comm = /* 已初始化的通信域句柄 */;

// 1. 新方式：先创建CCU实例，再绑定到通信域
CcuInsHandle insHandle = 0;
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

// 2. 查询已绑定的CCU实例（新方式）
CcuInsHandle queriedInsHandle = 0;
uint32_t insNum = 0;
hcclRet = HcclCommQueryAssignedCcuIns(comm, &queriedInsHandle, &insNum);
// 当前实现固定返回1个CCU实例，insNum != 1视为失败
if (hcclRet != HCCL_SUCCESS || insNum != 1) {
    // 错误处理：接口失败返回原始错误码，接口成功但实例数异常返回内部错误码
    HcclCommDestroy(comm);
    return (hcclRet != HCCL_SUCCESS) ? hcclRet : HCCL_E_INTERNAL;
}

// 校验查询到的句柄与绑定时的句柄一致
if (queriedInsHandle != insHandle) {
    HcclCommDestroy(comm);
    return HCCL_E_INTERNAL;
}

// 使用queriedInsHandle注册并下发CCU Kernel
CcuResult regStartRet = HcommCcuKernelRegisterStart(queriedInsHandle);
// ...

// 销毁通信域时，由通信域销毁已绑定的CCU实例。
return HcclCommDestroy(comm);
```
