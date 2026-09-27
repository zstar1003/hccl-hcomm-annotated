# HcclCommQueryCcuIns

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

查询指定HCCL通信域自有的CCU实例句柄，返回实例句柄用于后续CCU Kernel的注册与下发。

通信域在初始化阶段不再主动创建CCU实例。本接口查询通信域自有的CCU实例句柄（即以固定资源量创建、由通信域自有的句柄）；当该句柄尚未存在（值为`0`）时，接口会根据通信域的算子扩展模式（opExpansionMode）创建一个CCU实例并保存到通信域，随后返回该句柄。查询成功时固定返回1个实例：insHandles[0]为通信域自有的实例句柄，*insNum为1。

当前一个通信域最多持有一个自有的CCU实例，多次查询返回同一句柄，不会重复创建。

> [!NOTE]注意
> - 本接口查询的是通信域自有的CCU实例，与通过[HcclCommAssignCcuIns](HcclCommAssignCcuIns.md)绑定的CCU实例（新版本）相互独立。若需查询通过`HcclCommAssignCcuIns`绑定的实例，请使用[HcclCommQueryAssignedCcuIns](HcclCommQueryAssignedCcuIns.md)。
> - 查询结果仅供借用，不转移实例所有权。创建的CCU实例所有权归通信域，由通信域负责释放；调用者不可销毁该实例，否则会造成对同一实例的重复释放，破坏通信域的资源管理。

## 函数原型

```c
HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄，不可为nullptr。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| insHandles | 输出 | CCU实例句柄数组，不可为nullptr。查询成功后insHandles[0]返回通信域自有的CCU实例句柄（若调用前不存在则创建）。<br>调用方需分配至少1个CcuInsHandle元素的空间。<br>CcuInsHandle类型的定义如下：<br>typedef uint64_t CcuInsHandle; |
| insNum | 输出 | CCU实例数量，不可为nullptr。查询成功后返回值固定为1。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

- 当comm、insHandles或insNum为nullptr时，返回HCCL_E_PTR。
- 当通信域代际不支持CCU（早于Ascend 950PR/Ascend 950DT）时，返回HCCL_E_NOT_SUPPORT。
- 当通信域的自有CCU实例句柄为`0`、且算子扩展模式未使能CCU时，返回HCCL_E_UNAVAIL，不创建实例。
- 当创建CCU实例失败时，返回创建接口的原始错误码（透传CcuResult，如CCU_E_DRV_BUSY、CCU_E_UNAVAIL等）。

> [!NOTE]资源不足处理
> - 当算子扩展模式为CCU_MS且CCU资源不足以创建CCU_MS实例时，本接口会自动降级到CCU_SCHED重试创建一次。
> - 当CCU_SCHED实例也无法创建（资源仍不足）时，返回HCCL_E_UNAVAIL。**本接口不会自动降级到AICPU_TS**，调用方需在收到HCCL_E_UNAVAIL后自行决定是否降级。

## 约束说明

1. CCU特性仅在Ascend 950PR/Ascend 950DT及以上代际支持，在更早代际的通信域上调用返回HCCL_E_NOT_SUPPORT。
2. 需在通信域完成初始化后调用。当通信域的自有CCU实例尚未创建时，本接口会创建；若通信域展开模式未使能CCU，返回HCCL_E_UNAVAIL。
3. 本接口查询的CCU实例与通过`HcclCommAssignCcuIns`绑定的CCU实例相互独立，二者不会相互覆盖或影响。
4. 返回的CCU实例句柄仅供借用，所有权仍归通信域所有，由通信域负责释放。调用者不可销毁该实例，否则会造成对同一实例的重复释放，破坏通信域的资源管理。
5. 本接口仅保证多次`HcclCommQueryCcuIns`调用之间的幂等性（已创建则直接返回，不重复创建）。调用方须保证本接口不与`HcclCommAssignCcuIns`或`HcclCommDestroy`并发执行。
6. CCU资源不足时本接口仅在CCU_MS→CCU_SCHED之间自动降级一次，不会自动降级到AICPU_TS。调用方收到HCCL_E_UNAVAIL后需自行处理是否降级到AICPU_TS模式。

## 调用示例

```c
// comm 需通过 HcclCommInitClusterInfo 等接口完成初始化，此处仅为示例占位
HcclComm comm = /* 已初始化的通信域句柄 */;
CcuInsHandle insHandle = 0;
uint32_t insNum = 0;

// 查询通信域自有的CCU实例；若尚未创建则按通信域展开模式创建
HcclResult ret = HcclCommQueryCcuIns(comm, &insHandle, &insNum);
if (ret == HCCL_E_UNAVAIL) {
    // CCU资源不足，无法创建CCU实例，调用方需自行处理是否降级到AICPU_TS模式
    return fallbackToAicpuTs(comm, ...);
}
// 当前实现固定返回1个CCU实例，insNum != 1视为失败
if (ret != HCCL_SUCCESS || insNum != 1) {
    // 错误处理：接口失败返回原始错误码，接口成功但实例数异常返回内部错误码
    return (ret != HCCL_SUCCESS) ? ret : HCCL_E_INTERNAL;
}

// 使用insHandle注册并下发CCU Kernel
CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
// ...
```
