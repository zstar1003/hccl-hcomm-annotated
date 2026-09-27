# HcommCcuKernelRegisterStart

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

标记一轮Kernel注册流程的开始，清空指定CCU实例上的待翻译Kernel集合，为后续调用[HcommCcuKernelRegister](HcommCcuKernelRegister.md)做准备。

一个CCU实例支持多轮注册，每轮由本接口开始，由[HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)结束。每轮内可调用[HcommCcuKernelRegister](HcommCcuKernelRegister.md)注册一个或多个Kernel。

## 函数原型

```c
CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| insHandle | 输入 | CCU实例句柄，从Hccl通信域中获取。有效句柄从1开始；传入0或未注册的句柄将返回`CCU_E_PTR`。|

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | `insHandle`为0或无效，未找到对应实例。 |
| `CCU_E_INTERNAL` | 时序错误：上一轮注册流程尚未调用[HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)结束，就再次调用本接口开始新一轮注册。 |

## 约束说明

- CcuInsHandle 必须在Hccl通信域中先获取，且必须先于[HcommCcuKernelRegister](HcommCcuKernelRegister.md)。
- 本接口与[HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)须成对调用：开始新一轮注册前，上一轮必须已通过[HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)结束，否则本接口返回`CCU_E_INTERNAL`。
- 本接口只能在Host侧调用，不能在Kernel函数体内调用。

## 调用示例

```c
// insHandle 由Hccl通信域中获取
CcuInsHandle insHandle = 0;
// ... 此处省略HcommCcuInsCreate 调用 ...

// 开始一轮Kernel 注册
CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegisterStart failed, ret = %d\n", ret);
    return ret;
}
// 后续调用HcommCcuKernelRegister 注册一个或多个Kernel
```
