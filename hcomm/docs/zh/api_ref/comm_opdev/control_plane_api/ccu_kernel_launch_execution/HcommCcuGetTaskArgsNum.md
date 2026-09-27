# HcommCcuGetTaskArgsNum

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

查询已注册Kernel的任务参数个数（`taskArgsNum`）。该值等于Kernel注册期间所有`CcuLoadArg`调用中使用过的最大`argId`加1，即`taskArgs`数组所需的最小元素个数。算子层可在[HcommCcuKernelRegister](HcommCcuKernelRegister.md)之后调用本接口获取该数值，再据此构造`taskArgs`数组并作为`argNum`传入[HcommCcuKernelLaunch](HcommCcuKernelLaunch.md)，从而支持 `<<<>>>`的直调使用模式。

若Kernel注册期间未调用任何`CcuLoadArg`，本接口返回`0`。

## 函数原型

```c
CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle kernelHandle, uint32_t *taskArgsNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| kernelHandle | 输入 | Kernel句柄，须是通过[HcommCcuKernelRegister](HcommCcuKernelRegister.md)获取的有效句柄。取值不能为0。 |
| taskArgsNum | 输出 | 出参，指向`uint32_t`的指针，成功时写入任务参数个数。不能为空指针。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | `taskArgsNum`为空指针。 |
| `CCU_E_NOT_FOUND` | `kernelHandle`在Kernel管理表中不存在，句柄无效或已被注销。 |

## 约束说明

- 本接口调用前需确保HcommCcuKernelRegister已成功返回Kernel句柄。本接口与HcommCcuKernelRegisterEnd接口的调用顺序无强制要求，可在HcommCcuKernelRegisterEnd之前或之后，但需确保在kernel注销前调用。
- 返回值的计算方式为：Kernel注册期间所有`CcuLoadArg`调用的最大`argId`加1。`argId`从0开始编号，因此该值等于`taskArgs`数组按`argId`索引取值时所需的最小数组长度。若未调用任何`CcuLoadArg`，返回`0`。
- 返回值可直接作为[HcommCcuKernelLaunch](HcommCcuKernelLaunch.md)的`argNum`参数传入。
- 本接口为只读查询，不修改Kernel的注册状态，不影响后续启动。
- 本接口线程安全，内部通过锁保护Kernel管理表的访问。
- 本接口只能在Host侧调用，不能在Kernel函数体内调用。

## 调用示例

```c
// insHandle 从通信域中获取，kernelHandle 由 HcommCcuKernelRegister 返回
// Kernel 函数体内调用了 CcuLoadArg(0) 和 CcuLoadArg(1)
CcuKernelHandle kernelHandle = 0;
// ... 此处省略 HcommCcuKernelRegisterStart / HcommCcuKernelRegister / HcommCcuKernelRegisterEnd ...

// 查询 taskArgs 数组所需的元素个数
uint32_t taskArgsNum = 0;
CcuResult ret = HcommCcuGetTaskArgsNum(kernelHandle, &taskArgsNum);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuGetTaskArgsNum failed, ret = %d\n", ret);
    return ret;
}
// taskArgsNum 为 2（max(argId)=1，加1）

// 据 taskArgsNum 构造 taskArgs 数组并启动
uint64_t taskArgs[2] = { 100, 200 };
ret = HcommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, taskArgsNum);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelLaunch failed, ret = %d\n", ret);
    return ret;
}
```
