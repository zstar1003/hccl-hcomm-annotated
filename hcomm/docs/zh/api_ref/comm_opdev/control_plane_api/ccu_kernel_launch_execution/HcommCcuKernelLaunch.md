# HcommCcuKernelLaunch

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

启动一次CCU Kernel的执行。本接口将已翻译的Kernel提交给CCU硬件取指执行。同一Kernel句柄可多次调用本接口，每次可传入不同的`taskArgs`，实现"一次注册，多次启动"。

## 函数原型

```c
CcuResult HcommCcuKernelLaunch(ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void *taskArgs, uint32_t argNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| threadHandle | 输入 | 通信线程句柄，须通过[HcclThreadAcquireWithStream](../comms_domain_resource_mgmt/HcclThreadAcquireWithStream.md)获取的有效句柄。取值不能为0。 |
| kernelHandle | 输入 | Kernel句柄，须是通过[HcommCcuKernelRegister](HcommCcuKernelRegister.md)获取的有效句柄。取值不能为0。 |
| taskArgs | 输入 | 指向`uint64_t`数组的指针，数组元素按`CcuLoadArg`中的`argId`索引取值，为Kernel提供运行期可变参数。若Kernel内未使用`CcuLoadArg`，可传入空指针。 |
| argNum | 输入 | `taskArgs`数组的元素个数，单位为`uint64_t`元素个数，不是字节数。取值须等于Kernel注册阶段内使用过的不同`argId`总数；若Kernel内未使用`CcuLoadArg`，取值为0。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PARA` | 参数错误，`threadHandle`或`kernelHandle`为0。 |
| `CCU_E_PTR` | 该错误返回值包括`kernelHandle`对应Kernel不存在或者句柄无效、或线程stream为空，或`argNum > 0`时`taskArgs`为空指针。 |
| `CCU_E_INTERNAL` | 内部错误，`argNum`与Kernel内`CcuLoadArg`使用的不同`argId`总数不一致，或内部下发失败。 |

## 约束说明

- 必须在[HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)之后调用。
- `argNum`的单位是`uint64_t`元素个数，不是字节数。传入字节数会导致`CCU_E_INTERNAL`错误。
- 当`taskArgs`参数较多时，框架会自动分批下发，无需用户干预。
- 本接口只能在Host侧调用，不能在Kernel函数体内调用。

## 调用示例

```c
// threadHandle 由HcclThreadAcquireWithStream 返回
// kernelHandle 由HcommCcuKernelRegister 返回，且已完成RegisterEnd
ThreadHandle threadHandle = 0;
CcuKernelHandle kernelHandle = 0;
// ... 此处省略前序调用 ...

// 第一次启动，传入运行期参数（元素个数，不是字节数）
uint64_t taskArgs[2] = { 100, 200 };
CcuResult ret = HcommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, 2);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelLaunch failed, ret = %d\n", ret);
    return ret;
}

// 同一Kernel 可重复启动，每次可传入不同的taskArgs
taskArgs[0] = 300;
taskArgs[1] = 400;
ret = HcommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, 2);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelLaunch failed, ret = %d\n", ret);
    return ret;
}
```
