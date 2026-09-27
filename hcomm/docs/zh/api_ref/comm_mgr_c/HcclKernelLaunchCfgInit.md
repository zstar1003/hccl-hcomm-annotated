# HcclKernelLaunchCfgInit

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

初始化HcclKernelLaunchCfg结构体，用于描述AICPU运行核函数的配置信息。

该接口会先用0xFF填充整个结构体，再设置ABI头部信息（包含版本号、魔数字、结构体大小），使结构体处于可用初始状态。调用该接口后，开发者需根据实际任务按需设置timeOut等字段，再将配置传入[HcclAicpuKernelLaunch](./HcclAicpuKernelLaunch.md)接口下发任务。

## 函数原型

```c
static inline HcclResult HcclKernelLaunchCfgInit(HcclKernelLaunchCfg *kernelLaunchCfg)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| kernelLaunchCfg | 输出 | 需要初始化的KernelLaunch配置参数。<br>HcclKernelLaunchCfg类型的定义可参见数据类型[HcclKernelLaunchCfg](./data_type_definition/HcclKernelLaunchCfg.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS；传入空指针时返回HCCL_E_PTR。

## 约束说明

- 该接口仅完成ABI头部信息的初始化，结构体其余字段被填充为0xFF，调用者需在初始化后按需设置业务字段（如timeOut）。
- 传入的指针必须指向已分配内存的HcclKernelLaunchCfg对象，不能为空指针。

## 调用示例

```c
// 初始化核函数配置
HcclKernelLaunchCfg kernelLaunchCfg;
HcclKernelLaunchCfgInit(&kernelLaunchCfg);
kernelLaunchCfg.timeOut = 60;  // 设置超时时间，单位为秒

// 运行AICPU核函数
HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
if (ret != HCCL_SUCCESS) {
    // 处理错误
}
```
