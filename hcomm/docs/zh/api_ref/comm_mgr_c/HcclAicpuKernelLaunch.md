# HcclAicpuKernelLaunch

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

在AI CPU上运行核函数。该接口用于下发自定义通信算子到AI CPU执行，支持在Group通信场景和非Group通信场景下使用。

在Group通信场景下（调用了HcclGroupStart），该接口会将任务添加到Group任务队列中，等待HcclGroupEnd时统一执行；在非Group通信场景下，直接在AI CPU上运行核函数。

## 函数原型

```c
HcclResult HcclAicpuKernelLaunch(HcclComm comm, const HcclOpDesc *opInfo, const HcclKernelFuncInfo *funcInfo, ThreadHandle aicpuThreadHandle, aclrtStream userStream, const HcclKernelLaunchCfg *kernelLaunchCfg);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 集合通信操作所在的通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| opInfo | 输入 | 算子描述参数，包含算子类型和名称相关参数。<br>HcclOpDesc类型的定义可参见数据类型[HcclOpDesc](./data_type_definition/HcclOpDesc.md)。 |
| funcInfo | 输入 | 核函数信息，包含动态库名、函数名、参数及参数大小。<br>HcclKernelFuncInfo类型的定义可参见数据类型[HcclKernelFuncInfo](./data_type_definition/HcclKernelFuncInfo.md)。 |
| aicpuThreadHandle | 输入 | AICPU通信主流线程句柄。<br>ThreadHandle类型的定义可参见[ThreadHandle](../comm_opdev/datatype_definition/ThreadHandle.md) |
| userStream | 输入 | 本rank所使用的stream。 |
| kernelLaunchCfg | 输入 | 运行核函数的相关配置，包含超时时间等参数。<br>HcclKernelLaunchCfg类型的定义可参见数据类型[HcclKernelLaunchCfg](./data_type_definition/HcclKernelLaunchCfg.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 自定义通信算子场景使用（当前仅支持Send和Receive算子）。
- 调用前需确保通信域已正确初始化。

## 调用示例

```c
// 初始化算子描述参数
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
aclrtStream userStream;
aclrtCreateStream(&userStream);
ThreadHandle aicpuThreadHandle;
HcclResult ret = HcclThreadAcquire(comm, engine, 1, 1, &aicpuThreadHandle);
if (ret != HCCL_SUCCESS) {
    // 处理错误
}

HcclOpDesc opInfo;
HcclOpDescInit(&opInfo);
opInfo.opDescType = 1;  // 代表Send或Receive算子
opInfo.p2p.buffer = sendBuffer;
opInfo.p2p.cmdType = HCCL_CMD_SEND;
opInfo.p2p.dataType = HCCL_DATA_TYPE_FP32;
opInfo.p2p.count = dataSize;
opInfo.p2p.remoteRank = destRank;
opInfo.p2p.unfoldStream = userStream;

// 初始化核函数信息
HcclKernelFuncInfo funcInfo;
strncpy_s(funcInfo.kernelSoName, HCCL_KERNEL_SO_NAME_MAX_LEN, "libscatter_aicpu_kernel.so", strlen("libscatter_aicpu_kernel.so") + 1);
strncpy_s(funcInfo.kernelFuncName, HCCL_KERNEL_FUNC_NAME_MAX_LEN, "HcclLaunchP2pAicpuKernel", strlen("HcclLaunchP2pAicpuKernel") + 1);
funcInfo.args = kernelArgs;
funcInfo.argSize = argsSize;

// 初始化核函数配置
AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
u16 kernelLaunchTimeout = IsHcommDefaultTimeoutSupported() ? timeout.kernelLaunchTimeout :
ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
HcclKernelLaunchCfg kernelLaunchCfg;
HcclKernelLaunchCfgInit(&kernelLaunchCfg);
kernelLaunchCfg.timeOut = kernelLaunchTimeout;

// 运行AI CPU核函数
ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
if (ret != HCCL_SUCCESS) {
    // 处理错误
}
```
