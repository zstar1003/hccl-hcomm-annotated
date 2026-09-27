# HcclOpDescInit

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

初始化HcclOpDesc结构体，用于描述AI CPU运行核函数时的算子信息。

该接口会先用0xFF填充整个结构体，再设置ABI头部信息（包含版本号、魔数字、结构体大小），使结构体处于可用初始状态。调用该接口后，开发者需根据实际通信任务按需设置opDescType、opName及联合体成员（如p2p）等字段，再传入[HcclAicpuKernelLaunch](./HcclAicpuKernelLaunch.md)接口下发任务。

## 函数原型

```c
static inline HcclResult HcclOpDescInit(HcclOpDesc *opDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opDesc | 输出 | 需要初始化的算子描述参数。<br>HcclOpDesc类型的定义可参见数据类型[HcclOpDesc](./data_type_definition/HcclOpDesc.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS；传入空指针时返回HCCL_E_PTR。

## 约束说明

- 该接口仅完成ABI头部信息的初始化，结构体其余字段被填充为0xFF，调用者需在初始化后按需设置业务字段（如opDescType、p2p.buffer、p2p.cmdType等）。
- 传入的指针必须指向已分配内存的HcclOpDesc对象，不能为空指针。

## 调用示例

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
aclrtStream userStream;
aclrtCreateStream(&userStream);
ThreadHandle aicpuThreadHandle;
HcclResult ret = HcclThreadAcquire(comm, engine, 1, 1, &aicpuThreadHandle);
if (ret != HCCL_SUCCESS) {
    // 处理错误
}

// 初始化算子描述参数
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
HcclKernelLaunchCfg kernelLaunchCfg;
HcclKernelLaunchCfgInit(&kernelLaunchCfg);
kernelLaunchCfg.timeOut = 60;

// 运行AICPU核函数
ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
if (ret != HCCL_SUCCESS) {
    // 处理错误
}
```
