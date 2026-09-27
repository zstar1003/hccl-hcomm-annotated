# 算子下发

开发者完成通信算子任务编排后，需要将Kernel函数下发给具体的通信引擎执行。

对于AI CPU通信引擎，开发者需首先声明算子信息库文件，文件格式为json，文件内容如下：

```c
{
  "CustomAicpuKernel": {
    "opInfo": {
      "opKernelLib": "AICPUKernel",                // 固定值
      "kernelSo": "<aicpu_kernel_so_name>",        // AI CPU侧动态链接库，用户自定义，如：libp2p_aicpu_kernel.so
      "functionName": "<aicpu_kernel_func_name>"   // AI CPU Kernel函数名字，用户自定义，如：HcclLaunchP2PAicpuKernel
    }
  }
}
```

定义AI CPU Kernel函数，格式如下。其中<aicpu_kernel_func_name\>为AI CPU Kernel函数名字，函数名必须与算子信息库文件（json文件）中的functionName字段保持一致。

```c
extern "C" unsigned int <aicpu_kernel_func_name>(void *param)
{
    // Kernel实现
}
```

## 代码示例

以自定义Send/Receive算子为例，使用AI CPU通信引擎时，Host侧下发AI CPU Kernel函数的代码片段如下：

```c
// Host stream通知Device主thread
aclrtRecordNotify(g_notifies[0], stream);

std::string kernelName = "HcclLaunchP2PAicpuKernel";
aclrtFuncHandle funcHandle;
aclrtArgsHandle argsHandle;
ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle));
ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
aclrtParamHandle paraHandle;
aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle);
aclrtKernelArgsFinalize(argsHandle);

uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
aclrtLaunchKernelCfg cfg;
aclrtLaunchKernelAttr attr;
attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
cfg.numAttrs = 1;
cfg.attrs = &attr;
constexpr uint32_t blockDim = 1;

// 执行device侧的算法编排
aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, &cfg, argsHandle, nullptr);

// Host stream等待Device的通知
aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT);
```

在AI CPU侧需要定义Kernel函数入口，该函数需编译到Device侧：

```c
typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct AlgResourceCtx {
    ThreadHandle threadHandle;                              // 通信线程句柄
    CommBuffer localBuffer;                                 // 本端通信内存
    CommBuffer remoteBuffer;                                // 远端通信内存
    ChannelHandle channelHandle;                            // 通信通道资源
    uint32_t notifyIds[AICPU_CONTROL_NOTIFY_NUM];           // aicpu模式下device侧控制notify
};

struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];             // 通信域名称
    void* inputPtr = nullptr;                               // 算子输入数据地址
    void* outputPtr = nullptr;                              // 算子输出数据地址
    uint64_t count = 0;                                     // 算子数据量
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;        // 算子数据类型
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;     // 算子类型
    AlgResourceCtx* resCtx = nullptr;                       // 资源上下文
};

// 在AI CPU上执行的Kernel函数
extern "C" unsigned int HcclLaunchP2PAicpuKernel(OpParam *param)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) { 
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    // 获取Device侧主thread
    ThreadHandle thread = param->resCtx->threadHandle;
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed start batch mode");
        return 1;
    }

    // 主thread等待Host stream的通知
    if (HcommAclrtNotifyWaitOnThread(thread, param->resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to wait notify[%d] from host main stream", param->resCtx->notifyIds[0]);
        return 1;
    }

    // 执行任务编排
    if (ExecOp(*param, param->resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for op:%d", param->opType);
        return 1;
    }

    // 主thread通知Host stream
    if (HcommAclrtNotifyRecordOnThread(thread, param->resCtx->notifyIds[1]) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to record host main stream");
        return 1;
    }

    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed end batch mode");
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    return 0;
}
```
