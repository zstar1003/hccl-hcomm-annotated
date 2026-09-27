# 算法执行

下发kernel后，CCU硬件根据kernel中编排的任务执行算法，完成数据搬运与同步。

## 算法编排步骤

CCU的Kernel接收来自Host的2种参数：KernelArg和TaskArg。

- KernelArg：包含算法编排所需的参数，包括rank ID、通信域大小、reduce类算子的归约类型等。此类参数作为Kernel函数入参传入。
- TaskArg：包含算法执行时所需的参数，包括input地址、output地址、token等。此类参数需要通过HcommCcuKernelLaunch传入，并在Kernel函数中使用Load函数动态加载。

Kernel中算法主要包含以下几个步骤：

1. 初始化资源。

    算法需要使用的Variable、LocalAddr、RemoteAddr和CompetedEvent等变量，需要进行声明。如果是需要从远端rank来写入的Variable，则需要初始化时，和channel进行绑定，例如：

    ```c
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    AllGatherMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    
    // 初始化资源
    ctx.output.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    ```

2. 从TaskArg中加载参数值到CCU变量中，通常包含input地址、output地址、token等，需要调用Load接口进行加载。

    ```c
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    ```

3. 前同步，将本端的input地址等信息同步到对端rank，并等待对端rank的同步。

    常用的数据面接口为NotifyRecord和NotifyWait。其中，前同步的同时可以选择将Variable的值写到对端，需使用接口WriteVariableWithNotify，例如：

    ```c
    // 将output变量中的值，写到对端，并同步到0号同步寄存器的bit1
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << OUTPUT_XN_ID));
    }
    ```

4. 进行数据搬运，常用的数据面接口有Write、Read和LocalCopy等。
5. 后同步，通知对端Rank搬运结束，并等待对端Rank的同步。

## 代码示例

以自定义AllGather算子为例，其在CCU引擎下的任务编排代码片段如下：

```c
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;
CcuResult CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    AllGatherMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    // 1.初始化资源
    ctx.output.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    // 2.加载参数
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));
    // 3.前同步
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }
    // 4.执行算法
    ccu::LocalAddr src;
    ccu::LocalAddr localDst;
    std::vector<ccu::RemoteAddr> dst;
    dst.resize(ctx.arg->rankSize);
    src.addr = ctx.input;
    src.addr += ctx.currentRankSliceInputOffset;
    src.token = ctx.token[ctx.arg->rankId];
    for (uint32_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
        if (rankIdx == ctx.arg->rankId) {
            localDst.addr = ctx.output[rankIdx];
            localDst.addr += ctx.currentRankSliceOutputOffset;
            localDst.token = ctx.token[rankIdx];
        } else {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }
    }
    CCU_IF(ctx.sliceSize != 0)
    {
        uint32_t channelId = 0;
        for (uint64_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
            const uint16_t mask = 1 << rankIdx;
            if (rankIdx != ctx.arg->rankId) {
                CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelId], dst[rankIdx], src, ctx.sliceSize, ctx.event, mask)); // 本卡数据写入远端地址
                channelId++;
            } else {
                CCU_CHK_RET(ccu::LocalCopy(localDst, src, ctx.sliceSize, ctx.event, mask)); // 本rank数据拷贝到本地
            }
        }
    }
    const uint16_t totalMask = (1 << ctx.arg->rankSize) - 1;
    CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask)); // 等待本卡的数据搬运完成
    // 5.后同步
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID)); // 等待远端卡数据搬运完成
    }
    return CcuResult::CCU_SUCCESS;
}
```
