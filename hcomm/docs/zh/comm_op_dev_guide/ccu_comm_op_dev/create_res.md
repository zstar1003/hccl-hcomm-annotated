# 创建资源

## 通信资源计算

通信算子在执行时依赖底层的硬件通信资源，因此在算子任务编排下发前，首先需要申请所需的通信资源。HCCL通信资源主要分为四类：

- **Thread**：用于表达单个rank内的并发关系，相同Thread上的task严格顺序执行，不同Thread上的task可以并发执行。无论何种展开模式，Thread均可分为1个主Thread和若干个从Thread，由该主Thread来控制这些从Thread上的任务的开始和结束，其中主Thread必须申请，从Thread的申请数量由每个通信算法单独计算。
- **Notify**：用于实现同步机制，包括rank内Thread间同步和rank间同步。
- **Channel**：用于跨rank传输数据的通道。
- **Kernel**：在CCU硬件上执行的算法，每个Kernel都包含了CCU的各种资源，包括CCU指令空间、寄存器、片上缓存、并发引擎和channel表项等。Kernel所需的资源数量和CCU指令内容，在使用HcommCcuKernelRegister接口注册kernel时翻译产生，并完成申请。

不同的通信算法所需的通信资源数量往往不相同。下面以单机4卡的拓扑组网、使用CCU通信引擎为例，讲解Mesh全连接通信算法所需的通信资源数量计算逻辑。

- **Mesh全连接通信算法**

  Mesh算法的硬件拓扑如下图所示，每个rank与其他所有rank直接通信。

  ![](figures/mesh.png)

  上述组网中Mesh全连接通信算法所需要的通信资源数量如下：

  - Thread：CCU的Mesh算法集成在1个kernel中，kernel需要下发在一个Thread上，因此需要1个Thread资源。
  - Notify：当算法只需要1个Thread时，不需要Thread上的Notify资源。
  - Channel：每个rank需要与其他所有rank进行通信，因此每个rank共需要建立3条Channel通信通道。
  - Kernel：Mesh算法实现在1个Kernel中，因此需要1个Kernel资源。

综上所述，针对Mesh全连接通信算法，每个rank所需的通信资源数量如下表所示。

**表1**  单机4卡拓扑中，算法所需各类通信资源的数量对比

| 通信资源 | Mesh算法 |
| --- | --- |
| Thread | 1 |
| Notify | 0 |
| Channel | 3 |
| Kernel | 1 |

## 代码示例

下面以自定义AllGather通信算子为例，介绍其在Host上的资源创建代码片段。

申请Thread资源：

```c
CommEngine engine = CommEngine::COMM_ENGINE_CCU;  // CCU通信引擎
uint32_t threadNum = 1;             // 申请1个通信线程
uint32_t notifyNumPerThread = 0;    // 不额外申请通信线程中的同步资源
ThreadHandle thread;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);
```

建立通信链路Channel，与每个对端rank之间建立1条通信通道：

```c
std::vector<ChannelHandle> kernelChannels;
kernelChannels.resize(rankSize - 1);
for(uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
    if (remoteRank == myRank) {
        continue;
    }
    uint32_t netLayer = 0, listSize = 0;
    CommLink *linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));  // 获取srcRank和dstRank间link信息
    HcclChannelDesc desc;
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
    bool protocolExists = false;
    for (uint32_t idx = 0; idx < listSize; idx++) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol == protocol) {
            desc.remoteRank = remoteRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            protocolExists = true;
            break;
        }
    }
    CHK_RET(HcclChannelAcquire(comm, param.engine, &desc, 1, &kernelChannels[channelIndex])); // 获取channelhandle
    channelIndex++;
}
```

注册Kernel：

```c
// 设置kernel函数名和函数指针
char kernelFuncName[64];
strcpy_s(kernelFuncName, sizeof(kernelFuncName), "CcuAllGatherMesh1DMem2MemKernel");
kernelFunc = reinterpret_cast<void *>(CcuAllGatherMesh1DMem2MemKernel);
auto kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>(rankSize, myRank);
// 将channel与kernel绑定
auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelArg.get());
for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
    kernelArgBase->channels[i] = kernelChannels[i];  // 将channelhandle保存在kernelArgBase中
}
kernelArgBase->channelCount = static_cast<uint32_t>(kernelChannels.size());
// 获取insHandle
CcuInsHandle insHandle{0};
uint32_t insNum = 0;
CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
CHK_PRT_RET(insNum != 1,
    HCCL_ERROR("[GetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
    HCCL_E_INTERNAL);
// 注册kernel
CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
if (regStartRet != CCU_SUCCESS) {
    HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet);
    return ConvertCcuToHccl(regStartRet);
}
CcuKernelHandle kernelHandle;
constexpr uint32_t dieId = 0;
constexpr uint32_t kernelArgNum = 1;
const void *kernelArgsArr[] = { kernelArg.get() }; // 按HcommCcuKernelRegister签名构造入参指针数组
CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelFuncName, reinterpret_cast<void*>(kernelFunc), kernelArgsArr, kernelArgNum, &kernelHandle); // 注册kernel
if (regRet != CCU_SUCCESS) {
    HCCL_ERROR("ccu kernel register failed: ccuRet -> %d", regRet);
    return ConvertCcuToHccl(regRet);
}
CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
if (regEndRet != CCU_SUCCESS) {
    HCCL_ERROR("ccu kernel register end failed: ccuRet -> %d", regEndRet);
    return ConvertCcuToHccl(regEndRet);
}
```
