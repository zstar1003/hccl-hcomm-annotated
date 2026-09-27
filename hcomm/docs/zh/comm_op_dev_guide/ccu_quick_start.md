# CCU算子

本节以CCU的集合通信算子为例，介绍使用HCCL通信编程接口开发通信算子的整体流程，帮助用户快速了解通信算子的开发步骤。

## 集合通信算子

本节以AllGather集合通信算子为例进行介绍。AllGather操作是将通信域内所有节点的输入按照rank id重新排序（rank id按照从小到大的顺序排序），然后拼接起来，再将结果发送到所有节点的输出buffer。

![](./figures/allgather.png)

## 产品支持情况

本样例支持以下昇腾产品，组网要求为支持UB协议的Mesh互联，N卡（N>=2）：

- Ascend 950PR/Ascend 950DT

## 样例介绍

开发者可以点击[CCU样例](https://gitcode.com/cann/hccl/tree/master/examples/05_custom_ops_allgather/ccu)获取完整样例代码，该样例使用HCCL通信算子开发接口实现了基于CCU通信引擎的AllGather算子，主要实现流程如下所示：

- **查询通信域的拓扑信息**：调用拓扑信息查询接口HcclGetRankId\(\)和HcclGetRankSize\(\)获取当前线程操作的rank\_id和通信域的rank数量。
- **创建Thread资源**：调用资源管理接口HcclThreadAcquire\(\)分配通信线程资源。
- **建立通信通道**：调用HcclChannelAcquire\(\)接口创建rank间的channel链路。
- **注册CCU Kernel**：调用HcommCcuKernelRegister\(\)接口创建CCU Kernel，生成CCU指令，用于在CCU上执行。
- **获取本端通信内存**：调用HcclGetHcclBuffer\(\)接口获取本端的通信内存信息。
- **准备输入数据**：调用HcommLocalCopyOnThread\(\)接口将输入数据拷贝到本端通信内存。
- **生成Token秘钥**：调用HcommCcuGetMemToken\(\)接口生成秘钥，唯一标识一块地址信息，用于远端交互。
- **下发CCU Kernel**：调用HcommCcuKernelLaunch\(\)接口下发CCU Kernel。
- **前同步**：调用ccu::WriteVariableWithNotify\(\)接口告知远端数据已经准备完成；
- **写入数据**：调用ccu::Write\(\)接口将本端数据写入到远端通信内存。
- **后同步**：调用ccu::NotifyRecord\(\)、ccu::NotifyWait\(\)接口告知远端已经写入完成。

同时，该样例中还包含了测试程序，其中创建1个通信域，每个线程操作1个设备，共同完成AllGather操作。包含以下功能点：

- 设备检测，通过aclrtGetDeviceCount\(\)接口查询可用设备数量。
- 将rank0作为root节点，通过HcclGetRootInfo\(\)接口生成root节点的rootinfo标识信息。
- 在每个线程中，基于rootinfo标识信息通过HcclCommInitRootInfo\(\)接口初始化通信域。
- 调用算子接口HcclAllGatherCustom\(\)，并打印接收结果。

## 编译安装

在CANN/hccl代码仓根目录下执行如下命令，编译并安装自定义算子包：

```bash
# 设置CANN软件包环境变量，此处以root用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh

# 执行build.sh脚本进行编译，通过custom_ops_path指定自定义算子工程路径
bash build.sh --vendor=cust --ops=allgather --custom_ops_path=./examples/05_custom_ops_allgather/ccu

# 自定义算子安装包在代码仓的build_out目录下
./build_out/cann-hccl_custom_allgather_linux-<arch>.run --install
```

自定义算子包安装信息如下：

- 头文件：$\{ASCEND\_HOME\_PATH\}/opp/vendors/cust/include/hccl\_custom\_allgather.h
- 动态库：$\{ASCEND\_HOME\_PATH\}/opp/vendors/cust/lib64/libhccl\_custom\_allgather.so

- 安装脚本：$\{ASCEND\_HOME\_PATH\}/opp/vendors/cust/scripts/install.sh

## 执行样例

编译并执行测试样例。

```bash
# 进入样例代码目录
cd examples/05_custom_ops_allgather/ccu/testcase
# 编译
make
# 执行测试用例
make test
```

## 结果解析

所有节点的输入数据初始化为该节点的DeviceId。运行成功后，终端将输出类似以下的日志信息（以2卡运行为例）：

```text
Found 2 NPU device(s) available
rankId: 0, output: [ 0 1 ]
rankId: 1, output: [ 0 1 ]
```

## 关键代码解析

下面以自定义AllGather算子为例，讲解其实现细节。

1. 解析通信域的拓扑信息。

    ```c
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    ```

2. 创建资源。

    ```c
    // 申请thread资源。host模式下，将主流封装为thread，并创建主流上的notify
    ThreadHandle thread;
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 0, &thread)); 
    ```

3. 建立通信链路。

    ```c
    // 申请channel资源
    uint32_t netLayer = 0, listSize = 0;
    CommLink *linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &linkList, &listSize)); // 获取srcRank和dstRank间link信息
    
    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
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
    if (!protocolExists) {
        HCCL_ERROR("[GetChannelForCcu] Protocol %d not found between rank %u and rank %u",
            protocol, param.myRank, remoteRank);
        return HCCL_E_NOT_FOUND;
    }
    CHK_RET(HcclChannelAcquire(comm, param.engine, &desc, 1, &kernelChannels[channelIndex])); // 获取channelhandle
    ```

4. 注册CCU Kernel。

    ```c
    CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
    
    // 注册kernel
    CcuKernelHandle kernelHandle;
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    const void *kernelArgsArr[] = { kernelInfo.kernelArg }; // 按HcommCcuKernelRegister签名构造入参指针数组
    CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName, reinterpret_cast<void*>(kernelInfo.kernelFunc),
                                              kernelArgsArr, kernelArgNum, &kernelHandle);
    
    resCtxHost.ccuKernels[0] = kernelHandle;
    CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
    ```

5. 获取远端的通信内存地址。

    ```c
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize)); // 从通信域获取CCL buffer
    ```

6. 准备输入数据。

    ```c
    HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize);
    ```

7. 生成Token秘钥，唯一标识一块地址信息，用于远端交互。

    ```c
    uint64_t token = 0;
    uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    if (param.inputPtr != nullptr) {
        HcommCcuGetMemToken(baseInputAddr, static_cast<uint64_t>(dataSize), &token);
    } else if (param.outputPtr != nullptr) {
        HcommCcuGetMemToken(baseOutputAddr, static_cast<uint64_t>(dataSize), &token);
    }
    ```

8. 下发CCU Kernel。

    ```c
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        sliceSize,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
    };
    
    CcuResult launchRet = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(), taskArgs.size());
    ```

9. 前同步，告知远端数据已经准备完成。

    ```c
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    ```

10. 将本端数据写入到远端通信内存。

    ```c
    CCU_IF(ctx.sliceSize != 0)
    {
        uint32_t channelId = 0;
        for (uint64_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
            const uint16_t mask = 1 << rankIdx;
            if (rankIdx != ctx.arg->rankId) {
                CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelId], dst[rankIdx], src, ctx.sliceSize, ctx.event, mask)); // 本卡数据写入远端地址
                channelId++;
            }
        }
    }
    ```

11. 后同步，告知远端已经写入完成。

    ```c
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID)); // 等待远端卡数据搬运完成
    }
    ```
