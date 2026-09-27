# 创建资源

## 通信资源计算

通信算子在执行时依赖底层的硬件通信资源，因此在算子任务编排下发前，首先需要申请所需的通信资源。AIV通信算子需要的通信资源主要分为三类：

- **Vector Core**：用于表达单个rank内的并发关系。在算法编排过程中，可将不同rank或不同数据块拆分，并在多个计算核上并行处理。多个AI Core共享同一份代码逻辑，各个计算核上运行的实例之间唯一的区别在于核ID（下标）不同，每个核根据自己的下标来确定所走的算法分支流程。
- **Notify**：用于实现同步机制，包括rank内不同Vector Core之间的同步以及rank间的同步。由于AI Core内无法调用硬件能力来实现rank间同步，因此需要通过软件算法来模拟同步功能的实现，称为软同步，软同步使用到的Notify本质是一块片上内存。
- **Channel**：用于跨rank传输数据的通道。

不同的通信算法所需的通信资源类型和数量往往不相同。下面以单机4卡的拓扑组网、使用AIV通信引擎为例，讲解Mesh全连接通信算法所需的通信资源数量计算逻辑。

Mesh算法的硬件拓扑如下图所示，每个rank与其他所有rank直接通信。

![Mesh算法硬件拓扑](figures/mesh.png)

上述组网中Mesh全连接通信算法所需要申请或者占用的通信资源数量如下：

- Vector Core：每个rank需要跟其他所有rank进行通信，将每个rank上的通信任务（包含本rank的本地拷贝）分配给不同的核异步执行。因此，每个rank共需要占用4个Vector Core资源，这些资源在算子下发时由系统自动分配，无需提前申请。
- Notify：每个rank需与其他所有rank进行同步，同步操作分为前同步和后同步，二者需要配对使用。因此，每个rank共需要6个Notify资源，每个软同步Notify资源分配32 Byte的片上内存，通常可直接预留1KB至1MB左右的空间。
- Channel：每个rank需要与其他所有rank进行通信，因此每个rank共需要建立3条Channel通信通道。

## 代码示例

下面以AllGather通信算子为例，介绍其在Host上的资源创建代码片段。

1. 申请Context内存，用以存储资源信息。

    ```c
    uint64_t size = 需要申请的Context内存的大小;
    void *ctx = nullptr;
    char *tag = 保存资源需要用到的tag;
    CommEngine engine = COMM_ENGINE_CPU_TS; // 使用Host侧内存缓存资源
    HcclResult ret = HcclEngineCtxGet(comm, tag, engine, &ctx, &size);
    if (ret != HCCL_SUCCESS) {
       // 即之前没有创建过tag表示的资源
       HcclEngineCtxCreate(comm, tag, engine, size, ctx); 
    } else {
       // 说明之前创建过资源，直接用ctx就好
    }
    ```

2. 申请Notify资源。

    ```c
    char tag[] = "allgather";
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;                   // AIV通信引擎
    void* aivTagBufPtr = nullptr;                                      // 软同步标记区地址
    HcclEngineCtxCreate(comm, tag, engine, AIV_TAG_BUFF_LEN, &aivTagBufPtr);  // 创建Notify资源
    aclrtMemset(aivTagBufPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN);  // 清零标记区
    ```

3. 将Notify软同步内存注册到通信域，建立通信链路Channel，两个rank之间建立1条通信通道。

    ```c
    HcclMemHandle memHandle;
    CommMem regMem{COMM_MEM_TYPE_DEVICE, aivTagBufPtr, AIV_TAG_BUFF_LEN};
    HcclCommMemReg(comm, tag, &regMem, &memHandle); // 将软同步内存注册到通信域
    
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = destRank;
    channelDesc.channelProtocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    channelDesc.memHandles = &memHandle;           // 建链时指定交换软同步内存
    channelDesc.memHandleNum = 1;
    ChannelHandle channelHandle;
    HcclChannelAcquire(comm, engine, &channelDesc, 1, &channelHandle);
    ```

4. 分配通信内存。

    ```c
    // 分配并获取本端通信内存
    void *localAddr;
    uint64_t localSize;
    HcclGetHcclBuffer(comm, &localAddr, &localSize);
    // 获取远端通信内存，用于后续读写操作
    void *remoteAddr;
    uint64_t remoteSize;
    HcclChannelGetHcclBuffer(comm, channelHandle, &remoteAddr, &remoteSize);
    // 获取远端软同步标记区内存，用于后续同步操作
    uint32_t memNum;
    CommMem* remoteMems;
    char** memTags;
    HcclChannelGetRemoteMems(comm, channelHandle, &memNum, &remoteMems, &memTags);
    void *remoteTagBufAddr = remoteMems[0].addr;
    ```
