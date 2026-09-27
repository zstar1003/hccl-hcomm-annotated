# 创建资源

## 通信资源计算

通信算子在任务编排前，需要根据算法的并发方式和通信对象计算资源。AICPU通信主要使用以下资源：

- **Context**：保存Thread、Channel、通信内存等资源信息。Host侧完成资源创建后，将Context序列化并拷贝到Device，供AICPU Kernel反序列化使用。
- **Thread**：表达单个rank内任务的串行和并发关系。同一Thread上的任务顺序执行，不同Thread上的任务可以并发执行。算法Thread通常由1个主Thread和若干个从Thread组成。
- **Thread Notify**：用于同一rank内主从Thread之间的同步。
- **Channel**：连接本rank与远端rank，用于跨rank数据传输和同步。
- **Channel Notify**：属于Channel的同步资源，用于通知远端数据已准备、传输已完成等状态。
- **通信内存**：包括本端CCL Buffer和通过Channel获取的远端CCL Buffer。

Thread Notify和Channel Notify的作用域不同，计算资源时应分别统计，不能只用一个Notify总数表示。

## Mesh全连接资源示例

以当前自定义AllGather AICPU样例的单机4卡Mesh全连接为例，每个rank与其余3个rank直接通信：

![Mesh算法硬件拓扑示例](./figures/mesh.png)

### Thread与Thread Notify

样例为每条并发通信路径分配一个算法Thread，因此每个rank需要3个算法Thread：

- 1个主Thread，负责本地拷贝以及协调从Thread。
- 2个从Thread，负责其余并发通信路径。

主从Thread同步是成对组织的：

1. 开始阶段，主Thread分别通知两个从Thread开始执行。
2. 结束阶段，两个从Thread分别通知主Thread已完成。

因此，从逻辑需求看：

| Thread类型 | 数量 | 每个Thread所需Notify | Notify小计 |
| --- | ---: | ---: | ---: |
| 主Thread | 1 | 2 | 2 |
| 从Thread | 2 | 1 | 2 |
| **合计** | **3** | - | **4** |

一般化后，若算法共有 `T` 个Thread，则有1个主Thread和 `T - 1` 个从Thread：

- 主Thread需要 `T - 1` 个Notify，分别对应每个从Thread。
- 每个从Thread需要1个Notify，用于与主Thread同步。
- 算法逻辑所需的Thread Notify总数为 `2 * (T - 1)`。

资源接口可能按“每个Thread的最大Notify数”统一申请，因此物理申请数量可能大于上述逻辑使用数量。应以算法实际使用的Notify索引为准，避免把接口的统一申请上限误认为每个Thread都会使用相同数量的Notify。

### Channel与Channel Notify

4卡Mesh中，每个rank需要连接另外3个rank，因此需要3条Channel。当前样例为每条Channel申请3个Notify：

- `ACK`：双方确认数据已经准备完成。
- `DATA_SIGNAL`：双方确认本轮数据传输已经完成。
- `FIN_ACK`：预留的完成确认Notify，便于后续扩展完整的结束同步。

因此每个rank的Channel Notify申请数量为 `3条Channel * 3 = 9`。当前任务编排主要使用前两个Notify索引，资源申请仍以代码中的Channel描述为准。

### Host与AICPU控制同步

除算法Thread外，样例还创建了一对用于Host与AICPU Kernel控制同步的Thread：

- Host Thread在Kernel下发前通知AICPU控制Thread可以开始执行。
- AICPU控制Thread完成任务编排后通知Host Thread。
- 两侧各使用1个控制Notify，该资源不计入算法主从Thread的4个逻辑Notify中。

单机4卡样例的资源汇总如下：

| 通信资源 | 每个rank的数量 | 说明 |
| --- | ---: | --- |
| 算法Thread | 3 | 1个主Thread，2个从Thread |
| 算法Thread Notify | 4 | 主Thread 2个，两个从Thread各1个 |
| Channel | 3 | 分别连接其余3个rank |
| Channel Notify | 9 | 每条Channel申请3个 |
| Host/AICPU控制Notify | 2 | Host和AICPU控制Thread各1个 |

## 代码示例

下面的代码片段展示当前AICPU自定义AllGather样例的主要资源计算方式。

1. 计算算法Thread和Thread Notify。

    ```cpp
    uint32_t threadNum = rankSize > 1 ? rankSize - 1 : 1;
    resource.slaveThreadNum = threadNum - 1;
    resource.notifyNumOnMainThread = resource.slaveThreadNum;
    resource.notifyNumPerThread =
        std::vector<uint32_t>(resource.slaveThreadNum, 1);
    ```

2. 申请算法Thread。接口按每个Thread的最大Notify数统一申请；样例额外预留1个Notify用于控制同步。

    ```cpp
    uint32_t maxNotifyNum = resource.notifyNumOnMainThread;
    std::vector<ThreadHandle> threads(threadNum);
    HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS,
                      threadNum, maxNotifyNum + 1, threads.data());
    ```

3. 为每个远端rank建立Channel，并设置Channel Notify数量。

    ```cpp
    constexpr uint32_t channelNotifyNum = 3;
    HcclChannelDesc desc;
    HcclChannelDescInit(&desc, 1);
    desc.remoteRank = remoteRank;
    desc.notifyNum = channelNotifyNum;
    desc.channelProtocol = selectedProtocol;
    HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS,
                       &desc, 1, &channelHandle);
    ```

4. 获取本端和远端通信内存，并写入资源Context。

    ```cpp
    void *localAddr = nullptr;
    uint64_t localSize = 0;
    HcclGetHcclBuffer(comm, &localAddr, &localSize);

    void *remoteAddr = nullptr;
    uint64_t remoteSize = 0;
    HcclChannelGetHcclBuffer(comm, channelHandle,
                             &remoteAddr, &remoteSize);
    ```

资源创建完成后，Host将Context序列化并拷贝到Device。AICPU Kernel启动后反序列化该Context，再使用其中的Thread、Notify、Channel和通信内存进行任务编排。
