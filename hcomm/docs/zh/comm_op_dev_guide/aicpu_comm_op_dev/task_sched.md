# 任务编排

## 编排步骤

参与集合通信的各个rank协调有序地进行同步与数据搬运，完成一个集合通信操作的过程，称为任务编排。任务编排的主要目的是将不同通信线程上的任务并行执行，实现资源的最大化利用，从而提升整体性能。

任务编排主要包含以下几个步骤：

1. 获取本端通信内存，在HCCL中称其为HCCL Buffer。
2. 拷贝算子输入数据到HCCL Buffer，常用的数据面接口为[HcommLocalCopyOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommLocalCopyOnThread.md)。

   > [!NOTE]说明
   > HCCL Buffer是HCCL每个通信域所管理的一块Device上的锁页内存（Pinned Memory），由于通信任务是异步执行的，为确保用户输入的数据在通信任务实际执行时依然有效，因此需要首先将输入数据拷贝到拥有固定内存地址的HCCL Buffer中。

3. 切分输入数据，计算偏移量。

    HCCL Buffer默认大小是200MB，若输入数据大小超过该值，则需要切分成多个数据块，分别进行处理。

4. 前同步，主Thread通知从Thread启动任务，从Thread等待主Thread通知。常用的数据面接口为[HcommThreadNotifyRecordOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommThreadNotifyRecordOnThread.md)和[HcommThreadNotifyWaitOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommThreadNotifyWaitOnThread.md)。
5. 进行数据搬运，将远端数据拷贝至本端的HCCL Buffer中。常用的数据面接口为[HcommReadOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/communication_operations/HcommReadOnThread.md)、[HcommReadReduceOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/communication_operations/HcommReadReduceOnThread.md)等。
6. 后同步，从流通知主流任务完成，主流等待从流通知。
7. 拷贝HCCL Buffer中的结果数据到算子输出内存。常用的数据面接口为[HcommLocalCopyOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommLocalCopyOnThread.md)。

## 代码示例

- 以自定义Send算子为例，其在AI CPU侧的任务编排代码片段如下：

    ```c
    uint64_t size = count * dataTypeSize;  // 数据量 = 数据数量 * 数据类型大小
    // 1. 拷贝到中转内存
    HcommLocalCopyOnThread(threadHandle, localAddr, inputPtr, size);
    // 2. 通知接收端，本端已经准备好数据
    HcommChannelNotifyRecordOnThread(threadHandle, channelHandle, 0);
    // 3. 等待接收端，告知已经读完本端数据
    HcommChannelNotifyWaitOnThread(threadHandle, channelHandle, 1, 1800);
    ```

- 以自定义Receive算子为例，其在AI CPU侧的任务编排代码片段如下：

    ```c
    // 1. 等待发送端，告知本端可以开始读数据
    HcommChannelNotifyWaitOnThread(threadHandle, channelHandle, 0, 1800);
    // 2. 读取发送端的数据
    HcommReadOnThread(threadHandle, channelHandle, outputPtr, remoteAddr, size);
    // 3. 通知发送端，本端已经读完数据
    HcommChannelNotifyRecordOnThread(threadHandle, channelHandle, 1);
    ```
