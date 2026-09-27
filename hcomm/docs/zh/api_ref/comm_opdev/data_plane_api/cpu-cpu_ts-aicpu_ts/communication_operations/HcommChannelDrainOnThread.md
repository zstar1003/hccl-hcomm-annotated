# HcommChannelDrainOnThread

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

阻塞等待指定通道上已提交的通信操作自然完成，直到待完成任务为空。

本接口用于等待调用前已经提交的任务完成，不为调用后提交的任务增加保序约束。若需要保证屏障前后的通道读写操作顺序，请调用[HcommChannelFenceOnThread](HcommChannelFenceOnThread.md)。

## 函数原型

```c
int32_t HcommChannelDrainOnThread(ThreadHandle thread, ChannelHandle channel)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 通信线程句柄。AI CPU侧调用时，为通过[HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)接口获取到的thread；Host CPU侧调用时，可传入0。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| channel | 输入 | 通信通道句柄，为通过[HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口获取到的channel。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

<!-- npu="A3,910b" id6 -->
- 针对如下产品，在AI CPU侧调用此接口时，通信引擎为AICPU_TS，仅支持通信协议RoCE。
    <!-- npu="A3" id7 -->
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
    <!-- end id8 -->
<!-- end id6 -->
<!-- npu="950" id9 -->
- 针对Ascend 950PR/Ascend 950DT，在Host CPU侧调用时，申请入参`channel`使用的通信引擎须为`COMM_ENGINE_CPU`，且`channelDesc.remoteEndpoint.protocol`须为`COMM_PROTOCOL_ROCE`或`COMM_PROTOCOL_UB_CTP`。
<!-- end id9 -->
- 同一个`ChannelHandle`不支持多线程并发访问。

## 调用示例

### AI CPU侧调用示例

```c
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
HcclComm comm;
ThreadHandle thread;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 申请通信通道资源
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 获取本端通信内存信息
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// 获取对端通信内存信息
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// 将对端内存的内容读到本端内存上
HcommReadOnThread(thread, channel, localBuffer, remoteBuffer, len);

HcommChannelDrainOnThread(thread, channel);
```

### Host CPU侧调用示例

```c
// endpointHandle为已经创建的Endpoint，channelDesc已根据对端Endpoint信息完成配置。
HcommResult DrainHostCpuChannel(EndpointHandle endpointHandle, HcommChannelDesc *channelDesc)
{
    ChannelHandle channel = 0;
    HcommResult result = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU, channelDesc, 1, &channel);
    if (result != 0) {
        printf("Failed to create channel, result = %d\n", result);
        return result;
    }

    // 使用channel提交通信任务（省略）。

    // 等待channel上已经提交的通信任务完成。Host CPU侧的thread参数传入0。
    result = HcommChannelDrainOnThread(0, channel);
    if (result != 0) {
        printf("Failed to drain channel, result = %d\n", result);
        return result;
    }
    return 0;
}
```
