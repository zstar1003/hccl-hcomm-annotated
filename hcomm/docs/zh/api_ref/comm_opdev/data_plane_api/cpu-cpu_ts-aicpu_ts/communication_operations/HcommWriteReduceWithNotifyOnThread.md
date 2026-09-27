# HcommWriteReduceWithNotifyOnThread

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
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

向channel上的指定内存写数据，将src中长度为count\*sizeof\(dataType\)的内存数据，与dst所指向的相同长度的内存数据进行reduceOp操作，并将结果输出到dst中，并向dst所在节点发送同步信号。接口调用方为src所在节点，该接口为异步接口。

## 函数原型

```c
int32_t HcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 通信线程句柄，为通过[HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)接口获取到的threads。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| channel | 输入 | 通信通道句柄，为通过[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口获取到的channels。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。 |
| dst | 输出 | 目的内存地址，使用指定channel对端的HCCL通信内存。 |
| src | 输入 | 源内存地址，使用通信域中本端rank的HCCL通信内存。 |
| count | 输入 | 元素个数。 |
| dataType | 输入 | 数据类型。<br>HcommDataType类型的定义请参见[HcommDataType](../../../datatype_definition/HcommDataType.md)。<br>不同产品型号支持的数据类型定义不同，详细请参见[dataType说明](#dataType说明)。|
| reduceOp | 输入 | 归约操作类型，支持：sum、max、min。<br>HcommReduceOp类型的定义请参见[HcommReduceOp](../../../datatype_definition/HcommReduceOp.md)。 |
| remoteNotifyIdx | 输入 | 通信通道另一端的Notify索引。<br>取值范围：[0, [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口传入的channelDescs参数中的notifyNum)。 |

### dataType说明

<!-- npu="950" id6 -->
针对Ascend 950PR/Ascend 950DT，支持的数据类型：int8、int16、int32、uint8、uint16、uint32、float16、float32、bfp16。
<!-- end id6 -->

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

该接口需要配合[HcommChannelNotifyWaitOnThread](HcommChannelNotifyWaitOnThread.md)使用。

在Ascend 950PR/Ascend 950DT上，仅支持AICPU_TS模式下、在Device侧调用该接口。

## 调用示例

```c
// 申请通信线程资源
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
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

// 拷贝参数并Launch Kernel
// Device侧算法编排
uint64_t len = std::min(localBufferSize, remoteBufferSize);
uint64_t sizeOfFP32 = 4;
uint64_t count = len / sizeOfFP32;

// 将本端内存的内容写到对端内存上并通知对端
uint32_t rmtNotifyIdx = 0;
HcommWriteReduceWithNotifyOnThread(thread, channel, remoteBuffer, localBuffer, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM, rmtNotifyIdx);

// 数据面操作
// ...

// 等待对端通知本端
uint32_t lclNotifyIdx = 0;
uint32_t notifyTimeout = 0;
HcommChannelNotifyWaitOnThread(thread, channel, lclNotifyIdx, notifyTimeout);
```
