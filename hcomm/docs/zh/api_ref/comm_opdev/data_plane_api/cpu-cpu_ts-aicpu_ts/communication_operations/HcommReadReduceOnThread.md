# HcommReadReduceOnThread

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

从channel上的指定内存读数据，从src中读取长度为count\*sizeof\(dataType\)的内存数据，与dst所指向的相同长度的内存数据进行reduceOp操作，并将结果输出到dst中。接口调用方为dst所在节点。

## 函数原型

```c
int32_t HcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 通信线程句柄，为通过[HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)接口获取到的threads。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| channel | 输入 | 通信通道句柄，为通过[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口获取到的channels。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。 |
| dst | 输出 | 目的内存地址，使用[HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md)、[HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md)获取到的内存。 |
| src | 输入 | 源内存地址，使用[HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md)、[HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md)获取到的内存。 |
| count | 输入 | 元素个数。 |
| dataType | 输入 | 数据类型。<br>HcommDataType类型的定义请参见[HcommDataType](../../../datatype_definition/HcommDataType.md)。<br>不同产品型号下支持的数据类型不同，详细请参见[dataType说明](#dataType说明)。|
| reduceOp | 输入 | 归约操作类型，支持：sum、max、min。<br>HcommReduceOp类型的定义请参见[HcommReduceOp](../../../datatype_definition/HcommReduceOp.md)。 |

### dataType说明

<!-- npu="950" id6 -->
- 针对Ascend 950PR/Ascend 950DT，支持的数据类型：int8、int16、int32、uint8、uint16、uint32、fp16、fp32、bfp16。
<!-- end id6 -->
<!-- npu="A3" id7 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、float16、float32、bfp16。
<!-- end id7 -->
<!-- npu="910b" id8 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、float16、float32、bfp16。
<!-- end id8 -->

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

无

## 调用示例

```c
// 申请通信线程资源
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS; // Atlas A3 训练系列产品/Atlas A3 推理系列产品使用
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS; // Ascend 950PR/Ascend 950DT使用
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
uint64_t count = std::min(localBufferSize/sizeof(uint64_t), remoteBufferSize/sizeof(uint64_t));

// 将本端内存和对端内存数据进行reduce，输出到本端内存上
HcommReadReduceOnThread(thread, channel, localBuffer, remoteBuffer, count, HCOMM_DATA_TYPE_INT32, HCOMM_REDUCE_SUM);
```
