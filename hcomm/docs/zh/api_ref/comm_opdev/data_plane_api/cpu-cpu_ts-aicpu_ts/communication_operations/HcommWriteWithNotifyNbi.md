# HcommWriteWithNotifyNbi

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

向channel上的指定内存写数据，将src中长度为len的内存数据写入dst所指向的相同长度的内存区域，并向dst所在节点发送同步信号。接口调用方为src所在节点，该接口为非阻塞接口。

## 函数原型

```c
int32_t HcommWriteWithNotifyNbi(ChannelHandle channel, void *dst, const void *src, uint64_t len,
    uint32_t remoteNotifyIdx)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channel | 输入 | 通信通道句柄，为通过[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口获取到的channels。关于channel的约束参见约束说明。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。 |
| dst | 输出 | 目的内存地址，使用[HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md)、[HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md)获取到的对端内存地址。 |
| src | 输入 | 源内存地址，使用[HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md)、[HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md)获取到的本端内存地址。 |
| len | 输入 | 数据长度（字节），需大于0。 |
| remoteNotifyIdx | 输入 | 通信通道另一端的Notify索引。<br>取值范围：[0, notifyNum)。<br>notifyNum为[HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口传入的channelDesc参数中的notifyNum。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

<!-- npu="950" id6 -->
- 针对Ascend 950PR/Ascend 950DT，本接口仅支持在Host CPU上调用，不支持在AICPU侧调用。
- 调用[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)申请入参channel时，需传入`engine = COMM_ENGINE_CPU`，且`channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`。URMA/UBC等协议通道当前不支持本接口。
<!-- end id6 -->
- `[src, src + len)`必须落在本端已注册或获取的内存范围内，`[dst, dst + len)`必须落在对端已导入或获取的内存范围内。
- `remoteNotifyIdx`必须小于通信通道另一端的Notify数量，且通信通道创建时的`notifyNum`需大于0。
- 写入目标端应调用[HcommChannelNotifyWait](HcommChannelNotifyWait.md)等待同步信号。
- 该接口返回成功仅表示带通知写请求提交成功。如需确认本端通道上已提交的写操作完成，请调用[HcommChannelFence](HcommChannelFence.md)。

## 调用示例

```c
// 省略：创建通信域句柄comm

// 申请通信通道资源
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclResult result = HcclChannelDescInit(&channelDesc, channelNum);
channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
channelDesc.localEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.notifyNum = 1;
// 省略：channelDesc填充其他信息
ChannelHandle channel;
result = HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 获取本端通信内存信息
void * localBuffer;
uint64_t localBufferSize;
result = HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// 获取对端通信内存信息
void * remoteBuffer;
uint64_t remoteBufferSize;
result = HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// 将本端内存的内容写到对端内存上并通知对端
uint32_t rmtNotifyIdx = 0;
int32_t ret = HcommWriteWithNotifyNbi(channel, remoteBuffer, localBuffer, len, rmtNotifyIdx);

// 等待通信通道上已提交的写操作完成
ret = HcommChannelFence(channel);
```

以下代码在写入目标端调用，等待写发起端发送的同步信号。

```c
// 省略：资源创建

// 使用本地Notify索引等待写发起端发送的同步信号
uint32_t lclNotifyIdx = 0;
uint32_t notifyTimeout = 1800;
int32_t ret = HcommChannelNotifyWait(channel, lclNotifyIdx, notifyTimeout);
```
