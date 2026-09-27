# HcommChannelNotifyWait

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

等待同步信号，阻塞等待指定Channel上的Notify完成。

## 函数原型

```c
int32_t HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channel | 输入 | 通信通道句柄，为通过[HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口获取到的channels。关于channel的约束参见约束说明。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。 |
| localNotifyIdx | 输入 | 本地Notify索引。<br>取值范围：[0, notifyNum)。<br>notifyNum为[HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口传入的channelDesc参数中的notifyNum。 |
| timeOut | 输入 | 超时时间，单位：秒。针对Ascend 950PR/Ascend 950DT的CPU引擎RoCE场景，需配置大于0的超时时间。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

<!-- npu="950" id6 -->
- 针对Ascend 950PR/Ascend 950DT，本接口仅支持在Host CPU上调用，不支持在AICPU侧调用。
- 调用[HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)申请入参channel时，需传入`engine = COMM_ENGINE_CPU`，且`channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`。URMA/UBC等协议通道当前不支持本接口。
- `localNotifyIdx`必须小于本端通信通道的Notify数量，且通信通道创建时的`notifyNum`需大于0；`timeOut`需大于0。
<!-- end id6 -->
- 本接口需要配合[HcommChannelNotifyRecord](HcommChannelNotifyRecord.md)使用。

## 调用示例

```c
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
HcclComm comm;

// 申请通信通道资源
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
channelDesc.localEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.notifyNum = 1;
// 省略：channelDesc填充其他信息
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 以下接口在Host CPU侧调用

// 通知对端
HcommChannelNotifyRecord(channel, 0);

// 数据面操作
// ...

// 等待对端通知本端
uint32_t notifyTimeout = 1800;
HcommChannelNotifyWait(channel, 0, notifyTimeout);
```
