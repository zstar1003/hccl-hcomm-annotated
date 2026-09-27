# HcommChannelCreateWithConfig

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

通过配置对象创建通信通道，是[HcommChannelCreate](HcommChannelCreate.md)的增强版本，支持通过配置对象传入共享Jetty等高级配置。

当config为nullptr或未设置IS_SHARED_QUEUE时，行为与[HcommChannelCreate](HcommChannelCreate.md)完全等价。

当config中IS_SHARED_QUEUE=true时，启用共享Jetty模式：
- 使用相同endpointHandle多次调用本接口创建的Channel，若源目的endpointPair相同，则共享同一个底层Jetty资源，实现通信资源的复用。
- 适用于需要频繁创建/销毁通信通道的场景，可显著降低建链开销。

该接口执行后仅完成通道对象的创建，**不会立即建立链路连接**。调用者需在后续通过[HcommChannelGetStatus](HcommChannelGetStatus.md)接口推动建链状态机，直至通道状态变为就绪后方可进行通信操作。

## 函数原型

```c
HcommResult HcommChannelCreateWithConfig(EndpointHandle endpointHandle, CommEngine engine,
    HcommChannelDesc *channelDescs, uint32_t channelNum, HcommChannelConfig config, ChannelHandle *channels)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| endpointHandle | 输入 | 网络设备端点句柄，标识一个已创建的本地网络设备端点。<br>EndpointHandle类型的定义请参见[EndpointHandle](../../datatype_definition/EndpointHandle.md)，该句柄必须通过[HcommEndpointCreate](HcommEndpointCreate.md)成功创建，且未销毁。 |
| engine | 输入 | 通信引擎类型，指定通道的执行位置。<br>CommEngine类型的定义请参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| channelDescs | 输入 | 通道描述符数组，每个元素描述一个待创建通道的属性信息。<br>HcommChannelDesc类型的定义请参见[HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md)。 |
| channelNum | 输入 | 待创建的通道数量，取值范围：[1, 1048576]。 |
| config | 输入 | Channel配置对象指针，可为nullptr（等价于[HcommChannelCreate](HcommChannelCreate.md)）。<br>HcommChannelConfig类型的定义请参见[HcommChannelConfig](../../datatype_definition/HcommChannelConfig.md)。<br>通过[HcommChannelConfigCreate](HcommChannelConfigCreate.md)创建。 |
| channels | 输出 | 通道句柄数组，用于返回创建成功的通道句柄列表。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。<br>调用者分配的数组，需要至少包含channelNum个元素的空间。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. channelDescs数组长度必须与channelNum参数一致。

2. 当config为nullptr时，行为与[HcommChannelCreate](HcommChannelCreate.md)完全等价。

3. 当IS_SHARED_QUEUE=true时，有以下额外约束：
   - 仅支持UB网络语义协议（COMM_PROTOCOL_UB_CTP/COMM_PROTOCOL_UB_RTP），不支持UBMem/RoCE/UBOE。
   - 使用相同endpointHandle多次调用本接口创建的Channel共享同一个Jetty。
   - 使用不同endpointHandle调用创建的Channel不共享Jetty。
   - 共享Jetty的不同Channel不支持并发使用，需由调用者按业务顺序串行调用。
   - 销毁endpointHandle前需确保所有共享Jetty的Channel已销毁。

4. config对象在本接口调用完成后即可通过[HcommChannelConfigDestroy](HcommChannelConfigDestroy.md)销毁，不影响已创建的Channel。

5. 各CommEngine支持的通信协议与芯片型号有关，具体如下：

   <!-- npu="950" id6 -->
   针对Ascend 950PR/Ascend 950DT，仅支持AIV引擎的UB网络语义协议（UB_CTP/UB_RTP）：
   - COMM_ENGINE_AIV
     - COMM_PROTOCOL_UB_CTP
     - COMM_PROTOCOL_UB_RTP
   <!-- end id6 -->

## 调用示例

以共享Jetty模式创建AIV通信通道为例：

```c
EndpointHandle endpointHandle = nullptr;
// ... 创建端点的代码（省略）

// 1. 创建并配置Channel配置对象
HcommChannelConfig config = nullptr;
HcommChannelConfigCreate(&config);
HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);

// 2. 准备通道描述符
const uint32_t CHANNEL_NUM = 4;
HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};
ChannelHandle channels[CHANNEL_NUM] = {0};
for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
    HcommChannelDescInit(&channelDescs[i], 1);
    channelDescs[i].remoteEndpoint.protocol = COMM_PROTOCOL_UB_CTP;
    // 填充localEndpoint / remoteEndpoint ...
}

// 3. 使用配置创建通信通道
HcommResult ret = HcommChannelCreateWithConfig(endpointHandle, COMM_ENGINE_AIV,
    channelDescs, CHANNEL_NUM, config, channels);
if (ret != 0) {
    printf("Failed to create channels with config, ret = %d\n", ret);
    HcommChannelConfigDestroy(config);
    HcommEndpointDestroy(endpointHandle);
    return ret;
}

// 4. 销毁配置对象（Channel已创建，配置对象不再需要）
HcommChannelConfigDestroy(config);

// 5. 后续通过HcommChannelGetStatus推动建链状态机
```
