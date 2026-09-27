# HcclChannelAcquireWithConfig

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
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

基于通信域和配置对象创建通信通道，是[HcclChannelAcquire](HcclChannelAcquire.md)的增强版本，支持通过配置对象传入共享Jetty等高级配置。

当config为nullptr或未设置IS_SHARED_QUEUE时，行为与[HcclChannelAcquire](HcclChannelAcquire.md)完全等价。

当config中IS_SHARED_QUEUE=true时，启用共享Jetty模式：
- 使用相同SHARED_QUEUE_TAG多次调用本接口创建的Channel，若源目的endpointPair相同，则共享同一个底层Jetty资源，实现通信资源的复用。
- 适用于需要频繁创建/销毁通信通道的场景，可显著降低建链开销。

## 函数原型

```c
HcclResult HcclChannelAcquireWithConfig(HcclComm comm, CommEngine engine,
    const HcclChannelDesc *channelDescs, uint32_t channelNum, HcclChannelConfig config, ChannelHandle *channels)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义请参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| channelDescs | 输入 | 通信通道描述列表，列表长度为channelNum。<br>HcclChannelDesc类型的定义请参见[HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md)。<br>**必须使用[HcclChannelDescInit](HcclChannelDescInit.md)进行初始化。** |
| channelNum | 输入 | 通信通道数量，取值范围为(0, 1024 * 1024]。 |
| config | 输入 | Channel配置对象指针，可为nullptr（等价于[HcclChannelAcquire](HcclChannelAcquire.md)）。<br>HcclChannelConfig类型的定义请参见[HcclChannelConfig](../../datatype_definition/HcclChannelConfig.md)。<br>通过[HcclChannelConfigCreate](HcclChannelConfigCreate.md)创建。 |
| channels | 输出 | 通信通道句柄列表，列表长度为channelNum。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. channelDescs必须使用[HcclChannelDescInit](HcclChannelDescInit.md)进行初始化。

2. 当config为nullptr时，行为与[HcclChannelAcquire](HcclChannelAcquire.md)完全等价。

3. 当IS_SHARED_QUEUE=true时，有以下额外约束：
   - 仅支持AIV引擎。
   - 仅支持UB网络语义协议（COMM_PROTOCOL_UB_CTP/COMM_PROTOCOL_UB_RTP），不支持UBMem/RoCE/UBOE。
   - 必须设置SHARED_QUEUE_TAG（非空字符串）。
   - channelDescs中所有元素的localEndpoint必须相同（复用的Jetty只能关联一个endpoint）。
   - 仅支持V2通信器（需通过HcclCommInitClusterInfoConfig等V2接口创建通信域）。
   - 共享Jetty的不同Channel不支持并发使用，需由调用者按业务顺序串行调用。

4. 共享Jetty模式下，相同tag+endpointPair的Channel复用规则：
   - 重复调用时，若已有Channel数量不足则创建新的补充，足够则直接按序返回已有Channel。
   - Channel的销毁由通信域统一管理，调用者无需单独销毁。

5. config对象在本接口调用完成后即可通过[HcclChannelConfigDestroy](HcclChannelConfigDestroy.md)销毁，不影响已创建的Channel。

## 调用示例

以共享Jetty模式创建AIV通信通道为例：

```c
// 1. 创建并配置Channel配置对象
HcclChannelConfig config = nullptr;
HcclChannelConfigCreate(&config);
HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "layer1_shared");

// 2. 准备通道描述符
const uint32_t CHANNEL_NUM = 4;
HcclChannelDesc channelDescs[CHANNEL_NUM];
ChannelHandle channels[CHANNEL_NUM] = {0};
for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
    HcclChannelDescInit(&channelDescs[i], 1);
    channelDescs[i].remoteRank = remoteRanks[i];
    channelDescs[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
    // 填充localEndpoint / remoteEndpoint ...
}

// 3. 使用配置创建通信通道
HcclResult ret = HcclChannelAcquireWithConfig(comm, COMM_ENGINE_AIV,
    channelDescs, CHANNEL_NUM, config, channels);
if (ret != HCCL_SUCCESS) {
    printf("Failed to acquire channels with config, ret = %d\n", ret);
    HcclChannelConfigDestroy(config);
    return ret;
}

// 4. 销毁配置对象（Channel已创建，配置对象不再需要）
HcclChannelConfigDestroy(config);

// 5. 后续可重复调用，相同tag的Channel会复用底层Jetty
// HcclChannelAcquireWithConfig(comm, COMM_ENGINE_AIV,
//     channelDescs2, CHANNEL_NUM2, config2, channels2);
```
