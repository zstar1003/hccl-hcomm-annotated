# HcommChannelCreate

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

该接口为创建通信通道的资源管理接口，基于已创建的网络端点（Endpoint），根据给定的通道描述信息批量创建通信通道，为点对点通信或集合通信提供数据传输的基础设施。

该接口执行后仅完成通道对象的创建，**不会立即建立链路连接**。调用者需在后续通过[HcommChannelGetStatus](HcommChannelGetStatus.md)接口推动建链状态机，直至通道状态变为就绪后方可进行通信操作。

## 函数原型

```c
HcommResult HcommChannelCreate(EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| endpointHandle | 输入 | 网络设备端点句柄，标识一个已创建的本地网络设备端点。<br>EndpointHandle类型的定义请参见[EndpointHandle](../../datatype_definition/EndpointHandle.md)，该句柄必须通过[HcommEndpointCreate](HcommEndpointCreate.md)成功创建，且未销毁。 |
| engine | 输入 | 通信引擎类型，指定通道的执行位置。<br>CommEngine类型的定义请参见[CommEngine](../../datatype_definition/CommEngine.md)。<br>需要注意：必须是有效的引擎类型。 |
| channelDescs | 输入 | 通道描述符数组，每个元素描述一个待创建通道的属性信息。<br>HcommChannelDesc类型的定义请参见[HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md)。<br>数组元素数量必须等于channelNum，每个元素需正确填充必要字段。 |
| channelNum | 输入 | 待创建的通道数量。<br>单位：个，取值范围：[1, 1048576]。<br>该参数需要大于0。 |
| channels | 输出 | 通道句柄数组，用于返回创建成功的通道句柄列表。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。<br>调用者分配的数组，需要至少包含channelNum个元素的空间。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- channelDescs数组长度必须与channelNum参数一致。
- HcommChannelDesc中的remoteEndpoint必须正确填充远端端点信息。
- 当HcommChannelDesc中exchangeAllMems为false时，必须配置memHandles和memHandleNum。
- AIV直驱URMA场景下，本端与远端参与交换的内存数量必须一致，否则Channel创建失败。
- 当前CommEngine配置为CCU时，仅支持交换1份memHandle。
- 当前CommEngine配置为CCU时，不支持外部配置NotifyNum，默认为4个CCU Notify。
- 当前各CommEngine支持的通信协议与芯片型号有关，具体如下：

  <!-- npu="950" id6 -->
  针对Ascend 950PR/Ascend 950DT，各通信引擎支持的通信协议如下：

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UB_CTP
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_ROCE
  <!-- end id6 -->

  **注意事项：**
    - Atlas 350加速卡不支持COMM_ENGINE_CPU通信引擎以及相应的通信协议
    - Atlas 350加速卡不支持COMM_ENGINE_AICPU_TS通信引擎的COMM_PROTOCOL_ROCE通信协议

## 调用示例

```c
EndpointHandle endpointHandle = nullptr;
 // ... 创建端点的代码（省略）

 // 创建多个通道
 const uint32_t CHANNEL_NUM = 4;
 HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};
 ChannelHandle channels[CHANNEL_NUM] = {0};

 // 准备通道描述符并创建通道
 for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
     // ... 填充channelDescs[i]
 }

 HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU,
                                      channelDescs, CHANNEL_NUM, channels);
 if (ret != 0) {
     printf("Failed to create channels, ret = %d\n", ret);
     HcommEndpointDestroy(endpointHandle);
     return ret;
 }

 printf("%u channels created successfully\n", CHANNEL_NUM);
```
