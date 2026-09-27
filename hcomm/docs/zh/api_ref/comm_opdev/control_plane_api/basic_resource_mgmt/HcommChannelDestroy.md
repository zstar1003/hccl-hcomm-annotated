# HcommChannelDestroy

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

该接口为销毁通信通道的资源管理接口，释放由[HcommChannelCreate](HcommChannelCreate.md)创建的通信通道及其占用的所有系统资源，包括网络连接、同步信号、通信队列等。

该接口支持批量销毁，可在单次调用中释放多个通道，提高资源释放效率。

## 函数原型

```c
HcommResult HcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channels | 输入 | 待销毁的通道句柄数组，每个元素标识一个已创建的通信通道。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。<br>该参数不能为空指针，数组中每个通道句柄必须是通过[HcommChannelCreate](HcommChannelCreate.md)创建的有效句柄（未销毁的句柄）。 |
| channelNum | 输入 | 待销毁的通道数量。<br>单位为“个”，取值范围：[1, 1048576]。<br>该参数必须大于0，且等于channels数组中有效句柄的数量。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- channels数组长度需要与channelNum参数一致。
- 对于通过不同引擎创建的通道，可批量销毁。
- 该销毁操作逐个执行，遇到失败立即返回，已销毁通道不回滚，非原子操作。

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

 // 使用通道进行通信
 // ...

 // 批量销毁所有通道
 ret = HcommChannelDestroy(channels, CHANNEL_NUM);
 if (ret != 0) {
     printf("Failed to destroy channels, ret = %d\n", ret);
 } else {
     printf("All channels destroyed successfully\n");
 }

 // 销毁端点
 HcommEndpointDestroy(endpointHandle);
```
