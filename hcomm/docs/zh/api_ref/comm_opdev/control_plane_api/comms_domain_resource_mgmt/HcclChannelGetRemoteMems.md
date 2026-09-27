# HcclChannelGetRemoteMems

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

获取通信通道中交换的远端内存信息。返回的远端内存列表包含所有远端内存（含通信域默认的HcclBuffer），而非仅用户注册的内存。

## 函数原型

```c
HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t *memNum, CommMem **remoteMems, char ***memTags)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| channel | 输入 | 通信通道句柄。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。 |
| memNum | 输出 | 内存数量。 |
| remoteMems | 输出 | 远端内存列表，包含所有远端内存（含通信域默认的HcclBuffer）。<br>CommMem类型的定义请参见[CommMem](../../datatype_definition/CommMem.md)。 |
| memTags | 输出 | 远端内存字符串列表。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
uint32_t channelNum = 1;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);

// 省略channelDesc配置

HcclComm comm;
CommEngine engine = CommEngine::COMM_ENGINE_AIV;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDesc.data(), channelNum, channels.data());

uint32_t memNum = 0;
CommMem* remoteMems = nullptr;
char** memTags = nullptr;
HcclChannelGetRemoteMems(comm, channels[0], &memNum, &remoteMems, &memTags);
```
