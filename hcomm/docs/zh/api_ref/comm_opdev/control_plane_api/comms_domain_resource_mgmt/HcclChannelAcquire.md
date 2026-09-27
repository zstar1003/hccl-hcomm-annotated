# HcclChannelAcquire

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

基于通信域获取多个通信通道，如果通信域中没有对应的通信通道则直接创建。

channel是否复用，依据commId + engine + remoterank + channelProtocol组成的channel唯一标识判断。

## 函数原型

```c
HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| channelDescs | 输入 | 通信通道描述列表，列表长度为channelNum。<br>HcclChannelDesc类型的定义可参见[HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md)，可通过[HcclRankGraphGetLinks](../topo_info_query/HcclRankGraphGetLinks.md)获取链路信息后填充。 |
| channelNum | 输入 | 通信通道数量，channelNum的取值范围为(0, 1024 * 1024]。 |
| channels | 输出 | 通信通道句柄列表，通信通道句柄列表长度为channelNum。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 当前CommEngine配置为CCU时，不支持外部配置NotifyNum，默认分配4个CCU Notify。

2. 当前CommEngine配置为CCU时，不支持交换额外自定义内存，仅支持交换通信域的HcclBuffer。

3. AIV直驱URMA场景下，本端与远端参与交换的内存数量必须一致，否则Channel创建失败。

4. 同一个通信域内，本接口不支持并发调用，调用方须保证调用串行执行。

5. 当前各CommEngine支持的通信协议与芯片型号有关，具体如下：

   <!-- npu="950" id6 -->
   针对Ascend 950PR/Ascend 950DT，各通信引擎支持的通信协议如下：

   - COMM_ENGINE_CPU
     - COMM_PROTOCOL_ROCE
   - COMM_ENGINE_AICPU_TS
     - COMM_PROTOCOL_UBOE
     - COMM_PROTOCOL_UB_CTP
     - COMM_PROTOCOL_ROCE
   - COMM_ENGINE_AIV
     - COMM_PROTOCOL_UB_CTP
     - COMM_PROTOCOL_UB_MEM
     - COMM_PROTOCOL_ROCE
   - COMM_ENGINE_CCU
     - COMM_PROTOCOL_UB_CTP
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，各通信引擎支持的通信协议如下：

   - COMM_ENGINE_AICPU_TS
     - COMM_PROTOCOL_ROCE
     - COMM_PROTOCOL_HCCS
     - COMM_PROTOCOL_HCCS_ONLY
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，各通信引擎支持的通信协议如下：

   - COMM_ENGINE_CPU_TS
     - COMM_PROTOCOL_ROCE
     - COMM_PROTOCOL_HCCS
     - COMM_PROTOCOL_HCCS_ONLY
   <!-- end id8 -->

## 调用示例

以批量通信通道为例：

```c
// 1. 调用HcclRankGraphGetLinks获取链路信息
CommLink *linkList = nullptr;
uint32_t listSize;
CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));

// 2. 遍历每个CommLink，填充HcclChannelDesc
uint32_t channelNum = listSize;
std::vector<HcclChannelDesc> channelDescVec(channelNum);
for (uint32_t idx = 0; idx < listSize; idx++) {
  HcclChannelDesc channelDesc;
  HcclChannelDescInit(&channelDesc, 1);
  channelDesc.remoteRank = rank;

  CommLink link = linkList[idx];

  //  核心映射：从CommLink提取Endpoint信息
  channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
  channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
  channelDesc.localEndpoint.loc    = link.srcEndpointDesc.loc;
  channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
  channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
  channelDesc.remoteEndpoint.loc   = link.dstEndpointDesc.loc;
  channelDesc.channelProtocol     = link.linkAttr.linkProtocol;
  channelDesc.notifyNum = 4; // 由用户按需指定Notify数量

  channelDescVec[idx] = channelDesc;
}

// 3. 批量创建Channel
HcclComm comm;
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDescVec.data(), channelNum, channels.data());
```
