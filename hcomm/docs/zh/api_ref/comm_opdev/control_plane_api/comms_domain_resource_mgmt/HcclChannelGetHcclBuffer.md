# HcclChannelGetHcclBuffer

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

获取指定channel对端的HCCL通信内存。

## 函数原型

```c
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void **buffer, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| channel | 输入 | 通信通道句柄。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。 |
| buffer | 输出 | HCCL通信内存地址。 |
| size | 输出 | HCCL通信内存大小。内存大小为通信域初始化配置或环境变量HCCL_BUFFSIZE配置值的2倍，默认为400MB。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无。

## 调用示例

```c
// 设置当前线程操作的Device
aclrtSetDevice(0);

// 初始化Channel描述符
uint32_t channelNum = 1;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);

// 对端为Rank1
channelDesc[0].remoteRank = 1;
channelDesc[0].channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
channelDesc[0].notifyNum = 3;

// 通信域句柄
HcclComm comm;
// 创建Channel资源
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDesc.data(), channelNum, channels.data());

// 获取对端（Rank1）的HCCL Buffer地址和大小
void *remoteBufferAddr;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channels[0], &remoteBufferAddr, &remoteBufferSize);
```
