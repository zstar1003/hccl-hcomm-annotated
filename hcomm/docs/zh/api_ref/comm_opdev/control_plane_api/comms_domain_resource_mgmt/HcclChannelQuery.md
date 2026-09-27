# HcclChannelQuery

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

在调用[HcclChannelAcquire](HcclChannelAcquire.md)之前，查询指定的通信通道描述（channelDescs）对应的通道是否已存在：已存在则返回对应通道句柄，可直接复用；不存在则返回0（空句柄），表示后续调用[HcclChannelAcquire](HcclChannelAcquire.md)时需要新建。

## 函数原型

```c
HcclResult HcclChannelQuery(HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| channelDescs | 输入 | 通信通道描述列表，列表长度为channelNum，必须使用[HcclChannelDescInit](HcclChannelDescInit.md)初始化。<br>HcclChannelDesc类型的定义可参见[HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md)。 |
| channelNum | 输入 | 通信通道数量，channelNum的取值范围为(0, 1024 * 1024]。 |
| channels | 输出 | 查询结果句柄列表，长度为channelNum；通道已存在返回对应句柄，不存在返回0（空句柄）。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. channelDescs必须使用[HcclChannelDescInit](HcclChannelDescInit.md)进行初始化。

2. 本端与对端endpoint的loc类型不同（如hostNIC与deviceNIC之间）的通道不可复用，查询结果恒为0，表示每次调用[HcclChannelAcquire](HcclChannelAcquire.md)都会新建通道。

3. 查询返回的通道句柄与[HcclChannelAcquire](HcclChannelAcquire.md)返回的句柄可对应使用。

4. 同一通信域内，本接口不得与[HcclChannelAcquire](HcclChannelAcquire.md)、[HcclChannelDestroy](HcclChannelDestroy.md)并发调用，调用方须保证相关调用串行执行。

5. 同一个通信域内，本接口不支持并发调用，调用方须保证调用串行执行。

## 调用示例

以批量查询通信通道为例。channelDesc的构造与字段填充步骤与[HcclChannelAcquire](HcclChannelAcquire.md)调用示例一致：

```c
// 1. 构造channelDesc列表
HcclComm comm;   // 已创建的通信域
CommEngine engine = CommEngine::COMM_ENGINE_CCU;
uint32_t channelNum = 2;
std::vector<HcclChannelDesc> channelDescVec(channelNum);
for (uint32_t idx = 0; idx < channelNum; idx++) {
    HcclChannelDesc channelDesc;
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    // 按[HcclChannelAcquire](HcclChannelAcquire.md)调用示例填充channelDesc字段
    channelDescVec[idx] = channelDesc;
}

// 2. 查询哪些通道已存在
std::vector<ChannelHandle> channels(channelNum);
HcclResult ret = HcclChannelQuery(comm, engine, channelDescVec.data(), channelNum, channels.data());
if (ret != HCCL_SUCCESS) {
    // 错误处理
    return;
}
for (uint32_t idx = 0; idx < channelNum; idx++) {
    if (channels[idx] != 0) {
        // 通道已存在，可直接复用
    } else {
        // 通道不存在，后续调用HcclChannelAcquire时会被新建
    }
}
```
