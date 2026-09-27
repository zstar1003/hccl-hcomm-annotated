# HcclChannelDestroy

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

销毁指定通信域中已创建的通信通道，释放对应的通道资源。销毁后该通道句柄不可再使用，如再次使用需重新调用[HcclChannelAcquire](HcclChannelAcquire.md)获取。

## 函数原型

```c
HcclResult HcclChannelDestroy(HcclComm comm, const ChannelHandle* channels, uint32_t channelNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| channels | 输入 | 待销毁的通信通道句柄列表，列表长度为channelNum，句柄需为[HcclChannelAcquire](HcclChannelAcquire.md)返回的有效句柄。 |
| channelNum | 输入 | 通信通道数量，channelNum的取值范围为(0, 1024 * 1024]。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 本接口暂只支持销毁CCU通信引擎的channel。

2. channels中的句柄应为通过[HcclChannelAcquire](HcclChannelAcquire.md)获取的有效通道句柄；销毁后该句柄不可再使用。

3. 对无效句柄或已销毁的句柄再次调用本接口，返回HCCL_E_NOT_FOUND。

4. 批量销毁时，如果部分通道销毁失败，本接口返回首个失败的错误码，其余通道仍会继续尝试销毁。

5. 同一通信域内，本接口不得与[HcclChannelAcquire](HcclChannelAcquire.md)、[HcclChannelQuery](HcclChannelQuery.md)并发调用，调用方须保证相关调用串行执行。

6. 同一个通信域内，本接口不支持并发调用，调用方须保证调用串行执行。

## 调用示例

以通道的获取、使用与销毁为例。通道获取（含channelDesc构造与字段填充）步骤参见[HcclChannelAcquire](HcclChannelAcquire.md)调用示例：

```c
// 1. 获取通信通道
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
std::vector<ChannelHandle> channels(channelNum);
HcclResult ret = HcclChannelAcquire(comm, engine, channelDescVec.data(), channelNum, channels.data());
if (ret != HCCL_SUCCESS) {
    // 错误处理
    return;
}

// 2. 使用通道完成通信操作
// ...

// 3. 销毁通信通道
ret = HcclChannelDestroy(comm, channels.data(), channelNum);
if (ret != HCCL_SUCCESS) {
    // 错误处理
    return;
}
```
