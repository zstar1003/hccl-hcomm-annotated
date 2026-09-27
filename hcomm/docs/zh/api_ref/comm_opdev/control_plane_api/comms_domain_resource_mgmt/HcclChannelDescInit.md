# HcclChannelDescInit

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

初始化通信通道描述列表。

## 函数原型

```c
HcclResult HcclChannelDescInit(HcclChannelDesc *channelDesc, uint32_t descNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channelDesc | 输出 | 通信通道描述列表，列表长度为descNum，函数会初始化该结构体。<br>HcclChannelDesc类型的定义可参见[HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md)。 |
| descNum | 输入 | 通信通道描述数量。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

HcclChannelDesc结构体必须调用该接口进行初始化。

## 调用示例

以初始化通信通道描述数量为2的通信通道描述列表为例：

```c
uint32_t channelNum = 2;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);
```
