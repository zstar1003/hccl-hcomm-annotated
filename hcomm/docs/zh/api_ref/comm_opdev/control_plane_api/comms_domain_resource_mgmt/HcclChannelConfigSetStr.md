# HcclChannelConfigSetStr

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

设置Channel配置对象的字符串型属性。目前支持的属性类型请参见[HcclChannelConfigType](../../datatype_definition/HcclChannelConfigType.md)。

## 函数原型

```c
HcclResult HcclChannelConfigSetStr(HcclChannelConfig config, HcclChannelConfigType type, const char *value)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| config | 输入 | Channel配置对象句柄，须通过[HcclChannelConfigCreate](HcclChannelConfigCreate.md)创建。 |
| type | 输入 | 属性类型枚举值。<br>HcclChannelConfigType类型的定义请参见[HcclChannelConfigType](../../datatype_definition/HcclChannelConfigType.md)。 |
| value | 输入 | 属性值字符串，以'\0'结尾。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- config和value均不可为nullptr，否则返回参数错误。
- type必须为有效的字符串型属性枚举值，否则返回参数错误。
- 当前支持的字符串型属性：
  - HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG：共享队列的tag标识。仅在IS_SHARED_QUEUE=true时有效。tag不可为空，且最大字节长度不得超过255（含'\0'），否则返回参数错误。

## 调用示例

```c
HcclChannelConfig config = nullptr;
HcclChannelConfigCreate(&config);
HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);

// 设置共享队列tag
HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "group_a_tag");

HcclChannelAcquireWithConfig(comm, engine, channelDescs, channelNum, config, channels);
HcclChannelConfigDestroy(config);
```
