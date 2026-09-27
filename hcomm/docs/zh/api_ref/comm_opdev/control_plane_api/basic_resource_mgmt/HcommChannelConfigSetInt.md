# HcommChannelConfigSetInt

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

设置Channel配置对象的整型属性。目前支持的属性类型请参见[HcommChannelConfigType](../../datatype_definition/HcommChannelConfigType.md)。

## 函数原型

```c
HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| config | 输入 | Channel配置对象句柄，须通过[HcommChannelConfigCreate](HcommChannelConfigCreate.md)创建。 |
| type | 输入 | 属性类型枚举值。<br>HcommChannelConfigType类型的定义请参见[HcommChannelConfigType](../../datatype_definition/HcommChannelConfigType.md)。 |
| value | 输入 | 属性值。对于bool类型属性，0表示false，非0表示true。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- config不可为nullptr，否则返回参数错误。
- type必须为有效的整型属性枚举值，否则返回参数错误。
- 当前支持的整型属性：
  - HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE：是否启用共享队列模式。

## 调用示例

```c
HcommChannelConfig config = nullptr;
HcommChannelConfigCreate(&config);

// 启用共享队列模式
HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);

HcommChannelCreateWithConfig(endpointHandle, COMM_ENGINE_AIV,
    channelDescs, channelNum, config, channels);
HcommChannelConfigDestroy(config);
```
