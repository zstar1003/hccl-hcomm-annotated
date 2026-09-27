# HcclChannelConfigCreate

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

创建Channel配置对象，用于[HcclChannelAcquireWithConfig](HcclChannelAcquireWithConfig.md)接口创建通信通道时传入共享Jetty等高级配置。

配置对象创建后，可通过[HcclChannelConfigSetInt](HcclChannelConfigSetInt.md)和[HcclChannelConfigSetStr](HcclChannelConfigSetStr.md)设置属性，使用完毕后须通过[HcclChannelConfigDestroy](HcclChannelConfigDestroy.md)销毁。

## 函数原型

```c
HcclResult HcclChannelConfigCreate(HcclChannelConfig *config)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| config | 输出 | Channel配置对象的不透明句柄。<br>HcclChannelConfig类型的定义请参见[HcclChannelConfig](../../datatype_definition/HcclChannelConfig.md)。<br>调用者仅需传入指针，由接口内部分配并返回句柄。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 调用者须确保传入的config指针有效。
- 创建的配置对象使用完毕后必须调用[HcclChannelConfigDestroy](HcclChannelConfigDestroy.md)销毁，否则会导致内存泄漏。
- 配置对象在[HcclChannelAcquireWithConfig](HcclChannelAcquireWithConfig.md)调用完成后即可销毁，不影响已创建的Channel。

## 调用示例

```c
HcclChannelConfig config = nullptr;
HcclResult ret = HcclChannelConfigCreate(&config);
if (ret != HCCL_SUCCESS) {
    printf("Failed to create channel config, ret = %d\n", ret);
    return ret;
}

// 设置共享队列属性
HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "my_shared_tag");

// 使用配置创建Channel
HcclChannelAcquireWithConfig(comm, engine, channelDescs, channelNum, config, channels);

// 销毁配置对象
HcclChannelConfigDestroy(config);
```
