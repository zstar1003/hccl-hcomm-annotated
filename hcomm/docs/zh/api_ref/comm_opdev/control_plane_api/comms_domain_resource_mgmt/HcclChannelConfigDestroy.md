# HcclChannelConfigDestroy

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

销毁通过[HcclChannelConfigCreate](HcclChannelConfigCreate.md)创建的Channel配置对象，释放其占用的内存资源。

## 函数原型

```c
HcclResult HcclChannelConfigDestroy(HcclChannelConfig config)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| config | 输入 | 待销毁的Channel配置对象句柄。<br>HcclChannelConfig类型的定义请参见[HcclChannelConfig](../../datatype_definition/HcclChannelConfig.md)。|

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 销毁后不可再使用该句柄，否则会导致未定义行为。
- 配置对象在[HcclChannelAcquireWithConfig](HcclChannelAcquireWithConfig.md)调用完成后即可销毁，不影响已创建的Channel。

## 调用示例

```c
HcclChannelConfig config = nullptr;
HcclChannelConfigCreate(&config);
// ... 使用config创建Channel ...
HcclChannelConfigDestroy(config);
config = nullptr;
```
