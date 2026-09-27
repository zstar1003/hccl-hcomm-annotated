# HcclGetCommConfigCapability

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

该接口用于判断当前版本软件是否支持某项通信域初始化配置。

通信域初始化时支持的完整配置项可参见[HcclCommConfigCapability](./data_type_definition/HcclCommConfigCapability.md)，包括共享数据的缓存区大小、确定性计算开关、通信域名称、通信算子展开模式等。

使用HcclGetCommConfigCapability接口判断当前软件是否支持某项配置的流程为：

1. 调用HcclGetCommConfigCapability接口，获取一个代表当前软件通信域初始化配置能力的数值。
2. 比较该数值与[HcclCommConfigCapability](./data_type_definition/HcclCommConfigCapability.md)中某项配置枚举值的大小，若该数值大于枚举值，代表当前软件支持[HcclCommConfigCapability](./data_type_definition/HcclCommConfigCapability.md)中对应枚举值的配置能力；若该数值小于等于枚举值，代表不支持。

   例如，若想判断当前软件是否支持配置通信域名称，可使用HcclGetCommConfigCapability接口的返回值与枚举值“HCCL_COMM_CONFIG_COMM_NAME”做比较，若返回值大于“HCCL_COMM_CONFIG_COMM_NAME”，代表当前软件支持配置通信域名称；若返回值小于等于“HCCL_COMM_CONFIG_COMM_NAME”，代表当前软件不支持配置通信域名称。

## 函数原型

```c
uint32_t HcclGetCommConfigCapability()
```

## 参数说明

无

## 返回值

uint32_t：表示通信域初始化配置能力的数值。

该数值的具体含义可参见[功能说明](#功能说明)。

## 约束说明

无

## 调用示例

```c
uint32_t configCapability = HcclGetCommConfigCapability();
bool isSupportCommName = configCapability > HCCL_COMM_CONFIG_COMM_NAME;  // 判断是否支持配置通信域名称
```
