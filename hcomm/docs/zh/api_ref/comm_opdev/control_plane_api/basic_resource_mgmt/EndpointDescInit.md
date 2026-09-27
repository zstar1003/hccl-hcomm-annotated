# EndpointDescInit

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

初始化Endpoint描述列表。该接口会将[EndpointDesc](../../datatype_definition/EndpointDesc.md)结构体中的字段设置为保留值，确保结构体处于明确的无效初始状态。

## 函数原型

```c
HcommResult EndpointDescInit(EndpointDesc *endpoint, uint32_t num)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| endpoint | 输出 | Endpoint描述列表，列表长度为num，函数会初始化该结构体。<br>EndpointDesc类型的定义可参见[EndpointDesc](../../datatype_definition/EndpointDesc.md)。 |
| num | 输入 | Endpoint描述数量。 |

## 返回值

[HcommResult](../../datatype_definition/HcommResult.md)：接口成功返回0，其他失败。

## 约束说明

无。

## 调用示例

以初始化数量为2的Endpoint描述列表为例：

```c
uint32_t descNum = 2;
std::vector<EndpointDesc> endpointDesc(descNum);
HcommResult ret = EndpointDescInit(endpointDesc.data(), descNum);
if (ret != 0) {
    // 错误处理
}
```
