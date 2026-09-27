# HcommEndpointCheckFeature

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

该接口用于查询指定网络端点是否支持某种特性，无需实际创建端点。

通过传入端点描述信息和特性类型，查询底层硬件/网络驱动是否支持该特性，结果通过value参数返回。该接口主要用于在创建端点之前进行能力探测，以便上层根据硬件能力选择不同的通信策略。

当前仅支持查询NDA（NPU Direct RDMA Async）特性。

## 函数原型

```c
HcommResult HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc *endpointDesc, bool *value);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| featureType | 输入 | 待查询的特性类型。<br>HcommEndpointFeatureType类型的定义请参见[HcommEndpointFeatureType](../../datatype_definition/HcommEndpointFeatureType.md)。 |
| endpointDesc | 输入 | 端点描述信息，用于指定待查询的端点属性（协议、地址、位置等），无需实际创建端点即可查询。<br>EndpointDesc类型的定义请参见[EndpointDesc](../../datatype_definition/EndpointDesc.md)。<br>该参数不能为空指针。 |
| value | 输出 | 特性支持情况的结果。<br>true表示支持该特性，false表示不支持。<br>该参数不能为空指针。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- 调用前无需创建Endpoint，仅需填充EndpointDesc描述信息。
- 查询HCOMM_ENDPOINT_FEATURE_NDA特性时，EndpointDesc中的协议需为COMM_PROTOCOL_ROCE，端点位置类型需为ENDPOINT_LOC_TYPE_HOST，否则返回false。
- 该接口主要用于能力探测阶段，不应在通信过程中频繁调用。

## 调用示例

```c
// 填充端点描述信息
EndpointDesc endpointDesc;
// 参考HcommEndpointCreate进行EndpointDesc的填充
...

// 查询是否支持NDA特性
bool isNdaSupported = false;
HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_NDA, &endpointDesc, &isNdaSupported);
if (ret != 0) {
    printf("Failed to check feature, ret = %d\n", ret);
    return ret;
}

if (isNdaSupported) {
    // 使用NDA特性进行通信
    // ...
} else {
    // 回退到其他通信方式
    // ...
}
```
