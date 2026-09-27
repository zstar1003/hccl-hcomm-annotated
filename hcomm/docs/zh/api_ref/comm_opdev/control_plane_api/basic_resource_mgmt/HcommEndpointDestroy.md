# HcommEndpointDestroy

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

销毁通信设备Endpoint。

## 函数原型

```c
HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| endpointHandle | 输入 | Endpoint句柄。<br>EndpointHandle类型的定义请参见[EndpointHandle](../../datatype_definition/EndpointHandle.md)。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

无

## 调用示例

```c
struct in_addr ipAddr;
inet_pton(AF_INET, "192.168.1.100", &ipAddr);
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_UB_RTP,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = ipAddr
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_DEVICE,
        .device = {
            .devPhyId = 0,
            .superDevId = 0,
            .serverIdx = 0,
            .superPodIdx = 0
        }
    },
    .raws = {0}
};
EndpointHandle endpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&endpointDesc, &endpointHandle);
result = HcommEndpointDestroy(endpointHandle);
```
