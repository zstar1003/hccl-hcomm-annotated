# HcommEndpointGetDescs

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

查询指定NPU设备上的网络语义`Endpoint`描述，包括`Endpoint`的通信协议、通信地址和所在位置。

## 函数原型

```c
HcommResult HcommEndpointGetDescs(
    int32_t deviceLogicId, uint32_t *descNum, EndpointDesc *endpointDescs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `deviceLogicId` | 输入 | NPU设备逻辑ID。 |
| `descNum` | 输入/输出 | 输入时，表示`endpointDescs`数组实际可容纳的`EndpointDesc`元素数量；接口调用成功时，输出实际返回的`Endpoint`数量。接口调用失败时，该值保持为输入的数组容量。该参数不能为空指针。 |
| `endpointDescs` | 输出 | 返回`Endpoint`描述数组，内存由调用方申请和释放。接口调用成功时，前`descNum`个元素有效；接口调用失败时，不应使用数组内容。该参数不能为空指针。<br>`EndpointDesc`类型的定义请参见[`EndpointDesc`](../../datatype_definition/EndpointDesc.md)。 |

> [!CAUTION]注意
>
> - `descNum`输入值表示`endpointDescs`数组的实际可用容量。
> - 推荐按[`HcommEndpointGetDescNum`](HcommEndpointGetDescNum.md)返回的数量申请数组，并将该数量原值传入本接口。
> - 若两次调用之间设备配置发生变化，实际数量仍可能超过已申请的容量，此时应重新查询数量、申请数组并重试。

成功时，接口返回的关键字段如下。

| `protocol` | `commAddr.type` | 有效的地址字段 |
| --- | --- | --- |
| `COMM_PROTOCOL_UBC_CTP` | `COMM_ADDR_TYPE_EID` | `commAddr.eid` |
| `COMM_PROTOCOL_UBG` | `COMM_ADDR_TYPE_EID` | `commAddr.eid` |
| `COMM_PROTOCOL_UBOE` | `COMM_ADDR_TYPE_IP_V4` | `commAddr.addr` |

所有返回项的`loc.locType`均为`ENDPOINT_LOC_TYPE_DEVICE`，`loc.device.devPhyId`表示`Endpoint`所在NPU设备的物理ID。

## 返回值

`HcommResult`：接口返回`0`表示成功，其他值表示失败。

## 约束说明

- 当前仅支持Ascend 950PR/Ascend 950DT。
- 查询范围仅包含网络语义`Endpoint`，返回的通信协议包括`COMM_PROTOCOL_UBC_CTP`、`COMM_PROTOCOL_UBG`和`COMM_PROTOCOL_UBOE`，不返回`COMM_PROTOCOL_UB_MEM`、`COMM_PROTOCOL_HCCS`等内存语义`Endpoint`。
- `deviceLogicId`必须是有效且在位的NPU设备逻辑ID。
- `endpointDescs`必须由调用方提前申请，建议先调用[`HcommEndpointGetDescNum`](HcommEndpointGetDescNum.md)查询数组容量。
- `descNum`输入值应填写`endpointDescs`数组的实际可用容量。
- `endpointDescs`数组容量小于实际`Endpoint`数量时，接口返回参数错误，且`descNum`保持输入值不变；调用方需重新调用[`HcommEndpointGetDescNum`](HcommEndpointGetDescNum.md)、重新申请数组并重试。
- NPU设备未配置EID时，接口返回未找到错误。

## 调用示例

```c
#include <stdio.h>
#include <stdlib.h>

#include "hcomm_res.h"

int main(void)
{
    /* 第一步：查询Endpoint数量。 */
    const int32_t deviceLogicId = 0;
    uint32_t descNum = 0;
    HcommResult result = HcommEndpointGetDescNum(deviceLogicId, &descNum);
    if (result != 0) {
        (void)fprintf(stderr, "HcommEndpointGetDescNum failed, result = %d\n", (int)result);
        return 1;
    }
    if (descNum == 0) {
        (void)printf("No endpoint description found.\n");
        return 0;
    }

    /* 第二步：按查询到的数量申请描述数组。 */
    EndpointDesc *endpointDescs = (EndpointDesc *)calloc(descNum, sizeof(*endpointDescs));
    if (endpointDescs == NULL) {
        (void)fprintf(stderr, "Allocate endpoint description array failed.\n");
        return 1;
    }

    /* 第三步：输入数组容量，成功后descNum更新为实际返回数量。 */
    result = HcommEndpointGetDescs(deviceLogicId, &descNum, endpointDescs);
    if (result != 0) {
        (void)fprintf(stderr, "HcommEndpointGetDescs failed, result = %d\n", (int)result);
        free(endpointDescs);
        return 1;
    }

    /* 第四步：使用返回的描述。 */
    for (uint32_t i = 0; i < descNum; ++i) {
        const EndpointDesc *desc = &endpointDescs[i];
        (void)printf("endpoint[%u]: protocol=%d, addrType=%d, devPhyId=%u\n",
            i, (int)desc->protocol, (int)desc->commAddr.type, desc->loc.device.devPhyId);
    }

    free(endpointDescs);
    return 0;
}
```
