# HcommEndpointGetDescNum

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

查询指定NPU设备上的网络语义`Endpoint`数量。调用方可根据返回的数量申请`Endpoint`描述数组，再调用[`HcommEndpointGetDescs`](HcommEndpointGetDescs.md)获取`Endpoint`描述。

## 函数原型

```c
HcommResult HcommEndpointGetDescNum(int32_t deviceLogicId, uint32_t *descNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `deviceLogicId` | 输入 | NPU设备逻辑ID。 |
| `descNum` | 输出 | 接口调用成功时，返回网络语义`Endpoint`的数量；当该指针非空但接口调用失败时，值为`0`。该参数不能为空指针。 |

## 返回值

`HcommResult`：接口返回`0`表示成功，其他值表示失败。

## 约束说明

- 当前仅支持Ascend 950PR/Ascend 950DT。
- 查询范围仅包含网络语义`Endpoint`，支持的通信协议包括`COMM_PROTOCOL_UBC_CTP`、`COMM_PROTOCOL_UBG`和`COMM_PROTOCOL_UBOE`，不返回`COMM_PROTOCOL_UB_MEM`、`COMM_PROTOCOL_HCCS`等内存语义`Endpoint`。
- `deviceLogicId`必须是有效且在位的NPU设备逻辑ID。
- NPU设备未配置EID时，接口返回未找到错误，此时`descNum`为`0`。

## 调用示例

```c
#include <stdio.h>

#include "hcomm_res.h"

int main(void)
{
    /* 输入：待查询的NPU设备逻辑ID。 */
    const int32_t deviceLogicId = 0;

    /* 输出：网络语义Endpoint的数量。 */
    uint32_t descNum = 0;
    HcommResult result = HcommEndpointGetDescNum(deviceLogicId, &descNum);
    if (result != 0) {
        (void)fprintf(stderr, "HcommEndpointGetDescNum failed, result = %d\n", (int)result);
        return 1;
    }

    (void)printf("Endpoint description count: %u\n", descNum);
    return 0;
}
```
