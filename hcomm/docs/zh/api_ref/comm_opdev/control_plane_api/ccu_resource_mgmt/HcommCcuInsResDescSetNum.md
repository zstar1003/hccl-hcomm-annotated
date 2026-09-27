# HcommCcuInsResDescSetNum

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

按资源类型设置CCU资源描述符中的资源数量。可多次调用逐类型设置期望值。设置为“0”表示不申请该类型资源。

## 函数原型

```c
CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t resNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| resDesc | 输入/输出 | 资源描述符句柄，由 `HcommCcuInsResDescCreate` 创建。设置成功后，指定资源类型的数量更新为`resNum`。 |
| resType | 输入 | 资源类型，取[HcommCcuResType](../../datatype_definition/HcommCcuResType.md)枚举值。 |
| resNum | 输入 | 期望的资源数量。设置为“0”表示不申请该类型资源。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回 `CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 设置成功。 |
| `CCU_E_PARA` | `resType`超出合法范围。 |
| `CCU_E_NOT_FOUND` | `resDesc`未注册。 |

## 约束说明

- 调用本接口前须通过`HcommCcuInsResDescCreate`创建资源描述符。
- 多次调用本接口对同一资源类型进行设置，后一次的值覆盖前一次的值。
- 本接口为纯数据写入操作，不校验`resNum`是否超过硬件容量上限。容量校验在使用描述符创建 CCU 实例或查询剩余资源时进行。
- 本接口只能在主机侧调用。

## 依赖关系

- 依赖`HcommCcuInsResDescCreate`创建的资源描述符句柄。
- 依赖`HcommCcuInsResDescQueryNum`查询已设置的资源数量。
- 配合`HcommCcuQueryRemainResDesc`使用：先查询剩余资源，再按需调整描述符中的资源数量。

## 调用示例

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

ret = HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, 8);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(resDesc);
    return ret;
}
ret = HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_CCU_BUF, 16);

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
