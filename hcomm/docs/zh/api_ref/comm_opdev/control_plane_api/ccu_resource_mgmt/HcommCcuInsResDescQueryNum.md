# HcommCcuInsResDescQueryNum

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

按资源类型查询CCU资源描述符中已设置的资源数量。

## 函数原型

```c
CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t *resNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| resDesc | 输入 | 资源描述符句柄，由`HcommCcuInsResDescCreate`创建。 |
| resType | 输入 | 资源类型，取[HcommCcuResType](../../datatype_definition/HcommCcuResType.md)枚举值。 |
| resNum | 输出 | 查询成功后返回该资源类型已设置的数量，不能为空指针。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 查询成功，`*resNum`为已设置的资源数量。 |
| `CCU_E_PARA` | `resType`超出合法范围。 |
| `CCU_E_PTR` | `resNum`为空指针。 |
| `CCU_E_NOT_FOUND` | `resDesc`未注册。 |

## 约束说明

- 调用本接口前须通过 `HcommCcuInsResDescCreate` 创建资源描述符。
- 若从未调用 `HcommCcuInsResDescSetNum` 设置该资源类型，则查询结果为 0。
- 本接口只能在主机侧调用。

## 依赖关系

- 依赖 `HcommCcuInsResDescCreate` 创建的资源描述符句柄。
- 依赖 `HcommCcuInsResDescSetNum` 设置资源数量。

## 调用示例

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, 8);

uint32_t loopNum = 0;
ret = HcommCcuInsResDescQueryNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, &loopNum);
// loopNum == 8

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
