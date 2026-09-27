# HcommCcuInsResDescQueryDieId

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

查询 CCU 资源描述符归属的 IO Die 编号。

## 函数原型

```c
CcuResult HcommCcuInsResDescQueryDieId(HcommCcuResDescHandle resDesc, uint32_t *dieId)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| resDesc | 输入 | 资源描述符句柄，由`HcommCcuInsResDescCreate`创建。 |
| dieId | 输出 | 查询成功后返回创建描述符时指定的IO Die编号，不能为空指针。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 查询成功，`*dieId`为描述符归属的 IO Die 编号。 |
| `CCU_E_PTR` | `dieId`为空指针。 |
| `CCU_E_NOT_FOUND` | `resDesc`未注册。 |

## 约束说明

- 调用本接口前须通过 `HcommCcuInsResDescCreate`创建资源描述符。
- 本接口只能在主机侧调用。

## 依赖关系

- 依赖 `HcommCcuInsResDescCreate`创建的资源描述符句柄。
- 供 `HcommCcuQueryRemainResDesc`内部获取 die ID以查询对应Die的剩余资源。

## 调用示例

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(1, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

uint32_t dieId = 0;
ret = HcommCcuInsResDescQueryDieId(resDesc, &dieId);
// dieId == 1

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
