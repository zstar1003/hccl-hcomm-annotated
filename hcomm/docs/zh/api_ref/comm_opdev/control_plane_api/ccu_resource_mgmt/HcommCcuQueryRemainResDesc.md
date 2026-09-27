# HcommCcuQueryRemainResDesc

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

查询CCU资源描述符对应IO Die上的最大连续剩余资源数量，并将各类资源的查询结果写入该描述符。

调用后可通过`HcommCcuInsResDescQueryNum`查询各类资源的剩余数量，也可在查询结果的基础上调整申请数量并创建CCU实例。

## 函数原型

```c
CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle resDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| resDesc | 输入/输出 | 资源描述符句柄，由`HcommCcuInsResDescCreate`创建。创建时指定的`dieId`用于选择待查询的IO Die，查询结果覆盖描述符中原有的资源数量。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回 `CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 查询成功，各类资源的最大连续剩余数量已写入`resDesc`。 |
| `CCU_E_PARA` | `resDesc`对应的描述符中IO Die编号超出合法范围。 |
| `CCU_E_NOT_FOUND` | `resDesc`未注册。 |
| `CCU_E_UNAVAIL` | 指定的IO Die未使能。 |
| `CCU_E_INTERNAL` | 查询过程中发生内部错误（如资源规格查询失败）。 |

## 约束说明

- 调用本接口前须通过`HcommCcuInsResDescCreate`创建资源描述符。
- 创建`resDesc`与调用本接口时，当前线程须绑定到同一NPU Device。
- 调用方须保证查询期间不并发修改或销毁`resDesc`。
- 查询结果覆盖描述符中已有的资源数量。如需保留原始设置值，调用方应在调用前自行备份。
- 本接口只能在主机侧调用，不能在Kernel函数体内调用。

## 依赖关系

- 依赖`HcommCcuInsResDescCreate`创建的资源描述符句柄。
- 配合`HcommCcuInsResDescQueryNum`使用：查询返回后通过QueryNum读取各资源类型的剩余数量。

## 调用示例

```c
uint32_t dieId = 0;
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(dieId, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// 查询 dieId 对应 IO Die 上的剩余资源
ret = HcommCcuQueryRemainResDesc(resDesc);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(resDesc);
    return ret;
}

// 读取各资源类型的剩余数量
uint32_t loopRemain = 0;
HcommCcuInsResDescQueryNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, &loopRemain);

// 如需设置特定的资源规格，可在剩余量基础上调整
HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, loopRemain / 2);

// 基于调整后的描述符创建 CCU 实例（略）

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
