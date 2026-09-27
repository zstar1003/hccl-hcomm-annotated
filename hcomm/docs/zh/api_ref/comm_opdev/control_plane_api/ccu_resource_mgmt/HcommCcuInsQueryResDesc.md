# HcommCcuInsQueryResDesc

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

查询CCU实例在指定IO Die上实际占用的各类资源数量，并将结果写入调用方提供的资源描述符。

待查询的IO Die由创建`resDesc`时传入的`dieId`指定。接口返回成功后，可通过[HcommCcuInsResDescQueryNum](HcommCcuInsResDescQueryNum.md)逐类型读取查询结果。

## 函数原型

```c
CcuResult HcommCcuInsQueryResDesc(CcuInsHandle ccuInsHandle, HcommCcuResDescHandle resDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccuInsHandle | 输入 | 待查询的CCU实例句柄，不能为`0`。 |
| resDesc | 输入/输出 | 资源描述符句柄，不能为`0`。须通过[HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md)在当前Device上创建；创建时指定的`dieId`用于选择待查询的IO Die，查询结果覆盖描述符中原有的资源数量。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 查询成功，实例在指定IO Die上的资源占用量已写入`resDesc`。 |
| `CCU_E_PARA` | `ccuInsHandle`或`resDesc`为`0`，或`resDesc`中的IO Die编号非法。 |
| `CCU_E_NOT_FOUND` | 当前Device上不存在对应的CCU实例或资源描述符。 |
| `CCU_E_INTERNAL` | 查询或写入资源数量时发生内部错误。 |

## 约束说明

- 调用本接口前须通过[HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md)创建`resDesc`，并用其`dieId`指定待查询的IO Die。
- 调用本接口的线程须绑定到创建CCU实例和`resDesc`时使用的同一NPU Device。
- 查询结果覆盖`resDesc`中原有的各类资源数量，但不修改其IO Die编号。
- 对按对齐粒度分配的资源，本接口返回实例实际占用的数量，可能大于创建实例时请求的数量。
- 调用方须保证查询期间不并发销毁或修改`ccuInsHandle`和`resDesc`。
- 本接口只能在主机侧调用。

## 调用示例

```c
HcommCcuResDescHandle queryDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &queryDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

ret = HcommCcuInsQueryResDesc(insHandle, queryDesc);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(queryDesc);
    return ret;
}

uint32_t loopNum = 0;
ret = HcommCcuInsResDescQueryNum(
    queryDesc, HCOMM_CCU_RES_TYPE_LOOP, &loopNum);

HcommCcuInsResDescDestroy(queryDesc);
return ret;
```
