# HcclGroupStatusGet

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

获取Group特性状态，判断是否启用此特性。

## 函数原型

```c
HcclResult HcclGroupStatusGet(bool *isGroupEnabled)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| isGroupEnabled | 输出 | Group状态，TRUE表示已启用，FALSE表示未启用。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
bool isGroupEnabled = false;
HCCLCHECK(HcclGroupStatusGet(&isGroupEnabled));
```
