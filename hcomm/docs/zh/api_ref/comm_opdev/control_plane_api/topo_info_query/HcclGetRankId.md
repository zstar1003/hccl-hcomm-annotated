# HcclGetRankId

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

获取Device在指定通信域中对应的rank序号。

## 函数原型

```c
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 集合通信操作所在的通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| rank | 输出 | 集合通信域中rank序号的输出地址指针。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
uint32_t rank;
HcclComm comm;
HcclGetRankId(comm, &rank);
```
