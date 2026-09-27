# HcclCommGetStatus

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

算子下发时获取通信域状态，判断能否下发算子。

## 函数原型

```c
HcclResult HcclCommGetStatus(const char *commId, HcclCommStatus *status)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| commId | 输入 | 通信域名称。<br>const char*类型，最大长度支持128。 |
| status | 输出 | 通信域状态，HcclCommStatus类型的定义可参见[HcclCommStatus](./data_type_definition/HcclCommStatus.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

自定义通信算子场景使用。

## 调用示例

```c
HcclComm comm;
char commName[128];
HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;

... //创建通信域

HCCLCHECK(HcclGetCommName(comm, commName));
HCCLCHECK(HcclCommGetStatus(commName, &commStatus));
```
