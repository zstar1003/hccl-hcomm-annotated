# HcclGetCommName

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

获取指定通信操作所在的通信域的名称。

## 函数原型

```c
HcclResult HcclGetCommName(HcclComm comm, char* commName)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 集合通信操作所在的通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| commName | 输出 | 获取到的通信域名称。<br>char*类型，最大长度支持128。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
// 初始化通信域
HcclComm comm;
// 查询通信域名称
char commName[128] = {0};
HcclResult ret = HcclGetCommName(comm, commName);
// 错误处理
if (ret != HCCL_SUCCESS) {
}
```
