# HcclCommGetHandleWithName

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
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

根据通信域名称获取指向对应通信域的句柄。

## 函数原型

```c
HcclResult HcclCommGetHandleWithName(const char* commName, HcclComm* comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| commName | 输入 | 通信域名称。<br>const char*类型，最大长度支持128。 |
| comm | 输出 | 获取到的对应通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
// 定义需要获取通信域句柄的通信域名称
char commName[128] = "group_name_0";
HcclComm comm;
// 获取通信域名称对应的通信域句柄
HcclCommGetHandleWithName(commName, &comm);
```
