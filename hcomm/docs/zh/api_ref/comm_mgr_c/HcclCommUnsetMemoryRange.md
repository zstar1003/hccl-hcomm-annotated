# HcclCommUnsetMemoryRange

> [!NOTE]说明
> 本接口为试用接口，后续可能存在变更，暂不支持应用于生产环境。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
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

该接口用于通知HCCL通信域取消使用预留的虚拟内存。

## 函数原型

```c
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void *baseVirPtr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域，建议使用Server内最大的通信域，即覆盖最大卡数的通信域。 |
| baseVirPtr | 输入 | 预留的虚拟内存基地址。<br>指定的基地址需已成功执行[HcclCommSetMemoryRange](HcclCommSetMemoryRange.md)，否则会报错。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

如果本虚拟地址空间内仍存在激活的内存，此接口会执行失败。

## 调用示例

```c
// 假设已通过HcclCommSetMemoryRange设置了虚拟内存范围
// baseVirPtr为HcclCommSetMemoryRange传入的虚拟内存基地址

// 确保该地址范围内所有已激活的内存均已反激活
// HcclCommDeactivateCommMemory(hcclComm, activatedVirPtr);

// 取消预留的虚拟内存
HcclCommUnsetMemoryRange(hcclComm, baseVirPtr);
```
