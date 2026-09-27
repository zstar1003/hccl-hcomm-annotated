# HcclCommResume

> [!NOTE]说明
> 本接口为预留接口，后续有可能变更，不支持开发者使用。

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
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

本接口用于恢复通信域的状态。

若开发者调用[HcclCommSuspend](HcclCommSuspend.md)接口或者acl提供的aclrtDeviceTaskAbort接口挂起了通信域，故障恢复后，需要调用本接口将通信域恢复为正常状态。

## 函数原型

```c
HcclResult HcclCommResume(HcclComm comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 需要从挂起状态恢复为正常状态的通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 调用本接口前，需要调用acl提供的aclrtDeviceTaskAbort接口停止本Device上的任务执行。
- 调用本接口恢复通信域状态前，需要进行一次集群同步操作。

## 调用示例

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// 生成root节点的rank标识信息
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));
// 初始化通信域
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm));
// 假设通信域已通过HcclCommSuspend接口或者acl提供的aclrtDeviceTaskAbort接口被挂起，恢复通信域
HCCLCHECK(HcclCommResume(hcclComm));
// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));
```
