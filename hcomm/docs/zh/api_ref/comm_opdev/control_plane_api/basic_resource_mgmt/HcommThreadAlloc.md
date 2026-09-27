# HcommThreadAlloc

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

申请通信线程，当前支持AI CPU、AI CPU+TS、HOST CPU、HOST CPU+TS通信引擎。注意如果通信引擎是AI CPU+TS，需要额外下发一次kernel，AI CPU侧才能使用此通信线程。

## 函数原型

```c
HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t *notifyNumPerThread, ThreadHandle* threads)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| threadNum | 输入 | 通信线程数量，每次调用该接口申请的最大线程数量为200。 |
| notifyNumPerThread | 输入 | 每个通信线程中的同步资源（Notify）数量，每个通信线程调用该接口每次申请的notify数量最大为64。 |
| threads | 输出 | 返回的通信线程句柄。需传入threadNum大小的ThreadHandle类型数组。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. 调用此接口申请的thread，后续需要调用[HcommThreadFree](HcommThreadFree.md)接口释放，调用HcommThreadAlloc接口申请AICPU_TS或CPU_TS等通信引擎的线程前，必须先在同一线程上调用aclrtSetdevice接口指定deviceId。

2. 该接口不支持COMM_ENGINE_AIV和COMM_ENGINE_CCU两种通信引擎。

## 调用示例

```c
ThreadHandle thread[2];
//申请两条流，每条流Notify数量为3
const uint32_t notifyNumPerThread[2] = {3, 3};
HcommResult ret =  HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNumPerThread, thread);
```
