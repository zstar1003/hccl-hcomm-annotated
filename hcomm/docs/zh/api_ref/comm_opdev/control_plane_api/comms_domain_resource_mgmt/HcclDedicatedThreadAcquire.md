# HcclDedicatedThreadAcquire

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

基于通信域申请专用通信线程，并为该线程分配指定数量的同步资源（Notify）。根据useType不同，线程管理方式分为两类：

- **非保序类型**（HCCL_DED_THREAD_TYPE_AICPU_LAUNCH、HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE）：专用线程按useType在通信域内缓存，相同useType重复调用将直接返回已缓存的线程句柄，不会重复创建。

- **保序类型**（HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_*系列）：专用线程由进程粒度的OrderLaunchThreadMgr管理，不与单个通信域绑定，适用于保序任务下发场景。

该接口主要用于AI CPU任务下发等专用场景。相关概念可参见[通信算子开发指南-并发模型](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md)章节。

> [!NOTE]说明
> 与[HcclThreadAcquire](./HcclThreadAcquire.md)相比，该接口按useType缓存专用线程，相同useType重复调用复用同一线程；且仅支持专用下发场景，不支持通过CommEngine指定通信引擎。

## 函数原型

```c
HcclResult HcclDedicatedThreadAcquire(HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle *thread)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| useType | 输入 | 专用线程使用类型。HcclDedicatedThreadType类型的定义可参见[HcclDedicatedThreadType](../../datatype_definition/HcclDedicatedThreadType.md)。 |
| notifyNumPerThread | 输入 | 通信线程中的同步资源（Notify）数量。取值范围为：\[0, 64\]。建议根据业务场景合理配置，避免资源不足或浪费。 |
| thread | 输出 | 返回的专用通信线程句柄。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 返回的通信线程与同步资源由库内管理，调用者严禁释放。

2. 非保序类型（HCCL_DED_THREAD_TYPE_AICPU_LAUNCH、HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE）在同一个通信域内仅创建一次专用线程，重复调用返回已缓存的线程句柄；若本次申请的notifyNumPerThread大于已缓存线程的Notify数量，库内会按差额补充同步资源。

3. 保序类型（HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_*系列）由进程粒度的OrderLaunchThreadMgr管理，线程不与单个通信域绑定，各类型行为如下：

   - HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE：按aclrtContext缓存CPU_TS线程，相同context下重复调用复用同一线程。
   - HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH：按aclrtContext缓存CPU_TS线程，与OPBASE独立缓存。
   - HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE：从已设置的附属流获取线程，调用前需先通过HcomSetAttachedStream接口设置附属流；若未设置，返回的thread句柄为0。
   - HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE：按通信域标识缓存AICPU_TS线程。

4. 保序类型在以下场景返回的thread句柄为0（接口仍返回HCCL_SUCCESS），表示无需创建专用保序线程：

   - 当前context下已注册的通信域数量不超过设备AICPU block数量时，OPBASE/ACLGRAPH/DEVICE/GE类型均不创建专用线程。
   - GE类型未通过HcomSetAttachedStream设置附属流时。

5. 图模式下（useType为HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE）若通信域中尚未缓存该专用线程，接口不会实际创建线程，返回的thread句柄为0。

6. 调用HcclDedicatedThreadAcquire接口申请专用线程前，必须先在同一线程上调用aclrtSetDevice接口指定deviceId。

7. HCCL_DED_THREAD_TYPE_INVALID为无效值，传入将返回HCCL_E_PARA。

## 调用示例

申请AICPU任务下发的专用线程示例如下：

```c
// 通信域句柄
HcclComm comm;
// 申请AICPU_LAUNCH类型的专用线程，包含2个notify
ThreadHandle dedThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &dedThread);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
```

图模式下申请专用线程示例如下：

```c
// 通信域句柄
HcclComm comm;
// 图模式下申请AICPU_LAUNCH_GE类型的专用线程
// 若通信域中尚未缓存该线程，返回的thread为0，不会实际创建
ThreadHandle dedThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 2, &dedThread);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
```

申请单算子保序任务下发的专用线程示例如下：

```c
// 通信域句柄
HcclComm comm;
// 申请OPBASE保序类型的专用线程，包含1个notify
ThreadHandle orderThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, 1, &orderThread);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
// 若orderThread为0，表示当前无需专用保序线程，可直接使用默认下发路径
```

图模式下申请保序任务下发的专用线程示例如下：

```c
// 通信域句柄
HcclComm comm;
// 图模式下需先通过HcomSetAttachedStream设置附属流
aclrtStream stream;
aclrtCreateStream(&stream);
HcomSetAttachedStream(nullptr, graphId, &stream, 1);
// 申请GE保序类型的专用线程
ThreadHandle geOrderThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, 1, &geOrderThread);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
// 若geOrderThread为0，表示未设置附属流或当前无需专用保序线程
```

申请设备侧保序任务下发的专用线程示例如下：

```c
// 通信域句柄
HcclComm comm;
// 申请DEVICE保序类型的专用线程，包含1个notify
ThreadHandle deviceOrderThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &deviceOrderThread);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
```

