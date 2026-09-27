# HcclThreadAcquire

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

基于通信域获取通信线程，并为每条通信线程分配指定数量的同步资源（Notify）。相关概念可参见[通信算子开发指南-并发模型](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md)章节。

## 函数原型

```c
HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle *threads)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| threadNum | 输入 | 通信线程数量。 |
| notifyNumPerThread | 输入 | 每个通信线程中的同步资源（Notify）数量。取值范围为：\[0, 65535\]，具体数量限制由具体产品限定。建议根据业务场景合理配置，避免资源不足或浪费。 |
| threads | 输出 | 返回的通信线程句柄。需传入threadNum大小的ThreadHandle类型数组。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 返回的通信线程与同步资源由库内管理，调用者严禁释放。

2. 当前各产品形态支持的CommEngine范围：

  <!-- npu="950" id6 -->
  - Ascend 950PR/Ascend 950DT：
    - COMM_ENGINE_CPU_TS
    - COMM_ENGINE_AICPU_TS
  <!-- end id6 -->

  <!-- npu="A3" id7 -->
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    - COMM_ENGINE_CPU_TS
    - COMM_ENGINE_AICPU_TS
  <!-- end id7 -->

  <!-- npu="910b" id8 -->
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：
    - COMM_ENGINE_CPU_TS
    - COMM_ENGINE_AICPU_TS
  <!-- end id8 -->

3. 该接口不支持COMM_ENGINE_AIV和COMM_ENGINE_CCU两种通信引擎。

## 调用示例

创建线程资源示例如下：

```c
// 通信域句柄
HcclComm comm;
// 申请5条AICPU_TS类型的thread通信线程，每条thread包含2个notify
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[5];
HcclThreadAcquire(comm, engine, 5, 2, threads);
```

同步Host线程资源示例如下：

```c
// 申请1条host流
aclrtStream stream;
aclrtCreateStream(&stream);
// 根据申请的流创建CPU_TS类型thread
ThreadHandle cpuThread;
HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &cpuThread);
// 任务编排
// ...
// 流同步
aclrtSynchronizeStream(stream);
```

同步AICPU线程资源示例如下：

```c
// --Host侧调用流程--
// 申请1条host流
aclrtStream stream;
aclrtCreateStream(&stream);
// 根据申请的流创建CPU_TS类型thread
ThreadHandle cpuThread;
HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &cpuThread);

// 创建一个AICPU_TS类型的thread
ThreadHandle aicpuThread;
HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, &aicpuThread);

// 把创建的AICPU_TS类型的thread导出为CPU上可用的thread
ThreadHandle exportedCpuThread;
HcclThreadExportToCommEngine(comm, 1, &aicpuThread, COMM_ENGINE_CPU_TS, &exportedCpuThread);
// 把创建的CPU类型的thread导出为AICPU上可用的thread
ThreadHandle exportedAicpuThread;
HcclThreadExportToCommEngine(comm, 1, &cpuThread, COMM_ENGINE_AICPU_TS, &exportedAicpuThread);

// 发送同步信号
HcommThreadNotifyRecordOnThread(cpuThread, exportedCpuThread, 0);
// 下发kernel，把exportedAicpuThread，aicpuThread带到aicpu侧
// ...
uint32_t timeout = 1;
// 等待同步信号
HcommThreadNotifyWaitOnThread(cpuThread, 0, timeout);

// --Device侧调用流程--
// 等待同步信号
uint32_t timeout = 1;
HcommThreadNotifyWaitOnThread(aicpuThread, 0, timeout);
// 任务编排，将任务下发到aicpuThread
// ...
// 发送同步信号
HcommThreadNotifyRecordOnThread(aicpuThread, exportedAicpuThread, 0);

// --Host侧调用流程--
// 流同步
aclrtSynchronizeStream(stream);
```
