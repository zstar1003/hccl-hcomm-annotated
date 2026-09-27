# HcclThreadAcquireWithStream

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

基于通信域和Runtime Stream句柄获取通信线程，并为该通信线程分配指定数量的同步资源（Notify）。相关概念可参见[通信算子开发指南-并发模型](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md)章节。

## 函数原型

```c
HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| stream | 输入 | stream句柄。 |
| notifyNum | 输入 | 同步信号数量。 |
| thread | 输出 | 线程句柄。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

该接口只支持COMM_ENGINE_CPU、COMM_ENGINE_CPU_TS、COMM_ENGINE_CCU通信引擎。

## 调用示例

```c
// 通信域句柄
HcclComm comm;
// 创建runtime stream
aclrtStream stream;
aclrtCreateStream(&stream);
// 创建thread，并申请2个thread上的notify资源
ThreadHandle thread;
HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 2, &thread);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}

// 数据面操作
// ...

// 流同步
aclrtSynchronizeStream(stream);
```
