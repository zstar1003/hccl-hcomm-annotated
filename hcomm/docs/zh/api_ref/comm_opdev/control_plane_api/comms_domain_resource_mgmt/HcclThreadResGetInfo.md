# HcclThreadResGetInfo

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
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

该接口用于获取Thread底层资源，例如stream等。

## 函数原型

```c
HcclResult HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| thread | 输入 | 线程句柄。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。可通过[HcclThreadAcquire](./HcclThreadAcquire.md)、[HcclThreadAcquireWithConfig](./HcclThreadAcquireWithConfig.md)、[HcclThreadAcquireWithStream](./HcclThreadAcquireWithStream.md)接等接口创建通信线程。 |
| resType | 输入 | 底层资源类型（如stream等）。<br>ThreadResType类型的定义可参见[ThreadResType](../../datatype_definition/ThreadResType.md)。 |
| infoLen | 输入 | 目标资源信息大小。 |
| info | 输出 | 资源信息输出缓冲区。返回类型为获取的对应资源类型。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

该接口仅支持获取通信线程的底层stream资源（类型：[ThreadResTypeStream](../../datatype_definition/ThreadResTypeStream.md)）。

## 调用示例

```c
// 通信域句柄
HcclComm comm;
ThreadHandle thread;          // HcclThreadAcquire创建出来的thread的句柄
ThreadResTypeStream stream;   // info缓冲区必须按资源类型对齐且可写
uint32_t size = sizeof(ThreadResTypeStream);  // 必须等于目标类型大小
HcclResult ret = HcclThreadResGetInfo(comm, thread, THREAD_RES_TYPE_STREAM, size, &stream);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
// 使用stream资源
// ...
```
