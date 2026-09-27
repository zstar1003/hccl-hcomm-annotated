# HcommThreadResGetInfo

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

该接口用于获取Thread底层资源，例如stream等。

## 函数原型

```c
HcommResult HcommThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 线程句柄。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。可通过[HcommThreadAlloc](./HcommThreadAlloc.md)接口创建通信线程。 |
| resType | 输入 | 底层资源类型（如stream等）。<br>ThreadResType类型的定义可参见[ThreadResType](../../datatype_definition/ThreadResType.md)。 |
| infoLen | 输入 | 目标资源信息大小，须等于对应资源类型的大小。 |
| info | 输出 | 资源信息输出缓冲区。返回类型为获取的对应资源类型。 |

## 返回值

[HcommResult](../../datatype_definition/HcommResult.md)：接口成功返回0，其他失败。

## 约束说明

1. 该接口仅支持获取通信线程的底层stream资源（类型：[ThreadResTypeStream](../../datatype_definition/ThreadResTypeStream.md)）。
2. 参数infoLen必须等于sizeof(ThreadResTypeStream)，否则返回参数校验失败。
3. 参数info不能为空，参数thread不能为0，否则返回空指针错误。

## 调用示例

```c
ThreadHandle thread;          // HcommThreadAlloc创建出来的thread的句柄
const uint32_t notifyNumPerThread = 3;
HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 1, &notifyNumPerThread, &thread);
if (ret != 0) {
    // 错误处理
}

ThreadResTypeStream stream;   // info缓冲区必须按资源类型对齐且可写
uint32_t size = sizeof(ThreadResTypeStream);  // 必须等于目标类型大小
ret = HcommThreadResGetInfo(thread, THREAD_RES_TYPE_STREAM, size, (void**)&stream);
if (ret != 0) {
    // 错误处理
}
// 使用stream资源
// ...

HcommThreadFree(&thread, 1);
```
