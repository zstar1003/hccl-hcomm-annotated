# HcommThreadFree

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

使用HcommThreadAlloc接口申请的thread需要通过本接口进行释放。

## 函数原型

```c
HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| threads | 输入 | 通信线程句柄<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |
| threadNum | 输入 | 通信线程数量。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

只能释放HcommThreadAlloc接口申请的thread。

## 调用示例

```c
ThreadHandle thread[2];
const uint32_t notifyNumPerThread[2] = {3, 3};
HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNumPerThread, thread);
ret = HcommThreadFree(thread, 2);
```
