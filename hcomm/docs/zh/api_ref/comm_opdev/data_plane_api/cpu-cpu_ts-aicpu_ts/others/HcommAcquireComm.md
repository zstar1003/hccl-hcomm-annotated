# HcommAcquireComm

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

根据传入的commId获取对应通信域，并对该通信域加锁，防止该通信域被重复获取。

## 函数原型

```c
int32_t HcommAcquireComm(const char* commId)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| commId | 输入 | 通信域ID。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. HcommAcquireComm和HcommReleaseComm分别对应加锁和解锁动作，必须成对调用。接口内部会拦截重复加锁场景，避免同一个通信域被多个线程同时占用。

## 调用示例

该函数需编译到Device侧使用：

```c
// 在AI CPU上执行的Kernel函数
extern "C" unsigned int HcclAicpuKernel(const char* commId)
{
    // 对通信域加锁，防止该通信域被并发使用
    if (HcommAcquireComm(commId) != HCCL_SUCCESS) {
        return 1;
    }

    // 执行任务编排
    // ...

    // 释放通信域
    if (HcommReleaseComm(commId) != HCCL_SUCCESS) {
        return 1;
    }
    return 0;
}
```
