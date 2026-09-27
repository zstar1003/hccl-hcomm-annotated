# HcommThreadNotifyWaitOnThreadWithDefaultTimeout

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

等待其他Thread发送的同步信号，主要用于多Thread之间的同步等待场景。**使用通过[HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md)设置的默认超时时间**，无需手动传入超时参数。

## 函数原型

```c
int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, uint32_t notifyIdx)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 线程句柄，为通过[HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)接口获取到的threads。<br>ThreadHandle类型的定义请参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| notifyIdx | 输入 | 需等待的Notify通知索引。<br>取值范围为：[0, [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)接口传入的notifyNumPerThread参数的值)。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 超时机制说明

1. **默认超时时间**
   - 默认超时时间通过[HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md)接口设置
   - 如果未设置，AICPU_TS模式下，Device侧默认超时时间为1836秒，Host侧默认超时时间为1836+50秒

2. **超时生效条件**
   - 已设置的超时时间将应用于本接口的等待操作
   - 设置为0表示永久等待，不超时
   - 设置大于0表示具体的超时时间（单位：秒）

3. **非AICPU_TS模式特殊处理**
   - 在非AICPU_TS模式下，如果未手动设置默认超时时间，系统会自动在默认值基础上增加50秒的偏移量作为安全缓冲

## 约束说明

- 该接口需要配合[HcommThreadNotifyRecordOnThread](./HcommThreadNotifyRecordOnThread.md)使用。
<!-- npu="950" id6 -->
- 在Ascend 950PR/Ascend 950DT上，支持AICPU_TS模式在Device侧调用，也支持在Host CPU侧调用。
<!-- end id6 -->
- AICPU_TS模式下，在Host侧和Device侧调用该接口前，如需设置超时时间，需要各自调用[HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md)设置，不调用设置接口则默认超时时间为1836秒（Host侧默认为1836+50秒）。
<!-- npu="950" id7 -->
- 针对Ascend 950PR/Ascend 950DT的AICPU_TS模式，`notifyIdx`必须小于本端Thread的Notify数量，且Thread创建时的`notifyNumPerThread`需大于0。
<!-- end id7 -->

## 调用示例

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream streams[2];
ThreadHandle threads[2];
// 申请2条流，每条流2个Notify
aclrtCreateStream(&streams[0]);
aclrtCreateStream(&streams[1]);
HcclResult result = HcclThreadAcquireWithStream(comm, engine, streams[0], 2, &threads[0]);
result = HcclThreadAcquireWithStream(comm, engine, streams[1], 2, &threads[1]);

// 1. 设置默认超时时间（可选，未设置则使用默认值1836秒）
uint32_t defaultTimeout = 600;  // 10分钟
HcommSetNotifyWaitTimeOut(defaultTimeout);

uint32_t notifyIdx = 0;
// 发送同步信号
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
// 使用默认超时时间等待同步信号（无需手动传入超时参数）
HcommThreadNotifyWaitOnThreadWithDefaultTimeout(threads[1], notifyIdx);
```
<!-- npu="950" id8 -->
在Ascend 950PR/Ascend 950DT上，该函数需要编译到Device侧使用：

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[2];
uint32_t notifyNumPerThread = 2;
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[0]);
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[1]);

// 其余资源申请
// 拷贝参数并Launch Kernel

// 1. 设置默认超时时间（可选，未设置则使用默认值1836秒）
uint32_t defaultTimeout = 1800;  // 30分钟
HcommSetNotifyWaitTimeOut(defaultTimeout);

// Device侧算法编排
uint32_t notifyIdx = 0;
// 发送同步信号
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
// 使用默认超时时间等待同步信号（无需手动传入超时参数）
HcommThreadNotifyWaitOnThreadWithDefaultTimeout(threads[1], notifyIdx);
```
<!-- end id8 -->

## 相关接口

- [HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md) - 设置默认超时时间
- [HcommThreadNotifyRecordOnThread](./HcommThreadNotifyRecordOnThread.md) - 发送Thread同步信号
- [HcommThreadNotifyWaitOnThread](./HcommThreadNotifyWaitOnThread.md) - 等待Thread同步信号（手动指定超时时间）
