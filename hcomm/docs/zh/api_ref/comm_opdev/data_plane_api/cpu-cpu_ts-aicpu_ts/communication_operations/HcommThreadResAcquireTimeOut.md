# HcommThreadResAcquireTimeOut

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

设置线程资源获取的超时时间。该超时时间应用于RTSQ（Remote Transport Sequence Queue）的队列满等待场景。当RTSQ队列已满时，会等待直到有可用空间或超时。

## 函数原型

```c
int32_t HcommThreadResAcquireTimeOut(float timeOut)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| timeOut | 输入 | RTSQ队列满等待超时时间。<br>单位：秒。<br>取值说明：<br>  - 0：表示永不超时，等待直到RTSQ有可用空间。<br>  - >0：具体的超时时间（秒）。<br>默认值：1856秒。<br>当前只支持整数传入。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 超时机制说明

1. **超时值语义**
   - 当RTSQ队列已满时，会阻塞等待直到有可用空间
   - 如果设置了超时时间，在超时后仍未有可用空间则报错
   - 设置为0表示永不超时，会一直等待

2. **默认值**
   - 如果未调用本接口设置，默认超时时间为1856秒（约30分钟）

3. **设置范围**
   - 适用于所有通过线程获取的RTSQ资源
   - 设置后会影响到所有后续的资源获取操作

## 应用场景

该接口主要用于以下场景：

- **RTSQ队列满等待**：当发送缓冲区满时，等待可用空间
- **流控机制**：配合发送操作实现背压（backpressure）机制
- **资源获取**：确保资源可用后再进行后续操作

## 约束说明

- 该接口设置的超时时间会影响所有线程的资源获取操作
- 建议在初始化阶段或资源申请前设置合适的超时值
- 超时值为0时会一直等待，可能导致死锁风险，请谨慎使用
- 当前只支持整数传入`timeOut`，不支持小数传入。

## 调用示例

```c
// 1. 设置RTSQ队列满等待超时时间（例如设置为10分钟）
float timeout = 600;  // 600秒 = 10分钟
HcommThreadResAcquireTimeOut(timeout);

// 或者设置为永不超时（0表示永不超时）
float timeoutNever = 0;
HcommThreadResAcquireTimeOut(timeoutNever);

// 2. 申请通信线程资源
CommEngine engine = COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 3. 数据面操作（RTSQ满时会等待可用空间或超时）
// ... 执行数据发送操作 ...
```

## 相关接口

[HcommSetNotifyWaitTimeOut](./HcommSetNotifyWaitTimeOut.md)：设置通知等待超时时间。
