# HcommSetNotifyWaitTimeOut

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

设置通知等待的默认超时时间。该超时时间将应用于后续调用的[HcommChannelNotifyWaitOnThreadWithDefaultTimeout](./HcommChannelNotifyWaitOnThreadWithDefaultTimeout.md)和[HcommThreadNotifyWaitOnThreadWithDefaultTimeout](../local_operations/HcommThreadNotifyWaitOnThreadWithDefaultTimeout.md)接口。

## 函数原型

```c
int32_t HcommSetNotifyWaitTimeOut(float timeOut)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | -------------- |
| timeOut | 输入 | 通知等待超时时间。<br>单位：秒。<br>取值说明：<br>  - 0：表示永久等待，不超时。<br>  - >0：具体的超时时间（秒）。<br>默认值：1836秒（约30分钟）。<br>当前只支持整数传入。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 超时机制说明

1. **默认超时时间**
   - 如果未调用本接口设置，默认超时时间为1836秒（约30分钟）
   
2. **超时生效条件**
   - 本接口设置的超时时间将应用于后续的 `*WithDefaultTimeout`系列接口
   - 已调用 `HcommChannelNotifyWaitOnThread`或`HcommThreadNotifyWaitOnThread`设置的超时不受本接口影响

3. **非AICPU模式特殊处理**
   - 在非AICPU模式下，如果未调用本接口设置默认超时时间
   - 系统会自动在默认值基础上增加50秒的偏移量作为安全缓冲

## 约束说明

- 本接口设置的超时时间仅对`*WithDefaultTimeout`系列接口生效。
- 手动传入超时时间的 `HcommChannelNotifyWaitOnThread`和`HcommThreadNotifyWaitOnThread`不受本接口影响。
- 建议在数据面操作开始前设置默认超时时间。
- 当前只支持整数传入`timeOut`，不支持小数传入。

## 调用示例

```c
// 1. 设置默认超时时间（例如设置为10分钟）
float defaultTimeOut = 600;  // 600秒 = 10分钟
HcommSetNotifyWaitTimeOut(defaultTimeOut);

// 2. 申请通信线程资源
CommEngine engine = COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 3. 申请通信通道资源
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 4. 在Device侧记录通知
HcommChannelNotifyRecordOnThread(thread, channel, 0);

// 5. 数据面操作
// ...

// 6. 使用默认超时时间等待通知（无需手动传入超时参数）
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, 0);

// 或者使用线程通知等待
HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, 0);
```

## 相关接口

- [HcommChannelNotifyWaitOnThreadWithDefaultTimeout](./HcommChannelNotifyWaitOnThreadWithDefaultTimeout.md)：使用默认超时时间等待Channel通知。
- [HcommThreadNotifyWaitOnThreadWithDefaultTimeout](../local_operations/HcommThreadNotifyWaitOnThreadWithDefaultTimeout.md)：使用默认超时时间等待Thread通知。
- [HcommChannelNotifyWaitOnThread](./HcommChannelNotifyWaitOnThread.md)：手动指定超时时间等待Channel通知。
- [HcommThreadNotifyWaitOnThread](../local_operations/HcommThreadNotifyWaitOnThread.md)：手动指定超时时间等待Thread通知。
