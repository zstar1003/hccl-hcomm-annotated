# HcommBatchTransferOnThread

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

在指定线程和通道上异步提交一组传输任务。每个传输任务由[HcommBatchTransferDesc](../../../datatype_definition/HcommBatchTransferDesc.md)描述，支持单边写、单边读、写归约、带通知的写以及通知记录/等待等操作类型。

## 函数原型

```c
int32_t HcommBatchTransferOnThread(ThreadHandle thread, ChannelHandle channel,
    const HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 通信线程句柄。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| channel | 输入 | 通信通道句柄。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。 |
| transferDescs | 输入 | 批量传输描述符数组，数组中每个元素描述一个传输任务，按数组顺序依次提交。描述符定义参见[HcommBatchTransferDesc](../../../datatype_definition/HcommBatchTransferDesc.md)。 |
| transferDescNum | 输入 | 描述符数量，须大于0。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

<!-- npu="A3,910b" id6 -->
针对如下产品，该接口仅支持通信协议RoCE，且transType仅支持HCOMM_TRANSFER_TYPE_WRITE和HCOMM_TRANSFER_TYPE_READ类型，其他类型返回HCCL_E_NOT_SUPPORT。

<!-- npu="A3" id7 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id7 -->
<!-- npu="910b" id8 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id8 -->
<!-- end id6 -->

## 调用示例

```c
// 申请通信线程资源和通道资源（略）
// 获取本端和对端通信内存信息（略）

// 构造批量传输描述符
HcommBatchTransferDesc descs[2];

// 描述符1：单边写
descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
descs[0].transferInfo.write.len = 4096;
descs[0].transferInfo.write.dst = remoteBuffer;    // 远端地址
descs[0].transferInfo.write.src = localBuffer;     // 本地地址

// 描述符2：单边写（带通知）
descs[1].transType = HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY;
descs[1].transferInfo.writeWithNotify.len = 8192;
descs[1].transferInfo.writeWithNotify.dst = remoteBuffer2;    // 远端地址
descs[1].transferInfo.writeWithNotify.src = localBuffer2;     // 本地地址
descs[1].transferInfo.writeWithNotify.notifyIdx = 0;          // 通知索引

// 提交批量传输
HcommBatchTransferOnThread(thread, channel, descs, 2);
```
