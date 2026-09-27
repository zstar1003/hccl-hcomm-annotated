# HcommFenceOnThread

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

在Host CPU侧触发数据面全局Fence操作，对当前已初始化的内部Flush资源执行Flush并等待完成。

该接口不绑定具体通信通道。如需等待指定通信通道上已提交的读写操作完成，应先调用[HcommChannelFenceOnThread](HcommChannelFenceOnThread.md)。

## 函数原型

```c
int32_t HcommFenceOnThread(ThreadHandle thread)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 通信线程句柄。Host CPU侧调用时，该参数无作用，传入0即可。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

<!-- npu="950" id6 -->
- 针对Ascend 950PR/Ascend 950DT，仅支持Host CPU侧调用。
<!-- end id6 -->
- Host CPU侧调用时，通信引擎为CPU，支持通信协议RoCE，`thread`参数无作用，可传入0。
- 该接口用于Host CPU侧数据面Flush，不等待指定通信通道上的读写操作完成。如需等待通道上已提交的读写操作完成，应先调用[HcommChannelFenceOnThread](HcommChannelFenceOnThread.md)。
- 当前没有需要执行Flush的内部资源时，接口返回成功。

## 调用示例

### 集合通信示例

```c
// 申请通信通道资源
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
HcclComm comm;
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 获取本端通信内存信息
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// 获取对端通信内存信息
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// 将本端内存的内容写到对端内存上
int32_t ret = HcommWriteNbiOnThread(0, channel, remoteBuffer, localBuffer, len);
ret = HcommChannelFenceOnThread(0, channel);

// 执行Host CPU侧数据面Flush并等待完成
ret = HcommFenceOnThread(0);
```
