# HcommChannelGetStatus

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

该接口用于查询通信通道的建链状态，并推动建链状态机前进一步。每次调用会对所有传入的通道执行一轮建链推进操作，并返回当前状态。调用方需反复调用该接口直至通道状态变为就绪或失败，通道就绪后方可进行通信操作。


## 函数原型

```c
HcommResult HcommChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channelList | 输入 | 待查询状态的通道句柄数组，每个元素标识一个已创建的通信通道。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。<br>该参数不能为空指针，数组中每个通道句柄必须是通过[HcommChannelCreate](HcommChannelCreate.md)创建的有效句柄。 |
| listNum | 输入 | 待查询的通道数量。<br>单位为“个”，取值范围：[1, 1048576]。<br>该参数必须大于0。 |
| statusList | 输出 | 通道状态数组，用于返回每个通道的当前状态，与channelList一一对应。<br>该参数不能为空指针。<br>调用者分配的数组，至少包含listNum个元素的空间。<br>状态值定义如下：<br>0：建链完成，通道就绪。<br>1：建链进行中，需继续调用本接口推动建链。<br>2：建链失败。<br>3：建链超时。<br>4：本端资源不足，建链失败，当前版本仅能识别并上报CCU资源不足。<br>5：对端资源不足，建链失败，当前版本仅能识别并上报CCU资源不足。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- channelList数组长度需要与listNum参数一致。
- statusList\[i\]与channelList\[i\]一一对应，表示第i个通道的状态。
- 同一个ChannelHandle不能在多线程中并发访问，即同一个通道的建链状态查询和通信以及销毁操作必须串行执行。

## 调用示例

```c
uint32_t channelNum = 50;
std::vector<ChannelHandle> channels(channelNum);
// 参考HcommChannelCreate进行Channel创建
...

std::vector<int32_t> statuses(channelNum, 0);
bool hasFailed = false;
while (true) {
    HcommResult ret = HcommChannelGetStatus(channels.data(), channelNum, statuses.data());
    if (ret != HCCL_SUCCESS) {
        // 接口调用失败
        break;
    }

    bool allReady = true;
    for (uint32_t i = 0; i < channelNum; i++) {
        if (statuses[i] >= 2 && statuses[i] <= 5) {
            // 建链失败或超时，退出
            hasFailed = true;
            allReady = false;
            break;
        }
        if (statuses[i] != 0) {
            allReady = false;
        }
    }
    if (allReady || hasFailed) {
        break;
    }
    // 建链进行中，适当休眠后重试
    usleep(1000);
}

if (hasFailed) {
    // 建链失败或超时，执行错误处理
    // ...
    return;
}

 // 通道就绪，执行通信操作
 // ...
```
