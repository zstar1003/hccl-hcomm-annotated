# EI0002 Communication_Error_Timeout

## 错误信息

报错格式如下，占位符%s的含义依次为对端Rank ID、任务信息、通信算子信息、通信域信息：

```text
A timeout occurs when the Notify register waits for execution. Waiting peer rank: %s; task information: %s; communication operator information: %s; communicator: %s.
```

报错示例如下：

```text
A timeout occurs when the Notify register waits for execution. Waiting peer rank: 4; task information: streamID:[90], taskID[686], taskType[Notify Wait], tag[AllReduce_80.48.9.154%enp48s3u1u1_60000_0_1779783710697217ringAllReduceMeshSmallCountExecutor_device], AlgType(level 0-1-2):[ring-ring-NHR].; communication operator information: notify id:[0x00000000000018fc], stage:[0], remote rank:[4]; communicator: none.
```

## 可能原因

1. 集群中的某些NPU在执行过程中发生异常，导致集合通信操作失败。

2. 集群中的某些NPU执行速度过慢，无法在超时时间内完成通信操作（默认超时时间为1800秒，可通过环境变量HCCL_EXEC_TIMEOUT进行设置）。

3. 每个NPU的训练样本数量不一致。

4. 通信链路出现丢包或其他连通性问题。

## 解决方法

1. 如果这些rank中的某一部分报错，需检查其他rank是否已提前报错。

2. 如果所有rank都报错，需检查报错时间是否一致（最大差异不能超过1800秒），如果不一致，需定位原因或将HCCL_EXEC_TIMEOUT环境变量设置为更大的值。

3. 确保每个NPU的训练样本数量一致。

4. 使用grep -rn 'error cqe'命令检查plog中是否存在该错误的completion queue element \(CQE\)，如果存在，则需要检查网络连接状态。具体排查方法可参考[昇腾社区文档中心](https://www.hiascend.com/document)，搜索关键字“EI0002”。
