# EI0012 Execution_Error_SDMA

## 错误信息

报错格式如下，占位符%s的含义依次为对端Rank信息、基础信息、任务信息、通信域信息：

```text
SDMA memory copy task exception occurred. Remote rank: %s. Base information: %s. Task information: %s. Communicator information: %s.
```

报错示例如下：

```text
SDMA memory copy task exception occurred. Remote rank: [4800]. Base information: [streamID:[44], taskID[2], taskType[Memcpy], tag[AllReduce_group_name_0], AlgType(level 0-1-2):[ring-ring-NHR].]. Task information: [src:[0x320000], dst:[0x12c088120000], size:[0x320000], notify id:[0xffffffffffffffff], link type:[OnChip], remote rank:[local]]. Communicator information: [group:[group_name_0], user define information[Unspecified], rankSize[16], rankId[13]].
```

## 可能原因

1. SDMA任务执行过程中网络连接出现异常。
2. 对端进程先异常退出。
3. 输入或者输出的内存地址未分配、实际分配大小小于传入数据量或算子未执行完就释放了内存。

## 解决方法

1. 请检查执行过程中的网络链路是否有发生异常。
2. 请检查集群中是否有进程在报错之前先退出，如果有，则需排查该进程先退出的原因。
3. 检查输入/输出内存大小是否正确，以及内存或通信域是否提前释放。
