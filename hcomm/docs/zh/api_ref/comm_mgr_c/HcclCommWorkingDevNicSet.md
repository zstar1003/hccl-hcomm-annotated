# HcclCommWorkingDevNicSet

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

在集群场景下，配置通信域内的通信网卡。支持在同一NPU下，在某一Device网卡和备用网卡之间切换通信，备用网卡为同一NPU中的另一个Die网卡。

单次通信网卡配置时，同一通信域内的所有卡都需要调用该接口，且所有卡下发的ranks、useBackup和nRanks参数配置需保持一致。

> [!NOTE]说明
> 配置通信网卡过程中，请注意以下操作的影响：
>
> - 同时下发切换到备网卡和切换到主网卡的命令时，如果在链路两端，一端切换到主网卡，一端切换到备用网卡，则命令执行失败。
> - 如果存在一条卡1和卡2的链路，第一次下发命令使卡1切到备用网卡，第二次下发命令使卡2换到默认网卡，此时卡1和卡2之间的链路会使用默认网卡通信，卡1和其他卡的链路会使用备用网卡通信。
> - 如果卡1和卡2的网卡互为备用网卡，并且两张卡同时下发切换到备用网卡的命令，那么卡1和卡2使用的通信网卡将会互换。

## 函数原型

```c
HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 指定通信网卡的通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| ranks | 输入 | 指定通信网卡的rank在通信域中的rank id组成的数组。 |
| useBackup | 输入 | 指定ranks中的卡是否使用备用网卡。 |
| nRanks | 输入 | 指定切换网卡的rank数量。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 配置前，请确保当前环境上的通信任务已完成。
- 属于同一通信域的rank调用该接口时传入的ranks和useBackup数组长度与nRanks数量保持一致。
- 使用该接口需满足以下两种情况时，才会发生实际的网卡切换。
  - 超节点间通信开启重执行，即环境变量HCCL_OP_RETRY_ENABLE的“L2”配置为1。
  - 拓扑形态存在RDMA链路。

- 对于不支持的场景，如HCCL_OP_RETRY_ENABLE未开启重执行时，会返回HCCL_SUCCESS并在日志中打印WARNING，但实际未切换网卡。
- 针对同一个rank，HcclCommWorkingDevNicSet接口需要one by one保序调用，不支持并发下发。
- 针对整个通信域，调用HcclCommWorkingDevNicSet接口时，在不同rank间要保证同一下发顺序。
- comm句柄指向的通信域必须已下发过算子才能执行网卡配置，后续新下发同类型同参数的算子会继续使用同样的网卡配置，其他新下发的算子会使用默认网卡通信。不满足以上要求，会出现网卡配置失败。

## 调用示例

```c
HcclComm comm;
uint32_t rankSize = 4;
uint32_t rankIds[rankSize] = {0, 3, 7, 12};
bool useBackup[rankSize] = {true, true, true, true};
HcclCommWorkingDevNicSet(comm, rankIds, useBackup, rankSize);
```
