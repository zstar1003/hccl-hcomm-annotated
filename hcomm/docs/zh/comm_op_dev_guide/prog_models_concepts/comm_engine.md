# 通信引擎

HCCL可以使用不同通信引擎实现通信算子，不同的通信引擎适用不同的场景，对比结果如下：

**表1**  不同通信引擎的适用场景

| 通信引擎 | 优势 | 约束 | 适用场景 | 支持的产品型号 |
| --- | --- | --- | --- | --- |
| AI CPU+TS | 不占用计算核，通信效率高，适用于大数据高带宽场景。 | 通信静态开销较大，对小数据量通信场景不友好。 | 高带宽通信场景 | Ascend 950PR/Ascend 950DT<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
| Host CPU+TS | 不占用计算核。 | 下发开销大，随任务数线性增加。 | NA | Atlas A2 训练系列产品/Atlas A2 推理系列产品 |
| AIV | 低时延。 | 通信占用Vector计算核，需要多个Vector计算核才能打满通信带宽；通信算子与计算算子竞争计算核资源，可能互相影响。 | 低时延通信场景 | Ascend 950PR/Ascend 950DT<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品（仅支持超节点内通信场景）<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品（仅支持单机通信场景） |
| CCU | 减少访存带宽与计算核占用。<br>高带宽，低时延。 | 受限于片上资源，支持的通信域数量有限。 | 高带宽、低时延通信场景 | Ascend 950PR/Ascend 950DT |

下面分别介绍不同通信引擎的任务执行流程。

## AI CPU+TS

由AI CPU向任务调度系统（Task Scheduler，简写为TS）提交通信操作相关任务。

1. Host提交一个AI CPU Kernel至任务队列。
2. AI CPU Kernel被任务调度器调度后交给AI CPU执行。
3. AI CPU提交通信任务至任务队列。
4. AI CPU提交的通信任务被调度器调度至执行器执行。

**图1**  AI CPU+TS调度  
![](figures/aicpu_ts_schedule.png "AI-CPU+TS调度")

## Host CPU+TS

由Host CPU向Device侧的任务调度系统（Task Scheduler，TS）提交通信操作相关任务。

1. Host将通信过程中的各类操作（包括内存拷贝、同步操作等）提交至任务队列。
2. 调度器将下发至任务队列中的任务调度至对应的执行器上执行。

**图2**  Host CPU+TS调度  
![](figures/hostcpu_ts_schedule.png "Host-CPU+TS调度")

## AIV

通信算子的执行逻辑与操作步骤由Vector Core执行。

1. Host提交一个AIV Kernel至任务队列。
2. AIV Kernel被调度器调度后发送至Vector Core执行。
3. Vector Core可以利用不同协议完成数据搬运。

**图3**  AIV通信  
![](figures/aiv_communication.png "AIV通信")

## CCU

通信算子由CCU（Collective Communication Unit，集合通信加速单元）执行，如图所示。

1. Host将CCU指令序列（由CCU可识别的指令组成）下发至CCU指令空间中，同时提交对应CCU Kernel任务至任务队列。
2. CCU Kernel被调度器调度后发送至CCU执行。
3. CCU执行对应的指令流，并利用URMA（Unified Remote Memory Access，统一远端内存访问）完成数据搬运。

**图4**  CCU加速  
![](figures/ccu_communication.png "CCU加速")
