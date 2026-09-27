# HcclCommInitRootInfoScalable

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id2 -->

## 功能说明

根据nRoot个rootInfo初始化HCCL，创建支持多root（scalable）的HCCL通信域。集群中nRanks个rank被划分为nRoot个分组，每个分组由一个root管理，root间通过mesh全互联协同完成通信域初始化。当前rank所属分组的root信息由分组算法根据nRanks、nRoot和rank确定。

该接口在同一进程内支持多线程并发调用，但仅支持单卡单线程的场景，若是单卡多线程，不支持并发调用。

## 函数原型

```c
HcclResult HcclCommInitRootInfoScalable(uint32_t nRanks, uint32_t nRoot, const HcclRootInfo *rootInfoList, uint32_t rank, uint32_t nExtRoot, const HcclCommConfig *config, HcclComm *comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| nRanks | 输入 | 集群中的rank数量。 |
| nRoot | 输入 | root数量，即root分组的个数，取值需满足0 < nRoot ≤ nRanks。 |
| rootInfoList | 输入 | 长度为nRoot的rootInfo数组，由各root节点调用[HcclGetRootInfoScalable](HcclGetRootInfoScalable.md)生成，数组下标即root的index，需按固定顺序排列。 |
| rank | 输入 | 本rank的rank id。 |
| nExtRoot | 输入 | 预留参数，当前仅支持传入0，用于当前一级root基础上未来两级root扩展能力。 |
| config | 输入 | 通信域配置项，包括buffer大小、确定性计算开关、通信域名称、通信算子展开模式等信息，配置参数需确保在合法值域内，关于HcclCommConfig中的详细参数含义及优先级可参见[HcclCommConfig](./data_type_definition/HcclCommConfig.md)的定义。<br>需要注意：传入的config必须先调用[HcclCommConfigInit](HcclCommConfigInit.md)对其进行初始化。 |
| comm | 输出 | 初始化后的通信域指针。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 同一通信域中所有rank的nRanks、nRoot、rootInfoList、config均应相同，且rootInfoList的内容及顺序需完全一致。
- nRoot需满足0 < nRoot ≤ nRanks。

## 调用示例

```c
uint32_t rankSize = 32;
uint32_t nRoot = 4;
uint32_t deviceId = 0;
HcclRootInfo rootInfoList[4];
// 每个root节点生成自己的rank标识信息
if (isRootRank) {
    HcclGetRootInfoScalable(&rootInfoList[rootIndex]);
}
// 通过集合通信将rootInfoList广播给集群内所有rank

// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
// 初始化集合通信域
HcclComm hcclComm;
HcclCommInitRootInfoScalable(rankSize, nRoot, rootInfoList, deviceId, 0, &config, &hcclComm);
// 销毁通信域
HcclCommDestroy(hcclComm);
```
