# HcclGetRootInfoScalable

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id2 -->

## 功能说明

此接口需要在HCCL初始化接口[HcclCommInitRootInfoScalable](HcclCommInitRootInfoScalable.md)前调用，仅需在每个root节点调用，用于生成root节点的rank标识信息（HcclRootInfo），并建立root间mesh全互联监听。

- 该接口需要和初始化接口[HcclCommInitRootInfoScalable](HcclCommInitRootInfoScalable.md)接口配对使用，不能单独使用。
- 多root（scalable）场景下，集群中的nRoot个root节点需分别调用本接口生成各自的rootInfo，所有root的rootInfo构成rootInfoList，作为[HcclCommInitRootInfoScalable](HcclCommInitRootInfoScalable.md)的入参传给集群内所有rank。

## 函数原型

```c
HcclResult HcclGetRootInfoScalable(HcclRootInfo *rootInfo)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| rootInfo | 输出 | 本root的标识信息，主要包含device ip、device id等信息，并包含root间mesh全互联监听端口。此信息需与其他root生成的rootInfo组成rootInfoList广播至集群内所有rank，用于HCCL初始化。<br>HcclRootInfo类型的定义可参见[HcclRootInfo](./data_type_definition/HcclRootInfo.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 每个root节点需调用一次本接口，n个root节点共生成n个rootInfo，组成rootInfoList传入[HcclCommInitRootInfoScalable](HcclCommInitRootInfoScalable.md)。
- 所有root生成的rootInfo需在调用[HcclCommInitRootInfoScalable](HcclCommInitRootInfoScalable.md)前就绪。

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
// 初始化通信域
HcclComm hcclComm;
HcclCommInitRootInfoScalable(rankSize, nRoot, rootInfoList, deviceId, 0, &config, &hcclComm);
// 销毁通信域
HcclCommDestroy(hcclComm);
```
