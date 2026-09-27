# HcclCommInitRootInfoConfig

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

根据rootInfo初始化HCCL，创建具有特定配置的HCCL通信域。

该接口在同一进程内支持多线程并发调用，但仅支持单卡单线程的场景，若是单卡多线程，不支持并发调用。

如下图所示，不支持step0与step1并发调用，需要step0执行结束后，再串行执行step1。

![不支持并发调用](figures/commit_rootinfo_config.png)

## 函数原型

```c
HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, const HcclCommConfig *config, HcclComm *comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| nRanks | 输入 | 集群中的rank数量。 |
| rootInfo | 输入 | root rank信息，主要包含root rank的ip、id等信息，由[HcclGetRootInfo](HcclGetRootInfo.md)接口生成。 |
| rank | 输入 | 本rank的rank id。 |
| config | 输入 | 通信域配置项，包括buffer大小、确定性计算开关、通信域名称、通信算子展开模式等信息，配置参数需确保在合法值域内，关于HcclCommConfig中的详细参数含义及优先级可参见[HcclCommConfig](./data_type_definition/HcclCommConfig.md)的定义。<br>需要注意：传入的config必须先调用[HcclCommConfigInit](HcclCommConfigInit.md)对其进行初始化。 |
| comm | 输出 | 初始化后的通信域指针。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

同一通信域中所有rank的nRanks、rootInfo、config均应相同。

## 调用示例

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// 生成root节点的rank标识信息
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);

// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
// 按需修改通信域配置
config.hcclBufferSize = 1024;  // 共享数据的缓存区大小，单位为：MB，取值需 >= 1，默认值为：200
config.hcclDeterministic = 1;  // 开启归约类通信算子的确定性计算，默认值为：0，表示关闭确定性计算功能
std::strcpy(config.hcclCommName, "comm_1");
// 初始化集合通信域
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, deviceId, &config, &hcclComm));

// 销毁通信域
HcclCommDestroy(hcclComm);
```
