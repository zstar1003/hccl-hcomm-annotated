# HcclConfigGetInfo

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

获取指定通信域的HCCL配置信息。

根据配置项类型查询对应的配置信息，并写入调用者提供的缓冲区中，当前支持查询通信算子的展开模式、通信算法配置字符串及UB多channel数量。

## 函数原型

```c
HcclResult HcclConfigGetInfo(HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, void *info);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| cfgType | 输入 | 需要查询的配置项类型，HcclConfigType的定义可参见[HcclConfigType](./data_type_definition/HcclConfigType.md)。 |
| infoLen | 输入 | 目标配置类型的大小（字节数），查询HCCL_CONFIG_TYPE_OP_EXPANSION_MODE时必须等于待查询配置类型的实际大小,查询HCCL_CONFIG_TYPE_HCCL_ALGO时必须不小于HCCL_COMM_ALGO_MAX_LENGTH字节数，查询HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM时必须等于sizeof(uint32_t)。 |
| info | 输出 | 配置信息输出缓冲区，必须按目标配置类型对齐且可写。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
HcclConfigTypeOpExpansionMode mode;
uint32_t size = sizeof(HcclConfigTypeOpExpansionMode); // 必须等于目标类型大小
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, size, &mode);

// 查询通信算法字符串
char algoInfo[HCCL_COMM_ALGO_MAX_LENGTH];
uint32_t algoSize = HCCL_COMM_ALGO_MAX_LENGTH; // 必须不小于HCCL_COMM_ALGO_MAX_LENGTH
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_HCCL_ALGO, algoSize, algoInfo);

// 查询UB多channel数量
uint32_t multiChannelNum = 0;
uint32_t numSize = sizeof(uint32_t); // 必须等于目标类型大小
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM, numSize, &multiChannelNum);
```
