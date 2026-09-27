# HcclCreateSubCommConfig

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
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

基于既有的全局通信域，切分具有特定配置的子通信域。

该子通信域创建方式无需进行socket建链与rank信息交换，可应用于业务故障下的快速通信域创建。

说明：

如果组网中卡间存在负载不均衡的情况，使用该接口创建的子通信域可能会由于卡间不同步发生建链超时。此时可通过环境变量HCCL_CONNECT_TIMEOUT增加设备间的建链超时时间。配置示例：

```bash
export HCCL_CONNECT_TIMEOUT=600
```

## 函数原型

```c
HcclResult HcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum, uint32_t *rankIds, uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig *config, HcclComm *subComm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 被切分的全局通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| rankNum | 输入 | 需要切分的子通信域中的rank数量。 |
| rankIds | 输入 | 子通信域中rank在全局通信域中的rank id组成的数组。<br>需要注意：该数组应当是有序的，数组中每个rank的下标将映射为其在子通信域的rank id。 |
| subCommId | 输入 | 当前子通信域标识，用户自定义。<br>  - 若未在config参数中配置子通信域名称“hcclCommName”，系统会使用`{全局通信域名}_sub_{subCommId}`作为子通信域名称，此种场景下，需要确保“subCommId”在全局通信域中保持唯一。<br>  - 若在config参数中配置了子通信域名称“hcclCommName”，则优先以config中配置为准，此参数不再做校验。 |
| subCommRankId | 输入 | 本rank在子通信域中的rank id。<br>请配置为当前rank在rankIds数组中的下标索引。 |
| config | 输入 | 通信域配置项，包括buffer大小、确定性计算开关、通信域名称、通信算子展开模式等信息，配置参数需确保在合法值域内，关于HcclCommConfig中的详细参数含义及优先级可参见[HcclCommConfig](./data_type_definition/HcclCommConfig.md)的定义。<br>需要注意：传入的config必须先调用[HcclCommConfigInit](HcclCommConfigInit.md)对其进行初始化。 |
| subComm | 输出 | 将初始化后的子通信域以指针的信息回传给调用者。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 属于同一子通信域的rank调用该接口时传入的rankNum、rankIds、subCommId、config均应相同。
- 不需要创建子通信域的rank应当传入rankIds=nullptr和subCommId=0xFFFFFFFF，此场景不会对“subCommId”参数做校验。
- 只支持从全局通信域切分子通信域，不支持在子通信域中进一步切分子通信域。
- 不支持单卡在多个进程或者线程中调用该接口切分子通信域。

## 调用示例

```c
// 初始化全局通信域
HcclComm globalHcclComm;
HcclCommInitClusterInfo(rankTableFile, devId, &globalHcclComm);
// 通信域配置
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclBufferSize = 50;
strcpy(config.hcclCommName, "comm_1");
// 初始化子通信域
HcclComm hcclComm;
uint32_t rankIds[4] = {0, 1, 2, 3};  // 子通信域的Rank列表
// 当前rank在子通信域中的rank id设置为0
HcclCreateSubCommConfig(&globalHcclComm, 4, rankIds, 1, 0, &config, &hcclComm);
```
