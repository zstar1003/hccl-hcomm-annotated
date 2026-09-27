# HcclCommInitClusterInfoConfig

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

基于rank table初始化HCCL，创建具有特定配置的HCCL通信域。

## 函数原型

```c
HcclResult HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config, HcclComm *comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| clusterInfo | 输入 | rank table文件路径（含文件名），作为字符串最大长度为4096字节，含结束符。 |
| rank | 输入 | 本rank的id。<br>需要注意，此参数取值需要与rank table中对应的“rank_id”字段取值一致。 |
| config | 输入 | 通信域配置项，包括buffer大小、确定性计算开关、通信域名称、通信算子展开模式等信息，配置参数需确保在合法值域内，关于HcclCommConfig中的详细参数含义及优先级可参见[HcclCommConfig](./data_type_definition/HcclCommConfig.md)的定义。<br>需要注意：传入的config必须先调用[HcclCommConfigInit](HcclCommConfigInit.md)对其进行初始化。 |
| comm | 输出 | 将初始化后的通信域以指针的信息回传给调用者。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

同一通信域不支持重复初始化。

## 调用示例

```c
// 设备资源初始化
aclInit(NULL);
// rank table配置文件路径
const char *rankTableFile = "/path/to/rank_table.json";
// 指定集合通信操作使用的设备
uint32_t rankSize = 8;
uint32_t devId = 0;
aclrtSetDevice(devId);
// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
// 按需修改通信域配置
config.hcclBufferSize = 50;  // 共享数据的缓存区大小，单位为：MB，取值需 >= 1，默认值为：200
strncpy(config.hcclCommName, "comm_1", COMM_NAME_MAX_LENGTH - 1);
config.hcclCommName[COMM_NAME_MAX_LENGTH - 1] = '\0';
// 初始化通信域
HcclComm hcclComm;
// 此样例以devId作为当前rank的rank id
HcclCommInitClusterInfoConfig(rankTableFile, devId, &config, &hcclComm);
// 销毁通信域
HcclCommDestroy(hcclComm);
// 设备资源去初始化
aclFinalize();
```
