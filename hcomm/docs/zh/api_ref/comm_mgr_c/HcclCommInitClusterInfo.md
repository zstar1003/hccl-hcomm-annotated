# HcclCommInitClusterInfo

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

基于rank table初始化HCCL，创建HCCL通信域。

rank table文件是一个JSON格式的文件，配置了参与集合通信的NPU资源信息，关于rank table文件的配置可参见[集群信息配置](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/cluster_info_config/README.md)。

## 函数原型

```c
HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| clusterInfo | 输入 | rank table的文件路径（含文件名），作为字符串最大长度为4096字节，含结束符。 |
| rank | 输入 | 本rank的id。<br>需要注意，此参数取值需要与rank table中对应的“rank_id”字段取值一致。 |
| comm | 输出 | 将初始化后的通信域以指针的信息回传给调用者。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

重复初始化会报错。

## 调用示例

```c
// 设备资源初始化
aclInit(NULL);
// rank table配置文件路径
char *rankTableFile = "/path/rank_table.json";
// 指定集合通信操作使用的Device ID
uint32_t devId = 0;
aclrtSetDevice(devId);
// 创建通信域
HcclComm hcclComm;
// 此样例以devId作为当前rank的rank id
HcclCommInitClusterInfo(rankTableFile, devId, &hcclComm);
// 销毁通信域
HcclCommDestroy(hcclComm);
// 设备资源去初始化
aclFinalize();
```
