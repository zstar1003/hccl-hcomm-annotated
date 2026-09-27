# HcclRankGraphGetLinks

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
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

给定通信域和拓扑层级编号，查询源rank和目的rank之间的通信连接信息。

以Atlas A3 训练系列产品/Atlas A3 推理系列产品为例：

- 示例1：源rank与目的rank分别在两个超节点。

  netLayer = 0， 无连接。

  netLayer = 1， 无连接。

  netLayer = 2， RDMA连接。

- 示例2：源rank与目的rank在同一个超节点，但不在同一个AI Server内。

  netLayer = 0， 无连接。

  netLayer = 1， HCCS连接。

  netLayer = 2， 无连接。

- 示例3：源rank与目的rank在同一个AI Server内，不在同一个NPU中。

  netLayer = 0， HCCS连接。

  netLayer = 1， 无连接。

  netLayer = 2， 无连接。

## 函数原型

```c
HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink **links, uint32_t *linkNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| netLayer | 输入 | 拓扑层级编号。 |
| srcRank | 输入 | 源rank编号。 |
| dstRank | 输入 | 目的rank编号。 |
| links | 输出 | 通信连接列表。<br>CommLink类型的定义请参见[CommLink](../../datatype_definition/CommLink.md)。 |
| linkNum | 输出 | 通信连接数量。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 返回的内存由库内管理，调用者严禁释放。
- 应及时复制返回数据，同一通信域重复调用可能使前次结果失效。

## 调用示例

```c
HcclComm comm;
CommLink *links;
uint32_t linkNum;
uint32_t netlayer = 0;
// 查询机内rank0与rank1间的链路
HcclRankGraphGetLinks(comm, netlayer, 0, 1, &links, &linkNum);
```
