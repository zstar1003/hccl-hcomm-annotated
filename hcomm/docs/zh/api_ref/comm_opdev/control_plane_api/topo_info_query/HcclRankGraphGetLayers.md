# HcclRankGraphGetLayers

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

查询包含当前rank的拓扑层级编号列表以及拓扑层级数量。

![拓扑模型](figures/topo_model.png)

以上述拓扑模型为例，包含Layer 0和Layer 1两级拓扑，调用此接口后返回的拓扑层级编号列表为\[0,1\]，拓扑层级数量为2。

## 函数原型

```c
HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 当前rank所在的通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| netLayers | 输出 | 拓扑层级编号列表。 |
| netLayerNum | 输出 | 拓扑层级编号数量。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 返回的内存由库内管理，调用者严禁释放。
- 应及时复制返回数据，同一通信域重复调用可能使前次结果失效。

## 调用示例

以[功能说明](#功能说明)的拓扑模型为例：

```c
HcclComm comm;
uint32_t *netLayers;
uint32_t layerNum;
HcclRankGraphGetLayers(comm, &netLayers, &layerNum);
// netLayers=[0,1], layerNum=2
```
