# HcclRankGraphGetInstSizeListByLayer

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

给定通信域和拓扑层级编号，查询该层级下的拓扑实例数量，以及每个实例包含的rank数量。

![拓扑模型](figures/topo_model.png)

以上述拓扑模型为例：

- Layer 0中包含两个拓扑实例，为方便理解，定义拓扑实例ID分别为0和1，每个实例中分别包含3个rank。
- Layer 1中包含1个拓扑实例，实例中包含6个rank。

## 函数原型

```c
HcclResult HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| netLayer | 输入 | 拓扑层级编号。 |
| instSizeList | 输出 | 该层级下每个拓扑实例包含的rank数量组成的列表。 |
| listSize | 输出 | instSizeList的大小，即该层级下包含的拓扑实例的数量。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 返回的内存由库内管理，调用者严禁释放。
- 应及时复制返回数据，同一通信域重复调用可能使前次结果失效。

## 调用示例

以[功能说明](#功能说明)的拓扑模型为例：

```c
HcclComm comm;
uint32_t *instSizeList;
uint32_t listSize;
HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize);
// instSizeList=[3,3], listSize=2
HcclRankGraphGetInstSizeListByLayer(comm, 1, &instSizeList, &listSize);
// instSizeList=[6], listSize=1
```
