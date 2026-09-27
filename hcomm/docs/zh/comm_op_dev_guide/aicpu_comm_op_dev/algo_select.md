# 算法选择

## 选择策略

AICPU通信算子应根据实际物理拓扑选择通信算法，例如：

- 单Server多卡Mesh拓扑，建议采用Mesh算法，利用卡间直连链路并行通信。
- 多Server、每个Server单卡的拓扑，建议采用NHR算法，按跨Server链路组织通信。

AICPU通信算法选择示意图如下所示：

![](./figures/algo_select_new.png)

> [!NOTE]说明
>
> 1. 算法选择应以运行环境查询到的实际物理拓扑为准，不能只根据rank数量判断拓扑。
> 2. 如果自定义算子只支持一种物理拓扑和一种算法实现，可以跳过算法选择步骤，直接使用该算法。

## 代码示例

```c
CommEngine engine;
if (engine == CommEngine::COMM_ENGINE_AICPU_TS) {
    algName = "AicpuMesh";  // 选择使用aicpu展开的Mesh算法
} else {
    return HCCL_E_NOT_SUPPORT;
}
```
