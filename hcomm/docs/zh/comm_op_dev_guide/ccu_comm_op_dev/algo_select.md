# 算法选择

## 选择策略

使用CCU引擎可以节省HBM带宽，相比AI CPU模式可以减少通信展开时间，并且不占用计算硬件（例如AIV）。但同时，CCU的硬件资源是有限的，因此推荐在有极致性能要求的关键场景中使用，例如TP、EP并行。

**图1**  通信算法选择示意图  
![](figures/ccu_algo_select.png "通信算法选择示意图")

如上图所示，CCU通信算子支持多种算法实现，开发者可以根据拓扑信息选择最优通信算法：

- Mesh算法实现：适用于Server内物理拓扑为Mesh的场景。
- NHR算法实现：适用于多Server，且每个Server中取一个rank进行通信的场景。

> [!NOTE]说明
>
>1. 如果通信算子只有一种算法实现，那么可以跳过本节的算法选择步骤。
>2. 本章所述的算法是开发者自行实现的算法，HCCL\_ALGO环境变量配置的算法是HCCL内置算法。HCCL内置算法可参考《[HCCL集合通信库用户指南](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/README.md)》的相关参考 \> 集合通信算法介绍。

## 代码示例

以[选择策略](#选择策略)章节介绍的算法选择策略为例，对应的算法选择代码片段如下：

```c
CommEngine engine;
if (engine == CommEngine::COMM_ENGINE_CCU) {
    algName = "CCUAllGatherMesh";  // 选择使用CCU引擎的Mesh算法
} else {
    return HCCL_E_NOT_SUPPORT;
}
```
