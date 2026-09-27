# 算法选择

## 选择策略

通信引擎为AIV时，通信算子在Vector Core展开，下发与展开阶段的路径较短、时延开销比较低，适用于通信数据量较小、对时延要求较高的推理场景。

一个通信算子往往具有多种算法实现，但针对AIV通信算子，建议采用Mesh算法，因为Mesh算法具有极少的通信步数，可进一步降低通信时延。

AIV引擎通信算法选择示意图如下所示：

![AIV引擎通信算法选择示意图](figures/aiv_algo_select.png)

> [!NOTE]说明
>
> 1. 如果通信算子只支持一种通信引擎和算法实现，那么可以跳过本节的算法选择步骤。
> 2. 本章所述的算法是开发者自行实现的算法，HCCL_ALGO环境变量配置的算法是HCCL内置算法。HCCL内置算法可参考《[HCCL集合通信库用户指南](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/README.md)》的相关参考 \> 集合通信算法介绍。

## 代码示例

以[选择策略](#选择策略)介绍的算法选择策略为例，对应的算法选择代码片段如下：

```c
CommEngine engine;
if (engine == CommEngine::COMM_ENGINE_AIV) {
    algName = "AivMesh";  // 选择使用AIV展开的Mesh算法
} else {
    return HCCL_E_NOT_SUPPORT;
}
```
