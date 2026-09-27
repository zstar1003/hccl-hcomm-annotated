# 编程模型与概念

介绍HCCL通信算子开发中的核心编程模型与关键概念，帮助开发者理解通信引擎、通信模型、并发模型、拓扑模型及CCU编程模型。

- [通信引擎](comm_engine.md)：介绍HCCL支持的AI CPU+TS、Host CPU+TS、AIV和CCU四种通信引擎的适用场景与任务执行流程。
- [通信模型](comm_model.md)：描述HCCL通信模型中的通信内存、Endpoint、Channel等核心概念，以及网络语义、内存语义和CCU三种通信模型。
- [并发模型](concurrency_model.md)：介绍以Thread为并发单元、通过Notify机制实现通信任务并发执行与同步的编程模型。
- [拓扑模型](topology_model.md)：说明HCCL对通信域内rank间连接关系的拓扑建模，包括Node、Endpoint、Edge、Link及分层拓扑概念。
- [CCU编程模型与概念](CCU_models_concepts.md)：介绍CCU集合通信加速单元的架构、资源抽象、数据搬运与计算能力、并发模型、同步机制及流程控制。
