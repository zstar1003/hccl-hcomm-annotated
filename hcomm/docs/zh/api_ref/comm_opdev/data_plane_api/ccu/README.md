# CCU接口简介

本节提供CCU（Collective Communication Unit）kernel内的数据面接口，用于在算子kernel函数体内描述通信与同步逻辑。

CCU采用"Host注册+Device执行"两阶段模型：用户在kernel函数体内调用本节接口描述逻辑，框架在注册阶段先记录这些调用，结束注册时统一翻译为CCU设备指令，最终由CCU硬件执行。

按功能分为以下几类：

- [资源创建与操作](./resource_allocation_operation/README.md)
- [参数加载/存储](./arg_load_store/README.md)
- [数据搬运](./data_movement/README.md)
- [同步](./synchronization/README.md)
- [流程控制](./execution_control/README.md)
