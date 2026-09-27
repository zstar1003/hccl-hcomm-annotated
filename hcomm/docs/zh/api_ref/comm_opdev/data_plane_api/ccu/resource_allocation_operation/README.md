# 简介

本节提供CCU kernel内虚拟资源句柄的申请与绑定接口，以及对`Variable`/`Address`进行赋值与算术运算的接口。

CCU资源分配采用"先虚后实"两阶段模型：用户在kernel注册阶段（kernel函数体内）调用`Variable()`/`Address()`/`Event()`等默认构造函数仅产生虚拟句柄（不消耗硬件资源、恒成功），真正的物理资源分配在`HcommCcuKernelRegister`阶段（kernel函数执行完后）完成。默认构造的C++包装类均遵循构造即虚拟分配、析构不释放的语义；物理资源随CCU实例生命周期统一管理，不由C++析构释放。

按资源类型分为以下几类：

| 资源类型 | 单元分配 | 批量分配 | 通道引用 | 绑定Host侧预约资源 |
| --- | --- | --- | --- | --- |
| 标量寄存器 | [Variable](Variable.md) | [Array\<Variable\>](Array.md) | [GetResByChannel](GetResByChannel.md) | [`Variable(varHandle, index)`](Variable.md)、[`Array<Variable>(acqHandle, count)`](Array.md) |
| 地址寄存器 | [Address](Address.md) | — | — | — |
| 完成事件单元 | [Event](Event.md) | [Array\<Event\>](Array.md) | — | [`Event(acqHandle, index)`](Event.md)、[`Array<Event>(acqHandle, count)`](Array.md) |
| 片上CcuBuffer（4KB） | [CcuBuffer](CcuBuffer.md) | [Array\<CcuBuffer\>](Array.md) | — | — |
| 本端HBM复合地址 | [LocalAddr](LocalAddr.md) | — | — | — |
| 对端HBM复合地址 | [RemoteAddr](RemoteAddr.md) | — | — | — |

表中「通道引用」和「绑定Host侧预约资源」不走先虚后实：前者绑定channel建链时预留的槽位，后者绑定Host侧[HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md)、[HcommCcuEventAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md)预约的资源。二者都不申请新资源；参数不合法时抛出异常，不像默认构造那样恒成功。

[Variable](Variable.md)和[Address](Address.md)除资源分配外，还提供赋值与算术运算符；这些运算符描述的是device端执行的操作，在硬件执行时操作对应的寄存器，而非host端立即计算。

## 接口列表

- [Variable](Variable.md)
- [Address](Address.md)
- [Event](Event.md)
- [CcuBuffer](CcuBuffer.md)
- [LocalAddr](LocalAddr.md)
- [RemoteAddr](RemoteAddr.md)
- [Array](Array.md)
- [GetResByChannel](GetResByChannel.md)
