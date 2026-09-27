# 简介

CCU资源管理接口用于查询和描述CCU资源诉求、按需创建CCU实例、查询实例资源占用、在实例上预约Variable/Event资源，以及销毁调用方持有的实例。

## 创建并销毁CCU实例

按需创建CCU实例的典型流程如下：

1. 调用[HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md)为目标IO Die创建资源描述符。
2. 调用[HcommCcuKernelQueryResReq](HcommCcuKernelQueryResReq.md)统计Kernel资源诉求，或调用[HcommCcuInsResDescSetNum](HcommCcuInsResDescSetNum.md)直接设置资源数量。
3. 调用[HcommCcuInsCreate](HcommCcuInsCreate.md)创建CCU实例。
4. 可调用[HcommCcuInsQueryResDesc](HcommCcuInsQueryResDesc.md)查询实例实际占用的资源数量。
5. 不再使用实例时，调用[HcommCcuInsDestroy](HcommCcuInsDestroy.md)销毁；若通过[HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md)成功绑定到通信域，则由通信域管理实例生命周期。

## 预约Variable/Event资源

需要跨Kernel共享同一组标量寄存器/完成事件单元，或与CCU之外的目标模块共用时，可在实例上预约资源：

1. 调用[HcommCcuVariableAlloc](HcommCcuVariableAlloc.md)或[HcommCcuEventAlloc](HcommCcuEventAlloc.md)，从实例资源池中预约一段物理连续的Variable（标量寄存器）或Event（完成事件单元），得到预约句柄。预约完成后有两种用法，彼此独立，可只用其一，也可都用：
   - 在Host侧调用[HcommCcuVariableGetAddr](HcommCcuVariableGetAddr.md)或[HcommCcuEventGetAddr](HcommCcuEventGetAddr.md)，取出段内每个资源映射后的虚拟地址，须按原值交给目标模块。
   - 把预约句柄通过`kernelArgs`传入Kernel，在Kernel内用[Variable](../../data_plane_api/ccu/resource_allocation_operation/Variable.md)、[Event](../../data_plane_api/ccu/resource_allocation_operation/Event.md)或[Array](../../data_plane_api/ccu/resource_allocation_operation/Array.md)的预约句柄构造形式绑定到同一段资源。
2. 预约资源没有单独的释放接口，随[HcommCcuInsDestroy](HcommCcuInsDestroy.md)销毁实例时预约句柄一并失效。
