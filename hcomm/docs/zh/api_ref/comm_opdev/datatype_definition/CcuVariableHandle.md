# CcuVariableHandle

## 功能说明

CCU Variable相关的句柄类型（`uint64_t`）。同一类型在不同场景下有两种用法，不能混用：

| 用途 | 来源 | 用法 |
| --- | --- | --- |
| 预约句柄 | Host侧[HcommCcuVariableAlloc](../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md) | 传给[HcommCcuVariableGetAddr](../control_plane_api/ccu_resource_mgmt/HcommCcuVariableGetAddr.md)，或传给Kernel内[Variable](../data_plane_api/ccu/resource_allocation_operation/Variable.md) / [Array\<Variable\>](../data_plane_api/ccu/resource_allocation_operation/Array.md)的预约句柄构造形式 |
| Kernel内虚拟句柄 | Kernel注册阶段`Variable`默认构造等资源创建接口 | 用于同一Kernel的数据面接口；Kernel指令翻译完成后失效 |

不能把预约句柄当作`Variable::handle`使用，也不能把`Variable::handle`传给`HcommCcuVariableGetAddr`。

## 定义原型

```c
typedef uint64_t CcuVariableHandle;
```
