# CcuEventHandle

## 功能说明

CCU Event相关的句柄类型（`uint64_t`）。同一类型在不同场景下有两种用法，不能混用：

| 用途 | 来源 | 用法 |
| --- | --- | --- |
| 预约句柄 | Host侧[HcommCcuEventAlloc](../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md) | 传给[HcommCcuEventGetAddr](../control_plane_api/ccu_resource_mgmt/HcommCcuEventGetAddr.md)，或传给Kernel内[Event](../data_plane_api/ccu/resource_allocation_operation/Event.md) / [Array\<Event\>](../data_plane_api/ccu/resource_allocation_operation/Array.md)的预约句柄构造形式 |
| Kernel内虚拟句柄 | Kernel注册阶段`Event`默认构造等资源创建接口 | 用于同一Kernel的数据面同步接口（如`EventRecord`、`EventWait`）；Kernel指令翻译完成后失效 |

不能把预约句柄当作`Event::handle`使用，也不能把`Event::handle`传给`HcommCcuEventGetAddr`。

## 定义原型

```c
typedef uint64_t CcuEventHandle;
```
