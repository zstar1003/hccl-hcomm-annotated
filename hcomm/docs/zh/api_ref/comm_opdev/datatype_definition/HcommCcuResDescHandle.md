# HcommCcuResDescHandle

## 功能说明

CCU资源描述符句柄类型（`uint64_t`）。资源描述符用于描述CCU实例所需的各类资源（loop、buffer、variable、address、event、thread、instruction）的数量规格，不包含实际资源；一个资源描述符对应一个IO Die。

由Host侧[HcommCcuInsResDescCreate](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescCreate.md)创建，创建后各资源类型的数量初始化为0，可通过[HcommCcuInsResDescSetNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescSetNum.md)逐类型设置，通过[HcommCcuInsResDescQueryNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescQueryNum.md)查询。设置完成后传给[HcommCcuInsCreate](../control_plane_api/ccu_resource_mgmt/HcommCcuInsCreate.md)创建CCU实例。

此外还可作为查询结果的载体使用：

- [HcommCcuInsQueryResDesc](../control_plane_api/ccu_resource_mgmt/HcommCcuInsQueryResDesc.md)：查询CCU实例实际占用的资源数量并写入描述符。
- [HcommCcuQueryRemainResDesc](../control_plane_api/ccu_resource_mgmt/HcommCcuQueryRemainResDesc.md)：查询对应IO Die上的最大连续剩余资源数量并写入描述符。
- [HcommCcuKernelQueryResReq](../control_plane_api/ccu_resource_mgmt/HcommCcuKernelQueryResReq.md)：查询CCU Kernel的资源诉求并写入描述符。

不再使用时须通过[HcommCcuInsResDescDestroy](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescDestroy.md)销毁，销毁后句柄失效。

## 定义原型

```c
typedef uint64_t HcommCcuResDescHandle;
```
