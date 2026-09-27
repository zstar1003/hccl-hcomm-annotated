# CcuLocalAddrHandle

## 功能说明

CCU LocalAddr（本端地址）的句柄类型，用于标识CCU kernel内通过[LocalAddr](../data_plane_api/ccu/resource_allocation_operation/LocalAddr.md)资源创建的本端HBM地址资源。该句柄在CCU数据面接口（如LocalCopy、LocalReduce、Read、Write等）中作为本端地址参数使用。

## 定义原型

```c
typedef uint64_t CcuLocalAddrHandle;
```
