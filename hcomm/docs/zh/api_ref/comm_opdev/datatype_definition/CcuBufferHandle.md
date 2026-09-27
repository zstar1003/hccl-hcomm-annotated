# CcuBufferHandle

## 功能说明

CCU Buffer（MS Buffer）的句柄类型，用于标识CCU kernel内通过[CcuBuffer](../data_plane_api/ccu/resource_allocation_operation/CcuBuffer.md)资源创建的MS Buffer资源。MS Buffer是CCU die内的片上高速暂存区，用于在片上内存与对端之间中转数据。

## 定义原型

```c
typedef uint64_t CcuBufferHandle;
```
