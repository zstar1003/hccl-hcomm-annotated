# CcuRemoteAddrHandle

## 功能说明

CCU RemoteAddr（对端地址）的句柄类型，用于标识CCU kernel内通过[RemoteAddr](../data_plane_api/ccu/resource_allocation_operation/RemoteAddr.md)资源创建的对端片上内存地址资源。该句柄在CCU数据面跨rank接口（如Read、ReadReduce、Write、WriteReduce等）中作为对端地址参数使用。

## 定义原型

```c
typedef uint64_t CcuRemoteAddrHandle;
```
