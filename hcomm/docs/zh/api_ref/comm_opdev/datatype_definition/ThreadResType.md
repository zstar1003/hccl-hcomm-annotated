# ThreadResType

## 功能说明

支持获取的底层资源类型。

## 定义原型

```c
typedef enum {
    THREAD_RES_TYPE_INVALID = -1,
    THREAD_RES_TYPE_STREAM = 0,   // 获取的资源类型为stream
} ThreadResType;
```

## 字段说明

| 字段 | 值 | 说明 |
| --- | --- | --- |
| THREAD_RES_TYPE_INVALID | -1 | 无效资源类型。 |
| THREAD_RES_TYPE_STREAM | 0 | Stream资源。对应的资源类型为[ThreadResTypeStream](ThreadResTypeStream.md)，可通过[HcclThreadResGetInfo](../control_plane_api/comms_domain_resource_mgmt/HcclThreadResGetInfo.md)接口获取。 |
