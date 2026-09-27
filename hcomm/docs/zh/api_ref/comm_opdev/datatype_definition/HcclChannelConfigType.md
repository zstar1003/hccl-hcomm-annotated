# HcclChannelConfigType

## 功能说明

Channel配置属性类型枚举，用于[HcclChannelConfigSetInt](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetInt.md)和[HcclChannelConfigSetStr](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetStr.md)接口指定待设置的属性类型。

## 类型定义

```c
typedef enum {
    HCCL_CHANNEL_CONFIG_TYPE_INVALID = -1,
    HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
    HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG = 1,
} HcclChannelConfigType;
```

## 枚举值说明

| 枚举值 | 值 | 描述 | 适用接口 |
| --- | --- | --- | --- |
| HCCL_CHANNEL_CONFIG_TYPE_INVALID | -1 | 无效值，用于初始化校验。 | - |
| HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE | 0 | 是否启用共享队列模式（bool/int，默认0=false）。<br>设为true时，创建的多个Channel共享一个Jetty。<br>仅支持AIV引擎的UB网络语义协议（UB_CTP/UB_RTP）。 | [HcclChannelConfigSetInt](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetInt.md) |
| HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG | 1 | 共享队列的tag标识（string）。<br>仅IS_SHARED_QUEUE=true时有效，指定创建Channel的tag。<br>重复调用时，使用相同tag创建出的Channel共享一个Jetty。 | [HcclChannelConfigSetStr](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetStr.md) |
