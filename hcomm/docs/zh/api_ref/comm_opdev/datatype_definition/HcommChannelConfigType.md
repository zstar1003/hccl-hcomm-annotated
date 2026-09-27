# HcommChannelConfigType

## 功能说明

Channel配置属性类型枚举，用于[HcommChannelConfigSetInt](../control_plane_api/basic_resource_mgmt/HcommChannelConfigSetInt.md)接口指定待设置的属性类型。

## 类型定义

```c
typedef enum {
    HCOMM_CHANNEL_CONFIG_TYPE_INVALID = -1,
    HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
} HcommChannelConfigType;
```

## 枚举值说明

| 枚举值 | 值 | 描述 | 适用接口 |
| --- | --- | --- | --- |
| HCOMM_CHANNEL_CONFIG_TYPE_INVALID | -1 | 无效值，用于初始化校验。 | - |
| HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE | 0 | 是否启用共享队列模式（bool/int，默认0=false）。<br>设为true时，使用[HcommChannelCreateWithConfig](../control_plane_api/basic_resource_mgmt/HcommChannelCreateWithConfig.md)创建的多个Channel共享一个Jetty。<br>仅支持AIV引擎的UB网络语义协议（UB_CTP/UB_RTP）。 | [HcommChannelConfigSetInt](../control_plane_api/basic_resource_mgmt/HcommChannelConfigSetInt.md) |
