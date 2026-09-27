# HcclChannelConfig

## 功能说明

Channel配置对象的不透明句柄，用于[HcclChannelAcquireWithConfig](../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquireWithConfig.md)接口创建通信通道时传入共享Jetty等高级配置。

## 类型定义

```c
typedef void *HcclChannelConfig;
```

## 使用说明

- 通过[HcclChannelConfigCreate](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigCreate.md)创建。
- 通过[HcclChannelConfigSetInt](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetInt.md)和[HcclChannelConfigSetStr](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetStr.md)设置属性。
- 使用完毕后通过[HcclChannelConfigDestroy](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigDestroy.md)销毁。
