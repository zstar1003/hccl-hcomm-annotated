# HcommChannelConfig

## 功能说明

Channel配置对象的不透明句柄，用于[HcommChannelCreateWithConfig](../control_plane_api/basic_resource_mgmt/HcommChannelCreateWithConfig.md)接口创建通信通道时传入共享Jetty等高级配置。

## 类型定义

```c
typedef void *HcommChannelConfig;
```

## 使用说明

- 通过[HcommChannelConfigCreate](../control_plane_api/basic_resource_mgmt/HcommChannelConfigCreate.md)创建。
- 通过[HcommChannelConfigSetInt](../control_plane_api/basic_resource_mgmt/HcommChannelConfigSetInt.md)设置属性。
- 使用完毕后通过[HcommChannelConfigDestroy](../control_plane_api/basic_resource_mgmt/HcommChannelConfigDestroy.md)销毁。
