# HcommEndpointFeatureType

## 功能说明

底层特性枚举定义，用于[HcommEndpointCheckFeature](../control_plane_api/basic_resource_mgmt/HcommEndpointCheckFeature.md)接口中指定待查询的特性类型。

## 定义原型

```c
typedef enum {
    HCOMM_ENDPOINT_FEATURE_INVALID = -1,       /* 无效特性类型，暂不支持配置 */
    HCOMM_ENDPOINT_FEATURE_NDA = 0,            /* NPU Direct RDMA Async特性 */
} HcommEndpointFeatureType;
```
