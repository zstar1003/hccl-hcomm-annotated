# HcclDedicatedThreadType

## 功能说明

专用通信线程使用类型枚举，用于[HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md)接口指定专用线程的使用场景。

## 定义原型

```c
typedef enum {
    HCCL_DED_THREAD_TYPE_INVALID = -1,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH = 0,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE = 1,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE = 2,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH = 3,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE = 4,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE = 5
} HcclDedicatedThreadType;
```

## 成员说明

| 成员 | 描述 |
| --- | --- |
| HCCL_DED_THREAD_TYPE_INVALID | 无效的专用线程类型。 |
| HCCL_DED_THREAD_TYPE_AICPU_LAUNCH | 单算子用于AI CPU任务下发的专用线程。 |
| HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE | 图模式下用于AI CPU任务下发的专用线程。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE | 单算子模式下用于AI CPU保序任务下发的专用线程。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH | aclgraph模式下用于AI CPU保序任务下发的专用线程。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE | 图模式下用于AI CPU保序任务下发的专用线程，需预先通过HcomSetAttachedStream设置附属流。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE | 设备侧用于AI CPU保序任务下发的专用线程。 |

## 约束说明

1. HCCL_DED_THREAD_TYPE_INVALID为无效值，在[HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md)接口中传入将返回失败。

2. HCCL_DED_THREAD_TYPE_AICPU_LAUNCH与HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE为非保序类型，线程在通信域内按useType缓存。

3. HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_*系列为保序类型，线程由进程粒度的OrderLaunchThreadMgr管理，不与单个通信域绑定。各保序类型的行为差异详见[HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md)约束说明。

