/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_DEVICE_RESOURCE_V1_H
#define HCCL_SIM_DEVICE_RESOURCE_V1_H

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

constexpr uint32_t MAX_NOTIFY_NUM = 8192;

class DeviceResourceManager;

struct StreamNotify {
    uint32_t notifyCnts_[MAX_NOTIFY_NUM];
};

class DeviceResourceV1 {
public:
    DeviceResourceV1(uint32_t rankSize);
    ~DeviceResourceV1() = default;
    void Reset(uint32_t rankSize);

private:
    uint32_t rankSize_{0};
    uint32_t streamSize_{0};
    friend DeviceResourceManager; // 声明资源管理类为友元类
    // notify资源<rankId, streamId>
    std::vector<std::vector<StreamNotify>> notifyResources_;
};

#endif // HCCL_SIM_DEVICE_RESOURCE_V1_H
