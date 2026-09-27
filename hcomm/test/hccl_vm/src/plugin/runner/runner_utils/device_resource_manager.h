/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_DEVICE_RESOURCE_MANAGER_H
#define HCCL_SIM_DEVICE_RESOURCE_MANAGER_H

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "device_resource.h"

struct DeviceResData {
    std::unique_ptr<DeviceResourceV1> v1Res;
};

class DeviceResourceManager {
public:
    DeviceResourceManager(const DeviceResourceManager&) = delete;
    DeviceResourceManager& operator=(const DeviceResourceManager&) = delete;

    static DeviceResourceManager& GetInstance()
    {
        static DeviceResourceManager instance;
        return instance;
    }

    void Init(uint32_t rankSize);
    void InitRankRes(uint32_t rankId, uint32_t streamSize);
    void NotifyRecord(uint32_t rankId, uint32_t streamId, uint32_t notifyId, uint32_t notifyCnt);
    bool NotifyWait(uint32_t rankId, uint32_t streamId, uint32_t notifyId, uint32_t notifyCnt);
    bool CheckNotifyWait(uint32_t rankId, uint32_t streamId, uint32_t notifyId, uint32_t notifyCnt);

private:
    DeviceResourceManager() = default;
    ~DeviceResourceManager() = default;

private:
    DeviceResData deviceResData_{}; // device资源数据
};

#endif // HCCL_SIM_DEVICE_RESOURCE_MANAGER_H
