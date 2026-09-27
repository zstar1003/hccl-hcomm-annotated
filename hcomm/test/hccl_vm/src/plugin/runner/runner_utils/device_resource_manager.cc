/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_resource_manager.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <queue>

using namespace std;

void DeviceResourceManager::Init(uint32_t rankSize)
{
    if (deviceResData_.v1Res == nullptr) {
        deviceResData_.v1Res = std::make_unique<DeviceResourceV1>(rankSize);
    }
    deviceResData_.v1Res->Reset(rankSize);
}

void DeviceResourceManager::InitRankRes(uint32_t rankId, uint32_t streamSize)
{
    deviceResData_.v1Res->streamSize_ = streamSize;
    deviceResData_.v1Res->notifyResources_[rankId].resize(streamSize);

    for (auto& streamNotify : deviceResData_.v1Res->notifyResources_[rankId]) {
        memset(&streamNotify, 0, sizeof(streamNotify));
    }
}

void DeviceResourceManager::NotifyRecord(uint32_t rankId, uint32_t streamId, uint32_t notifyId, uint32_t notifyCnt)
{
    deviceResData_.v1Res->notifyResources_[rankId][streamId].notifyCnts_[notifyId] += notifyCnt;
}

bool DeviceResourceManager::NotifyWait(uint32_t rankId, uint32_t streamId, uint32_t notifyId, uint32_t notifyCnt)
{
    deviceResData_.v1Res->notifyResources_[rankId][streamId].notifyCnts_[notifyId] -= notifyCnt;
    return deviceResData_.v1Res->notifyResources_[rankId][streamId].notifyCnts_[notifyId] >= 0;
}

bool DeviceResourceManager::CheckNotifyWait(uint32_t rankId, uint32_t streamId, uint32_t notifyId, uint32_t notifyCnt)
{
    auto notifyRemain = deviceResData_.v1Res->notifyResources_[rankId][streamId].notifyCnts_[notifyId] - notifyCnt;
    return deviceResData_.v1Res->notifyResources_[rankId][streamId].notifyCnts_[notifyId] >= notifyRemain >= 0;
}
