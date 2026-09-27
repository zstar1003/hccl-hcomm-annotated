/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_resource.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <queue>

DeviceResourceV1::DeviceResourceV1(uint32_t rankSize)
{
    rankSize_ = rankSize;
    notifyResources_.resize(rankSize);
}

void DeviceResourceV1::Reset(uint32_t rankSize)
{
    rankSize_ = rankSize;
    notifyResources_.resize(rankSize);
    for (auto& rankRes : notifyResources_) {
        for (auto& streamNotify : rankRes) {
            memset(&streamNotify, 0, sizeof(streamNotify));
        }
    }
}
