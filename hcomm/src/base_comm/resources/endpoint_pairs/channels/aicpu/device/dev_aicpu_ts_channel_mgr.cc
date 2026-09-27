/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dev_aicpu_ts_channel_mgr.h"
#include "dev_aicpu_ts_roce_channel.h"
#include "dev_aicpu_ts_hccs_channel.h"
#include "log.h"

DevAicpuTsChannelMgr& DevAicpuTsChannelMgr::Instance()
{
    static DevAicpuTsChannelMgr inst;
    return inst;
}

DevAicpuTsChannel* DevAicpuTsChannelMgr::GetOrCreateAicpuTsChannel(hcomm::HcommChannelKind kind)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = channelMap_.find(kind);
    if (it != channelMap_.end()) {
        return it->second.get();
    }

    std::unique_ptr<DevAicpuTsChannel> channel;
    switch (kind) {
        case hcomm::HcommChannelKind::AICPU_TS_ROCE:
            channel = std::make_unique<DevAicpuTsRoceChannel>();
            break;
        case hcomm::HcommChannelKind::AICPU_TS_HCCS:
            channel = std::make_unique<hccl::DevAicpuTsHccsChannel>();
            break;
        default:
            HCCL_ERROR(
                "[DevAicpuTsChannelMgr][GetOrCreateAicpuTsChannel] unsupported kind[%u]", static_cast<uint32_t>(kind));
            return nullptr;
    }

    if (channel == nullptr) {
        HCCL_ERROR(
            "[DevAicpuTsChannelMgr][GetOrCreateAicpuTsChannel] alloc failed for kind[%u]", static_cast<uint32_t>(kind));
        return nullptr;
    }

    DevAicpuTsChannel* ptr = channel.get();
    channelMap_.emplace(kind, std::move(channel));
    HCCL_DEBUG(
        "[DevAicpuTsChannelMgr][GetOrCreateAicpuTsChannel] created channel for kind[%u]", static_cast<uint32_t>(kind));
    return ptr;
}

bool DevAicpuTsChannelMgr::DestroyChannel(ChannelHandle handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : channelMap_) {
        if (entry.second && entry.second->Destroy(handle)) {
            HCCL_DEBUG("[DevAicpuTsChannelMgr][DestroyChannel] destroyed handle[0x%llx]", handle);
            return true;
        }
    }
    HCCL_WARNING("[DevAicpuTsChannelMgr][DestroyChannel] handle[0x%llx] not found", handle);
    return false;
}
