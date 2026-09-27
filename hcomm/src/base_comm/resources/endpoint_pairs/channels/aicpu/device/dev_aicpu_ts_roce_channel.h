/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_AICPU_TS_ROCE_CHANNEL_H
#define DEV_AICPU_TS_ROCE_CHANNEL_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "dev_aicpu_ts_channel.h"
#include "hccl_dispatcher_ctx.h"
#include "transport_pub.h"

struct HcommRoceChannelRes;

class DevAicpuTsRoceChannel : public DevAicpuTsChannel {
public:
    DevAicpuTsRoceChannel() = default;
    ~DevAicpuTsRoceChannel() override;

    HcclResult
    Create(const void* blob, u64 blobBytes, const HcommDeviceInfo& deviceInfo, ChannelHandle& outHandle) override;
    bool Destroy(ChannelHandle handle) override;

private:
    struct RoceSlot {
        DispatcherCtxPtr ctx{nullptr};
        std::shared_ptr<hccl::Transport> link;
        char commId[128]{};
    };

    std::unordered_map<ChannelHandle, RoceSlot> slots_;
    std::mutex mutex_;
    static u64 commSeq_;
};

#endif // DEV_AICPU_TS_ROCE_CHANNEL_H
