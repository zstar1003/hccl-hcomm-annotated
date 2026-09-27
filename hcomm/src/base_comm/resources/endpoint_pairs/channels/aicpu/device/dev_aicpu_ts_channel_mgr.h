/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_AICPU_TS_CHANNEL_MGR_H
#define DEV_AICPU_TS_CHANNEL_MGR_H

#include "dev_aicpu_ts_channel.h"
#include "../channel.h"
#include <mutex>
#include <unordered_map>
#include <memory>

class DevAicpuTsChannelMgr {
public:
    static DevAicpuTsChannelMgr& Instance();

    DevAicpuTsChannel* GetOrCreateAicpuTsChannel(hcomm::HcommChannelKind kind);
    bool DestroyChannel(ChannelHandle handle);

private:
    DevAicpuTsChannelMgr() = default;
    ~DevAicpuTsChannelMgr() = default;
    DevAicpuTsChannelMgr(const DevAicpuTsChannelMgr&) = delete;
    DevAicpuTsChannelMgr& operator=(const DevAicpuTsChannelMgr&) = delete;

    std::unordered_map<hcomm::HcommChannelKind, std::unique_ptr<DevAicpuTsChannel>> channelMap_;
    std::mutex mutex_;
};

#endif // DEV_AICPU_TS_CHANNEL_MGR_H
