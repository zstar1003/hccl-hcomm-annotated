/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_AICPU_TS_HCCS_CHANNEL_H
#define DEV_AICPU_TS_HCCS_CHANNEL_H

#include "common.h"
#include "channel_param.h"
#include "hccl_dispatcher_ctx.h"
#include "dev_aicpu_ts_channel.h"

namespace hccl {
class DevAicpuTsHccsChannel : public DevAicpuTsChannel {
public:
    DevAicpuTsHccsChannel() = default;
    ~DevAicpuTsHccsChannel() override;

    HcclResult
    Create(const void* blob, u64 blobBytes, const HcommDeviceInfo& deviceInfo, ChannelHandle& outHandle) override;
    bool Destroy(ChannelHandle handle) override;

private:
    HcclResult SetTransportMachinePara(hccl::MachinePara& machinePara, const HcclChannelHccsRes& channelHccsRes);

    struct HccsSlot {
        DispatcherCtxPtr dispatcherCtx{nullptr};
        std::shared_ptr<hccl::Transport> transport;
        std::string tag{""};
    };

    std::unordered_map<ChannelHandle, HccsSlot> slots_;
    std::mutex mutex_;
};
} // namespace hccl

#endif // DEV_AICPU_TS_HCCS_CHANNEL_H
