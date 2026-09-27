/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_CHANNEL_PROCESS_H__
#define __AICPU_CHANNEL_PROCESS_H__

#include <memory>
#include <mutex>
#include <unordered_map>

#include "common.h"
#include "channel_param.h"
#include "base_transport_lite_impl.h"

class AicpuChannelProcess {
public:
    ~AicpuChannelProcess() = default;
    static HcclResult ParsePackData(std::vector<char>& data, ChannelHandle& handle);
    static HcclResult InitUrmaChannel(HcclChannelUrmaRes* commParam);
    static HcclResult AicpuChannelInit(HcclChannelUrmaRes* commParam);
    static HcclResult AicpuChannelDestroy(HcclChannelUrmaRes* commParam);
    static HcclResult InitHcommChannelRes(HcommChannelRes* commParam);

private:
    static std::mutex mutex_;
    static std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>> transportMap_;
};
#endif // __AICPU_CHANNEL_PROCESS_H__
