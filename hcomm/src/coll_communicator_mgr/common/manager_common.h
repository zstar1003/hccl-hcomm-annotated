/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MANAGER_COMMON_H
#define MANAGER_COMMON_H

#include "hccl_types.h"
#include "transport_manager.h"
#include "common.h"
#include <string>
#include <vector>

namespace hccl {

struct ManagerCallbacks {
    // 获取Aicpu通信域状态
    std::function<bool()> getAicpuCommState;
    // 设置Aicpu通信域状态
    std::function<void(bool)> setAicpuCommState;
    // Aicpu通信域初始化
    std::function<HcclResult()> kernelLaunchAicpuCommInit;
    // report kernel
    std::function<HcclResult(uint64_t, std::string)> reportProfilingKernel;
};

struct ChannelManagerCallbacks {
    // channel建链
    std::function<HcclResult(const std::string&, OpCommTransport&, bool, const HcclMemHandle*, uint32_t)>
        indOpTransportAlloc;
    std::function<std::vector<RankInfo>()> getRankLists;
};

} // namespace hccl

#endif // MANAGER_COMMON_H
