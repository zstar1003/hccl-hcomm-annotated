/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef P2P_MGMT_PUB_H
#define P2P_MGMT_PUB_H

#include <map>
#include <mutex>
#include <vector>
#include <atomic>
#include "hccl/hccl_types.h"

namespace hccl {
enum class P2PStatus { P2P_STATUS_DISABLED = 0, P2P_STATUS_ENABLING, P2P_STATUS_ENABLED };

using P2PConnectionInfo = struct P2PConnectionInfoDef {
    uint32_t reference = 0;
    P2PStatus status = P2PStatus::P2P_STATUS_DISABLED;
};

class P2PMgmt;
class P2PMgmtPub {
public:
    static HcclResult EnableP2P(std::vector<uint32_t> remoteDevices);
    static HcclResult DisableP2P(std::vector<uint32_t> remoteDevices);
    static HcclResult WaitP2PEnabled(
        std::vector<uint32_t> remoteDevices, std::function<bool()> needStop = []() {
            return false;
        });
};
} // namespace hccl

#endif // P2P_MGMT_H
