/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "p2p_mgmt.h"

namespace hccl {

HcclResult P2PMgmtPub::EnableP2P(std::vector<uint32_t> remoteDevices)
{
    return P2PMgmt::Instance().EnableP2P(remoteDevices);
}

HcclResult P2PMgmtPub::DisableP2P(std::vector<uint32_t> remoteDevices)
{
    return P2PMgmt::Instance().DisableP2P(remoteDevices);
}

HcclResult P2PMgmtPub::WaitP2PEnabled(std::vector<uint32_t> remoteDevices, std::function<bool()> needStop)
{
    return P2PMgmt::Instance().WaitP2PEnabled(remoteDevices, needStop);
}
} // namespace hccl
