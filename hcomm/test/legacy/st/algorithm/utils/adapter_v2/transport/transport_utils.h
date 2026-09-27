/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ADAPTERV2_TRANSPORT_UTILS_H
#define HCCL_ADAPTERV2_TRANSPORT_UTILS_H

#include <map>
#include <vector>
#include "llt_common.h"
#include "ccu_transport.h"

using namespace checker;
namespace Hccl {

struct RemoteDieInfo {
    RankId dstRank;
    uint32_t remoteDieId;
};

using ChannelsPerDie = std::map<uint32_t, RemoteDieInfo>;

// rank <-> dieId <-> channelId <-> dieInfo
extern std::map<RankId, std::map<u32, ChannelsPerDie>> g_allRankChannelInfo;

// localRank <-> remoteRank <-> transports
// 绕路场景下，两个rank之间会有多个多条链路，此时需要根据localAddr和remoteAddr进行区分
extern std::map<RankId, std::map<RankId, std::vector<CcuTransport*>>> g_allRankTransports;

extern std::map<CcuTransport*, CcuTransport*> g_transportsPair;

} // namespace Hccl

#endif
