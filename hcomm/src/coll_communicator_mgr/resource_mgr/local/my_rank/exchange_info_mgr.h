/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXCHANGE_INFO_MGR_H
#define EXCHANGE_INFO_MGR_H

#include <string>
#include <vector>
#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hccl/hccl_res.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "common.h"
#include "coll_comm_config_consistency.h"
#include "socket/socket.h"

namespace hccl {
class ExchangeInfoMgr {
public:
    ExchangeInfoMgr();
    ~ExchangeInfoMgr();

    HcclResult BatchExchangeAndCheckConsistency(
        const HcclChannelDesc* channelDescs, const std::vector<HcommChannelDesc>& hcommDescs, uint32_t channelNum,
        const std::vector<std::pair<u32, u32>>& newChannels, CollCommConfigConsistency& collCommConfigConsistency,
        CommEngine engine) const;
    HcclResult CheckHcommInfo(
        const HcclChannelDesc* channelDescs, const std::vector<HcommChannelDesc>& hcommDescs,
        const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
        const std::vector<HcommSocketRole>& roles, const std::vector<std::pair<u32, u32>>& newChannels,
        bool isAsync) const;
    HcclResult ExchangeUserInfo(
        const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
        const std::vector<HcommSocketRole>& roles, CollCommConfigConsistency& collCommConfigConsistency) const;
    HcclResult ExchangeUserInfoAsync(
        const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
        const std::vector<HcommSocketRole>& roles, CollCommConfigConsistency& collCommConfigConsistency) const;
    HcclResult BatchExchangeFixedData(
        const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
        const std::vector<HcommSocketRole>& roles, const u8* sendData, u32 sendLen, u8* recvData, u32 recvLen,
        bool isAsync) const;
    HcclResult
    WaitAllAsyncComplete(const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks) const;
    HcclResult WaitActiveAsyncComplete(
        const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
        const std::vector<HcommSocketRole>& roles, const std::vector<u32>& remoteExchangeInfoLens,
        u32 localExchangeInfoLen, bool isFirstPass) const;

private:
    HcclResult ExchangeAsyncDataPhase(
        const std::vector<Hccl::Socket*>& sockets, const std::vector<HcommSocketRole>& roles,
        const std::vector<u32>& remoteRanks, std::vector<std::vector<u8>>& remoteUserDatas,
        const std::vector<u32>& remoteExchangeInfoLens, u32 localExchangeInfoLen,
        CollCommConfigConsistency& collCommConfigConsistency, bool isServerRecv) const;
};
} // namespace hccl

#endif // EXCHANGE_INFO_MGR_H
