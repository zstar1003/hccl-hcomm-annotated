/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_INFO_DETECT_SERVICE_H
#define HCCLV2_RANK_INFO_DETECT_SERVICE_H

#include "socket.h"
#include <unordered_map>
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "json_parser.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "socket_agent.h"

namespace Hccl {

class RankInfoDetectService {
public:
    RankInfoDetectService(
        u32 devPhyId, std::shared_ptr<Socket> serverSocket, std::string identifier, vector<RaSocketWhitelist> wlistInfo)
        : devPhyId_(devPhyId),
          serverSocket_(serverSocket),
          hostIp_(serverSocket_->GetLocalIp()),
          identifier_(identifier),
          wlistInfo_(wlistInfo)
    {}
    ~RankInfoDetectService();

    void Setup();

private:
    u32 devPhyId_{0};
    shared_ptr<Socket> serverSocket_{nullptr};
    std::unordered_map<std::string, std::shared_ptr<Socket>> connSockets_{};
    IpAddress hostIp_{};
    RankTableInfo rankTable_{};
    std::string failedAgentIdList_{};
    u32 currentStep_{0};
    std::string identifier_{};
    vector<RaSocketWhitelist> wlistInfo_{};

    void GetConnections();
    void GetRankTable();
    void BroadcastRankTable();
    void Disconnect();
    void TearDown();

    bool RecvRemoteAgentId(SocketAgent& connSocketAgent, std::string& agentId);
    bool RecvRemoteRankSize(SocketAgent& connSocketAgent, u32& rankSize);
    void RecvRankInfoMsg(SocketAgent& connSocketAgent, vector<char>& rankInfoMsg);
    void SortRankTable();
    void ParseRankTable(vector<char>& rankInfoMsg);

    // 异常流程处理方法
    void FailedConnectionAgentIdString(u32 rankSize);

    // 校验相关方法
    bool RecvAndVerifyRemoteAgentIdAndRankSize(
        std::shared_ptr<Socket> connSocket, u32& expectedSocketNum, u32& previousRankSize);
    bool VerifyRemoteRankSize(u32& previousRankSize, u32 remoteRankSize) const;

    // DFX相关方法
    void DisplayConnectingStatus(u32 totalSockets, u32 waitSockets);
    void DisplayConnectedRanks();
};

} // namespace Hccl
#endif // HCCLV2_RANK_INFO_DETECT_SERVICE_H
