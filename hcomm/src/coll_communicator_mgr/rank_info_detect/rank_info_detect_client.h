/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_INFO_DETECT_CLIENT_H
#define HCCLV2_RANK_INFO_DETECT_CLIENT_H

#include <cstddef>
#include <functional>
#include <vector>
#include "socket.h"
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "json_parser.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "socket_agent.h"
#include "root_handle_v2.h"
#include "ip_address.h"

namespace Hccl {

#ifdef OPEN_HCCL_TEST
const u32 WAIT_ERROR_BROADCAST_TIME = 1;
#else
const u32 WAIT_ERROR_BROADCAST_TIME = 20;
#endif

class RankInfoDetectClient {
public:
    RankInfoDetectClient(u32 devPhyId, u32 rankSize, u32 rankId, const std::shared_ptr<Socket>& clientSocket)
        : devPhyId_(devPhyId),
          rankSize_(rankSize),
          rankId_(rankId),
          clientSocket_(clientSocket),
          socketAgent_(clientSocket.get())
    {}
    ~RankInfoDetectClient();

    void Setup(RankTableInfo& rankTable);

private:
    u32 devPhyId_{0};
    u32 rankSize_{0};
    u32 rankId_{0};
    std::shared_ptr<Socket> clientSocket_{nullptr};
    u32 currentStep_{0};
    RankTableInfo rankTable_{};
    SocketAgent socketAgent_;
    std::mutex hostSocketLock_;
    std::shared_ptr<Socket> hostSocket_ = nullptr;
    bool hostSocketRegistered_ = false; // 标记 hostSocket_ 是否已登记到 SocketManager::GetServerSocketMap()

    void SetupHostListenPort(u32 devLogicId, u32 devPhyId, const IpAddress& hostIp, uint32_t& hostPort);
    void SocketTearDown(u32 devPhyId);
    void Connect();
    void CheckStatus();
    void SendAgentIdAndRankSize();
    void SendLocalRankTable(const RankTableInfo& localRankTable);
    void ConstructRankTable(RankTableInfo& localRankTable);
    void VerifyRankTable();
    void RecvRankTable();
    void RecvRankTableMsg(vector<char>& rankInfoMsg);
    void ParseRankTable(vector<char>& rankInfoMsg);
    void GetLocalRankTableJson(const nlohmann::json& parseJson, nlohmann::json& localRankTableJson);
    void GetLocalDevInfoJson(const nlohmann::json& parseJson, nlohmann::json& localDevInfoJson);
    void SelectLocalHostBackupAddr(nlohmann::json& localDevInfoJson);
    void SelectAvailableHostAddr(nlohmann::json& addrJson);
    void UpdateSelectedHostAddr(
        nlohmann::json& addrJson, const std::vector<IpAddress>& candidates, std::size_t selectedIndex) const;
    s32 ProbeHostRoceAddr(const IpAddress& hostAddr) const;
    void ConstructSingleRank(RankTableInfo& localRankTable);
    HcclResult GetLocalTlsStatus(TlsStatus& tlsStatus) const;
    HcclResult GetLocalHostDpuTlsStatus(TlsStatus& tlsStatus) const;
    HcclResult VerifyTlsConsistency() const;
    HcclResult VerifyHostDpuTlsConsistency() const;
    HcclResult VerifyTlsConsistencyByTlsType(
        const std::string& tlsType, const std::function<TlsStatus(const NewRankInfo&)>& getTlsStatus) const;
    void GenerateTlsStatusStr(std::string& tlsStatusStr, const std::vector<u32>& tlsStatusRanks) const;
    void ReportTlsConfigurationError(
        const std::string& tlsType, const std::string& tlsInconsistentTlsType, const std::string& tlsEnableRankStr,
        const std::string& tlsDisableRankStr, const std::string& tlsUnknownRankStr) const;
    void TearDown();
    void HostListenPortDetect(NewRankInfo& rankInfo);
};

} // namespace Hccl
#endif // HCCLV2_RANK_INFO_DETECT_CLIENT_H
