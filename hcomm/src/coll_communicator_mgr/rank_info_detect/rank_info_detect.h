/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RANK_INFO_DETECT_H
#define HCCL_RANK_INFO_DETECT_H

#include "types.h"
#include "socket.h"
#include "ip_address.h"
#include "rank_table_info.h"
#include "root_handle_v2.h"
#include "orion_adapter_hccp.h"
#include "rank_info_detect_service.h"
#include "rank_info_detect_client.h"
#include "socket_handle_manager.h"
#include "universal_concurrent_map.h"
#include "adapter_error_manager_pub.h"

namespace Hccl {

const u32 RANKINFO_DETECT_SERVER_STATUS_IDLE = 0;
const u32 RANKINFO_DETECT_SERVER_STATUS_RUNING = 1;
const u32 RANKINFO_DETECT_SERVER_STATUS_ERROR = 2;

class RankInfoDetect {
public:
    RankInfoDetect();
    ~RankInfoDetect() = default;

    void SetupServer(HcclRootHandleV2& rootHandle);
    void SetupAgent(u32 rankSize, u32 rankId, const HcclRootHandleV2& rootHandle);
    void GetRankTable(RankTableInfo& ranktable) const;
    void WaitComplete(u32 listenPort, u32 listenStatus) const;

private:
    s32 devLogicId_{0};
    s32 userDevId_{0};
    u32 devPhyId_{0};
    RankTableInfo rankTable_{};
    IpAddress hostIp_{};
    u32 hostPort_{HCCL_INVALID_PORT};
    vector<IpAddress> hostSocketWlist_{};
    vector<RaSocketWhitelist> wlistInfo_{};
    std::string identifier_{};
    std::shared_ptr<RankInfoDetectClient> rankInfoDetectClient;
    void SetupRankInfoDetectService(
        ErrContextPub errorContext, shared_ptr<Socket> serverSocket, s32 devLogicId, u32 devPhyId,
        std::string identifier, vector<RaSocketWhitelist> wlistInfo);
    std::shared_ptr<Socket> ServerInit();
    std::shared_ptr<Socket> ClientInit(const HcclRootHandleV2& rootHandle);
    HcclResult GetHandleAndAddHostSocketWhitelist();
    HcclResult AddHostSocketWhitelist(SocketHandle& socketHandle, const vector<IpAddress>& hostSocketWlist);
    u32 GetHostListenPort();
    void GetRootHandle(HcclRootHandleV2& rootHandle);
    SocketHandle GetHostSocketHandle();

    static UniversalConcurrentMap<u32, u32> g_detectServerStatus_;
};

} // namespace Hccl

#endif // HCCL_VIRT_TOPO_DETECT_H
