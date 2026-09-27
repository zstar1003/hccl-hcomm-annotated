/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_DETECT_SCALABLE_H
#define TOPOINFO_DETECT_SCALABLE_H

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "topoinfo_detect.h"

namespace hccl {
// 多root（scalable）建链的拓扑探测入口。
// 组内收集 → root间全互联合并 → 组内广播 的全流程。
class TopoInfoDetectScalable : public TopoInfoDetect {
public:
    explicit TopoInfoDetectScalable() = default;
    ~TopoInfoDetectScalable() override = default;

    static u32 RootIdFromRank(u32 nRanks, u32 nRoot, u32 rank);
    static u32 FirstRankFromRoot(u32 nRanks, u32 nRoot, u32 rootIndex);
    static u32 NRankFromRoot(u32 root, u32 nRanks, u32 nRoots);
    static bool RankHasRoot(u32 rank, u32 nRanks, u32 nRoots);

    HcclResult SetupScalableRoot(HcclScalableRootHandle& outHandle);
    HcclResult SetScalableServerInfo(const ScalableServerInfo& info);
    // 拿到全部 root 信息后，按与 connect/accept 一致的 tag 下发 root 间 mesh 白名单
    HcclResult AddMeshSocketWhiteList(const ScalableServerInfo& info);

private:
    HcclResult SetupMeshListen(u32& meshPort, const std::vector<HcclSocketPortRange>& portRanges);
    HcclResult DeleteMeshSocketWhiteList(const ScalableServerInfo& info) const;
    void SetupTopoExchangeServerScalable(
        s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort, std::vector<HcclIpAddress> whitelist,
        HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket, bool hasPortRanges);
    HcclResult WaitScalableInfoReady() const;
    void ReleaseMeshListenSocket(s32 deviceLogicID, bool hasPortRanges);
    HcclResult BuildMeshSocketWlistInfos(
        const std::vector<std::string>& tags, const std::vector<HcclIpAddress>& whitelist,
        std::vector<SocketWlistInfo>& wlistInfosVec) const;
    HcclResult SetupScalableExchangeServer(
        HcclIpAddress& hostIP, u32 hostPort, const std::vector<HcclIpAddress>& whitelist, HcclNetDevCtx netDevCtx,
        std::shared_ptr<HcclSocket> listenSocket);

    ScalableServerInfo scalableInfo_{};
    std::atomic<bool> scalableInfoReady_{false};
    std::shared_ptr<HcclSocket> meshListenSocket_{nullptr};
    std::vector<HcclIpAddress> meshWhitelist_; // SetupScalableRoot 读取，供 init 侧按 peer tag 下发 mesh 白名单
    mutable std::mutex scalableInfoMutex_;
};
} // namespace hccl
#endif /* TOPOINFO_DETECT_SCALABLE_H */
