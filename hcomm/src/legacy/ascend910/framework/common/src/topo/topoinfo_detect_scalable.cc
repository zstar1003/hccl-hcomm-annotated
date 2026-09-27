/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_detect_scalable.h"
#include <chrono>
#include <string>
#include "adapter_rts_common.h"
#include "hccl_socket.h"
#include "sal_pub.h"
#include "preempt_port_manager.h"

using namespace std;
namespace hccl {
u32 TopoInfoDetectScalable::RootIdFromRank(u32 nRanks, u32 nRoot, u32 rank)
{
    // rank 所属 root：前 rmr 个 root 各 rpr+1 个 rank，其后各 rpr 个
    const u32 rpr = nRanks / nRoot;
    const u32 rmr = nRanks % nRoot;
    const u32 rlim = rmr * (rpr + 1); // 前 rmr 个 root 的 rank 总数边界
    if (rank < rlim) {
        return rank / (rpr + 1);
    } else {
        return (rank - rlim) / rpr + rmr;
    }
}

u32 TopoInfoDetectScalable::FirstRankFromRoot(u32 nRanks, u32 nRoot, u32 rootIndex)
{
    const u32 rpr = nRanks / nRoot;
    const u32 rmr = nRanks % nRoot;
    const u32 head = (rootIndex < rmr) ? rootIndex : rmr; // rootIndex之前多带rank的root个数
    return rootIndex * rpr + head;
}

u32 TopoInfoDetectScalable::NRankFromRoot(u32 root, u32 nRanks, u32 nRoots)
{
    const u32 rpr = nRanks / nRoots;
    const u32 rmr = nRanks % nRoots;
    return rpr + (root < rmr ? 1 : 0); // 前 rmr 个 root 各 rpr+1，其余 rpr
}

bool TopoInfoDetectScalable::RankHasRoot(u32 rank, u32 nRanks, u32 nRoots)
{
    const u32 rmr = nRanks % nRoots;  // 余数：前 rmr 个 root 各多 1 个 rank
    const u32 rpr = nRanks / nRoots;  // 基础 rank 数 per root
    const u32 rlim = rmr * (rpr + 1); // 前 rmr 个 root 的 rank 总数边界
    if (rank < rlim) {
        // 在前 rmr 个 root 的范围内，每 rpr+1 个 rank 为一组
        // 组内第一个 rank（rank % (rpr+1) == 0）是 root
        return (rank % (rpr + 1)) == 0;
    } else {
        // 在后 nRoots-rmr 个 root 的范围内，每 rpr 个 rank 为一组
        // 组内第一个 rank（(rank-rlim) % rpr == 0）是 root
        return ((rank - rlim) % rpr) == 0;
    }
}

HcclResult TopoInfoDetectScalable::SetupScalableRoot(HcclScalableRootHandle& outHandle)
{
    HcclIpAddress hostIP;
    u32 hostPort = HCCL_INVALID_PORT;
    std::vector<HcclSocketPortRange> portRanges;
    vector<HcclIpAddress> whitelist;
    CHK_RET(SetupRootServerNetwork(hostIP, hostPort, portRanges, whitelist));
    meshWhitelist_ = whitelist; // 保存白名单，init 侧拿到全部 root 信息后按 peer tag 下发 mesh 白名单

    // root间mesh全互联监听端口：与agent服务端口取自同一range，抢占得到互不相同的端口
    u32 meshPort = HCCL_INVALID_PORT;
    CHK_RET(SetupMeshListen(meshPort, portRanges));
    // mesh 白名单在 HcclCommInitRootInfoScalableInner 拿到全部 root 信息后按连接 tag 下发，
    // 此处不提前下发（此时无法预知对端 root index，无法构造与 accept 一致的 tag）

    g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
        status = TOPO_EXCHANGE_SERVER_STATUS_RUNING;
    });
    exchangeServerThreadPtr_.reset(new (nothrow) thread(
        &TopoInfoDetectScalable::SetupTopoExchangeServerScalable, this, devicePhysicID_, deviceLogicID_, hostIP,
        hostPort, whitelist, serverPortCtx_, listenSocket_, !portRanges.empty()));
    CHK_SMART_PTR_NULL(exchangeServerThreadPtr_);

    outHandle.rootHandle = rootInfo_;
    outHandle.meshPort = meshPort;
    HCCL_INFO(
        "setup scalable root complete, hostPort[%u], meshPort[%u], identifier[%s]", hostPort, meshPort,
        rootInfo_.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetectScalable::SetupMeshListen(u32& meshPort, const std::vector<HcclSocketPortRange>& portRanges)
{
    if (portRanges.empty()) {
        // 未配置端口范围（HCCL_IF_BASE_PORT 模式）：mesh 端口 = basePort + deviceId + 偏移，
        // 保证与 agent 服务端口（basePort + deviceId）互不相同
        meshPort = devicePhysicID_ + GetExternalInputHcclIfBasePort() + TOPO_MESH_PORT_OFFSET;
    }
    return StartListenNetwork(meshListenSocket_, serverPortCtx_, bootstrapHostIP_, meshPort, portRanges);
}

HcclResult TopoInfoDetectScalable::BuildMeshSocketWlistInfos(
    const std::vector<std::string>& tags, const std::vector<HcclIpAddress>& whitelist,
    std::vector<SocketWlistInfo>& wlistInfosVec) const
{
    for (auto ip : whitelist) {
        for (const auto& tag : tags) {
            SocketWlistInfo wlistInfo = {};
            wlistInfo.connLimit = HOST_SOCKET_CONN_LIMIT;
            wlistInfo.remoteIp.addr = ip.GetBinaryAddress().addr;
            wlistInfo.remoteIp.addr6 = ip.GetBinaryAddress().addr6;
            s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1);
            if (sRet != EOK) {
                HCCL_ERROR("[Build][MeshSocketWhiteList]memory copy failed. errorno[%d]", sRet);
                return HCCL_E_MEMORY;
            }
            wlistInfosVec.push_back(wlistInfo);
        }
    }
    return HCCL_SUCCESS;
}

// 拿到全部 root 信息后下发 mesh 白名单：为每个将 accept 的 smaller root 生成与连接一致的 tag
HcclResult TopoInfoDetectScalable::AddMeshSocketWhiteList(const ScalableServerInfo& info)
{
    if (GetExternalInputHcclEnableWhitelist() != HCCL_WHITELIST_ON || meshWhitelist_.empty() || !meshListenSocket_) {
        return HCCL_SUCCESS;
    }
    std::vector<std::string> tags;
    for (u32 peer = 0; peer < info.rootIndex; ++peer) {
        tags.push_back(
            TopoInfoExchangeBase::BuildMeshTag(peer, info.rootIndex, info.meshInfos[info.rootIndex].meshPort));
    }
    std::vector<SocketWlistInfo> wlistInfosVec;
    CHK_RET(BuildMeshSocketWlistInfos(tags, meshWhitelist_, wlistInfosVec));
    CHK_RET(meshListenSocket_->AddWhiteList(wlistInfosVec));
    HCCL_INFO(
        "add mesh socket white list success. rootIndex[%u], peerNum[%zu], total[%zu]", info.rootIndex, tags.size(),
        wlistInfosVec.size());
    return HCCL_SUCCESS;
}

// 按与下发一致的 peer tag 删除 mesh 白名单
HcclResult TopoInfoDetectScalable::DeleteMeshSocketWhiteList(const ScalableServerInfo& info) const
{
    std::vector<std::string> tags;
    for (u32 peer = 0; peer < info.rootIndex; ++peer) {
        tags.push_back(
            TopoInfoExchangeBase::BuildMeshTag(peer, info.rootIndex, info.meshInfos[info.rootIndex].meshPort));
    }
    std::vector<SocketWlistInfo> wlistInfosVec;
    CHK_RET(BuildMeshSocketWlistInfos(tags, meshWhitelist_, wlistInfosVec));
    if (!wlistInfosVec.empty()) {
        meshListenSocket_->DelWhiteList(wlistInfosVec);
    }
    HCCL_INFO("delete mesh socket white list success. rootIndex[%u], total[%zu]", info.rootIndex, wlistInfosVec.size());
    return HCCL_SUCCESS;
}

void TopoInfoDetectScalable::ReleaseMeshListenSocket(s32 deviceLogicID, bool hasPortRanges)
{
    if (meshListenSocket_) {
        if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
            (void)DeleteMeshSocketWhiteList(scalableInfo_);
        }
        if (!hasPortRanges) {
            (void)meshListenSocket_->DeInit();
        } else {
            (void)PreemptPortManager::GetInstance(deviceLogicID).Release(meshListenSocket_);
        }
        meshListenSocket_ = nullptr;
    }
}

HcclResult TopoInfoDetectScalable::SetScalableServerInfo(const ScalableServerInfo& info)
{
    {
        std::lock_guard<std::mutex> lock(scalableInfoMutex_);
        scalableInfo_ = info;
    }
    scalableInfoReady_.store(true, std::memory_order_release);
    HCCL_INFO(
        "[TopoInfoDetectScalable][SetScalableServerInfo] nRoot[%u], rootIndex[%u], groupSize[%u].", info.nRoot,
        info.rootIndex, info.groupSize);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetectScalable::WaitScalableInfoReady() const
{
    const auto start = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (!scalableInfoReady_.load(std::memory_order_acquire)) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
        if (elapsed > timeout) {
            HCCL_ERROR(
                "[TopoInfoDetectScalable][WaitScalableInfoReady]wait scalable info timeout[%lld s]", elapsed.count());
            return HCCL_E_TIMEOUT;
        }
        SaluSleep(ONE_MILLISECOND_OF_USLEEP);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetectScalable::SetupScalableExchangeServer(
    HcclIpAddress& hostIP, u32 hostPort, const std::vector<HcclIpAddress>& whitelist, HcclNetDevCtx netDevCtx,
    std::shared_ptr<HcclSocket> listenSocket)
{
    pTopoExchangeServer_.reset(new (nothrow) TopoInfoExchangeServer(
        hostIP, hostPort, whitelist, netDevCtx, listenSocket, rootInfo_.identifier));
    if (!pTopoExchangeServer_) {
        HCCL_ERROR("[SetupTopoExchangeServerScalable]build topoExchangeServer failed.");
        return HCCL_E_PTR;
    }

    ScalableServerInfo info;
    {
        std::lock_guard<std::mutex> lock(scalableInfoMutex_);
        info = scalableInfo_;
    }
    info.meshListenSocket = meshListenSocket_;
    // 供server内部mesh accept线程建立device上下文（该线程独立于server线程，未设置device）
    info.deviceLogicId = deviceLogicID_;
    CHK_RET(pTopoExchangeServer_->SetScalableInfo(info));
    return pTopoExchangeServer_->SetupScalable();
}

void TopoInfoDetectScalable::SetupTopoExchangeServerScalable(
    [[maybe_unused]] s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort,
    vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket,
    bool hasPortRanges)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_TopoDetect_Scalable");

    HcclResult ret = hrtSetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[SetupTopoExchangeServerScalable]set device[%d] failed, ret[%u]", deviceLogicID, ret);
        ReleaseMeshListenSocket(deviceLogicID, hasPortRanges);
        return;
    }

    // 等待 HcclCommInitRootInfoScalable 注入 root 间 mesh 全互联信息
    ret = WaitScalableInfoReady();
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[SetupTopoExchangeServerScalable]wait scalable info timeout, ret[%u]", ret);
        ReleaseMeshListenSocket(deviceLogicID, hasPortRanges);
        (void)hrtResetDevice(deviceLogicID);
        return;
    }

    ret = SetupScalableExchangeServer(hostIP, hostPort, whitelist, netDevCtx, listenSocket);
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[SetupTopoExchangeServerScalable]setup topoExchangeServer(scalable) failed, ret[%u]", ret);
    }

    // 释放root间mesh全互联监听端口
    ReleaseMeshListenSocket(deviceLogicID, hasPortRanges);

    ret = hrtResetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[SetupTopoExchangeServerScalable]reset device[%d] failed, ret[%u]", deviceLogicID, ret);
        return;
    }
    g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
        status = TOPO_EXCHANGE_SERVER_STATUS_IDLE;
    });
}
} // namespace hccl
