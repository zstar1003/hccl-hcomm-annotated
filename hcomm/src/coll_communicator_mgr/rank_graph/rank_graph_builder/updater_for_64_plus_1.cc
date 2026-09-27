/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../rank_graph_builder/updater_for_64_plus_1.h"
#include "topo_common_types.h"
#include "exception_util.h"
#include "invalid_params_exception.h"

namespace Hccl {

using namespace std;

void UpdaterFor64Plus1::SaveReplaceInfo(const NewRankInfo& rank)
{
    if (rank.localId != BACKUP_LOCAL_ID) {
        return;
    }
    string netInstId = "";
    for (const auto& netLayerInfo : rank.rankLevelInfos) {
        if (netLayerInfo.netLayer == 0) {
            netInstId = netLayerInfo.netInstId;
        }
    }
    if (netInstId.empty()) {
        THROW<InvalidParamsException>(StringFormat(
            "[UpdaterFor64Plus1::Init] "
            "Replaced rank[%d] has empty R0Id",
            rank.rankId));
    }
    if (replaceInfo.find(netInstId) != replaceInfo.end()) {
        THROW<InvalidParamsException>(StringFormat(
            "[UpdaterFor64Plus1::Init] "
            "R0Group[%s] has more than one backups",
            netInstId.c_str()));
    }
    replaceInfo[netInstId] = make_pair(rank.localId, rank.replacedLocalId);
}

void UpdaterFor64Plus1::UpdateRankGraph(RankGraph* rankGraph, const RankTableInfo* rankTable) const
{
    if (rankGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[UpdaterFor64Plus1][%s] rankGraph is nullptr", __func__));
    }
    if (rankTable == nullptr) {
        THROW<NullPtrException>(StringFormat("[UpdaterFor64Plus1][%s] rankTable is nullptr", __func__));
    }
    for (const auto& it : replaceInfo) {
        const auto& netInstId = it.first;
        s32 localId = it.second.first;
        s32 replacedLocalId = it.second.second;
        NetInstance* netInstance = rankGraph->GetNetInstanceByNetInstId(0, netInstId);
        if (netInstance == nullptr) {
            HCCL_WARNING(
                "[UpdaterFor64Plus1][%s] netInstance netlayer[0] netInstanceId[%s] not exist", __func__,
                netInstId.c_str());
            continue;
        }
        UpdateNetInstance(netInstance, localId, replacedLocalId, rankTable);
    }
}

void UpdaterFor64Plus1::UpdateNetInstance(
    NetInstance* netInstance, LocalId localId, LocalId replacedLocalId, const RankTableInfo* rankTable) const
{
    if (netInstance == nullptr) {
        THROW<NullPtrException>(StringFormat("[UpdaterFor64Plus1][%s] netInstance is nullptr", __func__));
    }
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph(0);
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[UpdaterFor64Plus1][%s] phyTopoGraph is nullptr", __func__));
    }
    HCCL_DEBUG(
        "[UpdaterFor64Plus1][%s] Updating NetInstance[%s]: localId[%u]->replacedId[%u]", __func__,
        netInstance->GetNetInstId().c_str(), localId, replacedLocalId);

    // 找到与故障D直连的D
    shared_ptr<NetInstance::Peer> backupPeer; // local[64]
    vector<shared_ptr<NetInstance::Peer>> backupLinkedPeers;
    for (const auto& it : netInstance->GetPeers()) {
        auto peer = it.second;
        if (peer->GetLocalId() == BACKUP_LOCAL_ID) {
            backupPeer = peer;
            continue;
        }
        if (!IsSameX(peer->GetLocalId(), replacedLocalId) && !IsSameY(peer->GetLocalId(), replacedLocalId)) {
            continue;
        }
        if (!phyTopoGraph->HasEdge(PhyTopo::Peer::GetId(peer->GetLocalId()), PhyTopo::Peer::GetId(replacedLocalId))) {
            continue;
        }
        backupLinkedPeers.emplace_back(peer);
    }

    // 添加与故障D直连的D到备份D的Link
    for (const auto& peer : backupLinkedPeers) {
        AddPeer2BackupLinks(peer, backupPeer, replacedLocalId, netInstance, rankTable);
    }
}

void UpdaterFor64Plus1::AddPeer2BackupLinks(
    shared_ptr<NetInstance::Peer> peer, shared_ptr<NetInstance::Peer> backupPeer, LocalId replacedLocalId,
    NetInstance* netInstance, [[maybe_unused]] const RankTableInfo* rankTable) const
{
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph(0);

    std::unordered_map<u64, u64> fabricIds;
    auto peer2AllPlaneEdges
        = phyTopoGraph->GetEdges(PhyTopo::Peer::GetId(backupPeer->GetLocalId()), PhyTopo::Fabric::GetId());
    for (auto edge : peer2AllPlaneEdges) {
        auto topoInstId = edge->GetTopoInstId();
        auto fabricId = static_cast<u64>(topoInstId) | (static_cast<u64>(1) << 32);
        fabricIds[topoInstId] = fabricId;
    }

    auto idx = GetLinkIndex(peer->GetLocalId(), replacedLocalId);
    auto backupPlaneId = idx.first;
    auto backupLinkIdx = idx.second;
    HCCL_DEBUG(
        "[UpdaterFor64Plus1][%s] Peer{rankId[%d], localId[%u]} will use BackupPlane[%u] addr[%u]", __func__,
        peer->GetRankId(), peer->GetLocalId(), backupPlaneId, backupLinkIdx);

    // 先获取phyTopoGraph中备份面和备份D的连接，因为只有一个fabric，所以会获取到全量16条备份面和备份D的连接
    // Edges中可能包含多个连接，但是在phytopo中保存为一条连接，内部有多个连接的端口
    // 拿到物理边后，根据backupPlaneId匹配，选择对应的一条物理边
    std::shared_ptr<PhyTopo::Link> backD2PlaneEdges = GetPeer2PlaneEdges(backupPlaneId, backupPeer, phyTopoGraph);
    std::shared_ptr<PhyTopo::Link> peer2PlaneEdges = GetPeer2PlaneEdges(backupPlaneId, peer, phyTopoGraph);

    // 取得端口列表，根据backupLinkIdx去选择对应的端口
    std::set<std::string> backD2PlanePorts = backD2PlaneEdges->GetSourceIFace()->GetPorts();
    std::set<std::string> peer2PlanePorts = peer2PlaneEdges->GetSourceIFace()->GetPorts();

    // 校验端口集合大小是否符合预期
    if (backD2PlanePorts.size() != BACKUP_TO_PLANE_ADDR_NUM) {
        THROW<InvalidParamsException>(
            "[UpdaterFor64Plus1][%s] Backup to BackupPlane[%u] port num error", __func__, backupPlaneId);
    }
    if (peer2PlanePorts.size() != 1) {
        THROW<InvalidParamsException>(
            "[UpdaterFor64Plus1][%s] Peer[%u] to BackupPlane[%u] port num error", __func__, peer->GetLocalId(),
            backupPlaneId);
    }

    // 从端口集合中取出对应逻辑位置的端口
    std::string backD2PlanePort = GetPortFromSet(backD2PlanePorts, backupLinkIdx);
    std::string peer2PlanePort = GetPortFromSet(peer2PlanePorts, 0);

    // 匹配了对应的端口后，去ranktableinfo中查对应端口的地址信息
    // todo 加一下peer->GetPortAddrMapLayer0()是否能找到port对应的地址。
    IpAddress backD2PlaneAddr = backupPeer->GetPortAddrMapLayer0()[backD2PlanePort][0];
    IpAddress peer2PlaneAddr = peer->GetPortAddrMapLayer0()[peer2PlanePort][0];

    // 组装成NetInstance的conninterface，加入peer和backupPeer
    if (backD2PlaneEdges->GetSourceIFace() == nullptr || peer2PlaneEdges->GetSourceIFace() == nullptr) {
        THROW<InvalidParamsException>("[UpdaterFor64Plus1][%s] source ConnInterface is nullptr", __func__);
    }
    std::set<string> peer2PlanePortSet;
    peer2PlanePortSet.insert(peer2PlanePort);
    std::set<string> backD2PlanePortSet;
    backD2PlanePortSet.insert(backD2PlanePort);

    shared_ptr<NetInstance::ConnInterface> backupIface = make_shared<NetInstance::ConnInterface>(
        backD2PlaneAddr, backD2PlanePortSet, backD2PlaneEdges->GetSourceIFace()->GetPos(), LinkType::PEER2PEER,
        backD2PlaneEdges->GetLinkProtocols(), backD2PlaneEdges->GetTopoType(), backD2PlaneEdges->GetTopoInstId());
    shared_ptr<NetInstance::ConnInterface> peerIface = make_shared<NetInstance::ConnInterface>(
        peer2PlaneAddr, peer2PlanePortSet, peer2PlaneEdges->GetSourceIFace()->GetPos(), LinkType::PEER2PEER,
        peer2PlaneEdges->GetLinkProtocols(), backD2PlaneEdges->GetTopoType(), backD2PlaneEdges->GetTopoInstId());

    backupPeer->AddConnInterface(0, backupIface);
    peer->AddConnInterface(0, peerIface);

    shared_ptr<NetInstance::Link> peer2Backup = make_shared<NetInstance::Link>(
        peer, backupPeer, peerIface, backupIface, LinkType::PEER2PEER, backD2PlaneEdges->GetLinkProtocols());
    netInstance->AddLink(peer2Backup);
    shared_ptr<NetInstance::Link> backup2Peer = make_shared<NetInstance::Link>(
        backupPeer, peer, backupIface, peerIface, LinkType::PEER2PEER, backD2PlaneEdges->GetLinkProtocols());
    netInstance->AddLink(backup2Peer);

    // peer2peer建立后删除graph中DB到所选planeId的对应的peer2net链路
    // 直接删除备份d和fabric的链接保证GetLinks接口只能获取到peer2db的一条peer2peer
    // 删除peer到fabric用到的peer2net
    for (auto id : fabricIds) {
        netInstance->DeleteLink(backupPeer->GetNodeId(), id.second);
    }
}

std::shared_ptr<PhyTopo::Link> UpdaterFor64Plus1::GetPeer2PlaneEdges(
    u32 backupPlaneId, shared_ptr<NetInstance::Peer> peer,
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> phyTopoGraph) const
{
    std::shared_ptr<PhyTopo::Link> peer2PlaneEdges = nullptr;
    auto peer2AllPlaneEdges
        = phyTopoGraph->GetEdges(PhyTopo::Peer::GetId(peer->GetLocalId()), PhyTopo::Fabric::GetId());
    if (peer2AllPlaneEdges.size() != BACKUP_PLANE_NUM) {
        THROW<InvalidParamsException>("[UpdaterFor64Plus1][%s] BackupPlane num error", __func__);
    }
    for (auto& edges : peer2AllPlaneEdges) {
        if (edges->GetTopoInstId() == backupPlaneId) {
            peer2PlaneEdges = edges;
            break;
        }
    }
    if (peer2PlaneEdges == nullptr) {
        THROW<NullPtrException>(StringFormat("[UpdaterFor64Plus1][%s] peer2PlaneEdges is nullptr", __func__));
    }
    return peer2PlaneEdges;
}

std::string UpdaterFor64Plus1::GetPortFromSet(std::set<string>& ports, u32 linkIdx) const
{
    std::string peer2PlanePort = "";
    if (linkIdx >= ports.size()) {
        THROW<InvalidParamsException>("[UpdaterFor64Plus1][%s] BackupPlane port num error", __func__);
    }
    for (auto& port : ports) {
        if (linkIdx == 0) {
            peer2PlanePort = port;
            break;
        }
        linkIdx--;
    }
    return peer2PlanePort;
}

bool UpdaterFor64Plus1::IsSameX(LocalId srcLocalId, LocalId dstLocalId) const
{
    // 两个localId对应的D是否在同一个X轴
    return (srcLocalId / DEVICE_NUM_PER_AXIS) == (dstLocalId / DEVICE_NUM_PER_AXIS);
}

bool UpdaterFor64Plus1::IsSameY(LocalId srcLocalId, LocalId dstLocalId) const
{
    // 两个localId对应的D是否在同一个Y轴
    return (srcLocalId % DEVICE_NUM_PER_AXIS) == (dstLocalId % DEVICE_NUM_PER_AXIS);
}

pair<u32, u32> UpdaterFor64Plus1::GetLinkIndex(LocalId localId, LocalId replacedLocalId) const
{
    if (localId >= BACKUP_LOCAL_ID || replacedLocalId >= BACKUP_LOCAL_ID || localId == replacedLocalId) {
        THROW<InvalidParamsException>(
            "[UpdaterFor64Plus1][%s] localId[%u] or replacedId[%u] invalid", __func__, localId, replacedLocalId);
        return {};
    }
    if (IsSameX(localId, replacedLocalId)) {
        auto idx = localId % DEVICE_NUM_PER_AXIS;
        if (idx < DEVICE_HALF_NUM_PER_AXIS) {
            return make_pair(0, idx); // X轴左边4个正常D走备份面0
        } else {
            return make_pair(1, idx % DEVICE_HALF_NUM_PER_AXIS); // X轴右边4个正常D走备份面1
        }
    } else if (IsSameY(localId, replacedLocalId)) {
        auto idx = localId / DEVICE_NUM_PER_AXIS;
        if (idx < DEVICE_HALF_NUM_PER_AXIS) {
            return make_pair(2, idx); // Y轴上边4个正常D走备份面2
        } else {
            return make_pair(3, idx % DEVICE_HALF_NUM_PER_AXIS); // Y轴下边4个正常D走备份面3
        }
    } else {
        THROW<InvalidParamsException>(
            "[UpdaterFor64Plus1][%s] localId[%u] and replacedId[%u] are not at same line", __func__, localId,
            replacedLocalId);
        return {};
    }
}

} // namespace Hccl
