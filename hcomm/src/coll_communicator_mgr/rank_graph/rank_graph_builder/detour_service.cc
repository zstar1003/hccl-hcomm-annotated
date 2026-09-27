/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "detour_service.h"
#include "env_config/env_config_v2.h"
#include "detour_rules.h"
#include "not_support_exception.h"

namespace Hccl {

using namespace std;

constexpr u32 DETOUR_NODE_NUM = 8;
constexpr u32 DETOUR_NODE_NUM_2P = 2;
constexpr u32 DETOUR_NODE_NUM_4P = 4;

DetourService& DetourService::GetInstance()
{
    static DetourService detourService(PhyTopo::GetInstance().get());
    return detourService;
}

DetourService::DetourService(const PhyTopo* phyTopo) { this->phyTopo = phyTopo; }

struct DetourData {
    NodeId detourPhyPeerId{0};
    NodeId srcPhyPeerId{0};
    NodeId dstPhyPeerId{0};
    shared_ptr<NetInstance::Peer> srcNetInstPeer{nullptr};
    shared_ptr<NetInstance::Peer> dstNetInstPeer{nullptr};
};

vector<shared_ptr<PhyTopo::Link>>
GetLinks(NodeId srcId, NodeId dstId, const shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>>& phyTopoGraph)
{
    vector<shared_ptr<PhyTopo::Link>> links;
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[GetLinks] phyTopoGraph is nullptr"));
    }
    phyTopoGraph->TraverseEdge(srcId, dstId, [&](shared_ptr<PhyTopo::Link> link) {
        if (link != nullptr) {
            links.emplace_back(link);
            return;
        }
    });

    return links;
}

void AddDetourLink(
    NetInstance* innerNetInst, const DetourData& data,
    const shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>>& phyTopoGraph,
    [[maybe_unused]] const RankTableInfo* rankTable)
{
    vector<shared_ptr<PhyTopo::Link>> src2detVec = GetLinks(data.srcPhyPeerId, data.detourPhyPeerId, phyTopoGraph);
    vector<shared_ptr<PhyTopo::Link>> det2dstVec = GetLinks(data.detourPhyPeerId, data.dstPhyPeerId, phyTopoGraph);
    if (src2detVec.size() == 0 || det2dstVec.size() == 0) {
        return;
    }
    if (innerNetInst == nullptr) {
        THROW<NullPtrException>(StringFormat("[AddDetourLink] innerGroup is nullptr"));
    }
    u32 hop = 2; // 目前绕路hop一定是2
    for (const auto& src2detLink : src2detVec) {
        for (const auto& det2dstLink : det2dstVec) {
            LinkType linkType = src2detLink->GetType();
            std::set<LinkProtocol> linkProtocols = src2detLink->GetLinkProtocols();
            if (linkType != det2dstLink->GetType()) {
                HCCL_WARNING(
                    "[DetourService][InsertDetourLinks][AddDetourLink] src2det[%s] det2dst[%s] linkType no match",
                    linkType.Describe().c_str(), det2dstLink->GetType().Describe().c_str());
                continue;
            };
            // todo 是不是应该改为判断两个set是否有交集,取绕路的协议集合？ 修改了一下，llt再check一下
            std::set<LinkProtocol> newLinkProtocols;
            std::set_intersection(
                linkProtocols.begin(), linkProtocols.end(), det2dstLink->GetLinkProtocols().begin(),
                det2dstLink->GetLinkProtocols().end(), std::inserter(newLinkProtocols, newLinkProtocols.begin()));

            if (newLinkProtocols.empty()) {
                // todo 先改编译，后面再实现日志打印
                continue;
            };

            // 从Link中获取端口集合
            if (src2detLink->GetSourceIFace() == nullptr || det2dstLink->GetTargetIFace() == nullptr) {
                THROW<InvalidParamsException>(
                    "[DetourService][InsertDetourLinks][AddDetourLink] source ConnInterface is nullptr");
            }
            std::set<string> src2detPorts = src2detLink->GetSourceIFace()->GetPorts();
            std::set<string> det2dstPorts = det2dstLink->GetTargetIFace()->GetPorts();
            if (src2detPorts.size() != 1 || det2dstPorts.size() != 1) {
                THROW<InvalidParamsException>(
                    "[DetourService][InsertDetourLinks][AddDetourLink] Peer to Peer port num error");
            }

            // 取出对应的端口，然后去ranktableInfo中查对应端口的地址信息
            // todo 逻辑判断一下只取第一个端口是否正确？
            IpAddress src2detAddr = data.srcNetInstPeer->GetPortAddrMapLayer0()[*src2detPorts.begin()][0];
            IpAddress det2dstAddr = data.dstNetInstPeer->GetPortAddrMapLayer0()[*det2dstPorts.begin()][0];

            // 构造InterFace对象用于后续生成Link
            shared_ptr<NetInstance::ConnInterface> sourceIface = make_shared<NetInstance::ConnInterface>(
                src2detAddr, src2detPorts, src2detLink->GetSourceIFace()->GetPos(), LinkType::PEER2PEER,
                src2detLink->GetLinkProtocols(), src2detLink->GetTopoType(), src2detLink->GetTopoInstId());
            shared_ptr<NetInstance::ConnInterface> targetIface = make_shared<NetInstance::ConnInterface>(
                det2dstAddr, det2dstPorts, det2dstLink->GetSourceIFace()->GetPos(), LinkType::PEER2PEER,
                det2dstLink->GetLinkProtocols(), det2dstLink->GetTopoType(), det2dstLink->GetTopoInstId());

            // 构造Link加入NetInstance中
            shared_ptr<NetInstance::Link> sendEdge = make_shared<NetInstance::Link>(
                data.srcNetInstPeer, data.dstNetInstPeer, sourceIface, targetIface, linkType, newLinkProtocols,
                LinkDirection::SEND_ONLY, hop);
            shared_ptr<NetInstance::Link> recvEdge = make_shared<NetInstance::Link>(
                data.dstNetInstPeer, data.srcNetInstPeer, targetIface, sourceIface, linkType, newLinkProtocols,
                LinkDirection::RECV_ONLY, hop);
            innerNetInst->AddLink(sendEdge);
            innerNetInst->AddLink(recvEdge);

            data.srcNetInstPeer->AddConnInterface(0, sourceIface);
            if (data.dstNetInstPeer == nullptr) {
                THROW<NullPtrException>(
                    StringFormat("[DetourService][InsertDetourLinks][AddDetourLink] dstVirtPeer is nullptr"));
            }
            data.dstNetInstPeer->AddConnInterface(0, targetIface);

            HCCL_DEBUG(
                "[DetourService][AddDetourLink] add SEND_ONLY and RECV_ONLY two links: linkType[%s]"
                " sourceIfaceAddress[%s] targetIfaceAddress[%s]",
                linkType.Describe().c_str(), sourceIface->GetAddr().Describe().c_str(),
                targetIface->GetAddr().Describe().c_str());
        }
    }

    HCCL_DEBUG(
        "[DetourService][AddDetourLink] srcRankId[%llu] dstRankId[%llu] srcLocalId[%llu] detourLocalId[%llu] "
        "dstLocalId[%llu] src2detVec.size[%u] det2dstVec.size[%u]",
        data.srcNetInstPeer->GetNodeId(), data.dstNetInstPeer->GetNodeId(), data.srcPhyPeerId, data.detourPhyPeerId,
        data.dstPhyPeerId, src2detVec.size(), det2dstVec.size());
}

std::vector<LocalId> GetInnerLocalIds(const RankGraph* rankGraph)
{
    if (rankGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[GetInnerLocalIds] rankGraph is nullptr"));
    }
    const NetInstance* innerNetInst = rankGraph->GetNetInstanceByRankId(0, rankGraph->GetMyRank());
    if (innerNetInst == nullptr) {
        THROW<NullPtrException>(StringFormat("[GetInnerLocalIds] innerNetInst is nullptr"));
    }
    set<RankId> innerRanks = innerNetInst->GetRankIds();
    std::vector<LocalId> localIds;
    for (auto& rankId : innerRanks) {
        localIds.emplace_back(rankGraph->GetLocalId(rankId));
    }
    return localIds;
}

bool IsInSameRow(const std::vector<LocalId>& localIds)
{
    int preRowId = -1;
    for (auto& localId : localIds) {
        int curRowId = localId / DETOUR_NODE_NUM;
        if (preRowId == -1) {
            preRowId = curRowId;
        } else {
            if (curRowId != preRowId) {
                return false;
            }
        }
    }
    return true;
}

bool IsInSameCol(const std::vector<LocalId>& localIds)
{
    int preColId = -1;
    for (auto& localId : localIds) {
        int curColId = localId % DETOUR_NODE_NUM;
        if (preColId == -1) {
            preColId = curColId;
        } else {
            if (curColId != preColId) {
                return false;
            }
        }
    }
    return true;
}

bool GetTableIds(
    const std::vector<LocalId>& localIds, std::unordered_map<LocalId, u32>& tableIds, std::set<u32>& tableIdSet)
{
    if (IsInSameRow(localIds)) {
        for (auto& localId : localIds) {
            u32 tableId = localId % DETOUR_NODE_NUM;
            tableIds[localId] = tableId;
            tableIdSet.emplace(tableId);
        }
    } else if (IsInSameCol(localIds)) {
        for (auto& localId : localIds) {
            u32 tableId = localId / DETOUR_NODE_NUM;
            tableIds[localId] = tableId;
            tableIdSet.emplace(tableId);
        }
    } else {
        HCCL_WARNING("[DetourService][GetTableIds] The ranks localIds are not in the same row or column.");
        return false;
    }
    return true;
}

void SetDetourTable4P(
    const std::set<u32>& tableIdSet, unordered_map<LocalId, unordered_map<LocalId, vector<LocalId>>>& detourTable)
{
    if (tableIdSet == std::set<u32>{0, 1, 2, 3}) { // tableId必须为 0 1 2 3，才能使用DETOUR4PTABLE_0123绕路
        detourTable = GetDetour4PTable0123();
        HCCL_DEBUG("[DetourService] selected detour table is DETOUR4PTABLE_0123");
    } else if (tableIdSet == std::set<u32>{4, 5, 6, 7}) { // tableId必须为 4 5 6 7，才能使用DETOUR4PTABLE_4567绕路
        detourTable = GetDetour4PTable4567();
        HCCL_DEBUG("[DetourService] selected detour table is DETOUR4PTABLE_4567");
    } else if (tableIdSet == std::set<u32>{0, 2, 4, 6}) { // tableId必须为 0 2 4 6，才能使用DETOUR4PTABLE_0246绕路
        detourTable = GetDetour4PTable0246();
        HCCL_DEBUG("[DetourService] selected detour table is DETOUR4PTABLE_0246");
    } else if (tableIdSet == std::set<u32>{1, 3, 5, 7}) { // tableId必须为 1 3 5 7，才能使用DETOUR4PTABLE_1357绕路
        detourTable = GetDetour4PTable1357();
        HCCL_DEBUG("[DetourService] selected detour table is DETOUR4PTABLE_1357");
    } else {
        HCCL_WARNING("no matched detourTable found");
    }
}

void SetDetourTable2P(
    const std::set<u32>& tableIdSet, unordered_map<LocalId, unordered_map<LocalId, vector<LocalId>>>& detourTable)
{
    if (tableIdSet == std::set<u32>{0, 1} || tableIdSet == std::set<u32>{2, 3} || tableIdSet == std::set<u32>{4, 5}
        || tableIdSet == std::set<u32>{6, 7}) {
        detourTable = GetDetour2PTable01();
    } else if (
        tableIdSet == std::set<u32>{0, 4} || tableIdSet == std::set<u32>{1, 5} || tableIdSet == std::set<u32>{2, 6}
        || tableIdSet == std::set<u32>{3, 7}) {
        detourTable = GetDetour2PTable04();
    } else {
        HCCL_WARNING("[DetourService][%s]no matched detourTable found", __func__);
    }
}

void GetDetourTableAndTableIds(
    const std::vector<LocalId>& localIds,
    std::unordered_map<LocalId, unordered_map<LocalId, vector<LocalId>>>& detourTable,
    std::unordered_map<LocalId, u32>& tableIds, const RankTableInfo* rankTable)
{
    std::set<u32> tableIdSet;
    HcclDetourType detourType = EnvConfig::GetInstance().GetDetourConfig().GetDetourType();
    if (rankTable->detour == false) {
        detourType = HcclDetourType::HCCL_DETOUR_DISABLE;
    }
    switch (detourType) {
        case HcclDetourType::HCCL_DETOUR_ENABLE_2P: {
            if (localIds.size() != DETOUR_NODE_NUM_2P) {
                return;
            }
            bool res = GetTableIds(localIds, tableIds, tableIdSet);
            if (res) {
                SetDetourTable2P(tableIdSet, detourTable);
                HCCL_DEBUG("[DetourService] selected detour type is DETOUR2PTABLE");
            } else {
                HCCL_WARNING("[DetourService] detourtype [%s] does not support.", detourType.Describe().c_str());
            }
            break;
        }
        case HcclDetourType::HCCL_DETOUR_ENABLE_4P: {
            if (localIds.size() != DETOUR_NODE_NUM_4P) {
                return;
            }
            bool res = GetTableIds(localIds, tableIds, tableIdSet);
            if (res) {
                SetDetourTable4P(tableIdSet, detourTable);
            } else {
                HCCL_WARNING("[DetourService] detourtype [%s] does not support.", detourType.Describe().c_str());
            }
            break;
        }
        case HcclDetourType::HCCL_DETOUR_DISABLE: {
            HCCL_DEBUG("[DetourService] detour is disable");
            break;
        }
        default: {
            THROW<NotSupportException>(
                StringFormat("[DetourService] detourtype [%s] does not support.", detourType.Describe().c_str()));
        }
    }
}

void AddDetourLinks(
    const PhyTopo* phyTopo, RankGraph* rankGraph,
    std::unordered_map<LocalId, unordered_map<LocalId, vector<LocalId>>>& detourTable,
    std::unordered_map<LocalId, u32>& tableIds, const RankTableInfo* rankTable)
{
    auto phyTopoGraph = phyTopo->GetTopoGraph(0);
    NetInstance* innerNetInst = rankGraph->GetNetInstanceByRankId(0, rankGraph->GetMyRank());
    if (innerNetInst == nullptr) {
        THROW<NullPtrException>(StringFormat("[DetourService] innerNetInst is nullptr"));
    }
    set<RankId> innerRanks = innerNetInst->GetRankIds();
    for (const auto& srcRankId : innerRanks) {
        LocalId srcLocalId = rankGraph->GetLocalId(srcRankId);
        u32 srcTableId = tableIds[srcLocalId];
        for (const auto& dstRankId : innerRanks) {
            LocalId dstLocalId = rankGraph->GetLocalId(dstRankId);
            u32 dstTableId = tableIds[dstLocalId];

            if (detourTable.count(srcTableId) == 0 || detourTable[srcTableId].count(dstTableId) == 0) {
                continue;
            }

            auto detourTableIds = detourTable[srcTableId][dstTableId];
            for (auto& detourTableId : detourTableIds) {
                LocalId detourLocalId = detourTableId + srcLocalId - srcTableId;

                // 插入detourlink
                DetourData detourData;
                detourData.detourPhyPeerId = PhyTopo::Peer::GetId(detourLocalId); // 绕路可能没有绕路节点的rankid
                detourData.srcPhyPeerId = PhyTopo::Peer::GetId(srcLocalId);
                detourData.dstPhyPeerId = PhyTopo::Peer::GetId(dstLocalId);
                detourData.srcNetInstPeer = rankGraph->GetPeer(srcRankId);
                detourData.dstNetInstPeer = rankGraph->GetPeer(dstRankId);

                AddDetourLink(innerNetInst, detourData, phyTopoGraph, rankTable);
            }
        }
    }
}

void DetourService::InsertDetourLinks(RankGraph* rankGraph, const RankTableInfo* rankTable)
{
    std::unordered_map<LocalId, u32> tableIds;
    std::unordered_map<LocalId, unordered_map<LocalId, vector<LocalId>>> detourTable;

    // 获取innerGroup的localIds
    std::vector<LocalId> localIds = GetInnerLocalIds(rankGraph);
    // 获取当前localIds的tableId和detourTable
    GetDetourTableAndTableIds(localIds, detourTable, tableIds, rankTable);

    if (!detourTable.empty()) {
        // 添加绕路links
        AddDetourLinks(phyTopo, rankGraph, detourTable, tableIds, rankTable);
    } else {
        HCCL_WARNING("no detour links found");
    }
}

} // namespace Hccl
