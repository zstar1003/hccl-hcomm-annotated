/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include "virtual_topo.h"
#include "string_util.h"
#include "binary_stream.h"
#include "exception_util.h"
#include "topo_common_types.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "not_support_exception.h"
#include "rank_gph.h"

namespace Hccl {

using namespace std;

void RankGraph::AddPeer(const shared_ptr<NetInstance::Peer>& peer)
{
    if (!initFlag_) {
        if (peer == nullptr) {
            THROW<NullPtrException>("[RankGraph][AddPeer] peer is nullptr.");
        }

        RankId rankId = peer->GetRankId();
        peers_[rankId] = peer;
    } else {
        THROW<InternalException>("RankGraph AddPeer fail, rankGraph has been initialized, please check.");
    }
}

void RankGraph::AddNetInstance(const shared_ptr<NetInstance>& netInst)
{
    if (netInst == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraph][AddNetInstance] netInst is nullptr"));
    }
    if (!initFlag_) {
        // 外部调用保证netInst非空
        // 处理添加的netInst中的rank在virtualTopo中没有的情况
        for (const auto rankId : netInst->GetRankIds()) {
            if (!HasRank(rankId)) {
                THROW<InvalidParamsException>(
                    StringFormat("[RankGraph][AddNetInstance] Non-innerRank[%d] exists netInst.", rankId));
            }
        }

        // 添加netInst, 若重复添加打印日志
        auto res = netInsts_[netInst->GetNetLayer()].emplace(netInst->GetNetInstId(), netInst);
        if (!res.second) {
            HCCL_WARNING(
                "[RankGraph][AddNetInstance] netLayer[%u] netInstId[%s] is existed.", netInst->GetNetLayer(),
                netInst->GetNetInstId().c_str());
        }
    } else {
        THROW<InternalException>("RankGraph AddNetInstance fail, rankGraph has been initialized, please check.");
    }
}

void RankGraph::InitInnerRanks()
{
    // 只支持在创建好InnerGroup之后调用, 否则抛异
    auto innerInstance = GetNetInstanceByRankId(0, myRank_);
    if (innerInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraph][SetInnerRanks] myRank[%d] netLayer[0] netInst is not existed.", myRank_));
    }

    innerRanks_ = innerInstance->GetRankIds();
}

void RankGraph::InitFinish() { initFlag_ = true; }

bool RankGraph::HasRank(RankId rankId) const
{
    if (peers_.find(rankId) == peers_.end()) {
        HCCL_DEBUG("[RankGraph][HasRank] rankId[%d] is not existed", rankId);
        return false;
    }
    return true;
}

u32 RankGraph::GetRankSize() const { return peers_.size(); }

u32 RankGraph::GetInnerRankSize() const { return innerRanks_.size(); }

RankId RankGraph::GetMyRank() const { return myRank_; }

LocalId RankGraph::GetLocalId(RankId rankId) const
{
    if (!HasRank(rankId)) {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][GetLocalId] rankId[%d] is not existed.", rankId));
    }

    return peers_.at(rankId)->GetLocalId();
}

LocalId RankGraph::GetReplacedLocalId(RankId rankId) const
{
    if (!HasRank(rankId)) {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][GetLocalId] rankId[%d] is not existed.", rankId));
    }

    return peers_.at(rankId)->GetReplacedLocalId();
}

set<u32> RankGraph::GetLevels(RankId rankId) const
{
    if (!HasRank(rankId)) {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][GetLevels] rankId[%d] is not existed.", rankId));
    }

    return peers_.at(rankId)->GetLevels();
}

u32 RankGraph::GetLevelNum() const
{
    u32 validLevelNum{0};
    for (const auto& netInst : netInsts_) {
        if (!netInst.empty()) {
            validLevelNum++;
        }
    }
    HCCL_INFO("[RankGraph][%s] validLevelNum[%u]", __func__, validLevelNum);
    return validLevelNum;
}

const NetInstance* RankGraph::GetNetInstanceByNetInstId(u32 netLayer, const string& netInstId) const
{
    // 不存在netInst, 则返回空
    if (netLayer >= netInsts_.size() || netInsts_.at(netLayer).count(netInstId) == 0) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance netLayer[%u] netInstId[%s] is not existed.", netLayer,
            netInstId.c_str());
        return nullptr;
    }
    return netInsts_.at(netLayer).at(netInstId).get();
}

NetInstance* RankGraph::GetNetInstanceByNetInstId(u32 netLayer, const std::string& netInstId)
{
    // 不存在netInst, 则返回空
    if (netLayer >= netInsts_.size() || netInsts_.at(netLayer).count(netInstId) == 0) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance netLayer[%u] netInstId[%s] is not existed.", netLayer,
            netInstId.c_str());
        return nullptr;
    }
    return netInsts_.at(netLayer).at(netInstId).get();
}

const NetInstance* RankGraph::GetNetInstanceByRankId(u32 netLayer, RankId rankId) const
{
    // 不存在netInst, 则返回空
    if (!HasRank(rankId)) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance rankId[%d] netLayer[%u] is not existed.", rankId, netLayer);
        return nullptr;
    }
    return peers_.at(rankId)->GetNetInstance(netLayer);
}

NetInstance* RankGraph::GetNetInstanceByRankId(u32 netLayer, RankId rankId)
{
    if (!HasRank(rankId)) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance rankId[%d] netLayer[%u] is not existed.", rankId, netLayer);
        return nullptr;
    }
    const NetInstance* constInstance = peers_.at(rankId)->GetNetInstance(netLayer);
    if (constInstance == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraph][GetGroup]GetNetInstance(rankId, netLayer) is nullptr"));
    }
    string netInstId = constInstance->GetNetInstId();
    return netInsts_.at(netLayer).at(netInstId).get();
}

const shared_ptr<NetInstance::Peer> RankGraph::GetPeer(RankId rankId) const
{
    if (!HasRank(rankId)) {
        HCCL_WARNING("[RankGraph][GetPeer] rankId[%d] is not existed.", rankId);
        return nullptr;
    }

    return peers_.at(rankId);
}

vector<NetInstance::Path> RankGraph::GetPaths(u32 netLayer, RankId sRankId, RankId dRankId) const
{
    // 若sRank和dRank均不在innerGroup里，则返回空
    if (innerRanks_.count(sRankId) == 0 && innerRanks_.count(dRankId) == 0) {
        HCCL_WARNING("[RankGraph][GetPaths] sRankId[%d] and dRankId[%d] are not in innerRanks", sRankId, dRankId);
        return {};
    }

    vector<NetInstance::Path> paths;
    auto netInst = GetNetInstanceByRankId(netLayer, sRankId);
    if (netInst == nullptr) {
        HCCL_WARNING("[RankGraph][GetPaths] netLayer[%u] sRankId[%d] netInst is not existed.", netLayer, sRankId);
        return {};
    }
    if (!netInst->HasNode(NetInstance::Peer::GenerateNodeId(dRankId))) {
        HCCL_WARNING(
            "[RankGraph][GetPaths] netLayer[%u] sRankId[%d] netInst has no dRankId[%d].", netLayer, sRankId, dRankId);
        return {};
    }

    paths = netInst->GetPaths(sRankId, dRankId);
    if (paths.size() == 0) {
        HCCL_WARNING(
            "[RankGraph][GetPaths] netLayer[%u] sRankId[%d] dRankId[%d] netInst has no path.", netLayer, sRankId,
            dRankId);
        return paths;
    }

    HCCL_DEBUG(
        "[RankGraph][GetPaths] netLayer[%u] sRankId[%d] dRankId[%d] pathsize[%u].", netLayer, sRankId, dRankId,
        paths.size());
    return paths;
}

u32 RankGraph::GetLayerRanks(const u32 netLayer) const
{
    u32 layerRankSize = 0;
    if (netInsts_.at(netLayer).size() == 0) {
        HCCL_WARNING("[RankGraph][GetLayerRanks] RankGraph has no netInstance on netLayer %u", netLayer);
        return 0;
    }
    for (const auto& netInst : netInsts_.at(netLayer)) {
        layerRankSize += netInst.second->GetRankSize();
    }
    return layerRankSize;
}

void RankGraph::GetLocalInstRanks(const u32 netLayer, vector<u32>& rankList, u32& rankNum) const
{
    const NetInstance* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    if (netInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraph][GetLocalInstRanks] myRank %u has no netInstance on layer %u", myRank_, netLayer));
    }
    set<RankId> rankSet = netInstance->GetRankIds();
    rankList.clear();
    for (const RankId& rank : rankSet) {
        rankList.push_back(static_cast<u32>(rank));
    }
    rankNum = rankSet.size();
}

u32 RankGraph::GetLocalInstSize(const u32 netLayer) const
{
    const NetInstance* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    if (netInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraph][GetLocalInstSize] myRank %u has no netInstance on layer %u", myRank_, netLayer));
    }
    return netInstance->GetRankSize();
}

const NetType RankGraph::GetNetType(const u32 netLayer) const
{
    const NetInstance* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    if (netInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraph][GetLocalInstSize] myRank %u has no netInstance on layer %u", myRank_, netLayer));
    }
    return netInstance->GetNetType();
}

HcclResult RankGraph::GetNetInstanceList(const u32 netLayer, vector<u32>& instSizeList, u32& listSize) const
{
    instSizeList.clear();
    listSize = 0;
    if (netInsts_.at(netLayer).size() == 0) {
        HCCL_WARNING("[RankGraph][GetNetInstanceList] RankGraph has no net instance on layer %u", netLayer);
        return HCCL_E_PARA;
    }
    for (const auto& netInst : netInsts_.at(netLayer)) {
        instSizeList.push_back(netInst.second->GetRankSize());
    }
    listSize = instSizeList.size();
    return HCCL_SUCCESS;
}

void RankGraph::GetTopoInstsByLayer(const u32 netLayer, std::vector<u32>& topoInsts, u32& topoInstNum) const
{
    auto* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    netInstance->GetTopoInstsByLayer(topoInsts, topoInstNum);
}

HcclResult RankGraph::GetTopoType(const u32 netLayer, const u32 topoInstId, TopoType& topoType) const
{
    auto* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    CHK_PRT_RET(
        netInstance == nullptr,
        HCCL_ERROR(
            "[RankGraph::GetTopoType] netInstance is nullptr, myRank[%d], netLayer[%u], "
            "topoInstId[%u]",
            myRank_, netLayer, topoInstId),
        HCCL_E_PTR);

    auto ret = netInstance->GetTopoType(topoInstId, topoType);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Failed to GetTopoType, myRank[%d], netLayer[%u], netInstId[%s], topoInstId[%u], "
            "ret[%d]",
            __func__, myRank_, netLayer, netInstance->GetNetInstId().c_str(), topoInstId, ret),
        ret);
    return HCCL_SUCCESS;
}

HcclResult
RankGraph::GetRanksByTopoInst(const u32 netLayer, const u32 topoInstId, std::vector<u32>& ranks, u32& rankNum) const
{
    auto* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    CHK_PRT_RET(
        netInstance == nullptr,
        HCCL_ERROR(
            "[RankGraph::GetRanksByTopoInst] netInstance is nullptr, myRank[%d], netLayer[%u], "
            "topoInstId[%u]",
            myRank_, netLayer, topoInstId),
        HCCL_E_PTR);

    auto ret = netInstance->GetRanksByTopoInst(topoInstId, ranks, rankNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Failed to GetRanksByTopoInst, myRank[%d], netLayer[%u], netInstId[%s], "
            "topoInstId[%u], ret[%d]",
            __func__, myRank_, netLayer, netInstance->GetNetInstId().c_str(), topoInstId, ret),
        ret);
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetEndpointNum(uint32_t layer, uint32_t topoInstId, uint32_t* num) const
{
    auto peer = GetPeer(myRank_);
    if (peer == nullptr) {
        HCCL_ERROR("[RankGraph::GetEndpointNum] Peer is nullptr at netLayer [%u]", layer);
        return HCCL_E_PTR;
    }
    auto ifacesVec = peer->GetIfacesByLayer(layer);
    uint32_t sum = 0;
    for (auto& iface : ifacesVec) {
        if (iface->GetTopoInstId() == topoInstId) {
            sum += iface->GetLinkProtocols().size();
        }
    }
    *num = sum;
    return HCCL_SUCCESS;
}

HcclResult GetCommAddr(CommAddr& commAddr, const IpAddress& ipAddr)
{
    s32 family = ipAddr.GetFamily();
    if (family == AF_INET) {
        string addr = ipAddr.GetIpStr();
        if (ipAddr.IsEID(addr)) {
            commAddr.type = COMM_ADDR_TYPE_EID;
            const auto& eid = ipAddr.GetEid();
            for (u32 i = 0; i < URMA_EID_LEN && i < sizeof(commAddr.eid); i++) {
                commAddr.eid[i] = eid.raw[i];
            }
        } else {
            commAddr.type = COMM_ADDR_TYPE_IP_V4;
            commAddr.addr = ipAddr.GetBinaryAddress().addr;
        }
    } else if (family == AF_INET6) {
        commAddr.type = COMM_ADDR_TYPE_IP_V6;
        commAddr.addr6 = ipAddr.GetBinaryAddress().addr6;
    } else {
        HCCL_ERROR("invalid commAddrType");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

EndpointLocType AddrPositionToEndpointLoc(AddrPosition pos)
{
    switch (pos) {
        case AddrPosition::HOST:
            return ENDPOINT_LOC_TYPE_HOST;
        case AddrPosition::DEVICE:
            return ENDPOINT_LOC_TYPE_DEVICE;
        default:
            return ENDPOINT_LOC_TYPE_RESERVED;
    }
}

HcclResult
RankGraph::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc) const
{
    auto peer = GetPeer(myRank_);
    CHK_PTR_NULL(peer);

    auto ifacesVec = peer->GetIfacesByLayer(layer);
    uint32_t count = 0;

    // 找到 topoInstId 匹配的 iface
    for (const auto& iface : ifacesVec) {
        if (iface->GetTopoInstId() != topoInstId) {
            continue;
        }

        // 对该 iface，从 endpointToIfaceMap_ 中找出所有匹配的 EndpointDesc
        // 一个 iface 可能对应多个 protocol（即多个 EndpointDesc）
        const auto& endpointMap = peer->GetEndpointToIfaceMap();
        for (const auto& entry : endpointMap) {
            std::pair<CommAddr, CommProtocol> endpoint = entry.first;
            const std::shared_ptr<NetInstance::ConnInterface>& mappedIface = entry.second;

            if (mappedIface != iface) {
                continue;
            }

            // 检查输出缓冲区是否足够
            if (count >= *descNum) {
                HCCL_ERROR(
                    "[RankGraph::GetEndpointDesc] endpointDesc array too small: "
                    "need %u, given %u",
                    count + 1, *descNum);
                return HCCL_E_PARA;
            }

            endpointDesc[count].commAddr = endpoint.first;
            endpointDesc[count].protocol = endpoint.second;
            endpointDesc[count].loc.locType = AddrPositionToEndpointLoc(iface->GetPos());

            HCCL_INFO(
                "[RankGraph::GetEndpointDesc] local type is %d, protocol %d", endpointDesc[count].loc.locType,
                endpointDesc[count].protocol);
            count++;
        }
    }

    *descNum = count;
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info) const
{
    if (endpointDesc == nullptr || info == nullptr) {
        HCCL_ERROR("[GetEndpointInfo] Invalid parameter");
        return HCCL_E_PTR;
    }
    const std::shared_ptr<NetInstance::Peer> peer = GetPeer(rankId);
    if (peer == nullptr) {
        HCCL_ERROR("[RankGraph::GetEndpointInfo] Peer is nullptr for rankId [%u]", rankId);
        return HCCL_E_PTR;
    }

    // 查找接口
    auto key = std::make_pair(endpointDesc->commAddr, endpointDesc->protocol);
    const auto& endpointToIfaceMap = peer->GetEndpointToIfaceMap();
    auto it = endpointToIfaceMap.find(key);
    if (it == endpointToIfaceMap.end()) {
        HCCL_ERROR("[GetEndpointInfo] No matching interface found");
        return HCCL_E_NOT_FOUND;
    }
    const auto& iface = it->second;
    // 填充信息
    switch (endpointAttr) {
        case ENDPOINT_ATTR_BW_COEFF: {
            if (infoLen != sizeof(EndpointAttrBwCoeff)) {
                HCCL_ERROR(
                    "[GetEndpointInfo] Size mismatch: expected %zu, actual %u", sizeof(EndpointAttrBwCoeff), infoLen);
                return HCCL_E_PARA;
            }
            *(static_cast<EndpointAttrBwCoeff*>(info)) = iface->GetPorts().size();
            break;
        }
        case ENDPOINT_ATTR_DIE_ID: {
            if (infoLen != sizeof(EndpointAttrDieId)) {
                HCCL_ERROR(
                    "[GetEndpointInfo] Size mismatch: expected %zu, actual %u", sizeof(EndpointAttrDieId), infoLen);
                return HCCL_E_PARA;
            }
            *(static_cast<EndpointAttrDieId*>(info)) = iface->GetLocalDieId();
            HCCL_INFO("GetEndpointInfo rankId[%u] iface[%s]", rankId, iface->Describe().c_str());
            break;
        }
        case ENDPOINT_ATTR_LOCATION: {
            if (infoLen != sizeof(EndpointAttrLocation)) {
                HCCL_ERROR(
                    "[GetEndpointInfo] Size mismatch: expected %zu, actual %u", sizeof(EndpointAttrLocation), infoLen);
                return HCCL_E_PARA;
            }
            *(static_cast<EndpointAttrLocation*>(info)) = iface->GetPos();
            break;
        }
        default: {
            HCCL_ERROR("[GetEndpointInfo] Invalid endpointAttr [%d]", endpointAttr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

bool RankGraph::IsSymmetric(const u32 netLayer) const
{
    if (netInsts_.at(netLayer).size() == 0) {
        THROW<NullPtrException>(
            StringFormat("[RankGraph][IsSymmetric] RankGraph no netInstance on net layer %u", netLayer));
    }
    std::unordered_set<u32> rankSize;
    for (const auto& netInst : netInsts_.at(netLayer)) {
        rankSize.insert(netInst.second->GetRankSize());
    }
    return rankSize.size() == 1;
}

std::shared_ptr<NetInstance> GetOrCreateNetInstance(
    u32 netLayer, const string& netInstId, NetType type, Level2Id2NetInst& netInsts, RankGraph* rankGraph)
{
    std::shared_ptr<NetInstance> netInstance;

    // 若netLayer和netInstId对应netInstance没创建则创建
    if (netInsts[netLayer].count(netInstId) == 0) {
        if (type == NetType::TOPO_FILE_DESC) {
            netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
        } else if (type == NetType::CLOS) {
            netInstance = std::make_shared<ClosNetInstance>(netLayer, netInstId);
        }
        netInsts[netLayer][netInstId] = netInstance;
        // netInstance添加到virtualTopo
        rankGraph->AddNetInstance(netInstance);

        HCCL_DEBUG(
            "[CreateNetInstance] create netInstance success, netLayer[%u] netInstId[%s] type[%s].", netLayer,
            netInstId.c_str(), type.Describe().c_str());
    } else {
        // 若netInstance存在, type不一致则报错
        NetType curType = netInsts[netLayer][netInstId]->GetNetType();
        if (curType != type) {
            HCCL_WARNING(
                "[CreateNetInstance]FabType [%s] and [%s] no match", curType.Describe().c_str(),
                type.Describe().c_str());
            return nullptr;
        }
        // 若netInstance存在, type一致则直接获取
        netInstance = netInsts[netLayer][netInstId];
    }
    return netInstance;
}

void GetNewNodeInfo(
    u32 layer, RankId newRankId, const NetInstance::Link& oldLink, shared_ptr<NetInstance>& newNetInstance,
    RankId2PeerMap& tmpPeers, shared_ptr<NetInstance::Node>& newNode, shared_ptr<NetInstance::ConnInterface>& newIface,
    bool isSource)
{
    shared_ptr<NetInstance::Node> oldNode;
    shared_ptr<NetInstance::ConnInterface> oldIface;
    if (isSource) {
        oldNode = oldLink.GetSourceNode();
        oldIface = oldLink.GetSourceIface();
    } else {
        oldNode = oldLink.GetTargetNode();
        oldIface = oldLink.GetTargetIface();
    }

    NetInstance::Node::NodeType type = oldNode->GetType();
    if (type == NetInstance::Node::NodeType::PEER) {
        newIface = oldIface;
        newNode = tmpPeers.at(newRankId);
        tmpPeers.at(newRankId)->AddConnInterface(layer, newIface);
    } else if (type == NetInstance::Node::NodeType::FABRIC) {
        newIface = nullptr;
        newNode = oldNode;
        if (!newNetInstance->HasNode(newNode->GetNodeId())) {
            newNetInstance->AddNode(newNode);
        }
    } else {
        THROW<NotSupportException>(StringFormat(
            "[CreateSubNetInstances][GetNewNodeInfo] newRankId[%d] oldLink Node isSource[%d] type[%s] "
            "is not supported.",
            newRankId, isSource, type.Describe().c_str()));
    }
}

bool NeedUpdateTopoInstForSubGraph(const NetInstance* oldNetInstance, u32 topoInstId, RankId parentMyRank)
{
    auto topoInstIter = oldNetInstance->topoInsts_.find(topoInstId);
    if (topoInstIter == oldNetInstance->topoInsts_.end()) {
        HCCL_DEBUG(
            "[SubRankGraph][NeedUpdateTopoInstForSubGraph] skip topoInstId[%u], parentMyRank[%d], "
            "not found in parent netInstId[%s]",
            topoInstId, parentMyRank, oldNetInstance->GetNetInstId().c_str());
        return false;
    }
    if (topoInstIter->second == nullptr) {
        HCCL_WARNING(
            "[SubRankGraph][NeedUpdateTopoInstForSubGraph] skip topoInstId[%u], parentMyRank[%u], "
            "topoInst is null in parent netInstId[%s]",
            topoInstId, parentMyRank, oldNetInstance->GetNetInstId().c_str());
        return false;
    }
    return topoInstIter->second->ranks.count(parentMyRank) > 0;
}

void AddNewLink(
    u32 layer, const NetInstance::Link& oldLink, RankId srcNewRankId, RankId dstNewRankId,
    shared_ptr<NetInstance>& newNetInstance, RankId2PeerMap& tmpPeers, const NetInstance* oldNetInstance,
    RankId parentMyRank)
{
    // 不添加绕路link
    if (oldLink.GetHop() > 1 && oldLink.GetType() != LinkType::PEER2NET) {
        return;
    }

    shared_ptr<NetInstance::ConnInterface> newSourceIface;
    shared_ptr<NetInstance::ConnInterface> newTargetIface;
    shared_ptr<NetInstance::Node> newSourceNode;
    shared_ptr<NetInstance::Node> newTargetNode;
    // oldLink有fabricNode需要先addFabricNode
    // SourceNode
    GetNewNodeInfo(layer, srcNewRankId, oldLink, newNetInstance, tmpPeers, newSourceNode, newSourceIface, true);
    // TargetNode
    GetNewNodeInfo(layer, dstNewRankId, oldLink, newNetInstance, tmpPeers, newTargetNode, newTargetIface, false);
    // link
    shared_ptr<NetInstance::Link> link = make_shared<NetInstance::Link>(
        newSourceNode, newTargetNode, newSourceIface, newTargetIface, oldLink.GetType(), oldLink.GetLinkProtocols(),
        oldLink.GetLinkDirection(), oldLink.GetHop());

    newNetInstance->AddLink(link);
    if (newSourceIface != nullptr) {
        u32 sourceTopoInstId = newSourceIface->GetTopoInstId();
        TopoType sourceTopoType = newSourceIface->GetTopoType();
        if (NeedUpdateTopoInstForSubGraph(oldNetInstance, sourceTopoInstId, parentMyRank)) {
            newNetInstance->UpdateTopoInst(sourceTopoInstId, sourceTopoType, srcNewRankId);
        }
    }
    if (newTargetIface != nullptr) {
        u32 targetTopoInstId = newTargetIface->GetTopoInstId();
        TopoType targetTopoType = newTargetIface->GetTopoType();
        if (NeedUpdateTopoInstForSubGraph(oldNetInstance, targetTopoInstId, parentMyRank)) {
            newNetInstance->UpdateTopoInst(targetTopoInstId, targetTopoType, dstNewRankId);
        }
    }

    for (const auto& pair : newNetInstance->topoInsts_) {
        uint32_t topoInstId = pair.first;
        if (pair.second == nullptr) {
            THROW<NullPtrException>(
                StringFormat("[SubRankGraph][AddNewLink] topoInstId %u has no TopoInst", topoInstId));
        }
        auto topoType = pair.second->topoType;
        if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
            HCCL_DEBUG("[SubRankGraph] topoInstId[%u] topoType[%d]", topoInstId, topoType);
        }
    }
    HCCL_DEBUG(
        "[RankGraph][AddNewLink] srcNewRankId[%d] dstNewRankId[%d] newLink[%s]", srcNewRankId, dstNewRankId,
        link->Describe().c_str());
}

void AddGroupLinks(
    const vector<RankId>& rankIds, const NetInstance* oldNetInstance, shared_ptr<NetInstance>& newNetInstance,
    RankId2PeerMap& tmpPeers, RankId parentMyRank)
{
    set<RankId> newRankIds = newNetInstance->GetRankIds();
    u32 layer = newNetInstance->GetNetLayer();
    if (oldNetInstance == nullptr) {
        THROW<NullPtrException>(StringFormat("[AddGroupLinks]oldNetInstance is nullptr"));
    }
    if (newRankIds.size() == 1) {
        // 子通信域单卡场景直接返回1DMESH
        RankId singleId = *newRankIds.begin();
        newNetInstance->UpdateTopoInst(0, TopoType::MESH_1D, singleId);
        return;
    }
    for (RankId srcRankId : newRankIds) {
        for (RankId dstRankId : newRankIds) {
            if (srcRankId == dstRankId) {
                continue;
            }
            // 对oldNetInstance中的每一条Link, 创建新的Link添加到newNetInstance
            vector<NetInstance::Path> oldPaths = oldNetInstance->GetPaths(rankIds[srcRankId], rankIds[dstRankId]);
            for (auto& oldPath : oldPaths) {
                for (auto& oldLink : oldPath.links) {
                    AddNewLink(
                        layer, oldLink, srcRankId, dstRankId, newNetInstance, tmpPeers, oldNetInstance, parentMyRank);
                }
            }
        }
    }
}

void RankGraph::AddSubPeers(const std::vector<RankId>& rankIds, RankGraph* subRankGraph, RankId2PeerMap& peers) const
{
    // 遍历rankIds将索引作为子虚拟拓扑的rankId构造subPeer并添加到subRankGraph
    s32 rankSize = rankIds.size();
    for (RankId subRankId = 0; subRankId < rankSize; ++subRankId) {
        RankId rankId = rankIds[subRankId];
        shared_ptr<NetInstance::Peer> oldPeer = GetPeer(rankId);
        LocalId localId = oldPeer->GetLocalId();
        LocalId replacedLocalId = oldPeer->GetReplacedLocalId();
        DeviceId deviceId = oldPeer->GetDeviceId();
        u32 devicePort = oldPeer->GetDevicePort();
        u32 hostPort = oldPeer->GetHostPort();
        shared_ptr<NetInstance::Peer> subPeer
            = make_shared<NetInstance::Peer>(subRankId, localId, replacedLocalId, deviceId, devicePort, hostPort);
        subRankGraph->AddPeer(subPeer);
        peers.emplace(subRankId, subPeer);
        const auto& oldEndpointMap = oldPeer->GetEndpointToIfaceMap();
        for (const auto& entry : oldEndpointMap) {
            subPeer->SetEndpointToIface(entry.first.first, entry.first.second, entry.second);
            HCCL_DEBUG(
                "[SubRankGraph][AddSubPeers] endpointToIfaceMap: protocol[%d] for subRankId[%d]", entry.first.second,
                subRankId);
        }
        HCCL_DEBUG(
            "[RankGraph][AddSubPeers] oldRankId[%d] subPeer[%s] add success.", rankId, subPeer->Describe().c_str());
    }
}

RankId GetSubRankId(const vector<RankId>& rankIds, RankId rank)
{
    RankId subRank;

    // rankIds中查找rank, 数组索引即为subMyRank
    auto iter = find(begin(rankIds), end(rankIds), rank);
    if (iter != end(rankIds)) {
        subRank = distance(begin(rankIds), iter);
    } else {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][CreateSubVirtTopo] rankIds has no rank[%d].", rank));
    }

    HCCL_DEBUG("[GetSubRank] rank[%d] subRank[%d].", rank, subRank);
    return subRank;
}

/**
 * 1. 创建子NetInstance
 * 2. peer添加对应netInstance
 */
void RankGraph::CreateSubNetInstances(
    const std::vector<RankId> rankIds, Level2Id2NetInst& subNetInstances, RankId2PeerMap& peers,
    RankGraph* subRankGraph) const
{
    // 遍历rankIds, 获取每个rankId所在的oldNetInstance, 创建subNetInstance
    RankId rankSize = rankIds.size();
    for (RankId subRankId = 0; subRankId < rankSize; ++subRankId) {
        set<u32> curLevels = GetLevels(rankIds[subRankId]);
        for (u32 netLayer : curLevels) {
            const NetInstance* oldNetInstance = GetNetInstanceByRankId(netLayer, rankIds[subRankId]);
            if (oldNetInstance == nullptr) {
                THROW<NullPtrException>(StringFormat("[RankGraph][CreateSubNetInstances] oldNetInstance is nullptr"));
            }
            // 创建subNetInstance (根据oldNetInstance.netLayer,id,type)
            NetType netType = oldNetInstance->GetNetType();
            string netInstId = oldNetInstance->GetNetInstId();
            shared_ptr<NetInstance> subNetInstance
                = GetOrCreateNetInstance(netLayer, netInstId, netType, subNetInstances, subRankGraph);
            if (subNetInstance == nullptr) {
                THROW<NullPtrException>(StringFormat("[RankGraph][CreateSubNetInstances] subNetInstance is nullptr"));
            }

            // subNetInstance Add RankId and subPeer
            shared_ptr<NetInstance::Peer> subPeer = peers.at(subRankId);
            subNetInstance->AddRankId(subRankId);
            subNetInstance->AddNode(subPeer);

            // subPeer Add subNetInstance
            subPeer->AddNetInstance(subNetInstance);
            HCCL_DEBUG(
                "[RankGraph][CreateSubNetInstances] subNetInstance subRankId[%d] subType[%s] subNetInstId[%s]",
                subRankId, netType.Describe().c_str(), netInstId.c_str());
        }
    }
}

void RankGraph::AddSubLinks(
    const std::vector<RankId>& rankIds, RankId2PeerMap& peers, Level2Id2NetInst& subNetInsts, RankId parentMyRank) const
{
    // 遍历subNetInstances，对每一个NetInstance插入Links
    for (u32 netLayer = 0; netLayer < subNetInsts.size(); ++netLayer) {
        for (auto& curNetInstance : subNetInsts[netLayer]) {
            const NetInstance* oldNetInstance = GetNetInstanceByNetInstId(netLayer, curNetInstance.first);
            AddGroupLinks(rankIds, oldNetInstance, curNetInstance.second, peers, parentMyRank);
        }
    }
}

unique_ptr<RankGraph> RankGraph::CreateSubRankGraph(const std::vector<u32>& rankIds) const
{
    // 参数类型转换
    vector<RankId> subRankIds;
    for_each(rankIds.begin(), rankIds.end(), [&](u32 rankId) {
        subRankIds.emplace_back(static_cast<RankId>(rankId));
    });

    // 参数检查, 若rankIds中存在当前virtualTopo不存在的rankId, 抛异
    for (const auto rankId : subRankIds) {
        if (!HasRank(rankId)) {
            THROW<InvalidParamsException>(
                StringFormat("[RankGraph][CreateSubVirtTopo] rankId[%d] is not existed.", rankId));
        }
    }

    // step1: 创建subRankGraph
    RankId subMyRankId = GetSubRankId(subRankIds, myRank_);
    unique_ptr<RankGraph> subRankGraph = make_unique<RankGraph>(subMyRankId);

    // step2: subRankGraph添加subPeers
    RankId2PeerMap peers; // 保存Peer指针以便后续执行Add操作
    AddSubPeers(subRankIds, subRankGraph.get(), peers);

    // step3: 构造subNetInstances, NetInstance添加RankId和Peer, Peer添加NetInstance
    Level2Id2NetInst subNetInstances(MAX_NET_LAYER); // 保存NetInstance指针以便后续执行Add操作
    CreateSubNetInstances(subRankIds, subNetInstances, peers, subRankGraph.get());

    // step4: subNetInstances添加Links, Peer添加ConnIfaces
    AddSubLinks(subRankIds, peers, subNetInstances, myRank_);

    // step5: 设置innerRanks
    subRankGraph->InitInnerRanks();
    // step6: 构造完成
    subRankGraph->InitFinish();

    HCCL_INFO("[subRankGraph] Build success!");
    subRankGraph->Dump();
    return subRankGraph;
}

std::vector<char> RankGraph::GetPackedData(const std::vector<std::pair<u32, RankId>>& netLayerRankPairs) const
{
    std::vector<u32> numVec;
    std::vector<LinkData> links;
    u32 netLayerRankPairsNum = netLayerRankPairs.size();

    HCCL_DEBUG("netLayerRankPairs Num=%u", netLayerRankPairsNum);

    for (const auto& it : netLayerRankPairs) {
        auto paths = GetPaths(it.first, myRank_, it.second);
        numVec.push_back(paths.size());
        HCCL_DEBUG("RankGraph::GetPackedData: netLayer=%u, srcRank=%u, dstRank=%u", it.first, myRank_, it.second);
        for (const auto& path : paths) {
            links.emplace_back(path);
            HCCL_DEBUG("RankGraph::GetPackedData: %s", links.back().Describe().c_str());
        }
    }
    if (links.empty()) {
        HCCL_WARNING("[RankGraph][GetPackedData] connected links are empty");
    }

    std::vector<char> result;
    BinaryStream binaryStream;
    u32 linkSize = links.size();
    binaryStream << netLayerRankPairsNum;
    binaryStream << linkSize;
    HCCL_DEBUG("netLayerRankPairsNum=%u, linkSize=%u", netLayerRankPairsNum, linkSize);
    u32 idx = 0;
    for (const auto& it : netLayerRankPairs) {
        binaryStream << it.first;
        binaryStream << it.second;
        binaryStream << numVec[idx];
        HCCL_DEBUG("netLayer=%u, RankId=%u, num=%u", it.first, it.second, numVec[idx]);
        idx++;
    }

    for (const auto& link : links) {
        binaryStream << link.GetUniqueId();
    }
    binaryStream.Dump(result);

    return result;
}

string RankIds2Str(const set<RankId>& rankIds)
{
    stringstream ranks;
    for (auto it = rankIds.begin(); it != rankIds.end(); ++it) {
        if (it != rankIds.begin()) {
            ranks << ", ";
        }
        ranks << *it;
    }
    return ranks.str();
}

void RankGraph::Dump() const
{
    HCCL_DEBUG("RankGraph Dump:");
    HCCL_DEBUG("myRank: %d", myRank_);
    HCCL_DEBUG("innerRanks: [%s]", RankIds2Str(innerRanks_).c_str());
    HCCL_DEBUG("peers:");
    for (const auto& peer : peers_) {
        HCCL_DEBUG("%s", peer.second->Describe().c_str());
    }
    HCCL_DEBUG("netInsts:");
    for (uint32_t i = 0; i < netInsts_.size(); ++i) {
        for (auto& netInst : netInsts_[i]) {
            HCCL_DEBUG("[netLayer=%u, netInstId=%s]", i, netInst.first.c_str());
            HCCL_DEBUG("%s", netInst.second->Describe().c_str());
            HCCL_DEBUG("rankIds: [%s]", RankIds2Str(netInst.second->GetRankIds()).c_str());
            HCCL_DEBUG("peers:");
            for (const auto& peerMapIt : netInst.second->GetPeers()) {
                HCCL_DEBUG("%s", peerMapIt.second->Describe().c_str());
            }
            HCCL_DEBUG("fabrics:");
            for (const auto& fabric : netInst.second->GetFabrics()) {
                HCCL_DEBUG("%s", fabric->Describe().c_str());
            }
            set<NodeId> nodeIds{};
            HCCL_DEBUG("Graph Nodes:");
            netInst.second->GetGraph().TraverseNode([&](shared_ptr<NetInstance::Node> node) {
                nodeIds.insert(node->GetNodeId());
                HCCL_DEBUG("%s", node->Describe().c_str());
            });
            HCCL_DEBUG("Graph Links:");
            for (NodeId nodeId : nodeIds) {
                netInst.second->GetGraph().TraverseEdge(nodeId, [&](shared_ptr<NetInstance::Link> link) {
                    HCCL_DEBUG("%s", link->Describe().c_str());
                });
            }
        }
    }
}
CommProtocol LinkProtocolToCommProtocol(const LinkProtocol& linkProtocol)
{
    constexpr std::pair<LinkProtocol, CommProtocol> protocolPairs[]
        = {{LinkProtocol::UB_CTP, COMM_PROTOCOL_UB_CTP}, {LinkProtocol::UB_TP, COMM_PROTOCOL_UBC_TP},
           {LinkProtocol::ROCE, COMM_PROTOCOL_ROCE},     {LinkProtocol::HCCS, COMM_PROTOCOL_HCCS},
           {LinkProtocol::PCIE, COMM_PROTOCOL_PCIE},     {LinkProtocol::UB_MEM, COMM_PROTOCOL_UB_MEM},
           {LinkProtocol::UBOE, COMM_PROTOCOL_UBOE},     {LinkProtocol::UB_RTP, COMM_PROTOCOL_UB_RTP}};

    for (const auto& p : protocolPairs) {
        if (p.first == linkProtocol) {
            return p.second;
        }
    }

    return COMM_PROTOCOL_RESERVED;
}
} // namespace Hccl
