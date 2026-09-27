/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_graph_interface.h"
#include <set>
#include <functional>
#include <unordered_map>
#include "topo_common_types.h"

namespace Hccl {

HcclResult IRankGraph::GetRankId(uint32_t* rank)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    *rank = rankGraph->GetMyRank();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetRankSize(uint32_t* rankSize)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    *rankSize = rankGraph->GetRankSize();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetDevicePort(const uint32_t rank, uint32_t* devPort)
{
    CHK_PTR_NULL(devPort);
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    auto peer = rankGraph->GetPeer(rank);
    CHK_PTR_NULL(peer);
    *devPort = peer->GetDevicePort();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetHostPort(const uint32_t rank, uint32_t* hostPort)
{
    CHK_PTR_NULL(hostPort);
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    auto peer = rankGraph->GetPeer(rank);
    CHK_PTR_NULL(peer);
    *hostPort = peer->GetHostPort();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetRankGraphInfo(void** graph, uint32_t* len)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    *graph = rankGraph;
    *len = sizeof(RankGraph);
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetDeviceId(uint32_t rankId, uint32_t* deviceId)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    if (rankGraph->GetPeer(rankId) == nullptr) {
        HCCL_ERROR("[GetDeviceId] rankGraph peer is null!");
        return HCCL_E_PTR;
    }
    *deviceId = rankGraph->GetPeer(rankId)->GetDeviceId();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    netLayersVec_.clear();
    netLayersVec_ = std::vector<u32>(levels.begin(), levels.end());
    *netLayers = netLayersVec_.data();
    *netLayerNum = rankGraph->GetLevelNum();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetInstTopoTypeByNetLayer] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    auto type = rankGraph->GetNetType(netLayer);
    static const std::unordered_map<NetType, CommTopo> netTypeMap
        = {{NetType::CLOS, CommTopo::COMM_TOPO_CLOS},
           {NetType::MESH_1D, CommTopo::COMM_TOPO_1DMESH},
           {NetType::A3_SERVER, CommTopo::COMM_TOPO_910_93},
           {NetType::A2_AX_SERVER, CommTopo::COMM_TOPO_A2AXSERVER},
           {NetType::TOPO_FILE_DESC, CommTopo::COMM_TOPO_CUSTOM}};

    auto it = netTypeMap.find(type);
    if (it == netTypeMap.end()) {
        HCCL_ERROR("[GetInstTopoTypeByNetLayer] netType[%s] not in netTypeMap", type.Describe().c_str());
        return HCCL_E_PARA;
    }
    *topoType = it->second;
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetInstSizeByNetLayer] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    u32 num = rankGraph->GetLocalInstSize(netLayer);
    *rankNum = static_cast<uint32_t>(num);
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetInstRanksByNetLayer] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    u32 num = 0;
    rankListVec_.clear();
    rankGraph->GetLocalInstRanks(netLayer, rankListVec_, num);
    *rankList = rankListVec_.data();
    *rankNum = num;
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetInstSizeListByNetLayer] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    u32 size = 0;
    instSizeVec_.clear();
    auto ret = rankGraph->GetNetInstanceList(netLayer, instSizeVec_, size);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[IRankGraph::GetInstSizeListByNetLayer] Failed to get instSizeList at netLayer[%u], "
            "myRank[%u], ret[%d]",
            netLayer, rankId, ret),
        ret);
    *instSizeList = instSizeVec_.data();
    *listSize = size;
    return HCCL_SUCCESS;
}

static HcclResult SetCommAddress(CommAddr& commAddr, const IpAddress& ipAddr)
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
        HCCL_ERROR("[SetCommAddress] invalid commAddrType");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

static HcclResult SetEndpointLoc(EndpointLocType& locType, const AddrPosition& position)
{
    if (position == AddrPosition::DEVICE) {
        locType = ENDPOINT_LOC_TYPE_DEVICE;
    } else if (position == AddrPosition::HOST) {
        locType = ENDPOINT_LOC_TYPE_HOST;
    } else {
        locType = ENDPOINT_LOC_TYPE_RESERVED;
    }
    return HCCL_SUCCESS;
}

static HcclResult InsertInnerLink(const NetInstance::Path& path, std::vector<CommLink>& linkListVec)
{
    for (const auto& link : path.links) {
        const NetInstance::Link* peer2peer = &link;
        for (LinkProtocol protocol : link.GetLinkProtocols()) {
            CommLink commLink;
            CommLinkInit(&commLink, 1);
            const CommProtocol& commProtocol = LinkProtocolToCommProtocol(protocol);
            commLink.linkAttr.linkProtocol = commProtocol;
            commLink.linkAttr.hop = peer2peer->GetHop();
            commLink.srcEndpointDesc.protocol = commProtocol;
            commLink.dstEndpointDesc.protocol = commProtocol;

            // 设置源端点
            std::shared_ptr<NetInstance::ConnInterface> srcConnInterface = link.GetSourceIface();
            CHK_PTR_NULL(srcConnInterface);
            HcclResult result = SetCommAddress(commLink.srcEndpointDesc.commAddr, srcConnInterface->GetAddr());
            if (result != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[IRankGraph::%s] SetCommAddress FAILED for srcConn: %s.", __func__,
                    srcConnInterface->Describe().c_str());
                return result;
            }
            CHK_RET(SetEndpointLoc(commLink.srcEndpointDesc.loc.locType, srcConnInterface->GetPos()));

            // 设置目标端点
            std::shared_ptr<NetInstance::ConnInterface> dstConnInterface = link.GetTargetIface();
            CHK_PTR_NULL(dstConnInterface);
            result = SetCommAddress(commLink.dstEndpointDesc.commAddr, dstConnInterface->GetAddr());
            if (result != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[IRankGraph::%s] SetCommAddress FAILED for dstConn: %s.", __func__,
                    dstConnInterface->Describe().c_str());
                return result;
            }

            CHK_RET(SetEndpointLoc(commLink.dstEndpointDesc.loc.locType, dstConnInterface->GetPos()));

            if (commLink.srcEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                std::shared_ptr<NetInstance::Node> srcNode = peer2peer->GetSourceNode();
                std::shared_ptr<NetInstance::Node> dstNode = peer2peer->GetTargetNode();
                std::shared_ptr<NetInstance::Peer> srcPeer = std::dynamic_pointer_cast<NetInstance::Peer>(srcNode);
                std::shared_ptr<NetInstance::Peer> dstPeer = std::dynamic_pointer_cast<NetInstance::Peer>(dstNode);
                commLink.srcEndpointDesc.loc.device.devPhyId = srcPeer->GetDeviceId();
                commLink.dstEndpointDesc.loc.device.devPhyId = dstPeer->GetDeviceId();
            }

            linkListVec.emplace_back(std::move(commLink));
        }
    }

    return HCCL_SUCCESS;
}

static HcclResult InsertClosLinks(const NetInstance::Path& path, std::vector<CommLink>& linkListVec)
{
    const NetInstance::Link* peer2net = nullptr;
    const NetInstance::Link* net2peer = nullptr;
    for (const auto& link : path.links) {
        bool srcNull = (link.GetSourceIface() == nullptr);
        bool dstNull = (link.GetTargetIface() == nullptr);
        if (!srcNull && dstNull) {
            peer2net = &link;
        } else if (srcNull && !dstNull) {
            net2peer = &link;
        }
    }
    CHK_PTR_NULL(peer2net);
    CHK_PTR_NULL(net2peer);

    auto srcInterface = peer2net->GetSourceIface();
    auto dstInterface = net2peer->GetTargetIface();
    CHK_PTR_NULL(srcInterface);
    CHK_PTR_NULL(dstInterface);
    for (LinkProtocol protocol : peer2net->GetLinkProtocols()) {
        CommLink commLink;
        CommLinkInit(&commLink, 1);
        const CommProtocol& commProtocol = LinkProtocolToCommProtocol(protocol);

        commLink.linkAttr.linkProtocol = commProtocol;
        commLink.linkAttr.hop = peer2net->GetHop();
        commLink.srcEndpointDesc.protocol = commProtocol;
        commLink.dstEndpointDesc.protocol = commProtocol;

        // 设置源端点
        CHK_RET(SetCommAddress(commLink.srcEndpointDesc.commAddr, srcInterface->GetAddr()));
        CHK_RET(SetEndpointLoc(commLink.srcEndpointDesc.loc.locType, srcInterface->GetPos()));
        if (commLink.srcEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            std::shared_ptr<NetInstance::Node> srcNode = peer2net->GetSourceNode();
            std::shared_ptr<NetInstance::Peer> srcPeer = std::dynamic_pointer_cast<NetInstance::Peer>(srcNode);
            commLink.srcEndpointDesc.loc.device.devPhyId = srcPeer->GetDeviceId();
        }

        // 设置目标端点
        CHK_RET(SetCommAddress(commLink.dstEndpointDesc.commAddr, dstInterface->GetAddr()));
        CHK_RET(SetEndpointLoc(commLink.dstEndpointDesc.loc.locType, dstInterface->GetPos()));
        if (commLink.dstEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            std::shared_ptr<NetInstance::Node> dstNode = net2peer->GetTargetNode();
            std::shared_ptr<NetInstance::Peer> dstPeer = std::dynamic_pointer_cast<NetInstance::Peer>(dstNode);
            commLink.dstEndpointDesc.loc.device.devPhyId = dstPeer->GetDeviceId();
        }

        linkListVec.emplace_back(std::move(commLink));
    }
    return HCCL_SUCCESS;
}

HcclResult
IRankGraph::GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetLinks] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    std::vector<NetInstance::Path> paths = rankGraph->GetPaths(netLayer, srcRank, dstRank);
    linkListVec_.clear();
    // 遍历每条path
    for (const auto& path : paths) {
        // 检查是否是Clos网络（有nullptr接口）
        bool isClos = false;
        for (const auto& link : path.links) {
            // fabric没有接口
            if (link.GetSourceIface() == nullptr || link.GetTargetIface() == nullptr) {
                isClos = true;
                break;
            }
        }
        if (!isClos) {
            // Peer2Peer网络：直接处理每条link
            HcclResult ret = InsertInnerLink(path, linkListVec_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[IRankGraph::%s] InsertInnerLink failed for Peer2Peer, linkNum[%zu], ret[%d]", __func__,
                    path.links.size(), ret),
                ret);
        } else {
            // Clos网络：找到peer2net和net2peer，组合成一条链路
            HcclResult ret = InsertClosLinks(path, linkListVec_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[IRankGraph::%s] InsertClosLinks failed for Clos, linkNum[%zu], ret[%d]", __func__,
                    path.links.size(), ret),
                ret);
        }
    }
    *linkList = linkListVec_.data();
    *listSize = linkListVec_.size();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetTopoInstsByLayer] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    u32 num = 0;
    topoInstsVec_.clear();
    rankGraph->GetTopoInstsByLayer(netLayer, topoInstsVec_, num);
    *topoInsts = topoInstsVec_.data();
    *topoInstNum = num;
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetTopoType] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    Hccl::TopoType type;
    HcclResult ret = rankGraph->GetTopoType(netLayer, topoInstId, type);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[IRankGraph::GetTopoType] Failed to get topo type at netLayer [%u] topoInstId [%u] ret[%d]", netLayer,
            topoInstId, ret);
        return ret;
    }
    static const std::unordered_map<Hccl::TopoType, CommTopo> topoTypeMap
        = {{Hccl::TopoType::CLOS, COMM_TOPO_CLOS},
           {Hccl::TopoType::MESH_1D, COMM_TOPO_1DMESH},
           {Hccl::TopoType::A3_SERVER, COMM_TOPO_910_93},
           {Hccl::TopoType::A2_AX_SERVER, COMM_TOPO_A2AXSERVER}};
    auto it = topoTypeMap.find(type);
    if (it != topoTypeMap.end()) {
        *topoType = it->second;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[IRankGraph::GetTopoType] topoType[%s] is not supported, netLayer[%u], "
        "topoInstId[%u], myRank[%u]",
        type.Describe().c_str(), netLayer, topoInstId, rankId);
    return HCCL_E_PARA;
}

HcclResult
IRankGraph::GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetRanksByTopoInst] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    u32 num = 0;
    auto ret = rankGraph->GetRanksByTopoInst(netLayer, topoInstId, ranksVec_, num);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[IRankGraph::GetRanksByTopoInst] Failed to get ranks at netLayer [%u] topoInstId [%u] ret[%d]", netLayer,
            topoInstId, ret);
        return ret;
    }
    *ranks = ranksVec_.data();
    *rankNum = ranksVec_.size();
    return HCCL_SUCCESS;
}

HcclResult IRankGraph::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetEndpointNum] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    auto ret = rankGraph->GetEndpointNum(netLayer, topoInstId, num);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[IRankGraph::GetEndpointNum] Failed to get endpoint num at netLayer[%u], "
            "topoInstId[%u], myRank[%u], ret[%d]",
            netLayer, topoInstId, rankId, ret),
        ret);
    return HCCL_SUCCESS;
}

HcclResult
IRankGraph::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    u32 rankId = rankGraph->GetMyRank();
    std::set<u32> levels = rankGraph->GetLevels(rankId);
    if (levels.find(netLayer) == levels.end()) {
        HCCL_ERROR("[IRankGraph::GetEndpointDesc] netLayer[%u] is invalid", netLayer);
        return HCCL_E_PARA;
    }
    auto ret = rankGraph->GetEndpointDesc(netLayer, topoInstId, descNum, endpointDesc);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[IRankGraph::GetEndpointDesc] Failed to get endpoint desc at netLayer[%u], "
            "topoInstId[%u], myRank[%u], descNum[%u], ret[%d]",
            netLayer, topoInstId, rankId, *descNum, ret),
        ret);
    return HCCL_SUCCESS;
}

static const char* EndpointAttrToString(EndpointAttr endpointAttr)
{
    switch (endpointAttr) {
        case ENDPOINT_ATTR_BW_COEFF:
            return "ENDPOINT_ATTR_BW_COEFF";
        case ENDPOINT_ATTR_DIE_ID:
            return "ENDPOINT_ATTR_DIE_ID";
        case ENDPOINT_ATTR_LOCATION:
            return "ENDPOINT_ATTR_LOCATION";
        default:
            return "ENDPOINT_ATTR_INVALID";
    }
}

HcclResult IRankGraph::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    CHK_PTR_NULL(rankGraphPtr_);
    RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
    HcclResult ret = rankGraph->GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[IRankGraph::GetEndpointInfo] Failed to get endpoint info, rankId[%u], "
            "endpointAttr[%s], ret[%d]",
            rankId, EndpointAttrToString(endpointAttr), ret),
        ret);
    return HCCL_SUCCESS;
}

} // namespace Hccl
