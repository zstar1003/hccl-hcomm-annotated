/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "rank_graph.h"
#include "externalinput_pub.h"
#include "comm_base_pub.h"

namespace hccl {

// 根据 rankId 获取 rank 信息
const RankInfo_t* RankGraphV1::FindRank(uint32_t rankId) const
{
    auto it = rankIndex_.find(rankId);
    if (it == rankIndex_.end()) {
        return nullptr;
    }
    return &(it->second.rankInfo);
}

HcclResult RankGraphV1::DevTypeToCommProtocol(DevType& type, CommProtocol& protocol) const
{
    CHK_RET(hrtGetDeviceType(type));
    switch (type) {
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
        case DevType::DEV_TYPE_910:
            protocol = CommProtocol::COMM_PROTOCOL_ROCE;
            break;
        case DevType::DEV_TYPE_310P1:
        case DevType::DEV_TYPE_310P3:
            protocol = CommProtocol::COMM_PROTOCOL_PCIE;
            break;
        case DevType::DEV_TYPE_NOSOC:
            protocol = CommProtocol::COMM_PROTOCOL_PCIE;
            break;
        case DevType::DEV_TYPE_950:
        case DevType::DEV_TYPE_960:
            // 待扩展UB的协议，当前先不支持
            protocol = CommProtocol::COMM_PROTOCOL_RESERVED;
            break;
        default:
            HCCL_ERROR("[RankGraphV1] Unknown comm devType: %d", type);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult
RankGraphV1::BuildRankGraphInfo(const RankInfo_t& rankItem, const CommProtocol& protocol, RankGraphInfo& outInfo) const
{
    HCCL_INFO(
        "[RankGraphV1][%s] rankId[%u] serverId[%s] serverIdx[%u] superDeviceId[%u] superPodId[%s] "
        "devicePhyId[%d]",
        __func__, rankItem.rankId, rankItem.serverId.c_str(), rankItem.serverIdx, rankItem.superDeviceId,
        rankItem.superPodId.c_str(), rankItem.deviceInfo.devicePhyId);
    outInfo.rankInfo = rankItem;
    std::vector<HcclIpAddress> addrs = rankItem.deviceInfo.deviceIp;
    for (const auto& addr : addrs) {
        EndpointDesc point;
        CHK_RET(static_cast<HcclResult>(EndpointDescInit(&point, 1)));

        // 初始化ROCE协议的基础点位信息
        if (addr.IsIPv6()) {
            point.commAddr.type = COMM_ADDR_TYPE_IP_V6;
            point.commAddr.addr6 = addr.GetBinaryAddress().addr6;
        } else {
            point.commAddr.type = COMM_ADDR_TYPE_IP_V4;
            point.commAddr.addr = addr.GetBinaryAddress().addr;
        }
        point.protocol = protocol;
        if (rankItem.deviceInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
            point.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        } else {
            point.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        }
        point.loc.device.devPhyId = rankItem.deviceInfo.devicePhyId;
        point.loc.device.superDevId = rankItem.superDeviceId;
        point.loc.device.serverIdx = rankItem.serverIdx;
        point.loc.device.superPodIdx = rankItem.superPodIdx;
        // ROCE协议
        outInfo.endPoints.push_back(std::move(point));

        // HCCS 协议
        point.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        if (devType_ == DevType::DEV_TYPE_910B || devType_ == DevType::DEV_TYPE_910_93
            || devType_ == DevType::DEV_TYPE_310P1 || devType_ == DevType::DEV_TYPE_310P3) {
            EndpointDesc hccsPoint = point;
            hccsPoint.protocol = COMM_PROTOCOL_HCCS;
            hccsPoint.commAddr.type = COMM_ADDR_TYPE_ID;
            outInfo.endPoints.push_back(std::move(hccsPoint));
        }

        // PCIE 协议
        if (devType_ == DevType::DEV_TYPE_910B || devType_ == DevType::DEV_TYPE_310P1
            || devType_ == DevType::DEV_TYPE_310P3) {
            EndpointDesc pciePoint = point;
            pciePoint.protocol = COMM_PROTOCOL_PCIE;
            pciePoint.commAddr.type = COMM_ADDR_TYPE_ID;
            outInfo.endPoints.push_back(std::move(pciePoint));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::Init(const RankTable_t& rankTable, const HcclTopoAttr& topoAttr)
{
    rankTable_ = rankTable;
    topoAttr_ = topoAttr;
    rankIndex_.clear();
    rankPairInfo_.clear();
    HCCL_INFO("[RankGraphV1][%s] rankNum[%zu]", __func__, rankTable_.rankList.size());
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_RESERVED;
    CHK_RET(DevTypeToCommProtocol(devType_, protocol));
    // 解析 rankTable，建立 rankId -> RankGraphInfo 映射
    for (const auto& r : rankTable.rankList) {
        RankGraphInfo info;
        CHK_RET(BuildRankGraphInfo(r, protocol, info));
        rankIndex_[r.rankId] = std::move(info);
    }
    rankGraph_ = rankTable_.rankList;
    CHK_RET(InitRankInfo());
    CHK_RET(InitNetLayer());
    CHK_RET(InitHeterogMode());
    HCCL_INFO("[RankGraphV1][%s] Init success", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::Init(const HcclTopoAttr& topoAttr)
{
    topoAttr_ = topoAttr;
    rankIndex_.clear();
    rankPairInfo_.clear();
    HCCL_INFO("[RankGraphV1][%s] rankNum[%zu]", __func__, rankTable_.rankList.size());
    CHK_RET(InitRankInfo());
    CHK_RET(InitNetLayer());
    CHK_RET(InitHeterogMode());
    return HCCL_SUCCESS;
}

bool RankGraphV1::IsRoceInSameServer(uint32_t netLayer, const RankInfo_t& srcInfo, const RankInfo_t& dstInfo)
{
    // 910B单机两种使能RoCE场景：1.A+X 两卡分别在两个MESH  2.标卡
    uint32_t srcPhyId = srcInfo.deviceInfo.devicePhyId;
    uint32_t dstPhyId = dstInfo.deviceInfo.devicePhyId;
    uint32_t intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    HCCL_INFO(
        "[%s] netLayer[%u], devType[%d], srcPhyId[%u], dstPhyId[%u], isStandardCard[%d], isDiffDeviceModule[%d] "
        "IntraRoceSwitch[%u]",
        __func__, netLayer, devType_, srcPhyId, dstPhyId, topoAttr_.isStandardCard, topoAttr_.isDiffDeviceModule,
        intraRoceSwitch);
    const uint32_t deviceMeshDivider = DEVICE_PER_MODULE;
    if (netLayer == HCCL_NETLAYER_1 && devType_ == DevType::DEV_TYPE_910B) {
        bool isSrcInLowerMesh = srcPhyId < deviceMeshDivider;
        bool isDstInLowerMesh = dstPhyId < deviceMeshDivider;
        bool isSrcInUpperMesh = srcPhyId >= deviceMeshDivider;
        bool isDstInUpperMesh = dstPhyId >= deviceMeshDivider;

        // 判定是否为跨MESH（一卡在低区、一卡在高区，匹配A+X跨MESH场景）
        bool isCrossMesh = (isSrcInLowerMesh || isDstInLowerMesh) && (isSrcInUpperMesh || isDstInUpperMesh);
        // 跨MESH或标卡直接满足
        bool isMeetRoceCondition = (isCrossMesh && topoAttr_.isDiffDeviceModule) || topoAttr_.isStandardCard;
        return isMeetRoceCondition && intraRoceSwitch == 1;
    }

    // 非910B的NETLAYER_1场景：仅标卡满足条件时取外部配置，否则返回false
    return topoAttr_.isStandardCard && intraRoceSwitch == 1 && netLayer == HCCL_NETLAYER_1;
}

CommProtocol RankGraphV1::GetCommProtocolInSameServer(const RankInfo_t& srcInfo, const RankInfo_t& dstInfo)
{
    // 310P间链路为PCIE或HCCS
    LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
    hrtGetPairDeviceLinkType(srcInfo.deviceInfo.devicePhyId, dstInfo.deviceInfo.devicePhyId, linkType);
    HCCL_INFO(
        "[RankGraphV1][%s] ranks[%u,%u] intra-server linkType[%d]", __func__, srcInfo.rankId, dstInfo.rankId, linkType);
    if (linkType == LinkTypeInServer::HCCS_TYPE || linkType == LinkTypeInServer::HCCS_SW_TYPE) {
        return CommProtocol::COMM_PROTOCOL_HCCS;
    } else if (linkType == LinkTypeInServer::SIO_TYPE) {
        return CommProtocol::COMM_PROTOCOL_SIO;
    } else if (linkType == LinkTypeInServer::PXI_TYPE) {
        bool isDiffDeviceModule = (topoAttr_.isDiffDeviceModule && devType_ == DevType::DEV_TYPE_910B);
        bool isRankModEqual = (srcInfo.rankId % DEVICE_PER_MODULE == dstInfo.rankId % DEVICE_PER_MODULE);
        bool isMeetPxiCondition = (!isDiffDeviceModule) || (isDiffDeviceModule && isRankModEqual);
        return isMeetPxiCondition ? CommProtocol::COMM_PROTOCOL_PCIE : CommProtocol::COMM_PROTOCOL_RESERVED;
    }
    return CommProtocol::COMM_PROTOCOL_RESERVED;
}

CommProtocol RankGraphV1::GetCommProtocolBetweenServers(const RankInfo_t& srcInfo, const RankInfo_t& dstInfo) const
{
    // srcInfo与dstInfo一定是相同数据类型
    if (devType_ == DevType::DEV_TYPE_310P3 || devType_ == DevType::DEV_TYPE_310P1) {
        return CommProtocol::COMM_PROTOCOL_PCIE;
    }
    if (devType_ == DevType::DEV_TYPE_910B) {
        return CommProtocol::COMM_PROTOCOL_ROCE;
    }
    HCCL_DEBUG(
        "[%s] srcInfo.superPodId %s dstInfo.superPodId %s", __func__, srcInfo.superPodId.c_str(),
        dstInfo.superPodId.c_str());
    if (devType_ == DevType::DEV_TYPE_910_93) {
        // 超节点内链路为HCCS
        if (!srcInfo.superPodId.empty() && srcInfo.superPodId == dstInfo.superPodId) {
            return CommProtocol::COMM_PROTOCOL_HCCS;
        }
    }
    return CommProtocol::COMM_PROTOCOL_RESERVED;
}

CommProtocol
RankGraphV1::GetCommProtocolFromRankInfo(const RankInfo_t& srcInfo, const RankInfo_t& dstInfo, uint32_t netLayer)
{
    if (srcInfo.deviceInfo.deviceType != dstInfo.deviceInfo.deviceType) {
        HCCL_ERROR(
            "[RankGraphV1][%s] srcType[%d] != dstType[%d]", __func__, srcInfo.deviceInfo.deviceType,
            dstInfo.deviceInfo.deviceType);
        return CommProtocol::COMM_PROTOCOL_RESERVED;
    }
    // 首先判断是否在同一机内
    if (srcInfo.serverIdx == dstInfo.serverIdx) {
        if (netLayer == HCCL_NETLAYER_0) {
            return GetCommProtocolInSameServer(srcInfo, dstInfo);
            // 超节点有HCCL_NETLAYER_1及以上的情况，为HCCS链路，或者同卡不同DIE
        } else if (
            netLayer == HCCL_NETLAYER_1 && devType_ == DevType::DEV_TYPE_910_93
            && (srcInfo.superPodId == dstInfo.superPodId
                || GetCommProtocolInSameServer(srcInfo, dstInfo) == CommProtocol::COMM_PROTOCOL_SIO)) {
            return CommProtocol::COMM_PROTOCOL_HCCS;
        } else if (IsRoceInSameServer(netLayer, srcInfo, dstInfo)) {
            return CommProtocol::COMM_PROTOCOL_ROCE;
        } else {
            // 接了交换机才会有HCCL_NETLAYER_1及以上的情况，当前无法判断是否连接交换机，接了交换机走RDMA
            return CommProtocol::COMM_PROTOCOL_RESERVED;
        }
    }
    if (srcInfo.serverIdx != dstInfo.serverIdx) {
        if (netLayer == HCCL_NETLAYER_0) {
            HCCL_INFO("[RankGraphV1][%s] ranks[%u,%u] not in same server", __func__, srcInfo.rankId, dstInfo.rankId);
            return CommProtocol::COMM_PROTOCOL_RESERVED;
        }
        if (netLayer == HCCL_NETLAYER_1) {
            HCCL_INFO(
                "[RankGraphV1][%s] ranks[%u,%u] inter-server but same superPod[%s]", __func__, srcInfo.rankId,
                dstInfo.rankId, srcInfo.superPodId.c_str());
            return GetCommProtocolBetweenServers(srcInfo, dstInfo);
            // 跨超走ROCE
        } else if (
            !srcInfo.superPodId.empty() && srcInfo.superPodId != dstInfo.superPodId && netLayer == HCCL_NETLAYER_2) {
            HCCL_INFO(
                "[RankGraphV1][%s] ranks[%u,%u] inter-superPod use ROCE", __func__, srcInfo.rankId, dstInfo.rankId);
            return CommProtocol::COMM_PROTOCOL_ROCE;
        }
    }
    return CommProtocol::COMM_PROTOCOL_RESERVED;
}

bool RankGraphV1::NeedIgnoreEndPoints(
    CommProtocol srcProtocol, CommProtocol dstProtocol, CommProtocol linkProtocol) const
{
    if (srcProtocol != dstProtocol) {
        return true;
    } else {
        // 两个hccs endpoints间可能是SIO链路
        // A + X 两个mesh间是PCIE链路, 310DUO卡两个DIE间链路是HCCS，主次DIE间是PCIE
        if (srcProtocol == COMM_PROTOCOL_HCCS && dstProtocol == COMM_PROTOCOL_HCCS
            && linkProtocol == COMM_PROTOCOL_SIO) {
            return false;
        } else if (dstProtocol != linkProtocol) {
            return true;
        }
    }
    return false;
}

void RankGraphV1::PrintLinksInfo(CommLink& link) const
{
    // 打印CommLink 头部基础信息
    HCCL_INFO(
        "[RankGraphV1][%s] link.header.version[%u] magicWord[0x%08x] size[%u] reserved[%u]", __func__,
        link.header.version, link.header.magicWord, link.header.size, link.header.reserved);

    // 打印【源端】srcEndpointDesc 完整信息
    HCCL_INFO(
        "[RankGraphV1][%s] srcProtocol[%d] srcCommAddrType[%d] srcLocType[%d] srcDevPhyId[%u] "
        "srcSuperDevId[%u] srcServerIdx[%u] srcSuperPodIdx[%u]",
        __func__, link.srcEndpointDesc.protocol, link.srcEndpointDesc.commAddr.type, link.srcEndpointDesc.loc.locType,
        link.srcEndpointDesc.loc.device.devPhyId, link.srcEndpointDesc.loc.device.superDevId,
        link.srcEndpointDesc.loc.device.serverIdx, link.srcEndpointDesc.loc.device.superPodIdx);

    // 打印【目的端】dstEndpointDesc 完整信息
    HCCL_INFO(
        "[RankGraphV1][%s] dstProtocol[%d] dstCommAddrType[%d] dstLocType[%d] dstDevPhyId[%u] "
        "dstSuperDevId[%u] dstServerIdx[%u] dstSuperPodIdx[%u]",
        __func__, link.dstEndpointDesc.protocol, link.dstEndpointDesc.commAddr.type, link.dstEndpointDesc.loc.locType,
        link.dstEndpointDesc.loc.device.devPhyId, link.dstEndpointDesc.loc.device.superDevId,
        link.dstEndpointDesc.loc.device.serverIdx, link.dstEndpointDesc.loc.device.superPodIdx);

    // 打印【链路属性】linkAttr 信息
    HCCL_INFO("[RankGraphV1][%s] linkProtocol[%d] hop[%u]", __func__, link.linkAttr.linkProtocol, link.linkAttr.hop);
}

HcclResult
RankGraphV1::GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    if (rankIndex_.find(srcRank) == rankIndex_.end() || rankIndex_.find(dstRank) == rankIndex_.end()
        || FindRank(srcRank) == nullptr || FindRank(dstRank) == nullptr) {
        HCCL_ERROR(
            "[RankGraphV1][%s] srcRank[%u] or dstRank[%u] is not existed in rankTable", __func__, srcRank, dstRank);
        return HCCL_E_PARA;
    }

    if (netLayer > HCCL_NETLAYER_2) {
        HCCL_ERROR(
            "[RankGraphV1][%s] srcRank[%u] and dstRank[%u] do not have netLayer[%u]", __func__, srcRank, dstRank,
            netLayer);
        return HCCL_E_PARA;
    }
    auto& srcEndpointDescs = rankIndex_[srcRank].endPoints;
    auto& dstEndpointDescs = rankIndex_[dstRank].endPoints;

    const RankInfo_t& srcInfo = rankIndex_[srcRank].rankInfo;
    const RankInfo_t& dstInfo = rankIndex_[dstRank].rankInfo;
    CommProtocol protocol = COMM_PROTOCOL_RESERVED;
    protocol = GetCommProtocolFromRankInfo(srcInfo, dstInfo, netLayer);
    if (protocol == COMM_PROTOCOL_RESERVED) {
        HCCL_WARNING("[RankGraphV1][%s] no links between srcRank[%u] dstRank[%u]", __func__, srcRank, dstRank);
        *linkList = nullptr;
        *listSize = 0;
        return HCCL_SUCCESS;
    }

    // 1. 查询是否有缓存CommLink信息
    auto key = std::make_tuple(netLayer, srcRank, dstRank);
    auto it = rankPairInfo_.find(key);
    if (it == rankPairInfo_.end()) {
        // 没有则创建
        HCCL_INFO("[RankGraphV1][%s] no cached links, build new srcRank[%u] dstRank[%u]", __func__, srcRank, dstRank);
        std::vector<CommLink> links;
        for (size_t i = 0; i < srcEndpointDescs.size(); i++) {
            for (size_t j = 0; j < dstEndpointDescs.size(); j++) {
                if (NeedIgnoreEndPoints(srcEndpointDescs[i].protocol, dstEndpointDescs[j].protocol, protocol)) {
                    continue;
                }
                CommLink link;
                CHK_RET(CommLinkInit(&link, 1));

                link.srcEndpointDesc = srcEndpointDescs[i];
                link.srcEndpointDesc.protocol = protocol;
                link.dstEndpointDesc = dstEndpointDescs[j];
                link.dstEndpointDesc.protocol = protocol;
                link.linkAttr.linkProtocol = protocol;
                PrintLinksInfo(link);
                links.push_back(std::move(link));
            }
        }
        it = rankPairInfo_.emplace(std::make_tuple(netLayer, srcRank, dstRank), std::move(links)).first;
    }
    HCCL_INFO(
        "[RankGraphV1][%s] links, netLayer[%u] srcRank[%u] dstRank[%u] protocol[%d]", __func__, netLayer, srcRank,
        dstRank, protocol);

    auto& links = it->second;
    *listSize = static_cast<uint32_t>(links.size());
    if (links.empty()) {
        *linkList = nullptr;
        HCCL_ERROR("[RankGraphV1][%s] links empty for srcRank[%u] dstRank[%u]", __func__, srcRank, dstRank);
    } else {
        *linkList = links.data(); // 连续数组首地址
        HCCL_INFO(
            "[RankGraphV1][%s] srcRank[%u] dstRank[%u] linkList[%p] linkNum[%u]", __func__, srcRank, dstRank, *linkList,
            *listSize);
    }

    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::InitHeterogMode()
{
    if (topoAttr_.rankInfoList.empty()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. rankInfoList is empty", __func__);
        return HCCL_E_INTERNAL;
    }

    std::set<DevType> devTypes;
    for (u32 index = 0; index < topoAttr_.rankInfoList.size(); index++) {
        devTypes.insert(topoAttr_.rankInfoList[index].deviceType);
    }

    // 只包含一种芯片的同构组网
    if (devTypes.size() == 1) {
        heterogMode_ = HcclHeterogMode::HCCL_HETEROG_MODE_HOMOGENEOUS;
        return HCCL_SUCCESS;
    }

    // 包含两种芯片的异构混合组网
    constexpr uint32_t MIX_CHIPS = 2;
    if (devTypes.size() == MIX_CHIPS && devTypes.find(DevType::DEV_TYPE_910B) != devTypes.end()
        && devTypes.find(DevType::DEV_TYPE_910_93) != devTypes.end()) {
        heterogMode_ = HcclHeterogMode::HCCL_HETEROG_MODE_MIX_A2_A3;
        return HCCL_SUCCESS;
    }

    std::string devStr;
    for (auto itSet = devTypes.begin(); itSet != devTypes.end(); itSet++) {
        if (itSet != devTypes.begin()) {
            devStr += ", ";
        }
        devStr += std::to_string(static_cast<int>(*itSet));
    }
    HCCL_ERROR(
        "[RankGraphV1][%s] Unknown mode[%d], devtypes[%s]", __func__, HcclHeterogMode::HCCL_HETEROG_MODE_INVALID,
        devStr.c_str());
    return HCCL_E_INTERNAL;
}

HcclResult RankGraphV1::GetHeterogMode(HcclHeterogMode* mode) const
{
    *mode = heterogMode_;
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    if (netLayer_.empty()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. netLayer is empty", __func__);
        return HCCL_E_INTERNAL;
    }
    *netLayers = netLayer_.data();
    *netLayerNum = netLayer_.size();
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType)
{
    if (netLayer >= netLayer_.size()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. netlayer[%u]", __func__, netLayer);
        return HCCL_E_PARA;
    }
    DevType deviceType = topoAttr_.deviceType;
    if (deviceType == DevType::DEV_TYPE_910_93) {
        if (netLayer == static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L0)) {
            *topoType = CommTopo::COMM_TOPO_910_93;
        } else if ((netLayer == static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L1)
                    || (netLayer == static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L2)))) {
            *topoType = CommTopo::COMM_TOPO_CLOS;
        }
    } else if (deviceType == DevType::DEV_TYPE_910B || deviceType == DevType::DEV_TYPE_910) {
        if (netLayer == static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L0)) {
            *topoType = CommTopo::COMM_TOPO_1DMESH;
        } else if (netLayer == static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L1)) {
            *topoType = CommTopo::COMM_TOPO_CLOS;
        }
    } else if (deviceType == DevType::DEV_TYPE_310P3) {
        if (netLayer == static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L0)) {
            *topoType = CommTopo::COMM_TOPO_310P;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    if (netLayer >= netLayer_.size()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. netlayer[%u]", __func__, netLayer);
        return HCCL_E_PARA;
    }

    if (rankList_.find(netLayer) == rankList_.end()) {
        HCCL_ERROR("[RankGraphV1][%s] failed to find rankList map. netlayer[%u]", __func__, netLayer);
        return HCCL_E_INTERNAL;
    }
    *rankNum = rankList_[netLayer].size();

    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum)
{
    if (netLayer >= netLayer_.size()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. netlayer[%u]", __func__, netLayer);
        return HCCL_E_PARA;
    }

    if (rankList_.find(netLayer) == rankList_.end()) {
        HCCL_ERROR("[RankGraphV1][%s] failed to find rankList map. netlayer[%u]", __func__, netLayer);
        return HCCL_E_INTERNAL;
    }
    *rankNum = rankList_[netLayer].size();
    *rankList = rankList_[netLayer].data();

    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    if (netLayer >= netLayer_.size()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. netlayer[%u]", __func__, netLayer);
        return HCCL_E_PARA;
    }

    if (rankSizeList_.find(netLayer) == rankSizeList_.end()) {
        HCCL_ERROR("[RankGraphV1][%s] failed to find rankSizeList map. netlayer[%u]", __func__, netLayer);
        return HCCL_E_INTERNAL;
    }
    *instSizeList = rankSizeList_[netLayer].data();
    *listSize = rankSizeList_[netLayer].size();

    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    if (netLayer >= netLayer_.size()) {
        return HCCL_E_PARA;
    }
    if (rankSizeList_.find(netLayer) == rankSizeList_.end()) {
        return HCCL_E_INTERNAL;
    }

    uint32_t instNum = rankSizeList_[netLayer].size();
    *topoInstNum = instNum;

    topoInstsVec_.clear();
    topoInstsVec_.resize(instNum);
    for (uint32_t i = 0; i < instNum; ++i) {
        topoInstsVec_[i] = i;
    }

    *topoInsts = topoInstsVec_.data();

    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetTopoType(uint32_t netLayer, CommTopo* topoType)
{
    if (netLayer >= netLayer_.size()) {
        return HCCL_E_PARA;
    }
    return GetInstTopoTypeByNetLayer(netLayer, topoType);
}

HcclResult RankGraphV1::GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    if (netLayer >= netLayer_.size()) {
        return HCCL_E_PARA;
    }
    if (devType_ != DevType::DEV_TYPE_910B) {
        return HCCL_E_NOT_SUPPORT;
    }
    auto rankListIt = rankList_.find(netLayer);
    auto rankSizeListIt = rankSizeList_.find(netLayer);
    if (rankListIt != rankList_.end() && rankSizeListIt != rankSizeList_.end()) {
        if (topoInstId >= rankSizeListIt->second.size()) {
            return HCCL_E_PARA;
        }
    }

    *ranks = rankListIt->second.data();
    if (topoInstId < rankSizeListIt->second.size()) {
        *rankNum = rankSizeListIt->second[topoInstId];
    } else {
        *rankNum = 0;
    }

    return HCCL_SUCCESS;
}

std::vector<const RankInfo_t*> RankGraphV1::GetRanksInTopoInst(uint32_t netLayer, uint32_t topoInstId)
{
    std::vector<const RankInfo_t*> ranks;

    if (netLayer >= netLayer_.size()) {
        return ranks;
    }
    if (netLayer == static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L0)) {
        auto it = serverToRank_.find(topoInstId);
        if (it == serverToRank_.end()) {
            return ranks;
        }
        for (const auto& rankInfo : it->second) {
            for (const auto& graphInfo : rankGraph_) {
                if (graphInfo.rankId == rankInfo.userRank) {
                    ranks.push_back(&graphInfo);
                }
            }
        }
    } else {
        auto listIt = rankList_.find(netLayer);
        auto sizeListIt = rankSizeList_.find(netLayer);
        if (listIt != rankList_.end() && sizeListIt != rankSizeList_.end()) {
            if (topoInstId >= sizeListIt->second.size()) {
                return ranks;
            }
        }

        for (uint32_t userRank : listIt->second) {
            for (const auto& rankInfo : rankGraph_) {
                if (rankInfo.rankId == userRank) {
                    ranks.push_back(&rankInfo);
                    break;
                }
            }
        }
    }
    return ranks;
}

std::set<CommProtocol>
RankGraphV1::GetProtocolsByConnections(uint32_t netLayer, const std::vector<const RankInfo_t*>& topoInstRanks)
{
    std::set<CommProtocol> protocols;

    const RankInfo_t* srcRankInfo = nullptr;
    for (const auto& rankInfo : rankGraph_) {
        if (rankInfo.rankId == rankData_.userRank) {
            srcRankInfo = &rankInfo;
            break;
        }
    }
    if (srcRankInfo == nullptr) {
        return protocols;
    }

    for (const RankInfo_t* dstRankInfo : topoInstRanks) {
        if (dstRankInfo->rankId == rankData_.userRank) {
            continue;
        }

        CommProtocol protocol = GetCommProtocolFromRankInfo(*srcRankInfo, *dstRankInfo, netLayer);
        if (protocol != COMM_PROTOCOL_RESERVED) {
            protocols.insert(protocol);
        }
    }

    return protocols;
}

HcclResult RankGraphV1::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num)
{
    if (netLayer >= netLayer_.size()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. netlayer[%u]", __func__, netLayer);
        return HCCL_E_PARA;
    }
    if (rankIndex_.empty()) {
        HCCL_ERROR("[RankGraphV1][%s] rankIndex is empty", __func__);
        return HCCL_E_INTERNAL;
    }
    std::vector<const RankInfo_t*> topoInstRanks = GetRanksInTopoInst(netLayer, topoInstId);
    if (topoInstRanks.empty()) {
        *num = 0;
        return HCCL_SUCCESS;
    }

    std::set<CommProtocol> protocols = GetProtocolsByConnections(netLayer, topoInstRanks);
    if (protocols.empty()) {
        HCCL_INFO(
            "[RankGraphV1][%s] no protocols found for netlayer [%u] topoInstId[%u]", __func__, netLayer, topoInstId);
    }

    const RankGraphInfo* currentRankInfo = nullptr;
    for (auto& pair : rankIndex_) {
        if (pair.second.rankInfo.rankId == rankData_.userRank) {
            currentRankInfo = &pair.second;
            break;
        }
    }
    if (currentRankInfo == nullptr) {
        return HCCL_E_INTERNAL;
    }

    uint32_t count = 0;
    for (const auto& endpoint : currentRankInfo->endPoints) {
        if (protocols.find(endpoint.protocol) != protocols.end()) {
            count++;
        }
    }
    *num = count;

    return HCCL_SUCCESS;
}

HcclResult
RankGraphV1::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    if (netLayer >= netLayer_.size()) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. netlayer[%u]", __func__, netLayer);
        return HCCL_E_PARA;
    }
    if (descNum == nullptr || endpointDesc == nullptr) {
        HCCL_ERROR("[RankGraphV1][%s] invalid para. null ptr", __func__);
        return HCCL_E_PARA;
    }
    if (rankIndex_.empty()) {
        HCCL_ERROR("[RankGraphV1][%s] rankIndex is empty", __func__);
        return HCCL_E_INTERNAL;
    }
    std::vector<const RankInfo_t*> topoInstRanks = GetRanksInTopoInst(netLayer, topoInstId);
    if (topoInstRanks.empty()) {
        *descNum = 0;
        return HCCL_SUCCESS;
    }

    std::set<CommProtocol> protocols = GetProtocolsByConnections(netLayer, topoInstRanks);
    if (protocols.empty()) {
        HCCL_INFO(
            "[RankGraphV1][GetEndpointDesc] no protocols found, netlayer [%u] topoInstId[%u]", netLayer, topoInstId);
    }

    const RankGraphInfo* currRankInfo = nullptr;
    for (auto& pair : rankIndex_) {
        if (pair.second.rankInfo.rankId == rankData_.userRank) {
            currRankInfo = &pair.second;
            break;
        }
    }
    if (currRankInfo == nullptr) {
        return HCCL_E_INTERNAL;
    }

    uint32_t count = 0;
    for (const auto& endpoint : currRankInfo->endPoints) {
        if (protocols.find(endpoint.protocol) != protocols.end()) {
            if (count >= *descNum) {
                return HCCL_E_PARA;
            }
            endpointDesc[count] = endpoint;
            count++;
        }
    }
    *descNum = count;

    return HCCL_SUCCESS;
}

const EndpointDesc*
RankGraphV1::MatchEndpointByAddr(const RankGraphInfo& rankGraphInfo, const EndpointDesc* endPointDesc) const
{
    for (const auto& endpoint : rankGraphInfo.endPoints) {
        if (endpoint.commAddr.type != endPointDesc->commAddr.type || endpoint.protocol != endPointDesc->protocol) {
            continue;
        }
        bool matched = false;
        switch (endpoint.commAddr.type) {
            case COMM_ADDR_TYPE_IP_V4:
                matched
                    = (memcmp(&endpoint.commAddr.addr, &endPointDesc->commAddr.addr, sizeof(endpoint.commAddr.addr))
                       == 0);
                break;
            case COMM_ADDR_TYPE_IP_V6:
                matched
                    = (memcmp(&endpoint.commAddr.addr6, &endPointDesc->commAddr.addr6, sizeof(endpoint.commAddr.addr6))
                       == 0);
                break;
            case COMM_ADDR_TYPE_ID:
                matched = true;
                break;
            default:
                break;
        }
        if (matched) {
            return &endpoint;
        }
    }
    return nullptr;
}

HcclResult
RankGraphV1::FillAttr(EndpointAttr endpointAttr, const EndpointDesc* foundEndpoint, uint32_t infoLen, void* info) const
{
    switch (endpointAttr) {
        case ENDPOINT_ATTR_BW_COEFF: {
            CHK_PRT_RET(
                infoLen != sizeof(EndpointAttrBwCoeff),
                HCCL_ERROR(
                    "[RankGraphV1::GetEndpointInfo] Size mismatch for ENDPOINT_ATTR_BW_COEFF: expected %zu, actual %u",
                    sizeof(EndpointAttrBwCoeff), infoLen),
                HCCL_E_PARA);
            *(static_cast<EndpointAttrBwCoeff*>(info)) = 1;
            break;
        }
        case ENDPOINT_ATTR_DIE_ID: {
            CHK_PRT_RET(
                infoLen != sizeof(EndpointAttrDieId),
                HCCL_ERROR(
                    "[RankGraphV1::GetEndpointInfo] Size mismatch for ENDPOINT_ATTR_DIE_ID: expected %zu, actual %u",
                    sizeof(EndpointAttrDieId), infoLen),
                HCCL_E_PARA);
            *(static_cast<EndpointAttrDieId*>(info)) = foundEndpoint->loc.device.superDevId;
            break;
        }
        case ENDPOINT_ATTR_LOCATION: {
            CHK_PRT_RET(
                infoLen != sizeof(EndpointAttrLocation),
                HCCL_ERROR(
                    "[RankGraphV1::GetEndpointInfo] Size mismatch for ENDPOINT_ATTR_LOCATION: expected %zu, actual %u",
                    sizeof(EndpointAttrLocation), infoLen),
                HCCL_E_PARA);
            *(static_cast<EndpointAttrLocation*>(info)) = foundEndpoint->loc.locType;
            break;
        }
        default: {
            HCCL_ERROR("[RankGraphV1::GetEndpointInfo] Invalid endpointAttr[%d]", endpointAttr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    if (endPointDesc == nullptr || info == nullptr) {
        HCCL_ERROR("[RankGraphV1::GetEndpointInfo] Invalid parameter, null pointer");
        return HCCL_E_PTR;
    }

    if (rankIndex_.empty()) {
        HCCL_ERROR("[RankGraphV1::GetEndpointInfo] rankIndex is empty");
        return HCCL_E_INTERNAL;
    }

    auto rankIt = rankIndex_.find(rankId);
    if (rankIt == rankIndex_.end()) {
        HCCL_ERROR("[RankGraphV1::GetEndpointInfo] rankId[%u] not found in rankIndex", rankId);
        return HCCL_E_NOT_FOUND;
    }

    const RankGraphInfo& rankGraphInfo = rankIt->second;
    const EndpointDesc* foundEndpoint = MatchEndpointByAddr(rankGraphInfo, endPointDesc);
    if (foundEndpoint == nullptr) {
        HCCL_ERROR("[RankGraphV1::GetEndpointInfo] No matching endpoint found for rankId[%u]", rankId);
        return HCCL_E_NOT_FOUND;
    }

    return FillAttr(endpointAttr, foundEndpoint, infoLen, info);
}

HcclResult RankGraphV1::GetRankSize(uint32_t* rankSize)
{
    CHK_PTR_NULL(rankSize);
    *rankSize = rankGraph_.size();
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetDevicePort(const uint32_t rank, uint32_t* devPort)
{
    CHK_PTR_NULL(devPort);
    const RankInfo_t* rankInfo = FindRank(rank);
    if (rankInfo == nullptr) {
        HCCL_ERROR("[RankGraphV1][%s] rank[%u] not found", __func__, rank);
        return HCCL_E_PARA;
    }
    *devPort = rankInfo->deviceInfo.port;
    return HCCL_SUCCESS;
}

bool RankGraphSort(const RankInfo& first, const RankInfo& second)
{
    if (first.serverIdx != second.serverIdx) {
        return first.serverIdx < second.serverIdx;
    } else {
        return first.userRank < second.userRank;
    }
}

HcclResult RankGraphV1::InitGraphRankInfo()
{
    for (u32 index = 0; index < rankGraph_.size(); index++) {
        struct GraphRankInfo graphRankInfo = {};
        graphRankInfo.rankId = rankGraph_[index].rankId;
        graphRankInfo.localRank = rankGraph_[index].localRank;
        graphRankInfo.serverId = rankGraph_[index].serverId;
        graphRankInfo.serverIdx = rankGraph_[index].serverIdx;
        graphRankInfo.superDeviceId = rankGraph_[index].superDeviceId;
        graphRankInfo.superPodId = rankGraph_[index].superPodId;
        graphRankInfo.superPodIdx = rankGraph_[index].superPodIdx;
        graphRankInfo.hostPort = rankGraph_[index].hostPort;
        graphRankInfo.nodeId = rankGraph_[index].nodeId;
        graphRankInfo.itemId = rankGraph_[index].itemId;
        graphRankInfo.deviceInfo.devicePhyId = rankGraph_[index].deviceInfo.devicePhyId;
        graphRankInfo.deviceInfo.deviceType = rankGraph_[index].deviceInfo.deviceType;
        graphRankInfo.deviceInfo.port = rankGraph_[index].deviceInfo.port;
        graphRankInfo.deviceInfo.vnicPort = rankGraph_[index].deviceInfo.vnicPort;
        graphRankInfo.deviceInfo.backupPort = rankGraph_[index].deviceInfo.backupPort;
        graphRankInfo.bindDeviceId = rankGraph_[index].bindDeviceId;
        graphRankInfo.originalSuperPodId = rankGraph_[index].originalSuperPodId;

        graphRankInfo_.push_back(graphRankInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetRankGraphInfo(GraphType type, void** graph, uint32_t* len)
{
    switch (type) {
        case RANK_GRAPH_910_93: {
            *graph = graphRankInfo_.data();
            *len = graphRankInfo_.size() * sizeof(GraphRankInfo);
            break;
        }
        default: {
            HCCL_ERROR("[RankGraphV1][%s]Graph type[%d] is invalid", __func__, type);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::GetDeviceId([[maybe_unused]] uint32_t rankId, [[maybe_unused]] uint32_t* deviceId)
{
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::InitRankInfo()
{
    auto& rankInfoList = topoAttr_.rankInfoList;
    for (u32 index = 0; index < rankInfoList.size(); index++) {
        if (topoAttr_.userRank == rankInfoList[index].userRank) {
            rankData_ = rankInfoList[index];
            break;
        }
    }
    CHK_RET(InitServerRankInfo());
    CHK_RET(InitSuperPodRankInfo());
    CHK_RET(InitGraphRankInfo());
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::InitServerRankInfo()
{
    u32 serverIdx = 0;
    auto& rankInfoList = topoAttr_.rankInfoList;
    for (u32 index = 0; index < rankInfoList.size(); index++) {
        serverIdx = rankInfoList[index].serverIdx;
        auto itServer = serverToRank_.find(serverIdx);
        if (itServer != serverToRank_.end()) {
            itServer->second.push_back(rankInfoList[index]);
        } else {
            std::vector<RankInfo> rankVecTmp;
            rankVecTmp.push_back(rankInfoList[index]);
            serverToRank_.insert(std::make_pair(serverIdx, rankVecTmp));
        }
    }
    // 调整每个server内的user_rank排序(server内userRank从小到大,一定连续)
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            std::sort(iterMap->second.begin(), iterMap->second.end(), RankGraphSort);
        }
    }
    serverIdx = rankData_.serverIdx;
    auto rankVec = serverToRank_.find(serverIdx);
    if (rankVec != serverToRank_.end()) {
        std::string rankIdListServer;
        for (auto iter : serverToRank_[serverIdx]) {
            rankIdListServer += std::to_string(iter.userRank) + " ";
        }
        HCCL_INFO(
            "[RankGraphV1][%s] devtype[%d], curRank[%u], serverToRanklist[%s]", __func__, topoAttr_.deviceType,
            rankData_.userRank, rankIdListServer.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::InitSuperPodRankInfo()
{
    auto& rankInfoList = topoAttr_.rankInfoList;
    for (u32 index = 0; index < rankInfoList.size(); index++) {
        // 填充superPodRankMap_, 记录superPodId -> rankInfo
        HCCL_DEBUG(
            "[RankGraphV1][%s] superPodIdx[%u],superPodId[%s]", __func__, rankInfoList[index].superPodIdx,
            rankInfoList[index].superPodId.c_str());
        auto itSuperPod = superPodToRank_.find(rankInfoList[index].superPodIdx);
        if (itSuperPod != superPodToRank_.end()) {
            itSuperPod->second.push_back(rankInfoList[index]);
        } else {
            std::vector<RankInfo> rankVecTmp;
            rankVecTmp.push_back(rankInfoList[index]);
            superPodToRank_.insert(std::make_pair(rankInfoList[index].superPodIdx, rankVecTmp));
        }
    }

    // 调整每个superPod内的user_rank排序, 按照serverIdx从小到大、userRank从小到大排序
    for (auto iterMap = superPodToRank_.begin(); iterMap != superPodToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            std::sort(iterMap->second.begin(), iterMap->second.end(), RankGraphSort);
        }
    }

    if (superPodToRank_.find(rankData_.superPodIdx) != superPodToRank_.end()) {
        std::string rankIdListPod;
        for (auto iter : superPodToRank_[rankData_.superPodIdx]) {
            rankIdListPod += std::to_string(iter.userRank) + " ";
        }
        HCCL_INFO(
            "[RankGraphV1][%s] curRank[%u], curSuperPod[%s] superPodToRanklist[%s]", __func__, rankData_.userRank,
            rankData_.superPodId.c_str(), rankIdListPod.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV1::InitNetLayer()
{
    netLayer_.clear();
    netLayer_.push_back(static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L0));

    u32 serverIdx = rankData_.serverIdx;
    auto rankVec = serverToRank_.find(serverIdx);
    if (rankVec == serverToRank_.end()) {
        HCCL_ERROR("[RankGraphV1][%s] find serverToRank failed, serverIdx[%u]", __func__, serverIdx);
        return HCCL_E_INTERNAL;
    }
    std::vector<u32> rankListTmp;
    for (auto iter : serverToRank_[serverIdx]) {
        rankListTmp.push_back(iter.userRank);
    }
    rankList_.insert({static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L0), rankListTmp});

    std::vector<u32> rankSizeListTmp;
    for (auto iter : serverToRank_) {
        rankSizeListTmp.push_back(iter.second.size());
    }
    rankSizeList_.insert({static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L0), rankSizeListTmp});

    DevType deviceType = topoAttr_.deviceType;
    if (serverToRank_.size() > 1) {
        netLayer_.push_back(static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L1));
        if (deviceType == DevType::DEV_TYPE_910B || deviceType == DevType::DEV_TYPE_910) {
            std::vector<u32> rankListTmp1;
            for (auto& pair : serverToRank_) {
                for (auto iter : pair.second) {
                    rankListTmp1.push_back(iter.userRank);
                }
            }
            rankList_.insert({static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L1), rankListTmp1});
            rankSizeList_.insert(
                {static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L1), {topoAttr_.userRankSize}});
        } else if (deviceType == DevType::DEV_TYPE_910_93) {
            auto it = superPodToRank_.find(rankData_.superPodIdx);
            if (it == superPodToRank_.end()) {
                HCCL_ERROR(
                    "[RankGraphV1][%s] find superPodToRank_ failed, superPodIdx[%u]", __func__, rankData_.superPodIdx);
                return HCCL_E_INTERNAL;
            }
            std::vector<u32> rankListTmp1;
            for (auto iter : superPodToRank_[rankData_.superPodIdx]) {
                rankListTmp1.push_back(iter.userRank);
            }
            std::vector<u32> rankSizeListTmp1;
            for (auto iter : superPodToRank_) {
                rankSizeListTmp1.push_back(iter.second.size());
            }
            rankList_.insert({static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L1), rankListTmp1});
            rankSizeList_.insert({static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L1), rankSizeListTmp1});
        }
    }

    if (deviceType == DevType::DEV_TYPE_910_93 && superPodToRank_.size() > 1) {
        netLayer_.push_back(static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L2));
        std::vector<u32> rankListTmp2;
        for (const auto& pair : superPodToRank_) {
            for (auto iter : pair.second) {
                rankListTmp2.push_back(iter.userRank);
            }
        }
        rankList_.insert({static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L2), rankListTmp2});
        rankSizeList_.insert({static_cast<uint32_t>(HcclNetLayerlevel::HCCL_NetLayer_L2), {topoAttr_.userRankSize}});
    }
    return HCCL_SUCCESS;
}
}; // namespace hccl
