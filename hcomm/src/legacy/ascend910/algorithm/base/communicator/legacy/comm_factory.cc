/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_factory.h"
#include <sstream>
#include <algorithm>
namespace hccl {

CommFactory::CommFactory(
    const std::string& identifier, const u32 userRank, const u32 userRankSize, const HcclDispatcher dispatcher,
    const std::unique_ptr<NotifyPool>& notifyPool, std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap,
    std::shared_ptr<TopoInfoExtractor> topoInfoEx, const bool isUsedRdmaLevel0, const TopoType topoFlag,
    const DevType deviceType, const std::vector<RankInfo> rankVector, const NICDeployment nicDeploymentInner,
    bool isHeterogComm, u32 meshAggregationRankSize, bool isHaveCpuRank, bool isUsedInterHccsMode, bool useSuperPodMode)
    : identifier_(identifier),
      userRank_(userRank),
      userRankSize_(userRankSize),
      topoFlag_(topoFlag),
      deviceType_(deviceType),
      dispatcher_(dispatcher),
      notifyPool_(notifyPool),
      netDevCtxMap_(netDevCtxMap),
      topoInfoEx_(topoInfoEx),
      isUsedRdmaLevel0_(isUsedRdmaLevel0),
      rankVector_(rankVector),
      nicDeployInner_(nicDeploymentInner),
      isHeterogComm_(isHeterogComm),
      isHaveCpuRank_(isHaveCpuRank),
      reusedSocketManager_(),
      deviceLogicId_(0),
      isUsedInterHccsMode_(isUsedInterHccsMode),
      useSuperPodMode_(useSuperPodMode)
{
    (void)meshAggregationRankSize;
}

CommFactory::~CommFactory()
{
    // 销毁资源
    CommPlaneVector_.clear();
    isBridgeVector_.clear();
    superPodToRank_.clear();
    serverToRank_.clear();
    deviceLinkTypeMap_.clear();
    rankVector_.clear();
}

HcclResult CommFactory::Init()
{
    CHK_RET(topoInfoEx_->CheckInitInfo());

    topoInfoEx_->GetCommPlaneVector(CommPlaneVector_);
    topoInfoEx_->GetIsBridgeVector(isBridgeVector_);
    topoInfoEx_->GetRankData(rankData_);
    topoInfoEx_->GetServerToRank(serverToRank_);
    topoInfoEx_->GetSuperPodToRank(superPodToRank_);
    topoInfoEx_->GetDeviceLinkTypeMap(deviceLinkTypeMap_);

    s32 deviceLogicID = 0;
    if (!isHeterogComm_ && rankVector_[userRank_].devicePhyId != HOST_DEVICE_ID) {
        CHK_RET(hrtGetDevice(&deviceLogicID));
        deviceLogicId_ = deviceLogicID;
    }

    reusedSocketManager_.reset(new (std::nothrow) HcclSocketManager(
        nicDeployInner_, deviceLogicId_, rankVector_[userRank_].devicePhyId, userRank_));
    CHK_PTR_NULL(reusedSocketManager_);

    return HCCL_SUCCESS;
}

HcclResult CommFactory::CheckCommPara(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem,
    const CommParaInfo& commParaInfo) const
{
    (void)inputMem;
    (void)outputMem;
    CHK_PRT_RET(
        commParaInfo.commPlane >= COMM_LEVEL_RESERVED,
        HCCL_ERROR(
            "[Check][CommPara]tag[%s], commPlane[%d] is invalid, is out of range [0, %d]", tag.c_str(),
            commParaInfo.commPlane, COMM_LEVEL_RESERVED - 1),
        HCCL_E_PARA);

    // 判断commPlane和commType的组合是否支持
    bool isSupport = true;
    switch (commParaInfo.commType) {
        case CommType::COMM_TAG_RING_INNER:
        case CommType::COMM_TAG_HALVING_DOUBLING: {
            isSupport = (commParaInfo.commPlane == COMM_LEVEL0) || (commParaInfo.commPlane == COMM_LEVEL1)
                        || (commParaInfo.commPlane == COMM_LEVEL2);
            break;
        }
        case CommType::COMM_TAG_MESH: {
            isSupport = (commParaInfo.commPlane == COMM_COMBINE) || (commParaInfo.commPlane == COMM_LEVEL0)
                        || (commParaInfo.commPlane == COMM_MESH_L0) || (commParaInfo.commPlane == COMM_MESH_L1)
                        || (commParaInfo.commPlane == COMM_LEVEL2) || (commParaInfo.commPlane == COMM_COMBINE_ORDER);
            break;
        }
        case CommType::COMM_TAG_RING_COMBINED:
        case CommType::COMM_TAG_P2P: {
            isSupport = commParaInfo.commPlane == COMM_COMBINE;
            break;
        }
        case CommType::COMM_TAG_MESH_COMBINED: {
            isSupport = commParaInfo.commPlane == COMM_COMBINE_ORDER;
            break;
        }
        case CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE:
        case CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE_BROKE:
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING:
        case CommType::COMM_TAG_NONUNIFORM_BRUCK: {
            isSupport = (commParaInfo.commPlane == COMM_LEVEL1);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1: {
            isSupport = (commParaInfo.commPlane == COMM_LEVEL1 && deviceType_ != DevType::DEV_TYPE_910_93);
            break;
        }
        case CommType::COMM_TAG_STAR:
            break;
        case CommType::COMM_TAG_WHOLE_NHR:
        case CommType::COMM_TAG_WHOLE_NHR_V1:
        case CommType::COMM_TAG_WHOLE_AHC:
        case CommType::COMM_TAG_WHOLE_AHC_BROKE:
        case CommType::COMM_TAG_WHOLE_NB: {
            isSupport = (deviceType_ != DevType::DEV_TYPE_910_93);
            break;
        }
        default: {
            HCCL_ERROR("[Check][CommPara]commType[%d] is invalid", commParaInfo.commType);
            return HCCL_E_PARA;
        }
    }

    CHK_PRT_RET(
        isSupport == false,
        HCCL_ERROR(
            "[Check][CommPara]tag[%s], deviceType[%d], commPlane[%d] and commType[%d] is not support", tag.c_str(),
            deviceType_, commParaInfo.commPlane, commParaInfo.commType),
        HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommPlane(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
    std::vector<std::unique_ptr<CommBase>>& commVec, DeviceMem expMem)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO(
        "[Create][CommPlane]tag[%s], identifier[%s], commPlane[%d], commType[%d]", tag.c_str(), identifier_.c_str(),
        commParaInfo.commPlane, commParaInfo.commType);

    CHK_RET(CheckCommPara(tag, inputMem, outputMem, commParaInfo));
    bool isUsedRdma = false;
    CHK_RET(GetIsUsedRdma(commParaInfo, isUsedRdma));

    switch (commParaInfo.commType) {
        case CommType::COMM_TAG_RING_INNER:
        case CommType::COMM_TAG_RING_COMBINED:
        case CommType::COMM_TAG_NONUNIFORM_BRUCK:
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING:
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1: {
            ret = CreateCommRing(
                tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane], isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_HALVING_DOUBLING: {
            ret = CreateCommHD(
                tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane], isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_STAR: {
            std::vector<std::vector<RankInfo>> commPlaneVec;
            std::vector<RankInfo> linkParas;
            CreateStarLinkPara(linkParas);
            commPlaneVec.push_back(linkParas);
            ret = CreateCommStar(tag, inputMem, outputMem, commParaInfo, commPlaneVec, isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_MESH:
        case CommType::COMM_TAG_MESH_COMBINED: {
            if (commParaInfo.meshSinglePlane == true) {
                // 910B非确定性计算场景，server内MESH组网只需要创建一个commbase平面
                std::vector<std::vector<RankInfo>> commPlaneVec;
                commPlaneVec.push_back(CommPlaneVector_[commParaInfo.commPlane][0]);
                ret = CreateCommMesh(tag, inputMem, outputMem, commParaInfo, commPlaneVec, isUsedRdma, commVec, expMem);
            } else {
                ret = CreateCommMesh(
                    tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane], isUsedRdma,
                    commVec, expMem);
            }
            break;
        }
        case CommType::COMM_TAG_P2P: {
            ret = CreateCommP2P(
                tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane], isUsedRdma, commVec);
            break;
        }
        default: {
            HCCL_ERROR("[Create][CommPlane]commType[%d] is invalid", commParaInfo.commType);
            return HCCL_E_PARA;
        }
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Create][CommPlane]failed, tag[%s], commPlane[%d], commType[%d]", tag.c_str(), commParaInfo.commPlane,
            commParaInfo.commType),
        ret);

    HCCL_INFO(
        "complete commPlane[%d] commType[%d] creation, Time:%lld us", commParaInfo.commPlane, commParaInfo.commType,
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetIsUsedRdma(const CommParaInfo& commParaInfo, bool& isUsedRdma)
{
    std::vector<std::vector<RankInfo>> commP2PPlaneVec;
    if (commParaInfo.commType == CommType::COMM_TAG_P2P) {
        // P2P只需要判断两张卡之间的连接关系
        bool invalidcheck = (rankVector_.size() <= userRank_) || (rankVector_.size() <= commParaInfo.peerUserRank);
        CHK_PRT_RET(
            invalidcheck,
            HCCL_ERROR(
                "[GetIsUsedRdma]dstUserRank[%u] or userRank[%u] is bigger than "
                "rankVector size[%u]",
                commParaInfo.peerUserRank, userRank_, rankVector_.size()),
            HCCL_E_PARA);

        std::vector<RankInfo> commP2PRankVec;
        commP2PRankVec.push_back(rankVector_[userRank_]);
        commP2PRankVec.push_back(rankVector_[commParaInfo.peerUserRank]);
        commP2PPlaneVec.push_back(commP2PRankVec);
    }

    std::vector<std::vector<RankInfo>>& commPlaneVec = (commParaInfo.commType == CommType::COMM_TAG_P2P) ?
                                                           commP2PPlaneVec :
                                                           CommPlaneVector_[commParaInfo.commPlane];

    bool isInterSuperPod = false;
    bool isInterServer = false;
    bool isConnectedWithPcie = false;
    for (const std::vector<RankInfo>& commPlane : commPlaneVec) {
        for (const RankInfo& dstRank : commPlane) {
            if (rankData_.superPodId != dstRank.superPodId) { // 跨超节点场景
                isInterSuperPod = true;
            } else if (rankData_.serverIdx != dstRank.serverIdx) { // 不跨超节点, 跨server场景
                isInterServer = true;
            } else { // 同server, PCIE互连场景
                auto it = deviceLinkTypeMap_.find(dstRank.devicePhyId);
                CHK_PRT_RET(
                    it == deviceLinkTypeMap_.end(),
                    HCCL_ERROR("can't find devicePhyId[%d] in deviceLinkTypeMap_", dstRank.devicePhyId),
                    HCCL_E_NOT_FOUND);
                isConnectedWithPcie |= (it->second == LinkTypeInServer::PXI_TYPE);
            }
        }
    }
    // 使能RDMA的场景: 1.跨超节点  2.跨server且不使能HCCS  3.PCIE连接且使能RDMA开关
    isUsedRdma
        = (isInterSuperPod) || (isInterServer && !isUsedInterHccsMode_) || (isConnectedWithPcie && isUsedRdmaLevel0_);
    HCCL_INFO(
        "[GetIsUsedRdma]isUsedRdma[%d], isInterSuperPod[%d], isInterServer[%d], isUsedInterHccsMode_[%d], "
        "isConnectedWithPcie[%d], isUsedRdmaLevel0_[%d]",
        isUsedRdma, isInterSuperPod, isInterServer, isUsedInterHccsMode_, isConnectedWithPcie, isUsedRdmaLevel0_);
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommRing(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
    const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase>>& commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        // 只有在当前环是bridge rank才需要创建comm实例
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        HCCL_DEBUG("[CommFactory][CreateCommRing]rank is %u", rank);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork{};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO(
            "[Create][CommRing]comm is used %s. userRank = %u, rank = %u", isUsedRdma ? "rdma" : "sdma", userRank_,
            rank);

        commVec[ringIndex].reset(new (std::nothrow) CommRing(
            identifier_, userRank_, userRankSize_, rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_,
            notifyPool_, netDevCtxMap_, exchangerNetwork, commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, tag,
            nicDeployInner_, false, false, isHaveCpuRank_, useSuperPodMode_));

        CHK_PRT_RET(
            !commVec[ringIndex], HCCL_ERROR("[Create][CommRing]comm array[%u] reset failed", ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(
            rankDevicePhyIdNicInfoMap_, ranksPort_, vnicRanksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommRing]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommHD(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
    const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase>>& commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    u32 subUserRankRoot = INVALID_VALUE_RANKID;
    if (commParaInfo.root != INVALID_VALUE_RANKID) {
        subUserRankRoot = GetSubRootUserRank(userRank_, commParaInfo.root);
        if (subUserRankRoot == INVALID_VALUE_RANKID) {
            HCCL_ERROR("[create][CommHD]get sub root userrank value[%u] invalid.", subUserRankRoot);
            return HCCL_E_PARA;
        }
    }

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        // 只有在当前环是bridge rank才需要创建comm实例
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork{};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO(
            "[create][CommHD]comm is used %s. userRank = %u, rank = %u", isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommHalvingDoubling(
            identifier_, userRank_, userRankSize_, rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_,
            notifyPool_, netDevCtxMap_, exchangerNetwork, commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, tag,
            nicDeployInner_, subUserRankRoot, HalvingDoublingType::RECURSIVE_HALVING_DOUBLING, isHaveCpuRank_,
            useSuperPodMode_));

        CHK_PRT_RET(
            !commVec[ringIndex], HCCL_ERROR("[create][CommHD]comm array[%u] reset failed", ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(
            rankDevicePhyIdNicInfoMap_, ranksPort_, vnicRanksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[create][CommHD]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

void CommFactory::CreateStarLinkPara(std::vector<RankInfo>& linkParas) { linkParas = rankVector_; }

HcclResult CommFactory::CreateCommStar(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
    const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase>>& commVec)
{
    HCCL_INFO("create comm star start");
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        IntraExchanger exchangerNetwork{};
        HCCL_INFO("[CreateCommStar] CommStar is used %s. userRank = %u", isUsedRdma ? "rdma" : "sdma", userRank_);

        commVec[ringIndex].reset(new (std::nothrow) CommStar(
            identifier_, userRank_, userRankSize_, userRank_, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_,
            notifyPool_, netDevCtxMap_, exchangerNetwork, commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, tag,
            nicDeployInner_, commParaInfo.root, isHaveCpuRank_));

        CHK_PRT_RET(
            !commVec[ringIndex], HCCL_ERROR("[create][CommStar]comm array[%u] reset failed", ringIndex), HCCL_E_PARA);
        commVec[ringIndex]->SetHDCModeInfo(
            rankDevicePhyIdNicInfoMap_, ranksPort_, vnicRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

        if (JudgmentSetHeterogP2p(userRank_)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[create][CommStar]comm array[%u] star rank[%u] init failed", ringIndex, userRank_);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommMesh(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
    const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase>>& commVec, DeviceMem& expMem)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);
    bool intraIsUseRdma = isUsedRdma;
    bool isAlltoAllCommMesh = false;

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        CHK_PRT_RET(rank == INVALID_VALUE_RANKID, HCCL_ERROR("[Create][CommMesh] invalid rank info."), HCCL_E_PARA);

        IntraExchanger exchangerNetwork{};
        exchangerNetwork.socketManager = reusedSocketManager_;

        u32 mc2MultiServerType = commVec[ringIndex]->IsSupportMC2(tag);
        HCCL_INFO(
            "[Create][CommMesh]comm is used %s. userRank = %u, rank = %u, mc2MultiServerType = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank, mc2MultiServerType);
        if (mc2MultiServerType == MC2_PLANE_MODE_HIERARCHY) { // 分层场景
            intraIsUseRdma = false;
            isAlltoAllCommMesh = false;
        } else if (mc2MultiServerType == MC2_PLANE_MODE_COMBINE) { // 非分层场景
            intraIsUseRdma = true;
        }
        HCCL_INFO(
            "[Create][CommMesh]comm is created by intraIsUseRdma = %u, isAlltoAllCommMesh = %u", intraIsUseRdma,
            isAlltoAllCommMesh);

        commVec[ringIndex].reset(new (std::nothrow) CommMesh(
            identifier_, userRank_, userRankSize_, rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_,
            notifyPool_, netDevCtxMap_, exchangerNetwork, commPlaneVec[ringIndex], inputMem, outputMem, intraIsUseRdma,
            tag, isAlltoAllCommMesh, nicDeployInner_, false, commParaInfo.isAicpuModeEn, isHaveCpuRank_,
            useSuperPodMode_, expMem));

        CHK_PRT_RET(
            !commVec[ringIndex], HCCL_ERROR("[Create][CommMesh]comm array[%u] reset failed", ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(
            rankDevicePhyIdNicInfoMap_, ranksPort_, vnicRanksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommMesh]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommP2P(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
    const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase>>& commVec)
{
    bool invalidcheck = (rankVector_.size() <= userRank_) || (rankVector_.size() <= commParaInfo.peerUserRank);
    CHK_PRT_RET(
        invalidcheck,
        HCCL_ERROR(
            "[Create][CommP2P]dstUserRank[%u] or userRank[%u] is bigger than rank vector size[%u].",
            commParaInfo.peerUserRank, userRank_, rankVector_.size()),
        HCCL_E_PARA);

    bool heterogP2P = ((rankVector_[userRank_].devicePhyId == HOST_DEVICE_ID)
                       && (rankVector_[commParaInfo.peerUserRank].devicePhyId != HOST_DEVICE_ID))
                      || ((rankVector_[userRank_].devicePhyId != HOST_DEVICE_ID)
                          && (rankVector_[commParaInfo.peerUserRank].devicePhyId == HOST_DEVICE_ID));

    if (heterogP2P) {
        return CreateCommP2PSync(
            tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane], isUsedRdma, commVec);
    }

    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        CHK_PRT_RET(rank == INVALID_VALUE_RANKID, HCCL_ERROR("[Create][CommP2P] invalid rank info."), HCCL_E_PARA);

        IntraExchanger exchangerNetwork{};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO(
            "[Create][CommP2P]comm is used %s. userRank = %u, rank = %u", isUsedRdma ? "rdma" : "sdma", userRank_,
            rank);

        commVec[ringIndex].reset(new (std::nothrow) CommP2P(
            identifier_, userRank_, userRankSize_, rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_,
            notifyPool_, netDevCtxMap_, exchangerNetwork, commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, tag,
            commParaInfo.peerUserRank, nicDeployInner_, isHaveCpuRank_, useSuperPodMode_));

        CHK_PRT_RET(
            !commVec[ringIndex], HCCL_ERROR("[Create][CommP2P]comm array[%u] reset failed", ringIndex), HCCL_E_PARA);

        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommP2P]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommP2PSync(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
    [[maybe_unused]] const std::vector<std::vector<RankInfo>>& commPlaneVec, [[maybe_unused]] bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase>>& commVec)
{
    u32 status = 0;
    commVec = CreateCommP2PAsync(tag, inputMem, outputMem, commParaInfo.peerUserRank, status);
    for (u32 index = 0; index < commVec.size(); index++) {
        CHK_PRT_RET(
            !commVec[index],
            HCCL_ERROR(
                "[Create][CommP2PSync]errNo[0x%016llx] tag[%s], created p2pComm[%u] is null.",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str(), index),
            HCCL_E_NOT_FOUND);
    }

    if (status == 0) {
        return HCCL_SUCCESS;
    }
    do {
        HcclResult ret = CreateCommP2PQuarry(commVec, status);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommP2P]comm p2p init failed");
            return ret;
        }
        SaluSleep(COMM_P2P_QUERRY_WAIT_TIME);
    } while (status == 1);
    return HCCL_SUCCESS;
}

std::vector<std::unique_ptr<CommBase>> CommFactory::CreateCommP2PAsync(
    const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const u32 dstUserRank, u32& status)
{
    u32 ringSize = CommPlaneVector_[COMM_COMBINE].size();
    std::vector<std::unique_ptr<CommBase>> commP2PArray(0); // 复用CommBase来实现P2P拓扑功能

    bool memFlag = !inputMem || !outputMem;
    CHK_PRT_RET(memFlag, HCCL_ERROR("[Create][CommP2P]inputMem is null or outputMem is null."), commP2PArray);

    commP2PArray.resize(ringSize); // ring_size即为网络平面数, 比如能组几条环

    bool invalidcheck = (rankVector_.size() <= userRank_) || (rankVector_.size() <= dstUserRank);
    CHK_PRT_RET(
        invalidcheck,
        HCCL_ERROR(
            "[Create][CommP2P]dstUserRank[%u] or userRank[%u] is bigger than rank vector.", dstUserRank, userRank_),
        commP2PArray);

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        u32 rank = GetSubCollectiveRank(CommPlaneVector_[COMM_COMBINE][ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork{};
        HCCL_INFO(
            "[CreateCommP2PAsync] CommP2P is used %s. userRank = %u, rank = %u", isUsedRdmaLevel0_ ? "rdma" : "sdma",
            userRank_, rank);
        commP2PArray[ringIndex].reset(new (std::nothrow) CommP2P(
            identifier_, userRank_, userRankSize_, rank, CommPlaneVector_[COMM_COMBINE][ringIndex].size(),
            TopoType::TOPO_TYPE_COMMON, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            CommPlaneVector_[COMM_COMBINE][ringIndex], inputMem, outputMem, isUsedRdmaLevel0_, tag, dstUserRank,
            nicDeployInner_));

        CHK_PRT_RET(
            !commP2PArray[ringIndex], HCCL_ERROR("[Create][CommP2P]comm p2p array[%u] reset failed.", ringIndex),
            commP2PArray);
        if (commP2PArray[ringIndex]->BuildAsync(status) != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommP2P]comm p2p array[%u] init failed", ringIndex);
            commP2PArray[ringIndex].reset(nullptr);
            return commP2PArray;
        }
        HCCL_DEBUG("BuildAsync %u", status);
    }
    return commP2PArray;
}

HcclResult CommFactory::CreateCommP2PQuarry(std::vector<std::unique_ptr<CommBase>>& comm, u32& status)
{
    HcclResult ret;
    std::vector<u32> commStatus(comm.size());
    for (u32 index = 0; index < comm.size(); index++) {
        CHK_SMART_PTR_NULL(comm[index]);
        ret = comm[index]->BuildQuerry(commStatus[index]);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Query][CommP2P]comm p2p array[%u] init failed", index);
            comm[index].reset(nullptr);
            return ret;
        }
    }
    status = (static_cast<int>(comm.size()) == std::count(commStatus.begin(), commStatus.end(), 0)) ? 0 : 1;
    HCCL_DEBUG("CreateCommP2PQuery %u", status);
    return HCCL_SUCCESS;
}

u32 CommFactory::GetSubRootUserRank(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;
    if ((rankVector_.size() > userRank) && (rankVector_.size() > rootUserRank)) {
        u32 moduleIdx = 0;
        CHK_PRT_RET(
            topoInfoEx_->GetModuleIdx(rankVector_[rootUserRank], moduleIdx) != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SubRootUserRank]get server id failed."), INVALID_VALUE_RANKID);

        auto iterRankRoot = serverToRank_.find(moduleIdx);
        CHK_PRT_RET(
            iterRankRoot == serverToRank_.end(),
            HCCL_ERROR(
                "[Get][SubRootUserRank]can't find root serverId[%s] in rank map",
                rankVector_[rootUserRank].serverId.c_str()),
            INVALID_VALUE_RANKID);

        CHK_PRT_RET(
            topoInfoEx_->GetModuleIdx(rankVector_[userRank], moduleIdx) != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SubRootUserRank]get server id failed."), INVALID_VALUE_RANKID);

        auto iterRankCurr = serverToRank_.find(moduleIdx);
        CHK_PRT_RET(
            iterRankCurr == serverToRank_.end(),
            HCCL_ERROR(
                "[Get][SubRootUserRank]can't find local serverId[%s] in rank map",
                rankVector_[userRank].serverId.c_str()),
            INVALID_VALUE_RANKID);

        for (u32 index = 0; index < (iterRankCurr->second).size(); index++) {
            /* 当userRank的server内rank号与rootUserRank所在服务器中某一个server内rank号相同，
            获取出rootUserRank所在服务器内的userrank */
            if (userRank == (iterRankCurr->second)[index].userRank) {
                tmpUserRank = (iterRankRoot->second)[index].userRank;
                break;
            }
        }
    }
    return tmpUserRank;
}

u32 CommFactory::GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;
    if ((rankVector_.size() <= userRank) || (rankVector_.size() <= rootUserRank)) {
        return tmpUserRank;
    }

    u32 rootSuperPodIdx = rankVector_[rootUserRank].superPodIdx;
    auto iterRankRoot = superPodToRank_.find(rootSuperPodIdx);
    CHK_PRT_RET(
        iterRankRoot == superPodToRank_.end(),
        HCCL_ERROR(
            "[Get][GetSubRootUserRankWithSuperPod]can't find root rootSuperPodIdx[%u] in rank map", rootSuperPodIdx),
        INVALID_VALUE_RANKID);

    u32 userSuperPodIdx = rankVector_[userRank].superPodIdx;
    auto iterRankCurr = superPodToRank_.find(userSuperPodIdx);
    CHK_PRT_RET(
        iterRankCurr == superPodToRank_.end(),
        HCCL_ERROR(
            "[Get][GetSubRootUserRankWithSuperPod]can't find local userSuperPodIdx[%u] in rank map", userSuperPodIdx),
        INVALID_VALUE_RANKID);

    for (u32 index = 0; index < (iterRankCurr->second).size(); index++) {
        /* 当userRank的superPod内rank号与rootUserRank所在服务器中某一个superPod内rank号相同，
        获取出rootUserRank所在服务器内的userrank */
        if (userRank == (iterRankCurr->second)[index].userRank) {
            tmpUserRank = (iterRankRoot->second)[index].userRank;
            break;
        }
    }

    return tmpUserRank;
}

u32 CommFactory::GetSubRootForScatter(const u32 root)
{
    // 通过root找到ringIndex, 通过userRank找到level1中的rank
    u32 subRoot = INVALID_VALUE_RANKID;
    u32 planeIdx = INVALID_VALUE_RANKID;
    u32 ringSize = CommPlaneVector_[COMM_LEVEL1].size();

    CHK_PRT_RET(
        CommPlaneVector_[COMM_LEVEL1].size() == 0,
        HCCL_ERROR("[GET][GetSubRootForScatter]bridgeRankVector size is zero."), HCCL_E_PARA);

    u32 rank = INVALID_VALUE_RANKID;
    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (isBridgeVector_[ringIndex]) {
            rank = GetSubCollectiveRank(CommPlaneVector_[COMM_LEVEL1][ringIndex]); // 确定userRank在level1中的rank号
        }
        for (u32 idx = 0; idx < CommPlaneVector_[COMM_LEVEL1][ringIndex].size(); idx++) {
            if (root == CommPlaneVector_[COMM_LEVEL1][ringIndex][idx].userRank) { // 获取root所在的平面
                planeIdx = ringIndex;
            }
        }
    }
    CHK_PRT_RET(
        rank == INVALID_VALUE_RANKID, HCCL_ERROR("[GET][GetSubRootForScatter]get rankId in level1 failed."),
        HCCL_E_PARA);
    CHK_PRT_RET(
        planeIdx == INVALID_VALUE_RANKID,
        HCCL_ERROR("[GET][GetSubRootForScatter]get root[%u] planeIdx[%u] failed.", root, planeIdx), HCCL_E_PARA);
    subRoot = CommPlaneVector_[COMM_LEVEL1][planeIdx][rank].userRank;
    HCCL_DEBUG("[GetSubRootForScatter] userRank_:[%u] subRoot:[%u]", userRank_, subRoot);
    return subRoot;
}

u32 CommFactory::GetSubCollectiveRank(const std::vector<RankInfo>& vecPara) const
{
    // 在vecPara数据中，查询本user rank，查询到的vec下标就是rank值
    u32 tmpRank = INVALID_VALUE_RANKID;

    for (u32 rankIndex = 0; rankIndex < vecPara.size(); rankIndex++) {
        if (userRank_ == vecPara[rankIndex].userRank) {
            tmpRank = rankIndex;
            break;
        }
    }

    return tmpRank;
}

u32 CommFactory::GetLevel1CommRank(const u32 ringIdx)
{
    return GetSubCollectiveRank(CommPlaneVector_[COMM_LEVEL1][ringIdx]);
}

bool CommFactory::JudgmentSetHeterogP2p([[maybe_unused]] u32 rank) const { return isHaveCpuRank_; }

HcclResult CommFactory::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& rankDevicePhyIdNicInfoMap,
    std::vector<u32>& ranksPort, std::vector<u32>& vnicRanksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    rankDevicePhyIdNicInfoMap_ = rankDevicePhyIdNicInfoMap;
    ranksPort_ = ranksPort;
    vnicRanksPort_ = vnicRanksPort;
    isSetHDCModeInfo_ = isSetHDCModeInfo;
    isUseRankPort_ = isUseRankPort;
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetIsUsedRdma(
    const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport, bool isUsedRdma) const
{
    u32 ringSize = commTransport.size();

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        SingleSubCommTransport& subCommTransport = commTransport[ringIndex];
        for (auto& transportRequest : subCommTransport.transportRequests) {
            transportRequest.isUsedRdma = isUsedRdma;
        }
    }
    HCCL_INFO("[CommFactory][SetIsUsedRdma] commPlane[%d] isUsedRdma[%d]", commParaInfo.commPlane, isUsedRdma);
    return HCCL_SUCCESS;
}

} // namespace hccl
