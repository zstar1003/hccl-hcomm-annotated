/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <condition_variable>
#include "externalinput_pub.h"
#include "coll_alg_param.h"
#include "search_path.h"
#include "calc_p2p_transport_req.h"
#include "calc_hccs_plus_sio_transport_req_pub.h"
#include "topo_matcher.h"
namespace hccl {

TopoMatcher::TopoMatcher(
    const std::vector<std::vector<std::vector<u32>>> CommPlaneRanks, const std::vector<bool> isBridgeVector,
    HcclTopoInfo& topoInfo, HcclAlgoInfo& algoInfo, HcclExternalEnable& externalEnable,
    std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank)
    : CommPlaneVector_(CommPlaneRanks),
      isBridgeVector_(isBridgeVector),
      topoInfo_(topoInfo),
      algoInfo_(algoInfo),
      externalEnable_(externalEnable),
      userRank_(topoInfo.userRank),
      serverAndsuperPodToRank_(serverAndsuperPodToRank)
{
    SetRankMap();
}

HcclResult TopoMatcher::CalcCommPlaneInfo(
    const std::string& tag, const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport,
    TransportMemType inputMemType, TransportMemType outputMemType)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO(
        "[Calc][CommPlane]tag[%s], commPlane[%d], commType[%d]", tag.c_str(), commParaInfo.commPlane,
        commParaInfo.commType);

    u32 subUserRankRoot = INVALID_VALUE_RANKID;
    if (commParaInfo.root != INVALID_VALUE_RANKID) {
        if (commParaInfo.commPlane == COMM_LEVEL2) {
            subUserRankRoot = GetSubRootUserRankWithSuperPod(userRank_, commParaInfo.root);
        } else {
            subUserRankRoot = GetSubRootUserRank(userRank_, commParaInfo.root);
        }
        if (subUserRankRoot == INVALID_VALUE_RANKID) {
            HCCL_ERROR("[TopoMatcher][CalcCommPlaneInfo]get sub root userrank value[%u] invalid.", subUserRankRoot);
            return HCCL_E_PARA;
        }
    }

    std::unique_ptr<CalcTransportReqBase> calcTransportReq;
    bool isAHCType = false;
    switch (commParaInfo.commType) {
        case CommType::COMM_TAG_RING_INNER:
        case CommType::COMM_TAG_RING_COMBINED: {
            calcTransportReq.reset(new (std::nothrow) CalcRingTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_HALVING_DOUBLING: {
            calcTransportReq.reset(new (std::nothrow) CalcHDTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING:
        case CommType::COMM_TAG_WHOLE_NHR: {
            calcTransportReq.reset(new (std::nothrow) CalcNHRTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1:
        case CommType::COMM_TAG_WHOLE_NHR_V1: {
            calcTransportReq.reset(new (std::nothrow) CalcNHRV1TransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE:
        case CommType::COMM_TAG_WHOLE_AHC: {
            isAHCType = true;
            CHK_PRT_RET(
                static_cast<u32>(topoInfo_.CommPlaneSubGroupVector.size())
                        < (static_cast<u32>(commParaInfo.commPlane) + 1)
                    || topoInfo_.CommPlaneSubGroupVector[commParaInfo.commPlane].size() == 0,
                HCCL_ERROR("[TopoMatcher][CalcCommPlaneInfo] CommPlaneSubGroupVector para init error."), HCCL_E_PARA);
            calcTransportReq.reset(new (std::nothrow) CalcAHCTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_,
                topoInfo_.CommPlaneSubGroupVector[commParaInfo.commPlane], topoInfo_.ahcAlgOption,
                topoInfo_.isUsedRdmaMap));
            break;
        }
        case CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE_BROKE:
        case CommType::COMM_TAG_WHOLE_AHC_BROKE: {
            isAHCType = true;
            CHK_PRT_RET(
                static_cast<u32>(topoInfo_.CommPlaneSubGroupVector.size())
                        < (static_cast<u32>(commParaInfo.commPlane) + 1)
                    || topoInfo_.CommPlaneSubGroupVector[commParaInfo.commPlane].size() == 0,
                HCCL_ERROR("[TopoMatcher][CalcCommPlaneInfo] CommPlaneSubGroupVector para init error."), HCCL_E_PARA);
            calcTransportReq.reset(new (std::nothrow) CalcAHCBrokeTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_,
                topoInfo_.CommPlaneSubGroupVector[commParaInfo.commPlane], topoInfo_.ahcAlgOption,
                topoInfo_.isUsedRdmaMap));
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_BRUCK:
        case CommType::COMM_TAG_WHOLE_NB: {
            calcTransportReq.reset(new (std::nothrow) CalcNBTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_MESH: {
            calcTransportReq.reset(new (std::nothrow) CalcMeshTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_PARTIAL_MESH_COMBINED: {
            calcTransportReq.reset(new (std::nothrow) CalcPartialMeshTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_P2P: {
            calcTransportReq.reset(new (std::nothrow) CalcP2PTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        case CommType::COMM_TAG_HCCS_PLUS_SIO: {
            calcTransportReq.reset(new (std::nothrow) CalcHccsPlusSioTransportReq(
                CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            break;
        }
        default: {
            HCCL_ERROR("[Calc][CommPlane]commType[%d] is invalid", commParaInfo.commType);
            return HCCL_E_PARA;
        }
    }

    CHK_SMART_PTR_NULL(calcTransportReq);
    ret = calcTransportReq->CalcTransportRequest(
        tag, inputMemType, outputMemType, commParaInfo, commTransport, subUserRankRoot);
    // AHC内部单独刷新，外部不需要再刷新
    if (!isAHCType) {
        CHK_RET(SetIsUsedRdma(commParaInfo, commTransport));
    }
    CHK_RET(GetRankMap(commParaInfo, commTransport));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Calc][CommPlane]failed, tag[%s], commPlane[%d], commType[%d]", tag.c_str(), commParaInfo.commPlane,
            commParaInfo.commType),
        ret);

    HCCL_INFO(
        "complete commPlane[%d] commType[%d] Calculation, Time:%lld us", commParaInfo.commPlane, commParaInfo.commType,
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetRankMap(const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport)
{
    u32 ringSize = commTransport.size();

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        SingleSubCommTransport& subCommTransport = commTransport[ringIndex];
        // 有建链诉求，则记录从userRank到subCommRank 和 从subCommRank到userRank的映射
        if (subCommTransport.transportRequests.size() != 0) {
            if (commParaInfo.commType == CommType::COMM_TAG_PARTIAL_MESH_COMBINED
                || commParaInfo.commType == CommType::COMM_TAG_HCCS_PLUS_SIO) {
                CHK_RET(GetSub2UserRankMap(commParaInfo.commPlane, 0, subCommTransport.subCommRank2UserRank));
                CHK_RET(GetUserRank2SubMap(commParaInfo.commPlane, 0, subCommTransport.userRank2subCommRank));
            } else {
                CHK_RET(GetSub2UserRankMap(commParaInfo.commPlane, ringIndex, subCommTransport.subCommRank2UserRank));
                CHK_RET(GetUserRank2SubMap(commParaInfo.commPlane, ringIndex, subCommTransport.userRank2subCommRank));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::SetRankMap()
{
    // 构建由UserRank到子通信域的映射
    CHK_PRT_RET(
        CommPlaneVector_.size() > COMM_LEVEL_RESERVED,
        HCCL_ERROR(
            "[TopoMatcher][SetRankMap]CommPlaneVector.size[%u] > COMM_LEVEL_RESERVED[%u]", CommPlaneVector_.size(),
            COMM_LEVEL_RESERVED),
        HCCL_E_PARA);
    subCommRank2UserRank_.resize(static_cast<u32>(COMM_LEVEL_RESERVED));
    userRank2subCommRank_.resize(static_cast<u32>(COMM_LEVEL_RESERVED));
    for (u32 levelIndex = 0; levelIndex < CommPlaneVector_.size(); levelIndex++) {
        u32 ringSize = CommPlaneVector_[levelIndex].size();
        subCommRank2UserRank_[levelIndex].resize(ringSize);
        userRank2subCommRank_[levelIndex].resize(ringSize);
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            u32 rankSize = CommPlaneVector_[levelIndex][ringIndex].size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                u32 userRank = CommPlaneVector_[levelIndex][ringIndex][rankIndex];
                subCommRank2UserRank_[levelIndex][ringIndex][rankIndex] = userRank;
                userRank2subCommRank_[levelIndex][ringIndex][userRank] = rankIndex;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetIsUsedRdma(const CommParaInfo& commParaInfo, bool& isUsedRdma)
{
    std::vector<std::vector<u32>> commP2PPlaneVec;
    if (commParaInfo.commType == CommType::COMM_TAG_P2P) {
        // P2P只需要判断两张卡之间的连接关系
        bool invalidcheck = (topoInfo_.isUsedRdmaMap.size() <= topoInfo_.userRank)
                            || (topoInfo_.isUsedRdmaMap.size() <= commParaInfo.peerUserRank);
        CHK_PRT_RET(
            invalidcheck,
            HCCL_ERROR(
                "[GetIsUsedRdma]dstUserRank[%u] or userRank[%u] is bigger than "
                "rankVector size[%u]",
                commParaInfo.peerUserRank, topoInfo_.userRank, topoInfo_.isUsedRdmaMap.size()),
            HCCL_E_PARA);

        std::vector<u32> commP2PRankVec;
        commP2PRankVec.push_back(topoInfo_.userRank);
        commP2PRankVec.push_back(commParaInfo.peerUserRank);
        commP2PPlaneVec.push_back(commP2PRankVec);
    }

    std::vector<std::vector<u32>>& commPlaneVec = (commParaInfo.commType == CommType::COMM_TAG_P2P) ?
                                                      commP2PPlaneVec :
                                                      CommPlaneVector_[commParaInfo.commPlane];

    for (const std::vector<u32>& commPlane : commPlaneVec) {
        for (const u32 dstRank : commPlane) {
            if (topoInfo_.isUsedRdmaMap[dstRank]) {
                isUsedRdma = true;
                return HCCL_SUCCESS;
            }
        }
    }
    isUsedRdma = false;
    return HCCL_SUCCESS;
}

HcclResult
TopoMatcher::SetIsUsedRdma(const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport)
{
    bool isUsedRdma = false;
    CHK_RET(GetIsUsedRdma(commParaInfo, isUsedRdma));
    u32 ringSize = commTransport.size();

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        SingleSubCommTransport& subCommTransport = commTransport[ringIndex];
        for (auto& transportRequest : subCommTransport.transportRequests) {
            transportRequest.isUsedRdma = isUsedRdma;
        }
    }
    HCCL_INFO("[TopoMatcher][SetIsUsedRdma] commPlane[%d] isUsedRdma[%d]", commParaInfo.commPlane, isUsedRdma);
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetSub2UserRankMap(CommPlane commPlane, u32 ringIndex, std::map<u32, u32>& subCommRank2UserRank)
{
    subCommRank2UserRank = subCommRank2UserRank_[static_cast<u32>(commPlane)][ringIndex];
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetUserRank2SubMap(CommPlane commPlane, u32 ringIndex, std::map<u32, u32>& userRank2subCommRank)
{
    userRank2subCommRank = userRank2subCommRank_[static_cast<u32>(commPlane)][ringIndex];
    return HCCL_SUCCESS;
}

HcclTopoInfo TopoMatcher::GetTopoInfo() { return topoInfo_; }

HcclAlgoInfo TopoMatcher::GetAlgoInfo() { return algoInfo_; }

u32 TopoMatcher::GetExternalInputHcclEnableFfts() { return externalEnable_.enableFfts; }

u32 TopoMatcher::GetExternalInputHcclDeterministic() { return externalEnable_.deterministic; }

u32 TopoMatcher::GetExternalInputIntraRoceSwitch() { return externalEnable_.intraRoceSwitch; }

u32 TopoMatcher::GetExternalInputHcclDumpDebug() { return externalEnable_.dumpDebug; }

u32 TopoMatcher::GetExternalInputInterHccsDisable() { return externalEnable_.interHccsDisable; }

bool TopoMatcher::GetARSFlag()
{
    bool isARSTrue = (topoInfo_.deviceType == DevType::DEV_TYPE_910_93)
                     && static_cast<bool>(topoInfo_.multiModuleDiffDeviceNumMode)
                     && !static_cast<bool>(topoInfo_.multiSuperPodDiffDeviceNumMode);
    return isARSTrue;
}

HcclResult TopoMatcher::EditCommPlaneVector(CommPlane commPlane, std::vector<std::vector<u32>> commVector)
{
    CommPlaneVector_[commPlane] = commVector;
    return HCCL_SUCCESS;
}

std::vector<std::vector<u32>> TopoMatcher::GetCommPlaneRanks(CommPlane commPlane)
{
    return CommPlaneVector_[commPlane];
}

bool CheckRankNeighbors(const std::vector<u32>& nicList)
{
    // 组成ROH环路必须偶数个,且2节点不能组成双环？
    if (nicList.size() % 2 != 0 || nicList.size() < HCCL_DEVICE_NUM_FOUR) {
        return false;
    }

    std::vector<u32> tmpNicList(nicList);
    std::sort(tmpNicList.begin(), tmpNicList.end());
    u32 halfNum = 2;
    for (u32 i = 0; i < tmpNicList.size() / halfNum; i++) {
        auto nicIndex = i * halfNum;
        // 检查相邻下标的节点，devID是否相邻
        if (tmpNicList[nicIndex] + 1 != tmpNicList[nicIndex + 1]) {
            return false;
        }
    }

    return true;
}

// 适配ROH平面网段隔离，奇数rank互通，偶数rank互通，奇偶不通
bool TopoMatcher::CheckSdmaWithRohTopo(const std::vector<u32>& nicList, std::vector<u32>& topoList)
{
    std::vector<u32> tmpNicList(nicList);
    std::sort(tmpNicList.begin(), tmpNicList.end());
    SearchPath searchPath;
    topoList = searchPath.Search(tmpNicList);
    if (topoList.empty()) {
        return false;
    }
    return true;
}

u32 TopoMatcher::GetSubCollectiveRank(const std::vector<u32>& vecPara) const
{
    // 在vecPara数据中，查询本user rank，查询到的vec下标就是rank值
    u32 tmpRank = INVALID_VALUE_RANKID;

    HCCL_DEBUG("[TopoMatcher]GetSubCollectiveRank begins.");
    for (u32 rankIndex = 0; rankIndex < vecPara.size(); rankIndex++) {
        if (userRank_ == vecPara[rankIndex]) {
            tmpRank = rankIndex;
            break;
        }
    }

    return tmpRank;
}

HcclResult TopoMatcher::GetSubRootForScatter(const u32 root, u32& subRoot)
{
    // 通过root找到ringIndex, 通过userRank找到level1中的rank
    u32 planeIdx = INVALID_VALUE_RANKID;
    u32 ringSize = CommPlaneVector_[COMM_LEVEL1_INDEX].size();

    CHK_PRT_RET(ringSize == 0, HCCL_ERROR("[GET][GetSubRootForScatter]bridgeRankVector size is zero."), HCCL_E_PARA);
    CHK_PRT_RET(
        isBridgeVector_.size() != ringSize,
        HCCL_ERROR("[GET][GetSubRootForScatter]bridgeRankVector is not equal ringSize."), HCCL_E_PARA);

    u32 rank = INVALID_VALUE_RANKID;
    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (isBridgeVector_[ringIndex]) {
            rank = GetSubCollectiveRank(
                CommPlaneVector_[COMM_LEVEL1_INDEX][ringIndex]); // 确定userRank在level1中的rank号
        }
        for (u32 idx = 0; idx < CommPlaneVector_[COMM_LEVEL1_INDEX][ringIndex].size(); idx++) {
            if (root == CommPlaneVector_[COMM_LEVEL1_INDEX][ringIndex][idx]) { // 获取root所在的平面
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
    subRoot = CommPlaneVector_[COMM_LEVEL1_INDEX][planeIdx][rank];
    HCCL_DEBUG("[GetSubRootForScatter] userRank_:[%u] subRoot:[%u]", userRank_, subRoot);
    return HCCL_SUCCESS;
}

u32 TopoMatcher::GetSubRootUserRank(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;

    u32 serverIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[0].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[0][i].size(); j++) {
            if (serverAndsuperPodToRank_[0][i][j] == rootUserRank) {
                serverIdx = i;
                break;
            }
        }
    }
    u32 rankIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[0].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[0][i].size(); j++) {
            if (serverAndsuperPodToRank_[0][i][j] == userRank) {
                rankIdx = j;
                break;
            }
        }
    }

    if (serverIdx != INVALID_VALUE_RANKID && rankIdx != INVALID_VALUE_RANKID) {
        tmpUserRank = serverAndsuperPodToRank_[0][serverIdx][rankIdx];
    }
    HCCL_DEBUG(
        "[GetSubRootUserRank] userRank:[%u] rootUserRank:[%u], tmpUserRank[%u]", userRank, rootUserRank, tmpUserRank);
    return tmpUserRank;
}

u32 TopoMatcher::GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;

    u32 superPodIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[1].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[1][i].size(); j++) {
            if (serverAndsuperPodToRank_[1][i][j] == rootUserRank) {
                superPodIdx = i;
                break;
            }
        }
    }
    u32 rankIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[1].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[1][i].size(); j++) {
            if (serverAndsuperPodToRank_[1][i][j] == userRank) {
                rankIdx = j;
                break;
            }
        }
    }

    if (superPodIdx != INVALID_VALUE_RANKID && rankIdx != INVALID_VALUE_RANKID) {
        tmpUserRank = serverAndsuperPodToRank_[1][superPodIdx][rankIdx];
    }
    HCCL_DEBUG(
        "GetSubRootUserRankWithSuperPod userRank[%u], rootUserRank[%u], ret[%u]", userRank, rootUserRank, tmpUserRank);
    return tmpUserRank;
}

u32 TopoMatcher::GetSubRootWithSuperPod(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;

    u32 superPodIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[1].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[1][i].size(); j++) {
            if (serverAndsuperPodToRank_[1][i][j] == userRank) {
                superPodIdx = i;
                break;
            }
        }
    }
    u32 rankIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[1].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[1][i].size(); j++) {
            if (serverAndsuperPodToRank_[1][i][j] == rootUserRank) {
                rankIdx = j;
                break;
            }
        }
    }

    if (superPodIdx != INVALID_VALUE_RANKID && rankIdx != INVALID_VALUE_RANKID) {
        tmpUserRank = serverAndsuperPodToRank_[1][superPodIdx][rankIdx];
    }
    HCCL_DEBUG("GetSubRootWithSuperPod superPodIdx[%u], rankIdx[%u], ret[%u]", superPodIdx, rankIdx, tmpUserRank);
    return tmpUserRank;
}

HcclResult TopoMatcher::GetLocalSuperPodRankSize(const u32 userRank, u32& devNumInlocalPod, u32& rankIdxInPod)
{
    u32 superPodIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[1].size(); i++) {
        std::vector<u32> userRankInSuperPod(serverAndsuperPodToRank_[1][i]);
        std::sort(userRankInSuperPod.begin(), userRankInSuperPod.end());
        for (u32 j = 0; j < userRankInSuperPod.size(); j++) {
            if (userRankInSuperPod[j] == userRank) {
                superPodIdx = i;
                rankIdxInPod = j;
                break;
            }
        }
    }
    if (superPodIdx == INVALID_VALUE_RANKID || rankIdxInPod == INVALID_VALUE_RANKID) {
        HCCL_ERROR("[GET][GetLocalSuperPodRankSize]get rankId in level1 failed.");
        return HCCL_E_PARA;
    }
    devNumInlocalPod = serverAndsuperPodToRank_[1][superPodIdx].size();
    HCCL_DEBUG(
        "[GetLocalSuperPodRankSize] userRank[%u], superPodIdx[%u], rankIdxInPod[%u] devNumInlocalPod[%u]", userRank,
        superPodIdx, rankIdxInPod, devNumInlocalPod);
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetLocalServerRankSize(const u32 userRank, u32& devNumInlocalServer, u32& rankIdxInServer)
{
    u32 serverIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[0].size(); i++) {
        std::vector<u32> userRankInServer(serverAndsuperPodToRank_[0][i]);
        std::sort(userRankInServer.begin(), userRankInServer.end());
        for (u32 j = 0; j < userRankInServer.size(); j++) {
            if (userRankInServer[j] == userRank) {
                serverIdx = i;
                rankIdxInServer = j;
                break;
            }
        }
    }
    if (serverIdx == INVALID_VALUE_RANKID || rankIdxInServer == INVALID_VALUE_RANKID) {
        HCCL_ERROR("[GET][GetLocalServerRankSize]get rankId in level1 failed.");
        return HCCL_E_PARA;
    }
    devNumInlocalServer = serverAndsuperPodToRank_[0][serverIdx].size();
    HCCL_DEBUG(
        "[GetLocalServerRankSize] userRank[%u], serverIdx[%u], rankIdxInServer[%u] devNumInlocalServer[%u]", userRank,
        serverIdx, rankIdxInServer, devNumInlocalServer);
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::SetDeterministicConfig(const u8 deterministic)
{
    if (deterministic > DETERMINISTIC_STRICT) {
        HCCL_ERROR("[SetDeterministicConfig] deterministic[%u] is invalid, should be 0, 1 or 2.", deterministic);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[SetDeterministicConfig]deterministic is set to [%d]", deterministic);
    externalEnable_.deterministic = deterministic;
    return HCCL_SUCCESS;
}

u8 TopoMatcher::GetDeterministicConfig() const { return externalEnable_.deterministic; }

HcclResult TopoMatcher::SetOnlyAivModeConfig(const bool isOnlyAiv)
{
    if (isOnlyAiv) {
        externalEnable_.aivMode = isOnlyAiv;
        externalEnable_.isOnlyAiv = isOnlyAiv;
    }
    HCCL_RUN_INFO("[SetOnlyAivModeConfig]isOnlyAiv is set to [%d]", isOnlyAiv);
    return HCCL_SUCCESS;
}

bool TopoMatcher::GetIsOnlyAivConfig() const { return externalEnable_.isOnlyAiv; }

HcclResult TopoMatcher::SetAivModeConfig(const bool aivMode)
{
    HCCL_INFO("[SetAivMode]AivMode is set to [%d]", aivMode);
    externalEnable_.aivMode = aivMode;
    return HCCL_SUCCESS;
}

bool TopoMatcher::GetAivModeConfig() const { return externalEnable_.aivMode; }

HcclResult TopoMatcher::SetAicpuUnfoldConfig(const bool aicpuUnfold)
{
    HCCL_INFO("[SetAicpuMode]Aicpu is set to [%d]", aicpuUnfold);
    externalEnable_.aicpuUnfold = aicpuUnfold;
    return HCCL_SUCCESS;
}

bool TopoMatcher::GetAicpuUnfoldConfig() const { return externalEnable_.aicpuUnfold; }

HcclResult TopoMatcher::SetExecTimeOutConfig(const s32 execTimeOut)
{
    HCCL_INFO("[SetExecTimeOutConfig]execTimeOut is set to [%d s]", execTimeOut);
    externalEnable_.execTimeOut = execTimeOut;
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap)
{
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        externalEnable_.algoConfig[static_cast<HcclCMDType>(opType)] = algoMap.at(static_cast<HcclCMDType>(opType));
    }
    return HCCL_SUCCESS;
}

s32 TopoMatcher::GetExecTimeOutConfig() const { return externalEnable_.execTimeOut; }

std::vector<HcclAlgoType> TopoMatcher::GetAlgoConfig(HcclCMDType opType) { return externalEnable_.algoConfig[opType]; }

HcclResult
TopoMatcher::GetGlobalSubGroups(const CommPlane level, std::vector<std::vector<std::vector<u32>>>& globalSubGroups)
{
    globalSubGroups = topoInfo_.CommPlaneSubGroupVector[level];
    CHK_PRT_RET(
        globalSubGroups.size() == 0, HCCL_ERROR("[TopoMatcher][GetGlobalSubGroups] globalSubGroups para init error."),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult
TopoMatcher::SetGlobalSubGroups(const CommPlane level, std::vector<std::vector<std::vector<u32>>>& globalSubGroups)
{
    topoInfo_.CommPlaneSubGroupVector[level] = globalSubGroups;
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetCommPlaneSubGroupVector(
    std::vector<std::vector<std::vector<std::vector<u32>>>>& commPlaneSubGroupVector)
{
    commPlaneSubGroupVector = topoInfo_.CommPlaneSubGroupVector;
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::SetCommPlaneSubGroupVector(
    std::vector<std::vector<std::vector<std::vector<u32>>>>& commPlaneSubGroupVector)
{
    topoInfo_.CommPlaneSubGroupVector = commPlaneSubGroupVector;
    return HCCL_SUCCESS;
}

void TopoMatcher::GetAHCAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    ahcAlgOption = topoInfo_.ahcAlgOption;
}

void TopoMatcher::SetAHCAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    topoInfo_.ahcAlgOption = ahcAlgOption;
}

} // namespace hccl
