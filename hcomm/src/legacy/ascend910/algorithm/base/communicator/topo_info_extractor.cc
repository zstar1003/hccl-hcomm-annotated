/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_info_extractor.h"
#include "externalinput_pub.h"
#include "device_capacity.h"
#include "search_path.h"
#include "comm_base_pub.h"
#include "hccl_impl_pub.h"

namespace hccl {

constexpr u32 SERVER_RANK_SIZE = 8;

TopoInfoExtractor::TopoInfoExtractor(HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr, const TopoType topoType)
    : identifier_(algoAttr.identifier),
      userRank_(topoAttr.userRank),
      userRankSize_(topoAttr.userRankSize),
      topoType_(topoType),
      deviceType_(topoAttr.deviceType),
      rankVector_(topoAttr.rankInfoList),
      meshAggregationRankSize_(topoAttr.meshAggregationRankSize),
      isUsedRdmaLevel0_(algoAttr.isUsedRdmaLevel0),
      isUsedInterHccsMode_(algoAttr.isUsedInterHccsMode),
      isDiffAggregation_(topoAttr.isDiffDeviceModule),
      isConfigAHC_(false),
      isConfigNULL_(false),
      multiModuleDiffDeviceNumMode_(topoAttr.multiModuleDiffDeviceNumMode),
      multiSuperPodDiffServerNumMode_(topoAttr.multiSuperPodDiffServerNumMode),
      multiSuperPodDiffDeviceNumMode_(topoAttr.multiSuperPodDiffDeviceNumMode),
      isDiffDeviceType_(topoAttr.isDiffDeviceType),
      gcdDeviceNumPerAggregation_(topoAttr.gcdDeviceNumPerAggregation),
      CommPlaneSubGroupVector_(COMM_LEVEL_RESERVED),
      CommPlaneVector_(COMM_LEVEL_RESERVED) {};

#ifdef CCL_LLT
// 为了适配老的LLT框架提供的构造函数
TopoInfoExtractor::TopoInfoExtractor(
    std::string identifier, u32 userRank, u32 userRankSize, TopoType topoType, DevType deviceType,
    std::vector<RankInfo>& rankVector, u32 meshAggregationRankSize, bool isUsedRdmaLevel0, bool isUsedInterHccsMode,
    bool multiModuleDiffDeviceNumMode, bool multiSuperPodDiffServerNumMode, bool multiSuperPodDiffDeviceNumMode,
    bool isDiffDeviceType, u32 gcdDeviceNumPerAggregation)
    : identifier_(identifier),
      userRank_(userRank),
      userRankSize_(userRankSize),
      topoType_(topoType),
      deviceType_(deviceType),
      rankVector_(rankVector),
      meshAggregationRankSize_(meshAggregationRankSize),
      isUsedRdmaLevel0_(isUsedRdmaLevel0),
      isUsedInterHccsMode_(isUsedInterHccsMode),
      isDiffAggregation_(false),
      multiModuleDiffDeviceNumMode_(multiModuleDiffDeviceNumMode),
      multiSuperPodDiffServerNumMode_(multiSuperPodDiffServerNumMode),
      multiSuperPodDiffDeviceNumMode_(multiSuperPodDiffDeviceNumMode),
      isDiffDeviceType_(isDiffDeviceType),
      gcdDeviceNumPerAggregation_(gcdDeviceNumPerAggregation),
      isConfigAHC_(false),
      isConfigNULL_(false),
      CommPlaneVector_(COMM_LEVEL_RESERVED),
      CommPlaneSubGroupVector_(COMM_LEVEL_RESERVED) {};
#endif

TopoInfoExtractor::~TopoInfoExtractor() {}

HcclResult TopoInfoExtractor::Init(std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfig)
{
    HCCL_INFO(
        "factory init:collective id[%s], user rank[%u], user rank size[%u], topo type[%d], device Type[%d], "
        "meshAggregationRankSize[%u]",
        identifier_.c_str(), userRank_, userRankSize_, topoType_, deviceType_, meshAggregationRankSize_);

    // 参数有效性校验
    CHK_RET(CheckInitInfo());

    // 初始化 AHC 相关信息
    InitAHCConfig(algoConfig);

    // 填充必要数据结构
    CHK_RET(SetRankInfo());

    if (IsGeneralServer() && GetRemoteIsHdc()) {
        HCCL_INFO("heterog ES ps factory init no need set topoInfo");
    } else {
        // 设置拓扑信息
        CHK_RET(SetTopologyInfo());
        // 根据拓扑类型以及芯片类型，校验两层拓扑(外层/内层)、单层拓扑的平面个数合法性
        CHK_RET(CheckPlaneInfo());
    }

    CHK_RET(SetRankMap());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetRankMap()
{
    // 构建由UserRank到子通信域的映射
    subCommRank2UserRank_.resize(static_cast<u32>(COMM_LEVEL_RESERVED));
    userRank2subCommRank_.resize(static_cast<u32>(COMM_LEVEL_RESERVED));
    HCCL_DEBUG("[TopoInfoExtractor]SetRankMap begin");

    for (u32 levelIndex = 0; levelIndex < CommPlaneVector_.size(); levelIndex++) {
        u32 ringSize = CommPlaneVector_[levelIndex].size();
        subCommRank2UserRank_[levelIndex].resize(ringSize);
        userRank2subCommRank_[levelIndex].resize(ringSize);
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            u32 rankSize = CommPlaneVector_[levelIndex][ringIndex].size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                u32 userRank = CommPlaneVector_[levelIndex][ringIndex][rankIndex].userRank;
                subCommRank2UserRank_[levelIndex][ringIndex][rankIndex] = userRank;
                userRank2subCommRank_[levelIndex][ringIndex][userRank] = rankIndex;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::CheckInitInfo()
{
    if (rankVector_.size() == 0) {
        HCCL_ERROR("[Check][InitInfo]Not support the scenes: rank_vector size is zero");
        return HCCL_E_PARA; // 没有rank_table的场景直接报错
    }

    // 构造函数入参有效性检查:user_rank_size与user_rank_
    if (userRankSize_ <= userRank_) {
        HCCL_ERROR("[Check][InitInfo]userRankSize_[%u] or userRank_[%u] is invalid.", userRankSize_, userRank_);
        return HCCL_E_PARA;
    }

    if (userRankSize_ != rankVector_.size()) {
        HCCL_ERROR(
            "[Check][InitInfo]userRankSize_[%u] is not equal to rank_vector size[%llu].", userRankSize_,
            rankVector_.size());
        return HCCL_E_PARA;
    }

    bool isParaInvalid = ((topoType_ == TopoType::TOPO_TYPE_RESERVED) || (deviceType_ >= DevType::DEV_TYPE_COUNT));
    if (isParaInvalid) {
        HCCL_ERROR(
            "[Check][InitInfo]Not support the scenes: TopoType[%d] or deviceType[%d] is invalid.", topoType_,
            deviceType_);
        return HCCL_E_PARA;
    }

    // 入参组合有效性检查:不支持4P_RING
    if ((deviceType_ == DevType::DEV_TYPE_910 || deviceType_ == DevType::DEV_TYPE_910B
         || deviceType_ == DevType::DEV_TYPE_910_93)
        && (topoType_ == TopoType::TOPO_TYPE_4P_RING)) {
        HCCL_ERROR(
            "[Check][InitInfo]Not support the scenes: TopoType[%d] with deviceType[%d] is invalid.", topoType_,
            deviceType_);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetRankInfo()
{
    for (u32 index = 0; index < rankVector_.size(); index++) {
        if (userRank_ == rankVector_[index].userRank) {
            rankData_ = rankVector_[index];
            HCCL_INFO(
                "[SetRankInfo]rankData_: userRank[%u], devicePhyId[%d], serverIdx[%u], superPodId[%s], superPodIdx[%u]",
                rankData_.userRank, rankData_.devicePhyId, rankData_.serverIdx, rankData_.superPodId.c_str(),
                rankData_.superPodIdx);
            break;
        }
    }

    std::set<u32> serverIdxs;
    std::set<u32> moduleIdxs;
    for (u32 index = 0; index < rankVector_.size(); index++) {
        // 填充superPodRankMap_, 记录superPodId -> rankInfo
        auto itSuperPod = superPodToRank_.find(rankVector_[index].superPodIdx);
        if (itSuperPod != superPodToRank_.end()) {
            itSuperPod->second.push_back(rankVector_[index]);
        } else {
            std::vector<RankInfo> rankVecTmp;
            rankVecTmp.push_back(rankVector_[index]);
            superPodToRank_.insert(std::make_pair(rankVector_[index].superPodIdx, rankVecTmp));
        }

        u32 moduleIdx = 0;
        CHK_RET(GetModuleIdx(rankVector_[index], moduleIdx));
        moduleIdxs.insert(moduleIdx);
        // 填充serverRankMap_, 只记录本superPod下的serverIdx -> rankInfo
        if (rankVector_[index].superPodId == rankData_.superPodId || isDiffDeviceType_) {
            auto itServer = serverToRank_.find(moduleIdx);
            if (itServer != serverToRank_.end()) { // 存在该服务器内相关rank的对应信息
                itServer->second.push_back(rankVector_[index]);
            } else { // 不存在则新增一条map记录
                std::vector<RankInfo> rankVecTmp;
                rankVecTmp.push_back(rankVector_[index]);
                serverToRank_.insert(std::make_pair(moduleIdx, rankVecTmp));
            }
        }

        // 填充 serverToRankMerge_, server 和 superPod 两层合并的通信域内所有 rank 信息
        auto itServer = serverToRankMerge_.find(moduleIdx);
        if (itServer != serverToRankMerge_.end()) { // 存在该服务器内相关rank的对应信息
            itServer->second.push_back(rankVector_[index]);
        } else { // 不存在则新增一条map记录
            std::vector<RankInfo> rankVecTmp;
            rankVecTmp.push_back(rankVector_[index]);
            serverToRankMerge_.insert(std::make_pair(moduleIdx, rankVecTmp));
        }

        // 同一个server内, 记录本rank和其他rank的链路
        if (rankVector_[index].serverIdx == rankData_.serverIdx) {
            LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
            if (rankData_.devicePhyId != rankVector_[index].devicePhyId && rankData_.devicePhyId != HOST_DEVICE_ID
                && rankVector_[index].devicePhyId != HOST_DEVICE_ID && topoType_ != TopoType::TOPO_TYPE_HETEROG) {
                CHK_RET(hrtGetPairDeviceLinkType(rankData_.devicePhyId, rankVector_[index].devicePhyId, linkType));
            }
            deviceLinkTypeMap_.insert(std::make_pair(rankVector_[index].devicePhyId, linkType));
        }

        u32 serverIdx = 0;
        CHK_RET(GetServerIdx(rankVector_[index], serverIdx));
        serverIdxs.insert(serverIdx);
    }

    u32 rankNumPerAggregation = userRankSize_ / static_cast<u32>(moduleIdxs.size());
    if (isDiffDeviceType_) {
        rankNumPerAggregation = gcdDeviceNumPerAggregation_;
        HCCL_INFO(
            "[SetRankInfo] isDiffDeviceType[%u] userRankSize[%u] moduleIdxs.size[%u] rankNumPerAggregation[%u]",
            isDiffDeviceType_, userRankSize_, moduleIdxs.size(), rankNumPerAggregation);
    }

    // 调整每个server内的user_rank排序(server内userRank从小到大,一定连续)
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            std::sort(iterMap->second.begin(), iterMap->second.end(), Ascending);
        }
    }

    // 调整每个superPod内的user_rank排序, 按照serverIdx从小到大、userRank从小到大排序
    for (auto iterMap = superPodToRank_.begin(); iterMap != superPodToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            std::sort(iterMap->second.begin(), iterMap->second.end(), Ascending);
        }
    }

    // 调整多个 superPod 合并的 user_rank 排序，按照 serverIdx 从小到大、userRank从小到大排序
    for (auto iterMap = serverToRankMerge_.begin(); iterMap != serverToRankMerge_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            std::sort(iterMap->second.begin(), iterMap->second.end(), Ascending);
        }
    }

    for (auto it = serverToRank_.begin(); it != serverToRank_.end(); it++) {
        HCCL_DEBUG("[SetRankInfo][MIX_DEBUG] serverID[%u]", it->first);
        for (auto index = it->second.begin(); index != it->second.end(); index++) {
            HCCL_DEBUG(
                "[SetRankInfo][MIX_DEBUG] userRank[%u], devicePhyId[%d], serverIdx[%u], superPodId[%s]",
                index->userRank, index->devicePhyId, index->serverIdx, index->superPodId.c_str());
        }
    }

    for (auto it = serverToRankMerge_.begin(); it != serverToRankMerge_.end(); it++) {
        HCCL_DEBUG("[SetRankInfo][AHC_DEBUG] serverID[%u]", it->first);
        for (auto index = it->second.begin(); index != it->second.end(); index++) {
            HCCL_DEBUG(
                "[SetRankInfo][AHC_DEBUG] userRank[%u], devicePhyId[%d], serverIdx[%u], superPodId[%s]",
                index->userRank, index->devicePhyId, index->serverIdx, index->superPodId.c_str());
        }
    }
    HCCL_DEBUG(
        "[SetRankInfo][AHC_DEBUG] rankNumPerAggregation[%u] moduleIdxs.size()=[%u]", rankNumPerAggregation,
        moduleIdxs.size());

    ranksOneNode_ = {0, 8, 4, 2, 1, 4, rankNumPerAggregation, 0, rankNumPerAggregation, rankNumPerAggregation};

    // 校验每个server内的设备个数与topo类型的组合是否正确
    if (topoType_ != TopoType::TOPO_TYPE_COMMON) {
        CHK_RET(CheckServerInfo());
    }
    // 校验每个superPod下的device数量相同
    CHK_RET(CheckSuperPodInfo());

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::CheckSuperPodInfo()
{
    for (auto iter = superPodToRank_.begin(); iter != superPodToRank_.end(); iter++) {
        u32 devNum = superPodToRank_.begin()->second.size();
        u32 curDevNum = iter->second.size();
        if (devNum != curDevNum) {
            HCCL_INFO(
                "[Check][SuperPodInfo]devNum[%u] in superPodIdx[%u] is inconsistent with "
                "devNum[%u] in superPodIdx[%u].",
                devNum, superPodToRank_.begin(), curDevNum, iter->first);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::CheckServerInfo()
{
    /*
     * HCOM模块：
     * 1、每个AI server之间的芯片个数必须一致，不一致则报错；
     * 2、每个AI server之间的芯片ID（device
     * ID）必须相同（server0里面devID分别是0、1、4、5；server1->server127也必须是相同的） ，不一致则报错； HCCL
     * API模块： 3、校验rank_table传进来devID，与rt_get_device查询到的devID，是否相同，不一致则报错（针对当前设备）
     * 因此，上层模块已经校验过的不再重复，本函数仅用于校验每个server内的设备个数与topo类型的组合是否正确
     */
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Check][ServerInfo]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    HcclResult ret = HCCL_SUCCESS;

    switch (topoType_) {
        case TopoType::TOPO_TYPE_NP_MESH:
        case TopoType::TOPO_TYPE_4P_MESH:
        case TopoType::TOPO_TYPE_2P_MESH:
        case TopoType::
            TOPO_TYPE_1P_MESH: { // 4p_mesh场景下，支持server(4P+4P)和server(4P)+server(4P)，2p_mesh/1p_mesh同理
            ret = (((iterRank->second).size() == ranksOneNode_[static_cast<u32>(topoType_)])
                   || ((iterRank->second).size()
                       == 2 * ranksOneNode_[static_cast<u32>(topoType_)])) // 2表示8P满配走4PMESH算法
                      ?
                      HCCL_SUCCESS :
                      HCCL_E_UNAVAIL;
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Check][ServerInfo]check server info err:server rank size[%llu], expected "
                    "value[%u], topo type[%d]",
                    (iterRank->second).size(), ranksOneNode_[static_cast<u32>(topoType_)], topoType_),
                HCCL_E_UNAVAIL);
            break;
        }
        case TopoType::TOPO_TYPE_NP_SINGLE_RING:
            ret = ((iterRank->second).size() == ranksOneNode_[static_cast<u32>(topoType_)]) ? HCCL_SUCCESS :
                                                                                              HCCL_E_UNAVAIL;
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Check][ServerInfo]check server info err:server rank size[%llu], expected "
                    "less than value[%u], topo type[%d]",
                    (iterRank->second).size(), ranksOneNode_[static_cast<u32>(topoType_)], topoType_),
                HCCL_E_UNAVAIL);
            break;
        case TopoType::TOPO_TYPE_HETEROG:
        case TopoType::TOPO_TYPE_ES_MESH:
            break;
        default: { // 8P_RING or 4P_RING
            ret = ((iterRank->second).size() == ranksOneNode_[static_cast<u32>(topoType_)]) ? HCCL_SUCCESS :
                                                                                              HCCL_E_UNAVAIL;
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Check][ServerInfo]check server info err:server rank size[%llu], expected "
                    "value[%u], topo type[%d]",
                    (iterRank->second).size(), ranksOneNode_[static_cast<u32>(topoType_)], topoType_),
                HCCL_E_UNAVAIL);
            break;
        }
    }

    HCCL_INFO(
        "check server info:server rank size[%llu], expected value[%u], topo type[%d]", (iterRank->second).size(),
        ranksOneNode_[static_cast<u32>(topoType_)], topoType_);
    return ret;
}

HcclResult TopoInfoExtractor::GetServerIdx(const RankInfo& rankInfo, u32& serverIdx) const
{
    // 通过ranktable指定集群信息场景，可以调整server在ranktable的排序(serverIdx）来指定server间通信的topo，从优化通信拓扑
    // rootInfo初始化场景，会自动收集集群信息，外部无法指定server的排序，可以无视serverIdx，使用serverID来代替
    // PS:返回的serverIdx，会影响rankMap_中server的排序，从而影响bridgeRank的选择，优化通信拓扑
    CHK_PRT_RET((rankInfo.serverIdx == INVALID_UINT), HCCL_ERROR("server idx is invalid."), HCCL_E_INTERNAL);
    serverIdx = rankInfo.serverIdx;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopologyInfo()
{
    CHK_RET(SetTopoDefaultInfo());

    CHK_RET(SetTopoInfoForLevel0());
    CHK_RET(SetTopoInfoForLevel1());
    CHK_RET(SetTopoInfoForLevel2());
    CHK_RET(SetTopoInfoForARS());
    CHK_RET(SetTopoInfoForCombineL1());

    CHK_RET(SetAHCSubGroupsAndAlgOption());

    // 是否支持按mesh划分通信拓扑
    bool isSupportMeshTopo = meshAggregationRankSize_ > 0 && userRankSize_ % meshAggregationRankSize_ == 0;
    if (isSupportMeshTopo) {
        CHK_RET(SetTopoInfoForMeshL0());
        CHK_RET(SetTopoInfoForMeshL1());
    } else {
        HCCL_INFO(
            "[Set][TopologyInfo]topo is not support Mesh, meshAggregationRankSize_[%u], userRankSize_[%u]",
            meshAggregationRankSize_, userRankSize_);
    }
    CommPlaneVector_[COMM_COMBINE_ORDER].push_back(rankVector_);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoDefaultInfo()
{
    // 填充combined_rank_vector_:不区分board_type,只生成default单层拓扑
    std::vector<RankInfo> tmpCombinedVector;

    bool incrementFlag = true; // 节点间建链的两个deviceID必须相同(同一个网段平面)，server间需要特殊处理
    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";
    RankInfo tempRankData;

    // 填充combined_rank_vector_的内层vector:combined场景只有一条固定的环
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            // TOPO_TYPE_COMMON为单环拓扑场景，不需要执行如下判断设置流程
            if (topoType_ != TopoType::TOPO_TYPE_COMMON) {
                if (((iterMap->second).size() == SERVER_RANK_SIZE) && !IsDiffDeviceModuleInServer()
                    && (topoType_ != TopoType::TOPO_TYPE_HETEROG) && !Is310PDevice()) {
                    CHK_RET(SetTopoDefaultInfoFor8P()); // 服务器内dev个数相同已在hcom层做过校验
                    return HCCL_SUCCESS;
                }
            }

            if (incrementFlag) {
                for (u32 incrementIndex = 0; incrementIndex < (iterMap->second).size(); incrementIndex++) {
                    u32 combinedUserRank = (iterMap->second)[incrementIndex].userRank;
                    bool checkError = (rankVector_.size() <= combinedUserRank);
                    CHK_PRT_RET(
                        checkError,
                        HCCL_ERROR(
                            "[Set][TopoDefaultInfo]combined userRank[%u] is bigger than "
                            "rank vector",
                            combinedUserRank),
                        HCCL_E_INTERNAL);
                    tempRankData = rankVector_[combinedUserRank];
                    outLogInfo.append(std::to_string(tempRankData.userRank));
                    outLogInfo.append("/");
                    outLogInfo.append(std::to_string(tempRankData.devicePhyId));
                    outLogInfo.append("; ");
                    tmpCombinedVector.push_back(tempRankData);
                }

                incrementFlag = false;
            } else {
                for (u32 decrementIndex = (iterMap->second).size(); decrementIndex > 0; decrementIndex--) {
                    u32 combinedUserRank = (iterMap->second)[decrementIndex - 1].userRank;
                    bool checkError = (rankVector_.size() <= combinedUserRank);
                    CHK_PRT_RET(
                        checkError,
                        HCCL_ERROR(
                            "[Set][TopoDefaultInfo]combined userRank[%u] is bigger than "
                            "rank vector",
                            combinedUserRank),
                        HCCL_E_INTERNAL);
                    tempRankData = rankVector_[combinedUserRank];
                    outLogInfo.append(std::to_string(tempRankData.userRank));
                    outLogInfo.append("/");
                    outLogInfo.append(std::to_string(tempRankData.devicePhyId));
                    outLogInfo.append("; ");
                    tmpCombinedVector.push_back(tempRankData);
                }

                incrementFlag = true;
            }
        }
    }
    if (topoType_ == TopoType::TOPO_TYPE_COMMON) {
        std::sort(tmpCombinedVector.begin(), tmpCombinedVector.end(), CompareWithUserRankAscend);
    }

    CommPlaneVector_[COMM_COMBINE].push_back(tmpCombinedVector);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::CheckPlaneInfo()
{
    bool isTopoComm = (topoType_ == TopoType::TOPO_TYPE_COMMON) && (CommPlaneVector_[COMM_COMBINE].size() != 1);
    CHK_PRT_RET(
        isTopoComm,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d] and combined plane num[%llu] are not match", topoType_,
            CommPlaneVector_[COMM_COMBINE].size()),
        HCCL_E_INTERNAL);

    bool isTopo8pring = (topoType_ == TopoType::TOPO_TYPE_8P_RING)
                        && ((CommPlaneVector_[COMM_LEVEL0].size() != meshAggregationRankSize_)
                            || (ranksOneNode_[static_cast<u32>(topoType_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(
        isTopo8pring,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    bool isTopo2pring = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING)
                        && ((CommPlaneVector_[COMM_LEVEL0].size() != 2) || // 2表示一个节点内通信域里面是否只有2个ring
                            (ranksOneNode_[static_cast<u32>(topoType_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(
        isTopo2pring,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    bool isTopo4pRing
        = (topoType_ == TopoType::TOPO_TYPE_4P_RING)
          && ((CommPlaneVector_[COMM_LEVEL0].size() != 1) || // 1表示一个节点内通信域里面是否只有一个device
              (ranksOneNode_[static_cast<u32>(topoType_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(
        isTopo4pRing,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    bool isTopo4pMesh = (topoType_ == TopoType::TOPO_TYPE_4P_MESH)
                        && ((CommPlaneVector_[COMM_LEVEL0].size() != (ranksOneNode_[static_cast<u32>(topoType_)] - 1))
                            || (ranksOneNode_[static_cast<u32>(topoType_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(
        isTopo4pMesh,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    bool isTopoNpMesh = (topoType_ == TopoType::TOPO_TYPE_NP_MESH)
                        && ((CommPlaneVector_[COMM_LEVEL0].size() != (ranksOneNode_[static_cast<u32>(topoType_)] - 1))
                            || (ranksOneNode_[static_cast<u32>(topoType_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(
        isTopoNpMesh,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    // 1表示一个module里面是否只有一个device
    bool isTopo2pMesh = (topoType_ == TopoType::TOPO_TYPE_2P_MESH)
                        && ((CommPlaneVector_[COMM_LEVEL0].size() != 1)
                            || (ranksOneNode_[static_cast<u32>(topoType_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(
        isTopo2pMesh,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    bool isTopo1pMesh
        = (topoType_ == TopoType::TOPO_TYPE_1P_MESH)
          && ((CommPlaneVector_[COMM_LEVEL0].size() != 1) || // 1表示一个节点内通信域里面是否只有一个device
              (ranksOneNode_[static_cast<u32>(topoType_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(
        isTopo1pMesh,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    bool isTopoNpSingleRing
        = (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) && (!IsDiffDeviceModuleInServer())
          && ((CommPlaneVector_[COMM_LEVEL0].size() != 1)
              || (CommPlaneVector_[COMM_LEVEL1].size() != ranksOneNode_[static_cast<u32>(topoType_)]));
    CHK_PRT_RET(
        isTopoNpSingleRing,
        HCCL_ERROR(
            "[Check][PlaneInfo]topo type[%d], level0 plane num[%llu], level1 plane num[%llu], is not match", topoType_,
            CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()),
        HCCL_E_INTERNAL);

    HCCL_RUN_INFO(
        "plane info:topo type[%d], device type[%d], COMM_COMBINE size[%llu], COMM_LEVEL0 size[%llu], COMM_LEVEL1 "
        "size[%llu], COMM_LEVEL2 size[%llu], COMM_MESH_L0 size[%llu], COMM_MESH_L1 size[%llu]",
        topoType_, deviceType_, CommPlaneVector_[COMM_COMBINE].size(), CommPlaneVector_[COMM_LEVEL0].size(),
        CommPlaneVector_[COMM_LEVEL1].size(), CommPlaneVector_[COMM_LEVEL2].size(),
        CommPlaneVector_[COMM_MESH_L0].size(), CommPlaneVector_[COMM_MESH_L1].size());

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoInfoForLevel0()
{
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Set][TopoInfoForLevel0]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);
    // 查询本rank所在服务器的rank数
    u32 ranksSize = (iterRank->second).size();

    multiLevel0Order_.clear();
    // 生成mockNicList
    std::vector<u32> mockNicList;
    mockNicList.reserve(ranksSize);
    for (u32 startIndex = 0; startIndex < ranksSize; startIndex++) {
        mockNicList.push_back(startIndex);
    }

    multiLevel0Order_ = GetRingsOrderByTopoType(ranksSize, topoType_, mockNicList);

    HCCL_DEBUG(
        "[TopoInfoExtractor] The ring number is %zu, the rank size is %lu.", multiLevel0Order_.size(), ranksSize);
    if (multiLevel0Order_.size() == 1) {
        CHK_RET(SetSingleLevel0());
    } else { // 8p-ring/np ring 环场景
        u32 ringNum = multiLevel0Order_.size();
        CHK_RET(SetMultiLevel0(ringNum)); // 8P_RING场景下，外层拓扑中有四个环; 910_93场景中适配双环
    }

    AHCCommSubgroupInit(); // 准备 AHC COMM 场景下的分组

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoInfoForLevel1(bool prepareAHC)
{
    std::map<u32, std::vector<RankInfo>>& serverToRank = (prepareAHC) ? serverToRankMerge_ : serverToRank_;

    CommPlane commPlaneLevel1 = (prepareAHC) ? COMM_LEVEL1_AHC : COMM_LEVEL1;

    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Set][TopoInfoForLevel1]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    u32 ringSize;
    if (topoType_ == TopoType::TOPO_TYPE_2P_MESH) { // 2P_MESH在任何情况下，内层拓扑平面始终为2
        ringSize = ranksOneNode_[static_cast<u32>(topoType_)];
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) { // 标卡内层拓扑的环数
        ringSize = (iterRank->second).size();
    } else { // 其他场景下内层拓扑平面为每个module中的device数量
        ringSize = ranksOneNode_[static_cast<u32>(topoType_)];
    }
    HCCL_INFO("[Set][TopoInfoForLevel1] topoType_[%u] ringSize[%u]", topoType_, ringSize);

    // 计算每个 level 环的超节点分组，每个环都一致，只计算一次
    std::vector<std::string> idsForIndexingVector;
    std::map<std::string, std::vector<u32>> aggregatedSubGroup;
    bool calcGroupDone = false;

    // 内层拓扑的每层环
    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        std::vector<RankInfo> tmpBridgeVector;
        bool bridgeRankFlag = false;
        std::string outLogInfo = ""; // 维护topo输出的信息
        outLogInfo.append(" ringIndex: ");
        outLogInfo.append(std::to_string(ringIndex));
        outLogInfo.append(", ");
        outLogInfo.append("userRank/serverId/devicePhyId/nicIp/isBridgeRank: ");

        // 2、填充bridge_rank_vector_的内层vector和is_bridge_vector_
        u32 subGroupIndex = 0;
        for (auto iterMap = serverToRank.begin(); iterMap != serverToRank.end(); iterMap++) {
            if (!(iterMap->second).empty()) {
                RankInfo tmpBridgePara;
                u32 bridgeUserRank = (iterMap->second)[ringIndex].userRank;
                u32 bridgeDevicePhyId = (iterMap->second)[ringIndex].devicePhyId;
                std::vector<u32> bridgeNic((iterMap->second)[ringIndex].nicIdx);
                bool checkError = (rankVector_.size() <= bridgeUserRank);
                CHK_PRT_RET(
                    checkError,
                    HCCL_ERROR(
                        "[Set][TopoInfoForLevel1]bridge userRank[%u] is bigger than rank "
                        "vector",
                        bridgeUserRank),
                    HCCL_E_INTERNAL);
                CHK_RET(SetBridgeLinkInfo(tmpBridgePara, bridgeUserRank));
                tmpBridgeVector.push_back(tmpBridgePara);
                std::vector<u32>::iterator iterNic = std::find(bridgeNic.begin(), bridgeNic.end(), bridgeDevicePhyId);
                if ((bridgeNic.size() == 0) || (iterNic != bridgeNic.end())) {
                    if (bridgeUserRank == static_cast<u32>(userRank_)) { // 本rank是否为bridge_rank
                        bridgeRankFlag = true;
                    }
                }

                outLogInfo.append(std::to_string(tmpBridgePara.userRank));
                outLogInfo.append("/");
                outLogInfo.append(tmpBridgePara.serverId);
                outLogInfo.append("/");
                outLogInfo.append(std::to_string(tmpBridgePara.devicePhyId));
                outLogInfo.append("/");
                outLogInfo.append(tmpBridgePara.nicIp[0].GetReadableAddress());
                outLogInfo.append("/");
                outLogInfo.append(std::to_string(bridgeRankFlag));
                outLogInfo.append("; ");

                // 环内填充 aggregatedSubGroup superPodId-> subGroupIndex 用于生成分组信息
                if (!calcGroupDone
                    && (deviceType_ == DevType::DEV_TYPE_910_93 || deviceType_ == DevType::DEV_TYPE_910B)) {
                    std::string IdForIndexing = (iterMap->second)[ringIndex].superPodId;
                    auto itIndex = aggregatedSubGroup.find(IdForIndexing);
                    if (itIndex != aggregatedSubGroup.end()) {
                        itIndex->second.push_back(subGroupIndex);
                    } else {
                        idsForIndexingVector.push_back(IdForIndexing);
                        std::vector<u32> subGroup;
                        subGroup.push_back(subGroupIndex);
                        aggregatedSubGroup.insert(std::make_pair(IdForIndexing, subGroup));
                    }
                    HCCL_INFO(
                        "[Set][TopoInfoForLevel1] calc subGroup superPodId[%s] subIndex[%u]", IdForIndexing.c_str(),
                        subGroupIndex);
                }

                subGroupIndex = subGroupIndex + 1;
            }
        }

        for (auto it = aggregatedSubGroup.begin(); it != aggregatedSubGroup.end(); it++) {
            HCCL_DEBUG("[Set][TopoInfoForLevel1][AHC_DEBUG] superPodId[%s]", it->first.c_str());
            for (auto index = it->second.begin(); index != it->second.end(); index++) {
                HCCL_DEBUG("[Set][TopoInfoForLevel1][AHC_DEBUG] groupIndex[%u]", (*index));
            }
        }

        for (auto it = tmpBridgeVector.begin(); it != tmpBridgeVector.end(); it++) {
            HCCL_DEBUG(
                "[Set][TopoInfoForLevel1][AHC_DEBUG] ringIndex[%u] tmpBridgevector userRank[%u]", ringIndex,
                it->userRank);
        }

        // 3、填充bridge_rank_vector_、isBridgeVector_
        if (!prepareAHC) {
            isBridgeVector_.push_back(bridgeRankFlag);
        }
        CommPlaneVector_[commPlaneLevel1].push_back(tmpBridgeVector);

        // 4、填充当前 level 的通信域内分组信息（用于层次化算法）
        if (!calcGroupDone) {
            std::vector<std::vector<u32>> SubGroups;
            for (u32 i = 0; i < idsForIndexingVector.size(); i++) {
                std::string idForIndexing = idsForIndexingVector[i];
                SubGroups.push_back(aggregatedSubGroup[idForIndexing]);
            }
            CommPlaneSubGroupVector_[commPlaneLevel1].push_back(SubGroups);
            calcGroupDone = true;
        }

        HCCL_INFO("SetTopoInfoForLevel1: topoRankInfo[%s]", outLogInfo.c_str());
    }

    HCCL_RUN_INFO(
        "SetTopoInfoForLevel1: identifier[%s], userRank[%u], userRankSize[%u], plane size[%u]", identifier_.c_str(),
        userRank_, userRankSize_, CommPlaneVector_[commPlaneLevel1].size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoInfoForLevel2()
{
    bool isLevel2Support
        = ((deviceType_ == DevType::DEV_TYPE_910_93) && multiModuleDiffDeviceNumMode_
           && !multiSuperPodDiffDeviceNumMode_);
    // 对称场景需要初始化多个平面，超节点内rank数一致也可切分平面；其他非对称场景 level1 和 level2 合并无需切分平面
    if (((!multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffServerNumMode_) || isLevel2Support)
        && !isDiffDeviceType_) {
        HCCL_INFO("[Set][TopoInfoForLevel2] select origin proc");

        // 找到当前rank在本超节点内部的序号
        auto it = superPodToRank_.find(rankData_.superPodIdx);
        CHK_PRT_RET(
            it == superPodToRank_.end(),
            HCCL_ERROR(
                "[Set][TopoInfoForLevel2]superPodIdx[%u] is not exist in superPodRankMap", rankData_.superPodIdx),
            HCCL_E_INTERNAL);

        u32 index = 0;
        for (; index < it->second.size(); ++index) {
            if (userRank_ == it->second[index].userRank) {
                break;
            }
        }
        CHK_PRT_RET(
            index >= it->second.size(),
            HCCL_ERROR(
                "[Set][TopoInfoForLevel2]userRank_[%u] superPodId[%s] superPodIdx[%u] not exist in superPodRankMap",
                userRank_, rankData_.superPodId.c_str(), rankData_.superPodIdx),
            HCCL_E_INTERNAL);

        std::vector<RankInfo> tmpRankVec;
        for (auto iterMap = superPodToRank_.begin(); iterMap != superPodToRank_.end(); iterMap++) {
            CHK_PRT_RET(
                iterMap->second.size() <= index,
                HCCL_ERROR(
                    "[Set][TopoInfoForLevel2]index[%u] is bigger than rank vector size[%u]", index,
                    iterMap->second.size()),
                HCCL_E_INTERNAL);

            RankInfo& tempRankData = iterMap->second[index];
            tmpRankVec.push_back(tempRankData);

            // 维护topo输出的信息
            std::string outLogInfo = "userRank/devicePhyId/serverIdx/superPodId: ";
            outLogInfo.append(std::to_string(tempRankData.userRank));
            outLogInfo.append("/");
            outLogInfo.append(std::to_string(tempRankData.devicePhyId));
            outLogInfo.append("/");
            outLogInfo.append(std::to_string(tempRankData.serverIdx));
            outLogInfo.append("/");
            outLogInfo.append(tempRankData.superPodId);
            outLogInfo.append("; ");
            HCCL_INFO("SetTopoInfoForLevel2: topoRankInfo[%s]", outLogInfo.c_str());
        }

        CommPlaneVector_[COMM_LEVEL2].push_back(tmpRankVec);
        HCCL_RUN_INFO(
            "SetTopoInfoForLevel2: identifier[%s], userRank[%u], userRankSize[%u], plane size[%u]", identifier_.c_str(),
            userRank_, userRankSize_, CommPlaneVector_[COMM_LEVEL2].size());
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoInfoForMeshL0()
{
    // 以MeshAggregation为粒度、MeshAggregation内各设备的mesh建链
    u32 rankSize = meshAggregationRankSize_;
    u32 userRankIndexBegin = userRank_ / meshAggregationRankSize_ * meshAggregationRankSize_;
    u32 userRankIndexEnd = userRankIndexBegin + meshAggregationRankSize_;
    std::vector<RankInfo> paraVector(rankSize);
    u32 rankIndex = 0;
    std::string outLogInfo = "userRank/devicePhyId: "; // 维护topo输出的信息

    CHK_PRT_RET(
        rankVector_.size() < userRankIndexEnd,
        HCCL_ERROR(
            "[Set][TopoInfoForMeshL0]rankVector_ size[%u] should be greater than userRankIndexEnd[%u]",
            rankVector_.size(), userRankIndexEnd),
        HCCL_E_PARA);

    for (u32 i = userRankIndexBegin; i < userRankIndexEnd; i++) {
        paraVector[rankIndex] = rankVector_[i];
        outLogInfo.append(std::to_string(paraVector[rankIndex].userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(paraVector[rankIndex].devicePhyId));
        outLogInfo.append("; ");
        rankIndex++;
    }
    CommPlaneVector_[COMM_MESH_L0].push_back(paraVector);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoInfoForMeshL1()
{
    // 以MeshAggregation为粒度、MeshAggregation间各平面的mesh建链
    u32 rankSize = userRankSize_ / meshAggregationRankSize_; // 1 = 7 / 4
    u32 planeID = userRank_ % meshAggregationRankSize_;      // 0
    std::vector<RankInfo> paraVector(rankSize);

    CHK_PRT_RET(
        rankVector_.size() < userRankSize_,
        HCCL_ERROR(
            "[Set][TopoInfoForMeshL1]rankVector_ size[%u] should be greater than userRankSize[%u]", rankVector_.size(),
            userRankSize_),
        HCCL_E_PARA);

    for (u32 i = planeID; i < userRankSize_; i += meshAggregationRankSize_) {
        u32 rankIndex = i / meshAggregationRankSize_;
        paraVector[rankIndex] = rankVector_[i];
        std::string outLogInfo = "userRank/devicePhyId"; // 维护topo输出的信息
        outLogInfo.append(std::to_string(paraVector[rankIndex].userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(paraVector[rankIndex].devicePhyId));
        outLogInfo.append("; ");
        HCCL_INFO("SetTopoInfoForMeshL1: topoRankInfo[%s]", outLogInfo.c_str());
    }
    CommPlaneVector_[COMM_MESH_L1].push_back(paraVector);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetSingleLevel0()
{
    // 填充level0_rank_vector_，该函数处理场景非8P_RING算法
    std::vector<RankInfo> tmpLevel0Vector;
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Set][SingleLevel0]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    std::vector<s32> devicePhyIdVector;
    for (u32 i = 0; i < rankVector_.size(); i++) {
        devicePhyIdVector.push_back(rankVector_[i].devicePhyId);
    }
    s32 maxPhyId = *max_element(devicePhyIdVector.begin(), devicePhyIdVector.end());
    // 8P满配场景:4PMESH算法 + 8Pfullmesh + 16P仅使用左边module
    if (((iterRank->second).size() == DEVICE_PER_MODULE && maxPhyId < DEVICE_PER_MODULE)
        && (topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_NP_MESH)) {
        return SetSingleLevel0For8P(); // 服务器内dev个数相同已在hcom层做过校验
    }

    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";
    RankInfo tempRankData;
    // 其他场景 + 16P使用右边module
    for (u32 startIndex = 0; startIndex < (iterRank->second).size(); startIndex++) {
        u32 level0StartRank = (iterRank->second)[startIndex].userRank;
        bool checkError = (rankVector_.size() <= level0StartRank);
        CHK_PRT_RET(
            checkError,
            HCCL_ERROR("[Set][SingleLevel0]level0 userRank[%u] is bigger than rank vector", level0StartRank),
            HCCL_E_INTERNAL);
        tempRankData = rankVector_[level0StartRank];

        outLogInfo.append(std::to_string(tempRankData.userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(tempRankData.devicePhyId));
        outLogInfo.append("; ");
        tmpLevel0Vector.push_back(tempRankData);
    }

    // NPmesh或4Pmesh场景下，外层拓扑平面为device数量-1
    u32 level0Size = (topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_NP_MESH) ?
                         (ranksOneNode_[static_cast<u32>(topoType_)] - 1) :
                         1;

    for (u32 index = 0; index < level0Size; index++) {
        CommPlaneVector_[COMM_LEVEL0].push_back(tmpLevel0Vector);
    }
    HCCL_RUN_INFO(
        "SetTopoInfoForLevel0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]", identifier_.c_str(),
        userRank_, userRankSize_, outLogInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetSingleLevel0For8P()
{
    // 该函数处理场景:8P满配、非8P_RING算法(8P满配下走4PMESH)
    std::vector<RankInfo> tmpLevel0Vector;
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Set][SingleLevel0For8P]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    u32 startIndex
        = (rankData_.devicePhyId < static_cast<s32>(meshAggregationRankSize_)) ? 0 : meshAggregationRankSize_;
    u32 devcount = 0;
    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";
    RankInfo tempRankData;
    while (devcount < meshAggregationRankSize_) {
        u32 level0StartRank = (iterRank->second)[startIndex].userRank;
        bool checkError = (rankVector_.size() <= level0StartRank);
        CHK_PRT_RET(
            checkError,
            HCCL_ERROR("[Set][SingleLevel0For8P]level0 userRank[%u] is bigger than rank vector", level0StartRank),
            HCCL_E_INTERNAL);
        tempRankData = rankVector_[level0StartRank];

        outLogInfo.append(std::to_string(tempRankData.userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(tempRankData.devicePhyId));
        outLogInfo.append("; ");
        tmpLevel0Vector.push_back(tempRankData);
        startIndex++;
        devcount++;
    }

    // 4PMESH场景下，外层拓扑3个平面
    u32 level0Size = ranksOneNode_[static_cast<u32>(topoType_)] - 1;
    for (u32 index = 0; index < level0Size; index++) {
        CommPlaneVector_[COMM_LEVEL0].push_back(tmpLevel0Vector);
    }
    HCCL_RUN_INFO(
        "SetTopoInfoForLevel0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]", identifier_.c_str(),
        userRank_, userRankSize_, outLogInfo.c_str());
    return HCCL_SUCCESS;
}

/*
 * *********************************************************************************
 * 用来标识集群中是否存在910B A+X形态
 * **********************************************************************************
 */
bool TopoInfoExtractor::IsDiffDeviceModuleInServer() const
{
    return deviceType_ == DevType::DEV_TYPE_910B && isDiffAggregation_;
}

HcclResult TopoInfoExtractor::SetMultiLevel0(u32 ringNum)
{
    std::vector<u32> tmpLevel0Order;
    u32 moduleIdx = 0;
    RankInfo tempRankData;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Set][MultiLevel0]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    // 维护topo输出的信息
    std::string outLogInfo = "";
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        tmpLevel0Order = multiLevel0Order_[ringIndex]; // 获取每一个环的设备物理ID排序
        std::vector<RankInfo> tmpLevel0Vector;
        outLogInfo = "userRank/devicePhyId: ";
        for (u32 startIndex = 0; startIndex < (iterRank->second).size(); startIndex++) {
            u32 devIndex = tmpLevel0Order[startIndex];
            u32 level0RingUserank = (iterRank->second)[devIndex].userRank;
            bool checkError = (rankVector_.size() <= level0RingUserank);
            CHK_PRT_RET(
                checkError,
                HCCL_ERROR("[Set][MultiLevel0]level0 userRank[%u] is bigger than rank vector", level0RingUserank),
                HCCL_E_INTERNAL);
            tempRankData = rankVector_[level0RingUserank];
            outLogInfo.append(std::to_string(tempRankData.userRank));
            outLogInfo.append("/");
            outLogInfo.append(std::to_string(tempRankData.devicePhyId));
            outLogInfo.append("; ");
            tmpLevel0Vector.push_back(tempRankData);
        }
        HCCL_RUN_INFO(
            "SetTopoInfoForLevel0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]",
            identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
        CommPlaneVector_[COMM_LEVEL0].push_back(tmpLevel0Vector);
    }
    return HCCL_SUCCESS;
}

// anypath创建通信域
HcclResult TopoInfoExtractor::SetMultiLevel0AnyPath(std::vector<std::vector<u32>> multiOrder)
{
    u32 ringNum = multiOrder.size();
    std::vector<u32> tmpLevel0Order;
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Set][MultiLevel0]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    // 维护topo输出的信息
    std::string outLogInfo = "";
    RankInfo tempRankData;
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        tmpLevel0Order = multiOrder[ringIndex]; // 获取每一个环的设备物理ID排序
        std::vector<RankInfo> tmpLevel0Vector;
        outLogInfo = "userRank/devicePhyId: ";
        for (u32 startIndex = 0; startIndex < (iterRank->second).size(); startIndex++) {
            u32 devIndex = tmpLevel0Order[startIndex];
            u32 level0RingUserank = (iterRank->second)[devIndex].userRank;
            bool checkError = (rankVector_.size() <= level0RingUserank);
            CHK_PRT_RET(
                checkError,
                HCCL_ERROR("[Set][MultiLevel0]level0 userRank[%u] is bigger than rank vector", level0RingUserank),
                HCCL_E_INTERNAL);
            tempRankData = rankVector_[level0RingUserank];
            outLogInfo.append(std::to_string(tempRankData.userRank));
            outLogInfo.append("/");
            outLogInfo.append(std::to_string(tempRankData.devicePhyId));
            outLogInfo.append("; ");
            tmpLevel0Vector.push_back(tempRankData);
        }
        HCCL_RUN_INFO(
            "[AnyPath]SetTopoInfoForLevel0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]",
            identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
        CommPlaneVector_[COMM_LEVEL0_ANYPATH_SDMA].push_back(tmpLevel0Vector);
        CommPlaneVector_[COMM_LEVEL0_ANYPATH_RDMA].push_back(tmpLevel0Vector);
    }

    return HCCL_SUCCESS;
}

// 集群中存在910B A+X时，0-7卡: moduleIdx = 2 * serverIdx; 8-15卡: moduleIdx = 2 * serverIdx + 1
// 集群中不存在910B A+X时，moduleIdx = serverIdx
HcclResult TopoInfoExtractor::GetModuleIdx(const RankInfo& rankInfo, u32& moduleIdx)
{
    // 获取moduleIdx，在16P同时使用左右两个module时，moduleIdx标识当前rank所在的module，其他场景下moduleIdx等同于serverIdx
    u32 serverIdx = 0;
    CHK_RET(GetServerIdx(rankInfo, serverIdx));
    if (isDiffDeviceType_) {
        moduleIdx = rankInfo.userRank / gcdDeviceNumPerAggregation_;
        HCCL_DEBUG(
            "[TopoInfoExtractor][GetModuleIdx]serverIdx [%u] devicePhyId[%u] userRank[%u] moduleIdx[%u] "
            "gcdDeviceNumPerAggregation[%u]",
            serverIdx, rankInfo.devicePhyId, rankInfo.userRank, moduleIdx, gcdDeviceNumPerAggregation_);
    } else if (IsDiffDeviceModuleInServer()) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO + rankInfo.devicePhyId / DEVICE_PER_MODULE;
    } else {
        moduleIdx = serverIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetBridgeLinkInfo(RankInfo& bridgePara, u32 bridgeUserRank)
{
    bool checkSize = (rankVector_.size() <= bridgeUserRank);
    CHK_PRT_RET(
        checkSize, HCCL_ERROR("[Set][BridgeLinkInfo]bridge UserRank %u is bigger than rank vector", bridgeUserRank),
        HCCL_E_INTERNAL);

    bridgePara = rankVector_[bridgeUserRank];
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoDefaultInfoFor8P()
{
    // 填充combined_rank_vector_:不区分board_type,只生成default单层拓扑
    std::vector<RankInfo> tmpCombinedVector;
    // 服务器内排序固定为0, 2, 3, 1, 5, 7, 6, 4，挑选8P多环中适用于combined的一组服务器内排序
    std::vector<u32> devOrder = {0, 2, 3, 1, 5, 7, 6, 4};
    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";

    // 填充combined_rank_vector_的内层vector:combined场景只有一条固定的环
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        /* 服务器内8P满配单环特殊适配逻辑 */
        for (u32 index = 0; index < devOrder.size(); index++) {
            u32 devIndex = devOrder[index];
            u32 combinedUserRank = (iterMap->second)[devIndex].userRank;

            bool checkError = (rankVector_.size() <= combinedUserRank);
            CHK_PRT_RET(
                checkError,
                HCCL_ERROR(
                    "[Set][TopoDefaultInfoFor8P]combined userRank[%u] is bigger than rank vector", combinedUserRank),
                HCCL_E_INTERNAL);

            RankInfo tmpCombinedPara = rankVector_[combinedUserRank];
            outLogInfo.append(std::to_string(tmpCombinedPara.userRank));
            outLogInfo.append("/");
            outLogInfo.append(std::to_string(tmpCombinedPara.devicePhyId));
            outLogInfo.append("; ");
            tmpCombinedVector.push_back(tmpCombinedPara);
        }
    }

    CommPlaneVector_[COMM_COMBINE].push_back(tmpCombinedVector);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>>& CommPlaneRanks)
{
    CommPlaneRanks.resize(CommPlaneVector_.size());
    for (u32 level = 0; level < CommPlaneVector_.size(); level++) {
        u32 ringSize = CommPlaneVector_[level].size();
        CommPlaneRanks[level].resize(ringSize);
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            u32 rankSize = CommPlaneVector_[level][ringIndex].size();
            CommPlaneRanks[level][ringIndex].resize(rankSize);
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                u32 userRank = CommPlaneVector_[level][ringIndex][rankIndex].userRank;
                CommPlaneRanks[level][ringIndex][rankIndex] = userRank;
                HCCL_DEBUG("GetCommPlaneRanks CommPlaneRanks[%u][%u][%u]=%u", level, ringIndex, rankIndex, userRank);
            }
        }
    }
    return HCCL_SUCCESS;
}

void TopoInfoExtractor::GetIsBridgeVector(std::vector<bool>& isBridgeVector)
{
    isBridgeVector = isBridgeVector_;
    return;
}

HcclResult TopoInfoExtractor::GetIsUsedRdmaMap(std::unordered_map<u32, bool>& isUsedRdmaMap)
{
    for (const RankInfo& dstRank : rankVector_) {
        bool isInterSuperPod = false;
        bool isInterServer = false;
        uint32_t isConnectedWithPcie = 0;
        if (rankData_.superPodId != dstRank.superPodId) { // 跨超节点场景
            isInterSuperPod = true;
        } else if (rankData_.serverIdx != dstRank.serverIdx) { // 不跨超节点, 跨server场景
            isInterServer = true;
        } else { // 同server, PCIE互连场景
            HCCL_DEBUG("[TopoInfoExtractor]GetIsUsedRdmaMap for interServer");
            auto it = deviceLinkTypeMap_.find(dstRank.devicePhyId);
            CHK_PRT_RET(
                it == deviceLinkTypeMap_.end(),
                HCCL_ERROR("can't find devicePhyId[%d] in deviceLinkTypeMap_", dstRank.devicePhyId), HCCL_E_NOT_FOUND);
            isConnectedWithPcie = (it->second == LinkTypeInServer::PXI_TYPE) ? true : false;
        }
        // 使能RDMA的场景: 1.跨超节点  2.跨server且不使能HCCS  3.PCIE连接且使能RDMA开关
        bool isUsedRdma = (isInterSuperPod) || (isInterServer && !isUsedInterHccsMode_)
                          || (isConnectedWithPcie && isUsedRdmaLevel0_);
        isUsedRdmaMap[dstRank.userRank] = isUsedRdma;
        HCCL_DEBUG(
            "[GetIsUsedRdma]isUsedRdma[%u], isInterSuperPod[%u], isInterServer[%u], isUsedInterHccsMode_[%u], "
            "isConnectedWithPcie[%u], isUsedRdmaLevel0_[%u], dstRank[%u]",
            isUsedRdma, isInterSuperPod, isInterServer, isUsedInterHccsMode_, isConnectedWithPcie, isUsedRdmaLevel0_,
            dstRank.userRank);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::GetRankVecInfo(std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank)
{
    std::vector<std::vector<u32>> serverToRank;
    std::vector<std::vector<u32>> superPodToRank;
    serverToRank.clear();
    superPodToRank.clear();
    u32 firstIdx = 0;

    serverToRank.resize(serverToRank_.size());
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        serverToRank[firstIdx].resize((iterMap->second).size());
        if (!(iterMap->second).empty()) {
            for (u32 i = 0; i < (iterMap->second).size(); i++) {
                serverToRank[firstIdx][i] = (iterMap->second)[i].userRank;
            }
        }
        firstIdx++;
    }

    u32 podFirstIdx = 0;
    superPodToRank.resize(superPodToRank_.size());
    for (auto iterMap = superPodToRank_.begin(); iterMap != superPodToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            superPodToRank[podFirstIdx].resize((iterMap->second).size());
            for (u32 i = 0; i < (iterMap->second).size(); i++) {
                superPodToRank[podFirstIdx][i] = (iterMap->second)[i].userRank;
                HCCL_DEBUG("GetRankVecInfo superPodToRank[%u][%u]=%u", podFirstIdx, i, superPodToRank[podFirstIdx][i]);
            }
        }
        podFirstIdx++;
    }
    serverAndsuperPodToRank.push_back(serverToRank);
    serverAndsuperPodToRank.push_back(superPodToRank);
    return HCCL_SUCCESS;
}
HcclResult TopoInfoExtractor::SetTopoInfoForCombineL1()
{
    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        // 按照superPodIdx 划分得所有rank信息
        for (auto iter = superPodToRank_.begin(); iter != superPodToRank_.end(); iter++) {
            if (iter->first != rankData_.superPodIdx) {
                continue; // 只在自己所在的超节点创建
            }
            CommPlaneVector_[COMM_COMBINE_L1].push_back(iter->second);
            HCCL_DEBUG("[SetTopoInfoForARS]Superpod rankSize[%u].", CommPlaneVector_[COMM_COMBINE_L1][0].size());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExtractor::SetTopoInfoForARS() // 针对ARS特性
{
    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        for (auto iter = superPodToRank_.begin(); iter != superPodToRank_.end(); iter++) {
            if (iter->first != rankData_.superPodIdx) {
                continue; // 只在自己所在的超节点创建
            }
            CommPlaneVector_[COMM_ARS].push_back(iter->second);
            HCCL_DEBUG("[SetTopoInfoForARS]Superpod rankSize[%u].", CommPlaneVector_[COMM_ARS][0].size());
        }
    }
    return HCCL_SUCCESS;
}

void TopoInfoExtractor::GetCommPlaneVector(std::vector<std::vector<std::vector<RankInfo>>>& commPlaneVector)
{
    commPlaneVector = CommPlaneVector_;
    return;
}

void TopoInfoExtractor::InitAHCConfig(std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfig)
{
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        std::vector<HcclAlgoType> algoType = algoConfig[static_cast<HcclCMDType>(opType)];
        isConfigAHC_
            = (algoType[HCCL_ALGO_LEVEL_1] == HcclAlgoType::HCCL_ALGO_TYPE_AHC
               || algoType[HCCL_ALGO_LEVEL_1] == HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE);
        if (isConfigAHC_) {
            HCCL_INFO("[InitAHCConfig] set AHC alg, opType[%u]", opType);
            break;
        }
    }

    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX);
         opType++) { // 没配置算法的情况下默认会走AHC嘛？  给测试用
        std::vector<HcclAlgoType> algoType = algoConfig[static_cast<HcclCMDType>(opType)];
        isConfigNULL_ = algoType[HCCL_ALGO_LEVEL_0] == HcclAlgoType::HCCL_ALGO_TYPE_NULL;
        if (isConfigNULL_) {
            HCCL_INFO("[InitAHCConfig] set NULL alg, opType[%u]", opType);
            break;
        }
    }
    return;
}

HcclResult TopoInfoExtractor::SetAHCSubGroupsAndAlgOption()
{
    // 解析subgroup
    CHK_RET(SetTopoInfoForLevel1(true));

    // 分组切分
    CHK_RET(AHCSubGroupInit(COMM_LEVEL1_AHC, CommPlaneSubGroupVector_));
    if (deviceType_ != DevType::DEV_TYPE_910_93) {
        CHK_RET(AHCSubGroupInit(COMM_COMBINE, CommPlaneSubGroupVector_));
    } else {
        CHK_RET(AHCSubGroupInit(COMM_COMBINE_ORDER, CommPlaneSubGroupVector_));
    }

    // 算法配置初始化
    CommAHCBaseInfo::InitConcAlgOption(ahcAlgOption_);
    return HCCL_SUCCESS;
}

void TopoInfoExtractor::AHCCommSubgroupInit()
{
    if (deviceType_ != DevType::DEV_TYPE_910_93) {
        // 用于910B AHC COMM_COMBINE 通信域分组场景测试
        std::map<std::string, std::vector<u32>> serverIDGroup;
        std::vector<std::string> serverIdsVector;
        for (u32 i = 0; i < rankVector_.size(); i++) {
            auto itServerID = serverIDGroup.find(rankVector_[i].serverId);
            if (itServerID != serverIDGroup.end()) {
                itServerID->second.push_back(i);
            } else {
                serverIdsVector.push_back(rankVector_[i].serverId);
                std::vector<u32> subGroup;
                subGroup.push_back(i);
                serverIDGroup.insert(std::make_pair(rankVector_[i].serverId, subGroup));
            }
        }
        std::vector<std::vector<u32>> serverSubGroups;
        for (u32 i = 0; i < serverIdsVector.size(); i++) {
            serverSubGroups.push_back(serverIDGroup[serverIdsVector[i]]);
        }
        CommPlaneSubGroupVector_[COMM_COMBINE].push_back(serverSubGroups);
    } else {
        // 用于910_93 AHC COMM_COMBINE_ORDER 通信域分组场景测试
        std::vector<std::string> idsForIndexingVector;
        std::map<std::string, std::vector<u32>> superPodIdGroup;
        for (u32 i = 0; i < rankVector_.size(); i++) {
            auto itSuperPodID = superPodIdGroup.find(rankVector_[i].superPodId);
            if (itSuperPodID != superPodIdGroup.end()) {
                itSuperPodID->second.push_back(i);
            } else {
                idsForIndexingVector.push_back(rankVector_[i].superPodId);
                std::vector<u32> subGroup;
                subGroup.push_back(i);
                superPodIdGroup.insert(std::make_pair(rankVector_[i].superPodId, subGroup));
            }
        }
        std::vector<std::vector<u32>> superPodSubGroups;
        for (u32 i = 0; i < idsForIndexingVector.size(); i++) {
            superPodSubGroups.push_back(superPodIdGroup[idsForIndexingVector[i]]);
            HCCL_DEBUG("[SetTopoInfoForLevel0][AHC_DEBUG 910_93] superPodId[%s]", idsForIndexingVector[i].c_str());
            for (u32 index = 0; index < superPodIdGroup[idsForIndexingVector[i]].size(); index++) {
                HCCL_DEBUG(
                    "[SetTopoInfoForLevel0][AHC_DEBUG 910_93] groupIdx[%u]",
                    superPodIdGroup[idsForIndexingVector[i]][index]);
            }
        }
        CommPlaneSubGroupVector_[COMM_COMBINE_ORDER].push_back(superPodSubGroups);
    }
    return;
}

void TopoInfoExtractor::GetCommPlaneSubGroupVector(
    std::vector<std::vector<std::vector<std::vector<u32>>>>& CommPlaneSubGroupVector)
{
    CommPlaneSubGroupVector = CommPlaneSubGroupVector_;
    return;
}

void TopoInfoExtractor::SetAHCLevelAlgOption(AHCConcOpType ahcConcOpType, TemplateType templateType)
{
    if (ahcAlgOption_.find(ahcConcOpType) == ahcAlgOption_.end()) {
        ahcAlgOption_[ahcConcOpType] = templateType;
    }
    return;
}

void TopoInfoExtractor::GetAHCAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    ahcAlgOption = ahcAlgOption_;
    return;
}

void TopoInfoExtractor::GetRankData(RankInfo& rankData)
{
    rankData = rankData_;
    return;
}

void TopoInfoExtractor::GetServerToRank(std::map<u32, std::vector<RankInfo>>& serverToRank)
{
    serverToRank = serverToRank_;
    return;
}

void TopoInfoExtractor::GetSuperPodToRank(std::map<u32, std::vector<RankInfo>>& superPodToRank)
{
    superPodToRank = superPodToRank_;
    return;
}

void TopoInfoExtractor::GetDeviceLinkTypeMap(std::map<s32, LinkTypeInServer>& deviceLinkTypeMap)
{
    deviceLinkTypeMap = deviceLinkTypeMap_;
    return;
}

HcclResult TopoInfoExtractor::AHCSubGroupInit(
    CommPlane algLevel, std::vector<std::vector<std::vector<std::vector<u32>>>>& CommPlaneSubGroupVector) const
{
    std::vector<std::vector<std::vector<u32>>>& globalSubGroups = CommPlaneSubGroupVector[algLevel];

    // globalSubGroups 参数检查
    if ((globalSubGroups.size() == 0) || (globalSubGroups[0].size() == 0)) {
        return HCCL_SUCCESS;
    }

    CHK_RET(CommAHCBaseInfo::CheckGlobalGroups(globalSubGroups));

    u32 minSubGroupSize = globalSubGroups[0][0].size();
    u32 maxSubGroupSize = globalSubGroups[0][0].size();
    u32 subGroupSizeGCD = globalSubGroups[0][0].size();
    for (u32 i = 1; i < globalSubGroups[0].size(); ++i) {
        subGroupSizeGCD = std::__gcd(subGroupSizeGCD, static_cast<u32>(globalSubGroups[0][i].size()));
        if (globalSubGroups[0][i].size() < minSubGroupSize) {
            minSubGroupSize = globalSubGroups[0][i].size();
        }
        if (globalSubGroups[0][i].size() > maxSubGroupSize) {
            maxSubGroupSize = globalSubGroups[0][i].size();
        }
    }

    u32 splitDivisor = minSubGroupSize;

    HCCL_DEBUG(
        "[TopoInfoExtractor][AHCSubGroupInit] begin groupSplit algLevel = %u minSubGroupSize = %u maxSubGroupSize = %u "
        "SubGroupSizeGCD = %u "
        "splitDivisor = %u",
        algLevel, minSubGroupSize, maxSubGroupSize, subGroupSizeGCD, splitDivisor);

    // 切分分组逻辑, 满足整数倍切分，满足最大偏差大于等于最小分组数时切分
    if ((subGroupSizeGCD == minSubGroupSize) || (maxSubGroupSize / splitDivisor > 1)) {
        // 设置为 BROKE 类型
        maxSubGroupSize = splitDivisor;
        // 将所有的分组切分成 splitDivisor 粒度的 subGroup
        for (u32 i = 0; i < globalSubGroups[0].size(); ++i) {
            if (globalSubGroups[0][i].size() / splitDivisor <= 1) {
                continue;
            }
            std::vector<u32> originGroup = globalSubGroups[0][i];
            globalSubGroups[0].erase(globalSubGroups[0].begin() + i);
            u32 splitGroupsNum = (originGroup.size() / splitDivisor);
            for (u32 j = 0; j < splitGroupsNum - 1; ++j) {
                globalSubGroups[0].insert(
                    globalSubGroups[0].begin() + i + j,
                    std::vector<u32>(
                        originGroup.begin() + j * splitDivisor, originGroup.begin() + (j + 1) * splitDivisor));
            }
            globalSubGroups[0].insert(
                globalSubGroups[0].begin() + i + splitGroupsNum - 1,
                std::vector<u32>(
                    originGroup.begin() + (splitGroupsNum - 1) * splitDivisor,
                    originGroup.begin() + originGroup.size()));
            maxSubGroupSize = std::max(
                maxSubGroupSize, static_cast<u32>(originGroup.size()) - ((splitGroupsNum - 1) * splitDivisor));
            i--;
        }
    }

    HCCL_DEBUG(
        "[TopoInfoExtractor][AHCSubGroupInit] after groupSplit algLevel = %u minSubGroupSize = %u maxSubGroupSize = %u "
        "SubGroupSizeGCD = %u "
        "splitDivisor = %u",
        algLevel, minSubGroupSize, maxSubGroupSize, subGroupSizeGCD, splitDivisor);

    CHK_RET(CommAHCBaseInfo::CheckGlobalGroups(globalSubGroups));

    return HCCL_SUCCESS;
}

bool Ascending(const RankInfo& first, const RankInfo& second)
{
    if (first.serverIdx != second.serverIdx) {
        return first.serverIdx < second.serverIdx;
    } else {
        return first.userRank < second.userRank;
    }
}

bool CompareWithUserRankAscend(const RankInfo& left, const RankInfo& right) { return left.userRank < right.userRank; }

// 适配ROH平面网段隔离，奇数rank互通，偶数rank互通，奇偶不通
bool CheckSdmaWithRohTopo(const std::vector<u32>& nicList, std::vector<u32>& topoList)
{
    std::vector<u32> tmpNicList(nicList);
    SearchPath searchPath;
    std::sort(tmpNicList.begin(), tmpNicList.end());
    topoList = searchPath.Search(tmpNicList);
    if (topoList.empty()) {
        return false;
    }
    return true;
}

std::vector<std::vector<u32>> GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList)
{
    std::vector<std::vector<u32>> multiRingOrder;
    if (topoType == TopoType::TOPO_TYPE_8P_RING) { // 4 ring 场景
        // 每个环的排序是按照设备物理ID进行的
        std::vector<u32> tmpLevel00 = {0, 1, 2, 6, 5, 4, 7, 3}; // 环0
        std::vector<u32> tmpLevel01 = {0, 3, 7, 4, 5, 6, 2, 1}; // 环1
        std::vector<u32> tmpLevel02 = {0, 2, 3, 1, 5, 7, 6, 4}; // 环2
        std::vector<u32> tmpLevel03 = {0, 4, 6, 7, 5, 1, 3, 2}; // 环3

        // 填充8pring 多环的comm level0 四个环的顺序
        multiRingOrder.push_back(tmpLevel00);
        multiRingOrder.push_back(tmpLevel01);
        multiRingOrder.push_back(tmpLevel02);
        multiRingOrder.push_back(tmpLevel03);
    } else if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING) { // 2 ring 场景
        std::vector<u32> tmpLevel00;                             // 环0
        std::vector<u32> tmpLevel01;                             // 环1
        tmpLevel00 = nicList;                                    // { 0, 1, 2, 3, 4, 5, 6, 7 };
        tmpLevel01.reserve(ranksSize);
        tmpLevel01.push_back(nicList[0]);
        tmpLevel01.insert(tmpLevel01.end(), tmpLevel00.rbegin(), tmpLevel00.rend() - 1);
        HCCL_INFO("[GetRingsOrderByTopoType] TopoType:TOPO_TYPE_NP_DOUBLE_RING");
        // 填充 double ring 两环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
        multiRingOrder.push_back(tmpLevel01);
    } else {                                   // 1 ring 场景
        std::vector<u32> tmpLevel00 = nicList; // 环0

        // 填充 single ring 单环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
    }
    // 打印多个环
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        for (size_t i = 0; i < multiRingOrder.size(); i++) {
            auto ring = multiRingOrder[i];
            std::ostringstream stringRepresentation;
            for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
                stringRepresentation << *it << " ";
            }
            std::string ringString = stringRepresentation.str();
            const char* charRing = ringString.c_str();
            HCCL_DEBUG("[GetRingsOrderByTopoType] The No.%zu ring: %s", i, charRing);
        }
    }
    return multiRingOrder;
}

std::vector<std::vector<u32>> GetRingsOrderForAnyPath(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList)
{
    std::vector<std::vector<u32>> multiRingOrder;
    if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING) { // 2 ring 场景
        std::vector<u32> tmpLevel00;                      // 环0
        std::vector<u32> tmpLevel01;                      // 环1
        std::vector<u32> rohLevel0;
        if (CheckSdmaWithRohTopo(nicList, rohLevel0)) {
            tmpLevel00 = rohLevel0;        // 环0, 8卡 { 0, 1, 3, 2, 4, 5, 7, 6 };
            tmpLevel01.reserve(ranksSize); // 环1, 8卡 { 0, 6, 7, 5, 4, 2, 3, 1 };
            tmpLevel01.push_back(rohLevel0[0]);
            tmpLevel01.insert(tmpLevel01.end(), rohLevel0.rbegin(), rohLevel0.rend() - 1);
        } else {
            tmpLevel00 = nicList; // { 0, 1, 2, 3, 4, 5, 6, 7 };
            tmpLevel01.reserve(ranksSize);
            tmpLevel01.push_back(nicList[0]);
            tmpLevel01.insert(tmpLevel01.end(), tmpLevel00.rbegin(), tmpLevel00.rend() - 1);
        }
        // 填充 double ring 两环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
        multiRingOrder.push_back(tmpLevel01);
    } else {                                   // 1 ring 场景
        std::vector<u32> tmpLevel00 = nicList; // 环0

        // 填充 single ring 单环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
    }
    // 打印多个环
    HCCL_DEBUG("[GetRingsOrderForAnyPath]print rings:");
    for (size_t i = 0; i < multiRingOrder.size(); i++) {
        auto ring = multiRingOrder[i];
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string ringString = stringRepresentation.str();
        const char* charRing = ringString.c_str();
        HCCL_INFO("[GetRingsOrderByRdmaSdmaConcurrent] The No.%zu ring: %s.", i, charRing);
    }
    return multiRingOrder;
}

} // namespace hccl
