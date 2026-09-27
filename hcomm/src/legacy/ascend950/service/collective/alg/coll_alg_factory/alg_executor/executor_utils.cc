/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_utils.h"

namespace Hccl {
bool IsEnableCounterNotifyByDevType(const RankId myRank, const DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_950:
        case DevType::DEV_TYPE_960:
            HCCL_DEBUG("[CollAlgFactory] Rank [%d], CounterNotify func enabled.", myRank);
            return true;
        default:
            HCCL_DEBUG("[CollAlgFactory] Rank [%d], CounterNotify func disabled.", myRank);
            return false;
    }
}

HcclResult InitOpInfo(const CollAlgOperator& op, OpType& opType, ReduceOp& redOp, u32& root)
{
    opType = op.opType;
    switch (opType) {
        case OpType::ALLREDUCE:
        case OpType::REDUCESCATTER:
            redOp = op.reduceOp;
            break;
        case OpType::SCATTER:
        case OpType::BROADCAST:
            root = op.root;
            break;
        case OpType::REDUCE:
            redOp = op.reduceOp;
            root = op.root;
            break;
        default:
            break;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InitDataInfo(const CollAlgOperator& op, DataType& dataType, DataType& outputDataType, u64& dataCount)
{
    dataType = op.dataType;
    dataCount = op.dataCount;
    outputDataType = op.outputDataType;
    if (outputDataType == DataType::INVALID) {
        outputDataType = dataType;
    }

    return HcclResult::HCCL_SUCCESS;
}

// Get Prior Link from virtual topo
const std::vector<NetInstance::Path>
GetPathsFromRankGraph(const RankGraph* rankGraph, const RankId srcRank, const RankId dstRank)
{
    // 遍历当前节点的所有层级，返回两个节点间查到到的所有path
    std::vector<NetInstance::Path> pathList;
    std::set<u32> levelSet = rankGraph->GetLevels(srcRank);
    for (u32 levelIdx : levelSet) {
        std::vector<NetInstance::Path> paths = rankGraph->GetPaths(levelIdx, srcRank, dstRank);
        pathList.insert(pathList.end(), paths.begin(), paths.end());
    }
    return pathList;
}

HcclResult AddToResLinks(const RankId vNeighborRank, const LinkData& linkData, ResLinks& resLinks)
{
    HCCL_DEBUG(
        "RankId [%d] linkData.des[%s] resLinks[%zu]", vNeighborRank, linkData.Describe().c_str(), resLinks.size());
    auto rankLinkIter = resLinks.find(vNeighborRank);
    if (rankLinkIter == resLinks.end()) {
        std::vector<LinkData> tmpLinks = {linkData};
        resLinks.insert(std::pair<RankId, std::vector<LinkData>>(vNeighborRank, tmpLinks));
    } else {
        rankLinkIter->second.push_back(linkData);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult PrepResLinks(
    const RankId myRank, const RankGraph* rankGraph, const std::vector<BasePortType>& linkPriority,
    const LinkReq& linkReq, ResLinks& resLinks)
{
    HCCL_DEBUG("PrepResLinks linkPriority.size()[%zu], linkReq.size()[%zu]", linkPriority.size(), linkReq.size());
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        const std::vector<NetInstance::Path> tmpPaths = GetPathsFromRankGraph(rankGraph, myRank, resReqIter->first);
        if (resReqIter->second == 1) {
            CHK_PRT_RET(
                tmpPaths.size() == 0,
                HCCL_ERROR(
                    "[CollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank,
                    resReqIter->first),
                HcclResult::HCCL_E_INTERNAL);
            LinkData requiredLinkData(tmpPaths[0]); // 当前只取第一条path
            // update res
            CHK_PRT_RET(
                AddToResLinks(resReqIter->first, requiredLinkData, resLinks) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CollAlgFactory] Rank [%d], Fail to prepare links.", myRank), HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                tmpPaths.size() < resReqIter->second,
                HCCL_ERROR("[CollAlgFactory] Rank [%d], available linkNum smaller than required.", myRank),
                HcclResult::HCCL_E_INTERNAL);
            // 从所有path中选择前resReqIter->second条
            for (u32 linkNum = 0; linkNum < resReqIter->second; linkNum++) {
                LinkData requiredLinkData(tmpPaths[linkNum]);
                // update res
                CHK_PRT_RET(
                    AddToResLinks(resReqIter->first, requiredLinkData, resLinks) != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[CollAlgFactory] Rank [%d], Fail to prepare links.", myRank),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult PrepResLinks(const RankId myRank, const LinkReq& linkReq, ConnectedLinkMgr* linkMgr, ResLinks& resLinks)
{
    CHK_PTR_NULL(linkMgr);
    HCCL_DEBUG("PrepResLinks linkReq.size()[%zu]", linkReq.size());
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        if (resReqIter->second == 1) {
            auto rankId = resReqIter->first;
            auto links = linkMgr->GetLinks(rankId);
            CHK_PRT_RET(
                links.size() == 0, HCCL_ERROR("[PrepResLinks] Rank [%d], Fail to get peer links.", myRank),
                HcclResult::HCCL_E_INTERNAL);
            LinkData requiredLinkData = links[0];
            // update res
            CHK_PRT_RET(
                AddToResLinks(resReqIter->first, requiredLinkData, resLinks) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CollAlgFactory] Rank [%d], Fail to prepare links.", myRank), HcclResult::HCCL_E_INTERNAL);
        } else {
            for (u32 linkNum = 0; linkNum < resReqIter->second; linkNum++) {
                LinkData requiredLinkData = linkMgr->GetLinks(resReqIter->first)[linkNum];
                // update res
                CHK_PRT_RET(
                    AddToResLinks(resReqIter->first, requiredLinkData, resLinks) != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[CollAlgFactory] Rank [%d], Fail to prepare links.", myRank),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcResLinks(
    const RankId myRank, const RankGraph* rankGraph, const std::vector<BasePortType>& linkPriority,
    const LinkReq& linkReq, std::vector<LinkData>& links)
{
    HCCL_DEBUG("CalcResLinks linkPriority.size()[%zu]", linkPriority.size());
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        const std::vector<NetInstance::Path> tmpPaths = GetPathsFromRankGraph(rankGraph, myRank, resReqIter->first);
        if (resReqIter->second == 1) {
            CHK_PRT_RET(
                tmpPaths.size() == 0,
                HCCL_ERROR(
                    "[CollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank,
                    resReqIter->first),
                HcclResult::HCCL_E_INTERNAL);
            // update res
            links.emplace_back(tmpPaths[0]);
        } else {
            CHK_PRT_RET(
                tmpPaths.size() < resReqIter->second,
                HCCL_ERROR("[CollAlgFactory] Rank [%d], available linkNum smaller than required.", myRank),
                HcclResult::HCCL_E_INTERNAL);
            for (u32 linkNum = 0; linkNum < resReqIter->second; linkNum++) {
                // update res
                links.emplace_back(tmpPaths[linkNum]);
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcLinkInfo(
    const RankId myRank, const RankGraph* rankGraph, const LinkReq& linkReq,
    std::vector<std::pair<u32, RankId>>& algTempLinksInfo)
{
    std::set<u32> levelSet = rankGraph->GetLevels(myRank);
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        RankId remoteRank = resReqIter->first;
        if (resReqIter->second == 0) {
            continue;
        }
        if (levelSet.size() == 1) {
            algTempLinksInfo.push_back(std::make_pair(0, remoteRank));
            continue;
        }
        // 当前场景只考虑两层拓扑场景
        u32 levelIdx = 0;
        const NetInstance* netInstance = rankGraph->GetNetInstanceByRankId(levelIdx, myRank);
        std::set<RankId> rankSet = netInstance->GetRankIds();
        auto rankInRankSet = std::find(rankSet.begin(), rankSet.end(), remoteRank);
        if (rankInRankSet != rankSet.end()) {
            algTempLinksInfo.push_back(std::make_pair(0, remoteRank));
        } else {
            algTempLinksInfo.push_back(std::make_pair(1, remoteRank));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SetPathNumMapByRankGraphMultiLevel(
    const RankGraph* rankGraph, std::vector<std::vector<RankId>>& virtRanks_, RankId myRank_,
    std::vector<map<u32, u32>>& rank2PathNumMap)
{
    uint64_t levelNum = 2;
    for (uint64_t levelNumIdx = 0; levelNumIdx < levelNum; levelNumIdx++) {
        rank2PathNumMap.emplace_back();
        for (auto rankIdx : virtRanks_[levelNumIdx]) {
            if (rankIdx == myRank_) {
                continue;
            }
            std::vector<NetInstance::Path> tmpPaths = rankGraph->GetPaths(levelNumIdx, myRank_, rankIdx);
            auto pathNum = 0;
            for (const auto& path : tmpPaths) {
                bool isWithPcie = false;
                for (const auto& link : path.links) {
                    if (*link.GetLinkProtocols().begin() == LinkProtocol::PCIE) {
                        isWithPcie = true;
                        break;
                    }
                }
                if (!isWithPcie) {
                    pathNum++;
                }
            }
            rank2PathNumMap[levelNumIdx][rankIdx] = pathNum;
            HCCL_INFO("[%s]levelNumIdx[%u] rankIdx[%d] pathNum[%d]", __func__, levelNumIdx, rankIdx, pathNum);
        }
    }
    if (rank2PathNumMap.size() == 0) {
        HCCL_ERROR("No path to all remoteRank");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SetPathNumMapByRankGraphMultiLevel(
    const RankGraph* rankGraph, std::vector<RankId>& virtRanks_, RankId myRank_, std::map<u32, u32>& rank2PathNumMap)
{
    std::set<u32> levelSet = rankGraph->GetLevels(myRank_);
    for (auto level : levelSet) {
        bool levelFlag = 1;
        for (auto rankIdx : virtRanks_) {
            if (rankIdx == myRank_) {
                continue;
            }
            std::vector<NetInstance::Path> tmpPaths = rankGraph->GetPaths(level, myRank_, rankIdx);
            if (tmpPaths.size() == 0) {
                rank2PathNumMap.clear();
                levelFlag = 0;
                break;
            }
            auto pathNum = 0;
            for (const auto& path : tmpPaths) {
                bool isWithPcie = false;
                for (const auto& link : path.links) {
                    if (*link.GetLinkProtocols().begin() == LinkProtocol::PCIE) {
                        isWithPcie = true;
                        break;
                    }
                }
                if (!isWithPcie) {
                    pathNum++;
                }
            }
            rank2PathNumMap[rankIdx] = pathNum;
            HCCL_INFO("[%s]rankIdx[%d] pathNum[%d]", __func__, rankIdx, pathNum);
        }
        if (levelFlag) {
            break;
        }
    }
    if (rank2PathNumMap.size() == 0) {
        HCCL_ERROR("No path to all remoteRank");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SetPathNumMapByLinkMgrMultiLevel(
    ConnectedLinkMgr* linkMgr, std::vector<std::vector<RankId>>& virtRanks_, RankId myRank_,
    std::vector<map<u32, u32>>& rank2PathNumMap)
{
    (void)myRank_;
    uint64_t levelNum = 2;
    for (uint64_t levelNumIdx = 0; levelNumIdx < levelNum; levelNumIdx++) {
        rank2PathNumMap.emplace_back();
        for (auto rankIdx : virtRanks_[levelNumIdx]) {
            auto links = linkMgr->GetLinks(levelNumIdx, rankIdx);
            auto linkNum = 0;
            for (const auto& link : links) {
                if (link.GetLinkProtocol() != LinkProtocol::PCIE) {
                    linkNum++;
                }
            }
            if (linkNum != 0) {
                rank2PathNumMap[levelNumIdx][rankIdx] = linkNum;
            }
            HCCL_INFO("[%s]levelNumIdx[%u] rankIdx[%d] linkNum[%d]", __func__, levelNumIdx, rankIdx, linkNum);
        }
    }
    if (rank2PathNumMap.size() == 0) {
        HCCL_ERROR("No path to all remoteRank");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SetPathNumMapByLinkMgrMultiLevel(
    ConnectedLinkMgr* linkMgr, std::vector<RankId>& virtRanks_, RankId myRank_, map<u32, u32>& rank2PathNumMap)
{
    (void)myRank_;
    for (u32 rankIdx : virtRanks_) {
        auto links = linkMgr->GetLinks(rankIdx);
        auto linkNum = 0;
        for (const auto& link : links) {
            if (link.GetLinkProtocol() != LinkProtocol::PCIE) {
                linkNum++;
            }
        }
        if (linkNum != 0) {
            rank2PathNumMap[rankIdx] = linkNum;
        }
        HCCL_INFO("[%s]rankIdx[%u] linkNum[%d]", __func__, rankIdx, linkNum);
    }
    if (rank2PathNumMap.size() == 0) {
        HCCL_ERROR("No path to all remoteRank");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
