/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_selector.h"

#include <numeric>

namespace Hccl {
BaseSelector& BaseSelector::SetVirtualTopo(RankGraph* rankGraph)
{
    rankGraph_ = rankGraph;
    return *this;
}

BaseSelector& BaseSelector::SetDevType(DevType devType)
{
    devType_ = devType;
    return *this;
}

BaseSelector& BaseSelector::SetMyRank(RankId myRank)
{
    myRank_ = myRank;
    return *this;
}

BaseSelector& BaseSelector::SetRankSize(u32 rankSize)
{
    rankSize_ = rankSize;
    return *this;
}

BaseSelector& BaseSelector::SetSeverId(std::string severId)
{
    severId_ = severId;
    return *this;
}

BaseSelector& BaseSelector::SetDeviceNumPerSever(u32 deviceNumPerSever)
{
    deviceNumPerSever_ = deviceNumPerSever;
    return *this;
}

BaseSelector& BaseSelector::SetServerNum(u32 serverNum)
{
    serverNum_ = serverNum;
    return *this;
}

BaseSelector& BaseSelector::SetIsMc2(bool isMc2)
{
    isMc2_ = isMc2;
    return *this;
}

BaseSelector& BaseSelector::SetOpConfig(OpExecuteConfig opConfig)
{
    opConfig_ = opConfig;
    return *this;
}

RankGraph* BaseSelector::GetVirtualTopo() { return rankGraph_; }

DevType BaseSelector::GetDevType() { return devType_; }

RankId BaseSelector::GetMyRank() const { return myRank_; }

u32 BaseSelector::GetRankSize() const { return rankSize_; }

std::string BaseSelector::GetSeverId() { return severId_; }

u32 BaseSelector::GetDeviceNumPerSever() const { return deviceNumPerSever_; }

u32 BaseSelector::GetServerNum() const { return serverNum_; }

u32 BaseSelector::Gcd(u32 a, u32 b) const
{
    while (b != 0) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

u32 BaseSelector::GcdOfArray(const std::vector<u32>& numbers) const
{
    if (numbers.empty()) {
        return 0;
    }
    u32 result = numbers[0];
    for (size_t i = 1; i < numbers.size(); ++i) {
        result = Gcd(result, numbers[i]); // C++17 及以上推荐使用 std::gcd
    }
    return result;
}

u32 BaseSelector::GetLevel0Gcd() const
{
    std::vector<u32> instSizeList = {};
    u32 listSize = 0;
    rankGraph_->GetNetInstanceList(0, instSizeList, listSize);
    return GcdOfArray(instSizeList);
}

bool BaseSelector::IsAsymmetricTopoShapeLevel1Nhr(
    const std::vector<std::vector<u32>>& localIdPerBoard, u32 gcdRankSizeLevel0) const
{
    // Level0的gcd为1分支
    if (gcdRankSizeLevel0 == 1) {
        return true;
    }
    // Pod形状不规则分支
    if (localIdPerBoard.size() > 1) {
        if (!IsTopoShapeLevel0Regular(localIdPerBoard)) {
            return true;
        }
    }
    return false;
}

bool BaseSelector::IsTopoShapeLevel0Regular(const std::vector<std::vector<u32>>& localIdPerBoard) const
{
    u32 rankSizeOfFirstBoard = localIdPerBoard[0].size();
    u32 rankSize = 8;
    for (u32 boardIdx = 1; boardIdx < localIdPerBoard.size(); ++boardIdx) {
        // 条件1：与第一行rank数是否一致
        if (localIdPerBoard[boardIdx].size() != rankSizeOfFirstBoard) {
            return false;
        }
        // 条件2：同一slot内rank数差异是否能被8整除
        for (u32 slotIdx = 0; slotIdx < rankSizeOfFirstBoard; ++slotIdx) {
            if ((localIdPerBoard[boardIdx][slotIdx] - localIdPerBoard[0][slotIdx]) % rankSize != 0) {
                return false;
            }
        }
    }
    return true;
}

HcclResult BaseSelector::ExtractNetLayerDetails(TopoInfo& topoInfo) const
{
    CHK_PRT_RET(
        rankGraph_ == nullptr, HCCL_ERROR("[BaseSelector][ExtractNetLayerDetails] rankGraph_ is null"), HCCL_E_PTR);

    auto& topoLevelNum = topoInfo.levelNum;
    auto& netLayerNum = topoInfo.netLayerDetails.netLayerNum;
    auto& netLayers = topoInfo.netLayerDetails.netLayers;
    auto& netInstNumOfLayer = topoInfo.netLayerDetails.netInstNumOfLayer;
    auto& instSizeListOfLayer = topoInfo.netLayerDetails.instSizeListOfLayer;
    auto& localNetInsSizeOfLayer = topoInfo.netLayerDetails.localNetInsSizeOfLayer;

    netLayers = rankGraph_->GetLevels(myRank_); // 有那几层网络 如：[0,1]
    netLayerNum = rankGraph_->GetLevelNum();
    netInstNumOfLayer.resize(netLayerNum);   // 每层网络中有几个网络实例
    instSizeListOfLayer.resize(netLayerNum); // 每层网络中的各个网络实例的大小
    localNetInsSizeOfLayer.resize(netLayerNum);

    HcclResult ret;
    // 获取并校验每一层的网路实例大小
    for (auto layerIdx : netLayers) {
        std::vector<u32>& currLayerInstSizeList = instSizeListOfLayer[layerIdx];
        u32& currLayerNetInstNum = netInstNumOfLayer[layerIdx];
        ret = rankGraph_->GetNetInstanceList(layerIdx, currLayerInstSizeList, currLayerNetInstNum);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[BaseSelector][ExtractNetLayerDetails] GetNetInstanceList failed, netLayer[%u]", layerIdx),
            ret);
        for (u32 i = 0; i < currLayerInstSizeList.size(); i++) {
            HCCL_DEBUG(
                "[BaseSelector][ExtractNetLayerDetails] netInstanceSize[%u] is [%u]", i, currLayerInstSizeList[i]);
        }
        u32 currLayerRankSize = std::accumulate(currLayerInstSizeList.begin(), currLayerInstSizeList.end(), 0);
        HCCL_INFO("[BaseSelector][ExtractNetLayerDetails] Net layer[%u] instNum[%u]", layerIdx, currLayerNetInstNum);
        CHK_PRT_RET(
            currLayerRankSize != rankSize_,
            HCCL_ERROR(
                "[BaseSelector][ExtractNetLayerDetails] NetLayer[%u], totalRankSize[%u] is not equal to comm "
                "rankSize[%u]",
                layerIdx, currLayerRankSize, rankSize_),
            HCCL_E_PARA);
        localNetInsSizeOfLayer[layerIdx] = rankGraph_->GetLocalInstSize(layerIdx);
    }

    topoLevelNum = 0;
    // 获取最小的能覆盖所有卡的 layer
    for (auto layerIdx : netLayers) {
        if (netInstNumOfLayer[layerIdx] == 1) {
            // 当本层只有一个网络实例时, 认为这个就是当前的 topoLevelNum
            topoLevelNum = layerIdx + 1;
            break;
        }
    }

    HCCL_INFO(
        "[BaseSelector][ExtractNetLayerDetails] topoLevelNum[%u], netLayerNum[%u], netLayers.size[%u]", topoLevelNum,
        netLayerNum, netLayers.size());

    CHK_PRT_RET(
        topoLevelNum == 0,
        HCCL_ERROR(
            "[BaseSelector][ExtractNetLayerDetails] topoLevelNum[%u] is invalid, netLayerNum[%u]", topoLevelNum,
            netLayerNum),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult BaseSelector::ExtractTopoDetails(TopoInfo& topoInfo) const
{
    HcclResult ret;
    CHK_PRT_RET(rankGraph_ == nullptr, HCCL_ERROR("[BaseSelector][ExtractTopoDetails] rankGraph_ is null"), HCCL_E_PTR);
    u32 netLayerNum = topoInfo.netLayerDetails.netLayerNum;

    // 初始化每一层的 TopoInstDetails
    topoInfo.topoInstDetailsOfLayer.resize(netLayerNum);
    for (u32 netLayerIdx = 0; netLayerIdx < netLayerNum; netLayerIdx++) {
        auto& currentNetLayerTopoTopoDetail = topoInfo.topoInstDetailsOfLayer[netLayerIdx];
        auto& currentLayerTopoSize = currentNetLayerTopoTopoDetail.sizeOfTopo;
        auto& currentLayerTopoType = currentNetLayerTopoTopoDetail.typeOfTopo;
        auto& currentLayerTopoRanks = currentNetLayerTopoTopoDetail.ranksInTopo;
        auto& currentLayerTopo2SizeMap = currentNetLayerTopoTopoDetail.rankNumForTopoType;
        auto& topoInstNum = currentNetLayerTopoTopoDetail.topoInstNum;

        std::vector<u32> topoInsts;
        rankGraph_->GetTopoInstsByLayer(netLayerIdx, topoInsts, topoInstNum);
        HCCL_INFO("[BaseSelector][ExtractTopoDetails] netLayerIdx[%u], topoInstNum[%u]", netLayerIdx, topoInstNum);
        // 初始化当前层的拓扑信息
        currentLayerTopoSize.resize(topoInstNum);
        currentLayerTopoType.resize(topoInstNum);
        currentLayerTopoRanks.resize(topoInstNum);
        currentLayerTopo2SizeMap.clear();

        // 填充当前层的拓扑信息
        for (u32 topoInstIdx = 0; topoInstIdx < topoInstNum; topoInstIdx++) {
            u32& topoInstId = topoInsts[topoInstIdx];
            u32& topoSize = currentLayerTopoSize[topoInstIdx];
            TopoType& topoType = currentLayerTopoType[topoInstIdx];
            std::vector<u32>& ranks = currentLayerTopoRanks[topoInstIdx];

            // 获取拓扑实例的类型
            ret = rankGraph_->GetTopoType(netLayerIdx, topoInstId, topoType);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[BaseSelector][ExtractTopoDetails] GetTopoType failed, netLayerIdx[%u], topoInstId[%u]",
                    netLayerIdx, topoInstId),
                ret);

            // 获取拓扑实例中包含的rank
            ret = rankGraph_->GetRanksByTopoInst(netLayerIdx, topoInstId, ranks, topoSize);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[BaseSelector][ExtractTopoDetails] GetRanksByTopoInst failed, netLayerIdx[%u], topoInstId[%u]",
                    netLayerIdx, topoInstId),
                ret);

            // 将topoInstId按照topoType进行归类
            currentLayerTopo2SizeMap[topoType].push_back(topoSize);

            HCCL_INFO(
                "[BaseSelector][ExtractTopoDetails] netLayerIdx[%u], topoInstIdx[%u] type is[%s], topoInstId is[%u], "
                "topoSize is[%u]",
                netLayerIdx, topoInstIdx, topoType.Describe().c_str(), topoInstId, topoSize);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BaseSelector::CalcLevel0TopoShape(TopoInfo& topoInfo) const
{
    u32 netLayer = 0;
    u32 topoInstNum2 = 2;
    u32 topoInstNum3 = 3;
    CHK_PRT_RET(
        topoInfo.topoInstDetailsOfLayer.size() <= netLayer,
        HCCL_ERROR(
            "[BaseSelector][CalcLevel0TopoShape] topoInstNumOfLayer size[%zu] <= netLayer[%u]",
            topoInfo.topoInstDetailsOfLayer.size(), netLayer),
        HCCL_E_INTERNAL);
    TopoInstDetails& level0TopoInstDetails = topoInfo.topoInstDetailsOfLayer[netLayer];
    CHK_PRT_RET(
        topoInfo.netLayerDetails.localNetInsSizeOfLayer.size() <= netLayer,
        HCCL_ERROR(
            "[BaseSelector][CalcLevel0TopoShape] localNetInsSizeOfLayer size[%zu] <= netLayer[%u]",
            topoInfo.netLayerDetails.localNetInsSizeOfLayer.size(), netLayer),
        HCCL_E_INTERNAL);
    u32 level0LocalRankSize = topoInfo.netLayerDetails.localNetInsSizeOfLayer[netLayer];

    auto& topoInstNum = level0TopoInstDetails.topoInstNum;
    auto& rankNumForTopoType = level0TopoInstDetails.rankNumForTopoType;
    HCCL_INFO("[%s]topoInstNum[%u]", __func__, topoInstNum);
    for (const auto& iter : rankNumForTopoType) {
        HCCL_INFO("[%s]topoType[%d] size[%lu]", __func__, iter.first, iter.second.size());
    }

    if (topoInstNum == 1 && rankNumForTopoType[TopoType::MESH_1D].size() == 1) {
        // MESH_1D 拓扑校验
        CHK_PRT_RET(
            rankNumForTopoType[TopoType::MESH_1D][0] != level0LocalRankSize,
            HCCL_ERROR(
                "[BaseSelector][CalcLevel0TopoShape] MESH_1D rankSize[%u] is not equal to level0LocalRankSize[%u]",
                rankNumForTopoType[TopoType::MESH_1D][0], level0LocalRankSize),
            HCCL_E_INTERNAL);
        topoInfo.level0Shape = Level0Shape::MESH_1D;
        return HCCL_SUCCESS;
    } else if (topoInstNum == 1 && rankNumForTopoType[TopoType::CLOS].size() == 1) {
        // CLOS 拓扑校验
        CHK_PRT_RET(
            rankNumForTopoType[TopoType::CLOS][0] != level0LocalRankSize,
            HCCL_ERROR(
                "[BaseSelector][CalcLevel0TopoShape] CLOS rankSize[%u] is not equal to level0LocalRankSize[%u]",
                rankNumForTopoType[TopoType::CLOS][0], level0LocalRankSize),
            HCCL_E_INTERNAL);
        topoInfo.level0Shape = Level0Shape::CLOS;
        if (IsLevel0PcieMix()) {
            topoInfo.level0PcieMix = true;
        }
        return HCCL_SUCCESS;
    } else if (
        topoInstNum == topoInstNum2 && rankNumForTopoType[TopoType::CLOS].size() == 1
        && rankNumForTopoType[TopoType::MESH_1D].size() == 1) {
        // MESH_1D_CLOS 拓扑校验
        CHK_PRT_RET(
            rankNumForTopoType[TopoType::CLOS][0] != level0LocalRankSize,
            HCCL_ERROR(
                "[BaseSelector][CalcLevel0TopoShape] CLOS rankSize[%u] is not equal to level0LocalRankSize[%u]",
                rankNumForTopoType[TopoType::CLOS][0], level0LocalRankSize),
            HCCL_E_INTERNAL);
        topoInfo.level0Shape = Level0Shape::MESH_1D_CLOS;

        if (IsLevel0PcieMix()) {
            topoInfo.level0PcieMix = true;
        }
        if (rankNumForTopoType[TopoType::CLOS].at(0) > BIG_CLOS_RANGE) {
            topoInfo.level0BigClosRange = true;
        }
        return HCCL_SUCCESS;
    } else if (
        topoInstNum == topoInstNum3 && rankNumForTopoType[TopoType::MESH_1D].size() == topoInstNum2
        && rankNumForTopoType[TopoType::CLOS].size() == 1) {
        // MESH_2D 拓扑校验
        CHK_PRT_RET(
            rankNumForTopoType[TopoType::MESH_1D][0] * rankNumForTopoType[TopoType::MESH_1D][1] != level0LocalRankSize,
            HCCL_ERROR(
                "[BaseSelector][CalcLevel0TopoShape] mesh rankSize[%u] * [%u] is not equal to level0LocalRankSize[%u]",
                rankNumForTopoType[TopoType::MESH_1D][0], rankNumForTopoType[TopoType::MESH_1D][1],
                level0LocalRankSize),
            HCCL_E_INTERNAL);
        topoInfo.level0Shape = Level0Shape::MESH_2D;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("Unknown topo for level 0, topoInstNum[%u]", topoInstNum);
    return HCCL_E_INTERNAL;
}

void BaseSelector::CalcTopoShape(TopoInfo& topoInfo) const
{
    CHK_PRT_THROW(
        ExtractNetLayerDetails(topoInfo) != HCCL_SUCCESS,
        HCCL_ERROR("[BaseSelector][CalcTopoShape] ExtractNetLayerDetails Failed"), InvalidParamsException,
        "ExtractNetLayerDetails Failed");
    HCCL_INFO("[BaseSelector][CalcTopoShape] topoInfo.levelNum is [%u]", topoInfo.levelNum);

    CHK_PRT_THROW(
        ExtractTopoDetails(topoInfo) != HCCL_SUCCESS,
        HCCL_ERROR("[BaseSelector][CalcTopoShape] ExtractTopoDetails Failed"), InvalidParamsException,
        "ExtractTopoDetails Failed");
    HCCL_INFO("[BaseSelector][ExtractTopoDetails] topoInstDetails size[%zu]", topoInfo.topoInstDetailsOfLayer.size());

    CHK_PRT_THROW(
        CalcLevel0TopoShape(topoInfo), HCCL_ERROR("[BaseSelector][CalcTopoShape] CalcLevel0TopoShape Failed"),
        InvalidParamsException, "CalcLevel0TopoShape Failed");
    HCCL_INFO(
        "[BaseSelector][CalcTopoShape] topoInfo.level0Shape is [%d], level0PcieMix is [%d], level0BigClosRange is [%d]",
        topoInfo.level0Shape, topoInfo.level0PcieMix, topoInfo.level0BigClosRange);
}

bool BaseSelector::IsLayerAllConnetedWithTopo(
    const TopoInfo& topoInfo, const u32 netLayer, const TopoType topoType) const
{
    CHK_PRT_THROW(
        rankGraph_ == nullptr, HCCL_ERROR("[BaseSelector][IsLayerAllConnetedWithTopo] rankGraph is nullptr"),
        NullPtrException, "[IsLayerAllConnetedWithTopo] rankGraph is nullptr");

    CHK_PRT_RET(
        topoInfo.netLayerDetails.localNetInsSizeOfLayer.size() <= netLayer,
        HCCL_WARNING(
            "[BaseSelector][IsLayerAllConnetedWithTopo] localNetInsSizeOfLayer size[%zu] <= netLayer[%u]",
            topoInfo.netLayerDetails.localNetInsSizeOfLayer.size(), netLayer),
        false);
    u32 localRankSize = topoInfo.netLayerDetails.localNetInsSizeOfLayer[netLayer];

    CHK_PRT_RET(
        topoInfo.topoInstDetailsOfLayer.size() <= netLayer,
        HCCL_WARNING(
            "[BaseSelector][IsLayerAllConnetedWithTopo] topoInstDetailsOfLayer size[%zu] <= netLayer[%u]",
            topoInfo.topoInstDetailsOfLayer.size(), netLayer),
        false);

    auto rankNumForTopoTypeItr = topoInfo.topoInstDetailsOfLayer[netLayer].rankNumForTopoType.find(topoType);
    if (rankNumForTopoTypeItr == topoInfo.topoInstDetailsOfLayer[netLayer].rankNumForTopoType.end()) {
        return false;
    }

    for (auto topoRankNum : rankNumForTopoTypeItr->second) {
        if (topoRankNum == localRankSize) {
            return true;
        }
    }
    return false;
}

bool BaseSelector::IsInputOutputOverlap(
    const std::shared_ptr<Buffer>& inputMem, const std::shared_ptr<Buffer>& outputMem) const
{
    CHK_PRT_RET(
        inputMem == nullptr || outputMem == nullptr,
        HCCL_INFO("[Algo][BaseSelector][IsInputOutputOverlap] The input or output buffer is null. Not overlap."),
        false);

    u64 inputStart = inputMem->GetAddr();
    u64 outputStart = outputMem->GetAddr();

    CHK_PRT_RET(
        inputStart == 0 || outputStart == 0,
        HCCL_INFO("[Algo][BaseSelector][IsInputOutputOverlap] The input or output buffer addr is null. Not overlap."),
        false);

    u64 inputDataSize = inputMem->GetSize();
    u64 outputDataSize = outputMem->GetSize();

    CHK_PRT_RET(
        inputDataSize == 0 || outputDataSize == 0,
        // 不存在overlap情况
        HCCL_INFO("[Algo][BaseSelector][IsInputOutputOverlap] The input or output buffer size is 0. Not overlap."),
        false);

    u64 inputEnd = inputStart + inputDataSize - 1;
    u64 outputEnd = outputStart + outputDataSize - 1;

    HCCL_DEBUG(
        "[Algo][BaseSelector][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
        "outputEnd[%llu].",
        inputStart, inputEnd, outputStart, outputEnd);

    CHK_PRT_RET(
        inputStart <= outputEnd && outputStart <= inputEnd,
        HCCL_INFO(
            "[Algo][BaseSelector][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
            "outputEnd[%llu]. Overlap detected.",
            inputStart, inputEnd, outputStart, outputEnd),
        true);

    HCCL_DEBUG("[Algo][BaseSelector][IsInputOutputOverlap]No overlap between input and output memory.");
    return false;
}

bool BaseSelector::Is2DieFullMesh() const
{
    u32 netLayer = 0; // 0 级拓扑
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(netLayer, myRank_);
    std::set<RankId> rankSet = netInstance->GetRankIds();
    if (rankSet.size() <= 2) { // 小于2张卡的话，肯定不是2die全互连
        return false;
    }
    // 遍历所有对端，校验是否和所有卡有全连链路，并判断链路中本端端口所所对应的 CCU die 是否一致;
    u32 dieNum = 2; // 一共2个die
    std::vector<u32> dieLinkCounter(dieNum, 0);
    for (RankId rankId : rankSet) {
        if (rankId == myRank_) {
            continue;
        }
        std::vector<NetInstance::Path> paths = rankGraph_->GetPaths(netLayer, myRank_, rankId);
        CHK_PRT_RET(
            paths.size() == 0 || paths[0].links.size() == 0,
            HCCL_INFO(
                "[BaseSelector][Is2DieFullMesh], Can not find path from Local[%d] to Rmt[%d], in netLayer %u. "
                "Topo is not mesh",
                myRank_, rankId, netLayer),
            false);
        NetInstance::Link& link = paths[0].links[0]; // 只取第一条路径的第一条link
        std::shared_ptr<NetInstance::ConnInterface> connInterface = link.GetSourceIface();
        u32 dieID = connInterface->GetLocalDieId();
        CHK_PRT_RET(
            dieID >= dieNum,
            HCCL_WARNING(
                "[BaseSelector][Is2DieFullMesh], Link from Local[%d] to Rmt[%d] die id[%u] is out of range[%u].",
                myRank_, rankId, dieID, dieNum),
            false);
        dieLinkCounter[dieID]++;
        HCCL_INFO(
            "[BaseSelector][Is2DieFullMesh], Link from Local[%d] to Rmt[%d] use die[%u], current counter[%u]", myRank_,
            rankId, dieID, dieLinkCounter[dieID]);
    }
    for (u32 i = 0; i < dieNum; i++) {
        if (dieLinkCounter[i] == 0) {
            return false;
        }
    }
    return true;
}

bool BaseSelector::IsLevel0PcieMix() const
{
    u32 netLayer = 0; // 0 级拓扑
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(netLayer, myRank_);
    std::set<RankId> rankSet = netInstance->GetRankIds();
    for (RankId rankId : rankSet) {
        if (rankId == myRank_) {
            continue;
        }
        std::vector<NetInstance::Path> paths = rankGraph_->GetPaths(netLayer, myRank_, rankId);
        CHK_PRT_RET(
            paths.size() == 0 || paths[0].links.size() == 0,
            HCCL_INFO(
                "[BaseSelector][IsLevel0PcieMix], Can not find path from Local[%d] to Rmt[%d], in netLayer %u. "
                "Topo is not mesh",
                myRank_, rankId, netLayer),
            false);
        NetInstance::Link& link = paths[0].links[0]; // 只取第一条路径的第一条link
        std::shared_ptr<NetInstance::ConnInterface> connInterface = link.GetSourceIface();
        std::set<LinkProtocol> linkProtocolsSet = connInterface->GetLinkProtocols();
        // 判断协议类型包含PCIE
        if (linkProtocolsSet.find(LinkProtocol::PCIE) != linkProtocolsSet.end()) {
            HCCL_INFO("IsLevel0PcieMix[true]");
            return true;
        }
    }
    HCCL_INFO("IsLevel0PcieMix[false]");
    return false;
}
} // namespace Hccl
