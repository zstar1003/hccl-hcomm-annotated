/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ios>
#include <iostream>

#include "log.h"

#include "ccu_instruction_all_gather_mesh1d_detour.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "executor_utils.h"
#include "ccu_context_all_gather_mesh1d_detour.h"
#include "ccu_temp_all_gather_mesh_1D_detour.h"

namespace Hccl {
constexpr u32 DETOUR_RANK_SIZE_2P = 2;
constexpr u32 DETOUR_SPLIT_LINK_NUM = 2;

static CcuInstRegister<CcuContextAllGatherMeshDetour1D> g_registrarAllGather(CcuInstType::CCU_ALLGATHER_MESH_1D_DETOUR);

CcuTempAllGatherMeshDetour1D::CcuTempAllGatherMeshDetour1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllGatherMeshDetour1D::~CcuTempAllGatherMeshDetour1D() {}

HcclResult CcuTempAllGatherMeshDetour1D::CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq)
{
    // 当前仅支持2P或4P
    CHK_PRT_RET(
        tempRankSize_ != DETOUR_RANK_SIZE_2P && tempRankSize_ != 4,
        HCCL_ERROR("[CcuTempAllGatherMeshDetour1D] Invalid RankSize[%u].", tempRankSize_), HcclResult::HCCL_E_INTERNAL);

    tempResReq.queNum = 1; // 当前只有一个ccu mission，暂定1条流
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CcuTempAllGatherMeshDetour1D][CalcResDetour] tempResReq.queNum[%u]", tempResReq.queNum);
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    for (u32 queIdx = 0; queIdx < tempVTopo_[0].size() - 1; queIdx++) {
        // find neighbors : virtualRank
        RankId neighborRank = tempVTopo_[0][(myAlgRank + 1 + queIdx) % tempRankSize_];
        uint32_t linkNum = GetPathsFromRankGraph(rankGraph, myRank_, neighborRank).size();
        tempResReq.links[neighborRank] = linkNum;
        HCCL_INFO(
            "[CcuTempAllGatherMeshDetour1D][CalcResDetour] RankSize[%u], MyRank[%d]--Neighbor[%d], linkNum[%u]",
            tempRankSize_, myRank_, neighborRank, linkNum);

        // 2P支持2,3,4条link，4P支持2条link，注意绕路link分两条
        CHK_PRT_RET(
            (tempRankSize_ == DETOUR_RANK_SIZE_2P && (linkNum <= 1 || linkNum > 1 + 3 * DETOUR_SPLIT_LINK_NUM))
                || (tempRankSize_ == 4 && linkNum != 1 + 1 * 2), // 4P场景下，1条直连，绕路拆成2条
            HCCL_ERROR(
                "[CcuTempAllGatherMeshDetour1D][CalcResDetour] Invalid linkNum[%u] for RankSize[%u].", linkNum,
                tempRankSize_),
            HcclResult::HCCL_E_INTERNAL);
        if (queIdx == 0) {
            detourPathNum_ = (tempRankSize_ == DETOUR_RANK_SIZE_2P) ? (linkNum - 1) / 2 :
                                                                      1; // 2P时去掉直连有2N条绕路link，对应N个绕路路径
            pathNumPerPeer_ = (tempRankSize_ == DETOUR_RANK_SIZE_2P) ? (detourPathNum_ + 1) :
                                                                       detourPathNum_ + 2; // 4P直连有2条，固定3条
            HCCL_INFO(
                "[CcuTempAllGatherMeshDetour1D][CalcResDetour] detourPathNum[%u], pathNum[%u]", detourPathNum_,
                pathNumPerPeer_);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMeshDetour1D::CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq)
{
    (void)linkMgr;
    (void)tempResReq;
    HCCL_ERROR("[InsCollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuTempAllGatherMeshDetour1D::CalcSliceInfo(
    const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAllGatherMeshDetour1D::CalcDetourOffset(
    uint64_t sliceSize, uint64_t& tailOffset, uint64_t& tailSize, uint64_t& loopIterNum)
{
    constexpr uint64_t MS_SIZE = 4096;
    uint64_t loopSize = pathNumPerPeer_ * MS_SIZE * CcuRep::CCU_MS_DEFAULT_LOOP_COUNT; // 整块迭代
    loopIterNum = sliceSize / loopSize;
    tailSize = sliceSize % loopSize;
    tailOffset = sliceSize - tailSize;

    singleTransportSize_ = 0;
    lengths_.clear(); // 多轮情况下每轮都需要清零
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        lengths_.emplace_back(MS_SIZE);
        singleTransportSize_ += MS_SIZE;
    }
    return;
}

void CcuTempAllGatherMeshDetour1D::ProcessLinks(std::vector<LinkData>& links, const ResLinks& tempLinks) const
{
    // 整理links，要区分sendOnly与recvOnly，根据读写操作选择不同的绕路link
    // 固定2P用2-4条链路，每个链路用一个ms；4P用2条链路，其中直连用2个ms，绕路用1个
    std::vector<LinkData> directLinks;
    std::vector<LinkData> sendLinks; // sendOnly
    std::vector<LinkData> recvLinks; // recvOnly
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        HCCL_INFO(
            "[CcuTempAllGatherMeshDetour1D][ProcessLinks] rankId[%d], linkSize[%zu]", pair.first, pair.second.size());
        for (uint32_t i = 0; i < pair.second.size(); i++) {
            LinkData curLink = pair.second[i];
            if (curLink.GetHop() == 1) {
                directLinks.emplace_back(curLink);
            } else if (curLink.GetDirection() == LinkDirection::SEND_ONLY) {
                sendLinks.emplace_back(curLink);
            } else if (curLink.GetDirection() == LinkDirection::RECV_ONLY) {
                recvLinks.emplace_back(curLink);
            } else {
                THROW<InvalidParamsException>(StringFormat(
                    "[CcuTempAllGatherMeshDetour1D][ProcessLinks] Rank[%d]--Peer[%d]--link[%d], unexpected link type.",
                    myRank_, pair.first, i));
            }
        }
    }

    // 校验link
    if (sendLinks.size() != recvLinks.size() || directLinks.size() != tempRankSize_ - 1
        || sendLinks.size() % directLinks.size() != 0 || recvLinks.size() % directLinks.size() != 0) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAllGatherMeshDetour1D] directSize[%zu]-sendSize[%zu]-recvSize[%zu].", directLinks.size(),
            sendLinks.size(), recvLinks.size()));
    }
    for (uint32_t i = 0; i < directLinks.size(); i++) {
        HCCL_INFO("Peer[%d][%s]", directLinks[i].GetRemoteRankId(), directLinks[i].GetDirection().Describe().c_str());
        links.emplace_back(directLinks[i]);
    }
    for (uint32_t i = 0; i < sendLinks.size(); i++) {
        HCCL_INFO("Peer[%d][%s]", sendLinks[i].GetRemoteRankId(), sendLinks[i].GetDirection().Describe().c_str());
        links.emplace_back(sendLinks[i]);
    }
    for (uint32_t i = 0; i < recvLinks.size(); i++) {
        HCCL_INFO("Peer[%d][%s]", recvLinks[i].GetRemoteRankId(), recvLinks[i].GetDirection().Describe().c_str());
        links.emplace_back(recvLinks[i]);
    }

    return;
}

void CcuTempAllGatherMeshDetour1D::GetAddrInfo(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, uint64_t& inputAddr, uint64_t& outputAddr,
    uint64_t& offset)
{
    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isForepart) {
            inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType())
                        + tempFuncs.usrData.usrInSlices[0].GetOffset();
        } else {
            inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        }
        if (tempFuncs.isBottom) {
            outputAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType());
            // 需要加上 UserOUt 的偏移，包含了 loop 偏移和 rank 偏移
            offset = tempFuncs.usrData.usrOutSlices[myRank_].GetOffset();
        } else {
            outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
            // 从 inBuff 获取数据，只需要加上 rank 偏移
            offset = sliceInfoVec[myRank_][0].offset;
        }
    } else {
        // 图模式没有 tempFuncs.usrData，直接通过 buffInfo_ 获取输入输出地址
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff
                    + tempFuncs.usrData.usrInSlices[0].GetOffset();
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
        offset = tempFuncs.usrData.usrOutSlices[myRank_].GetOffset();
    }

    return;
}

HcclResult CcuTempAllGatherMeshDetour1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAllGatherMeshDetour1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;
    CcuInstructionAllGatherMeshDetour1D ccuInsAllGatherMeshDetour1D;

    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t offset;
    GetAddrInfo(tempFuncs, sliceInfoVec, inputAddr, outputAddr, offset);

    std::vector<LinkData> links;
    uint64_t sliceSize = sliceInfoVec[myRank_][0].size; // 获取本rank需要处理的数据量
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t tailOffset;
    uint64_t tailSize;
    uint64_t loopIterNum;
    CalcDetourOffset(sliceSize, tailOffset, tailSize, loopIterNum);
    ProcessLinks(links, tempLinks);

    ccuInsAllGatherMeshDetour1D.InitDetourInfo(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, token, offset, tailOffset, tailSize, loopIterNum,
        lengths_, singleTransportSize_, detourPathNum_, pathNumPerPeer_, op_, tempVTopo_);

    HCCL_INFO(
        "[CcuTempAllGatherMeshDetour1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],outputAddr[%llu],"
        "sliceSize[%llu], baseOffset[%llu], tailOffset[%llu], tailSize[%llu], loopIterNum[%llu],"
        "singleTransportSize[%llu], detourPathNum[%u], pathNumPerPeer[%u], links.size[%llu]",
        myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, offset, tailOffset, tailSize, loopIterNum,
        singleTransportSize_, detourPathNum_, pathNumPerPeer_, links.size());

    ccuInsAllGatherMeshDetour1D.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 3;
    ccuInsAllGatherMeshDetour1D.SetCntCkeNum(cntCkeNum);
    ccuInsAllGatherMeshDetour1D.SetRankGroup(rankGroup);
    HCCL_INFO("ccuInsAllGatherMeshDetour1D is [%s]", ccuInsAllGatherMeshDetour1D.Describe().c_str());
    ccuInsAllGatherMeshDetour1D.Describe();
    tempInsQues[0]->Append(
        std::move(std::make_unique<CcuInstructionAllGatherMeshDetour1D>(ccuInsAllGatherMeshDetour1D)));

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
