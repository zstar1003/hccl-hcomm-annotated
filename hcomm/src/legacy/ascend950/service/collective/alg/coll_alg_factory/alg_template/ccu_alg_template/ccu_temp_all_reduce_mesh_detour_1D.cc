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
#include "executor_utils.h"

#include "ccu_instruction_all_reduce_mesh1d_detour.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_reduce_mesh1d_detour.h"
#include "ccu_temp_all_reduce_mesh_detour_1D.h"

namespace Hccl {

constexpr uint64_t MS_SIZE = 4096;
constexpr u32 DETOUR_RANK_SIZE_2P = 2;
constexpr u32 DETOUR_SPLIT_LINK_NUM = 2;

static CcuInstRegister<CcuContextAllReduceMeshDetour1D>
    g_registrarAllReduce(CcuInstType::CCU_ALL_REDUCE_MESH_1D_DETOUR);

CcuTempAllReduceMeshDetour1D::CcuTempAllReduceMeshDetour1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllReduceMeshDetour1D::~CcuTempAllReduceMeshDetour1D() {}

void CcuTempAllReduceMeshDetour1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempAllReduceMeshDetour1D::CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq)
{
    (void)linkMgr;
    (void)tempResReq;
    HCCL_INFO("[InsCollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuTempAllReduceMeshDetour1D::CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq)
{
    // 当前仅支持2P或4P
    CHK_PRT_RET(
        tempRankSize_ != DETOUR_RANK_SIZE_2P && tempRankSize_ != 4,
        HCCL_INFO("[CcuTempAllReduceMeshDetour1D] Invalid RankSize[%u].", tempRankSize_), HcclResult::HCCL_E_INTERNAL);

    tempResReq.queNum = 1; // 当前只有一个ccu mission，暂定1条流
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CcuTempAllReduceMeshDetour1D][CalcResDetour] tempResReq.queNum[%u]", tempResReq.queNum);
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    for (u32 queIdx = 0; queIdx < tempVTopo_[0].size() - 1; queIdx++) {
        // find neighbors : virtualRank
        RankId neighborRank = tempVTopo_[0][(myAlgRank + 1 + queIdx) % tempRankSize_];
        uint32_t linkNum = GetPathsFromRankGraph(rankGraph, myRank_, neighborRank).size();
        tempResReq.links[neighborRank] = linkNum;
        HCCL_INFO(
            "[CcuTempAllReduceMeshDetour1D][CalcResDetour] RankSize[%u], MyRank[%d]--Neighbor[%d], linkNum[%u]",
            tempRankSize_, myRank_, neighborRank, linkNum);

        // 2P支持2,3,4条link，4P支持2条link，注意绕路link分两条
        CHK_PRT_RET(
            (tempRankSize_ == DETOUR_RANK_SIZE_2P && (linkNum <= 1 || linkNum > 1 + 3 * DETOUR_SPLIT_LINK_NUM))
                || (tempRankSize_ == 4 && linkNum != 1 + 1 * 2), // 4P场景下，1条直连，绕路拆成2条
            HCCL_ERROR(
                "[CcuTempAllReduceMeshDetour1D][CalcResDetour] Invalid linkNum[%u] for RankSize[%u].", linkNum,
                tempRankSize_),
            HcclResult::HCCL_E_INTERNAL);
        if (queIdx == 0) {
            detourPathNum_ = (tempRankSize_ == DETOUR_RANK_SIZE_2P) ? (linkNum - 1) / 2 :
                                                                      1; // 2P时去掉直连有2N条绕路link，对应N个绕路路径
            pathNumPerPeer_ = (tempRankSize_ == DETOUR_RANK_SIZE_2P) ? (detourPathNum_ + 1) :
                                                                       detourPathNum_ + 2; // 4P直连有2条，固定3条
            HCCL_INFO(
                "[CcuTempAllReduceMeshDetour1D][CalcResDetour] detourPathNum[%u], pathNum[%u]", detourPathNum_,
                pathNumPerPeer_);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMeshDetour1D::CalcSliceInfo(
    const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    CHK_RET(CalcSliceInfoAllReduce(allignInfo, tempRankSize_, dataSize, sliceInfoVec)); // ***
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAllReduceMeshDetour1D::CalcDetourOffset(
    uint64_t sliceSize, uint64_t& tailOffset, uint64_t& tailSize, uint64_t& iterNum)
{
    uint64_t loopSize = pathNumPerPeer_ * MS_SIZE * CcuRep::CCU_MS_DEFAULT_LOOP_COUNT; // 整块迭代
    tailSize = sliceSize % loopSize;
    tailOffset = sliceSize - tailSize;
    iterNum = sliceSize / loopSize;

    singleTransportSize_ = 0;
    lengths_.clear(); // 多轮情况下每轮都需要清零
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        lengths_.emplace_back(MS_SIZE);
        singleTransportSize_ += MS_SIZE;
    }
    return;
}

void CcuTempAllReduceMeshDetour1D::ProcessLinks(std::vector<LinkData>& links, const ResLinks& tempLinks) const
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
            "[CcuTempAllReduceMeshDetour1D][ProcessLinks] rankId[%d], linkSize[%zu]", pair.first, pair.second.size());
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
                    "[CcuTempAllReduceMeshDetour1D][ProcessLinks] Rank[%d]--Peer[%d]--link[%d], unexpected link type.",
                    myRank_, pair.first, i));
            }
        }
    }

    // 校验link
    if (sendLinks.size() != recvLinks.size() || directLinks.size() != tempRankSize_ - 1
        || sendLinks.size() % directLinks.size() != 0 || recvLinks.size() % directLinks.size() != 0) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAllReduceMeshDetour1D][ProcessLinks] Unexpected "
            "directLinkSize[%u]--sendLinkSize[%u]--recvLinkSize[%u].",
            directLinks.size(), sendLinks.size(), recvLinks.size()));
    }
    for (uint32_t i = 0; i < directLinks.size(); i++) {
        HCCL_INFO(
            "[CcuTempAllReduceMeshDetour1D][ProcessLinks] directLinks[%u]: peer[%d], linkType[%s]", i,
            directLinks[i].GetRemoteRankId(), directLinks[i].GetDirection().Describe().c_str());
        links.emplace_back(directLinks[i]);
    }
    for (uint32_t i = 0; i < sendLinks.size(); i++) {
        HCCL_INFO(
            "[CcuTempAllReduceMeshDetour1D][ProcessLinks] sendLinks[%u]: peer[%d], linkType[%s]", i,
            sendLinks[i].GetRemoteRankId(), sendLinks[i].GetDirection().Describe().c_str());
        links.emplace_back(sendLinks[i]);
    }
    for (uint32_t i = 0; i < recvLinks.size(); i++) {
        HCCL_INFO(
            "[CcuTempAllReduceMeshDetour1D][ProcessLinks] recvLinks[%u]: peer[%d], linkType[%s]", i,
            recvLinks[i].GetRemoteRankId(), recvLinks[i].GetDirection().Describe().c_str());
        links.emplace_back(recvLinks[i]);
    }

    return;
}

void CcuTempAllReduceMeshDetour1D::GetAddrInfo(const TempFuncs& tempFuncs, uint64_t& inputAddr, uint64_t& outputAddr)
{
    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isBottom) {
            outputAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType())
                         + tempFuncs.usrData.usrOutSlices[0].GetOffset();
        } else {
            outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
        }
        if (tempFuncs.isForepart) {
            inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType())
                        + tempFuncs.usrData.usrInSlices[0].GetOffset();
        } else {
            inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        }
    } else {
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff
                     + tempFuncs.usrData.usrOutSlices[0].GetOffset();
    }
    HCCL_INFO("inputAddr[%llu], outputAddr[%llu]", inputAddr, outputAddr);
    return;
}

HcclResult CcuTempAllReduceMeshDetour1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAllReduceMeshDetour1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;
    CcuInstructionAllReduceMeshDetour1D ccuInsAllReduceMeshDetour1D;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);
    uint64_t inputAddr;
    uint64_t outputAddr;
    GetAddrInfo(tempFuncs, inputAddr, outputAddr);

    uint64_t sliceSize = sliceInfoVec[myRank_][0].size; // 获取本rank需要处理的数据量
    uint64_t offset = sliceInfoVec[myRank_][0].offset;  // 自己需要 reduce 的数据基于 inputAddr 的偏移
    uint64_t token;
    CHK_RET(GetToken(op_, token));

    uint64_t tailOffset;
    uint64_t tailSize;
    uint64_t iterNum;
    CalcDetourOffset(sliceSize, tailOffset, tailSize, iterNum);
    std::vector<LinkData> links;
    ProcessLinks(links, tempLinks);

    ccuInsAllReduceMeshDetour1D.Init(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, offset, token, op_, tempVTopo_, iterNum, tailOffset,
        tailSize, singleTransportSize_, detourPathNum_, pathNumPerPeer_, lengths_);
    HCCL_INFO(
        "[CcuTempAllReduceMeshDetour1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu], outputAddr[%llu],"
        "sliceSize[%llu], offset[%llu], iterNum[%llu], tailOffset[%llu], tailSize[%llu], singleTransportSize_[%u], "
        "detourPathNum_[%u], pathNumPerPeer_[%u]",
        myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, offset, iterNum, tailOffset, tailSize,
        singleTransportSize_, detourPathNum_, pathNumPerPeer_);

    HCCL_INFO("[CcuTempAllReduceMeshDetour1D] links.size[%zu]", links.size());
    ccuInsAllReduceMeshDetour1D.SetLinks(links);

    RankGroup rankGroup;

    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 4;
    ccuInsAllReduceMeshDetour1D.SetCntCkeNum(cntCkeNum);
    ccuInsAllReduceMeshDetour1D.SetRankGroup(rankGroup);
    HCCL_INFO("CCUInsAllReduceMeshDetour1D is [%s]", ccuInsAllReduceMeshDetour1D.Describe().c_str());
    ccuInsAllReduceMeshDetour1D.Describe();
    tempInsQues[0]->Append(
        std::move(std::make_unique<CcuInstructionAllReduceMeshDetour1D>(ccuInsAllReduceMeshDetour1D)));
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
