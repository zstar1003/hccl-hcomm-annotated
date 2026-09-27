/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_to_all_v_mesh_2Die.h"

#include <ios>
#include <iostream>

#include "log.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_ins_group.h"
#include "ccu_context_all_to_all_v_mesh2die.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllToAllVMesh2Die> registerAlltoAllV2Die(CcuInstType::CCU_ALLTOALLV_MESH_2DIE_DIRECT);

CcuTempAlltoAllVMesh2Die::CcuTempAlltoAllVMesh2Die(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    if (tempRankSize_ <= 1 || tempRankSize_ % 2 != 0) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuTempAlltoAllVMesh2Die] Rank[%d], Invalid rankSize[%u].", myRank_, tempRankSize_));
    }

    if (tempVTopo_.size() != 1) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAlltoAllVMesh2Die] Rank[%d], Invalid tempVTopo size[%u].", myRank_, tempVTopo_.size()));
    }
    if (tempVTopo_[0].size() != tempRankSize_) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAlltoAllVMesh2Die] Rank[%d], Invalid tempVTopo[0] size[%u] is not equal to rankSize[%u].", myRank_,
            tempVTopo_[0].size(), tempRankSize_));
    }
    // 填充框内的维度大小
    dimSize_.push_back(tempRankSize_);
}

CcuTempAlltoAllVMesh2Die::~CcuTempAlltoAllVMesh2Die() {}

void CcuTempAlltoAllVMesh2Die::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
    return;
}

HcclResult CcuTempAlltoAllVMesh2Die::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1; // 只申请一个insQue，填充一个insGroup，由框架将其中的ins放在多个stream上
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup

    HCCL_INFO(
        "[CcuTempAlltoAllVMesh2Die] Rank[%d] requiredQueNum[%u] VtopoSize[%u], VtopoSize0[%u].", myRank_,
        tempResReq.queNum, tempVTopo_.size(), tempVTopo_[0].size());
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2Die::FillLinks(const ResLinks& tempLinks)
{
    for (const auto& pair : tempLinks) {
        CHK_PRT_RET(
            pair.second.empty(), // ESL环境上暂只有直连链路
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die] Rank[%d]--peerRank[%d] no link.", myRank_, pair.first),
            HcclResult::HCCL_E_PARA);
        for (const auto& linkData : pair.second) {
            const u32 dieId = linkData.GetLocalDieId();
            if (dieId == 0) {
                HCCL_INFO(
                    "[CcuTempAlltoAllVMesh2Die][Run] Rank[%d] insert link to peerRank[%d] in links(Die0).", myRank_,
                    pair.first);
            } else if (dieId == 1) {
                HCCL_INFO(
                    "[CcuTempAlltoAllVMesh2Die][Run] Rank[%d] insert link to peerRank[%d] in links(Die1).", myRank_,
                    pair.first);
            } else {
                HCCL_ERROR(
                    "[CcuTempAlltoAllVMesh2Die] Rank[%d]--peerRank[%d], Unexpected dieId[%u] in link.", myRank_,
                    pair.first, dieId);
                return HcclResult::HCCL_E_PARA;
            }
            links_[dieId].emplace_back(linkData);
            rankGroup_[dieId].AddRank(pair.first);
        }
    }
    rankGroup_[0].AddRank(myRank_); // keep myRank_ at last, sync with context
    rankGroup_[1].AddRank(myRank_);
    uint32_t minLinks = std::min(links_[0].size(), links_[1].size());
    uint32_t maxLinks = std::max(links_[0].size(), links_[1].size());
    CHK_PRT_RET(
        minLinks + 1 != maxLinks,
        HCCL_ERROR(
            "[CcuTempAlltoAllVMesh2Die] Rank[%d], Unexpected links size, minLinks+1[%u] != maxLinks[%u].", myRank_,
            minLinks + 1, maxLinks),
        HcclResult::HCCL_E_PARA);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2Die::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    (void)sliceInfoVec;
    (void)buffInfo;

    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][Run] tempInsQues is empty."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        op_.inputMem == nullptr || op_.outputMem == nullptr,
        HCCL_ERROR(
            "[CcuTempAlltoAllVMesh2Die][Run] Rank[%d] inputmem[%p] or outputmem[%p] is null", myRank_,
            op_.inputMem.get(), op_.outputMem.get()),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = op_.inputMem == nullptr ? 0 : op_.inputMem->GetAddr();
    uint64_t outputAddr = op_.outputMem == nullptr ? 0 : op_.outputMem->GetAddr();
    uint64_t scratchAddr = op_.scratchMem == nullptr ? 0 : op_.scratchMem->GetAddr();

    HCCL_INFO(
        "[CcuTempAlltoAllVMesh2Die][Run] Rank[%d], input[%#llx], output[%#llx], scratch[%#llx], dataType[%d], "
        "sendType[%d].",
        myRank_, inputAddr, outputAddr, scratchAddr, op_.dataType, op_.all2AllVDataDes.sendType);

    // 分别记录两个Die上的link，构造rankGroup
    CHK_RET(FillLinks(tempLinks));

    uint64_t token;
    CHK_RET(GetToken(op_, token));

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2Die算法，需要执行两次
        CcuInstructionAllToAllVMesh2Die ins = CcuInstructionAllToAllVMesh2Die(op_, dimSize_, tempVTopo_);
        bool withMyRank = links_[axisId].size() < links_[1 - axisId].size();
        ins.Init(myRank_, withMyRank, inputAddr, outputAddr, scratchAddr, token, localSendRecvInfo_);
        ins.SetLinks(links_[axisId]);
        ins.SetRankGroup(rankGroup_[axisId]);
        const u32 ckeNum = 5;
        ins.SetCntCkeNum(ckeNum);
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAllToAllVMesh2Die>(ins)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
