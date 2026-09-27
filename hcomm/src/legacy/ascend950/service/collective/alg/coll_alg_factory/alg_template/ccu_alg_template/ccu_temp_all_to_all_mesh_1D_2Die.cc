/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "ccu_temp_all_to_all_mesh_1D_2Die.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_all_to_all_mesh1d_2Die.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_to_all_mesh1d_2Die.h"
#include "ccu_ins_group.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllToAllMesh1D2Die> registrarAllToAll(CcuInstType::CCU_ALLTOALL_MESH_1D_2DIE);

CcuTempAllToAllMesh1D2Die::CcuTempAllToAllMesh1D2Die(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllToAllMesh1D2Die::~CcuTempAllToAllMesh1D2Die() {}

HcclResult CcuTempAllToAllMesh1D2Die::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum + 1;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::SetBuffBlockSize(const u64 buffBlockSize)
{
    CHK_PRT_RET(
        buffBlockSize == 0,
        HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][SetBuffBlockSize] buffBlockSize should not be zero"),
        HcclResult::HCCL_E_PARA);
    buffBlockSize_ = buffBlockSize;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::SetConcurrentSendRecvNum(const u32 concurrentSendRecvNum)
{
    CHK_PRT_RET(
        concurrentSendRecvNum == 0,
        HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][SetConcurrentSendRecvNum] concurrentSendRecvNum should not be zero"),
        HcclResult::HCCL_E_PARA);
    concurrentSendRecvNum_ = concurrentSendRecvNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Run");
    opMode_ = tempFuncs.opMode;
    buffInfo_ = templateDataParams.buffInfo;

    CcuInstructionAllToAllMesh1D2Die ccuInsAllToAllMesh1D2Die;
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAllToAllMesh1D2Die] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);

    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint64_t inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t sliceSize = templateDataParams.sliceSize;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t outBuffBaseOff = buffInfo_.outBuffBaseOff;

    HCCL_INFO(
        "[CcuTempAllToAllMesh1D2Die] myRank_[%d], dimSize[%llu], inputAddr[%llu],"
        "outputAddr[%llu], sliceSize[%llu], outBuffBaseOff[%llu], inputSliceStride[%llu],",
        myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, outBuffBaseOff, inputSliceStride);

    // key表示为dieId
    std::map<uint32_t, std::vector<LinkData>> linksDie;
    std::map<uint32_t, RankGroup> rankGroup;

    for (auto& link : tempLinks) {
        std::vector<LinkData> linkData = link.second;
        RankId peerRankId = link.first;
        if (link.second.empty()) {
            continue;
        }
        linksDie[linkData[0].GetLocalDieId()].push_back(linkData[0]);
        rankGroup[linkData[0].GetLocalDieId()].AddRank(peerRankId);
    }

    HCCL_INFO(
        "[CcuTempAllToAllMesh1D2Die] linksDie0Size[%zu], linksDie1Size[%zu]", linksDie[0].size(), linksDie[1].size());

    rankGroup[0].AddRank(myRank_);
    rankGroup[1].AddRank(myRank_);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t dieId = 0; dieId < 2; dieId++) { // 2Die算法，需要下发 2 条通信指令
        CcuInstructionAllToAllMesh1D2Die ccuInstruction;
        bool withMyRank = linksDie[dieId].size() > linksDie[1 - dieId].size() ? false : true;
        u32 bitNum = min(linksDie[dieId].size(), linksDie[1 - dieId].size()) + 1;
        ccuInstruction.Init(
            static_cast<uint32_t>(myRank_), inputAddr, outputAddr, sliceSize, token, inputSliceStride,
            outputSliceStride, outBuffBaseOff, op_, tempVTopo_, withMyRank, bitNum);
        ccuInstruction.SetLinks(linksDie[dieId]);
        ccuInstruction.SetRankGroup(rankGroup[dieId]);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAllToAllMesh1D2Die>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条que
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
