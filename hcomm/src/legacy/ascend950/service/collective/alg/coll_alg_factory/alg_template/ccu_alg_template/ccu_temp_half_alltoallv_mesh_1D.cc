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
#include <algorithm>

#include "log.h"
#include "template_utils.h"

#include "ccu_temp_half_alltoallv_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_half_alltoallv_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_half_alltoallv_mesh1d.h"

namespace Hccl {

static CcuInstRegister<CcuContextHalfAllToAllVMesh1D> registrarAllToAll(CcuInstType::CCU_HALF_ALLTOALLV_MESH_1D);

CcuTempHalfAllToAllVMesh1D::CcuTempHalfAllToAllVMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    if (tempVTopo_.size() != 1 || tempVTopo_[0].size() == 0) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempHalfAllToAllVMesh1D] Invalid tempVTopo "
            "Size[%u] or Invalid tempVTopo[0] size [%u].",
            tempVTopo_.size(), tempVTopo_[0].size()));
    }
}

CcuTempHalfAllToAllVMesh1D::~CcuTempHalfAllToAllVMesh1D() {}

HcclResult CcuTempHalfAllToAllVMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempHalfAllToAllVMesh1D::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo) const
{
    (void)sendRecvInfo;
    return;
}

HcclResult CcuTempHalfAllToAllVMesh1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempHalfAllToAllVMesh1D] Run");
    (void)sliceInfoVec;
    (void)tempFuncs;
    (void)buffInfo;

    if (!op_.scratchMem) {
        HCCL_ERROR("[CcuTempHalfAllToAllVMesh1D][Run] Rank[%d] op_.scratchMem is null", myRank_);
        return HcclResult::HCCL_E_PTR;
    }

    uint64_t scratchAddr = static_cast<uint64_t>(op_.scratchMem->GetAddr());

    // 暂只支持在1D Fullmesh场景下使用两个context，故使用的link相同
    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    HCCL_INFO("[CcuTempHalfAllToAllVMesh1D] links.size[%zu]", links.size());
    int32_t signalNum = (tempRankSize_ + 16 - 1) / 16; // 每个CKE有16个bit

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t mId = 0; mId < 2; mId++) { // 2个mission并发
        HCCL_INFO("[CcuTempHalfAllToAllVMesh1D][Run] MissionId[%u], Rank[%d], SignalNum[%d].", mId, myRank_, signalNum);
        CcuInstructionHalfAllToAllVMesh1D ins = CcuInstructionHalfAllToAllVMesh1D();
        ins.Init(mId, myRank_, scratchAddr, op_, tempVTopo_);
        ins.SetLinks(links);
        ins.SetRankGroup(rankGroup);
        ins.SetCntCkeNum(std::max(3, signalNum)); // 每个transport默认用3个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionHalfAllToAllVMesh1D>(ins)));
    }
    if (tempInsQues.size() == 0) {
        HCCL_ERROR("[CcuTempHalfAllToAllVMesh1D][Run] tempInsQues is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    CHK_PTR_NULL(tempInsQues[0]);
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
