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

#include "ccu_temp_broadcast_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_broadcast_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_broadcast_mesh1d.h"

namespace Hccl {

static CcuInstRegister<CcuContextBroadcastMesh1D> registrarBroadcast(CcuInstType::CCU_BROADCAST_MESH_1D_DIRECT);

CcuTempBroadcastMesh1D::CcuTempBroadcastMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempBroadcastMesh1D::~CcuTempBroadcastMesh1D() {}

void CcuTempBroadcastMesh1D::GetInAndOutAddr(const TempFuncs& tempFuncs, uint64_t& inputAddr, uint64_t& outputAddr)
{
    if (tempFuncs.isForepart) {
        // 从 UserIn 获取数据
        inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType())
                    + tempFuncs.usrData.usrInSlices[0].GetOffset();
        HCCL_INFO(
            "[CcuTempBroadcastMesh1D] BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType()) is [%llu], "
            "offset is [%llu]",
            BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType()), tempFuncs.usrData.usrInSlices[0].GetOffset());
    } else {
        // 从 inBuff 获取数据
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        HCCL_INFO(
            "[CcuTempBroadcastMesh1D] BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType()) is [%llu], "
            "offset is [%llu]",
            BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType()), tempFuncs.usrData.usrInSlices[0].GetOffset());
    }
    if (tempFuncs.isBottom) {
        // 把数据写入 UserOut
        outputAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType())
                     + tempFuncs.usrData.usrOutSlices[0].GetOffset();
        HCCL_INFO(
            "[CcuTempBroadcastMesh1D] BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType()) is [%llu], "
            "offset is [%llu]",
            BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType()),
            tempFuncs.usrData.usrOutSlices[0].GetOffset());
    } else {
        // 把数据写入 outBuff
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
        HCCL_INFO(
            "[CcuTempBroadcastMesh1D] BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType()) is [%llu], "
            "offset is [%llu]",
            BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType()),
            tempFuncs.usrData.usrOutSlices[0].GetOffset());
    }
    HCCL_INFO("[CcuTempBroadcastMesh1D] inputAddr[%llu], outputAddr[%llu]", inputAddr, outputAddr);
    return;
}

HcclResult CcuTempBroadcastMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastMesh1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)sliceInfoVec;
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempBroadcastMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    buffInfo_ = buffInfo;
    opMode_ = tempFuncs.opMode;
    CcuInstructionBroadcastMesh1D ccuInsBroadcastMesh1D;
    rootId_ = op_.root;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    // 只传userIn的起始位置，不带偏移，偏移已在offset中包含
    uint64_t inputAddr;
    // userOut 的位置，需要带上偏移
    uint64_t outputAddr;
    GetInAndOutAddr(tempFuncs, inputAddr, outputAddr);
    uint64_t sliceSize = tempFuncs.usrData.usrInSlices[0].GetSize(); //

    uint64_t offset = 0;

    uint64_t token;
    CHK_RET(GetToken(op_, token));

    ccuInsBroadcastMesh1D.Init(
        static_cast<uint32_t>(myRank_), static_cast<uint32_t>(rootId_), inputAddr, outputAddr, sliceSize, offset, token,
        op_, tempVTopo_);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempBroadcastMesh1D] links.size[%zu]", links.size());
    ccuInsBroadcastMesh1D.SetLinks(links);
    if (tempVTopo_[0].size() == 0) {
        HCCL_ERROR("[CcuTempBroadcastMesh1D] Invalid tempVTopo[0] size [%zu].", tempVTopo_[0].size());
        return HcclResult::HCCL_E_PARA;
    }
    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 3;
    ccuInsBroadcastMesh1D.SetCntCkeNum(cntCkeNum);
    ccuInsBroadcastMesh1D.SetRankGroup(rankGroup);
    ccuInsBroadcastMesh1D.Describe();
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionBroadcastMesh1D>(ccuInsBroadcastMesh1D)));

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
