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

#include "ccu_instruction_all_gather_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_gather_mesh1d.h"
#include "ccu_temp_all_gather_mesh_1D.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllGatherMesh1D> g_registrarAllGather(CcuInstType::CCU_ALLGATHER_MESH_1D_DIRECT);

CcuTempAllGatherMesh1D::CcuTempAllGatherMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllGatherMesh1D::~CcuTempAllGatherMesh1D() {}

HcclResult CcuTempAllGatherMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuTempAllGatherMesh1D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempAllGatherMesh1D::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

HcclResult CcuTempAllGatherMesh1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempAllGatherMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;
    CcuInstructionAllGatherMesh1D ccuInsAllGatherMesh1D;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t offset;
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
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff
                    + tempFuncs.usrData.usrInSlices[0].GetOffset();
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
        offset = tempFuncs.usrData.usrOutSlices[myRank_].GetOffset();
    }

    uint64_t sliceSize = sliceInfoVec[myRank_][0].size; // 获取本rank需要处理的数据量
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    ccuInsAllGatherMesh1D.Init(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, sliceSize, offset, token, op_, tempVTopo_);
    HCCL_INFO(
        "[CcuTempAllGatherMesh1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],"
        "outputAddr[%llu], sliceSize[%llu], offset[%llu]",
        myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, offset);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempAllGatherMesh1D] links.size[%zu]", links.size());
    ccuInsAllGatherMesh1D.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 3;
    ccuInsAllGatherMesh1D.SetCntCkeNum(cntCkeNum);
    ccuInsAllGatherMesh1D.SetRankGroup(rankGroup);
    HCCL_INFO("CCUInsAllGathermesh1D is [%s]", ccuInsAllGatherMesh1D.Describe().c_str());
    ccuInsAllGatherMesh1D.Describe();
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>(ccuInsAllGatherMesh1D)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1D::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    // 框架解析aicpuIns，算法的algCompnnetLite在device侧直接调用Run（）
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
