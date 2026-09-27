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

#include "ccu_ins_group.h"
#include "ccu_instruction_all_reduce_mesh1d_one_shot.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_reduce_mesh1d_one_shot.h"
#include "ccu_temp_all_reduce_mesh_1D_one_shot.h"
#include "ccu_context_reduce_tail_block.h"

namespace Hccl {
static CcuInstRegister<CcuContextAllReduceMesh1DOneShot>
    g_registrarAllReduce(CcuInstType::CCU_ALL_REDUCE_MESH_1D_ONE_SHOT_DIRECT);

static CcuInstRegister<CcuContextReduceTailBlock> g_registrarReduceTailBlock(CcuInstType::CCU_REDUCE_TAILBLOCK_DIRECT);

CcuTempAllReduceMesh1DOneShot::CcuTempAllReduceMesh1DOneShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllReduceMesh1DOneShot::~CcuTempAllReduceMesh1DOneShot() {}

void CcuTempAllReduceMesh1DOneShot::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempAllReduceMesh1DOneShot::CalcSliceInfo(
    const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    (void)allignInfo;
    SliceInfo basicSlice;
    basicSlice.offset = 0;
    basicSlice.size = dataSize;
    HCCL_INFO("[CcuTempAllReduceMesh1DOneShot][CalcSliceInfo] basicSlice.size[%u]", basicSlice.size);
    std::vector<SliceInfo> singleRankSliceInfoVector{basicSlice};
    sliceInfoVec.resize(tempRankSize_, singleRankSliceInfoVector);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    HCCL_INFO("[CcuTempAllReduceMesh1DOneShot][CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempAllReduceMesh1DOneShot][Run] start");
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAllReduceMesh1DOneShot] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;
    CHK_PRT_RET(
        myRank_ == INVALID_RANKID, HCCL_ERROR("[CcuTempAllReduceMesh1DOneShot][Run]myRank[%d] is invalid", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    uint32_t rankId = static_cast<uint32_t>(myRank_);

    uint64_t inputAddr;
    uint64_t outputAddr;
    CHK_RET(CalcInputOutputAddr(tempFuncs, inputAddr, outputAddr));

    std::vector<LinkData> links;
    CHK_RET(PrepareLinks(tempLinks, links));

    uint64_t token;
    CHK_RET(GetToken(op_, token));

    uint64_t totalSliceSize = sliceInfoVec[myRank_][0].size; // 本rank需要处理的数据量

    u32 cntCkeNum = 4;

    RankGroup rankGroup;
    CHK_RET(PrepareRankGroup(rankGroup));

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();

    std::string notifySignal = "AllReduceMesh1DOneShot_TailBlock";
    // 添加主拓展指令
    CcuInstructionAllReduceMesh1DOneShot ccuInstructionAllReduceMesh1DOneShot;
    ccuInstructionAllReduceMesh1DOneShot.Init(
        rankId, inputAddr, outputAddr, totalSliceSize, token, notifySignal, op_, tempVTopo_);
    ccuInstructionAllReduceMesh1DOneShot.SetLinks(links);
    ccuInstructionAllReduceMesh1DOneShot.SetCntCkeNum(cntCkeNum);
    ccuInstructionAllReduceMesh1DOneShot.SetRankGroup(rankGroup);
    tempInsQues[0]->Append(
        std::move(std::make_unique<CcuInstructionAllReduceMesh1DOneShot>(ccuInstructionAllReduceMesh1DOneShot)));
    HCCL_INFO("[CcuTempAllReduceMesh1DOneShot][Run] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::CalcInputOutputAddr(
    const TempFuncs& tempFuncs, uint64_t& inputAddr, uint64_t& outputAddr)
{
    HCCL_INFO("[CcuTempAllReduceMesh1DOneShot][CalcInputOutputAddr] start");
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
        // 图模式
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff
                     + tempFuncs.usrData.usrOutSlices[0].GetOffset();
    }
    HCCL_INFO(
        "[CcuTempAllReduceMesh1DOneShot][CalcInputOutputAddr] end, inputAddr[%llu], outputAddr[%llu]", inputAddr,
        outputAddr);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::PrepareLinks(const ResLinks& tempLinks, std::vector<LinkData>& links) const
{
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempAllReduceMesh1DOneShot][PrepareLinks] end, links.size[%zu]", links.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::PrepareRankGroup(RankGroup& rankGroup)
{
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
