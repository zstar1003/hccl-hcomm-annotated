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

#include "ccu_temp_all_to_all_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_all_to_all_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_to_all_mesh1d.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllToAllMesh1D> registrarAllToAll(CcuInstType::CCU_ALLTOALL_MESH_1D_DIRECT);

CcuTempAllToAllMesh1D::CcuTempAllToAllMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllToAllMesh1D::~CcuTempAllToAllMesh1D() {}

HcclResult CcuTempAllToAllMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}
/*
dataSize / (rankSize) --> chunkSize
dataSize / (rankSize * queNum) --> sliceSize

SliceInfoVecforNHR: [1st chunk: [1st Slice, 2nd Slice, ...], 2nd chunk: [1st Slice, 2nd Slice, ...], ...]
*/
HcclResult
CcuTempAllToAllMesh1D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAllToAllMesh1D::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAllToAllMesh1D::SetBuffBlockSize(const u64 buffBlockSize)
{
    CHK_PRT_RET(
        buffBlockSize == 0, HCCL_ERROR("[CcuTempAllToAllMesh1D][SetBuffBlockSize] buffBlockSize should not be zero"),
        HcclResult::HCCL_E_PARA);
    buffBlockSize_ = buffBlockSize;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D::SetConcurrentSendRecvNum(const u32 concurrentSendRecvNum)
{
    CHK_PRT_RET(
        concurrentSendRecvNum == 0,
        HCCL_ERROR("[CcuTempAllToAllMesh1D][SetConcurrentSendRecvNum] concurrentSendRecvNum should not be zero"),
        HcclResult::HCCL_E_PARA);
    concurrentSendRecvNum_ = concurrentSendRecvNum;
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempAllToAllMesh1D::DataSliceToAddr(const DataSlice& dataSlice)
{
    if (dataSlice.GetType() == BufferType::INPUT) {
        return static_cast<uint64_t>(op_.inputMem->GetAddr());
    } else if (dataSlice.GetType() == BufferType::OUTPUT) {
        return static_cast<uint64_t>(op_.outputMem->GetAddr());
    } else {
        return static_cast<uint64_t>(op_.scratchMem->GetAddr());
    }
}

HcclResult CcuTempAllToAllMesh1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D] Run");
    (void)sliceInfoVec;
    (void)tempFuncs;
    (void)buffInfo;
    CcuInstructionAllToAllMesh1D ccuInsAllToAllMesh1D;
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempAllToAllMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);
    // 拿到input和output的首地址,和每片小数据的大小
    uint64_t totalSliceSize = localSendRecvInfo_.sendLength[0]; // Bytes
    uint64_t inputAddr = op_.inputMem == nullptr ? 0 : static_cast<uint64_t>(op_.inputMem->GetAddr());
    uint64_t outputAddr = op_.outputMem == nullptr ? 0 : static_cast<uint64_t>(op_.outputMem->GetAddr());
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t srcStride = totalSliceSize + sendStrideSize_;
    uint64_t dstStride = totalSliceSize + recvStrideSize_;

    uint64_t loopCnt = totalSliceSize / UB_MAX_DATA_SIZE + (totalSliceSize % UB_MAX_DATA_SIZE == 0 ? 0 : 1);
    uint64_t sliceBias = 0;
    for (uint64_t i = 0; i < loopCnt; i++) {
        if (tempRankSize_ == 1) {
            // ccu-alltoall算子的单P场景单独处理
            DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, totalSliceSize);
            DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, 0, totalSliceSize);
            std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
            tempInsQues[0]->Append(std::move(insLocalCopy));
            HCCL_INFO("[CcuTempAllToAllMesh1D] rankSize = 1, use InsLocalCopy for sliceSize[%llu].", totalSliceSize);
            break;
        }
        //  prepare parameters & ccuIns init
        uint64_t sliceSize = ((i == loopCnt - 1) ? (totalSliceSize - i * UB_MAX_DATA_SIZE) : UB_MAX_DATA_SIZE);
        uint64_t srcOffset = sliceBias;
        uint64_t dstOffset = sliceBias + myRank_ * dstStride;

        ccuInsAllToAllMesh1D.Init(
            static_cast<uint32_t>(myRank_), inputAddr, outputAddr, sliceSize, token, srcOffset, dstOffset, srcStride,
            op_, tempVTopo_, loadFromMem_);
        HCCL_INFO(
            "[CcuTempAllToAllMesh1D] Run Init: loadFromMem_[%d], myRank_[%d], dimSize[%llu], inputAddr[%llu],"
            "outputAddr[%llu], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu], loopCnt[%llu]",
            loadFromMem_, myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, srcOffset, dstOffset, loopCnt);
        //  init links
        std::vector<LinkData> links;
        for (auto& pair : tempLinks) {
            if (pair.second.empty()) {
                continue;
            }
            links.push_back(pair.second[0]);
        }
        HCCL_INFO("[CcuTempAllToAllMesh1D] links.size[%zu]", links.size());
        ccuInsAllToAllMesh1D.SetLinks(links);
        RankGroup rankGroup;
        for (auto& peer : tempVTopo_[0]) {
            rankGroup.AddRank(peer);
        }
        u32 cntCkeNum = 3;
        ccuInsAllToAllMesh1D.SetCntCkeNum(cntCkeNum);
        ccuInsAllToAllMesh1D.SetRankGroup(rankGroup);
        HCCL_INFO("CCUInsAllToAllmesh1D is [%s]", ccuInsAllToAllMesh1D.Describe().c_str());
        ccuInsAllToAllMesh1D.Describe();
        tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionAllToAllMesh1D>(ccuInsAllToAllMesh1D)));
        sliceBias += sliceSize;
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
