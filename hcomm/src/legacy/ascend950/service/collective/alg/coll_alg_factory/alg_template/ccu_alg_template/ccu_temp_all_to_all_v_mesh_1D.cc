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

#include "ccu_temp_all_to_all_v_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_all_to_all_v_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_to_all_v_mesh1d.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllToAllVMesh1D> registrarAllToAllV(CcuInstType::CCU_ALLTOALLV_MESH_1D_DIRECT);

CcuTempAlltoAllVMesh1D::CcuTempAlltoAllVMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAlltoAllVMesh1D::~CcuTempAlltoAllVMesh1D() {}

HcclResult CcuTempAlltoAllVMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuTempAlltoAllVMesh1D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D::GetScratchBufferInfo(const uint64_t scratchBufferSize, DataType dataType)
{
    (void)scratchBufferSize;
    (void)dataType;
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAlltoAllVMesh1D::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAlltoAllVMesh1D::SetBuffBlockSize(const u64 buffBlockSize)
{
    CHK_PRT_RET(
        buffBlockSize == 0, HCCL_ERROR("[CcuTempAlltoAllVMesh1D][SetBuffBlockSize] buffBlockSize should not be zero"),
        HcclResult::HCCL_E_PARA);
    buffBlockSize_ = buffBlockSize;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D::SetConcurrentSendRecvNum(const u32 concurrentSendRecvNum)
{
    CHK_PRT_RET(
        concurrentSendRecvNum == 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D][SetConcurrentSendRecvNum] concurrentSendRecvNum "
                   "should not be zero"),
        HcclResult::HCCL_E_PARA);
    concurrentSendRecvNum_ = concurrentSendRecvNum;
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempAlltoAllVMesh1D::CalcSendRecvNumSubStep()
{
    sendNumSubStep_.clear();
    recvNumSubStep_.clear();
    uint64_t numSubStep = 0;
    u32 rankSize = localSendRecvInfo_.sendLength.size();

    for (u32 destRank = 0; destRank < rankSize; destRank++) {
        uint64_t currRankSendSubStep
            = ((localSendRecvInfo_.sendLength[destRank] + UB_MAX_DATA_SIZE - 1) / UB_MAX_DATA_SIZE);
        sendNumSubStep_[destRank] = currRankSendSubStep;

        uint64_t currRankRecvSubStep
            = ((localSendRecvInfo_.recvLength[destRank] + UB_MAX_DATA_SIZE - 1) / UB_MAX_DATA_SIZE);
        recvNumSubStep_[destRank] = currRankRecvSubStep;
        HCCL_INFO(
            "[CcuTempAlltoAllVMesh1D][CalcNumSubStep] myRank [%d] currRankSendSubStep[%llu] "
            "currRankRecvSubStep[%llu]",
            myRank_, currRankSendSubStep, currRankRecvSubStep);
        numSubStep = std::max(numSubStep, std::max(currRankSendSubStep, currRankRecvSubStep));
    }
    HCCL_INFO("[CcuTempAlltoAllVMesh1D][CalcNumSubStep] myRank [%d] max communication step[%u]", myRank_, numSubStep);
    return numSubStep;
}

HcclResult CcuTempAlltoAllVMesh1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempAlltoAllVMesh1D] Run");
    (void)sliceInfoVec;
    (void)tempFuncs;
    (void)buffInfo;

    if (tempRankSize_ == 1) {
        // ccu-alltoall算子的单P场景单独处理
        CHK_PRT_RET(
            localSendRecvInfo_.sendLength[myRank_] != localSendRecvInfo_.recvLength[myRank_],
            HCCL_ERROR(
                "[CcuTempAlltoAllVMesh1D] rankSize = 1, sendLength[%llu] and recvLength[%llu] "
                "should be equal.",
                localSendRecvInfo_.sendLength[myRank_], localSendRecvInfo_.recvLength[myRank_]),
            HcclResult::HCCL_E_PARA);
        CHK_PRT_RET(
            localSendRecvInfo_.sendLength[myRank_] == 0,
            HCCL_INFO("[CcuTempAlltoAllVMesh1D] Single Rank and DataSlice size is 0, no need to process."),
            HcclResult::HCCL_SUCCESS);

        DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, localSendRecvInfo_.sendLength[myRank_]);
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, 0, localSendRecvInfo_.recvLength[myRank_]);

        std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
        tempInsQues[0]->Append(std::move(insLocalCopy));
        HCCL_INFO(
            "[CcuTempAlltoAllVMesh1D] rankSize = 1, use InsLocalCopy for sliceSize[%llu].",
            localSendRecvInfo_.sendLength[myRank_]);
        return HcclResult::HCCL_SUCCESS;
    }

    CcuInstructionAllToAllVMesh1D ccuInsAllToAllVMesh1D;

    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    std::vector<uint64_t> sliceSize;
    sliceSize.reserve(localSendRecvInfo_.sendLength.size());
    for (auto& slice : localSendRecvInfo_.sendLength) {
        sliceSize.push_back(slice);
    }

    uint64_t inputAddr = op_.inputMem == nullptr ? 0 : static_cast<uint64_t>(op_.inputMem->GetAddr()); // usrIN起始地址
    uint64_t outputAddr = op_.outputMem == nullptr ? 0 : static_cast<uint64_t>(op_.outputMem->GetAddr());
    uint64_t srcOffset = 0; // alltoallv假设src起始地址为发送rank的对应块起始地址
    uint64_t dstOffset = 0; // alltoallv假设dst起始地址为接收rank的对应块起始地址
    uint64_t token = 0;
    GetToken(op_, token);

    ccuInsAllToAllVMesh1D.Init(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, sliceSize, token, srcOffset, dstOffset, op_, tempVTopo_,
        localSendRecvInfo_, loadFromMem_);
    HCCL_INFO(
        "[CcuTempAlltoAllVMesh1D] Run Init: LoadFromMem[%d], myRank_[%d], dimSize[%llu], inputAddr[%llu],"
        "outputAddr[%llu], sliceSize size[%llu], srcOffset[%llu], dstOffset[%llu]",
        loadFromMem_, myRank_, dimSize[0], inputAddr, outputAddr, sliceSize.size(), srcOffset, dstOffset);
    //  init links
    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempAlltoAllVMesh1D] links.size[%zu]", links.size());
    ccuInsAllToAllVMesh1D.SetLinks(links);
    RankGroup rankGroup;
    if (tempVTopo_.size() == 0 || tempInsQues.size() == 0 || tempVTopo_[0].size() == 0) {
        HCCL_ERROR(
            "[CcuTempAlltoAllVMesh1D] invalid tempVTopo size is [%u] or invalid tempInsQues size is [%u] or "
            "invalid tempVTopo_[0].size is [%u].",
            tempVTopo_.size(), tempInsQues.size(), tempVTopo_[0].size());
        return HcclResult::HCCL_E_PARA;
    }
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 3; // 默认为3，实际alltoallv只需要2个
    ccuInsAllToAllVMesh1D.SetCntCkeNum(cntCkeNum);
    ccuInsAllToAllVMesh1D.SetRankGroup(rankGroup);
    HCCL_INFO("ccuInsAllToAllVMesh1D is [%s]", ccuInsAllToAllVMesh1D.Describe().c_str());
    ccuInsAllToAllVMesh1D.Describe();
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionAllToAllVMesh1D>(ccuInsAllToAllVMesh1D)));

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
