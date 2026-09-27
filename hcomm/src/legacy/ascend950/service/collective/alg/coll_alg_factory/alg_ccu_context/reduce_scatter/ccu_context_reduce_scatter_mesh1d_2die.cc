/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_mesh1d_2die.h"
#include "ccu_instruction_reduce_scatter_mesh1d_2die.h"
#include "ccu_assist.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 1;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int LOOP_NUM = 128;

constexpr int MISSION_NUM = 2;
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};

CcuContextReduceScatterMesh1D2Die::CcuContextReduceScatterMesh1D2Die(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceScatterMesh1D2Die* ctxArg = dynamic_cast<const CcuCtxArgReduceScatterMesh1D2Die*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh1D2Die::ctxArg ptr is null"));
    }
    moConfig.loopCount = LOOP_NUM;

    rmtReduceWithMyRank_ = ctxArg->rmtReduceWithMyRank_;
    myRankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];

    rmtReduceRankNum_ = transports.size() + (rmtReduceWithMyRank_ == true ? 1 : 0);

    rmtSyncMyBit_ = 1 << (myRankId_ % rmtReduceRankNum_);
    rmtSyncWaitBit_
        = rmtReduceWithMyRank_ ? ((1 << rmtReduceRankNum_) - 1) & (~rmtSyncMyBit_) : (1 << rmtReduceRankNum_) - 1;

    ctxName_ = ctxArg->GetCtxSignature().Describe();
    myMissionSignalName_ = ctxName_ + (rmtReduceWithMyRank_ ? "_withMyRank" : "_withoutMyRank");
    otherMissionSignalName_ = ctxName_ + (!rmtReduceWithMyRank_ ? "_withMyRank" : "_withoutMyRank");

    missionSyncMybit_ = 1 << (rmtReduceWithMyRank_ ? 1 : 0);
    missionSyncWaitBit_ = 1 << (!rmtReduceWithMyRank_ ? 1 : 0);

    // 数据类型处理
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceScatterMesh1D2Die] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
}

void CcuContextReduceScatterMesh1D2Die::InitResources()
{
    moConfig.loopCount = LOOP_NUM;

    myInput_ = CreateVariable();
    myOutput_ = CreateVariable();
    myScratch_ = CreateVariable();
    myToken_ = CreateVariable();

    for (auto& t : transports) {
        peerInput_.push_back(CreateVariable(*t, INPUT_XN_ID));
        peerToken_.push_back(CreateVariable(*t, TOKEN_XN_ID));
    }

    sliceSize_ = CreateVariable();

    rmtReduceSliceOffset_ = CreateVariable();

    rmtReduceGoSize_ = CreateGroupOpSize();

    AllocGoResource(LOOP_NUM);

    myMissionSignal_ = CreateMaskSignal();
    ExportMaskSignal(myMissionSignal_, myMissionSignalName_);
    otherMissionSignal_ = ImportMaskSignal(otherMissionSignalName_);
}

void CcuContextReduceScatterMesh1D2Die::LoadArgs()
{
    Load(myInput_);
    Load(myOutput_);
    Load(myToken_);
    Load(myScratch_);
    Load(sliceSize_);
    Load(rmtReduceSliceOffset_);
    Load(rmtReduceGoSize_);
}

void CcuContextReduceScatterMesh1D2Die::PreSync()
{
    for (auto& t : transports) {
        WriteVariableWithSignal(*t, myInput_, INPUT_XN_ID, CKE_IDX_1, rmtSyncMyBit_);
        WriteVariableWithSignal(*t, myToken_, TOKEN_XN_ID, CKE_IDX_2, rmtSyncMyBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_1, rmtSyncWaitBit_);
    GroupWait(*transportGroup, CKE_IDX_2, rmtSyncWaitBit_);
}

void CcuContextReduceScatterMesh1D2Die::PostSync(uint32_t signalIndex)
{
    for (auto& t : transports) {
        RemotePost(*t, signalIndex, rmtSyncMyBit_);
    }
    GroupWait(*transportGroup, signalIndex, rmtSyncWaitBit_);
}

void CcuContextReduceScatterMesh1D2Die::MissionSync(uint32_t signalIndex)
{
    HCCL_INFO(
        "[CcuContextReduceScatterMesh1D2Die] MissionSync, missionSyncMybit_[%u], missionSyncWaitBit_[%u]",
        missionSyncMybit_, missionSyncWaitBit_);
    LocalCtxPost(otherMissionSignal_, missionSyncMybit_ << (signalIndex * MISSION_NUM));
    LocalWait(myMissionSignal_, missionSyncWaitBit_ << (signalIndex * MISSION_NUM));
    return;
}

void CcuContextReduceScatterMesh1D2Die::RmtReduce()
{
    std::vector<CcuRep::Memory> src;
    src.reserve(rmtReduceRankNum_);
    for (uint32_t peerIdx = 0; peerIdx < transports.size(); peerIdx++) {
        src.push_back(CreateMemory());
        src.back().token = peerToken_[peerIdx];
        src.back().addr = peerInput_[peerIdx];
        src.back().addr += rmtReduceSliceOffset_;
    }
    if (rmtReduceWithMyRank_) {
        src.push_back(CreateMemory());
        src.back().token = myToken_;
        src.back().addr = myInput_;
        src.back().addr += rmtReduceSliceOffset_;
    }

    CcuRep::Memory dst = CreateMemory();
    dst.token = myToken_;
    dst.addr = rmtReduceWithMyRank_ ? myOutput_ : myScratch_;

    if (rmtReduceWithMyRank_) {
        GroupReduce(transports, dst, src, rmtReduceGoSize_, dataType_, outputDataType_, reduceOp_);
    } else {
        GroupReduceWithoutMyRank(transports, dst, src, rmtReduceGoSize_, dataType_, outputDataType_, reduceOp_);
    }
}

std::string CcuContextReduceScatterMesh1D2Die::GetLoopBlockTag(std::string loopType, int32_t index) const
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}

void CcuContextReduceScatterMesh1D2Die::CreateReduceLoop(
    uint32_t size, DataType dataType, DataType outputDataType, ReduceOp opType)
{
    std::string loopType = CcuRep::GetReduceTypeStr(dataType, opType);
    loopType = "local_reduce_" + loopType;
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::Memory dst = CreateMemory();
        std::vector<CcuRep::Memory> src;
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateMemory());
        }
        CcuRep::Variable len = CreateVariable();
        CcuRep::Variable lenForExpansion = CreateVariable();
        CcuRep::LoopBlock lb(this, GetLoopBlockTag(loopType, index));
        lb(dst, src, len, lenForExpansion);

        std::vector<CcuRep::CcuBuffer> bufs
            = {moRes.ccuBuffer.begin() + index * moConfig.msInterleave,
               moRes.ccuBuffer.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::MaskSignal sem = moRes.maskSignal[index];

        for (uint32_t i = 0; i < size; i++) {
            LocalCopy(bufs[i], src[i], len, sem, 1 << i);
        }
        LocalWait(sem, (1 << size) - 1);

        if (size > 1) {
            LocalReduce(bufs, size, dataType, outputDataType, opType, sem, len);
            LocalWait(sem);
        }

        LocalCopy(dst, bufs[0], lenForExpansion, sem);
        LocalWait(sem);
    }

    registeredLoop.insert(loopType);
}

void CcuContextReduceScatterMesh1D2Die::ReduceLoopGroup(
    CcuRep::Memory& outDstOrg, std::vector<CcuRep::Memory>& srcOrg, GroupOpSize goSize, DataType dataType,
    DataType outputDataType, ReduceOp opType)
{
    const uint32_t size = srcOrg.size();

    CcuRep::Memory dst = CreateMemory();
    dst = outDstOrg;

    std::vector<CcuRep::Memory> src;
    for (uint32_t idx = 0; idx < size; idx++) {
        src.push_back(CreateMemory());
        src[idx] = srcOrg[idx];
    }

    CreateReduceLoop(size, dataType, outputDataType, opType);

    std::string loopType = CcuRep::GetReduceTypeStr(dataType, opType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();
    loopType = "local_reduce_" + loopType;
    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = CcuRep::GetExpansionParam(expansionNum);
        dst.token += tmp;
    }

    // m部分
    CCU_IF(goSize.loopParam != 0) // goSize1
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, src, sliceSize, sliceSizeExpansion);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(goSize.parallelParam != 0) // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual; // goSize3
        }

        auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, src, goSize.residual, sliceSizeExpansion);

        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.residual;
        }

        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, src, sliceSize, sliceSizeExpansion);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}

void CcuContextReduceScatterMesh1D2Die::Algorithm()
{
    InitResources();
    LoadArgs();
    PreSync();
    RmtReduce();
    PostSync(CKE_IDX_0);
    return;
}

std::vector<uint64_t> CcuContextReduceScatterMesh1D2Die::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterMesh1D2Die* taskArg = dynamic_cast<const CcuTaskArgReduceScatterMesh1D2Die*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh1D2Die::taskArg ptr is null"));
    }
    moConfig.loopCount = LOOP_NUM;
    uint64_t myInput = taskArg->inputAddr_;
    uint64_t myOutput = taskArg->outputAddr_;
    uint64_t myToken = taskArg->token_;
    uint64_t myScratch = taskArg->scratchAddr_;

    uint64_t sliceSize = taskArg->sliceSize_;

    uint64_t rmtReduceSliceOffset = sliceSize * myRankId_;

    u32 dataTypeSize = DataTypeSizeGet(dataType_);

    uint64_t localRedcueSize0 = (sliceSize / dataTypeSize) / MISSION_NUM * dataTypeSize;
    uint64_t localRedcueSize1 = sliceSize - localRedcueSize0;

    auto rmtReduceGoSize = CalGoSize(sliceSize);
    auto localReduceGoSize0 = CalGoSize(localRedcueSize0);
    auto localReduceGoSize1 = CalGoSize(localRedcueSize1);

    HCCL_INFO(
        "[CcuContextReduceScatterMesh1D2Die][GeneArgs] myInput[%llu], myOutput[%llu], myScratch[%llu], "
        "rmtReduceSliceOffset[%llu], sliceSize[%llu], localReduceSize0[%llu], localReduceSize1[%llu]",
        myInput, myOutput, myScratch, rmtReduceSliceOffset, sliceSize, localRedcueSize0, localRedcueSize1);

    std::vector<uint64_t> taskArgs = {myInput, myOutput, myToken, myScratch, sliceSize, rmtReduceSliceOffset};

    for (auto& goSize : {rmtReduceGoSize}) {
        for (auto& element : goSize) {
            taskArgs.push_back(element);
        }
    }
    return taskArgs;
}
} // namespace Hccl
