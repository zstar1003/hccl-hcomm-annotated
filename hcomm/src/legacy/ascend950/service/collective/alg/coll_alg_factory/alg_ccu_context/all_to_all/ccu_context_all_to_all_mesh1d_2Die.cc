/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_to_all_mesh1d_2Die.h"
#include "ccu_instruction_all_to_all_mesh1d_2Die.h"

namespace Hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int INPUT_XN_ID = 0;
constexpr int OUPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;

constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;

CcuContextAllToAllMesh1D2Die::CcuContextAllToAllMesh1D2Die(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllToAllMesh1D2Die* ctxArg = dynamic_cast<const CcuCtxArgAllToAllMesh1D2Die*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D2Die::ctxArg ptr is null"));
    }

    rankId_ = ctxArg->rankId_;
    withMyRank_ = ctxArg->withMyRank_;
    rankGroup_ = ctxArg->rankGroup;
    if (ctxArg->dimSize_.size() > 0) {
        rankSize_ = ctxArg->dimSize_[0];
    }
    bitNumPerCKE_ = ctxArg->bitNum_;
}

void CcuContextAllToAllMesh1D2Die::InitResource()
{
    // 创建Variable，用于交换地址及token
    u32 transportId = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D2Die transports is empty"));
    }
    virRankSize = transports.size() + 1;

    for (u64 id = 0; id < transports.size(); id++) {
        // 非本地，使用远端Variable
        CHK_PRT_RET(
            transports[transportId] == nullptr,
            HCCL_ERROR("[CcuContextAllToAllMesh1D2Die] Algorithm transport ptr is null"), );
        input_.push_back(CreateVariable((*transports[transportId]), CKE_IDX_0));
        output_.push_back(CreateVariable((*transports[transportId]), CKE_IDX_1));
        token_.push_back(CreateVariable((*transports[transportId]), CKE_IDX_2));
        transportId++;
    }
    // 最后一个位置放自己地址
    input_.push_back(CreateVariable());
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    sliceSize_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputoffset_ = CreateVariable();
    outBuffBaseOff_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    moConfig.loopCount = 8;                          // loop展开8次、16次
    moConfig.msInterleave = LOCAL_COPY_MS;           // 一个loop 8个MS
    moConfig.memSlice = LOCAL_COPY_MS * CCU_MS_SIZE; // 32k
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.maskSignal = CreateBlockMaskSignal(moConfig.loopCount);
        moRes.ccuBuffer = CreateBlockCcuBuffer(moConfig.loopCount * moConfig.msInterleave);
    }

    logicRankSize = withMyRank_ ? transports.size() + 1 : transports.size();
    signalNum_ = (rankSize_ + bitNumPerCKE_ - 1) / bitNumPerCKE_;
    HCCL_INFO(
        "[CcuContextAlltoAll2Die] CtxArg: rankId_[%u], rankSize_[%u], signalNum_[%u]", rankId_, rankSize_, signalNum_);
    return;
}

void CcuContextAllToAllMesh1D2Die::LoadArgs()
{
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, srcOffset, dstOffset, groupOpSize
    Load(input_[virRankSize - 1]);
    Load(output_[virRankSize - 1]);
    Load(token_[virRankSize - 1]);
    Load(sliceSize_); // 本轮传输的分片大小
    Load(inputSliceStride_);
    Load(outputoffset_);
    Load(outBuffBaseOff_);
    Load(groupOpSize_);
    return;
}

void CcuContextAllToAllMesh1D2Die::PreSync()
{
    if (withMyRank_) {
        uint16_t logicId = rankId_ % logicRankSize;
        selfBit = 1 << logicId;
        allBit = ((1 << logicRankSize) - 1) & (~(1 << logicId));
        for (auto t : transports) {
            // （transport, param, paramID, SemID, mask）
            WriteVariableWithSignal(
                *t, output_[virRankSize - 1], OUPUT_XN_ID, CKE_IDX_1,
                selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(
                *t, token_[virRankSize - 1], TOKEN_XN_ID, CKE_IDX_2,
                selfBit); // index = 2，传递token信息
        }
        GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
        GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息
    } else {
        uint16_t selfSignalId = rankId_ / bitNumPerCKE_;
        uint16_t selfBit = 1 << (rankId_ % bitNumPerCKE_);
        for (auto t : transports) {
            // （transport, param, paramID, SemID, mask）
            WriteVariableWithSignal(
                *t, output_[virRankSize - 1], OUPUT_XN_ID, selfSignalId + signalNum_ * CKE_IDX_1,
                selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(
                *t, token_[virRankSize - 1], TOKEN_XN_ID, selfSignalId + signalNum_ * CKE_IDX_2,
                selfBit); // index = 2，传递token信息
        }
        std::vector<uint16_t> waitBitVector(signalNum_, 0);
        for (uint16_t sId = 0; sId < waitBitVector.size(); sId++) {
            waitBitVector[sId] = (1 << bitNumPerCKE_) - 1;
            if (sId == selfSignalId) {
                waitBitVector[sId] = 0;
            }
            GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_1, waitBitVector[sId]); // index = 1，传递output信息
            GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_2, waitBitVector[sId]); // index = 2，传递token信息
        }
    }
    return;
}

void CcuContextAllToAllMesh1D2Die::PostSync()
{
    if (withMyRank_) {
        uint16_t logicId = rankId_ % logicRankSize;
        selfBit = 1 << logicId;
        allBit = ((1 << logicRankSize) - 1) & (~(1 << logicId));
        for (auto t : transports) {
            if (t == nullptr) {
                THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D2Die::Algorithm transport ptr is null"));
            }
            RemotePost(*t, CKE_IDX_0, selfBit);
        }
        GroupWait(*transportGroup, CKE_IDX_0, allBit);
    } else {
        uint16_t selfSignalId = rankId_ / bitNumPerCKE_;
        uint16_t selfBit = 1 << (rankId_ % bitNumPerCKE_);
        for (auto t : transports) {
            if (t == nullptr) {
                THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D2Die::Algorithm transport ptr is null"));
            }
            RemotePost(*t, selfSignalId + signalNum_ * CKE_IDX_0, selfBit);
        }
        std::vector<uint16_t> waitBitVector(signalNum_, 0);
        for (uint16_t sId = 0; sId < waitBitVector.size(); sId++) {
            waitBitVector[sId] = (1 << bitNumPerCKE_) - 1;
            if (sId == selfSignalId) {
                waitBitVector[sId] = 0;
            }
            GroupWait(*transportGroup, CKE_IDX_0, allBit);
        }
    }

    return;
}

uint32_t CcuContextAllToAllMesh1D2Die::CalcDstRank(uint32_t peerId) const
{
    if (peerId > rankGroup_.size()) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextAllToAllMesh1D2Die][CalcDstRank] Unexpected peerId[%u]", peerId));
    }
    return rankGroup_[peerId];
}

void CcuContextAllToAllMesh1D2Die::DoRepeatAllToAll()
{
    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    std::vector<CcuRep::Memory> src;
    for (uint64_t rankIdx = 0; rankIdx < logicRankSize; rankIdx++) {
        src.push_back(CreateMemory());
    }
    std::vector<CcuRep::Memory> dst;
    for (uint64_t rankIdx = 0; rankIdx < logicRankSize; rankIdx++) {
        dst.push_back(CreateMemory());
    }

    // 考虑stride信息
    for (uint64_t r = 0; r < logicRankSize; r++) {
        const u32 dstRank = CalcDstRank(r);

        src[r].token = token_[r];
        dst[r].token = token_[r];

        src[r].addr = input_[virRankSize - 1];
        dst[r].addr = output_[r];
        dst[r].addr += outputoffset_;
        for (uint64_t i = 0; i < dstRank; i++) {
            src[r].addr += inputSliceStride_;
        }
    }

    //  all2all 数据搬运
    u32 transportIdx = 0;
    if (withMyRank_) {
        uint64_t allBit_
            = withMyRank_ ? ((1 << logicRankSize) - 1) & (~(1 << transports.size())) : (1 << logicRankSize) - 1;
        CcuRep::MaskSignal locMask = CreateMaskSignal();
        for (uint64_t r = 0; r < logicRankSize; r++) {
            if (withMyRank_ && r == logicRankSize - 1) {
                LocalCopyByLoopGroup(dst[r], src[r]);
                continue;
            }
            Write(*transports[transportIdx], dst[r], src[r], sliceSize_, locMask, 1 << r);
            transportIdx++;
        }
        LocalWait(locMask, allBit_);
    } else {
        vector<CcuRep::MaskSignal> locMask;
        uint16_t signalNum = logicRankSize / bitNumPerCKE_;
        std::vector<uint16_t> waitBitVector(signalNum, 0);
        for (uint16_t sId = 0; sId < signalNum; sId++) {
            locMask.push_back(CreateMaskSignal());
        }
        for (uint16_t r = 0; r < logicRankSize; r++) {
            uint16_t rmtSignalId = r / bitNumPerCKE_;
            uint16_t rmtSignalBit = 1 << (r % bitNumPerCKE_);
            Write(*transports[transportIdx], dst[r], src[r], sliceSize_, locMask[rmtSignalId], rmtSignalBit);
            transportIdx++;
        }
        for (uint16_t sId = 0; sId < signalNum; sId++) {
            waitBitVector[sId] = (1 << bitNumPerCKE_) - 1;
            LocalWait(locMask[sId], waitBitVector[sId]);
        }
    }
}

void CcuContextAllToAllMesh1D2Die::CreateLocalCopyLoop()
{
    std::string loopType = "all_to_all";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    for (uint32_t index = 0; index < 2; index++) { // 需要2个Loop
        CcuRep::Variable len = CreateVariable();
        CcuRep::Memory src = CreateMemory();
        CcuRep::Memory dst = CreateMemory();
        CcuRep::LoopBlock lb(this, loopType + "_localcopy_loop_" + std::to_string(index));
        lb(src, dst, len);

        std::vector<CcuRep::CcuBuffer> bufs;
        CcuRep::MaskSignal sem = moRes.maskSignal[index];
        for (uint32_t i = 0; i < LOCAL_COPY_MS; i++) {
            bufs.push_back(moRes.ccuBuffer[i]);
        }

        LocalCopy(bufs[0], src, len, sem);
        LocalWait(sem);
        LocalCopy(dst, bufs[0], len, sem);
        LocalWait(sem);
    }
    registeredLoop.insert(loopType);
    return;
}

void CcuContextAllToAllMesh1D2Die::LocalCopyByLoopGroup(CcuRep::Memory dst, CcuRep::Memory src)
{
    CreateLocalCopyLoop();

    CCU_IF(groupOpSize_.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += groupOpSize_.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc = Loop("all_to_all_localcopy_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(groupOpSize_.parallelParam != 0)
    {
        CcuRep::Condition cond(this, groupOpSize_.parallelParam != 0);

        src.addr += groupOpSize_.addrOffset;
        dst.addr += groupOpSize_.addrOffset;
        auto lc0 = Loop("all_to_all_localcopy_loop_0")(src, dst, groupOpSize_.residual);

        src.addr += groupOpSize_.residual;
        dst.addr += groupOpSize_.residual;
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc1 = Loop("all_to_all_localcopy_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, groupOpSize_.parallelParam, offsetCfg);
    }
}

void CcuContextAllToAllMesh1D2Die::Algorithm()
{
    HCCL_INFO("[ccuAllToAllMesh1D2Die_context] AllToAllMesh1D2Die run.");
    InitResource();

    LoadArgs();

    PreSync();

    DoRepeatAllToAll();

    PostSync();
    HCCL_INFO("[ccuAllToAllMesh1D2Die_context] AllToAllMesh1D2Die end.");
    return;
}

std::vector<uint64_t> CcuContextAllToAllMesh1D2Die::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllToAllMesh1D2Die* taskArg = dynamic_cast<const CcuTaskArgAllToAllMesh1D2Die*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D2Die::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t sliceSize = taskArg->sliceSize_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_ * rankId_;
    uint64_t outBuffBaseOff = taskArg->outBuffBaseOff_;

    auto goSize = CalGoSize(sliceSize);
    HCCL_INFO(
        "[CcuContextAllToAllMesh1D2Die] inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], "
        "inputSliceStride[%llu], outputSliceStride[%llu], outBuffBaseOff[%llu].",
        inputAddr, outputAddr, sliceSize, inputSliceStride, outputSliceStride, outBuffBaseOff);

    return {inputAddr,      outputAddr, tokenInfo, sliceSize, inputSliceStride, outputSliceStride,
            outBuffBaseOff, goSize[0],  goSize[1], goSize[2], goSize[3]};
}

} // namespace Hccl
