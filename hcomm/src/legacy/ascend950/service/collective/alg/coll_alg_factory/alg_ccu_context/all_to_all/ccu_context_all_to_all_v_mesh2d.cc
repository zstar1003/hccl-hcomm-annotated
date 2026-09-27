/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_to_all_v_mesh2d.h"
#include "ccu_instruction_all_to_all_v_mesh2d.h"

namespace Hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr int FST_AXIS_ID = 0;
constexpr int SEC_AXIS_ID = 1;

constexpr int SEND_LOOP_UPDATE_FLAG = 1;
constexpr int RECV_LOOP_UPDATE_FLAG = 2;

CcuContextAllToAllVMesh2D::CcuContextAllToAllVMesh2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    localAxisSignal_ = CreateMaskSignal();

    firstScratchBaseOffset_ = CreateVariable();
    secondScratchBaseOffset_ = CreateVariable();
    firstScratchSliceOffset_ = CreateVariable();
    firstScratchSliceStep_ = CreateVariable();
    secondScratchSliceOffset_ = CreateVariable();
    secondScratchSliceStep_ = CreateVariable();

    xnConst1_ = CreateVariable();
    completedRankCount_ = CreateVariable();
    xnHalfTransportSize_ = CreateVariable();
    xnMaxTransportSize_ = CreateVariable();
    curSendTailSize_ = CreateVariable();
    xnHalfTransportGoSize_ = CreateGroupOpSize();
    curSendTailGoSize_ = CreateGroupOpSize();

    if (transports.size() == 0) {
        THROW<InvalidParamsException>(StringFormat("CcuContextAllToAllVMesh2D transports is empty"));
    }

    const CcuCtxArgAllToAllVMesh2D* ctxArg = dynamic_cast<const CcuCtxArgAllToAllVMesh2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId;
    axisId_ = ctxArg->axisId;
    dimSize_ = ctxArg->dimSize;
    if (dimSize_.size() != 2 || axisId_ > 1) { // dimSize不为2，或axisId超过1，则不为2D场景
        THROW<InvalidParamsException>(
            StringFormat("CcuContextAlltoAllVMesh2D::dimSize[%u] or axisId[%u] is invalid", dimSize_.size(), axisId_));
    }
    if (dimSize_[0] <= 1 || dimSize_[1] <= 1) {
        THROW<InvalidParamsException>(StringFormat(
            "CcuContextAlltoAllVMesh2D::dimSize[0] is [%u], dimSize[1] is [%u] are invalid", dimSize_[0], dimSize_[1]));
    }
    dimId_.emplace_back(rankId_ % dimSize_[0]);
    dimId_.emplace_back(rankId_ / dimSize_[0]);
    localId_ = dimId_[axisId_];
    localSize_ = dimSize_[axisId_];
    anotherId_ = dimId_[1 - axisId_]; // 本rank在另一个轴上的Id
    anotherSize_ = dimSize_[1 - axisId_];
    rankSize_ = dimSize_[0] * dimSize_[1];
    HCCL_INFO(
        "[CcuContextAlltoAllVMesh2D] RankId[%u], DimSize: D0[%u]--D1[%u], localId[%u], localSize[%u]", rankId_,
        dimSize_[0], dimSize_[1], localId_, localSize_);

    localAxisSignalName_ = "CcuContextAlltoAllVMesh2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextAlltoAllVMesh2DAxisSync_" + std::to_string(1 - axisId_);
}

void CcuContextAllToAllVMesh2D::InitResources()
{
    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);

    uint32_t transportIdx = 0;
    u32 ckeNum = 2;
    input_ = CreateVariable();

    sendLoopNumRecorder_.resize(localSize_, std::vector<CcuRep::Variable>(anotherSize_));
    recvLoopNumRecorder_.resize(localSize_, std::vector<CcuRep::Variable>(anotherSize_));
    LocSendLoopNumRecorder_.resize(localSize_, std::vector<CcuRep::Variable>(anotherSize_));
    LocRecvLoopNumRecorder_.resize(localSize_, std::vector<CcuRep::Variable>(anotherSize_));
    for (uint32_t peerId = 0; peerId < localSize_; peerId++) {
        isPostFlag_.emplace_back(CreateVariable());
        sendRecorder_.emplace_back(CreateVariable());
        sendRecorder_[peerId] = 0;
        if (peerId == localId_) {
            scratch_.emplace_back(CreateVariable());
            output_.emplace_back(CreateVariable());
            token_.emplace_back(CreateVariable());
            for (uint16_t anotherId = 0; anotherId < anotherSize_; anotherId++) {
                sendLoopNumRecorder_[peerId][anotherId] = CreateVariable();
                recvLoopNumRecorder_[peerId][anotherId] = CreateVariable();
                LocSendLoopNumRecorder_[peerId][anotherId] = CreateVariable();
                LocRecvLoopNumRecorder_[peerId][anotherId] = CreateVariable();
            }
        } else {
            HCCL_INFO(
                "[CcuContextAllToAllVMesh2D]Rank[%u], PeerId[%u], TransportId[%u]", rankId_, peerId, transportIdx);
            scratch_.emplace_back(CreateVariable(*(transports[transportIdx]), CKE_IDX_1));
            output_.emplace_back(CreateVariable(*(transports[transportIdx]), CKE_IDX_2));
            token_.emplace_back(CreateVariable(*(transports[transportIdx]), CKE_IDX_3));
            for (uint16_t anotherId = 0; anotherId < anotherSize_; anotherId++) {
                LocSendLoopNumRecorder_[peerId][anotherId] = CreateVariable();
                LocRecvLoopNumRecorder_[peerId][anotherId] = CreateVariable();
                sendLoopNumRecorder_[peerId][anotherId]
                    = (CreateVariable(*(transports[transportIdx]), CKE_IDX_4 + anotherId * ckeNum));
                recvLoopNumRecorder_[peerId][anotherId]
                    = (CreateVariable(*(transports[transportIdx]), CKE_IDX_4 + anotherId * ckeNum + 1));
            }
            transportIdx++;
        }
    }

    for (uint16_t i = 0; i < localSize_; i++) {
        inputAddrs_.emplace_back(CreateMemory());
        bufferAddrs_.emplace_back(CreateMemory());
        outputAddrs_.emplace_back(CreateMemory());
    }

    for (uint16_t sliceId = 0; sliceId < anotherSize_; sliceId++) {
        firstSignal_.emplace_back(
            CreateMaskSignal()); // 每个对端发anotherSize个分片，localSize个分片共用一个信号，共anotherSize个
        secondSignal_.emplace_back(CreateMaskSignal());
    }

    sendRecvInfo_.resize(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        sendRecvInfo_[rankIdx].sendOffset = CreateVariable();
        sendRecvInfo_[rankIdx].recvOffset = CreateVariable();
        sendRecvInfo_[rankIdx].sendTailSizeA = CreateVariable();
        sendRecvInfo_[rankIdx].sendTailSizeB = CreateVariable();
        sendRecvInfo_[rankIdx].sendTailGoSizeA = CreateGroupOpSize();
        sendRecvInfo_[rankIdx].sendTailGoSizeB = CreateGroupOpSize();
        sendRecvInfo_[rankIdx].sendTailSize = CreateVariable();
        sendRecvInfo_[rankIdx].recvTailSizeA = CreateVariable();
        sendRecvInfo_[rankIdx].recvTailSizeB = CreateVariable();
        sendRecvInfo_[rankIdx].sendLoopNum = CreateVariable();
        sendRecvInfo_[rankIdx].recvLoopNum = CreateVariable();
    }

    return;
}

void CcuContextAllToAllVMesh2D::LoadArgs()
{
    Load(input_);
    Load(output_[localId_]);
    Load(token_[localId_]);
    Load(scratch_[localId_]);

    Load(firstScratchBaseOffset_);
    Load(secondScratchBaseOffset_);
    Load(firstScratchSliceOffset_);
    Load(firstScratchSliceStep_);
    Load(secondScratchSliceOffset_);
    Load(secondScratchSliceStep_);
    Load(xnHalfTransportSize_);
    Load(xnHalfTransportGoSize_);

    xnMaxTransportSize_ = xnHalfTransportSize_;
    xnMaxTransportSize_ += xnHalfTransportSize_;

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        Load(sendRecvInfo_[rankIdx].sendOffset);
        Load(sendRecvInfo_[rankIdx].recvOffset);
        Load(sendRecvInfo_[rankIdx].sendTailSizeA);
        Load(sendRecvInfo_[rankIdx].sendTailSizeB);
        Load(sendRecvInfo_[rankIdx].sendTailGoSizeA);
        Load(sendRecvInfo_[rankIdx].sendTailGoSizeB);
        Load(sendRecvInfo_[rankIdx].sendTailSize);
        Load(sendRecvInfo_[rankIdx].recvTailSizeA);
        Load(sendRecvInfo_[rankIdx].recvTailSizeB);
        Load(sendRecvInfo_[rankIdx].sendLoopNum);
        Load(sendRecvInfo_[rankIdx].recvLoopNum);
    }

    return;
}

void CcuContextAllToAllVMesh2D::ExchangeInfoAndSync()
{
    // 交换信息并做同步，前同步固定用1,2,3号信号
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    CcuRep::Variable tempDst = CreateVariable();
    u32 transportId = 0;
    u32 ckeNum = 2;
    for (u32 id = 0; id < localSize_; id++) {
        if (id == localId_) {
            continue;
        }
        uint32_t dst = CalcDstRank(anotherId_, id);
        tempDst = output_[localId_];
        tempDst += sendRecvInfo_[dst].recvOffset;

        WriteVariableWithSignal(*transports[transportId], scratch_[localId_], CKE_IDX_1, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*transports[transportId], tempDst, CKE_IDX_2, CKE_IDX_2, selfBit);
        WriteVariableWithSignal(*transports[transportId], token_[localId_], CKE_IDX_3, CKE_IDX_3, selfBit);

        for (u32 anotherId = 0; anotherId < anotherSize_; anotherId++) {
            dst = CalcDstRank(anotherId, id);
            WriteVariableWithSignal(
                *transports[transportId], sendRecvInfo_[dst].sendLoopNum, CKE_IDX_4 + anotherId * ckeNum,
                CKE_IDX_4 + anotherId * ckeNum, selfBit);
            WriteVariableWithSignal(
                *transports[transportId], sendRecvInfo_[dst].recvLoopNum, CKE_IDX_4 + anotherId * ckeNum + 1,
                CKE_IDX_4 + anotherId * ckeNum + 1, selfBit);
        }
        transportId++;
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    GroupWait(*transportGroup, CKE_IDX_3, allBit);
    for (u32 anotherId = 0; anotherId < anotherSize_; anotherId++) {
        GroupWait(*transportGroup, CKE_IDX_4 + anotherId * ckeNum, allBit);
        GroupWait(*transportGroup, CKE_IDX_4 + anotherId * ckeNum + 1, allBit);
    }

    return;
}

void CcuContextAllToAllVMesh2D::RankSync(uint32_t signalIndex)
{
    // 与远端做同步
    uint16_t selfBit = 1 << localId_;
    uint16_t waitBit = 0;
    uint16_t transportId = 0;
    for (u32 id = 0; id < localSize_; id++) {
        isPostFlag_[id] = 0;
        if (id == localId_) {
            continue;
        }
        for (uint16_t anotherId = 0; anotherId < anotherSize_; anotherId++) {
            u32 dstRank = CalcDstRank(anotherId, id);
            CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX) { isPostFlag_[id] = 1; }
            CCU_IF(sendRecvInfo_[dstRank].recvLoopNum != UINT64_MAX) { isPostFlag_[id] = 1; }
            if (anotherId == anotherId_) {
                continue;
            }
            CCU_IF(LocSendLoopNumRecorder_[id][anotherId] != UINT64_MAX) { isPostFlag_[id] = 1; }
            CCU_IF(LocRecvLoopNumRecorder_[id][anotherId] != UINT64_MAX) { isPostFlag_[id] = 1; }
        }
        CCU_IF(isPostFlag_[id] == 1) { RemotePost(*transports[transportId], signalIndex, selfBit); }
        transportId++;
    }
    for (u32 id = 0; id < localSize_; id++) {
        if (id == localId_) {
            continue;
        }
        waitBit = 1 << id;
        CCU_IF(isPostFlag_[id] == 1) { GroupWait(*transportGroup, signalIndex, waitBit); }
    }

    return;
}

void CcuContextAllToAllVMesh2D::PostSync()
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh2D::Algorithm transport ptr is null"));
        }
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    return;
}

void CcuContextAllToAllVMesh2D::UpdateLoopRecorder(uint16_t flag)
{
    if (flag == SEND_LOOP_UPDATE_FLAG) {
        for (uint16_t peerId = 0; peerId < localSize_; peerId++) {
            for (uint16_t anotherId = 0; anotherId < anotherSize_; anotherId++) {
                u32 dstRank = CalcDstRank(anotherId, peerId);
                CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX)
                {
                    sendRecvInfo_[dstRank].sendLoopNum += xnConst1_;
                    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX) { completedRankCount_ += xnConst1_; }
                }
                if (anotherId == anotherId_) {
                    CCU_IF(sendRecvInfo_[dstRank].recvLoopNum != UINT64_MAX)
                    {
                        sendRecvInfo_[dstRank].recvLoopNum += xnConst1_;
                        CCU_IF(sendRecvInfo_[dstRank].recvLoopNum == UINT64_MAX) { completedRankCount_ += xnConst1_; }
                    }
                }
                if (anotherId == anotherId_ || peerId == localId_) {
                    continue;
                }
                CCU_IF(LocSendLoopNumRecorder_[peerId][anotherId] != UINT64_MAX)
                {
                    LocSendLoopNumRecorder_[peerId][anotherId] += xnConst1_;
                    CCU_IF(LocSendLoopNumRecorder_[peerId][anotherId] == UINT64_MAX)
                    {
                        completedRankCount_ += xnConst1_;
                    }
                }
            }
        }
    } else if (flag == RECV_LOOP_UPDATE_FLAG) {
        for (uint16_t peerId = 0; peerId < localSize_; peerId++) {
            for (uint16_t anotherId = 0; anotherId < anotherSize_; anotherId++) {
                u32 srcRank = CalcDstRank(anotherId, peerId);
                if (anotherId != anotherId_) {
                    CCU_IF(sendRecvInfo_[srcRank].recvLoopNum != UINT64_MAX)
                    {
                        sendRecvInfo_[srcRank].recvLoopNum += xnConst1_;
                        CCU_IF(sendRecvInfo_[srcRank].recvLoopNum == UINT64_MAX) { completedRankCount_ += xnConst1_; }
                    }
                }
                if (anotherId == anotherId_ || peerId == localId_) {
                    continue;
                }
                CCU_IF(LocRecvLoopNumRecorder_[peerId][anotherId] != UINT64_MAX)
                {
                    LocRecvLoopNumRecorder_[peerId][anotherId] += xnConst1_;
                    CCU_IF(LocRecvLoopNumRecorder_[peerId][anotherId] == UINT64_MAX)
                    {
                        completedRankCount_ += xnConst1_;
                    }
                }
            }
        }
    }

    return;
}

void CcuContextAllToAllVMesh2D::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2; // 2个die
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextAllToAllVMesh2D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    return;
}

uint32_t CcuContextAllToAllVMesh2D::CalcDstRank(uint32_t sliceId, uint32_t peerId) const
{
    uint32_t dstRank;
    if (axisId_ == 0) {
        dstRank = sliceId * localSize_ + peerId;
    } else {
        dstRank = sliceId + anotherSize_ * peerId;
    }
    return dstRank;
}

uint32_t CcuContextAllToAllVMesh2D::CalcTransIdx(uint32_t peerId) const
{
    uint32_t transIdx;
    if (peerId < localId_) {
        transIdx = peerId;
    } else {
        transIdx = peerId - 1;
    }
    return transIdx;
}

void CcuContextAllToAllVMesh2D::DoAll2AllVMultiLoop()
{
    // 需要等待的次数：2 * rankSize_ + (localSize_ - 1) * (anotherSize_ - 1) * 2
    completedRankCount_ = 0;
    xnConst1_ = 1;
    uint64_t targetCount = 2 * rankSize_ + (localSize_ - 1) * (anotherSize_ - 1) * 2;
    CCU_WHILE(completedRankCount_ != targetCount)
    {
        // 第一轮，直连的rank间直接搬运数据。将需要中转的数据搬到中转rank的scratchBuf上
        FirstStep();
        RankSync(CKE_IDX_1);
        UpdateLoopRecorder(SEND_LOOP_UPDATE_FLAG);
        AxisSync(FST_AXIS_ID);

        // 第二轮，从input和buffer中将剩余的本端分片以及待转发分片发给对端；其中给每个对端发1个本端分片，localSize-1个转发分片
        HCCL_INFO("[CcuContextAlltoAllVMesh2D] Algorithm second step begins.");
        RankSync(CKE_IDX_2);
        SecondStep();
        RankSync(CKE_IDX_3);
        UpdateLoopRecorder(RECV_LOOP_UPDATE_FLAG);
        AxisSync(SEC_AXIS_ID);
    }
}

void CcuContextAllToAllVMesh2D::WriteToDstOutput(uint16_t sliceId, uint16_t peerId)
{
    uint32_t dstRank = CalcDstRank(sliceId, peerId);
    uint32_t transIdx = CalcTransIdx(peerId);

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX)
    { // 已经搬完了，仅同步
        LocalPost(firstSignal_[sliceId], (1 << peerId));
    }

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX)
    { // 还没有搬完
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX - 1)
        { // 最后一次搬运
            CCU_IF(sendRecvInfo_[dstRank].sendTailSize == 0) { LocalPost(firstSignal_[sliceId], (1 << peerId)); }
            CCU_IF(sendRecvInfo_[dstRank].sendTailSize != 0)
            {
                Write(
                    *(transports[transIdx]), outputAddrs_[peerId], inputAddrs_[peerId],
                    sendRecvInfo_[dstRank].sendTailSize, firstSignal_[sliceId], (1 << peerId));
            }
        }
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX - 1)
        { // 正常搬运
            Write(
                *(transports[transIdx]), outputAddrs_[peerId], inputAddrs_[peerId], xnMaxTransportSize_,
                firstSignal_[sliceId], (1 << peerId));
            sendRecvInfo_[dstRank].sendOffset += xnMaxTransportSize_;
            sendRecorder_[peerId] += xnMaxTransportSize_;
        }
    }
    return;
}

void CcuContextAllToAllVMesh2D::GroupCopyToDstOutput(uint16_t sliceId, uint16_t peerId)
{
    HCCL_DEBUG("[CcuContextAlltoAllVMesh2D] GroupCopyToDstOutput Start.");
    uint32_t dstRank = CalcDstRank(sliceId, peerId);

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX)
    { // 已经搬完了，仅同步
        LocalPost(firstSignal_[sliceId], (1 << peerId));
    }

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX)
    { // 还没有完成，则继续循环
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX - 1)
        { // 最后一轮循环, 发送尾块数据
            curSendTailSize_
                = (axisId_ == 0) ? sendRecvInfo_[dstRank].sendTailSizeA : sendRecvInfo_[dstRank].sendTailSizeB;
            curSendTailGoSize_
                = (axisId_ == 0) ? sendRecvInfo_[dstRank].sendTailGoSizeA : sendRecvInfo_[dstRank].sendTailGoSizeB;
            if (axisId_ == 1) {
                inputAddrs_[peerId].addr += sendRecvInfo_[dstRank].sendTailSizeA;
                outputAddrs_[peerId].addr += sendRecvInfo_[dstRank].sendTailSizeA;
            }

            CCU_IF(curSendTailSize_ == 0) { LocalPost(firstSignal_[sliceId], (1 << peerId)); }
            CCU_IF(curSendTailSize_ != 0)
            {
                outputAddrs_[peerId].addr += sendRecvInfo_[dstRank].recvOffset;
                GroupCopy(outputAddrs_[peerId], inputAddrs_[peerId], curSendTailGoSize_);
                LocalPost(firstSignal_[sliceId], (1 << peerId));
            }
        }
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX - 1)
        {
            outputAddrs_[peerId].addr += sendRecvInfo_[dstRank].recvOffset;
            if (axisId_ == 1) {
                inputAddrs_[peerId].addr += xnHalfTransportSize_;
                outputAddrs_[peerId].addr += xnHalfTransportSize_;
            }
            GroupCopy(outputAddrs_[peerId], inputAddrs_[peerId], xnHalfTransportGoSize_);
            LocalPost(firstSignal_[sliceId], (1 << peerId));
            sendRecvInfo_[dstRank].sendOffset += xnMaxTransportSize_;
            sendRecorder_[peerId] += xnMaxTransportSize_;
        }
    }
    HCCL_DEBUG("[CcuContextAlltoAllVMesh2D] GroupCopyToDstOutput end.");
}

void CcuContextAllToAllVMesh2D::WriteToDstScratch(uint16_t sliceId, uint16_t peerId)
{
    uint32_t dstRank = CalcDstRank(sliceId, peerId);
    uint32_t transIdx = CalcTransIdx(peerId);

    if (peerId == localId_) {
        LocalPost(firstSignal_[sliceId], (1 << peerId));
    } else {
        if (axisId_ == 0) {
            CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX)
            { // 已经搬完了，仅同步
                LocalPost(firstSignal_[sliceId], (1 << peerId));
            }
            CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX)
            { // 还没有搬完
                CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX - 1)
                { // 最后一次搬运
                    CCU_IF(sendRecvInfo_[dstRank].sendTailSizeA == 0)
                    {
                        LocalPost(firstSignal_[sliceId], (1 << peerId));
                    }
                    CCU_IF(sendRecvInfo_[dstRank].sendTailSizeA != 0)
                    {
                        Write(
                            *(transports[transIdx]), outputAddrs_[peerId], inputAddrs_[peerId],
                            sendRecvInfo_[dstRank].sendTailSizeA, firstSignal_[sliceId], (1 << peerId));
                    }
                }
                CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX - 1)
                { // 正常搬运
                    Write(
                        *(transports[transIdx]), outputAddrs_[peerId], inputAddrs_[peerId], xnHalfTransportSize_,
                        firstSignal_[sliceId], (1 << peerId));
                }
            }
        } else {
            CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX)
            { // 已经搬完了，仅同步
                LocalPost(firstSignal_[sliceId], (1 << peerId));
            }
            CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX)
            { // 还没有搬完
                CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX - 1)
                { // 最后一次搬运
                    CCU_IF(sendRecvInfo_[dstRank].sendTailSizeB == 0)
                    {
                        LocalPost(firstSignal_[sliceId], (1 << peerId));
                    }
                    CCU_IF(sendRecvInfo_[dstRank].sendTailSizeB != 0)
                    {
                        inputAddrs_[peerId].addr += sendRecvInfo_[dstRank].sendTailSizeA;
                        Write(
                            *(transports[transIdx]), outputAddrs_[peerId], inputAddrs_[peerId],
                            sendRecvInfo_[dstRank].sendTailSizeB, firstSignal_[sliceId], (1 << peerId));
                    }
                }
                CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX - 1)
                { // 正常搬运
                    inputAddrs_[peerId].addr += xnHalfTransportSize_;
                    Write(
                        *(transports[transIdx]), outputAddrs_[peerId], inputAddrs_[peerId], xnHalfTransportSize_,
                        firstSignal_[sliceId], (1 << peerId));
                }
            }
        }
        sendRecvInfo_[dstRank].sendOffset += xnMaxTransportSize_;
    }

    return;
}

void CcuContextAllToAllVMesh2D::ReadFromSrc(uint16_t sliceId, uint16_t peerId)
{
    uint32_t srcRank = CalcDstRank(sliceId, peerId);
    uint32_t transIdx = CalcTransIdx(peerId);

    CCU_IF(sendRecvInfo_[srcRank].recvLoopNum == UINT64_MAX) { LocalPost(secondSignal_[sliceId], (1 << peerId)); }
    CCU_IF(sendRecvInfo_[srcRank].recvLoopNum != UINT64_MAX)
    {
        CCU_IF(sendRecvInfo_[srcRank].recvLoopNum == UINT64_MAX - 1)
        { // 最后一次搬运
            if (axisId_ == 0) {
                CCU_IF(sendRecvInfo_[srcRank].recvTailSizeB == 0) { LocalPost(secondSignal_[sliceId], (1 << peerId)); }
                CCU_IF(sendRecvInfo_[srcRank].recvTailSizeB != 0)
                {
                    outputAddrs_[peerId].addr += sendRecvInfo_[srcRank].recvTailSizeA;
                    Read(
                        *(transports[transIdx]), outputAddrs_[peerId], bufferAddrs_[peerId],
                        sendRecvInfo_[srcRank].recvTailSizeB, secondSignal_[sliceId], (1 << peerId));
                }
            } else {
                CCU_IF(sendRecvInfo_[srcRank].recvTailSizeA == 0) { LocalPost(secondSignal_[sliceId], (1 << peerId)); }
                CCU_IF(sendRecvInfo_[srcRank].recvTailSizeA != 0)
                {
                    Read(
                        *(transports[transIdx]), outputAddrs_[peerId], bufferAddrs_[peerId],
                        sendRecvInfo_[srcRank].recvTailSizeA, secondSignal_[sliceId], (1 << peerId));
                }
            }
        }
        CCU_IF(sendRecvInfo_[srcRank].recvLoopNum != UINT64_MAX - 1)
        { // 正常搬运
            if (axisId_ == 0) {
                outputAddrs_[peerId].addr += xnHalfTransportSize_;
                Read(
                    *(transports[transIdx]), outputAddrs_[peerId], bufferAddrs_[peerId], xnHalfTransportSize_,
                    secondSignal_[sliceId], (1 << peerId));
            } else {
                Read(
                    *(transports[transIdx]), outputAddrs_[peerId], bufferAddrs_[peerId], xnHalfTransportSize_,
                    secondSignal_[sliceId], (1 << peerId));
            }
            sendRecvInfo_[srcRank].recvOffset += xnMaxTransportSize_;
        }
    }
    return;
}

void CcuContextAllToAllVMesh2D::FirstStep()
{
    // 统一处理token，访问第i个对端需要使用对应的token
    for (uint16_t peerId = 0; peerId < localSize_; peerId++) {
        inputAddrs_[peerId].token = token_[peerId];
        bufferAddrs_[peerId].token = token_[peerId];
        outputAddrs_[peerId].token = token_[peerId];
    }

    // 统一处理bufferAddrs的初始值
    for (uint16_t peerId = 0; peerId < localSize_; peerId++) {
        bufferAddrs_[peerId].addr = scratch_[peerId];
        bufferAddrs_[peerId].addr += firstScratchBaseOffset_;
        if (peerId < localId_) {
            for (uint16_t i = 1; i < localId_; i++) {
                bufferAddrs_[peerId].addr += firstScratchSliceOffset_;
            }
        } else {
            for (uint16_t i = 0; i < localId_; i++) {
                bufferAddrs_[peerId].addr += firstScratchSliceOffset_;
            }
        }
    }

    for (uint16_t sliceId = 0; sliceId < anotherSize_; sliceId++) { // sliceId等于dstRank在另一个维度上的id
        for (uint32_t peerId = 0; peerId < localSize_; peerId++) {
            u32 dstRank = CalcDstRank(sliceId, peerId);
            if (peerId == localId_ && sliceId == anotherId_) {
                continue;
            }
            if (sliceId == anotherId_) {
                inputAddrs_[peerId].addr = input_;
                inputAddrs_[peerId].addr += sendRecvInfo_[dstRank].sendOffset;
                outputAddrs_[peerId].addr = output_[peerId];
                outputAddrs_[peerId].addr += sendRecorder_[peerId];
                WriteToDstOutput(sliceId, peerId);
            } else {
                inputAddrs_[peerId].addr = input_;
                inputAddrs_[peerId].addr += sendRecvInfo_[dstRank].sendOffset;
                outputAddrs_[peerId].addr = bufferAddrs_[peerId].addr;
                WriteToDstScratch(sliceId, peerId);
                bufferAddrs_[peerId].addr += firstScratchSliceStep_;
            }
        }
    }
    uint32_t dstRankForSelf = CalcDstRank(anotherId_, localId_);
    inputAddrs_[localId_].addr = input_;
    inputAddrs_[localId_].addr += sendRecvInfo_[dstRankForSelf].sendOffset;
    outputAddrs_[localId_].addr = output_[localId_];
    outputAddrs_[localId_].addr += sendRecorder_[localId_];
    GroupCopyToDstOutput(anotherId_, localId_);

    // 检查第一轮的数据是否已发完
    for (uint16_t sliceId = 0; sliceId < anotherSize_; sliceId++) {
        LocalWait(firstSignal_[sliceId], (1 << localSize_) - 1); // 等待第一轮所有分片都发完
    }

    return;
}

void CcuContextAllToAllVMesh2D::SecondStep()
{
    // 统一处理bufferAddrs的初始值
    for (uint16_t peerId = 0; peerId < localSize_; peerId++) {
        bufferAddrs_[peerId].addr = scratch_[peerId];
        bufferAddrs_[peerId].addr += secondScratchBaseOffset_;
        if (peerId < localId_) {
            for (uint16_t i = 1; i < localId_; i++) {
                bufferAddrs_[peerId].addr += secondScratchSliceOffset_;
            }
        } else {
            for (uint16_t i = 0; i < localId_; i++) {
                bufferAddrs_[peerId].addr += secondScratchSliceOffset_;
            }
        }
    }

    // 本端从直连rank的scratchmem上读取数据
    for (uint16_t sliceId = 0; sliceId < anotherSize_; sliceId++) {
        for (uint32_t peerId = 0; peerId < localSize_; peerId++) {
            if (peerId == localId_ || sliceId == anotherId_) { // 直连链路之前已经搬过了
                LocalPost(secondSignal_[sliceId], (1 << peerId));
                continue;
            } else {
                u32 srcRank = CalcDstRank(sliceId, peerId);
                outputAddrs_[peerId].addr = output_[localId_];
                outputAddrs_[peerId].addr += sendRecvInfo_[srcRank].recvOffset;
                ReadFromSrc(sliceId, peerId);
            }
            bufferAddrs_[peerId].addr += secondScratchSliceStep_;
        }
    }

    for (uint16_t sliceId = 0; sliceId < anotherSize_; sliceId++) {
        LocalWait(secondSignal_[sliceId], (1 << localSize_) - 1); // 等待第二轮所有分片都发完
    }

    return;
}

void CcuContextAllToAllVMesh2D::CopyLoopNumRecorder()
{
    for (uint16_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            continue;
        }
        for (uint16_t anotherId = 0; anotherId < anotherSize_; anotherId++) {
            LocSendLoopNumRecorder_[peerId][anotherId] = sendLoopNumRecorder_[peerId][anotherId];
            LocRecvLoopNumRecorder_[peerId][anotherId] = recvLoopNumRecorder_[peerId][anotherId];
        }
    }
}

void CcuContextAllToAllVMesh2D::Algorithm()
{
    // 初始化寄存器资源 & 加载外部输入参数
    HCCL_INFO("[CcuContextAlltoAllVMesh2D] AllgatherMesh1D Algorithm Init Begins.");
    InitResources();
    LoadArgs();

    // 第一轮，X方向发a，Y方向发后b，到对端的块均放在output，要沿X转发的b块放在对端的bufferX，根据转发目的、自身locId两级偏移
    HCCL_INFO("[CcuContextAlltoAllVMesh2D] Algorithm first step begins.");
    ExchangeInfoAndSync();
    PostSync();
    AxisSync(SEC_AXIS_ID);
    CopyLoopNumRecorder();

    DoAll2AllVMultiLoop();
    PostSync();
    AxisSync(FST_AXIS_ID);
    HCCL_INFO("[CcuContextAlltoAllVMesh2D] Algorithm Ends.");
    return;
}

void CcuContextAllToAllVMesh2D::CalculateArgs()
{
    if (axisId_ == 0) {
        firstScratchBaseOffset = 0;
        secondScratchBaseOffset = scratchSliceSize * (localSize_ - 1) * (anotherSize_ - 1);

        firstScratchSliceOffset = scratchSliceSize * (anotherSize_ - 1);
        firstScratchSliceStep = scratchSliceSize;
        secondScratchSliceOffset = scratchSliceSize;
        secondScratchSliceStep = scratchSliceSize * (localSize_ - 1);
    } else {
        firstScratchBaseOffset = scratchSliceSize * (localSize_ - 1) * (anotherSize_ - 1);
        secondScratchBaseOffset = 0;

        firstScratchSliceOffset = scratchSliceSize * (anotherSize_ - 1);
        firstScratchSliceStep = scratchSliceSize;
        secondScratchSliceOffset = scratchSliceSize;
        secondScratchSliceStep = scratchSliceSize * (localSize_ - 1);
    }

    return;
}

std::vector<uint64_t> CcuContextAllToAllVMesh2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllToAllVMesh2D* taskArg = dynamic_cast<const CcuTaskArgAllToAllVMesh2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh2D::taskArg ptr is null"));
    }

    uint64_t inputAddr = taskArg->inputAddr;
    uint64_t outputAddr = taskArg->outputAddr;
    uint64_t scratchAddr = taskArg->scratchAddr;
    uint64_t tokenInfo = taskArg->token;

    scratchSliceSize = std::min(taskArg->scratchSliceSize, UB_MAX_TRANS_SIZE / MESH_2D_NUM); // 最小值
    CalculateArgs();
    auto scratchGoSliceSize = CalGoSize(scratchSliceSize);

    HCCL_INFO(
        "[CcuContextAllToAllVMesh2D][GeneArgs] inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu], "
        "scratchSliceSize[%llu], firstScratchBaseOffset[%llu], secondScratchBaseOffset[%llu], "
        "firstScratchSliceOffset[%llu], firstScratchSliceStep[%llu], secondScratchSliceOffset[%llu], "
        "secondScratchSliceStep[%llu]",
        inputAddr, outputAddr, scratchAddr, scratchSliceSize, firstScratchBaseOffset, secondScratchBaseOffset,
        firstScratchSliceOffset, firstScratchSliceStep, secondScratchSliceOffset, secondScratchSliceStep);

    std::vector<uint64_t> processReturn
        = {inputAddr,
           outputAddr,
           tokenInfo,
           scratchAddr,
           firstScratchBaseOffset,
           secondScratchBaseOffset,
           firstScratchSliceOffset,
           firstScratchSliceStep,
           secondScratchSliceOffset,
           secondScratchSliceStep,
           scratchSliceSize};

    processReturn.insert(processReturn.end(), scratchGoSliceSize.begin(), scratchGoSliceSize.end());

    for (uint16_t i = 0; i < rankSize_; i++) {
        uint64_t perTranSize = scratchSliceSize * MESH_2D_NUM;
        uint64_t sendLoopNum = UINT64_MAX - 1 - taskArg->localSendRecvInfo.sendLength[i] / perTranSize;
        uint64_t recvLoopNum = UINT64_MAX - 1 - taskArg->localSendRecvInfo.recvLength[i] / perTranSize;

        uint64_t sendTailSize = taskArg->localSendRecvInfo.sendLength[i]
                                - taskArg->localSendRecvInfo.sendLength[i] / perTranSize * perTranSize;
        uint64_t recvTailSize = taskArg->localSendRecvInfo.recvLength[i]
                                - taskArg->localSendRecvInfo.recvLength[i] / perTranSize * perTranSize;

        uint64_t sendTailSizeA = sendTailSize / MESH_2D_NUM;
        uint64_t sendTailSizeB = sendTailSize - sendTailSizeA;
        auto sendTailGoSizeA = CalGoSize(sendTailSizeA);
        auto sendTailGoSizeB = CalGoSize(sendTailSizeB);
        uint64_t recvTailSizeA = recvTailSize / MESH_2D_NUM;
        uint64_t recvTailSizeB = recvTailSize - recvTailSizeA;

        uint64_t sendOffset = taskArg->localSendRecvInfo.sendOffset[i];
        uint64_t recvOffset = taskArg->localSendRecvInfo.recvOffset[i];

        processReturn.push_back(sendOffset);
        processReturn.push_back(recvOffset);
        processReturn.push_back(sendTailSizeA);
        processReturn.push_back(sendTailSizeB);
        processReturn.insert(processReturn.end(), sendTailGoSizeA.begin(), sendTailGoSizeA.end());
        processReturn.insert(processReturn.end(), sendTailGoSizeB.begin(), sendTailGoSizeB.end());
        processReturn.push_back(sendTailSize);
        processReturn.push_back(recvTailSizeA);
        processReturn.push_back(recvTailSizeB);
        processReturn.push_back(sendLoopNum);
        processReturn.push_back(recvLoopNum);
        HCCL_INFO(
            "[CcuContextAllToAllVMesh2D][sliceInfo] curRankIdx[%u], dstrankIdx[%u]: sendOffset[%llu], "
            "recvOffset[%llu], sendTailSizeA[%llu], sendTailSizeB[%llu], recvTailSizeA[%llu], recvTailSizeB[%llu],"
            "sendLoopNum[%llu], recvLoopNum[%llu]",
            rankId_, i, sendOffset, recvOffset, sendTailSizeA, sendTailSizeB, recvTailSizeA, recvTailSizeB, sendLoopNum,
            recvLoopNum);
    }

    return processReturn;
}

} // namespace Hccl
