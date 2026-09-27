/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_to_all_v_mesh2die.h"
#include "ccu_instruction_all_to_all_v_mesh2die.h"

namespace Hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

CcuContextAllToAllVMesh2Die::CcuContextAllToAllVMesh2Die(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    if (transports.empty()) {
        THROW<InvalidParamsException>(StringFormat("CcuContextAllToAllVMesh2Die transports is empty"));
    }

    const CcuCtxArgAllToAllVMesh2Die* ctxArg = dynamic_cast<const CcuCtxArgAllToAllVMesh2Die*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh2Die::ctxArg ptr is null"));
    }

    auto dimSize = ctxArg->dimSize;
    if (dimSize.size() != 1) { // 2Die场景dimSize为1
        THROW<InvalidParamsException>(
            StringFormat("CcuContextAllToAllVMesh2Die::dimSize[%u] is invalid", dimSize.size()));
    }

    rankSize_ = dimSize[0];
    if (rankSize_ <= 1 || rankSize_ % RANK_EVEN != 0) {
        THROW<InvalidParamsException>(StringFormat("CcuContextAllToAllVMesh2Die::rankSize[%u] is invalid", rankSize_));
    }

    rankId_ = ctxArg->rankId;
    withMyRank_ = ctxArg->withMyRank;
    rankGroup_ = ctxArg->rankGroup;

    localSize_ = transports.size() + 1;
    localId_ = localSize_ - 1; // 本rank所在DIE的编号，固定放在末尾

    peerSize_ = transports.size() + (withMyRank_ ? 1 : 0);
    logicId_ = rankId_ % peerSize_;

    selfBit_ = 1 << logicId_;
    allBit_ = ((1 << peerSize_) - 1) & (~(withMyRank_ ? selfBit_ : 0));

    HCCL_INFO(
        "[CcuContextAllToAllVMesh2Die] RankId[%u], rankSize[%llu], localSize[%u], peerSize[%u], logicId[%u], "
        "withMyRank[%u]",
        rankId_, rankSize_, localSize_, peerSize_, logicId_, withMyRank_);
}

void CcuContextAllToAllVMesh2Die::InitResources()
{
    locSignal_ = CreateMaskSignal();

    input_ = CreateVariable();

    for (uint32_t peerId = 0; peerId < transports.size(); peerId++) {
        HCCL_DEBUG("[CcuContextAllToAllVMesh2Die]RankId[%u], PeerId[%u]", rankId_, peerId);
        output_.emplace_back(CreateVariable(*(transports[peerId]), CKE_IDX_1));
        token_.emplace_back(CreateVariable(*(transports[peerId]), CKE_IDX_2));
    }
    // 本rank固定放在末尾
    output_.emplace_back(CreateVariable());
    token_.emplace_back(CreateVariable());

    xnMaxTransportSize_ = CreateVariable();
    xnMaxTransportGoSize_ = CreateGroupOpSize();

    xnMaxTransportSize_ = MAX_TRANSPORT_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(MAX_TRANSPORT_SIZE);
    xnMaxTransportGoSize_.addrOffset = xnMaxTransportGoSize[GO_ADDR_OFFSET_IDX];
    xnMaxTransportGoSize_.loopParam = xnMaxTransportGoSize[GO_LOOP_PARAM_IDX];
    xnMaxTransportGoSize_.parallelParam = xnMaxTransportGoSize[GO_PARALLEL_PARAM_IDX];
    xnMaxTransportGoSize_.residual = xnMaxTransportGoSize[GO_RESIDUAL_IDX];

    sendRecvInfo_.resize(localSize_);
    for (uint64_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        sendRecvInfo_[rankIdx].sendOffset = CreateVariable();
        sendRecvInfo_[rankIdx].recvOffset = CreateVariable();
        sendRecvInfo_[rankIdx].sendTailSize = CreateVariable();
        sendRecvInfo_[rankIdx].sendTailGoSize = CreateGroupOpSize();
        sendRecvInfo_[rankIdx].sendLoopNum = CreateVariable();
    }

    for (uint16_t i = 0; i < localSize_; i++) {
        src_.emplace_back(CreateMemory());
        dst_.emplace_back(CreateMemory());
    }

    curSendTailSize_ = CreateVariable();
    curSendTailGoSize_ = CreateGroupOpSize();

    xnConst1_ = CreateVariable();
    completedRankCount_ = CreateVariable();
}

void CcuContextAllToAllVMesh2Die::LoadArgs()
{
    Load(input_);
    Load(output_[localId_]);
    Load(token_[localId_]);

    for (uint64_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        Load(sendRecvInfo_[rankIdx].sendOffset);
        Load(sendRecvInfo_[rankIdx].recvOffset);
        Load(sendRecvInfo_[rankIdx].sendTailSize);
        Load(sendRecvInfo_[rankIdx].sendTailGoSize);
        Load(sendRecvInfo_[rankIdx].sendLoopNum);
    }
}

void CcuContextAllToAllVMesh2Die::ExchangeInfoAndSync()
{
    // 交换信息并做同步，前同步固定用1,2,3号信号
    CcuRep::Variable tempDst = CreateVariable();
    for (u32 peerId = 0; peerId < transports.size(); peerId++) {
        uint32_t dst = CalcDstRank(peerId);
        tempDst = output_[localId_];
        tempDst += sendRecvInfo_[dst].recvOffset;

        WriteVariableWithSignal(*transports[peerId], tempDst, CKE_IDX_1, CKE_IDX_1, selfBit_);
        WriteVariableWithSignal(*transports[peerId], token_[localId_], CKE_IDX_2, CKE_IDX_2, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
}

void CcuContextAllToAllVMesh2Die::PostSync()
{
    for (const auto& t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh2Die::PostSync transport ptr is null"));
        }
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit_);
}

uint32_t CcuContextAllToAllVMesh2Die::CalcDstRank(uint32_t peerId) const { return peerId; }

uint32_t CcuContextAllToAllVMesh2Die::CalcTransIdx(uint32_t peerId) const { return peerId; }

void CcuContextAllToAllVMesh2Die::DoAll2AllVMultiLoop()
{
    completedRankCount_ = 0;
    xnConst1_ = 1;
    CCU_WHILE(completedRankCount_ != peerSize_)
    {
        HCCL_DEBUG("[CcuContextAllToAllVMesh2Die] Algorithm loops[%u].", peerSize_);
        LoopStep();
    }
}

void CcuContextAllToAllVMesh2Die::WriteToDstOutput(uint32_t peerId)
{
    uint32_t dstRank = CalcDstRank(peerId);
    uint32_t transIdx = CalcTransIdx(peerId);

    HCCL_DEBUG(
        "[CcuContextAllToAllVMesh2Die] WriteToDstOutput[%u] Start. RankId[%u] dstRank[%u] transIdx[%u]", peerId,
        rankId_, dstRank, transIdx);

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX) // 已经搬完了，仅同步
    {
        LocalPost(locSignal_, (1 << peerId));
    }

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX) // 还没有搬完
    {
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX - 1) // 最后一次搬运, 发送尾块数据
        {
            curSendTailSize_ = sendRecvInfo_[dstRank].sendTailSize;
            CCU_IF(curSendTailSize_ == 0) { LocalPost(locSignal_, (1 << peerId)); }
            CCU_IF(curSendTailSize_ != 0)
            {
                Write(*(transports[transIdx]), dst_[peerId], src_[peerId], curSendTailSize_, locSignal_, (1 << peerId));
            }
            completedRankCount_ += xnConst1_;
        }
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX - 1) // 正常搬运
        {
            Write(*(transports[transIdx]), dst_[peerId], src_[peerId], xnMaxTransportSize_, locSignal_, (1 << peerId));
            dst_[peerId].addr += xnMaxTransportSize_;
            src_[peerId].addr += xnMaxTransportSize_;
        }
        sendRecvInfo_[dstRank].sendLoopNum += xnConst1_;
    }
    HCCL_DEBUG("[CcuContextAllToAllVMesh2Die] WriteToDstOutput end.");
}

void CcuContextAllToAllVMesh2Die::GroupCopyToDstOutput(uint32_t peerId)
{
    uint32_t dstRank = CalcDstRank(peerId);

    HCCL_DEBUG(
        "[CcuContextAllToAllVMesh2Die] GroupCopyToDstOutput[%u] Start. RankId[%u] dstRank[%u]", peerId, rankId_,
        dstRank);

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX) // 已经搬完了，仅同步
    {
        LocalPost(locSignal_, (1 << peerId));
    }

    CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX) // 还没有搬完
    {
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum == UINT64_MAX - 1) // 最后一次搬运, 发送尾块数据
        {
            curSendTailSize_ = sendRecvInfo_[dstRank].sendTailSize;
            curSendTailGoSize_ = sendRecvInfo_[dstRank].sendTailGoSize;
            CCU_IF(curSendTailSize_ == 0) { LocalPost(locSignal_, (1 << peerId)); }
            CCU_IF(curSendTailSize_ != 0)
            {
                GroupCopy(dst_[peerId], src_[peerId], curSendTailGoSize_);
                LocalPost(locSignal_, (1 << peerId));
            }
            completedRankCount_ += xnConst1_;
        }
        CCU_IF(sendRecvInfo_[dstRank].sendLoopNum != UINT64_MAX - 1) // 正常搬运
        {
            GroupCopy(dst_[peerId], src_[peerId], xnMaxTransportGoSize_);
            dst_[peerId].addr += xnMaxTransportSize_;
            src_[peerId].addr += xnMaxTransportSize_;
            LocalPost(locSignal_, (1 << peerId));
        }
        sendRecvInfo_[dstRank].sendLoopNum += xnConst1_;
    }
    HCCL_DEBUG("[CcuContextAllToAllVMesh2Die] GroupCopyToDstOutput end.");
}

void CcuContextAllToAllVMesh2Die::CalcGroupSrcDst()
{
    for (uint32_t peerId = 0; peerId < transports.size(); peerId++) {
        const u32 dstRank = CalcDstRank(peerId);

        src_[peerId].addr = input_;
        src_[peerId].addr += sendRecvInfo_[dstRank].sendOffset;
        src_[peerId].token = token_[peerId];

        dst_[peerId].addr = output_[peerId]; // recvOffset在前同步时已经计算
        dst_[peerId].token = token_[peerId];
    }

    if (withMyRank_) {
        src_[localId_].addr = input_;
        src_[localId_].addr += sendRecvInfo_[localId_].sendOffset;
        src_[localId_].token = token_[localId_];
        dst_[localId_].addr = output_[localId_];
        dst_[localId_].addr += sendRecvInfo_[localId_].recvOffset;
        dst_[localId_].token = token_[localId_];
    }
}

void CcuContextAllToAllVMesh2Die::LoopStep()
{
    for (uint32_t peerId = 0; peerId < transports.size(); peerId++) {
        WriteToDstOutput(peerId);
    }

    if (withMyRank_) {
        GroupCopyToDstOutput(localId_);
    }

    LocalWait(locSignal_, (1 << peerSize_) - 1);
}

void CcuContextAllToAllVMesh2Die::Algorithm()
{
    // 初始化寄存器资源 & 加载外部输入参数
    HCCL_INFO("[CcuContextAllToAllVMesh2Die] Algorithm Init Begins.");
    InitResources();
    LoadArgs();

    HCCL_INFO("[CcuContextAllToAllVMesh2Die] Algorithm begins.");

    // 框架已经默认做了前后轴同步，算法不需要再重复做
    ExchangeInfoAndSync();

    CalcGroupSrcDst();
    DoAll2AllVMultiLoop();

    PostSync();

    HCCL_INFO("[CcuContextAllToAllVMesh2Die] Algorithm Ends.");
}

std::vector<uint64_t> CcuContextAllToAllVMesh2Die::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllToAllVMesh2Die* taskArg = dynamic_cast<const CcuTaskArgAllToAllVMesh2Die*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh2Die::taskArg ptr is null"));
    }

    uint64_t inputAddr = taskArg->inputAddr;
    uint64_t outputAddr = taskArg->outputAddr;
    uint64_t tokenInfo = taskArg->token;

    std::vector<uint64_t> taskParams = {inputAddr, outputAddr, tokenInfo}; // 不需要ScratchMem

    for (auto peerId : rankGroup_) {
        const uint64_t floorLoopNum = taskArg->localSendRecvInfo.sendLength[peerId] / MAX_TRANSPORT_SIZE;
        uint64_t sendLoopNum = UINT64_MAX - 1 - floorLoopNum;
        uint64_t sendTailSize = taskArg->localSendRecvInfo.sendLength[peerId] - floorLoopNum * MAX_TRANSPORT_SIZE;
        auto sendTailGoSize = CalGoSize(sendTailSize);
        uint64_t sendOffset = taskArg->localSendRecvInfo.sendOffset[peerId];
        uint64_t recvOffset = taskArg->localSendRecvInfo.recvOffset[peerId];
        taskParams.push_back(sendOffset);
        taskParams.push_back(recvOffset);
        taskParams.push_back(sendTailSize);
        taskParams.insert(taskParams.cend(), sendTailGoSize.cbegin(), sendTailGoSize.cend());
        taskParams.push_back(sendLoopNum);
        HCCL_DEBUG(
            "[CcuContextAllToAllVMesh2Die][sliceInfo] RankId[%u], dstRank[%d]: sendOffset[%llu], "
            "recvOffset[%llu], sendLength[%llu], sendTailSize[%llu], sendLoopNum[%llu]",
            rankId_, peerId, sendOffset, recvOffset, taskArg->localSendRecvInfo.sendLength[peerId], sendTailSize,
            sendLoopNum);
    }

    HCCL_DEBUG(
        "[CcuContextAllToAllVMesh2Die][GeneArgs] RankId[%u], inputAddr[%#llx], outputAddr[%#llx], "
        "xnMaxTransportSize[%llu], args[%zu]",
        rankId_, inputAddr, outputAddr, MAX_TRANSPORT_SIZE, taskParams.size());

    return taskParams;
}

} // namespace Hccl
