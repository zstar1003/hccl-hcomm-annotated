/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_for_310p.h"
#include "alg_template_register.h"

namespace hccl {
AlltoAllVFor310P::AlltoAllVFor310P(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AlltoAllVFor310P::~AlltoAllVFor310P() {}

HcclResult AlltoAllVFor310P::Prepare(
    DeviceMem& userInput, DeviceMem& userOutput, DeviceMem& cclInMem, DeviceMem& cclOutMem,
    const std::vector<std::shared_ptr<LocalNotify>>& signalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>>& signalSubToMain, Stream& mainStream,
    std::vector<Stream>& subStreams, const std::vector<LINK>& links, u32 userRank, u32 userRankSize,
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo)
{
    mainStream_ = mainStream;
    subStream_ = subStreams;
    links_ = links;
    userRank_ = userRank;
    userRankSize_ = userRankSize;
    CHK_PRT_RET(userRankSize_ == 0, HCCL_ERROR("[AlltoAllVFor310P][Prepare]userRankSize_ is zero."), HCCL_E_PARA);
    allMeshAggregationSendRecvInfoPtr_ = &allMeshAggregationSendRecvInfo;

    userInput_ = userInput;
    userOutput_ = userOutput;
    cclInMem_ = cclInMem;
    cclOutMem_ = cclOutMem;
    memList_.push_back(cclInMem_);   // Id 0
    memList_.push_back(cclOutMem_);  // Id 1
    memList_.push_back(userOutput_); // Id 2

    if (userRank_ % COMPUTE_CONST == 0) {
        mainRank_ = true;
        myMinor_ = userRank_ + 1;
        if (subStream_.size() != COMPUTE_CONST) {
            HCCL_ERROR("[AlltoAllVFor310P][Prepare]main subStream.size[%zu] != 2", subStream_.size());
            return HCCL_E_INTERNAL;
        }
    } else {
        minorRank_ = true;
        myMain_ = (userRank_ - 1 + userRankSize_) % userRankSize_;
        if (subStream_.size() != 1) {
            HCCL_ERROR("[AlltoAllVFor310P][Prepare]minor subStream.size[%zu] != 1", subStream_.size());
            return HCCL_E_INTERNAL;
        }
    }

    CHK_PRT_RET(
        signalMainToSub.size() != subStream_.size() || signalSubToMain.size() != subStream_.size(),
        HCCL_ERROR(
            "[AlltoAllVFor310P][Prepare] Signal size not equal to subStream size, signalMainToSub.size[%llu],"
            "signalSubToMain.size[%llu], subStream_.size[%llu]",
            signalMainToSub.size(), signalSubToMain.size(), subStream_.size()),
        HCCL_E_INTERNAL);

    HCCL_DEBUG(
        "userRank[%u], subStream.size[%zu], signalMainToSub.size[%zu], signalSubToMain.size[%zu]", userRank_,
        subStream_.size(), signalMainToSub.size(), signalSubToMain.size());

    for (u32 index = 0; index < signalMainToSub.size(); index++) {
        CHK_PTR_NULL(signalMainToSub[index]);
        signalMainToSub_.push_back(signalMainToSub[index]);
    }

    for (u32 index = 0; index < signalSubToMain.size(); index++) {
        CHK_PTR_NULL(signalSubToMain[index]);
        signalSubToMain_.push_back(signalSubToMain[index]);
    }

    cclBlockSize_ = ((cclInMem.size() / COMPUTE_CONST) / ALIGN_CONST) * ALIGN_CONST; // 除以128取整
    maxSizePerLoop_ = cclBlockSize_ - ALIGN_CONST;

    CHK_PRT_RET(cclBlockSize_ == 0, HCCL_ERROR("[AlltoAllVFor310P][Prepare]DataBlockSize_is zero."), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

std::string AlltoAllVFor310P::GetStreamIndexString()
{
    std::string res = "";
    for (u32 streamIndex = 0; streamIndex < subStream_.size(); streamIndex++) {
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

HcclResult AlltoAllVFor310P::WaitSubStreamFinish()
{
    // 从流通知主流做完
    for (u32 streamIndex = 0; streamIndex < subStream_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            subStream_[streamIndex], dispatcher_, signalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, signalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[AlltoAllVFor310P][WaitSubStreamFinish] userRank [%u] main stream wait stream [%s]", userRank_,
        GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStream_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, signalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            subStream_[streamIndex], dispatcher_, signalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[AlltoAllVFor310P][NotifySubStreamStart] userRank [%u] main stream notify sdma stream [%s]", userRank_,
        GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::CalcSendInfo(
    const u32 srcDataRank, const u32 dstDataRank, const u32 times, const u64 subStepLen, SendMemBlock& sendInfo) const
{
    const std::vector<u64>& sendLength = (*allMeshAggregationSendRecvInfoPtr_)[srcDataRank].sendLength;
    const std::vector<u64>& sendOffset = (*allMeshAggregationSendRecvInfoPtr_)[srcDataRank].sendOffset;
    const std::vector<u64>& recvOffset = (*allMeshAggregationSendRecvInfoPtr_)[dstDataRank].recvOffset;

    u32 sendLen = 0;
    if (sendLength[dstDataRank] > times * maxSizePerLoop_) {
        u32 leftLen = sendLength[dstDataRank] - times * maxSizePerLoop_;
        sendLen = leftLen > subStepLen ? subStepLen : leftLen;
        sendInfo.userInOffset = sendOffset[dstDataRank] + times * maxSizePerLoop_;
    } else {
        sendInfo.userInOffset
            = sendOffset[dstDataRank] + sendLength[dstDataRank]; // 已经发完了，offset变成最大值，sendLen为0
    }
    sendInfo.dstRank = dstDataRank;
    sendInfo.sendLen = sendLen;
    sendInfo.cclDstOffset = (recvOffset[srcDataRank] + times * maxSizePerLoop_) % ALIGN_CONST;
    HCCL_DEBUG(
        "[AlltoAllVFor310P] [CalcSendInfo]srcDataRank[%u], dstDataRank[%u], times[%u], subStepLen[%llu], "
        "sendLen[%llu], userInOffset[%llu], cclDstOffset[%llu]",
        srcDataRank, dstDataRank, times, subStepLen, sendInfo.sendLen, sendInfo.userInOffset, sendInfo.cclDstOffset);

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::CalcRecvInfo(
    const u32 srcDataRank, const u32 dstDataRank, const u32 times, const u64 subStepLen, RecvMemBlock& recvInfo) const
{
    const std::vector<u64>& recvLength = (*allMeshAggregationSendRecvInfoPtr_)[dstDataRank].recvLength;
    const std::vector<u64>& recvOffset = (*allMeshAggregationSendRecvInfoPtr_)[dstDataRank].recvOffset;

    u32 recvLen = 0;
    if (recvLength[srcDataRank] > times * maxSizePerLoop_) {
        u32 leftLen = recvLength[srcDataRank] - times * maxSizePerLoop_;
        recvLen = leftLen > subStepLen ? subStepLen : leftLen;
        recvInfo.userOutOffset = recvOffset[srcDataRank] + times * maxSizePerLoop_;
    } else {
        recvInfo.userOutOffset
            = recvOffset[srcDataRank] + recvLength[srcDataRank]; // 已经收完了，offset变成最大值，recvLen为0
    }
    recvInfo.srcRank = srcDataRank;
    recvInfo.recvLen = recvLen;
    recvInfo.cclSrcOffset = (recvOffset[srcDataRank] + times * maxSizePerLoop_) % ALIGN_CONST;
    HCCL_DEBUG(
        "[AlltoAllVFor310P] [CalcRecvInfo]dstDataRank[%u], srcDataRank[%u], times[%u], subStepLen[%llu], "
        "recvLen[%llu], userOutOffset[%llu], cclSrcOffset[%llu]",
        dstDataRank, srcDataRank, times, subStepLen, recvInfo.recvLen, recvInfo.userOutOffset, recvInfo.cclSrcOffset);

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::MainFirstLocalCopy(const u32 times, const u32 roundIdx, const u64 subStepLen)
{
    if (roundIdx == 0) {
        // 给本次卡的数据
        SendMemBlock sendData0;
        CHK_RET(CalcSendInfo(userRank_, myMinor_, times, subStepLen, sendData0));
        DeviceMem src0 = userInput_.range(sendData0.userInOffset, sendData0.sendLen);
        DeviceMem dst0 = cclInMem_.range(sendData0.cclDstOffset, sendData0.sendLen);
        HCCL_DEBUG(
            "[AlltoAllVFor310P] rank[%u] localCopy to cclIn, src.Offset [%llu], dst.Offset[%llu], len[%llu]", userRank_,
            sendData0.userInOffset, sendData0.cclDstOffset, sendData0.sendLen);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst0, src0, mainStream_));
    }

    // 给右次卡的数据
    SendMemBlock sendData1;
    CHK_RET(CalcSendInfo(userRank_, rightMinor_, times, subStepLen, sendData1));
    DeviceMem src1 = userInput_.range(sendData1.userInOffset, sendData1.sendLen);
    DeviceMem dst1 = cclInMem_.range(cclBlockSize_ + sendData1.cclDstOffset, sendData1.sendLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] localCopy to cclIn, src.Offset [%llu], dst.Offset[%llu], len[%llu]", userRank_,
        sendData1.userInOffset, cclBlockSize_ + sendData1.cclDstOffset, sendData1.sendLen);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst1, src1, mainStream_));

    HCCL_DEBUG("[AlltoAllVFor310P] MainFirstLocalCopy finish.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::MinorFirstLocalCopy(const u32 times, const u32 roundIdx, const u64 subStepLen)
{
    (void)roundIdx;
    // 给右次卡的数据
    SendMemBlock sendData;
    CHK_RET(CalcSendInfo(userRank_, rightMinor_, times, subStepLen, sendData));
    DeviceMem src = userInput_.range(sendData.userInOffset, sendData.sendLen);
    DeviceMem dst = cclInMem_.range(sendData.cclDstOffset, sendData.sendLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] localCopy to cclIn, src.Offset [%llu], dst.Offset[%llu], len[%llu]", userRank_,
        sendData.userInOffset, sendData.cclDstOffset, sendData.sendLen);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::RunAlltoAllVFor310P()
{
    u32 roundNum = userRankSize_ == DUO_RANK_NUM ? 1 : DUO_RANK_NUM - 1;
    u64 remainLen = CalcMaxSendLength();
    u64 subStepLen = std::min(remainLen, maxSizePerLoop_);
    u32 totalTimes = (remainLen + subStepLen - 1) / subStepLen;
    HCCL_INFO(
        "[AlltoAllVFor310P] roundNum[%u], maxLength[%llu], subStepLen[%llu], totalTimes[%u]", roundNum, remainLen,
        subStepLen, totalTimes);
    for (u32 times = 0; times < totalTimes && remainLen > 0; times++) {
        subStepLen = std::min(remainLen, maxSizePerLoop_);
        for (u32 roundIdx = 0; roundIdx < roundNum; roundIdx++) {
            SetNeighborRanks(roundIdx);
            for (u32 stepIdx = 0; stepIdx < STEP_NUM; stepIdx++) {
                CHK_RET(UpdateSendRecvRankInfo(roundIdx, stepIdx));
                CHK_RET(RunSendRecvBuffer(times, roundIdx, stepIdx, subStepLen));
            }
            HCCL_INFO("[AlltoAllVFor310P] Round[%u] finish.", roundIdx);
        }
        remainLen = remainLen - subStepLen;
        HCCL_INFO("[AlltoAllVFor310P] Times[%u] finish.", times);
    }
    return HCCL_SUCCESS;
}

void AlltoAllVFor310P::SetNeighborRanks(const u32 roundIdx)
{
    if (userRank_ % COMPUTE_CONST == 0) {
        rightMain_ = ((userRank_ + COMPUTE_CONST * (roundIdx + 1))) % userRankSize_;
        rightMinor_ = ((userRank_ + MAX_RANK_GAP * (roundIdx + 1))) % userRankSize_;
        leftMain_ = ((userRank_ - COMPUTE_CONST * (roundIdx + 1)) + userRankSize_) % userRankSize_;
        leftMinor_ = ((userRank_ - 1 * (roundIdx + 1)) + userRankSize_) % userRankSize_;
    } else {
        rightMain_ = ((userRank_ + 1 * (roundIdx + 1))) % userRankSize_;
        rightMinor_ = ((userRank_ + COMPUTE_CONST * (roundIdx + 1))) % userRankSize_;
        leftMain_ = ((userRank_ - MAX_RANK_GAP * (roundIdx + 1)) + userRankSize_) % userRankSize_;
        leftMinor_ = ((userRank_ - COMPUTE_CONST * (roundIdx + 1)) + userRankSize_) % userRankSize_;
    }
    HCCL_DEBUG("[AlltoAllVFor310P] SetNeighborRanks finish.");
}

HcclResult AlltoAllVFor310P::UpdateSendRecvRankInfo(const u32 roundIdx, const u32 stepIdx)
{
    sendRecvRankInfo_.clear();
    if (stepIdx <= THIRD_STEP && mainRank_) {
        sendRecvRankInfo_.push_back(std::make_pair(rightMain_, leftMain_)); // first send, second recv
        sendRecvRankInfo_.push_back(std::make_pair(myMinor_, myMinor_));
    } else if (stepIdx <= THIRD_STEP && minorRank_) {
        sendRecvRankInfo_.push_back(std::make_pair(myMain_, myMain_));
    } else if (stepIdx > THIRD_STEP && mainRank_) {
        sendRecvRankInfo_.push_back(std::make_pair(rightMain_, leftMain_));
    }
    HCCL_DEBUG(
        "[AlltoAllVFor310P][UpdateSendRecvRankInfo] update send/recv rank finished, roundIdx[%u], stepIdx[%u]",
        roundIdx, stepIdx);
    return HCCL_SUCCESS;
}

HcclResult
AlltoAllVFor310P::RunMainCommonSteps(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen)
{
    UpdateMainStepMemInfo(roundIdx, stepIdx);
    if (stepIdx == THIRD_STEP) {
        // 给其他主卡的数据拷到cclOut
        SendMemBlock sendData;
        CHK_RET(CalcSendInfo(userRank_, rightMain_, times, subStepLen, sendData));
        DeviceMem src = userInput_.range(sendData.userInOffset, sendData.sendLen);
        DeviceMem dst = cclOutMem_.range(sendData.cclDstOffset, sendData.sendLen);
        HCCL_DEBUG(
            "[AlltoAllVFor310P] rank[%u] localCopy to cclOut, src.Offset [%llu], dst.Offset[%llu], len[%llu]",
            userRank_, sendData.userInOffset, sendData.cclDstOffset, sendData.sendLen);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    }

    RecvMemBlock recvData0;
    CHK_RET(CalcRecvInfo(mainStepInfo_.readMinor.first, mainStepInfo_.readMinor.second, times, subStepLen, recvData0));
    u64 dstOffset = 0;
    if (stepIdx == THIRD_STEP) {
        dstOffset = recvData0.userOutOffset;
    } else {
        dstOffset = cclBlockSize_ + recvData0.cclSrcOffset;
    }
    const LINK& readMinorTransport = links_[sendRecvRankInfo_[1].second];
    CHK_PTR_NULL(readMinorTransport);
    void* remMemPtr0 = nullptr;
    CHK_RET(readMinorTransport->GetRemoteMem(mainStepInfo_.srcMemType, &remMemPtr0));
    DeviceMem remoteMem0 = DeviceMem::create(static_cast<u8*>(remMemPtr0), memList_[mainStepInfo_.srcBuffId].size());
    DeviceMem src0 = remoteMem0.range(recvData0.cclSrcOffset, recvData0.recvLen);
    DeviceMem dst0 = memList_[mainStepInfo_.dstBuffId].range(dstOffset, recvData0.recvLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] Memcpy to rank[%u], src.Offset [%llu], dst.Offset[%llu], len[%llu]",
        sendRecvRankInfo_[1].second, userRank_, recvData0.cclSrcOffset, dstOffset, recvData0.recvLen);
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, dst0, src0, subStream_[1], readMinorTransport->GetRemoteRank(),
        readMinorTransport->GetLinkType()));

    RecvMemBlock recvData1;
    CHK_RET(CalcRecvInfo(mainStepInfo_.readMain.first, mainStepInfo_.readMain.second, times, subStepLen, recvData1));
    if (stepIdx == THIRD_STEP) {
        dstOffset = recvData1.userOutOffset;
    } else {
        dstOffset = recvData1.cclSrcOffset;
    }
    const LINK& readMainTransport = links_[sendRecvRankInfo_[0].second];
    CHK_PTR_NULL(readMainTransport);
    void* remMemPtr1 = nullptr;
    CHK_RET(readMainTransport->GetRemoteMem(mainStepInfo_.srcMemType, &remMemPtr1));
    DeviceMem remoteMem1 = DeviceMem::create(static_cast<u8*>(remMemPtr1), memList_[mainStepInfo_.srcBuffId].size());
    DeviceMem src1 = remoteMem1.range(cclBlockSize_ + recvData1.cclSrcOffset, recvData1.recvLen);
    DeviceMem dst1 = memList_[mainStepInfo_.dstBuffId].range(dstOffset, recvData1.recvLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] Memcpy to rank[%u], src.Offset [%llu], dst.Offset[%llu], len[%llu]",
        sendRecvRankInfo_[0].second, userRank_, cclBlockSize_ + recvData1.cclSrcOffset, dstOffset, recvData1.recvLen);
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, dst1, src1, subStream_[0], readMainTransport->GetRemoteRank(), readMainTransport->GetLinkType()));
    HCCL_DEBUG("[AlltoAllVFor310P] RunMainStep %u finish.", stepIdx);
    return HCCL_SUCCESS;
}

void AlltoAllVFor310P::UpdateMainStepMemInfo(const u32 roundIdx, const u32 stepIdx)
{
    (void)roundIdx;
    if (stepIdx % COMPUTE_CONST == 1) {
        mainStepInfo_.srcBuffId = 0;
        mainStepInfo_.srcMemType = UserMemType::INPUT_MEM;
    } else {
        mainStepInfo_.srcBuffId = 1;
        mainStepInfo_.srcMemType = UserMemType::OUTPUT_MEM;
    }
    if (stepIdx == 1) {
        mainStepInfo_.dstBuffId = 1;
        mainStepInfo_.readMain = std::make_pair(leftMain_, myMinor_);
        mainStepInfo_.readMinor = std::make_pair(myMinor_, rightMinor_);
    } else if (stepIdx == COMPUTE_CONST) {
        mainStepInfo_.dstBuffId = 0;
        mainStepInfo_.readMain = std::make_pair(leftMinor_, myMinor_);
        mainStepInfo_.readMinor = std::make_pair(myMinor_, rightMain_);
    } else {
        mainStepInfo_.dstBuffId = COMPUTE_CONST;
        mainStepInfo_.readMain = std::make_pair(leftMinor_, userRank_);
        mainStepInfo_.readMinor = std::make_pair(myMinor_, userRank_);
    }
}

HcclResult AlltoAllVFor310P::RunMainStep4(const u32 times, const u64 subStepLen)
{
    // 拷本卡的数据
    SendMemBlock sendDataLocal;
    CHK_RET(CalcSendInfo(userRank_, userRank_, times, subStepLen, sendDataLocal));
    RecvMemBlock recvDataLocal;
    CHK_RET(CalcRecvInfo(userRank_, userRank_, times, subStepLen, recvDataLocal));

    DeviceMem src0 = userInput_.range(sendDataLocal.userInOffset, sendDataLocal.sendLen);
    DeviceMem dst0 = userOutput_.range(recvDataLocal.userOutOffset, recvDataLocal.recvLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] localCopy to userOut, src.Offset [%llu], dst.Offset[%llu], len[%llu]", userRank_,
        sendDataLocal.userInOffset, recvDataLocal.userOutOffset, recvDataLocal.recvLen);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst0, src0, mainStream_));

    // 读其他主卡的数据到usrOut
    RecvMemBlock recvData;
    CHK_RET(CalcRecvInfo(leftMain_, userRank_, times, subStepLen, recvData));
    const LINK& readMainTransport = links_[sendRecvRankInfo_[0].second];
    CHK_PTR_NULL(readMainTransport);
    void* remMemPtr = nullptr;
    CHK_RET(readMainTransport->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem remoteCCLInMem = DeviceMem::create(static_cast<u8*>(remMemPtr), cclOutMem_.size());
    DeviceMem src1 = remoteCCLInMem.range(recvData.cclSrcOffset, recvData.recvLen);
    DeviceMem dst1 = userOutput_.range(recvData.userOutOffset, recvData.recvLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] Memcpy to rank[%u] userOut, src.Offset [%llu], dst.Offset[%llu], len[%llu]",
        sendRecvRankInfo_[0].second, userRank_, recvData.cclSrcOffset, recvData.userOutOffset, recvData.recvLen);
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, dst1, src1, subStream_[0], readMainTransport->GetRemoteRank(), readMainTransport->GetLinkType()));
    HCCL_DEBUG("[AlltoAllVFor310P] RunMainStep 4 finish.");
    return HCCL_SUCCESS;
}

u64 AlltoAllVFor310P::CalcMaxSendLength()
{
    u64 maxSendDataLen = 0;
    for (u32 i = 0; i < allMeshAggregationSendRecvInfoPtr_->size(); i++) {
        for (u32 j = 0; j < (*allMeshAggregationSendRecvInfoPtr_)[i].sendLength.size(); j++) {
            u64 sendLength = (*allMeshAggregationSendRecvInfoPtr_)[i].sendLength[j];
            maxSendDataLen = std::max(maxSendDataLen, sendLength);
        }
    }
    HCCL_DEBUG("[AlltoAllVFor310P][CalcMaxSendLength] maxSendDataLen[%llu]", maxSendDataLen);
    return maxSendDataLen;
}

HcclResult
AlltoAllVFor310P::RunMainSendRecvBuffer(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen)
{
    // 读次卡的都用从流1，读主卡的都用从流0
    if (stepIdx == 0) {
        CHK_RET(MainFirstLocalCopy(times, roundIdx, subStepLen));
    } else if (stepIdx <= THIRD_STEP) {
        CHK_RET(NotifySubStreamStart());
        const LINK& rightMainTransport = links_[rightMain_];
        const LINK& leftMainTransport = links_[leftMain_];
        const LINK& minorTransport = links_[myMinor_];
        CHK_RET(rightMainTransport->TxAck(subStream_[0]));
        CHK_RET(leftMainTransport->RxAck(subStream_[0]));
        CHK_RET(minorTransport->TxAck(subStream_[1]));
        CHK_RET(minorTransport->RxAck(subStream_[1]));
        CHK_RET(RunMainCommonSteps(times, roundIdx, stepIdx, subStepLen));
        CHK_RET(leftMainTransport->TxDataSignal(subStream_[0]));
        CHK_RET(rightMainTransport->RxDataSignal(subStream_[0]));
        CHK_RET(minorTransport->TxDataSignal(subStream_[1]));
        CHK_RET(minorTransport->RxDataSignal(subStream_[1]));
        CHK_RET(WaitSubStreamFinish());
    } else {
        CHK_RET(NotifySubStreamStart());
        const LINK& rightMainTransport = links_[rightMain_];
        const LINK& leftMainTransport = links_[leftMain_];
        CHK_RET(rightMainTransport->TxAck(subStream_[0]));
        CHK_RET(leftMainTransport->RxAck(subStream_[0]));
        CHK_RET(RunMainStep4(times, subStepLen));
        CHK_RET(leftMainTransport->TxDataSignal(subStream_[0]));
        CHK_RET(rightMainTransport->RxDataSignal(subStream_[0]));
        CHK_RET(WaitSubStreamFinish());
    }
    return HCCL_SUCCESS;
}

HcclResult
AlltoAllVFor310P::RunMinorSendRecvBuffer(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen)
{
    if (stepIdx == 0) {
        CHK_RET(MinorFirstLocalCopy(times, roundIdx, subStepLen));
        // 主流告诉从流已拷贝完
    } else if (stepIdx <= THIRD_STEP) {
        CHK_RET(NotifySubStreamStart());
        const LINK& mainTransport = links_[myMain_]; // 次die读主die 0
        CHK_RET(mainTransport->TxAck(subStream_[0]));
        CHK_RET(mainTransport->RxAck(subStream_[0]));
        CHK_RET(RunMinorCommonSteps(times, roundIdx, stepIdx, subStepLen));
        CHK_RET(mainTransport->TxDataSignal(subStream_[0]));
        CHK_RET(mainTransport->RxDataSignal(subStream_[0]));
        CHK_RET(WaitSubStreamFinish());
    } else {
        CHK_RET(NotifySubStreamStart());
        CHK_RET(RunMinorStep4(times, subStepLen));
        CHK_RET(WaitSubStreamFinish());
    }
    return HCCL_SUCCESS;
}

void AlltoAllVFor310P::UpdateMinorStepMemInfo(const u32 roundIdx, const u32 stepIdx)
{
    (void)roundIdx;
    if (stepIdx % COMPUTE_CONST == 1) {
        minorStepInfo_.srcBuffId = 0; // 读主卡的src buffer
        minorStepInfo_.srcMemType = UserMemType::INPUT_MEM;
        minorStepInfo_.dstBuffId = 1; // 本地拷贝的dst buffer
    } else {
        minorStepInfo_.srcBuffId = 1;
        minorStepInfo_.srcMemType = UserMemType::OUTPUT_MEM;
        minorStepInfo_.dstBuffId = 0;
    }
    if (stepIdx == 1) {
        minorStepInfo_.readMain = std::make_pair(myMain_, userRank_);
        minorStepInfo_.readMinor = std::make_pair(userRank_, rightMain_); // 本地拷贝的数据的src/dst rank
    } else if (stepIdx == COMPUTE_CONST) {
        minorStepInfo_.readMain = std::make_pair(leftMain_, userRank_);
        minorStepInfo_.readMinor = std::make_pair(userRank_, myMain_);
    } else {
        minorStepInfo_.readMain = std::make_pair(leftMinor_, userRank_);
    }
}

HcclResult
AlltoAllVFor310P::RunMinorCommonSteps(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen)
{
    UpdateMinorStepMemInfo(roundIdx, stepIdx);
    if (stepIdx != THIRD_STEP) {
        SendMemBlock sendData;
        CHK_RET(
            CalcSendInfo(minorStepInfo_.readMinor.first, minorStepInfo_.readMinor.second, times, subStepLen, sendData));
        DeviceMem src = userInput_.range(sendData.userInOffset, sendData.sendLen);
        DeviceMem dst = memList_[minorStepInfo_.dstBuffId].range(sendData.cclDstOffset, sendData.sendLen);
        HCCL_DEBUG(
            "[AlltoAllVFor310P] rank[%u] localcopy, src.Offset [%llu], dst.Offset[%llu], len[%llu]", userRank_,
            sendData.userInOffset, sendData.cclDstOffset, sendData.sendLen);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    }

    RecvMemBlock recvData;
    CHK_RET(CalcRecvInfo(minorStepInfo_.readMain.first, minorStepInfo_.readMain.second, times, subStepLen, recvData));
    const LINK& readMainTransport = links_[sendRecvRankInfo_[0].second];
    CHK_PTR_NULL(readMainTransport);
    void* remMemPtr = nullptr;
    CHK_RET(readMainTransport->GetRemoteMem(minorStepInfo_.srcMemType, &remMemPtr));
    DeviceMem remoteMem = DeviceMem::create(static_cast<u8*>(remMemPtr), memList_[minorStepInfo_.srcBuffId].size());
    DeviceMem src0 = remoteMem.range(recvData.cclSrcOffset, recvData.recvLen);
    DeviceMem dst0 = userOutput_.range(recvData.userOutOffset, recvData.recvLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] Memcpy to rank[%u] userOut, src.Offset [%llu], dst.Offset[%llu], len[%llu]",
        sendRecvRankInfo_[0].second, userRank_, recvData.cclSrcOffset, recvData.userOutOffset, recvData.recvLen);
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, dst0, src0, subStream_[0], readMainTransport->GetRemoteRank(), readMainTransport->GetLinkType()));
    HCCL_DEBUG("[AlltoAllVFor310P] RunMinorStep %u finish.", stepIdx);

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::RunMinorStep4(const u32 times, const u64 subStepLen)
{
    SendMemBlock sendDataLocal;
    CHK_RET(CalcSendInfo(userRank_, userRank_, times, subStepLen, sendDataLocal));
    RecvMemBlock recvDataLocal;
    CHK_RET(CalcRecvInfo(userRank_, userRank_, times, subStepLen, recvDataLocal));

    DeviceMem src = userInput_.range(sendDataLocal.userInOffset, sendDataLocal.sendLen);
    DeviceMem dst = userOutput_.range(recvDataLocal.userOutOffset, recvDataLocal.recvLen);
    HCCL_DEBUG(
        "[AlltoAllVFor310P] rank[%u] localCopy to userOut, src.Offset [%llu], dst.Offset[%llu], len[%llu]", userRank_,
        sendDataLocal.userInOffset, recvDataLocal.userOutOffset, recvDataLocal.recvLen);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    HCCL_DEBUG("[AlltoAllVFor310P] RunMinorStep 4 finish.");
    return HCCL_SUCCESS;
}

HcclResult
AlltoAllVFor310P::RunSendRecvBuffer(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen)
{
    // 读次卡的都用从流0，读主卡的都用从流1
    HCCL_INFO(
        "[AlltoAllVFor310P] RunSendRecvBuffer start, times[%u], roundIdx[%u], stepIdx[%u]", times, roundIdx, stepIdx);
    if (mainRank_) {
        CHK_RET(RunMainSendRecvBuffer(times, roundIdx, stepIdx, subStepLen));
    } else {
        CHK_RET(RunMinorSendRecvBuffer(times, roundIdx, stepIdx, subStepLen));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVFor310P::RunAsync()
{
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, cclInMem_.size(), true);
    CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
    CHK_RET(RunAlltoAllVFor310P());
    CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, subStream_));
    HCCL_INFO("[AlltoAllVFor310P][RunAsync] finished");
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_V_FOR310P, AlltoAllVFor310P);
// namespace hccl
} // namespace hccl
