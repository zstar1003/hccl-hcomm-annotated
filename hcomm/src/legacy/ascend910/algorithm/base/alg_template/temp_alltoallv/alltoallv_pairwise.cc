/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_pairwise.h"
#include "alg_template_register.h"

namespace hccl {
AlltoAllVPairWise::AlltoAllVPairWise(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AlltoAllVPairWise::~AlltoAllVPairWise() {}

HcclResult AlltoAllVPairWise::Prepare(
    AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer, bool isAlltoAllZCopyMode, const Stream& stream,
    HcclWorkflowMode workMode, std::map<u32, std::vector<u64>>& rankSendDisplsMap,
    std::map<u32, std::vector<u64>>& rankRecvDisplsMap)
{
    DeviceMem scratchInputMem = DeviceMem();
    DeviceMem scratchOutputMem = DeviceMem();
    CHK_RET(AlltoAllVPairWise::Prepare(
        sendBuffer, recvBuffer, scratchInputMem, scratchOutputMem, isAlltoAllZCopyMode, stream, workMode,
        rankSendDisplsMap, rankRecvDisplsMap));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWise::Prepare(
    AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer, DeviceMem& scratchInputMem,
    DeviceMem& scratchOutputMem, bool isAlltoAllZCopyMode, const Stream& stream, HcclWorkflowMode workMode,
    std::map<u32, std::vector<u64>>& rankSendDisplsMap, std::map<u32, std::vector<u64>>& rankRecvDisplsMap)
{
    HCCL_INFO("[AlltoAllVPairWise][Prepare] Begin");
    scratchMemSize_ = 0;
    sendDataUnitBytes_ = 0;
    recvDataUnitBytes_ = 0;
    isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;
    workMode_ = workMode;
    rankSendDisplsMapPtr_ = &rankSendDisplsMap;
    rankRecvDisplsMapPtr_ = &rankRecvDisplsMap;

    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_PRT_RET(
            (!isAlltoAllZCopyMode_ && scratchInputMem.size() != scratchOutputMem.size()),
            HCCL_ERROR(
                "[AlltoAllVPairWise][Prepare]scratchInputMem and scratchOutputMem should be the same size, "
                "ScratchInputMem[%llu] ScratchOutputMem[%llu]",
                scratchInputMem.size(), scratchOutputMem.size()),
            HCCL_E_MEMORY);

        CHK_PRT_RET(
            scratchInputMem.size() == 0 || scratchOutputMem.size() == 0,
            HCCL_ERROR("[AlltoAllVPairWise][Prepare] invalid scratchMemSize[%llu]", scratchInputMem.size()),
            HCCL_E_PARA);
        scratchInputMem_ = scratchInputMem;
        scratchOutputMem_ = scratchOutputMem;
        scratchMemSize_ = scratchInputMem.size();
    }

    sendBuffer_ = sendBuffer;
    recvBuffer_ = recvBuffer;
    stream_ = stream;

    CHK_RET(SalGetDataTypeSize(sendBuffer_.dataType, sendDataUnitBytes_));
    CHK_RET(SalGetDataTypeSize(recvBuffer_.dataType, recvDataUnitBytes_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWise::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO("[AlltoAllVPairWise][RunAsync]: rank[%u] transportSize[%llu]", rank, links.size());
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    CHK_PRT_RET(rankSize == 0, HCCL_ERROR("[AlltoAllVPairWise][Prepare] invalid rankSize[%u]", rankSize), HCCL_E_PARA);

    CHK_PRT_RET(
        rankSize != links.size(),
        HCCL_ERROR(
            "[AlltoAllVPairWise][RunAsync]: rankSize[%u] and transport size[%llu] do not match", rankSize,
            links.size()),
        HCCL_E_PARA);

    CHK_RET(LocalCopy(rank));
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        CHK_RET(RunBCopyAlltoAll(rank, rankSize, links));
    } else {
        CHK_RET(RunZCopyAlltoAll(rank, rankSize, links));
    }
    return HCCL_SUCCESS;
}

// 从本rank的sendbuffer拷贝到本rank的recvbuffer
HcclResult AlltoAllVPairWise::LocalCopy(const u32 rank)
{
    DeviceMem dstMem = recvBuffer_.mem.range(
        recvDataUnitBytes_ * recvBuffer_.displs[rank], recvBuffer_.counts[rank] * recvDataUnitBytes_);
    DeviceMem srcMem = sendBuffer_.mem.range(
        sendDataUnitBytes_ * sendBuffer_.displs[rank], sendBuffer_.counts[rank] * sendDataUnitBytes_);
    HCCL_DEBUG(
        "[AlltoAllVPairWise][LocalCopy]: Rank[%u] destAddr[%p], destMax[%llu], srcAddr[%p], size[%llu]", rank,
        dstMem.ptr(), dstMem.size(), srcMem.ptr(), srcMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWise::RunBCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    for (u32 i = 1; i < rankSize; i++) {
        u32 prevRank = (rank + rankSize - i) % rankSize;
        u32 nextRank = (rank + i) % rankSize;
        std::shared_ptr<Transport> prevTransport = links[prevRank];
        std::shared_ptr<Transport> nextTransport = links[nextRank];

        CHK_SMART_PTR_NULL(prevTransport);
        CHK_SMART_PTR_NULL(nextTransport);

        HCCL_DEBUG("[AlltoAllVPairWise][RunBCopyAlltoAll]: prevRank[%u] nextRank[%u], step[%u]", prevRank, nextRank, i);

        u64 sendBytes = sendBuffer_.counts[nextRank] * sendDataUnitBytes_;
        u64 recvBytes = recvBuffer_.counts[prevRank] * recvDataUnitBytes_;

        u64 sendDispBytes = sendBuffer_.displs[nextRank] * sendDataUnitBytes_;
        u64 recvDispBytes = recvBuffer_.displs[prevRank] * recvDataUnitBytes_;

        // scratchMemSize_ 的合法性已经在 Prepare 函数中校验
        u32 sendTimes = (sendBytes / scratchMemSize_) + ((sendBytes % scratchMemSize_) == 0 ? 0 : 1);
        u32 recvTimes = (recvBytes / scratchMemSize_) + ((recvBytes % scratchMemSize_) == 0 ? 0 : 1);

        HCCL_DEBUG(
            "[AlltoAllVPairWise][RunBCopyAlltoAll]: rank[%u] "
            "sendTimes[%u] recvTimes[%u] sendBytes[%llu] recvBytes[%llu] scratchMemSize_[%llu]",
            rank, sendTimes, recvTimes, sendBytes, recvBytes, scratchMemSize_);

        u32 curSendTime = 0;
        u32 curRecvTime = 0;
        while (sendTimes != 0 || recvTimes != 0) {
            u8* sendAddr = reinterpret_cast<u8*>(sendBuffer_.mem.ptr()) + sendDispBytes + curSendTime * scratchMemSize_;
            u8* recvAddr = reinterpret_cast<u8*>(recvBuffer_.mem.ptr()) + recvDispBytes + curRecvTime * scratchMemSize_;
            u64 curSendBytes = 0;
            u64 curRecvBytes = 0;
            CHK_RET(CalcSendRecvCounts(sendTimes, curSendTime, sendBytes, curSendBytes));
            CHK_RET(CalcSendRecvCounts(recvTimes, curRecvTime, recvBytes, curRecvBytes));

            HCCL_DEBUG(
                "[AlltoAllVPairWise][RunBCopyAlltoAll]: "
                "curSendTime[%llu] curRecvTime[%llu] curSendBytes[%llu] curRecvBytes[%llu]",
                curSendTime, curRecvTime, curSendBytes, curRecvBytes);

            HcclResult ret = SendRecv(curSendBytes, curRecvBytes, sendAddr, recvAddr, prevTransport, nextTransport);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AlltoAllVPairWise][RunBCopyAlltoAll]: errNo[0x%016llx] "
                    "curSendBytes[%llu] curRecvBytes[%llu] sendAddr[%p] recvAddr[%p]",
                    HCCL_ERROR_CODE(ret), curSendBytes, curRecvBytes, sendAddr, recvAddr),
                ret);

            curSendTime = curSendBytes != 0 ? curSendTime + 1 : curSendTime;
            curRecvTime = curRecvBytes != 0 ? curRecvTime + 1 : curRecvTime;
            if (curSendTime == sendTimes && curRecvTime == recvTimes) {
                break;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWise::CalcSendRecvCounts(u32 times, u32 curTime, u64 totalBytes, u64& curBytes) const
{
    if (times == 0) { // 不需要发送
        curBytes = 0;
    } else if (times == 1 && curTime == times - 1) { // 只发一次
        curBytes = totalBytes;
    } else if (times > 1 && totalBytes % scratchMemSize_ == 0 && curTime < times) {
        curBytes = scratchMemSize_;
    } else if (times > 1 && totalBytes % scratchMemSize_ != 0 && curTime < times - 1) {
        curBytes = scratchMemSize_;
    } else if (times > 1 && totalBytes % scratchMemSize_ != 0 && curTime == times - 1) {
        curBytes = totalBytes % scratchMemSize_;
    } else {
        curBytes = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWise::SendRecv(
    u64 curSendBytes, u64 curRecvBytes, u8* sendAddr, u8* recvAddr, std::shared_ptr<Transport> prevTransport,
    std::shared_ptr<Transport> nextTransport)
{
    if (curRecvBytes > 0) {
        CHK_RET(prevTransport->TxAck(stream_)); // transport sync record
    }
    if (curSendBytes > 0) {
        CHK_RET(nextTransport->RxAck(stream_)); // transport sync wait
        DeviceMem srcMem1 = DeviceMem::create(sendAddr, curSendBytes);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, scratchInputMem_, srcMem1, stream_));
        // send payload + notify
        CHK_RET(nextTransport->TxAsync(UserMemType::OUTPUT_MEM, 0, scratchInputMem_.ptr(), curSendBytes, stream_));
    }
    if (curRecvBytes > 0) {
        CHK_RET(prevTransport->RxAsync(UserMemType::INPUT_MEM, 0, scratchOutputMem_.ptr(), curRecvBytes, stream_));
        DeviceMem dstMem = DeviceMem::create(recvAddr, curRecvBytes);
        DeviceMem srcMem = scratchOutputMem_.range(0, curRecvBytes);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream_));
        CHK_RET(prevTransport->TxAck(stream_)); // record
    }
    if (curSendBytes > 0) {
        CHK_RET(nextTransport->RxAck(stream_));        // wait
        CHK_RET(nextTransport->TxDataSignal(stream_)); // record
    }
    if (curRecvBytes > 0) {
        CHK_RET(prevTransport->RxDataSignal(stream_)); // wait
        CHK_RET(prevTransport->RxWaitDone(stream_));
    }
    if (curSendBytes > 0) {
        CHK_RET(nextTransport->TxWaitDone(stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWise::SendRecv(
    TxMemoryInfo txMemoryInfo, RxMemoryInfo rxMemoryInfo, std::shared_ptr<Transport> prevTransport,
    std::shared_ptr<Transport> nextTransport)
{
    // send payload + notify
    CHK_RET(nextTransport->TxAsync(
        txMemoryInfo.dstMemType, txMemoryInfo.dstOffset, txMemoryInfo.src, txMemoryInfo.len, stream_));
    CHK_RET(prevTransport->RxAsync(
        rxMemoryInfo.srcMemType, rxMemoryInfo.srcOffset, rxMemoryInfo.dst, rxMemoryInfo.len, stream_));
    CHK_RET(prevTransport->TxAck(stream_));        // record
    CHK_RET(nextTransport->RxAck(stream_));        // wait
    CHK_RET(nextTransport->TxDataSignal(stream_)); // record
    CHK_RET(prevTransport->RxDataSignal(stream_)); // wait
    CHK_RET(prevTransport->RxWaitDone(stream_));
    CHK_RET(nextTransport->TxWaitDone(stream_));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWise::RunZCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    for (u32 i = 1; i < rankSize; i++) {
        u32 prevRank = (rank + rankSize - i) % rankSize;
        u32 nextRank = (rank + i) % rankSize;
        std::shared_ptr<Transport> prevTransport = links[prevRank];
        std::shared_ptr<Transport> nextTransport = links[nextRank];

        CHK_SMART_PTR_NULL(prevTransport);
        CHK_SMART_PTR_NULL(nextTransport);

        HCCL_DEBUG("[AlltoAllVPairWise][RunZCopyAlltoAll]: prevRank[%u] nextRank[%u], step[%u]", prevRank, nextRank, i);

        CHK_RET(prevTransport->TxAck(stream_)); // transport sync record
        CHK_RET(nextTransport->RxAck(stream_)); // transport sync wait

        u64 sendBytes = sendBuffer_.counts[nextRank] * sendDataUnitBytes_;
        u64 recvBytes = recvBuffer_.counts[prevRank] * recvDataUnitBytes_;
        u64 sendDispBytes = sendBuffer_.displs[nextRank] * sendDataUnitBytes_;
        u64 recvDispBytes = recvBuffer_.displs[prevRank] * recvDataUnitBytes_;
        u8* sendAddr = reinterpret_cast<u8*>(sendBuffer_.mem.ptr()) + sendDispBytes;
        u8* recvAddr = reinterpret_cast<u8*>(recvBuffer_.mem.ptr()) + recvDispBytes;

        u64 dstOffset = rankRecvDisplsMapPtr_->at(nextRank)[rank];
        u64 srcOffset = rankSendDisplsMapPtr_->at(prevRank)[rank];

        TxMemoryInfo txMemoryInfo{UserMemType::OUTPUT_MEM, dstOffset, sendAddr, sendBytes};
        RxMemoryInfo rxMemoryInfo{UserMemType::INPUT_MEM, srcOffset, recvAddr, recvBytes};

        HCCL_DEBUG(
            "[AlltoAllVPairWise][RunZCopyAlltoAll]: sendBytes[%llu] recvBytes[%llu] sendDispBytes[%llu]"
            " dstOffset[%llu]",
            sendBytes, recvBytes, sendDispBytes, dstOffset);
        HcclResult ret = SendRecv(txMemoryInfo, rxMemoryInfo, prevTransport, nextTransport);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AlltoAllVPairWise][RunZCopyAlltoAll]errNo[0x%016llx] "
                "sendBytes[%llu] recvBytes[%llu] sendAddr[%p] dstOffset[%llu]",
                HCCL_ERROR_CODE(ret), sendBytes, recvBytes, sendAddr, dstOffset),
            ret);
    }

    return HCCL_SUCCESS;
}
HcclResult AlltoAllVPairWise::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    (void)links;
    for (u32 i = 1; i < rankSize; i++) {
        u32 nextRank = (rank + i) % rankSize;
        if (i < NSLBDP_PAIRWISE_MAXPHASE) {
            NslbDpAdjInfo adjInfoStep = {};
            adjInfoStep.dstLocalRankId = nextRank;
            adjInfoStep.phaseId = i;
            adjInfoStep.rev = 0;
            HCCL_INFO(
                "AlltoAllVPairWise-NSLB-DP: adjInfoStep.phaseId[%u], remoteuserRank[%u]", adjInfoStep.phaseId,
                nextRank);
            nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
        }
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_V_PAIRWISE, AlltoAllVPairWise);
} // namespace hccl
