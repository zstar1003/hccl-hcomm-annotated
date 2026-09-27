/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"

#include "ins_coll_alg_registry.h"
#include "ins_batchSendRecv_executor.h"
#include "alg_data_trans_wrapper.h"

using namespace std;

namespace Hccl {

template <typename AlgTopoMatch>
InsBatchSendRecvExecutor<AlgTopoMatch>::InsBatchSendRecvExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch>
InsBatchSendRecvExecutor<AlgTopoMatch>::~InsBatchSendRecvExecutor()
{}

template <typename AlgTopoMatch>
void InsBatchSendRecvExecutor<AlgTopoMatch>::SetRmaDataBufferMgr(const RmtDataBufferMgr* rmaDataBufferMgr)
{
    rmaDataBufferMgr_ = const_cast<RmtDataBufferMgr*>(rmaDataBufferMgr);
    return;
}

template <typename AlgTopoMatch>
void InsBatchSendRecvExecutor<AlgTopoMatch>::SetOp(const CollAlgOperator& op)
{
    op_ = op;
    HcclSendRecvItem* itemPtr = reinterpret_cast<HcclSendRecvItem*>(op.batchSendRecvDataDes.sendRecvItemsPtr);
    u32 itemNum = op.batchSendRecvDataDes.itemNum;
    if (itemPtr == nullptr) {
        THROW<NullPtrException>(StringFormat("itemPtr is null!"));
    }
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum; i++) {
        commTargetUserRankSet_.insert((itemPtr + i)->remoteRank);
        HCCL_DEBUG(
            "[InsBatchSendRecvExecutor][ParseParam] insert remoteUserRank[%u] to Set ", (itemPtr + i)->remoteRank);
    }
    HCCL_DEBUG("[SetOp]commTargetUserRankSet_ size[%zu]", commTargetUserRankSet_.size());
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::InitParams(const CollAlgOperator& op, const CollAlgParams& params)
{
    opMode_ = params.opMode;
    maxTmpMemSize_ = params.maxTmpMemSize;
    CHK_PRT_RET(
        (maxTmpMemSize_ == 0), HCCL_ERROR("[InitParams] maxTmpMemSize equals to zero for OPBASE."),
        HcclResult::HCCL_E_PARA);
    HcclSendRecvItem* itemPtr = reinterpret_cast<HcclSendRecvItem*>(op.batchSendRecvDataDes.sendRecvItemsPtr);
    u32 itemNum = op.batchSendRecvDataDes.itemNum;
    CHK_PTR_NULL(itemPtr);
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum; i++) {
        commTargetUserRankSet_.insert((itemPtr + i)->remoteRank);
        HCCL_DEBUG(
            "[InsBatchSendRecvExecutor][ParseParam] insert remoteUserRank[%u] to Set ", (itemPtr + i)->remoteRank);
    }
    HCCL_DEBUG("[InitParams]commTargetUserRankSet_ size[%zu]", commTargetUserRankSet_.size());
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
bool InsBatchSendRecvExecutor<AlgTopoMatch>::SortSendItems(HcclSendRecvItem* a, HcclSendRecvItem* b) const
{
    u32 aFlag = (a->remoteRank <= static_cast<uint32_t>(myRank_)) ? (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank <= static_cast<uint32_t>(myRank_)) ? (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag > bFlag) {
        return true;
    } else if (aFlag < bFlag) {
        return false;
    }
    return a->count > b->count;
}

template <typename AlgTopoMatch>
bool InsBatchSendRecvExecutor<AlgTopoMatch>::SortRecvItems(HcclSendRecvItem* a, HcclSendRecvItem* b) const
{
    u32 aFlag = (a->remoteRank < static_cast<uint32_t>(myRank_)) ? (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank < static_cast<uint32_t>(myRank_)) ? (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag > bFlag) {
        return false;
    } else if (aFlag < bFlag) {
        return true;
    }
    return a->count > b->count;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::GetPairWiseList(HcclSendRecvItem* sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[InsBatchSendRecvExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO(
            "[InsBatchSendRecvExecutor][GetPairWiseList] index is %u, itemNum is %u,"
            "localRankID is %d, remoteRank is %u, sendRecvType is %u, rankSize is %u.",
            i, itemNum, myRank_, sendRecvInfo->remoteRank, static_cast<u32>(sendRecvInfo->sendRecvType), rankSize_);
        CHK_PTR_NULL(sendRecvInfo->buf);

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            sendDeque_.push_back(sendRecvInfo);
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            recvDeque_.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR(
                "[InsBatchSendRecvExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, "
                "rankID is %d, remoteRank is %u.",
                sendRecvInfo->sendRecvType, myRank_, sendRecvInfo->remoteRank);
            return HcclResult::HCCL_E_PARA;
        }
        sendRecvInfo++;
    }

    /* 此处的排序逻辑(pair-wise算法):
        1.sendDeque元素顺序是:先放remoteRank号小于等于root rank的第一个任务，依次减小(循环索引)直至放完
        2.recvDeque元素顺序是:先放remoteRank号大于等于root rank的第一个任务，依次增大(循环索引)直至放完
        如果有rank间重复send/recv场景，按照收发数据从大到小排序
    */
    auto sendCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        return this->SortSendItems(a, b);
    };

    auto recvCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        return this->SortRecvItems(a, b);
    };

    std::stable_sort(sendDeque_.begin(), sendDeque_.end(), sendCompare);
    std::stable_sort(recvDeque_.begin(), recvDeque_.end(), recvCompare);

    // 筛选自收发任务
    while ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))
           && (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
        sendToSelfDeque_.push_back(sendDeque_.front());
        recvFromSelfDeque_.push_back(recvDeque_.front());
        sendDeque_.pop_front();
        recvDeque_.pop_front();
    }
    // 自收发任务按照收发长度大小排序
    auto selfDequeCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        return a->count > b->count;
    };

    std::stable_sort(sendToSelfDeque_.begin(), sendToSelfDeque_.end(), selfDequeCompare);
    std::stable_sort(recvFromSelfDeque_.begin(), recvFromSelfDeque_.end(), selfDequeCompare);

    // 如果自发自收任务没有完全匹配
    if ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))
        || (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
        HCCL_ERROR("[CollBatchSendRecvExecutor] SendTask and Recv Task to rank itself do not match,"
                   "please check the task list.");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_INFO("[CollBatchSendRecvExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::ProcessSelfSendRecvTasks(InsQuePtr& queue)
{
    while (!sendToSelfDeque_.empty() && !recvFromSelfDeque_.empty()) {
        if (sendToSelfDeque_.front()->count == recvFromSelfDeque_.front()->count
            && sendToSelfDeque_.front()->dataType == recvFromSelfDeque_.front()->dataType) {
            HcclDataType hccldataTypeSelf = sendToSelfDeque_.front()->dataType;
            DataType dataTypeSelf = HcclDataTypeToDataType(hccldataTypeSelf);
            u64 dataSize = sendToSelfDeque_.front()->count * DataTypeSizeGet(dataTypeSelf);

            // 搬运本卡到本卡的数据 使用扩展InsLocalCopyExtend接口
            DataBuffer inputBuffer(reinterpret_cast<uintptr_t>(sendToSelfDeque_.front()->buf), dataSize);
            DataBuffer outputBuffer(reinterpret_cast<uintptr_t>(recvFromSelfDeque_.front()->buf), dataSize);
            HCCL_DEBUG(
                "inputBuffer[%llu], outputBuffer[%llu], dataSize[%llu]", inputBuffer.GetAddr(), outputBuffer.GetAddr(),
                dataSize);
            queue->Append(std::make_unique<InsLocalCopyExtend>(inputBuffer, outputBuffer)); // localcopy

            sendToSelfDeque_.pop_front();
            recvFromSelfDeque_.pop_front();
        } else {
            HCCL_ERROR("[HcclBatchSendRecv] Send task and recv task to self : count or dataType do not equal, please "
                       "check the task list.");
            return HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::ProcessSendRecv(
    const CollAlgOperator& op, InsQuePtr& queue, u32 remoteRank, std::vector<SendRecvSlice>& sendRemoteSlices,
    std::vector<SendRecvSlice>& recvRemoteSlices, LinkData& link) const
{
    HCCL_INFO("[InsBatchSendRecvExecutor][ProcessSendRecv] Start to send/recv with rank[%u].", remoteRank);
    u32 maxSendRecvStep = std::max(sendRemoteSlices.size(), recvRemoteSlices.size());
    HCCL_DEBUG("[InsBatchSendRecvExecutor][ProcessSendRecv] maxSendRecvStep[%u].", maxSendRecvStep);

    CHK_PTR_NULL(op.scratchMem);
    uint64_t scratchBufferAddr = op.scratchMem->GetAddr();

    // 当前是write的逻辑，整理一套read的逻辑出来
    if (dmaMode_ == DmaMode::GET) {
        for (u32 step = 0; step < maxSendRecvStep; step++) {
            if (step < sendRemoteSlices.size()) {
                // 先做localCopy local copy: usrin->cclin
                DataBuffer inputBuffer(sendRemoteSlices[step].addr_, sendRemoteSlices[step].size_);
                DataBuffer inScratchSlice(
                    scratchBufferAddr + (remoteRank % rankSize_) * maxRoundTransferSize_, sendRemoteSlices[step].size_);
                HCCL_DEBUG(
                    "scratchBufferAddr[%llu], offset[%llu], dataSize[%llu]", scratchBufferAddr,
                    (remoteRank % rankSize_) * maxRoundTransferSize_, sendRemoteSlices[step].size_);

                queue->Append(std::make_unique<InsLocalCopyExtend>(
                    inputBuffer, inScratchSlice)); // 这里还没做DMA消减，可以到时看下性能再优化
                // 然后通知对端来读
                queue->Append(std::make_unique<InsPostReady>(static_cast<RankId>(remoteRank), link));
            }

            if (step < recvRemoteSlices.size()) {
                CHK_RET(ProcessRecvDataSlice(queue, recvRemoteSlices[step], remoteRank, scratchBufferAddr, link));
            }

            if (step < sendRemoteSlices.size()) {
                queue->Append(std::make_unique<InsWaitFin>(static_cast<RankId>(remoteRank), link));
            }
        }

        return HcclResult::HCCL_SUCCESS;
    }

    for (u32 step = 0; step < maxSendRecvStep; step++) {
        if (step < recvRemoteSlices.size()) {
            // tell sendRank ready to write
            queue->Append(std::make_unique<InsPostReady>(static_cast<RankId>(remoteRank), link));
        }
        if (step < sendRemoteSlices.size()) {
            CHK_RET(ProcessSendDataSlice(queue, sendRemoteSlices[step], remoteRank, scratchBufferAddr, link));
        }
        if (step < recvRemoteSlices.size()) {
            // wait sendRank write done
            queue->Append(std::make_unique<InsWaitFin>(static_cast<RankId>(remoteRank), link));
            // local copy
            CHK_RET(CopyRecvDataSliceToUsrOut(queue, recvRemoteSlices[step], remoteRank, scratchBufferAddr));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::RunLoopSendRecv(
    const CollAlgOperator& op, std::vector<InsQuePtr>& queues, InsTempAllGatherMesh1D& tempAlg)
{
    // pre sync
    CHK_RET(tempAlg.PreSyncInterQueues(queues));

    // sendrecv options
    u32 queIdx = 1;
    for (const u32& remoteRank : commTargetUserRankSet_) {
        HCCL_INFO("[InsBatchSendRecvExecutor][RunLoopSendRecv] remoteRank[%u].", remoteRank);
        if (remoteRank == static_cast<uint32_t>(myRank_)) {
            continue;
        }
        if (queIdx >= queues.size()) {
            HCCL_ERROR(
                "[InsBatchSendRecvExecutor][RunLoopSendRecv] queIdx[%u] is bigger than queues size[%u].", queIdx,
                queues.size());
            return HCCL_E_PARA;
        }
        auto sendIt = SendSliceMapByRemoteRank_.find(remoteRank);
        auto recvIt = RecvSliceMapByRemoteRank_.find(remoteRank);
        if (sendIt == SendSliceMapByRemoteRank_.end() && recvIt == RecvSliceMapByRemoteRank_.end()) {
            continue;
        }
        LinkData link = tempResLinks_.at(remoteRank)[0];
        if (sendIt != SendSliceMapByRemoteRank_.end() && recvIt != RecvSliceMapByRemoteRank_.end()) {
            CHK_RET(ProcessSendRecv(
                op, queues[queIdx], remoteRank, SendSliceMapByRemoteRank_[remoteRank],
                RecvSliceMapByRemoteRank_[remoteRank], link));
        } else if (sendIt != SendSliceMapByRemoteRank_.end()) {
            std::vector<SendRecvSlice> empty;
            CHK_RET(
                ProcessSendRecv(op, queues[queIdx], remoteRank, SendSliceMapByRemoteRank_[remoteRank], empty, link));
        } else if (recvIt != RecvSliceMapByRemoteRank_.end()) {
            std::vector<SendRecvSlice> empty;
            CHK_RET(
                ProcessSendRecv(op, queues[queIdx], remoteRank, empty, RecvSliceMapByRemoteRank_[remoteRank], link));
        }
        queIdx++;
    }

    // post sync
    CHK_RET(tempAlg.PostSyncInterQueues(queues));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::GenSendSlicesMapRank()
{
    // 遍历 sendDataSlices_，将每个元素根据其 remoteRank 放入相应的 vector 中
    for (const auto& slice : sendDataSilces_) {
        // 获取 remoteRank
        int remoteRank = slice.remoteRank_;

        // 将当前 slice 放入对应的 vector 中
        SendSliceMapByRemoteRank_[remoteRank].emplace_back(slice);
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::CalcSendSlices(u64 maxRoundTransferSize)
{
    while (!sendDeque_.empty()) {
        HcclSendRecvItem* sendRecvItem = sendDeque_.front();
        HCCL_INFO(
            "[InsBatchSendRecvExecutor][CalcSendSlices] remoteRank[%u], buf[%p], count[%llu],"
            "dataType[%u], sendRecvType[%d].",
            sendRecvItem->remoteRank, sendRecvItem->buf, sendRecvItem->count, sendRecvItem->dataType,
            sendRecvItem->sendRecvType);
        u8* curInputPtr = static_cast<u8*>(sendRecvItem->buf);
        CHK_PTR_NULL(curInputPtr);

        HcclDataType hccldataTypeSend = sendRecvItem->dataType;
        DataType dataTypeSend = HcclDataTypeToDataType(hccldataTypeSend);
        u32 unitSize = DataTypeSizeGet(dataTypeSend);

        u64 resDataSize = sendRecvItem->count * unitSize;
        u64 curOffset = 0;

        while (resDataSize > 0) {
            // 判断本轮需搬运的数据量
            u64 transferSize = resDataSize > maxRoundTransferSize ? maxRoundTransferSize : resDataSize;
            curInputPtr = static_cast<u8*>(sendRecvItem->buf) + curOffset;
            sendDataSilces_.emplace_back(
                reinterpret_cast<uintptr_t>(curInputPtr), transferSize, sendRecvItem->remoteRank);
            HCCL_DEBUG(
                "[InsBatchSendRecvExecutor][CalcSendSlices] slice curOffset[%llu], slice size[%llu] curInputPtr [%p].",
                curOffset, transferSize, curInputPtr);
            curOffset += transferSize;
            resDataSize -= transferSize;
        }
        sendDeque_.pop_front();
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::GenRecvSlicesMapRank()
{
    // 遍历 recvDataSilces_, 将每个元素根据其 remoteRank 放入相应的 vector 中
    for (const auto& slice : recvDataSilces_) {
        // 获取 remoteRank
        int remoteRank = slice.remoteRank_;

        // 将当前 slice 放入对应的 vector 中
        RecvSliceMapByRemoteRank_[remoteRank].emplace_back(slice);
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::CalcRecvSlices(u64 maxRoundTransferSize)
{
    while (!recvDeque_.empty()) {
        HcclSendRecvItem* sendRecvItem = recvDeque_.front();
        HCCL_INFO(
            "[InsBatchSendRecvExecutor][CalcSendSlices] remoteRank[%u], buf[%p], count[%llu],"
            "dataType[%u], sendRecvType[%d].",
            sendRecvItem->remoteRank, sendRecvItem->buf, sendRecvItem->count, sendRecvItem->dataType,
            sendRecvItem->sendRecvType);
        u8* curInputPtr = static_cast<u8*>(sendRecvItem->buf);
        CHK_PTR_NULL(curInputPtr);

        HcclDataType hccldataTypeRecv = sendRecvItem->dataType;
        DataType dataTypeRecv = HcclDataTypeToDataType(hccldataTypeRecv);
        u32 unitSize = DataTypeSizeGet(dataTypeRecv);

        u64 resDataSize = sendRecvItem->count * unitSize;
        u64 curOffset = 0;

        while (resDataSize > 0) {
            // 判断本轮需搬运的数据量
            u64 transferSize = resDataSize > maxRoundTransferSize ? maxRoundTransferSize : resDataSize;
            curInputPtr = static_cast<u8*>(sendRecvItem->buf) + curOffset;
            recvDataSilces_.emplace_back(
                reinterpret_cast<uintptr_t>(curInputPtr), transferSize, sendRecvItem->remoteRank);
            HCCL_DEBUG(
                "[InsBatchSendRecvExecutor][CalcRecvSlices] slice curOffset[%llu], slice size[%llu], curInputPtr [%p].",
                curOffset, transferSize, curInputPtr);
            curOffset += transferSize;
            resDataSize -= transferSize;
        }

        recvDeque_.pop_front();
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::ProcessRecvDataSlice(
    InsQuePtr& queue, SendRecvSlice& recvRemoteSlice, u32 remoteRank, uint64_t scratchBufferAddr, LinkData& link) const
{
    // 从远端读
    queue->Append(std::make_unique<InsWaitReady>(static_cast<RankId>(remoteRank), link));

    // 获取远端内存地址, 获取的是scratch的基起始地址
    DataBuffer remoteBuffer = rmaDataBufferMgr_->GetBuffer(link, BufferType::SCRATCH);
    uint64_t remoteBufferAddr = remoteBuffer.GetAddr();
    DataBuffer srcScratchSlice(remoteBufferAddr + (myRank_ % rankSize_) * maxRoundTransferSize_, recvRemoteSlice.size_);

    // 准备数据偏移
    u64 offsetOfRemoteScratchBase
        = maxRoundTransferSize_ * rankSize_ + maxRoundTransferSize_ * (remoteRank % rankSize_);
    DataBuffer dstScratchSlice(scratchBufferAddr + offsetOfRemoteScratchBase, recvRemoteSlice.size_);
    HCCL_DEBUG(
        "[InsBatchSendRecvExecutor][ProcessRecvDataSlice] myRank[%d] recv Size[%llu], remoteBuffer[%llu], "
        "remoteUserRank[%u].",
        myRank_, recvRemoteSlice.size_, remoteBufferAddr, remoteRank);

    // Recv
    queue->Append(
        std::make_unique<InsReadExtend>(static_cast<RankId>(remoteRank), link, dstScratchSlice, srcScratchSlice));

    queue->Append(std::make_unique<InsPostFin>(static_cast<RankId>(remoteRank), link));

    // 最后把数据拷贝回自己的output
    CHK_RET(CopyRecvDataSliceToUsrOut(queue, recvRemoteSlice, remoteRank, scratchBufferAddr));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::ProcessSendDataSlice(
    InsQuePtr& queue, SendRecvSlice& sendRemoteSlice, u32 remoteRank, uint64_t scratchBufferAddr, LinkData& link) const
{
    // local copy: usrin->cclin
    DataBuffer inputBuffer(sendRemoteSlice.addr_, sendRemoteSlice.size_);
    DataBuffer inScratchSlice(
        scratchBufferAddr + (remoteRank % rankSize_) * maxRoundTransferSize_, sendRemoteSlice.size_);
    HCCL_DEBUG(
        "scratchBufferAddr[%llu], offset[%llu], dataSize[%llu]", scratchBufferAddr,
        (remoteRank % rankSize_) * maxRoundTransferSize_, sendRemoteSlice.size_);

    queue->Append(std::make_unique<InsLocalCopyExtend>(inputBuffer, inScratchSlice));

    CHK_RET(SendRun(inScratchSlice, remoteRank, queue, link));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::CopyRecvDataSliceToUsrOut(
    InsQuePtr& queue, SendRecvSlice& slice, u32 remoteRank, uint64_t scratchBufferAddr) const
{
    // local copy : cclout->usrout
    DataBuffer outScratchSlice(
        scratchBufferAddr + (remoteRank % rankSize_ + rankSize_) * maxRoundTransferSize_, slice.size_);

    DataBuffer outputBuffer(slice.addr_, slice.size_);
    HCCL_DEBUG(
        "[InsBatchSendRecvExecutor][CopyRecvDataSliceToUsrOut] scratchMem Addr[%llu] localcopy"
        "size[%llu] to outputBuffer[%llu].",
        outScratchSlice.GetAddr(), slice.size_, outputBuffer.GetAddr());
    queue->Append(std::make_unique<InsLocalCopyExtend>(outScratchSlice, outputBuffer));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::SendRun(
    DataBuffer& execBufferSlice, u32 remoteUserRank, InsQuePtr& queue, LinkData& link) const
{
    if (execBufferSlice.GetSize() == 0) {
        HCCL_ERROR("[InsBatchSendRecvExecutor][SendRun] SendRun input is null");
        return HCCL_E_PTR;
    }

    u64 sendSize = execBufferSlice.GetSize();

    // 准备数据偏移
    u64 offsetOfRemoteScratchBase = maxRoundTransferSize_ * rankSize_ + maxRoundTransferSize_ * (myRank_ % rankSize_);

    // 获取远端内存地址, 获取的是scratch的基起始地址
    DataBuffer remoteBuffer = rmaDataBufferMgr_->GetBuffer(link, BufferType::SCRATCH);
    uint64_t remoteBufferAddr = remoteBuffer.GetAddr();
    DataBuffer sendRemoteBuffer(remoteBufferAddr + offsetOfRemoteScratchBase, sendSize);
    HCCL_DEBUG(
        "[InsBatchSendRecvExecutor][SendRun] myRank[%d] send Size[%llu], remoteBuffer[%llu], remoteUserRank[%u].",
        myRank_, sendSize, remoteBuffer.GetAddr(), remoteUserRank);

    // wait recvRank ready
    queue->Append(std::make_unique<InsWaitReady>(static_cast<RankId>(remoteUserRank), link));

    // Send
    queue->Append(std::make_unique<InsWriteWithFinExtend>(
        static_cast<RankId>(remoteUserRank), link, execBufferSlice, sendRemoteBuffer));

    return HcclResult::HCCL_SUCCESS;
}

// 算子执行ccu接口
template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    (void)rankGraph;
    (void)op;
    (void)params;
    (void)insQue;

    return HcclResult::HCCL_E_NOT_SUPPORT;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsBatchSendRecvExecutor][Orchestrate] Begin to Generate Instruction Queue for BatchSendRecv.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    CHK_PRT_RET(
        topoInfo.vTopo.size() == 0, HCCL_ERROR("[InsBatchSendRecvExecutor] Rank[%d], vTopo size is zero.", myRank_),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        topoInfo.virtRankMap.size() == 0,
        HCCL_ERROR("[InsBatchSendRecvExecutor] Rank[%d], virtRankMap size is zero.", myRank_), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        rankSize_ == 1, HCCL_ERROR("BatchSendRecv Executor orchestrate failed, do not support single rank."),
        HcclResult::HCCL_E_PARA);

    virtRankMap_ = topoInfo.virtRankMap[0];
    vTopo_ = topoInfo.vTopo[0];

    InsTempAllGatherMesh1D tempAlg(myRank_, rankSize_, topoInfo.vTopo[0], topoInfo.virtRankMap[0]);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    CHK_RET(CalcRes(tempResReq));

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_DEBUG("[InsBatchSendRecvExecutor] Rank[%d], requiredQue Num [%u].", myRank_, tempResReq.queNum);

    CHK_PTR_NULL(linkMgr);
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    if (tempAlg.IsPcieLink(tempResLinks_)) {
        dmaMode_ = DmaMode::GET;
    }

    // cclbuffer
    buffInfo_.inBuffType = BufferType::SCRATCH;
    buffInfo_.outBuffType = BufferType::SCRATCH;
    buffInfo_.inBuffBaseOff = 0;
    buffInfo_.outBuffBaseOff = maxTmpMemSize_ / 2; // 占据scratch memory的后半部分，除以2

    // batchsendrecv实现
    CHK_RET(GetPairWiseList(
        static_cast<HcclSendRecvItem*>(op.batchSendRecvDataDes.sendRecvItemsPtr), op.batchSendRecvDataDes.itemNum));
    CHK_RET(ProcessSelfSendRecvTasks(requiredQue_[0]));

    // 当需要多轮搬运时，需保证一次数据的搬运量需为单个数据size的整数倍
    u64 maxRoundTransferSize = params.maxTmpMemSize / MULTIPLY_TWO / rankSize_; // scratch分成2*ranksize份
    maxRoundTransferSize_ = maxRoundTransferSize;
    HCCL_DEBUG("[InsBatchSendRecvExecutor][Orchestrate] Max scratch buffer size [%u].", params.maxTmpMemSize);

    CHK_RET(CalcSendSlices(maxRoundTransferSize));
    CHK_RET(GenSendSlicesMapRank());

    CHK_RET(CalcRecvSlices(maxRoundTransferSize));
    CHK_RET(GenRecvSlicesMapRank());

    // aicpu mode
    CHK_RET(RunLoopSendRecv(op, requiredQue_, tempAlg));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::CalcResLinksPartialMesh(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const u32 linkNumBtwPeers,
    AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    u32 partialRankSize = commTargetUserRankSet_.size() + 1;

    if (tempVTopo.size() < 1) {
        HCCL_ERROR("[InsBatchSendRecvExecutor][CalcResLinksPartialMesh] Rank[%d], tempVTopo size is zero.", myRank);
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < tempVTopo.size(); i++) { // 遍历level0的2个平面
        CHK_RET(GetAlgRank(myRank, tempVTopo[i], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempResReq.queNum; queIdx++) {
            // find neighbors : virtualRank
            u32 remoteAlgRank = (myAlgRank + 1 + queIdx + partialRankSize) % partialRankSize;
            if (remoteAlgRank >= tempVTopo[i].size()) {
                continue;
            }
            RankId neighborRank = tempVTopo[i][remoteAlgRank];
            HCCL_DEBUG("tempVTopo[%u] index[%u] value[%d]", i, remoteAlgRank, neighborRank);
            auto rankInRankSet = std::find(
                commTargetUserRankSet_.begin(), commTargetUserRankSet_.end(), static_cast<u32>(neighborRank));
            if (rankInRankSet != commTargetUserRankSet_.end() && neighborRank != myRank) {
                // LinkNum
                tempResReq.links[neighborRank] = linkNumBtwPeers;
                HCCL_DEBUG("myRank[%d] neighborRank[%d] links is [%u]", myRank, neighborRank, linkNumBtwPeers);
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::CalcRes(AlgTempResReq& tempResReq)
{
    InsTempAllGatherMesh1D tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempResReq.queNum = commTargetUserRankSet_.size() + 1; // 使用n条从流
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = tempAlg.CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    CHK_RET(CalcResLinksPartialMesh(myRank_, vTopo_, 1, tempResReq));
    HCCL_DEBUG(
        "[InsBatchSendRecvExecutor][CalcRes] Rank[%d] vTopoSize[%lu] requiredQue Num[%u].", myRank_, vTopo_[0].size(),
        tempResReq.queNum);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 0;
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.SetTargetRanks(commTargetUserRankSet_));
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsBatchSendRecvExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[InsBatchSendRecvExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = commTargetUserRankSet_.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsBatchSendRecvExecutor<AlgTopoMatch>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.SetTargetRanks(commTargetUserRankSet_));
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    for (u32 i = 0; i < vTopo_.size(); i++) {        // 遍历level0
        for (u32 j = 0; j < vTopo_[i].size(); j++) { // 遍历平面内的所有rank
            HCCL_DEBUG(
                "[InsBatchSendRecvExecutor][CalcResLinksPartialMesh] vTopo_[%u][%u] is [%d].", i, j, vTopo_[i][j]);
        }
    }
    HCCL_DEBUG(
        "[InsBatchSendRecvExecutor][CalcRes]topoInfo.virtRanks[%zu], topoInfo.virtRankMap[%zu],"
        "topoInfo.vTopo[%zu]",
        algResReq.topoInfo.virtRanks.size(), algResReq.topoInfo.virtRankMap.size(), algResReq.topoInfo.vTopo.size());

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsBatchSendRecvExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[InsBatchSendRecvExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(CalcRes(tempResReq));
    }

    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    HCCL_DEBUG("[InsBatchSendRecvExecutor] Rank[%d], requiredQueNum [%u].", myRank_, algResReq.primQueueNum);

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));
    HCCL_DEBUG("[InsBatchSendRecvExecutor] Rank[%d], algResReq.links size[%zu].", myRank_, algResReq.links.size());

    return HcclResult::HCCL_SUCCESS;
}

// 注册
INS_REGISTER_IMPL_BY_TOPO(OpType::BATCHSENDRECV, InsBatchSendRecv, InsBatchSendRecvExecutor, TopoMatchPartialMesh);

} // namespace Hccl
