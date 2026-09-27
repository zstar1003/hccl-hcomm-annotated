/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_data_trans_wrapper.h"
#include "log.h"

namespace Hccl {
HcclResult
PreSyncQues(const std::vector<InsQuePtr>& syncQueues, const u32 postQueIdx, u32 topicId, bool enableCounterNotify)
{
    if (syncQueues.size() <= 1) {
        HCCL_WARNING(
            "[InsCollAlgFactory] [AlgDataTrans] PreSyncQues: syncQueues size [%zu], do nothing.", syncQueues.size());
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        postQueIdx >= syncQueues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] PreSyncQues: postQueIdx [%u] out of idx range for syncQueues [%zu].",
            postQueIdx, syncQueues.size()),
        HcclResult::HCCL_E_INTERNAL);

    if (enableCounterNotify) {
        std::unique_ptr<InsLocalBcastPost> insLocalBcastPost = std::make_unique<InsLocalBcastPost>(topicId);
        CHK_PTR_NULL(insLocalBcastPost);
        for (size_t queIdx = 0; queIdx < syncQueues.size(); queIdx++) {
            if (queIdx != postQueIdx) {
                insLocalBcastPost->Append(syncQueues[queIdx]->GetId()); // add queIdx to semaphore post
                std::unique_ptr<Instruction> insLocalWaitFrom
                    = std::make_unique<InsLocalWaitFrom>(syncQueues[postQueIdx]->GetId(), NotifyType::COUNTER);
                CHK_PTR_NULL(insLocalWaitFrom);
                syncQueues[queIdx]->Append(std::move(insLocalWaitFrom)); // semaphore wait
            }
        }
        syncQueues[postQueIdx]->Append(std::move(insLocalBcastPost)); // semaphore post
    } else {
        for (size_t queIdx = 0; queIdx < syncQueues.size(); queIdx++) {
            if (queIdx != postQueIdx) {
                // semaphore post
                std::unique_ptr<Instruction> insLocalPostTo
                    = std::make_unique<InsLocalPostTo>(syncQueues[queIdx]->GetId());
                CHK_PTR_NULL(insLocalPostTo);
                syncQueues[postQueIdx]->Append(std::move(insLocalPostTo));
                // semaphore wait
                std::unique_ptr<Instruction> insLocalWaitFrom
                    = std::make_unique<InsLocalWaitFrom>(syncQueues[postQueIdx]->GetId());
                CHK_PTR_NULL(insLocalWaitFrom);
                syncQueues[queIdx]->Append(std::move(insLocalWaitFrom));
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
PostSyncQues(const std::vector<InsQuePtr>& syncQueues, const u32 waitQueIdx, u32 topicId, bool enableCounterNotify)
{
    if (syncQueues.size() <= 1) {
        HCCL_WARNING(
            "[InsCollAlgFactory] [AlgDataTrans] PreSyncQues: syncQueues size [%zu], do nothing.", syncQueues.size());
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        waitQueIdx >= syncQueues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] PostSyncQues: waitQueIdx [%u] out of idx range for syncQueues [%zu].",
            waitQueIdx, syncQueues.size()),
        HcclResult::HCCL_E_INTERNAL);

    if (enableCounterNotify) {
        std::unique_ptr<InsLocalWaitGroup> insLocalWaitGroup = std::make_unique<InsLocalWaitGroup>(topicId);
        CHK_PTR_NULL(insLocalWaitGroup);
        for (size_t queIdx = 0; queIdx < syncQueues.size(); queIdx++) {
            if (queIdx != waitQueIdx) {
                insLocalWaitGroup->Append(syncQueues[queIdx]->GetId()); // add queIdx to semaphore wait

                std::unique_ptr<Instruction> insLocalPostTo
                    = std::make_unique<InsLocalPostTo>(syncQueues[waitQueIdx]->GetId(), NotifyType::COUNTER);
                CHK_PTR_NULL(insLocalPostTo);
                syncQueues[queIdx]->Append(std::move(insLocalPostTo)); // semaphore post
            }
        }
        syncQueues[waitQueIdx]->Append(std::move(insLocalWaitGroup)); // semaphore wait
    } else {
        for (size_t queIdx = 0; queIdx < syncQueues.size(); queIdx++) {
            if (queIdx != waitQueIdx) {
                // semaphore post
                std::unique_ptr<Instruction> insLocalPostTo
                    = std::make_unique<InsLocalPostTo>(syncQueues[waitQueIdx]->GetId());
                CHK_PTR_NULL(insLocalPostTo);
                syncQueues[queIdx]->Append(std::move(insLocalPostTo));
                // semaphore wait
                std::unique_ptr<Instruction> insLocalWaitFrom
                    = std::make_unique<InsLocalWaitFrom>(syncQueues[queIdx]->GetId());
                CHK_PTR_NULL(insLocalWaitFrom);
                syncQueues[waitQueIdx]->Append(std::move(insLocalWaitFrom));
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxReady(const LinkData& link, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsWaitReady>(link.GetRemoteRankId(), link));
    } else {
        queue->Append(std::make_unique<InsPostReady>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RxReady(const LinkData& link, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsPostReady>(link.GetRemoteRankId(), link));
    } else {
        queue->Append(std::make_unique<InsWaitReady>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxFin(const LinkData& link, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsPostFin>(link.GetRemoteRankId(), link));
    } else {
        queue->Append(std::make_unique<InsWaitFin>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RxFin(const LinkData& link, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    CHK_PTR_NULL(queue);
    if (mode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsWaitFin>(link.GetRemoteRankId(), link));
    } else {
        queue->Append(std::make_unique<InsPostFin>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxFinAck(const LinkData& link, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    (void)dmaMode;
    if ((link.GetType() == PortDeploymentType::DEV_NET) && (!DevCapability::GetInstance().IsSupportStarsPollNetCq())) {
        // DmaMode of DEV_NET can only be PUT
        queue->Append(std::make_unique<InsWaitFinAck>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RxFinAck(const LinkData& link, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    (void)dmaMode;
    if ((link.GetType() == PortDeploymentType::DEV_NET) && (!DevCapability::GetInstance().IsSupportStarsPollNetCq())) {
        // DmaMode of DEV_NET can only be PUT
        queue->Append(std::make_unique<InsPostFinAck>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxData(const LinkData& link, InsQuePtr queue, const SlicesList& slices, DmaMode dmaMode)
{
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices), DmaMode::PUT));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RxData(const LinkData& link, InsQuePtr queue, const SlicesList& slices, DmaMode dmaMode)
{
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::GET) {
        CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices), DmaMode::GET));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxReduce(const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, DmaMode dmaMode)
{
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices), DmaMode::PUT));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RxReduce(const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, DmaMode dmaMode)
{
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::GET) {
        CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices), DmaMode::GET));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxDataWithFin(const LinkData& link, InsQuePtr queue, const SlicesList& slices, u32 topicId, DmaMode dmaMode)
{
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        if (DevCapability::GetInstance().IsSupportWriteWithNotify()) {
            CHK_RET(WriteSlicesListsWithFin(link, queue, TransSlicesInfo(slices), topicId));
        } else {
            CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices),
                                     DmaMode::PUT)); // Write Data
            queue->Append(std::make_unique<InsPostFin>(link.GetRemoteRankId(), link));
        }
    } else {
        queue->Append(std::make_unique<InsWaitFin>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RxDataWithFin(const LinkData& link, InsQuePtr queue, const SlicesList& slices, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsWaitFin>(link.GetRemoteRankId(), link));
    } else {
        CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices), DmaMode::GET)); // Read Data
        queue->Append(std::make_unique<InsPostFin>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
TxReduceWithFin(const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, u32 topicId, DmaMode dmaMode)
{
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        if (DevCapability::GetInstance().IsSupportWriteWithNotify()) {
            CHK_RET(WriteSlicesListsWithFin(link, queue, TransSlicesInfo(slices), topicId));
        } else {
            CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices), DmaMode::PUT)); // WriteReduce Data
            queue->Append(std::make_unique<InsPostFin>(link.GetRemoteRankId(), link));
        }
    } else {
        queue->Append(std::make_unique<InsWaitFin>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
RxReduceWithFin(const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaMode, link.GetType(), mode));
    if (mode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsWaitFin>(link.GetRemoteRankId(), link));
    } else {
        CHK_RET(TransSlicesLists(link, queue, TransSlicesInfo(slices), DmaMode::GET)); // ReadReduce Data
        queue->Append(std::make_unique<InsPostFin>(link.GetRemoteRankId(), link));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiTxDataWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues, const std::vector<SlicesList>& slices,
    u32 topicId, DmaMode dmaMode)
{
    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportWriteWithNotify(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxDataWithFinCounter: inter-rank counterNotify is "
                   "supported only when the device supports WriteWithNotify."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxDataWithFinCounter: num of links [%zu] given non-equal "
            "with num of queues given [%zu].",
            links.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != slices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxDataWithFinCounter: num of links [%zu] given non-equal "
            "with num of slices given [%zu].",
            links.size(), slices.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = links.begin();
    auto queIter = queues.begin();
    auto sliceListIter = slices.begin();

    DmaMode mode;
    for (; linkIter != links.end(); linkIter++, queIter++, sliceListIter++) {
        CHK_RET(GetDMAMode(dmaMode, linkIter->GetType(), mode));
        CHK_PRT_RET(
            mode != DmaMode::PUT,
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxDataWithFinCounter: inter-rank counterNotify is "
                       "supported only in PUT MODE."),
            HcclResult::HCCL_E_INTERNAL);

        CHK_RET(WriteSlicesListsWithFin((*linkIter), (*queIter), TransSlicesInfo((*sliceListIter), true), topicId));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiRxDataWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues, const std::vector<SlicesList>& slices,
    u32 topicId, DmaMode dmaMode)
{
    (void)slices;
    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportWriteWithNotify(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxDataWithMultiFinCounter: inter-rank counterNotify is "
                   "supported only when the device supports WriteWithNotify."),
        HcclResult::HCCL_E_INTERNAL);

    std::unique_ptr<InsWaitGroupFin> insWaitGroupFin = std::make_unique<InsWaitGroupFin>(topicId);

    DmaMode mode;
    for (auto linkIter = links.begin(); linkIter != links.end(); linkIter++) {
        CHK_RET(GetDMAMode(dmaMode, linkIter->GetType(), mode));
        CHK_PRT_RET(
            mode != DmaMode::PUT,
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] RxDataWithMultiFinCounter: inter-rank counterNotify is "
                       "supported only in PUT MODE."),
            HcclResult::HCCL_E_INTERNAL);

        insWaitGroupFin->Append((*linkIter));
    }
    queues[0]->Append(std::move(insWaitGroupFin));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiTxReduceWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<ReduceSlicesList>& slices, u32 topicId, DmaMode dmaMode)
{
    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportWriteWithNotify(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxReduceWithFinCounter: inter-rank counterNotify is "
                   "supported only when the device supports WriteWithNotify."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxReduceWithFinCounter: num of links [%u] given non-equal "
            "with num of queues given [%u].",
            links.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != slices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxReduceWithFinCounter: num of links [%u] given non-equal "
            "with num of slices given [%u].",
            links.size(), slices.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = links.begin();
    auto queIter = queues.begin();
    auto sliceListIter = slices.begin();

    DmaMode mode;
    for (; linkIter != links.end(); linkIter++, queIter++, sliceListIter++) {
        CHK_RET(GetDMAMode(dmaMode, linkIter->GetType(), mode));
        CHK_PRT_RET(
            mode != DmaMode::PUT,
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxReduceWithFinCounter: inter-rank counterNotify is "
                       "supported only in PUT MODE."),
            HcclResult::HCCL_E_INTERNAL);

        CHK_RET(WriteSlicesListsWithFin((*linkIter), (*queIter), TransSlicesInfo((*sliceListIter), true), topicId));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiRxReduceWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<ReduceSlicesList>& slices, u32 topicId, DmaMode dmaMode)
{
    (void)slices;
    CHK_PRT_RET(
        queues.empty(), HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiRxReduceWithFinCounter: queue is empty"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(queues[0]);
    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportWriteWithNotify(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiRxReduceWithFinCounter: inter-rank counterNotify is "
                   "supported only when the device supports WriteWithNotify."),
        HcclResult::HCCL_E_INTERNAL);

    std::unique_ptr<InsWaitGroupFin> insWaitGroupFin = std::make_unique<InsWaitGroupFin>(topicId);

    DmaMode mode;
    for (auto linkIter = links.begin(); linkIter != links.end(); linkIter++) {
        CHK_RET(GetDMAMode(dmaMode, linkIter->GetType(), mode));
        CHK_PRT_RET(
            mode != DmaMode::PUT,
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiRxReduceWithFinCounter: inter-rank counterNotify is "
                       "supported only in PUT MODE."),
            HcclResult::HCCL_E_INTERNAL);

        insWaitGroupFin->Append((*linkIter));
    }
    queues[0]->Append(std::move(insWaitGroupFin));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxRxReady(const TxRxLinks& txRxlinks, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode txMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.txLink_.GetType(), txMode));
    DmaMode rxMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.rxLink_.GetType(), rxMode));
    CHK_PRT_RET(
        txMode != rxMode,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] TxRxReady: DmaMode of txLink inconsistent with rxLink."),
        HcclResult::HCCL_E_INTERNAL);

    if (txMode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsPostReady>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_));
        queue->Append(std::make_unique<InsWaitReady>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_));
    } else {
        queue->Append(std::make_unique<InsPostReady>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_));
        queue->Append(std::make_unique<InsWaitReady>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxRxFin(const TxRxLinks& txRxlinks, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    DmaMode txMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.txLink_.GetType(), txMode));
    DmaMode rxMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.rxLink_.GetType(), rxMode));
    CHK_PRT_RET(
        txMode != rxMode,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] TxRxFin: DmaMode of txLink inconsistent with rxLink."),
        HcclResult::HCCL_E_INTERNAL);

    if (txMode == DmaMode::PUT) {
        queue->Append(std::make_unique<InsPostFin>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_));
        queue->Append(std::make_unique<InsWaitFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_));
    } else {
        queue->Append(std::make_unique<InsPostFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_));
        queue->Append(std::make_unique<InsWaitFin>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxRxFinAck(const TxRxLinks& txRxlinks, InsQuePtr queue, u32 topicId, DmaMode dmaMode)
{
    (void)topicId;
    if (!DevCapability::GetInstance().IsSupportStarsPollNetCq()) {
        bool isTxLinkNet = txRxlinks.txLink_.GetType() == PortDeploymentType::DEV_NET;
        bool isRxLinkNet = txRxlinks.rxLink_.GetType() == PortDeploymentType::DEV_NET;
        if (isTxLinkNet && isRxLinkNet) {
            queue->Append(std::make_unique<InsPostFinAck>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_));
            queue->Append(std::make_unique<InsWaitFinAck>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_));
        } else if (isTxLinkNet) {
            DmaMode mode;
            CHK_RET(GetDMAMode(dmaMode, txRxlinks.rxLink_.GetType(), mode));
            CHK_PRT_RET(
                mode != DmaMode::PUT,
                HCCL_ERROR(
                    "[InsCollAlgFactory] [AlgDataTrans] TxRxFinAck: DmaMode of txLink inconsistent with rxLink."),
                HcclResult::HCCL_E_INTERNAL);
            queue->Append(std::make_unique<InsWaitFinAck>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_));
        } else if (isRxLinkNet) {
            DmaMode mode;
            CHK_RET(GetDMAMode(dmaMode, txRxlinks.txLink_.GetType(), mode));
            CHK_PRT_RET(
                mode != DmaMode::PUT,
                HCCL_ERROR(
                    "[InsCollAlgFactory] [AlgDataTrans] TxRxFinAck: DmaMode of txLink inconsistent with rxLink."),
                HcclResult::HCCL_E_INTERNAL);
            queue->Append(std::make_unique<InsPostFinAck>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxRxData(const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxSlicesList& txRxSlices, DmaMode dmaMode)
{
    DmaMode txMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.txLink_.GetType(), txMode));
    DmaMode rxMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.rxLink_.GetType(), rxMode));
    CHK_PRT_RET(
        txMode != rxMode,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] TxRxData: DmaMode of txLink inconsistent with rxLink."),
        HcclResult::HCCL_E_INTERNAL);

    if (txMode == DmaMode::PUT) {
        TransSlicesInfo transSlicesInfo = TransSlicesInfo(txRxSlices.txSlicesList_);
        CHK_RET(TransSlicesLists(txRxlinks.txLink_, queue, transSlicesInfo, DmaMode::PUT));
    } else {
        TransSlicesInfo transSlicesInfo = TransSlicesInfo(txRxSlices.rxSlicesList_);
        CHK_RET(TransSlicesLists(txRxlinks.rxLink_, queue, transSlicesInfo, DmaMode::GET));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
TxRxReduce(const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxReduceSlicesList& txRxSlices, DmaMode dmaMode)
{
    DmaMode txMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.txLink_.GetType(), txMode));
    DmaMode rxMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.rxLink_.GetType(), rxMode));
    CHK_PRT_RET(
        txMode != rxMode,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] TxRxReduce: DmaMode of txLink inconsistent with rxLink."),
        HcclResult::HCCL_E_INTERNAL);

    if (txMode == DmaMode::PUT) {
        CHK_RET(TransSlicesLists(
            txRxlinks.txLink_, queue,
            TransSlicesInfo(txRxSlices.txSlicesList_, txRxSlices.dataType_, txRxSlices.reduceOp_), DmaMode::PUT));
    } else {
        CHK_RET(TransSlicesLists(
            txRxlinks.rxLink_, queue,
            TransSlicesInfo(txRxSlices.rxSlicesList_, txRxSlices.dataType_, txRxSlices.reduceOp_), DmaMode::GET));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxRxDataWithFin(
    const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxSlicesList& txRxSlices, u32 topicId, DmaMode dmaMode)
{
    CHK_PTR_NULL(queue);
    DmaMode txMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.txLink_.GetType(), txMode));
    DmaMode rxMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.rxLink_.GetType(), rxMode));
    CHK_PRT_RET(
        txMode != rxMode,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] TxRxReduce: DmaMode of txLink inconsistent with rxLink."),
        HcclResult::HCCL_E_INTERNAL);
    if (txMode == DmaMode::PUT) {
        if (DevCapability::GetInstance().IsSupportWriteWithNotify()) {
            CHK_RET(WriteSlicesListsWithFin(
                txRxlinks.txLink_, queue, TransSlicesInfo(txRxSlices.txSlicesList_),
                topicId)); // write + postFin

            queue->Append(
                std::make_unique<InsWaitFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_)); // waitFin
        } else {
            CHK_RET(TransSlicesLists(
                txRxlinks.txLink_, queue, TransSlicesInfo(txRxSlices.txSlicesList_),
                DmaMode::PUT)); // write data
            queue->Append(
                std::make_unique<InsPostFin>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_)); // postFin
            queue->Append(
                std::make_unique<InsWaitFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_)); // waitFin
        }
    } else {
        CHK_RET(TransSlicesLists(
            txRxlinks.rxLink_, queue, TransSlicesInfo(txRxSlices.rxSlicesList_),
            DmaMode::GET)); // read data
        queue->Append(std::make_unique<InsPostFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_)); // postFin
        queue->Append(std::make_unique<InsWaitFin>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_)); // waitFin
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TxRxReduceWithFin(
    const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxReduceSlicesList& txRxSlices, u32 topicId, DmaMode dmaMode)
{
    DmaMode txMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.txLink_.GetType(), txMode));
    DmaMode rxMode;
    CHK_RET(GetDMAMode(dmaMode, txRxlinks.rxLink_.GetType(), rxMode));
    CHK_PRT_RET(
        txMode != rxMode,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] TxRxReduceWithFin: DmaMode of txLink inconsistent with rxLink."),
        HcclResult::HCCL_E_INTERNAL);

    if (txMode == DmaMode::PUT) {
        TransSlicesInfo transSlicesInfo
            = TransSlicesInfo(txRxSlices.txSlicesList_, txRxSlices.dataType_, txRxSlices.reduceOp_);

        if (DevCapability::GetInstance().IsSupportWriteWithNotify()) {
            CHK_RET(WriteSlicesListsWithFin(txRxlinks.txLink_, queue, transSlicesInfo, topicId)); // write + postFin

            queue->Append(
                std::make_unique<InsWaitFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_)); // waitFin
        } else {
            CHK_RET(TransSlicesLists(txRxlinks.txLink_, queue, transSlicesInfo, DmaMode::PUT)); // writeReduce data
            queue->Append(
                std::make_unique<InsPostFin>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_)); // postFin
            queue->Append(
                std::make_unique<InsWaitFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_)); // waitFin
        }
    } else {
        CHK_RET(TransSlicesLists(
            txRxlinks.rxLink_, queue,
            TransSlicesInfo(txRxSlices.rxSlicesList_, txRxSlices.dataType_, txRxSlices.reduceOp_),
            DmaMode::GET)); // readReduce data
        queue->Append(std::make_unique<InsPostFin>(txRxlinks.rxLink_.GetRemoteRankId(), txRxlinks.rxLink_)); // postFin
        queue->Append(std::make_unique<InsWaitFin>(txRxlinks.txLink_.GetRemoteRankId(), txRxlinks.txLink_)); // waitFin
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiTxRxDataWithFinCounter(
    const std::vector<TxRxLinks>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<TxRxSlicesList>& slices, u32 topicId, DmaMode dmaMode)
{
    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportWriteWithNotify(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxRxDataWithFinCounter: inter-rank counterNotify is "
                   "supported only when the device supports WriteWithNotify."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxRxDataWithFinCounter: num of links [%u] given non-equal "
            "with num of queues given [%u].",
            links.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != slices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxRxDataWithFinCounter: num of links [%zu] given non-equal "
            "with num of slices given [%zu].",
            links.size(), slices.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto txRxLinkIter = links.begin();
    auto queIter = queues.begin();
    auto sliceListIter = slices.begin();

    DmaMode txMode;
    DmaMode rxMode;
    std::unique_ptr<InsWaitGroupFin> insWaitGroupFin = std::make_unique<InsWaitGroupFin>(topicId);
    for (; txRxLinkIter != links.end(); txRxLinkIter++, queIter++, sliceListIter++) {
        CHK_RET(GetDMAMode(dmaMode, (*txRxLinkIter).txLink_.GetType(), txMode));
        CHK_RET(GetDMAMode(dmaMode, (*txRxLinkIter).rxLink_.GetType(), rxMode));
        CHK_PRT_RET(
            ((txMode != DmaMode::PUT) || (rxMode != DmaMode::PUT)),
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxRxDataWithFinCounter: inter-rank counterNotify is "
                       "supported only in PUT MODE."),
            HcclResult::HCCL_E_INTERNAL);

        TransSlicesInfo transSlicesInfo = TransSlicesInfo((*sliceListIter).txSlicesList_, true);
        CHK_RET(WriteSlicesListsWithFin((*txRxLinkIter).txLink_, (*queIter), transSlicesInfo, topicId));

        insWaitGroupFin->Append((*txRxLinkIter).rxLink_);
    }

    queues[0]->Append(std::move(insWaitGroupFin));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiTxRxReduceWithFinCounter(
    const std::vector<TxRxLinks>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<TxRxReduceSlicesList>& slices, u32 topicId, DmaMode dmaMode)
{
    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportWriteWithNotify(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxRxReduceWithFinCounter: inter-rank counterNotify is "
                   "supported only when the device supports WriteReduceWithNotify."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxRxReduceWithFinCounter: num of links [%u] given non-equal "
            "with num of queues given [%u].",
            links.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        links.size() != slices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiTxRxReduceWithFinCounter: num of links [%u] given non-equal "
            "with num of slices given [%u].",
            links.size(), slices.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto txRxLinkIter = links.begin();
    auto queIter = queues.begin();
    auto sliceListIter = slices.begin();

    DmaMode txMode;
    DmaMode rxMode;
    std::unique_ptr<InsWaitGroupFin> insWaitGroupFin = std::make_unique<InsWaitGroupFin>(topicId);
    for (; txRxLinkIter != links.end(); txRxLinkIter++, queIter++, sliceListIter++) {
        CHK_RET(GetDMAMode(dmaMode, (*txRxLinkIter).txLink_.GetType(), txMode));
        CHK_RET(GetDMAMode(dmaMode, (*txRxLinkIter).rxLink_.GetType(), rxMode));
        CHK_PRT_RET(
            ((txMode != DmaMode::PUT) || (rxMode != DmaMode::PUT)),
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiTxRxReduceWithFinCounter: inter-rank counterNotify "
                       "is supported only in PUT MODE."),
            HcclResult::HCCL_E_INTERNAL);

        TransSlicesInfo transSlicesInfo
            = TransSlicesInfo(sliceListIter->txSlicesList_, sliceListIter->dataType_, sliceListIter->reduceOp_, true);
        CHK_RET(WriteSlicesListsWithFin((*txRxLinkIter).txLink_, (*queIter), transSlicesInfo, topicId));

        insWaitGroupFin->Append((*txRxLinkIter).rxLink_);
    }

    queues[0]->Append(std::move(insWaitGroupFin));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult LocalReduce(
    InsQuePtr queue, const DataSlice& srcSlice, const DataSlice& dstSlice, const DataType dataType,
    const ReduceOp reduceOp)
{
    CHK_PRT_RET(
        srcSlice.GetSize() != dstSlice.GetSize(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] LocalReduce: src slice size [%zu] is not equal to dst slice size "
            "[%zu].",
            srcSlice.GetSize(), dstSlice.GetSize()),
        HcclResult::HCCL_E_INTERNAL);

    std::unique_ptr<InsLocalReduce> insLocalReduce
        = std::make_unique<InsLocalReduce>(srcSlice, dstSlice, dataType, reduceOp);
    queue->Append(std::move(insLocalReduce));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult LocalReduceSlices(
    InsQuePtr queue, const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices,
    const DataType dataType, const ReduceOp reduceOp)
{
    CHK_PRT_RET(
        srcSlices.size() != dstSlices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] LocalReduceSlices: num of src slices [%zu], is not equal "
            "to num of dst slices [%zu].",
            srcSlices.size(), dstSlices.size()),
        HcclResult::HCCL_E_INTERNAL);

    // tmpSlices: slices to be transfer in this loop
    DataSlice tmpSrcSlice = srcSlices[0];
    DataSlice tmpDstSlice = dstSlices[0];

    for (u32 sliceIdx = 0; sliceIdx < srcSlices.size(); sliceIdx++) {
        CHK_PRT_RET(
            srcSlices[sliceIdx].GetSize() != dstSlices[sliceIdx].GetSize(),
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTrans] LocalReduceSlices: [%zu]-th slice, src slice size [%zu] "
                "is not equal to dst slice size [%zu].",
                sliceIdx, srcSlices[sliceIdx].GetSize(), dstSlices[sliceIdx].GetSize()),
            HcclResult::HCCL_E_INTERNAL);
        try {
            if (sliceIdx == (srcSlices.size() - 1)) {
                // last slice
                std::unique_ptr<InsLocalReduce> insLocalReduce
                    = std::make_unique<InsLocalReduce>(tmpSrcSlice, tmpDstSlice, dataType, reduceOp);
                queue->Append(std::move(insLocalReduce));
            } else if (
                IsContinuousSlice(srcSlices[sliceIdx + 1], tmpSrcSlice)
                && IsContinuousSlice(dstSlices[sliceIdx + 1], tmpDstSlice)) {
                // nxtSlice is continuous with tmpSlice, update tmpSlice
                u64 newTmpSize = tmpSrcSlice.GetSize() + srcSlices[sliceIdx + 1].GetSize();
                tmpSrcSlice = DataSlice(tmpSrcSlice.GetType(), tmpSrcSlice.GetOffset(), newTmpSize);
                tmpDstSlice = DataSlice(tmpDstSlice.GetType(), tmpDstSlice.GetOffset(), newTmpSize);
            } else {
                // nxtSlice is not continuous with tmpSlice, copy tmpSlice, update tmpSlice with nxtSlice
                std::unique_ptr<InsLocalReduce> insLocalReduce
                    = std::make_unique<InsLocalReduce>(tmpSrcSlice, tmpDstSlice, dataType, reduceOp);
                queue->Append(std::move(insLocalReduce));

                tmpSrcSlice = srcSlices[sliceIdx + 1];
                tmpDstSlice = dstSlices[sliceIdx + 1];
            }
        } catch (const std::bad_alloc& e) {
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] LocalReduceSlices: memory allocation failed");
            return HcclResult::HCCL_E_MEMORY;
        } catch (const std::exception& e) {
            HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] LocalReduceSlices: exception occurred - %s", e.what());
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult LocalCopy(InsQuePtr queue, const DataSlice& srcSlice, const DataSlice& dstSlice)
{
    CHK_PRT_RET(
        srcSlice.GetSize() != dstSlice.GetSize(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] LocalCopy: src slice size [%zu] is not equal to dst slice size [%zu].",
            srcSlice.GetSize(), dstSlice.GetSize()),
        HcclResult::HCCL_E_INTERNAL);

    std::unique_ptr<InsLocalCopy> insLocalCopy = std::make_unique<InsLocalCopy>(srcSlice, dstSlice);
    queue->Append(std::move(insLocalCopy));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
LocalCopySlices(InsQuePtr queue, const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices)
{
    CHK_PRT_RET(
        srcSlices.size() != dstSlices.size(),
        HCCL_ERROR(
            "[AlgDataTransWrapper] [AlgDataTrans] LocalCopySlices: num of src slices [%u], is not equal "
            "to num of dst slices [%u].",
            srcSlices.size(), dstSlices.size()),
        HcclResult::HCCL_E_INTERNAL);

    // tmpSlices: slices to be transfer in this loop
    DataSlice tmpSrcSlice = srcSlices[0];
    DataSlice tmpDstSlice = dstSlices[0];

    for (u32 sliceIdx = 0; sliceIdx < srcSlices.size(); sliceIdx++) {
        CHK_PRT_RET(
            srcSlices[sliceIdx].GetSize() != dstSlices[sliceIdx].GetSize(),
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTrans] LocalCopySlices: [%u]-th slice, src slice size [%zu] "
                "is not equal to dst slice size [%zu].",
                sliceIdx, srcSlices[sliceIdx].GetSize(), dstSlices[sliceIdx].GetSize()),
            HcclResult::HCCL_E_INTERNAL);

        if (sliceIdx == (srcSlices.size() - 1)) {
            // last slice
            std::unique_ptr<InsLocalCopy> insLocalCopy = std::make_unique<InsLocalCopy>(tmpSrcSlice, tmpDstSlice);
            queue->Append(std::move(insLocalCopy));
        } else if (
            IsContinuousSlice(srcSlices[sliceIdx + 1], tmpSrcSlice)
            && IsContinuousSlice(dstSlices[sliceIdx + 1], tmpDstSlice)) {
            // nxtSlice is continuous with tmpSlice, update tmpSlice
            u64 newTmpSize = tmpSrcSlice.GetSize() + srcSlices[sliceIdx + 1].GetSize();
            tmpSrcSlice = DataSlice(tmpSrcSlice.GetType(), tmpSrcSlice.GetOffset(), newTmpSize);
            tmpDstSlice = DataSlice(tmpDstSlice.GetType(), tmpDstSlice.GetOffset(), newTmpSize);
        } else {
            // nxtSlice is not continuous with tmpSlice, copy tmpSlice, update tmpSlice with nxtSlice
            std::unique_ptr<InsLocalCopy> insLocalCopy = std::make_unique<InsLocalCopy>(tmpSrcSlice, tmpDstSlice);
            queue->Append(std::move(insLocalCopy));

            tmpSrcSlice = srcSlices[sliceIdx + 1];
            tmpDstSlice = dstSlices[sliceIdx + 1];
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult StreamSync(std::vector<InsQuePtr>& queues)
{
    CHK_PRT_RET(
        queues.empty(), HCCL_ERROR("[alg_data_trans_wrapper_mid][StreamSync] empty queue"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(queues[0]);
    for (auto& queue : queues) {
        std::unique_ptr<InsPreStreamSync> insPreStreamSync = std::make_unique<InsPreStreamSync>();
        queue->Append(std::move(insPreStreamSync));
    }
    std::unique_ptr<InsStreamSync> insStreamSync = std::make_unique<InsStreamSync>();
    queues[0]->Append(std::move(insStreamSync));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AicpuReduce(
    InsQuePtr queue, const DataSlice& srcSlice, const DataSlice& dstSlice, const DataType dataType,
    const ReduceOp reduceOp)
{
    CHK_PRT_RET(
        srcSlice.GetSize() != dstSlice.GetSize(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] AicpuReduce: src slice size [%zu] is not equal to dst slice size "
            "[%zu].",
            srcSlice.GetSize(), dstSlice.GetSize()),
        HcclResult::HCCL_E_INTERNAL);

    std::unique_ptr<InsAicpuReduce> insAicpuReduce
        = std::make_unique<InsAicpuReduce>(srcSlice, dstSlice, dataType, reduceOp);
    queue->Append(std::move(insAicpuReduce));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AicpuReduceSlices(
    InsQuePtr queue, const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices,
    const DataType dataType, const ReduceOp reduceOp)
{
    CHK_PRT_RET(
        srcSlices.size() != dstSlices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] AicpuReduceSlices: num of src slices [%zu], is not equal "
            "to num of dst slices [%zu].",
            srcSlices.size(), dstSlices.size()),
        HcclResult::HCCL_E_INTERNAL);

    // tmpSlices: slices to be transfer in this loop
    DataSlice tmpSrcSlice = srcSlices[0];
    DataSlice tmpDstSlice = dstSlices[0];

    for (u32 sliceIdx = 0; sliceIdx < srcSlices.size(); sliceIdx++) {
        CHK_PRT_RET(
            srcSlices[sliceIdx].GetSize() != dstSlices[sliceIdx].GetSize(),
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTrans] AicpuReduceSlices: [%u]-th slice, src slice size [%zu] "
                "is not equal to dst slice size [%zu].",
                sliceIdx, srcSlices[sliceIdx].GetSize(), dstSlices[sliceIdx].GetSize()),
            HcclResult::HCCL_E_INTERNAL);

        if (sliceIdx == (srcSlices.size() - 1)) {
            // last slice
            std::unique_ptr<InsAicpuReduce> insAicpuReduce
                = std::make_unique<InsAicpuReduce>(tmpSrcSlice, tmpDstSlice, dataType, reduceOp);
            queue->Append(std::move(insAicpuReduce));
        } else if (
            IsContinuousSlice(srcSlices[sliceIdx + 1], tmpSrcSlice)
            && IsContinuousSlice(dstSlices[sliceIdx + 1], tmpDstSlice)) {
            // nxtSlice is continuous with tmpSlice, update tmpSlice
            u64 newTmpSize = tmpSrcSlice.GetSize() + srcSlices[sliceIdx + 1].GetSize();
            tmpSrcSlice = DataSlice(tmpSrcSlice.GetType(), tmpSrcSlice.GetOffset(), newTmpSize);
            tmpDstSlice = DataSlice(tmpDstSlice.GetType(), tmpDstSlice.GetOffset(), newTmpSize);
        } else {
            // nxtSlice is not continuous with tmpSlice, copy tmpSlice, update tmpSlice with nxtSlice
            std::unique_ptr<InsAicpuReduce> insAicpuReduce
                = std::make_unique<InsAicpuReduce>(tmpSrcSlice, tmpDstSlice, dataType, reduceOp);
            queue->Append(std::move(insAicpuReduce));

            tmpSrcSlice = srcSlices[sliceIdx + 1];
            tmpDstSlice = dstSlices[sliceIdx + 1];
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
