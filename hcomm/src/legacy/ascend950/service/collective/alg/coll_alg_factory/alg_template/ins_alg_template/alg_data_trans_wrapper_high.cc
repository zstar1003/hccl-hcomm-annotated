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
HcclResult Send(const DataInfo& sendInfo, InsQuePtr queue, u32 topicId, bool needNetFinAck, DmaMode dmaMode)
{
    CHK_RET(TxReady(sendInfo.link_, queue, topicId, dmaMode));
    CHK_RET(TxDataWithFin(sendInfo.link_, queue, sendInfo.slices_, topicId, dmaMode));
    if (needNetFinAck) {
        CHK_RET(TxFinAck(sendInfo.link_, queue, topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Recv(const DataInfo& recvInfo, InsQuePtr queue, u32 topicId, bool needNetFinAck, DmaMode dmaMode)
{
    CHK_RET(RxReady(recvInfo.link_, queue, topicId, dmaMode));
    CHK_RET(RxDataWithFin(recvInfo.link_, queue, recvInfo.slices_, topicId, dmaMode));
    if (needNetFinAck) {
        CHK_RET(RxFinAck(recvInfo.link_, queue, topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult SendRecv(const SendRecvInfo& sendRecvInfo, InsQuePtr queue, u32 topicId, bool needNetFinAck, DmaMode dmaMode)
{
    CHK_RET(TxRxReady(sendRecvInfo.sendRecvLinks_, queue, topicId, dmaMode));
    CHK_RET(TxRxDataWithFin(sendRecvInfo.sendRecvLinks_, queue, sendRecvInfo.sendRecvSlices_, topicId, dmaMode));
    if (needNetFinAck) {
        CHK_RET(TxRxFinAck(sendRecvInfo.sendRecvLinks_, queue, topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
SendReduce(const DataReduceInfo& sendReduceInfo, InsQuePtr queue, u32 topicId, bool needNetFinAck, DmaMode dmaMode)
{
    CHK_RET(TxReady(sendReduceInfo.link_, queue, topicId, dmaMode));
    CHK_RET(TxReduceWithFin(
        sendReduceInfo.link_, queue, {sendReduceInfo.slices_, sendReduceInfo.dataType_, sendReduceInfo.reduceOp_},
        topicId, dmaMode));
    if (needNetFinAck) {
        CHK_RET(TxFinAck(sendReduceInfo.link_, queue, topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
RecvReduce(const DataReduceInfo& recvReduceInfo, InsQuePtr queue, u32 topicId, bool needNetFinAck, DmaMode dmaMode)
{
    CHK_RET(RxReady(recvReduceInfo.link_, queue, topicId, dmaMode));
    CHK_RET(RxReduceWithFin(
        recvReduceInfo.link_, queue, {recvReduceInfo.slices_, recvReduceInfo.dataType_, recvReduceInfo.reduceOp_},
        topicId, dmaMode));
    if (needNetFinAck) {
        CHK_RET(RxFinAck(recvReduceInfo.link_, queue, topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult SendRecvReduce(
    const SendRecvReduceInfo& sendRecvReduceInfo, InsQuePtr queue, u32 topicId, bool needNetFinAck, DmaMode dmaMode)
{
    CHK_RET(TxRxReady(sendRecvReduceInfo.sendRecvLinks_, queue, topicId, dmaMode));
    CHK_RET(TxRxReduceWithFin(
        sendRecvReduceInfo.sendRecvLinks_, queue,
        {sendRecvReduceInfo.sendRecvSlices_, sendRecvReduceInfo.dataType_, sendRecvReduceInfo.reduceOp_}, topicId,
        dmaMode));
    if (needNetFinAck) {
        CHK_RET(TxRxFinAck(sendRecvReduceInfo.sendRecvLinks_, queue, topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiSendCounter(const MultiDataInfo& sendInfo, std::vector<InsQuePtr>& queues, u32 topicId, DmaMode dmaMode)
{
    if (sendInfo.links_.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] MultiSendCounter: link size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportStarsPollNetCq(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiSendCounter: inter-rank CounterNotify is "
                   "supported only when device supports StarsPollNetCq."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (sendInfo.links_.size() != queues.size()) || (sendInfo.slices_.size() != queues.size()),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiSendCounter: invalid input with link num [%zu], "
            "slice num [%zu], queue num [%zu].",
            sendInfo.links_.size(), sendInfo.slices_.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = sendInfo.links_.begin();
    auto queIter = queues.begin();

    for (; linkIter != sendInfo.links_.end(); linkIter++, queIter++) {
        CHK_RET(TxReady((*linkIter), (*queIter), topicId, dmaMode));
    }

    CHK_RET(MultiTxDataWithFinCounter(sendInfo.links_, queues, sendInfo.slices_, topicId, dmaMode));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiRecvCounter(const MultiDataInfo& recvInfo, std::vector<InsQuePtr>& queues, u32 topicId, DmaMode dmaMode)
{
    if (recvInfo.links_.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] MultiRecvCounter: link size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportStarsPollNetCq(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiRecvCounter: inter-rank CounterNotify is "
                   "supported only when device supports StarsPollNetCq."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (recvInfo.links_.size() != queues.size()) || (recvInfo.slices_.size() != queues.size()),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiRecvCounter: invalid input with link num [%zu], "
            "slice num [%zu], queue num [%zu].",
            recvInfo.links_.size(), recvInfo.slices_.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = recvInfo.links_.begin();
    auto queIter = queues.begin();

    for (; linkIter != recvInfo.links_.end(); linkIter++, queIter++) {
        CHK_RET(RxReady((*linkIter), (*queIter), topicId, dmaMode));
    }

    CHK_RET(MultiRxDataWithFinCounter(recvInfo.links_, queues, recvInfo.slices_, topicId, dmaMode));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiSendRecvCounter(
    const MultiSendRecvInfo& sendRecvInfo, std::vector<InsQuePtr>& queues, u32 topicId, DmaMode dmaMode)
{
    if (sendRecvInfo.txRxLinks_.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] MultiSendRecvCounter: link size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportStarsPollNetCq(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiSendRecvCounter: inter-rank CounterNotify is "
                   "supported only when device supports StarsPollNetCq."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (sendRecvInfo.txRxLinks_.size() != queues.size()) || (sendRecvInfo.txRxSlices_.size() != queues.size()),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiSendRecvCounter: invalid input with link num [%zu], "
            "slice num [%zu], queue num [%zu].",
            sendRecvInfo.txRxLinks_.size(), sendRecvInfo.txRxSlices_.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = sendRecvInfo.txRxLinks_.begin();
    auto queIter = queues.begin();

    for (; linkIter != sendRecvInfo.txRxLinks_.end(); linkIter++, queIter++) {
        CHK_RET(TxRxReady((*linkIter), (*queIter), topicId, dmaMode));
    }

    CHK_RET(MultiTxRxDataWithFinCounter(sendRecvInfo.txRxLinks_, queues, sendRecvInfo.txRxSlices_, topicId, dmaMode));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiSendReduceCounter(
    const MultiDataReduceInfo& sendInfo, std::vector<InsQuePtr>& queues, u32 topicId, DmaMode dmaMode)
{
    if (sendInfo.links_.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] MultiSendReduceCounter: link size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportStarsPollNetCq(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiSendReduceCounter: inter-rank CounterNotify is "
                   "supported only when device supports StarsPollNetCq."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (sendInfo.links_.size() != queues.size()) || (sendInfo.slices_.size() != queues.size()),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiSendReduceCounter: invalid input with link num [%zu], "
            "slice num [%zu], queue num [%zu].",
            sendInfo.links_.size(), sendInfo.slices_.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = sendInfo.links_.begin();
    auto queIter = queues.begin();

    for (; linkIter != sendInfo.links_.end(); linkIter++, queIter++) {
        CHK_RET(TxReady((*linkIter), (*queIter), topicId, dmaMode));
    }

    CHK_RET(MultiTxReduceWithFinCounter(sendInfo.links_, queues, sendInfo.slices_, topicId, dmaMode));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiRecvReduceCounter(
    const MultiDataReduceInfo& recvInfo, std::vector<InsQuePtr>& queues, u32 topicId, DmaMode dmaMode)
{
    if (recvInfo.links_.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] MultiRecvReduceCounter: link size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportStarsPollNetCq(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiRecvReduceCounter: inter-rank CounterNotify is "
                   "supported only when device supports StarsPollNetCq."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (recvInfo.links_.size() != queues.size()) || (recvInfo.slices_.size() != queues.size()),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiRecvReduceCounter: invalid input with link num [%zu], "
            "slice num [%zu], queue num [%zu].",
            recvInfo.links_.size(), recvInfo.slices_.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = recvInfo.links_.begin();
    auto queIter = queues.begin();

    for (; linkIter != recvInfo.links_.end(); linkIter++, queIter++) {
        CHK_RET(RxReady((*linkIter), (*queIter), topicId, dmaMode));
    }

    CHK_RET(MultiRxReduceWithFinCounter(recvInfo.links_, queues, recvInfo.slices_, topicId, dmaMode));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MultiSendRecvReduceCounter(
    const MultiSendRecvReduceInfo& sendRecvInfo, std::vector<InsQuePtr>& queues, u32 topicId, DmaMode dmaMode)
{
    if (sendRecvInfo.txRxLinks_.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] MultiSendRecvReduceCounter: link size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        !DevCapability::GetInstance().IsSupportStarsPollNetCq(),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] MultiSendRecvReduceCounter: inter-rank CounterNotify is "
                   "supported only when device supports StarsPollNetCq."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (sendRecvInfo.txRxLinks_.size() != queues.size()) || (sendRecvInfo.txRxSlices_.size() != queues.size()),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] MultiSendRecvReduceCounter: invalid input with link num [%zu], "
            "slice num [%zu], queue num [%zu].",
            sendRecvInfo.txRxLinks_.size(), sendRecvInfo.txRxSlices_.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto linkIter = sendRecvInfo.txRxLinks_.begin();
    auto queIter = queues.begin();

    for (; linkIter != sendRecvInfo.txRxLinks_.end(); linkIter++, queIter++) {
        CHK_RET(TxRxReady((*linkIter), (*queIter), topicId, dmaMode));
    }

    CHK_RET(MultiTxRxReduceWithFinCounter(sendRecvInfo.txRxLinks_, queues, sendRecvInfo.txRxSlices_, topicId, dmaMode));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult SendThruMultiLinks(
    const std::vector<DataInfo>& sendInfo, std::vector<InsQuePtr>& queues, u32 topicId, bool needNetFinAck,
    DmaMode dmaMode)
{
    if (sendInfo.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] SendThruMultiLinks: sendInfo size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        sendInfo.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] SendThruMultiLinks: sendInfo size [%zu] is non-equal to queue num "
            "[%zu].",
            sendInfo.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    // only those worker queues required to be sync: put mode in send
    std::vector<InsQuePtr> syncQues = {queues[0]};
    bool hasDiffDmaMode = false;

    CHK_RET(ProceedMultiLinks(
        sendInfo, queues, MultiDataLinksDmaModeInfo(DmaMode::PUT, dmaMode), syncQues, hasDiffDmaMode));

    if (hasDiffDmaMode) {
        HCCL_DEBUG("[InsCollAlgFactory] [AlgDataTrans] SendThruMultiLinks: current send links have two DmaMode.");
        CHK_RET(TxRxReady({sendInfo[0].link_, sendInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(TxReady(sendInfo[0].link_, queues[0], topicId, dmaMode));
    }

    CHK_RET(PreSyncQues(syncQues, 0));

    auto dataInfoIter = sendInfo.begin();
    auto queIter = queues.begin();
    for (; dataInfoIter != sendInfo.end(); dataInfoIter++, queIter++) {
        if (std::find(syncQues.begin(), syncQues.end(), (*queIter)) != syncQues.end()) {
            CHK_RET(TxData(dataInfoIter->link_, (*queIter), dataInfoIter->slices_, dmaMode));
        }
    }

    CHK_RET(PostSyncQues(syncQues, 0));

    if (hasDiffDmaMode) {
        CHK_RET(TxRxFin({sendInfo[0].link_, sendInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(TxFin(sendInfo[0].link_, queues[0], topicId, dmaMode));
    }

    if (needNetFinAck) {
        TxFinAck(sendInfo[0].link_, queues[0], topicId, dmaMode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult RecvThruMultiLinks(
    const std::vector<DataInfo>& recvInfo, std::vector<InsQuePtr>& queues, u32 topicId, bool needNetFinAck,
    DmaMode dmaMode)
{
    if (recvInfo.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] RecvThruMultiLinks: recvInfo size equals 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        recvInfo.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] RecvThruMultiLinks: invalid input with recvInfo size [%u], "
            "queue num [%u].",
            recvInfo.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    // only those worker queues required to be sync: put mode in send
    std::vector<InsQuePtr> syncQues = {queues[0]};
    bool hasDiffDmaMode = false;

    CHK_RET(ProceedMultiLinks(
        recvInfo, queues, MultiDataLinksDmaModeInfo(DmaMode::GET, dmaMode), syncQues,
        hasDiffDmaMode)); // Get mode should be sync for Recv

    if (hasDiffDmaMode) {
        HCCL_DEBUG("[InsCollAlgFactory] [AlgDataTrans] RecvThruMultiLinks: current recv links have two DmaMode.");
        CHK_RET(TxRxReady({recvInfo[0].link_, recvInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(RxReady(recvInfo[0].link_, queues[0], topicId, dmaMode));
    }

    CHK_RET(PreSyncQues(syncQues, 0));

    auto dataInfoIter = recvInfo.begin();
    auto queIter = queues.begin();
    for (; dataInfoIter != recvInfo.end(); dataInfoIter++, queIter++) {
        if (std::find(syncQues.begin(), syncQues.end(), (*queIter)) != syncQues.end()) {
            CHK_RET(RxData(dataInfoIter->link_, (*queIter), dataInfoIter->slices_, dmaMode));
        }
    }

    CHK_RET(PostSyncQues(syncQues, 0));

    if (hasDiffDmaMode) {
        CHK_RET(TxRxFin({recvInfo[0].link_, recvInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(RxFin(recvInfo[0].link_, queues[0], topicId, dmaMode));
    }

    if (needNetFinAck) {
        RxFinAck(recvInfo[0].link_, queues[0], topicId, dmaMode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult SendRecvThruMultiLinks(
    const std::vector<SendRecvInfo>& sendRecvInfo, std::vector<InsQuePtr>& queues, u32 topicId, bool needNetFinAck,
    DmaMode dmaMode)
{
    if (sendRecvInfo.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] SendRecvThruMultiLinks: empty sendRecvInfo, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        sendRecvInfo.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] SendRecvThruMultiLinks: invalid input with recvInfo size [%zu], "
            "queue num [%zu].",
            sendRecvInfo.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto sendRecvInfoIter = sendRecvInfo.begin();
    auto queIter = queues.begin();
    u32 netTxLinksNum = 0;
    u32 netRxLinksNum = 0;

    CHK_RET(TxRxReady(sendRecvInfoIter->sendRecvLinks_, (*queIter), topicId, dmaMode));

    u32 mainQueIdx = 0;
    CHK_RET(PreSyncQues(queues, mainQueIdx));

    for (; sendRecvInfoIter != sendRecvInfo.end(); sendRecvInfoIter++, queIter++) {
        if (((sendRecvInfoIter->sendRecvLinks_).txLink_).GetType() == PortDeploymentType::DEV_NET) {
            netTxLinksNum++;
        }
        if (((sendRecvInfoIter->sendRecvLinks_).rxLink_).GetType() == PortDeploymentType::DEV_NET) {
            netRxLinksNum++;
        }
        CHK_RET(TxRxData(sendRecvInfoIter->sendRecvLinks_, (*queIter), sendRecvInfoIter->sendRecvSlices_, dmaMode));
    }

    CHK_PRT_RET(
        ((netTxLinksNum > 1) || (netRxLinksNum > 1)),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] SendRecvThruMultiLinks: multi net links is not "
                   "supported as NET operations are async, use mid-level wrapper instead."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (((netTxLinksNum == 1) && (sendRecvInfo[0].sendRecvLinks_.txLink_.GetType() == PortDeploymentType::P2P))
         || ((netRxLinksNum == 1) && (sendRecvInfo[0].sendRecvLinks_.rxLink_.GetType() == PortDeploymentType::P2P))),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] SendRecvThruMultiLinks: first link must be NET when there "
                   "exists NET links."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_RET(PostSyncQues(queues, mainQueIdx));

    CHK_RET(TxRxFin(sendRecvInfo[0].sendRecvLinks_, queues[0], topicId, dmaMode));

    if (needNetFinAck) {
        CHK_RET(TxRxFinAck(sendRecvInfo[0].sendRecvLinks_, queues[0], topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult SendReduceThruMultiLinks(
    const std::vector<DataReduceInfo>& sendReduceInfo, std::vector<InsQuePtr>& queues, u32 topicId, bool needNetFinAck,
    DmaMode dmaMode)
{
    if (sendReduceInfo.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] SendReduceThruMultiLinks: empty sendReduceInfo, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        sendReduceInfo.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] SendReduceThruMultiLinks: sendReduceInfo size [%zu] is "
            "non-equal to queue num [%zu].",
            sendReduceInfo.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    // only those worker queues required to be sync: put mode in send
    std::vector<InsQuePtr> syncQues = {queues[0]};
    bool hasDiffDmaMode = false;

    CHK_RET(ProceedMultiLinks(
        sendReduceInfo, queues, MultiDataLinksDmaModeInfo(DmaMode::PUT, dmaMode), syncQues, hasDiffDmaMode));

    if (hasDiffDmaMode) {
        HCCL_DEBUG("[InsCollAlgFactory] [AlgDataTrans] SendReduceThruMultiLinks: current send links have two DmaMode.");
        CHK_RET(TxRxReady({sendReduceInfo[0].link_, sendReduceInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(TxReady(sendReduceInfo[0].link_, queues[0], topicId, dmaMode));
    }

    CHK_RET(PreSyncQues(syncQues, 0));

    auto dataInfoIter = sendReduceInfo.begin();
    auto queIter = queues.begin();
    for (; dataInfoIter != sendReduceInfo.end(); dataInfoIter++, queIter++) {
        if (std::find(syncQues.begin(), syncQues.end(), (*queIter)) != syncQues.end()) {
            CHK_RET(TxReduce(
                dataInfoIter->link_, (*queIter),
                {dataInfoIter->slices_, dataInfoIter->dataType_, dataInfoIter->reduceOp_}, dmaMode));
        }
    }

    CHK_RET(PostSyncQues(syncQues, 0));

    if (hasDiffDmaMode) {
        CHK_RET(TxRxFin({sendReduceInfo[0].link_, sendReduceInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(TxFin(sendReduceInfo[0].link_, queues[0], topicId, dmaMode));
    }

    if (needNetFinAck) {
        TxFinAck(sendReduceInfo[0].link_, queues[0], topicId, dmaMode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult RecvReduceThruMultiLinks(
    const std::vector<DataReduceInfo>& recvReduceInfo, std::vector<InsQuePtr>& queues, u32 topicId, bool needNetFinAck,
    DmaMode dmaMode)
{
    if (recvReduceInfo.size() == 0) {
        HCCL_WARNING("[InsCollAlgFactory] [AlgDataTrans] RecvReduceThruMultiLinks: empty recvReduceInfo, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        recvReduceInfo.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] RecvReduceThruMultiLinks: invalid input with recvReduceInfo size "
            "[%zu], "
            "queue num [%zu].",
            recvReduceInfo.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    // only those worker queues required to be sync: put mode in send
    std::vector<InsQuePtr> syncQues = {queues[0]};
    bool hasDiffDmaMode = false;

    CHK_RET(ProceedMultiLinks(
        recvReduceInfo, queues, MultiDataLinksDmaModeInfo(DmaMode::GET, dmaMode), syncQues,
        hasDiffDmaMode)); // Get mode should be sync for Recv

    if (hasDiffDmaMode) {
        HCCL_DEBUG("[InsCollAlgFactory] [AlgDataTrans] RecvReduceThruMultiLinks: current recv links have two DmaMode.");
        CHK_RET(TxRxReady({recvReduceInfo[0].link_, recvReduceInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(RxReady(recvReduceInfo[0].link_, queues[0], topicId, dmaMode));
    }

    CHK_RET(PreSyncQues(syncQues, 0));

    auto dataInfoIter = recvReduceInfo.begin();
    auto queIter = queues.begin();
    for (; dataInfoIter != recvReduceInfo.end(); dataInfoIter++, queIter++) {
        if (std::find(syncQues.begin(), syncQues.end(), (*queIter)) != syncQues.end()) {
            CHK_RET(RxReduce(
                dataInfoIter->link_, (*queIter),
                {dataInfoIter->slices_, dataInfoIter->dataType_, dataInfoIter->reduceOp_}, dmaMode));
        }
    }

    CHK_RET(PostSyncQues(syncQues, 0));

    if (hasDiffDmaMode) {
        CHK_RET(TxRxFin({recvReduceInfo[0].link_, recvReduceInfo[0].link_}, queues[0], topicId, dmaMode));
    } else {
        CHK_RET(RxFin(recvReduceInfo[0].link_, queues[0], topicId, dmaMode));
    }

    if (needNetFinAck) {
        RxFinAck(recvReduceInfo[0].link_, queues[0], topicId, dmaMode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult SendRecvReduceThruMultiLinks(
    const std::vector<SendRecvReduceInfo>& sendRecvReduceInfo, std::vector<InsQuePtr>& queues, u32 topicId,
    bool needNetFinAck, DmaMode dmaMode)
{
    if (sendRecvReduceInfo.size() == 0) {
        HCCL_WARNING(
            "[InsCollAlgFactory] [AlgDataTrans] SendRecvReduceThruMultiLinks: empty sendRecvReduceInfo, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        sendRecvReduceInfo.size() != queues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] SendRecvReduceThruMultiLinks: sendRecvReduceInfo size [%u] is "
            "non-equal to queue num [%u].",
            sendRecvReduceInfo.size(), queues.size()),
        HcclResult::HCCL_E_INTERNAL);

    auto dataInfoIter = sendRecvReduceInfo.begin();
    auto queIter = queues.begin();
    u32 netTxLinksNum = 0;
    u32 netRxLinksNum = 0;

    CHK_RET(TxRxReady(dataInfoIter->sendRecvLinks_, (*queIter), topicId, dmaMode));

    u32 mainQueIdx = 0;
    CHK_RET(PreSyncQues(queues, mainQueIdx));

    for (; dataInfoIter != sendRecvReduceInfo.end(); dataInfoIter++, queIter++) {
        if (((dataInfoIter->sendRecvLinks_).txLink_).GetType() == PortDeploymentType::DEV_NET) {
            netTxLinksNum++;
        }
        if (((dataInfoIter->sendRecvLinks_).rxLink_).GetType() == PortDeploymentType::DEV_NET) {
            netRxLinksNum++;
        }
        CHK_RET(TxRxReduce(
            dataInfoIter->sendRecvLinks_, (*queIter),
            {dataInfoIter->sendRecvSlices_, dataInfoIter->dataType_, dataInfoIter->reduceOp_}, dmaMode));
    }

    CHK_PRT_RET(
        ((netTxLinksNum > 1) || (netRxLinksNum > 1)),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] SendRecvThruMultiLinks: multi net links is not "
                   "supported as NET operations are async, use mid-level wrapper instead."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (((netTxLinksNum == 1) && (sendRecvReduceInfo[0].sendRecvLinks_.txLink_.GetType() == PortDeploymentType::P2P))
         || ((netRxLinksNum == 1)
             && (sendRecvReduceInfo[0].sendRecvLinks_.rxLink_.GetType() == PortDeploymentType::P2P))),
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] SendRecvThruMultiLinks: first link must be NET when there "
                   "exists NET links."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_RET(PostSyncQues(queues, mainQueIdx));

    CHK_RET(TxRxFin(sendRecvReduceInfo[0].sendRecvLinks_, queues[0], topicId, dmaMode));

    if (needNetFinAck) {
        CHK_RET(TxRxFinAck(sendRecvReduceInfo[0].sendRecvLinks_, queues[0], topicId, dmaMode));
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
