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
#include "alg_common_interface.h"
#include "log.h"

namespace Hccl {

HcclResult GetDMAMode(const DmaMode setMode, const PortDeploymentType linkPortType, DmaMode& mode)
{
    if (setMode == DmaMode::GET) {
        mode = DmaMode::GET;
    } else if (setMode == DmaMode::PUT) {
        mode = DmaMode::PUT;
    } else {
        if (linkPortType == PortDeploymentType::P2P) {
            mode = DmaMode::GET;
        } else {
            mode = DmaMode::PUT;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

bool IsContinuousSlice(const DataSlice& nxtSlice, const DataSlice& currSlice)
{
    if (nxtSlice.GetType() != currSlice.GetType()) {
        return false;
    }
    if (nxtSlice.GetOffset() != currSlice.GetOffset() + currSlice.GetSize()) {
        return false;
    }
    return true;
}

bool isSupportBatchTransfer()
{
    if (IsAicpuMode()) {
        return true;
    } else {
        return false;
    }
}

void TransSlice(const LinkData& link, InsQuePtr queue, const SlicePair& txRxSlice, DmaMode dmaMode, bool reduceFlag)
{
    if ((dmaMode == DmaMode::PUT) && (!reduceFlag)) {
        queue->Append(std::make_unique<InsWrite>(
            link.GetRemoteRankId(), link, txRxSlice.srcSlice_,
            txRxSlice.dstSlice_)); // src as local
    } else if ((dmaMode == DmaMode::PUT) && (reduceFlag)) {
        queue->Append(std::make_unique<InsWriteReduce>(
            link.GetRemoteRankId(), link, txRxSlice.srcSlice_, txRxSlice.dstSlice_, txRxSlice.dataType_,
            txRxSlice.reduceOp_));
    } else if ((dmaMode == DmaMode::GET) && (!reduceFlag)) {
        queue->Append(std::make_unique<InsRead>(
            link.GetRemoteRankId(), link, txRxSlice.dstSlice_,
            txRxSlice.srcSlice_)); // dst as local
    } else {
        queue->Append(std::make_unique<InsReadReduce>(
            link.GetRemoteRankId(), link, txRxSlice.dstSlice_, txRxSlice.srcSlice_, txRxSlice.dataType_,
            txRxSlice.reduceOp_));
    }

    return;
}

HcclResult
IndividualTransSlicesLists(const LinkData& link, InsQuePtr queue, const TransSlicesInfo& slices, DmaMode dmaMode)
{
    CHK_PRT_RET(
        slices.dstSlices.size() != slices.srcSlices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] IndividualTransSlicesLists: recv slice num [%zu] is not equal to "
            "send slice num [%zu].",
            slices.dstSlices.size(), slices.srcSlices.size()),
        HcclResult::HCCL_E_INTERNAL);

    if (slices.srcSlices.size() == 0) {
        HCCL_DEBUG("[InsCollAlgFactory] [AlgDataTrans] IndividualTransSlicesLists: empty slices do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    // tmpSlices: slices to be transfer in this loop
    DataSlice tmpSrcSlice = slices.srcSlices[0];
    DataSlice tmpDstSlice = slices.dstSlices[0];

    for (u32 sliceIdx = 0; sliceIdx < slices.srcSlices.size(); sliceIdx++) {
        CHK_PRT_RET(
            slices.srcSlices[sliceIdx].GetSize() != slices.dstSlices[sliceIdx].GetSize(),
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTrans] TransSlicesLists: [%u]-th slice, recv slice size [%zu] "
                "is not equal to send slice size [%zu].",
                sliceIdx, slices.dstSlices[sliceIdx].GetSize(), slices.srcSlices[sliceIdx].GetSize()),
            HcclResult::HCCL_E_INTERNAL);

        if (sliceIdx == (slices.srcSlices.size() - 1)) {
            // last slice, transfer immediately
            SlicePair txRxSlice = SlicePair(tmpSrcSlice, tmpDstSlice);
            if (slices.reduceFlag) {
                txRxSlice.dataType_ = slices.dataType_;
                txRxSlice.reduceOp_ = slices.reduceOp_;
            }
            TransSlice(link, queue, txRxSlice, dmaMode, slices.reduceFlag);
        } else if (
            IsContinuousSlice(slices.srcSlices[sliceIdx + 1], tmpSrcSlice)
            && IsContinuousSlice(slices.dstSlices[sliceIdx + 1], tmpDstSlice)) {
            // nxtSlice is continuous with tmpSlice, update tmpSlice
            u64 newTmpSize = tmpSrcSlice.GetSize() + slices.srcSlices[sliceIdx + 1].GetSize();
            tmpSrcSlice = DataSlice(tmpSrcSlice.GetType(), tmpSrcSlice.GetOffset(), newTmpSize);
            tmpDstSlice = DataSlice(tmpDstSlice.GetType(), tmpDstSlice.GetOffset(), newTmpSize);
        } else {
            // nxtSlice is not continuous with tmpSlice, transfer tmpSlice, update tmpSlice with nxtSlice
            SlicePair txRxSlice = SlicePair(tmpSrcSlice, tmpDstSlice);
            if (slices.reduceFlag) {
                txRxSlice.reduceOp_ = slices.reduceOp_;
                txRxSlice.dataType_ = slices.dataType_;
            }
            TransSlice(link, queue, txRxSlice, dmaMode, slices.reduceFlag);

            tmpSrcSlice = slices.srcSlices[sliceIdx + 1];
            tmpDstSlice = slices.dstSlices[sliceIdx + 1];
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
IndividualWriteSlicesListsWithFin(const LinkData& link, InsQuePtr queue, const TransSlicesInfo& slices, u32 topicId)
{
    CHK_PRT_RET(
        slices.srcSlices.size() == 0,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] WriteSlicesListsWithFin: invalid input with empty slices."),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        slices.dstSlices.size() != slices.srcSlices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] WriteSlicesListsWithFin: dst slice num [%zu] is not equal to "
            "src slice num [%zu].",
            slices.dstSlices.size(), slices.srcSlices.size()),
        HcclResult::HCCL_E_INTERNAL);

    DataSlice tmpSrcSlice = slices.srcSlices[0];
    DataSlice tmpDstSlice = slices.dstSlices[0];

    for (u32 sliceIdx = 0; sliceIdx < slices.srcSlices.size(); sliceIdx++) {
        CHK_PRT_RET(
            slices.srcSlices[sliceIdx].GetSize() != slices.dstSlices[sliceIdx].GetSize(),
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTrans] WriteSlicesListsWithFin: [%u]-th slice, recv slice "
                "size [%zu] is not equal to send slice size [%zu].",
                sliceIdx, slices.dstSlices[sliceIdx].GetSize(), slices.srcSlices[sliceIdx].GetSize()),
            HcclResult::HCCL_E_INTERNAL);

        if (sliceIdx == (slices.srcSlices.size() - 1)) {
            // last slice, transfer immediately, src as local
            NotifyType notifyType = slices.enableCounterNotify_ ? NotifyType::COUNTER : NotifyType::NORMAL;
            if (!slices.reduceFlag) {
                queue->Append(std::make_unique<InsWriteWithFin>(
                    link.GetRemoteRankId(), link, tmpSrcSlice, tmpDstSlice, notifyType, topicId)); // src as local
            } else {
                queue->Append(std::make_unique<InsWriteReduceWithFin>(
                    link.GetRemoteRankId(), link, tmpSrcSlice, tmpDstSlice, slices.dataType_, slices.reduceOp_,
                    notifyType, topicId));
            }
        } else if (
            IsContinuousSlice(slices.srcSlices[sliceIdx + 1], tmpSrcSlice)
            && IsContinuousSlice(slices.dstSlices[sliceIdx + 1], tmpDstSlice)) {
            // nxtSlice is continuous with tmpSlice, update tmpSlice
            u64 newTmpSize = tmpSrcSlice.GetSize() + slices.srcSlices[sliceIdx + 1].GetSize();
            tmpSrcSlice = DataSlice(tmpSrcSlice.GetType(), tmpSrcSlice.GetOffset(), newTmpSize);
            tmpDstSlice = DataSlice(tmpDstSlice.GetType(), tmpDstSlice.GetOffset(), newTmpSize);
        } else {
            // nxtSlice is not continuous with tmpSlice, transfer tmpSlice, update tmpSlice with nxtSlice
            SlicePair txRxSlice = SlicePair(tmpSrcSlice, tmpDstSlice);
            if (slices.reduceFlag) {
                txRxSlice.dataType_ = slices.dataType_;
                txRxSlice.reduceOp_ = slices.reduceOp_;
            }

            // not last slice, therefore no Fin sync
            TransSlice(link, queue, txRxSlice, DmaMode::PUT, slices.reduceFlag);

            tmpSrcSlice = slices.srcSlices[sliceIdx + 1];
            tmpDstSlice = slices.dstSlices[sliceIdx + 1];
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult BatchTransSlicesLists(const LinkData& link, InsQuePtr queue, const TransSlicesInfo& slices, DmaMode dmaMode)
{
    CHK_PRT_RET(
        slices.dstSlices.size() != slices.srcSlices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] BatchTransSlicesLists: recv slice num [%zu] is not equal to "
            "send slice num [%zu].",
            slices.dstSlices.size(), slices.srcSlices.size()),
        HcclResult::HCCL_E_INTERNAL);

    if (slices.srcSlices.size() == 0) {
        HCCL_DEBUG("[InsCollAlgFactory] [AlgDataTrans] BatchTransSlicesLists: empty slices do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }

    if (dmaMode == DmaMode::PUT) {
        unique_ptr<InsBatchWrite> batchInstruction = std::make_unique<InsBatchWrite>(link.GetRemoteRankId(), link);
        for (u32 sliceIdx = 0; sliceIdx < slices.srcSlices.size(); sliceIdx++) {
            CHK_PRT_RET(
                slices.srcSlices[sliceIdx].GetSize() != slices.dstSlices[sliceIdx].GetSize(),
                HCCL_ERROR(
                    "[InsCollAlgFactory] [AlgDataTrans] BatchTransSlicesLists: [%u]-th slice, recv slice size [%zu] "
                    "is not equal to send slice size [%zu].",
                    sliceIdx, slices.dstSlices[sliceIdx].GetSize(), slices.srcSlices[sliceIdx].GetSize()),
                HcclResult::HCCL_E_INTERNAL);
            if (!slices.reduceFlag) {
                batchInstruction->PushWriteIns(std::make_unique<InsWrite>(
                    link.GetRemoteRankId(), link, slices.srcSlices[sliceIdx], slices.dstSlices[sliceIdx]));
            } else {
                batchInstruction->PushWriteIns(std::make_unique<InsWriteReduce>(
                    link.GetRemoteRankId(), link, slices.srcSlices[sliceIdx], slices.dstSlices[sliceIdx],
                    slices.dataType_, slices.reduceOp_));
            }
        }
        queue->Append(std::move(batchInstruction));
    } else if (dmaMode == DmaMode::GET) {
        unique_ptr<InsBatchRead> batchInstruction = std::make_unique<InsBatchRead>(link.GetRemoteRankId(), link);
        for (u32 sliceIdx = 0; sliceIdx < slices.srcSlices.size(); sliceIdx++) {
            CHK_PRT_RET(
                slices.srcSlices[sliceIdx].GetSize() != slices.dstSlices[sliceIdx].GetSize(),
                HCCL_ERROR(
                    "[InsCollAlgFactory] [AlgDataTrans] BatchTransSlicesLists: [%u]-th slice, recv slice size [%zu] "
                    "is not equal to send slice size [%zu].",
                    sliceIdx, slices.dstSlices[sliceIdx].GetSize(), slices.srcSlices[sliceIdx].GetSize()),
                HcclResult::HCCL_E_INTERNAL);
            if (!slices.reduceFlag) {
                batchInstruction->PushReadIns(std::make_unique<InsRead>(
                    link.GetRemoteRankId(), link, slices.dstSlices[sliceIdx], slices.srcSlices[sliceIdx]));
            } else {
                batchInstruction->PushReadIns(std::make_unique<InsReadReduce>(
                    link.GetRemoteRankId(), link, slices.dstSlices[sliceIdx], slices.srcSlices[sliceIdx],
                    slices.dataType_, slices.reduceOp_));
            }
        }
        queue->Append(std::move(batchInstruction));
    } else {
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] BatchTransSlicesLists: dmaMode [%s] is not supported.",
            dmaMode.Describe().c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TransSlicesLists(const LinkData& link, InsQuePtr queue, const TransSlicesInfo& slices, DmaMode dmaMode)
{
    if (isSupportBatchTransfer()) {
        CHK_RET(BatchTransSlicesLists(link, queue, slices, dmaMode));
    } else {
        CHK_RET(IndividualTransSlicesLists(link, queue, slices, dmaMode));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult WriteSlicesListsWithFin(const LinkData& link, InsQuePtr queue, const TransSlicesInfo& slices, u32 topicId)
{
    if (isSupportBatchTransfer() && slices.srcSlices.size() > 1) {
        CHK_RET(BatchTransSlicesLists(link, queue, slices, DmaMode::PUT));
        queue->Append(std::make_unique<InsPostFin>(link.GetRemoteRankId(), link));
    } else {
        CHK_RET(IndividualWriteSlicesListsWithFin(link, queue, slices, topicId));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult ProceedMultiLinks(
    const std::vector<DataInfo>& dataInfo, const std::vector<InsQuePtr>& queues,
    const MultiDataLinksDmaModeInfo& dmaModeInfo, std::vector<InsQuePtr>& syncQues, bool& hasDiffDmaMode)
{
    auto dataInfoIter = dataInfo.begin();
    auto queIter = queues.begin();

    RankId remoteRank = dataInfoIter->link_.GetRemoteRankId();
    PortDeploymentType linkType = dataInfoIter->link_.GetType();
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaModeInfo.modeSet_, linkType, mode));
    dataInfoIter++;
    queIter++;

    DmaMode tmpMode;
    u32 netLinkNum = 0;
    for (; dataInfoIter != dataInfo.end(); dataInfoIter++, queIter++) {
        CHK_PRT_RET(
            dataInfoIter->link_.GetRemoteRankId() != remoteRank,
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTrans] Send/RecvThruMultiLinks: only support identical "
                "remote rank, now we have got rank [%d] and rank [%d].",
                remoteRank, dataInfoIter->link_.GetRemoteRankId()),
            HcclResult::HCCL_E_INTERNAL);

        PortDeploymentType tmpLinkType = (dataInfoIter->link_).GetType();
        if (tmpLinkType == PortDeploymentType::DEV_NET) {
            CHK_PRT_RET(
                linkType == PortDeploymentType::P2P,
                HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] Send/RecvThruMultiLinks: portType of first link "
                           "should be DEV_NET when there exists NET links."),
                HcclResult::HCCL_E_INTERNAL);
            netLinkNum++;
        }

        CHK_RET(GetDMAMode(dmaModeInfo.modeSet_, (dataInfoIter->link_).GetType(), tmpMode));

        if (tmpMode != mode) {
            hasDiffDmaMode = true;
        }

        if (tmpMode == dmaModeInfo.modeNeedSync_) {
            syncQues.push_back(*queIter); // que sync is required only when mode is PUT for send and GET for recv
        }
    }

    CHK_PRT_RET(
        netLinkNum > 1,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] Send/RecvThruMultiLinks: more than one net links, use "
                   "mid-level wrapper instead as DEV_NET is async."),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ProceedMultiLinks(
    const std::vector<DataReduceInfo>& dataInfo, const std::vector<InsQuePtr>& queues,
    const MultiDataLinksDmaModeInfo& dmaModeInfo, std::vector<InsQuePtr>& syncQues, bool& hasDiffDmaMode)
{
    RankId remoteRank = dataInfo[0].link_.GetRemoteRankId();
    PortDeploymentType linkType = dataInfo[0].link_.GetType();
    DmaMode mode;
    CHK_RET(GetDMAMode(dmaModeInfo.modeSet_, linkType, mode));

    auto dataInfoIter = dataInfo.begin();
    auto queIter = queues.begin();
    dataInfoIter++;
    queIter++;

    DmaMode tmpMode;
    u32 netLinkNum = 0;
    for (; dataInfoIter != dataInfo.end(); dataInfoIter++, queIter++) {
        CHK_PRT_RET(
            dataInfoIter->link_.GetRemoteRankId() != remoteRank,
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTrans] Send/RecvReduceThruMultiLinks: only support "
                "identical remote rank, now we have got rank [%d] and rank [%d].",
                remoteRank, dataInfoIter->link_.GetRemoteRankId()),
            HcclResult::HCCL_E_INTERNAL);

        PortDeploymentType tmpLinkType = (dataInfoIter->link_).GetType();
        if (tmpLinkType == PortDeploymentType::DEV_NET) {
            CHK_PRT_RET(
                linkType == PortDeploymentType::P2P,
                HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] Send/RecvReduceThruMultiLinks: portType of first link "
                           "should be DEV_NET when there exists NET links."),
                HcclResult::HCCL_E_INTERNAL);
            netLinkNum++;
        }

        CHK_RET(GetDMAMode(dmaModeInfo.modeSet_, (dataInfoIter->link_).GetType(), tmpMode));

        if (tmpMode == dmaModeInfo.modeNeedSync_) {
            syncQues.push_back(*queIter); // que sync is required only when mode is PUT for send and GET for recv
        }

        if (tmpMode != mode) {
            hasDiffDmaMode = true;
        }
    }

    CHK_PRT_RET(
        netLinkNum > 1,
        HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] Send/RecvReduceThruMultiLinks: more than one net links, "
                   "use mid-level wrapper instead as DEV_NET is async."),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
