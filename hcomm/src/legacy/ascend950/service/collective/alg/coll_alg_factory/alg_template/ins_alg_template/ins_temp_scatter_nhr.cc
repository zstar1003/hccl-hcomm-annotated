/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "alg_data_trans_wrapper.h"
#include "dev_mode.h"
#include "log.h"

namespace Hccl {
InsTempScatterNHR::InsTempScatterNHR(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempScatterNHR::~InsTempScatterNHR() {}

HcclResult InsTempScatterNHR::CalcRes(AlgTempResReq& tempResReq)
{
    CHK_PRT_RET(
        CalcResLinksNHR(myRank_, tempRankSize_, tempVTopo_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempScatterNHR] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    auto& linkReq = tempResReq.links;
    u32 pathNum = 0;
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        auto remoteRank = resReqIter->first;
        if (rank2PathNumMap_.find(remoteRank) == rank2PathNumMap_.end() || rank2PathNumMap_[remoteRank] == 0) {
            HCCL_ERROR("[InsTempScatterNHR] No path to remoteRank[%d]", remoteRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (pathNum == 0) {
            pathNum = rank2PathNumMap_[remoteRank];
        } else if (rank2PathNumMap_[remoteRank] != pathNum) {
            HCCL_ERROR(
                "[InsTempScatterNHR] Inconsistency pathNum to remoteRanks, Previous consistent pathNum=[%u], "
                "mismatched "
                "remoteRank=[%d], pathNum=[%u]",
                pathNum, remoteRank, rank2PathNumMap_[remoteRank]);
            return HcclResult::HCCL_E_INTERNAL;
        }
        resReqIter->second = pathNum;
    }

    // NHR 需要的 que Num 为 1
    tempResReq.queNum = 1 * pathNum;
    HCCL_INFO("[InsTempScatterNHR] tempResReq.queNum = %u", tempResReq.queNum);
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    return HcclResult::HCCL_SUCCESS;
}

uint64_t InsTempScatterNHR::GetExpandedMode() const { return DeviceMode::AICPU; }

HcclResult InsTempScatterNHR::PreCopy(const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    if (u32(myRank_) != root_ || buffInfo_.inBuffType == BufferType::SCRATCH) {
        return HCCL_SUCCESS;
    }
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 tailSize = templateDataParams.tailSize == 0 ? templateDataParams.sliceSize : templateDataParams.tailSize;
    for (u32 r = 0; r < templateDataParams.repeatNum; r++) {
        for (u32 algRank = 0; algRank < tempRankSize_; algRank++) {
            u64 srcOffset = r * templateDataParams.inputRepeatStride + templateDataParams.inputSliceStride * algRank
                            + buffInfo_.inBuffBaseOff;
            DataSlice srcSlice(BufferType::INPUT, srcOffset, tailSize);
            u64 dstOffset = r * ((tempRankSize_ - 1) * templateDataParams.sliceSize + tailSize)
                            + buffInfo_.scratchBuffBaseOff + algRank * templateDataParams.sliceSize;
            DataSlice dstSlice(BufferType::SCRATCH, dstOffset, tailSize);
            LocalCopy(tempInsQues[0], srcSlice, dstSlice);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterNHR::PostCopy(const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    u32 myAlgRank;
    GetAlgRank(myRank_, tempVTopo_[0], myAlgRank);
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 tailSize = templateDataParams.tailSize == 0 ? templateDataParams.sliceSize : templateDataParams.tailSize;
    // 支持不均匀切分的情况下需要把尾部数据放到最后一张卡上
    u64 sliceSize = myAlgRank == tempVTopo_[0].size() - 1 ? tailSize : templateDataParams.sliceSize;

    for (u32 r = 0; r < templateDataParams.repeatNum; r++) {
        u64 dstOffset = buffInfo_.outBuffBaseOff + r * templateDataParams.sliceSize;
        u64 srcOffset = r * ((tempRankSize_ - 1) * templateDataParams.sliceSize + tailSize)
                        + buffInfo_.scratchBuffBaseOff + myAlgRank * templateDataParams.sliceSize;
        DataSlice dstSlice(buffInfo_.outBuffType, dstOffset, sliceSize);
        DataSlice srcSlice(BufferType::SCRATCH, srcOffset, sliceSize);
        if (buffInfo_.outBuffType == BufferType::SCRATCH && srcOffset == dstOffset) {
            continue;
        }
        LocalCopy(tempInsQues[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::RunNHR(
    TemplateDataParams& templateDataParams, ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) const
{
    u32 mainQueIdx = 0;
    // 流间前同步，主流通知从流，只有一个流则不做任何事
    CHK_RET(PreSyncQues(tempInsQues, mainQueIdx));

    // nhr主体部分
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 r = 0; r < templateDataParams.repeatNum; r++) {
        for (u32 step = 0; step < nSteps; step++) {
            AicpuNHRStepInfo stepInfo;
            GetStepInfo(step, nSteps, stepInfo);

            // 只有Tx,使用send指令
            if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() == 0) {
                CHK_RET(BatchSend(stepInfo, tempLinks, tempInsQues, templateDataParams, r));
            }
            // 只有Rx，使用recv指令
            else if (stepInfo.txSliceIdxs.size() == 0 && stepInfo.rxSliceIdxs.size() > 0) {
                CHK_RET(BatchRecv(stepInfo, tempLinks, tempInsQues, templateDataParams, r));
            }
            // 既有Tx又有Rx，使用SendRecv指令
            else if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() > 0) {
                CHK_RET(BatchSR(stepInfo, tempLinks, tempInsQues, templateDataParams, r));
            }
        }
    }
    // 流间后同步，从流通知主流
    CHK_RET(PostSyncQues(tempInsQues, mainQueIdx));
    return HCCL_SUCCESS;
}

// 需要支持input->scratch, scratch->output, input->output
HcclResult InsTempScatterNHR::GenExtIns(
    TempFuncs& tempFuncs, TemplateDataParams& templateDataParams, ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    } else {
        dmaMode_ = DmaMode::PUT;
    }
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = templateDataParams.buffInfo;

    HCCL_INFO("[InsTempScatterNHR] Run start");
    uint32_t linkNum = tempLinks.begin()->second.size();
    // 流的数量不能少于linkNum
    CHK_PRT_RET(
        linkNum < tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempScatterNHR] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO(
        "[InsTempScatterNHR Run]RankID:[%d], root:[%u], isForepart:[%d], isBottom:[%d]", myRank_, root_,
        tempFuncs.isForepart, tempFuncs.isBottom);
    CHK_RET(PreCopy(templateDataParams, tempInsQues));
    CHK_RET(RunNHR(templateDataParams, tempLinks, tempInsQues));
    CHK_RET(PostCopy(templateDataParams, tempInsQues));
    return HCCL_SUCCESS;
}

u32 InsTempScatterNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult InsTempScatterNHR::BatchSend(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    TemplateDataParams& templateDataParams, u32 repeat) const
{
    const std::vector<LinkData>& linkSend = tempLinks.at(stepInfo.toRank);
    u32 linkNum = rank2PathNumMap_.at(stepInfo.toRank);
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(linkSend, dataSplitRate));
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 tailSize = templateDataParams.tailSize == 0 ? templateDataParams.sliceSize : templateDataParams.tailSize;

    const u32 sliceCount = stepInfo.txSliceIdxs.size();
    const u64 baseOffset
        = repeat * ((tempRankSize_ - 1) * templateDataParams.sliceSize + tailSize) + buffInfo_.scratchBuffBaseOff;
    std::vector<DataSlice> srcDstSlices;
    srcDstSlices.reserve(sliceCount);
    for (u32 j = 0; j < linkNum; j++) {
        srcDstSlices.clear();
        for (u32 i = 0; i < sliceCount; i++) {
            u64 srcDstOffset = baseOffset + stepInfo.txSliceIdxs[i] * templateDataParams.sliceSize;
            // 发送的一定是root
            DataSlice srcDstSlice(templateDataParams.buffInfo.scratBuffType, srcDstOffset, tailSize);
            srcDstSlices.emplace_back(CalcDataSliceForLinks(srcDstSlice, dataSplitRate, j, dataType_));
        }
        SlicesList txSlicesList(srcDstSlices, srcDstSlices);
        DataInfo sendData(linkSend[j], std::move(txSlicesList));
        CHK_PRT_RET(
            Send(sendData, tempInsQues[j], 0, true, dmaMode_), HCCL_ERROR("[InsTempScatterNHR] BatchSend failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchRecv(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    TemplateDataParams& templateDataParams, u32 repeat) const
{
    u32 myAlgRank;
    GetAlgRank(myRank_, tempVTopo_[0], myAlgRank);
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 tailSize = templateDataParams.tailSize == 0 ? templateDataParams.sliceSize : templateDataParams.tailSize;
    // 支持不均匀切分的情况下需要把尾部数据放到最后一张卡上
    u64 sliceSize = myAlgRank == tempVTopo_[0].size() - 1 ? tailSize : templateDataParams.sliceSize;
    const std::vector<LinkData>& linkRecv = tempLinks.at(stepInfo.fromRank);
    u32 linkNum = rank2PathNumMap_.at(stepInfo.fromRank);
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(linkRecv, dataSplitRate));
    const u32 sliceCount = stepInfo.txSliceIdxs.size();
    const u64 baseOffset = repeat * ((tempRankSize_ - 1) * templateDataParams.sliceSize + templateDataParams.tailSize)
                           + buffInfo_.scratchBuffBaseOff;
    std::vector<DataSlice> srcDstSlices;
    srcDstSlices.reserve(sliceCount);

    for (u32 j = 0; j < linkNum; j++) {
        srcDstSlices.clear();
        for (u32 i = 0; i < sliceCount; i++) {
            u64 srcDstOffset = baseOffset + stepInfo.rxSliceIdxs[i] * templateDataParams.sliceSize;
            DataSlice srcDstSlice(templateDataParams.buffInfo.scratBuffType, srcDstOffset, sliceSize);
            srcDstSlices.emplace_back(CalcDataSliceForLinks(srcDstSlice, dataSplitRate, j, dataType_));
        }
        SlicesList rxSlicesList(srcDstSlices, srcDstSlices);
        DataInfo recvData(linkRecv[j], std::move(rxSlicesList));
        CHK_PRT_RET(
            Recv(recvData, tempInsQues[j], 0, true, dmaMode_), HCCL_ERROR("[InsTempScatterNHR] BatchTxRx Recv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchSR(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    TemplateDataParams& templateDataParams, u32 repeat) const
{
    const std::vector<LinkData>& linkRecv = tempLinks.at(stepInfo.fromRank);
    const std::vector<LinkData>& linkSend = tempLinks.at(stepInfo.toRank);

    HCCL_INFO("BatchSR(stepInfo.fromRank)=%u", stepInfo.fromRank);
    u32 linkNum = rank2PathNumMap_.at(stepInfo.fromRank);
    if (linkNum > linkRecv.size()) {
        HCCL_ERROR("InsTempScatterNHR::RunMesh linkNum > linkRecv.size()");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(linkRecv, dataSplitRate));

    const u32 txSliceCount = stepInfo.txSliceIdxs.size();
    const u32 rxSliceCount = stepInfo.rxSliceIdxs.size();
    std::vector<DataSlice> txSrcDstSlices;
    std::vector<DataSlice> rxSrcDstSlices;
    txSrcDstSlices.reserve(txSliceCount);
    rxSrcDstSlices.reserve(rxSliceCount);
    u32 myAlgRank;
    GetAlgRank(myRank_, tempVTopo_[0], myAlgRank);
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 tailSize = templateDataParams.tailSize == 0 ? templateDataParams.sliceSize : templateDataParams.tailSize;
    // 支持不均匀切分的情况下需要把尾部数据放到最后一张卡上
    u64 sliceSize = myAlgRank == tempVTopo_[0].size() - 1 ? tailSize : templateDataParams.sliceSize;
    const u64 baseOffset
        = repeat * ((tempRankSize_ - 1) * templateDataParams.sliceSize + tailSize) + buffInfo_.scratchBuffBaseOff;

    for (u32 j = 0; j < linkNum; j++) {
        txSrcDstSlices.clear();
        rxSrcDstSlices.clear();
        for (u32 i = 0; i < txSliceCount; i++) {
            u64 srcDstOffset = baseOffset + stepInfo.txSliceIdxs[i] * templateDataParams.sliceSize;
            u64 sliceSize = templateDataParams.sliceSize;
            DataSlice txSrcDstSlice(templateDataParams.buffInfo.scratBuffType, srcDstOffset, sliceSize);
            txSrcDstSlices.emplace_back(CalcDataSliceForLinks(txSrcDstSlice, dataSplitRate, j, dataType_));
        }
        SlicesList txSlicesList(txSrcDstSlices, txSrcDstSlices);
        for (u32 i = 0; i < rxSliceCount; i++) {
            u64 srcDstOffset = baseOffset + stepInfo.rxSliceIdxs[i] * templateDataParams.sliceSize;
            DataSlice rxSrcDstSlice(templateDataParams.buffInfo.scratBuffType, srcDstOffset, sliceSize);
            rxSrcDstSlices.emplace_back(CalcDataSliceForLinks(rxSrcDstSlice, dataSplitRate, j, dataType_));
        }
        SlicesList rxSlicesList(rxSrcDstSlices, rxSrcDstSlices);
        TxRxSlicesList txRxSlicesList(std::move(txSlicesList), std::move(rxSlicesList));
        TxRxLinks sendRecvLinks(linkSend[j], linkRecv[j]);
        SendRecvInfo sendRecvInfo(sendRecvLinks, std::move(txRxSlicesList));
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[j], 0, true, dmaMode_),
            HCCL_ERROR("[InsTempScatterNHR] sendrecv failed (j=%u)", j), HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult InsTempScatterNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const
{
    u32 rankSize = tempRankSize_;
    u32 myAlgRank;
    u32 rootAlgRank;
    GetAlgRank(myRank_, tempVTopo_[0], myAlgRank);
    GetAlgRank(root_, tempVTopo_[0], rootAlgRank);
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = myRank_;

    u32 deltaRoot = (rootAlgRank + rankSize - myAlgRank) % rankSize;
    u32 deltaRankPair = 1 << step;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // 判断是否是2的幂
    u32 nRanks = 0; // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (myAlgRank + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = tempVTopo_[0][sendTo];
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (myAlgRank + deltaRankPair) % rankSize;
        u32 rxSliceIdx = myAlgRank;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.fromRank = tempVTopo_[0][recvFrom];
        stepInfo.nSlices = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
