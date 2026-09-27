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

#include "alg_data_trans_wrapper.h"
#include "ins_alg_template/ins_temp_broadcast_nhr.h"

namespace Hccl {
InsTempBroadcastNHR::InsTempBroadcastNHR(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempBroadcastNHR::~InsTempBroadcastNHR() {}

HcclResult InsTempBroadcastNHR::CalcRes(AlgTempResReq& tempResReq)
{
    // NHR 需要的 que Num 为 1
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    CHK_PRT_RET(
        CalcResLinksNHR(myRank_, tempRankSize_, tempVTopo_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempBroadCastNHR] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempBroadcastNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    if (op_.opMode == OpMode::OPBASE) {
        return 1;
    } else {
        return 0;
    }
}

// SliceInfoVec for NHR
HcclResult InsTempBroadcastNHR::CalcDataSliceInfo(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    AllignInfo allignInfo = {false, 0, dataType_};

    u64 unitAlignSize;
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));
    sliceInfoVec.clear();
    sliceInfoVec.resize(tempRankSize_);

    u64 chunkSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx].push_back(slice);
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR("[InsTempBroadcastNHR] Rank [%d], SliceInfo calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempBroadcastNHR::PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const
{
    if (opMode_ == OpMode::OPBASE && (u32(myRank_) != root_)) {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Opbase && isBottom, copy from outBuff to userOut");
        u64 inOffset = tempAlgParams.buffInfo.scratchBuffBaseOff;

        DataSlice usrInSlice = DataSlice(BufferType::SCRATCH, inOffset, tempAlgParams.sliceSize);
        DataSlice usrOutSlice
            = DataSlice(BufferType::INPUT, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize);

        HCCL_INFO(
            "PostCopy usrInSlice: %s, usrOutSlice: %s", usrInSlice.Describe().c_str(), usrOutSlice.Describe().c_str());

        CHK_RET(LocalCopy(tempInsQues[0], usrInSlice, usrOutSlice));
    } else {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Offload Model, skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempBroadcastNHR::PreCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const
{
    if (opMode_ == OpMode::OPBASE && (u32(myRank_) == root_)) {
        DataSlice usrInSlice
            = DataSlice(BufferType::INPUT, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize);
        DataSlice usrOutSlice
            = DataSlice(BufferType::SCRATCH, tempAlgParams.buffInfo.scratchBuffBaseOff, tempAlgParams.sliceSize);
        CHK_PRT_RET(
            LocalCopy(tempInsQues[0], usrInSlice, usrOutSlice),
            HCCL_ERROR("[InsTempBroadcastNHR] RunScatter userIn to cclIn copy failed"),

            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Offload Model, skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::GetAllGatherStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
{
    u32 rankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (rankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 sendTo = (rankIdx + deltaRank) % tempRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (tempRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + tempRankSize_) % tempRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[InsTempBroadcastNHR][GetStepInfoForAllGather] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx,
            rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult InsTempBroadcastNHR::GetScatterStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const
{
    u32 rankIdx = tempVirtRankMap_.at(myRank_);
    u32 rootIdx = tempVirtRankMap_.at(root_);
    u32 rankSize = tempRankSize_;

    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = INVALID_U32;
    stepInfo.fromRank = INVALID_U32;
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    u32 deltaRoot = (rootIdx + rankSize - rankIdx) % rankSize;
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
        u32 sendTo = (rankIdx + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = sendTo;
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (rankIdx + deltaRankPair) % rankSize;
        u32 rxSliceIdx = rankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.fromRank = recvFrom;
        stepInfo.nSlices = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::RunScatter(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    // nhr主体部分,从ScratchIn计算，结果放至ScratchOut上, 该部分均从inType搬运到outType
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetScatterStepInfo(step, nSteps, stepInfo));
        CHK_PRT_RET(
            BatchTxRx(stepInfo, tempLinks, tempInsQues[0], sliceInfoVec),
            HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempBroadcastNHR::GetRankFromMap(const u32 rankIdx) const
{
    RankId rank = -1;
    for (auto& pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}

HcclResult InsTempBroadcastNHR::RunAllGather(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    u32 nSteps = GetNHRStepNum(tempRankSize_);

    BufferType buffType = opMode_ == OpMode::OPBASE ? buffInfo_.scratBuffType : buffInfo_.outBuffType;
    u64 memOffset = opMode_ == OpMode::OPBASE ? buffInfo_.scratchBuffBaseOff : buffInfo_.outBuffBaseOff;

    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetAllGatherStepInfo(step, nSteps, stepInfo));

        const std::vector<LinkData>& linkRecv = tempLinks.at(GetRankFromMap(stepInfo.fromRank));
        const std::vector<LinkData>& linkSend = tempLinks.at(GetRankFromMap(stepInfo.toRank));

        std::vector<DataSlice> txSlices;
        std::vector<DataSlice> rxSlices;

        HCCL_DEBUG(
            "[InsTempBroadcastNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, tempRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].offset + memOffset;
            u64 txSize = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].offset + memOffset;
            u64 rxSize = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].size;
            DataSlice txSlice = DataSlice(buffType, txOffset, txSize);
            DataSlice rxSlice = DataSlice(buffType, rxOffset, rxSize);
            txSlices.push_back(txSlice);
            rxSlices.push_back(rxSlice);
        }

        TxRxLinks sendRecvLinks(linkSend[0], linkRecv[0]);
        TxRxSlicesList sendRecvSlicesList({txSlices, txSlices}, {rxSlices, rxSlices});

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[0], 0, true, dmaMode_),
            HCCL_ERROR("[InsTempBroadcastNHR] RunAllGather send failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

// Send multiple DataSlices
HcclResult InsTempBroadcastNHR::BatchTxRx(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec)
{
    BufferType memType = (opMode_ == OpMode::OPBASE) ? BufferType::SCRATCH : BufferType::INPUT;
    u64 memOffset = (opMode_ == OpMode::OPBASE) ? buffInfo_.scratchBuffBaseOff : buffInfo_.inBuffBaseOff;
    // 只有Tx,使用send指令
    if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() == 0) {
        CHK_RET(BatchSend(stepInfo, tempLinks, queue, sliceInfoVec, memType, memOffset));
    }
    // 只有Rx，使用recv指令
    else if (stepInfo.txSliceIdxs.size() == 0 && stepInfo.rxSliceIdxs.size() > 0) {
        CHK_RET(BatchRecv(stepInfo, tempLinks, queue, sliceInfoVec, memType, memOffset));
    }
    // 既有Tx又有Rx，使用SendRecv指令
    else if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() > 0) {
        CHK_RET(BatchSR(stepInfo, tempLinks, queue, sliceInfoVec, memType, memOffset));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchSend(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
    BufferType memType, u64 memOffset) const
{
    const LinkData& linkSend = tempLinks.at(GetRankFromMap(stepInfo.toRank))[0];
    std::vector<DataSlice> txSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs[i];
        DataSlice txSrcDstSlice
            = DataSlice(memType, memOffset + sliceInfoVec[txId][0].offset, sliceInfoVec[txId][0].size);
        txSlices.push_back(txSrcDstSlice);
    }
    SlicesList txSlicesList(txSlices, txSlices);
    DataInfo sendData(linkSend, txSlicesList);
    CHK_PRT_RET(
        Send(sendData, queue), HCCL_ERROR("[InsTempBroadcastNHR] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchRecv(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
    BufferType memType, u64 memOffset) const
{
    const LinkData& linkRecv = tempLinks.at(GetRankFromMap(stepInfo.fromRank))[0];
    std::vector<DataSlice> rxSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs[i];
        DataSlice rxSrcDstSlice
            = DataSlice(memType, memOffset + sliceInfoVec[rxId][0].offset, sliceInfoVec[rxId][0].size);
        rxSlices.push_back(rxSrcDstSlice);
    }
    SlicesList rxSlicesList(rxSlices, rxSlices);
    DataInfo recvData(linkRecv, rxSlicesList);
    CHK_PRT_RET(
        Recv(recvData, queue), HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx Recv failed"), HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchSR(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
    BufferType memType, u64 memOffset) const
{
    const LinkData& linkSend = tempLinks.at(GetRankFromMap(stepInfo.toRank))[0];
    const LinkData& linkRecv = tempLinks.at(GetRankFromMap(stepInfo.fromRank))[0];
    TxRxLinks linkSendRecv = {linkSend, linkRecv};

    std::vector<DataSlice> txSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs[i];
        DataSlice txSrcDstSlice
            = DataSlice(memType, memOffset + sliceInfoVec[txId][0].offset, sliceInfoVec[txId][0].size);
        txSlices.push_back(txSrcDstSlice);
    }
    SlicesList txSlicesList(txSlices, txSlices);
    std::vector<DataSlice> rxSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs[i];
        DataSlice rxSrcDstSlice
            = DataSlice(memType, memOffset + sliceInfoVec[rxId][0].offset, sliceInfoVec[rxId][0].size);
        rxSlices.push_back(rxSrcDstSlice);
    }
    SlicesList rxSlicesList(rxSlices, rxSlices);
    TxRxSlicesList txRxSlicesList(txSlicesList, rxSlicesList);
    SendRecvInfo sendRecvInfo(linkSendRecv, txRxSlicesList);
    CHK_PRT_RET(
        SendRecv(sendRecvInfo, queue, 0, true, dmaMode_), HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx SendRecv failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    }
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = templateDataParams.buffInfo;
    HCCL_INFO("[InsTempBroadcastNHR] BroadcastNHR entry.");

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcDataSliceInfo(templateDataParams.sliceSize, sliceInfoVec));

    queNum_ = tempVTopo_.size();
    ;
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempBroadcastNHR] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO("[InsTempBroadcastNHR Run]RankID:[%d], root:[%u]", myRank_, root_);

    CHK_RET(PreCopy(templateDataParams, tempInsQues));
    CHK_RET(RunScatter(sliceInfoVec, tempLinks, tempInsQues));
    CHK_RET(RunAllGather(sliceInfoVec, tempLinks, tempInsQues));
    CHK_RET(PostCopy(templateDataParams, tempInsQues));

    HCCL_INFO("[InsTempBroadcastNHR] BroadcastNHR finish.");

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
