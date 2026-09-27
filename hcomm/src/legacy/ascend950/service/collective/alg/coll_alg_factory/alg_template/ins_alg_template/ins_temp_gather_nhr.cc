/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_gather_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "alg_data_trans_wrapper.h"
#include "dev_mode.h"
#include "log.h"

namespace Hccl {
InsTempGatherNHR::InsTempGatherNHR(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempGatherNHR::~InsTempGatherNHR() {}

//  NHR 算法需要的资源计算
HcclResult InsTempGatherNHR::CalcRes(AlgTempResReq& tempResReq)
{
    // NHR 需要的 que Num 为 1
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    CHK_PRT_RET(
        CalcResLinksNHR(myRank_, tempRankSize_, tempVTopo_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempGatherNHR] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

uint64_t InsTempGatherNHR::GetExpandedMode() const { return DeviceMode::AICPU; }

/*
按照mesh的方式计算SliceInfo，例如N张卡，就是N份slice
*/
HcclResult
InsTempGatherNHR::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    // 一般情况下，nhr的temp是单级的, Gather nhr的dataSize为output大小
    CHK_PRT_RET(
        tempVTopo_.size() != 1,
        HCCL_ERROR(
            "[CollAlgFactory] [InsTempGatherNHR], tempVtopo size is [%zu] one stage NHR only support one template.",
            tempVTopo_.size()),
        HcclResult::HCCL_E_INTERNAL);

    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoNHR(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherNHR::PreCopy(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues)
{
    // 前拷贝数据量
    u64 preCopyDataSize = sliceInfoVec[tempVirtRankMap_[myRank_]][0].size;
    // 前拷贝数据在 main buffer 上的偏移
    u64 preCopyOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset + mainBufferBaseOffset_;

    HCCL_INFO(
        "[InsTempGatherNHR][PreCopy] mainBufferType[%d], preCopyOffset[%llu], preCopyDataSize[%llu]", mainBufferType_,
        preCopyOffset, preCopyDataSize);
    if (preCopyDataSize == 0) {
        HCCL_INFO("[InsTempGatherNHR][PreCopy] preCopyDataSize is 0, no need copy");
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempFuncs.isForepart && opMode_ == OpMode::OPBASE) {
        // 单算子模式下，第一个算子，直接使用 user data 拷贝
        HCCL_INFO("[InsTempGatherNHR][PreCopy] Opbase Forepart, copy base on user data");
        CHK_RET(LocalCopySlices(tempInsQues[0], tempFuncs.usrData.usrInSlices, tempFuncs.usrData.scratchInSlices));
    } else if (tempFuncs.isForepart && opMode_ == OpMode::OFFLOAD) {
        // 图模式下，第一个算子，从 inBuff 拷贝到 scratchBuff
        if (buffInfo_.inBuffType == mainBufferType_ && buffInfo_.inBuffBaseOff == preCopyOffset) {
            // 如果 inBuff 就是 mainBuffer，不需要拷贝
            HCCL_INFO("[InsTempGatherNHR][PreCopy] isForepart inBuff is same as scratchBuff, no need pre copy");
        } else {
            HCCL_INFO("[InsTempGatherNHR][PreCopy] isForepart, copy from inBuff to scratchBuff");
            DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, buffInfo_.inBuffBaseOff, preCopyDataSize);
            DataSlice dstSlice = DataSlice(mainBufferType_, preCopyOffset, preCopyDataSize);
            CHK_RET(LocalCopy(tempInsQues[0], srcSlice, dstSlice));
        }
    } else if (tempFuncs.forAlgSeqComb) {
        // 作为融合算子的一部分，需要将 inBuff 拷贝到 scratchbuff
        if (buffInfo_.inBuffType == mainBufferType_ && buffInfo_.inBuffBaseOff == preCopyOffset) {
            // 如果 inBuff 就是 mainBuffer，不需要拷贝
            HCCL_INFO("[InsTempGatherNHR][PreCopy] forAlgSeqComb inBuff is same as scratchBuff, no need pre copy");
        } else {
            HCCL_INFO("[InsTempGatherNHR][PreCopy] forAlgSeqComb, copy from inBuff to scratchBuff");
            DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, buffInfo_.inBuffBaseOff, preCopyDataSize);
            DataSlice dstSlice = DataSlice(mainBufferType_, preCopyOffset, preCopyDataSize);
            CHK_RET(LocalCopy(tempInsQues[0], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherNHR::PostCopy(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues)
{
    if (u32(myRank_) != root_) { // 非root节点不需要后拷贝
        return HcclResult::HCCL_SUCCESS;
    }

    // 后拷贝数据量
    u64 postCopyDataSize = sliceInfoVec.back().back().size + sliceInfoVec.back().back().offset;
    // 后拷贝数据在 main buffer 上的偏移
    u64 postCopyOffset = mainBufferBaseOffset_;
    HCCL_INFO(
        "[InsTempGatherNHR][PostCopy] mainBufferType[%d], postCopyOffset[%llu], postCopyDataSize[%llu]",
        mainBufferType_, postCopyOffset, postCopyDataSize);

    // 通信后数据全部在 scratch 上，如果是单算子模式, 并且是最后一步算子，需要将数据从 scratch 拷贝到 userOut
    if (tempFuncs.isBottom && opMode_ == OpMode::OPBASE) {
        HCCL_INFO("[InsTempGatherNHR][PostCopy] Opbase && isBottom, copy from outBuff to userOut");
        CHK_RET(LocalCopySlices(tempInsQues[0], tempFuncs.usrData.scratchOutSlices, tempFuncs.usrData.usrOutSlices));
    } else {
        if (buffInfo_.outBuffType == mainBufferType_ && buffInfo_.outBuffBaseOff == postCopyOffset) {
            // 如果 mainBuffer 就是 OutBuffer，不需要拷贝
            HCCL_INFO("[InsTempGatherNHR][PostCopy] outBuff is same as scratchBuff, no need post copy");
        } else {
            HCCL_INFO("[InsTempGatherNHR][PostCopy] , copy from scratchBuff to outBuff");
            DataSlice srcSlice = DataSlice(mainBufferType_, postCopyOffset, postCopyDataSize);
            DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, buffInfo_.outBuffBaseOff, postCopyDataSize);
            CHK_RET(LocalCopy(tempInsQues[0], srcSlice, dstSlice));
        }

        // 图模式或者单算子模式下非第一个算子，需要将数据从 scratch 拷贝到 outBuff
        HCCL_INFO("[InsTempGatherNHR][PreCopy] not first op, not seq comb, copy from inBuff to outBuff");
        DataSlice srcSlice = DataSlice(mainBufferType_, postCopyOffset, postCopyDataSize);
        DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, buffInfo_.outBuffBaseOff, postCopyDataSize);
        CHK_RET(LocalCopy(tempInsQues[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherNHR::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempGatherNHR] Run start");
    // 初始化参数
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = buffInfo;
    queNum_ = tempVTopo_.size();
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempGatherNHR] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // Gather NHR 算子在通信过程中需要使用 scratch, 主要的通信过程也是在scratch上进行的
    mainBufferType_ = buffInfo_.scratBuffType;
    mainBufferBaseOffset_ = buffInfo_.scratchBuffBaseOff;

    HCCL_INFO(
        "[InsTempGatherNHR Run]RankID:[%d], root:[%u], isForepart:[%d], isBottom:[%d]", myRank_, root_,
        tempFuncs.isForepart, tempFuncs.isBottom);

    // 前拷贝
    PreCopy(tempFuncs, sliceInfoVec, tempInsQues);

    std::vector<AicpuNHRStepInfo> nhrSteps;
    GetGatherStepInfo(nhrSteps);

    for (auto& nhrstep : nhrSteps) {
        CHK_PRT_RET(
            BatchTxRx(nhrstep, tempLinks, tempInsQues[0], sliceInfoVec),
            HCCL_ERROR("[InsTempGatherNHR] BatchTxRx failed"), HcclResult::HCCL_E_INTERNAL);
    }

    // 后拷贝
    PostCopy(tempFuncs, sliceInfoVec, tempInsQues);
    return HcclResult::HCCL_SUCCESS;
}

// Send multiple DataSlices
HcclResult InsTempGatherNHR::BatchTxRx(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec)
{
    // 只有Tx,使用send指令
    if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() == 0) {
        BatchSend(stepInfo, tempLinks, queue, sliceInfoVec, mainBufferType_, mainBufferBaseOffset_);
    }
    // 只有Rx，使用recv指令
    else if (stepInfo.txSliceIdxs.size() == 0 && stepInfo.rxSliceIdxs.size() > 0) {
        BatchRecv(stepInfo, tempLinks, queue, sliceInfoVec, mainBufferType_, mainBufferBaseOffset_);
    }
    // 既有Tx又有Rx，使用SendRecv指令
    else if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() > 0) {
        BatchSR(stepInfo, tempLinks, queue, sliceInfoVec, mainBufferType_, mainBufferBaseOffset_);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherNHR::BatchSend(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
    BufferType memType, u32 memOffset) const
{
    CHK_PRT_RET(
        tempLinks.count(stepInfo.toRank) == 0,
        HCCL_ERROR("[InsTempGatherNHR][BatchSend] rank [%u] not found in links map", stepInfo.toRank),
        HcclResult::HCCL_E_INTERNAL);

    const LinkData& linkSend = tempLinks.at(stepInfo.toRank)[0];
    std::vector<DataSlice> txSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs[i];
        DataSlice txSrcDstSlice
            = DataSlice(memType, memOffset + sliceInfoVec[txId][0].offset, sliceInfoVec[txId][0].size);
        txSlices.push_back(txSrcDstSlice);
    }
    SlicesList txSlicesList(txSlices, txSlices);
    DataInfo sendData(linkSend, txSlicesList);
    CHK_PRT_RET(Send(sendData, queue), HCCL_ERROR("[InsTempGatherNHR] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherNHR::BatchRecv(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
    BufferType memType, u32 memOffset) const
{
    CHK_PRT_RET(
        tempLinks.count(stepInfo.fromRank) == 0,
        HCCL_ERROR("[InsTempGatherNHR][BatchRecv] rank [%u] not found in links map", stepInfo.fromRank),
        HcclResult::HCCL_E_INTERNAL);

    const LinkData& linkRecv = tempLinks.at(stepInfo.fromRank)[0];
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
        Recv(recvData, queue), HCCL_ERROR("[InsTempGatherNHR] BatchTxRx Recv failed"), HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherNHR::BatchSR(
    AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
    BufferType memType, u32 memOffset) const
{
    CHK_PRT_RET(
        tempLinks.count(stepInfo.toRank) == 0,
        HCCL_ERROR("[InsTempGatherNHR][BatchSR] rank [%u] not found in links map", stepInfo.toRank),
        HcclResult::HCCL_E_INTERNAL);
    const LinkData& linkSend = tempLinks.at(stepInfo.toRank)[0];
    CHK_PRT_RET(
        tempLinks.count(stepInfo.fromRank) == 0,
        HCCL_ERROR("[InsTempGatherNHR][BatchSR] rank [%u] not found in links map", stepInfo.fromRank),
        HcclResult::HCCL_E_INTERNAL);
    const LinkData& linkRecv = tempLinks.at(stepInfo.fromRank)[0];
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
        SendRecv(sendRecvInfo, queue, 0, true, dmaMode_), HCCL_ERROR("[InsTempGatherNHR] BatchTxRx SendRecv failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult InsTempGatherNHR::GetScatterStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const
{
    u32 rankSize = tempRankSize_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = myRank_;

    u32 deltaRoot = (root_ + rankSize - myRank_) % rankSize;
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
        u32 sendTo = (myRank_ + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = sendTo;
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (myRank_ + deltaRankPair) % rankSize;
        u32 rxSliceIdx = myRank_;
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

HcclResult InsTempGatherNHR::GetGatherStepInfo(std::vector<AicpuNHRStepInfo>& nhrSteps) const
{
    nhrSteps.clear();
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    nhrSteps.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        // 复用 Scatter NHR 的计算方法,只不过要把步骤顺序和收发端反过来
        u32 stepIdx = nSteps - step - 1;
        GetScatterStepInfo(step, nSteps, nhrSteps[stepIdx]);
        nhrSteps[stepIdx].step = step;
        u32 tmp = nhrSteps[stepIdx].toRank;
        nhrSteps[stepIdx].toRank = nhrSteps[stepIdx].fromRank;
        nhrSteps[stepIdx].fromRank = tmp;
        nhrSteps[stepIdx].txSliceIdxs.swap(nhrSteps[stepIdx].rxSliceIdxs);
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
