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
#include "ins_temp_all_reduce_nhr.h"

namespace Hccl {
InsTempAllReduceNHR::InsTempAllReduceNHR(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAllReduceNHR::~InsTempAllReduceNHR() {}

HcclResult InsTempAllReduceNHR::CalcRes(AlgTempResReq& tempResReq)
{
    // NHR 需要的 que Num 为 1
    CHK_PRT_RET(
        CalcResLinksNHR(myRank_, tempRankSize_, tempVTopo_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempAllReduceNHR] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    auto& linkReq = tempResReq.links;
    u32 pathNum = 0;
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        auto remoteRank = resReqIter->first;
        if (rank2PathNumMap_.find(remoteRank) == rank2PathNumMap_.end() || rank2PathNumMap_[remoteRank] == 0) {
            HCCL_ERROR("[InsTempAllReduceNHR] No path to remoteRank[%d]", remoteRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (pathNum == 0) {
            pathNum = rank2PathNumMap_[remoteRank];
        } else if (rank2PathNumMap_[remoteRank] != pathNum) {
            HCCL_ERROR(
                "[InsTempAllReduceNHR] Inconsistency pathNum to remoteRanks, Previous consistent pathNum=[%u], "
                "mismatched "
                "remoteRank=[%d], pathNum=[%u]",
                pathNum, remoteRank, rank2PathNumMap_[remoteRank]);
            return HcclResult::HCCL_E_INTERNAL;
        }
        resReqIter->second = pathNum;
    }
    tempResReq.queNum = 1 * pathNum;
    HCCL_INFO("[InsTempAllReduceNHR] tempResReq.queNum = %u", tempResReq.queNum);
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chuck 块，给后续的allreduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult InsTempAllReduceNHR::CalcSlice(const u64 dataSize, const u64 baseOff, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    u64 unitAlignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff + baseOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != baseOff + dataSize),
        HCCL_ERROR(
            "[InsTempAllReduceNHR] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 返回当前rank能处理的数据量和scratch buffer之间的比例关系
 * param: input: 输入数据位置
 * param: output 输出数据位置
 */
u32 InsTempAllReduceNHR::CalcScratchMultiple(BufferType input, BufferType output)
{
    (void)input;
    (void)output;
    // 单算子模式，cclBuffer和usrIn一样大，图模式，不需要cclBuffer
    u32 multiple = 0;
    if (op_.opMode == OpMode::OPBASE) {
        multiple = 1;
    }

    return multiple;
}

HcclResult InsTempAllReduceNHR::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceNHR][GenExtIns] AllReduceNHR begin: rank[%d] start", myRank_);
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    }

    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;

    uint32_t linkNum = tempLinks.begin()->second.size();
    // 流的数量不能少于linkNum
    CHK_PRT_RET(
        linkNum > tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempAllReduceNHR] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(tempLinks.begin()->second, dataSplitRate));

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(tempAlgParams.sliceSize, 0, sliceInfoVec));

    // 将一个RankSliceInfo，拆分成linkNum 个RankSliceInfo
    u64 typeSize = DataTypeSizeGet(dataType_);
    std::vector<RankSliceInfo> sliceInfoVecForAllLinks(linkNum);
    for (auto sliceInfoPerRank : sliceInfoVec) {
        std::vector<std::vector<SliceInfo>> sliceInfoPerRankForAllLinks(linkNum);
        for (auto sliceInfo : sliceInfoPerRank) {
            u64 size = sliceInfo.size;
            u64 offset = sliceInfo.offset;
            u64 AccSize = 0;
            u64 dataCnt = size / typeSize;
            vector<SliceInfo> sliceInfoForAllLinks(linkNum);
            for (u32 linkIdx = 0; linkIdx < linkNum; linkIdx++) {
                if (linkIdx != linkNum - 1) {
                    sliceInfoForAllLinks[linkIdx].size
                        = static_cast<u64>(static_cast<float>(dataCnt) * dataSplitRate[linkIdx]) * typeSize;
                } else {
                    sliceInfoForAllLinks[linkIdx].size = size - AccSize;
                }
                sliceInfoForAllLinks[linkIdx].offset = offset + AccSize;
                AccSize += sliceInfoForAllLinks[linkIdx].size;
            }
            for (u32 linkIdx = 0; linkIdx < linkNum; linkIdx++) {
                sliceInfoPerRankForAllLinks[linkIdx].emplace_back(sliceInfoForAllLinks[linkIdx]);
            }
        }
        for (u32 linkIdx = 0; linkIdx < linkNum; linkIdx++) {
            sliceInfoVecForAllLinks[linkIdx].emplace_back(sliceInfoPerRankForAllLinks[linkIdx]);
        }
    }

    // 预拷贝
    CHK_RET(PreCopy(tempAlgParams, tempInsQues));

    u32 mainQueIdx = 0;
    // 流间前同步，主流通知从流，只有一个流则不做任何事
    CHK_RET(PreSyncQues(tempInsQues, mainQueIdx));

    // 主从流执行nhr
    // 待修改数据切分方式
    for (uint32_t linkIdx = 0; linkIdx < linkNum; linkIdx++) {
        CHK_RET(RunReduceScatter(sliceInfoVecForAllLinks[linkIdx], tempLinks, tempInsQues, linkIdx));
        CHK_RET(PrepareDataForAllGather(sliceInfoVecForAllLinks[linkIdx], tempInsQues, linkIdx));
        CHK_RET(RunAllGather(sliceInfoVecForAllLinks[linkIdx], tempLinks, tempInsQues, linkIdx));
    }
    // 流间后同步，从流通知主流
    CHK_RET(PostSyncQues(tempInsQues, mainQueIdx));
    // 结果拷贝
    CHK_RET(PostCopy(tempAlgParams, tempInsQues));
    HCCL_INFO("[InsTempAllReduceNHR][GenExtIns] AllReduceNHR finished: rank[%d] end", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::PreCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // 单算子模式，需要先将数据拷贝到cclBuffer
    if (opMode_ == OpMode::OPBASE) {
        nhrInBuffType_ = BufferType::SCRATCH;
        nhrInBuffBaseOff_ = tempAlgParams.buffInfo.inBuffBaseOff;

        if (tempAlgParams.buffInfo.inBuffType != BufferType::SCRATCH) {
            HCCL_INFO("[InsTempAllReduceNHR][PreCopy] Opbase copy from userIn to scratchBuffer");
            DataSlice usrInSlices = DataSlice(
                tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize);
            DataSlice scratchSlices
                = DataSlice(BufferType::SCRATCH, tempAlgParams.buffInfo.scratchBuffBaseOff, tempAlgParams.sliceSize);
            CHK_RET(LocalCopy(tempInsQues[0], usrInSlices, scratchSlices));

            nhrInBuffBaseOff_ = tempAlgParams.buffInfo.scratchBuffBaseOff;
        } else {
            HCCL_INFO("[InsTempAllReduceNHR][PreCopy] skip precopy");
        }
    } else {
        HCCL_INFO("[InsTempAllReduceNHR][PreCopy] offload skip precopy");
        nhrInBuffType_ = tempAlgParams.buffInfo.inBuffType;
        nhrInBuffBaseOff_ = tempAlgParams.buffInfo.inBuffBaseOff;
    }

    nhrOutBuffType_ = tempAlgParams.buffInfo.outBuffType;
    nhrOutBuffBaseOff_ = tempAlgParams.buffInfo.outBuffBaseOff;

    return HcclResult::HCCL_SUCCESS;
}

// 将reduceScatter之后的数据先放到usrOut
HcclResult InsTempAllReduceNHR::PrepareDataForAllGather(
    const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues, u32 linkIdx)
{
    // 如果是单算子模式，在原来的位置要先做完allGather，然后postCopy把数据放到usrOut
    // 如果是图模式，直接把数据放到usrOUt，然后在usrOut上做allGather
    HCCL_INFO("[InsTempAllReduceNHR][PrepareDataForAllGather] prepare data for allGather");

    if (opMode_ == OpMode::OFFLOAD) {
        u64 size = sliceInfoVec[tempVirtRankMap_[myRank_]][0].size;
        u64 srcOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
        u64 dstOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
        DataSlice srcSlice = DataSlice(nhrInBuffType_, nhrInBuffBaseOff_ + srcOffset, size);
        DataSlice dstSlice = DataSlice(nhrOutBuffType_, nhrOutBuffBaseOff_ + dstOffset, size);
        CHK_RET(LocalCopy(tempInsQues[linkIdx], srcSlice, dstSlice));

        nhrInBuffType_ = nhrOutBuffType_;
        nhrInBuffBaseOff_ = nhrOutBuffBaseOff_;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // 单算子模式，需要将数据拷贝到usrOut
    if (opMode_ == OpMode::OPBASE) {
        HCCL_INFO("[InsTempAllReduceNHR][PostCopy] Opbase copy from scratchBuffer to userOut");
        DataSlice scratchSlices = DataSlice(nhrInBuffType_, nhrInBuffBaseOff_, tempAlgParams.sliceSize);
        DataSlice usrOutSlices = DataSlice(nhrOutBuffType_, nhrOutBuffBaseOff_, tempAlgParams.sliceSize);
        CHK_RET(LocalCopy(tempInsQues[0], scratchSlices, usrOutSlices));
    } else {
        HCCL_INFO("[InsTempAllReduceNHR][PostCopy] offload skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::RunReduceScatter(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, u32 linkIdx)
{
    std::vector<AicpuNHRStepInfo> stepInfoList;
    GetStepInfoList(stepInfoList);
    for (auto& stepInfo : stepInfoList) {
        HCCL_DEBUG(
            "[InsTempAllReduceNHR][RunReduceScatter] step[%u], myRank[%u], toRank[%u], fromRank[%u], nSlices[%u].",
            stepInfo.step, stepInfo.myRank, stepInfo.toRank, stepInfo.fromRank, stepInfo.nSlices);

        const std::vector<LinkData>& linkRecv = tempLinks.at(GetRankFromMap(stepInfo.fromRank));
        const std::vector<LinkData>& linkSend = tempLinks.at(GetRankFromMap(stepInfo.toRank));
        std::vector<DataSlice> txSlices;
        std::vector<DataSlice> rxSlices;

        // 在 nhrInBuffType_ 上进行 ReduceScatter 操作
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].offset + nhrInBuffBaseOff_;
            u64 txSize = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].offset + nhrInBuffBaseOff_;
            u64 rxSize = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].size;
            DataSlice txSlice = DataSlice(nhrInBuffType_, txOffset, txSize);
            DataSlice rxSlice = DataSlice(nhrInBuffType_, rxOffset, rxSize);
            txSlices.push_back(txSlice);
            rxSlices.push_back(rxSlice);
        }
        SendRecvReduceInfo sendRecvReduceInfo{
            {linkSend[linkIdx], linkRecv[linkIdx]}, {{txSlices, txSlices}, {rxSlices, rxSlices}}, dataType_, redOp_};
        CHK_PRT_RET(
            SendRecvReduce(sendRecvReduceInfo, tempInsQues[linkIdx], 0, true, dmaMode_) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[InsTempAllReduceNHR] RunReduceScatter SendRecvReduce failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::RunAllGather(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, u32 linkIdx)
{
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        const std::vector<LinkData>& linkRecv = tempLinks.at(GetRankFromMap(stepInfo.fromRank));
        const std::vector<LinkData>& linkSend = tempLinks.at(GetRankFromMap(stepInfo.toRank));

        std::vector<DataSlice> txSlices;
        std::vector<DataSlice> rxSlices;

        HCCL_DEBUG(
            "[InsTempAllReduceNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, tempRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].offset + nhrInBuffBaseOff_;
            u64 txSize = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].offset + nhrInBuffBaseOff_;
            u64 rxSize = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].size;
            DataSlice txSlice = DataSlice(nhrInBuffType_, txOffset, txSize);
            DataSlice rxSlice = DataSlice(nhrInBuffType_, rxOffset, rxSize);
            txSlices.push_back(txSlice);
            rxSlices.push_back(rxSlice);
        }

        TxRxLinks sendRecvLinks(linkSend[linkIdx], linkRecv[linkIdx]);
        TxRxSlicesList sendRecvSlicesList({txSlices, txSlices}, {rxSlices, rxSlices});

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[linkIdx], 0, true, dmaMode_) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[InsTempAllReduceNHR] RunAllGather send/recv failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
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

        HCCL_DEBUG("[InsTempAllReduceNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult InsTempAllReduceNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList)
{
    // 将本 rank 号转换成算法使用的索引号
    u32 rankIdx = tempVirtRankMap_[myRank_];
    stepInfoList.clear();

    u32 nSteps = GetNHRStepNum(tempRankSize_);
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 sendTo = (rankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
        u32 recvFrom = (rankIdx + deltaRank) % tempRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (tempRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 txSliceIdx = sendTo;
        u32 rxSliceIdx = rankIdx;

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = rankIdx;
        currStepInfo.nSlices = nSlices;
        currStepInfo.toRank = sendTo;
        currStepInfo.fromRank = recvFrom;

        // 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG(
                "[InsTempAllReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx,
                rxSliceIdx);
            txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
            rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempAllReduceNHR::GetRankFromMap(const u32 rankIdx)
{
    RankId rank = -1;
    HCCL_INFO("[InsTempAllReduceNHR] GetRankFromMap");
    for (auto& pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}
} // namespace Hccl
