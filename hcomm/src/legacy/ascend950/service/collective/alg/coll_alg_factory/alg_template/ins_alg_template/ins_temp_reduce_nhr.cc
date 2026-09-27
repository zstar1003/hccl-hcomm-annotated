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
#include "ins_temp_reduce_nhr.h"

namespace Hccl {
InsTempReduceNHR::InsTempReduceNHR(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceNHR::~InsTempReduceNHR() {}

HcclResult InsTempReduceNHR::CalcRes(AlgTempResReq& tempResReq)
{
    // NHR 需要的 que Num 为 1
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    CHK_PRT_RET(
        CalcResLinksNHR(myRank_, tempRankSize_, tempVTopo_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceNHR] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chuck 块，给后续的reduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult InsTempReduceNHR::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    // 按 rank 切分数据（与 AllReduceNHR 保持一致）
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    u64 unitAlignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR("[InsTempReduceNHR] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 返回当前rank能处理的数据量和scratch buffer之间的比例关系
 * param: input: 输入数据位置
 * param: output 输出数据位置
 */
u32 InsTempReduceNHR::CalcScratchMultiple(BufferType input, BufferType output)
{
    (void)input;
    (void)output;
    // 单算子模式下需要 1 倍的 scratch（ccl buffer），图/流水（OFFLOAD）模式下不需要
    u32 multiple = 0;
    if (op_.opMode == OpMode::OPBASE) {
        multiple = 1;
    }
    return multiple;
}

HcclResult InsTempReduceNHR::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempReduceNHR][GenExtIns] ReduceNHR begin: rank[%d] start", myRank_);
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    }
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    queNum_ = tempVTopo_.size();

    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceNHR] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // 1. 切片
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(tempAlgParams.sliceSize, sliceInfoVec));

    // 2. PreCopy (OPBASE 模式下将 userIn -> scratch)
    CHK_RET(PreCopy(tempAlgParams, tempInsQues));

    // 3. ReduceScatter 阶段 (pairwise reduce)
    CHK_RET(RunReduceScatter(sliceInfoVec, tempLinks, tempInsQues));

    // 4. PrepareDataForGather 阶段
    CHK_RET(PrepareDataForGather(sliceInfoVec, tempInsQues));

    // 5. Gather 阶段 (将每个 chunk 聚合到 root)
    CHK_RET(RunGather(sliceInfoVec, tempLinks, tempInsQues));

    // 6. PostCopy (OPBASE 且在 root 上将 scratch -> userOut)
    CHK_RET(PostCopy(tempAlgParams, tempInsQues));

    HCCL_INFO("[InsTempReduceNHR][GenExtIns] ReduceNHR finished: rank[%d] end", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::PreCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // 单算子模式，需要先将数据拷贝到cclBuffer
    if (opMode_ == OpMode::OPBASE) {
        reduceInBuffType_ = BufferType::SCRATCH;
        reduceInBuffBaseOff_ = tempAlgParams.buffInfo.inBuffBaseOff;

        if (tempAlgParams.buffInfo.inBuffType != BufferType::SCRATCH) {
            HCCL_INFO("[InsTempReduceNHR][PreCopy] Opbase copy from userIn to scratchBuffer");
            DataSlice usrInSlices = DataSlice(
                tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize);
            DataSlice scratchSlices
                = DataSlice(BufferType::SCRATCH, tempAlgParams.buffInfo.scratchBuffBaseOff, tempAlgParams.sliceSize);
            CHK_RET(LocalCopy(tempInsQues[0], usrInSlices, scratchSlices));
            reduceInBuffBaseOff_ = tempAlgParams.buffInfo.scratchBuffBaseOff;
        } else {
            HCCL_INFO("[InsTempReduceNHR][PreCopy] skip precopy");
        }
    } else {
        // OFFLOAD 图模式直接在用户 buffer 上操作
        HCCL_INFO("[InsTempReduceNHR][PreCopy] offload skip precopy");
        reduceInBuffType_ = tempAlgParams.buffInfo.inBuffType;
        reduceInBuffBaseOff_ = tempAlgParams.buffInfo.inBuffBaseOff;
    }

    reduceOutBuffType_ = tempAlgParams.buffInfo.outBuffType;
    reduceOutBuffBaseOff_ = tempAlgParams.buffInfo.outBuffBaseOff;

    return HcclResult::HCCL_SUCCESS;
}

// 将reduceScatter之后的数据先放到usrOut
HcclResult
InsTempReduceNHR::PrepareDataForGather(const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues)
{
    // 如果是单算子模式，在原来的位置要先做完Gather，然后postCopy把数据放到usrOut
    // 如果是图模式，直接把数据放到usrOUt，然后在usrOut上做Gather
    HCCL_INFO("[InsTempReduceNHR][PrepareDataForGather] prepare data for Gather");

    if (opMode_ == OpMode::OFFLOAD) {
        u64 size = sliceInfoVec[tempVirtRankMap_[myRank_]][0].size;
        u64 srcOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
        u64 dstOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
        DataSlice srcSlice = DataSlice(reduceInBuffType_, reduceInBuffBaseOff_ + srcOffset, size);
        DataSlice dstSlice = DataSlice(reduceOutBuffType_, reduceOutBuffBaseOff_ + dstOffset, size);
        CHK_RET(LocalCopy(tempInsQues[0], srcSlice, dstSlice));
        reduceInBuffType_ = reduceOutBuffType_;
        reduceInBuffBaseOff_ = reduceOutBuffBaseOff_;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // PostCopy 仅在 OPBASE 并且在 root 上执行（root 收到完整结果后写回用户 out）
    RankId rootRank = this->root_; // Executor 在 CreateTemplates 时已调用 SetRoot(op_.root)

    if (myRank_ != rootRank) {
        HCCL_DEBUG("[InsTempReduceNHR][PostCopy] not root, skip postcopy rank[%d]", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    if (opMode_ == OpMode::OPBASE) {
        HCCL_INFO("[InsTempReduceNHR][PostCopy] Opbase root copy from scratchBuffer to userOut");
        DataSlice scratchSlices = DataSlice(reduceInBuffType_, reduceInBuffBaseOff_, tempAlgParams.sliceSize);
        DataSlice usrOutSlices = DataSlice(reduceOutBuffType_, reduceOutBuffBaseOff_, tempAlgParams.sliceSize);
        CHK_RET(LocalCopy(tempInsQues[0], scratchSlices, usrOutSlices));
    } else {
        HCCL_INFO("[InsTempReduceNHR][PostCopy] offload skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::RunReduceScatter(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    std::vector<AicpuNHRStepInfo> stepInfoList;
    CHK_RET(GetStepInfoList(stepInfoList));

    for (auto& stepInfo : stepInfoList) {
        HCCL_DEBUG(
            "[InsTempReduceNHR][RunReduceScatter] step[%u], myRank[%u], toRank[%u], fromRank[%u], nSlices[%u].",
            stepInfo.step, stepInfo.myRank, stepInfo.toRank, stepInfo.fromRank, stepInfo.nSlices);

        const std::vector<LinkData>& linkRecv = tempLinks.at(GetRankFromMap(stepInfo.fromRank));
        const std::vector<LinkData>& linkSend = tempLinks.at(GetRankFromMap(stepInfo.toRank));

        std::vector<DataSlice> txSlices;
        std::vector<DataSlice> rxSlices;

        // 发送和接收 slice 都发生在 reduceInBuffType_ 上（scratch 或用户 buffer）
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].offset + reduceInBuffBaseOff_;
            u64 txSize = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].offset + reduceInBuffBaseOff_;
            u64 rxSize = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].size;

            txSlices.push_back(DataSlice(reduceInBuffType_, txOffset, txSize));
            rxSlices.push_back(DataSlice(reduceInBuffType_, rxOffset, rxSize));
        }

        SendRecvReduceInfo sendRecvReduceInfo{
            {linkSend[0], linkRecv[0]}, {{txSlices, txSlices}, {rxSlices, rxSlices}}, dataType_, redOp_};

        CHK_PRT_RET(
            SendRecvReduce(sendRecvReduceInfo, tempInsQues[0], 0, true, dmaMode_) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[InsTempReduceNHR] RunReduceScatter SendRecvReduce failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::RunGather(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        const std::vector<LinkData>& linkRecv = tempLinks.at(GetRankFromMap(stepInfo.fromRank));
        const std::vector<LinkData>& linkSend = tempLinks.at(GetRankFromMap(stepInfo.toRank));

        std::vector<DataSlice> txSlices;
        std::vector<DataSlice> rxSlices;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].offset + reduceInBuffBaseOff_;
            u64 txSize = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].offset + reduceInBuffBaseOff_;
            u64 rxSize = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].size;

            txSlices.push_back(DataSlice(reduceInBuffType_, txOffset, txSize));
            rxSlices.push_back(DataSlice(reduceInBuffType_, rxOffset, rxSize));
        }

        TxRxLinks sendRecvLinks(linkSend[0], linkRecv[0]);
        TxRxSlicesList sendRecvSlicesList({txSlices, txSlices}, {rxSlices, rxSlices});

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[0], 0, true, dmaMode_) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[InsTempReduceNHR] RunGather send/recv failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
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

        HCCL_DEBUG("[InsTempReduceNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult InsTempReduceNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList)
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
                "[InsTempReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
            txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
            rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempReduceNHR::GetRankFromMap(const u32 rankIdx)
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

} // namespace Hccl
