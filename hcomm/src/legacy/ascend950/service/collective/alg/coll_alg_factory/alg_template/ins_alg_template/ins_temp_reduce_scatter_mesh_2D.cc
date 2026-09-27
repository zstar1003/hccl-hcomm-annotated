/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_2D.h"

#include "log.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
InsTempReduceScatterMesh2D::InsTempReduceScatterMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    xQueNum_ = tempVTopo_[0].size() - 1; // x轴的卡数-1
    yQueNum_ = tempVTopo_[1].size() - 1; // y轴的卡数-1
    xRankSize_ = tempVTopo_[0].size();   // x轴的卡数
    yRankSize_ = tempVTopo_[1].size();   // y轴的卡数
}

InsTempReduceScatterMesh2D::~InsTempReduceScatterMesh2D() {}

u64 InsTempReduceScatterMesh2D::CalcScratchMultiple(const BufferType& inBuffType, const BufferType& outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u32 xyMaxRankSize = max(xRankSize_, yRankSize_);
    u64 scratchMultiple = xyMaxRankSize * (xRankSize_ + yRankSize_);
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh2D::CalcResLinksMesh2D(const u32 linkNumBtwPeers, AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            RankId neighborRank = tempVTopo_[dim][(myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size())];
            tempResReq.links[neighborRank] = linkNumBtwPeers;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    // Mesh 需要的 que Num 为 tempVTopo_[0].size() + tempVTopo_[1].size() - 2
    tempResReq.queNum = (xRankSize_ > 1 && yRankSize_ > 1) ? (xQueNum_ + yQueNum_) : 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    // linkNumBtwPeers_这个在没有绕路的情况下，是设置成1
    CHK_PRT_RET(
        CalcResLinksMesh2D(linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh2D] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], xRankId_)); // 得到当前卡在x轴上的编号
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[1], yRankId_)); // 得到当前卡在y轴上的编号
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    queNum_ = xQueNum_ + yQueNum_;
    u64 sliceNum = tempAlgParams.sliceSize / DataTypeSizeGet(dataType_); // 先计算得到本次迭代处理的数据量
    halfDataSize_ = sliceNum / PARALLEL_SIZE * DataTypeSizeGet(dataType_); // 前一半数据的size
    HCCL_INFO("[InsTempReduceScatterMesh2D] Run Start");
    // 这里不支持绕路的时候，应该就用原始的tempInsQues就行
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh2D] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    PreCopy(tempAlgParams, tempInsQues); // stream 0作为主流，负责把本卡的数据拷贝到scratchbuffer上
    if (queNum_ > 1) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }
    CHK_RET(RunFirstLevel(tempLinks, tempInsQues, tempAlgParams));
    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }
    CHK_RET(RunFirstReduce(tempInsQues, tempAlgParams));
    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }
    RunSecondLevel(tempLinks, tempInsQues, tempAlgParams);
    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }
    RunSecondReduce(tempInsQues, tempAlgParams);
    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterMesh2D::PreCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    u32 xyMaxRankSize = max(xRankSize_, yRankSize_);
    u64 remainDataSize = tempAlgParams.sliceSize - halfDataSize_;
    // 前一半数据，将本卡数据从input拷贝到scratchbuffer
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
        for (u32 yRankId = 0; yRankId < yRankSize_; yRankId++) {
            u32 rankId = yRankId * xRankSize_ + xRankId_; // 同y轴平面的所有卡，
            DataSlice inputRankSlice = DataSlice(
                tempAlgParams.buffInfo.inBuffType,
                tempAlgParams.buffInfo.inBuffBaseOff + rankId * tempAlgParams.inputSliceStride
                    + rpt * tempAlgParams.inputRepeatStride,
                halfDataSize_);
            DataSlice scratchRankSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                tempAlgParams.buffInfo.scratchBuffBaseOff
                    + tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId + xRankId_) + scratchRepeatStride,
                halfDataSize_);
            CHK_RET(LocalCopy(tempInsQues[0], inputRankSlice, scratchRankSlice));
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][PreCopy] myRank[%d] top inputRankSlice: %s, scratchRankSlice: %s",
                myRank_, inputRankSlice.Describe().c_str(), scratchRankSlice.Describe().c_str());
        }
    }
    // 后一半数据，将本卡数据从input拷贝到scratchbuffer
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride
            = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum)
              + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
        for (u32 xRankId = 0; xRankId < xRankSize_; xRankId++) {
            u32 rankId = yRankId_ * xRankSize_ + xRankId; // 同x轴平面的所有卡，
            DataSlice inputRankSlice = DataSlice(
                tempAlgParams.buffInfo.inBuffType,
                tempAlgParams.buffInfo.inBuffBaseOff + rankId * tempAlgParams.inputSliceStride + halfDataSize_
                    + rpt * tempAlgParams.inputRepeatStride,
                remainDataSize);
            DataSlice scratchRankSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                tempAlgParams.buffInfo.scratchBuffBaseOff
                    + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId + yRankId_) + scratchRepeatStride,
                remainDataSize);
            CHK_RET(LocalCopy(tempInsQues[0], inputRankSlice, scratchRankSlice));
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][PreCopy] myRank[%d] bottom inputRankSlice: %s, scratchRankSlice: %s",
                myRank_, inputRankSlice.Describe().c_str(), scratchRankSlice.Describe().c_str());
        }
    }
    HCCL_INFO("[InsTempReduceScatterMesh2D][PreCopy], copy from userIn to scratch");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::SendRecvProcess(
    const ResLinks& tempLinks, std::vector<std::vector<DataSlice>> allSliceVec, std::vector<InsQuePtr>& tempInsQues,
    u32 remoteRank, u32 queIdx) const
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[InsTempReduceScatterMesh2D][SendRecvProcess] empty queue"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    HCCL_DEBUG("[InsTempReduceScatterMesh2D][SendRecvProcess] SendRecvProcess start");
    const std::vector<LinkData>& linkRecv = tempLinks.at(remoteRank);
    const std::vector<LinkData>& linkSend = tempLinks.at(remoteRank);
    SendRecvInfo sendRecvInfo{
        {linkSend[0], linkRecv[0]}, {{allSliceVec[2], allSliceVec[3]}, {allSliceVec[0], allSliceVec[1]}}};

    CHK_PRT_THROW(
        queIdx >= tempInsQues.size(),
        HCCL_ERROR(
            "[InsTempReduceScatterMesh2D] queIdx[%u] is bigger than tempInsQues size[%zu].", queIdx,
            tempInsQues.size()),
        InvalidParamsException, "queIdx is invalid");
    // 做了DMA消减之后只支持PUT
    CHK_PRT_RET(
        SendRecv(sendRecvInfo, tempInsQues[queIdx], 0, true, DmaMode::PUT),
        HCCL_ERROR("[InsTempReduceScatterMesh2D] RunReduceScatter SendReduce failed"), HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

// 前一半数据的先x轴 和 后一半数据的先y轴
HcclResult InsTempReduceScatterMesh2D::RunFirstLevel(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams)
{
    HCCL_INFO("[InsTempReduceScatterMesh2D][RunFirstLevel] myRank[%d]", myRank_);
    u32 xyMaxRankSize = max(xRankSize_, yRankSize_);
    u64 processSize;
    for (u32 queIdx = 0; queIdx < queNum_; queIdx++) {
        u32 remoteRank;
        u32 index;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        if (queIdx < xQueNum_) { // 前xRankSize-1个stream，首先拉取前一半数据
            index = (xRankId_ + 1 + queIdx) % (tempVTopo_[0].size());
            remoteRank = tempVTopo_[0][index];
            processSize = halfDataSize_;
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][RunFirstLevel] queID < xQueNum myRank[%d] toRank[%u] fromRank[%u] "
                "rpt[%u], index[%u]",
                myRank_, remoteRank, remoteRank, tempAlgParams.repeatNum, index);
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
                for (u32 yRankId = 0; yRankId < yRankSize_; yRankId++) {
                    u32 readRankId = yRankId * xRankSize_ + xRankId_;
                    u32 writeRankId = yRankId * xRankSize_ + index;
                    // 数据从其他卡，传输到本卡，接收数据
                    rxSrcSlices.emplace_back(
                        tempAlgParams.buffInfo.inBuffType,
                        tempAlgParams.buffInfo.inBuffBaseOff + readRankId * tempAlgParams.inputSliceStride
                            + rpt * tempAlgParams.inputRepeatStride,
                        processSize);
                    rxDstSlices.emplace_back(
                        tempAlgParams.buffInfo.scratBuffType,
                        tempAlgParams.buffInfo.scratchBuffBaseOff
                            + tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId + index) + scratchRepeatStride,
                        processSize);
                    txSrcSlices.emplace_back(
                        tempAlgParams.buffInfo.inBuffType,
                        tempAlgParams.buffInfo.inBuffBaseOff + writeRankId * tempAlgParams.inputSliceStride
                            + rpt * tempAlgParams.inputRepeatStride,
                        processSize);
                    txDstSlices.emplace_back(
                        tempAlgParams.buffInfo.scratBuffType,
                        tempAlgParams.buffInfo.scratchBuffBaseOff
                            + tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId + xRankId_)
                            + scratchRepeatStride,
                        processSize);
                    HCCL_DEBUG(
                        "[InsTempReduceScatterMesh2D][RunFirstLevel] queID < xQueNum myRank[%d] *****sendrecv*****, "
                        "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s",
                        myRank_, rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                        txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
                }
            }
        } else { // 后yRankSize-1个stream,首先拉取后一半数据
            index = (yRankId_ + 1 + queIdx - xQueNum_) % (tempVTopo_[1].size());
            remoteRank = tempVTopo_[1][index];
            processSize = tempAlgParams.sliceSize - halfDataSize_;
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][RunFirstLevel] queId >= xQueNum myRank[%d] toRank[%u] fromRank[%u], "
                "rpt[%u], index[%u]",
                myRank_, remoteRank, remoteRank, tempAlgParams.repeatNum, index);
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride
                    = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum)
                      + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
                for (u32 xRankId = 0; xRankId < xRankSize_; xRankId++) {
                    u32 readRankId = yRankId_ * xRankSize_ + xRankId; // 同x轴平面的所有卡，
                    u32 writeRankId = index * xRankSize_ + xRankId;
                    rxSrcSlices.emplace_back(
                        tempAlgParams.buffInfo.inBuffType,
                        tempAlgParams.buffInfo.inBuffBaseOff + readRankId * tempAlgParams.inputSliceStride
                            + halfDataSize_ + rpt * tempAlgParams.inputRepeatStride,
                        processSize);
                    rxDstSlices.emplace_back(
                        tempAlgParams.buffInfo.scratBuffType,
                        tempAlgParams.buffInfo.scratchBuffBaseOff
                            + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId + index) + scratchRepeatStride,
                        processSize);
                    txSrcSlices.emplace_back(
                        tempAlgParams.buffInfo.inBuffType,
                        tempAlgParams.buffInfo.inBuffBaseOff + writeRankId * tempAlgParams.inputSliceStride
                            + halfDataSize_ + rpt * tempAlgParams.inputRepeatStride,
                        processSize);
                    txDstSlices.emplace_back(
                        tempAlgParams.buffInfo.scratBuffType, // tempAlgParams.buffInfo.scratBuffType,
                        tempAlgParams.buffInfo.scratchBuffBaseOff
                            + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId + yRankId_)
                            + scratchRepeatStride,
                        processSize);
                    HCCL_DEBUG(
                        "[InsTempReduceScatterMesh2D][RunFirstLevel] queId >= xQueNum myRank[%d] *****sendrecv*****, "
                        "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s",
                        myRank_, rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                        txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
                }
            }
        }
        if (processSize == 0) {
            continue;
        }
        std::vector<std::vector<DataSlice>> allSliceVec = {rxSrcSlices, rxDstSlices, txSrcSlices, txDstSlices};
        CHK_RET(SendRecvProcess(tempLinks, allSliceVec, tempInsQues, remoteRank, queIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterMesh2D::RunFirstReduce(std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams)
{
    HCCL_INFO("[InsTempReduceScatterMesh2D][RunFirstReduce] myRank[%d] rpt[%u]", myRank_, tempAlgParams.repeatNum);
    u32 xyMaxRankSize = max(xRankSize_, yRankSize_);
    u64 processSize = 0;
    // 这里的stream 0和stream xRankSize-1分别负责前一半数据与后一半数据的本地reduce
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
        for (u32 tmpRank = 0; tmpRank < yRankSize_; tmpRank++) { // 前一半数据做local reduce,由这部分的第一个stream做
            processSize = halfDataSize_;
            for (u32 dataIdx = 1; dataIdx < xRankSize_;
                 dataIdx++) { // 原始这个位置已经有数据了，因此从后一片数据开始累加
                DataSlice srcDataSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + (xyMaxRankSize * tmpRank + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize);
                DataSlice dstDataSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + (xyMaxRankSize * tmpRank) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize);
                HCCL_DEBUG(
                    "[InsTempReduceScatterMesh2D][RunFirstReduce] myRank[%d] queId < xQueNum *****LocalReduce*****, "
                    "srcDataSlice: %s, dstDataSlice: %s",
                    myRank_, srcDataSlice.Describe().c_str(), dstDataSlice.Describe().c_str());
                CHK_RET(LocalReduce(tempInsQues[0], srcDataSlice, dstDataSlice, dataType_, redOp_));
            }
        }
    }
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride
            = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum)
              + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
        for (u32 tmpRank = 0; tmpRank < xRankSize_; tmpRank++) { // 后一半数据做local reduce，由这部分的第一个stream做
            processSize = tempAlgParams.sliceSize - halfDataSize_;
            for (u32 dataIdx = 1; dataIdx < yRankSize_;
                 dataIdx++) { // 原始这个位置已经有数据了，因此从后一片数据开始累加
                DataSlice srcDataSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + (xyMaxRankSize * tmpRank + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize);
                DataSlice dstDataSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + (xyMaxRankSize * tmpRank) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize);
                HCCL_DEBUG(
                    "[InsTempReduceScatterMesh2D][RunFirstReduce] myRank[%d] queId >= xQueNum *****LocalReduce*****, "
                    "srcDataSlice: %s, dstDataSlice: %s",
                    myRank_, srcDataSlice.Describe().c_str(), dstDataSlice.Describe().c_str());
                CHK_RET(LocalReduce(tempInsQues[xQueNum_], srcDataSlice, dstDataSlice, dataType_, redOp_));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 后一半数据的后x轴 和 前一半数据的后y轴
HcclResult InsTempReduceScatterMesh2D::RunSecondLevel(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams)
{
    u32 xyMaxRankSize = max(xRankSize_, yRankSize_);
    u64 processSize;
    for (u32 queIdx = 0; queIdx < queNum_; queIdx++) {
        u32 remoteRank;
        u32 index;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        if (queIdx < xQueNum_) { // 前xRankSize-1个stream，后一半数据
            index = (xRankId_ + 1 + queIdx) % (tempVTopo_[0].size());
            remoteRank = tempVTopo_[0][index];
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][RunSecondLevel] queIdx < xQueNum myRank[%d] toRank[%u] fromRank[%u]",
                myRank_, remoteRank, remoteRank);
            processSize = tempAlgParams.sliceSize - halfDataSize_;
            // 这里过来的数据，直接按照queIdx的顺序放置，不一定是按照rankId顺序排列的
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride
                    = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum)
                      + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
                DataSlice rxSrcSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId_) + scratchRepeatStride,
                    processSize);
                DataSlice rxDstSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId_ + queIdx + 1)
                        + scratchRepeatStride,
                    processSize);
                DataSlice txSrcSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * index) + scratchRepeatStride,
                    processSize);
                DataSlice txDstSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * index + queIdx + 1) + scratchRepeatStride,
                    processSize);

                rxSrcSlices.emplace_back(rxSrcSlice);
                rxDstSlices.emplace_back(rxDstSlice);
                txSrcSlices.emplace_back(txSrcSlice);
                txDstSlices.emplace_back(txDstSlice);

                HCCL_DEBUG(
                    "[InsTempReduceScatterMesh2D][RunSecondLevel] queId < xQueNum myRank[%d] *****sendrecv*****, "
                    "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s",
                    myRank_, rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                    txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
            }
        } else { // 后yRankSize-1个stream,前一半数据
            index = (yRankId_ + 1 + queIdx - xQueNum_) % (tempVTopo_[1].size());
            remoteRank = tempVTopo_[1][index];
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][RunSecondLevel] queId >= xQueNum myRank[%d] toRank[%u] fromRank[%u] "
                "rpt[%u]",
                myRank_, remoteRank, remoteRank, tempAlgParams.repeatNum);
            processSize = halfDataSize_;
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
                DataSlice rxSrcSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId_) + scratchRepeatStride,
                    processSize);
                DataSlice rxDstSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId_ + queIdx - xQueNum_ + 1)
                        + scratchRepeatStride,
                    processSize);
                DataSlice txSrcSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * index) + scratchRepeatStride,
                    processSize);
                DataSlice txDstSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff
                        + tempAlgParams.outputSliceStride * (xyMaxRankSize * index + queIdx - xQueNum_ + 1)
                        + scratchRepeatStride,
                    processSize);

                rxSrcSlices.emplace_back(rxSrcSlice);
                rxDstSlices.emplace_back(rxDstSlice);
                txSrcSlices.emplace_back(txSrcSlice);
                txDstSlices.emplace_back(txDstSlice);

                HCCL_DEBUG(
                    "[InsTempReduceScatterMesh2D][RunSecondLevel] queId >= xQueNum myRank[%d] *****sendrecv*****, "
                    "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s",
                    myRank_, rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                    txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
            }
        }
        if (processSize == 0) {
            continue;
        }
        std::vector<std::vector<DataSlice>> allSliceVec = {rxSrcSlices, rxDstSlices, txSrcSlices, txDstSlices};
        CHK_RET(SendRecvProcess(tempLinks, allSliceVec, tempInsQues, remoteRank, queIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::RunSecondReduce(
    std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams)
{
    HCCL_INFO("[InsTempReduceScatterMesh2D][RunSecondReduce] myRank[%d] rpt[%u]", myRank_, tempAlgParams.repeatNum);
    u32 xyMaxRankSize = max(xRankSize_, yRankSize_);
    u64 processSize = 0;
    // 这里的stream 0和stream xRankSize-1分别负责后一半数据与前一半数据的本地reduce
    // 后一半数据做local reduce,由这部分的第一个stream做
    processSize = tempAlgParams.sliceSize - halfDataSize_;
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride
            = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum)
              + tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
        for (u32 dataIdx = 0; dataIdx < xRankSize_; dataIdx++) {
            DataSlice srcSecDataSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                tempAlgParams.buffInfo.scratchBuffBaseOff
                    + (xyMaxRankSize * xRankId_ + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                processSize);
            u64 outOffset
                = tempAlgParams.buffInfo.outBuffBaseOff + halfDataSize_ + rpt * tempAlgParams.outputRepeatStride;
            DataSlice dstSecDataSlice = DataSlice(
                tempAlgParams.buffInfo.outBuffType, // BufferType::OUTPUT,
                outOffset, processSize);
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][RunSecondReduce] myRank[%d] queId < xQueNum *****LocalReduce*****, "
                "srcDataSlice: %s, dstDataSlice: %s",
                myRank_, srcSecDataSlice.Describe().c_str(), dstSecDataSlice.Describe().c_str());
            if (srcSecDataSlice != dstSecDataSlice) {
                if (dataIdx == 0) {
                    CHK_RET(LocalCopy(tempInsQues[0], srcSecDataSlice, dstSecDataSlice));
                } else {
                    CHK_RET(LocalReduce(tempInsQues[0], srcSecDataSlice, dstSecDataSlice, dataType_, redOp_));
                }
            }
        }
    }
    // 前一半数据做local reduce，由这部分的第一个stream做
    processSize = halfDataSize_;
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
        bool hasInplace = false;
        std::vector<DataSlice> srcFirDataSlices;
        std::vector<DataSlice> dstFirDataSlices;
        for (u32 dataIdx = 0; dataIdx < yRankSize_; dataIdx++) {
            DataSlice srcFirDataSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                tempAlgParams.buffInfo.scratchBuffBaseOff
                    + (xyMaxRankSize * yRankId_ + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                processSize);
            u64 outOffset = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
            DataSlice dstFirDataSlice = DataSlice(
                tempAlgParams.buffInfo.outBuffType, // BufferType::OUTPUT,
                outOffset, processSize);
            HCCL_DEBUG(
                "[InsTempReduceScatterMesh2D][RunSecondReduce] myRank[%d] queId >= xQueNum *****LocalReduce*****, "
                "srcDataSlice: %s, dstDataSlice: %s",
                myRank_, srcFirDataSlice.Describe().c_str(), dstFirDataSlice.Describe().c_str());
            if (srcFirDataSlice != dstFirDataSlice) {
#if DATASLICE_ONE
                srcFirDataSlices.push_back(srcFirDataSlice);
                dstFirDataSlices.push_back(dstFirDataSlice);
#else
                if (dataIdx == 0) {
                    CHK_RET(LocalCopy(tempInsQues[xQueNum_], srcFirDataSlice, dstFirDataSlice));
                } else {
                    CHK_RET(LocalReduce(tempInsQues[xQueNum_], srcFirDataSlice, dstFirDataSlice, dataType_, redOp_));
                }
#endif
            } else {
                hasInplace = true;
            }
        }
        for (u32 dataIdx = 0; dataIdx < srcFirDataSlices.size(); dataIdx++) {
            if (!hasInplace && dataIdx == 0) {
                CHK_RET(LocalCopy(tempInsQues[xQueNum_], srcFirDataSlices[dataIdx], dstFirDataSlices[dataIdx]));
            } else {
                CHK_RET(LocalReduce(
                    tempInsQues[xQueNum_], srcFirDataSlices[dataIdx], dstFirDataSlices[dataIdx], dataType_, redOp_));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
