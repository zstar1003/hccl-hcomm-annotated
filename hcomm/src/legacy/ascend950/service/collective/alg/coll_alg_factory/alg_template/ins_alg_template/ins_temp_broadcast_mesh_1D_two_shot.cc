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
#include "ins_alg_template/ins_temp_broadcast_mesh_1D_two_shot.h"

namespace Hccl {
InsTempBroadcastMesh1DTwoShot::InsTempBroadcastMesh1DTwoShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempBroadcastMesh1DTwoShot::~InsTempBroadcastMesh1DTwoShot() {}

HcclResult InsTempBroadcastMesh1DTwoShot::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = (tempVTopo_[0].size() > 1) ? (tempVTopo_[0].size() - 1) : 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_DEBUG(
        "[InsTempBroadcastMesh1DTwoShot] Rank[%d], VtopoSize[%lu], requiredQue Num [%u].", myRank_,
        tempVTopo_[0].size(), tempResReq.queNum);

    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempBroadcastMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    if (op_.opMode == OpMode::OPBASE) {
        return 1;
    } else {
        return 0;
    }
}

// 按照mesh的方式计算SliceInfo，例如N张卡，就是N份slice
HcclResult InsTempBroadcastMesh1DTwoShot::CalcDataSliceInfo(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    // 一般情况下，mesh的temp是单级的
    u64 unitAlignSize;
    AllignInfo allignInfo = {false, 0, dataType_};
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));
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
        HCCL_ERROR("[InsTempBroadcastMesh1DTwoShot] Rank [%d], SliceInfo calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

// 计算scatter的通信rank集合
HcclResult
InsTempBroadcastMesh1DTwoShot::CalcCommRankSetforScatter(const u32 groupRankSize, std::vector<u32>& commRanks) const
{
    (void)groupRankSize;
    commRanks.clear();

    if (u32(myRank_) != root_) {
        commRanks.emplace_back(root_);
        return HcclResult::HCCL_SUCCESS;
    }

    for (auto& rankIter : tempVirtRankMap_) {
        if (u32(myRank_) != u32(rankIter.first)) {
            commRanks.emplace_back(u32(rankIter.first));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

// 计算allgather的通信rank集合
HcclResult
InsTempBroadcastMesh1DTwoShot::CalcCommRankSetforAllGather(const u32 groupRankSize, std::vector<u32>& commRanks) const
{
    (void)groupRankSize;
    commRanks.clear();

    for (auto& rankIter : tempVirtRankMap_) {
        if (u32(myRank_) != u32(rankIter.first) && root_ != u32(rankIter.first)) {
            commRanks.emplace_back(u32(rankIter.first));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RootSendData(
    const u64 memOffset, const s32 remoteRank, const TemplateDataParams& tempAlgParams, const InsQuePtr& queue,
    const LinkData& link, const RankSliceInfo& sliceInfoVec) const
{
    u32 myRankIdx = tempVirtRankMap_.at(myRank_);
    u32 remoteRankIdx = tempVirtRankMap_.at(remoteRank);

    // root执行常规scatter发送，将remoteRank的数据分片发送至remoteRank的buf中
    u64 sendSrcOffset0 = sliceInfoVec[remoteRankIdx][0].offset + memOffset;
    u64 sendDstOffset0 = sliceInfoVec[remoteRankIdx][0].offset;
    if (dstBufferType_ == BufferType::SCRATCH) {
        sendDstOffset0 += tempAlgParams.buffInfo.scratchBuffBaseOff;
    } else {
        sendDstOffset0 += tempAlgParams.buffInfo.outBuffBaseOff;
    }

    DataSlice sendSrcSlice0 = DataSlice(BufferType::INPUT, sendSrcOffset0, sliceInfoVec[remoteRankIdx][0].size);
    DataSlice sendDstSlice0 = DataSlice(dstBufferType_, sendDstOffset0, sliceInfoVec[remoteRankIdx][0].size);

    std::vector<DataSlice> sendSrcSliceVec0 = {sendSrcSlice0};
    std::vector<DataSlice> sendDstSliceVec0 = {sendDstSlice0};
    SlicesList sendDataSlice0(sendSrcSliceVec0, sendDstSliceVec0);
    DataInfo sendDataInfo0(link, sendDataSlice0);
    CHK_RET(Send(sendDataInfo0, queue, 0, true, DmaMode::PUT));

    // root将自己数据分片发送至对端
    u64 sendSrcOffset1 = sliceInfoVec[myRankIdx][0].offset + memOffset;
    u64 sendDstOffset1 = sliceInfoVec[myRankIdx][0].offset;
    if (dstBufferType_ == BufferType::SCRATCH) {
        sendDstOffset1 += tempAlgParams.buffInfo.scratchBuffBaseOff;
    } else {
        sendDstOffset1 += tempAlgParams.buffInfo.outBuffBaseOff;
    }

    DataSlice sendSrcSlice1 = DataSlice(BufferType::INPUT, sendSrcOffset1, sliceInfoVec[myRankIdx][0].size);
    DataSlice sendDstSlice1 = DataSlice(dstBufferType_, sendDstOffset1, sliceInfoVec[myRankIdx][0].size);

    std::vector<DataSlice> sendSrcSliceVec1 = {sendSrcSlice1};
    std::vector<DataSlice> sendDstSliceVec1 = {sendDstSlice1};
    SlicesList sendDataSlice1(sendSrcSliceVec1, sendDstSliceVec1);
    DataInfo sendDataInfo1(link, sendDataSlice1);
    CHK_RET(Send(sendDataInfo1, queue, 0, true, DmaMode::PUT));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RankRecvData(
    const u64 memOffset, const TemplateDataParams& tempAlgParams, const InsQuePtr& queue, const LinkData& link,
    const RankSliceInfo& sliceInfoVec) const
{
    u32 myRankIdx = tempVirtRankMap_.at(myRank_);
    u32 rootIdx = tempVirtRankMap_.at(root_);

    // 非root执行常规scatter接收，从root接收本rank的数据分片
    u64 sendSrcOffset0 = sliceInfoVec[myRankIdx][0].offset + memOffset;
    u64 sendDstOffset0 = sliceInfoVec[myRankIdx][0].offset;
    if (dstBufferType_ == BufferType::SCRATCH) {
        sendDstOffset0 += tempAlgParams.buffInfo.scratchBuffBaseOff;
    } else {
        sendDstOffset0 += tempAlgParams.buffInfo.outBuffBaseOff;
    }

    DataSlice recvSrcSlice0 = DataSlice(BufferType::INPUT, sendSrcOffset0, sliceInfoVec[myRankIdx][0].size);
    DataSlice recvDstSlice0 = DataSlice(dstBufferType_, sendDstOffset0, sliceInfoVec[myRankIdx][0].size);

    std::vector<DataSlice> recvSrcSliceVec0 = {recvSrcSlice0};
    std::vector<DataSlice> recvDstSliceVec0 = {recvDstSlice0};
    SlicesList recvDataSlice0(recvSrcSliceVec0, recvDstSliceVec0);
    DataInfo recvDataInfo0(link, recvDataSlice0);
    CHK_RET(Recv(recvDataInfo0, queue, 0, true, DmaMode::PUT));

    // 非root接收root的数据分片
    u64 sendSrcOffset1 = sliceInfoVec[rootIdx][0].offset + memOffset;
    u64 sendDstOffset1 = sliceInfoVec[rootIdx][0].offset;
    if (dstBufferType_ == BufferType::SCRATCH) {
        sendDstOffset1 += tempAlgParams.buffInfo.scratchBuffBaseOff;
    } else {
        sendDstOffset1 += tempAlgParams.buffInfo.outBuffBaseOff;
    }

    DataSlice recvSrcSlice1 = DataSlice(BufferType::INPUT, sendSrcOffset1, sliceInfoVec[rootIdx][0].size);
    DataSlice recvDstSlice1 = DataSlice(dstBufferType_, sendDstOffset1, sliceInfoVec[rootIdx][0].size);

    std::vector<DataSlice> recvSrcSliceVec1 = {recvSrcSlice1};
    std::vector<DataSlice> recvDstSliceVec1 = {recvDstSlice1};
    SlicesList recvDataSlice1(recvSrcSliceVec1, recvDstSliceVec1);
    DataInfo recvDataInfo1(link, recvDataSlice1);
    CHK_RET(Recv(recvDataInfo1, queue, 0, true, DmaMode::PUT));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RunScatter(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& queues, const RankSliceInfo& sliceInfoVec) const
{
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: Scatter entry.");

    // 主从流同步
    if (commRanks.size() > 1) {
        CHK_RET(PreSyncInterQueues(queues));
    }

    u64 memOffset = tempAlgParams.buffInfo.inBuffBaseOff;

    // DMA消减，直接从root的inputbuf传输数据至对端buf
    for (u32 i = 0; i < commRanks.size(); i++) {
        s32 remoteRank = static_cast<s32>(commRanks[i]);
        InsQuePtr queue = queues[i];
        LinkData link = tempLinks.at(remoteRank)[0];
        if (u32(myRank_) == root_) {
            // root只发不收
            CHK_RET(RootSendData(memOffset, remoteRank, tempAlgParams, queue, link, sliceInfoVec));
        } else {
            // 非root只收不发
            CHK_RET(RankRecvData(memOffset, tempAlgParams, queue, link, sliceInfoVec));
        }
    }

    // 主从流同步
    if (commRanks.size() > 1) {
        CHK_RET(PostSyncInterQueues(queues));
    }

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: Scatter finish.");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RunAllGather(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& queues, const RankSliceInfo& sliceInfoVec) const
{
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: AllGather entry.");

    if (commRanks.size() > 1) {
        CHK_RET(PreSyncInterQueues(queues));
    }

    for (u32 i = 0; i < commRanks.size(); i++) {
        s32 remoteRank = static_cast<s32>(commRanks[i]);
        InsQuePtr queue = queues[i];
        LinkData link = tempLinks.at(remoteRank)[0];

        u32 myRankIdx = tempVirtRankMap_.at(myRank_);
        u32 remoteRankIdx = tempVirtRankMap_.at(remoteRank);

        u64 sendSrcOffset = sliceInfoVec[myRankIdx][0].offset;
        u64 sendDstOffset = sliceInfoVec[myRankIdx][0].offset;
        u64 recvSrcOffset = sliceInfoVec[remoteRankIdx][0].offset;
        u64 recvDstOffset = sliceInfoVec[remoteRankIdx][0].offset;

        if (srcBufferType_ == BufferType::SCRATCH) {
            sendSrcOffset += tempAlgParams.buffInfo.scratchBuffBaseOff;
            recvSrcOffset += tempAlgParams.buffInfo.scratchBuffBaseOff;
        } else {
            sendSrcOffset += tempAlgParams.buffInfo.inBuffBaseOff;
            recvSrcOffset += tempAlgParams.buffInfo.inBuffBaseOff;
        }

        if (dstBufferType_ == BufferType::SCRATCH) {
            sendDstOffset += tempAlgParams.buffInfo.scratchBuffBaseOff;
            recvDstOffset += tempAlgParams.buffInfo.scratchBuffBaseOff;
        } else {
            sendDstOffset += tempAlgParams.buffInfo.outBuffBaseOff;
            recvDstOffset += tempAlgParams.buffInfo.outBuffBaseOff;
        }

        DataSlice sendSrcSlice = DataSlice(srcBufferType_, sendSrcOffset, sliceInfoVec[myRankIdx][0].size);
        DataSlice sendDstSlice = DataSlice(dstBufferType_, sendDstOffset, sliceInfoVec[myRankIdx][0].size);
        std::vector<DataSlice> sendSrcSliceVec = {sendSrcSlice};
        std::vector<DataSlice> sendDstSliceVec = {sendDstSlice};
        SlicesList sendDataSlice(sendSrcSliceVec, sendDstSliceVec);

        DataSlice recvSrcSlice = DataSlice(srcBufferType_, recvSrcOffset, sliceInfoVec[remoteRankIdx][0].size);
        DataSlice recvDstSlice = DataSlice(dstBufferType_, recvDstOffset, sliceInfoVec[remoteRankIdx][0].size);
        std::vector<DataSlice> recvSrcSliceVec = {recvSrcSlice};
        std::vector<DataSlice> recvDstSliceVec = {recvDstSlice};
        SlicesList recvDataSlice(recvSrcSliceVec, recvDstSliceVec);

        TxRxSlicesList sendRecvSlice(sendDataSlice, recvDataSlice);
        TxRxLinks sendRecvLinks(link, link);

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlice);
        CHK_RET(SendRecv(sendRecvInfo, queue, 0, true, DmaMode::PUT));
    }

    if (commRanks.size() > 1) {
        CHK_RET(PostSyncInterQueues(queues));
    }

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: AllGather finish.");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::PostCopy(
    const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const
{
    u64 inOffset = tempAlgParams.buffInfo.scratchBuffBaseOff;

    DataSlice usrInSlice = DataSlice(BufferType::SCRATCH, inOffset, tempAlgParams.sliceSize);
    DataSlice usrOutSlice
        = DataSlice(BufferType::INPUT, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize);

    HCCL_INFO(
        "PostCopy usrInSlice: %s, usrOutSlice: %s", usrInSlice.Describe().c_str(), usrOutSlice.Describe().c_str());

    CHK_RET(LocalCopy(tempInsQues[0], usrInSlice, usrOutSlice));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot entry.");

    if (opMode_ == OpMode::OPBASE) {
        srcBufferType_ = BufferType::SCRATCH;
        dstBufferType_ = BufferType::SCRATCH;
    }

    RankSliceInfo sliceInfoVec{};
    CHK_RET(CalcDataSliceInfo(templateDataParams.sliceSize, sliceInfoVec));

    queNum_ = tempVTopo_[0].size() - 1;
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempBroadcastMesh1DTwoShot] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO(
        "[InsTempBroadcastMesh1DTwoShot Run]RankID:[%d], root:[%u], isForepart:[%d], isBottom:[%d]", myRank_, root_,
        tempFuncs.isForepart, tempFuncs.isBottom);

    std::vector<u32> scatterCommRanks;
    CHK_RET(CalcCommRankSetforScatter(tempRankSize_, scatterCommRanks)); // 计算scatter步骤的通信对象
    CHK_RET(RunScatter(scatterCommRanks, templateDataParams, tempLinks, tempInsQues, sliceInfoVec)); // 运行scatter步骤

    if (u32(myRank_) != root_) {
        std::vector<u32> allgatherCommRanks;
        CHK_RET(CalcCommRankSetforAllGather(tempRankSize_, allgatherCommRanks)); // 计算allgather步骤的通信对象
        CHK_RET(RunAllGather(
            allgatherCommRanks, templateDataParams, tempLinks, tempInsQues, sliceInfoVec)); // 运行allgather步骤
    }

    // 单算子模式
    if (opMode_ == OpMode::OPBASE && (u32(myRank_) != root_)) {
        CHK_RET(PostCopy(templateDataParams, tempInsQues));
    }

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot finish.");

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
