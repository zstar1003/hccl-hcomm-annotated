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
#include "ins_temp_reduce_aicpu_reduce.h"

namespace Hccl {
InsTempReduceAicpuReduce::InsTempReduceAicpuReduce(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceAicpuReduce::~InsTempReduceAicpuReduce() {}

HcclResult InsTempReduceAicpuReduce::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempVTopo_[0].size();
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_DEBUG(
        "[InsTempReduceAicpuReduce]CalcRes: queNum[%u], myRank[%d], tempRankSize[%u]", tempResReq.queNum, myRank_,
        tempRankSize_);
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceAicpuReduce::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult InsTempReduceAicpuReduce::RunGatherMesh(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    if (u32(myRank_) == root_) {
        // 本端rank数据从本端input -> 本端scratch
        CHK_RET(PreSyncInterQueues(tempInsQues));
        u64 srcOffset = templateDataParams.buffInfo.inBuffBaseOff;
        u64 srcSize = templateDataParams.sliceSize;
        u64 dstOffset = templateDataParams.sliceSize * root_;
        DataSlice srcSlice = DataSlice(BufferType::INPUT, srcOffset, srcSize);
        DataSlice dstSlice = DataSlice(BufferType::SCRATCH, dstOffset, srcSize);
        std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(srcSlice, dstSlice);
        tempInsQues[0]->Append(std::move(insLocalCopy));
        // recv from other rank
        u32 myAlgRank;
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
        for (u32 queIdx = 1; queIdx < tempVTopo_[0].size(); queIdx++) {
            RankId neighborRank = tempVTopo_[0][(myAlgRank + queIdx) % tempRankSize_];
            LinkData neighborLinkData = tempLinks.at(neighborRank)[0];
            std::vector<DataSlice> srcSlices;
            std::vector<DataSlice> dstSlices;
            srcSlices.emplace_back(BufferType::INPUT, srcOffset, srcSize);
            dstSlices.emplace_back(BufferType::SCRATCH, srcSize * neighborRank, srcSize);
            SlicesList rxSlicesList(srcSlices, dstSlices);
            DataInfo recvData(neighborLinkData, rxSlicesList);
            CHK_PRT_RET(
                Recv(recvData, tempInsQues[queIdx], 0, true, dmaMode_),
                HCCL_ERROR("[InsTempReduceAicpuReduce] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
        }
        CHK_RET(PostSyncInterQueues(tempInsQues));
    } else {
        // send to root rank
        u64 srcOffset = templateDataParams.buffInfo.inBuffBaseOff;
        u64 srcSize = templateDataParams.sliceSize;
        LinkData linkSend = tempLinks.at(root_)[0];
        std::vector<DataSlice> srcSlices;
        std::vector<DataSlice> dstSlices;
        srcSlices.emplace_back(BufferType::INPUT, srcOffset, srcSize);
        dstSlices.emplace_back(BufferType::SCRATCH, srcSize * u32(myRank_), srcSize);
        SlicesList txSlicesList(srcSlices, dstSlices);
        DataInfo sendData(linkSend, txSlicesList);
        CHK_PRT_RET(
            Send(sendData, tempInsQues[0], 0, true, dmaMode_),
            HCCL_ERROR("[InsTempReduceAicpuReduce] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduce::RunAicpuLocalReduce(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    DataSlice dataSlice = DataSlice(BufferType::SCRATCH, 0, templateDataParams.sliceSize);
    for (u32 rankId = 1; rankId < tempRankSize_; rankId++) {
        DataSlice reduceSlice
            = DataSlice(BufferType::SCRATCH, templateDataParams.sliceSize * rankId, templateDataParams.sliceSize);
        AicpuReduce(tempInsQues[0], reduceSlice, dataSlice, dataType_, redOp_);
    }
    DataSlice outputSlice
        = DataSlice(BufferType::OUTPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize);
    LocalCopy(tempInsQues[0], dataSlice, outputSlice);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduce::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempReduceAicpuReduce] Run start");

    dmaMode_ = DmaMode::PUT;
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    }
    if (tempVTopo_[0].size() == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size();
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceAicpuReduce] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    RunGatherMesh(tempFuncs, templateDataParams, tempLinks, tempInsQues);
    StreamSync(tempInsQues);
    RunAicpuLocalReduce(templateDataParams, tempInsQues);
    HCCL_INFO("[InsTempReduceAicpuReduce] Run finished");
    return HCCL_SUCCESS;
}

} // namespace Hccl
