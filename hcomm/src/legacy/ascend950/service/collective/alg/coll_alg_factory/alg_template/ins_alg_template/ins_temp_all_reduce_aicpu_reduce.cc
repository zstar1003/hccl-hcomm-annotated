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
#include "ins_temp_all_reduce_aicpu_reduce.h"

namespace Hccl {
InsTempAllReduceAicpuReduce::InsTempAllReduceAicpuReduce(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAllReduceAicpuReduce::~InsTempAllReduceAicpuReduce() {}

HcclResult InsTempAllReduceAicpuReduce::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempVTopo_[0].size() - 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_DEBUG(
        "[InsTempAllReduceAicpuReduce]CalcRes: queNum[%u], myRank[%d], tempRankSize[%u]", tempResReq.queNum, myRank_,
        tempRankSize_);
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllReduceAicpuReduce::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult InsTempAllReduceAicpuReduce::RunAllGatherMesh(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    // 本端rank数据从本端input -> 本端scratch
    u64 srcOffset = templateDataParams.buffInfo.inBuffBaseOff;
    u64 srcSize = templateDataParams.sliceSize;
    u64 dstOffset = templateDataParams.sliceSize * myRank_;
    DataSlice srcSlice = DataSlice(BufferType::INPUT, srcOffset, srcSize);
    DataSlice dstSlice = DataSlice(BufferType::SCRATCH, dstOffset, srcSize);
    std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(srcSlice, dstSlice);
    tempInsQues[0]->Append(std::move(insLocalCopy));

    CHK_RET(PreSyncInterQueues(tempInsQues));
    // 本端rank数据从本端input -> 对端scratch
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    DataSlice currSendSliceSrc
        = DataSlice(BufferType::INPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize);
    DataSlice currSendSliceDst
        = DataSlice(BufferType::SCRATCH, templateDataParams.sliceSize * myRank_, templateDataParams.sliceSize);
    txSrcSlices.push_back(currSendSliceSrc);
    txDstSlices.push_back(currSendSliceDst);
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    for (u32 queIdx = 0; queIdx < tempVTopo_[0].size() - 1; queIdx++) {
        RankId neighborRank = tempVTopo_[0][(myAlgRank + 1 + queIdx) % tempRankSize_];
        LinkData neighborLinkData = tempLinks.at(neighborRank)[0];
        TxRxLinks sendRecvLinks(neighborLinkData, neighborLinkData);
        DataSlice currRecvSliceSrc
            = DataSlice(BufferType::SCRATCH, templateDataParams.sliceSize * neighborRank, templateDataParams.sliceSize);
        DataSlice currRecvSliceDst
            = DataSlice(BufferType::SCRATCH, templateDataParams.sliceSize * neighborRank, templateDataParams.sliceSize);
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        rxSrcSlices.push_back(currRecvSliceSrc);
        rxDstSlices.push_back(currRecvSliceDst);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[queIdx], 0, true, dmaMode_),
            HCCL_ERROR("[InsTempAllReduceAicpuReduce] RunAllGather sendrecv failed"), HcclResult::HCCL_E_INTERNAL);
    }
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceAicpuReduce::RunAicpuLocalReduce(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    DataSlice dataSlice = DataSlice(BufferType::SCRATCH, 0, templateDataParams.sliceSize);
    for (u32 rankId = 1; rankId < tempRankSize_; rankId++) {
        DataSlice addSlice
            = DataSlice(BufferType::SCRATCH, templateDataParams.sliceSize * rankId, templateDataParams.sliceSize);
        AicpuReduce(tempInsQues[0], addSlice, dataSlice, dataType_, redOp_);
    }
    DataSlice outputSlice
        = DataSlice(BufferType::OUTPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize);
    LocalCopy(tempInsQues[0], dataSlice, outputSlice);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceAicpuReduce::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceAicpuReduce] Run start");
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    } else {
        dmaMode_ = DmaMode::PUT;
    }
    if (tempVTopo_[0].size() == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size() - 1;
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempAllReduceAicpuReduce] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    RunAllGatherMesh(tempFuncs, templateDataParams, tempLinks, tempInsQues);
    StreamSync(tempInsQues);
    RunAicpuLocalReduce(templateDataParams, tempInsQues);
    HCCL_INFO("[InsTempAllReduceAicpuReduce] Run finished");
    return HCCL_SUCCESS;
}

} // namespace Hccl
