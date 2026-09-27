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
#include "ins_temp_reduce_scatter_aicpu_reduce.h"

namespace Hccl {
InsTempReduceScatterAicpuReduce::InsTempReduceScatterAicpuReduce(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceScatterAicpuReduce::~InsTempReduceScatterAicpuReduce() {}

HcclResult InsTempReduceScatterAicpuReduce::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempVTopo_[0].size();
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_DEBUG(
        "[InsTempReduceScatterAicpuReduce]CalcRes: queNum[%u], myRank[%d], tempRankSize[%u]", tempResReq.queNum,
        myRank_, tempRankSize_);
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceScatterAicpuReduce::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult InsTempReduceScatterAicpuReduce::RunAlltoAllMesh(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    CHK_RET(PreSyncInterQueues(tempInsQues));
    // 本端rank数据从本端input -> 本端scratch
    u64 srcOffset = templateDataParams.inputSliceStride * u32(myRank_) + templateDataParams.buffInfo.inBuffBaseOff;
    u64 srcSize = templateDataParams.sliceSize;
    u64 dstOffset = templateDataParams.sliceSize * u32(myRank_);
    DataSlice srcSlice = DataSlice(BufferType::INPUT, srcOffset, srcSize);
    DataSlice dstSlice = DataSlice(BufferType::SCRATCH, dstOffset, srcSize);
    std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(srcSlice, dstSlice);
    tempInsQues[0]->Append(std::move(insLocalCopy));

    // 本端rank数据从本端input -> 对端scratch

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    for (u32 queIdx = 1; queIdx < tempVTopo_[0].size(); queIdx++) {
        RankId neighborRank = tempVTopo_[0][(myAlgRank + queIdx) % tempRankSize_];
        LinkData neighborLinkData = tempLinks.at(neighborRank)[0];
        TxRxLinks sendRecvLinks(neighborLinkData, neighborLinkData);
        // send
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        DataSlice currSendSliceSrc = DataSlice(
            BufferType::INPUT,
            templateDataParams.inputSliceStride * u32(neighborRank) + templateDataParams.buffInfo.inBuffBaseOff,
            templateDataParams.sliceSize);
        DataSlice currSendSliceDst
            = DataSlice(BufferType::SCRATCH, templateDataParams.sliceSize * u32(myRank_), templateDataParams.sliceSize);
        txSrcSlices.push_back(currSendSliceSrc);
        txDstSlices.push_back(currSendSliceDst);
        // recv
        DataSlice currRecvSliceSrc = DataSlice(
            BufferType::SCRATCH, templateDataParams.inputSliceStride * u32(neighborRank), templateDataParams.sliceSize);
        DataSlice currRecvSliceDst = DataSlice(
            BufferType::SCRATCH,
            templateDataParams.inputSliceStride * u32(neighborRank) + templateDataParams.buffInfo.inBuffBaseOff,
            templateDataParams.sliceSize);
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        rxSrcSlices.push_back(currRecvSliceSrc);
        rxDstSlices.push_back(currRecvSliceDst);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[queIdx], 0, true, dmaMode_),
            HCCL_ERROR("[InsTempReduceScatterAicpuReduce] RunReduceScatter sendrecv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduce::RunAicpuLocalReduce(
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

HcclResult InsTempReduceScatterAicpuReduce::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempReduceScatterAicpuReduce] Run start");
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    } else {
        dmaMode_ = DmaMode::PUT;
    }
    if (tempVTopo_[0].size() == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size();
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterAicpuReduce] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    RunAlltoAllMesh(tempFuncs, templateDataParams, tempLinks, tempInsQues);
    StreamSync(tempInsQues);
    RunAicpuLocalReduce(templateDataParams, tempInsQues);
    HCCL_INFO("[InsTempReduceScatterAicpuReduce] Run finished");
    return HCCL_SUCCESS;
}

} // namespace Hccl
