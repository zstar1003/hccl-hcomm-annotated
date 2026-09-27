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
#include "ins_alg_template/ins_temp_all_reduce_mesh_1D_one_shot.h"

namespace Hccl {
InsTempAllReduceMesh1DOneShot::InsTempAllReduceMesh1DOneShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAllReduceMesh1DOneShot::~InsTempAllReduceMesh1DOneShot() {}

u32 InsTempAllReduceMesh1DOneShot::CalcScratchMultiple(BufferType input, BufferType output) const
{
    (void)input;
    (void)output;
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    return tempRankSize_;
}

HcclResult InsTempAllReduceMesh1DOneShot::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempVTopo_[0].size();
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    CHK_PRT_RET(
        CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempAllReduceMesh1DOneShot] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DOneShot::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);
    AllignInfo allignInfo = {false, 0, dataType_}; // 参数填充，CalcRsAgSliceInfoMesh实际没用到

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DOneShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][Run] AllReduceMesh1DOneShot begin: rank[%d] start", myRank_);

    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size();
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempAllReduceMesh1DOneShot] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(tempAlgParams.sliceSize, sliceInfoVec));

    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][PreCopy] write userIn data directly to the ScratchBuffer, skip precopy");
    CHK_RET(RunAllReduce(tempAlgParams, sliceInfoVec, tempLinks, tempInsQues));
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][PostCopy] data is already in the userOut, skip postcopy");

    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][Run] AllReduceMesh1DOneShot finished: rank[%d] end", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DOneShot::RunAllReduce(
    const TemplateDataParams& tempAlgParams, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][RunAllReduce] send/recv: rank[%d]", myRank_);

    // semaphore sync
    if (tempVTopo_[0].size() > 1) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }

    DataSlice usrInSlices = DataSlice(BufferType::INPUT, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize);
    DataSlice usrOutSlices
        = DataSlice(BufferType::OUTPUT, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize);

    // 主流动作
    CHK_RET(LocalCopy(tempInsQues[0], usrInSlices, usrOutSlices));

    // 从流动作
    for (u32 queIdx = 1; queIdx < queNum_; queIdx++) {
        u32 nextRank = (myRank_ + queIdx) % tempRankSize_; // 让rank和que对应上
        RankId fromRank = nextRank;
        RankId toRank = nextRank;

        const std::vector<LinkData>& linkRecv = tempLinks.at(fromRank);
        const std::vector<LinkData>& linkSend = tempLinks.at(toRank);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;

        u64 txDstOffset = sliceInfoVec[myRank_][0].offset + tempAlgParams.buffInfo.scratchBuffBaseOff;
        u64 txDstSize = sliceInfoVec[myRank_][0].size;
        DataSlice txSrcSlice = usrInSlices;
        DataSlice txDstSlice = DataSlice(BufferType::SCRATCH, txDstOffset, txDstSize);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);

        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        u64 rxDstOffset = sliceInfoVec[fromRank][0].offset + tempAlgParams.buffInfo.scratchBuffBaseOff;
        u64 rxDstSize = sliceInfoVec[fromRank][0].size;
        DataSlice rxSrcSlice = usrInSlices;
        DataSlice rxDstSlice = DataSlice(BufferType::SCRATCH, rxDstOffset, rxDstSize);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);

        TxRxLinks sendRecvLinks(linkSend[0], linkRecv[0]);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[queIdx], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempAllReduceMesh1DOneShot] RunAllReduce SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
    }

    // semaphore sync
    if (tempVTopo_[0].size() > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }

    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][RunAllReduce] reduce: rank[%d]", myRank_);

    for (u32 rankIdx = 0; rankIdx < tempVTopo_[0].size(); rankIdx++) {
        RankId myRank = myRank_;
        RankId curRank = rankIdx;
        if (curRank == myRank) {
            continue;
        }

        u64 curSrcOffset = sliceInfoVec[curRank][0].offset + tempAlgParams.buffInfo.scratchBuffBaseOff;
        u64 curSrcSize = sliceInfoVec[curRank][0].size;
        DataSlice curSrcSlice = DataSlice(BufferType::SCRATCH, curSrcOffset, curSrcSize);
        DataSlice curDstSlice = usrOutSlices;

        CHK_RET(LocalReduce(tempInsQues[0], curSrcSlice, curDstSlice, dataType_, redOp_));
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
