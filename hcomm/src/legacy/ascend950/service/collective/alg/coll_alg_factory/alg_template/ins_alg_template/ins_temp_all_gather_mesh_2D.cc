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
#include "executor_utils.h"
#include "ins_temp_all_gather_mesh_2D.h"

namespace Hccl {
InsTempAllGatherMesh2D::InsTempAllGatherMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAllGatherMesh2D::~InsTempAllGatherMesh2D() {}

HcclResult InsTempAllGatherMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_DEBUG("Enter InsTempAllGatherMesh2D::CalcRes");
    const int TwoD = 2;
    CHK_PRT_RET(
        tempVTopo_.size() < TwoD,
        HCCL_ERROR("[InsTempAllGatherMesh2D] tempVTopo_ mismatch size:%zu", tempVTopo_.size()),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempVTopo_[0].size() <= 1 || tempVTopo_[1].size() <= 1,
        HCCL_ERROR(
            "[InsTempAllGatherMesh2D] tempVTopo_ size error, size:%zu %zu", tempVTopo_[0].size(), tempVTopo_[1].size()),
        HcclResult::HCCL_E_INTERNAL);
    tempResReq.queNum = tempVTopo_[0].size() - 1 + tempVTopo_[1].size() - 1;

    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    HCCL_DEBUG("InsTempAllGatherMesh2D::CalcRes queNotifys size[%zu]", tempResReq.queNotifys.size());

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "InsTempAllGatherMesh2D::CalcRes Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim, neighborRank);
            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }
    HCCL_INFO("InsTempAllGatherMesh2D::CalcRes done");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh2D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempGatherMesh2D] Run start");

    opMode_ = tempFuncs.opMode;
    tempAlgParams_ = tempAlgParams;
    tempLinks_ = tempLinks;
    tempFuncs_ = tempFuncs;
    const int TwoD = 2;
    CHK_PRT_RET(
        tempVTopo_.size() < TwoD,
        HCCL_ERROR("[InsTempAllGatherMesh2D] tempVTopo_ mismatch size:%zu", tempVTopo_.size()),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempVTopo_[0].size() <= 1 || tempVTopo_[1].size() <= 1,
        HCCL_ERROR(
            "[InsTempAllGatherMesh2D] tempVTopo_ size error, size:%zu %zu", tempVTopo_[0].size(), tempVTopo_[1].size()),
        HcclResult::HCCL_E_INTERNAL);
    majorQueNum_ = tempVTopo_[0].size() - 1 + tempVTopo_[1].size() - 1;
    xQueNum_ = tempVTopo_[0].size() - 1;
    yQueNum_ = tempVTopo_[1].size() - 1;

    // queue arrangement
    std::vector<InsQuePtr> mainInsQues;
    std::vector<InsQuePtr> xInsQues;
    std::vector<InsQuePtr> yInsQues;
    for (u32 queIdx = 0; queIdx < majorQueNum_; queIdx++) {
        mainInsQues.push_back(tempInsQues[queIdx]);
        if (queIdx < xQueNum_) {
            xInsQues.push_back(tempInsQues[queIdx]);
        } else {
            yInsQues.push_back(tempInsQues[queIdx]);
        }
    }

    // Local Copy from Input to Output
    CHK_RET(LocalDataCopy(mainInsQues));
    if (tempRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    // semaphore sync
    CHK_RET(PreSyncInterQueues(mainInsQues));

    // // step1
    CHK_RET(Run2DStep1(xInsQues, yInsQues));

    // semaphore sync
    CHK_RET(PostSyncInterQueues(mainInsQues));
    CHK_RET(PreSyncInterQueues(mainInsQues));

    // step2 run Mesh
    CHK_RET(Run2DStep2(xInsQues, yInsQues));
    CHK_RET(PostSyncInterQueues(mainInsQues));
    // LocalCopy: from scratch to output for opbase
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isBottom) {
        CHK_RET(PostLocalCopy(mainInsQues));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh2D::Run2DStep1(std::vector<InsQuePtr>& xInsQues, std::vector<InsQuePtr>& yInsQues)
{
    u32 myAlgRankX;
    u32 myAlgRankY;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRankX));
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[1], myAlgRankY));

    CHK_PRT_RET(
        RunMesh(myAlgRankX, myRank_, 0, tempVTopo_[0], xInsQues, 0, tempAlgParams_.sliceSize, DmaMode::PUT)
            != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsCollAlgFactory] [InsTempAllGatherMesh2D] Rank [%d], unable to run the mesh x0 algorithm.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        RunMesh(myAlgRankY, myRank_, 0, tempVTopo_[1], yInsQues, 0, tempAlgParams_.sliceSize, DmaMode::PUT)
            != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsCollAlgFactory] [InsTempAllGatherMesh2D] Rank [%d], unable to run the mesh y0 algorithm.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh2D::Run2DStep2(std::vector<InsQuePtr>& xInsQues, std::vector<InsQuePtr>& yInsQues)
{
    u32 myAlgRankX;
    u32 myAlgRankY;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRankX));
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[1], myAlgRankY));
    u64 xSize = tempAlgParams_.sliceSize / 2;
    u64 ySize = tempAlgParams_.sliceSize - xSize;
    for (u32 rank = 0; rank < tempVTopo_[0].size(); rank++) { // 转发哪一个rank
        if (rank == myAlgRankX) {                             // 只处理转发，直连的在step1传输完成
            continue;
        }
        RankId globalRank = tempVTopo_[0][rank];
        int rankOffset = rank - myAlgRankX;
        // 上半部分数据通过Y轴传输
        CHK_PRT_RET(
            RunMesh(myAlgRankY, globalRank, rankOffset, tempVTopo_[1], yInsQues, 0, xSize, DmaMode::GET)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[InsCollAlgFactory] [InsTempAllGatherMesh2D] Rank [%d], unable to run the mesh y1 algorithm.",
                myRank_),
            HcclResult::HCCL_E_INTERNAL);
    }
    for (u32 rank = 0; rank < tempVTopo_[1].size(); rank++) {
        if (rank == myAlgRankY) {
            continue;
        }
        RankId globalRank = tempVTopo_[1][rank];
        int rankOffset = (rank - myAlgRankY) * tempVTopo_[0].size();
        // 下半部分数据通过X轴传输
        CHK_PRT_RET(
            RunMesh(myAlgRankX, globalRank, rankOffset, tempVTopo_[0], xInsQues, xSize, ySize, DmaMode::GET)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[InsCollAlgFactory] [InsTempAllGatherMesh2D] Rank [%d], unable to run the mesh x1 algorithm.",
                myRank_),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh2D::RunMesh(
    const u32 myAlgRank, RankId globalSrcRank, int rankOffset, const std::vector<RankId>& vTopo,
    std::vector<InsQuePtr>& tempInsQues, u64 xyOffset, u64 size, DmaMode dmaMode)
{
    if (size == 0) {
        HCCL_INFO("[InsTempAllGatherMesh2D] 0 data skip sendrecv");
        return HcclResult::HCCL_SUCCESS;
    }
    const u64 scratchRepeatStride = tempAlgParams_.sliceSize * tempRankSize_;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        for (u32 queIdx = 0; queIdx < vTopo.size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            RankId connectedRank = vTopo[(myAlgRank + 1 + queIdx) % vTopo.size()];
            RankId globalDstRank = connectedRank + rankOffset;
            HCCL_INFO(
                "[InsTempAllGatherMesh2D] RunAllGather opbase find neighbors: ==============="
                "myRank=%d, connectedRank=%d, globalSrcRank=%d, globalDstRank=%d, myAlgRank=%u, queIdx=%u,",
                myRank_, connectedRank, globalSrcRank, globalDstRank, myAlgRank, queIdx);

            RankId srcAlgRank = globalSrcRank % tempRankSize_;
            RankId dstAlgRank = globalDstRank % tempRankSize_;

            CHK_PRT_RET(
                queIdx >= tempInsQues.size() or tempLinks_.at(connectedRank).size() <= 0,
                HCCL_ERROR(
                    "InsTempAllGatherMesh2D: tempInsQues.size()=%zu, connectedRank=%d, tempLinks_.size()=%zu, ",
                    tempInsQues.size(), connectedRank, tempLinks_.size()),
                HcclResult::HCCL_E_INTERNAL);
            InsQuePtr currQue = tempInsQues[queIdx];
            LinkData& neighborLinkData = tempLinks_.at(connectedRank)[0];

            BufferType type
                = (opMode_ == OpMode::OPBASE || !tempFuncs_.isBottom) ? BufferType::SCRATCH : BufferType::OUTPUT;
            const u64 txOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride
                                    + tempAlgParams_.outputSliceStride * srcAlgRank + xyOffset;
            const u64 txScratchOffset = tempAlgParams_.buffInfo.scratchBuffBaseOff + rpt * scratchRepeatStride
                                        + +tempAlgParams_.sliceSize * srcAlgRank + xyOffset;
            const u64 txDstOffset = (opMode_ == OpMode::OPBASE || !tempFuncs_.isBottom) ? txScratchOffset : txOutOffset;
            HCCL_DEBUG(
                "[InsTempAllGatherMesh2D] RunAllGather opbase sendrecv: "
                "txOutOffset=%llu, txScratchOffset=%llu, txDstOffset=%llu "
                "(globalSrcRank=%d, globalDstRank=%d, opMode=%d)",
                txOutOffset, txScratchOffset, txDstOffset, globalSrcRank, globalDstRank, opMode_);
            const u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride
                                    + tempAlgParams_.outputSliceStride * dstAlgRank + xyOffset;
            const u64 rxScratchOffset = tempAlgParams_.buffInfo.scratchBuffBaseOff + rpt * scratchRepeatStride
                                        + tempAlgParams_.sliceSize * dstAlgRank + xyOffset;
            const u64 rxSrcOffset = (opMode_ == OpMode::OPBASE || !tempFuncs_.isBottom) ? rxScratchOffset : rxOutOffset;
            HCCL_DEBUG(
                "[InsTempAllGatherMesh2D] RunAllGather opbase sendrecv: "
                "rxOutOffset=%llu, rxScratchOffset=%llu, rxSrcOffset=%llu "
                "(globalSrcRank=%d, globalDstRank=%d, opMode=%d)",
                rxOutOffset, rxScratchOffset, rxSrcOffset, globalSrcRank, globalDstRank, opMode_);

            BufferType txrxBufType = !tempFuncs_.isBottom ? BufferType::SCRATCH : BufferType::OUTPUT;
            vector<DataSlice> txSrcSlice = vector<DataSlice>{DataSlice(txrxBufType, txOutOffset, size)}; // 发送源
            vector<DataSlice> txDstSlice = vector<DataSlice>{DataSlice(type, txDstOffset, size)};        // 发送目标
            HCCL_INFO(
                "[InsTempAllGatherMesh2D] RunAllGather opbase *****sendrecv*****, txSrcSlice: %s, txDstSlice: %s",
                txSrcSlice[0].Describe().c_str(), txDstSlice[0].Describe().c_str());

            vector<DataSlice> rxSrcSlice = vector<DataSlice>{DataSlice(type, rxSrcOffset, size)};        // 接收源
            vector<DataSlice> rxDstSlice = vector<DataSlice>{DataSlice(txrxBufType, rxOutOffset, size)}; // 接收目标
            HCCL_INFO(
                "[InsTempAllGatherMesh2D] RunAllGather opbase *****sendrecv*****, rxSrcSlice: %s, rxDstSlice: %s",
                rxSrcSlice[0].Describe().c_str(), rxDstSlice[0].Describe().c_str());

            TxRxSlicesList sendRecvSlicesList({txSrcSlice, txDstSlice}, {rxSrcSlice, rxDstSlice});
            TxRxLinks sendRecvLinks(neighborLinkData, neighborLinkData);
            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, currQue, 0, true, dmaMode),
                HCCL_ERROR("[InsTempAllGatherMesh2D] RunAllGather opbase sendrecv failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh2D::LocalDataCopy(std::vector<InsQuePtr>& tempInsQues)
{
    if (tempAlgParams_.buffInfo.inBuffType == tempAlgParams_.buffInfo.outBuffType) {
        return HcclResult::HCCL_SUCCESS;
    }
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        RankId algRank = myRank_ % tempRankSize_;
        const u64 inOffset = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, inOffset, tempAlgParams_.sliceSize);
        const u64 outOffset = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride
                              + tempAlgParams_.outputSliceStride * algRank;
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, outOffset, tempAlgParams_.sliceSize);
        HCCL_INFO(
            "[InsTempAllGatherMesh2D] PreCopy usrInSlice: %s, usrOutSlice: %s", usrInSlice.Describe().c_str(),
            usrOutSlice.Describe().c_str());
        std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
        tempInsQues[0]->Append(std::move(insLocalCopy));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh2D::PostLocalCopy(std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[InsTempAllGatherMesh2D][PostLocalCopy] empty tempInsQues"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    const u64 scratchRepeatStride = tempAlgParams_.sliceSize * tempRankSize_;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        for (u32 i = 0; i < tempVTopo_.size(); i++) {
            for (auto rank : tempVTopo_[i]) {
                if (rank == myRank_) {
                    continue;
                }
                // 只拷贝step1的对端
                RankId algRank = (rank % tempRankSize_);
                u64 scratchOffset = tempAlgParams_.buffInfo.scratchBuffBaseOff + rpt * scratchRepeatStride
                                    + tempAlgParams_.sliceSize * algRank;
                u64 outOffset = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride
                                + tempAlgParams_.outputSliceStride * algRank;
                DataSlice usrInSlice = DataSlice(BufferType::SCRATCH, scratchOffset, tempAlgParams_.sliceSize);
                DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, outOffset, tempAlgParams_.sliceSize);
                HCCL_INFO(
                    "[InsTempAllGatherMesh2D] rank[%d] algRank[%d] PostCopy usrInSlice: %s, usrOutSlice: %s", myRank_,
                    algRank, usrInSlice.Describe().c_str(), usrOutSlice.Describe().c_str());
                std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
                tempInsQues[0]->Append(std::move(insLocalCopy));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
