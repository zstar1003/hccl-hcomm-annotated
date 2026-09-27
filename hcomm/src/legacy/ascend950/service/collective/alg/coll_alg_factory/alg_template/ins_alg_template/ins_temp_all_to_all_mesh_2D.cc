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
#include "ins_temp_all_to_all_mesh_2D.h"

namespace Hccl {
InsTempAlltoAllMesh2D::InsTempAlltoAllMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAlltoAllMesh2D::~InsTempAlltoAllMesh2D() {}

HcclResult InsTempAlltoAllMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    if (tempVTopo_.size() >= TEMPVTOPOSIZE) {
        rankId_ = myRank_;
        rankSize_ = tempRankSize_;
        xRankSize_ = tempVTopo_[0].size();
        yRankSize_ = tempVTopo_[1].size();
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], xRankId_));
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[1], yRankId_));
        CHK_PRT_RET((xRankSize_ == 0), HCCL_ERROR("xRankSize_ equals to zero."), HcclResult::HCCL_E_PARA);
        CHK_PRT_RET((yRankSize_ == 0), HCCL_ERROR("yRankSize_ equals to zero."), HcclResult::HCCL_E_PARA);
    } else {
        HCCL_ERROR("tempVTopo_.size() is [%zu]", tempVTopo_.size());
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_DEBUG(
        "rankId_ is [%u], rankSize_ is [%u], xRankSize_ is [%u], yRankSize_ is [%u]", rankId_, rankSize_, xRankSize_,
        yRankSize_);

    tempResReq.queNum = tempVTopo_[0].size() + tempVTopo_[1].size();
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

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
                "InsTempAlltoAllMesh2D::CalcRes Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim, neighborRank);
            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }
    HCCL_INFO(
        "[InsTempAlltoAllMesh2D] Calculate resource, stream number is[%u], queNotifys size is[%u]",
        tempResReq.streamNum, tempResReq.queNotifys.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2D::RunMeshX(
    std::vector<u64>& xDataInAddr, std::vector<u64>& xDataOutAddr, u64 xSize, BufferType srcBufferType,
    BufferType dstBufferType, DmaMode dmaMode, std::vector<InsQuePtr>& xInsQues, const ResLinks& tempLinks) const
{
    HCCL_DEBUG("RunMeshX begin, xSize is [%u]", xSize);
    if (xSize == 0) {
        HCCL_INFO("[InsTempAlltoAllMesh2D] RunMeshX, xSize is 0");
        return HcclResult::HCCL_SUCCESS;
    }
    std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;
    for (u32 i = 0; i < rankSize_; i++) {
        // 计算send
        u64 xOffset = xRankId_;
        u64 yOffset = i / xRankSize_;
        u64 dstOffset = yOffset * xRankSize_ + xOffset;

        DataSlice txSrcSlice = DataSlice(srcBufferType, xDataInAddr[i], xSize);
        DataSlice txDstSlice = DataSlice(dstBufferType, xDataOutAddr[dstOffset], xSize);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);

        // 计算recv,recv侧的 xOffset，yOffset，dstOffset的计算方式和send侧一样
        DataSlice rxSrcSlice = DataSlice(srcBufferType, xDataInAddr[dstOffset], xSize);
        DataSlice rxDstSlice = DataSlice(dstBufferType, xDataOutAddr[i], xSize);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }

    // 同一列的用一个队列
    std::vector<DataSlice> txLocalSrcSlices, txLocalDstSlices;
    for (u32 i = 0; i < rankSize_; i++) {
        if (i % xRankSize_ == xRankId_) {
            txLocalSrcSlices.push_back(txSrcSlices[i]);
            txLocalDstSlices.push_back(txDstSlices[i]);
        }
    }
    // 本地拷贝
    CHK_RET(LocalCopySlices(xInsQues[xRankId_], txLocalSrcSlices, txLocalDstSlices));

    // 拷贝到其他卡
    for (u32 queIdx = 0; queIdx < xRankSize_; queIdx++) {
        if (queIdx == xRankId_) {
            continue;
        }

        std::vector<DataSlice> txRmtSrcSlices, txRmtDstSlices, rxRmtSrcSlices, rxRmtDstSlices;
        for (u32 i = 0; i < rankSize_; i++) {
            if (i % xRankSize_ == queIdx) {
                txRmtSrcSlices.push_back(txSrcSlices[i]);
                txRmtDstSlices.push_back(txDstSlices[i]);
                rxRmtSrcSlices.push_back(rxSrcSlices[i]);
                rxRmtDstSlices.push_back(rxDstSlices[i]);
            }
        }
        TxRxSlicesList sendRecvSlicesList({txRmtSrcSlices, txRmtDstSlices}, {rxRmtSrcSlices, rxRmtDstSlices});

        RankId rankSendRecv = queIdx + yRankId_ * xRankSize_;
        const std::vector<LinkData>& linkSendRecv = tempLinks.at(rankSendRecv);
        TxRxLinks sendRecvLinks(linkSendRecv[0], linkSendRecv[0]);

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, xInsQues[queIdx], 0, true, dmaMode),
            HCCL_ERROR("[InsTempAlltoAllMesh2D] RunMeshX SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2D::RunMeshY(
    std::vector<u64>& yDataInAddr, std::vector<u64>& yDataOutAddr, u64 ySize, BufferType srcBufferType,
    BufferType dstBufferType, DmaMode dmaMode, std::vector<InsQuePtr>& yInsQues, const ResLinks& tempLinks) const
{
    HCCL_DEBUG("RunMeshX begin, ySize is [%u]", ySize);
    if (ySize == 0) {
        HCCL_INFO("[InsTempAlltoAllMesh2D] RunMeshY, ySize is 0");
        return HcclResult::HCCL_SUCCESS;
    }

    std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;
    for (u32 i = 0; i < rankSize_; i++) {
        // 计算send
        u64 xOffset = i % xRankSize_;
        u64 yOffset = yRankId_;
        u64 dstOffset = yOffset * xRankSize_ + xOffset;

        DataSlice txSrcSlice = DataSlice(srcBufferType, yDataInAddr[i], ySize);
        DataSlice txDstSlice = DataSlice(dstBufferType, yDataOutAddr[dstOffset], ySize);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);

        // 计算recv,recv侧的 xOffset，yOffset，dstOffset的计算方式和send侧一样
        DataSlice rxSrcSlice = DataSlice(srcBufferType, yDataInAddr[dstOffset], ySize);
        DataSlice rxDstSlice = DataSlice(dstBufferType, yDataOutAddr[i], ySize);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }

    // 同一行的用一个队列
    std::vector<DataSlice> txLocalSrcSlices, txLocalDstSlices;
    for (u32 i = 0; i < rankSize_; i++) {
        if (i / xRankSize_ == yRankId_) {
            txLocalSrcSlices.push_back(txSrcSlices[i]);
            txLocalDstSlices.push_back(txDstSlices[i]);
        }
    }
    // 本地拷贝
    CHK_RET(LocalCopySlices(yInsQues[yRankId_], txLocalSrcSlices, txLocalDstSlices));

    // 拷贝到其他卡
    for (u32 queIdx = 0; queIdx < yRankSize_; queIdx++) {
        if (queIdx == yRankId_) {
            continue;
        }

        std::vector<DataSlice> txRmtSrcSlices, txRmtDstSlices, rxRmtSrcSlices, rxRmtDstSlices;
        for (u32 i = 0; i < rankSize_; i++) {
            if (i / xRankSize_ == queIdx) {
                txRmtSrcSlices.push_back(txSrcSlices[i]);
                txRmtDstSlices.push_back(txDstSlices[i]);
                rxRmtSrcSlices.push_back(rxSrcSlices[i]);
                rxRmtDstSlices.push_back(rxDstSlices[i]);
            }
        }
        TxRxSlicesList sendRecvSlicesList({txRmtSrcSlices, txRmtDstSlices}, {rxRmtSrcSlices, rxRmtDstSlices});

        RankId rankSendRecv = queIdx * xRankSize_ + xRankId_;
        const std::vector<LinkData>& linkSendRecv = tempLinks.at(rankSendRecv);
        TxRxLinks sendRecvLinks(linkSendRecv[0], linkSendRecv[0]);

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, yInsQues[queIdx], 0, true, dmaMode),
            HCCL_ERROR("[InsTempAlltoAllMesh2D] RunMeshY SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues) const
{
    (void)tempFuncs;
    HCCL_INFO("[InsTempAlltoAllMesh2D] Run algorithm start: rank[%d]", myRank_);

    u64 xSize = tempAlgParams.sliceSize / 2;
    u64 ySize = tempAlgParams.sliceSize - xSize;

    // queue arrangement
    std::vector<InsQuePtr> xInsQues, yInsQues;
    for (u32 queIdx = 0; queIdx < xRankSize_ + yRankSize_; queIdx++) {
        if (queIdx < xRankSize_) {
            xInsQues.push_back(tempInsQues[queIdx]);
        } else {
            yInsQues.push_back(tempInsQues[queIdx]);
        }
    }

    // stage1
    if (rankSize_ > 1) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }

    std::vector<u64> stage1XDataInAddr, stage1YDataInAddr, stage1XDataOutAddr, stage1YDataOutAddr;
    for (u32 i = 0; i < rankSize_; i++) {
        stage1XDataInAddr.push_back(tempAlgParams.inputSliceStride * i + tempAlgParams.buffInfo.inBuffBaseOff);
        stage1YDataInAddr.push_back(tempAlgParams.inputSliceStride * i + tempAlgParams.buffInfo.inBuffBaseOff + xSize);
        stage1XDataOutAddr.push_back(tempAlgParams.buffInfo.scratchBuffBaseOff + tempAlgParams.sliceSize * i);
        stage1YDataOutAddr.push_back(tempAlgParams.buffInfo.scratchBuffBaseOff + tempAlgParams.sliceSize * i + xSize);
    }

    CHK_RET(RunMeshX(
        stage1XDataInAddr, stage1XDataOutAddr, xSize, BufferType::INPUT, BufferType::SCRATCH, DmaMode::PUT, xInsQues,
        tempLinks));
    CHK_RET(RunMeshY(
        stage1YDataInAddr, stage1YDataOutAddr, ySize, BufferType::INPUT, BufferType::SCRATCH, DmaMode::PUT, yInsQues,
        tempLinks));

    if (rankSize_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }

    // stage2
    if (rankSize_ > 1) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }

    std::vector<u64> stage2XDataInAddr, stage2YDataInAddr, stage2XDataOutAddr, stage2YDataOutAddr;
    for (u32 i = 0; i < rankSize_; i++) {
        stage2XDataInAddr.push_back(tempAlgParams.buffInfo.scratchBuffBaseOff + tempAlgParams.sliceSize * i);
        stage2YDataInAddr.push_back(tempAlgParams.buffInfo.scratchBuffBaseOff + tempAlgParams.sliceSize * i + xSize);
        stage2XDataOutAddr.push_back(tempAlgParams.outputSliceStride * i + tempAlgParams.buffInfo.outBuffBaseOff);
        stage2YDataOutAddr.push_back(
            tempAlgParams.outputSliceStride * i + tempAlgParams.buffInfo.outBuffBaseOff + xSize);
    }

    BufferType outType = !tempFuncs.isBottom ? BufferType::SCRATCH : BufferType::OUTPUT;
    CHK_RET(RunMeshY(
        stage2XDataInAddr, stage2XDataOutAddr, xSize, BufferType::SCRATCH, outType, DmaMode::GET, yInsQues, tempLinks));
    CHK_RET(RunMeshX(
        stage2YDataInAddr, stage2YDataOutAddr, ySize, BufferType::SCRATCH, outType, DmaMode::GET, xInsQues, tempLinks));

    if (rankSize_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }

    HCCL_INFO("[InsTempAlltoAllMesh2D] Run algorithm end: rank[%d]", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
