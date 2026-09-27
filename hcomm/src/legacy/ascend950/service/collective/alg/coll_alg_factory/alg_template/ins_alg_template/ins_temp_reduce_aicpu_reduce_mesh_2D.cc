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
#include "ins_temp_reduce_aicpu_reduce_mesh_2D.h"

namespace Hccl {
InsTempReduceAicpuReduceMesh2D::InsTempReduceAicpuReduceMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap),
      sizeX_(static_cast<u32>(tempVTopo[0].size())),
      sizeY_(static_cast<u32>(tempVTopo[1].size())),
      curX_(myRank_ / sizeX_),
      curY_(myRank_ % sizeX_)
{}

InsTempReduceAicpuReduceMesh2D::~InsTempReduceAicpuReduceMesh2D() {}

HcclResult InsTempReduceAicpuReduceMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = sizeX_ - 1 + sizeY_ - 1 > 0 ? sizeX_ - 1 + sizeY_ - 1 : 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    CHK_RET(CalcResLinksMesh2D(myRank_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_INFO(
        "[InsTempReduceAicpuReduceMesh2D]CalcRes: queNum[%u], myRank[%d], tempRankSize[%u]", tempResReq.queNum, myRank_,
        tempRankSize_);
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceAicpuReduceMesh2D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult InsTempReduceAicpuReduceMesh2D::RunAicpuLocalReduce(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D][RunAicpuLocalReduce] empty queue"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    if (u32(myRank_) != root_) {
        return HCCL_SUCCESS;
    }
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

HcclResult InsTempReduceAicpuReduceMesh2D::RunGatherToRootX(
    const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    // send from x-axis
    DataSlice srcX(BufferType::INPUT, 0, dataSizeX_);
    DataSlice dstX(BufferType::SCRATCH, templateDataParams.sliceSize * myRank_, dataSizeX_);
    if (curY_ == rootY_) {
        LocalCopy(tempInsQues[0], srcX, dstX);
        for (u32 y = 0; y < sizeX_ - 1; y++) {
            u32 calcY = (rootY_ + y + 1) % sizeX_;
            u32 peerRank = curX_ * sizeX_ + calcY;
            const LinkData& linkRecv = tempLinks.at(peerRank)[0];
            std::vector<DataSlice> recvSrc;
            std::vector<DataSlice> recvDst;
            recvSrc.emplace_back(BufferType::INPUT, 0, dataSizeX_);
            recvDst.emplace_back(BufferType::SCRATCH, templateDataParams.sliceSize * peerRank, dataSizeX_);
            SlicesList rxSlicesList(recvSrc, recvDst);
            DataInfo recvData(linkRecv, rxSlicesList);
            CHK_PRT_RET(
                Recv(recvData, tempInsQues[y], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
        }
    } else {
        u32 peerRankX = curX_ * sizeX_ + rootY_;
        const LinkData& linkSendX = tempLinks.at(peerRankX)[0];
        std::vector<DataSlice> srcSlicesX = {srcX};
        std::vector<DataSlice> dstSlicesX = {dstX};
        SlicesList txSlicesListX(srcSlicesX, dstSlicesX);
        DataInfo sendDataX(linkSendX, txSlicesListX);
        CHK_PRT_RET(
            Send(sendDataX, tempInsQues[0], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduceMesh2D::RunGatherToRootY(
    const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    // send from y-axis
    DataSlice srcY(BufferType::INPUT, rankOffsetY_, dataSizeY_);
    DataSlice dstY(BufferType::SCRATCH, templateDataParams.sliceSize * myRank_ + rankOffsetY_, dataSizeY_);
    if (curX_ == rootX_) {
        LocalCopy(tempInsQues[sizeX_ - 1], srcY, dstY);
        for (u32 x = 0; x < sizeY_ - 1; x++) {
            u32 calcX = (rootX_ + x + 1) % sizeY_;
            u32 peerRank = calcX * sizeX_ + curY_;
            const LinkData& linkRecv = tempLinks.at(peerRank)[0];
            std::vector<DataSlice> recvSrc;
            std::vector<DataSlice> recvDst;
            recvSrc.emplace_back(BufferType::INPUT, rankOffsetY_, dataSizeY_);
            recvDst.emplace_back(
                BufferType::SCRATCH, templateDataParams.sliceSize * peerRank + rankOffsetY_, dataSizeY_);
            SlicesList rxSlicesList(recvSrc, recvDst);
            DataInfo recvData(linkRecv, rxSlicesList);
            CHK_PRT_RET(
                Recv(recvData, tempInsQues[x + sizeX_ - 1], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
        }
    } else {
        u32 peerRankY = rootX_ * sizeX_ + curY_;
        const LinkData& linkSendY = tempLinks.at(peerRankY)[0];
        std::vector<DataSlice> srcSlicesY = {srcY};
        std::vector<DataSlice> dstSlicesY = {dstY};
        SlicesList txSlicesListY(srcSlicesY, dstSlicesY);
        DataInfo sendDataY(linkSendY, txSlicesListY);
        CHK_PRT_RET(
            Send(sendDataY, tempInsQues[sizeX_ - 1], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduceMesh2D::RunGatherToRootXY(
    const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    // Step 1
    CHK_RET(PreSyncInterQueues(tempInsQues));
    CHK_RET(RunGatherToRootX(templateDataParams, tempLinks, tempInsQues));
    CHK_RET(RunGatherToRootY(templateDataParams, tempLinks, tempInsQues));
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduceMesh2D::RunXGatherToRoot(
    const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) const
{
    if (curX_ == rootX_ && curY_ == rootY_) {
        // recv data from x-axis
        for (u32 y = 0; y < sizeX_ - 1; y++) {
            u32 calcY = (rootY_ + y + 1) % sizeX_;
            u32 recvRank = rootX_ * sizeX_ + calcY;
            const LinkData& linkRecv = tempLinks.at(recvRank)[0];
            // calc recv data
            std::vector<DataSlice> srcDstRecvSlices;
            for (u32 x = 0; x < sizeY_ - 1; x++) {
                u32 calcX = (rootX_ + x + 1) % sizeY_;
                u32 peerRank = calcX * sizeX_ + calcY;
                DataSlice srcDstSlice(
                    BufferType::SCRATCH, templateDataParams.sliceSize * peerRank + rankOffsetY_, dataSizeY_);
                srcDstRecvSlices.emplace_back(srcDstSlice);
            }
            SlicesList rxSlicesList(srcDstRecvSlices, srcDstRecvSlices);
            DataInfo recvData(linkRecv, rxSlicesList);
            CHK_PRT_RET(
                Recv(recvData, tempInsQues[y], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
        }
    } else if (curX_ == rootX_) {
        // x-axis gather to root
        std::vector<DataSlice> srcDstSlicesX;
        const LinkData& linkSendXY = tempLinks.at(root_)[0];
        for (u32 x = 0; x < sizeY_; x++) {
            u32 peerRank = x * sizeX_ + curY_;
            DataSlice srcDstSlice(
                BufferType::SCRATCH, templateDataParams.sliceSize * peerRank + rankOffsetY_, dataSizeY_);
            srcDstSlicesX.emplace_back(srcDstSlice);
        }
        SlicesList txSlicesList(srcDstSlicesX, srcDstSlicesX);
        DataInfo sendDataX(linkSendXY, txSlicesList);
        CHK_PRT_RET(
            Send(sendDataX, tempInsQues[0], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduceMesh2D::RunYGatherToRoot(
    const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) const
{
    if (curX_ == rootX_ && curY_ == rootY_) {
        // recv data from y-axis
        for (u32 x = 0; x < sizeY_ - 1; x++) {
            u32 calcX = (rootX_ + x + 1) % sizeY_;
            u32 recvRank = calcX * sizeX_ + rootY_;
            const LinkData& linkRecv = tempLinks.at(recvRank)[0];
            // calc recv data
            std::vector<DataSlice> srcDstRecvSlices;
            for (u32 y = 0; y < sizeX_ - 1; y++) {
                u32 calcY = (rootY_ + y + 1) % sizeX_;
                u32 peerRank = calcX * sizeX_ + calcY;
                DataSlice srcDstSlice(BufferType::SCRATCH, templateDataParams.sliceSize * peerRank, dataSizeX_);
                srcDstRecvSlices.emplace_back(srcDstSlice);
            }
            SlicesList rxSlicesList(srcDstRecvSlices, srcDstRecvSlices);
            DataInfo recvData(linkRecv, rxSlicesList);
            CHK_PRT_RET(
                Recv(recvData, tempInsQues[sizeX_ - 1 + x], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
        }
    } else if (curY_ == rootY_) {
        // y-axis gather to root
        std::vector<DataSlice> srcDstSlicesY;
        const LinkData& linkSendXY = tempLinks.at(root_)[0];
        for (u32 y = 0; y < sizeX_; y++) {
            u32 peerRank = curX_ * sizeX_ + y;
            DataSlice srcDstSlice(BufferType::SCRATCH, templateDataParams.sliceSize * peerRank, dataSizeX_);
            srcDstSlicesY.emplace_back(srcDstSlice);
        }
        SlicesList txSlicesList(srcDstSlicesY, srcDstSlicesY);
        DataInfo sendDataY(linkSendXY, txSlicesList);
        CHK_PRT_RET(
            Send(sendDataY, tempInsQues[sizeX_ - 1], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceAicpuReduceMesh2D] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduceMesh2D::RunXYGatherToRoot(
    const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) const
{
    // Step 2
    CHK_RET(PreSyncInterQueues(tempInsQues));
    CHK_RET(RunXGatherToRoot(templateDataParams, tempLinks, tempInsQues));
    CHK_RET(RunYGatherToRoot(templateDataParams, tempLinks, tempInsQues));
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceAicpuReduceMesh2D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempReduceAicpuReduceMesh2D] Run start");
    if (tempVTopo_[0].size() == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size() - 1 + tempVTopo_[1].size() - 1;
    CHK_PRT_RET(
        queNum_ > tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceAicpuReduceMesh2D] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    rootX_ = root_ / sizeX_;
    rootY_ = root_ % sizeX_;
    const int splitDataXYFactor = 2;
    dataSizeX_ = templateDataParams.sliceSize / dataTypeSize_ / splitDataXYFactor * dataTypeSize_;
    rankOffsetY_ = dataSizeX_;
    dataSizeY_ = templateDataParams.sliceSize - dataSizeX_;
    RunGatherToRootXY(templateDataParams, tempLinks, tempInsQues);
    RunXYGatherToRoot(templateDataParams, tempLinks, tempInsQues);
    StreamSync(tempInsQues);
    RunAicpuLocalReduce(templateDataParams, tempInsQues);
    HCCL_INFO("[InsTempReduceAicpuReduceMesh2D] Run finished");
    return HCCL_SUCCESS;
}

} // namespace Hccl
