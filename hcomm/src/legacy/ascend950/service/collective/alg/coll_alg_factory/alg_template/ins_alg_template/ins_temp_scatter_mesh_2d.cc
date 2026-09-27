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
#include "ins_temp_scatter_mesh_2d.h"

const u32 CONST_NUM_TWO = 2;

namespace Hccl {
InsTempScatterMesh2D::InsTempScatterMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempScatterMesh2D::~InsTempScatterMesh2D() {}

HcclResult InsTempScatterMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_DEBUG("Enter InsTempScatterMesh2D::CalcRes");

    // 计算所需streamNum
    u32 queNum = (tempVTopo_[0].size() + tempVTopo_[1].size()) - CONST_NUM_TWO;
    tempResReq.queNum = queNum > 0 ? queNum : 1;
    tempResReq.streamNum = queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    HCCL_DEBUG("InsTempScatterMesh2D::CalcRes queNotifys size[%zu]", tempResReq.queNotifys.size());

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    // 计算建链关系
    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            if (neighborAlgRank > (tempVTopo_[dim].size() - 1)) {
                HCCL_ERROR(
                    "[CollAlgFactory] [InsTempScatterMesh2D] neighborAlgRank[%u] is invalid, "
                    "the Max rank[%zu].",
                    neighborAlgRank, tempVTopo_[dim].size() - 1);
                return HcclResult::HCCL_E_PARA;
            }

            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_DEBUG(
                "[CollAlgFactory] [InsTempScatterMesh2D] Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim,
                neighborRank);
            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }

    HCCL_INFO("InsTempScatterMesh2D::CalcRes done");
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempScatterMesh2D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;

    return tempRankSize_;
}

HcclResult InsTempScatterMesh2D::GenExtIns(
    TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsScatterMesh2D][GenExtIns] Root[%u] Rank[%d], start.", root_, myRank_);
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;

    queNum_ = (tempVTopo_[0].size() + tempVTopo_[1].size()) - CONST_NUM_TWO;
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempScatterMesh2D] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    if (tempVTopo_.size() < CONST_NUM_TWO) {
        HCCL_DEBUG("[InsTempScatterMesh2D] The size Of tempVTopo_ is less than two.");
        return HcclResult::HCCL_E_PARA;
    }

    xRankSize_ = tempVTopo_[0].size();   // x轴的ranksize
    yRankSize_ = tempVTopo_[1].size();   // y轴的ranksize
    xQueNum_ = tempVTopo_[0].size() - 1; // x轴的streamNum
    yQueNum_ = tempVTopo_[1].size() - 1; // y轴的streamNum

    for (u32 queIdx = 0; queIdx < queNum_; queIdx++) {
        if (queIdx < xQueNum_) {
            xInsQues_.push_back(tempInsQues[queIdx]);
        } else {
            yInsQues_.push_back(tempInsQues[queIdx]);
        }
    }

    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myRankX_));
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[1], myRankY_));
    // root节点的x, y坐标
    rootX_ = root_ % xRankSize_;
    rootY_ = root_ / xRankSize_;

    if (static_cast<u32>(myRank_) == root_ || tempRankSize_ == 1) {
        // if root Copy from Input to Output
        CHK_RET(PreDataCopy(tempAlgParams, tempInsQues));
    }

    u64 xDataSize = tempAlgParams.sliceSize / 2;
    u64 yDataSize = tempAlgParams.sliceSize - xDataSize;

    // 主从流前同步
    CHK_RET(PreSyncInterQueues(tempInsQues));
    // first Step
    if (static_cast<u32>(myRank_) == root_) {
        // root发送直达和中转数据块
        CHK_RET(RootMeshSend(tempAlgParams, tempResLinks, tempVTopo_[0], yRankSize_, 0, 1, 0, xDataSize, xInsQues_));
        CHK_RET(RootMeshSend(
            tempAlgParams, tempResLinks, tempVTopo_[1], xRankSize_, 1, 0, xDataSize, yDataSize, yInsQues_));
    } else if (myRankX_ == rootX_) {
        // root直连rank接收数据块
        CHK_RET(RankRecvFromRoot(tempAlgParams, tempResLinks, xRankSize_, 1, xDataSize, yDataSize, tempInsQues));
    } else if (myRankY_ == rootY_) {
        // root直连rank接收数据块
        CHK_RET(RankRecvFromRoot(tempAlgParams, tempResLinks, yRankSize_, xRankSize_, 0, xDataSize, tempInsQues));
    }

    // 主从流后同步
    CHK_RET(PostSyncInterQueues(tempInsQues));
    CHK_RET(PreSyncInterQueues(tempInsQues));

    // second
    if (static_cast<u32>(myRank_) != root_) {
        // 中转数据块的收发
        CHK_RET(RunDataCombine(tempAlgParams, tempResLinks, xDataSize, yDataSize, tempInsQues));
    }

    CHK_RET(PostSyncInterQueues(tempInsQues));

    if (static_cast<u32>(myRank_) != root_) {
        // if root Copy from Scratch to Output
        CHK_RET(PostDataCopy(tempAlgParams, tempInsQues));
    }

    HCCL_INFO("[InsScatterMesh2D][GenExtIns] Root[%u] Rank[%d], success.", root_, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterMesh2D::PreDataCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const
{
    u64 inOffset = tempAlgParams.inputSliceStride * myRank_ + tempAlgParams.buffInfo.inBuffBaseOff;

    DataSlice usrInSlice = DataSlice(BufferType::INPUT, inOffset, tempAlgParams.sliceSize);
    DataSlice usrOutSlice
        = DataSlice(BufferType::OUTPUT, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize);

    HCCL_INFO("PreCopy usrInSlice: %s, usrOutSlice: %s", usrInSlice.Describe().c_str(), usrOutSlice.Describe().c_str());

    CHK_RET(LocalCopy(tempInsQues[0], usrInSlice, usrOutSlice));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterMesh2D::PostDataCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const
{
    u64 inOffset = tempAlgParams.sliceSize * myRank_ + tempAlgParams.buffInfo.scratchBuffBaseOff;

    DataSlice usrInSlice = DataSlice(BufferType::SCRATCH, inOffset, tempAlgParams.sliceSize);
    DataSlice usrOutSlice
        = DataSlice(BufferType::OUTPUT, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize);

    HCCL_INFO(
        "PostCopy usrInSlice: %s, usrOutSlice: %s", usrInSlice.Describe().c_str(), usrOutSlice.Describe().c_str());

    CHK_RET(LocalCopy(tempInsQues[0], usrInSlice, usrOutSlice));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::RootMeshSend(
    TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, const std::vector<RankId> vTopo, const u32 xyRankSize,
    u32 rankDistX, u32 rankDistY, u64 dataOffSet, u64 tranDataSize, std::vector<InsQuePtr>& tempInsQues) const
{
    HCCL_DEBUG("[InsTempScatterMesh2D] Entry RootMeshSend");

    u32 insQueIndex = 0;
    for (u32 i = 0; i < vTopo.size(); i++) {
        u32 remoteRank = vTopo[i];
        if (static_cast<RankId>(remoteRank) == myRank_) {
            continue;
        }
        CHK_RET(SendDirect(tempAlgParams, tempInsQues[insQueIndex], tempResLinks[remoteRank][0], remoteRank));
        CHK_RET(SendTransit(
            tempAlgParams, tempInsQues[insQueIndex], tempResLinks[remoteRank][0], remoteRank, xyRankSize, rankDistX,
            rankDistY, dataOffSet, tranDataSize));

        insQueIndex++;
    }

    HCCL_DEBUG("[InsTempScatterMesh2D] RootMeshSend Done");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::RankRecvFromRoot(
    TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, const u32 xyRankSize, u32 rankDist, u64 dataOffSet,
    u64 tranDataSize, std::vector<InsQuePtr>& tempInsQues) const
{
    HCCL_DEBUG("[InsTempScatterMesh2D] Entry RankRecvFromRoot");

    CHK_RET(RecvDirect(tempAlgParams, tempInsQues[0], tempResLinks[root_][0], root_));
    CHK_RET(RecvTransit(
        tempAlgParams, tempInsQues[0], tempResLinks[root_][0], root_, xyRankSize, rankDist, dataOffSet, tranDataSize));

    HCCL_DEBUG("[InsTempScatterMesh2D] RankRecvFromRoot Done");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::RunDataCombine(
    TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, u64 tranDataSizeX, u64 tranDataSizeY,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_DEBUG("[InsTempScatterMesh2D] Entry RunDataCombine");

    if (myRankX_ == rootX_) {
        CHK_RET(DataCombineSend(tempAlgParams, tempResLinks, tempVTopo_[0], tranDataSizeX, tranDataSizeY, tempInsQues));
    } else if (myRankY_ == rootY_) {
        CHK_RET(DataCombineSend(tempAlgParams, tempResLinks, tempVTopo_[1], 0, tranDataSizeX, tempInsQues));
    } else {
        // 通过x, y坐标计算rank
        u32 recvSrcRank = 0;
        CHK_RET(GetRankId(rootX_, myRankY_, recvSrcRank));
        CHK_RET(
            DataCombineRecv(tempAlgParams, tempResLinks[recvSrcRank][0], tranDataSizeX, tranDataSizeY, tempInsQues[0]));

        CHK_RET(GetRankId(myRankX_, rootY_, recvSrcRank));
        CHK_RET(DataCombineRecv(tempAlgParams, tempResLinks[recvSrcRank][0], 0, tranDataSizeX, tempInsQues[0]));
    }

    HCCL_DEBUG("[InsTempScatterMesh2D] RunDataCombine Done");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::DataCombineSend(
    const TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, const std::vector<RankId> vTopo, u64 dataOffSet,
    u64 tranDataSize, std::vector<InsQuePtr>& tempInsQues) const
{
    HCCL_DEBUG("[InsTempScatterMesh2D] Entry DataCombineSend");

    u32 insQueIndex = 0;

    for (u32 i = 0; i < vTopo.size(); i++) {
        u32 remoteRank = vTopo[i];
        if (static_cast<RankId>(remoteRank) == myRank_) {
            continue;
        }

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;

        u64 txSrcOffset = tempAlgParams.sliceSize * remoteRank + tempAlgParams.buffInfo.scratchBuffBaseOff + dataOffSet;
        u64 txDstOffset = tempAlgParams.sliceSize * remoteRank + tempAlgParams.buffInfo.scratchBuffBaseOff + dataOffSet;

        BufferType txSrcBuffType = BufferType::SCRATCH;
        BufferType txDtsBuffType = BufferType::SCRATCH;

        txSrcSlices.emplace_back(txSrcBuffType, txSrcOffset, tranDataSize);
        txDstSlices.emplace_back(txDtsBuffType, txDstOffset, tranDataSize);

        SlicesList sendDataSlice(txSrcSlices, txDstSlices);
        DataInfo sendDataInfo(tempResLinks[remoteRank][0], sendDataSlice);

        CHK_RET(Send(sendDataInfo, tempInsQues[insQueIndex++], 0, true, DmaMode::PUT));
    }

    HCCL_DEBUG("[InsTempScatterMesh2D] DataCombineSend Done");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::DataCombineRecv(
    const TemplateDataParams& tempAlgParams, LinkData link, u64 dataOffSet, u64 tranDataSize, InsQuePtr queue) const
{
    HCCL_DEBUG("[InsTempScatterMesh2D] Entry DataCombineRecv");

    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    u64 rxSrcOffset = tempAlgParams.sliceSize * myRank_ + tempAlgParams.buffInfo.scratchBuffBaseOff + dataOffSet;
    u64 rxDstOffset = tempAlgParams.sliceSize * myRank_ + tempAlgParams.buffInfo.scratchBuffBaseOff + dataOffSet;

    BufferType txSrcBuffType = BufferType::SCRATCH;
    BufferType txDtsBuffType = BufferType::SCRATCH;

    rxSrcSlices.emplace_back(txSrcBuffType, rxSrcOffset, tranDataSize);
    rxDstSlices.emplace_back(txDtsBuffType, rxDstOffset, tranDataSize);

    SlicesList rendDataSlice(rxSrcSlices, rxDstSlices);
    DataInfo rendDataInfo(link, rendDataSlice);

    CHK_RET(Recv(rendDataInfo, queue, 0, true, DmaMode::PUT));

    HCCL_DEBUG("[InsTempScatterMesh2D] DataCombineRecv Done");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::GetRankId(u32 xRank, u32 yRank, u32& rank) const
{
    rank = (yRank * xRankSize_ + xRank) % tempRankSize_;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::SendDirect(
    TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank) const
{
    // 发送直达数据块
    HCCL_DEBUG("[InsTempScatterMesh2D] Entry SendDirect");

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    u64 txSrcOffset = tempAlgParams.inputSliceStride * remoteRank + tempAlgParams.buffInfo.inBuffBaseOff;
    u64 txDstOffset = tempAlgParams.sliceSize * remoteRank + tempAlgParams.buffInfo.scratchBuffBaseOff;

    txSrcSlices.emplace_back(BufferType::INPUT, txSrcOffset, tempAlgParams.sliceSize);
    txDstSlices.emplace_back(BufferType::SCRATCH, txDstOffset, tempAlgParams.sliceSize);

    SlicesList sendDataSlice(txSrcSlices, txDstSlices);
    DataInfo sendDataInfo(link, sendDataSlice);

    CHK_RET(Send(sendDataInfo, queue, 0, true, DmaMode::PUT));

    HCCL_DEBUG("[InsTempScatterMesh2D] SendDirect Done");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::SendTransit(
    const TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank, u32 xyRankSize,
    u32 rankDistX, u32 rankDistY, u64 xyOffSet, u64 tranDataSize) const
{
    // 发送中转数据块
    u32 remoteRankX = remoteRank % xRankSize_;
    u32 remoteRankY = remoteRank / xRankSize_;

    for (u32 i = 0; i < xyRankSize - 1; i++) {
        remoteRankX = (remoteRankX + rankDistX) % xRankSize_;
        remoteRankY = (remoteRankY + rankDistY) % yRankSize_;

        u32 tranRank = 0;
        CHK_RET(GetRankId(remoteRankX, remoteRankY, tranRank));

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;

        u64 inputSrcOffset
            = tempAlgParams.inputSliceStride * tranRank + tempAlgParams.buffInfo.inBuffBaseOff + xyOffSet;
        u64 scartSrcOffset = tempAlgParams.sliceSize * tranRank + tempAlgParams.buffInfo.scratchBuffBaseOff + xyOffSet;
        u64 txDstOffset = tempAlgParams.sliceSize * tranRank + tempAlgParams.buffInfo.scratchBuffBaseOff + xyOffSet;

        u64 txSrcOffset = static_cast<u32>(myRank_) == root_ ? inputSrcOffset : scartSrcOffset;

        BufferType txSrcBuffType = static_cast<u32>(myRank_) == root_ ? BufferType::INPUT : BufferType::SCRATCH;
        BufferType txDtsBuffType = BufferType::SCRATCH;

        txSrcSlices.emplace_back(txSrcBuffType, txSrcOffset, tranDataSize);
        txDstSlices.emplace_back(txDtsBuffType, txDstOffset, tranDataSize);

        SlicesList sendDataSlice(txSrcSlices, txDstSlices);
        DataInfo sendDataInfo(link, sendDataSlice);

        CHK_RET(Send(sendDataInfo, queue, 0, true, DmaMode::PUT));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::RecvDirect(
    TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank) const
{
    // 接收直达数据块
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    u64 rxSrcOffset = tempAlgParams.inputSliceStride * remoteRank + tempAlgParams.buffInfo.inBuffBaseOff;
    u64 rxDstOffset = tempAlgParams.sliceSize * remoteRank + tempAlgParams.buffInfo.scratchBuffBaseOff;

    rxSrcSlices.emplace_back(BufferType::INPUT, rxSrcOffset, tempAlgParams.sliceSize);
    rxDstSlices.emplace_back(BufferType::SCRATCH, rxDstOffset, tempAlgParams.sliceSize);

    SlicesList recvDataSlice(rxSrcSlices, rxDstSlices);
    DataInfo recvDataInfo(link, recvDataSlice);

    CHK_RET(Recv(recvDataInfo, queue, 0, true, DmaMode::PUT));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh2D::RecvTransit(
    const TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank, u32 xyRankSize,
    u32 rankDist, u64 xyOffSet, u64 tranDataSize) const
{
    // 接收直达数据块
    u32 tranIndex = remoteRank;
    for (u32 i = 0; i < xyRankSize - 1; i++) {
        tranIndex += rankDist;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        u32 tranRank = tranIndex % tempRankSize_;

        u64 inputSrcOffset
            = tempAlgParams.inputSliceStride * remoteRank + tempAlgParams.buffInfo.inBuffBaseOff + xyOffSet;
        u64 scratDstOffset
            = tempAlgParams.sliceSize * remoteRank + tempAlgParams.buffInfo.scratchBuffBaseOff + xyOffSet;
        u64 rxDstOffset = tempAlgParams.sliceSize * tranRank + tempAlgParams.buffInfo.scratchBuffBaseOff + xyOffSet;

        u64 rxSrcOffset = inputSrcOffset;

        BufferType txSrcBuffType = BufferType::INPUT;
        BufferType txDtsBuffType = BufferType::SCRATCH;

        if (myRankX_ != rootX_ && myRankY_ != rootY_) {
            rxSrcOffset = scratDstOffset;
            txSrcBuffType = BufferType::SCRATCH;
        }

        rxSrcSlices.emplace_back(txSrcBuffType, rxSrcOffset, tranDataSize);
        rxDstSlices.emplace_back(txDtsBuffType, rxDstOffset, tranDataSize);

        SlicesList rendDataSlice(rxSrcSlices, rxDstSlices);
        DataInfo rendDataInfo(link, rendDataSlice);

        CHK_RET(Recv(rendDataInfo, queue, 0, true, DmaMode::PUT));
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
