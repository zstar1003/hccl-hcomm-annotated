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
#include "ins_alg_template/ins_temp_broadcast_mesh_2D_two_shot.h"

namespace Hccl {

InsTempBroadcastMesh2DTwoShot::InsTempBroadcastMesh2DTwoShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap),
      sizeX_(static_cast<u32>(tempVTopo[0].size())),
      sizeY_(static_cast<u32>(tempVTopo[1].size()))
{
    std::tie(curX_, curY_) = GetRankPos(myRank_);
}

InsTempBroadcastMesh2DTwoShot::~InsTempBroadcastMesh2DTwoShot() {}

HcclResult InsTempBroadcastMesh2DTwoShot::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = sizeX_ - 1 + sizeY_ - 1 > 0 ? sizeX_ - 1 + sizeY_ - 1 : 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    CHK_RET(CalcResLinksMesh2D(myRank_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_INFO(
        "[InsTempBroadcastMesh2DTwoShot]CalcRes: queNum[%u], myRank[%d], tempRankSize[%u]", tempResReq.queNum, myRank_,
        tempRankSize_);
    return HcclResult::HCCL_SUCCESS;
}
u32 InsTempBroadcastMesh2DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    if (op_.opMode == OpMode::OPBASE) {
        return 1;
    } else {
        return 0;
    }
}

HcclResult InsTempBroadcastMesh2DTwoShot::RunScatter(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, u64 baseOffset, u64 dataSize, u32 root,
    bool isRootX)
{
    if (dataSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 rootX, rootY;
    u64 rankStride = isRootX ? dataSize / sizeX_ / dataTypeSize_ * dataTypeSize_ :
                               dataSize / sizeY_ / dataTypeSize_ * dataTypeSize_;
    std::tie(rootX, rootY) = GetRankPos(root);
    if (root == u32(myRank_)) {
        if (isRootX) {
            // x轴(板内)SendRecvInfo
            for (u32 i = 0, index = 0; i < sizeX_; i++) {
                if (i == rootY)
                    continue;
                u64 dataOffset = baseOffset + i * rankStride;
                u64 curSize = i == sizeX_ - 1 ? dataSize - rankStride * i : rankStride;
                u32 peerRank = rootX * sizeX_ + i;
                const LinkData& linkSend = tempLinks.at(peerRank)[0];
                u64 dataOffsetAll = opMode_ == OpMode::OFFLOAD ? inputOffset_ + dataOffset : dataOffset;
                BufferType srcDstBufferType = opMode_ == OpMode::OFFLOAD ? BufferType::INPUT : BufferType::SCRATCH;
                std::vector<DataSlice> srcSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
                std::vector<DataSlice> dstSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
                SlicesList txSlicesList(srcSlices, dstSlices);
                DataInfo sendData(linkSend, txSlicesList);
                CHK_PRT_RET(
                    Send(sendData, tempInsQues[index], 0, true, DmaMode::GET),
                    HCCL_ERROR("[InsTempBroadcastMesh2DTwoShot] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
                index++;
            }
        } else {
            // y轴(板间)SendRecvInfo
            for (u32 i = 0, index = 0; i < sizeY_; i++) {
                if (i == rootX)
                    continue;
                u64 dataOffset = baseOffset + i * rankStride;
                u64 curSize = i == sizeY_ - 1 ? dataSize - rankStride * i : rankStride;
                u32 peerRank = i * sizeX_ + rootY;
                const LinkData& linkSend = tempLinks.at(peerRank)[0];
                u64 dataOffsetAll = opMode_ == OpMode::OFFLOAD ? inputOffset_ + dataOffset : dataOffset;
                BufferType srcDstBufferType = opMode_ == OpMode::OFFLOAD ? BufferType::INPUT : BufferType::SCRATCH;
                std::vector<DataSlice> srcSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
                std::vector<DataSlice> dstSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
                SlicesList txSlicesList(srcSlices, dstSlices);
                DataInfo sendData(linkSend, txSlicesList);
                CHK_PRT_RET(
                    Send(sendData, tempInsQues[sizeX_ - 1 + index], 0, true, DmaMode::GET),
                    HCCL_ERROR("[InsTempBroadcastMesh2DTwoShot] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
                index++;
            }
        }
    } else if (curX_ == rootX && isRootX) {
        u64 dataOffset = baseOffset + curY_ * rankStride;
        u64 curSize = curY_ == sizeX_ - 1 ? dataSize - rankStride * curY_ : rankStride;
        const LinkData& linkRecv = tempLinks.at(root)[0];
        u64 dataOffsetAll = opMode_ == OpMode::OFFLOAD ? inputOffset_ + dataOffset : dataOffset;
        BufferType srcDstBufferType = opMode_ == OpMode::OFFLOAD ? BufferType::INPUT : BufferType::SCRATCH;
        std::vector<DataSlice> srcSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
        std::vector<DataSlice> dstSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
        SlicesList rxSlicesList(srcSlices, dstSlices);
        DataInfo recvData(linkRecv, rxSlicesList);
        u32 index = curY_ > rootY ? curY_ - 1 : curY_;
        CHK_PRT_RET(
            Recv(recvData, tempInsQues[index], 0, true, DmaMode::GET),
            HCCL_ERROR("[InsTempBroadcastMesh2DTwoShot] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
    } else if (curY_ == rootY && !isRootX) {
        u64 dataOffset = baseOffset + curX_ * rankStride;
        u64 curSize = curX_ == sizeY_ - 1 ? dataSize - rankStride * curX_ : rankStride;
        const LinkData& linkRecv = tempLinks.at(root)[0];
        u64 dataOffsetAll = opMode_ == OpMode::OFFLOAD ? inputOffset_ + dataOffset : dataOffset;
        BufferType srcDstBufferType = opMode_ == OpMode::OFFLOAD ? BufferType::INPUT : BufferType::SCRATCH;
        std::vector<DataSlice> srcSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
        std::vector<DataSlice> dstSlices = {DataSlice(srcDstBufferType, dataOffsetAll, curSize)};
        SlicesList rxSlicesList(srcSlices, dstSlices);
        DataInfo recvData(linkRecv, rxSlicesList);
        u32 index = curX_ > rootX ? curX_ - 1 : curX_;
        CHK_PRT_RET(
            Recv(recvData, tempInsQues[sizeX_ - 1 + index], 0, true, DmaMode::GET),
            HCCL_ERROR("[InsTempBroadcastMesh2DTwoShot] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::RunAllgather(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, u64 baseOffset, u64 dataSize, bool isX, bool isDma)
{
    if (dataSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    u64 rankStride
        = isX ? dataSize / sizeX_ / dataTypeSize_ * dataTypeSize_ : dataSize / sizeY_ / dataTypeSize_ * dataTypeSize_;

    if (isX) {
        for (u32 i = 0, index = 0; i < sizeX_; i++) {
            if (i == curY_)
                continue;
            // send to peer rank
            u64 sendOffset = baseOffset + curY_ * rankStride;
            u64 sendSize = curY_ == sizeX_ - 1 ? dataSize - rankStride * curY_ : rankStride;
            u32 peerRank = curX_ * sizeX_ + i;
            // recv from peer rank
            u64 recvOffset = baseOffset + i * rankStride;
            u64 recvSize = i == sizeX_ - 1 ? dataSize - rankStride * i : rankStride;
            const LinkData& linkSendRecv = tempLinks.at(peerRank)[0];
            u64 sendOffsetSrc = opMode_ == OpMode::OFFLOAD ? inputOffset_ + sendOffset : sendOffset;
            u64 sendOffsetDst = opMode_ == OpMode::OFFLOAD || isDma ? inputOffset_ + sendOffset : sendOffset;
            u64 recvOffsetSrc = opMode_ == OpMode::OFFLOAD ? inputOffset_ + recvOffset : recvOffset;
            u64 recvOffsetDst = opMode_ == OpMode::OFFLOAD || isDma ? inputOffset_ + recvOffset : recvOffset;
            BufferType srcBufferType = opMode_ == OpMode::OFFLOAD ? BufferType::INPUT : BufferType::SCRATCH;
            BufferType dstBufferType = opMode_ == OpMode::OFFLOAD || isDma ? BufferType::INPUT : BufferType::SCRATCH;
            std::vector<DataSlice> txSrcSlices = {DataSlice(srcBufferType, sendOffsetSrc, sendSize)};
            std::vector<DataSlice> txDstSlices = {DataSlice(dstBufferType, sendOffsetDst, sendSize)};
            std::vector<DataSlice> rxSrcSlices = {DataSlice(srcBufferType, recvOffsetSrc, recvSize)};
            std::vector<DataSlice> rxDstSlices = {DataSlice(dstBufferType, recvOffsetDst, recvSize)};
            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            TxRxLinks sendRecvLinks(linkSendRecv, linkSendRecv);
            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, tempInsQues[index], 0, true, DmaMode::GET),
                HCCL_ERROR("[InsTempBroadcastMesh2DTwoShot] BatchSendRecv failed"), HcclResult::HCCL_E_INTERNAL);
            index++;
        }
    } else {
        for (u32 i = 0, index = 0; i < sizeY_; i++) {
            if (i == curX_)
                continue;
            // send to peer rank
            u64 sendOffset = baseOffset + curX_ * rankStride;
            u64 sendSize = curX_ == sizeY_ - 1 ? dataSize - rankStride * curX_ : rankStride;
            u32 peerRank = i * sizeX_ + curY_;
            // recv from peer rank
            u64 recvOffset = baseOffset + i * rankStride;
            u64 recvSize = i == sizeY_ - 1 ? dataSize - rankStride * i : rankStride;
            const LinkData& linkSendRecv = tempLinks.at(peerRank)[0];
            u64 sendOffsetSrc = opMode_ == OpMode::OFFLOAD ? inputOffset_ + sendOffset : sendOffset;
            u64 sendOffsetDst = opMode_ == OpMode::OFFLOAD || isDma ? inputOffset_ + sendOffset : sendOffset;
            u64 recvOffsetSrc = opMode_ == OpMode::OFFLOAD ? inputOffset_ + recvOffset : recvOffset;
            u64 recvOffsetDst = opMode_ == OpMode::OFFLOAD || isDma ? inputOffset_ + recvOffset : recvOffset;
            BufferType srcBufferType = opMode_ == OpMode::OFFLOAD ? BufferType::INPUT : BufferType::SCRATCH;
            BufferType dstBufferType = opMode_ == OpMode::OFFLOAD || isDma ? BufferType::INPUT : BufferType::SCRATCH;
            std::vector<DataSlice> txSrcSlices = {DataSlice(srcBufferType, sendOffsetSrc, sendSize)};
            std::vector<DataSlice> txDstSlices = {DataSlice(dstBufferType, sendOffsetDst, sendSize)};
            std::vector<DataSlice> rxSrcSlices = {DataSlice(srcBufferType, recvOffsetSrc, recvSize)};
            std::vector<DataSlice> rxDstSlices = {DataSlice(dstBufferType, recvOffsetDst, recvSize)};
            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            TxRxLinks sendRecvLinks(linkSendRecv, linkSendRecv);
            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, tempInsQues[sizeX_ - 1 + index], 0, true, DmaMode::GET),
                HCCL_ERROR("[InsTempBroadcastMesh2DTwoShot] BatchSendRecv failed"), HcclResult::HCCL_E_INTERNAL);
            index++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::Run1RootScatterXY(
    u64 dataSize, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    u64 perRank = dataSize / (sizeX_ + sizeY_) / dataTypeSize_ * dataTypeSize_;
    u32 rootX, rootY;
    std::tie(rootX, rootY) = GetRankPos(root_);
    u64 step14DataSizeX = perRank * sizeX_;
    u64 step14DataSizeY = dataSize - step14DataSizeX;
    u64 step14OffsetX = 0;
    u64 step14OffsetY = step14DataSizeX;
    CHK_RET(PreSyncInterQueues(tempInsQues));
    CHK_RET(RunScatter(tempLinks, tempInsQues, step14OffsetX, step14DataSizeX, root_, true));
    CHK_RET(RunScatter(tempLinks, tempInsQues, step14OffsetY, step14DataSizeY, root_, false));
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::RunNRootScatterYX(
    u64 dataSize, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    u64 perRank = dataSize / (sizeX_ + sizeY_) / dataTypeSize_ * dataTypeSize_;
    u32 rootX, rootY;
    std::tie(rootX, rootY) = GetRankPos(root_);
    u64 step23DataSizeX = perRank * sizeX_;
    u64 step23BaseOffsetY = perRank * sizeX_;
    u64 step23DataSizeY = dataSize - perRank * sizeX_;
    u64 step23PerRankY = step23DataSizeY / sizeY_ / dataTypeSize_ * dataTypeSize_;
    CHK_RET(PreSyncInterQueues(tempInsQues));
    // n root scatter
    for (u32 i = 0; i < sizeX_; i++) {
        u64 dataOffset = i * perRank;
        u64 curSize = i == sizeX_ - 1 ? step23DataSizeX - perRank * i : perRank;
        u32 curRank = rootX * sizeX_ + i;
        CHK_RET(RunScatter(tempLinks, tempInsQues, dataOffset, curSize, curRank, false));
    }

    for (u32 i = 0; i < sizeY_; i++) {
        u64 dataOffset = step23BaseOffsetY + i * step23PerRankY;
        u64 curSize = i == sizeY_ - 1 ? step23DataSizeY - step23PerRankY * i : step23PerRankY;
        u64 curRank = i * sizeX_ + rootY;
        CHK_RET(RunScatter(tempLinks, tempInsQues, dataOffset, curSize, curRank, true));
    }
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::RunAllgatherYX(
    u64 dataSize, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    u64 perRank = dataSize / (sizeX_ + sizeY_) / dataTypeSize_ * dataTypeSize_;
    u32 rootX, rootY;
    std::tie(rootX, rootY) = GetRankPos(root_);
    u64 step23DataSizeX = perRank * sizeX_;
    u64 step23BaseOffsetY = perRank * sizeX_;
    u64 step23DataSizeY = dataSize - perRank * sizeX_;
    u64 step23PerRankY = step23DataSizeY / sizeY_ / dataTypeSize_ * dataTypeSize_;
    // allgather yx轴
    CHK_RET(PreSyncInterQueues(tempInsQues));
    u64 dataOffsetY = curY_ * perRank;
    u64 curSizeY = curY_ == sizeX_ - 1 ? step23DataSizeX - perRank * curY_ : perRank;
    CHK_RET(RunAllgather(tempLinks, tempInsQues, dataOffsetY, curSizeY, false, false));

    u64 dataOffsetX = step23BaseOffsetY + curX_ * step23PerRankY;
    u64 curSizeX = curX_ == sizeY_ - 1 ? step23DataSizeY - step23PerRankY * curX_ : step23PerRankY;
    CHK_RET(RunAllgather(tempLinks, tempInsQues, dataOffsetX, curSizeX, true, false));

    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::RunAllgatherXY(
    u64 dataSize, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    u64 perRank = dataSize / (sizeX_ + sizeY_) / dataTypeSize_ * dataTypeSize_;
    u32 rootX, rootY;
    std::tie(rootX, rootY) = GetRankPos(root_);
    u64 step14DataSizeX = perRank * sizeX_;
    u64 step14DataSizeY = dataSize - step14DataSizeX;
    u64 step14OffsetX = 0;
    u64 step14OffsetY = step14DataSizeX;
    // allgather xy轴
    CHK_RET(PreSyncInterQueues(tempInsQues));
    CHK_RET(RunAllgather(tempLinks, tempInsQues, step14OffsetX, step14DataSizeX, true, true));
    CHK_RET(RunAllgather(tempLinks, tempInsQues, step14OffsetY, step14DataSizeY, false, true));
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::PreCopy(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    if (opMode_ == OpMode::OFFLOAD || u32(myRank_) != root_) {
        return HcclResult::HCCL_SUCCESS;
    }
    std::vector<DataSlice> scratchSlices = {DataSlice(BufferType::SCRATCH, 0, templateDataParams.sliceSize)};
    std::vector<DataSlice> inputSlices
        = {DataSlice(BufferType::INPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize)};
    CHK_RET(LocalCopySlices(tempInsQues[0], inputSlices, scratchSlices));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::PostCopy(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    if (opMode_ == OpMode::OFFLOAD) {
        return HcclResult::HCCL_SUCCESS;
    }
    u64 perRank = templateDataParams.sliceSize / (sizeX_ + sizeY_) / dataTypeSize_ * dataTypeSize_;
    u64 scratchDataSizeX = perRank * sizeX_;
    u64 scratchDataSizeY = templateDataParams.sliceSize - scratchDataSizeX;
    u64 scratchOffsetX = 0;
    u64 scratchOffsetY = scratchDataSizeX;
    // x轴当前卡
    u64 curRankDataOffsetX = scratchOffsetX + perRank * curY_;
    u64 curRankDataSizeX = perRank;
    u64 perRankY = scratchDataSizeY / sizeY_ / dataTypeSize_ * dataTypeSize_;
    u64 curRankDataOffsetY = scratchOffsetY + perRankY * curX_;
    u64 curRankDataSizeY = curX_ == sizeY_ - 1 ? scratchDataSizeY - curX_ * perRankY : perRankY;

    std::vector<DataSlice> scratchSlices
        = {DataSlice(BufferType::SCRATCH, curRankDataOffsetX, curRankDataSizeX),
           DataSlice(BufferType::SCRATCH, curRankDataOffsetY, curRankDataSizeY)};
    std::vector<DataSlice> inputSlices = {
        DataSlice(BufferType::INPUT, templateDataParams.buffInfo.inBuffBaseOff + curRankDataOffsetX, curRankDataSizeX),
        DataSlice(BufferType::INPUT, templateDataParams.buffInfo.inBuffBaseOff + curRankDataOffsetY, curRankDataSizeY)};
    CHK_RET(LocalCopySlices(tempInsQues[0], scratchSlices, inputSlices));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh2DTwoShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempBroadcastMesh2DTwoShot][GenExtIns] Broadcast2DMesh start: rank[%d] end", myRank_);
    opMode_ = tempFuncs.opMode;
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    inputOffset_ = templateDataParams.buffInfo.inBuffBaseOff;
    if (sizeX_ == 1 && sizeY_ == 1) {
        HCCL_INFO("[InsTempBroadcastMesh2DTwoShot][GenExtIns] Broadcast2DMesh finished: rank[%d] end", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    queNum_ = sizeX_ - 1 + sizeY_ - 1;
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempBroadcastMesh2DTwoShot] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    CHK_RET(PreCopy(templateDataParams, tempInsQues));

    CHK_RET(Run1RootScatterXY(templateDataParams.sliceSize, tempLinks, tempInsQues));
    CHK_RET(RunNRootScatterYX(templateDataParams.sliceSize, tempLinks, tempInsQues));
    CHK_RET(RunAllgatherYX(templateDataParams.sliceSize, tempLinks, tempInsQues));
    CHK_RET(RunAllgatherXY(templateDataParams.sliceSize, tempLinks, tempInsQues));

    CHK_RET(PostCopy(templateDataParams, tempInsQues));

    HCCL_INFO("[InsTempBroadcastMesh2DTwoShot][GenExtIns] Broadcast2DMesh finished: rank[%d] end", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
