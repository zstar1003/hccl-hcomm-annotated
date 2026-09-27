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
#include "ins_temp_scatter_mesh_1d.h"

namespace Hccl {
InsTempScatterMesh1D::InsTempScatterMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempScatterMesh1D::~InsTempScatterMesh1D() {}

HcclResult InsTempScatterMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_DEBUG("Enter InsTempScatterMesh1D::CalcRes");
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    auto& linkReq = tempResReq.links;
    u32 pathNum = 0;
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        auto remoteRank = resReqIter->first;
        if (rank2PathNumMap_.find(remoteRank) == rank2PathNumMap_.end() || rank2PathNumMap_[remoteRank] == 0) {
            HCCL_ERROR("[InsTempScatterMesh1D] No path to remoteRank[%d]", remoteRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (pathNum == 0) {
            pathNum = rank2PathNumMap_[remoteRank];
        } else if (rank2PathNumMap_[remoteRank] != pathNum) {
            HCCL_ERROR(
                "[InsTempScatterMesh1D] Inconsistency pathNum to remoteRanks, Previous consistent pathNum=[%u], "
                "mismatched "
                "remoteRank=[%d], pathNum=[%u]",
                pathNum, remoteRank, rank2PathNumMap_[remoteRank]);
            return HcclResult::HCCL_E_INTERNAL;
        }
        resReqIter->second = pathNum;
    }

    tempResReq.queNum = tempVTopo_[0].size() * pathNum;
    HCCL_INFO("[InsTempScatterMesh1D] tempResReq.queNum = %u", tempResReq.queNum);
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    if (op_.opMode == OpMode::OPBASE) {
        return 1;
    } else {
        return 0;
    }
}

// 需要支持 input->output, input->scratch, scratch->output
HcclResult InsTempScatterMesh1D::GenExtIns(
    TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempScatterMesh1D][Run] start: Rank [%d]", myRank_);

    opMode_ = tempFuncs.opMode;
    buffInfo_ = tempAlgParams.buffInfo;
    majorQueNum_ = tempVTopo_[0].size();
    isZeroCopy_ = opMode_ == OpMode::OFFLOAD && buffInfo_.inBuffType == BufferType::INPUT
                  && buffInfo_.outBuffType == BufferType::OUTPUT;

    uint32_t linkNum = tempResLinks.begin()->second.size();
    // 流的数量不能少于linkNum
    CHK_PRT_RET(
        linkNum > tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempScatterMesh1D] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(tempResLinks.begin()->second, dataSplitRate));
    queNumPerNeighbor_ = linkNum;
    std::vector<InsQuePtr> localInsQues;
    localInsQues.push_back(tempInsQues[0]);
    localInsQues.push_back(tempInsQues[tempInsQues.size() - 1]);
    // queNumPerNeighbor_初始化是1
    CHK_PRT_RET(
        majorQueNum_ * queNumPerNeighbor_ > tempInsQues.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [InsTempScatterMesh1D] Rank [%d], requiredQueNum [%u] not equals to "
            "templateQueNum [%u].",
            myRank_, majorQueNum_ * queNumPerNeighbor_, tempInsQues.size()),
        HcclResult::HCCL_E_INTERNAL);

    PreCopy(tempAlgParams, tempInsQues);
    // semaphore sync
    if (majorQueNum_ > 1) { // more than one rank
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }

    // run Mesh
    CHK_RET(RunMesh(tempAlgParams, tempResLinks, tempInsQues));

    // semaphore sync
    if (majorQueNum_ > 1) { // more than one rank
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }
    PostCopy(tempAlgParams, tempInsQues);
    return HcclResult::HCCL_SUCCESS;
}

uint64_t InsTempScatterMesh1D::GetExpandedMode() const { return 1; }

HcclResult InsTempScatterMesh1D::RunMeshTx(
    u32 myAlgRank, u32 repeatTimes, const TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 sliceSize = tempAlgParams.tailSize == 0 ? tempAlgParams.sliceSize : tempAlgParams.tailSize;
    u32 count = 0;
    for (u32 algRank = 0; algRank < tempVTopo_[0].size(); algRank++) {
        if (myAlgRank == algRank) {
            continue;
        }
        if (tempInsQues.size() < tempVTopo_[0].size()) {
            HCCL_ERROR(
                "tempInsQues size [%zu] is smaller than tempVTopo_[0].size() [%zu]", tempInsQues.size(),
                tempVTopo_[0].size());
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 peerRank = tempVTopo_[0][algRank];
        std::vector<LinkData>& neighborLinkDatas = tempResLinks.at(peerRank);
        u32 linkNum = rank2PathNumMap_.at(peerRank);
        if (linkNum != neighborLinkDatas.size()) {
            HCCL_ERROR("InsTempScatterMesh1D::RunMeshTx linkNum != neighborLinkDatas.size()");
            return HcclResult::HCCL_E_INTERNAL;
        }
        std::vector<float> dataSplitRate(linkNum);
        CHK_RET(CalcDataSplitRateForLinks(neighborLinkDatas, dataSplitRate));

        u64 srcOffset = buffInfo_.inBuffType == BufferType::SCRATCH ?
                            buffInfo_.scratchBuffBaseOff + repeatTimes * tempAlgParams.inputRepeatStride
                                + algRank * tempAlgParams.inputSliceStride :
                            repeatTimes * tempAlgParams.inputRepeatStride + algRank * tempAlgParams.inputSliceStride
                                + buffInfo_.inBuffBaseOff;
        u64 dstOffset = isZeroCopy_ ? buffInfo_.outBuffBaseOff + repeatTimes * tempAlgParams.outputRepeatStride :
                                      buffInfo_.scratchBuffBaseOff + repeatTimes * tempAlgParams.outputRepeatStride;
        BufferType dstBuffType = isZeroCopy_ ? BufferType::OUTPUT : BufferType::SCRATCH;
        DataSlice srcSlice(buffInfo_.inBuffType, srcOffset, sliceSize);
        DataSlice dstSlice(dstBuffType, dstOffset, sliceSize);
        for (u32 j = 0; j < linkNum; j++) {
            CHK_PRT_RET(
                count >= tempInsQues.size(),
                HCCL_ERROR("[InsTempScatterMesh1D] count=%u, tempInsQues.size=%zu", count, tempInsQues.size()),
                HcclResult::HCCL_E_INTERNAL);
            LinkData& neighborLinkData = neighborLinkDatas[j];

            vector<DataSlice> txSrcSlices{CalcDataSliceForLinks(srcSlice, dataSplitRate, j, dataType_)};
            vector<DataSlice> txDstSlices{CalcDataSliceForLinks(dstSlice, dataSplitRate, j, dataType_)};

            SlicesList txSlicesList({txSrcSlices}, {txDstSlices});
            DataInfo sendData(neighborLinkData, txSlicesList);
            CHK_PRT_RET(
                Send(sendData, tempInsQues[++count], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempScatterMesh1D] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::RunMeshRx(
    u32 myAlgRank, u32 repeatTimes, const TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    u64 srcOffset = buffInfo_.inBuffType == BufferType::SCRATCH ?
                        buffInfo_.scratchBuffBaseOff + repeatTimes * tempAlgParams.inputRepeatStride
                            + myAlgRank * tempAlgParams.inputSliceStride :
                        repeatTimes * tempAlgParams.inputRepeatStride + myAlgRank * tempAlgParams.inputSliceStride
                            + buffInfo_.inBuffBaseOff;
    u64 dstOffset = isZeroCopy_ ? buffInfo_.outBuffBaseOff + repeatTimes * tempAlgParams.outputRepeatStride :
                                  buffInfo_.scratchBuffBaseOff + repeatTimes * tempAlgParams.outputRepeatStride;
    BufferType dstBuffType = isZeroCopy_ ? BufferType::OUTPUT : BufferType::SCRATCH;

    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 tailSize = tempAlgParams.tailSize == 0 ? tempAlgParams.sliceSize : tempAlgParams.tailSize;
    // 支持不均匀切分的情况下需要把尾部数据放到最后一张卡上
    u64 sliceSize = myAlgRank == tempVTopo_[0].size() - 1 ? tailSize : tempAlgParams.sliceSize;

    DataSlice srcSlice(buffInfo_.inBuffType, srcOffset, sliceSize);
    DataSlice dstSlice(dstBuffType, dstOffset, sliceSize);

    u32 currQueIdx = 0;
    for (currQueIdx = 1; currQueIdx < tempVTopo_[0].size(); currQueIdx++) {
        if ((myAlgRank + currQueIdx) % tempVTopo_[0].size() == root_ % tempVTopo_[0].size()) {
            break;
        }
    }
    std::vector<LinkData>& neighborLinkDatas = tempResLinks.at(root_);
    u32 linkNum = rank2PathNumMap_.at(root_);
    if (linkNum != neighborLinkDatas.size()) {
        HCCL_ERROR("InsTempScatterMesh1D::RunMeshTx linkNum != neighborLinkDatas.size()");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(neighborLinkDatas, dataSplitRate));
    for (u32 j = 0; j < linkNum; j++) {
        const LinkData& linkRecv = tempResLinks.at(root_)[j];
        vector<DataSlice> rxSrcSlices{CalcDataSliceForLinks(srcSlice, dataSplitRate, j, dataType_)};
        vector<DataSlice> rxDstSlices{CalcDataSliceForLinks(dstSlice, dataSplitRate, j, dataType_)};
        SlicesList rxSlicesList({rxSrcSlices}, {rxDstSlices});
        DataInfo recvData(linkRecv, rxSlicesList);
        CHK_PRT_RET(
            Recv(recvData, tempInsQues[currQueIdx++], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempScatterMesh1D] RunMeshRx failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::RunMesh(
    TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, std::vector<InsQuePtr>& tempInsQues)
{
    u32 myAlgRank;
    GetAlgRank(myRank_, tempVTopo_[0], myAlgRank);
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        if (root_ == u32(myRank_)) {
            CHK_PRT_RET(
                RunMeshTx(myAlgRank, r, tempAlgParams, tempResLinks, tempInsQues),
                HCCL_ERROR("[InsTempScatterMesh1D] RunMeshTx failed"), HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                RunMeshRx(myAlgRank, r, tempAlgParams, tempResLinks, tempInsQues),
                HCCL_ERROR("[InsTempScatterMesh1D] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::PreCopy(TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    if (u32(myRank_) != root_) {
        return HCCL_SUCCESS;
    }
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 sliceSize = tempAlgParams.tailSize == 0 ? tempAlgParams.sliceSize : tempAlgParams.tailSize;
    u32 myAlgRank;
    GetAlgRank(myRank_, tempVTopo_[0], myAlgRank);
    // 零拷贝而且不是最后一张卡的情况需要拷贝sliceSize
    if (isZeroCopy_ && myAlgRank != tempVTopo_[0].size() - 1) {
        sliceSize = tempAlgParams.sliceSize;
    }
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        u64 srcOffset
            = buffInfo_.inBuffType == BufferType::SCRATCH ? buffInfo_.scratchBuffBaseOff : buffInfo_.inBuffBaseOff;
        srcOffset += r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * myAlgRank;
        BufferType dstBufferType
            = buffInfo_.outBuffType == BufferType::INPUT ? buffInfo_.scratBuffType : buffInfo_.outBuffType;
        u64 dstOffset = buffInfo_.outBuffType == BufferType::SCRATCH || buffInfo_.outBuffType == BufferType::INPUT ?
                            r * tempAlgParams.outputRepeatStride + buffInfo_.scratchBuffBaseOff :
                            r * tempAlgParams.outputRepeatStride + buffInfo_.outBuffBaseOff;
        DataSlice srcSlice(buffInfo_.inBuffType, srcOffset, sliceSize);
        DataSlice dstSlice(dstBufferType, dstOffset, sliceSize);
        LocalCopy(tempInsQues[0], srcSlice, dstSlice);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // 零拷贝或者输出地址是SCRATCH场景在PreCopy阶段就已经拷贝完了
    if (isZeroCopy_ || buffInfo_.outBuffType == BufferType::SCRATCH) {
        return HCCL_SUCCESS;
    }
    if (buffInfo_.outBuffType == BufferType::OUTPUT && (static_cast<u32>(myRank_) == root_)) {
        return HCCL_SUCCESS;
    }
    u32 myAlgRank;
    GetAlgRank(myRank_, tempVTopo_[0], myAlgRank);
    // root卡需要将发送的数据刷新为尾部数据长度,保护一下tailSize=0认为没有
    u64 tailSize = tempAlgParams.tailSize == 0 ? tempAlgParams.sliceSize : tempAlgParams.tailSize;
    // 支持不均匀切分的情况下需要把尾部数据放到最后一张卡上
    u64 sliceSize = myAlgRank == tempVTopo_[0].size() - 1 ? tailSize : tempAlgParams.sliceSize;

    DataSlice dstSlice(buffInfo_.outBuffType, buffInfo_.outBuffBaseOff, sliceSize * tempAlgParams.repeatNum);
    DataSlice srcSlice(buffInfo_.scratBuffType, buffInfo_.scratchBuffBaseOff, sliceSize * tempAlgParams.repeatNum);
    if (buffInfo_.outBuffType == buffInfo_.scratBuffType && buffInfo_.outBuffBaseOff == buffInfo_.scratchBuffBaseOff) {
        return HCCL_SUCCESS;
    }
    LocalCopy(tempInsQues[0], srcSlice, dstSlice);

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
