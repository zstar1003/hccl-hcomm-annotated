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
#include "ins_temp_all_gather_mesh.h"

namespace Hccl {
InsTempAllGatherMesh1D::InsTempAllGatherMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAllGatherMesh1D::~InsTempAllGatherMesh1D() {}

HcclResult InsTempAllGatherMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_DEBUG("[InsTempAllGatherMesh1D] Enter CalcRes");
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));

    auto& linkReq = tempResReq.links;
    u32 pathNum = 0;
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        auto remoteRank = resReqIter->first;
        if (rank2PathNumMap_.find(remoteRank) == rank2PathNumMap_.end() || rank2PathNumMap_[remoteRank] == 0) {
            HCCL_ERROR("[InsTempAllGatherMesh1D] No path to remoteRank[%d]", remoteRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (pathNum == 0) {
            pathNum = rank2PathNumMap_[remoteRank];
        } else if (rank2PathNumMap_[remoteRank] != pathNum) {
            HCCL_ERROR(
                "[InsTempAllGatherMesh1D] Inconsistency pathNum to remoteRanks, Previous consistent "
                "pathNum=[%u], mismatched "
                "remoteRank=[%d], pathNum=[%u]",
                pathNum, remoteRank, rank2PathNumMap_[remoteRank]);
            return HcclResult::HCCL_E_INTERNAL;
        }
        resReqIter->second = pathNum;
    }

    tempResReq.queNum = tempVTopo_[0].size() * pathNum;
    HCCL_INFO("[InsTempAllGatherMesh1D] tempResReq.queNum = %u", tempResReq.queNum);
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    HCCL_DEBUG("[InsTempAllGatherMesh1D] CalcRes queNotifys size[%zu]", tempResReq.queNotifys.size());

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    HCCL_DEBUG("[InsTempAllGatherMesh1D] CalcRes done");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempAllGatherMesh1D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(1);
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGather start");

    opMode_ = tempFuncs.opMode;
    tempAlgParams_ = tempAlgParams;
    tempLinks_ = tempLinks;

    uint32_t linkNum = tempLinks.begin()->second.size();
    // 流的数量不能少于linkNum
    CHK_PRT_RET(
        linkNum > tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempAllGatherMesh] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(tempLinks.begin()->second, dataSplitRate));
    queNumPerNeighbor_ = linkNum;
    std::vector<InsQuePtr> localInsQues;
    localInsQues.push_back(tempInsQues[0]);
    localInsQues.push_back(tempInsQues[tempInsQues.size() - 1]);

    CHK_RET(LocalCopyToScratch(tempInsQues[0]));
    // 如果输入和输出内存是同一片地址则必须先拷贝完成再运算Mesh，否则的话提升性能同步并行
    if (tempAlgParams.buffInfo.inBuffType == tempAlgParams.buffInfo.outBuffType) {
        // 因为后续RunMesh会写Output，如果inBuffType和outBuffType先同步后拷贝会导致改写
        CHK_RET(LocalCopyToUsrOut(tempInsQues[0]));
        CHK_RET(PreSyncInterQueues(tempInsQues));
    } else {
        CHK_RET(PreSyncInterQueues(tempInsQues));
        CHK_RET(LocalCopyToUsrOut(tempInsQues[0]));
    }

    // locate myRank in tempVTopo -> algRank
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    // run Mesh 使用第1至rankSize条queue
    CHK_PRT_RET(
        RunMesh(myAlgRank, tempVTopo_[0], tempInsQues) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[InsTempAllGatherMesh1D] Rank [%d], unable to run mesh algorithm.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    // semaphore sync
    CHK_RET(PostSyncInterQueues(tempInsQues));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::LocalCopyToUsrOut(InsQuePtr tempInsQue)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    // 做个保护，tailSize填写为0就认为尾块是正常块
    u64 tailSize = (tempAlgParams_.tailSize == 0) ? tempAlgParams_.sliceSize : tempAlgParams_.tailSize;
    u64 sliceSize = (myAlgRank == tempRankSize_ - 1) ? tailSize : tempAlgParams_.sliceSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;

        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
        if (tempAlgParams_.buffInfo.inBuffType == tempAlgParams_.buffInfo.outBuffType && inOff == outOff) {
            continue;
        }
        // OPBASE模式下scratch中已有备份数据，从scratch读取避免INPUT src/dst重叠导致数据损坏
        if (opMode_ == OpMode::OPBASE && tempAlgParams_.buffInfo.inBuffType == tempAlgParams_.buffInfo.outBuffType) {
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * (tempRankSize_ - 1) + tailSize;
            const u64 scratchOff = tempAlgParams_.buffInfo.scratchBuffBaseOff + rpt * scratchRepeatStride
                                   + tempAlgParams_.sliceSize * myAlgRank;
            DataSlice src(tempAlgParams_.buffInfo.scratBuffType, scratchOff, sliceSize);
            DataSlice dst(tempAlgParams_.buffInfo.outBuffType, outOff, sliceSize);
            HCCL_INFO("[InsTempAllGatherMesh1D] scratch:%s -> out:%s", src.Describe().c_str(), dst.Describe().c_str());
            auto ins = std::make_unique<InsLocalCopy>(src, dst);
            tempInsQue->Append(std::move(ins));
        } else {
            DataSlice src(tempAlgParams_.buffInfo.inBuffType, inOff, sliceSize);
            DataSlice dst(tempAlgParams_.buffInfo.outBuffType, outOff, sliceSize);
            HCCL_INFO("[InsTempAllGatherMesh1D] in:%s -> out:%s", src.Describe().c_str(), dst.Describe().c_str());
            auto ins = std::make_unique<InsLocalCopy>(src, dst);
            tempInsQue->Append(std::move(ins));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::LocalCopyToScratch(InsQuePtr tempInsQue)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    // 做个保护，tailSize填写为0就认为尾块是正常块
    u64 tailSize = (tempAlgParams_.tailSize == 0) ? tempAlgParams_.sliceSize : tempAlgParams_.tailSize;
    u64 sliceSize = (myAlgRank == tempRankSize_ - 1) ? tailSize : tempAlgParams_.sliceSize;

    if (opMode_ == OpMode::OPBASE) {
        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * (tempRankSize_ - 1) + tailSize;
            const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
            const u64 outBaseOff = tempAlgParams_.buffInfo.scratchBuffBaseOff + rpt * scratchRepeatStride;
            const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
            const u64 outOff = tempAlgParams_.sliceSize * myAlgRank + outBaseOff;

            DataSlice src(tempAlgParams_.buffInfo.inBuffType, inOff, sliceSize);
            DataSlice dst(tempAlgParams_.buffInfo.scratBuffType, outOff, sliceSize);
            HCCL_INFO("[InsTempAllGatherMesh1D] in:%s -> scratch:%s", src.Describe().c_str(), dst.Describe().c_str());

            auto ins = std::make_unique<InsLocalCopy>(src, dst);
            tempInsQue->Append(std::move(ins));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1D::SingleRunMesh(
    const u32 myAlgRank, u32 connectedRank, std::vector<InsQuePtr>& tempInsQues, u32& queIdx)
{
    u32 connectedAlgRank = 0;
    CHK_RET(GetAlgRank(connectedRank, tempVTopo_[0], connectedAlgRank));
    // 做个保护，tailSize填写为0就认为尾块是正常块
    tempAlgParams_.tailSize = (tempAlgParams_.tailSize == 0) ? tempAlgParams_.sliceSize : tempAlgParams_.tailSize;
    std::vector<LinkData>& neighborLinkDatas = tempLinks_.at(connectedRank);
    u32 linkNum = rank2PathNumMap_.at(connectedRank);
    if (linkNum != neighborLinkDatas.size()) {
        HCCL_ERROR("InsTempAllGatherMesh1D::RunMesh linkNum != neighborLinkDatas.size()");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(neighborLinkDatas, dataSplitRate));
    for (u32 j = 0; j < linkNum; j++) {
        CHK_PRT_RET(
            queIdx >= tempInsQues.size(),
            HCCL_ERROR("[SingleRunMesh] queIdx=%u, tempInsQues.size=%zu", queIdx, tempInsQues.size()),
            HcclResult::HCCL_E_INTERNAL);
        LinkData& neighborLinkData = neighborLinkDatas[j];
        std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;
        PrepareLinkSlices(
            myAlgRank, connectedAlgRank, dataSplitRate, j, txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxLinks sendRecvLinks(neighborLinkData, neighborLinkData);
        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        InsQuePtr curInsQues = tempInsQues.at(++queIdx);
        CHK_PRT_RET(
            SendRecv(sendRecvInfo, curInsQues, 0, true, DmaMode::GET),
            HCCL_ERROR("[InsTempAllGatherMesh1D] sendrecv failed (connectedRank=%d)", connectedRank),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::RunMesh(
    const u32 myAlgRank, const std::vector<RankId>& vTopo, std::vector<InsQuePtr>& tempInsQues)
{
    u32 queIdx = 0;
    for (u32 rankIdx = 0; rankIdx < vTopo.size() - 1; rankIdx++) {
        RankId connectedRank = vTopo[(myAlgRank + 1 + rankIdx) % vTopo.size()];
        auto it = tempLinks_.find(connectedRank);
        if (it == tempLinks_.end()) {
            HCCL_ERROR("[InsTempAllGatherMesh1D] connectedRank does not exist");
            return HcclResult::HCCL_E_PARA;
        }
        CHK_PRT_RET(
            tempLinks_.at(connectedRank).empty(),
            HCCL_ERROR(
                "[InsTempAllGatherMesh1D] connectedRank=%d, tempLinks_.size=%zu", connectedRank, tempLinks_.size()),
            HcclResult::HCCL_E_INTERNAL);
        // SingleRunMesh函数中会自动移动curInsQues队列下标
        CHK_PRT_RET(
            SingleRunMesh(myAlgRank, connectedRank, tempInsQues, queIdx),
            HCCL_ERROR("[SingleRunMesh] myAlgRank=%u, connectedRank=%d", myAlgRank, connectedRank),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1D::PrepareLinkSlices(
    const u32 myAlgRank, const u32 connectedAlgRank, const std::vector<float>& dataSplitRate, const u32 linkIndex,
    std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices, std::vector<DataSlice>& rxSrcSlices,
    std::vector<DataSlice>& rxDstSlices)
{
    BufferType writeType
        = (opMode_ == OpMode::OPBASE) ? tempAlgParams_.buffInfo.scratBuffType : tempAlgParams_.buffInfo.inBuffType;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * (tempRankSize_ - 1) + tempAlgParams_.tailSize;
        const u64 scratchBase = tempAlgParams_.buffInfo.scratchBuffBaseOff + rpt * scratchRepeatStride;
        u64 txInOffset = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
        u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;
        u64 txDstOffset = (opMode_ == OpMode::OPBASE) ? txScratchOffset : txOutOffset;
        u64 rxInOffset = tempAlgParams_.inputSliceStride * connectedAlgRank + inBaseOff;
        u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
        u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;
        u64 rxSrcOffset = (opMode_ == OpMode::OPBASE) ? rxScratchOffset : rxInOffset;
        u64 sendSliceSize = (myAlgRank == tempRankSize_ - 1) ? tempAlgParams_.tailSize : tempAlgParams_.sliceSize;
        u64 recvSliceSize
            = (connectedAlgRank == tempRankSize_ - 1) ? tempAlgParams_.tailSize : tempAlgParams_.sliceSize;
        DataSlice txSrcSlice(tempAlgParams_.buffInfo.inBuffType, txInOffset, sendSliceSize);
        DataSlice txDstSlice(writeType, txDstOffset, sendSliceSize);
        DataSlice rxSrcSlice(writeType, rxSrcOffset, recvSliceSize);
        DataSlice rxDstSlice(tempAlgParams_.buffInfo.outBuffType, rxOutOffset, recvSliceSize);
        txSrcSlices.push_back(CalcDataSliceForLinks(txSrcSlice, dataSplitRate, linkIndex, dataType_));
        txDstSlices.push_back(CalcDataSliceForLinks(txDstSlice, dataSplitRate, linkIndex, dataType_));
        rxSrcSlices.push_back(CalcDataSliceForLinks(rxSrcSlice, dataSplitRate, linkIndex, dataType_));
        rxDstSlices.push_back(CalcDataSliceForLinks(rxDstSlice, dataSplitRate, linkIndex, dataType_));
    }
    return;
}

} // namespace Hccl
