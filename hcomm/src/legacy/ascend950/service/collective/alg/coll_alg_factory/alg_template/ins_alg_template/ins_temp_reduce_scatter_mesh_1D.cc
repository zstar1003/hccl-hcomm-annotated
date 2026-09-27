/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "log.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
InsTempReduceScatterMesh1D::InsTempReduceScatterMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceScatterMesh1D::~InsTempReduceScatterMesh1D() {}

HcclResult InsTempReduceScatterMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    CHK_PRT_RET(
        CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh1D] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    auto& linkReq = tempResReq.links;
    u32 pathNum = 0;
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        auto remoteRank = resReqIter->first;
        if (rank2PathNumMap_.find(remoteRank) == rank2PathNumMap_.end() || rank2PathNumMap_[remoteRank] == 0) {
            HCCL_ERROR("[InsTempReduceScatterMesh1D] No path to remoteRank[%d]", remoteRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (pathNum == 0) {
            pathNum = rank2PathNumMap_[remoteRank];
        } else if (rank2PathNumMap_[remoteRank] != pathNum) {
            HCCL_ERROR(
                "[InsTempReduceScatterMesh1D] Inconsistency pathNum to remoteRanks, Previous consistent pathNum=[%u], "
                "mismatched "
                "remoteRank=[%d], pathNum=[%u]",
                pathNum, remoteRank, rank2PathNumMap_[remoteRank]);
            return HcclResult::HCCL_E_INTERNAL;
        }
        resReqIter->second = pathNum;
    }

    // Mesh 需要的 que Num 为 tempVTopo_[0].size()-1
    tempResReq.queNum = (tempVTopo_[0].size() > 1) ? (tempVTopo_[0].size()) * pathNum : pathNum;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1D::CalcScratchMultiple(const BufferType& inBuffType, const BufferType& outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = tempRankSize_;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    queNum_ = tempInsQues.size();
    HCCL_INFO("[InsTempReduceScatterMesh1D] Run Start");
    uint32_t linkNum = tempLinks.begin()->second.size();
    CHK_PRT_RET(
        linkNum > tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh1D] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    if (queNum_ > 1) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }
    CHK_RET(RunReduceScatter(tempLinks, tempInsQues, tempAlgParams));
    HCCL_INFO("[InsTempReduceScatterMesh1D][PostCopy] Rank [%d].", myRank_);
    // 流间后同步，从流通知主流
    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }
    PostCopy(tempAlgParams, tempInsQues);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterMesh1D::PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // 通信结束之后，数据都在 inbuff 上，需要搬运到对应的输出位置。
    u32 rankIdx = tempVirtRankMap_[myRank_];
    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 inBuff 拷贝到 userOut
    // 是否需要将数据搬运到 OutBuff 上再搬运到 UserOut 上？？
    HCCL_INFO("[InsTempReduceScatterMesh1D][PostCopy], copy from outBuff to userOut");
    // 先把本卡的数据从input搬运到output
    HCCL_INFO("[InsTempReduceScatterMesh1D][PostCopy]tempAlgParams.repeatNum=%llu", tempAlgParams.repeatNum);
    u64 sliceSize = ((rankIdx == tempRankSize_ - 1) && (tempAlgParams.tailSize != 0)) ? tempAlgParams.tailSize :
                                                                                        tempAlgParams.sliceSize;
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        DataSlice myRankSlice = DataSlice(
            tempAlgParams.buffInfo.inBuffType,
            tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                + rankIdx * tempAlgParams.inputSliceStride,
            sliceSize);
        DataSlice outputSlice = DataSlice(
            tempAlgParams.buffInfo.outBuffType,
            tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, sliceSize);
        // myRankSlice与outputSlice一致就不进行LocalCopy
        if (!(tempAlgParams.buffInfo.inBuffType == tempAlgParams.buffInfo.outBuffType
              && myRankSlice.GetOffset() == outputSlice.GetOffset())) {
            CHK_RET(LocalCopy(tempInsQues[0], myRankSlice, outputSlice));
        }

        // 把其他卡的数据input累加到output
        for (u32 tmpRank = 0; tmpRank < tempRankSize_; tmpRank++) {
            if (tmpRank != rankIdx) {
                DataSlice srcDataSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                        + tmpRank * sliceSize,
                    sliceSize);
                DataSlice dstDataSlice = DataSlice(
                    tempAlgParams.buffInfo.outBuffType,
                    tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, sliceSize);
                CHK_RET(LocalReduce(tempInsQues[0], srcDataSlice, dstDataSlice, dataType_, redOp_));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1D::RunReduceScatter(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    // 控制mesh通信的rankSize - 1个对端
    u32 queIdx = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_ - 1; rankIdx++) {
        u32 nextRank = (myAlgRank + 1 + rankIdx) % tempRankSize_;
        RankId remoteRank = tempVTopo_[0][nextRank];
        u32 rmAlgRank;
        CHK_RET(GetAlgRank(remoteRank, tempVTopo_[0], rmAlgRank));
        HCCL_DEBUG(
            "[InsTempReduceScatterMesh1D][RunReduceScatter] myRank[%d], toRank[%d], fromRank[%d], rmAlgRank[%u]",
            myRank_, remoteRank, remoteRank, rmAlgRank);
        CHK_PRT_RET(
            tempLinks.at(remoteRank).empty(),
            HCCL_ERROR(
                "[InsTempReduceScatterMesh1D][RunReduceScatter] Rank [%d], remoteRank[%d] required links Error.",
                myRank_, remoteRank),
            HcclResult::HCCL_E_INTERNAL);
        const std::vector<LinkData>& neighborLinkDatas = tempLinks.at(remoteRank);
        u32 linkNum = rank2PathNumMap_.at(remoteRank);
        CHK_PRT_RET(
            linkNum != neighborLinkDatas.size(),
            HCCL_ERROR(
                "[InsTempReduceScatterMesh1D][RunReduceScatter] Rank [%d], remoteRank[%d] linkNum != "
                "neighborLinkDatas.size().",
                myRank_, remoteRank),
            HcclResult::HCCL_E_INTERNAL);
        std::vector<float> dataSplitRate(linkNum);
        CHK_RET(CalcDataSplitRateForLinks(neighborLinkDatas, dataSplitRate));
        for (u32 linkIdx = 0; linkIdx < linkNum; linkIdx++) {
            CHK_PRT_RET(
                queIdx >= tempInsQues.size(),
                HCCL_ERROR(
                    "[InsTempReduceScatterMesh1D][RunReduceScatter] queIdx [%u] >= tempInsQues.size() [%zu].", queIdx,
                    tempInsQues.size()),
                HcclResult::HCCL_E_INTERNAL);
            InsQuePtr currQue = tempInsQues[queIdx + 1];
            queIdx++;
            const LinkData& neighborLinkData = neighborLinkDatas[linkIdx];
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            u64 sendSlice = ((rmAlgRank == tempRankSize_ - 1) && (tempAlgParams.tailSize != 0)) ?
                                tempAlgParams.tailSize :
                                tempAlgParams.sliceSize;
            for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
                DataSlice rxSrcSlice = DataSlice(
                    tempAlgParams.buffInfo.inBuffType,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                        + myAlgRank * tempAlgParams.inputSliceStride,
                    tempAlgParams.sliceSize); // 接收源
                DataSlice rxDstSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                        + nextRank * tempAlgParams.sliceSize,
                    tempAlgParams.sliceSize); // 接收目标
                DataSlice txSrcSlice = DataSlice(
                    tempAlgParams.buffInfo.inBuffType,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                        + nextRank * tempAlgParams.inputSliceStride,
                    sendSlice); // 发送源
                DataSlice txDstSlice = DataSlice(
                    tempAlgParams.buffInfo.scratBuffType,
                    tempAlgParams.buffInfo.scratchBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                        + myAlgRank * sendSlice,
                    sendSlice); // 发送目标

                txSrcSlices.push_back(CalcDataSliceForLinks(txSrcSlice, dataSplitRate, linkIdx, dataType_));
                txDstSlices.push_back(CalcDataSliceForLinks(txDstSlice, dataSplitRate, linkIdx, dataType_));
                rxSrcSlices.push_back(CalcDataSliceForLinks(rxSrcSlice, dataSplitRate, linkIdx, dataType_));
                rxDstSlices.push_back(CalcDataSliceForLinks(rxDstSlice, dataSplitRate, linkIdx, dataType_));
            }
            SendRecvInfo sendRecvInfo{
                {neighborLinkData, neighborLinkData}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, currQue, 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempReduceScatterMesh1D] RunReduceScatter SendReduce failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempReduceScatterMesh1D::GetRankFromMap(const u32 rankIdx)
{
    RankId rank = -1;
    HCCL_INFO("[InsTempReduceScatterMesh1D] GetRankFromMap");
    for (auto& pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}

} // namespace Hccl
