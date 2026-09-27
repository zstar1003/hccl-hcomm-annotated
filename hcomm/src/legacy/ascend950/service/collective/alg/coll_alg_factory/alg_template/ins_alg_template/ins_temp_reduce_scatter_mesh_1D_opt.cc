/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_opt.h"
#include "log.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
InsTempReduceScatterMesh1DOpt::InsTempReduceScatterMesh1DOpt(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceScatterMesh1DOpt::~InsTempReduceScatterMesh1DOpt() {}

HcclResult InsTempReduceScatterMesh1DOpt::CalcRes(AlgTempResReq& tempResReq)
{
    // Mesh 需要的 que Num 为 tempVTopo_[0].size()-1
    tempResReq.queNum = (tempVTopo_[0].size() > 1) ? tempVTopo_[0].size() : 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    // linkNumBtwPeers_这个在没有绕路的情况下，是设置成1
    CHK_PRT_RET(
        CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh1DOpt] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DOpt::CalcScratchMultiple(
    const BufferType& inBuffType, const BufferType& outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = tempRankSize_;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1DOpt::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    queNum_ = tempVTopo_[0].size();
    HCCL_INFO(
        "[InsTempReduceScatterMesh1DOpt] tempVTopo_.size() is [%zu], ReduceScatter queNum is [%llu]", tempVTopo_.size(),
        queNum_);
    processSize_ = tempAlgParams.sliceSize;
    HCCL_INFO("[InsTempReduceScatterMesh1DOpt] Run Start");
    // 这里不支持绕路的时候，应该就用原始的tempInsQues就行
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh1DOpt] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    if (queNum_ > 1) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }
    CHK_RET(RunReduceScatter(tempLinks, tempInsQues, tempAlgParams));
    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }
    PostCopy(tempAlgParams, tempInsQues);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterMesh1DOpt::PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // 通信结束之后，数据都在 inbuff 上，需要搬运到对应的输出位置。
    u32 rankIdx = tempVirtRankMap_[myRank_];
    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 inBuff 拷贝到 userOut
    // 是否需要将数据搬运到 OutBuff 上再搬运到 UserOut 上？？
    HCCL_INFO("[InsTempReduceScatterMesh1DOpt][PostCopy], copy from outBuff to userOut");
    // 先把本卡的数据从input搬运到output
    HCCL_INFO("[InsTempReduceScatterMesh1DOpt][PostCopy]tempAlgParams.repeatNum=%llu", tempAlgParams.repeatNum);
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
        CHK_RET(LocalCopy(tempInsQues[0], myRankSlice, outputSlice));
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

HcclResult InsTempReduceScatterMesh1DOpt::RunReduceScatter(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    for (u32 queIdx = 1; queIdx < queNum_; queIdx++) {
        u32 nextRank = (myAlgRank + queIdx) % tempRankSize_;
        RankId remoteRank = tempVTopo_[0][nextRank];

        u32 rmAlgRank;
        CHK_RET(GetAlgRank(remoteRank, tempVTopo_[0], rmAlgRank));

        HCCL_DEBUG(
            "[InsTempReduceScatterMesh1DOpt][RunReduceScatter] myRank[%d], toRank[%d], fromRank[%d]", myRank_,
            remoteRank, remoteRank);
        const std::vector<LinkData>& linkRecv = tempLinks.at(remoteRank);
        const std::vector<LinkData>& linkSend = tempLinks.at(remoteRank);
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        u64 sendSlice = ((rmAlgRank == tempRankSize_ - 1) && (tempAlgParams.tailSize != 0)) ? tempAlgParams.tailSize :
                                                                                              tempAlgParams.sliceSize;
        u64 recvSlice = ((myAlgRank == tempRankSize_ - 1) && (tempAlgParams.tailSize != 0)) ? tempAlgParams.tailSize :
                                                                                              tempAlgParams.sliceSize;

        // 在 inBuff 上进行 ReduceScatter 操作
        // 数据从其他卡，传输到本卡，接收数据
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
            DataSlice rxSrcSlice = DataSlice(
                tempAlgParams.buffInfo.inBuffType,
                tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                    + myAlgRank * tempAlgParams.inputSliceStride,
                recvSlice); // 接收源
            DataSlice rxDstSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                tempAlgParams.buffInfo.scratchBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                    + nextRank * recvSlice,
                recvSlice); // 接收目标
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

            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
        }
        SendRecvInfo sendRecvInfo{{linkSend[0], linkRecv[0]}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};

        CHK_PRT_RET(
            SendRecv(sendRecvInfo, tempInsQues[queIdx], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceScatterMesh1DOpt] RunReduceScatter SendReduce failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempReduceScatterMesh1DOpt::GetRankFromMap(const u32 rankIdx)
{
    RankId rank = -1;
    for (auto& pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}

} // namespace Hccl
