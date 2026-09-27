/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_meshchunk.h"
#include "log.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
InsTempReduceScatterMesh1DMeshChunk::InsTempReduceScatterMesh1DMeshChunk(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceScatterMesh1DMeshChunk::~InsTempReduceScatterMesh1DMeshChunk() {}

HcclResult InsTempReduceScatterMesh1DMeshChunk::CalcRes(AlgTempResReq& tempResReq)
{
    // Mesh 需要的 que Num 为 tempVTopo_[0].size()-1
    tempResReq.queNum = (tempVTopo_[0].size() > 1) ? tempVTopo_[0].size() - 1 : 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    // linkNumBtwPeers_这个在没有绕路的情况下，是设置成1
    CHK_PRT_RET(
        CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlgFactory] [InsTempReduceScatterMesh1DMeshChunk] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DMeshChunk::CalcScratchMultiple(
    const BufferType& inBuffType, const BufferType& outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = tempRankSize_ - 1;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::CalcSliceInfoVec(const u64& dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        SliceInfo slice = {accumOff, dataSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += dataSize;
    }
    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size
         != dataSize * tempRankSize_),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank_), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    queNum_ = tempVTopo_[0].size() - 1;
    processSize_ = tempAlgParams.sliceSize;
    rankIdx_ = tempVirtRankMap_[myRank_];
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSliceInfoVec(tempAlgParams.sliceSize, sliceInfoVec));
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk] Run Start");
    // 这里不支持绕路的时候，应该就用原始的tempInsQues就行
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh1DMeshChunk] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    PreCopy(tempAlgParams, tempInsQues);
    if (queNum_ > 1) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }

    CHK_RET(RunReduceScatter(tempLinks, tempInsQues, tempAlgParams, sliceInfoVec));

    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
    }
    PostCopy(tempAlgParams, tempInsQues);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::PreCopy(
    const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const
{
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][PreCopy], copy from userIn to scratch");
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        DataSlice srcSlice = DataSlice(
            tempAlgParams.buffInfo.inBuffType,
            tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                + rankIdx_ * tempAlgParams.inputSliceStride,
            processSize_);
        DataSlice dstSlice
            = DataSlice(tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff, processSize_);
        CHK_RET(LocalCopy(tempInsQues[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::RunReduceScatter(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams,
    RankSliceInfo& sliceInfoVec)
{
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][RunReduceScatter] myRank[%d]", myRank_);
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    uint64_t sliceNum = tempRankSize_ - 1;
    uint64_t mySliceSize = sliceInfoVec[myAlgRank][0].size; // 获取本rank需要处理的数据量
    uint64_t mySliceCount = mySliceSize / DataTypeSizeGet(op_.dataType);
    // 数据切分为sliceNum块，当数据量不能均匀切分时，后面smallDataSliceNum个数据块比前面bigDataSliceNum个数据块每块少1个数据
    uint64_t bigDataSliceNum = mySliceCount % sliceNum;
    uint64_t bigDataSliceSize = (mySliceCount / sliceNum + 1) * DataTypeSizeGet(op_.dataType);
    uint64_t smallDataSliceNum = sliceNum - mySliceCount % sliceNum;
    uint64_t smallDataSliceSize = mySliceCount / sliceNum * DataTypeSizeGet(op_.dataType);

    std::vector<uint64_t> sliceSize;
    for (uint64_t i = 0; i < bigDataSliceNum; i++) {
        sliceSize.push_back(bigDataSliceSize);
    }
    for (uint64_t i = 0; i < smallDataSliceNum; i++) {
        sliceSize.push_back(smallDataSliceSize);
    }
    uint64_t sliceRecvBaseOffset = 0;
    uint16_t rankNum = 2;
    for (uint16_t i = 0; i < (tempRankSize_ - rankNum); i++) {
        sliceRecvBaseOffset += sliceSize[i];
    }
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        uint64_t sliceSendOffset_;
        uint64_t sliceRecvOffset_;
        DoMeshChunk(
            tempLinks, tempInsQues, tempAlgParams, sliceSize, repeatIdx, myAlgRank, sliceSendOffset_, sliceRecvOffset_,
            sliceRecvBaseOffset);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::DoMeshChunk(
    const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams,
    const std::vector<uint64_t>& sliceSize, const u32& repeatIdx, const u32& myAlgRank, uint64_t& sliceSendOffset_,
    uint64_t& sliceRecvOffset_, const uint64_t& sliceRecvBaseOffset)
{
    for (uint16_t stepIdx = 0; stepIdx < (tempRankSize_ - 1); stepIdx++) {
        sliceSendOffset_ = 0;
        sliceRecvOffset_ = sliceRecvBaseOffset;
        uint16_t rankNum = 2;
        uint16_t tempNum = 3;
        for (uint16_t i = 0; i < (tempRankSize_ - 1); i++) {
            uint16_t nextNum = stepIdx + i + 1;
            if (nextNum >= tempRankSize_) {
                nextNum += 1;
            }
            uint16_t nextRank = (myAlgRank + nextNum) % tempRankSize_;
            uint16_t frontNum = 2 * myAlgRank - nextRank + tempRankSize_;
            uint16_t frontRank = frontNum % tempRankSize_;
            RankId toRank = tempVTopo_[0][frontRank];
            uint16_t queIdx;
            if (frontRank < myAlgRank) {
                queIdx = frontRank;
            } else {
                queIdx = frontRank - 1;
            }
            DataSlice rxSrcSlice = DataSlice(
                tempAlgParams.buffInfo.inBuffType,
                tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                    + myAlgRank * tempAlgParams.inputSliceStride + sliceRecvOffset_,
                sliceSize[i]); // 接收源
            DataSlice rxDstSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff + sliceRecvOffset_,
                sliceSize[i]); // 接收目标
            DataSlice txSrcSlice = DataSlice(
                tempAlgParams.buffInfo.inBuffType,
                tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                    + frontRank * tempAlgParams.inputSliceStride + sliceSendOffset_,
                sliceSize[i]); // 发送源
            DataSlice txDstSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff + sliceSendOffset_,
                sliceSize[i]); // 发送目标

            u32 rankFromRank = GetRankFromMap(toRank);
            auto it = tempLinks.find(rankFromRank);
            if (it == tempLinks.end()) {
                HCCL_ERROR("rankFromRank [%u] not in tempLinks.", rankFromRank);
                return HcclResult::HCCL_E_PARA;
            }
            const std::vector<LinkData>& linkRecv = tempLinks.at(GetRankFromMap(toRank));
            const std::vector<LinkData>& linkSend = tempLinks.at(GetRankFromMap(toRank));
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);

            SendRecvReduceInfo sendRecvReduceInfo{
                {linkSend[0], linkRecv[0]},
                {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                dataType_,
                redOp_};

            CHK_PRT_RET(
                SendRecvReduce(sendRecvReduceInfo, tempInsQues[queIdx], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempReduceScatterMesh1DMeshChunk] RunReduceScatter SendRecvReduce failed"),
                HcclResult::HCCL_E_INTERNAL);

            sliceSendOffset_ += sliceSize[i];
            if (tempRankSize_ > rankNum && i < (tempRankSize_ - rankNum)) {
                sliceRecvOffset_ -= sliceSize[tempRankSize_ - tempNum - i];
            }
        }
        if (queNum_ > 1 && stepIdx < (tempRankSize_ - rankNum)) {
            CHK_RET(PostSyncInterQueues(tempInsQues));
            CHK_RET(PreSyncInterQueues(tempInsQues));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::PostCopy(
    const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues)
{
    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 scratch 拷贝到 userOut
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][PostCopy], copy from scratch to userOut");
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));
    // 先把本卡的数据从input搬运到output
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        DataSlice myRankSlice
            = DataSlice(tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff, processSize_);
        DataSlice outputSlice
            = DataSlice(tempAlgParams.buffInfo.outBuffType, tempAlgParams.buffInfo.outBuffBaseOff, processSize_);
        CHK_RET(LocalCopy(tempInsQues[0], myRankSlice, outputSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempReduceScatterMesh1DMeshChunk::GetRankFromMap(const u32 rankIdx)
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
