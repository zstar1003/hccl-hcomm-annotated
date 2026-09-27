/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.h"

#include "log.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
InsTempAllReduceMesh1DTwoShotMeshChunk::InsTempAllReduceMesh1DTwoShotMeshChunk(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShotMeshChunk] Init.");
}

InsTempAllReduceMesh1DTwoShotMeshChunk::~InsTempAllReduceMesh1DTwoShotMeshChunk()
{
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShotMeshChunk] exit.");
}

/*
 * Desc: 计算资源需求
 * return: tempResReq: 资源计算结果存储，包括notify信息，links信息等
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh1DTwoShotMeshChunk::CalcRes(AlgTempResReq& tempResReq)
{
    // 1D Mesh 需要的 que Num 为 ranksize
    tempResReq.queNum = tempVTopo_[0].size();
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    CHK_PRT_RET(
        CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlgFactory] [InsTempAllReduceMesh1DTwoShotMeshChunk] Rank [%d], resLinks calculation error!",
            myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chunk 块，给后续的allreduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh1DTwoShotMeshChunk::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    u64 unitAlignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR(
            "[InsAllReduceCombExecutor] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 返回当前rank能处理的数据量和scratch buffer之间的比例关系
 * param: input: 输入数据位置
 * param: output 输出数据位置
 */
u32 InsTempAllReduceMesh1DTwoShotMeshChunk::CalcScratchMultiple(BufferType input, BufferType output) const
{
    (void)input;
    (void)output;
    u32 multiple = 2;
    return multiple;
}

/*
 * Desc: GenExtIns 算子执行入口
 * param: tempFuncs: 辅助信息包括userIn/OutSlices, opMode等标记信息
 * param: tempAlgParams: 每个rank的数据切片信息
 * param: tempLinks: 当前rank通信链接信息
 * param: tempInsQues: 通信队列
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh1DTwoShotMeshChunk::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShotMeshChunk] start.");

    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;

    queNum_ = tempVTopo_[0].size();
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR(
            "[InsTempAllReduceMesh1DTwoShotMeshChunk] Rank [%d], queNum_:[%u], tempInsQues size:[%zu],requiredQue "
            "Error.",
            myRank_, queNum_, tempInsQues.size()),
        HcclResult::HCCL_E_INTERNAL);

    u64 dataSizePerVolume = DataTypeSizeGet(dataType_);
    CHK_PRT_RET(
        (tempRankSize_ * dataSizePerVolume) + tempAlgParams.sliceSize > tempAlgParams.buffInfo.scratchBuffSize,
        HCCL_ERROR(
            "[InsTempAllReduceMesh1DTwoShotMeshChunk]Rank [%d], Input size:[%llu], BfSize:[%llu] Insufficient buffer!",
            myRank_, tempAlgParams.sliceSize, tempAlgParams.buffInfo.scratchBuffSize),
        HcclResult::HCCL_E_INTERNAL);

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(tempAlgParams.sliceSize, sliceInfoVec));

    HCCL_INFO("[InsTempAllReduce1DMeshTwoShot][PreCopy] Rank [%d].", myRank_);
    CHK_RET(PreCopy(tempAlgParams, sliceInfoVec, tempInsQues));
    CHK_RET(RunReduceScatter(sliceInfoVec, tempLinks, tempInsQues, tempAlgParams));
    CHK_RET(RunAllgather(sliceInfoVec, tempLinks, tempInsQues, tempAlgParams));
    HCCL_INFO("[InsTempAllReduce1DMeshTwoShot][PostCopy] Rank [%d].", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShotMeshChunk::PreCopy(
    const TemplateDataParams& tempAlgParams, const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShotMeshChunk][PreCopy], copy from userIn to scratch");
    u64 inBuffBaseOff = tempAlgParams.buffInfo.inBuffBaseOff;
    u32 myAlgRank = tempVirtRankMap_[myRank_];
    for (u32 rankId = 0; rankId < tempRankSize_; rankId++) {
        DataSlice localsrcSlice = DataSlice(
            tempAlgParams.buffInfo.inBuffType, sliceInfoVec[rankId][0].offset + inBuffBaseOff,
            sliceInfoVec[rankId][0].size);
        DataSlice loacldestSlice = DataSlice(
            tempAlgParams.buffInfo.scratBuffType,
            sliceInfoVec[rankId][0].offset + tempAlgParams.buffInfo.scratchBuffBaseOff, sliceInfoVec[rankId][0].size);

        if (rankId == u32(myAlgRank)) {
            // 本地rank对应一片直接拷贝到scratch对应位置
            CHK_PRT_RET(
                LocalCopy(tempInsQues[0], localsrcSlice, loacldestSlice),
                HCCL_ERROR(
                    "[InsTempAllReduceMesh1DTwoShotMeshChunk][RunReduceScatter] RunAllReduce scatter LocalCopy failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 1D Mesh twoshot AllReduce: Scatter+reduce
 * param: sliceInfoVec: 每个rank的数据切片信息
 * param: tempLinks: 当前rank通信链接信息
 * param: tempInsQues: 通信队列
 * param: tempFuncs: 辅助信息包括userIn/OutSlices, opMode等标记信息
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh1DTwoShotMeshChunk::RunReduceScatter(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    const TemplateDataParams& tempAlgParams)
{
    u32 myAlgRank = tempVirtRankMap_[myRank_];
    // 计算单个rank内一片数据再次分片成ranksize-1大小
    u64 sliceNum = tempRankSize_ - 1;
    vector<vector<u64>> sliceSize(tempRankSize_, vector<u64>(tempRankSize_ - 1));
    for (u32 rankId = 0; rankId < tempRankSize_; rankId++) {
        u64 rankIdSliceSize = sliceInfoVec[rankId][0].size;
        u64 rankIdSliceCount = rankIdSliceSize / DataTypeSizeGet(dataType_);
        // 数据切分为sliceNum块，当数据量不能均匀切分时，后面smallDataSliceNum个数据块比前面bigDataSliceNum个数据块每块少1个数据
        u64 bigDataSliceNum = rankIdSliceCount % sliceNum;
        u64 bigDataSliceSize = (rankIdSliceCount / sliceNum + 1) * DataTypeSizeGet(dataType_);
        u64 smallDataSliceNum = sliceNum - rankIdSliceCount % sliceNum;
        u64 smallDataSliceSize = rankIdSliceCount / sliceNum * DataTypeSizeGet(dataType_);
        for (uint64_t i = 0; i < bigDataSliceNum; i++) {
            sliceSize[rankId][i] = bigDataSliceSize;
        }
        for (uint64_t i = 0; i < smallDataSliceNum; i++) {
            sliceSize[rankId][i + bigDataSliceNum] = smallDataSliceSize;
        }
    }
    CHK_RET(PreSyncInterQueues(tempInsQues));
    for (u32 stepIndex = 0; stepIndex < (tempRankSize_ - 1); stepIndex++) {
        ReduceScatterMeshChunk(sliceInfoVec, tempLinks, tempInsQues, tempAlgParams, sliceSize, stepIndex, myAlgRank);
    }
    CHK_RET(PostSyncInterQueues(tempInsQues));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShotMeshChunk::ReduceScatterMeshChunk(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    const TemplateDataParams& tempAlgParams, const std::vector<vector<u64>>& sliceSize, const u32& stepIndex,
    const u32& myAlgRank)
{
    u64 inBuffBaseOff = tempAlgParams.buffInfo.inBuffBaseOff;
    u64 scratchBuffBaseOff = tempAlgParams.buffInfo.scratchBuffBaseOff;
    for (u32 chunkIndex = 0; chunkIndex < (tempRankSize_ - 1); chunkIndex++) {
        u64 sliceRxOffset_ = 0;
        u64 sliceTxOffset_ = 0;
        u32 nextNum = stepIndex + chunkIndex + 1;
        if (nextNum >= tempRankSize_) {
            nextNum += 1;
        }
        u32 nextRank = (myAlgRank + nextNum) % tempRankSize_;
        u32 preNum = 2 * myAlgRank + tempRankSize_ - nextRank;
        u32 preRank = preNum % tempRankSize_;
        RankId fromRank = tempVTopo_[0][nextRank];
        RankId toRank = tempVTopo_[0][preRank];
        u32 queIdx;
        for (u32 m = 0; m < chunkIndex; m++) {
            sliceRxOffset_ += sliceSize[fromRank][m];
            sliceTxOffset_ += sliceSize[toRank][m];
        }
        if (preRank < myAlgRank) {
            queIdx = preRank;
        } else {
            queIdx = preRank - 1;
        }
        DataSlice rxSrcSlice = DataSlice(
            tempAlgParams.buffInfo.inBuffType, inBuffBaseOff + sliceInfoVec[myAlgRank][0].offset + sliceRxOffset_,
            sliceSize[fromRank][chunkIndex]); // 接收源
        DataSlice rxDstSlice = DataSlice(
            tempAlgParams.buffInfo.scratBuffType,
            scratchBuffBaseOff + sliceInfoVec[myAlgRank][0].offset + sliceRxOffset_,
            sliceSize[fromRank][chunkIndex]); // 接收目标
        DataSlice txSrcSlice = DataSlice(
            tempAlgParams.buffInfo.inBuffType, inBuffBaseOff + sliceInfoVec[toRank][0].offset + sliceTxOffset_,
            sliceSize[toRank][chunkIndex]); // 发送源
        DataSlice txDstSlice = DataSlice(
            tempAlgParams.buffInfo.scratBuffType, scratchBuffBaseOff + sliceInfoVec[toRank][0].offset + sliceTxOffset_,
            sliceSize[toRank][chunkIndex]); // 发送目标

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
            {linkSend[0], linkRecv[0]}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_, redOp_};
        CHK_PRT_RET(
            SendRecvReduce(sendRecvReduceInfo, tempInsQues[queIdx], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceScatterMesh1DMeshChunk] RunReduceScatter SendRecvReduce failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    u32 rankNum = 2;
    if (stepIndex < (tempRankSize_ - rankNum)) {
        CHK_RET(PostSyncInterQueues(tempInsQues));
        CHK_RET(PreSyncInterQueues(tempInsQues));
    }
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 1D Mesh twoshot AllReduce: Allgather
 * param: sliceInfoVec: 每个rank的数据切片信息
 * param: tempLinks: 当前rank通信链接信息
 * param: tempInsQues: 通信队列
 * param: tempFuncs: 辅助信息包括userIn/OutSlices, opMode等标记信息
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh1DTwoShotMeshChunk::RunAllgather(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    const TemplateDataParams& tempAlgParams)
{
    u64 outBuffBaseOff = tempAlgParams.buffInfo.outBuffBaseOff;
    // sync:前同步
    PreSyncInterQueues(tempInsQues);
    u32 myAlgRank = tempVirtRankMap_[myRank_];
    // allgather
    for (u32 rankId = 0; rankId < tempRankSize_; rankId++) {
        DataSlice rsrcSlice = DataSlice(
            tempAlgParams.buffInfo.scratBuffType,
            tempAlgParams.buffInfo.scratchBuffBaseOff + sliceInfoVec[rankId][0].offset, sliceInfoVec[rankId][0].size);
        DataSlice rdestSlice = DataSlice(
            tempAlgParams.buffInfo.outBuffType, sliceInfoVec[rankId][0].offset + outBuffBaseOff,
            sliceInfoVec[rankId][0].size);
        if (u32(myAlgRank) == rankId) {
            if (sliceInfoVec[rankId][0].size != 0) {
                // copy本端计算的结果到user output
                CHK_PRT_RET(
                    LocalCopy(tempInsQues[rankId], rsrcSlice, rdestSlice),
                    HCCL_ERROR("[InsTempAllReduceMesh1DTwoShotMeshChunk][RunAllgather] RunAllReduce AllGather "
                               "LocalCopy failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        } else {
            u32 queIdx;
            if (rankId < myAlgRank) {
                queIdx = rankId;
            } else {
                queIdx = rankId - 1;
            }
            const std::vector<LinkData>& linkSendRecv = tempLinks.at(GetRankFromMap(rankId));
            // 接收, 未过滤size为0的情况
            std::vector<DataSlice> recvSrcSlices{rsrcSlice};
            std::vector<DataSlice> recvDestSlices{rdestSlice};

            // 发送，未过滤size为0的情况
            DataSlice ssrcSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                tempAlgParams.buffInfo.scratchBuffBaseOff + sliceInfoVec[myAlgRank][0].offset,
                sliceInfoVec[myAlgRank][0].size);
            DataSlice sdestSlice = DataSlice(
                tempAlgParams.buffInfo.outBuffType, sliceInfoVec[myAlgRank][0].offset + outBuffBaseOff,
                sliceInfoVec[myAlgRank][0].size);

            std::vector<DataSlice> sendSrcSlices{ssrcSlice};
            std::vector<DataSlice> sendDestSlices{sdestSlice};

            TxRxLinks sendRecvLinks(linkSendRecv[0], linkSendRecv[0]);
            TxRxSlicesList sendRecvSlicesList({sendSrcSlices, sendDestSlices}, {recvSrcSlices, recvDestSlices});

            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, tempInsQues[queIdx], 0, true, DmaMode::GET),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShotMeshChunk][RunAllgather] RunAllReduce AllGather failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    PostSyncInterQueues(tempInsQues);
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempAllReduceMesh1DTwoShotMeshChunk::GetRankFromMap(const u32 rankIdx)
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
