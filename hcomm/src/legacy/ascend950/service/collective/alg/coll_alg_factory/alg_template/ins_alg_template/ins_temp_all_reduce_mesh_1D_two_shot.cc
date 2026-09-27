/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_reduce_mesh_1D_two_shot.h"

#include "log.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
InsTempAllReduceMesh1DTwoShot::InsTempAllReduceMesh1DTwoShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAllReduceMesh1DTwoShot::~InsTempAllReduceMesh1DTwoShot() {}

/*
 * Desc: 计算资源需求
 * return: tempResReq: 资源计算结果存储，包括notify信息，links信息等
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh1DTwoShot::CalcRes(AlgTempResReq& tempResReq)
{
    CHK_PRT_RET(
        CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [InsTempAllReduceMesh1DTwoShot] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    auto& linkReq = tempResReq.links;
    u32 pathNum = 0;
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        auto remoteRank = resReqIter->first;
        if (rank2PathNumMap_.find(remoteRank) == rank2PathNumMap_.end() || rank2PathNumMap_[remoteRank] == 0) {
            HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot] No path to remoteRank[%d]", remoteRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (pathNum == 0) {
            pathNum = rank2PathNumMap_[remoteRank];
        } else if (rank2PathNumMap_[remoteRank] != pathNum) {
            HCCL_ERROR(
                "[InsTempAllReduceMesh1DTwoShot] Inconsistency pathNum to remoteRanks, Previous consistent "
                "pathNum=[%u], mismatched "
                "remoteRank=[%d], pathNum=[%u]",
                pathNum, remoteRank, rank2PathNumMap_[remoteRank]);
            return HcclResult::HCCL_E_INTERNAL;
        }
        resReqIter->second = pathNum;
    }

    // 1D Mesh 需要的 que Num 为 ranksize
    tempResReq.queNum = tempVTopo_[0].size() * pathNum;
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShot] tempResReq.queNum = %u", tempResReq.queNum);
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chuck 块，给后续的allreduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh1DTwoShot::CalcSlice(const u64 dataSize, const u64 baseOff, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    u64 unitAlignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff + baseOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != baseOff + dataSize),
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
u32 InsTempAllReduceMesh1DTwoShot::CalcScratchMultiple(BufferType input, BufferType output) const
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
HcclResult InsTempAllReduceMesh1DTwoShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShot] start.");
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;

    uint32_t linkNum = tempLinks.begin()->second.size();
    CHK_PRT_RET(
        tempInsQues.size() != tempVTopo_[0].size() * linkNum,
        HCCL_ERROR(
            "[InsTempAllReduceMesh1DTwoShot] RankSize [%lu], linkNum_:[%u], tempInsQues size:[%zu],requiredQue Error.",
            tempVTopo_[0].size(), linkNum, tempInsQues.size()),
        HcclResult::HCCL_E_INTERNAL);

    u64 dataSizePerVolume = DataTypeSizeGet(dataType_);
    CHK_PRT_RET(
        (tempRankSize_ * dataSizePerVolume) + tempAlgParams.sliceSize > tempAlgParams.buffInfo.scratchBuffSize,
        HCCL_ERROR(
            "[InsTempAllReduceMesh1DTwoShot]Rank [%d], Input size:[%llu], BfSize:[%llu] Insufficient buffer!", myRank_,
            tempAlgParams.sliceSize, tempAlgParams.buffInfo.scratchBuffSize),
        HcclResult::HCCL_E_INTERNAL);

    u64 inBuffSize = tempAlgParams.sliceSize;
    std::vector<RankSliceInfo> sliceInfoVecForAllLinks(linkNum);
    std::vector<float> dataSplitRate(linkNum);
    CHK_RET(CalcDataSplitRateForLinks(tempLinks.begin()->second, dataSplitRate));
    u64 typeSize = DataTypeSizeGet(dataType_);
    u64 dataCnt = inBuffSize / typeSize;
    std::vector<u64> inBuffSizeForLinks(linkNum);
    std::vector<u64> baseOff(linkNum);
    u64 offset = 0;
    for (u32 linkIdx = 0; linkIdx < linkNum; linkIdx++) {
        if (linkIdx != linkNum - 1) {
            inBuffSizeForLinks[linkIdx]
                = static_cast<u64>(static_cast<float>(dataCnt) * dataSplitRate[linkIdx]) * typeSize;
        } else {
            inBuffSizeForLinks[linkIdx] = inBuffSize - offset;
        }
        // 对每块输入都CalcSlice
        CHK_RET(CalcSlice(inBuffSizeForLinks[linkIdx], offset, sliceInfoVecForAllLinks[linkIdx]));
        offset += inBuffSizeForLinks[linkIdx];
    }

    std::vector<std::vector<InsQuePtr>> tempInsQuesVec;
    u32 queNumPerlink = tempInsQues.size() / linkNum;
    HCCL_INFO("tempInsQues.size()=%zu,queNumPerlink=%u", tempInsQues.size(), queNumPerlink);
    for (uint32_t linkIdx = 0; linkIdx < linkNum; linkIdx++) {
        tempInsQuesVec.emplace_back(
            tempInsQues.begin() + linkIdx * queNumPerlink, tempInsQues.begin() + (linkIdx + 1) * queNumPerlink);
    }
    u32 mainQueIdx = 0;
    CHK_RET(PreSyncQues(tempInsQues, mainQueIdx));
    HCCL_INFO("[InsTempAllReduce1DMeshTwoShot][PreCopy] Rank [%d].", myRank_);
    for (uint32_t linkIdx = 0; linkIdx < linkNum; linkIdx++) {
        CHK_RET(RunAllReduceScatter(
            sliceInfoVecForAllLinks[linkIdx], tempLinks, tempInsQuesVec[linkIdx], tempAlgParams, linkIdx));
        CHK_RET(RunAllReduceAllgather(
            sliceInfoVecForAllLinks[linkIdx], tempLinks, tempInsQuesVec[linkIdx], tempAlgParams, linkIdx));
    }
    HCCL_INFO("[InsTempAllReduce1DMeshTwoShot][PostCopy] Rank [%d].", myRank_);
    // 流间后同步，从流通知主流
    CHK_RET(PostSyncQues(tempInsQues, mainQueIdx));

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
HcclResult InsTempAllReduceMesh1DTwoShot::RunAllReduceScatter(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    const TemplateDataParams& tempAlgParams, u32 linkIdx)
{
    u64 inBuffBaseOff = tempAlgParams.buffInfo.inBuffBaseOff;
    u32 MyAlgRank = tempVirtRankMap_[myRank_];
    // scatter
    for (u32 rankId = 0; rankId < tempRankSize_; rankId++) {
        DataSlice ssrcSlice = DataSlice(
            tempAlgParams.buffInfo.inBuffType, sliceInfoVec[rankId][0].offset + inBuffBaseOff,
            sliceInfoVec[rankId][0].size);
        DataSlice sdestSlice = DataSlice(
            tempAlgParams.buffInfo.scratBuffType,
            MyAlgRank * sliceInfoVec[rankId][0].size + tempAlgParams.buffInfo.scratchBuffBaseOff,
            sliceInfoVec[rankId][0].size);
        if (rankId == u32(MyAlgRank)) {
            if (sliceInfoVec[rankId][0].size != 0) {
                // 如果是本地rank，直接拷贝到scratch对应位置
                CHK_PRT_RET(
                    LocalCopy(tempInsQues[rankId], ssrcSlice, sdestSlice),
                    HCCL_ERROR(
                        "[InsTempAllReduceMesh1DTwoShot][RunAllReduceScatter] RunAllReduce scatter LocalCopy failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        } else {
            const std::vector<LinkData>& linkSendRecv = tempLinks.at(GetRankFromMap(rankId));
            // 发送, 未过滤size为0的情况
            std::vector<DataSlice> sendSrcSlices{ssrcSlice};
            std::vector<DataSlice> sendDestSlices{sdestSlice};

            // 接收，未过滤size为0的情况
            DataSlice rsrcSlice = DataSlice(
                tempAlgParams.buffInfo.inBuffType, sliceInfoVec[MyAlgRank][0].offset + inBuffBaseOff,
                sliceInfoVec[MyAlgRank][0].size);
            DataSlice rdestSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                rankId * sliceInfoVec[MyAlgRank][0].size + tempAlgParams.buffInfo.scratchBuffBaseOff,
                sliceInfoVec[MyAlgRank][0].size);
            std::vector<DataSlice> recvSrcSlices{rsrcSlice};
            std::vector<DataSlice> recvDestSlices{rdestSlice};

            TxRxLinks sendRecvLinks(linkSendRecv[linkIdx], linkSendRecv[linkIdx]);
            TxRxSlicesList sendRecvSlicesList({sendSrcSlices, sendDestSlices}, {recvSrcSlices, recvDestSlices});

            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, tempInsQues[rankId], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][RunAllReduceScatter] RunAllReduce scatter failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    // 从流同步,等待所有并发的send和copy完成
    PostSyncInterQueues(tempInsQues);

    // local reduce
    if (sliceInfoVec[MyAlgRank][0].size != 0) {
        DataSlice ldestSlice = DataSlice(
            tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff,
            sliceInfoVec[MyAlgRank][0].size);
        for (u32 rankId = 1; rankId < tempRankSize_; rankId++) {
            DataSlice lsrcSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType,
                rankId * sliceInfoVec[MyAlgRank][0].size + tempAlgParams.buffInfo.scratchBuffBaseOff,
                sliceInfoVec[MyAlgRank][0].size);
            // 所有reduce操作在同一个insque中才能保序；
            CHK_PRT_RET(
                LocalReduce(tempInsQues[0], lsrcSlice, ldestSlice, dataType_, redOp_),
                HCCL_ERROR(
                    "[InsTempAllReduceMesh1DTwoShot][RunAllReduceScatter] RunAllReduce reduce LocalReduce failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
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
HcclResult InsTempAllReduceMesh1DTwoShot::RunAllReduceAllgather(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    const TemplateDataParams& tempAlgParams, u32 linkIdx)
{
    u64 outBuffBaseOff = tempAlgParams.buffInfo.outBuffBaseOff;
    // sync:前同步
    PreSyncInterQueues(tempInsQues);
    u32 MyAlgRank = tempVirtRankMap_[myRank_];
    // allgather
    for (u32 rankId = 0; rankId < tempRankSize_; rankId++) {
        DataSlice rsrcSlice = DataSlice(
            tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff,
            sliceInfoVec[rankId][0].size);
        DataSlice rdestSlice = DataSlice(
            tempAlgParams.buffInfo.outBuffType, sliceInfoVec[rankId][0].offset + outBuffBaseOff,
            sliceInfoVec[rankId][0].size);
        if (u32(MyAlgRank) == rankId) {
            if (sliceInfoVec[rankId][0].size != 0) {
                // copy本端计算的结果到user output
                CHK_PRT_RET(
                    LocalCopy(tempInsQues[rankId], rsrcSlice, rdestSlice),
                    HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][RunAllReduceAllgather] RunAllReduce AllGather "
                               "LocalCopy failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        } else {
            const std::vector<LinkData>& linkSendRecv = tempLinks.at(GetRankFromMap(rankId));
            // 接收, 未过滤size为0的情况
            std::vector<DataSlice> recvSrcSlices{rsrcSlice};
            std::vector<DataSlice> recvDestSlices{rdestSlice};

            // 发送，未过滤size为0的情况
            DataSlice ssrcSlice = DataSlice(
                tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff,
                sliceInfoVec[MyAlgRank][0].size);
            DataSlice sdestSlice = DataSlice(
                tempAlgParams.buffInfo.outBuffType, sliceInfoVec[MyAlgRank][0].offset + outBuffBaseOff,
                sliceInfoVec[MyAlgRank][0].size);
            std::vector<DataSlice> sendSrcSlices{ssrcSlice};
            std::vector<DataSlice> sendDestSlices{sdestSlice};

            TxRxLinks sendRecvLinks(linkSendRecv[linkIdx], linkSendRecv[linkIdx]);
            TxRxSlicesList sendRecvSlicesList({sendSrcSlices, sendDestSlices}, {recvSrcSlices, recvDestSlices});

            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, tempInsQues[rankId], 0, true, DmaMode::GET),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][RunAllReduceAllgather] RunAllReduce AllGather failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempAllReduceMesh1DTwoShot::GetRankFromMap(const u32 rankIdx)
{
    RankId rank = -1;
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShot] GetRankFromMap");
    for (auto& pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}
} // namespace Hccl
