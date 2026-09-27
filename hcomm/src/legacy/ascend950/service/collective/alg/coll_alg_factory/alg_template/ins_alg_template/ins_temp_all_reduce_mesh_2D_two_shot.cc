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
#include "ins_temp_all_reduce_mesh_2D_two_shot.h"

namespace Hccl {
InsTempAllReduceMesh2DTwoShot::InsTempAllReduceMesh2DTwoShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    HCCL_INFO("[InsTempAllReduceMesh2DTwoShot] Init.");
}

InsTempAllReduceMesh2DTwoShot::~InsTempAllReduceMesh2DTwoShot() { HCCL_INFO("[InsTempAllReduceMesh2DTwoShot] exit."); }

/*
 * Desc: 计算资源需求
 * return: tempResReq: 资源计算结果存储，包括notify信息，links信息等
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh2DTwoShot::CalcRes(AlgTempResReq& tempResReq)
{
    // 1D Mesh 需要的 que Num 为 ranksize
    tempResReq.queNum = tempVTopo_[0].size() + tempVTopo_[1].size();
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateQueNotifiesRequest(tempResReq.queNum, 1, 0, tempVTopo_[0].size());

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "InsTempAllReduceMesh2DTwoShot::CalcRes Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim,
                neighborRank);
            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }
    HCCL_INFO("InsTempAllReduceMesh2DTwoShot::CalcRes done");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<std::tuple<QId, QId, u32>>
InsTempAllReduceMesh2DTwoShot::CreateQueNotifiesRequest(u32 queueNum, u32 pairNum, QId masterIdX, QId masterIdY) const
{
    std::vector<std::tuple<QId, QId, u32>> notifyRequests;
    HCCL_DEBUG("[Create][MasterSlaveQueNotifiesRequest] queueNum[%u], pairNum[%u]", queueNum, pairNum);
    if (queueNum == 0 || pairNum == 0) {
        HCCL_INFO("[Create][MasterSlaveQueNotifiesRequest] queueNum or pairNum is zero, "
                  "return empty notifyRequests");
        return notifyRequests;
    };

    u32 slaveNum = queueNum - 1;
    HCCL_INFO("[Create][MasterSlaveQueNotifiesRequest] slaveNum[%u]", slaveNum);
    if (slaveNum < 1 || pairNum < 1) {
        return notifyRequests;
    }

    notifyRequests.reserve((slaveNum + queueNum - masterIdY - 1) * pairNum);
    // masterX(master0)跟所有的stream有同步关系
    for (QId q = 0; q < queueNum; q++) {
        if (q == masterIdX) {
            continue;
        }
        for (u32 i = 0; i < pairNum; i++) {
            notifyRequests.emplace_back(std::make_tuple(masterIdX, q, i));
            notifyRequests.emplace_back(std::make_tuple(q, masterIdX, i));
        }
    }

    for (QId q = masterIdY + 1; q < queueNum; q++) {
        for (u32 i = 0; i < pairNum; i++) {
            notifyRequests.emplace_back(std::make_tuple(masterIdY, q, i));
            notifyRequests.emplace_back(std::make_tuple(q, masterIdY, i));
        }
    }
    return notifyRequests;
}

/*
 * Desc: 返回当前rank能处理的数据量和scratch buffer之间的比例关系
 * param: input: 输入数据位置
 * param: output 输出数据位置
 */
u32 InsTempAllReduceMesh2DTwoShot::CalcScratchMultiple(BufferType input, BufferType output) const
{
    // scratchbuffer如果能够通过ranksize规整：buffersize%2*ranksize_M*ranksize_N=0，则这里只需要返回1，最大化利用scratchbuffer

    // 否则返回2，使用1倍的buffer保证能缓存所有其他rank发来的数据，理论上数据被分成2*M*N块，假设有尾块，每个数据块的大小(inputcount/（2*M*N)+1）
    // 总共需要(inputCount/（2*M*N)+1）*（2*M*N）=[inputCount+2*M*N]*elembytesize,
    // 而预留的buffersize=inputcount*elembytesize，
    // 所以如果2*M*N>inputcount(预留)则缓存buffer仍然不够，但是由于最小的scratchbuffersize=1M，而2*M*N很难大于1M/2(一半数据一半缓存)，
    // 所以返回2的时候要判断(2*M*N+inputcount)*elembytesize>scratchbufferSize(即预留缓存buffer+输入数据占用的buffer)；2*M*N是常量，只需要增加buffersize解决
    (void)input;
    (void)output;
    u32 multiple = 2;
    return multiple;
}

HcclResult InsTempAllReduceMesh2DTwoShot::BuildSlice(
    const std::vector<RankId>& rankInfo, const u64 dataSize, const u64 chunkSize, RankSliceInfo& sliceInfoVec) const
{
    std::vector<SliceInfo> tmp(1);
    sliceInfoVec.resize(rankInfo.size(), tmp);

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < rankInfo.size(); rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: GenExtIns 算子执行入口
 * param: tempAlgParams: slice和stride信息
 * param: tempFuncs: 辅助信息包括userIn/OutSlices, opMode等标记信息
 * param: tempLinks: 当前rank通信链接信息
 * param: tempInsQues: 通信队列
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh2DTwoShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    InitInnerParams(tempFuncs, tempAlgParams, tempLinks, tempInsQues);
    // step1: reducescatter, X轴划分为M个块，每个块大小N*chunksize, Y轴划分为N个块，每个块M*chunksize
    CHK_RET(PreSyncQues(tempInsQues, 0));
    CHK_RET(PostSyncQues(tempInsQues, 0));
    SubStageArgs bufferInfo
        = {tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.scratBuffType,
           tempAlgParams.buffInfo.inBuffBaseOff, 0};
    CHK_RET(RunReduceScatter(bufferInfo, XsliceInfoVec_, tempLinks, XtempInsQues_, tempVTopo_[0]));

    if (YDataSize_ != 0) {
        bufferInfo
            = {tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.scratBuffType,
               tempAlgParams.buffInfo.inBuffBaseOff + M_ * N_ * chunkSize_, M_ * N_ * chunkSize_};
        CHK_RET(RunReduceScatter(bufferInfo, YsliceInfoVec_, tempLinks, YtempInsQues_, tempVTopo_[1]));
    }

    // step2: 换轴reducescatter
    CHK_RET(PreSyncQues(tempInsQues, 0));
    CHK_RET(PostSyncQues(tempInsQues, 0));
    if (XDataSizeS2_ != 0) {
        bufferInfo
            = {tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratBuffType, M_ * N_ * chunkSize_,
               M_ * N_ * chunkSize_ + M_ * chunkSize_};
        CHK_RET(RunReduceScatter(bufferInfo, XsliceInfoVecS2_, tempLinks, XtempInsQues_, tempVTopo_[0]));
    }
    if (YDataSizeS2_ != 0) {
        bufferInfo = {tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratBuffType, 0, N_ * chunkSize_};
        CHK_RET(RunReduceScatter(bufferInfo, YsliceInfoVecS2_, tempLinks, YtempInsQues_, tempVTopo_[1]));
    }

    // step3: allgather
    CHK_RET(PreSyncQues(tempInsQues, 0));
    CHK_RET(PostSyncQues(tempInsQues, 0));
    if (XDataSizeS2_ != 0) { // X轴allgather
        bufferInfo
            = {tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratBuffType,
               M_ * N_ * chunkSize_ + M_ * chunkSize_, M_ * N_ * chunkSize_};
        CHK_RET(RunAllgather(bufferInfo, XsliceInfoVecS2_, tempLinks, XtempInsQues_, tempVTopo_[0]));
    }
    if (YDataSizeS2_ != 0) { // Y轴allgather
        bufferInfo = {tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratBuffType, N_ * chunkSize_, 0};
        CHK_RET(RunAllgather(bufferInfo, YsliceInfoVecS2_, tempLinks, YtempInsQues_, tempVTopo_[1]));
    }

    // step4: 换轴allgather
    CHK_RET(PreSyncQues(tempInsQues, 0));
    CHK_RET(PostSyncQues(tempInsQues, 0));
    bufferInfo
        = {tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.outBuffType, 0,
           tempAlgParams.buffInfo.outBuffBaseOff};
    CHK_RET(RunAllgather(bufferInfo, XsliceInfoVec_, tempLinks, XtempInsQues_, tempVTopo_[0]));
    if (YDataSize_ != 0) {
        bufferInfo
            = {tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.outBuffType, M_ * N_ * chunkSize_,
               tempAlgParams.buffInfo.outBuffBaseOff + M_ * N_ * chunkSize_};
        CHK_RET(RunAllgather(bufferInfo, YsliceInfoVec_, tempLinks, YtempInsQues_, tempVTopo_[1]));
    }
    CHK_RET(PreSyncQues(tempInsQues, 0));
    CHK_RET(PostSyncQues(tempInsQues, 0));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh2DTwoShot::InitInnerParams(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempLinks;
    HCCL_INFO("[InsTempAllReduceMesh2DTwoShot] start.");
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;

    M_ = tempVTopo_[0].size();
    N_ = tempVTopo_[1].size();
    queNum_ = M_ + N_;
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR(
            "[InsTempAllReduceMesh2DTwoShot] Rank [%d], queNum_:[%u], tempInsQues size:[%u],requiredQue Error.",
            myRank_, queNum_, tempInsQues.size()),
        HcclResult::HCCL_E_INTERNAL);

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_); // 均分为2MN块
    u32 times = 2;
    chunkSize_ = RoundUp(tempAlgParams.sliceSize, (M_ * N_ * times * dataSizePerVolume)) * dataSizePerVolume;
    CHK_PRT_RET(
        (chunkSize_ * M_ * N_ * times) > tempAlgParams.buffInfo.scratchBuffSize,
        HCCL_ERROR(
            "[InsTempAllReduceMesh2DTwoShot]Rank [%d], Input size:[%llu], BfSize:[%llu] Insufficient buffer!", myRank_,
            tempAlgParams.sliceSize, tempAlgParams.buffInfo.scratchBuffSize),
        HcclResult::HCCL_E_INTERNAL);
    XtempInsQues_ = std::vector<InsQuePtr>(tempInsQues.begin(), tempInsQues.begin() + M_);
    YtempInsQues_ = std::vector<InsQuePtr>(tempInsQues.begin() + M_, tempInsQues.end());

    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], XAlgrankId_));
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[1], YAlgrankId_));

    // for step1
    XDataSize_ = tempAlgParams.sliceSize >= M_ * N_ * chunkSize_ ? M_ * N_ * chunkSize_ : tempAlgParams.sliceSize;
    YDataSize_ = tempAlgParams.sliceSize - XDataSize_;
    BuildSlice(tempVTopo_[0], XDataSize_, N_ * chunkSize_, XsliceInfoVec_);
    BuildSlice(tempVTopo_[1], YDataSize_, M_ * chunkSize_, YsliceInfoVec_);

    // for step2
    YDataSizeS2_ = XsliceInfoVec_[XAlgrankId_][0].size; // 找到前一步切分时本rank负责的数据块大小
    XDataSizeS2_ = YsliceInfoVec_[YAlgrankId_][0].size;
    BuildSlice(tempVTopo_[1], YDataSizeS2_, chunkSize_, YsliceInfoVecS2_);
    BuildSlice(tempVTopo_[0], XDataSizeS2_, chunkSize_, XsliceInfoVecS2_);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 2D Mesh twoshot AllReduce: Scatter+reduce
 * param: sliceInfoVec: 每个rank的数据切片信息
 * param: tempLinks: 当前rank通信链接信息
 * param: tempInsQues: 通信队列
 * param: tempFuncs: 辅助信息包括userIn/OutSlices, opMode等标记信息
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh2DTwoShot::RunReduceScatter(
    SubStageArgs& subparams, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues, const std::vector<RankId>& rankInfo) const
{
    u32 myAlgrankId;
    CHK_RET(GetAlgRank(myRank_, rankInfo, myAlgrankId));

    CHK_RET(PreSyncQues(tempInsQues, 0));
    // scatter
    for (u32 rankId = 0; rankId < rankInfo.size(); rankId++) { // 写模式
        DataSlice ssrcSlice = DataSlice(
            subparams.inType, sliceInfoVec[rankId][0].offset + subparams.inbaseOff, sliceInfoVec[rankId][0].size);
        DataSlice sdestSlice = DataSlice(
            subparams.outType, myAlgrankId * sliceInfoVec[rankId][0].size + subparams.outbaesOff,
            sliceInfoVec[rankId][0].size);
        if (rankId == myAlgrankId) {
            if (sliceInfoVec[rankId][0].size != 0) { // 如果是本地rank，直接拷贝到scratch对应位置
                CHK_PRT_RET(
                    LocalCopy(tempInsQues[rankId], ssrcSlice, sdestSlice),
                    HCCL_ERROR(
                        "[InsTempAllReduceMesh2DTwoShot][RunReduceScatter] RunAllReduce scatter LocalCopy failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        } else {
            const std::vector<LinkData>& linkSendRecv = tempLinks.at(rankInfo[rankId]);
            // 发送, 未过滤size为0的情况
            std::vector<DataSlice> sendSrcSlices{ssrcSlice};
            std::vector<DataSlice> sendDestSlices{sdestSlice};
            // 接收，未过滤size为0的情况
            DataSlice rsrcSlice = DataSlice(
                subparams.inType, sliceInfoVec[myAlgrankId][0].offset + subparams.inbaseOff,
                sliceInfoVec[myAlgrankId][0].size);
            DataSlice rdestSlice = DataSlice(
                subparams.outType, rankId * sliceInfoVec[myAlgrankId][0].size + subparams.outbaesOff,
                sliceInfoVec[myAlgrankId][0].size);
            std::vector<DataSlice> recvSrcSlices{rsrcSlice};
            std::vector<DataSlice> recvDestSlices{rdestSlice};
            TxRxLinks sendRecvLinks(linkSendRecv[0], linkSendRecv[0]);
            TxRxSlicesList sendRecvSlicesList({sendSrcSlices, sendDestSlices}, {recvSrcSlices, recvDestSlices});
            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, tempInsQues[rankId], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempAllReduceMesh2DTwoShot][RunReduceScatter] RunAllReduce scatter failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    CHK_RET(PostSyncQues(tempInsQues, 0));        // 从流同步,等待所有并发的send和copy完成
    if (sliceInfoVec[myAlgrankId][0].size != 0) { // local reduce, 计算结果都放在最开始的位置
        DataSlice ldestSlice = DataSlice(subparams.outType, subparams.outbaesOff, sliceInfoVec[myAlgrankId][0].size);
        for (u32 rankId = 1; rankId < rankInfo.size(); rankId++) {
            DataSlice lsrcSlice = DataSlice(
                subparams.outType, rankId * sliceInfoVec[myAlgrankId][0].size + subparams.outbaesOff,
                sliceInfoVec[myAlgrankId][0].size);
            // 所有reduce操作在同一个insque中才能保序；
            CHK_PRT_RET(
                LocalReduce(tempInsQues[0], lsrcSlice, ldestSlice, dataType_, redOp_),
                HCCL_ERROR("[InsTempAllReduceMesh2DTwoShot]LocalReduce failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 2D Mesh twoshot AllReduce: Allgather
 * param: sliceInfoVec: 每个rank的数据切片信息
 * param: tempLinks: 当前rank通信链接信息
 * param: tempInsQues: 通信队列
 * param: tempFuncs: 辅助信息包括userIn/OutSlices, opMode等标记信息
 * return: HcclResult
 */
HcclResult InsTempAllReduceMesh2DTwoShot::RunAllgather(
    SubStageArgs& subparams, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues, const std::vector<RankId>& rankInfo) const
{
    u32 myAlgrankId;
    CHK_RET(GetAlgRank(myRank_, rankInfo, myAlgrankId));

    // sync:前同步
    CHK_RET(PreSyncQues(tempInsQues, 0));

    // allgather
    for (u32 rankId = 0; rankId < rankInfo.size(); rankId++) {
        DataSlice rsrcSlice = DataSlice(subparams.inType, subparams.inbaseOff, sliceInfoVec[rankId][0].size);
        DataSlice rdestSlice = DataSlice(
            subparams.outType, sliceInfoVec[rankId][0].offset + subparams.outbaesOff, sliceInfoVec[rankId][0].size);
        if (u32(myAlgrankId) == rankId) {
            if (sliceInfoVec[rankId][0].size != 0) {
                // copy本端计算的结果到user output
                CHK_PRT_RET(
                    LocalCopy(tempInsQues[rankId], rsrcSlice, rdestSlice),
                    HCCL_ERROR("[InsTempAllReduceMesh2DTwoShot][RunAllgather] RunAllReduce AllGather "
                               "LocalCopy failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        } else {
            const std::vector<LinkData>& linkSendRecv = tempLinks.at(rankInfo[rankId]);
            // 接收, 未过滤size为0的情况
            std::vector<DataSlice> recvSrcSlices{rsrcSlice};
            std::vector<DataSlice> recvDestSlices{rdestSlice};

            // 发送，未过滤size为0的情况
            DataSlice ssrcSlice = DataSlice(subparams.inType, subparams.inbaseOff, sliceInfoVec[myAlgrankId][0].size);
            DataSlice sdestSlice = DataSlice(
                subparams.outType, sliceInfoVec[myAlgrankId][0].offset + subparams.outbaesOff,
                sliceInfoVec[myAlgrankId][0].size);
            std::vector<DataSlice> sendSrcSlices{ssrcSlice};
            std::vector<DataSlice> sendDestSlices{sdestSlice};

            TxRxLinks sendRecvLinks(linkSendRecv[0], linkSendRecv[0]);
            TxRxSlicesList sendRecvSlicesList({sendSrcSlices, sendDestSlices}, {recvSrcSlices, recvDestSlices});

            SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecv(sendRecvInfo, tempInsQues[rankId], 0, true, DmaMode::GET),
                HCCL_ERROR("[InsTempAllReduceMesh2DTwoShot][RunAllgather] RunAllReduce AllGather failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    CHK_RET(PostSyncQues(tempInsQues, 0));
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
