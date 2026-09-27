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

#include "ins_coll_alg_registry.h"
#include "ins_v2_batchSendRecv_executor.h"
#include "alg_data_trans_wrapper.h"

#include "hccl_aiv_utils.h"
#include "aiv_ins.h"
#include "executor_utils.h"

using namespace std;

namespace Hccl {

template <typename AlgTopoMatch>
InsV2BatchSendRecvExecutor<AlgTopoMatch>::InsV2BatchSendRecvExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch>
InsV2BatchSendRecvExecutor<AlgTopoMatch>::~InsV2BatchSendRecvExecutor()
{}

template <typename AlgTopoMatch>
void InsV2BatchSendRecvExecutor<AlgTopoMatch>::SetRmaDataBufferMgr(const RmtDataBufferMgr* rmaDataBufferMgr)
{
    rmaDataBufferMgr_ = const_cast<RmtDataBufferMgr*>(rmaDataBufferMgr);
    return;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::InitParams(const CollAlgOperator& op, const CollAlgParams& params)
{
    op_ = op;
    opMode_ = params.opMode;
    maxTmpMemSize_ = params.maxTmpMemSize;
    CHK_PRT_RET(
        (maxTmpMemSize_ == 0), HCCL_ERROR("[InitParams] maxTmpMemSize equals to zero for OPBASE."),
        HcclResult::HCCL_E_PARA);
    HcclSendRecvItem* itemPtr = reinterpret_cast<HcclSendRecvItem*>(op.batchSendRecvDataDes.sendRecvItemsPtr);
    u32 itemNum = op.batchSendRecvDataDes.itemNum;
    CHK_PTR_NULL(itemPtr);
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum; i++) {
        commTargetUserRankSet_.insert((itemPtr + i)->remoteRank);
        HCCL_DEBUG(
            "[InsV2BatchSendRecvExecutor][ParseParam] insert remoteUserRank[%u] to Set ", (itemPtr + i)->remoteRank);
    }
    HCCL_DEBUG("[InitParams]commTargetUserRankSet_ size[%zu]", commTargetUserRankSet_.size());
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.SetTargetRanks(commTargetUserRankSet_));
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    CHK_PRT_RET(
        topoInfo.vTopo.size() == 0, HCCL_ERROR("[InsV2BatchSendRecvExecutor] Rank[%d], vTopo size is zero.", myRank_),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        topoInfo.virtRankMap.size() == 0,
        HCCL_ERROR("[InsV2BatchSendRecvExecutor] Rank[%d], virtRankMap size is zero.", myRank_),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        topoInfo.virtRanks.size() == 0,
        HCCL_ERROR("[InsV2BatchSendRecvExecutor] Rank[%d], virtRanks size is zero.", myRank_), HcclResult::HCCL_E_PARA);

    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::CalNumBlocks(u32& blockDim, u64 dataSize, u32 blockDimLimit)
{
    (void)dataSize;
    u32 rankNum = 2;
    HCCL_INFO("[InsV2BatchSendRecvExecutor] Limit core num[%u]", blockDimLimit);

    if (blockDimLimit < rankNum) { // batchSendRecv至少需要两个核，分别去收发
        HCCL_ERROR("[InsV2BatchSendRecvExecutor] core num[%u] is less than 2", blockDimLimit);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    blockDim = blockDimLimit / rankNum * rankNum;
    HCCL_INFO("[InsV2BatchSendRecvExecutor] Actually use core num[%u]", blockDim);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
bool InsV2BatchSendRecvExecutor<AlgTopoMatch>::SortSendItems(HcclSendRecvItem* a, HcclSendRecvItem* b) const
{
    u32 aFlag = (a->remoteRank <= static_cast<uint32_t>(myRank_)) ? (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank <= static_cast<uint32_t>(myRank_)) ? (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag > bFlag) {
        return true;
    } else if (aFlag < bFlag) {
        return false;
    }
    return a->count > b->count;
}

template <typename AlgTopoMatch>
bool InsV2BatchSendRecvExecutor<AlgTopoMatch>::SortRecvItems(HcclSendRecvItem* a, HcclSendRecvItem* b) const
{
    u32 aFlag = (a->remoteRank < static_cast<uint32_t>(myRank_)) ? (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank < static_cast<uint32_t>(myRank_)) ? (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag > bFlag) {
        return false;
    } else if (aFlag < bFlag) {
        return true;
    }
    return a->count > b->count;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::GetPairWiseList()
{
    HCCL_INFO("[InsV2BatchSendRecvExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");

    HcclSendRecvItem* sendRecvInfo = static_cast<HcclSendRecvItem*>(op_.batchSendRecvDataDes.sendRecvItemsPtr);
    u32 itemNum = op_.batchSendRecvDataDes.itemNum;
    if (itemNum > BATCH_SEND_RECV_ITEM_SIZE) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvExecutor][GetPairWiseList] itemNum [%u] is greater than BATCH_SEND_RECV_ITEM_SIZE [%u]",
            itemNum, BATCH_SEND_RECV_ITEM_SIZE);
        return HcclResult::HCCL_E_PARA;
    }

    CHK_PTR_NULL(sendRecvInfo);
    std::set<DataType> hcclDataTypeSet;

    for (u32 i = 0; i < itemNum; i++) {
        CHK_PTR_NULL(sendRecvInfo->buf);
        HCCL_INFO(
            "[InsV2BatchSendRecvExecutor][GetPairWiseList] index is %u, itemNum is %u,"
            "localRankID is %d, sendRecvType is %u, buf is %p, count is %u, dataType is %u, remoteRank is %u, rankSize "
            "is %u.",
            i, itemNum, myRank_, static_cast<u32>(sendRecvInfo->sendRecvType), sendRecvInfo->buf, sendRecvInfo->count,
            static_cast<u32>(sendRecvInfo->dataType), sendRecvInfo->remoteRank, rankSize_);

        hcclDataTypeSet.insert(HcclDataTypeToDataType(sendRecvInfo->dataType));

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            sendDeque_.push_back(sendRecvInfo);
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            recvDeque_.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR(
                "[InsV2BatchSendRecvExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, "
                "rankID is %d, remoteRank is %u.",
                sendRecvInfo->sendRecvType, myRank_, sendRecvInfo->remoteRank);
            return HcclResult::HCCL_E_PARA;
        }
        sendRecvInfo++;
    }
    // 如果item里面的数据类型都一样，那就用item里面的，如果不一样，就统一用UINT8
    dataType_ = DataType::UINT8;
    if (hcclDataTypeSet.size() == 1) {
        dataType_ = *hcclDataTypeSet.begin();
    }
    HCCL_INFO(
        "[InsV2BatchSendRecvExecutor][GetPairWiseList] dataType num is %u, so the final dataType_ is %u",
        hcclDataTypeSet.size(), static_cast<u32>(dataType_));

    /* 此处的排序逻辑(pair-wise算法):
        1.sendDeque元素顺序是:先放remoteRank号小于等于root rank的第一个任务，依次减小(循环索引)直至放完
        2.recvDeque元素顺序是:先放remoteRank号大于等于root rank的第一个任务，依次增大(循环索引)直至放完
        如果有rank间重复send/recv场景，按照收发数据从大到小排序
    */
    auto sendCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        return this->SortSendItems(a, b);
    };

    auto recvCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        return this->SortRecvItems(a, b);
    };

    std::stable_sort(sendDeque_.begin(), sendDeque_.end(), sendCompare);
    std::stable_sort(recvDeque_.begin(), recvDeque_.end(), recvCompare);

    // 校验自收发任务，校验数据量和数据类型是否一一对应
    // 遍历收发队列
    for (auto& item : sendDeque_) {
        if (item->remoteRank == static_cast<uint32_t>(myRank_)) {
            sendToSelfDeque_.push_back(item);
        }
    }

    for (auto& item : recvDeque_) {
        if (item->remoteRank == static_cast<uint32_t>(myRank_)) {
            recvFromSelfDeque_.push_back(item);
        }
    }

    if (sendToSelfDeque_.size() != recvFromSelfDeque_.size()) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvExecutor][GetPairWiseList] selfSendRecv is not equal, sendQue size is [%u], recvQue "
            "size is [%u]",
            sendToSelfDeque_.size(), recvFromSelfDeque_.size());
        return HcclResult::HCCL_E_PARA;
    }

    // 收发队列应该一一对应
    for (u32 i = 0; i < sendToSelfDeque_.size(); i++) {
        if ((sendToSelfDeque_[i]->count != recvFromSelfDeque_[i]->count)
            || (sendToSelfDeque_[i]->dataType != recvFromSelfDeque_[i]->dataType)) {
            HCCL_ERROR(
                "[InsV2BatchSendRecvExecutor][GetPairWiseList] selfSendRecv is not equal, "
                "sendQue count is [%u], sendQue dataType is [%u]; recvQue count is [%u], recvQue dataType is [%u]",
                sendToSelfDeque_[i]->count, static_cast<u32>(sendToSelfDeque_[i]->dataType),
                recvFromSelfDeque_[i]->count, static_cast<u32>(recvFromSelfDeque_[i]->dataType));
            return HcclResult::HCCL_E_PARA;
        }
    }

    HCCL_INFO("[CollBatchSendRecvExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aiv接口，这个接口需要补齐
template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsV2BatchSendRecvExecutor][Orchestrate] Begin to Generate Instruction Queue for BatchSendRecv.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));

    CHK_PRT_RET(
        rankSize_ == 1, HCCL_ERROR("BatchSendRecv Executor orchestrate failed, do not support single rank."),
        HcclResult::HCCL_E_PARA);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    CHK_RET(CalcRes(tempResReq));

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor] Rank[%d], requiredQue Num [%u].", myRank_, tempResReq.queNum);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    CHK_RET(ExecAiv()); // 这里进入算法编排，把参数按照顺序构造好发下去
    HCCL_INFO("[InsV2BatchSendRecvExecutor][Orchestrate] Orchestrate AIV End");
    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsV2BatchSendRecvExecutor][Orchestrate] Begin to Generate Instruction Queue for BatchSendRecv.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(topoInfo));

    CHK_PRT_RET(
        rankSize_ == 1, HCCL_ERROR("BatchSendRecv Executor orchestrate failed, do not support single rank."),
        HcclResult::HCCL_E_PARA);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    CHK_RET(CalcRes(tempResReq));

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor] Rank[%d], requiredQue Num [%u].", myRank_, tempResReq.queNum);

    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    CHK_RET(ExecAiv()); // 这里进入算法编排，把参数按照顺序构造好发下去
    HCCL_INFO("[InsV2BatchSendRecvExecutor][Orchestrate] Orchestrate AICPU End");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::ExecAiv()
{
    HCCL_INFO("[InsV2SendExecutor][ExecAiv] start: rank[%d]", myRank_);

    CHK_RET(GetPairWiseList());

    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxScratchDataSize = std::min(transportBoundDataSize, maxTmpMemSize_);
    std::vector<LinkData> allLinks;
    for (auto iter = tempResLinks_.begin(); iter != tempResLinks_.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    sliceId_++;

    AivOpArgs aivBatchSendRecvArgs;
    aivBatchSendRecvArgs.cmdType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    aivBatchSendRecvArgs.input = 0; // ins_rules.cc里面，这里会和起始地址累加起来作为input
    aivBatchSendRecvArgs.output = 0;
    aivBatchSendRecvArgs.rank = u32(myRank_);
    aivBatchSendRecvArgs.rankSize = rankSize_;
    aivBatchSendRecvArgs.count
        = maxScratchDataSize; // 把整个 CCLBuffer的size发过去，因为这里没法确认单次send/recv的dataType
    aivBatchSendRecvArgs.dataType = dataType_;
    aivBatchSendRecvArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivBatchSendRecvArgs.isOpBase = (opMode_ == OpMode::OPBASE);
    aivBatchSendRecvArgs.xRankSize = rankSize_;
    aivBatchSendRecvArgs.yRankSize = 0;
    aivBatchSendRecvArgs.zRankSize = 0;
    CHK_RET(CalNumBlocks(
        aivBatchSendRecvArgs.numBlocks, 0, op_.numBlocksLimit)); // 为什么前面计算的值不能用吗，这里要再计算一遍

    aivBatchSendRecvArgs.extraArgs.itemNum = op_.batchSendRecvDataDes.itemNum;

    // 遍历收、发队列
    u32 curQue = 0;
    for (auto& item : sendDeque_) {
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].sendRecvType = static_cast<uint32_t>(item->sendRecvType);
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].bufAddr = reinterpret_cast<uint64_t>(item->buf);
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].count = item->count;
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].dataTypeSize
            = DATA_TYPE_SIZE_MAP.at(HcclDataTypeToDataType(item->dataType));
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].remoteRank = item->remoteRank;
        curQue++;
    }

    for (auto& item : recvDeque_) {
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].sendRecvType = static_cast<uint32_t>(item->sendRecvType);
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].bufAddr = reinterpret_cast<uint64_t>(item->buf);
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].count = item->count;
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].dataTypeSize
            = DATA_TYPE_SIZE_MAP.at(HcclDataTypeToDataType(item->dataType));
        aivBatchSendRecvArgs.extraArgs.sendRecvInfo[curQue].remoteRank = item->remoteRank;
        curQue++;
    }

    aivBatchSendRecvArgs.inputSliceStride = 0;
    aivBatchSendRecvArgs.outputSliceStride = 0;
    aivBatchSendRecvArgs.repeatNum = 1; // 不重复
    aivBatchSendRecvArgs.inputRepeatStride = 0;
    aivBatchSendRecvArgs.outputRepeatStride = 0;

    std::unique_ptr<Instruction> aivInsBatchSendRecv = std::make_unique<AivInstruction>(allLinks, aivBatchSendRecvArgs);

    requiredQue_[0]->Append(std::move(aivInsBatchSendRecv));

    HCCL_INFO("[InsV2BatchSendRecvExecutor][ExecAiv] end: rank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::CalcResLinksPartialMesh(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const u32 linkNumBtwPeers,
    AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    u32 partialRankSize = commTargetUserRankSet_.size() + 1;

    if (tempVTopo.size() < 1) {
        HCCL_ERROR("[InsV2BatchSendRecvExecutor][CalcResLinksPartialMesh] Rank[%d], tempVTopo size is zero.", myRank);
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < tempVTopo.size(); i++) { // 遍历level0的2个平面
        CHK_RET(GetAlgRank(myRank, tempVTopo[i], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempResReq.queNum; queIdx++) {
            // find neighbors : virtualRank
            u32 remoteAlgRank = (myAlgRank + 1 + queIdx + partialRankSize) % partialRankSize;
            if (remoteAlgRank >= tempVTopo[i].size()) {
                continue;
            }
            RankId neighborRank = tempVTopo[i][remoteAlgRank];
            HCCL_DEBUG("tempVTopo[%u] index[%u] value[%d]", i, remoteAlgRank, neighborRank);
            auto rankInRankSet = std::find(
                commTargetUserRankSet_.begin(), commTargetUserRankSet_.end(), static_cast<u32>(neighborRank));
            if (rankInRankSet != commTargetUserRankSet_.end() && neighborRank != myRank) {
                // LinkNum
                tempResReq.links[neighborRank] = linkNumBtwPeers;
                HCCL_DEBUG("myRank[%d] neighborRank[%d] links is [%u]", myRank, neighborRank, linkNumBtwPeers);
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1; // aiv只需要1条流
    tempResReq.streamNum = tempResReq.queNum;

    CHK_RET(CalcResLinksPartialMesh(myRank_, vTopo_, 1, tempResReq));
    HCCL_DEBUG(
        "[InsV2BatchSendRecvExecutor][CalcRes] Rank[%d] vTopoSize[%lu] requiredQue Num[%u].", myRank_, vTopo_[0].size(),
        tempResReq.queNum);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)rankGraph;
    (void)dataSize;
    (void)resReq;
    HCCL_ERROR("[InsCollAlgFactory][InsV2BatchSendRecvExecutor][CalcResOffload] offload is not supported");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

template <typename AlgTopoMatch>
HcclResult InsV2BatchSendRecvExecutor<AlgTopoMatch>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    for (u32 i = 0; i < vTopo_.size(); i++) {        // 遍历level0
        for (u32 j = 0; j < vTopo_[i].size(); j++) { // 遍历平面内的所有rank
            HCCL_DEBUG(
                "[InsV2BatchSendRecvExecutor][CalcResLinksPartialMesh] vTopo_[%u][%u] is [%d].", i, j, vTopo_[i][j]);
        }
    }
    HCCL_DEBUG(
        "[InsV2BatchSendRecvExecutor][CalcRes]topoInfo.virtRanks[%zu], topoInfo.virtRankMap[%zu],"
        "topoInfo.vTopo[%zu]",
        algResReq.topoInfo.virtRanks.size(), algResReq.topoInfo.virtRankMap.size(), algResReq.topoInfo.vTopo.size());

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2BatchSendRecvExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[InsV2BatchSendRecvExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(CalcRes(tempResReq));
    }

    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor] Rank[%d], requiredQueNum [%u].", myRank_, algResReq.primQueueNum);

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor] Rank[%d], algResReq.links size[%zu].", myRank_, algResReq.links.size());

    return HcclResult::HCCL_SUCCESS;
}

// 注册
INS_REGISTER_IMPL_BY_TOPO(OpType::BATCHSENDRECV, AivBatchSendRecv, InsV2BatchSendRecvExecutor, TopoMatchPartialMesh);

} // namespace Hccl
