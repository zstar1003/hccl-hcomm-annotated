/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_send_recv_retry_executor.h"
namespace hccl {
constexpr u32 PAIRSIZE_TWO = 2;

CollBatchSendRecvRetryExecutor::CollBatchSendRecvRetryExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBatchSendRecvExecutor(dispatcher, topoMatcher)
{}

HcclResult CollBatchSendRecvRetryExecutor::CreatePairWiseList(HcclSendRecvItem* sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[CollBatchSendRecvRetryExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO(
            "[CollBatchSendRecvRetryExecutor][GetPairWiseList] index is %u, itemNum is %u, localRankID is %u, "
            "remoteRank is %u, sendRecvType is %u, rankSize is %u.",
            i, itemNum, topoAttr_.userRank, sendRecvInfo->remoteRank, static_cast<u32>(sendRecvInfo->sendRecvType),
            topoAttr_.userRankSize);
        CHK_PTR_NULL(sendRecvInfo->buf);

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            sendDeque_.push_back(sendRecvInfo);
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            recvDeque_.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR(
                "[CollBatchSendRecvRetryExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, "
                "rankID is %u, remoteRank is %u.",
                sendRecvInfo->sendRecvType, topoAttr_.userRank, sendRecvInfo->remoteRank);
            return HCCL_E_PARA;
        }
        sendRecvInfo++;
    }
    /* 此处的排序逻辑(pair-wise算法):
        1.sendDeque元素顺序是:先放remoteRank号小于等于root rank的第一个任务，依次减小(循环索引)直至放完
        2.recvDeque元素顺序是:先放remoteRank号大于等于root rank的第一个任务，依次增大(循环索引)直至放完
    */
    auto sendCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        u32 aFlag = (a->remoteRank <= topoAttr_.userRank) ? (a->remoteRank + topoAttr_.userRankSize) : a->remoteRank;
        u32 bFlag = (b->remoteRank <= topoAttr_.userRank) ? (b->remoteRank + topoAttr_.userRankSize) : b->remoteRank;
        return aFlag > bFlag;
    };

    auto recvCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        u32 aFlag = (a->remoteRank < topoAttr_.userRank) ? (a->remoteRank + topoAttr_.userRankSize) : a->remoteRank;
        u32 bFlag = (b->remoteRank < topoAttr_.userRank) ? (b->remoteRank + topoAttr_.userRankSize) : b->remoteRank;
        return aFlag < bFlag;
    };

    std::sort(sendDeque_.begin(), sendDeque_.end(), sendCompare);
    std::sort(recvDeque_.begin(), recvDeque_.end(), recvCompare);

    // 生成SendRecvPair
    u32 pairNum = std::max(sendDeque_.size(), recvDeque_.size());
    for (u32 pairIndex = 0; pairIndex < pairNum; pairIndex++) {
        std::vector<HcclSendRecvItem*> sendRecvPair;
        if (sendDeque_.size() > pairIndex) {
            sendRecvPair.push_back(sendDeque_[pairIndex]);
        }
        if (recvDeque_.size() > pairIndex) {
            sendRecvPair.push_back(recvDeque_[pairIndex]);
        }
        sendRecvPairList_.push_back(sendRecvPair);
    }
    HCCL_INFO("[CollBatchSendRecvRetryExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

HcclResult
CollBatchSendRecvRetryExecutor::GetPairWiseList(std::vector<std::vector<HcclSendRecvItem*>>& sendRecvPairList)
{
    sendRecvPairList = sendRecvPairList_;
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::GetPairWiseList(HcclSendRecvItem* sendRecvInfo, u32 itemNum)
{
    return CollBatchSendRecvExecutor::GetPairWiseList(sendRecvInfo, itemNum);
}

HcclResult CollBatchSendRecvRetryExecutor::CheckSendRecvPair(const std::vector<HcclSendRecvItem*>& sendRecvPair)
{
    if (sendRecvPair.empty()) {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor] please check the pair list.");
        return HCCL_E_PARA;
    }
    if (sendRecvPair.size() == 1 && sendRecvPair[0]->remoteRank == topoAttr_.userRank) {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor] SendTask and Recv Task to rank itself do not match,"
                   "please check the task list.");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBatchSendRecvRetryExecutor] batchsendrecv retry starts.");
    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));

    // 校验当前sendRecvPair
    std::vector<HcclSendRecvItem*> sendRecvPair;
    if (param.BatchSendRecvDataDes.curIterNum < sendRecvPairList_.size()) {
        sendRecvPair = sendRecvPairList_[param.BatchSendRecvDataDes.curIterNum];
    } else {
        HCCL_ERROR(
            "[CollBatchSendRecvRetryExecutor] the curIterNum[%u] is out of range[0, %zu].",
            param.BatchSendRecvDataDes.curIterNum, sendRecvPairList_.size());
        return HCCL_E_PARA;
    }
    CHK_RET(CheckSendRecvPair(sendRecvPair));

    // 自发自收场景
    if (sendRecvPair.size() == PAIRSIZE_TWO && sendRecvPair[0]->remoteRank == topoAttr_.userRank
        && sendRecvPair[1]->remoteRank == topoAttr_.userRank) {
        if (sendRecvPair[0]->count == sendRecvPair[1]->count
            && sendRecvPair[0]->dataType == sendRecvPair[1]->dataType) {
            u64 dataSize = sendRecvPair[0]->count * SIZE_TABLE[sendRecvPair[0]->dataType];
            DeviceMem inUserMem = DeviceMem::create(static_cast<u8*>(sendRecvPair[0]->buf), dataSize);
            DeviceMem outUserMem = DeviceMem::create(static_cast<u8*>(sendRecvPair[1]->buf), dataSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outUserMem, inUserMem, param.stream));
            return HCCL_SUCCESS;
        } else {
            HCCL_ERROR("[HcclBatchSendRecvRetry] Send task and recv task to self : data size do not equal, please "
                       "check the task list.");
            return HCCL_E_PARA;
        }
    }

    // 重执行正常执行场景，前后需和控制流做同步
    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND_RECV) {
        HCCL_INFO("[BatchSendRecv] Stream sync: main stream record, subStream wait.");
        CHK_RET(LocalNotify::Post(param.stream, dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(
            algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_0],
            PROF_STAGE_0));
        CHK_RET(LocalNotify::Post(param.stream, dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_1], PROF_STAGE_1));
        CHK_RET(LocalNotify::Wait(
            algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_1],
            PROF_STAGE_1));
    }
    // run sendrecv
    CHK_RET(RunLoop(param, algResource, sendRecvPair));

    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND_RECV) {
        HCCL_INFO("[BatchSendRecv] Stream sync: subStream record, main stream wait.");
        CHK_RET(LocalNotify::Post(
            algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_0],
            PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(param.stream, dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Post(
            algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_1],
            PROF_STAGE_1));
        CHK_RET(LocalNotify::Wait(param.stream, dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_1], PROF_STAGE_1));
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND) {
        CHK_RET(LocalNotify::Post(
            algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_0],
            PROF_STAGE_0));
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::RECV) {
        CHK_RET(LocalNotify::Post(
            algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_1],
            PROF_STAGE_1));
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    HCCL_INFO("[info][print] LaunchTaskExtend success.");
    HCCL_INFO(
        "tag[%s] BatchSendRecv Executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::RunLoop(
    OpParam& param, AlgResourceResponse& algRes, const std::vector<HcclSendRecvItem*>& sendRecvPair)
{
    // 判断当前需执行的算子
    std::vector<HcclSendRecvItem*> curSendRecvPair;
    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND) {
        curSendRecvPair.push_back(sendRecvPair[0]);
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::RECV) {
        curSendRecvPair.push_back(sendRecvPair[sendRecvPair.size() - 1]);
    } else {
        curSendRecvPair = sendRecvPair;
    }

    // 执行当前需执行的算子
    for (const auto& itemPtr : curSendRecvPair) {
        if (static_cast<bool>(param.BatchSendRecvDataDes.isDirectRemoteRank[itemPtr->remoteRank])) {
            // device direct链路的任务会在host侧下发，此处需要跳过
            continue;
        }
        HCCL_INFO("[CollBatchSendRecvRetryExecutor][RunLoop] remoteRank %u", itemPtr->remoteRank);
        if (itemPtr->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            CHK_RET(CalcSendSlices(algRes, itemPtr));
        } else if (itemPtr->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            CHK_RET(CalcRecvSlices(algRes, itemPtr));
        } else {
            HCCL_ERROR("[CollBatchSendRecvRetryExecutor][RunLoop] sendRecvType is Wrong.");
            return HCCL_E_PARA;
        }
    }

    u32 loopInOnceLaunch = 0;
    // 每隔200个loop launch一次
    while (!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if (!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendDataSlice(algResResp_->slaveStreams[STREAM_INDEX_0], false, true));
            sendDataSilces_.pop_front();
        }
        if (!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvDataSlice(algResResp_->slaveStreams[STREAM_INDEX_1], true));
            recvDataSilces_.pop_front();
        }
        loopInOnceLaunch++;
        if (loopInOnceLaunch == MAX_LOOP_IN_ONCE_LAUNCH || (sendDataSilces_.empty() && recvDataSilces_.empty())) {
            CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
            HCCL_INFO(
                "[BatchSendRecv] LaunchTaskExtend, unprocessed send slices[%u], recv slices[%u].",
                sendDataSilces_.size(), recvDataSilces_.size());
            loopInOnceLaunch = 0;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    HCCL_INFO("[CollBatchSendRecvRetryExecutor][CalcScratchMemSize] tag_[%s], streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::CalcSendSlices(AlgResourceResponse& algRes, HcclSendRecvItem* sendRecvItem)
{
    HCCL_INFO(
        "[CollBatchSendRecvExecutor][CalcSendSlices] tag[%s], remoteRank[%u], buf[%p], count[%llu],"
        "dataType[%s], sendRecvType[%d].",
        tag_.c_str(), sendRecvItem->remoteRank, sendRecvItem->buf, sendRecvItem->count,
        GetDataTypeEnumStr(sendRecvItem->dataType).c_str(), sendRecvItem->sendRecvType);
    u8* curInputPtr = static_cast<u8*>(sendRecvItem->buf);
    CHK_PTR_NULL(curInputPtr);
    u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
    u64 maxCountPerLoop = CalcSendLoopMaxCount(const_cast<DeviceMem&>(algRes.cclInputMem), unitSize);

    for (u64 countLeft = sendRecvItem->count, curCount = 0, curOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += curOffset;
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节
        sendDataSilces_.emplace_back(curInputPtr, curSize, sendRecvItem->remoteRank);
        HCCL_DEBUG(
            "[CollBatchSendRecvExecutor][CalcSendSlices] tag[%s], slice userAddr[%p], slice size[%llu].", tag_.c_str(),
            curInputPtr, curSize);
        curOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::CalcRecvSlices(AlgResourceResponse& algRes, HcclSendRecvItem* sendRecvItem)
{
    HCCL_INFO(
        "[CollBatchSendRecvRetryExecutor][CalcSendSlices] tag[%s], remoteRank[%u], buf[%p], count[%llu],"
        "dataType[%s], sendRecvType[%d].",
        tag_.c_str(), sendRecvItem->remoteRank, sendRecvItem->buf, sendRecvItem->count,
        GetDataTypeEnumStr(sendRecvItem->dataType).c_str(), sendRecvItem->sendRecvType);
    u8* curOutputPtr = static_cast<u8*>(sendRecvItem->buf);
    CHK_PTR_NULL(curOutputPtr);
    u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
    u64 maxCountPerLoop = CalcRecvLoopMaxCount(const_cast<DeviceMem&>(algRes.cclOutputMem), unitSize);
    HCCL_DEBUG("[CollBatchSendRecvRetryExecutor][CalcSendSlices]maxCountPerLoop is %llu", maxCountPerLoop);

    for (u64 countLeft = sendRecvItem->count, curCount = 0, curOffset = 0; countLeft > 0; countLeft -= curCount) {
        curOutputPtr += curOffset;
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节
        recvDataSilces_.emplace_back(curOutputPtr, curSize, sendRecvItem->remoteRank);
        HCCL_DEBUG(
            "[CollBatchSendRecvRetryExecutor][CalcRecvSlices] tag[%s], slice userAddr[%p], slice size[%llu].",
            tag_.c_str(), curOutputPtr, curSize);
        curOffset = curSize;
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BatchSendRecvRetry", BatchSendRecvRetryExecutor, CollBatchSendRecvRetryExecutor);
} // namespace hccl
