/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_dispatcher.h"

#include "common/aicpu_sqe_context.h"
#include "common/aicpu_hccl_common.h"
#include "adapter_hal_pub.h"

using namespace hccl;

HcclResult AicpuDispatcher::SignalWait(u16 streamId, u16 notifyId, bool innerChip, bool preNotify)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo* notifyInfo
        = innerChip ? (preNotify ? &ctx->noIpcPreNotify[notifyId] : &ctx->noIpcPostNotify[notifyId]) :
                      (preNotify ? &ctx->ipcPreWaitNotify[notifyId] : &ctx->ipcPostWaitNotify[notifyId]);
    return SignalWaitWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult AicpuDispatcher::AicpuUnfoldSignalWait(u16 streamId, u16 notifyId, bool innerChip)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo* notifyInfo = &ctx->aicpuOpNotify[notifyId];
    return SignalWaitWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult
AicpuDispatcher::SignalWaitWithNotify(u16 streamId, u16 notifyId, bool innerChip, AicpuComSignalInfo* notifyInfo)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));
    if (innerChip || (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3)) {
        AicpuAddOneNotifyWaitSqe addOneNotifyWaitSqe = AicpuGetAddOneNotifyWaitSqe();
        if (addOneNotifyWaitSqe == nullptr) {
            HCCL_ERROR("AicpuAddOneNotifyWaitSqe is null.");
            return HCCL_SUCCESS;
        }
        if (ctx->debugMode == MC2_DEBUG_NOTIFY_WAIT_TIMEOUT) {
            addOneNotifyWaitSqe(
                streamInfo->actualStreamId, taskId, INVALID_U64, sqeBuffer, sqeTypeAddr,
                ctx->dfxExtendInfo.dfxTimeOutConfig);
        } else {
            addOneNotifyWaitSqe(
                streamInfo->actualStreamId, taskId, notifyInfo->actualNotifyId, sqeBuffer, sqeTypeAddr,
                ctx->dfxExtendInfo.dfxTimeOutConfig);
        }

        if (innerChip) {
            CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, ctx->rankId));
        } else {
            CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, notifyId));
        }
    } else {
        u32 notifyRevisedOffset = 15U; // eventid偏移15位后为1
        u32 notifyGetEventId = 0x3FFU; // 取低15位
        if ((static_cast<u32>(notifyInfo->actualNotifyId) >> notifyRevisedOffset) != 0) {
            AicpuAddOneEventWaitSqe addOneEventWaitSqe = AicpuGetAddOneEventWaitSqe();
            if (addOneEventWaitSqe == nullptr) {
                HCCL_ERROR("addOneEventWaitSqe is null");
                return HCCL_SUCCESS;
            }
            addOneEventWaitSqe(
                streamInfo->actualStreamId, (static_cast<u32>(notifyInfo->actualNotifyId) & notifyGetEventId), taskId,
                sqeBuffer, sqeTypeAddr);
            CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, notifyId));

            uint8_t* sqeBuffer1 = nullptr;
            uint8_t* sqeTypeAddr1 = nullptr;
            CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer1, sqeTypeAddr1, taskId));

            AicpuAddOneEventResetSqe addOneEventResetSqe = AicpuGetAddOneEventResetSqe();
            if (addOneEventResetSqe == nullptr) {
                HCCL_ERROR("addOneEventResetSqe is null");
                return HCCL_SUCCESS;
            }
            addOneEventResetSqe(
                streamInfo->actualStreamId, (static_cast<u32>(notifyInfo->actualNotifyId) & notifyGetEventId), taskId,
                streamId, 0, notifyInfo->address, sqeBuffer1, sqeTypeAddr1);
            CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, notifyId));
        } else {
            HCCL_WARNING("SignalWait id is not event, please check %d", notifyInfo->actualNotifyId);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuDispatcher::SignalRecord(u16 streamId, u16 notifyId, bool innerChip, bool preNotify)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo* notifyInfo
        = innerChip ? (preNotify ? &ctx->noIpcPreNotify[notifyId] : &ctx->noIpcPostNotify[notifyId]) :
                      (preNotify ? &ctx->ipcPreRecordNotify[notifyId] : &ctx->ipcPostRecordNotify[notifyId]);
    return SignalRecordWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult AicpuDispatcher::AicpuUnfoldSignalRecord(u16 streamId, u16 notifyId, bool innerChip)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo* notifyInfo = &ctx->aicpuOpNotify[notifyId];
    return SignalRecordWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult
AicpuDispatcher::SignalRecordWithNotify(u16 streamId, u16 notifyId, bool innerChip, AicpuComSignalInfo* notifyInfo)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    if (innerChip) {
        AicpuAddOneRecordSqe addOneRecordSqe = AicpuGetAddOneRecordSqe();
        if (addOneRecordSqe == nullptr) {
            HCCL_ERROR("AicpuAddOneRecordSqe is null");
            return HCCL_SUCCESS;
        }
        addOneRecordSqe(streamInfo->actualStreamId, taskId, notifyInfo->actualNotifyId, sqeBuffer, sqeTypeAddr);
        CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, ctx->rankId));
    } else {
        AicpuAddOneWriteValueRecordSqe addOneWriteValueRecordSqe = AicpuGetAddOneWriteValueRecordSqe();
        if (addOneWriteValueRecordSqe == nullptr) {
            HCCL_ERROR("AicpuAddOneWriteValueRecordSqe is null");
            return HCCL_SUCCESS;
        }
        addOneWriteValueRecordSqe(streamInfo->actualStreamId, taskId, notifyInfo->address, sqeBuffer, sqeTypeAddr);
        CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, notifyId));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuDispatcher::CopyData(
    u16 streamId, void* src, void* dst, u32 len, HcclDataType dataType, HcclReduceOp reduceOp, u32 remoteRank)
{
    if (len == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];

    aclDataType rtDataType = DT_MAP_TABLE[dataType];
    aclrtReduceKind rtReduceOp = RK_MAP_TABLE[reduceOp];

    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    AicpuAddOneMemcpySqe addOneMemcpySqe = AicpuGetAddOneMemcpySqe();
    if (addOneMemcpySqe == nullptr) {
        HCCL_ERROR("addOneMemcpySqe is null");
        return HCCL_SUCCESS;
    }
    if (ctx->debugMode == MC2_DEBUG_SDMA_ERROR) {
        src = nullptr;
    }
    addOneMemcpySqe(
        streamInfo->actualStreamId, taskId, src, len, rtDataType, rtReduceOp, dst, 0, ctx->ssid, ctx->devId,
        ctx->overflowAddr, static_cast<uint8_t>(LinkType::LINK_RESERVED), sqeBuffer, sqeTypeAddr, SDMA_QOS_DEFAULT);
    CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, (remoteRank << 16) + static_cast<uint32_t>(dataType))); // 16 bit
    return HCCL_SUCCESS;
}

HcclResult AicpuDispatcher::CopyData(
    uint16_t streamId, u64 src, u64 dst, uint32_t len, HcclDataType dataType, HcclReduceOp reduceOp, u32 remoteRank)
{
    return CopyData(
        streamId, reinterpret_cast<void*>(src), reinterpret_cast<void*>(dst), len, dataType, reduceOp, remoteRank);
}

HcclResult AicpuDispatcher::LaunchTask(uint32_t streamId)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];
    auto& sqeContextBuffer = GetSqeContext()->buffPtr[streamId];
    const auto cnt = sqeContextBuffer.sqeCnt;
    if (cnt == 0U) {
        HCCL_DEBUG("no sqe, rankid:%u, streamId:%d, sqId:%u", streamId, streamInfo->actualStreamId, streamInfo->sqId);
        return HCCL_SUCCESS;
    }
    auto& head = sqeContextBuffer.sqHead;
    auto& tail = sqeContextBuffer.sqTail;
    u32 newTail = (tail + cnt) % streamInfo->sqDepth;
    HCCL_INFO("Before send sqe:%d cnt:%u head:%u curtail:%u newTail:%u", streamInfo->sqId, cnt, head, tail, newTail);

    u64 startUsec = GetCurCpuTimestamp();
    while ((tail < head ? streamInfo->sqDepth : 0U) + tail - head + cnt >= streamInfo->sqDepth) { // 存在回绕
        CHK_RET(QuerySqStatusByType(ctx->devId, streamInfo->sqId, DRV_SQCQ_PROP_SQ_HEAD, head));
        if (GetCurCpuTimestamp() - startUsec > NSEC_PER_SEC * ctx->dfxExtendInfo.dfxTimeOutConfig.sqFullWaitTimeOut) {
            HCCL_ERROR(
                "Rtsq full, timeout %lus. cur head:%u, sqId:%d", ctx->dfxExtendInfo.dfxTimeOutConfig.sqFullWaitTimeOut,
                head, streamInfo->sqId);
            return HCCL_E_INTERNAL;
        }
    }

    auto memcpyFunc = [&](uint32_t dst, uint32_t dstMax, uint32_t src, uint32_t length) -> HcclResult {
        HCCL_DEBUG("Memcpy rank:%u , dst:%u, dstMax:%u, src:%u, length:%u", streamId, dst, dstMax, src, length);
        if (length == 0U) {
            return HCCL_SUCCESS;
        }
        errno_t ret = memcpy_s(
            reinterpret_cast<uint8_t*>(streamInfo->sqBaseAddr) + dst * AC_SQE_SIZE, dstMax * AC_SQE_SIZE,
            sqeContextBuffer.localBuff + src * AC_SQE_SIZE, length * AC_SQE_SIZE);
        if (ret != EOK) {
            HCCL_ERROR("Memcpy ret %d, dst:%u, dstMax:%u, src:%u, length:%u", ret, dst, dstMax, src, length);
            return HCCL_E_MEMORY;
        }
        return HCCL_SUCCESS;
    };
    uint32_t left = streamInfo->sqDepth - tail; // sqeAddr 剩余空间
    const auto tailSqeIdx = sqeContextBuffer.tailSqeIdx;
    HCCL_INFO("cpy sqe, left:%u, tailSqeId:%u, cnt:%u", left, tailSqeIdx, cnt);
    if (cnt <= left) { // 剩余buffer放得下新增sqe
        CHK_RET(memcpyFunc(tail, left, tailSqeIdx - cnt, cnt));
    } else {
        CHK_RET(memcpyFunc(tail, left, tailSqeIdx - cnt, left));
        CHK_RET(memcpyFunc(0, streamInfo->sqDepth, tailSqeIdx - cnt + left, cnt - left));
    }
    CHK_RET(ConfigSqStatusByType(ctx->devId, streamInfo->sqId, DRV_SQCQ_PROP_SQ_TAIL, newTail));

    tail = newTail;
    HCCL_INFO(
        "After send sqe:%d, sqe_num:%u, curHead:%u, curtail:%u, sqeCnt:%u, tailSqeIdx:%u", streamInfo->sqId, cnt, head,
        tail, sqeContextBuffer.sqeCnt, sqeContextBuffer.tailSqeIdx);
    sqeContextBuffer.sqeCnt = 0;
    // StartMC2MaintenanceThread函数如果为空，说明是老的驱动包，为了解决老的驱动包可能的异常cq占满物理cq队列的情况，
    // 我们这里使用物理cq查询接口来清理队列； 如果是新的驱动包，会在StartMC2MaintenanceThread线程中使用logic
    // cq进行查询并解析
    if (!IsSupportStartMC2MaintenanceThread()
        && (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3)) {
        CqeQueryInput cqeQueryInput;
        cqeQueryInput.devId = ctx->devId;
        cqeQueryInput.streamId = streamInfo->actualStreamId;
        cqeQueryInput.sqId = streamInfo->sqId;
        cqeQueryInput.cqId = streamInfo->sqId; // 使用sqid替代cqid，只有在sq cq成对申请，sqid cqid一样时才可以
        cqeQueryInput.type = static_cast<uint32_t>(DRV_NORMAL_TYPE);
        uint8_t tmpAddr[MAX_REPORT_CNT * 16]; // 16 cqe byte size
        cqeQueryInput.cqeAddr = tmpAddr;
        HCCL_DEBUG("Start to call cq report with [%s]", cqeQueryInput.ToString().c_str());
        rtLogicCqReport_t cqeException;
        (void)CqReportRecv(cqeQueryInput, cqeException);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuDispatcher::AddCcoreWait(uint16_t streamId, u64 waitAddr, uint32_t turnNum, bool isLast)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];

    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    HCCL_INFO(
        "[SQE]Add ccore wait addr %p, workSpaceAddr %p, notifyOff %u, turnNum %u, streamId=%u, isLast=%d", waitAddr,
        ctx->workSpaceAddr, ctx->notifyOff, turnNum, streamInfo->actualStreamId, isLast);
    if (ctx->debugMode == MC2_DEBUG_COMMIT_TIMEOUT) {
        ctx->turnValue[turnNum] = 0xFF;
    }
    AddOneWaitStartSqe(
        streamInfo->actualStreamId, taskId, waitAddr, reinterpret_cast<u64>(&ctx->turnValue[turnNum]), isLast,
        reinterpret_cast<rtStarsCcoreWaitStartSqe_t*>(sqeBuffer), sqeTypeAddr);
    CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, (turnNum << 16) + static_cast<uint32_t>(isLast))); // 16 bit
    return HCCL_SUCCESS;
}

HcclResult AicpuDispatcher::AddWaitStartTaskOnMainStream(u16 streamId)
{
    auto ctx = AicpuGetComContext();
    // 保持和AIC的间消息长度一致，每隔64字节(sizeof(u8)*AC_SQE_SIZE)写一个地址。
    u64 waitAddr = 0;
    uint32_t turnNum = 0;
    bool isLast = 0;
    if (ctx->preparePosition == TASK_PREPARE_KERNEL) {
        waitAddr = ctx->workSpaceAddr + offsetof(HcclApi::HcclMsgArea, commMsg.singleMsg.commitTurnCnt)
                   + ctx->msgPosForKernel * sizeof(HcclApi::TurnCnt) + offsetof(HcclApi::TurnCnt, cnt);

        turnNum = ctx->curTurnCntForKernel;
        isLast = ctx->curTurnCntForKernel >= ctx->totalTurnCntForKernel;
        HCCL_INFO(
            "aicpu kernel mode, curTurnCnt %u, totalTurnCnt %u", ctx->curTurnCntForKernel, ctx->totalTurnCntForKernel);
    } else {
        waitAddr = ctx->workSpaceAddr + ctx->notifyOff + offsetof(AivAicpuOpParam, sendCnt);
        turnNum = (ctx->curTurnCnt + 1);
        isLast = (ctx->curTurnCnt + 1 >= ctx->totalTurnCnt);
    }
    return AddCcoreWait(streamId, waitAddr, turnNum, isLast);
}

HcclResult AicpuDispatcher::AddCcoreNotify(uint16_t streamId, uint32_t turnNum)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    u64 recordAddr = 0;
    if (ctx->preparePosition == TASK_PREPARE_KERNEL) {
        recordAddr = ctx->workSpaceAddr + offsetof(HcclApi::HcclMsgArea, commMsg.singleMsg.finishedTurnCnt)
                     + ctx->msgPosForKernel * sizeof(HcclApi::TurnCnt) + offsetof(HcclApi::TurnCnt, cnt);
    } else {
        recordAddr = ctx->workSpaceAddr + ctx->notifyOff + ctx->notifyBeginCnt * AC_SQE_SIZE
                     + offsetof(AivAicpuOpParam, rcvCnt);
    }
    HCCL_INFO(
        "[SQE]Add ccore notify recordAddr %p, workSpaceAddr %p, notifyOff %u, notifyBeginCnt %u,"
        "streamId=%u, curTurnCnt %u, turnNum %u, preparePosition %u, msgPos %u",
        recordAddr, ctx->workSpaceAddr, ctx->notifyOff, ctx->notifyBeginCnt, streamInfo->actualStreamId,
        ctx->curTurnCnt, turnNum, ctx->preparePosition, ctx->msgPosForKernel);

    if (ctx->debugMode == MC2_DEBUG_AICORE_WAIT_TIMEOUT) {
        ctx->turnValue[turnNum] = 0;
    }
    AddOneWriteValueStartSqe(
        streamInfo->actualStreamId, taskId, recordAddr, reinterpret_cast<u64>(&ctx->turnValue[turnNum]),
        reinterpret_cast<rtStarsCcoreWriteValueSqe_t*>(sqeBuffer), sqeTypeAddr);
    CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, turnNum));
    return HCCL_SUCCESS;
}

HcclResult AicpuDispatcher::AddExecEndTaskOnMainStream(u16 streamId)
{
    auto ctx = AicpuGetComContext();
    uint32_t turnNum = ctx->preparePosition == TASK_PREPARE_KERNEL ? ctx->curTurnCntForKernel : ctx->curTurnCnt;
    return AddCcoreNotify(streamId, turnNum);
}

HcclResult AicpuDispatcher::AddAllEndTaskOnMainStream(u16 streamId)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    AicpuAddOneRecordSqe addOneRecordSqe = AicpuGetAddOneRecordSqe();
    if (addOneRecordSqe == nullptr) {
        HCCL_ERROR("AicpuAddOneRecordSqe is null");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[SQE]Add all end task kfcNotifyId %lu, streamId %d", ctx->kfcNotifyId, streamInfo->actualStreamId);
    addOneRecordSqe(streamInfo->actualStreamId, taskId, ctx->kfcNotifyId, sqeBuffer, sqeTypeAddr);
    CHK_RET(AicpuSqeContext::RecordAddInfo(streamId, ctx->rankId));
    return HCCL_SUCCESS;
}

HcclResult AicpuDispatcher::RdmaSend(uint16_t streamId, u64 dbInfo, u64 dbAddr, u32 userRank)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(AicpuSqeContext::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    AicpuAddOneRdmaDbSendSqe AddOneRdmaDbSendSqe = AicpuGetAddOneRdmaDbSendSqe();
    if (AddOneRdmaDbSendSqe == nullptr) {
        HCCL_ERROR("[AicpuDispatcher][RdmaSend] AddOneRdmaDbSendSqe is null");
        return HCCL_E_PTR;
    }
    AddOneRdmaDbSendSqe(
        streamInfo->actualStreamId, taskId, dbInfo, dbAddr, 0, static_cast<uint8_t>(hccl::RdmaType::RDMA_TYPE_RESERVED),
        sqeBuffer, sqeTypeAddr);

    HCCL_INFO(
        "[AicpuDispatcher][RdmaSend] Call RdmaSend. para: rankId[%u] "
        "taskId[%u], streamId[%u]",
        userRank, taskId, streamInfo->actualStreamId);

    return HCCL_SUCCESS;
}
