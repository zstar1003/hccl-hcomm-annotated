/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_allgather.h"
#include "common/aicpu_hccl_common.h"

bool AicpuAllgather::isMCFirstCall = true;

HcclResult AicpuAllgather::RunAlgorithm(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen,
    AivAicpuOpParam* nextTask)
{
    CHK_PTR_NULL(ctx_);
    switch (ctx_->commAlg) {
        case CommAlgType::COMM_ALG_FULL_MESH: {
            if (ctx_->commLen < ctx_->windowSize / AC_DEFAULT_WINDOW_DIM) {
                if (nextTask == nullptr
                    || DataUnitSize(nextTask->hcclDataType) * nextTask->count
                           > ctx_->windowSize / AC_DEFAULT_WINDOW_DIM) {
                    return RunAllGathervMC(opType, sendBuffer, recvBuffer, dataCount, dataType, strideLen, nullptr);
                } else {
                    return RunAllGathervMC(opType, sendBuffer, recvBuffer, dataCount, dataType, strideLen, nextTask);
                }
            }
            return RunAllGatherv(opType, sendBuffer, recvBuffer, dataCount, dataType, strideLen);
        }
        case CommAlgType::COMM_ALG_DOUBLE_RING: {
            return RunDoubleRingAllGather(
                opType, reinterpret_cast<u64>(sendBuffer), reinterpret_cast<u64>(recvBuffer), dataCount, dataType);
        }
        default: {
            HCCL_ERROR("CommAlg %d is not supported.", ctx_->commAlg);
            return HCCL_E_NOT_SUPPORT;
        }
    }
}

HcclResult AicpuAllgather::RunAllGatherv(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideCnt)
{
    CHK_PTR_NULL(ctx_);

    // dataCount 为 gather前的数据量
    u64 windowSize = ctx_->windowSize;
    u32 unitSize = ctx_->unitSize;
    u64 maxCountPerLoop = windowSize / unitSize; // 中转内存单次最多能够接受的output count

    u8* curInputPtr = static_cast<u8*>(sendBuffer);
    u8* curOutputPtr = static_cast<u8*>(recvBuffer);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = dataCount;

    u64 displs[AC_MAX_RANK_NUM] = {0};
    for (u32 i = 0; i < ctx_->rankNum; i++) {
        displs[i] = i * strideCnt * ctx_->unitSize;
    }

    u32 loopIdx = 0;
    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG(
            "RunAllGatherv: loop %u, curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]", loopIdx++,
            curInputPtr, curOutputPtr, curCount, curSize);

        // 1. 片内数据拷贝 snd->win
        CHK_RET(TaskOrchestrator::SelfCpySnd2Win(curInputPtr, curSize, 0, 0, HCCL_REDUCE_RESERVED, dataType));
        // 2. 前同步
        CHK_RET(TaskOrchestrator::DoPreSync());

        // 缺省情况allgather时op_type没用，融合时aic设置op_type为1时表示不需要本卡数据。
        if (opType != HCCL_REDUCE_PROD) {
            CHK_RET(TaskOrchestrator::SelfCpyWin2Rcv(
                curOutputPtr, curSize, 0, displs[ctx_->rankId], HCCL_REDUCE_RESERVED, dataType));
        }

        // 3. 跨片SDMA，其他win->当前rcv buff
        CHK_RET(
            TaskOrchestrator::IpcCpyWin2Rcv(curOutputPtr, curSize, nullptr, displs, HCCL_REDUCE_RESERVED, dataType));

        // 4. 后同步
        CHK_RET(TaskOrchestrator::DoPostSync());

        CHK_RET(TaskOrchestrator::LaunchTasks());

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

u64 AicpuAllgather::GetWindowOffset(u32 curTurnCnt, u64 curSize, u64 strideCnt, u64 recvBuffer)
{
    (void)recvBuffer;
    u64 windowOffset = 0;
    u64 bufferFlag = static_cast<uint64_t>(curTurnCnt % AC_DEFAULT_WINDOW_DIM);
    u64 recvOffset
        = (static_cast<uint64_t>(ctx_->rankId) * strideCnt * static_cast<uint64_t>(ctx_->unitSize)) % HCCL_COPY_ALIGN;
    u64 windowOffsetTmp
        = bufferFlag * (ctx_->windowSize / AC_DEFAULT_WINDOW_DIM / HCCL_COPY_ALIGN + 1) * HCCL_COPY_ALIGN + recvOffset;
    if ((bufferFlag == 0 && (recvOffset + curSize) < ctx_->windowSize / AC_DEFAULT_WINDOW_DIM)
        || (bufferFlag == 1 && (windowOffsetTmp + curSize) < ctx_->windowSize)) {
        windowOffset = windowOffsetTmp;
    } else {
        windowOffset = bufferFlag * (ctx_->windowSize / AC_DEFAULT_WINDOW_DIM);
    }
    return windowOffset;
}

HcclResult AicpuAllgather::RunAllGathervMC(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideCnt,
    AivAicpuOpParam* nextTask)
{
    u64 displs[AC_MAX_RANK_NUM] = {0};
    for (u32 i = 0; i < ctx_->rankNum; i++) {
        displs[i] = i * strideCnt * ctx_->unitSize;
    }

    u64 curSize = ctx_->unitSize * dataCount;
    u64 windowOffset = 0;
    if (curSize < ctx_->windowSize / AC_DEFAULT_WINDOW_DIM) {
        windowOffset = GetWindowOffset(ctx_->curTurnCnt, curSize, strideCnt, reinterpret_cast<uint64_t>(recvBuffer));
    }
    HCCL_INFO(
        "current task: windowSize[%llu], windowOffset[%llu], dataCount[%llu], "
        "curSize[%llu], strideCnt[%llu], gatherOut [%#llx]",
        ctx_->windowSize, windowOffset, dataCount, curSize, strideCnt, ctx_->gatherOut);

    // 1. 片内数据拷贝 snd->win
    if (isMCFirstCall) {
        CHK_RET(TaskOrchestrator::SelfCpySnd2Win(sendBuffer, curSize, 0, windowOffset, HCCL_REDUCE_RESERVED, dataType));
        isMCFirstCall = false;
    }

    // 2. 前同步
    CHK_RET(TaskOrchestrator::DoPreSync());

    // 缺省情况allgather时op_type没用，融合时aic设置op_type为1时表示不需要本卡数据。
    if (opType != HCCL_REDUCE_PROD) {
        CHK_RET(TaskOrchestrator::SelfCpyWin2Rcv(
            recvBuffer, curSize, windowOffset, displs[ctx_->rankId], HCCL_REDUCE_RESERVED, dataType));
    }

    // 3. 跨片SDMA，其他win->当前rcv buff
    u64 winOffsets[AC_MAX_RANK_NUM] = {0};
    for (u32 i = 0; i < ctx_->rankNum; i++) {
        winOffsets[i] = (ctx_->curTurnCnt % AC_DEFAULT_WINDOW_DIM)
                            * (ctx_->windowSize / AC_DEFAULT_WINDOW_DIM / HCCL_COPY_ALIGN + 1) * HCCL_COPY_ALIGN
                        + (i * strideCnt * ctx_->unitSize) % HCCL_COPY_ALIGN;
    }
    CHK_RET(TaskOrchestrator::IpcCpyWin2Rcv(recvBuffer, curSize, winOffsets, displs, HCCL_REDUCE_RESERVED, dataType));

    if (nextTask != nullptr) {
        // 1. 片内数据拷贝 snd->win
        u64 nextSize = DataUnitSize(nextTask->hcclDataType) * nextTask->count;
        u64 nextWindowOffset = 0;
        if (curSize < ctx_->windowSize / AC_DEFAULT_WINDOW_DIM) {
            nextWindowOffset = GetWindowOffset(ctx_->curTurnCnt + 1, nextSize, strideCnt, nextTask->recvBuffer);
        }

        HCCL_INFO(
            "curTurnCnt[%u], windowSize[%llu], windowOffset[%llu], dataCount[%llu], nextSize[%llu]", ctx_->curTurnCnt,
            ctx_->windowSize, nextWindowOffset, nextTask->count, nextSize);

        CHK_RET(TaskOrchestrator::SelfCpySnd2Win(
            reinterpret_cast<void*>(nextTask->sendBuffer), nextSize, 0, nextWindowOffset, HCCL_REDUCE_RESERVED,
            nextTask->hcclDataType));
    } else {
        isMCFirstCall = true;
    }

    // 4. 后同步
    CHK_RET(TaskOrchestrator::DoPostSync());

    CHK_RET(TaskOrchestrator::LaunchTasks());

    return HCCL_SUCCESS;
}

HcclResult AicpuAllgather::GenRingTask(
    [[maybe_unused]] HcclReduceOp opType, u64 sndAddr, u64 rcvAddr, u64 gatherSize, HcclDataType dataType,
    uint32_t streamId, bool isClockwise, uint32_t step, bool isWindowLast) const
{
    const u64 winIn = ctx_->rankInfo[rankId_].window + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const u64 winOut = ctx_->rankInfo[rankId_].windowOut + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const uint32_t preRankId = isClockwise ? (rankId_ + rankNum_ - 1U) % rankNum_ : (rankId_ + 1U) % rankNum_;
    const uint32_t postRankId = isClockwise ? (rankId_ + 1U) % rankNum_ : (rankId_ + rankNum_ - 1U) % rankNum_;
    const u64 preWinIn = ctx_->rankInfo[preRankId].window + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const u64 preWinOut = ctx_->rankInfo[preRankId].windowOut + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const bool evenStep = (step % 2U == 0U); // 2: 环上winIn winOut轮流接收, 奇数轮in->out, 偶数轮out->in

    HcclResult ret = HCCL_SUCCESS;
    if (step == 1U) { // 首轮send->winIn
        ret = AicpuDispatcher::CopyData(streamId, sndAddr, winIn, gatherSize, dataType, HCCL_REDUCE_RESERVED, rankId_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u send to block winIn failed", turn_, step), ret);
    }
    // 片间同步 notify后卡 wait前卡
    ret = AicpuDispatcher::SignalRecord(streamId, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u notify post rank failed", turn_, step), ret);
    ret = AicpuDispatcher::SignalWait(streamId, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u wait pre rank failed", turn_, step), ret);

    // 片间memcpy 前卡window->recv
    auto recvBufAddr = rcvAddr
                       + (isClockwise ? (rankId_ + rankNum_ - step) % rankNum_ : (rankId_ + step) % rankNum_)
                             * ctx_->totalCnt * unitSize_;
    ret = AicpuDispatcher::CopyData(
        streamId, evenStep ? preWinOut : preWinIn, recvBufAddr, gatherSize, dataType, HCCL_REDUCE_RESERVED, preRankId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u cpy pre window failed", turn_, step), ret);

    if (isClockwise) {
        if (!((step == rankNum_ - 1U) && isWindowLast)) {
            CHK_RET(AicpuDispatcher::SignalWait(
                streamId, (rankId_ + 1U) % rankNum_, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));
        }
        // ccore notify
        if ((step < rankNum_ - 1U) && isWindowLast) {
            ret = AicpuDispatcher::AddCcoreNotify(streamId, turn_ * (rankNum_ - 1U) + step);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add ccore notify failed", turn_, step), ret);
        }
    } else {
        if (!((step == rankNum_ - 1U) && isWindowLast)) {
            CHK_RET(AicpuDispatcher::SignalRecord(
                streamId, (rankId_ + 1U) % rankNum_, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));
        }
    }
    // recv->windowout
    if (step < rankNum_ - 1U) {
        ret = AicpuDispatcher::CopyData(
            streamId, recvBufAddr, evenStep ? winIn : winOut, gatherSize, dataType, HCCL_REDUCE_RESERVED, rankId_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u cpy pre window failed", turn_, step), ret);
    }

    // 片间同步 notify前卡 wait后卡
    ret = AicpuDispatcher::SignalRecord(streamId, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u notify pre rank failed", turn_, step), ret);
    ret = AicpuDispatcher::SignalWait(streamId, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u wait post rank failed", turn_, step), ret);
    return HCCL_SUCCESS;
}

HcclResult AicpuAllgather::RunDoubleRingAllGather(
    HcclReduceOp opType, u64 sendBuffer, u64 recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    const u64 gatherSize = dataCount * unitSize_;
    if (gatherSize > ctx_->windowSize / RING_NUM) {
        HCCL_INFO("Tile gather size %lu max less than window size %lu/RING_NUM", gatherSize, ctx_->windowSize);
    }
    const u64 rankDataSize = ctx_->totalCnt * unitSize_;
    thread_local static u64 sndAddr[RING_NUM] = {0UL};
    thread_local static u64 rcvAddr[RING_NUM] = {0UL};
    if (turn_ == 0U) { // 首个tile时初始化snd rcv地址
        sndAddr[0] = sendBuffer;
        sndAddr[1] = sendBuffer + rankDataSize;
        rcvAddr[0] = recvBuffer;
        rcvAddr[1] = recvBuffer + rankDataSize;
    }

    HCCL_INFO(
        "DR AllGather snd addr %p %p, rcv addr %p %p, size %lu", sndAddr[0], sndAddr[1], rcvAddr[0], rcvAddr[1],
        gatherSize);
    uint32_t mainStream = rankId_;
    uint32_t subStream = (rankId_ + 1U) % rankNum_;
    HcclResult ret = HCCL_SUCCESS;

    u64 maxCountPerLoop = ctx_->windowSize / RING_NUM / unitSize_; // 中转内存单次最多能够接受的output count
    u64 countLeft = dataCount;
    bool isWindowFirst = true;
    bool isWindowLast = false;
    // windowSize循环
    u32 loopIdx = 0;
    while (countLeft > 0) {
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize_; // 单位 byte

        HCCL_DEBUG(
            "DR AllGather: loop %u, snd addr[%u %u], rcv addr[%u %u], curCount[%llu], curSize[%llu]", loopIdx++,
            sndAddr[0], sndAddr[1], rcvAddr[0], rcvAddr[1], curCount, curSize);

        sndAddr[1] -= curSize;
        rcvAddr[1] -= curSize; // 逆时针输出在执行前向上偏移

        isWindowLast = ((countLeft - curCount) == 0);

        // 顺、逆时针各平移ranknum-1次
        for (uint32_t step = 1U; step <= rankNum_ - 1U; step++) {
            if (isWindowFirst) {
                // ccore wait
                u64 waitAddr = ctx_->workSpaceAddr + ctx_->notifyOff + offsetof(AivAicpuOpParam, sendCnt);
                ret = AicpuDispatcher::AddCcoreWait(
                    mainStream, waitAddr, turn_ * (rankNum_ - 1U) + step,
                    (turn_ + 1U >= ctx_->totalTurnCnt) && (step == rankNum_ - 1U));
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add ccore wait failed", turn_, step), ret);
            }

            // 主->从
            CHK_RET(AicpuDispatcher::SignalRecord(
                mainStream, subStream, AicpuDispatcher::NO_IPC, AicpuDispatcher::PRE_SYNC));
            CHK_RET(
                AicpuDispatcher::SignalWait(subStream, subStream, AicpuDispatcher::NO_IPC, AicpuDispatcher::PRE_SYNC));

            // 顺时针环
            CHK_RET(
                GenRingTask(opType, sndAddr[0], rcvAddr[0], curSize, dataType, mainStream, true, step, isWindowLast));

            // 逆时针环
            CHK_RET(
                GenRingTask(opType, sndAddr[1], rcvAddr[1], curSize, dataType, subStream, false, step, isWindowLast));

            // 缺省情况allgather时op_type没用，融合时aic设置op_type为1时表示不需要本卡数据。
            if ((step == rankNum_ - 1U) && isWindowLast) { // 拷贝全量本卡直接从send->recv
                if (opType != HCCL_REDUCE_PROD) {
                    ret = AicpuDispatcher::CopyData(
                        mainStream, sendBuffer, recvBuffer + rankId_ * rankDataSize, rankDataSize, dataType,
                        HCCL_REDUCE_RESERVED, rankId_);
                    CHK_PRT_RET(
                        ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u send to recv winIn failed", turn_, step), ret);
                }

                // 从->主
                CHK_RET(AicpuDispatcher::SignalRecord(
                    subStream, subStream, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));
                CHK_RET(AicpuDispatcher::SignalWait(
                    mainStream, subStream, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));

                ret = TaskOrchestrator::AddBarrier(mainStream, rankId_, rankNum_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add barrier failed.", turn_, step), ret);

                // ccore notify 非step7由GenRingTask处理AddCcoreNotify
                ret = AicpuDispatcher::AddCcoreNotify(mainStream, turn_ * (rankNum_ - 1U) + step);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add ccore notify failed", turn_, step), ret);
            }
        }
        isWindowFirst = false;
        countLeft -= curCount;
        sndAddr[0] += curSize;
        rcvAddr[0] += curSize; // 顺时针输出在执行后向下偏移
    }

    ret = TaskOrchestrator::LaunchTasks();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Launch tasks failed"), ret);
    return HCCL_SUCCESS;
}
