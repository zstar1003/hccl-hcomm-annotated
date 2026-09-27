/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_reduce_scatter.h"

namespace {
template <typename T>
inline T MathCeil(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline T AlignUp(T num1, T num2)
{
    return MathCeil(num1, num2) * num2;
}
} // namespace

HcclResult AicpuReduceScatter::RunAlgorithm(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen,
    AivAicpuOpParam* /* nextTask */)
{
    CHK_PTR_NULL(ctx_);
    // 通信域 rankNum 为 0 时，后续取模/除法会除零，作为非法入参直接返回
    if (rankNum_ == 0) {
        HCCL_ERROR("Reduce scatter rankNum_ is 0, division by zero.");
        return HCCL_E_PARA;
    }
    // dataCount 为tile的输入的数据量（scatter前）
    if (dataCount % rankNum_ != 0) { // 每个tile数据量必须能均分至每张卡
        HCCL_ERROR("Reduce scatter dataCount %lu must be a multiple of rankNum %u.", dataCount, rankNum_);
        return HCCL_E_NOT_SUPPORT;
    }

    switch (ctx_->commAlg) {
        case CommAlgType::COMM_ALG_FULL_MESH: {
            if (ctx_->determinism) {
                return RunDeterministicReduceScatterLocal(
                    opType, sendBuffer, recvBuffer, dataCount, dataType, strideLen);
            }
            return RunReduceScatterWriteMode(opType, sendBuffer, recvBuffer, dataCount, dataType, strideLen);
        }
        case CommAlgType::COMM_ALG_DOUBLE_RING: {
            return RunDoubleRingReduceScatter(
                opType, reinterpret_cast<u64>(sendBuffer), reinterpret_cast<u64>(recvBuffer), dataCount, dataType);
        }
        case CommAlgType::COMM_ALG_SWITCH_WING: {
            return RunSwitchReduceScatter(
                opType, reinterpret_cast<u64>(sendBuffer), reinterpret_cast<u64>(recvBuffer), dataCount, dataType);
        }
        default: {
            HCCL_ERROR("CommAlg %d is not supported.", ctx_->commAlg);
            return HCCL_E_NOT_SUPPORT;
        }
    }
}

HcclResult AicpuReduceScatter::RunDeterministicReduceScatterLocal(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen)
{
    u8* curInputPtr = static_cast<u8*>(sendBuffer);
    u8* curOutputPtr = static_cast<u8*>(recvBuffer);
    u64 windowSize = ctx_->windowSize;
    u64 countLeft = dataCount / rankNum_;
    u64 maxCountPerLoop = windowSize / unitSize_ / rankNum_;
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u32 loopIdx = 0;
    u64 displs[AC_MAX_RANK_NUM] = {0};
    u64 windowSlices = rankId_ * countLeft * unitSize_;

    for (u32 i = 0; i < ctx_->rankNum; i++) {
        displs[i] = strideLen * i * unitSize_;
    }

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize_; // 单位 byte
        windowSlices = rankId_ * curSize;

        HCCL_DEBUG(
            "RunDeterministicReduceScatterLocal: countLeft = %llu, loop %u, curInputPtr[%p], curOutputPtr[%p],"
            "curCount[%llu], curSize[%llu], strideLen[%llu]",
            countLeft, loopIdx++, curInputPtr, curOutputPtr, curCount, curSize, strideLen);
        // 1. 片内数据拷贝：send->win
        TaskOrchestrator::SelfCpySnd2Win(
            curInputPtr, curSize, displs[rankId_], windowSlices, HCCL_REDUCE_RESERVED, dataType);

        // 2. 前同步
        TaskOrchestrator::DoPreSync();

        // 3. 跨片SDMA，send->对端win
        TaskOrchestrator::IpcCpySnd2Win(curInputPtr, curSize, displs, windowSlices, HCCL_REDUCE_RESERVED, dataType);

        // 4. 后同步
        TaskOrchestrator::DoPostSync();

        // 5. 折半计算
        TaskOrchestrator::SelfLocalReduce(curSize, opType, dataType);

        // 6. 片内数据拷贝 本端win->当前rcv buff
        TaskOrchestrator::SelfCpyWin2Rcv(curOutputPtr, curSize, 0, 0, HCCL_REDUCE_RESERVED, dataType);

        TaskOrchestrator::LaunchTasks();

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuReduceScatter::RunReduceScatterWriteMode(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen)
{
    dataCount /= rankNum_;

    u64 windowSize = ctx_->windowSize;
    u64 maxCountPerLoop = windowSize / unitSize_; // 中转内存单次最多能够接受的output count

    uint8_t* curInputPtr = static_cast<uint8_t*>(sendBuffer);
    uint8_t* curOutputPtr = static_cast<uint8_t*>(recvBuffer);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = dataCount;

    u64 displs[AC_MAX_RANK_NUM] = {0};
    u64 windowOffsets[AC_MAX_RANK_NUM] = {0};
    for (u32 i = 0; i < rankNum_; i++) {
        displs[i] = i * strideLen * unitSize_;
        windowOffsets[i] = (i * strideLen * unitSize_) % HCCL_COPY_ALIGN;
    }

    uint32_t loopIdx = 0;
    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize_; // 单位 byte

        HCCL_DEBUG(
            "RunReduceScatterWriteMode: loop %u, curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu], "
            "strideLen[%llu]",
            loopIdx++, curInputPtr, curOutputPtr, curCount, curSize, strideLen);

        // 1. 片内数据 snd->win
        CHK_RET(TaskOrchestrator::SelfCpySnd2Win(
            curInputPtr, curSize, displs[rankId_], windowOffsets[rankId_], HCCL_REDUCE_RESERVED, dataType)); // 1, 0
        // 2. 前同步
        CHK_RET(TaskOrchestrator::DoPreSync()); // 15 sqe, 35

        // 3. 跨片SDMA send->其他window
        CHK_RET(TaskOrchestrator::IpcCpySnd2Win(curInputPtr, curSize, displs, windowOffsets, opType,
                                                dataType)); // 0, 7

        // 4. 后同步
        CHK_RET(TaskOrchestrator::DoPostSync()); // 8, 21

        // 5. 片内数据 拷贝到recv
        CHK_RET(TaskOrchestrator::SelfCpyWin2Rcv(
            curOutputPtr, curSize, windowOffsets[rankId_], 0, HCCL_REDUCE_RESERVED,
            dataType)); // 1, 0

        CHK_RET(TaskOrchestrator::LaunchTasks()); // 25, 63

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuReduceScatter::RunReduceScatterReadMode(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType)
{
    uint32_t dataSize = dataCount * unitSize_; // 输入大小
    u64 scatterSize = dataSize / rankNum_;     // 输出大小

    HCCL_DEBUG(
        "RunReduceScatterReadMode: sendBuffer[%p], recvBuffer[%p], dataCount[%llu], dataSize[%llu]", sendBuffer,
        recvBuffer, dataCount, dataSize);

    // 1. 片内数据 snd->win 一次性全拷贝
    CHK_RET(TaskOrchestrator::SelfCpySnd2Win(sendBuffer, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType));
    // 2. 前同步
    CHK_RET(TaskOrchestrator::DoPreSync()); // 15 sqe, 35

    // 3. 片内win->recv   实测放在和跨片sdma并行时性能更好
    CHK_RET(TaskOrchestrator::SelfCpyWin2Rcv(
        recvBuffer, scatterSize, rankId_ * scatterSize, 0, HCCL_REDUCE_RESERVED, dataType));

    // 3. 跨片SDMA 其他win->recv
    u64 winOffsets[AC_MAX_RANK_NUM] = {0};
    for (size_t i = 0; i < rankNum_; i++) {
        winOffsets[i] = rankId_ * scatterSize;
    }
    CHK_RET(TaskOrchestrator::IpcCpyWin2Rcv(recvBuffer, scatterSize, winOffsets, nullptr, opType, dataType));

    // 4. 后同步
    CHK_RET(TaskOrchestrator::DoPostSync()); // 8 sqe, 21

    CHK_RET(TaskOrchestrator::LaunchTasks()); // 25, 63

    return HCCL_SUCCESS;
}

std::vector<Slice> AicpuReduceScatter::PrepareMeshSlice(u64 dataSize, uint32_t rankNum_)
{
    std::vector<Slice> meshSlices;
    uint32_t roundCnt = rankNum_ - 1;

    u64 sizePerRound = AlignUp<u64>((dataSize + roundCnt - 1) / roundCnt, HCCL_MIN_SLICE_ALIGN);
    int64_t rankResidueSize = static_cast<int64_t>(dataSize);

    while (rankResidueSize > 0) {
        Slice singleRoundSlice;
        singleRoundSlice.offset = dataSize - rankResidueSize;
        singleRoundSlice.size = std::min<u64>(sizePerRound, rankResidueSize);
        rankResidueSize -= singleRoundSlice.size;
        meshSlices.push_back(singleRoundSlice);
    }

    for (const auto& slice : meshSlices) {
        HCCL_DEBUG("Slice offset:%lu, size:%lu", slice.offset, slice.size);
    }
    return meshSlices;
}

HcclResult AicpuReduceScatter::RunDeterministicReduceScatter(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideCount)
{
    HCCL_INFO("RunDeterministicReduceScatter strideCount:%u", strideCount);
    dataCount /= rankNum_;

    u64 maxCountPerLoop = ctx_->windowSize / (rankNum_ * unitSize_); // 中转内存单次最多能够接受的output count
    u64 curCount = 0;
    uint8_t* curInputPtr = static_cast<uint8_t*>(sendBuffer);
    uint8_t* curOutputPtr = static_cast<uint8_t*>(recvBuffer);
    for (u64 countLeft = dataCount, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        curCount = ((countLeft * unitSize_ * rankNum_) > ctx_->windowSize) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize_;

        CHK_RET(RunCurrentDeterministicReduceScatter(
            opType, curInputPtr, curOutputPtr, strideCount * unitSize_, curSize, dataType));

        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuReduceScatter::RunCurrentDeterministicReduceScatter(
    HcclReduceOp opType, uint8_t* curInputPtr, uint8_t* curOutputPtr, u64 strideSize, u64 curSize,
    HcclDataType dataType)
{
    std::vector<Slice> slices = PrepareMeshSlice(curSize, rankNum_);
    std::vector<uint32_t> srcRankOrder;
    std::vector<uint32_t> dstRankOrder;
    for (uint32_t i = 1; i <= rankNum_ - 1; i++) {
        srcRankOrder.push_back((rankId_ + rankNum_ - i) % rankNum_);
        dstRankOrder.push_back((rankId_ + i) % rankNum_);
        HCCL_DEBUG("SrcRank:%u dstRank:%u", (rankId_ + rankNum_ - i) % rankNum_, (rankId_ + i) % rankNum_);
    }
    // 1. 片内send->recv
    AicpuDispatcher::CopyData(
        rankId_, static_cast<void*>(curInputPtr + rankId_ * strideSize), curOutputPtr, curSize, dataType,
        HCCL_REDUCE_RESERVED, rankId_);
    for (uint32_t round = 0; round < rankNum_ - 1; round++) {
        // 2. 片内send->win
        for (size_t i = 0; i < slices.size(); i++) {
            uint32_t idx = (round + i) % (rankNum_ - 1);
            u64 sendOff = dstRankOrder[idx] * strideSize + slices[i].offset;
            HCCL_DEBUG(
                "Cpy send to win, dstRank:%u, srcOffset:%lu, dstOffset:%lu, size:%lu", dstRankOrder[idx], sendOff,
                slices[i].offset, slices[i].size);
            CHK_RET(TaskOrchestrator::SelfCpySnd2Win(
                curInputPtr, slices[i].size, sendOff, slices[i].offset, HCCL_REDUCE_RESERVED, dataType));
        }
        // 3. 前同步
        CHK_RET(TaskOrchestrator::DoPreSync());
        // 4. 跨片读 inline reduce
        for (size_t i = 0; i < slices.size(); i++) {
            uint32_t idx = (round + i) % (rankNum_ - 1);
            HCCL_DEBUG(
                "Ipc read win to rcv, srcRank:%u, srcOffset:%lu, dstOffset:%lu, size:%lu", srcRankOrder[idx],
                slices[i].offset, slices[i].offset, slices[i].size);
            CHK_RET(TaskOrchestrator::IpcCpyWin2RcvP2P(
                curOutputPtr, srcRankOrder[idx], slices[i].size, slices[i].offset, slices[i].offset, opType, dataType));
        }
        // 5. 后同步
        CHK_RET(TaskOrchestrator::DoPostSync());
    }
    CHK_RET(TaskOrchestrator::LaunchTasks());
    return HCCL_SUCCESS;
}

HcclResult AicpuReduceScatter::GenRingTask(
    HcclReduceOp opType, u64 sndAddr, u64 rcvAddr, u64 curSize, u64 scatterSize, HcclDataType dataType,
    uint32_t streamId, bool isClockwise, uint32_t step) const
{
    const u64 winIn = ctx_->rankInfo[rankId_].window + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const u64 winOut = ctx_->rankInfo[rankId_].windowOut + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const uint32_t preRankId = isClockwise ? (rankId_ + rankNum_ - 1U) % rankNum_ : (rankId_ + 1) % rankNum_;
    const uint32_t postRankId = isClockwise ? (rankId_ + 1) % rankNum_ : (rankId_ + rankNum_ - 1U) % rankNum_;
    const u64 preWinIn = ctx_->rankInfo[preRankId].window + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const u64 preWinOut = ctx_->rankInfo[preRankId].windowOut + (isClockwise ? 0U : ctx_->windowSize / RING_NUM);
    const bool evenStep = (step % 2U == 0U); // 2: 环上winIn winOut轮流接收, 奇数轮in->out, 偶数轮out->in

    HcclResult ret = HCCL_SUCCESS;
    if (step == 1U) { // 首轮send->winIn winOut
        ret = AicpuDispatcher::CopyData(streamId, sndAddr, winIn, curSize, dataType, HCCL_REDUCE_RESERVED, rankId_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u send to block winIn failed", turn_, step), ret);

        ret = AicpuDispatcher::CopyData(
            streamId, sndAddr + scatterSize, winOut, curSize, dataType, HCCL_REDUCE_RESERVED, rankId_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u send to clock winOut failed", turn_, step), ret);
    } else {
        ret = AicpuDispatcher::CopyData(
            streamId, sndAddr, evenStep ? winIn : winOut, curSize, dataType, HCCL_REDUCE_RESERVED, rankId_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u send to clock winOut failed", turn_, step), ret);
    }
    // 片间同步 notify后卡 wait前卡
    ret = AicpuDispatcher::SignalRecord(streamId, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u notify post rank failed", turn_, step), ret);

    ret = AicpuDispatcher::SignalWait(streamId, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u wait pre rank failed", turn_, step), ret);

    // 片间memcpy 前卡window->window
    ret = AicpuDispatcher::CopyData(
        streamId, evenStep ? preWinOut : preWinIn, evenStep ? winIn : winOut, curSize, dataType, opType, preRankId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u cpy pre window failed", turn_, step), ret);
    if (step == rankNum_ - 1U) { // 最后一轮 win输出至recv
        ret = AicpuDispatcher::CopyData(
            streamId, evenStep ? winIn : winOut, rcvAddr, curSize, dataType, HCCL_REDUCE_RESERVED, rankId_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u window to recv failed", turn_, step), ret);
    }
    ret = AicpuDispatcher::SignalRecord(streamId, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u notify pre rank failed", turn_, step), ret);
    // 片间同步 notify前卡 wait后卡
    ret = AicpuDispatcher::SignalWait(streamId, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u wait post rank failed", turn_, step), ret);
    return HCCL_SUCCESS;
}

HcclResult AicpuReduceScatter::RunDoubleRingReduceScatter(
    HcclReduceOp opType, u64 sendBuffer, u64 recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    const u64 scatterSize = dataCount / rankNum_ * unitSize_;
    if (scatterSize > ctx_->windowSize / RING_NUM) {
        HCCL_INFO("Tile scatter size %lu max less than window size %lu/RING_NUM", scatterSize, ctx_->windowSize);
    }
    thread_local static u64 sndAddr[RING_NUM] = {0UL};
    thread_local static u64 rcvAddr[RING_NUM] = {0UL};
    if (turn_ == 0U) { // 首个tile时初始化snd rcv地址
        sndAddr[0] = sendBuffer;
        rcvAddr[0] = recvBuffer;
        rcvAddr[1] = recvBuffer + ctx_->totalCnt * unitSize_ / rankNum_;
    }
    sndAddr[1] = sndAddr[0] + 2U * scatterSize + scatterSize; // 2 首轮需要同时输出到winIn和winOut

    HCCL_INFO(
        "DR reducescatter snd addr %p %p, rcv addr %p %p, size %lu", sndAddr[0], sndAddr[1], rcvAddr[0], rcvAddr[1],
        scatterSize);
    uint32_t mainStream = rankId_;
    uint32_t subStream = (rankId_ + 1U) % rankNum_;
    HcclResult ret = HCCL_SUCCESS;

    u64 maxCountPerLoop = ctx_->windowSize / RING_NUM / unitSize_; // 中转内存单次最多能够接受的output count
    u64 countLeft = dataCount / rankNum_;
    bool isWindowFirst = true;
    bool isWindowLast = false;
    // windowSize循环
    u32 loopIdx = 0;
    while (countLeft > 0) {
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize_; // 单位 byte

        HCCL_DEBUG(
            "DR reducescatter: loop %u, snd addr[%p %p], rcv addr[%p %p], curCount[%llu], curSize[%llu]", loopIdx++,
            sndAddr[0], sndAddr[1], rcvAddr[0], rcvAddr[1], curCount, curSize);

        sndAddr[1] -= curSize; // 逆时针输入在执行前向上偏移
        rcvAddr[1] -= curSize; // 逆时针输出在执行前向上偏移

        isWindowLast = ((countLeft - curCount) == 0);

        u64 stepSndAddrClockwise = sndAddr[0];
        u64 stepSndAddrAnticlockwise = sndAddr[1];
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
            CHK_RET(GenRingTask(
                opType, stepSndAddrClockwise, rcvAddr[0], curSize, scatterSize, dataType, mainStream, true, step));
            stepSndAddrClockwise
                += ((step == 1U) ? 2U * scatterSize * RING_NUM :
                                   scatterSize * RING_NUM); // 2 首轮需要同时输出到winIn和winOut

            // 逆时针环
            CHK_RET(GenRingTask(
                opType, stepSndAddrAnticlockwise, rcvAddr[1], curSize, scatterSize, dataType, subStream, false, step));
            stepSndAddrAnticlockwise
                += ((step == 1U) ? 2U * scatterSize + scatterSize :
                                   scatterSize * RING_NUM); // 2 首轮需要同时输出到winIn和winOut

            // 从->主
            CHK_RET(AicpuDispatcher::SignalRecord(
                subStream, subStream, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(
                mainStream, subStream, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));
            if (isWindowLast) {
                ret = TaskOrchestrator::AddBarrier(mainStream, rankId_, rankNum_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add barrier failed", turn_, step), ret);
                // ccore notify
                ret = AicpuDispatcher::AddCcoreNotify(mainStream, turn_ * (rankNum_ - 1U) + step);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add ccore notify failed", turn_, step), ret);
            }
        }
        isWindowFirst = false;
        countLeft -= curCount;
        sndAddr[0] += curSize; // 顺时针输入向下偏移
        rcvAddr[0] += curSize; // 顺时针输出向下偏移
    }

    // tile之间回退一个scatterSize再偏移
    sndAddr[0] = sndAddr[0] - scatterSize + rankNum_ * RING_NUM * scatterSize;

    ret = TaskOrchestrator::LaunchTasks();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Launch tasks failed"), ret);
    return HCCL_SUCCESS;
}

HcclResult AicpuReduceScatter::RunSwitchReduceScatter(
    HcclReduceOp opType, u64 sendBuffer, u64 recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    const u64 scatterSize = dataCount / rankNum_ * unitSize_;
    if (scatterSize > ctx_->windowSize) {
        HCCL_INFO("Tile scatter size %lu max less than window size %lu", scatterSize, ctx_->windowSize);
    }
    thread_local static u64 sndAddr = 0UL;
    thread_local static u64 rcvAddr = 0UL;
    if (turn_ == 0U) { // 首个tile时初始化snd rcv地址
        sndAddr = sendBuffer;
        rcvAddr = recvBuffer;
    }
    uint32_t mainStream = rankId_;
    HCCL_INFO("SW reducescatter snd addr %p, rcv addr %p, size %lu", sndAddr, rcvAddr, scatterSize);

    HcclResult ret = HCCL_SUCCESS;
    u64 maxCountPerLoop = ctx_->windowSize / unitSize_; // 中转内存单次最多能够接受的output count
    u64 countLeft = dataCount / rankNum_;
    bool isWindowFirst = true;
    bool isWindowLast = false;
    // windowSize循环
    while (countLeft > 0) {
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize_; // 单位 byte

        isWindowLast = ((countLeft - curCount) == 0);

        u64 stepSndAddr = sndAddr;
        for (uint32_t step = 1U; step <= rankNum_; step++) {
            uint32_t preRankId = (rankId_ + rankNum_ - step) % rankNum_;
            uint32_t postRankId = (rankId_ + step) % rankNum_;
            if (isWindowFirst) {
                // ccore wait
                u64 waitAddr = ctx_->workSpaceAddr + ctx_->notifyOff + offsetof(AivAicpuOpParam, sendCnt);
                ret = AicpuDispatcher::AddCcoreWait(
                    mainStream, waitAddr, turn_ * rankNum_ + step,
                    (turn_ + 1u >= ctx_->totalTurnCnt) && (step == rankNum_));
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add ccore wait failed", turn_, step), ret);
            }

            if (step != rankNum_) {
                // 片间同步 notify前卡 wait后卡
                ret = AicpuDispatcher::SignalRecord(
                    mainStream, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u notify pre rank failed", turn_, step), ret);

                ret = AicpuDispatcher::SignalWait(
                    mainStream, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u wait post rank failed", turn_, step), ret);

                // 片间memcpy snd->后序卡win 首轮采用覆盖
                ret = AicpuDispatcher::CopyData(
                    mainStream, stepSndAddr, ctx_->rankInfo[postRankId].window, curSize, dataType,
                    step == 1U ? HCCL_REDUCE_RESERVED : opType, postRankId);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u cpy snd to post win failed", turn_, step), ret);
                stepSndAddr += scatterSize;

                // 片间同步 notify后卡 wait前卡
                ret = AicpuDispatcher::SignalRecord(
                    mainStream, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u notify post rank failed", turn_, step), ret);

                ret = AicpuDispatcher::SignalWait(
                    mainStream, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u wait pre rank failed", turn_, step), ret);
            } else { // 最后一轮仅作本卡内拷贝
                // snd->recv
                ret = AicpuDispatcher::CopyData(
                    mainStream, stepSndAddr, rcvAddr, curSize, dataType, HCCL_REDUCE_RESERVED, rankId_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u cpy snd to rcv failed", turn_, step), ret);
                stepSndAddr += scatterSize;

                // win->recv
                ret = AicpuDispatcher::CopyData(
                    mainStream, ctx_->rankInfo[rankId_].window, rcvAddr, curSize, dataType, opType, rankId_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u cpy win to rcv failed", turn_, step), ret);
            }
            if (isWindowLast) {
                ret = TaskOrchestrator::AddBarrier(mainStream, rankId_, rankNum_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add barrier failed", turn_, step), ret);

                // ccore notify
                ret = AicpuDispatcher::AddCcoreNotify(mainStream, turn_ * rankNum_ + step);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS, HCCL_ERROR("Turn %u step %u add ccore notify failed", turn_, step), ret);
            }
        }
        isWindowFirst = false;
        countLeft -= curCount;
        sndAddr += curSize;
        rcvAddr += curSize;
    }

    // 回退一个scatterSize再偏移
    sndAddr = sndAddr - scatterSize + rankNum_ * scatterSize;

    ret = TaskOrchestrator::LaunchTasks();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Launch tasks failed"), ret);
    return HCCL_SUCCESS;
}
