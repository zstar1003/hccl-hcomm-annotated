/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kfc_rpc_serverv2.h"

#include <numeric>
#include "log_control.h"
#include "common/aicpu_hccl_common.h"
#include "utils/hccl_aicpu_utils.h"
#include "common/aicpu_kfc_utils.h"
#include "framework/aicpu_kfc_prof.h"

using namespace HcclApi;
static constexpr uint16_t TURN_LEFT_SHIFT_BIT = 16;

HcclResult
AicpuKfcRpcServerV2::Init(const HcclMC2WorkSpace& workspaceInfo, const HcclApi::Mc2InitTilingInner* tilingData)
{
    // 为提升效率，workspace 必须512 对齐
    u64 addr = workspaceInfo.workSpace;
    if (addr & 0x1ff) {
        addr = (addr & (~((uint64_t)0x1ff))) + 0x200;
    }
    Reset();
    blockNum_ = HcclAicpuUtils::GetBlockNum();
    CHK_PRT_RET(blockNum_ == 0U, HCCL_ERROR("Invalid block number."), HCCL_E_INTERNAL);
    HCCL_INFO(
        "Align hcclmsgarea from %p to %p, block number %u, current block idx %u.", workspaceInfo.workSpace, addr,
        blockNum_, HcclAicpuUtils::GetBlockIdx());
    if (tilingData != nullptr && tilingData->queueNum > 0U) {
        totalQueueNum_ = tilingData->commBlockNum * tilingData->queueNum;
        CHK_PRT_RET(
            totalQueueNum_ > LOCAL_STREAM_MAX_NUM || blockNum_ > std::min(MAX_AICPU_NUM_BLOCKS, totalQueueNum_),
            HCCL_ERROR(
                "Invalid para, comm block %u, aicpu block %u, queue number %u.", tilingData->commBlockNum, blockNum_,
                tilingData->queueNum),
            HCCL_E_INTERNAL);
    } else {
        totalQueueNum_ = 0U;
    }
    hcclMsgArea_ = reinterpret_cast<HcclMsgArea*>(addr);
    turnNumAddr_ = addr + sizeof(HcclMsgArea);
    if (turnNumAddr_ + sizeof(u32) * TILING_TURN_MAX * HCCL_MAX_RANK_NUM_V2
        > workspaceInfo.workSpace + workspaceInfo.workSpaceSize) {
        HCCL_ERROR(
            "Turn number addr %#llx, space for turn number is %lu, the space after workspace %#llx will "
            "be overwritten.",
            turnNumAddr_, sizeof(u32) * TILING_TURN_MAX * HCCL_MAX_RANK_NUM_V2,
            workspaceInfo.workSpace + workspaceInfo.workSpaceSize);
        return HCCL_E_INTERNAL;
    }
    uint32_t* turnNums = reinterpret_cast<uint32_t*>(turnNumAddr_);
    std::iota(&turnNums[0], &turnNums[TILING_TURN_MAX * HCCL_MAX_RANK_NUM_V2], 0);
    tilingBaseAddr_ = reinterpret_cast<u64>(tilingData);
    return HCCL_SUCCESS;
}

void AicpuKfcRpcServerV2::GetLocalQueueRange(u32& start, u32& end)
{
    if (blockNum_ == 0U || totalQueueNum_ == 0U) {
        start = end = 0U;
        return;
    }
    const u32 base = totalQueueNum_ / blockNum_;
    const u32 remainder = totalQueueNum_ % blockNum_;
    const u32 blockIdx = HcclAicpuUtils::GetBlockIdx();
    if (blockIdx < remainder) {
        start = blockIdx * base + blockIdx;
        end = start + base;
    } else {
        start = blockIdx * base + remainder;
        end = start + base - 1U;
    }
}

void AicpuKfcRpcServerV2::Reset()
{
    (void)memset_s(msgPos_, sizeof(msgPos_), 0, sizeof(msgPos_));
    msgPosForKernel_ = 0;
    for (int8_t i = 0; i < HCCL_MAX_HANDLE_ID; i++) {
        handleIdMsgPosition_[i] = -1;
    }
    (void)memset_s(isFinalize_, sizeof(isFinalize_), 0, sizeof(isFinalize_));
    (void)memset_s(barrierFlags_, sizeof(barrierFlags_), 0, sizeof(barrierFlags_));
    (void)memset_s(barrierFinishCnt_, sizeof(barrierFinishCnt_), 0, sizeof(barrierFinishCnt_));
    const u64 ts = GetCurCpuTimestamp();
    for (u32 i = 0U; i < MAX_QUE_NUM; ++i) {
        prepareTime_[i] = ts;
    }
    eventPrintTurn_ = 1U;
}

void AicpuKfcRpcServerV2::SetMsgHandlePos(uint32_t msgPos, HcclHandle handleId)
{
    if (handleId >= HCCL_MAX_HANDLE_ID || handleId < 0) {
        return;
    }
    handleIdMsgPosition_[handleId] = msgPos;
}

int32_t AicpuKfcRpcServerV2::GetMsgHandlePos(HcclHandle handleId)
{
    if (handleId >= HCCL_MAX_HANDLE_ID || handleId < 0) {
        HCCL_ERROR("[GetMsgHandlePos] invalid handleId %d", handleId);
        return -1;
    }
    if (handleIdMsgPosition_[handleId] < 0) {
        HCCL_WARNING("[GetMsgHandlePos] invalid handleIdMsgPosition %d", handleIdMsgPosition_[handleId]);
        return -1;
    }
    return handleIdMsgPosition_[handleId];
}

bool AicpuKfcRpcServerV2::IsPrintLog() const { return isPrintLog_; }

bool AicpuKfcRpcServerV2::GetIsFinalize(u32 queueId)
{
    if (queueId < MAX_QUE_NUM) {
        return isFinalize_[queueId];
    }
    u32 start = 0U;
    u32 end = 0U;
    GetLocalQueueRange(start, end);
    for (u32 i = start; i <= end; ++i) {
        if (!isFinalize_[i]) {
            return false;
        }
    }
    return true;
}

void AicpuKfcRpcServerV2::SetIsFinalize(u32 queueId, bool finalize) { isFinalize_[queueId] = finalize; }

HcclMsgExt* AicpuKfcRpcServerV2::GetHcclMsgExtPtr() { return msgExt_.get(); }

HcclMsgArea* AicpuKfcRpcServerV2::GetHcclMsgArea(void) { return hcclMsgArea_; }

HcclMsg (*AicpuKfcRpcServerV2::GetMsgWorkSpace())[HCCL_MSG_CNT]
{
    if (totalQueueNum_ == 0U) {
        return &(hcclMsgArea_->commMsg.singleMsg.sendMsgs);
    } else {
        return hcclMsgArea_->commMsg.multiMsg.sendMsgs;
    }
}

uint64_t AicpuKfcRpcServerV2::GetFinishAddr(int32_t idx) const
{
    if (idx >= static_cast<int32_t>(HCCL_MSG_CNT) || hcclMsgArea_ == nullptr) {
        HCCL_ERROR("idx %d exceeds max idx %u or msg area is not initialized.", idx, HCCL_MSG_CNT);
        return 0;
    }
    return reinterpret_cast<uint64_t>(&(hcclMsgArea_->commMsg.singleMsg.finishedTurnCnt[idx].cnt));
}

uint64_t AicpuKfcRpcServerV2::GetCommitareaAddr(int32_t idx) const
{
    if (idx >= static_cast<int32_t>(HCCL_MSG_CNT) || hcclMsgArea_ == nullptr) {
        HCCL_ERROR("idx %d exceeds max idx %u or hcclMsgArea_ is not initialized.", idx, HCCL_MSG_CNT);
        return 0;
    }
    return reinterpret_cast<uint64_t>(&(hcclMsgArea_->commMsg.singleMsg.commitTurnCnt[idx].cnt));
}

HcclResult AicpuKfcRpcServerV2::AddFlipTask(HcclDispatcher dispatcherPtr, hccl::Stream* stream)
{
    if (!dfx::ProfilingManager::GetProfL0State()) {
        return HCCL_SUCCESS;
    }
    hccl::HcclSqeContext* sqeCtx = stream->GetSqeContextPtr();
    CHK_PTR_NULL(sqeCtx);
    hccl::SqeRingBuffer& buff = sqeCtx->buffer;
    // nextTaskId=0的时候下发PlaceHolder
    if (UNLIKELY(buff.tailSqeTaskId == 0 && buff.filpNum != 0)) {
        CHK_RET(AddRetryPreamble(dispatcherPtr, *stream));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuKfcRpcServerV2::AddCcoreWait(
    HcclDispatcher dispatcherPtr, u64 waitAddr, uint32_t turnNum, hccl::Stream* stream,
    bool isLast) // client commit wait
{
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;

    CHK_RET(AddFlipTask(dispatcherPtr, stream));
    CHK_RET(stream->GetNextSqeBufferAddr(sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    const HcclComStreamInfo& streamInfo = stream->GetHcclStreamInfo();
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_COMMIT_TIMEOUT)) {
        uint32_t* turnNums = reinterpret_cast<uint32_t*>(turnNumAddr_);
        turnNums[turnNum] = 0xFF;
    }
    AddOneWaitStartSqe(
        streamInfo.actualStreamId, taskId, waitAddr, turnNumAddr_ + turnNum * sizeof(u32), isLast,
        reinterpret_cast<rtStarsCcoreWaitStartSqe_t*>(sqeBuffer), sqeTypeAddr);
    hccl::HcclSqeContext* sqeCtx = stream->GetSqeContextPtr();
    if (sqeCtx == nullptr) {
        HCCL_ERROR("AddCcoreWait sqeCtx is nullptr");
        return HCCL_E_INTERNAL;
    }
    sqeCtx->buffer.addInfo[taskId % hccl::HCCL_SQE_MAX_CNT]
        = ((turnNum << TURN_LEFT_SHIFT_BIT) + static_cast<uint32_t>(isLast));
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcRpcServerV2::AddCcoreNotify(
    HcclDispatcher dispatcherPtr, u64 recordAddr, uint32_t turnNum,
    hccl::Stream* stream) // client finish notify
{
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;

    CHK_RET(AddFlipTask(dispatcherPtr, stream));
    CHK_RET(stream->GetNextSqeBufferAddr(sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    const HcclComStreamInfo& streamInfo = stream->GetHcclStreamInfo();
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_AICORE_WAIT_TIMEOUT)) {
        uint32_t* turnNums = reinterpret_cast<uint32_t*>(turnNumAddr_);
        turnNums[turnNum] = 0;
    }
    AddOneWriteValueStartSqe(
        streamInfo.actualStreamId, taskId, recordAddr, turnNumAddr_ + turnNum * sizeof(u32),
        reinterpret_cast<rtStarsCcoreWriteValueSqe_t*>(sqeBuffer), sqeTypeAddr);
    hccl::HcclSqeContext* sqeCtx = stream->GetSqeContextPtr();
    if (sqeCtx == nullptr) {
        HCCL_ERROR("AddCcoreNotify sqeCtx is nullptr");
        return HCCL_E_INTERNAL;
    }
    sqeCtx->buffer.addInfo[taskId % hccl::HCCL_SQE_MAX_CNT] = turnNum;
    return HCCL_SUCCESS;
}

uint64_t AicpuKfcRpcServerV2::GetFinishAddrByHandleId(HcclHandle handleId)
{
    int32_t msgPos = GetMsgHandlePos(handleId);
    if (msgPos < 0) {
        return 0;
    }
    return GetFinishAddr(msgPos);
}

void AicpuKfcRpcServerV2::SetMsgRepeatCnt(u8 repeatCnt)
{
    repeatCnt_[msgPos_[0U]] = (totalStep_ == 0U ? repeatCnt : repeatCnt * totalStep_);
}

int32_t AicpuKfcRpcServerV2::GetMsgRepeatCnt(HcclHandle handleId)
{
    int32_t msgPos = GetMsgHandlePos(handleId);
    if (msgPos < 0) {
        return -1;
    }
    return repeatCnt_[msgPos];
}

HcclResult AicpuKfcRpcServerV2::ProcessExpectPrepareMsg(uint8_t seqNum, uint8_t expectId)
{
    // 当前无翻转场景，只需考虑单个通信域提前Finializa场景
    if (seqNum == 0 && expectId > 0) {
        return HCCL_SUCCESS;
    }
    if (seqNum < expectId) {
        HCCL_ERROR("curMsg seqNum %d is smaller than expect %d ignore.", seqNum, expectId);
        return HCCL_E_INTERNAL;
    }
    if (totalQueueNum_ == 0U && seqNum > expectId) {
        HCCL_INFO("curMsg seqNum %d is bigger than expect %d ignore.", seqNum, expectId);
        return HCCL_E_UNAVAIL;
    }
    return HCCL_SUCCESS;
}

void AicpuKfcRpcServerV2::SetNeedRetryFlag(bool needRetryFlag) { needReProcess_ = needRetryFlag; }

bool AicpuKfcRpcServerV2::ReadValidMsgExtArea(int32_t idx, u32 rankSize)
{
#ifdef __aarch64__
    __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
    __asm__ __volatile__("" : : : "memory");
#endif
    auto& extMsgList = hcclMsgArea_->commMsg.singleMsg.paramExtMsgList[idx];
    if (hcclMsgArea_ == nullptr || extMsgList.valid != static_cast<u64>(HCCL_MSG_VALID_MASK)) {
        return false;
    }
    uint64_t msgExtXorCheck = AicpuKfcUtils::GenXor(&extMsgList, rankSize);
    static uint32_t msgExtXorCheckTurn = 0;
    if (UNLIKELY(msgExtXorCheck != extMsgList.xorCheck)) {
        if (msgExtXorCheckTurn++ % MC2_API_XORCHECK_PRINT_NUM == 0) {
            HCCL_RUN_INFO(
                "Extend data is modified! modified_xor:%llu, origin_xor:%llu.", msgExtXorCheck, extMsgList.xorCheck);
        }
        return false;
    }
    HCCL_INFO("hcclMsgArea xorCheck[%llu]", extMsgList.xorCheck);
    const size_t copySize = sizeof(uint64_t) * rankSize;
    (void)memcpy_s(msgExt_->sendCounts, copySize, extMsgList.sendCounts, copySize);
    (void)memcpy_s(msgExt_->sendOffset, copySize, extMsgList.sendOffset, copySize);
    (void)memcpy_s(msgExt_->recvCounts, copySize, extMsgList.recvCounts, copySize);
    (void)memcpy_s(msgExt_->recvOffset, copySize, extMsgList.recvOffset, copySize);
    (void)memcpy_s(
        msgExt_->reserved, sizeof(HcclMsgExt) - offsetof(HcclMsgExt, reserved), extMsgList.reserved,
        sizeof(HcclMsgExt) - offsetof(HcclMsgExt, reserved));

#ifdef __aarch64__
    __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
    __asm__ __volatile__("" : : : "memory");
#endif
    extMsgList.valid = static_cast<u64>(~HCCL_MSG_VALID_MASK);
#ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
#endif
    HCCL_INFO("reset paramExtMsgList valid value %lu", extMsgList.valid);
    return true;
}

bool AicpuKfcRpcServerV2::IsExceedLimit(HcclCMDType commType, u32 rankSize)
{
    if (rankSize > HCCL_MAX_RANK_NUM_V2
        && (commType == HcclCMDType::HCCL_CMD_ALLTOALLV || commType == HcclCMDType::HCCL_CMD_ALLTOALL)) {
        HCCL_ERROR(
            "The number[%u] of ranks exceeds the 256p limit supported by the ALLTOALL/ALLTOALLV algorithm.", rankSize);
        return true;
    }
    return false;
}

bool AicpuKfcRpcServerV2::CheckDebugMode(HcclMsg* rMsg)
{
    if (UNLIKELY(
            AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PREPARE_TIMEOUT)
            && (rMsg->commType.msgType != ControlMsgType::HCCL_CMD_FINALIZE))) {
        return false;
    }
    if (UNLIKELY(
            AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_FINALIZE_TIMEOUT)
            && (rMsg->commType.msgType == ControlMsgType::HCCL_CMD_FINALIZE))) {
        return false;
    }
    return true;
}

bool AicpuKfcRpcServerV2::ReadValidMsg(HcclMsg* rMsg, HcclMsg* msg, bool needReProcess, uint32_t msgPos, u32 rankSize)
{
#if defined(__aarch64__)
    __asm__ __volatile__("dsb ld" : : : "memory");
#elif defined(__amd64__)
    __asm__ __volatile__("" : : : "memory");
#endif
    // 重处理消息
    if (needReProcess) {
        *rMsg = *msg;
        return true;
    }
    if (msg->addMsg.v0Msg.valid != HCCL_MSG_VALID_MASK) {
        return false;
    }
    if (memcpy_s(rMsg, sizeof(HcclMsg), msg, sizeof(HcclMsg)) != EOK) {
        HCCL_ERROR("rMsg memcpy_s failed!,size:%d", sizeof(HcclMsg));
        return false;
    }
    uint32_t msgXorCheck = AicpuKfcUtils::GenXor(rMsg);
    static uint32_t msgXorCheckTurn = 0;
    if (UNLIKELY(msgXorCheck != rMsg->addMsg.v0Msg.xorCheck)) {
        if (msgXorCheckTurn++ % MC2_API_XORCHECK_PRINT_NUM == 0) {
            AicpuKfcUtils::PrintMsg("Rcv src msg", *msg, true);
            AicpuKfcUtils::PrintMsg("Rcv dst msg", *rMsg, true);
            HCCL_RUN_INFO(
                "data is modified! modified_xor:%u, origin_xor:%u.", msgXorCheck, rMsg->addMsg.v0Msg.xorCheck);
        }
        return false;
    }
    if (UNLIKELY(IsExceedLimit(static_cast<HcclCMDType>(rMsg->commType.prepareType), rankSize))) {
        return false;
    }
    if (UNLIKELY(
            static_cast<HcclCMDType>(rMsg->commType.prepareType) == HCCL_CMD_ALLTOALLV
            && !ReadValidMsgExtArea(msgPos, rankSize))) {
        return false;
    }
    msg->addMsg.v0Msg.valid = ~HCCL_MSG_VALID_MASK;
    if (UNLIKELY(!CheckDebugMode(rMsg))) {
        return false;
    }
    HCCL_INFO("reset valid value 0x%x", msg->addMsg.v0Msg.valid);
    return true;
}

bool AicpuKfcRpcServerV2::ReadAddrMsg(HcclMsg* hcclMsg, HcclMsg* msgList, u32 queueIdx, u32 msgPos, u32 rankSize)
{
    bool ret = ReadValidMsg(hcclMsg, &(msgList[msgPos]), needReProcess_, msgPos, rankSize);
    isPrintLog_ = false;
    if (LIKELY(ret)) {
        HCCL_DEBUG("read valid msg msgPos %u commType %u", msgPos, static_cast<uint32_t>(hcclMsg->commType.msgType));
        PrintMsg(hcclMsg, msgPos, rankSize);
        // Prepare 成功，打印耗时
        u64 prepareTime = GetCurCpuTimestamp();
        if (eventPrintTurn_ > 1) {
            HCCL_RUN_INFO(
                "[AicpuKfcRpcServerV2][ReadAddrMsg] Read HcclMsg[%u] cost[%llu ns]", msgPos,
                prepareTime - prepareTime_[queueIdx]);
        } else {
            HCCL_INFO(
                "[AicpuKfcRpcServerV2][ReadAddrMsg] Read HcclMsg[%u] cost[%llu ns]", msgPos,
                prepareTime - prepareTime_[queueIdx]);
        }
        prepareTime_[queueIdx] = prepareTime;
        eventPrintTurn_ = 1;
    } else if (
        GetCurCpuTimestamp() - prepareTime_[queueIdx]
        > static_cast<unsigned long long>(NSEC_PER_SEC) * MC2_API_MSG_TIMEOUT * eventPrintTurn_) {
        // Prepare 等待 20s
        HCCL_RUN_WARNING(
            "[AicpuKfcRpcServerV2][ReadAddrMsg] ReadValidMsg[%u] timeout %lus", msgPos,
            MC2_API_MSG_TIMEOUT * eventPrintTurn_);
        eventPrintTurn_ *= 2; // 2 is print event log times
        LogControl logControl(false, true);
        PrintAllHcclMsgArea(rankSize);
        isPrintLog_ = true;
    }
    return ret;
}

// reset消息区msgPos的commitTurnId
HcclResult AicpuKfcRpcServerV2::ResetCommitTaskAdd(HcclDispatcher dispatcherPtr, hccl::Stream* stream)
{
    // reset函数复用 AddCcoreWait,turnNum保证条件算子恒成立
    for (uint32_t i = 0; i < static_cast<uint32_t>(msgPos_[0U]); i++) {
        uint64_t waitAddr = GetCommitareaAddr(i);
        CHK_RET(AddCcoreWait(dispatcherPtr, waitAddr, 0, stream, true));
    }
    return HCCL_SUCCESS;
}

void AicpuKfcRpcServerV2::WriteFinishWhenAllFinalize()
{
    if (hcclMsgArea_ == nullptr || totalQueueNum_ != 0U) {
        return;
    }
    uint32_t msgPos = GetMsgPos();
    hcclMsgArea_->commMsg.singleMsg.finishedTurnCnt[msgPos].cnt = FINALIZE_FINISH_CNT; // 用于校验的非法值
    HCCL_INFO(
        "Post finishedTurnCnt[%u].cnt = %lu.", msgPos, hcclMsgArea_->commMsg.singleMsg.finishedTurnCnt[msgPos].cnt);
#ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
#endif
}

void AicpuKfcRpcServerV2::WriteRestartFlag()
{
    if (totalQueueNum_ == 0U) {
        for (uint32_t i = 0; i < HCCL_MSG_CNT; i++) {
            hcclMsgArea_->commMsg.singleMsg.sendMsgs[i].addMsg.v0Msg.valid = ~HCCL_MSG_VALID_MASK;
            hcclMsgArea_->commMsg.singleMsg.commitTurnCnt[i].cnt = 0;
            hcclMsgArea_->commMsg.singleMsg.finishedTurnCnt[i].cnt = 0;
        }
    } else {
        for (uint32_t i = 0; i < MAX_QUE_NUM; i++) {
            for (uint32_t j = 0; j < HCCL_MSG_CNT; j++) {
                hcclMsgArea_->commMsg.multiMsg.sendMsgs[i][j].addMsg.v1Msg.valid = ~HCCL_MSG_VALID_MASK;
            }
        }
    }
    hcclMsgArea_->controlMsg.restart = 1;
#ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
#endif
}

void AicpuKfcRpcServerV2::PrintAllHcclMsgArea(u32 rankSize)
{
    if (totalQueueNum_ == 0U) {
        AicpuKfcUtils::PrintAllHcclMsgArea(hcclMsgArea_, rankSize, true);
    } else if (HcclAicpuUtils::GetBlockIdx() == 0U) {
        AicpuKfcUtils::PrintAllHcclMsgAreaForMulti(hcclMsgArea_, true);
    }
}

void AicpuKfcRpcServerV2::PrintMsg(HcclMsg* hcclMsg, uint32_t msgPos, u32 rankSize)
{
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PRINT_MSG)) {
        AicpuKfcUtils::PrintMsg("ReadAddrMsg msgPos " + std::to_string(msgPos), *hcclMsg, true);
        if (totalQueueNum_ == 0U) {
            AicpuKfcUtils::PrintAllHcclMsgArea(hcclMsgArea_, rankSize);
        } else {
            AicpuKfcUtils::PrintAllHcclMsgAreaForMulti(hcclMsgArea_);
        }
    } else {
        AicpuKfcUtils::PrintMsg("ReadAddrMsg msgPos " + std::to_string(msgPos), *hcclMsg);
    }
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
        AicpuKfcUtils::PrintApiBufferByMsgPos(*hcclMsg, msgPos);
    }
}

void AicpuKfcRpcServerV2::PrintAllHcclMsgAreaData()
{
    if (totalQueueNum_ != 0U) {
        return;
    }
    for (uint32_t i = 0; i < HCCL_MSG_CNT; ++i) {
        AicpuKfcUtils::PrintApiBufferByMsgPos(hcclMsgArea_->commMsg.singleMsg.sendMsgs[i], i);
    }
}

void AicpuKfcRpcServerV2::DumpBarrierInfo(u32 groupIdx, u32 sqId, u32 devId)
{
    const auto& barrierInfos = barrierFlags_[groupIdx];
    for (u32 i = 0U; i < totalQueueNum_; ++i) {
        const BarrierStatus status = barrierInfos[i].status;
        HCCL_ERROR("Queue:%u, msg pos:%u, status:%u.", i, GetMsgPos(i), static_cast<u32>(status));
        if (status == BarrierStatus::SELF_BARRIER) {
            u32 sqHead, sqTail;
            (void)QuerySqStatusByType(devId, sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead);
            (void)QuerySqStatusByType(devId, sqId, DRV_SQCQ_PROP_SQ_TAIL, sqTail);
            HCCL_ERROR("Queue:%u, sq head:%u, tail:%u.", i, sqHead, sqTail);
        }
    }
}
