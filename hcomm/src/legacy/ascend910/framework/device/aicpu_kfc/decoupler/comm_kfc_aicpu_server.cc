/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_kfc_aicpu_server.h"
#include <numeric>
#include "log.h"
#include "common/aicpu_kfc_utils.h"
#include "hccl_mc2_ex.h"

using namespace HcclApi;
namespace {
static constexpr u64 TIMEOUT_ERROR_THRESHOLD = 960UL;
static const std::vector<HcclCMDType> SUPPORT_OP_LIST{
    HCCL_CMD_ALLREDUCE, HCCL_CMD_ALLGATHER, HCCL_CMD_REDUCE_SCATTER, HCCL_CMD_ALLTOALLV, HCCL_CMD_ALLTOALL};

void FormatOpData(const HcclMsg& msg, HcclMsgExt& extMsg, u32 rankNum, u32 repeat, HcclOpData& data)
{
    if (repeat == 0U) {
        data.opType = static_cast<HcclCMDType>(msg.commType.prepareType);
        data.reduceOp = static_cast<HcclReduceOp>(msg.opType);
        data.dataType = data.outputDataType = static_cast<HcclDataType>(msg.addMsg.v1Msg.hcclDataType);
        data.dataCount = msg.dataCnt;
        if (data.opType == HCCL_CMD_ALLTOALLV) {
            data.all2AllVDataDes.sendType = data.all2AllVDataDes.recvType = data.dataType;
            data.all2AllVDataDes.sendCounts = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(extMsg.sendCounts));
            data.all2AllVDataDes.recvCounts = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(extMsg.recvCounts));
            data.all2AllVDataDes.sdispls = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(extMsg.sendOffset));
            data.all2AllVDataDes.rdispls = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(extMsg.recvOffset));
        } else if (data.opType == HCCL_CMD_ALLTOALL) {
            data.all2AllDataDes.sendType = data.all2AllDataDes.recvType = data.dataType;
            data.all2AllDataDes.sendCount = data.all2AllDataDes.recvCount = data.dataCount;
        } else {
            data.dataDes.dataType = data.dataType;
            data.dataDes.dataCount = data.dataCount;
            data.dataDes.strideCount = msg.strideCount;
        }
    } else if (data.opType == HCCL_CMD_ALLTOALLV) {
        for (u32 i = 0U; i < rankNum; ++i) {
            extMsg.sendOffset[i] += extMsg.sendCounts[i];
            extMsg.recvOffset[i] += extMsg.recvCounts[i];
            HCCL_INFO(
                "Formatted alltoallv info: repeat %u, rank id %u, send offset %llu, recv offset %llu.", repeat, i,
                static_cast<u64*>(data.all2AllVDataDes.sdispls)[i], static_cast<u64*>(data.all2AllVDataDes.rdispls)[i]);
        }
    }
    const u64 offset = data.dataCount * DataUnitSize(data.dataType);
    data.input = msg.sendBuffer + offset * repeat;
    data.output = msg.recvBuffer + offset * repeat;
    HCCL_INFO(
        "Formatted op info: repeat index %u, op type %u, reduce type %u, data type %u, "
        "data count %llu, input addr %#llx, output addr %#llx.",
        static_cast<u32>(repeat), static_cast<u32>(data.opType), static_cast<u32>(data.reduceOp),
        static_cast<u32>(data.dataType), data.dataCount, data.input, data.output);
}
} // namespace

HcclResult CommKfcAicpuServer::AddOpContext(const CommKfcContext* ctx)
{
    CHK_PTR_NULL(ctx);
    if (ctxToOpHandle_.find(ctx->hcclContext) != ctxToOpHandle_.end()) {
        HCCL_INFO("Group %u: ctx %#llx is already added.", groupIdx_, ctx->hcclContext);
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        msgArea_ != nullptr && reinterpret_cast<u64>(msgArea_) != ctx->apiCtx.workSpace,
        HCCL_ERROR(
            "Group %u: message area addr should be %#llx, not %#llx.", groupIdx_, msgArea_, ctx->apiCtx.workSpace),
        HCCL_E_PARA);
    void* opHandle = nullptr;
    HcclResult ret = HcclGetCommHandleByCtx(reinterpret_cast<void*>(ctx->hcclContext), &opHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || opHandle == nullptr,
        HCCL_ERROR("Group %u: failed to get op handle by HCCL ctx %#llx.", groupIdx_, ctx->hcclContext), HCCL_E_PARA);
    ctxToOpHandle_[ctx->hcclContext] = opHandle;
    if (msgArea_ == nullptr) {
        msgArea_ = reinterpret_cast<HcclMsgArea*>(ctx->apiCtx.workSpace);
        turnNumsAddr_ = reinterpret_cast<u64>(msgArea_ + 1);
        rankNum_ = ctx->apiCtx.rankNum;
        std::iota(reinterpret_cast<u32*>(turnNumsAddr_), reinterpret_cast<u32*>(turnNumsAddr_) + UINT8_MAX + 1U, 0U);
        KeepAlive();
    }
    HCCL_INFO(
        "Group %u: add op handle %#llx, HCCL context %#llx, message area address %#llx.", groupIdx_, opHandle,
        ctx->hcclContext, msgArea_);
    return HCCL_SUCCESS;
}

HcclResult CommKfcAicpuServer::Orchestrate(const HcclMsg& msg, HcclMsgExt& extMsg, u32 msgPos)
{
    KeepAlive();
    CHK_PTR_NULL(msgArea_);
    auto handleIter = ctxToOpHandle_.find(reinterpret_cast<uintptr_t>(msg.addMsg.v1Msg.ccOpTilingData));
    CHK_PRT_RET(
        handleIter == ctxToOpHandle_.end(),
        HCCL_ERROR("Group %u: op handle %#llx is not added by host.", groupIdx_, msg.addMsg.v1Msg.ccOpTilingData),
        HCCL_E_PARA);
    const auto opIter
        = std::find(SUPPORT_OP_LIST.begin(), SUPPORT_OP_LIST.end(), static_cast<HcclCMDType>(msg.commType.prepareType));
    CHK_PRT_RET(
        opIter == SUPPORT_OP_LIST.end(),
        HCCL_ERROR("Unsupported comm type %u.", static_cast<u32>(msg.commType.prepareType)), HCCL_E_PARA);
    const HcclHandle handle = msg.addMsg.v1Msg.selfHandleID;
    CHK_PRT_RET(handle < 0, HCCL_ERROR("Group %u: invalid handle id %d.", groupIdx_, handle), HCCL_E_INTERNAL);
    const u32 repeatCnt = static_cast<u32>(msg.addMsg.v1Msg.repeatCnt);
    const u64 waitAddr = reinterpret_cast<u64>(&(msgArea_->commMsg.singleMsg.commitTurnCnt[msgPos].cnt));
    const u64 recordAddr = reinterpret_cast<u64>(&(msgArea_->commMsg.singleMsg.finishedTurnCnt[msgPos].cnt));

    HcclOpData data{};
    void* opHandle = handleIter->second;
    for (u32 i = 0U; i < repeatCnt; ++i) {
        FormatOpData(msg, extMsg, rankNum_, i, data);
        const u32 turnIdx = i + 1U;
        CHK_RET(HcclLaunchCcoreWait(opHandle, waitAddr, turnIdx, turnNumsAddr_, turnIdx == repeatCnt));
        CHK_RET(HcclLaunchOp(opHandle, &data));
        CHK_RET(HcclLaunchCcorePost(opHandle, recordAddr, turnIdx, turnNumsAddr_));
    }
    SetMsgPosByHandle(handle, msgPos);
    SetRepeatByHandle(handle, repeatCnt);
    return HCCL_SUCCESS;
}

HcclResult CommKfcAicpuServer::Finalize([[maybe_unused]] u32 msgPos)
{
    KeepAlive();
    return HCCL_SUCCESS;
}

HcclResult CommKfcAicpuServer::IsAllTaskFinished(u32 msgPos, bool& isFinish)
{
    CHK_PTR_NULL(msgArea_);

    isFinish = false;
    // opHandles_能保证不为空，同一个通信域检查任何一个ophandle即可
    void* firstHandle = ctxToOpHandle_.begin()->second;
    if (HcclCheckFinishByStream(firstHandle) != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    HcclTaskStatus status;
    if (HcclGetTaskStatus(firstHandle, &status) != HCCL_SUCCESS || status != HcclTaskStatus::HCCL_NORMAL_STATUS) {
        HCCL_ERROR("Group %u: abnormal task status %u.", groupIdx_, static_cast<u32>(status));
        return HCCL_E_INTERNAL;
    }

    msgArea_->commMsg.singleMsg.finishedTurnCnt[msgPos].cnt = FINALIZE_FINISH_CNT;
#ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
#endif
    isFinish = true;
    HCCL_INFO("Group %u: all tasks are finished at message pos %u.", groupIdx_, msgPos);
    for (auto it : ctxToOpHandle_) {
        CHK_RET(HcclReleaseComm(it.second));
        HCCL_INFO("Group %u: Op handle %#llx is released, HCCL context %#llx.", groupIdx_, it.second, it.first);
    }
    return HCCL_SUCCESS;
}

HcclResult CommKfcAicpuServer::InterGroupSync(const CommKfcAicpuServer& otherServer, HcclHandle handle)
{
    KeepAlive();
    u32 msgPos, repeat;
    HcclResult ret = otherServer.GetServerInfoForSync(handle, msgPos, repeat);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO("Group %u: group sync info is not obtained, return code %u.", groupIdx_, ret);
        return ret;
    }
    CHK_PRT_RET(
        msgPos >= HCCL_MSG_CNT, HCCL_ERROR("Group %u: invalid message index %u.", groupIdx_, msgPos), HCCL_E_PARA);

    HcclMsgArea* msgArea = otherServer.GetMsgAreaAddr();
    CHK_PTR_NULL(msgArea);
    const u64 waitAddr = reinterpret_cast<u64>(&(msgArea->commMsg.singleMsg.finishedTurnCnt[msgPos].cnt));
    HCCL_INFO(
        "Group %u: group sync for handle %d: message index %u, finish count %u.", groupIdx_, handle, msgPos, repeat);
    return HcclLaunchCcoreWait(ctxToOpHandle_.begin()->second, waitAddr, repeat, turnNumsAddr_, false);
}

HcclResult CommKfcAicpuServer::CheckTimeOut(u32 msgPos)
{
    if (!IsTimeout()) {
        return HCCL_SUCCESS;
    }
    const bool error = (timeout_ >= TIMEOUT_ERROR_THRESHOLD);
    HcclResult ret;
    if (error) {
        HCCL_ERROR("Group %u: timeout %u seconds at message pos %u.", groupIdx_, timeout_, msgPos);
        ret = HCCL_E_TIMEOUT;
    } else {
        HCCL_RUN_INFO("Group %u: timeout %u seconds at message pos %u.", groupIdx_, timeout_, msgPos);
        ret = HCCL_E_AGAIN;
    }
    timeout_ *= 2U;
    return ret;
}

HcclResult CommKfcAicpuServer::GetServerInfoForSync(HcclHandle handle, u32& msgPos, u32& repeat) const
{
    CHK_PRT_RET(handle < 0, HCCL_ERROR("Group %u: invalid handle id %d.", groupIdx_, handle), HCCL_E_PARA);
    auto it = handleIdToMsgPos_.find(handle);
    if (it == handleIdToMsgPos_.end()) {
        HCCL_INFO("Group %u: handle %d in this group is not ready.", groupIdx_, handle);
        return HCCL_E_AGAIN;
    }
    msgPos = it->second;

    it = handleIdToRepeat_.find(handle);
    CHK_PRT_RET(
        it == handleIdToRepeat_.end(), HCCL_ERROR("Group %u: handle %d in this group is not ready.", groupIdx_, handle),
        HCCL_E_INTERNAL);
    repeat = it->second;
    return HCCL_SUCCESS;
}

HcclResult CommKfcAicpuServer::ErrorDfxProcess(HcclResult errorCode)
{
    void* firstHandle = ctxToOpHandle_.begin()->second;
    if (errorCode == HCCL_SUCCESS) {
        return errorCode;
    } else if (errorCode == HCCL_E_AGAIN) {
        AicpuKfcUtils::PrintAllHcclMsgArea(msgArea_, rankNum_);
        HcclPrintTaskExceptionAllComm(firstHandle);
        errorCode = HCCL_SUCCESS;
    } else {
        AicpuKfcUtils::PrintAllHcclMsgArea(msgArea_, rankNum_, true);
        HcclPrintTaskExceptionAllComm(firstHandle);
    }
    return errorCode;
}
