/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_engine_res_manager.h"

namespace hccl {
CommEngineResMgr::CommEngineResMgr() {};

HcclResult CommEngineResMgr::Init(
    uint32_t threadNum, uint32_t notifyNumPerThread, const std::string& commId, const aclrtBinHandle binHandle,
    const ManagerCallbacks& callbacks)
{
    std::lock_guard<std::mutex> lock(mtx_);
    HCCL_INFO(
        "[CommEngineResMgr][%s] Hcom[%s] threadNum[%u], notifyPerThread[%u]", __func__, commId.c_str(), threadNum,
        notifyNumPerThread);
    if (!threadMgr_) {
        EXCEPTION_CATCH(
            threadMgr_ = std::make_unique<ThreadMgr>(threadNum, notifyNumPerThread, commId, binHandle, callbacks),
            return HCCL_E_PTR);
    }
    if (!notifyMgr_) {
        EXCEPTION_CATCH(notifyMgr_ = std::make_unique<NotifyManager>(commId, binHandle, callbacks), return HCCL_E_PTR);
    }
    return HCCL_SUCCESS;
}

HcclResult CommEngineResMgr::HcclThreadAcquireV2(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
    std::vector<uint32_t>& threadId)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    uint32_t setThreadNum = threadMgr_->GetThreadNum();
    CHK_PRT_RET(
        threadNum > setThreadNum,
        HCCL_ERROR(
            "[%s]Alloced thread num[%u] more than num[%u] in config type[%d]", __func__, threadNum, setThreadNum,
            static_cast<int32_t>(type)),
        HCCL_E_PARA);
    return threadMgr_->HcclThreadAcquireV2(engine, threadNum, type, config, threads, threadId);
}

HcclResult CommEngineResMgr::HcclThreadAcquire(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
    std::vector<uint32_t>& threadId)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    uint32_t setThreadNum = threadMgr_->GetThreadNum();
    CHK_PRT_RET(
        threadNum > setThreadNum,
        HCCL_ERROR(
            "[%s] Alloced thread num[%u] more than num[%u] in config type[%d]", __func__, threadNum, setThreadNum,
            static_cast<int32_t>(type)),
        HCCL_E_PARA);
    return threadMgr_->HcclThreadAcquire(engine, threadNum, type, config, threads, threadId);
}

HcclResult CommEngineResMgr::HcclThreadAcquireWithStream(
    CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return threadMgr_->HcclThreadAcquireWithStream(engine, stream, notifyNum, thread);
}

HcclResult
CommEngineResMgr::HcclGetNotifyNumInThread(ThreadHandle thread, [[maybe_unused]] CommEngine engine, uint32_t* notifyNum)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return threadMgr_->HcclGetNotifyNumInThread(thread, notifyNum);
}

HcclResult CommEngineResMgr::HcclAllocNotify(
    CommEngine commEngine, ::NotifyType notifyType, uint32_t notifyNum, NotifyHandle** notifyHandleList)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return notifyMgr_->HcclAllocNotify(commEngine, notifyType, notifyNum, notifyHandleList);
}

HcclResult CommEngineResMgr::HcommFreeNotify(uint32_t notifyNum, NotifyHandle* notifyHandleList)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return notifyMgr_->HcommFreeNotify(notifyNum, notifyHandleList);
}

HcclResult CommEngineResMgr::HcclThreadExportToCommEngine(
    uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return threadMgr_->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
}

HcclResult
CommEngineResMgr::HcclThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return threadMgr_->HcclThreadResGetInfo(thread, resType, infoLen, info);
}

HcclResult CommEngineResMgr::HcclDedicatedThreadAcquire(
    HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return threadMgr_->HcclDedicatedThreadAcquire(useType, notifyNumPerThread, thread);
}

HcclResult CommEngineResMgr::RegisterOrderLaunchThread(ThreadHandle thread)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return threadMgr_->RegisterOrderLaunchThread(thread);
}

HcclResult CommEngineResMgr::ResetCommLocalNotifies()
{
    CHK_SMART_PTR_NULL(threadMgr_);
    HcclResult ret = threadMgr_->ResetThreadLocalNotifies();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CommEngineResMgr][ResetCommLocalNotifies] ResetThreadLocalNotifies failed, ret[0x%016llx]",
            HCCL_ERROR_CODE(ret));
        return ret;
    }
    if (notifyMgr_ != nullptr) {
        ret = notifyMgr_->ResetAllocLocalNotifies();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CommEngineResMgr][ResetCommLocalNotifies] ResetAllocLocalNotifies failed, ret[0x%016llx]",
                HCCL_ERROR_CODE(ret));
            return ret;
        }
    }
    return ret;
}

} // namespace hccl
