/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include "hccl/hccl_res.h"
#include "stream_pub.h"
#include "hccl_comm_pub.h"
#include "hccl_independent_common.h"
#include "coll_comm_profiling.h"
#include "comm_engine_utils.h"
#include "coll_comm_mgr.h"
#include "orion_adapter_rts.h"
#include "hccl_common.h"
#include "dfx_dlprof_function.h"
#include "adapter_rts.h"
#include "hcclCommOp.h"
using namespace hccl;
constexpr u32 MAX_EXPORT_THREAD_NUM = 40U;
static const std::unordered_set<HcclDedicatedThreadType> ORDER_LAUNCH_TYPES = {
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE,
};

HcclResult HcclGetNotifyNumInThread(HcclComm comm, ThreadHandle thread, CommEngine engine, uint32_t* notifyNum)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        !IsValidCommEngine(engine),
        HCCL_ERROR(
            "[%s] commEngine[%s] is invalid", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str()),
        HCCL_E_PARA);
    CHK_PRT_RET(notifyNum == nullptr, HCCL_ERROR("[%s] notifyNum is null", __func__), HCCL_E_PTR);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s] engine[%s]", __func__, commId.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclGetNotifyNumInThread(thread, engine, notifyNum);
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclGetNotifyNumInThread(thread, engine, notifyNum);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclGetNotifyNumInThread] Failed to get notifyNum for engine[%s] ret[%d]",
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret);
        return ret;
    }
    HCCL_INFO(
        "[HcclGetNotifyNumInThread] threads for engine[%s], notifyNum[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), *notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithConfigDfx(
    hccl::CollComm* collComm, const std::string& commId, CommEngine engine, u64 beginTime, uint32_t threadNum,
    ThreadHandle* threads, std::vector<uint32_t>& threadId)
{
    CHK_PTR_NULL(threads);
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);
    if (engine == CommEngine::COMM_ENGINE_AICPU) {
        Mc2CommInfo mc2CommInfo;
        mc2CommInfo.FreeStreamId = 0;
        mc2CommInfo.streamsId = threadId;
        mc2CommInfo.groupname = commId;
        mc2CommInfo.myRankId = collComm->GetMyRankId();
        mc2CommInfo.rankSize = collComm->GetRankSize();
        CHK_RET(collComm->GetParentRankId(mc2CommInfo.parentRankId));
        hcclCommDfx->ReportMc2CommInfo(mc2CommInfo);
        HCCL_INFO("[HcclThreadAcquireWithConfigDfx] ReportThreadAcquireKernel begin");
        const std::string KernelName = "RunAicpuIndOpThreadInit";
        // 这个地方获取不到当前是单算子还是图模式，所以全部都不保存
        CHK_RET(hcclCommDfx->ReportKernel(beginTime, commId, KernelName, SalGetTid(), false));
        HCCL_INFO("[HcclThreadAcquireWithConfigDfx] ReportThreadAcquireKernel success");
    } else {
        auto hcclCommDfxCallBack = collComm->GetDfxCallback();
        for (u32 num = 0; num < threadNum; ++num) {
            int ret = HcommThreadRegisterDfx(threads[num], hcclCommDfxCallBack);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[HcclThreadAcquireWithConfigDfx] ReportThreadAcquireKernel HcommThreadRegisterDfx failed"
                    " ret:[%d], num:[%u]",
                    ret, num);
                return HCCL_E_INTERNAL;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
ValidateThreadAcquireParams(CommEngine engine, ThreadType type, const ThreadConfig* config, uint32_t threadNum)
{
    CHK_PRT_RET(
        type == THREAD_TYPE_INVALID,
        HCCL_ERROR("[%s] thread type[%d] is invalid", __func__, static_cast<int32_t>(type)), HCCL_E_PARA);
    CHK_PRT_RET(
        !IsValidCommEngine(engine),
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);
    CHK_PRT_RET(threadNum == 0, HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum), HCCL_E_PARA);
    CHK_PRT_RET(config == nullptr, HCCL_ERROR("[%s] config is null", __func__), HCCL_E_PTR);
    for (uint32_t i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x], call ThreadConfigInit first", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            HCCL_E_PARA);
    }
    CHK_PRT_RET(
        engine == CommEngine::COMM_ENGINE_AICPU_TS || engine == CommEngine::COMM_ENGINE_CPU_TS,
        HCCL_ERROR(
            "[%s] commEngine[%d] CPU_TS/AICPU_TS not supported, use CPU/AICPU engine with THREAD_TYPE_TS instead",
            __func__, static_cast<int32_t>(engine)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        engine == CommEngine::COMM_ENGINE_AIV || engine == CommEngine::COMM_ENGINE_CCU,
        HCCL_ERROR(
            "[%s] commEngine[%d] AIV/CCU not supported, supported engines: CPU/AICPU", __func__,
            static_cast<int32_t>(engine)),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithConfig(
    HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    ThreadHandle* threads)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(threads == nullptr, HCCL_ERROR("[%s] threads is null", __func__), HCCL_E_PTR);
    CHK_RET(ValidateThreadAcquireParams(engine, type, config, threadNum));

    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s] engine[%s] ThreadNum[%u].", __func__, commId.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum);

    HcclResult ret = HCCL_SUCCESS;
    std::vector<uint32_t> threadId;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadAcquireV2(engine, threadNum, type, config, threads, threadId);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] failed to create threads for engine[%s], threadsNum[%u], ret[%d].", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret);
            return ret;
        }
        CHK_RET(HcclThreadAcquireWithConfigDfx(collComm, commId, engine, beginTime, threadNum, threads, threadId));
        return HCCL_SUCCESS;
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadAcquire(engine, threadNum, type, config, threads, threadId);
        if (engine == CommEngine::COMM_ENGINE_AICPU) {
            // 上报流
            if (threadNum != threadId.size()) {
                HCCL_ERROR("[%s] threadNum [%u] != threadId.size[%zu]", __func__, threadNum, threadId.size());
                return HCCL_E_PARA;
            }
            CHK_RET(HcclStreamProfilingReport(comm, threadNum, threadId.data()));
        }
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] failed to create threads for engine[%s], threadsNum[%u], ret[%d].", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret);
        return ret;
    }

    HCCL_INFO("[%s] Allocated %u threads for engine[%d]", __func__, threadNum, engine);
    return HCCL_SUCCESS;
}

static CommEngine ConvertEngineToTsType(CommEngine engine)
{
    if (engine == COMM_ENGINE_CPU_TS) {
        return COMM_ENGINE_CPU;
    }
    if (engine == COMM_ENGINE_AICPU_TS) {
        return COMM_ENGINE_AICPU;
    }
    return engine;
}

HcclResult HcclThreadAcquire(
    HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads)
{
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(threads == nullptr, HCCL_ERROR("[%s] threads is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        !IsValidCommEngine(engine),
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);
    CHK_PRT_RET(threadNum == 0, HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s] engine[%u] ThreadNum[%u] notifyNumPerThread[%u]", __func__, commId.c_str(), engine,
        threadNum, notifyNumPerThread);

    CommEngine newEngine = ConvertEngineToTsType(engine);
    ThreadType type = THREAD_TYPE_TS;
    std::unique_ptr<ThreadConfig[]> config = std::make_unique<ThreadConfig[]>(threadNum);
    CHK_PTR_NULL(config);
    CHK_PRT_RET(
        ThreadConfigInit(config.get(), threadNum) != 0, HCCL_ERROR("[%s] ThreadConfigInit failed", __func__),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        notifyNumPerThread >= HCCL_THREAD_NOTIFY_MAX_NUM,
        HCCL_ERROR("[%s] notifyNumPerThread[%u] exceeds HCCL_THREAD_NOTIFY_MAX_NUM", __func__, notifyNumPerThread),
        HCCL_E_PARA);
    for (u32 i = 0; i < threadNum; i++) {
        config[i].notifyNumPerThread = static_cast<uint16_t>(notifyNumPerThread);
    }

    HcclResult ret = HCCL_SUCCESS;
    std::vector<uint32_t> threadId;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadAcquireV2(newEngine, threadNum, type, config.get(), threads, threadId);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] failed to create threads for engine[%d], threadsNum[%u], ret[%d]", __func__, newEngine, threadNum,
                ret);
            return ret;
        }
        CHK_RET(HcclThreadAcquireWithConfigDfx(collComm, commId, newEngine, beginTime, threadNum, threads, threadId));
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadAcquire(newEngine, threadNum, type, config.get(), threads, threadId);
        if (newEngine == CommEngine::COMM_ENGINE_AICPU) {
            if (threadNum != threadId.size()) {
                HCCL_ERROR("[%s] threadNum [%u] != threadId.size[%zu]", __func__, threadNum, threadId.size());
                return HCCL_E_PARA;
            }
            CHK_RET(HcclStreamProfilingReport(comm, threadNum, threadId.data()));
        }
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to create threads for engine[%s], threadNum[%u], ret[%d]", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), threadNum, ret);
        return ret;
    }

    HCCL_INFO(
        "[%s] Allocated %u threads for engine[%s], notifyPerThread[%u]", __func__, threadNum,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), notifyNumPerThread);
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithStreamDfx(
    hccl::CollComm* collComm, const std::string& commId, CommEngine engine, ThreadHandle thread)
{
    auto hcclCommDfxCallback = collComm->GetDfxCallback();
    int ret = HcommThreadRegisterDfx(thread, hcclCommDfxCallback);
    if (ret != 0) {
        HCCL_ERROR("[HcclThreadAcquire] HcclThreadAcquire HcommThreadRegisterDfx failed, ret:[%d]", ret);
        return HCCL_E_INTERNAL;
    }
    if (engine == CommEngine::COMM_ENGINE_AICPU) {
        Thread* threadPtr = reinterpret_cast<Thread*>(thread);
        CHK_PTR_NULL(threadPtr);
        Stream* threadStream = threadPtr->GetStream();
        CHK_PTR_NULL(threadStream);
        Mc2CommInfo mc2CommInfo;
        mc2CommInfo.FreeStreamId = 0;
        mc2CommInfo.streamsId.push_back(static_cast<u32>(threadStream->sqId()));
        mc2CommInfo.groupname = commId;
        mc2CommInfo.myRankId = collComm->GetMyRankId();
        mc2CommInfo.rankSize = collComm->GetRankSize();
        CHK_RET(collComm->GetParentRankId(mc2CommInfo.parentRankId));
        HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
        CHK_PTR_NULL(hcclCommDfx);
        hcclCommDfx->ReportMc2CommInfo(mc2CommInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithStream(
    HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(thread);
    CHK_PRT_RET(
        !IsValidCommEngine(engine),
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);

    CommEngine newEngine = ConvertEngineToTsType(engine);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "Entry-%s:comm[%s] engine[%s] notifyNum[%u] stream[%p]", __func__, commId.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), notifyNum, stream);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadAcquireWithStream(newEngine, stream, notifyNum, thread);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclThreadAcquireWithStream failed, ret[%d]", __func__, ret), ret);
        CHK_RET(HcclThreadAcquireWithStreamDfx(collComm, commId, newEngine, *thread));
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadAcquireWithStream(newEngine, stream, notifyNum, thread);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclThreadAcquireWithStream failed, ret[%d]", __func__, ret), ret);
    }

    HCCL_INFO(
        "[HcclThreadAcquireWithStream] Allocated thread for engine[%s], stream[%p], notifyNum[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), stream, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclDedicatedThreadAcquire(
    HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    EXCEPTION_HANDLE_BEGIN
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        useType == HCCL_DED_THREAD_TYPE_INVALID, HCCL_ERROR("[%s] dedThreadType is invalid", __func__), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::string& commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "Entry-%s:comm[%s] dedThreadType[%u] notifyNumPerThread[%u]", __func__, commId.c_str(), useType,
        notifyNumPerThread);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    /* 保序场景：委托给 OrderLaunchThreadMgr（进程粒度） */
    if (ORDER_LAUNCH_TYPES.find(useType) != ORDER_LAUNCH_TYPES.end()) {
        s32 deviceLogicId = Hccl::HrtGetDevice();
        auto& resMgr = hccl::CollCommMgr::GetInstance().GetOrderLaunchThreadMgr(deviceLogicId);
        ThreadHandle th = 0;
        HcclResult ret = resMgr.OrderLaunchThreadAcquire(useType, collComm, commId, notifyNumPerThread, th);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] OrderLaunchThreadAcquire fail, ret[%d], useType[%d]", __func__, ret, static_cast<s32>(useType)),
            ret);
        *thread = th;
        return HCCL_SUCCESS;
    }

    CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
    CHK_PTR_NULL(engineResMgr);
    CHK_RET(engineResMgr->HcclDedicatedThreadAcquire(useType, notifyNumPerThread, thread));
    HCCL_INFO(
        "[%s] success, dedThreadType[%u], thread[0x%llx], notifyNumPerThread[%u]", __func__, useType, *thread,
        notifyNumPerThread);
    EXCEPTION_HANDLE_END

    return HCCL_SUCCESS;
}

HcclResult HcclAllocNotify(
    HcclComm comm, CommEngine commEngine, ::NotifyType notifyType, uint32_t notifyNum, NotifyHandle** notifyHandleList)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        !IsValidCommEngine(commEngine),
        HCCL_ERROR(
            "[%s] commEngine[%s] is invalid", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str()),
        HCCL_E_PARA);
    CHK_PRT_RET(
        !IsValidNotify(notifyType), HCCL_ERROR("[%s] notifyType[%u] is invalid", __func__, notifyType), HCCL_E_PARA);
    CHK_PRT_RET(
        notifyNum > NOTIFY_MAX_NUM || notifyNum == 0, HCCL_ERROR("[%s] notifyNum[%u] is invalid", __func__, notifyNum),
        HCCL_E_PARA);
    CHK_PRT_RET(notifyHandleList == nullptr, HCCL_ERROR("[%s] notifyHandleList is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(*notifyHandleList != nullptr, HCCL_ERROR("[%s] notifyHandleList is not null", __func__), HCCL_E_PARA);

    if (commEngine == CommEngine::COMM_ENGINE_CPU || commEngine == CommEngine::COMM_ENGINE_CPU_TS
        || commEngine == CommEngine::COMM_ENGINE_CCU) {
        if (notifyType != ::NOTIFY_TYPE_RTS_NOTIFY) {
            HCCL_ERROR(
                "[%s] commEngine[%s] and notifyType[%u] are mismatch", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType);
            return HCCL_E_PARA;
        }
    } else {
        if (notifyType != ::NOTIFY_TYPE_DEVICE_MEM) {
            HCCL_ERROR(
                "[%s] commEngine[%s] and notifyType[%u] are mismatch", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType);
            return HCCL_E_PARA;
        }
    }

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s] commEngine[%s] notifyType[%u] notifyNum[%u]", __func__, commId.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType, notifyNum);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclAllocNotify(commEngine, notifyType, notifyNum, notifyHandleList);
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclAllocNotify(commEngine, notifyType, notifyNum, notifyHandleList);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to create notify for commEngine[%s]", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str());
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] Allocated notify for commEngine[%s], notifyType[%u], notifyNum[%u]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcommFreeNotify(HcclComm comm, uint32_t notifyNum, NotifyHandle* notifyHandleList)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(notifyHandleList == nullptr, HCCL_ERROR("[%s] notifyHandleList is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        notifyNum > NOTIFY_MAX_NUM || notifyNum == 0, HCCL_ERROR("[%s] notifyNum[%u] is invalid", __func__, notifyNum),
        HCCL_E_PARA);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s:comm[%s] notifyNum[%u]", __func__, commId.c_str(), notifyNum);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcommFreeNotify(notifyNum, notifyHandleList);
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcommFreeNotify(notifyNum, notifyHandleList);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to free notify", __func__);
        return ret;
    }

    HCCL_RUN_INFO("[%s] Free notify for notifyNum[%u]", __func__, notifyNum);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
HcclResult HcclThreadExportToCommEngine(
    HcclComm comm, uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine,
    ThreadHandle* exportedThreads)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(exportedThreads);
    CHK_PRT_RET(
        !IsValidCommEngine(dstCommEngine),
        HCCL_ERROR(
            "[%s] commEngine[%s] is invalid", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str()),
        HCCL_E_PARA);
    if (threadNum == 0 || threadNum > MAX_EXPORT_THREAD_NUM) {
        HCCL_ERROR("[%s] threadNum[%u] is 0 or greater than %u", __func__, threadNum, MAX_EXPORT_THREAD_NUM);
        return HCCL_E_PARA;
    }

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "Entry-[%s]:comm[%s], threadNum[%u], commEngine[%s], threadsPtr[%p], exportedThreadsPtr[%p]", __func__,
        commId.c_str(), threadNum, GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str(), threads,
        exportedThreads);
    HcclResult ret;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Thread export failed. Export threadNum[%u], commEngine[%s], threadsPtr[%p], exportedThreadsPtr[%p]",
            __func__, threadNum, GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str(), threads,
            exportedThreads),
        ret);
    HCCL_INFO("[%s]:comm[%s] export success.", __func__, commId.c_str());
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif

HcclResult
HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(info);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "Entry-[%s]:comm[%s], thread[0x%llx], resType[%d], infoLen[%u], info[%p]", __func__, commId.c_str(),
        static_cast<unsigned long long>(thread), static_cast<int32_t>(resType), infoLen, info);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadResGetInfo(thread, resType, infoLen, info);
    } else {
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (devType != DevType::DEV_TYPE_910B) { // 910B HOST网卡需要走此流程，不打印错误日志
            HCCL_ERROR("[%s] communicatorType is not supported.", __func__);
        }
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] thread resource get info failed. thread[0x%llx], resType[%d], infoLen[%u], info[%p]", __func__,
            static_cast<unsigned long long>(thread), static_cast<int32_t>(resType), infoLen, info),
        ret);
    HCCL_INFO("[%s]:comm[%s] get thread resource success.", __func__, commId.c_str());
    return HCCL_SUCCESS;
}
