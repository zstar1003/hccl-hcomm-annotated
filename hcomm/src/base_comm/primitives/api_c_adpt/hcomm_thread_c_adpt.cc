/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "hcomm_thread_c_adpt.h"
#include "hcomm_res.h"
#include "hcomm_res_defs.h"
#include "../hcomm_res_mgr.h"
#include "log.h"
#include "thread.h"
#include "cpu_ts_thread.h"
#include "param_check_pub.h"
#include "comm_engine_utils.h"
#include "exception_handler.h"
#include "adapter_rts_common.h"
#include "aicpu_ts_channel_helper.h"
#include "aicpu_launch_manager.h"

namespace hcomm {
static std::unordered_map<ThreadHandle, std::shared_ptr<hccl::Thread>> g_ThreadMap;
static std::mutex g_ThreadMapMtx;
} // namespace hcomm

using namespace hcomm;

HcommResult
HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNumPerThread, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(notifyNumPerThread);
    (void)HcommResMgrInit();
    const uint32_t notifyNum = notifyNumPerThread[0];
    if (threadNum > 1U) {
        HCCL_RUN_WARNING(
            "[%s] only notifyNumPerThread[0] is used currently, threadNum[%u], notifyNum[0][%u].", __func__, threadNum,
            notifyNum);
    }
    HCCL_INFO(
        "[%s] ThreadAcquire begin. engine[%s], threadNum[%u], notifyPerThread[%u], threads[%p]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, notifyNum, threads);
    CHK_RET(RefreshCommEngineContext(engine));

    // 1. 参数校验
    CHK_RET(hccl::ValidateThreadParams(threadNum, notifyNum));

    // 2. 获取引擎对应的类型
    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::CommEngineToNotifyLoadType(engine, notifyLoadType));
    CHK_RET(hccl::CommEngineToStreamType(engine, streamType));

    // 3. 创建线程
    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    hccl::ThreadCreateParams params(engine, threadNum, notifyNum, notifyLoadType, streamType);
    CHK_RET(hccl::CreateAndInitThreads(params, newThreads));

    // 4. 插入全局映射表
    CHK_RET(hccl::SaveThreads(newThreads));

    // 5. 储存线程句柄
    CHK_RET(AicpuTsChannelHelper::EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, threads, engine, AicpuTsChannelHelper::GetBinHandle()));

    HCCL_INFO(
        "[HcommThreadAlloc] ThreadAcquire done: engine[%s] threadNum[%u], notifyPerThread[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, notifyNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads)
{
    return ::HcommThreadAlloc(engine, threadNum, &notifyNumPerThread, threads);
}

HcommResult HcommThreadAllocWithConfig(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(config);
    CHK_PRT_RET(
        type == THREAD_TYPE_INVALID,
        HCCL_ERROR("[%s] thread type[%d] is invalid", __func__, static_cast<int32_t>(type)),
        static_cast<HcommResult>(HCCL_E_PARA));
    CHK_PRT_RET(
        engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_CPU_TS,
        HCCL_ERROR(
            "[%s] commEngine[%d] CPU_TS/AICPU_TS not supported, use engine with ThreadType instead", __func__,
            static_cast<int32_t>(engine)),
        static_cast<HcommResult>(HCCL_E_PARA));
    CHK_PRT_RET(
        engine == COMM_ENGINE_AIV || engine == COMM_ENGINE_CCU,
        HCCL_ERROR(
            "[%s] commEngine[%d] AIV/CCU not supported, supported engines: CPU/AICPU", __func__,
            static_cast<int32_t>(engine)),
        static_cast<HcommResult>(HCCL_E_PARA));
    CHK_PRT_RET(
        threadNum == 0, HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum),
        static_cast<HcommResult>(HCCL_E_PARA));
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    CHK_RET(RefreshCommEngineContext(engine));

    HCCL_INFO(
        "[%s] begin. engine[%d], threadType[%d], threadNum[%u], threads[%p]", __func__, engine,
        static_cast<int32_t>(type), threadNum, threads);

    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::GetNotifyLoadType(engine, type, notifyLoadType));
    CHK_RET(hccl::GetStreamType(engine, type, streamType));

    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    newThreads.reserve(threadNum);
    for (uint32_t i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x], call ThreadConfigInit first", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            (HcommResult)HCCL_E_PARA);
        CHK_RET(hccl::ValidateThreadParams(1, config[i].notifyNumPerThread));
        std::shared_ptr<hccl::Thread> threadPtr;
        HcclResult ret
            = hccl::CreateThread(engine, streamType, config[i].notifyNumPerThread, notifyLoadType, threadPtr);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Failed to create thread at index[%u], ret[%d]", __func__, i, ret),
            static_cast<HcommResult>(ret));
        ret = threadPtr->Init();
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Failed to init thread at index[%u], ret[%d]", __func__, i, ret),
            (HcommResult)ret);
        newThreads.emplace_back(std::move(threadPtr));
    }

    CHK_RET(hccl::SaveThreads(newThreads));
    CHK_RET(AicpuTsChannelHelper::EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, threads, engine, AicpuTsChannelHelper::GetBinHandle()));

    HCCL_INFO(
        "[%s] done: engine[%d] threadType[%d] threadNum[%u]", __func__, engine, static_cast<int32_t>(type), threadNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    (void)HcommResMgrInit();
    return hccl::FreeThreads(threads, threadNum, AicpuTsChannelHelper::GetBinHandle());
}

HcommResult HcommThreadAllocWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_PTR_NULL(thread);
    hccl::NotifyLoadType notifyLoadType;
    CHK_RET(CommHostEngineToNotifyLoadType(engine, notifyLoadType));
    std::shared_ptr<hccl::Thread> handle;
    EXCEPTION_CATCH(handle = std::make_shared<hccl::CpuTsThread>(stream, notifyNum, notifyLoadType), return HCCL_E_PTR);
    CHK_RET(handle->Init());

    // 返回第一个句柄
    *thread = ReinterpretAs<ThreadHandle>(handle.get());
    {
        std::lock_guard<std::mutex> lock(hcomm::g_ThreadMapMtx);
        hcomm::g_ThreadMap.emplace(*thread, handle);
    }
    std::vector<std::shared_ptr<hccl::Thread>> threads{handle};
    CHK_RET(hccl::SaveThreads(threads));

    HCCL_INFO(
        "[ThreadMgr] ThreadAcquireWithStream done: engine[%s] stream[%p], "
        "notifyNum[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), stream, notifyNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadFreeWithStream(const ThreadHandle* threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    if (threadNum == 0U) {
        HCCL_ERROR("[%s] threadNum is 0", __func__);
        return HCCL_E_PARA;
    }
    HcommResult ret = HCCL_SUCCESS;
    std::lock_guard<std::mutex> lock(hcomm::g_ThreadMapMtx);
    for (uint32_t i = 0; i < threadNum; ++i) {
        ThreadHandle handle = threads[i];
        auto it = hcomm::g_ThreadMap.find(handle);
        if (it == hcomm::g_ThreadMap.end()) {
            HCCL_WARNING("[%s] thread handle[0x%llx] not found in g_ThreadMap, skip", __func__, handle);
            continue;
        }
        HcclResult deInitRet = it->second->DeInit();
        if (deInitRet != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] thread DeInit failed, ret[%d], handle[0x%llx]", __func__, deInitRet, handle);
            ret = static_cast<HcommResult>(deInitRet);
        }
        hcomm::g_ThreadMap.erase(it);
        HCCL_INFO("[%s] thread freed, handle[0x%llx]", __func__, handle);
    }
    return ret;
}

HcommResult HcommThreadSupplementNotify(
    CommEngine engine, ThreadHandle* handles, uint32_t threadNum, uint32_t* supplementNotifyNums)
{
    CHK_PTR_NULL(handles);
    CHK_PTR_NULL(supplementNotifyNums);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);

    std::vector<std::shared_ptr<hccl::Thread>> needSupplementThread;
    std::unique_ptr<ThreadHandle[]> threadHandle;
    EXCEPTION_CATCH(
        threadHandle = std::make_unique<ThreadHandle[]>(threadNum), return static_cast<HcommResult>(HCCL_E_PTR));

    for (uint32_t i = 0; i < threadNum; ++i) {
        std::shared_ptr<hccl::Thread> threadPtr;
        CHK_RET(hccl::LookupThreadByHandle(handles[i], threadPtr));
        CHK_RET(threadPtr->SupplementNotify(supplementNotifyNums[i]));
        needSupplementThread.push_back(std::move(threadPtr));
        threadHandle[i] = handles[i];
    }

    // 设备侧 kernel launch（仅 AICPU 引擎触发）
    if (engine == COMM_ENGINE_AICPU && !needSupplementThread.empty()) {
        CHK_RET(HcommResMgr::EnsureKernelBinLoaded(engine));
        HcclResult ret = hccl::AicpuLaunchMgr::SupplementNotifyKernelLaunch(
            needSupplementThread, std::string(""), threadHandle, HcommResMgr::GetBinHandle());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] SupplementNotifyKernelLaunch failed, ret[%d]", __func__, ret),
            static_cast<HcommResult>(ret));
    }
    return HCCL_SUCCESS;
}

HcommResult HcommThreadGetNotifyNum(ThreadHandle thread, uint32_t* notifyNum)
{
    CHK_PTR_NULL(notifyNum);
    (void)HcommResMgrInit();
    std::shared_ptr<hccl::Thread> threadPtr;
    CHK_RET(hccl::LookupThreadByHandle(thread, threadPtr));
    *notifyNum = threadPtr->GetNotifyNum();
    HCCL_INFO("[%s] thread[0x%llx] notifyNum[%u]", __func__, thread, *notifyNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadExportToCommEngineAiCpu(
    ThreadHandle* handles, const std::string& commIdStr, uint32_t threadNum, CommEngine dstEngine,
    ThreadHandle* outHandles)
{
    // AICPU 方向：正查 FindThreadByCommEngine + miss 批量建 + 入表 + 映射
    std::vector<std::shared_ptr<hccl::Thread>> hostThreads;
    std::vector<uint32_t> missIdx;
    for (uint32_t i = 0; i < threadNum; ++i) {
        std::shared_ptr<hccl::Thread> threadPtr;
        CHK_RET(hccl::LookupThreadByHandle(handles[i], threadPtr));
        hccl::Thread* exported = threadPtr->FindThreadByCommEngine(dstEngine);
        if (exported != nullptr) {
            outHandles[i] = ReinterpretAs<ThreadHandle>(exported);
        } else {
            hostThreads.push_back(std::move(threadPtr));
            missIdx.push_back(i);
        }
    }
    if (!hostThreads.empty()) {
        CHK_RET(HcommResMgr::EnsureKernelBinLoaded(dstEngine));
        std::unique_ptr<ThreadHandle[]> aicpuHandle;
        EXCEPTION_CATCH(
            aicpuHandle = std::make_unique<ThreadHandle[]>(hostThreads.size()),
            return static_cast<HcommResult>(HCCL_E_PTR));
        HcclResult ret = hccl::AicpuLaunchMgr::ThreadKernelLaunchForComm(
            hostThreads, commIdStr, aicpuHandle, HcommResMgr::GetBinHandle());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] ThreadKernelLaunchForComm failed, ret[%d]", __func__, ret),
            static_cast<HcommResult>(ret));
        for (size_t i = 0; i < hostThreads.size(); ++i) {
            outHandles[missIdx[i]] = aicpuHandle[i];
            CHK_RET(hostThreads[i]->AddThreadHandleToMap(dstEngine, aicpuHandle[i]));
            // 入 g_ThreadD2HMap（device->host）
            ThreadHandle hostHandle = ReinterpretAs<ThreadHandle>(hostThreads[i].get());
            CHK_RET(hccl::FillThreadD2HMap(&aicpuHandle[i], &hostHandle, 1));
        }
    }

    return HCCL_SUCCESS;
}

HcommResult HcommThreadExportToCommEngine(
    ThreadHandle* hostHandles, const char* commId, uint32_t threadNum, CommEngine dstEngine,
    ThreadHandle* outDeviceHandles)
{
    CHK_PTR_NULL(hostHandles);
    CHK_PTR_NULL(outDeviceHandles);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    CHK_RET(RefreshCommEngineContext(dstEngine));
    const std::string commIdStr = (commId != nullptr) ? std::string(commId) : std::string();
    switch (dstEngine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU: {
            // CPU 方向：反向查询 g_ThreadD2HMap（device 到 host 映射）
            for (uint32_t i = 0; i < threadNum; ++i) {
                CHK_RET(hccl::LookupD2HHandle(hostHandles[i], outDeviceHandles[i]));
            }
            return HCCL_SUCCESS;
        }
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS: {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadExportToCommEngineAiCpu(hostHandles, commIdStr, threadNum, dstEngine, outDeviceHandles)));
            break;
        }
        default:
            HCCL_ERROR("[%s] unsupported dstEngine[%d]", __func__, static_cast<int32_t>(dstEngine));
            return static_cast<HcommResult>(HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcommResult HcommThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info)
{
    CHK_PTR_NULL(info);
    CHK_PRT_RET(thread == 0, HCCL_ERROR("[%s] thread is 0", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        resType == ThreadResType::THREAD_RES_TYPE_INVALID,
        HCCL_ERROR("[%s] resType[%d] is invalid", __func__, static_cast<int32_t>(resType)), HCCL_E_PARA);

    HCCL_INFO(
        "[%s] begin, thread[0x%llx], resType[%d], infoLen[%u]", __func__, thread, static_cast<int32_t>(resType),
        infoLen);

    std::shared_ptr<hccl::Thread> threadPtr;
    CHK_RET(hccl::LookupThreadByHandle(thread, threadPtr));

    if (resType != ThreadResType::THREAD_RES_TYPE_STREAM) {
        HCCL_ERROR("[%s] resType[%d] is not supported", __func__, static_cast<int32_t>(resType));
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PRT_RET(
        infoLen != sizeof(ThreadResTypeStream),
        HCCL_ERROR(
            "[%s] infoLen[%u] mismatch sizeof(ThreadResTypeStream)[%zu]", __func__, infoLen,
            sizeof(ThreadResTypeStream)),
        HCCL_E_PARA);

    hccl::Stream* streamPtr = threadPtr->GetStream();
    CHK_PTR_NULL(streamPtr);
    ThreadResTypeStream stream = streamPtr->ptr();
    CHK_PTR_NULL(stream);

    *info = stream;

    HCCL_INFO(
        "[%s] success, thread[0x%llx] resType[%d] stream[%p]", __func__, thread, static_cast<int32_t>(resType), *info);
    return HCCL_SUCCESS;
}
