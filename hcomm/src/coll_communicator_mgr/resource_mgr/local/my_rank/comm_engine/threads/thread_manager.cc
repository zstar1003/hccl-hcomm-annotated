/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_manager.h"
#include <cstring>
#include "adapter_rts_common.h"
#include "aicpu_launch_manager.h"
#include "independent_op.h"
#include "comm_engine_utils.h"
#include "hcomm_res.h"
#include "thread.h"
#include "dfx_dlprof_function.h"

namespace hccl {

ThreadMgr::ThreadMgr(
    uint32_t threadNum, uint32_t notifyNumPerThread, std::string commId, aclrtBinHandle binHandle,
    const ManagerCallbacks& callbacks)
    : threadNum_(threadNum),
      notifyNumPerThread_(notifyNumPerThread),
      commId_(commId),
      binHandle_(binHandle),
      callbacks_(callbacks)
{}

ThreadMgr::~ThreadMgr()
{
    auto it = dedicatedThreadMap_.find(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH);
    if (it != dedicatedThreadMap_.end()) {
        ThreadHandle thread = it->second;
        HcommThreadFree(&thread, 1);
    }
}

uint64_t ThreadMgr::GetMaxNotifyTotal()
{
    // 如果没设定最大值，设置一下
    uint64_t maxNotifyTotal = 0;
    if (threadNum_ == HCCL_COMM_THREADNUM_CONFIG_NOT_SET
        && notifyNumPerThread_ == HCCL_COMM_NOTIFY_NUM_PER_THREAD_CONFIG_NOT_SET) {
        maxNotifyTotal = HCCL_THREAD_NOTIFY_MAX_NUM;
        threadNum_ = SIGNAL_DEV_STREAM_MAX_NUM;
        notifyNumPerThread_ = HCCL_THREAD_NOTIFY_MAX_NUM;
    } else {
        maxNotifyTotal = static_cast<uint64_t>(threadNum_) * static_cast<uint64_t>(notifyNumPerThread_);
        maxNotifyTotal = maxNotifyTotal > HCCL_THREAD_NOTIFY_MAX_NUM ? HCCL_THREAD_NOTIFY_MAX_NUM : maxNotifyTotal;
    }
    return maxNotifyTotal;
}

HcclResult ThreadMgr::CheckNotifyNum(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread)
{
    uint64_t maxNotifyTotal = GetMaxNotifyTotal();
    const uint64_t used = usedNotifyNum_;
    uint64_t remainNotifyQuota = (maxNotifyTotal > used) ? (maxNotifyTotal - used) : 0;
    uint64_t needNotifyTotal = static_cast<uint64_t>(threadNum) * static_cast<uint64_t>(notifyNumPerThread);
    if (remainNotifyQuota < needNotifyTotal || notifyNumPerThread > notifyNumPerThread_
        || maxNotifyTotal > HCCL_THREAD_NOTIFY_MAX_NUM) {
        HCCL_ERROR(
            "[ThreadMgr][%s] Notify quota exhausted: remainQuota[%llu], total[%llu], used[%llu], need[%llu], "
            "setPreNum[%u], allocPreNum[%u]",
            __func__, remainNotifyQuota, maxNotifyTotal, used, needNotifyTotal, notifyNumPerThread_,
            notifyNumPerThread);
        return HCCL_E_UNAVAIL;
    }

    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire quota: engine[%s], "
        "remainNotifyQuota[%llu]",
        __func__, commId_.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), remainNotifyQuota);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::CheckThreadNum(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread)
{
    GetMaxNotifyTotal();
    uint32_t remainQuota = (threadNum_ > threads_.size()) ? (threadNum_ - threads_.size()) : 0;
    if (remainQuota == 0 || threadNum > remainQuota) {
        HCCL_ERROR(
            "[ThreadMgr][%s] Threads quota exhausted: remainQuota[%u], need[%u].", __func__, remainQuota, threadNum);
        return HCCL_E_UNAVAIL;
    }

    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire quota: engine[%s] threadNum[%u].", __func__, commId_.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), remainQuota);
    return CheckNotifyNum(engine, threadNum, notifyNumPerThread);
}

HcclResult ThreadMgr::SupplementNotify(CommEngine engine, std::vector<std::shared_ptr<Thread>>& needSupplementThread)
{
    HcclResult ret = HCCL_E_INTERNAL;
    // thread资源 AICPU侧展开
    if ((engine == COMM_ENGINE_AICPU) && needSupplementThread.size() > 0) {
        if (!callbacks_.getAicpuCommState()) {
            HCCL_INFO("ThreadMgr::SupplementNotify kernelLaunchAicpuCommInit start");
            ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret),
                ret);
            callbacks_.setAicpuCommState(true);
        }

        std::unique_ptr<ThreadHandle[]> threadHandle;
        EXCEPTION_CATCH(
            threadHandle = std::make_unique<ThreadHandle[]>(needSupplementThread.size()), return HCCL_E_PTR);
        for (size_t i = 0; i < needSupplementThread.size(); ++i) {
            threadHandle[i] = hostToDeviceThreadHandle_[reinterpret_cast<ThreadHandle>(needSupplementThread[i].get())];
        }
        HCCL_INFO(
            "ThreadMgr::SupplementNotify ThreadKernelLaunchForComm start needSupplementThread size[%zu]",
            needSupplementThread.size());
        ret = AicpuLaunchMgr::SupplementNotifyKernelLaunch(needSupplementThread, commId_, threadHandle, binHandle_);
        HCCL_INFO("ThreadMgr::SupplementNotify ThreadKernelLaunchForComm end");
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[ThreadMgr][SupplementNotify] AiCpuKernelLaunch failed, return [%d].", ret), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::SupplementNotify(CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config)
{
    NotifyLoadType notifyLoadType;
    CHK_RET(GetNotifyLoadType(engine, type, notifyLoadType));
    std::vector<std::shared_ptr<Thread>>& threads = engineToThreadsMap_[std::make_pair(engine, type)];
    std::vector<std::shared_ptr<Thread>> needSupplementThread;
    HcclResult ret = HCCL_E_INTERNAL;
    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] threadNum[%zu] notifyLoadType[%d]", __func__, commId_.c_str(), threads.size(),
        static_cast<int32_t>(notifyLoadType));

    for (size_t i = 0; i < threads.size() && i < threadNum; ++i) {
        HCCL_INFO(
            "[ThreadMgr][%s] Hcom[%s] AicpuTsThread[%u] notifyLoadType[%d] notifyNum[%u]", __func__, commId_.c_str(),
            static_cast<uint32_t>(i), static_cast<int32_t>(notifyLoadType), config[i].notifyNumPerThread);
        const u32 notifyNum = threads[i]->GetNotifyNum();
        if (config[i].notifyNumPerThread > notifyNum) {
            u32 supplementNotifyNum = config[i].notifyNumPerThread - notifyNum;
            CHK_RET(CheckNotifyNum(engine, 1, supplementNotifyNum));
            ret = threads[i]->SupplementNotify(supplementNotifyNum);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[ThreadMgr][%s] Failed to supplement notify[%u] on thread index[%u], ret[%d]", __func__,
                    supplementNotifyNum, i, ret);
                return ret;
            }
            needSupplementThread.push_back(threads[i]);
            usedNotifyNum_ += supplementNotifyNum;
        }
    }
    CHK_RET(SupplementNotify(engine, needSupplementThread));
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::SupplementThread(
    CommEngine engine, std::vector<std::shared_ptr<Thread>>& newThreads, std::unique_ptr<ThreadHandle[]>& hostHandle)
{
    // thread资源 AICPU侧展开
    if (engine == COMM_ENGINE_AICPU) {
        HcclResult ret = HCCL_E_INTERNAL;
        if (!callbacks_.getAicpuCommState()) {
            HCCL_INFO("ThreadMgr::HcclAllocThreadRes kernelLaunchAicpuCommInit start");
            ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret),
                ret);
            callbacks_.setAicpuCommState(true);
        }

        EXCEPTION_CATCH(hostHandle = std::make_unique<ThreadHandle[]>(newThreads.size()), return HCCL_E_PTR);
        HCCL_INFO("ThreadMgr::HcclAllocThreadRes ThreadKernelLaunchForComm start");
        ret = AicpuLaunchMgr::ThreadKernelLaunchForComm(newThreads, commId_, hostHandle, binHandle_);
        HCCL_INFO("ThreadMgr::HcclAllocThreadRes ThreadKernelLaunchForComm end");
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] AiCpuKernelLaunch failed, return [%d].", ret), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::SupplementThread(
    CommEngine engine, uint32_t supplementThreadNum, ThreadType type, const ThreadConfig* config)
{
    NotifyLoadType notifyLoadType;
    StreamType streamType;
    CHK_RET(GetNotifyLoadType(engine, type, notifyLoadType));
    CHK_RET(GetStreamType(engine, type, streamType));
    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] supplementThreadNum[%u] notifyLoadType[%d], streamType[%d]", __func__,
        commId_.c_str(), supplementThreadNum, static_cast<int32_t>(notifyLoadType), static_cast<int32_t>(streamType));
    std::vector<std::shared_ptr<Thread>> newThreads;
    newThreads.reserve(supplementThreadNum);
    HcclResult ret = HCCL_E_INTERNAL;

    for (uint32_t i = 0; i < supplementThreadNum; ++i) {
        std::shared_ptr<Thread> handle;
        HCCL_INFO(
            "[ThreadMgr][%s] Hcom[%s] [%u]notifyLoadType[%d], streamType[%d] notifyNum[%u]", __func__, commId_.c_str(),
            i, static_cast<int32_t>(notifyLoadType), static_cast<int32_t>(streamType), config[i].notifyNumPerThread);
        CHK_RET(CheckThreadNum(engine, 1, config[i].notifyNumPerThread));
        CHK_RET(CreateThread(engine, streamType, config[i].notifyNumPerThread, notifyLoadType, handle));
        ret = handle->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] Failed to init thread index[%u], ret[%d]", i, ret);
            return ret;
        }
        usedNotifyNum_ += config[i].notifyNumPerThread;
        newThreads.emplace_back(std::move(handle));
    }

    std::unique_ptr<ThreadHandle[]> hostHandle;
    CHK_RET(SupplementThread(engine, newThreads, hostHandle));
    threads_.reserve(threads_.size() + newThreads.size());
    auto iter = engineToThreadsMap_.find(std::make_pair(engine, type));
    if (iter == engineToThreadsMap_.end()) {
        HCCL_ERROR(
            "[SupplementThread] engine+type pair not found in engineToThreadsMap_ "
            "engine[%d], type[%d].",
            engine, type);
        return HCCL_E_INTERNAL;
    }
    auto& threadVec = iter->second;
    threadVec.reserve(threadVec.size() + newThreads.size());
    threadVec.insert(threadVec.end(), newThreads.begin(), newThreads.end());
    threads_.insert(threads_.end(), newThreads.begin(), newThreads.end());

    if (engine == COMM_ENGINE_AICPU) {
        for (size_t i = 0; i < newThreads.size(); ++i) {
            ThreadHandle cpuTsHandle = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            newThreads[i]->AddThreadHandleToMap(engine, hostHandle[i]);
            hostToDeviceThreadHandle_[cpuTsHandle] = hostHandle[i];
            threadHandleOthersToCpu_[hostHandle[i]] = cpuTsHandle;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadAcquireV2(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
    std::vector<uint32_t>& threadId)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(config);
    if (threadNum == 0) {
        HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] threadNum is 0");
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[ThreadMgr][%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x]", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            HCCL_E_PARA);
    }

    std::lock_guard<std::mutex> lock(threadMutex_);
    std::lock_guard<std::mutex> lockMap(threadMapMutex_);
    std::lock_guard<std::mutex> engineToThreadMtx(engineToThreadMutex_);
    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire begin, max: engine[%s] threadNum[%u],"
        "notifyPerThread[%u], need: threadNum[%u], threadType[%d]",
        __func__, commId_.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum_,
        notifyNumPerThread_, threadNum, static_cast<int32_t>(type));

    // 1、thread上的notify数量不够，需要给thread补充notify
    auto it = engineToThreadsMap_.find(std::make_pair(engine, type));
    if (it == engineToThreadsMap_.end()) {
        it = engineToThreadsMap_.emplace(std::make_pair(engine, type), std::vector<std::shared_ptr<Thread>>{}).first;
    }
    auto& threadVec = it->second;
    if (threadVec.size() > 0) {
        // 调用补充函数，如果engine是COMM_ENGINE_AICPU，需要去device恢复
        CHK_RET(SupplementNotify(engine, threadNum, type, config));
    }

    // 2、补充thread
    if (threadVec.size() < threadNum) {
        u32 supplementThreadNum = threadNum - threadVec.size();
        // 调用补充函数，如果engine是COMM_ENGINE_AICPU，需要去device恢复
        CHK_RET(SupplementThread(engine, supplementThreadNum, type, &config[threadVec.size()]));
    }
    // 3、返回threadHandle和id
    for (u32 idx = 0; idx < threadNum; idx++) {
        ThreadHandle handle = reinterpret_cast<ThreadHandle>(threadVec[idx].get());
        threads[idx] = (engine == COMM_ENGINE_AICPU) ? hostToDeviceThreadHandle_[handle] : handle;
        uint32_t id = threadVec[idx]->GetStream()->sqId();
        HCCL_DEBUG("[%s]idx[%u] threadHandle[%llu] thread id = [%u]", __func__, idx, threads[idx], id);
        threadId.push_back(id);

        std::lock_guard<std::mutex> threadhandleToThreadMtx(threadhandleToThreadMutex_);
        threadMap_[threads[idx]] = threadVec[idx];
    }

    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire done: engine[%s] threadNum[%u]%s", __func__, commId_.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum,
        (engine == COMM_ENGINE_AICPU) ? " (AICPU token ready)" : "");
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::CreateAndInitThreads(
    CommEngine engine, StreamType streamType, NotifyLoadType notifyLoadType, uint32_t threadNum,
    const ThreadConfig* config, std::vector<std::shared_ptr<Thread>>& newThreads)
{
    for (uint32_t i = 0; i < threadNum; ++i) {
        std::shared_ptr<Thread> handle;
        HCCL_INFO(
            "[ThreadMgr][%s] Hcom[%s] AicpuTsThread notifyLoadType[%d], streamType[%d], notifyPerThread[%u]", __func__,
            commId_.c_str(), static_cast<int32_t>(notifyLoadType), static_cast<int32_t>(streamType),
            config[i].notifyNumPerThread);
        CHK_RET(CheckThreadNum(engine, 1, config[i].notifyNumPerThread));
        CHK_RET(CreateThread(engine, streamType, config[i].notifyNumPerThread, notifyLoadType, handle));
        HcclResult ret = handle->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] Failed to init thread index %u", i);
            return ret;
        }
        usedNotifyNum_ += config[i].notifyNumPerThread;
        newThreads.emplace_back(std::move(handle));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::AssignThreadHandles(
    CommEngine engine, std::vector<std::shared_ptr<Thread>>& newThreads, ThreadHandle* threads,
    std::unique_ptr<ThreadHandle[]>& hostHandle)
{
    if (engine == COMM_ENGINE_AICPU) {
        CHK_RET(SupplementThread(engine, newThreads, hostHandle));
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = hostHandle[i];
            HCCL_INFO("[ThreadMgr][%s] aicpu threadArray[%u] = [%llu]", __func__, i, threads[i]);
        }
    } else {
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            HCCL_INFO("[ThreadMgr][%s] host threadArray[%u] = [%llu]", __func__, i, threads[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::StoreThreadsAndBuildHandleMap(
    CommEngine engine, std::vector<std::shared_ptr<Thread>>& newThreads, std::unique_ptr<ThreadHandle[]>& hostHandle)
{
    auto threadsIt = threads_.insert(
        threads_.end(), std::make_move_iterator(newThreads.begin()), std::make_move_iterator(newThreads.end()));

    if (engine == COMM_ENGINE_AICPU) {
        for (size_t i = 0; i < newThreads.size(); ++i, ++threadsIt) {
            ThreadHandle cpuTsHandle = reinterpret_cast<ThreadHandle>((*threadsIt).get());
            (*threadsIt)->AddThreadHandleToMap(engine, hostHandle[i]);
            hostToDeviceThreadHandle_[cpuTsHandle] = hostHandle[i];
            threadHandleOthersToCpu_[hostHandle[i]] = cpuTsHandle;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadAcquire(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
    std::vector<uint32_t>& threadId)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(config);
    std::lock_guard<std::mutex> lock(threadMutex_);
    std::lock_guard<std::mutex> lockMap(threadMapMutex_);
    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire begin, max: engine[%d] threadNum[%u],"
        "notifyPerThread[%u], need: threadNum[%u], threadType[%d]",
        __func__, commId_.c_str(), engine, threadNum_, notifyNumPerThread_, threadNum, static_cast<int32_t>(type));

    if (threadNum == 0) {
        HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] threadNum is 0");
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[ThreadMgr][%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x]", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            HCCL_E_PARA);
    }

    NotifyLoadType notifyLoadType;
    StreamType streamType;
    CHK_RET(GetNotifyLoadType(engine, type, notifyLoadType));
    CHK_RET(GetStreamType(engine, type, streamType));

    std::vector<std::shared_ptr<Thread>> newThreads;
    newThreads.reserve(threadNum);
    CHK_RET(CreateAndInitThreads(engine, streamType, notifyLoadType, threadNum, config, newThreads));

    std::unique_ptr<ThreadHandle[]> hostHandle;
    CHK_RET(AssignThreadHandles(engine, newThreads, threads, hostHandle));

    for (size_t i = 0; i < newThreads.size(); ++i) {
        uint32_t id = newThreads[i]->GetStream()->id();
        HCCL_DEBUG("[%s] thread id = [%u]", __func__, id);
        threadId.push_back(id);
    }
    threads_.reserve(threads_.size() + newThreads.size());
    CHK_RET(StoreThreadsAndBuildHandleMap(engine, newThreads, hostHandle));

    HCCL_INFO(
        "[ThreadMgr][HcclThreadAcquire] Hcom[%s] HcclThreadAcquire done: engine[%s] threadNum[%u]%s", commId_.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum,
        (engine == COMM_ENGINE_AICPU) ? " (AICPU token ready)" : "");
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclGetNotifyNumInThread(ThreadHandle thread, uint32_t* notifyNum)
{
    CHK_PTR_NULL(notifyNum);
    Thread* hcclThread = reinterpret_cast<Thread*>(thread);
    CHK_PTR_NULL(hcclThread);
    *notifyNum = hcclThread->GetNotifyNum();
    HCCL_INFO("[ThreadMgr] Hcom[%s] HcclGetNotifyNumInThread done: notifyPerThread[%u]", commId_.c_str(), *notifyNum);
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::HcclThreadAcquireWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_PTR_NULL(thread);

    if (mainThread_.find(stream) != mainThread_.end()) {
        if (mainThread_[stream]->GetNotifyNum() < notifyNum) {
            u32 supplementNotifyNum = notifyNum - mainThread_[stream]->GetNotifyNum();
            CHK_RET(mainThread_[stream]->SupplementNotify(supplementNotifyNum));
        }
        *thread = reinterpret_cast<ThreadHandle>(mainThread_[stream].get());
        return HCCL_SUCCESS;
    }

    NotifyLoadType notifyLoadType;
    CHK_RET(CommHostEngineToNotifyLoadType(engine, notifyLoadType));
    std::shared_ptr<CpuTsThread> handle;
    EXCEPTION_CATCH(handle = std::make_shared<CpuTsThread>(stream, notifyNum, notifyLoadType), return HCCL_E_PTR);
    CHK_RET(handle->Init());
    handle->SetIsMaster(true);

    // 返回第一个句柄
    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    mainThread_.emplace(stream, std::move(handle));
    *thread = reinterpret_cast<ThreadHandle>(mainThread_[stream].get());
    std::lock_guard<std::mutex> threadhandleToThreadMtx(threadhandleToThreadMutex_);
    threadMap_[*thread] = mainThread_[stream];
    HCCL_INFO(
        "[ThreadMgr] Hcom[%s] HcclThreadAcquireWithStream done: engine[%s] stream[%p],"
        "notifyNum[%u]",
        commId_.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), stream, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::ThreadExportToCommEngineCpu(uint32_t threadNum, const ThreadHandle* threads, ThreadHandle* exportedThreads)
{
    std::lock_guard<std::mutex> lock(threadMapMutex_);
    for (u32 i = 0; i < threadNum; i++) {
        if (threadHandleOthersToCpu_.find(threads[i]) == threadHandleOthersToCpu_.end()) {
            HCCL_ERROR("[CommEngineResMgr]%s Unknown ThreadHandle[%llu]", __func__, threads[i]);
            return HCCL_E_PARA;
        }
        exportedThreads[i] = threadHandleOthersToCpu_[threads[i]];
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::GetExportedThread(
    const ThreadHandle threadHandle, CommEngine commEngine, Thread*& exportedThread, std::shared_ptr<Thread>& threadOut)
{
    Thread* threadPtr = reinterpret_cast<Thread*>(threadHandle);
    for (auto& thread : threads_) {
        if (thread.get() == threadPtr) {
            exportedThread = thread->FindThreadByCommEngine(commEngine);
            threadOut = thread;
            return HCCL_SUCCESS;
        }
    }

    for (auto& pair : mainThread_) {
        if (pair.second.get() == threadPtr) {
            exportedThread = pair.second->FindThreadByCommEngine(commEngine);
            threadOut = pair.second;
            return HCCL_SUCCESS;
        }
    }

    if (orderLaunchThreads_.find(threadHandle) != orderLaunchThreads_.end()) {
        exportedThread = threadPtr->FindThreadByCommEngine(commEngine);
        threadOut = std::shared_ptr<Thread>(threadPtr, [](Thread*) {});
        HCCL_DEBUG(
            "[ThreadMgr][%s] found order launch thread[0x%llx] in comm[%s]", __func__, threadHandle, commId_.c_str());
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[ThreadMgr][%s]Unknown ThreadHandle[%llu]", __func__, threadHandle);
    return HCCL_E_PARA;
}
HcclResult ThreadMgr::RegisterOrderLaunchThread(ThreadHandle thread)
{
    orderLaunchThreads_.insert(thread);
    HCCL_INFO(
        "[ThreadMgr][%s] register order launch thread[0x%llx], comm[%s], total[%zu]", __func__, thread, commId_.c_str(),
        orderLaunchThreads_.size());
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ExportHostThreadsToAicpu(
    std::vector<std::shared_ptr<Thread>>& hostThreads, const std::vector<u32>& index, const ThreadHandle* threads,
    CommEngine dstCommEngine, ThreadHandle* exportedThreads)
{
    if (hostThreads.empty()) {
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(threadMapMutex_);
    if (!callbacks_.getAicpuCommState()) {
        HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret), ret);
        callbacks_.setAicpuCommState(true);
    }
    std::unique_ptr<ThreadHandle[]> aicpuHandle;
    EXCEPTION_CATCH(aicpuHandle = std::make_unique<ThreadHandle[]>(hostThreads.size()), return HCCL_E_PTR);
    uint64_t beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HcclResult ret = AicpuLaunchMgr::ThreadKernelLaunchForComm(hostThreads, commId_, aicpuHandle, binHandle_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[ThreadMgr][HcclThreadExportToCommEngine] AiCpuKernelLaunch failed, return [%d].", ret), ret);
    if (callbacks_.reportProfilingKernel != nullptr) {
        ret = callbacks_.reportProfilingKernel(beginTime, "RunAicpuIndOpThreadInit");
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][HcclThreadExportToCommEngine] ReportProfilingAiCpuKernelLaunch failed, return [%d].", ret),
            ret);
    }
    for (size_t i = 0; i < hostThreads.size(); ++i) {
        exportedThreads[index[i]] = aicpuHandle[i];
        CHK_RET(hostThreads[i]->AddThreadHandleToMap(dstCommEngine, aicpuHandle[i]));
        threadHandleOthersToCpu_[aicpuHandle[i]] = threads[index[i]];
        HCCL_INFO("[ThreadMgr][%s] aicpu threadArray[%u] = [%llu]", __func__, i, aicpuHandle[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ExportOrderLaunchThreadsToAicpu(
    std::vector<std::shared_ptr<Thread>>& orderLaunchHostThreads, const std::vector<u32>& orderLaunchIndex,
    const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads)
{
    if (orderLaunchHostThreads.empty()) {
        return HCCL_SUCCESS;
    }
    std::unique_ptr<ThreadHandle[]> aicpuHandle;
    EXCEPTION_CATCH(aicpuHandle = std::make_unique<ThreadHandle[]>(orderLaunchHostThreads.size()), return HCCL_E_PTR);
    HcclResult ret = AicpuLaunchMgr::ThreadKernelLaunchForBase(orderLaunchHostThreads, aicpuHandle, binHandle_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ThreadMgr][HcclThreadExportToCommEngine] order launch ThreadKernelLaunchForBase failed, return [%d].",
            ret),
        ret);
    for (size_t i = 0; i < orderLaunchHostThreads.size(); ++i) {
        exportedThreads[orderLaunchIndex[i]] = aicpuHandle[i];
        CHK_RET(orderLaunchHostThreads[i]->AddThreadHandleToMap(dstCommEngine, aicpuHandle[i]));
        threadHandleOthersToCpu_[aicpuHandle[i]] = threads[orderLaunchIndex[i]];
        HCCL_INFO("[ThreadMgr][%s] order launch aicpu threadArray[%u] = [%llu]", __func__, i, aicpuHandle[i]);
    }
    return HCCL_SUCCESS;
}
HcclResult ThreadMgr::ThreadExportToCommEngineAicpu(
    uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads)
{
    std::vector<std::shared_ptr<Thread>> hostThreads;
    std::vector<std::shared_ptr<Thread>> orderLaunchHostThreads;
    std::vector<u32> index;
    Thread* exportedThread;
    std::vector<u32> orderLaunchIndex;
    HCCL_INFO(
        "[ThreadMgr][%s] begin, threadNum[%u], dstCommEngine[%s], comm[%s]", __func__, threadNum,
        GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str(), commId_.c_str());
    for (u32 i = 0; i < threadNum; i++) {
        std::shared_ptr<Thread> handle;
        CHK_RET(GetExportedThread(threads[i], dstCommEngine, exportedThread, handle));
        if (exportedThread != nullptr) {
            exportedThreads[i] = reinterpret_cast<ThreadHandle>(exportedThread);
            continue;
        } else {
            bool inOrderLaunch = (orderLaunchThreads_.find(threads[i]) != orderLaunchThreads_.end());
            if (inOrderLaunch) {
                orderLaunchHostThreads.push_back(handle);
                orderLaunchIndex.push_back(i);
            } else {
                hostThreads.push_back(handle);
                index.push_back(i);
            }
        }
    }

    CHK_RET(ExportHostThreadsToAicpu(hostThreads, index, threads, dstCommEngine, exportedThreads));
    CHK_RET(ExportOrderLaunchThreadsToAicpu(
        orderLaunchHostThreads, orderLaunchIndex, threads, dstCommEngine, exportedThreads));
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadExportToCommEngine(
    uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads)
{
    switch (dstCommEngine) {
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CCU:
            CHK_RET(ThreadExportToCommEngineCpu(threadNum, threads, exportedThreads));
            break;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            CHK_RET(ThreadExportToCommEngineAicpu(threadNum, threads, dstCommEngine, exportedThreads));
            break;
        case COMM_ENGINE_AIV:
        default:
            HCCL_ERROR(
                "[ThreadMgr] Unknown comm engine type: %s",
                GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str());
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info)
{
    CHK_PRT_RET(
        resType != ThreadResType::THREAD_RES_TYPE_STREAM,
        HCCL_ERROR("[%s] failed. resType[%d] is not supported.", __func__, static_cast<int32_t>(resType)),
        HCCL_E_NOT_SUPPORT);

    std::lock_guard<std::mutex> threadhandleToThreadMtx(threadhandleToThreadMutex_);
    auto it = threadMap_.find(thread);
    CHK_PRT_RET(
        it == threadMap_.end(),
        HCCL_ERROR("[%s] failed to find handle mapping in threadMap_, thread[0x%llx].", __func__, thread),
        HCCL_E_NOT_FOUND);
    std::shared_ptr<Thread> threadPtr = it->second;
    CHK_PTR_NULL(threadPtr);
    if (resType == ThreadResType::THREAD_RES_TYPE_STREAM) {
        CHK_PRT_RET(
            infoLen != sizeof(ThreadResTypeStream),
            HCCL_ERROR(
                "[%s] failed. infoLen[%u] is mismatch sizeof(ThreadResTypeStream)[%zu]", __func__, infoLen,
                sizeof(ThreadResTypeStream)),
            HCCL_E_PARA);
        CHK_PTR_NULL(threadPtr->GetStream());
        ThreadResTypeStream stream = threadPtr->GetStream()->ptr();
        CHK_PTR_NULL(stream);
        *info = stream;
    } else {
        HCCL_ERROR("[%s] unsupported resType[%d]", __func__, static_cast<int32_t>(resType));
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO(
        "[%s] success. thread[0x%llx] resType[%d] info[%p]", __func__, thread, static_cast<int32_t>(resType), *info);
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::HcclUnfoldThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    auto it = dedicatedThreadMap_.find(useType);
    if (it != dedicatedThreadMap_.end()) {
        *thread = it->second;
        HCCL_INFO("[%s] reuse dedicated thread, dedThreadType[%u], thread[0x%llx]", __func__, useType, *thread);
        CHK_RET(SupplementThreadNotify(*thread, notifyNumPerThread));
    } else {
        if (useType == HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE) {
            *thread = 0;
            HCCL_WARNING(
                "[%s] dedicated thread not found, dedThreadType[%u], return threadHandle[0]", __func__, useType);
            return HCCL_SUCCESS;
        }
        CommEngine engine = CommEngine::COMM_ENGINE_CPU;
        uint32_t notifyNumPerThreadVec[1] = {notifyNumPerThread};
        HcclResult ret = static_cast<HcclResult>(HcommThreadAlloc(engine, 1, notifyNumPerThreadVec, thread));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] Failed to cache dedicated thread, dedThreadType[%u], ret[%d]", __func__, useType, ret);
            return ret;
        }
        dedicatedThreadMap_[useType] = *thread;
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclDeviceOrderThreadCreate(
    HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    auto it = dedicatedThreadMap_.find(useType);
    if (it != dedicatedThreadMap_.end() && it->second != 0) {
        *thread = it->second;
        HCCL_INFO("[%s] reuse device order thread[0x%llx], comm[%s]", __func__, *thread, commId_.c_str());
        return HCCL_SUCCESS;
    }

    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;
    ThreadType type = THREAD_TYPE_TS;
    ThreadConfig config;
    HcommResult initRet = ThreadConfigInit(&config, 1);
    CHK_PRT_RET(initRet != 0, HCCL_ERROR("[%s] ThreadConfigInit failed, ret[%d]", __func__, initRet), HCCL_E_INTERNAL);
    config.notifyNumPerThread = static_cast<uint16_t>(notifyNumPerThread);

    NotifyLoadType notifyLoadType;
    StreamType streamType;
    CHK_RET(GetNotifyLoadType(engine, type, notifyLoadType));
    CHK_RET(GetStreamType(engine, type, streamType));

    std::lock_guard<std::mutex> threadMtx(threadMutex_);
    std::lock_guard<std::mutex> mapMtx(threadMapMutex_);

    std::vector<std::shared_ptr<Thread>> newThreads;
    CHK_RET(CreateAndInitThreads(engine, streamType, notifyLoadType, 1, &config, newThreads));

    std::unique_ptr<ThreadHandle[]> hostHandle;
    CHK_RET(AssignThreadHandles(engine, newThreads, thread, hostHandle));

    threads_.reserve(threads_.size() + newThreads.size());
    CHK_RET(StoreThreadsAndBuildHandleMap(engine, newThreads, hostHandle));

    dedicatedThreadMap_[useType] = *thread;

    HCCL_INFO("[%s] created device order thread[0x%llx], comm[%s]", __func__, *thread, commId_.c_str());
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclDedicatedThreadAcquire(
    HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        useType == HCCL_DED_THREAD_TYPE_INVALID, HCCL_ERROR("[%s] dedThreadType is invalid", __func__), HCCL_E_PARA);
    HCCL_INFO("Entry-%s: dedThreadType[%u] notifyNumPerThread[%u]", __func__, useType, notifyNumPerThread);

    std::lock_guard<std::mutex> lock(dedicatedThreadMutex_);
    if (useType == HCCL_DED_THREAD_TYPE_AICPU_LAUNCH || useType == HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE) {
        CHK_RET(HcclUnfoldThreadAcquire(useType, notifyNumPerThread, thread));
    } else if (useType == HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
        CHK_RET(HcclDeviceOrderThreadCreate(useType, notifyNumPerThread, thread));
    } else {
        HCCL_ERROR("[%s] unsupported dedThreadType[%u]", __func__, useType);
        return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO(
        "[%s] success, useType[%u], thread[0x%llx], notifyNumPerThread[%u]", __func__, useType, *thread,
        notifyNumPerThread);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetLocalNotify(LocalNotify* notify, uint32_t notifyIdx, uint64_t threadHandle)
{
    if (notify == nullptr) {
        return HCCL_SUCCESS;
    }
    HcclRtNotify rtNotify = notify->ptr();
    if (rtNotify == nullptr) {
        return HCCL_SUCCESS;
    }
    HcclResult ret = hrtNotifyReset(rtNotify);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[ThreadMgr][ResetLocalNotify] hrtNotifyReset failed, Hcom[%s], thread[0x%llx], notifyIdx[%u], "
            "notifyId[%u], ret[0x%016llx]",
            commId_.c_str(), threadHandle, notifyIdx, notify->notifyId_, HCCL_ERROR_CODE(ret));
        return ret;
    }
    HCCL_INFO(
        "[ThreadMgr][ResetLocalNotify] reset notify success, Hcom[%s], thread[0x%llx], notifyIdx[%u], "
        "notifyId[%u], notifyPtr[%p]",
        commId_.c_str(), threadHandle, notifyIdx, notify->notifyId_, rtNotify);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetNotifiesInThread(Thread* thread)
{
    if (thread == nullptr) {
        return HCCL_SUCCESS;
    }
    const uint64_t threadHandle = reinterpret_cast<uint64_t>(thread);
    const uint32_t notifyNum = thread->GetNotifyNum();
    for (uint32_t i = 0; i < notifyNum; ++i) {
        CHK_RET(ResetLocalNotify(thread->GetNotify(i), i, threadHandle));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetThreadPoolLocalNotifies()
{
    std::lock_guard<std::mutex> lock(threadMutex_);
    for (auto& threadPtr : threads_) {
        HcclResult ret = ResetNotifiesInThread(threadPtr.get());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][ResetThreadPoolLocalNotifies] reset notifies failed, Hcom[%s], thread[0x%llx], "
                "ret[0x%016llx]",
                commId_.c_str(), reinterpret_cast<uint64_t>(threadPtr.get()), HCCL_ERROR_CODE(ret)),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetMainThreadLocalNotifies()
{
    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    for (auto& pair : mainThread_) {
        HcclResult ret = ResetNotifiesInThread(pair.second.get());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][ResetMainThreadLocalNotifies] reset notifies failed, Hcom[%s], thread[0x%llx], "
                "ret[0x%016llx]",
                commId_.c_str(), reinterpret_cast<uint64_t>(pair.second.get()), HCCL_ERROR_CODE(ret)),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetDedicatedThreadLocalNotifies()
{
    std::lock_guard<std::mutex> lock(dedicatedThreadMutex_);
    for (auto& pair : dedicatedThreadMap_) {
        // device侧保序流不会保存在g_ThreadMap，不在此处清理，在ResetThreadPoolLocalNotifiles清
        if (pair.first == HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
            continue;
        }
        if (pair.second == 0) {
            continue;
        }
        std::shared_ptr<Thread> threadPtr;
        // dedicated 由 HcommThreadAlloc 创建，对象在 hccl::g_ThreadMap；不进 ThreadMgr::threadMap_。
        // Lookup 未命中即 Thread/notify 已 Free，勿裸指针解引用。
        if (LookupThreadByHandle(pair.second, threadPtr) != HCCL_SUCCESS || threadPtr == nullptr) {
            HCCL_WARNING(
                "[ThreadMgr][ResetDedicatedThreadLocalNotifies] thread not found (likely freed), skip reset, "
                "Hcom[%s], useType[%u], thread[0x%llx]",
                commId_.c_str(), static_cast<uint32_t>(pair.first), static_cast<uint64_t>(pair.second));
            continue;
        }
        HcclResult ret = ResetNotifiesInThread(threadPtr.get());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][ResetDedicatedThreadLocalNotifies] reset notifies failed, Hcom[%s], useType[%u], "
                "thread[0x%llx], ret[0x%016llx]",
                commId_.c_str(), static_cast<uint32_t>(pair.first), reinterpret_cast<uint64_t>(threadPtr.get()),
                HCCL_ERROR_CODE(ret)),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetOrderLaunchThreadLocalNotifies()
{
    for (ThreadHandle handle : orderLaunchThreads_) {
        if (handle == 0) {
            continue;
        }
        std::shared_ptr<Thread> threadPtr;
        // orderLaunch 所有权在 OrderLaunchThreadMgr；Lookup 未命中即 Thread/notify 已 Free，勿裸指针解引用。
        if (LookupThreadByHandle(handle, threadPtr) != HCCL_SUCCESS || threadPtr == nullptr) {
            HCCL_WARNING(
                "[ThreadMgr][ResetOrderLaunchThreadLocalNotifies] thread not found (likely freed), skip reset, "
                "Hcom[%s], thread[0x%llx]",
                commId_.c_str(), static_cast<uint64_t>(handle));
            continue;
        }
        HcclResult ret = ResetNotifiesInThread(threadPtr.get());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][ResetOrderLaunchThreadLocalNotifies] reset notifies failed, Hcom[%s], thread[0x%llx], "
                "ret[0x%016llx]",
                commId_.c_str(), reinterpret_cast<uint64_t>(threadPtr.get()), HCCL_ERROR_CODE(ret)),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetThreadLocalNotifies()
{
    // pool / main / dedicated / orderLaunch 可能指向同一 Thread（orderLaunch 常是已 Acquire
    // 的 handle 再 Register）。同一 LocalNotify 可能对 RTS 多次 hrtNotifyReset；
    // hrtNotifyReset → aclrtNotifyBatchReset 清零触发态，重复调用视为幂等，可接受。
    HCCL_INFO("[ThreadMgr][ResetThreadLocalNotifies] start, Hcom[%s]", commId_.c_str());
    CHK_RET(ResetThreadPoolLocalNotifies());
    CHK_RET(ResetMainThreadLocalNotifies());
    CHK_RET(ResetDedicatedThreadLocalNotifies());
    CHK_RET(ResetOrderLaunchThreadLocalNotifies());
    HCCL_INFO("[ThreadMgr][ResetThreadLocalNotifies] finish, Hcom[%s]", commId_.c_str());
    return HCCL_SUCCESS;
}
} // namespace hccl
