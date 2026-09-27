/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread.h"
#include "cpu_ts_thread.h"
#include "aicpu_ts_thread.h"
#include "sal_pub.h"
#include "stream_lite.h"
#include "task_info.h"
#include "comm_engine_utils.h"
#include "aicpu_launch_manager.h"
#include "dfx_profiling_handler_lite.h"
#include "dfx_dlprof_function.h"
#include "aicpu_indop_env.h"
#include "adapter_rts_common.h"

using namespace std;

namespace hccl {

struct DeviceThreadKey {
    int32_t deviceId;
    ThreadHandle handle;

    bool operator==(const DeviceThreadKey& other) const { return deviceId == other.deviceId && handle == other.handle; }
};

struct DeviceThreadKeyHash {
    std::size_t operator()(const DeviceThreadKey& key) const
    {
        return std::hash<int32_t>()(key.deviceId) ^ (std::hash<ThreadHandle>()(key.handle) << 1);
    }
};

static unordered_map<ThreadHandle, shared_ptr<Thread>> g_ThreadMap;
static unordered_map<DeviceThreadKey, ThreadHandle, DeviceThreadKeyHash> g_ThreadD2HMap;
static mutex g_ThreadMapMtx;

HcclResult CreateThread(
    CommEngine engine, StreamType streamType, uint32_t notifyNum, NotifyLoadType loadType,
    shared_ptr<Thread>& out_thread)
{
    out_thread = nullptr; // 初始化出参

    if (engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CCU
        || engine == COMM_ENGINE_AIV) {
        EXCEPTION_CATCH(out_thread = make_shared<CpuTsThread>(streamType, notifyNum, loadType), return HCCL_E_PTR);
    } else if (engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AICPU) {
        EXCEPTION_CATCH(out_thread = make_shared<AicpuTsThread>(streamType, notifyNum, loadType), return HCCL_E_PTR);
    } else {
        return HCCL_E_NOT_SUPPORT;
    }

    return HCCL_SUCCESS;
}

HcclResult CommHostEngineToNotifyLoadType(CommEngine engine, NotifyLoadType& type)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU:
            type = NotifyLoadType::HOST_NOTIFY;
            break;
        default:
            HCCL_ERROR(
                "[ThreadMgr] Unsupported comm engine type: %s",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CommEngineToNotifyLoadType(CommEngine engine, NotifyLoadType& type)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU:
            type = NotifyLoadType::HOST_NOTIFY;
            break;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            type = NotifyLoadType::HOST_NOTIFY;
            break;
        default:
            HCCL_ERROR(
                "[ThreadMgr] Unknown comm engine type: %s",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CommEngineToStreamType(CommEngine engine, StreamType& type)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU:
            type = StreamType::STREAM_TYPE_ONLINE; // 单算子使用online，图模式使用offline
            break;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            type = StreamType::STREAM_TYPE_DEVICE;
            break;
        // 暂不支持AIV
        case COMM_ENGINE_AIV:
        default:
            HCCL_ERROR(
                "[ThreadMgr] Unknown comm engine type: %s",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

struct EnumPairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        const std::size_t h1 = std::hash<T1>{}(p.first);
        const std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

const std::unordered_map<std::pair<CommEngine, ThreadType>, NotifyLoadType, EnumPairHash> NOTIFY_TYPE_CONVERT = {
    {{COMM_ENGINE_CPU, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
    {{COMM_ENGINE_CCU, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
    {{COMM_ENGINE_AIV, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
    {{COMM_ENGINE_AICPU, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
};

const std::unordered_map<std::pair<CommEngine, ThreadType>, StreamType, EnumPairHash> STREAM_TYPE_CONVERT = {
    {{COMM_ENGINE_CPU, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_ONLINE},
    {{COMM_ENGINE_CCU, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_ONLINE},
    {{COMM_ENGINE_AIV, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_ONLINE},
    {{COMM_ENGINE_AICPU, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_DEVICE},
};

HcclResult GetNotifyLoadType(CommEngine engine, ThreadType threadType, NotifyLoadType& type)
{
    auto iter = NOTIFY_TYPE_CONVERT.find(std::make_pair(engine, threadType));
    if (iter == NOTIFY_TYPE_CONVERT.end()) {
        HCCL_ERROR("[GetNotifyLoadType] not support comm engine type: %d, thread type: %d", engine, threadType);
        return HCCL_E_PARA;
    }
    type = iter->second;
    return HCCL_SUCCESS;
}

HcclResult GetStreamType(CommEngine engine, ThreadType threadType, StreamType& type)
{
    auto iter = STREAM_TYPE_CONVERT.find(std::make_pair(engine, threadType));
    if (iter == STREAM_TYPE_CONVERT.end()) {
        HCCL_ERROR("[GetStreamType] not support comm engine type: %d, thread type: %d", engine, threadType);
        return HCCL_E_PARA;
    }
    type = iter->second;
    return HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
HcclResult ValidateThreadParams(uint32_t threadNum, uint32_t notifyNumPerThread)
{
    if (threadNum == 0 || threadNum > HCOMM_THREADNUM_MAX_NUM) {
        HCCL_ERROR(
            "[%s] Validate thread params failed. ThreadNum %u, range (0, %u]", __func__, threadNum,
            HCOMM_THREADNUM_MAX_NUM);
        return HCCL_E_PARA;
    }
    if (notifyNumPerThread > HCOMM_NOTIFY_MAX_NUM) {
        HCCL_ERROR(
            "[%s] Validate thread params failed. notifyNumPerThread %u, range [0, %u]", __func__, notifyNumPerThread,
            HCOMM_NOTIFY_MAX_NUM);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult SaveThreads(const vector<shared_ptr<Thread>>& newThreads)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    lock_guard<mutex> lock(g_ThreadMapMtx);
    for (const auto& threadPtr : newThreads) {
        ThreadHandle handle = reinterpret_cast<ThreadHandle>(threadPtr.get());

        if (g_ThreadMap.find(handle) != g_ThreadMap.end()) {
            HCCL_ERROR("[%s] thread handle already exists [0x%llx] in ThreadMap", __func__, handle);
            return HCCL_E_INTERNAL;
        }
        DeviceThreadKey key{deviceId, handle};
        if (g_ThreadD2HMap.find(key) != g_ThreadD2HMap.end()) {
            HCCL_ERROR(
                "[%s] thread handle already exists [0x%llx] in g_ThreadD2HMap, deviceId[%d]", __func__, handle,
                deviceId);
            return HCCL_E_INTERNAL;
        }

        g_ThreadMap.emplace(handle, threadPtr);
        g_ThreadD2HMap.emplace(key, handle);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateAndInitThreads(const ThreadCreateParams& params, vector<shared_ptr<Thread>>& outThreads)
{
    HCCL_INFO(
        "[%s] Creating threads with params: engine[%s], threadNum[%u], "
        "notifyNumPerThread[%u], notifyLoadType[%u], streamType[%u]",
        __func__, GetEnumToString(GetCommEngineStatusStrMap(), params.engine).c_str(), params.threadNum,
        params.notifyNumPerThread, static_cast<int32_t>(params.notifyLoadType),
        static_cast<int32_t>(params.streamType));
    outThreads.reserve(params.threadNum);

    for (uint32_t i = 0; i < params.threadNum; ++i) {
        shared_ptr<Thread> threadPtr;
        // 创建线程
        HcclResult ret = CreateThread(
            params.engine, params.streamType, params.notifyNumPerThread, params.notifyLoadType, threadPtr);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Failed to create thread at index %u, error: %d", __func__, i, ret),
            ret);

        // 初始化线程
        ret = threadPtr->Init();
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] Failed to initialize thread at index %u, error: %d", __func__, i, ret), ret);

        // 添加到输出列表
        outThreads.emplace_back(move(threadPtr));
    }
    HCCL_INFO("[%s] Successfully created and initialized %u threads", __func__, params.threadNum);
    return HCCL_SUCCESS;
}

HcclResult
FillThreadD2HMap(const ThreadHandle* deviceThreadHandles, const ThreadHandle* hostThreadHandles, uint32_t listNum)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    lock_guard<mutex> lock(g_ThreadMapMtx);
    for (uint32_t idx = 0; idx < listNum; idx++) {
        auto deviceThreadHandle = deviceThreadHandles[idx];
        auto hostThreadHandle = hostThreadHandles[idx];
        HCCL_INFO(
            "%s deviceId[%d], deviceThreadHandle[0x%llx], hostThreadHandle[0x%llx]", __func__, deviceId,
            deviceThreadHandle, hostThreadHandle);
        DeviceThreadKey key{deviceId, deviceThreadHandle};
        g_ThreadD2HMap.emplace(key, hostThreadHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult StoreThreadHandles(
    vector<shared_ptr<Thread>>& newThreads, ThreadHandle* threads, CommEngine engine, aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(threads);
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        // AICPU引擎处理逻辑
        unique_ptr<ThreadHandle[]> aicpuHandle;
        EXCEPTION_CATCH(aicpuHandle = make_unique<ThreadHandle[]>(newThreads.size()), return HCCL_E_PTR);
        CHK_PTR_NULL(binHandle);
        HcclResult ret = AicpuLaunchMgr::ThreadKernelLaunchForBase(newThreads, aicpuHandle, binHandle);

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[StoreThreadHandles] AiCpuKernelLaunch failed, engine[%s], return[%d].",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret),
            ret);

        // 保存并映射AICPU线程句柄
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = aicpuHandle[i];
            ThreadHandle hostHandle = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            CHK_RET(FillThreadD2HMap(&aicpuHandle[i], &hostHandle, 1));
            HCCL_INFO(
                "[StoreThreadHandles] AICPU engine[%s] threadArray[%zu] = [%lu]",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), i, threads[i]);
        }
    } else {
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            HCCL_INFO(
                "[StoreThreadHandles] Host engine[%s] threadArray[%zu] = [%lu]",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), i, threads[i]);
        }
    }
    return HCCL_SUCCESS;
}

static HcclResult FreeThreadHandlesLocked(
    const ThreadHandle* threads, uint32_t threadNum, vector<ThreadHandle>& deviceHandles,
    vector<shared_ptr<Thread>>& hostThreadsToRelease)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    lock_guard<mutex> lock(g_ThreadMapMtx);
    for (uint32_t i = 0; i < threadNum; ++i) {
        const ThreadHandle inHandle = threads[i];

        DeviceThreadKey key{deviceId, inHandle};
        auto itH = g_ThreadD2HMap.find(key);
        if (itH == g_ThreadD2HMap.end()) {
            HCCL_ERROR(
                "[%s] failed to find handle mapping in g_ThreadD2HMap, deviceId[%d], inHandle[0x%llx].", __func__,
                deviceId, inHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        const ThreadHandle mappedHandle = itH->second;

        auto itC = g_ThreadMap.find(mappedHandle);
        if (itC == g_ThreadMap.end()) {
            HCCL_ERROR(
                "[%s] failed to find thread in g_ThreadMap, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].",
                __func__, deviceId, inHandle, mappedHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        if (inHandle != mappedHandle) {
            deviceHandles.push_back(inHandle);
        }

        // 暂缓析构 host 侧线程，待 device 侧线程销毁（停止守护线程轮询）后再释放其 SQ/CQ 资源
        hostThreadsToRelease.emplace_back(itC->second);

        HCCL_INFO(
            "[%s] erase thread: deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx], ptr[%p]", __func__, deviceId,
            inHandle, mappedHandle, itC->second.get());
        g_ThreadMap.erase(itC);

        for (auto it = g_ThreadD2HMap.begin(); it != g_ThreadD2HMap.end();) {
            if (it->second == mappedHandle && it->first.deviceId == deviceId) {
                it = g_ThreadD2HMap.erase(it);
            } else {
                ++it;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult FreeThreads(const ThreadHandle* threads, uint32_t threadNum, aclrtBinHandle binHandle)
{
    CHK_PRT_RET(threads == nullptr, HCCL_ERROR("[HcommThreadfree] threads is null."), HCCL_E_PARA);
    if (threadNum == 0 || threadNum > HCOMM_THREADNUM_MAX_NUM) {
        HCCL_ERROR(
            "[%s] Validate thread params failed. ThreadNum %u, range (0, %u]", __func__, threadNum,
            HCOMM_THREADNUM_MAX_NUM);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[%s] begin to free %u threads", __func__, threadNum);

    vector<ThreadHandle> deviceHandles; // 存放device侧的handle
    vector<shared_ptr<Thread>> hostThreadsToRelease; // 暂缓析构的 host 侧线程，避免 CQ 先于 device 侧线程销毁被释放

    CHK_RET(FreeThreadHandlesLocked(threads, threadNum, deviceHandles, hostThreadsToRelease));

    // 先销毁 device 侧线程（将其从守护线程轮询列表 threads_ 移除），再释放 host 侧 SQ/CQ 资源，
    // 避免守护线程在 CQ 已释放后仍对其发起查询导致「参数无效」报错
    if (!deviceHandles.empty()) {
        CHK_RET(AicpuLaunchMgr::ThreadKernelLaunchDestroy(deviceHandles.data(), deviceHandles.size(), binHandle));
    }
    HCCL_INFO("[%s] %u threads freed successfully.", __func__, threadNum);
    return HCCL_SUCCESS;
}

HcclResult SupplementThreadNotify(ThreadHandle handle, uint32_t notifyNum)
{
    lock_guard<mutex> lock(g_ThreadMapMtx);
    auto it = g_ThreadMap.find(handle);
    CHK_PRT_RET(
        it == g_ThreadMap.end(), HCCL_ERROR("[%s] thread handle[0x%llx] not found in g_ThreadMap.", __func__, handle),
        HCCL_E_NOT_FOUND);
    if (it->second->GetNotifyNum() >= notifyNum) {
        return HCCL_SUCCESS;
    }
    u32 supplementNum = notifyNum - it->second->GetNotifyNum();
    HCCL_INFO(
        "[%s] supplement notify num:[%u], current notify num:[%u], target notify num:[%u]", __func__, supplementNum,
        it->second->GetNotifyNum(), notifyNum);
    return it->second->SupplementNotify(supplementNum);
}

HcclResult LookupThreadByHandle(ThreadHandle handle, std::shared_ptr<Thread>& outThread)
{
    lock_guard<mutex> lock(g_ThreadMapMtx);
    auto it = g_ThreadMap.find(handle);
    if (it == g_ThreadMap.end()) {
        // try find device handle
        int32_t deviceId = 0;
        CHK_RET(hrtGetDevice(&deviceId));
        DeviceThreadKey key{deviceId, handle};
        auto device_it = g_ThreadD2HMap.find(key);
        CHK_PRT_RET(
            device_it == g_ThreadD2HMap.end(),
            HCCL_ERROR(
                "[%s] device handle[0x%llx] not found in g_ThreadD2HMap, deviceId[%d].", __func__, handle, deviceId),
            HCCL_E_NOT_FOUND);
        it = g_ThreadMap.find(device_it->second);
        CHK_PRT_RET(
            it == g_ThreadMap.end(),
            HCCL_ERROR("[%s] thread handle[0x%llx] not found in g_ThreadMap.", __func__, handle), HCCL_E_NOT_FOUND);
    }

    outThread = it->second;
    return HCCL_SUCCESS;
}

HcclResult LookupD2HHandle(ThreadHandle deviceHandle, ThreadHandle& outHostHandle)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));
    lock_guard<mutex> lock(g_ThreadMapMtx);
    DeviceThreadKey key{deviceId, deviceHandle};
    auto it = g_ThreadD2HMap.find(key);
    CHK_PRT_RET(
        it == g_ThreadD2HMap.end(),
        HCCL_ERROR(
            "[%s] device handle[0x%llx] not found in g_ThreadD2HMap, deviceId[%d].", __func__, deviceHandle, deviceId),
        HCCL_E_NOT_FOUND);
    outHostHandle = it->second;
    return HCCL_SUCCESS;
}
#endif

HcclResult Thread::AddThreadHandleToMap(CommEngine commEngine, ThreadHandle threadHandle)
{
    if (threadHandleMap_.find(commEngine) != threadHandleMap_.end() && threadHandleMap_[commEngine] != threadHandle) {
        HCCL_ERROR(
            "[Thread][%s]Mapping already exists:commEngine[%s], threadHandle[%lu], new threadHandle[%lu]", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), threadHandleMap_[commEngine],
            threadHandle);
        return HCCL_E_PARA;
    }

    threadHandleMap_[commEngine] = threadHandle;
    return HCCL_SUCCESS;
}

Thread* Thread::FindThreadByCommEngine(CommEngine commEngine)
{
    if (threadHandleMap_.find(commEngine) != threadHandleMap_.end()) {
        return reinterpret_cast<Thread*>(threadHandleMap_[commEngine]);
    }

    return nullptr;
}

HcclResult Thread::ReportAicpuNotifyWaitTask(u64 notifyId, u64 beginTime, u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.endTime = ProfGetCurCpuTimestamp();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], notifyId[%llu], %s", __func__, sqId, taskId, notifyId,
        taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostNotifyWaitTask(
    [[maybe_unused]] u64 notifyId, [[maybe_unused]] u64 beginTime, [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.isMaster = isMaster;
    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO("[ReportHostNotifyWaitTask] time is %llu", taskParam.endTime);
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], notifyId[%llu], %s", __func__, streamId, taskId, notifyId,
        taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportAicpuNotifyRecordTask(u64 notifyId, u64 beginTime, u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.endTime = ProfGetCurCpuTimestamp();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], notifyId[%llu], %s", __func__, sqId, taskId, notifyId,
        taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostNotifyRecordTask(
    [[maybe_unused]] u64 notifyId, [[maybe_unused]] u64 beginTime, [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.isMaster = isMaster;
    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO("[ReportHostNotifyRecordTask] time is %llu", taskParam.endTime);
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], notifyId[%llu], %s", __func__, streamId, taskId, notifyId,
        taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostLocalCopyTask(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] u64 beginTime, [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_SDMA;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.DMA.src = src;
    taskParam.taskPara.DMA.dst = dst;
    taskParam.taskPara.DMA.size = sizeByte;
    taskParam.taskPara.DMA.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.DMA.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.DMA.dmaOp = Hccl::DmaOp::HCCL_DMA_READ;
    taskParam.isMaster = isMaster;

    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], src[%p], dst[%p], len[%llu] %s", __func__, streamId, taskId, src, dst,
        sizeByte, taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportAicpuLocalCopyTask(
    void* dst, const void* src, uint64_t sizeByte, u64 beginTime, u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_SDMA;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.DMA.src = src;
    taskParam.taskPara.DMA.dst = dst;
    taskParam.taskPara.DMA.size = sizeByte;
    taskParam.taskPara.DMA.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.DMA.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.DMA.dmaOp = Hccl::DmaOp::HCCL_DMA_READ;
    taskParam.endTime = ProfGetCurCpuTimestamp();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], src[%p], dst[%p], len[%llu] %s", __func__, sqId, taskId, src, dst, sizeByte,
        taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportAicpuLocalReduceTask(
    void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp, u64 beginTime,
    u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_REDUCE_INLINE;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Reduce.src = src;
    taskParam.taskPara.Reduce.dst = dst;
    taskParam.taskPara.Reduce.size = sizeByte;
    taskParam.taskPara.Reduce.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.Reduce.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.dataType = static_cast<HcclDataType>(dataType);
    taskParam.taskPara.Reduce.reduceOp = static_cast<HcclReduceOp>(reduceOp);
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], src[%p], dst[%p], len[%llu], dataType[%d], reduceOp[%d], %s", __func__,
        sqId, taskId, src, dst, sizeByte, dataType, reduceOp, taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostLocalReduceTask(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] HcommDataType dataType, [[maybe_unused]] HcommReduceOp reduceOp, [[maybe_unused]] u64 beginTime,
    [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_REDUCE_INLINE;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Reduce.src = src;
    taskParam.taskPara.Reduce.dst = dst;
    taskParam.taskPara.Reduce.size = sizeByte;
    taskParam.taskPara.Reduce.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.Reduce.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.dataType = static_cast<HcclDataType>(dataType);
    taskParam.taskPara.Reduce.reduceOp = static_cast<HcclReduceOp>(reduceOp);
    taskParam.isMaster = isMaster;
    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();

    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], src[%p], dst[%p], len[%llu], dataType[%d], reduceOp[%d] %s", __func__,
        streamId, taskId, src, dst, sizeByte, dataType, reduceOp, taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

bool Thread::IsReportTask() const
{
#ifdef CCL_KERNEL_AICPU
    return hcomm::GetTaskExceptionEnable() || Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State();
#endif
    return true;
}

} // namespace hccl
