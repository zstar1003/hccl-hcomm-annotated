/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_thread_process.h"
#include <iomanip>
#include "exception_handle.h"
#include "stream_lite.h"

using namespace hccl;

std::shared_mutex AicpuThreadProcess::mutex_;
std::vector<std::shared_ptr<hccl::Thread>> AicpuThreadProcess::threads_;
std::mutex AicpuThreadProcess::bgThreadMutex_;
bool AicpuThreadProcess::daemonFuncRegistered_ = false;
Hccl::CommandToBackGroud AicpuThreadProcess::commandToBackGroud_ = Hccl::CommandToBackGroud::Default;
std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> AicpuThreadProcess::defaultDfxCallback_;

HcclResult AicpuThreadProcess::InitThreads(ThreadMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    std::vector<std::shared_ptr<Thread>> outThreads;
    outThreads.reserve(threadNum);
    std::string hcomId(param->hcomId);
    CHK_RET(AicpuThreadProcess::ResumeThread(param, outThreads, false));

    // 由调用方 AicpuThreadInit 持有 mutex_ 写锁保护，此处 check-then-set 无并发风险
    if (!defaultDfxCallback_) { // HcommThreadAlloc接口暂未适配profiling和上报task的能力
        defaultDfxCallback_ = [](u32 streamId, u32 taskId, [[maybe_unused]] const Hccl::TaskParam& taskParam,
                                 [[maybe_unused]] u64 handle) {
            HCCL_DEBUG("[AicpuThreadProcess] order launch dfx callback, streamId[%u], taskId[%u]", streamId, taskId);
            return HCCL_SUCCESS;
        };
    }

    ThreadHandle* threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    // 空指针校验
    CHK_PTR_NULL(threadArray);
    for (size_t i = 0; i < threadNum; ++i) {
        threadArray[i] = reinterpret_cast<ThreadHandle>(outThreads[i].get()); // 拷贝裸指针
        HCCL_INFO("[AicpuThreadProcess][%s] threadArray[%zu] = [%lu]", __func__, i, threadArray[i]);
        int32_t ret = HcommThreadRegisterDfx(threadArray[i], defaultDfxCallback_);
        if (ret != 0) {
            HCCL_WARNING(
                "[AicpuThreadProcess][%s] HcommThreadRegisterDfx failed, ret[%d], threadArray[%zu]", __func__, ret, i);
        }
    }
    threads_.insert(
        threads_.end(), std::make_move_iterator(outThreads.begin()), std::make_move_iterator(outThreads.end()));
    HCCL_INFO(
        "[AicpuThreadProcess][%s] comm identifier[%s], init threads num[%u] success", __func__, hcomId.c_str(),
        threadNum);
    return HCCL_SUCCESS;
}

const std::vector<std::shared_ptr<hccl::Thread>>& AicpuThreadProcess::GetThreads() { return threads_; }

std::shared_mutex& AicpuThreadProcess::GetMutex() { return mutex_; }

HcclResult AicpuThreadProcess::AicpuThreadInit(ThreadMgrAicpuParam* param)
{
    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(param->deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(param->deviceType)));
    {
        std::unique_lock<std::shared_mutex> rwlock(mutex_);
        HcclResult ret = InitThreads(param);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AicpuThreadProcess][AicpuThreadInit]errNo[0x%016llx] Failed to init threads", HCCL_ERROR_CODE(ret)),
            ret);
    }

    if (static_cast<DevType>(param->deviceType) == DevType::DEV_TYPE_950
        || static_cast<DevType>(param->deviceType) == DevType::DEV_TYPE_960) {
        InitBackGroundThread();
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuThreadProcess::AicpuThreadDestroy(ThreadMgrAicpuParam* param)
{
    HCCL_INFO("[AicpuThreadProcess][%s] threadNum[%u]", __func__, param->threadNum);

    bool needStopBgThread = false;
    {
        std::unique_lock<std::shared_mutex> rwlock(mutex_);
        ThreadHandle* threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
        if (threadArray == nullptr) {
            HCCL_ERROR("[AicpuThreadProcess][%s] threadArray is nullptr", __func__);
            return HCCL_E_PTR;
        }

        for (u32 i = 0; i < param->threadNum; ++i) {
            ThreadHandle handle = threadArray[i];
            auto it = std::find_if(threads_.begin(), threads_.end(), [handle](const std::shared_ptr<Thread>& ptr) {
                return reinterpret_cast<ThreadHandle>(ptr.get()) == handle;
            });
            if (it == threads_.end()) {
                HCCL_WARNING("[AicpuThreadProcess][%s] thread handle[0x%llx] not found in threads_", __func__, handle);
                continue;
            }
            Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>((*it)->GetStreamLitePtr());
            if (streamLite != nullptr) {
                hcomm::ExceptionHandle::GetInstance().ClearStreamState(streamLite->GetSqId());
            }
            threads_.erase(it);
            HCCL_DEBUG("[AicpuThreadProcess][%s] destroyed thread handle[0x%llx]", __func__, handle);
        }

        if (threads_.empty()) {
            needStopBgThread = true;
        }
    }

    if (needStopBgThread) {
        StopBackGroundThread();
    }

    HCCL_INFO("[AicpuThreadProcess][%s] success", __func__);
    return HCCL_SUCCESS;
}

void AicpuThreadProcess::InitBackGroundThread()
{
    std::lock_guard<std::mutex> lock(bgThreadMutex_);
    if (daemonFuncRegistered_) {
        HCCL_INFO("[AicpuThreadProcess][%s] background thread already started, skip.", __func__);
        return;
    }
    // 注册守护进程函数
    Hccl::AicpuDaemonService::GetInstance().Register(&hcomm::ExceptionHandle::GetInstance());
    daemonFuncRegistered_ = true;

    static auto daemonServiceRun = [](void* info) {
        Hccl::AicpuDaemonService::GetInstance().ServiceRun(info);
    };
    static auto daemonServiceStop = [](void* info) {
        Hccl::AicpuDaemonService::GetInstance().ServiceStop(info);
    };

    commandToBackGroud_ = Hccl::CommandToBackGroud::Default;

    // 启动背景线程，背景线程在runtime实现有保护，背景线程已经启动后会直接返回。
    if (Hccl::StartMC2MaintenanceThread != nullptr) {
        Hccl::StartMC2MaintenanceThread(
            daemonServiceRun, &commandToBackGroud_, daemonServiceStop, &commandToBackGroud_);
        HCCL_RUN_INFO("[%s]start BackGround thread success.", __func__);
    } else {
        HCCL_WARNING("[%s]StartMC2MaintenanceThread func is nullptr", __func__);
    }
}

void AicpuThreadProcess::StopBackGroundThread()
{
    std::lock_guard<std::mutex> lock(bgThreadMutex_);
    // 背景线程是同集合通信共用，这里不停止背景线程，只是将守护函数注销
    Hccl::AicpuDaemonService::GetInstance().Unregister(&hcomm::ExceptionHandle::GetInstance());
    daemonFuncRegistered_ = false;
    HCCL_INFO("[AicpuThreadProcess][%s] success", __func__);
}

HcclResult AicpuThreadProcess::ResumeThread(
    ThreadMgrAicpuParam* param, std::vector<std::shared_ptr<Thread>>& outThreads, bool isSupplementNotify)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    std::string hcomId(param->hcomId);
    ThreadHandle* threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    for (u32 i = 0; i < threadNum; ++i) {
        std::string thdUniqueId(param->threadParam[i], THREAD_UNIQUE_ID_MAX_SIZE);
        if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            std::ostringstream oss;
            oss << "threadParam[" << i << "] raw bytes: ";
            constexpr u32 HEX_WIDTH = 2;
            for (u32 j = 0; j < THREAD_UNIQUE_ID_MAX_SIZE; ++j) {
                oss << std::hex << std::setw(HEX_WIDTH) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(param->threadParam[i][j])) << " ";
            }
            HCCL_INFO("[AicpuThreadProcess][%s] %s", __func__, oss.str().c_str());
        }
        std::shared_ptr<AicpuTsThread> thread;
        EXCEPTION_CATCH((thread = std::make_shared<AicpuTsThread>(thdUniqueId)), return HCCL_E_PTR);
        thread->SetCommEngine(param->engine);
        u32 notifyNum = 0;
        std::string notifyDesc;
        CHK_RET(thread->GetNotifyByUniqueId(notifyNum, notifyDesc));
        if (isSupplementNotify) {
            AicpuTsThread* threadPtr = reinterpret_cast<AicpuTsThread*>(threadArray[i]);
            CHK_PTR_NULL(threadPtr);
            HCCL_INFO(
                "[%s]threadIdx[%u], threadHandle[%llu], notifyNum[%u], newNotifyNum[%u]", __func__, i, threadArray[i],
                threadPtr->GetNotifyNum(), notifyNum);
            CHK_RET(threadPtr->SupplementNotify(notifyNum, notifyDesc));
        } else {
            HcclResult ret = thread->Init();
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[AicpuThreadProcess][%s] comm identifier[%s], init threads num[%u] failed at index %u", __func__,
                    hcomId.c_str(), param->threadNum, i);
                return ret;
            }
            outThreads.emplace_back(thread);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuThreadProcess::AicpuThreadSupplementNotify(ThreadMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    std::string hcomId(param->hcomId);
    std::vector<std::shared_ptr<Thread>> outThreads;
    CHK_RET(AicpuThreadProcess::ResumeThread(param, outThreads, true));

    HCCL_INFO(
        "[AicpuThreadProcess][%s] comm identifier[%s], init threads num[%u] success", __func__, hcomId.c_str(),
        threadNum);
    return HCCL_SUCCESS;
}
