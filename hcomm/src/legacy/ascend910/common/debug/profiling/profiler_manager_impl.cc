/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlog_pub.h>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "adapter_prof.h"
#include "profiling_manager.h"
#include "profiler_manager_impl.h"
#include "stream_utils.h"

namespace hccl {
ProfilerManagerImpl::ProfilerManagerImpl(s32 devicePhyId, s32 deviceLogicId, u32 realUserRank, u32 rankSize)
    : devicePhyId_(devicePhyId),
      deviceLogicId_(deviceLogicId),
      realUserRank_(realUserRank),
      rankSize_(rankSize),
      profiler_(nullptr),
      taskExceptionHandler_(nullptr),
      taskOverflowHandler_(nullptr)
{}
ProfilerManagerImpl::~ProfilerManagerImpl() {}

void ProfilerManagerImpl::RegisterCallBack(ProfilerType name, hccl::PluginRunner& callback)
{
    // 容器操作并非线程安全的, 加锁
    std::unique_lock<std::mutex> lock(mutex_);

    if (callbacks_.find(name) != callbacks_.end()) {
        HCCL_WARNING("callback[%d] already registered", name);
        return;
    }

    callbacks_.insert(std::make_pair<ProfilerType&, hccl::PluginRunner&>(name, callback));
}

HcclResult ProfilerManagerImpl::InitProfiler()
{
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        return HCCL_SUCCESS;
    }
    CHK_RET(DlProfFunction::GetInstance().DlProfFunctionInit());
    CHK_RET(DlRtFunction::GetInstance().DlRtFunctionInit());
    profiler_.reset(new (std::nothrow) TaskProfiling(deviceLogicId_, realUserRank_, rankSize_, true));
    CHK_SMART_PTR_NULL(profiler_);
    PluginRunner profrunner(profiler_.get());
    RegisterCallBack(ProfilerType::TASK_PROFILING, profrunner);

    taskExceptionHandler_.reset(new (std::nothrow) TaskExceptionHandler(static_cast<u32>(deviceLogicId_)));
    CHK_SMART_PTR_NULL(taskExceptionHandler_);
    PluginRunner runner(taskExceptionHandler_.get());
    RegisterCallBack(ProfilerType::TASK_EXCEPTION, runner);

    // 记录可能导致算子溢出的task信息
    taskOverflowHandler_.reset(new (std::nothrow) TaskOverflow(static_cast<u32>(deviceLogicId_)));
    CHK_SMART_PTR_NULL(taskOverflowHandler_);
    PluginRunner dumprunner(taskOverflowHandler_.get());
    RegisterCallBack(ProfilerType::TASK_OVERFLOW, dumprunner);

    rtProfCtrlHandle callback = CommandHandle;
    HcclResult ret = hrtProfRegisterCtrlCallback(HCCL, callback);
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[ProfilerManager][InitProfiler]Register CtrlCallBack failed."), HCCL_E_PARA);

    for (const auto& it : PROF_TASK_OP_NAME) {
        std::string nameInfo = it.second;
        uint64_t ret = hrtMsprofGetHashId(nameInfo.c_str(), nameInfo.length());
        HCCL_DEBUG("[PROF_TASK_OP_NAME] nameInfo[%s] ret[%llu]", nameInfo.c_str(), ret);
    }

    HCCL_INFO("[ProfilerManager][InitProfiler]Register CtrlCallBack success");

    return HCCL_SUCCESS;
}

HcclResult ProfilerManagerImpl::GetandClearOverFlowTasks(std::vector<HcclDumpInfo>& hcclDumpInfo)
{
    if (taskOverflowHandler_ != nullptr) {
        CHK_RET(taskOverflowHandler_->GetandClearOverFlowTasks(hcclDumpInfo));
    } else {
        HCCL_WARNING("[ProfilerManager][GetDumpTask] taskOverflowHandler_ not set");
    }
    return HCCL_SUCCESS;
}

void ProfilerManagerImpl::TaskSdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(stream, hccl::TaskType::TASK_SDMA, para);
        }
    }
}

void ProfilerManagerImpl::TaskRdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(stream, hccl::TaskType::TASK_RDMA, para);
        }
    }
}

void ProfilerManagerImpl::TaskReduceInlineProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(stream, hccl::TaskType::TASK_REDUCE_INLINE, para);
        }
    }
}

void ProfilerManagerImpl::TaskReduceTbeProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(stream, hccl::TaskType::TASK_REDUCE_TBE, para);
        }
    }
}

void ProfilerManagerImpl::TaskRecordProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(stream, hccl::TaskType::TASK_NOTIFY_RECORD, para);
        }
    }
}

void ProfilerManagerImpl::TaskWaitProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(stream, hccl::TaskType::TASK_NOTIFY_WAIT, para);
        }
    }
}

void ProfilerManagerImpl::TaskAivProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaAiv& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(stream, para);
        }
    }
}

void ProfilerManagerImpl::TaskProfiler(
    ProfilerType profilerType, HcclRtStream stream, const void* descBuf, size_t descBufLen)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            if (profilerType == ProfilerType::TASK_EXCEPTION) {
                callback.second(stream, descBuf, descBufLen);
            } else {
                callback.second(stream);
            }
        }
    }
}

void ProfilerManagerImpl::TaskProfiler(ProfilerType profilerType, TaskParaHost& para)
{
    if (!callbacks_.empty()) {
        for (auto& callback : callbacks_) {
            if (profilerType != ProfilerType::TASK_ALL && callback.first != profilerType) {
                continue;
            }
            callback.second(para);
        }
    }
}

void ProfilerManagerImpl::TaskProfilerHandle(void* param, u32 length)
{
    if (UNLIKELY(param == nullptr)) {
        HCCL_ERROR("[ProfilerManagerImpl][%s]param is nullptr.", __func__);
        return;
    }
    struct TaskPara* taskPara = (struct TaskPara*)param;

    if (sizeof(TaskPara) < length) {
        return;
    }
    HCCL_INFO(
        "[ProfilerManagerImpl][%s]Start handle task profiler, taskType[%d], profilerType[%d]", __func__, taskPara->type,
        taskPara->profilerType);

    u32 ctxId = 0;
    ProfTaskType profTaskType;
    auto& profilingManager = hccl::ProfilingManager::Instance();
    if (taskPara->isFftsDispatcher) {
        profilingManager.SetFftsDispatcherMode();
    }

    HandleTask(taskPara, ctxId, profTaskType);

    if (taskPara->isFftsDispatcher) {
        profilingManager.ReSetFftsDispatcherMode();
    }

    if (GetIfProfile() && ctxId == INVALID_UINT) {
        (void)profilingManager.CallMsprofReportTaskApi(taskPara->isMainStream, taskPara->beginTime, profTaskType);
    }
}

void ProfilerManagerImpl::TaskAivProfilerHandle(void* param, u32 length)
{
    if (UNLIKELY(param == nullptr)) {
        HCCL_ERROR("[ProfilerManagerImpl][%s]param is nullptr.", __func__);
        return;
    }

    struct TaskParaGeneral* taskParaGeneral = static_cast<struct TaskParaGeneral*>(param);

    if (sizeof(TaskParaGeneral) < length) {
        return;
    }

    TaskAivProfiler(ProfilerType::TASK_ALL, taskParaGeneral->stream, taskParaGeneral->aiv);

    if (GetIfProfile()) {
        auto& profilingManager = hccl::ProfilingManager::Instance();
        (void)profilingManager.CallMsprofReportTaskApi(
            taskParaGeneral->isMainStream, taskParaGeneral->beginTime, ProfTaskType::TASK_AIV);
    }
}

void ProfilerManagerImpl::HandleTask(struct TaskPara* taskPara, u32& ctxId, ProfTaskType& profTaskType)
{
    switch (taskPara->type) {
        case TaskType::TASK_NOTIFY_RECORD:
            TaskRecordProfiler(taskPara->profilerType, taskPara->stream, taskPara->notify);
            ctxId = taskPara->notify.ctxId;
            profTaskType = ProfTaskType::TASK_NOTIFY_RECORD;
            break;

        case TaskType::TASK_NOTIFY_WAIT:
            TaskWaitProfiler(taskPara->profilerType, taskPara->stream, taskPara->notify);
            ctxId = taskPara->notify.ctxId;
            profTaskType = ProfTaskType::TASK_NOTIFY_WAIT;
            break;

        case TaskType::TASK_SDMA:
            TaskSdmaProfiler(taskPara->profilerType, taskPara->stream, taskPara->dma);
            ctxId = taskPara->dma.ctxId;
            profTaskType = ProfTaskType::TASK_SDMA;
            break;

        case TaskType::TASK_RDMA:
            TaskRdmaProfiler(taskPara->profilerType, taskPara->stream, taskPara->dma);
            ctxId = taskPara->dma.ctxId;
            profTaskType = ProfTaskType::TASK_RDMA;
            break;

        case TaskType::TASK_REDUCE_TBE:
            TaskReduceTbeProfiler(taskPara->profilerType, taskPara->stream, taskPara->reduce);
            ctxId = taskPara->reduce.ctxId;
            profTaskType = ProfTaskType::TASK_REDUCE_TBE;
            break;

        case TaskType::TASK_REDUCE_INLINE:
            TaskReduceInlineProfiler(taskPara->profilerType, taskPara->stream, taskPara->reduce);
            ctxId = taskPara->reduce.ctxId;
            profTaskType = ProfTaskType::TASK_REDUCE_INLINE;
            break;

        case TaskType::TASK_HOST:
            (void)ProfilerBase::GetTagByStream(taskPara->host.streamID, taskPara->host.tag);
            TaskProfiler(ProfilerType::TASK_PROFILING, taskPara->host);
            break;

        case TaskType::TASK_GRAPH_LAUNCH:
            HandleGraphLaunchTask(taskPara);
            break;

        default:
            return;
    }
}

void ProfilerManagerImpl::HandleGraphLaunchTask(struct TaskPara* taskPara)
{
    if (GetIfProfile()) {
        auto& profilingManager = hccl::ProfilingManager::Instance();
        rtModel_t rtModel = nullptr;
        bool isCapture = false;
        HcclResult retCapture = GetStreamCaptureInfo(taskPara->stream, rtModel, isCapture);
        CHK_PRT_CONT(
            retCapture != HCCL_SUCCESS, HCCL_ERROR("Get capture status error. return[%d], capture mode", retCapture));
        if (profilingManager.GetFftsLaunchApiState() || isCapture) {
            // 上报批量下发的ContextId信息
            (void)profilingManager.CallMsprofReportContextIdInfo((taskPara->graphLaunch.ctxNum - 1));

            if (profilingManager.GetTaskApiState() || isCapture) {
                // 上报编排的task(memcpy\notify等) addition Info
                profilingManager.ReportStoragedFftsInfo();
            }

            (void)profilingManager.CallMsprofReportTaskApi(
                taskPara->isMainStream, taskPara->beginTime, ProfTaskType::TASK_LAUNCH_FFTS_TASK);
        }
        TaskProfiler(ProfilerType::TASK_PROFILING, taskPara->stream);
    }
    TaskProfiler(
        ProfilerType::TASK_EXCEPTION, taskPara->stream, taskPara->graphLaunch.descBuf,
        taskPara->graphLaunch.descBufLen);
    TaskProfiler(ProfilerType::TASK_OVERFLOW, taskPara->stream);
}
} // namespace hccl
