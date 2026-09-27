/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_types.h>

#include "log.h"
#include "hccl/base.h"
#include "profiler_manager_impl.h"
#include "profiler_manager.h"

namespace hccl {
ProfilerManager::ProfilerManager(s32 devicePhyId, s32 deviceLogicId, u32 realUserRank, u32 rankSize)
{
    pimpl_.reset(new (std::nothrow) ProfilerManagerImpl(devicePhyId, deviceLogicId, realUserRank, rankSize));
}

ProfilerManager::~ProfilerManager() { pimpl_ = nullptr; }

HcclResult ProfilerManager::InitProfiler()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->InitProfiler();
}

HcclResult ProfilerManager::GetandClearOverFlowTasks(std::vector<HcclDumpInfo>& hcclDumpInfo)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetandClearOverFlowTasks(hcclDumpInfo);
}

void ProfilerManager::TaskSdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskSdmaProfiler(profilerType, stream, para);
}

void ProfilerManager::TaskRdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskRdmaProfiler(profilerType, stream, para);
}

void ProfilerManager::TaskReduceInlineProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskReduceInlineProfiler(profilerType, stream, para);
}

void ProfilerManager::TaskReduceTbeProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskReduceTbeProfiler(profilerType, stream, para);
}

void ProfilerManager::TaskRecordProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskRecordProfiler(profilerType, stream, para);
}

void ProfilerManager::TaskWaitProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskWaitProfiler(profilerType, stream, para);
}

void ProfilerManager::TaskAivProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaAiv& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskAivProfiler(profilerType, stream, para);
}

void ProfilerManager::TaskProfiler(ProfilerType profilerType, HcclRtStream stream)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskProfiler(profilerType, stream);
}

void ProfilerManager::TaskProfiler(ProfilerType profilerType, TaskParaHost& para)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskProfiler(profilerType, para);
}

void ProfilerManager::TaskProfilerHandle(void* param, u32 length)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskProfilerHandle(param, length);
}

void ProfilerManager::TaskAivProfilerHandle(void* param, u32 length)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->TaskAivProfilerHandle(param, length);
}

} // namespace hccl
