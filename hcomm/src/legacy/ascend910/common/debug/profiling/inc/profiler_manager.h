/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROFILER_MANAGER_H
#define PROFILER_MANAGER_H
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "profiler_base_pub.h"

namespace hccl {
class ProfilerManagerImpl;
class ProfilerManager {
public:
    ProfilerManager(s32 devicePhyId, s32 deviceLogicId, u32 realUserRank, u32 rankSize);
    ~ProfilerManager();
    HcclResult InitProfiler();
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo>& hcclDumpInfo);
    void TaskSdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA& para);
    void TaskRdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA& para);
    void TaskReduceInlineProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce& para);
    void TaskReduceTbeProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce& para);
    void TaskRecordProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify& para);
    void TaskWaitProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify& para);
    void TaskAivProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaAiv& para);
    void TaskProfiler(ProfilerType profilerType, HcclRtStream stream);
    void TaskProfiler(ProfilerType profilerType, TaskParaHost& para);
    void TaskProfilerHandle(void* param, u32 length);
    void TaskAivProfilerHandle(void* param, u32 length);

private:
    std::unique_ptr<ProfilerManagerImpl> pimpl_;
};
} // namespace hccl
#endif
