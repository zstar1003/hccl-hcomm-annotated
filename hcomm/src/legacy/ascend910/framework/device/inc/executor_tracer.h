/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_TRACE_EXECUTOR_TRACER_H_
#define ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_TRACE_EXECUTOR_TRACER_H_
#include "common/aicpu_hccl_def.h"
#include "utils/aicpu_hdc_utils.h"
#include "framework/aicpu_communicator.h"
#include "cann_error_reporter.h"

namespace dfx_tracer {
class ExecutorTracer {
public:
    explicit ExecutorTracer();
    static void BackGroundDfx(void* info);
    static void StopBackGroundDfx(void* info);
    static void
    SetCqeQueryInput(const uint32_t devId, const HcclComStreamInfo& streamInfo, CqeQueryInput& cqeQueryInput);

private:
    static void HandleCqeStatus(AicpuComContext* const ctx);
    static void StopLaunchCommandHandle(AicpuComContext* const ctx);
    static void KfcCommandHandle(AicpuComContext* const ctx);
    static void HandleBackGround(AicpuComContext* const ctx);
    static void HandleCqeStatusInComm();
    static void HandleReportStatusInComm();
    static void HandleAICPUCommand(hccl::HcclCommAicpu* const commInfo);
    static void StopBackGround(AicpuComContext* const ctx, bool& isNotStop);
    static void HandleDestroyComm(AicpuComContext* const ctx);
    static void HandleSwitchNic(AicpuComContext* const ctx);
    static void HandleResumeChangeLink(AicpuComContext* const ctx);
    static void TaskMonitor(void);
};
class AICPUcommandHandles {
public:
    static void NsCommStop(hccl::HcclCommAicpu* const commInfo);
    static void NsCommClean(hccl::HcclCommAicpu* const commInfo);
};
} // namespace dfx_tracer
#endif // ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_TRACE_EXECUTOR_TRACER_H_
