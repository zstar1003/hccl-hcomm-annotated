/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <array>
#include "adapter_rts.h"
#include "workflow_pub.h"

static std::mutex g_workFlowModeMutex;

// workflowMode 调整成线程变量，避免全局变量被并行修改
static thread_local HcclWorkflowMode g_workflowMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;

static bool g_isLaunchKernel = false; // 是否为对接GE接口场景
static bool g_isTaskNumCal = false;   // 是否为tasknum精确评估

HcclResult InitWorkflowMode(HcclWorkflowMode mode)
{
    CHK_RET(SetWorkflowMode(mode));
    return HCCL_SUCCESS;
}

HcclResult SetWorkflowMode(HcclWorkflowMode mode)
{
    g_workflowMode = mode;
    return HCCL_SUCCESS;
}

HcclWorkflowMode GetWorkflowMode() { return g_workflowMode; }

void SetLaunchKernelMode(bool state) { g_isLaunchKernel = state; }

bool IsLaunchKernelMode(void) { return g_isLaunchKernel; }

void SetTaskNumCalMode(bool state) { g_isTaskNumCal = state; }

bool IsTaskNumCalMode(void) { return g_isTaskNumCal; }
