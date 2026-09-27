/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROFILING_COMMON_H
#define PROFILING_COMMON_H
#include <map>
#include <string>

#include "task_param.h"
#include "enum_factory.h"

namespace Hccl {

MAKE_ENUM(
    HcclWorkflowMode, HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB = 0, HCCL_WORKFLOW_MODE_OP_BASE = 1,
    HCCL_WORKFLOW_MODE_RESERVED)

MAKE_ENUM(SimpleTaskType, SDMA = 0, RDMA, LOCAL, UB, ROCE)

MAKE_ENUM(TaskRole, DST = 0, SRC)

const std::map<TaskParamType, std::string> PROF_TASK_OP_NAME_V2 = {
    {TaskParamType::TASK_SDMA, "Memcpy"},
    {TaskParamType::TASK_RDMA, "RDMASend"},
    {TaskParamType::TASK_REDUCE_INLINE, "Reduce_Inline"},
    {TaskParamType::TASK_REDUCE_TBE, "Reduce_TBE"},
    {TaskParamType::TASK_NOTIFY_RECORD, "Notify_Record"},
    {TaskParamType::TASK_NOTIFY_WAIT, "Notify_Wait"},
    {TaskParamType::TASK_SEND_NOTIFY, "Send_Notify"},
    {TaskParamType::TASK_SEND_PAYLOAD, "Send_Payload"},
    {TaskParamType::TASK_WRITE_WITH_NOTIFY, "Write_With_Notify"},
    {TaskParamType::TASK_UB_INLINE_WRITE, "Ub_Inline_Write"},
    {TaskParamType::TASK_UB_REDUCE_INLINE, "Ub_Reduce_Inline"},
    {TaskParamType::TASK_UB, "Ub_Write_Or_Read"},
    {TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY, "Reuce_With_Notify"},
    {TaskParamType::TASK_CCU, "Ccu"},
    {TaskParamType::TASK_AICPU_KERNEL, "AicpuKernel"},
    {TaskParamType::TASK_AICPU_REDUCE, "Aicpu_Reduce"},
    {TaskParamType::TASK_AIV, "AivKernel"},
    {TaskParamType::TASK_DPU_KERNEL, "DpuKernel"},
    {TaskParamType::TASK_DPU_THREAD_FENCE, "Dpu_ThreadFence"},
    {TaskParamType::TASK_DPU_CHANNEL_FENCE, "Dpu_ChannelFence"},
    {TaskParamType::TASK_DPU_INLINE_WRITE, "Dpu_Notify_Record"},
    {TaskParamType::TASK_DPU_NOTIFY_WAIT, "Dpu_Notify_Wait"},
    {TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY, "Dpu_Write_With_Notify"},
};

inline std::string GetProfTaskOpNameV2(TaskParamType type)
{
    CHK_PRT_RET(PROF_TASK_OP_NAME_V2.empty(), HCCL_ERROR("PROF_TASK_OP_NAME_V2 has not inited."), "invalid");
    auto it = PROF_TASK_OP_NAME_V2.find(type);
    if (it != PROF_TASK_OP_NAME_V2.end()) {
        return it->second;
    }
    return "unknown";
}

} // namespace Hccl

#endif // PROFILING_COMMON_H
