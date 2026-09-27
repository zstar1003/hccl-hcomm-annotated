/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_PROFILING_MANAGER_H__
#define __AICPU_PROFILING_MANAGER_H__

#include "prof_common.h"
#include "common/aicpu_hccl_def.h"

namespace dfx {
class AicpuProfilingManager {
public:
    static HcclResult ReportTaskExecTimeLine(AicpuComProf* acprof, u32 turnOffset = 0U);
    static HcclResult ReportTaskInfo();
    static void Init(const AicpuComContext* ctx);

private:
    static void Ctx2MsprofAicpuMC2HcclInfo(const AicpuComContext* ctx, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo);
    static std::string AicpuKfcProfCommTurnToString(const AicpuKfcProfCommTurn& aicpuKfcProfCommTurn);
};
} // namespace dfx
#endif // __AICPU_PROFILING_MANAGER_H__
