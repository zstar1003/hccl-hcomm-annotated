/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_EXECUTOR_TRACER_H__
#define __AICPU_EXECUTOR_TRACER_H__

#include "common/aicpu_hccl_def.h"
#include "framework/aicpu_communicator.h"

namespace dfx_tracer {
class AicpuExecutorTracer {
    explicit AicpuExecutorTracer();

public:
    static void HandleBackGround(AicpuComContext* const ctx);
    static void StopLaunchCommandHandle(AicpuComContext* const ctx);
    static void KfcCommandHandle(AicpuComContext* const ctx);
    static void HandleCqeStatus(AicpuComContext* const ctx);
    static void
    StopKfcThread(AicpuComContext* const ctx, std::vector<std::pair<std::string, hccl::HcclCommAicpu*>> aicpuCommInfo);
    static void
    SetCqeQueryInput(const uint32_t devId, const HcclComStreamInfo& streamInfo, CqeQueryInput& cqeQueryInput);

private:
    static void HandleCqeStatusByRank(AicpuComContext* const ctx, uint32_t rank);
    static void PrintTaskException(const rtLogicCqReport_t& reportOfOne);
    static uint8_t getTrailingZeros(uint8_t num);
};

class KfcCommandHandles {
public:
    static void ClearFunc(AicpuComContext* const ctx);
    static void StopFunc(AicpuComContext* const ctx);

private:
    static void ClearCq(AicpuComContext* const ctx);
};

} // namespace dfx_tracer
#endif // __AICPU_EXECUTOR_TRACER_H__
