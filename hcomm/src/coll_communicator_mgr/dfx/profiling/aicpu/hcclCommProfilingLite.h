/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_PROFILING_LITE_H
#define HCCL_COMM_PROFILING_LITE_H
#include "dfx_profiling_reporter_lite.h"
#include "hccl/hccl_types.h"
#include <vector>

namespace hccl {

class HcclCommProfilingLite {
public:
    HcclCommProfilingLite(u32 deviceId);
    ~HcclCommProfilingLite();
    HcclResult Init();
    void ReportAllTasks(const std::vector<hccl::Thread*>& threads, const Hccl::DfxCommContext& ctx);
    void UpdateProfStat();
    void ReportStreamTask(Hccl::TaskInfoCircularQueue* taskQueue, const Hccl::DfxCommContext& ctx);

private:
    Hccl::DfxProfilingReporterLite* profilingReporterLite_{nullptr};
    bool initializedFlag_{false};
};
} // namespace hccl

#endif
