/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DFXPROFILING_REPORTER_LITE_H
#define HCCL_DFXPROFILING_REPORTER_LITE_H
#include "dfx_profiling_handler_lite.h"
#include "dfx_circular_queue.h"
#include <vector>

namespace hccl {
class Thread;
}

namespace Hccl {

class DfxProfilingReporterLite {
public:
    explicit DfxProfilingReporterLite(DfxProfilingHandlerLite* profilingHandlerLite);
    virtual ~DfxProfilingReporterLite();
    HcclResult Init();

    void ReportAllTasks(const std::vector<hccl::Thread*>& threads, const DfxCommContext& ctx);
    void ReportStreamTask(TaskInfoCircularQueue* taskQueue, const DfxCommContext& ctx);
    void UpdateProfStat();

private:
    void ReportAllTasksLog(const std::vector<hccl::Thread*>& threads) const;
    DfxProfilingHandlerLite* profilingHandlerLite_{nullptr};
    bool initializedFlag_{false};
};
} // namespace Hccl

#endif // HCCL_DFXPROFILING_REPORTER_LITE_H
