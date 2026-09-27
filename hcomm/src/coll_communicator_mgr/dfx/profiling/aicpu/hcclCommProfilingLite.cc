/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcclCommProfilingLite.h"
#include "dfx_profiling_reporter_lite.h"
#include "log.h"

namespace hccl {
HcclCommProfilingLite::HcclCommProfilingLite(u32 deviceId) { (void)deviceId; }

HcclCommProfilingLite::~HcclCommProfilingLite()
{
    delete profilingReporterLite_;
    profilingReporterLite_ = nullptr;
}

HcclResult HcclCommProfilingLite::Init()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    CHK_RET(Hccl::DfxProfilingHandlerLite::GetInstance().Init());
    profilingReporterLite_ = new Hccl::DfxProfilingReporterLite(&Hccl::DfxProfilingHandlerLite::GetInstance());
    CHK_RET(profilingReporterLite_->Init());
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

void HcclCommProfilingLite::ReportAllTasks(const std::vector<hccl::Thread*>& threads, const Hccl::DfxCommContext& ctx)
{
    profilingReporterLite_->ReportAllTasks(threads, ctx);
}

void HcclCommProfilingLite::ReportStreamTask(Hccl::TaskInfoCircularQueue* taskQueue, const Hccl::DfxCommContext& ctx)
{
    profilingReporterLite_->ReportStreamTask(taskQueue, ctx);
}

void HcclCommProfilingLite::UpdateProfStat() { profilingReporterLite_->UpdateProfStat(); }
} // namespace hccl
