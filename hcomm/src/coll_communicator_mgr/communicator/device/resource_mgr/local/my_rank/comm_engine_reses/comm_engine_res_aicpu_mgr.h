/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_ENGINE_RES_AICPU_MGR_H
#define COMM_ENGINE_RES_AICPU_MGR_H

#include "common.h"
#include "threads/thread_aicpu_mgr.h"
#include "notify/notify_aicpu_mgr.h"
#include <memory>

class CommEngineResAicpuMgr {
public:
    CommEngineResAicpuMgr(hccl::HcclCommDfxLite& dfx, std::function<HcclResult(bool)> checkExecStatusCallback);
    ~CommEngineResAicpuMgr();

    HcclResult InitThreads(ThreadMgrAicpuParam* param);
    HcclResult NotifyFree(NotifyMgrAicpuParam* param);
    HcclResult NotifyAlloc(NotifyMgrAicpuParam* param);
    void ReserveNotifyCapacity(size_t n);
    const std::vector<std::shared_ptr<hccl::Thread>>& GetAllThread();
    std::shared_mutex& GetThreadMutex();

private:
    std::unique_ptr<ThreadAicpuMgr> threadMgr_;
    std::unique_ptr<NotifyAicpuMgr> notifyMgr_;
};

#endif // COMM_ENGINE_RES_AICPU_MGR_H
