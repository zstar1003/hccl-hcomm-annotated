/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_ABORT_HANDLER__H
#define HCCL_TASK_ABORT_HANDLER__H
#include <mutex>
#include "coll_comm.h"
#include "orion_adapter_rts.h"

namespace hccl {
int32_t
ProcessTaskAbortHandleCallback(int32_t deviceLogicId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args);

MAKE_ENUM(
    TaskAbortResult,
    TASK_ABORT_SUCCESS = 0,  // taskabortSuccess
    TASK_ABORT_FAIL = 1,     // taskabortFail
    TASK_ABORT_TIMEOUT = 2); // taskabortTimeout

class CollComm;

class HcclTaskAbortHandler {
public:
    HcclTaskAbortHandler();
    ~HcclTaskAbortHandler();
    HcclResult Register(CollComm* communicator);
    HcclResult UnRegister(CollComm* communicator);

    friend int32_t ProcessTaskAbortHandleCallback(
        int32_t deviceLogicId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args);

private:
    std::vector<CollComm*> commVector_;
    std::mutex vecMutex_;

    HcclTaskAbortHandler(const HcclTaskAbortHandler&) = delete;
    HcclTaskAbortHandler& operator=(const HcclTaskAbortHandler&) = delete;
};
} // namespace hccl

#endif
