/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TASK_ABORT_HANDLER_H
#define HCCLV2_TASK_ABORT_HANDLER_H
#include <mutex>
#include "hccl_communicator.h"
#include "orion_adapter_rts.h"

namespace Hccl {
int32_t
ProcessTaskAbortHandleCallback(int32_t deviceLogicId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args);
MAKE_ENUM(
    TaskAbortResult,
    TASK_ABORT_SUCCESS = 0, // taskabortSuccess
    TASK_ABORT_FAIL = 1,    // taskabortFail
    TASK_ABORT_TIMEOUT = 2) // taskabortTimeout

// class HcclCommunicator;

class TaskAbortHandler {
public:
    TaskAbortHandler();
    ~TaskAbortHandler();
    static TaskAbortHandler& GetInstance();
    HcclResult Register(HcclCommunicator* communicator);
    HcclResult UnRegister(HcclCommunicator* communicator);

private:
    std::vector<HcclCommunicator*> commVector;
    std::mutex vecMutex;

    TaskAbortHandler(const TaskAbortHandler&) = delete;
    TaskAbortHandler& operator=(const TaskAbortHandler&) = delete;
};
} // namespace Hccl

#endif
