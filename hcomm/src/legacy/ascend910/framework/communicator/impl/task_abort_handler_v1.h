/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_ABORT_HANDLER_H
#define HCCL_TASK_ABORT_HANDLER_H
#include "task_abort_handler_pub.h"
#include "adapter_rts_common.h"
int32_t
ProcessTaskAbortHandleCallback(int32_t deviceLogicId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args);
enum class TaskAbortResult {
    TaskAbort_Success = 0, // taskabortSuccess
    TaskAbort_Fail = 1,    // taskabortFail
    TaskAbort_TimeOut = 2  // taskabortTimeout
};
#endif
