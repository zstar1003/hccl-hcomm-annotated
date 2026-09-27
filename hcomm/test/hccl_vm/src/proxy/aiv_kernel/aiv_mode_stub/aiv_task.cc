/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_task.h"

using namespace std;

namespace AivSim {
string GetTypeName(AivTaskType taskType)
{
    switch (taskType) {
        case AivTaskType::MEM_COPY:
            return "MemCopy";
        case AivTaskType::REDUCE:
            return "Reduce";
        case AivTaskType::SET_FLAG:
            return "SetFlag";
        case AivTaskType::WAIT_FLAG:
            return "WaitFlag";
        case AivTaskType::PIPE_BARRIER:
            return "PipeBarrier";
        case AivTaskType::SYNC_ALL:
            return "SyncAll";
        case AivTaskType::SEND_FLAG:
            return "SendFlag";
        case AivTaskType::RECV_FLAG:
            return "RecvFlag";
        default:
            return "UnknownTask";
    }
}
} // namespace AivSim
