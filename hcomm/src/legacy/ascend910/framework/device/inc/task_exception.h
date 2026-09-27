/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_EXCEPTION_H
#define TASK_EXCEPTION_H

#include "hccl_types.h"
#include "hccl_common.h"
#include "aicpu_operator_pub.h"
#include "hcomm_diag.h"

namespace hccl {
constexpr u32 OP_INFO_MAX_SIZE = 2048;
struct IndOpInfo {
    uint8_t opInfo[OP_INFO_MAX_SIZE]{0};
    HcommGetOpInfoCallback callback = nullptr;
    uint32_t opIndex = 0; // 记录算子下发index
};

class TaskException {
public:
    TaskException();
    ~TaskException();

    HcclResult Init(u32 devId, u32 localUserRank, const std::string& identifier);
    HcclResult RegisterOpInfo(void* opInfo, u32 size);
    HcclResult RegisterOpInfoCallback(HcommGetOpInfoCallback callback);
    HcclResult PrintTaskException(Stream& stream);
    HcclResult PrintTaskExceptionByTaskId(u8 sqeType, u16 taskId, hccl::Stream& stream, u32 tail);

    inline u32 GetOpRingBufferIdx() const { return opRingBufferIdx_; }

private:
    std::string GetTaskExceptionTaskInfo(u32 sqHead, SqeRingBuffer* sqeContextBuffer);
    void PrintTaskExceptionTaskQue(u32 sqIdx, SqeRingBuffer* sqeContextBuffer); // 打印当前位置的前序task
    std::string GetTaskBriefsInfo(u32 idx, SqeRingBuffer* sqeContextBuffer);
    void PrintTaskExceptionOpInfo(IndOpInfo& indOp);

    bool IsRepeatPrint(u32 streamId, u32 opIndex, u32 sqHead);

    IndOpInfo indOpInfos_[OPINFO_RING_BUFFER_MAX];
    u32 opRingBufferIdx_ = OPINFO_RING_BUFFER_MAX - 1;
    u32 devId_ = 0;
    u32 localUserRank_ = 0;
    std::string identifier_;
    // streamId -> <opIndex, sqHead> : 记录每条流上一次打印taskException的opIndex和head，避免重复打印
    std::unordered_map<u32, std::pair<u32, u32>> threadPrintState_;
};
} // namespace hccl
#endif
