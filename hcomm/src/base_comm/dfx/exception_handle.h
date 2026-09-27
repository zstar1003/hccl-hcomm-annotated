/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXCEPTION_HANDLE_H
#define HCOMM_EXCEPTION_HANDLE_H

#include "aicpu/aicpu_hccl_sqcq.h"
#include "daemon_func.h"
#include "hccl_types.h"
#include "hcomm/hcomm_exception.h"
#include "thread.h"
#include "stream_lite.h"
#include <unordered_map>
#include <unordered_set>

namespace hcomm {

constexpr uint8_t DFX_SQE_TYPE_UDMA = 9;

class ExceptionHandle : public Hccl::DaemonFunc {
public:
    static ExceptionHandle& GetInstance();
    HcclResult HandleExceptionCqe();
    void Call() override;
    void ClearStreamState(uint32_t sqId);

private:
    HcclResult CheckExceptionCqe(hccl::Thread* thread, uint32_t devId);
    HcclResult FillExceptionInfo(
        HcommExceptionInfo& exceptionInfo, hccl::Thread* thread, uint32_t taskId,
        const rtLogicCqReport_t& cqeException);
    uint32_t SwitchCqeErrCodeToHcclErrCode(uint32_t cqeErrCode);
    HcclResult CheckRepeatBySqeId(Hccl::StreamLite* streamLite, uint32_t devId, uint16_t taskId, uint16_t streamId);
    dfx::CqeStatus ReceiveCqeReport(uint32_t devId, Hccl::StreamLite* streamLite, rtLogicCqReport_t& cqeException);
    static uint32_t GetSqeId(uint16_t taskId, uint16_t streamId);

    std::unordered_map<uint32_t, uint32_t> threadsPrinted_; // sqId -> sqeId
    std::unordered_set<uint32_t> sqCqeErrorSet_;            // 记录CqReportRecv返回kCqeInnerError的sqId
};

} // namespace hcomm

#endif // HCOMM_EXCEPTION_HANDLE_H
