/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AICPU_MC2_HANDLER_H
#define HCCLV2_AICPU_MC2_HANDLER_H

#include <memory>
#include <shared_mutex>
#include "kernel_param_lite.h"
#include "mc2_data_type.h"
#include "stream_lite.h"
#include "ascend_hal_define.h"
#include "data_type.h"
#include "communicator_impl_lite.h"
#include "aicpu_utils.h"

namespace Hccl {

class AicpuMc2Handler {
public:
    ~AicpuMc2Handler() = default;
    static AicpuMc2Handler& GetInstance();

    HcclResult HcclGetCommHandleByCtx(void* ctx, void** opHandle) const;
    HcclResult HcclReleaseComm(void* opHandle) const;
    HcclResult HcclGetTaskStatus(void* opHandle, HcclTaskStatus* status) const;
    HcclResult HcclCheckFinishByStream(void* opHandle) const;
    HcclResult HcclPrintTaskExceptionAllComm(void* opHandle) const;
    HcclResult
    HcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast) const;
    HcclResult HcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr) const;
    HcclResult HcclLaunchOp(void* opHandle, HcclOpData* data) const;

private:
    AicpuMc2Handler();
};

} // namespace Hccl

#endif // HCCLV2_AICPU_MC2_HANDLER_H
