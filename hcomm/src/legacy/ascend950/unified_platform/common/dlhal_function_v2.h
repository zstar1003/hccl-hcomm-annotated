/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLHALFUNCTION_H_V2
#define HCCL_SRC_DLHALFUNCTION_H_V2

#include <functional>
#include <mutex>

#include "ascend_hal.h"
#include "ascend_hal_define.h"
#include <hccl/hccl_types.h>

namespace Hccl {
class DlHalFunctionV2 {
public:
    virtual ~DlHalFunctionV2();
    static DlHalFunctionV2& GetInstance();
    HcclResult DlHalFunctionInit();
    std::function<drvError_t(unsigned int, struct event_summary*)> dlHalEschedSubmitEvent;
    std::function<drvError_t(int, unsigned int*, unsigned int*, unsigned int*, unsigned int*)>
        dlHalDrvQueryProcessHostPid;
    std::function<drvError_t(uint32_t, int32_t, int32_t, int64_t*)> dlHalGetDeviceInfo;

protected:
private:
    void* handle_;
    std::mutex handleMutex_;
    DlHalFunctionV2(const DlHalFunctionV2&);
    DlHalFunctionV2& operator=(const DlHalFunctionV2&);
    DlHalFunctionV2();
    HcclResult DlHalFunctionEschedInit();
};
} // namespace Hccl

#endif // HCCL_SRC_DLHALFUNCTION_H
