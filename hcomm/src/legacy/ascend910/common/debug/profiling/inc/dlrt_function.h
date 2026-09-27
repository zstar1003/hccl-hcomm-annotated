/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLRTFUNCTION_H
#define HCCL_SRC_DLRTFUNCTION_H

#include <functional>
#include <securec.h>
#include <mutex>
#include <dlfcn.h>
#include "hccl/base.h"
#include "rt_external.h"
#include "externalinput_pub.h"
#include "log.h"

namespace hccl {
class DlRtFunction {
public:
    virtual ~DlRtFunction();
    static DlRtFunction& GetInstance();
    HcclResult DlRtFunctionInit();
    std::function<rtError_t(uint32_t, rtProfCtrlHandle)> dlrtProfRegisterCtrlCallback;

protected:
private:
    void* handle_{};
    std::mutex handleMutex_;
    DlRtFunction(const DlRtFunction&) = delete;
    DlRtFunction& operator=(const DlRtFunction&) = delete;
    DlRtFunction();
    HcclResult DlRtFunctionInterInit();
    HcclResult DlRtFunctionStubInit();
};
} // namespace hccl
#endif
