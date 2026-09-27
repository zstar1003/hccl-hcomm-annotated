/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLPROFFUNC_H
#define HCCL_SRC_DLPROFFUNC_H

#include <functional>
#include <mutex>
#include <dlfcn.h>
#include <hccl/hccl_types.h>
#include "prof_common.h"
#include "hccl/base.h"

namespace hccl {
class DlProfFunc {
public:
    virtual ~DlProfFunc();
    static DlProfFunc& GetInstance();
    HcclResult DlProfFunctionInit();
    std::function<uint64_t(void)> dlMsprofSysCycleTime{};

private:
    void* handle_{};
    std::mutex handleMutex_;
    DlProfFunc(const DlProfFunc&) = delete;
    DlProfFunc& operator=(const DlProfFunc&) = delete;
    DlProfFunc();
    HcclResult DlProfFunctionInterInit();
    void DlProfFunctionStubInit();
};
} // namespace hccl
#endif
