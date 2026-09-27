/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXCEPTION_CALLBACK_MGR_H
#define EXCEPTION_CALLBACK_MGR_H

#include <vector>
#include <mutex>
#include <shared_mutex>
#include <cstdint>
#include "hccl_types.h"
#include "hcomm/hcomm_exception.h"

namespace hcomm {

class ExceptionCallbackMgr {
public:
    static ExceptionCallbackMgr& GetInstance();

    HcclResult Register(HcommExceptionCallback cb, void* userData);
    HcclResult Unregister(HcommExceptionCallback cb);
    void NotifyAll(const HcommExceptionInfo& exceptionInfo);
    bool IsEmpty();

private:
    ExceptionCallbackMgr() = default;
    ~ExceptionCallbackMgr() = default;
    ExceptionCallbackMgr(const ExceptionCallbackMgr&) = delete;
    ExceptionCallbackMgr& operator=(const ExceptionCallbackMgr&) = delete;

    struct CallbackEntry {
        HcommExceptionCallback cb{nullptr};
        void* userData{nullptr};
    };

    std::shared_mutex mutex_;
    std::vector<CallbackEntry> callbacks_;
};

} // namespace hcomm

#endif
