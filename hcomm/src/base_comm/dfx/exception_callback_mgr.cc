/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "exception_callback_mgr.h"
#include "exception_util.h"
#include "log.h"

using Hccl::HcclException;
using std::exception;
using std::string;

namespace hcomm {

ExceptionCallbackMgr& ExceptionCallbackMgr::GetInstance()
{
    static ExceptionCallbackMgr instance;
    return instance;
}

HcclResult ExceptionCallbackMgr::Register(HcommExceptionCallback cb, void* userData)
{
    CHK_PTR_NULL(cb);
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = std::find_if(callbacks_.begin(), callbacks_.end(), [cb](const CallbackEntry& entry) {
        return entry.cb == cb;
    });
    if (it != callbacks_.end()) {
        it->userData = userData;
        HCCL_INFO("[%s] update existing cb[%p], userData[%p], total[%zu]", __func__, cb, userData, callbacks_.size());
        return HCCL_SUCCESS;
    }
    callbacks_.push_back({cb, userData});
    HCCL_INFO("[%s] success, cb[%p], userData[%p], total[%zu]", __func__, cb, userData, callbacks_.size());
    return HCCL_SUCCESS;
}

HcclResult ExceptionCallbackMgr::Unregister(HcommExceptionCallback cb)
{
    CHK_PTR_NULL(cb);
    std::unique_lock<std::shared_mutex> lock(mutex_);
    size_t beforeSize = callbacks_.size();
    callbacks_.erase(
        std::remove_if(
            callbacks_.begin(), callbacks_.end(),
            [cb](const CallbackEntry& entry) {
                return entry.cb == cb;
            }),
        callbacks_.end());
    size_t removed = (beforeSize > callbacks_.size()) ? (beforeSize - callbacks_.size()) : 0;
    HCCL_INFO("[%s] cb[%p], removed[%zu], remaining[%zu]", __func__, cb, removed, callbacks_.size());
    return HCCL_SUCCESS;
}

void ExceptionCallbackMgr::NotifyAll(const HcommExceptionInfo& exceptionInfo)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& entry : callbacks_) {
        if (entry.cb == nullptr) {
            continue;
        }
        TRY_CATCH_PRINT_ERROR(entry.cb(&exceptionInfo, entry.userData));
    }
}

bool ExceptionCallbackMgr::IsEmpty()
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return callbacks_.empty();
}

} // namespace hcomm
