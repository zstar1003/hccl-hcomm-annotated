/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DLRTS_FUNCTION_V2_H
#define DLRTS_FUNCTION_V2_H

#include <mutex>
#include <dlfcn.h>
#include "hccl_dl.h"
#include "log.h"
namespace Hccl {
template <const char* soName>
class DlRtsFunctionV2 {
public:
    explicit DlRtsFunctionV2() = default;
    virtual ~DlRtsFunctionV2()
    {
        if (handle_ != nullptr) {
            HcclDlclose(handle_);
            handle_ = nullptr;
        }
    }

    template <const char* funcName>
    void* Handle()
    {
        Init();
        if (handle_ == nullptr) {
            return nullptr;
        }
        return HcclDlsym(handle_, funcName);
    }

private:
    void Init()
    {
        std::lock_guard<std::mutex> lock(handleMutex_);
        if (isInit_) {
            return;
        }
        HCCL_INFO("DlRtsFunctionInit start, open so: %s", soName);
        handle_ = HcclDlopen(soName, RTLD_NOW);
        const char* errMsg = dlerror();
        if (handle_ == nullptr) {
            HCCL_ERROR(
                "dlopen [%s] failed, %s", soName,
                (errMsg == nullptr) ? "check the file exist or permission denied." : errMsg);
            return;
        }
        isInit_ = true;
    }
    void* handle_ = nullptr;
    std::mutex handleMutex_;
    bool isInit_ = false;
};
} // namespace Hccl
#endif // DLRTS_FUNCTION_V2_H
