/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DLRTS_FUNCTION_H
#define DLRTS_FUNCTION_H

#include <mutex>
#include "hccl_dl.h"
#include "log.h"

template <const char* soName>
class DlRtsFunction {
public:
    explicit DlRtsFunction() = default;
    virtual ~DlRtsFunction()
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
        HCCL_INFO("open so: %s", soName);
        handle_ = HcclDlopen(soName, RTLD_NOW);
        const char* errMsg = dlerror();
        CHK_PRT_RET(
            handle_ == nullptr,
            HCCL_ERROR(
                "dlopen [%s] failed, %s", soName,
                (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg), );
        isInit_ = true;
    }
    void* handle_ = nullptr;
    std::mutex handleMutex_;
    bool isInit_ = false;
};
#endif
