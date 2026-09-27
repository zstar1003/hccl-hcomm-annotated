/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_HCCL_EXCEPTION_H
#define HCCL_HCCL_EXCEPTION_H

#include <exception>
#include <string>
#include <vector>
#include "exception_definition.h"

#include <execinfo.h>

namespace Hccl {
class HcclException : public std::exception {
public:
    explicit HcclException(const ExceptionType& exceptionType, const std::string& userDefinedMsg)
        : exceptionType(exceptionType),
          userDefinedMsg(userDefinedMsg),
          errorMsg(ExceptionInfo::GetErrorMsg(exceptionType) + userDefinedMsg)
    {
        StoreBackTrace();
    }

    const char* what() const noexcept override { return errorMsg.c_str(); }

    HcclResult GetErrorCode() const { return ExceptionInfo::GetErrorCode(exceptionType); }

    std::vector<std::string> GetBackTraceStrings() const { return backtraceStrings; }

private:
    void StoreBackTrace()
    {
        constexpr int BACKTRACE_DEPTH = 15;
        void* array[BACKTRACE_DEPTH];
        int size = backtrace(array, BACKTRACE_DEPTH);

        char** callBackStrings = backtrace_symbols(array, size);
        if (callBackStrings == nullptr) {
            backtraceStrings.emplace_back("Failed to get backtrace symbols");
            return;
        }

        for (auto i = 0; i < size; ++i) {
            backtraceStrings.emplace_back(callBackStrings[i]);
        }
        free(callBackStrings);
    };

    std::vector<std::string> backtraceStrings;
    ExceptionType exceptionType;
    std::string userDefinedMsg{""};
    std::string errorMsg{""};
};

} // namespace Hccl

#endif // HCCL_HCCL_EXCEPTION_H
