/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXCEPTION_HANDLER_H
#define EXCEPTION_HANDLER_H

#include <stdexcept>
#include <string>
#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "log.h"

namespace hccl {

// 异常处理器类
class ExceptionHandler {
public:
    static HcclResult HandleException(const char* functionName);

    static void ThrowIfErrorCode(
        HcclResult errorCode, const std::string& errString, const char* fileName, s32 lineNum,
        const char* functionName);
};

// HCCL异常码类，用于识别特定HCCL错误码
class HcclException : public std::exception {
public:
    HcclException(HcclResult code, const std::string& msg) : code_(code), msg_(msg) {}

    HcclResult code() const { return code_; }

    const char* what() const noexcept override { return msg_.c_str(); }

private:
    HcclResult code_;
    std::string msg_;
};

// 宏定义，用于包装 C 接口函数的异常处理
#define EXCEPTION_HANDLE_BEGIN try {
#define EXCEPTION_HANDLE_END_INFO(func_name) \
    }                                        \
    catch (...) { return hccl::ExceptionHandler::HandleException(func_name); }

#define EXCEPTION_HANDLE_END EXCEPTION_HANDLE_END_INFO(__func__)

#define EXCEPTION_THROW_IF_ERR(call, errString)                                                 \
    do {                                                                                        \
        HcclResult ret = call;                                                                  \
        hccl::ExceptionHandler::ThrowIfErrorCode(ret, errString, __FILE__, __LINE__, __func__); \
    } while (0)

#define EXCEPTION_THROW_IF_COND_ERR(condition, errString)                                           \
    do {                                                                                            \
        if (condition) {                                                                            \
            HcclResult ret = HCCL_E_INTERNAL;                                                       \
            hccl::ExceptionHandler::ThrowIfErrorCode(ret, errString, __FILE__, __LINE__, __func__); \
        }                                                                                           \
    } while (0)

} // namespace hccl

#endif
