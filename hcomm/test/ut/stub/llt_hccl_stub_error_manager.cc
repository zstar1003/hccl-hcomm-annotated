/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdio.h>
#include <cstdarg>
#include <securec.h>
#include <vector>
#include "log.h"

using namespace std;

// old, will be delete after all caller transfer to new
namespace ErrorMessage {
int FormatErrorMessage(char* str_dst, size_t dst_max, const char* format, ...)
{
    int len = 0;
    va_list arg;
    (void)va_start(arg, format); // lint !e530
    (void)memset_s(str_dst, dst_max, 0, dst_max);
    /*
        C库标准的vsnprintf()函数在字符串超出缓存长度后返回需要的缓存空间.
        公司的安全函数库包装后的vsnprintf_s()在字符串超出缓存长度后返回值为-1, 无法根据返回值重新申请堆内存.
    */
    len = vsnprintf_s(str_dst, dst_max, (dst_max - 1), format, arg);
    va_end(arg);
    HCCL_INFO("lkk_test1 is %d\n", len);
    return len;
}
} // namespace ErrorMessage

namespace error_message {
std::string TrimPath(const std::string& str);
struct Context {
    uint64_t work_stream_id;
    std::string first_stage;
    std::string second_stage;
    std::string log_header;
};
struct ErrorManagerContext {
    uint64_t work_stream_id; // default value 0, invalid value
    uint64_t reserved[7];
    ErrorManagerContext() : work_stream_id(0), reserved{} {}
};
int FormatErrorMessage(char* str_dst, size_t dst_max, const char* format, ...)
{
    int ret;
    va_list arg_list;

    va_start(arg_list, format);
    ret = vsprintf(str_dst, format, arg_list);
    va_end(arg_list);
    (void)arg_list;

    if (ret < 0) {
        printf("FormatErrorMessage failed, ret:%d, pattern:%s", ret, format);
    }
    return ret;
}

void ReportInnerError(
    const char* file_name, const char* func, uint32_t line, const std::string error_code, const char* format, ...)
{
    return;
}

thread_local ErrorManagerContext errorManagerContext_;

ErrorManagerContext& GetErrMgrContext() { return errorManagerContext_; }
void SetErrMgrContext(ErrorManagerContext error_context) { return; }
int32_t ReportInnerErrMsg(
    const char* file_name, const char* func, uint32_t line, const char* error_code, const char* format, ...)
{
    return 0;
}

int32_t ReportPredefinedErrMsg(
    const char* error_code, const std::vector<const char*>& key, const std::vector<const char*>& value)
{
    return 0;
}

int32_t RegisterFormatErrorMessage(const char* error_msg, size_t error_msg_len) { return 0; }
} // namespace error_message

using namespace error_message;

class ErrorManager {
public:
    static ErrorManager& GetInstance();

    void ATCReportErrMessage(
        std::string error_code, const std::vector<std::string>& key = {}, const std::vector<std::string>& value = {});

    int ReportInterErrMessage(std::string error_code, const std::string& error_msg);

    const std::string& GetLogHeader();

    void SetErrorContext(error_message::Context error_context);

    error_message::Context& GetErrorManagerContext();

    ErrorManager() {}
    ~ErrorManager() {}

    ErrorManager(const ErrorManager&) = delete;
    ErrorManager(ErrorManager&&) = delete;
    ErrorManager& operator=(const ErrorManager&) = delete;
    ErrorManager& operator=(ErrorManager&&) = delete;
};

ErrorManager& ErrorManager::GetInstance()
{
    static ErrorManager instance;
    return instance;
}

thread_local error_message::Context error_context_ = {0, " ", " ", " "};

error_message::Context& ErrorManager::GetErrorManagerContext() { return error_context_; }

void ErrorManager::SetErrorContext(error_message::Context error_context) { return; }

void ErrorManager::ATCReportErrMessage(
    std::string error_code, const std::vector<std::string>& key, const std::vector<std::string>& value)
{
    cout << "ErrorMessage is:" << error_code << endl;
    for (int errMsgIdx = 0; errMsgIdx < key.size(); errMsgIdx++) {
        cout << key[errMsgIdx] << " is:" << value[errMsgIdx] << endl;
    }
}

int ErrorManager::ReportInterErrMessage(std::string error_code, const std::string& error_msg)
{
    cout << "ErrorMessage is:" << error_code << endl;
    cout << error_msg << endl;

    return 0;
}

std::string Stage = "[Stage0][Stage1]";
const std::string& ErrorManager::GetLogHeader() { return Stage; }

#ifdef __GNUC__
std::string error_message::TrimPath(const std::string& str)
{
    if (str.find_last_of('/') != std::string::npos) {
        return str.substr(str.find_last_of('/') + 1U);
    }
    return str;
}
#else
std::string error_message::TrimPath(const std::string& str)
{
    if (str.find_last_of('\\') != std::string::npos) {
        return str.substr(str.find_last_of('\\') + 1U);
    }
    return str;
}
#endif
