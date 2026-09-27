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

using namespace std;

namespace error_message {
std::string TrimPath(const std::string& str);
struct Context {
    uint64_t work_stream_id;
    std::string first_stage;
    std::string second_stage;
    std::string log_header;
};

void ReportInnerError(
    const char* file_name, const char* func, uint32_t line, const std::string error_code, const char* format, ...)
{
    return;
}

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

void ErrorManager::SetErrorContext(error_message::Context error_context) { return; }

void ErrorManager::ATCReportErrMessage(
    std::string error_code, const std::vector<std::string>& key, const std::vector<std::string>& value)
{
    return;
}

int ErrorManager::ReportInterErrMessage(std::string error_code, const std::string& error_msg) { return 0; }
