/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_string_util.h"

#include <cstdarg>
#include <cstdio>
#include <vector>

namespace HcclSim {
std::string StringFormat(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    va_list args_copy;
    va_copy(args_copy, args);

    int size = vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);

    if (size < 0) {
        va_end(args);
        return "";
    }

    std::vector<char> buffer(size + 1);
    vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);

    return std::string(buffer.data(), size);
}
} // namespace HcclSim
