/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <securec.h>
#include <stdio.h>
#include "semantics_utils.h"

namespace checker {

std::string GenMsg(const char* fileName, s32 lineNum, const char* fmt, ...)
{
    char buffer[LOG_TMPBUF_SIZE];
    va_list arg;
    (void)va_start(arg, fmt);
    (void)memset_s(buffer, LOG_TMPBUF_SIZE, 0, sizeof(buffer));
    vsnprintf_s(buffer, sizeof(buffer), (sizeof(buffer) - 1), fmt, arg);
    va_end(arg);

    char ret[LOG_TMPBUF_SIZE];
    snprintf_s(ret, sizeof(ret), (sizeof(ret) - 1), "[%s:%d]%s", fileName, lineNum, buffer);

    return std::string(ret);
}

} // namespace checker
