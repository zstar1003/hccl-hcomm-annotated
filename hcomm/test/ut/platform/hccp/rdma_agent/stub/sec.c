/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include <stdlib.h>

extern int sec_cpy_ret;

SECUREC_API errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    memcpy(dest, src, count);
    return 0;
}

SECUREC_API errno_t memset_s(void* dest, size_t destMax, int c, size_t count)
{
    memset(dest, c, count);
    return 0;
}

SECUREC_API errno_t strcpy_s(char* strDest, size_t destMax, const char* strSrc)
{
    strcpy(strDest, strSrc);
    return 0;
}

SECUREC_API errno_t strncpy_s(char* strDest, size_t destMax, const char* strSrc, size_t count)
{
    strncpy(strDest, strSrc, count);
    return 0;
}

SECUREC_API int sprintf_s(char* strDest, size_t destMax, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsprintf(strDest, format, args);
    va_end(args);
    return 1;
}
