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
int g_sprintf_s_flag = 0;
int g_sprintf_s_flag1 = 1;

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

errno_t memcpy_sOptAsm(void* dest, size_t destMax, const void* src, size_t count)
{
    return memcpy_s(dest, destMax, src, count);
}

errno_t memcpy_sOptTc(void* dest, size_t destMax, const void* src, size_t count)
{
    return memcpy_s(dest, destMax, src, count);
}

SECUREC_API errno_t memmove_s(void* dest, size_t destMax, const void* src, size_t count)
{
    memmove(dest, src, count);
    return 0;
}

int vsnprintf_s(char* strDest, size_t destMax, size_t count, const char* format, va_list arglist)
{
    vsnprintf(strDest, count, format, arglist);
    if (g_sprintf_s_flag1 > 0) {
        return 0;
    } else {
        return -1;
    }
}

errno_t strcat_s(char* strDest, size_t destMax, const char* strSrc)
{
    if (strcat(strDest, strSrc) != strDest) {
        return -1;
    }
    return 0;
}
