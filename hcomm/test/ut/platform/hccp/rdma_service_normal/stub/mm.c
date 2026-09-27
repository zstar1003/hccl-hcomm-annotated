/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
/*
void *mmap(void *addr, size_t length, int prot, int flags,
                  int fd, off_t offset)
{
    return malloc(length);
};

int munmap(void *addr, size_t length)
{
    free(addr);
};
*/

int memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    memcpy(dest, src, count);
    return 0;
}

int memset_s(void* dest, size_t destMax, int c, size_t count)
{
    memset(dest, c, count);
    return 0;
}

int strcpy_s(void* dest, size_t destMax, const void* src)
{
    if (src == NULL) {
        printf("src is null\n");
        return -1;
    }
    if (strlen(src) > destMax) {
        printf("error %d %d\n", strlen(src), destMax);
        return -1;
    }
    strcpy(dest, src);
    return 0;
}

int strncpy_s(char* strDest, size_t destMax, const char* strSrc, size_t count)
{
    strncpy(strDest, strSrc, count);
    return 0;
}

int sprintf_s(char* strDest, size_t destMax, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsprintf(strDest, format, args);
    va_end(args);
    return 1;
}

int snprintf_s(char* strDest, size_t destMax, size_t count, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(strDest, count, format, args);
    va_end(args);
    return 1;
}

int strcat_s(char* strDest, size_t destMax, const char* strSrc)
{
    strcat(strDest, strSrc);
    return 0;
}

int sscanf_s(const char* buffer, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsscanf(buffer, format, args);
    va_end(args);
    return 1;
}
