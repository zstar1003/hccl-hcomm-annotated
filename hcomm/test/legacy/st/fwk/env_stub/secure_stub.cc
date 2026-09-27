/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "secure_stub.h"

#ifdef __cplusplus
extern "C" {

errno_t memset_s(void* dest, size_t destMax, int c, size_t count)
{
    int safeCount = count < destMax ? count : destMax;
    memset(dest, c, safeCount);
    return 0;
}

errno_t strcpy_s(char* strDest, size_t destMax, const char* strSrc)
{
    strcpy(strDest, strSrc);
    return 0;
}

int snprintf_s(char* strDest, size_t destMax, size_t count, const char* format, ...) SECUREC_ATTRIBUTE(4, 5)
{
    return 0;
}

errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    if (count > destMax) {
        printf("\n count %zu > destMax %zu failed\n", count, destMax);
        return -1; // 或者其他错误代码
    }
    memcpy(dest, src, count);
    return 0;
}

SECUREC_API errno_t strncpy_s(char* strDest, size_t destMax, const char* strSrc, size_t count) { return 0; }
}
#endif
