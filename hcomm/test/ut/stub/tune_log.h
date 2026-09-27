/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TUNE_LOG_H
#define TUNE_LOG_H

#include <unistd.h>
#include <stdio.h>

#define TUNE_ERROR(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define TUNE_WARN(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define TUNE_INFO(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define TUNE_DEBUG(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define TUNE_EVENT(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define TUNE_CHK_RET(result, exeLog, retCode) \
    do {                                      \
        if (result) {                         \
            exeLog;                           \
            return retCode;                   \
        }                                     \
    } while (0)
#endif
