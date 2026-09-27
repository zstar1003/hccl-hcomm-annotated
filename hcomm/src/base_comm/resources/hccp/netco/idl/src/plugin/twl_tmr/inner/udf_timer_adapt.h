/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_TIMER_ADAPT_H
#define UDF_TIMER_ADAPT_H

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UDF_OK 0
#define UDF_ERROR (-1)

#define UDF_TASK_PROC_NAME_MAX_LEN (32)

#define LOG_INNER_DEBUG printf
#define LOG_INNER_ERR printf
#define LOG_INNER_WARN printf
#define UDF_CMD_OUTPUT printf
#define UDF_CMD_SPLIT_LINE                                                                                             \
    "----------------------------------------------------------------------------------------------------------------" \
    "------"

void DbgGetFuncName(void *funcAddr, char *funcName, size_t size);
uint32_t SystimeGetMilliSec(uint64_t *milliSec);
uint32_t SchFuncNameDeal(char *funcName, size_t size, const char *funcStr);
void UdfTimeStrGet(char *timeStr, size_t targetimeStrSize);

#ifdef __cplusplus
}
#endif

#endif
