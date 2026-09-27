/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_slog.h"

struct nslb_slog_api_ops g_nslb_slog_api_ops = {.DlogRecord = NULL, .CheckLogLevel = NULL};

static void *g_slog_api_handle = NULL;

int NetCoOpenSlogSo(void)
{
    if (g_slog_api_handle == NULL) {
        g_slog_api_handle = dlopen("libunified_dlog.so", RTLD_NOW);
        if (g_slog_api_handle != NULL) {
            return 0;
        }

        return -1;
    }

    return 0;
}

int NetCoCloseSlogSo(void)
{
    int ret = dlclose(g_slog_api_handle);
    return ret;
}

int NetCoSlogApiInit(void)
{
    g_nslb_slog_api_ops.DlogRecord = (void (*)(int moduleId, int level, const char *fmt, ...))dlsym(g_slog_api_handle,
        "DlogRecord");
    if (g_nslb_slog_api_ops.DlogRecord == NULL) {
        printf("DlogRecord is NULL!");
        return -1;
    }

    g_nslb_slog_api_ops.CheckLogLevel = (int (*)(int moduleId, int logLevel))dlsym(g_slog_api_handle, "CheckLogLevel");
    if (g_nslb_slog_api_ops.CheckLogLevel == NULL) {
        printf("CheckLogLevel is NULL!");
        return -1;
    }
    return 0;
}
