/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_DL_H
#define HCCP_DL_H

#include <stdio.h>
#include <dlfcn.h>
#include <errno.h>
#include "user_log.h"

#define DL_ATTRI_VISI_DEF __attribute__((visibility("default")))

#define DL_API_RET_IS_NULL_CHECK(p, str)                                                                               \
    do {                                                                                                               \
        if ((p) == NULL) {                                                                                             \
            hccp_err("ptr is NULL!, [%s]", (str));                                                                     \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

#define DL_API_RET_IS_NULL_INFO(p, str)                                                                                \
    do {                                                                                                               \
        if ((p) == NULL) {                                                                                             \
            hccp_info("ptr is NULL!, [%s]", (str));                                                                    \
        }                                                                                                              \
    } while (0)

void *HccpDlopen(const char *libName, int mode);
void *HccpDlsym(void *handle, const char *funcName);
int HccpDlclose(void *handle);
#endif // HCCP_DL_H
