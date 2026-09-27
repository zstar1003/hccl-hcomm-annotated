/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include "net_co_main_data.h"
#include "securec.h"
#include "net_co_mem.h"

#ifdef __cplusplus
extern "C" {
#endif
STATIC void *NetCoMemAdpOnMalloc(NetCo *co, uint32_t len, const char *str, const uint16_t num)
{
    (void)co;
    (void)str;
    (void)num;
    if (len == 0) {
        return VOS_NULL;
    }
    return calloc(len, sizeof(char));
}
STATIC void NetCoMemAdpOnFree(NetCo *co, void *ptr, const char *str, const uint16_t num)
{
    (void)co;
    (void)str;
    (void)num;
    free(ptr);
}
uint32_t NetCoMemInit(NetCo *co)
{
    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_mem", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfIMem arg = {
        .name = name,
        .cookie = co,
        .malloc = (F_BKF_IMEM_MALLOC)NetCoMemAdpOnMalloc,
        .free = (F_BKF_IMEM_FREE)NetCoMemAdpOnFree,
    };
    co->memMng = BkfMemInit(&arg);
    return (co->memMng != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void NetCoMemUninit(NetCo *co)
{
    if (co->memMng != VOS_NULL) {
        BkfMemUninit(co->memMng);
        co->memMng = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
