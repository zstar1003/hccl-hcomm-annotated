/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_co_xmap.h"
#include "net_co_main_data.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t NetCoXMapInit(NetCo *co)
{
    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_xMap", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfXMapInitArg arg = {
        .name = name,
        .memMng = co->memMng,
    };
    co->xMap = BkfXMapInit(&arg);
    return (co->xMap != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void NetCoXMapUninit(NetCo *co)
{
    if (co->xMap != VOS_NULL) {
        BkfXMapUninit(co->xMap);
        co->xMap = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
