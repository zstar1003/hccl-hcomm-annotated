/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((co)->log)
#define BKF_MOD_NAME ((co)->name)
#include "net_co_pfm.h"
#include "net_co_main_data.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t NetCoPfmInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_pfm", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfPfmInitArg arg = {
        .name = name,
        .memMng = co->memMng,
        .disp = co->disp,
        .logCnt = co->logCnt,
        .log = co->log,
        .enable = co->pfmOn,
    };
    co->pfm = BkfPfmInit(&arg);
    return (co->pfm != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void NetCoPfmUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "pfm(%#x)\n", BKF_MASK_ADDR(co->pfm));

    if (co->pfm != VOS_NULL) {
        BkfPfmUninit(co->pfm);
        co->pfm = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
