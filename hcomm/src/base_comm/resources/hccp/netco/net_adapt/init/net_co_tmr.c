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
#include "net_co_tmr.h"
#include "net_co_main_data.h"
#include "wfw_twl_timer_by_linux.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t NetCoTmrInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    BkfTwlTmrInitArg arg = {
        .memMng = co->memMng,
        .dbgSwitch = co->dbgOn,
        .disp = co->disp,
        .logCnt = co->logCnt,
        .log = co->log,
        .mux = co->mux,
        .appHandle = co,
        .appCid = co->selfCid,
    };
    int err = snprintf_truncated_s(arg.name, sizeof(arg.name), "%s_tmr", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    co->tmrMngAdpee = BkfAdptTwlTimerInit(&arg);
    if (co->tmrMngAdpee == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tmrMngAdpee(%#x), ng\n", BKF_MASK_ADDR(co->tmrMngAdpee));
        return BKF_ERR;
    }
    co->tmrMng = BkfAdptTwlTimerInitTmrMng((BkfTwlTmrHandle)co->tmrMngAdpee);
    if (co->tmrMngAdpee == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tmrMng(%#x), ng\n", BKF_MASK_ADDR(co->tmrMng));
        return BKF_ERR;
    }

    return BKF_OK;
}

void NetCoTmrUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "tmrMng(%#x)/tmrMngAdpee(%#x)\n", BKF_MASK_ADDR(co->tmrMng),
        BKF_MASK_ADDR(co->tmrMngAdpee));

    if (co->tmrMng != VOS_NULL) {
        BkfAdptBkfTmrMngUnInit(co->tmrMng);
        co->tmrMng = VOS_NULL;
    }
    if (co->tmrMngAdpee != VOS_NULL) {
        BkfAdptTwlTimerUnInit((BkfTwlTmrHandle)co->tmrMngAdpee);
        co->tmrMngAdpee = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
