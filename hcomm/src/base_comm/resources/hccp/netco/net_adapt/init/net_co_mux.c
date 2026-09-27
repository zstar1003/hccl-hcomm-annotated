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
#include "net_co_mux.h"
#include "net_co_main_data.h"
#include "wfw_mux_linux_epoll.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t NetCoMuxInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "epFd(%d)\n", co->argInit.epFd);
    BOOL argIsInvalid = (co->argInit.epFd < 0);
    if (argIsInvalid) {
        BKF_LOG_ERROR(BKF_LOG_HND, "argIsInvalid(%u), ng\n", argIsInvalid);
        return BKF_ERR;
    }

    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_muxLe", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    WfwMuxLeInitArg arg = {
        .name = name,
        .dbgOn = co->dbgOn,
        .memMng = co->memMng,
        .disp = co->disp,
        .logCnt = co->logCnt,
        .log = co->log,
        .pfm = co->pfm,
        .cookie = co,
        .onceProcEvCntMax = 0x10,
        .injectEpoll = VOS_TRUE,
        .injectEpollFd = co->argInit.epFd,
    };
    co->muxAdpee = WfwMuxLeInit(&arg);
    if (co->muxAdpee == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "muxAdpee(%#x), ng\n", BKF_MASK_ADDR(co->muxAdpee));
        return BKF_ERR;
    }
    WfwMuxInitArg temp;
    co->mux = WfwMuxInit(WfwMuxLeBuildMuxInitArg((WfwMuxLe *)co->muxAdpee, &temp));
    if (co->mux == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "mux(%#x), ng\n", BKF_MASK_ADDR(co->mux));
        return BKF_ERR;
    }

    return BKF_OK;
}

void NetCoMuxUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "mux(%#x)/muxAdpee(%#x)\n", BKF_MASK_ADDR(co->mux), BKF_MASK_ADDR(co->muxAdpee));

    if (co->mux != VOS_NULL) {
        WfwMuxUninit(co->mux);
        co->mux = VOS_NULL;
    }
    if (co->muxAdpee != VOS_NULL) {
        WfwMuxLeUninit((WfwMuxLe *)co->muxAdpee);
        co->muxAdpee = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
