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
#include "net_co_sys_log.h"
#include "net_co_main_data.h"
#include "securec.h"
#include "net_slog.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NET_CO_SYS_LOG_RESTRAIN_INTERVAL 3000
STATIC uint32_t NetCoSysLogPrintf(uint32_t sysLogParam1, uint32_t infoId, const char *fmt, ...)
{
    /*
    1. format
    2. 输出 sys log
    3. spy输出
    */
    uint32_t bufLen = BKF_1K;
    char buf[BKF_1K];
    char *temp = buf;
    uint32_t leftLen = sizeof(buf);
    int err = snprintf_truncated_s(temp, leftLen, "sysLogParam1(%u)/infoId(%u) ", sysLogParam1, infoId);
    if (err < 0 || (uint32_t)err > leftLen) {
        return BKF_ERR;
    }

    temp += err;
    leftLen -= (uint32_t)err;

    va_list va;
    va_start(va, fmt);
    err = vsnprintf_truncated_s(temp, leftLen, fmt, va);
    va_end(va);
    if (err < 0) {
        return BKF_ERR;
    }
    nslb_info("%s", buf);
    NetCoSysLogOutStrSpy(buf, bufLen);
    return BKF_OK;
}

uint32_t NetCoSysLogInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_sysLog", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfSysLogInitArg arg = {
        .name = name,
        .dbgOn = co->dbgOn,
        .memMng = co->memMng,
        .disp = co->disp,
        .log = co->log,
        .tmrMng = co->tmrMng,
        .printf = (F_BKF_SYS_LOG_PRINTF)NetCoSysLogPrintf, /* 函数原型不好，没有cookie，先容忍 */
        .printfParam1 = co->selfCid,
        .restrainIntervalMs = NET_CO_SYS_LOG_RESTRAIN_INTERVAL,
    };
    co->sysLogMng = BkfSysLogInit(&arg);
    if (co->sysLogMng == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "sysLogMng(%#x), ng\n", BKF_MASK_ADDR(co->sysLogMng));
        return BKF_ERR;
    }

    return BKF_OK;
}

void NetCoSysLogUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "sysLogMng(%#x)\n", BKF_MASK_ADDR(co->sysLogMng));

    if (co->sysLogMng != VOS_NULL) {
        BkfSysLogUninit(co->sysLogMng);
        co->sysLogMng = VOS_NULL;
    }
}

void NetCoSysLogOutStrSpy(char *outStr, uint32_t bufLen)
{
    (void)outStr;
    (void)bufLen;
}

#ifdef __cplusplus
}
#endif
