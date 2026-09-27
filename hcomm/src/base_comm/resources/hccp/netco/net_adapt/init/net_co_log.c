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
#include "net_co_log.h"
#include "net_co_main_data.h"
#include "bkf_str.h"
#include "securec.h"
#include "net_slog.h"
#include "bkf_assert.h"

#ifdef __cplusplus
extern "C" {
#endif
void NetCoOutAssertInfo(const char *fmt, ...);

uint32_t NetCoLogInit(NetCo *co)
{
    BkfMemMng *memMng = co->memMng;
    uint8_t buf[0x200];
    char *path = NetCoLogGetLogFilePath(buf, sizeof(buf));
    if (path == VOS_NULL) {
        return BKF_ERR;
    }
    co->logFilePathName = BkfStrNew(memMng, "%s/%s_%#x_log.txt", path, co->name, co->selfCid);
    if (co->logFilePathName == VOS_NULL) {
        return BKF_ERR;
    }

    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_log", co->name);
    if (err <= 0) {
        return BKF_ERR;
    };
    BkfLogInitArg arg = {
        .name = name,
        .memMng = memMng,
        .disp = co->disp,
        .logCnt = co->logCnt,
        .cookie = co,
        .outputOrNull = (F_BKF_LOG_OUTPUT)NetCoLogOutput2File,
        .outputEnable = co->dbgOn,
        .logFuncName = co->dbgOn,
        .logTime = co->dbgOn,
        .moduleDefalutLvl = co->dbgOn ? BKF_LOG_LVL_DEBUG : BKF_LOG_LVL_WARN,
        .memBufLen = 0,
    };
    co->log = BkfLogInit(&arg);
    if (co->log == VOS_NULL) {
        return BKF_ERR;
    }
    BkfSetLogFunc((F_ASSERT_OUTFUNC)NetCoOutAssertInfo);
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "log(%#x), init ok\n", BKF_MASK_ADDR(co->log));
    return BKF_OK;
}

void NetCoLogUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "log(%#x)\n", BKF_MASK_ADDR(co->log));

    if (co->log != VOS_NULL) {
        BkfLogUninit(co->log);
        co->log = VOS_NULL;
    }
    if (co->logFilePathName != VOS_NULL) {
        BkfStrDel(co->logFilePathName);
        co->logFilePathName = VOS_NULL;
    }
}

char *NetCoLogGetLogFilePath(uint8_t *buf, int32_t bufLen)
{
    if ((buf == VOS_NULL) || (bufLen <= 0)) {
        return VOS_NULL;
    }

    int err = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, ".");
    return (err > 0) ? (char *)buf : VOS_NULL;
}

void NetCoLogOutput2File(NetCo *co, const char *outStr)
{
    if (co->dbgOn) {
        nslb_dbg("%s", outStr);
    } else {
        nslb_info("%s", outStr);
    }
}

void NetCoOutAssertInfo(const char *fmt, ...)
{
    char buf[BKF_1K];
    buf[0] = '\0';
    int32_t usedLen = 0;
    int32_t len = 0;

    va_list args;
    va_start(args, fmt);
    len = vsnprintf_truncated_s(&buf[usedLen], sizeof(buf) - usedLen, fmt, args);
    va_end(args);
    if (len < 0) {
        return;
    }
    usedLen += len;
    nslb_err("%s", (char *)buf);
    return;
}

#ifdef __cplusplus
}
#endif
