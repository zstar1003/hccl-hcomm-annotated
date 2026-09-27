/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SYS_LOG_H
#define BKF_SYS_LOG_H

#include "bkf_bas_type_mthd.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_tmr.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
typedef struct tagBkfSysLogMng BkfSysLogMng;

/* init */
typedef uint32_t (*F_BKF_SYS_LOG_PRINTF)(uint32_t sysLogParam1, uint32_t infoId, const char *fmt, ...);

typedef uint32_t (*F_BKF_SYS_LOG_OUT)(void *cookieReg, void *key, void *val, F_BKF_SYS_LOG_PRINTF logPrintf,
    uint32_t logPrintfParam1);

typedef struct tagBkfSysLogInitArg {
    char *name;
    BOOL dbgOn;
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLog *log;
    BkfTmrMng *tmrMng;
    F_BKF_SYS_LOG_PRINTF printf;
    uint32_t printfParam1;
    uint32_t restrainIntervalMs;
    uint8_t rsv1[0x10];
} BkfSysLogInitArg;

typedef struct tagBkfSysLogTypeVTbl {
    char *name;
    void *cookie;

    uint32_t typeId;
    BOOL needRestrain;
    uint16_t keyLen;
    uint16_t valLen;
    F_BKF_CMP keyCmp;
    F_BKF_GET_STR keyGetStrOrNull;
    F_BKF_GET_STR valGetStrOrNull;
    F_BKF_SYS_LOG_OUT out;
    uint8_t rsv1[4];
} BkfSysLogTypeVTbl;

/* func */
BkfSysLogMng *BkfSysLogInit(BkfSysLogInitArg *arg);
void BkfSysLogUninit(BkfSysLogMng *sysLogMng);
uint32_t BkfSysLogReg(BkfSysLogMng *sysLogMng, BkfSysLogTypeVTbl *vTbl);
uint32_t BkfSysLogFunc(BkfSysLogMng *sysLogMng, uint32_t typeId, void *key, void *val);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
