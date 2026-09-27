/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WFW_TWL_TIMER_BY_LINUX_H
#define WFW_TWL_TIMER_BY_LINUX_H

#include <stdint.h>
#include "wfw_mux.h"
#include "bkf_mem.h"
#include "bkf_tmr.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_log_cnt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BKF_TWL_TMR_NAME_LEN 16

/* 时间轮定时器初始化参数 */
typedef struct {
    char name[BKF_TWL_TMR_NAME_LEN]; /* （必选参数）名字标识 */
    BkfMemMng *memMng;               /* （必选参数）内存管理库句柄 */
    unsigned char dbgSwitch;         /* （必选参数）是否开启诊断 */
    unsigned char pad[3];            /* （填充参数）无需设置 */
    WfwMux *mux;                     /* （必选参数）多路复用句柄 */
    BkfDisp *disp;                   /* （必选参数）诊断显示库句柄 */
    BkfLogCnt *logCnt;               /* （必选参数）logCnt库句柄 */
    BkfLog *log;                     /* （必选参数）日志库句柄 */

    void *appHandle; /* （必选参数）app handle */
    uint32_t appCid; /* （可选参数）组件cid，如果无效填0 */
} BkfTwlTmrInitArg;

typedef void *BkfTwlTmrHandle;

/* 初始化BKF时间轮定时器，返回时间轮定时器句柄 */
BkfTwlTmrHandle BkfAdptTwlTimerInit(BkfTwlTmrInitArg *initArg);
/* 初始化BKF定时器管理模块，返回BKF定时器管理模块句柄 */
BkfTmrMng *BkfAdptTwlTimerInitTmrMng(BkfTwlTmrHandle twlWfwAdapt);

/* 析构BKF时间轮定时器 */
void BkfAdptTwlTimerUnInit(BkfTwlTmrHandle twlWfwAdapt);
/* 析构BKF定时器管理模块 */
void BkfAdptBkfTmrMngUnInit(BkfTmrMng *bkfTmrMng);

#ifdef __cplusplus
}
#endif

#endif
