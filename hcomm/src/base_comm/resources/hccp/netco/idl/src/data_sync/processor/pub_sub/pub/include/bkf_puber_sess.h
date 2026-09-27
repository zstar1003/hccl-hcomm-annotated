/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_SESS_H
#define BKF_PUBER_SESS_H

#include "bkf_puber_adef.h"
#include "bkf_msg.h"
#include "bkf_msg_codec.h"
#include "bkf_puber_table_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
typedef struct TagBkfPuberSessMng BkfPuberSessMng;

typedef void (*F_BKF_PUBER_SESS_TRIG_SCHED_SELF)(void *cookieInit);
typedef void (*F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF)(void *cookieInit);

#define BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH 200
#define BKF_PUBER_SESS_SCHED_FINISH 201
#define BKF_PUBER_SESS_FATAL_ERR 202
#define BKF_PUBER_SESS_SCHED_YIELD 203

typedef struct {
    int32_t procCntMax;
    int32_t procCnt;
} BkfPuberSessSlowSchedCtx;
#pragma pack()

BkfPuberSessMng *BkfPuberSessInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng,
    F_BKF_PUBER_SESS_TRIG_SCHED_SELF trigSchedSess, F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF trigSlowSchedSess,
    void *cookie);
void BkfPuberSessUninit(BkfPuberSessMng *sessMng);
uint32_t BkfPuberSessProcRcvData(BkfPuberSessMng *sessMng, BkfMsgHead *msgHead, BkfMsgDecoder *decoder);
uint32_t BkfPuberSessProcSched(BkfPuberSessMng *sessMng, BkfMsgCoder *coder, BOOL isSlowSched, void *ctx);
void BkfPuberSessDispGetSummary(BkfPuberSessMng *sessMng, BkfDispTempCtx *ctx);
void BkfPuberSessDispPrintfSummary(BkfDisp *disp, BkfDispTempCtx *ctx);
int32_t BkfPuberSessDispSess(BkfPuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastCtx, BkfDispTempCtx *curCtx);
int32_t BkfPuberSessDispSessFsm(BkfPuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastCtx,
    BkfDispTempCtx *curCtx);

uint32_t BkfPuberSessDispCloseBatchTimeout(BkfPuberSessMng *sessMng, BkfDisp *disp);
uint32_t BkfPuberSessDispBatchTimeoutTest(BkfPuberSessMng *sessMng, BkfDisp *disp);
#ifdef __cplusplus
}
#endif

#endif
