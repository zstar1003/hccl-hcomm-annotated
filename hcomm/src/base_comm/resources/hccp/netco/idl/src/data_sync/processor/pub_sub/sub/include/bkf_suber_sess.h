/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_SUB_SESS_H
#define BKF_SUBER_SUB_SESS_H

#include "bkf_suber_env.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_suber_data.h"
#include "bkf_msg_codec.h"
#include "bkf_disp.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

#define BKF_SUBER_SESS_SEND_BUF_NOT_ENOUGH (300)
#define BKF_SUBER_SESS_FATAL_ERR (301)
#define BKF_SUBER_SESS_NEED_DELETE (302)

typedef void (*F_BKF_SUBER_SESS_TRIG_SCHED_SELF)(void *cookieInit);
typedef struct tagBkfSuberSessMngInitArg {
    BkfSuberEnv *env;
    void *cookie;
    F_BKF_SUBER_SESS_TRIG_SCHED_SELF trigSchedSelf;
    BkfSuberDataMng *dataMng;
    BkfUrl pubUrl;
    BkfUrl locUrl;
} BkfSuberSessMngInitArg;
typedef struct tagBkfSuberSessMng BkfSuberSessMng;
typedef struct tagBkfSuberSess BkfSuberSess;

BkfSuberSessMng *BkfSuberSessMngInit(BkfSuberSessMngInitArg *initArg);
void BkfSuberSessMngUnInit(BkfSuberSessMng *sessMng);

BkfSuberSess *BkfSuberSessCreate(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId, BOOL isVerify);
void BkfSuberSessDel(BkfSuberSess *sess);
void BkfSuberSessUnVerify(BkfSuberSess *sess);
void BkfSuberSessDelByKey(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId);
BkfSuberSess *BkfSuberSessFind(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId);
void BkfSuberSessReLoad(BkfSuberSess *sess, BOOL isVerify);
uint32_t BkfSuberSessProcDisconn(BkfSuberSessMng *sessMng);
uint32_t BkfSuberSessProcSched(BkfSuberSessMng *sessMng, BkfMsgCoder *coder);
uint32_t BkfSuberSessProcRcvData(BkfSuberSessMng *sessMng, BkfMsgDecoder *decoder, BkfMsgHead *msgHead);

uint32_t BkfSuberSessDispOneSessSummary(BkfSuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastTempCtxOrNull,
    BkfDispTempCtx *curTempCtx);
uint32_t BkfSuberSessDispOneSessFsm(BkfSuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastTempCtxOrNull,
    BkfDispTempCtx *curTempCtx);

uint32_t BkfSuberSessDispBatchTimeoutTest(BkfSuberSessMng *sessMng, BkfDisp *disp);
uint32_t BkfSuberSessDispCloseBatchTimeout(BkfSuberSessMng *sessMng, BkfDisp *disp);
void BkfSuberSessSetDisconnectReason(BkfSuberSessMng *sessMng, uint32_t reason);
uint32_t BkfSuberSessGetDisconnectReason(BkfSuberSessMng *sessMng);
void BkfSuberSessResetDisconnectReason(BkfSuberSessMng *sessMng);
void BkfSuberSessSetUnBlockFlag(BkfSuberSessMng *sessMng);
void BkfSuberSessResetUnBlockFlag(BkfSuberSessMng *sessMng);
#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif