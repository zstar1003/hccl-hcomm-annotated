/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_SUB_SESS_FSM_H
#define BKF_SUBER_SUB_SESS_FSM_H

#include "bkf_suber_sess.h"
#include "bkf_suber_sess_data.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

enum {
    BACKFIN_SUB_SESS_STATE_DOWN,
    BACKFIN_SUB_SESS_STATE_WAITSUBACK,
    BACKFIN_SUB_SESS_STATE_UP,
    BACKFIN_SUB_SESS_STATE_SNDUNSUB,
    BACKFIN_SUB_SESS_STATE_BATCH,
    BACKFIN_SUB_SESS_STATE_BATOK,
    BACKFIN_SUB_SESS_STATE_VERIFYWAITACK, /* wait for peer verify hello */
    BACKFIN_SUB_SESS_STATE_VERIFYRDY,     /* ready for verify */
    BACKFIN_SUB_SESS_STATE_VERIFY,        /* verify doing */
    BACKFIN_SUB_SESS_STATE_MAX
};

enum {
    BACKFIN_SUB_SESS_INPUT_SUB,
    BACKFIN_SUB_SESS_INPUT_SUBACK,
    BACKFIN_SUB_SESS_INPUT_UNSUB,
    BACKFIN_SUB_SESS_INPUT_BATCHBEGIN,
    BACKFIN_SUB_SESS_INPUT_BATCHEND,
    BACKFIN_SUB_SESS_INPUT_DATA,
    BACKFIN_SUB_SESS_INPUT_NTF,
    BACKFIN_SUB_SESS_INPUT_DISCONN,
    BACKFIN_SUB_SESS_INPUT_SCHED,
    BACKFIN_SUB_SESS_INPUT_VERIFYSUB,
    BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK,
    BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN,
    BACKFIN_SUB_SESS_INPUT_VERIFYEND,
    BACKFIN_SUB_SESS_INPUT_MAX
};
uint32_t BkfSuberSessFsmInputEvt(BkfSuberSess *sess, uint32_t event, void *param1OrNull, void *param2OrNull);
uint32_t BkfSuberSessFsmInitFsmTmp(BkfSuberSessMng *sessMng);
void BkfSuberSessFsmUnInitFsmTmp(BkfSuberSessMng *sessMng);

void BkfSuberSessBatchChkTmrStop(BkfSuberSess *sess);
void BkfSuberSessNtfyAppBatchTimeout(BkfSuberSess *sess);
#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif