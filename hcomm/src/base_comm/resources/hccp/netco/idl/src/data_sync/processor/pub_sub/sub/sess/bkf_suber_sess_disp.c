/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_sess_fsm.h"
#include "bkf_suber_sess_data.h"
#include "bkf_suber_sess.h"
#include "bkf_disp.h"
#include "securec.h"
#include "vos_base.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
typedef struct tagBkfSuberSessDispSessCtx {
    int32_t sessCnt;
    uint16_t tableTypeId;
    uint8_t pad1[2];
    uint8_t sliceKey[BKF_SUBER_SLICE_KEY_LEN_MAX];
} BkfSuberSessDispSessCtx;
STATIC BkfSuberSess *BkfSuberSessDispGetSesByCtx(BkfSuberSessMng *sessMng, BkfDispTempCtx *lastTempCtxOrNull,
    BkfDispTempCtx *curTempCtx, BkfSuberSessDispSessCtx **curCtx);
STATIC uint32_t BkfSuberSessDispSaveCtx(BkfSuberSessDispSessCtx *curCtx, BkfSuberSess *sess);
#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
uint32_t BkfSuberSessDispOneSessFsm(BkfSuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastTempCtxOrNull,
    BkfDispTempCtx *curTempCtx)
{
    BkfSuberSessDispSessCtx *curCtx;
    BkfSuberSess *sess = BkfSuberSessDispGetSesByCtx(sessMng, lastTempCtxOrNull, curTempCtx, &curCtx);
    if (sess != VOS_NULL) {
        uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
        uint8_t fsmStr[BKF_1K / 2] = {0};
        curCtx->sessCnt++;
        BKF_DISP_PRINTF(disp, "==========\n");
        BKF_DISP_PRINTF(disp, "++sess[%d], slice(%s)/tableTypeId(%u), fsm\n%s\n", curCtx->sessCnt,
            BkfSuberGetSliceKeyStr(sessMng->env, sess->key.sliceKey, dispSliceStr, sizeof(dispSliceStr)),
            sess->key.tableTypeId, BkfFsmGetStr(&sess->fsm, fsmStr, sizeof(fsmStr)));
        return BkfSuberSessDispSaveCtx(curCtx, sess);
    } else {
        BKF_DISP_PRINTF(disp, "Sess Cnt: %u\n", curCtx->sessCnt);
        BKF_DISP_PRINTF(disp, "==========\n");
        return 0;
    }
}

uint32_t BkfSuberSessDispOneSessSummary(BkfSuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastTempCtxOrNull,
    BkfDispTempCtx *curTempCtx)
{
    BkfSuberSessDispSessCtx *curCtx;
    BkfSuberSess *sess = BkfSuberSessDispGetSesByCtx(sessMng, lastTempCtxOrNull, curTempCtx, &curCtx);
    if (sess != VOS_NULL) {
        uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
        curCtx->sessCnt++;
        BKF_DISP_PRINTF(disp, "==========\n");
        BKF_DISP_PRINTF(disp,
            "++sess[%d], slice(%s)/tableTypeId(%u), state(%u, %s), "
            "seq(%" VOS_PRIu64 ")/verify(%u)\n",
            curCtx->sessCnt,
            BkfSuberGetSliceKeyStr(sessMng->env, sess->key.sliceKey, dispSliceStr, sizeof(dispSliceStr)),
            sess->key.tableTypeId, BKF_FSM_GET_STATE(&sess->fsm), BkfFsmGetStateStr(&sess->fsm), sess->seq,
            sess->isVerify);
        return BkfSuberSessDispSaveCtx(curCtx, sess);
    } else {
        BKF_DISP_PRINTF(disp, "Sess Cnt: %u\n", curCtx->sessCnt);
        BKF_DISP_PRINTF(disp, "==========\n");
        return 0;
    }
}
#endif

#if BKF_BLOCK("私有函数定义")
STATIC BkfSuberSess *BkfSuberSessDispGetSesByCtx(BkfSuberSessMng *sessMng, BkfDispTempCtx *lastTempCtxOrNull,
    BkfDispTempCtx *curTempCtx, BkfSuberSessDispSessCtx **curCtx)
{
    BkfSuberSessDispSessCtx *lastCtx = BKF_DISP_TMEP_CTX_2X(lastTempCtxOrNull, BkfSuberSessDispSessCtx);
    *curCtx = BKF_DISP_TMEP_CTX_2X(curTempCtx, BkfSuberSessDispSessCtx);
    BkfSuberSess *sess;
    if (lastCtx == VOS_NULL) {
        sess = BkfSuberSessDataGetFirst(sessMng, VOS_NULL);
    } else {
        **curCtx = *lastCtx;
        sess = BkfSuberSessDataFindNext(sessMng, (*curCtx)->sliceKey, (*curCtx)->tableTypeId);
    }
    return sess;
}

STATIC uint32_t BkfSuberSessDispSaveCtx(BkfSuberSessDispSessCtx *curCtx, BkfSuberSess *sess)
{
    curCtx->tableTypeId = sess->key.tableTypeId;
    int32_t err = 0;
    BkfSuberSessMng *sessMng = sess->sessMng;
    err = memcpy_s(curCtx->sliceKey, sizeof(curCtx->sliceKey), sess->key.sliceKey, sessMng->env->sliceVTbl.keyLen);
    BKF_ASSERT(err == EOK);
    return sizeof(*curCtx) - sizeof(curCtx->sliceKey) + sessMng->env->sliceVTbl.keyLen;
}

uint32_t BkfSuberSessDispBatchTimeoutTest(BkfSuberSessMng *sessMng, BkfDisp *disp)
{
    (void)disp;
    BkfSuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t testCnt = 0;
    for (sess = BkfSuberSessDataGetFirst(sessMng, &itor); sess != VOS_NULL;
         sess = BkfSuberSessDataGetNext(sessMng, &itor)) {
        BkfSuberSessNtfyAppBatchTimeout(sess);
        testCnt++;
    }
    return testCnt;
}

uint32_t BkfSuberSessDispCloseBatchTimeout(BkfSuberSessMng *sessMng, BkfDisp *disp)
{
    (void)disp;
    BkfSuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;
    sessMng->batchTimeoutChkFlag = VOS_TRUE;
    uint32_t closeCnt = 0;
    for (sess = BkfSuberSessDataGetFirst(sessMng, &itor); sess != VOS_NULL;
         sess = BkfSuberSessDataGetNext(sessMng, &itor)) {
        BkfSuberSessBatchChkTmrStop(sess);
        closeCnt++;
    }
    return closeCnt;
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif