/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "vos_base.h"
#include "bkf_puber_sess.h"
#include "bkf_puber_sess_data.h"
#include "bkf_puber_sess_utils.h"
#include "bkf_puber_sess_fsm.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
typedef struct {
    BkfFsmTmpl *fsmTmpl;
    int32_t totalCnt;
    int32_t eachStateCnt[BKF_PUBER_SESS_STATE_CNT];
    int32_t invalidStateCnt;
} BkfPuberSessDispCtx;
BKF_STATIC_ASSERT(sizeof(BkfPuberSessDispCtx) <= BKF_DISP_TEMP_CTX_LEN_MAX);

typedef struct {
    int32_t sessCnt;
    uint16_t tableTypeId;
    uint8_t pad1[2];
    uint8_t sliceKey[BKF_DC_SLICE_KEY_LEN_MAX];
} BkfPuberDispSessCtx;
#pragma pack()

STATIC void BkfPuberSessDispGetSummaryCtx(BkfPuberSessMng *sessMng, BkfPuberSessDispCtx *ctx)
{
    BkfPuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;
    ctx->fsmTmpl = sessMng->sessFsmTmpl;

    for (sess = BkfPuberSessGetFirst(sessMng, &itor); sess != VOS_NULL; sess = BkfPuberSessGetNext(sessMng, &itor)) {
        ctx->totalCnt++;

        uint8_t state = BkfFsmGetState(&sess->fsm);
        if (state < BKF_GET_ARY_COUNT(ctx->eachStateCnt)) {
            ctx->eachStateCnt[state]++;
        } else {
            ctx->invalidStateCnt++;
        }
    }
}

void BkfPuberSessDispGetSummary(BkfPuberSessMng *sessMng, BkfDispTempCtx *ctx)
{
    BkfPuberSessDispGetSummaryCtx(sessMng, BKF_DISP_TMEP_CTX_2X(ctx, BkfPuberSessDispCtx));
}

void BkfPuberSessDispPrintfSummary(BkfDisp *disp, BkfDispTempCtx *ctx)
{
    BkfPuberSessDispCtx *sessctx = BKF_DISP_TMEP_CTX_2X(ctx, BkfPuberSessDispCtx);
    BKF_DISP_PRINTF(disp, "----sess----\n");
    BKF_DISP_PRINTF(disp, "sessTotalCnt         : %d\n", sessctx->totalCnt);

    uint8_t state;
    for (state = 0; state < BKF_GET_ARY_COUNT(sessctx->eachStateCnt); state++) {
        BKF_DISP_PRINTF(disp, "sessCnt[%-3d, %-50s]      : %d\n", state, BkfFsmTmplGetStateStr(sessctx->fsmTmpl, state),
            sessctx->eachStateCnt[state]);
    }
    BKF_DISP_PRINTF(disp, "sessCnt[%-3d, %-50s]      : %d\n", BKF_FSM_STATE_INVALID, "state_invalid",
        sessctx->invalidStateCnt);
}

STATIC BkfPuberSess *BkfPuberSessDispSessGetSessAndCtx(BkfPuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastCtx,
    BkfDispTempCtx *curCtx, BkfPuberDispSessCtx **curSessCtx)
{
    BkfPuberDispSessCtx *tempLastSessCtx = BKF_DISP_TMEP_CTX_2X(lastCtx, BkfPuberDispSessCtx);
    BkfPuberDispSessCtx *tempCurSessCtx = BKF_DISP_TMEP_CTX_2X(curCtx, BkfPuberDispSessCtx);
    BkfPuberSess *sess = VOS_NULL;
    if (tempLastSessCtx == VOS_NULL) {
        BkfPuberSessDispCtx cnt = {0};
        BkfPuberSessDispGetSummaryCtx(sessMng, &cnt);
        BKF_DISP_PRINTF(disp, "totalSessCnt(%d)\n", cnt.totalCnt);
        BKF_DISP_PRINTF(disp, "--------\n");
        sess = BkfPuberSessGetFirst(sessMng, VOS_NULL);
    } else {
        *tempCurSessCtx = *tempLastSessCtx;
        sess = BkfPuberSessFindNext(sessMng, tempCurSessCtx->sliceKey, tempCurSessCtx->tableTypeId);
    }

    *curSessCtx = tempCurSessCtx;
    return sess;
}

STATIC int32_t BkfPuberSessDispOneSess(BkfPuberSessMng *sessMng, BkfDisp *disp, BkfPuberSess *sess,
    BkfPuberDispSessCtx *curSessCtx)
{
    uint8_t buf[BKF_LOG_LEN] = {0};
    BKF_DISP_PRINTF(disp,
        "++sess[%d], slice(%s)/tableTypeId(%u, %s), state(%u, %s), "
        "subTransNum(%" VOS_PRIu64 ")/verify(%u)/needTblCplt(%u), "
        "itorB_R(%#x, %#x)\n",
        curSessCtx->sessCnt, BkfDcGetSliceKeyStr(sessMng->argInit->dc, sess->key.sliceKey, buf, sizeof(buf)),
        sess->key.tableTypeId, BkfDcGetTableTypeIdStr(sessMng->argInit->dc, sess->key.tableTypeId),
        BKF_FSM_GET_STATE(&sess->fsm), BkfFsmGetStateStr(&sess->fsm), sess->subTransNum, sess->subWithVerify,
        sess->subWithNeedTblCplt, BKF_MASK_ADDR(sess->itorBatchData), BKF_MASK_ADDR(sess->itorRealData));

    curSessCtx->sessCnt++;
    curSessCtx->tableTypeId = sess->key.tableTypeId;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    int32_t err = memcpy_s(curSessCtx->sliceKey, sizeof(curSessCtx->sliceKey), sess->key.sliceKey, sliceVTbl->keyLen);
    if (err != EOK) {
        return 0;
    }
    BkfBlackBoxDispOneInstLog(sessMng->argInit->log, BKF_BLACKBOX_TYPE_SESS, (void *)sess, 0);
    return sizeof(*curSessCtx) - sizeof(curSessCtx->sliceKey) + sliceVTbl->keyLen;
}

int32_t BkfPuberSessDispSess(BkfPuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastCtx, BkfDispTempCtx *curCtx)
{
    BkfPuberDispSessCtx *curSessCtx = VOS_NULL;
    BkfPuberSess *sess = BkfPuberSessDispSessGetSessAndCtx(sessMng, disp, lastCtx, curCtx, &curSessCtx);
    if (sess == VOS_NULL) {
        return 0;
    }

    return BkfPuberSessDispOneSess(sessMng, disp, sess, curSessCtx);
}

STATIC int32_t BkfPuberSessDispOneSessFsm(BkfPuberSessMng *sessMng, BkfDisp *disp, BkfPuberSess *sess,
    BkfPuberDispSessCtx *curSessCtx)
{
    uint8_t buf[BKF_DISP_PRINTF_BUF_LEN_MAX];
    BKF_DISP_PRINTF(disp, "++sess[%d], key_slice(%s)/tableTypeId(%u, %s), fsmInfo:\n", curSessCtx->sessCnt,
        BkfDcGetSliceKeyStr(sessMng->argInit->dc, sess->key.sliceKey, buf, sizeof(buf)), sess->key.tableTypeId,
        BkfDcGetTableTypeIdStr(sessMng->argInit->dc, sess->key.tableTypeId));
    BKF_DISP_PRINTF(disp, "%s\n", BkfFsmGetStr(&sess->fsm, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "--------\n");

    curSessCtx->sessCnt++;
    curSessCtx->tableTypeId = sess->key.tableTypeId;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    int32_t err = memcpy_s(curSessCtx->sliceKey, sizeof(curSessCtx->sliceKey), sess->key.sliceKey, sliceVTbl->keyLen);
    if (err != EOK) {
        return 0;
    }

    return sizeof(*curSessCtx) - sizeof(curSessCtx->sliceKey) + sliceVTbl->keyLen;
}

int32_t BkfPuberSessDispSessFsm(BkfPuberSessMng *sessMng, BkfDisp *disp, BkfDispTempCtx *lastCtx,
    BkfDispTempCtx *curCtx)
{
    BkfPuberDispSessCtx *curSessCtx = VOS_NULL;
    BkfPuberSess *sess = BkfPuberSessDispSessGetSessAndCtx(sessMng, disp, lastCtx, curCtx, &curSessCtx);

    if (sess == VOS_NULL) {
        return 0;
    }

    return BkfPuberSessDispOneSessFsm(sessMng, disp, sess, curSessCtx);
}

uint32_t BkfPuberSessDispCloseBatchTimeout(BkfPuberSessMng *sessMng, BkfDisp *disp)
{
    BkfPuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t closeCnt = 0;
    sessMng->batchTimeoutChkFlag = VOS_TRUE;
    for (sess = BkfPuberSessGetFirst(sessMng, &itor); sess != VOS_NULL; sess = BkfPuberSessGetNext(sessMng, &itor)) {
        closeCnt++;
        BkfPuberSessBatchChkTmrStop(sess);
    }

    BKF_DISP_PRINTF(disp, "close batch timeout count[%u]", closeCnt);
    return closeCnt;
}

uint32_t BkfPuberSessDispBatchTimeoutTest(BkfPuberSessMng *sessMng, BkfDisp *disp)
{
    BkfPuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t testCnt = 0;
    for (sess = BkfPuberSessGetFirst(sessMng, &itor); sess != VOS_NULL; sess = BkfPuberSessGetNext(sessMng, &itor)) {
        testCnt++;
        BkfPuberSessNtfAppBatchTimeout(sessMng, sess->key.sliceKey, sess->key.tableTypeId);
    }

    BKF_DISP_PRINTF(disp, "batch timeout test count[%u]", testCnt);
    return testCnt;
}

#ifdef __cplusplus
}
#endif
