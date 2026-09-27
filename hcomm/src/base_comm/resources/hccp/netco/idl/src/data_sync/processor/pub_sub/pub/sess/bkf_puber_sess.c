/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((sessMng)->log)
#define BKF_MOD_NAME ((sessMng)->name)

#include "bkf_puber_sess.h"
#include "bkf_puber_sess_data.h"
#include "bkf_puber_sess_fsm.h"
#include "bkf_puber_sess_utils.h"
#include "bkf_dc.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
/* on msg */
STATIC void BkfPuberSessOnTupleChg(BkfPuberSess *sess);
STATIC void BkfPuberSessOnTableRel(BkfPuberSess *sess);
STATIC void BkfPuberSessOnTableCptl(BkfPuberSess *sess);

/* proc */
STATIC void BkfPuberSessProcRcvDataParseLeftTlv(BkfMsgDecoder *decoder, BkfTlvSliceKey **tlvSliceKey,
    BkfTlvTableType **tlvTableType);
STATIC BOOL BkfPuberSessProcRcvDataMayProc(BkfPuberSessMng *sessMng, BkfTlvSliceKey *tlvSliceKey,
    BkfTlvTableType *tlvTableType);
STATIC uint32_t BkfPuberSessProcRcvDataDo(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfMsgDecoder *decoder);
STATIC uint32_t BkfPuberSessSchedOneSess(BkfPuberSess *sess, BkfMsgCoder *coder, BOOL isSlowSched, void *ctx,
    uint8_t state);

#endif

#if BKF_BLOCK("公有函数定义")
BkfPuberSessMng *BkfPuberSessInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng,
    F_BKF_PUBER_SESS_TRIG_SCHED_SELF trigSchedSess, F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF trigSlowSchedSess,
    void *cookie)
{
    BkfPuberSessMng *sessMng = BkfPuberSessDataInit(arg, tableTypeMng, trigSchedSess, trigSlowSchedSess, cookie);
    if (sessMng == VOS_NULL) {
        return VOS_NULL;
    }

    sessMng->sessFsmTmpl = BkfPuberSessFsmTmplInit(sessMng);
    if (sessMng->sessFsmTmpl == VOS_NULL) {
        BkfPuberSessUninit(sessMng);
        return VOS_NULL;
    }

    sessMng->batchTimeoutChkFlag = VOS_FALSE;
    BKF_LOG_INFO(BKF_LOG_HND, "sessMng(%#x, %s), init ok\n", BKF_MASK_ADDR(sessMng), arg->name);
    return sessMng;
}

void BkfPuberSessUninit(BkfPuberSessMng *sessMng)
{
    if (sessMng == VOS_NULL) {
        return;
    }

    BKF_LOG_INFO(BKF_LOG_HND, "sessMng(%#x, %s), uninit\n", BKF_MASK_ADDR(sessMng), sessMng->argInit->name);
    BkfFsmTmpl *tmpl = sessMng->sessFsmTmpl;
    BkfPuberSessDataUninit(sessMng);
    BkfFsmTmplUninit(tmpl);
}

uint32_t BkfPuberSessProcRcvData(BkfPuberSessMng *sessMng, BkfMsgHead *msgHead, BkfMsgDecoder *decoder)
{
    BkfTlvSliceKey *tlvSliceKey = VOS_NULL;
    BkfTlvTableType *tlvTableType = VOS_NULL;
    BkfPuberSessProcRcvDataParseLeftTlv(decoder, &tlvSliceKey, &tlvTableType);
    BOOL rcvDataIsOk = (tlvSliceKey != VOS_NULL) && (tlvTableType != VOS_NULL);
    if (!rcvDataIsOk) {
        BKF_LOG_WARN(BKF_LOG_HND, "rcvDataIsOk(%u), ng\n", rcvDataIsOk);
        return BKF_ERR;
    }

    BOOL needNtfApp = (msgHead->msgId == BKF_MSG_SUB) && !BKF_BIT_TEST(msgHead->flag, BKF_FLAG_VERIFY);
    if (needNtfApp) {
        BkfPuberSessNtfAppSub(sessMng, tlvSliceKey->sliceKey, tlvTableType->tableTypeId);
    }

    BOOL rcvDataMayProc = BkfPuberSessProcRcvDataMayProc(sessMng, tlvSliceKey, tlvTableType);
    if (!rcvDataMayProc) {
        BKF_LOG_ERROR(BKF_LOG_HND, "rcvDataMayProc(%u), ng\n", rcvDataMayProc);
        return BKF_ERR;
    }

    BkfPuberSess *sess = BkfPuberSessFind(sessMng, tlvSliceKey->sliceKey, tlvTableType->tableTypeId);
    if (sess == VOS_NULL) {
        BkfDcTupleItorVTbl vTbl = {.afterTupleChangeOrNull = (F_BKF_DC_NTF_BY_ITOR)BkfPuberSessOnTupleChg,
            .afterTableReleaseOrNull = (F_BKF_DC_NTF_BY_ITOR)BkfPuberSessOnTableRel,
            .afterTableCompleteOrNull = (F_BKF_DC_NTF_BY_ITOR)BkfPuberSessOnTableCptl};
        sess = BkfPuberSessAdd(sessMng, tlvSliceKey->sliceKey, tlvTableType->tableTypeId, &vTbl);
        if (sess == VOS_NULL) {
            BkfPuberSessNtfAppUnsub(sessMng, tlvSliceKey->sliceKey, tlvTableType->tableTypeId);
            return BKF_PUBER_SESS_FATAL_ERR;
        }
    }

    uint32_t ret = BkfPuberSessProcRcvDataDo(sess, msgHead, decoder);
    BKF_LOG_DEBUG(BKF_LOG_HND, "ret(%u)\n", ret);
    if (ret == BKF_PUBER_SESS_NEED_DELETE) {
        BkfPuberSessDel(sess);
        BkfPuberSessNtfAppUnsub(sessMng, tlvSliceKey->sliceKey, tlvTableType->tableTypeId);
        return BKF_ERR;
    }

    needNtfApp = (msgHead->msgId == BKF_MSG_UNSUB) && !BKF_BIT_TEST(msgHead->flag, BKF_FLAG_VERIFY);
    if (needNtfApp) {
        BkfPuberSessNtfAppUnsub(sessMng, tlvSliceKey->sliceKey, tlvTableType->tableTypeId);
    }
    return ret;
}

static inline BOOL BkfPuberSessStateMaySched(uint8_t state)
{
    return (state != BKF_PUBER_SESS_STATE_WAIT_SUB) && (state != BKF_PUBER_SESS_STATE_WAIT_MAY_SEND_BATCH_BEGIN) &&
           (state != BKF_PUBER_SESS_STATE_IDLE) && (state != BKF_PUBER_SESS_STATE_SLOW_BATCH_DATA);
}

static inline BOOL BkfPuberSessStateMaySlowSched(uint8_t state)
{
    return (state == BKF_PUBER_SESS_STATE_SLOW_BATCH_DATA);
}

uint32_t BkfPuberSessProcSched(BkfPuberSessMng *sessMng, BkfMsgCoder *coder, BOOL isSlowSched, void *ctx)
{
    uint8_t state;
    for (state = 0; state < BKF_GET_ARY_COUNT(sessMng->sessSetByState); state++) {
        /* 链表即sess状态。下面是不需要调度的状态链表，提高效率 */
        BOOL maySched = isSlowSched ? BkfPuberSessStateMaySlowSched(state) : BkfPuberSessStateMaySched(state);
        if (!maySched) {
            continue;
        }

        BkfPuberSess *sess = BkfPuberSessGetFirstByState(sessMng, state, VOS_NULL);
        if (sess != VOS_NULL) {
            return BkfPuberSessSchedOneSess(sess, coder, isSlowSched, ctx, state);
        }
    }

    return BKF_PUBER_SESS_SCHED_FINISH;
}

#endif

#if BKF_BLOCK("私有函数定义")
STATIC void BkfPuberSessOnTupleChg(BkfPuberSess *sess)
{
    (void)BkfPuberSessFsmProc(sess, BKF_PUBER_SESS_EVT_TUPLE_CHG, VOS_NULL, VOS_NULL);
}

STATIC void BkfPuberSessOnTableRel(BkfPuberSess *sess)
{
    (void)BkfPuberSessFsmProc(sess, BKF_PUBER_SESS_EVT_TABLE_REL, VOS_NULL, VOS_NULL);
}

STATIC void BkfPuberSessOnTableCptl(BkfPuberSess *sess)
{
    (void)BkfPuberSessFsmProc(sess, BKF_PUBER_SESS_EVT_TABLE_CPLT, VOS_NULL, VOS_NULL);
}

STATIC void BkfPuberSessProcRcvDataParseLeftTlv(BkfMsgDecoder *decoder, BkfTlvSliceKey **tlvSliceKey,
    BkfTlvTableType **tlvTableType)
{
    uint32_t errCode = BKF_OK;
    BkfTL *tl = BkfMsgDecodeTLV(decoder, &errCode);
    BOOL tlvIsOk = (tl != VOS_NULL) && (tl->typeId == BKF_TLV_SLICE_KEY);
    if (!tlvIsOk) {
        return;
    }

    *tlvSliceKey = (BkfTlvSliceKey *)tl;
    tl = BkfMsgDecodeTLV(decoder, &errCode);
    tlvIsOk = (tl != VOS_NULL) && (tl->typeId == BKF_TLV_TABLE_TYPE);
    if (!tlvIsOk) {
        return;
    }

    *tlvTableType = (BkfTlvTableType *)tl;
}

STATIC BOOL BkfPuberSessProcRcvDataMayProc(BkfPuberSessMng *sessMng, BkfTlvSliceKey *tlvSliceKey,
    BkfTlvTableType *tlvTableType)
{
    BOOL isTableRelease = VOS_FALSE;
    BOOL isTableExist = BkfDcIsTableExist(sessMng->argInit->dc, tlvSliceKey->sliceKey, tlvTableType->tableTypeId,
        &isTableRelease);
    return (isTableExist && !isTableRelease);
}

STATIC uint32_t BkfPuberSessProcRcvDataDo(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfMsgDecoder *decoder)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    switch (msgHead->msgId) {
        case BKF_MSG_SUB: {
            return BkfPuberSessFsmProc(sess, BKF_PUBER_SESS_EVT_SUB, decoder, VOS_NULL);
        }
        case BKF_MSG_UNSUB: {
            return BkfPuberSessFsmProc(sess, BKF_PUBER_SESS_EVT_UNSUB, decoder, VOS_NULL);
        }
        default: {
            BKF_LOG_WARN(BKF_LOG_HND, "msgId(%u, %s), ng\n", msgHead->msgId, BkfMsgGetIdStr(msgHead->msgId));
            return BKF_ERR;
        }
    }
}

STATIC uint32_t BkfPuberSessSchedOneSess(BkfPuberSess *sess, BkfMsgCoder *coder, BOOL isSlowSched, void *ctx,
    uint8_t state)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    uint8_t evtId = isSlowSched ? BKF_PUBER_SESS_EVT_SLOW_SCHED : BKF_PUBER_SESS_EVT_SCHED;
    uint32_t ret = BkfPuberSessFsmProc(sess, evtId, coder, ctx);

    if (BkfPuberSessGetFirstByState(sessMng, state, VOS_NULL) == sess) { /* 当前会话还没有处理完, 轮转 */
        (void)BkfPuberSessMoveFirst2LastByState(sessMng, state);
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "ret(%u)\n", ret);
    if (ret == BKF_PUBER_SESS_SCHED_FINISH) {
        BKF_ASSERT(0); /* 一个sess不可能决策finish，防止错误 */
        BkfPuberSessDel(sess);
        return BKF_ERR;
    }

    if (ret == BKF_PUBER_SESS_NEED_DELETE) {
        BkfPuberSessDel(sess);
        return BKF_ERR;
    }

    return ret;
}

#endif

#ifdef __cplusplus
}
#endif
