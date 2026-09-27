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
#define BKF_LINE (__LINE__ + 15000)

#include "bkf_puber_sess_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

STATIC uint32_t BkfPuberSessChgState(BkfPuberSess *sess, uint8_t newState)
{
    uint32_t ret = BKF_FSM_CHG_STATE(&sess->fsm, newState);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return ret;
    }

    if (BKF_DL_NODE_IS_IN(&sess->dlStateNode)) {
        BKF_DL_REMOVE(&sess->dlStateNode);
    }

    BkfPuberSessMng *sessMng = sess->sessMng;
    BKF_DL_ADD_LAST(&sessMng->sessSetByState[newState], &sess->dlStateNode);
    return BKF_OK;
}

void BkfPuberSessChgStateAndTrigSched(BkfPuberSess *sess, uint8_t newState)
{
    uint32_t ret = BkfPuberSessChgState(sess, newState);
    if (ret == BKF_OK) {
        BkfPuberSessMng *sessMng = sess->sessMng;
        sessMng->trigSchedSelf(sessMng->cookieInit);
    }
}

void BkfPuberSessChgStateAndTrigSlowSched(BkfPuberSess *sess, uint8_t newState)
{
    uint32_t ret = BkfPuberSessChgState(sess, newState);
    if (ret == BKF_OK) {
        BkfPuberSessMng *sessMng = sess->sessMng;
        sessMng->trigSlowSchedSelf(sessMng->cookieInit);
    }
}

void BkfPuberSessNtfAppSub(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId)
{
    BkfPuberTableTypeVTbl *vTbl = BkfPuberTableTypeGetVTbl(sessMng->tableTypeMng, tableTypeId);
    if (vTbl == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }

    if (vTbl->tableOnSub != VOS_NULL) {
        vTbl->tableOnSub(vTbl->cookie, sliceKey, VOS_NULL, 0);
    }
}

void BkfPuberSessNtfAppUnsub(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId)
{
    BkfPuberTableTypeVTbl *vTbl = BkfPuberTableTypeGetVTbl(sessMng->tableTypeMng, tableTypeId);
    if (vTbl == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }

    if (vTbl->tableOnUnsub != VOS_NULL) {
        vTbl->tableOnUnsub(vTbl->cookie, sliceKey);
    }
}

int32_t BkfPuberSessNtfAppCode(BkfPuberSess *sess, BkfDcTupleInfo *tupleInfo, void *buf, int32_t bufLen)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BkfPuberTableTypeVTbl *vTbl = BkfPuberTableTypeGetVTbl(sessMng->tableTypeMng, sess->key.tableTypeId);
    if (vTbl == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    if (tupleInfo->isAddUpd) {
        return vTbl->tupleUpdateCode(vTbl->cookie, sess->key.sliceKey, tupleInfo->key, tupleInfo->valOrNull, buf,
            bufLen);
    }

    return vTbl->tupleDeleteCode(vTbl->cookie, sess->key.sliceKey, tupleInfo->key, buf, bufLen);
}

BkfTlvTransNum *BkfPuberSessParseTransNum(BkfMsgDecoder *decoder)
{
    uint32_t errCode;
    BkfTL *tl = BkfMsgDecodeTLV(decoder, &errCode);
    if ((tl == VOS_NULL) || (tl->typeId != BKF_TLV_TRANS_NUM)) {
        return VOS_NULL;
    }

    BkfTlvTransNum *tlvTransNum = (BkfTlvTransNum *)tl;
    if (tlvTransNum->num == BKF_TRANS_NUM_INVALID) {
        return VOS_NULL;
    }

    return tlvTransNum;
}

uint32_t BkfPuberSessPackSubAck(BkfPuberSess *sess, BkfMsgCoder *coder)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    BkfTlvReasonCode reasonCode = {.reasonCode = BKF_RC_OK};
    BkfTlvTableType tableType = {.tableTypeId = sess->key.tableTypeId};
    BkfTlvTransNum transNum = {.num = sess->subTransNum};

    uint8_t flag = 0;
    if (sess->subWithVerify) {
        BKF_BIT_SET(flag, BKF_FLAG_VERIFY);
    }

    uint32_t ret = BkfMsgCodeMsgHead(coder, BKF_MSG_SUB_ACK, flag);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_SLICE_KEY, 0, sess->key.sliceKey, sliceVTbl->keyLen, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TABLE_TYPE, 0, &tableType.tl + 1, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TRANS_NUM, 0, &transNum.tl + 1, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_REASON_CODE, 0, &reasonCode.tl + 1, VOS_TRUE);
    return ret;
}

STATIC uint32_t BkfPuberSessPackBatchBeginEnd(BkfPuberSess *sess, BkfMsgCoder *coder, uint8_t msgId)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    BkfTlvTableType tableType = {.tableTypeId = sess->key.tableTypeId};
    BkfTlvTransNum transNum = {.num = sess->subTransNum};

    uint8_t flag = 0;
    if (sess->subWithVerify) {
        BKF_BIT_SET(flag, BKF_FLAG_VERIFY);
    }

    uint32_t ret = BkfMsgCodeMsgHead(coder, msgId, flag);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_SLICE_KEY, 0, sess->key.sliceKey, sliceVTbl->keyLen, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TABLE_TYPE, 0, &tableType.tl + 1, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TRANS_NUM, 0, &transNum.tl + 1, VOS_TRUE);
    return ret;
}

uint32_t BkfPuberSessPackBatchBegin(BkfPuberSess *sess, BkfMsgCoder *coder)
{
    return BkfPuberSessPackBatchBeginEnd(sess, coder, BKF_MSG_BATCH_BEGIN);
}

uint32_t BkfPuberSessPackBatchEnd(BkfPuberSess *sess, BkfMsgCoder *coder)
{
    return BkfPuberSessPackBatchBeginEnd(sess, coder, BKF_MSG_BATCH_END);
}

uint32_t BkfPuberSessPackSubDelNtf(BkfPuberSess *sess, BkfMsgCoder *coder)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    BkfTlvTableType tableType = {.tableTypeId = sess->key.tableTypeId};
    BkfTlvTransNum transNum = {.num = sess->subTransNum};

    uint32_t ret = BkfMsgCodeMsgHead(coder, BKF_MSG_DEL_SUB_NTF, 0);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_SLICE_KEY, 0, sess->key.sliceKey, sliceVTbl->keyLen, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TABLE_TYPE, 0, &tableType.tl + 1, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TRANS_NUM, 0, &transNum.tl + 1, VOS_TRUE);
    return ret;
}

uint32_t BkfPuberSessPackDataHead(BkfPuberSess *sess, BkfMsgCoder *coder)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    BkfTlvTableType tableType = {.tableTypeId = sess->key.tableTypeId};
    BkfTlvTransNum transNum = {.num = sess->subTransNum};

    uint32_t ret = BkfMsgCodeMsgHead(coder, BKF_MSG_DATA, 0);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_SLICE_KEY, 0, sess->key.sliceKey, sliceVTbl->keyLen, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TABLE_TYPE, 0, &tableType.tl + 1, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TRANS_NUM, 0, &transNum.tl + 1, VOS_TRUE);
    return ret;
}

void BkfPuberSessNtfAppBatchTimeout(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId)
{
    BkfPuberTableTypeVTbl *vTbl = BkfPuberTableTypeGetVTbl(sessMng->tableTypeMng, tableTypeId);
    if (vTbl == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }

    if (vTbl->tableOnBatchTimeout != VOS_NULL && vTbl->batchTimeout != 0) {
        vTbl->tableOnBatchTimeout(vTbl->cookie, sliceKey);
    }
}

#ifdef __cplusplus
}
#endif
