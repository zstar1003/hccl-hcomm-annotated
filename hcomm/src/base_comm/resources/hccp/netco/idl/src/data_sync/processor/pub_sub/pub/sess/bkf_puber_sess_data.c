/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_puber_sess_data.h"
#include "bkf_str.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

STATIC int32_t BkfPuberSessKeyCmp(BkfPuberSessKey *key1Input, BkfPuberSessKey *key2InDs)
{
    if (key1Input->tableTypeId > key2InDs->tableTypeId) {
        return 1;
    }
    if (key1Input->tableTypeId < key2InDs->tableTypeId) {
        return -1;
    }

    if (key1Input->sliceKeyCmp > key2InDs->sliceKeyCmp) {
        return 1;
    }
    if (key1Input->sliceKeyCmp < key2InDs->sliceKeyCmp) {
        return -1;
    }

    return key1Input->sliceKeyCmp(key1Input->sliceKey, key2InDs->sliceKey);
}

BkfPuberSessMng *BkfPuberSessDataInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng,
    F_BKF_PUBER_SESS_TRIG_SCHED_SELF trigSchedSess, F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF trigSlowSchedSess,
    void *cookie)
{
    uint32_t len = sizeof(BkfPuberSessMng);
    BkfPuberSessMng *sessMng = BKF_MALLOC(arg->memMng, len);
    if (sessMng == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(sessMng, len, 0, len);
    sessMng->argInit = arg;
    sessMng->name = BkfStrNew(arg->memMng, "%s_sess", arg->name);
    if (sessMng->name == VOS_NULL) {
        BKF_FREE(arg->memMng, sessMng);
        return VOS_NULL;
    }
    sessMng->log = arg->log;

    sessMng->tableTypeMng = tableTypeMng;
    sessMng->trigSchedSelf = trigSchedSess;
    sessMng->trigSlowSchedSelf = trigSlowSchedSess;
    sessMng->cookieInit = cookie;
    VOS_AVLL_INIT_TREE(sessMng->sessSet, (AVLL_COMPARE)BkfPuberSessKeyCmp, BKF_OFFSET(BkfPuberSess, key),
        BKF_OFFSET(BkfPuberSess, avlNode));

    uint32_t i;
    for (i = 0; i < BKF_GET_ARY_COUNT(sessMng->sessSetByState); i++) {
        BKF_DL_INIT(&sessMng->sessSetByState[i]);
    }

    return sessMng;
}

void BkfPuberSessDataUninit(BkfPuberSessMng *sessMng)
{
    BkfPuberSessDelAll(sessMng);
    BkfStrDel(sessMng->name);
    BKF_FREE(sessMng->argInit->memMng, sessMng);
}

STATIC void BkfPuberSessInitItor(BkfPuberSess *sess, void *sliceKey, uint16_t tableTypeId, BkfDcTupleItorVTbl *vTbl)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BkfDcTupleItorVTbl temp = *vTbl;
    temp.name = "sessSeqItor";
    temp.cookie = sess;
    sess->itorBatchData = BkfDcNewTupleKeyItor(sessMng->argInit->dc, sliceKey, tableTypeId);
    sess->itorRealData = BkfDcNewTupleSeqItor(sessMng->argInit->dc, sliceKey, tableTypeId, &temp);
}

STATIC void BkfPuberSessDelByFlag(BkfPuberSess *sess, BOOL fsmInitOk, BOOL avlInsOk)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    if (sess->itorBatchData != VOS_NULL) {
        BkfDcDeleteTupleKeyItor(sessMng->argInit->dc, sess->itorBatchData);
    }

    if (sess->itorRealData != VOS_NULL) {
        BkfDcDeleteTupleSeqItor(sessMng->argInit->dc, sess->itorRealData);
    }

    if (fsmInitOk) {
        BkfFsmUninit(&sess->fsm);
    }

    if (BKF_DL_NODE_IS_IN(&sess->dlStateNode)) {
        BKF_DL_REMOVE(&sess->dlStateNode);
    }

    if (avlInsOk) {
        VOS_AVLL_DELETE(sessMng->sessSet, sess->avlNode);
    }
    BKF_BLACKBOX_DELLOGINST(sessMng->argInit->log, BKF_BLACKBOX_TYPE_SESS, (void *)sess, 0);
    BKF_FREE(sessMng->argInit->memMng, sess);
}

BkfPuberSess *BkfPuberSessAdd(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId,
    BkfDcTupleItorVTbl *vTbl)
{
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    uint32_t len = sizeof(BkfPuberSess) + sliceVTbl->keyLen;
    BkfPuberSess *sess = BKF_MALLOC(sessMng->argInit->memMng, len);
    if (sess == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(sess, len, 0, len);
    sess->sessMng = sessMng;
    BKF_DL_NODE_INIT(&sess->dlStateNode);
    BKF_DL_ADD_LAST(&sessMng->sessSetByState[BKF_PUBER_SESS_STATE_WAIT_SUB], &sess->dlStateNode);
    sess->subTransNum = BKF_TRANS_NUM_INVALID;

    BOOL fsmInitOk = VOS_FALSE;
    BOOL avlInsOk = VOS_FALSE;
    uint32_t ret = BkfFsmInit(&sess->fsm, sessMng->sessFsmTmpl);
    if (ret != BKF_OK) {
        BkfPuberSessDelByFlag(sess, fsmInitOk, avlInsOk);
        return VOS_NULL;
    }

    fsmInitOk = VOS_TRUE;
    sess->key.tableTypeId = tableTypeId;
    sess->key.sliceKeyCmp = sliceVTbl->keyCmp;
    sess->key.sliceKey = sess->sliceKey;
    (void)memcpy_s(sess->key.sliceKey, sliceVTbl->keyLen, sliceKey, sliceVTbl->keyLen);
    BkfPuberSessInitItor(sess, sliceKey, tableTypeId, vTbl);
    if ((sess->itorBatchData == VOS_NULL) || (sess->itorRealData == VOS_NULL)) {
        BkfPuberSessDelByFlag(sess, fsmInitOk, avlInsOk);
        return VOS_NULL;
    }

    VOS_AVLL_INIT_NODE(sess->avlNode);
    avlInsOk = VOS_AVLL_INSERT(sessMng->sessSet, sess->avlNode);
    if (!avlInsOk) {
        BkfPuberSessDelByFlag(sess, fsmInitOk, avlInsOk);
        return VOS_NULL;
    }

    return sess;
}

void BkfPuberSessDel(BkfPuberSess *sess)
{
    BkfPuberSessDelByFlag(sess, VOS_TRUE, VOS_TRUE);
}

void BkfPuberSessDelAll(BkfPuberSessMng *sessMng)
{
    BkfPuberSess *sess = VOS_NULL;
    while ((sess = VOS_AVLL_FIRST(sessMng->sessSet)) != VOS_NULL) {
        BkfPuberSessDel(sess);
    }
}

BkfPuberSess *BkfPuberSessFind(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId)
{
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    BkfPuberSessKey sessKey = {.tableTypeId = tableTypeId, .sliceKeyCmp = sliceVTbl->keyCmp, .sliceKey = sliceKey};
    return VOS_AVLL_FIND(sessMng->sessSet, &sessKey);
}

BkfPuberSess *BkfPuberSessFindNext(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId)
{
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(sessMng->argInit->dc);
    BkfPuberSessKey sessKey = {.tableTypeId = tableTypeId, .sliceKeyCmp = sliceVTbl->keyCmp, .sliceKey = sliceKey};
    return VOS_AVLL_FIND_NEXT(sessMng->sessSet, &sessKey);
}

BkfPuberSess *BkfPuberSessGetFirst(BkfPuberSessMng *sessMng, void **itorOutOrNull)
{
    BkfPuberSess *sess = VOS_AVLL_FIRST(sessMng->sessSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (sess != VOS_NULL) ? VOS_AVLL_NEXT(sessMng->sessSet, sess->avlNode) : VOS_NULL;
    }
    return sess;
}

BkfPuberSess *BkfPuberSessGetNext(BkfPuberSessMng *sessMng, void **itorInOut)
{
    BkfPuberSess *sess = (*itorInOut);
    *itorInOut = (sess != VOS_NULL) ? VOS_AVLL_NEXT(sessMng->sessSet, sess->avlNode) : VOS_NULL;
    return sess;
}

BkfPuberSess *BkfPuberSessGetFirstByState(BkfPuberSessMng *sessMng, uint8_t sessState, void **itorOutOrNull)
{
    if (!BkfPuberSessStateIsValid(sessState)) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }

    BkfDl *dl = &sessMng->sessSetByState[sessState];
    BkfDlNode *tempNode = BKF_DL_GET_FIRST(dl);
    if (tempNode != VOS_NULL) {
        BkfPuberSess *sess = BKF_DL_GET_ENTRY(tempNode, BkfPuberSess, dlStateNode);
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = BKF_DL_GET_NEXT(dl, &sess->dlStateNode);
        }
        return sess;
    }

    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }
    return VOS_NULL;
}

uint32_t BkfPuberSessMoveFirst2LastByState(BkfPuberSessMng *sessMng, uint8_t sessState)
{
    if (!BkfPuberSessStateIsValid(sessState)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    BkfDl *dl = &sessMng->sessSetByState[sessState];
    BkfDlNode *tempNode = BKF_DL_GET_FIRST(dl);
    if (tempNode != VOS_NULL) {
        BKF_DL_REMOVE(tempNode);
        BKF_DL_ADD_LAST(dl, tempNode);
    }
    return BKF_OK;
}

#ifdef __cplusplus
}
#endif
