/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_sess_data.h"
#include "bkf_suber_sess.h"
#include "bkf_mem.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

STATIC int32_t BkfSuberSessAndSubKeyCmp(BkfSuberSessKey *key1Input, BkfSuberSessKey *key2InDs)
{
    if (key1Input->tableTypeId > key2InDs->tableTypeId) {
        return 1;
    } else if (key1Input->tableTypeId < key2InDs->tableTypeId) {
        return -1;
    } else if (key1Input->sliceKeyCmp > key2InDs->sliceKeyCmp) {
        return 1;
    } else if (key1Input->sliceKeyCmp < key2InDs->sliceKeyCmp) {
        return -1;
    } else {
        return key1Input->sliceKeyCmp(key1Input->sliceKey, key2InDs->sliceKey);
    }
}

#if BKF_BLOCK("公有函数定义")
BkfSuberSessMng *BkfSuberSessMngDataInit(BkfSuberSessMngInitArg *initArg)
{
    uint32_t len = sizeof(BkfSuberSessMng);
    BkfSuberSessMng *sessMng = BKF_MALLOC(initArg->env->memMng, len);
    if (sessMng == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(sessMng, len, 0, len);
    if (initArg->env->dbgOn) {
        sessMng->log = initArg->env->log;
    }
    sessMng->env = initArg->env;
    sessMng->tirgSchedSelf = initArg->trigSchedSelf;
    sessMng->cookie = initArg->cookie;
    sessMng->pubUrl = initArg->pubUrl;
    sessMng->locUrl = initArg->locUrl;
    VOS_AVLL_INIT_TREE(sessMng->sessSet, (AVLL_COMPARE)BkfSuberSessAndSubKeyCmp, BKF_OFFSET(BkfSuberSess, key),
        BKF_OFFSET(BkfSuberSess, avlNode));
    sessMng->dataMng = initArg->dataMng;
    BKF_DL_INIT(&sessMng->subSess);
    BKF_DL_INIT(&sessMng->unSubSess);
    BKF_DL_INIT(&sessMng->verifySubSess);
    return sessMng;
}

void BkfSuberSessMngDataUnInit(BkfSuberSessMng *sessMng)
{
    BkfSuberSessDataDelAll(sessMng);
    BKF_FREE(sessMng->env->memMng, sessMng);
}

BkfSuberSess *BkfSuberSessDataAdd(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId)
{
    BkfSuberSess *sess = VOS_NULL;

    sess = BkfSuberSessDataFind(sessMng, sliceKey, tableTypeId);
    if (sess != VOS_NULL) {
        return sess;
    }

    uint32_t len = sizeof(BkfSuberSess) + sessMng->env->sliceVTbl.keyLen;
    sess = BKF_MALLOC(sessMng->env->memMng, len);
    if (sess == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(sess, len, 0, len);
    if (memcpy_s(sess->sliceKey, sessMng->env->sliceVTbl.keyLen, sliceKey, sessMng->env->sliceVTbl.keyLen) != 0) {
        goto error;
    }
    uint32_t ret = BkfFsmInit(&sess->fsm, sessMng->sessFsmTmpl);
    if (ret != BKF_OK) {
        goto error;
    }
    sess->key.tableTypeId = tableTypeId;
    sess->key.sliceKeyCmp = sessMng->env->sliceVTbl.keyCmp;
    sess->key.sliceKey = sess->sliceKey;
    sess->sessMng = sessMng;
    VOS_AVLL_INIT_NODE(sess->avlNode);
    BOOL isInsOk = VOS_AVLL_INSERT(sessMng->sessSet, sess->avlNode);
    if (!isInsOk) {
        goto error;
    }
    BKF_DL_NODE_INIT(&sess->dlStateNode);
    BKF_DL_ADD_LAST(&sessMng->subSess, &sess->dlStateNode);

    return sess;

error:
    BKF_FREE(sessMng->env->memMng, sess);
    return VOS_NULL;
}

void BkfSuberSessDataDel(BkfSuberSess *sess)
{
    if (BKF_DL_NODE_IS_IN(&sess->dlStateNode)) {
        BKF_DL_REMOVE(&sess->dlStateNode);
    }
    if (VOS_AVLL_IN_TREE(sess->avlNode)) {
        VOS_AVLL_DELETE(sess->sessMng->sessSet, sess->avlNode);
    }
    BkfFsmUninit(&sess->fsm);
    BKF_FREE(sess->sessMng->env->memMng, sess);
}

void BkfSuberSessDataDelByKey(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId)
{
    BkfSuberSess *sess = BkfSuberSessDataFind(sessMng, sliceKey, tableTypeId);
    if (sess == VOS_NULL) {
        return;
    }
    BkfSuberSessDataDel(sess);
}

BkfSuberSess *BkfSuberSessDataFind(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId)
{
    BkfSuberSessKey key = {
        .tableTypeId = tableTypeId, .sliceKey = sliceKey, .sliceKeyCmp = sessMng->env->sliceVTbl.keyCmp};
    return VOS_AVLL_FIND(sessMng->sessSet, &key);
}

BkfSuberSess *BkfSuberSessDataFindNext(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId)
{
    BkfSuberSessKey key = {
        .tableTypeId = tableTypeId, .sliceKey = sliceKey, .sliceKeyCmp = sessMng->env->sliceVTbl.keyCmp};
    return VOS_AVLL_FIND_NEXT(sessMng->sessSet, &key);
}

BkfSuberSess *BkfSuberSessDataGetFirst(BkfSuberSessMng *sessMng, void **itorOutOrNull)
{
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }
    BkfSuberSess *sess = VOS_AVLL_FIRST(sessMng->sessSet);
    if (sess != VOS_NULL && itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_AVLL_NEXT(sessMng->sessSet, sess->avlNode);
    }
    return sess;
}

BkfSuberSess *BkfSuberSessDataGetNext(BkfSuberSessMng *sessMng, void **itorInOut)
{
    BkfSuberSess *sess = *(BkfSuberSess **)itorInOut;
    if (sess == VOS_NULL) {
        *itorInOut = VOS_NULL;
    } else {
        *itorInOut = VOS_AVLL_NEXT(sessMng->sessSet, sess->avlNode);
    }
    return sess;
}

void BkfSuberSessDataDelAll(BkfSuberSessMng *sessMng)
{
    BkfSuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;

    for (sess = BkfSuberSessDataGetFirst(sessMng, &itor); sess != VOS_NULL;
         sess = BkfSuberSessDataGetNext(sessMng, &itor)) {
        BkfSuberSessDataDel(sess);
    }
}

uint32_t BkfSuberSessCreateReSubTmr(BkfSuberSessMng *sessMng)
{
    /* 创建握手定时器 */
    if (sessMng->reSubTmrId != VOS_NULL) {
        return BKF_OK;
    }
    sessMng->reSubTmrId = BkfTmrStartLoop(sessMng->env->tmrMng, (F_BKF_TMR_TIMEOUT_PROC)BkfSubSessReSubTmrSvcProc,
        BKF_SUBER_SESS_RESUB_TIMER_INTERVAL, sessMng);
    if (sessMng->reSubTmrId == VOS_NULL) {
        BKF_LOG_ERROR(sessMng->log, "Sess Create Timer failed.\n");
        return BKF_ERR;
    }
    return BKF_OK;
}

void BkfSuberSessDeleteReSubTmr(BkfSuberSessMng *sessMng)
{
    if (sessMng->reSubTmrId != VOS_NULL) {
        BkfTmrStop(sessMng->env->tmrMng, sessMng->reSubTmrId);
        sessMng->reSubTmrId = VOS_NULL;
    }
    return;
}

uint32_t BkfSubSessReSubTmrSvcProc(void *paramTmrStart, void *noUse)
{
    (void)noUse;
    BkfSuberSessMng *sessMng = (BkfSuberSessMng *)paramTmrStart;
    BkfSuberSessResetUnBlockFlag(sessMng);
    if (BKF_DL_IS_EMPTY(&sessMng->subSess)) {
        BKF_LOG_DEBUG(sessMng->log, "Sess Timer Proc no need proc, timer delete.\n"); /* NEED DELETE */
        BkfSuberSessDeleteReSubTmr(sessMng);
        return BKF_OK;
    }

    sessMng->tirgSchedSelf(sessMng->cookie);
    return BKF_OK;
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif