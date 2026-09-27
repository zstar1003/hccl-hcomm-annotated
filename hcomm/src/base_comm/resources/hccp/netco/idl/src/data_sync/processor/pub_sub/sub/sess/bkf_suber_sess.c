/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_sess.h"
#include "bkf_suber_conn.h"
#include "bkf_suber_sess_data.h"
#include "bkf_suber_sess_fsm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
STATIC void BkfSuberSessTrigSub(BkfSuberSessMng *sessMng, BkfSuberSess *sess);
STATIC void BkfSuberSessTrigUnSub(BkfSuberSessMng *sessMng, BkfSuberSess *sess);
STATIC BkfSuberSess *BkfSuberSessGetSessByRcvData(BkfSuberSessMng *sessMng, BkfMsgDecoder *decoder,
    BkfMsgHead *msgHead);
STATIC uint32_t BkfSuberSessProcOneSessData(BkfSuberSess *sess, BkfMsgDecoder *decoder, BkfMsgHead *msgHead);
#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
BkfSuberSessMng *BkfSuberSessMngInit(BkfSuberSessMngInitArg *initArg)
{
    BkfSuberSessMng *sessMng = BkfSuberSessMngDataInit(initArg);
    if (sessMng == VOS_NULL) {
        return VOS_NULL;
    }

    uint32_t ret = BkfSuberSessFsmInitFsmTmp(sessMng);
    if (ret != BKF_OK) {
        BkfSuberSessMngUnInit(sessMng);
        return VOS_NULL;
    }
    sessMng->batchTimeoutChkFlag = VOS_FALSE;
    return sessMng;
}

void BkfSuberSessMngUnInit(BkfSuberSessMng *sessMng)
{
    BkfSuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;

    for (sess = BkfSuberSessDataGetFirst(sessMng, &itor); sess != VOS_NULL;
         sess = BkfSuberSessDataGetNext(sessMng, &itor)) {
        BkfSuberSessDel(sess);
        BkfSuberSessDataDel(sess);
    }
    BkfSuberSessFsmUnInitFsmTmp(sessMng);
    BkfSuberSessMngDataUnInit(sessMng);
}

BkfSuberSess *BkfSuberSessCreate(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId, BOOL isVerify)
{
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t locUrlStr[BKF_URL_STR_LEN_MAX];
    (void)BkfSuberGetSliceKeyStr(sessMng->env, sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(sessMng->log, "create sess slice %s, table id %u, pubUrl %s, locUrl %s, isverify %u\n", dispSliceStr,
        tableTypeId, BkfUrlGetStr(&sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sessMng->locUrl, locUrlStr, sizeof(locUrlStr)), isVerify);

    BkfSuberSess *sess = BkfSuberSessDataAdd(sessMng, sliceKey, tableTypeId);
    if (sess == VOS_NULL) {
        BKF_LOG_DEBUG(sessMng->log, "create sess slice %s, table id %u, isverify %u. create data fail\n", dispSliceStr,
            tableTypeId, isVerify);
        return VOS_NULL;
    }
    if (isVerify) {
        sess->isVerify = VOS_TRUE;
    }
    BkfSuberSessTrigSub(sessMng, sess);
    return sess;
}

/* 1.对账->去对账->发送unsub后切换为batok,sess标记不变 */
/* 2.any->去订阅->发送unsub后删除sess，sess标记强制为false */
/* 3.非对账->去对账->不处理 */

void BkfSuberSessDel(BkfSuberSess *sess)
{
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BkfSuberSessMng *sessMng = sess->sessMng;
    (void)BkfSuberGetSliceKeyStr(sessMng->env, sess->key.sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(sessMng->log, "del sess slice %s, table id %u, pubUrl %s, locUrl %s, isverify %u\n", dispSliceStr,
        sess->key.tableTypeId, BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)), sess->isVerify);
    sess->isVerify = VOS_FALSE;
    BkfSuberSessTrigUnSub(sessMng, sess);
}

void BkfSuberSessUnVerify(BkfSuberSess *sess)
{
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BkfSuberSessMng *sessMng = sess->sessMng;
    (void)BkfSuberGetSliceKeyStr(sessMng->env, sess->key.sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(sessMng->log, "sess unverify slice %s, table id %u, pubUrl %s locUrl %s, isverify %u\n", dispSliceStr,
        sess->key.tableTypeId, BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)), sess->isVerify);

    if (!sess->isVerify) {
        return;
    }
    BkfSuberSessTrigUnSub(sessMng, sess);
}

void BkfSuberSessDelByKey(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId)
{
    BkfSuberSess *sess = BkfSuberSessDataFind(sessMng, sliceKey, tableTypeId);
    if (sess == VOS_NULL) {
        BKF_LOG_ERROR(sessMng->log, "can't find sess data.\n");
        return;
    }
    BkfSuberSessDel(sess);
}

BkfSuberSess *BkfSuberSessFind(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId)
{
    return BkfSuberSessDataFind(sessMng, sliceKey, tableTypeId);
}

void BkfSuberSessReLoad(BkfSuberSess *sess, BOOL isVerify)
{
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    BkfSuberSessMng *sessMng = sess->sessMng;
    (void)BkfSuberGetSliceKeyStr(sessMng->env, sess->key.sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(sessMng->log, "sess reload slice %s, table id %u, isverify %u\n", dispSliceStr, sess->key.tableTypeId,
        isVerify);

    sess->isVerify = isVerify;
    BkfSuberSessTrigSub(sessMng, sess);
}

uint32_t BkfSuberSessProcDisconn(BkfSuberSessMng *sessMng)
{
    BkfSuberSess *sess = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret;

    for (sess = BkfSuberSessDataGetFirst(sessMng, &itor); sess != VOS_NULL;
         sess = BkfSuberSessDataGetNext(sessMng, &itor)) {
        ret = BkfSuberSessFsmInputEvt(sess, BACKFIN_SUB_SESS_INPUT_DISCONN, VOS_NULL, VOS_NULL);
        if (ret == BKF_SUBER_CONN_NEED_DELETE) {
            return ret;
        } else if (ret == BKF_SUBER_SESS_NEED_DELETE) {
            BkfSuberSessDataDel(sess);
        } else if (ret != BKF_OK) {
            BKF_ASSERT(0);
        }
    }
    return BKF_OK;
}

uint32_t BkfSuberSessProcSched(BkfSuberSessMng *sessMng, BkfMsgCoder *coder)
{
    BkfSuberSess *sess = VOS_NULL;
    BkfDlNode *temp;
    BkfDlNode *tempNext;
    BkfDl *list[3] = {&sessMng->subSess, &sessMng->unSubSess, &sessMng->verifySubSess};
    uint32_t i;
    uint32_t ret;

    for (i = 0; i < BKF_GET_ARY_COUNT(list); i++) {
        for (temp = BKF_DL_GET_FIRST(list[i]); temp != VOS_NULL; temp = tempNext) {
            tempNext = BKF_DL_GET_NEXT(list[i], temp);
            sess = BKF_DL_GET_ENTRY(temp, BkfSuberSess, dlStateNode);
            ret = BkfSuberSessFsmInputEvt(sess, BACKFIN_SUB_SESS_INPUT_SCHED, coder, VOS_NULL);
            if (ret == BKF_SUBER_SESS_NEED_DELETE) {
                BkfSuberSessDataDel(sess);
            } else if (ret == BKF_SUBER_SESS_SEND_BUF_NOT_ENOUGH) {
                return ret;
            }
        }
    }

    return BKF_OK;
}

uint32_t BkfSuberSessProcRcvData(BkfSuberSessMng *sessMng, BkfMsgDecoder *decoder, BkfMsgHead *msgHead)
{
    BkfSuberSess *sess = BkfSuberSessGetSessByRcvData(sessMng, decoder, msgHead);
    if (sess == VOS_NULL) {
        return BKF_ERR;
    }
    return BkfSuberSessProcOneSessData(sess, decoder, msgHead);
}

void BkfSuberSessSetDisconnectReason(BkfSuberSessMng *sessMng, uint32_t reason)
{
    if (sessMng == VOS_NULL) {
        return;
    }
    sessMng->disconnReason = reason;
    return;
}

uint32_t BkfSuberSessGetDisconnectReason(BkfSuberSessMng *sessMng)
{
    if (sessMng == VOS_NULL) {
        return BKF_SUBER_DISCONNECT_REASON_LOCALFIN;
    }
    return sessMng->disconnReason;
}

void BkfSuberSessResetDisconnectReason(BkfSuberSessMng *sessMng)
{
    if (sessMng == VOS_NULL) {
        return;
    }
    sessMng->disconnReason = BKF_SUBER_DISCONNECT_REASON_LOCALFIN;
}

void BkfSuberSessSetUnBlockFlag(BkfSuberSessMng *sessMng)
{
    if (sessMng == VOS_NULL) {
        return;
    }
    sessMng->deCongest = VOS_TRUE;
    return;
}

void BkfSuberSessResetUnBlockFlag(BkfSuberSessMng *sessMng)
{
    if (sessMng == VOS_NULL) {
        return;
    }
    sessMng->deCongest = VOS_FALSE;
}
#endif

#if BKF_BLOCK("私有函数定义")
STATIC void BkfSuberSessTrigSub(BkfSuberSessMng *sessMng, BkfSuberSess *sess)
{
    (void)sessMng;
    if (sess->isVerify) {
        BkfSuberSessFsmInputEvt(sess, BACKFIN_SUB_SESS_INPUT_VERIFYSUB, VOS_NULL, VOS_NULL);
    } else {
        BkfSuberSessFsmInputEvt(sess, BACKFIN_SUB_SESS_INPUT_SUB, VOS_NULL, VOS_NULL);
    }
}

STATIC void BkfSuberSessTrigUnSub(BkfSuberSessMng *sessMng, BkfSuberSess *sess)
{
    (void)sessMng;
    BkfSuberSessFsmInputEvt(sess, BACKFIN_SUB_SESS_INPUT_UNSUB, VOS_NULL, VOS_NULL);
}

STATIC BkfSuberSess *BkfSuberSessGetSessByRcvData(BkfSuberSessMng *sessMng, BkfMsgDecoder *decoder, BkfMsgHead *msgHead)
{
    (void)msgHead;
    BkfTL *tl = VOS_NULL;
    uint32_t errCode;
    BOOL tlvIsOk = VOS_FALSE;
    tl = BkfMsgDecodeTLV(decoder, &errCode);
    tlvIsOk = (tl != VOS_NULL) && (tl->typeId == BKF_TLV_SLICE_KEY);
    if (!tlvIsOk) {
        return VOS_NULL;
    }
    BkfTlvSliceKey *tlvSliceKey = VOS_NULL;
    tlvSliceKey = (BkfTlvSliceKey *)tl;

    tl = BkfMsgDecodeTLV(decoder, &errCode);
    tlvIsOk = (tl != VOS_NULL) && (tl->typeId == BKF_TLV_TABLE_TYPE);
    if (!tlvIsOk) {
        return VOS_NULL;
    }
    BkfTlvTableType *tlvTableType = VOS_NULL;
    tlvTableType = (BkfTlvTableType *)tl;

    return BkfSuberSessDataFind(sessMng, tlvSliceKey->sliceKey, tlvTableType->tableTypeId);
}

STATIC uint32_t BkfSuberSessProcOneSessData(BkfSuberSess *sess, BkfMsgDecoder *decoder, BkfMsgHead *msgHead)
{
    uint32_t inputFlag[BKF_MSG_MAX][2] = {

        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {BACKFIN_SUB_SESS_INPUT_SUBACK, BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK}, /* subACK */
        {0, 0},
        {BACKFIN_SUB_SESS_INPUT_BATCHBEGIN, BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN}, /* batchBegin */
        {BACKFIN_SUB_SESS_INPUT_BATCHEND, BACKFIN_SUB_SESS_INPUT_VERIFYEND},     /* batchEnd */
        {BACKFIN_SUB_SESS_INPUT_DATA, BACKFIN_SUB_SESS_INPUT_DATA},
        {BACKFIN_SUB_SESS_INPUT_NTF, BACKFIN_SUB_SESS_INPUT_NTF},
    };
    uint32_t tmpIdx = 0;
    BKF_RETURNVAL_IF(sess == VOS_NULL, BKF_ERR);
    BKF_RETURNVAL_IF(decoder == VOS_NULL, BKF_ERR);

    if (sess->isVerify) {
        tmpIdx = 1;
    }

    switch (msgHead->msgId) {
        case BKF_MSG_SUB_ACK:
        case BKF_MSG_BATCH_BEGIN:
        case BKF_MSG_BATCH_END:
        case BKF_MSG_DATA:
        case BKF_MSG_DEL_SUB_NTF:
            return BkfSuberSessFsmInputEvt(sess, inputFlag[msgHead->msgId][tmpIdx], decoder, VOS_NULL);
        default:
            BKF_LOG_ERROR(sess->sessMng->log, "Sess (%#x) msgId(%u, %s), ng\n", BKF_MASK_ADDR(sess), msgHead->msgId,
                BkfMsgGetIdStr(msgHead->msgId));
            return BKF_ERR;
    }
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif