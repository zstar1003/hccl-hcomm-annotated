/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((ch)->log)
#define BKF_MOD_NAME ((ch)->name)
#define BKF_LOG_CNT_HND ((ch)->argInit.base->logCnt)

#include <unistd.h>
#include <fcntl.h>
#include "bkf_str.h"
#include "bkf_bas_type_mthd.h"
#include "securec.h"
#include "bkf_ch_ser_letcp_data.h"

#ifdef __cplusplus
extern "C" {
#endif
/* func */
BkfChSer *BkfChSerLetcpDataInit(BkfChSerInitArg *arg)
{
    uint32_t len = sizeof(BkfChSer);
    BkfChSer *ch = BKF_MALLOC(arg->base->memMng, len);
    if (ch == VOS_NULL) {
        goto error;
    }
    (void)memset_s(ch, len, 0, len);
    ch->argInit = *arg;
    if (arg->base->dbgOn) {
        ch->log = arg->base->log;
    }
    VOS_AVLL_INIT_TREE(ch->lsnSet, (AVLL_COMPARE)BkfUrlCmp, BKF_OFFSET(BkfChSerLetcpLsn, keyUrlSelf),
        BKF_OFFSET(BkfChSerLetcpLsn, avlNode));
    ch->name = BkfStrNew(arg->base->memMng, "ser_letcp_%s", arg->name);
    if (ch->name == VOS_NULL) {
        goto error;
    }

    BKF_SIGN_SET(ch->sign, BKF_CH_SER_LETCP_SIGN);
    return ch;

error:

    if (ch != VOS_NULL) {
        /* ch->name 一定为空 */
        BKF_FREE(arg->base->memMng, ch);
    }
    return VOS_NULL;
}

void BkfChSerLetcpDataUninit(BkfChSer *ch)
{
    BkfChSerLetcpDelAllLsn(ch);
    BkfStrDel(ch->name);
    BKF_SIGN_CLR(ch->sign);
    BKF_FREE(ch->argInit.base->memMng, ch);
    return;
}

BkfChSerLetcpLsn *BkfChSerLetcpAddLsn(BkfChSer *ch, BkfUrl *urlSelf)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;
    uint32_t len;
    BOOL insOk = VOS_FALSE;

    len = sizeof(BkfChSerLetcpLsn);
    lsn = BKF_MALLOC(ch->argInit.base->memMng, len);
    if (lsn == VOS_NULL) {
        goto error;
    }
    (void)memset_s(lsn, len, 0, len);
    lsn->ch = ch;
    lsn->keyUrlSelf = *urlSelf;
    lsn->lsnFd = -1;
    VOS_AVLL_INIT_TREE(lsn->connIdSet, (AVLL_COMPARE)BkfInt32Cmp, BKF_OFFSET(BkfChSerConnId, keyConnFd),
        BKF_OFFSET(BkfChSerConnId, avlNode));
    VOS_AVLL_INIT_NODE(lsn->avlNode);
    insOk = VOS_AVLL_INSERT(ch->lsnSet, lsn->avlNode);
    if (!insOk) {
        goto error;
    }

    BKF_SIGN_SET(lsn->sign, BKF_CH_SER_LETCP_LSN_SIGN);
    ch->lsnCache = lsn;
    return lsn;
error:

    if (lsn != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(ch->argInit.base->memMng, lsn);
    }
    return VOS_NULL;
}

void BkfChSerLetcpDelLsn(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    BkfChSerLetcpDelAllConnId(ch, lsn);
    if (lsn->lsnFd != -1) {
        if (lsn->lsnFdAttachOk) {
            (void)WfwMuxDetachFd(ch->argInit.base->mux, lsn->lsnFd);
            lsn->lsnFdAttachOk = VOS_FALSE;
        }
        (void)close(lsn->lsnFd);
        lsn->lsnFd = -1;
    }
    if (ch->lsnCache == lsn) {
        ch->lsnCache = VOS_NULL;
    }
    BkfChSerLetcpStopTmrReListen(ch, lsn);
    VOS_AVLL_DELETE(ch->lsnSet, lsn->avlNode);
    BKF_SIGN_CLR(lsn->sign);
    BKF_FREE(ch->argInit.base->memMng, lsn);
    return;
}

void BkfChSerLetcpDelAllLsn(BkfChSer *ch)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;
    void *itor = VOS_NULL;

    for (lsn = BkfChSerLetcpGetFirstLsn(ch, &itor); lsn != VOS_NULL; lsn = BkfChSerLetcpGetNextLsn(ch, &itor)) {
        BkfChSerLetcpDelLsn(ch, lsn);
    }
    return;
}

BkfChSerLetcpLsn *BkfChSerLetcpFindLsn(BkfChSer *ch, BkfUrl *urlSelf)
{
    BkfChSerLetcpLsn *lsn = ch->lsnCache;
    BOOL hit = (lsn != VOS_NULL) && BKF_URL_IS_EQUAL(urlSelf, &lsn->keyUrlSelf);
    if (hit) {
        return lsn;
    }

    lsn = VOS_AVLL_FIND(ch->lsnSet, urlSelf);
    if (lsn != VOS_NULL) {
        ch->lsnCache = lsn;
    }
    return lsn;
}

BkfChSerLetcpLsn *BkfChSerLetcpFindNextLsn(BkfChSer *ch, BkfUrl *urlSelf)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;

    lsn = VOS_AVLL_FIND_NEXT(ch->lsnSet, urlSelf);
    return lsn;
}

BkfChSerLetcpLsn *BkfChSerLetcpGetFirstLsn(BkfChSer *ch, void **itorOutOrNull)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;

    lsn = VOS_AVLL_FIRST(ch->lsnSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (lsn != VOS_NULL) ? VOS_AVLL_NEXT(ch->lsnSet, lsn->avlNode) : VOS_NULL;
    }
    return lsn;
}

BkfChSerLetcpLsn *BkfChSerLetcpGetNextLsn(BkfChSer *ch, void **itorInOut)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;

    lsn = (*itorInOut);
    *itorInOut = (lsn != VOS_NULL) ? VOS_AVLL_NEXT(ch->lsnSet, lsn->avlNode) : VOS_NULL;
    return lsn;
}

BkfChSerConnId *BkfChSerLetcpAddConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, int32_t connFd, BkfUrl *urlCli)
{
    uint32_t len = sizeof(BkfChSerConnId);
    BkfChSerConnId *connId = BKF_MALLOC(ch->argInit.base->memMng, len);
    if (connId == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(connId, len, 0, len);
    uint32_t baseInitRet = BkfChSerConnIdBaseInit(&connId->base, lsn->keyUrlSelf.type);
    if (baseInitRet != BKF_OK) {
        goto error;
    }
    connId->lsn = lsn;
    VOS_AVLL_INIT_NODE(connId->avlNode);
    connId->keyConnFd = connFd;
    connId->urlCli = *urlCli;
    BOOL insOk = VOS_AVLL_INSERT(lsn->connIdSet, connId->avlNode);
    if (!insOk) {
        goto error;
    }

    BKF_SIGN_SET(connId->sign, BKF_CH_SER_LETCP_CONN_ID_SIGN);
    lsn->connIdCache = connId;
    return connId;
error:
    if (connId != VOS_NULL) {
        /* insOk一定为false */
        if (baseInitRet == BKF_OK) {
            BkfChSerConnIdBaseUninit(&connId->base);
        }
        BKF_FREE(ch->argInit.base->memMng, connId);
    }
    return VOS_NULL;
}

void BkfChSerLetcpDelConnId(BkfChSer *ch, BkfChSerConnId *connId)
{
    BkfChSerLetcpLsn *lsn = connId->lsn;
    if (connId->lockDel) {
        connId->softDel = VOS_TRUE;
        return;
    }
    if (lsn->connIdCache == connId) {
        lsn->connIdCache = VOS_NULL;
    }
    BkfChSerLetcpStopConnTmrWriteErr(ch, connId);
    if (connId->connFdAttachOk) {
        (void)WfwMuxDetachFd(ch->argInit.base->mux, connId->keyConnFd);
    }
    (void)close(connId->keyConnFd);
    connId->keyConnFd = -1;
    BkfChSerConnIdBaseUninit(&connId->base);
    VOS_AVLL_DELETE(lsn->connIdSet, connId->avlNode);
    BKF_SIGN_CLR(connId->sign);
    BKF_FREE(ch->argInit.base->memMng, connId);
    return;
}

// 设置阻塞式
STATIC void BkfChSerLetcpSetFdBlock(int32_t fd)
{
    if (fd < 0) {
        return;
    }
    int32_t flags = fcntl(fd, F_GETFL, 0);
    (void)fcntl(fd, F_SETFL, (uint32_t)flags & ~O_NONBLOCK);
}

void BkfChSerLetcpDelAllConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    BkfChSerConnId *connId = VOS_NULL;
    void *itor = VOS_NULL;

    for (connId = BkfChSerLetcpGetFirstConnId(ch, lsn, &itor); connId != VOS_NULL;
         connId = BkfChSerLetcpGetNextConnId(ch, lsn, &itor)) {
        /* mod block mod */
        BkfChSerLetcpSetFdBlock(connId->keyConnFd);
        BkfChSerLetcpDelConnId(ch, connId);
    }
    return;
}

BkfChSerConnId *BkfChSerLetcpFindConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, int32_t connFd)
{
    (void)ch;
    BkfChSerConnId *connId = VOS_NULL;
    BOOL hit = VOS_FALSE;

    connId = lsn->connIdCache;
    hit = (connId != VOS_NULL) && (connId->keyConnFd == connFd);
    if (hit) {
        return connId;
    }

    connId = VOS_AVLL_FIND(lsn->connIdSet, &connFd);
    if (connId != VOS_NULL) {
        lsn->connIdCache = connId;
    }
    return connId;
}

BkfChSerConnId *BkfChSerLetcpFindNextConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, int32_t connFd)
{
    (void)ch;
    BkfChSerConnId *connId = VOS_NULL;

    connId = VOS_AVLL_FIND_NEXT(lsn->connIdSet, &connFd);
    return connId;
}

BkfChSerConnId *BkfChSerLetcpGetFirstConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, void **itorOutOrNull)
{
    (void)ch;
    BkfChSerConnId *connId = VOS_NULL;

    connId = VOS_AVLL_FIRST(lsn->connIdSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (connId != VOS_NULL) ? VOS_AVLL_NEXT(lsn->connIdSet, connId->avlNode) : VOS_NULL;
    }
    return connId;
}

BkfChSerConnId *BkfChSerLetcpGetNextConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, void **itorInOut)
{
    (void)ch;
    BkfChSerConnId *connId = VOS_NULL;

    connId = (*itorInOut);
    *itorInOut = (connId != VOS_NULL) ? VOS_AVLL_NEXT(lsn->connIdSet, connId->avlNode) : VOS_NULL;
    return connId;
}

uint32_t BkfChSerLetcpStartConnOnceTmrWriteErr(BkfChSer *ch, BkfChSerConnId *connId, F_BKF_TMR_TIMEOUT_PROC proc,
    uint32_t intervalMs)
{
    if (connId->tmrIdWriteErr == VOS_NULL) {
        connId->tmrIdWriteErr = BkfTmrStartOnce(ch->argInit.base->tmrMng, proc, intervalMs, connId);
    }
    return (connId->tmrIdWriteErr != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void BkfChSerLetcpStopConnTmrWriteErr(BkfChSer *ch, BkfChSerConnId *connId)
{
    if (connId->tmrIdWriteErr != VOS_NULL) {
        BkfTmrStop(ch->argInit.base->tmrMng, connId->tmrIdWriteErr);
        connId->tmrIdWriteErr = VOS_NULL;
    }
}

uint32_t BkfChSerLetcpStartOnceTmrReListen(BkfChSer *ch, BkfChSerLetcpLsn *lsn, F_BKF_TMR_TIMEOUT_PROC proc,
    uint32_t intervalMs)
{
    if (lsn->tmrIdReLsn == VOS_NULL) {
        lsn->tmrIdReLsn = BkfTmrStartOnce(ch->argInit.base->tmrMng, proc, intervalMs, lsn);
    }
    BKF_LOG_WARN(BKF_LOG_HND, "fd(%d) relisten tmrId %#x\n", lsn->lsnFd, BKF_MASK_ADDR(lsn->tmrIdReLsn));
    return (lsn->tmrIdReLsn != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void BkfChSerLetcpStopTmrReListen(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    if (lsn->tmrIdReLsn != VOS_NULL) {
        BkfTmrStop(ch->argInit.base->tmrMng, lsn->tmrIdReLsn);
        lsn->tmrIdReLsn = VOS_NULL;
    }
}

BkfChSerLetcpMsgHead *BkfChSerLetcpMsgMalloc(BkfChSer *ch, int32_t dataBufLen)
{
    int32_t headLen;
    int32_t mallocLen;
    BkfChSerLetcpMsgHead *chMsgHead = VOS_NULL;

    headLen = sizeof(BkfChSerLetcpMsgHead);
    mallocLen = headLen + dataBufLen;
    chMsgHead = BKF_MALLOC(ch->argInit.base->memMng, mallocLen);
    if (chMsgHead == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(chMsgHead, headLen, 0, headLen);
    BKF_SIGN_SET(chMsgHead->sign, BKF_CH_SER_LETCP_MSG_SIGN);

    BKF_LOG_CNT(BKF_LOG_CNT_HND);
    return chMsgHead;
}

void BkfChSerLetcpMsgFree(BkfChSer *ch, BkfChSerLetcpMsgHead *chMsgHead)
{
    BKF_LOG_CNT(BKF_LOG_CNT_HND);
    BKF_FREE(ch->argInit.base->memMng, chMsgHead);
}

#ifdef __cplusplus
}
#endif
