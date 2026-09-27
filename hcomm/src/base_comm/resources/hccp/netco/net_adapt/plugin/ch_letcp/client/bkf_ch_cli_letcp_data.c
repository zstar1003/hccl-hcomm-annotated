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
#include "bkf_ch_cli_letcp_data.h"
#include "bkf_str.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t BkfChCliLeTcpConnCmp(BkfChCliConnId *input1, BkfChCliConnId *input2)
{
    int32_t ret;
    ret = BkfUrlCmp(&input1->keyUrlCli, &input2->keyUrlCli);
    if (ret != 0) {
        return ret;
    }
    return BkfUrlCmp(&input1->keyUrlSer, &input2->keyUrlSer);
}

BkfChCli *BkfChCliLetcpDataInit(BkfChCliInitArg *arg)
{
    uint32_t len = sizeof(BkfChCli);
    BkfChCli *ch = BKF_MALLOC(arg->base->memMng, len);
    if (ch == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(ch, len, 0, len);
    if (arg->base->dbgOn) {
        ch->log = arg->base->log;
    }

    VOS_AVLL_INIT_TREE(ch->connSet, (AVLL_COMPARE)BkfChCliLeTcpConnCmp, BKF_OFFSET(BkfChCliConnId, keyUrlSer),
        BKF_OFFSET(BkfChCliConnId, avlNode));

    ch->name = BkfStrNew(arg->base->memMng, "cli_letcp_%s", arg->name);
    if (ch->name == VOS_NULL) {
        BKF_FREE(arg->base->memMng, ch);
        return VOS_NULL;
    }

    ch->argInit = *arg;
    BKF_SIGN_SET(ch->sign, BKF_CH_CLI_LETCP_SIGN);
    return ch;
}

void BkfChCliLetcpDataUninit(BkfChCli *ch)
{
    BkfStrDel(ch->name);
    BKF_SIGN_CLR(ch->sign);
    BKF_FREE(ch->argInit.base->memMng, ch);
    return;
}

void BkfChCliLetcpFreeConnId(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;

    if (ch->connIdCache == connId) {
        ch->connIdCache = VOS_NULL;
    }

    BKF_SIGN_CLR(connId->sign);
    BkfChCliLetcpStopConnTmrWriteErr(connId);

    BkfChCliConnIdBaseUninit(&connId->base);
    VOS_AVLL_DELETE(ch->connSet, connId->avlNode);
    BKF_FREE(ch->argInit.base->memMng, connId);
    return;
}

BkfChCliConnId *BkfChCliLetcpNewConnId(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlSelf)
{
    uint32_t baseInitRet = BKF_ERR;
    BOOL insOk = VOS_FALSE;

    BkfChCliConnId *connId = BKF_MALLOC(ch->argInit.base->memMng, sizeof(BkfChCliConnId));
    if (unlikely(connId == VOS_NULL)) {
        return VOS_NULL;
    }

    (void)memset_s(connId, sizeof(BkfChCliConnId), 0, sizeof(BkfChCliConnId));

    baseInitRet = BkfChCliConnIdBaseInit(&connId->base, urlServer->type);
    if (unlikely(baseInitRet != BKF_OK)) {
        goto error;
    }

    VOS_AVLL_INIT_NODE(connId->avlNode);
    connId->keyUrlSer = *urlServer;
    connId->keyUrlCli = *urlSelf;
    connId->ch = ch;
    connId->connFd = -1;
    insOk = VOS_AVLL_INSERT(ch->connSet, connId->avlNode);
    if (unlikely(!insOk)) {
        goto error;
    }
    ch->connIdCache = connId;
    BKF_SIGN_SET(connId->sign, BKF_CH_CLI_LETCP_CONN_ID_SIGN);
    return connId;

error:
    if (connId != VOS_NULL) {
        if (baseInitRet == BKF_OK) {
            BkfChCliConnIdBaseUninit(&connId->base);
        }
        BKF_FREE(ch->argInit.base->memMng, connId);
    }
    return VOS_NULL;
}

BkfChCliConnId *BkfChCliLetcpFindConnId(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlself)
{
    BkfChCliConnId *connId = ch->connIdCache;
    BOOL hit = (connId != VOS_NULL) && BKF_URL_IS_EQUAL(urlServer, &connId->keyUrlSer) &&
               BKF_URL_IS_EQUAL(urlself, &connId->keyUrlCli);
    if (hit) {
        return connId;
    }
    BkfChCliConnId connIdKey;
    connIdKey.keyUrlCli = *urlself;
    connIdKey.keyUrlSer = *urlServer;
    connId = VOS_AVLL_FIND(ch->connSet, &connIdKey);
    if (connId != VOS_NULL) {
        ch->connIdCache = connId;
    }
    return connId;
}

BkfChCliConnId *BkfChCliLetcpFindNextConnId(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlself)
{
    BkfChCliConnId *connId = VOS_NULL;
    BkfChCliConnId tmpConnId;
    tmpConnId.keyUrlCli = *urlself;
    tmpConnId.keyUrlSer = *urlServer;
    connId = VOS_AVLL_FIND_NEXT(ch->connSet, &tmpConnId);
    return connId;
}

BkfChCliConnId *BkfChCliLetcpGetFirstConnId(BkfChCli *ch, void **itorOutOrNull)
{
    BkfChCliConnId *connId = VOS_NULL;

    connId = VOS_AVLL_FIRST(ch->connSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (connId != VOS_NULL) ? VOS_AVLL_NEXT(ch->connSet, connId->avlNode) : VOS_NULL;
    }
    return connId;
}

BkfChCliConnId *BkfChCliLetcpGetNextConnId(BkfChCli *ch, void **itorInOut)
{
    BkfChCliConnId *connId = VOS_NULL;

    connId = (*itorInOut);
    *itorInOut = (connId != VOS_NULL) ? VOS_AVLL_NEXT(ch->connSet, connId->avlNode) : VOS_NULL;
    return connId;
}

uint32_t BkfChCliLetcpStartConnOnceTmrWriteErr(BkfChCliConnId *connId, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs)
{
    BkfChCli *ch = connId->ch;

    if (connId->tmrIdWriteErr == VOS_NULL) {
        connId->tmrIdWriteErr = BkfTmrStartOnce(ch->argInit.base->tmrMng, proc, intervalMs, connId);
    }
    return (connId->tmrIdWriteErr != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void BkfChCliLetcpStopConnTmrWriteErr(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;

    if (connId->tmrIdWriteErr != VOS_NULL) {
        BkfTmrStop(ch->argInit.base->tmrMng, connId->tmrIdWriteErr);
        connId->tmrIdWriteErr = VOS_NULL;
    }

    return;
}

BkfChCliLetcpMsgHead *BkfChCliLetcpMsgMalloc(BkfChCli *ch, int32_t dataBufLen)
{
    int32_t headLen = sizeof(BkfChCliLetcpMsgHead);
    int32_t mallocLen = headLen + dataBufLen;

    BkfChCliLetcpMsgHead *chMsgHead = BKF_MALLOC(ch->argInit.base->memMng, mallocLen);
    if (chMsgHead == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(chMsgHead, headLen, 0, headLen);
    BKF_SIGN_SET(chMsgHead->sign, BKF_CH_CLI_LETCP_MSG_SIGN);

    BKF_LOG_CNT(BKF_LOG_CNT_HND);
    return chMsgHead;
}

void BkfChCliLetcpMsgFree(BkfChCli *ch, BkfChCliLetcpMsgHead *chMsgHead)
{
    BKF_LOG_CNT(BKF_LOG_CNT_HND);
    BKF_FREE(ch->argInit.base->memMng, chMsgHead);
    return;
}

#ifdef __cplusplus
}
#endif
