/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_ch_ser_letcp_data.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct tagBkfChSerLetcpDispCnt {
    int32_t lsnCnt;
    int32_t connCnt;
} BkfChSerLetcpDispCnt;
STATIC void BkfChSerLetcpDispGetCntOneLsn(BkfChSer *ch, BkfChSerLetcpLsn *lsn, BkfChSerLetcpDispCnt *cnt)
{
    BkfChSerConnId *connId = VOS_NULL;
    void *itor = VOS_NULL;

    for (connId = BkfChSerLetcpGetFirstConnId(ch, lsn, &itor); connId != VOS_NULL;
         connId = BkfChSerLetcpGetNextConnId(ch, lsn, &itor)) {
        cnt->connCnt++;
    }
}
STATIC void BkfChSerLetcpDispGetCnt(BkfChSer *ch, BkfChSerLetcpDispCnt *cnt)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;
    void *itor = VOS_NULL;

    for (lsn = BkfChSerLetcpGetFirstLsn(ch, &itor); lsn != VOS_NULL; lsn = BkfChSerLetcpGetNextLsn(ch, &itor)) {
        cnt->lsnCnt++;
        BkfChSerLetcpDispGetCntOneLsn(ch, lsn, cnt);
    }
}

STATIC void BkfChSerLetcpDispSelf(BkfChSer *ch)
{
    BkfDisp *disp = ch->argInit.base->disp;
    BkfChSerInitArg *argInit = &ch->argInit;
    BkfChSerLetcpDispCnt cnt = {0};

    BKF_DISP_PRINTF(disp, "%s\n", ch->name);
    BKF_DISP_PRINTF(disp, "====init info====\n");
    BKF_DISP_PRINTF(disp, "dbgOn(%u)\n", argInit->base->dbgOn);
    BKF_DISP_PRINTF(disp, "memMng(%#x)\n", BKF_MASK_ADDR(argInit->base->memMng));
    BKF_DISP_PRINTF(disp, "disp(%#x)\n", BKF_MASK_ADDR(argInit->base->disp));
    BKF_DISP_PRINTF(disp, "logCnt(%#x)\n", BKF_MASK_ADDR(argInit->base->logCnt));
    BKF_DISP_PRINTF(disp, "log(%#x)\n", BKF_MASK_ADDR(argInit->base->log));
    BKF_DISP_PRINTF(disp, "pfm(%#x)\n", BKF_MASK_ADDR(argInit->base->pfm));
    BKF_DISP_PRINTF(disp, "tmrMng(%#x)\n", BKF_MASK_ADDR(argInit->base->tmrMng));
    BKF_DISP_PRINTF(disp, "mux(%#x)\n", BKF_MASK_ADDR(argInit->base->mux));

    BKF_DISP_PRINTF(disp, "====runtime info====\n");
    BKF_DISP_PRINTF(disp, "sign(%#x)\n", ch->sign);
    BKF_DISP_PRINTF(disp, "hasEnable(%u)\n", ch->hasEnable);
    BKF_DISP_PRINTF(disp, "log(%#x)\n", BKF_MASK_ADDR(ch->log));

    BKF_DISP_PRINTF(disp, "lsnCache(%#x)\n", BKF_MASK_ADDR(ch->lsnCache));
    BkfChSerLetcpDispGetCnt(ch, &cnt);
    BKF_DISP_PRINTF(disp, "lsnTotalCnt(%d)\n", cnt.lsnCnt);
    BKF_DISP_PRINTF(disp, "connTotalCnt(%d)\n", cnt.connCnt);
}

STATIC void BkfChSerLetcpDispOneLsn(BkfChSer *ch, BkfChSerLetcpLsn *lsn, int32_t idx)
{
    BkfDisp *disp = ch->argInit.base->disp;
    BkfChSerLetcpDispCnt cnt = {0};
    uint8_t buf[BKF_1K / 8];

    BKF_DISP_PRINTF(disp, "++lsn[%d]============\n", idx);
    BKF_DISP_PRINTF(disp, "sign(%#x)\n", lsn->sign);
    BKF_DISP_PRINTF(disp, "keyUrlSelf(%s)\n", BkfUrlGetStr(&lsn->keyUrlSelf, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "lsnFd(%u)\n", lsn->lsnFd);
    BKF_DISP_PRINTF(disp, "lsnFdAttachOk(%u)\n", lsn->lsnFdAttachOk);
    BkfChSerLetcpDispGetCntOneLsn(ch, lsn, &cnt);
    BKF_DISP_PRINTF(disp, "connCnt(%d)\n", cnt.connCnt);
}

void BkfChSerLetcpDispSummary(BkfChSer *ch)
{
    BkfDisp *disp = ch->argInit.base->disp;
    BkfUrl *lastLsnUrlSelf = VOS_NULL;
    BkfChSerLetcpLsn *curLsn = VOS_NULL;
    int32_t lsnCnt;

    lastLsnUrlSelf = BKF_DISP_GET_LAST_CTX(disp, &lsnCnt);
    if (lastLsnUrlSelf == VOS_NULL) {
        BkfChSerLetcpDispSelf(ch);
        curLsn = BkfChSerLetcpGetFirstLsn(ch, VOS_NULL);
        lsnCnt = 0;
    } else {
        curLsn = BkfChSerLetcpFindNextLsn(ch, lastLsnUrlSelf);
    }

    if (curLsn != VOS_NULL) {
        BkfChSerLetcpDispOneLsn(ch, curLsn, lsnCnt);
        lsnCnt++;
        BKF_DISP_SAVE_CTX(disp, &curLsn->keyUrlSelf, sizeof(BkfUrl), &lsnCnt, sizeof(lsnCnt));
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d lsn(s), ***\n", lsnCnt);
    }
}

typedef struct tagBkfChSerLetcpDispLsnCtx {
    BkfUrl lsnUrlSelf;
    int32_t lsnCnt;
    int32_t connFd;
    int32_t lsnConnCnt;
    int32_t connTotalCnt;
} BkfChSerLetcpDispLsnCtx;

STATIC void BkfChSerLetcpDispLsnGetFirst(BkfChSer *ch, BkfChSerLetcpLsn **outCurLsn, BOOL *outCurLsnIsNew,
    BkfChSerConnId **outCurConnId)
{
    *outCurLsnIsNew = VOS_TRUE;
    *outCurLsn = BkfChSerLetcpGetFirstLsn(ch, VOS_NULL);
    if (*outCurLsn != VOS_NULL) {
        *outCurConnId = BkfChSerLetcpGetFirstConnId(ch, *outCurLsn, VOS_NULL);
    }
}

STATIC void BkfChSerLetcpDispLsnGetNext(BkfChSer *ch, BkfChSerLetcpDispLsnCtx *lastCtx, BkfChSerLetcpLsn **outCurLsn,
    BOOL *outCurLsnIsNew, BkfChSerConnId **outCurConnId)
{
    if (lastCtx->connFd == -1) {
        *outCurLsnIsNew = VOS_TRUE;
        *outCurLsn = BkfChSerLetcpFindNextLsn(ch, &lastCtx->lsnUrlSelf);
        if (*outCurLsn != VOS_NULL) {
            *outCurConnId = BkfChSerLetcpGetFirstConnId(ch, *outCurLsn, VOS_NULL);
        }
    } else {
        *outCurLsn = BkfChSerLetcpFindLsn(ch, &lastCtx->lsnUrlSelf);
        if (*outCurLsn != VOS_NULL) {
            *outCurConnId = BkfChSerLetcpFindNextConnId(ch, *outCurLsn, lastCtx->connFd);
        } else {
            *outCurLsnIsNew = VOS_TRUE;
            *outCurLsn = BkfChSerLetcpFindNextLsn(ch, &lastCtx->lsnUrlSelf);
            if (*outCurLsn != VOS_NULL) {
                *outCurConnId = BkfChSerLetcpGetFirstConnId(ch, *outCurLsn, VOS_NULL);
            }
        }
    }
}
STATIC void BkfChSerLetcpDispOneConnId(BkfChSer *ch, BkfChSerConnId *connId, int32_t idx)
{
    BkfDisp *disp = ch->argInit.base->disp;
    uint8_t buf[BKF_1K / 8];

    BKF_DISP_PRINTF(disp, "++++connId[%d]============\n", idx);
    BKF_DISP_PRINTF(disp, "sign(%#x)\n", connId->sign);
    BKF_DISP_PRINTF(disp, "keyConnFd(%d)\n", connId->keyConnFd);
    BKF_DISP_PRINTF(disp, "urlCli(%s)\n", BkfUrlGetStr(&connId->urlCli, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "urlLocal(%s)\n", BkfUrlGetStr(&connId->urlLocal, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "tmrIdWriteErr(%#x)\n", BKF_MASK_ADDR(connId->tmrIdWriteErr));
    BKF_DISP_PRINTF(disp, "connFdAttachOk(%u)\n", connId->connFdAttachOk);
    BKF_DISP_PRINTF(disp, "lockDel(%u)\n", connId->lockDel);
    BKF_DISP_PRINTF(disp, "softDel(%u)\n", connId->softDel);
    BKF_DISP_PRINTF(disp, "tcpstate(%u)\n", connId->tcpState);
}
void BkfChSerLetcpDispLsn(BkfChSer *ch)
{
    BkfDisp *disp = ch->argInit.base->disp;
    BkfChSerLetcpDispLsnCtx *lastCtx = VOS_NULL;
    BkfChSerLetcpDispLsnCtx curCtx = {0};
    BkfChSerLetcpLsn *curLsn = VOS_NULL;
    BOOL curLsnIsNew = VOS_FALSE;
    BkfChSerConnId *curConnId = VOS_NULL;

    lastCtx = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (lastCtx == VOS_NULL) {
        BkfChSerLetcpDispLsnGetFirst(ch, &curLsn, &curLsnIsNew, &curConnId);
    } else {
        curCtx = *lastCtx;
        BkfChSerLetcpDispLsnGetNext(ch, lastCtx, &curLsn, &curLsnIsNew, &curConnId);
    }

    if (curLsn != VOS_NULL) {
        if (curLsnIsNew) {
            BkfChSerLetcpDispOneLsn(ch, curLsn, curCtx.lsnCnt);
            curCtx.lsnUrlSelf = curLsn->keyUrlSelf;
            curCtx.lsnCnt++;
        }
        if (curConnId != VOS_NULL) {
            BkfChSerLetcpDispOneConnId(ch, curConnId, curCtx.lsnConnCnt);
            curCtx.connFd = curConnId->keyConnFd;
            curCtx.lsnConnCnt++;
            curCtx.connTotalCnt++;
        } else {
            curCtx.connFd = -1;
        }
        BKF_DISP_SAVE_CTX(disp, &curCtx, sizeof(BkfChSerLetcpDispLsnCtx), VOS_NULL, 0);
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d lsn(s), %d connId(s) ***\n", curCtx.lsnCnt, curCtx.connTotalCnt);
    }
}

void BkfChSerLetcpDispInit(BkfChSer *ch)
{
    BkfDisp *disp = ch->argInit.base->disp;
    char *objName = (char *)ch->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, ch);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfChSerLetcpDispSummary, "disp ch linux tcp summary", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfChSerLetcpDispLsn, "disp ch linux tcp lsn", objName, 0);
}

void BkfChSerLetcpDispUninit(BkfChSer *ch)
{
    BkfDispUnregObj(ch->argInit.base->disp, ch->name);
}

#ifdef __cplusplus
}
#endif
