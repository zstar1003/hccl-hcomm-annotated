/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_conn.h"
#include "bkf_suber_conn_data.h"
#include "bkf_suber_sess.h"
#include "bkf_disp.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
typedef struct {
    uint32_t connCnt;
    uint32_t sessCtxLen;
    BkfUrl puberUrl;
    BkfUrl localUrl;
} BkfSuberConnDispConnCtx;

typedef void (*BKF_SUBER_CONN_DISP_PROC_FIST_CONN)(BkfSuberConnMng *connMng, BkfSuberConn *conn);
typedef void (*BKF_SUBER_CONN_DISP_PROC_END)(BkfSuberConnMng *connMng, BkfSuberConnDispConnCtx *curCtx);
typedef void (*BKF_SUBER_CONN_DISP_PROC_ONE_CONN)(BkfSuberConnMng *connMng, BkfSuberConn *conn,
    BkfSuberConnDispConnCtx *curCtx);
typedef uint32_t (*BKF_SUBER_CONN_DISP_PROC_SESS)(BkfSuberSessMng *sessMng, BkfDisp *disp,
    BkfDispTempCtx *lastTempCtxOrNull, BkfDispTempCtx *curTempCtx);
#pragma pack()
#endif

#if BKF_BLOCK("私有函数定义")
BkfSuberConn *BkfSuberGetNextConnByStartUrlType(BkfSuberConnMng *connMng, uint8_t startUrlType)
{
    BkfSuberConnUrlTypeMng *urlTypeMng = VOS_NULL;
    BkfSuberConn *conn = VOS_NULL;
    uint32_t i;
    for (i = startUrlType; i < BKF_GET_ARY_COUNT(connMng->urlTypeSet); i++) {
        urlTypeMng = connMng->urlTypeSet[i];
        if (urlTypeMng != VOS_NULL) {
            conn = BkfSuberConnDataGetFirstByUrlType(connMng, urlTypeMng->urlType, VOS_NULL);
            if (conn != VOS_NULL) {
                break;
            }
        }
    }
    return conn;
}

void BkfSuberConnDispSummary(BkfSuberConnMng *connMng)
{
    BkfDisp *disp = connMng->env->disp;
    BKF_DISP_PRINTF(disp, "===============================\n");
    BKF_DISP_PRINTF(disp, "isEnable: %u\n", connMng->isEnable);
    BkfSuberConnUrlTypeMng *urlTypeMng = VOS_NULL;
    uint32_t i;
    uint8_t urlBuf[BKF_1K / 4] = {0};
    for (i = BKF_URL_TYPE_INVALID + 1; i < BKF_GET_ARY_COUNT(connMng->urlTypeSet); i++) {
        urlTypeMng = connMng->urlTypeSet[i];
        if (urlTypeMng != VOS_NULL) {
            BKF_DISP_PRINTF(disp, "====================\n");
            BKF_DISP_PRINTF(disp, "urlType: %u\n", urlTypeMng->urlType);
            BKF_DISP_PRINTF(disp, "selfUrl: %s\n", BkfUrlGetStr(&urlTypeMng->selfUrl, urlBuf, sizeof(urlBuf)));
            BKF_DISP_PRINTF(disp, "connCnt: %u\n", urlTypeMng->connCnt);
        }
    }
    BKF_DISP_PRINTF(disp, "===============================\n");
    return;
}

void BkfSuberConnDispAllConnEndPrintf(BkfSuberConnMng *connMng, BkfSuberConnDispConnCtx *curCtx)
{
    BkfDisp *disp = connMng->env->disp;
    BKF_DISP_PRINTF(disp, "============================================\n");
    BKF_DISP_PRINTF(disp, "Total Conn Cnt: %u\n", curCtx->connCnt);
    BKF_DISP_PRINTF(disp, "============================================\n");
    return;
}

void BkfSuberConnDispOneConnFsm(BkfSuberConnMng *connMng, BkfSuberConn *conn, BkfSuberConnDispConnCtx *curCtx)
{
    BkfDisp *disp = connMng->env->disp;
    uint8_t buf[BKF_1K / 4] = {0};
    uint8_t fsmStrBuf[BKF_1K / 2] = {0};
    BKF_DISP_PRINTF(disp, "============================================\n");
    BKF_DISP_PRINTF(disp, "+conn[%d], puberUrl(%s), fsm:\n%s\n", curCtx->connCnt,
        BkfUrlGetStr(&conn->puberUrl, buf, sizeof(buf)), BkfFsmGetStr(&conn->fsm, fsmStrBuf, sizeof(fsmStrBuf)));
    return;
}

void BkfSuberConnDispAllConnSched(BkfSuberConnMng *connMng, BKF_SUBER_CONN_DISP_PROC_FIST_CONN firstConnPrintf,
    BKF_SUBER_CONN_DISP_PROC_ONE_CONN oneConnPrintf, BKF_SUBER_CONN_DISP_PROC_END endPrintf)
{
    BkfDisp *disp = connMng->env->disp;
    BkfSuberConnDispConnCtx *lastCtx = VOS_NULL;
    BkfSuberConnDispConnCtx curCtx = {0};
    BkfSuberConn *conn = VOS_NULL;

    lastCtx = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (lastCtx == VOS_NULL) {
        conn = BkfSuberGetNextConnByStartUrlType(connMng, BKF_URL_TYPE_INVALID + 1);
        if (firstConnPrintf != VOS_NULL) {
            firstConnPrintf(connMng, conn);
        }
    } else {
        curCtx = *lastCtx;
        conn = BkfSuberConnDataFindNext(connMng, &curCtx.puberUrl, &curCtx.localUrl);
        if (conn == VOS_NULL) {
            conn = BkfSuberGetNextConnByStartUrlType(connMng, curCtx.puberUrl.type + 1);
        }
    }

    if (conn == VOS_NULL) {
        if (endPrintf != VOS_NULL) {
            endPrintf(connMng, &curCtx);
        }
    } else {
        curCtx.connCnt++;
        if (oneConnPrintf != VOS_NULL) {
            oneConnPrintf(connMng, conn, &curCtx);
        }
        curCtx.puberUrl = conn->puberUrl;
        curCtx.localUrl = conn->localUrl;
        BKF_DISP_SAVE_CTX(disp, &curCtx, sizeof(curCtx), VOS_NULL, 0);
    }
    return;
}

void BkfSuberConnDispAllConnFsm(BkfSuberConnMng *connMng)
{
    BkfSuberConnDispAllConnSched(connMng, VOS_NULL, BkfSuberConnDispOneConnFsm, BkfSuberConnDispAllConnEndPrintf);
    return;
}

BkfSuberConn *BkfSuberConnDispGetConnByCtx(BkfSuberConnMng *connMng, BkfSuberConnDispConnCtx *curCtx, BOOL *isNewConn,
    BKF_SUBER_CONN_DISP_PROC_FIST_CONN firstConnPrintf)
{
    BkfDisp *disp = connMng->env->disp;
    BkfSuberConnDispConnCtx *lastCtx = VOS_NULL;
    BkfSuberConn *conn = VOS_NULL;

    *isNewConn = VOS_FALSE;
    lastCtx = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (lastCtx == VOS_NULL) {
        conn = BkfSuberGetNextConnByStartUrlType(connMng, BKF_URL_TYPE_INVALID + 1);
        *isNewConn = VOS_TRUE;
        if (firstConnPrintf != VOS_NULL) {
            firstConnPrintf(connMng, conn);
        }
    } else {
        *curCtx = *lastCtx;
        if (curCtx->sessCtxLen != 0) {
            conn = BkfSuberConnDataFind(connMng, &curCtx->puberUrl, &curCtx->localUrl);
        } else {
            conn = BkfSuberConnDataFindNext(connMng, &curCtx->puberUrl, &curCtx->localUrl);
            *isNewConn = VOS_TRUE;
        }
        if (conn == VOS_NULL) {
            conn = BkfSuberGetNextConnByStartUrlType(connMng, curCtx->puberUrl.type + 1);
            *isNewConn = VOS_TRUE;
        }
    }

    return conn;
}

BkfDispTempCtx *BkfSuberConnDispGetSessCtx(BkfDisp *disp, BkfDispTempCtx *lastSessTempCtx,
    BkfDispTempCtx *curSessTempCtx, BkfSuberConnDispConnCtx *curConnCtx)
{
    BKF_DISP_TEMP_CTX_INIT(lastSessTempCtx);
    BKF_DISP_TEMP_CTX_INIT(curSessTempCtx);
    (void)BKF_DISP_GET_LAST_CTX(disp, lastSessTempCtx);

    BkfDispTempCtx *lastSessCtx;
    if (curConnCtx->sessCtxLen == 0) {
        lastSessCtx = VOS_NULL;
    } else {
        lastSessCtx = lastSessTempCtx;
    }

    return lastSessCtx;
}

void BkfSuberConnDispAllConnAndSessSched(BkfSuberConnMng *connMng, BKF_SUBER_CONN_DISP_PROC_FIST_CONN firstConnPrintf,
    BKF_SUBER_CONN_DISP_PROC_ONE_CONN printfOneConn, BKF_SUBER_CONN_DISP_PROC_END endPrintf,
    BKF_SUBER_CONN_DISP_PROC_SESS sessPrintf)
{
    BkfDisp *disp = connMng->env->disp;
    BkfSuberConnDispConnCtx curCtx = {0};
    BkfSuberConn *conn = VOS_NULL;
    BOOL isNewConn = VOS_FALSE;

    conn = BkfSuberConnDispGetConnByCtx(connMng, &curCtx, &isNewConn, firstConnPrintf);
    if (conn == VOS_NULL) {
        if (endPrintf != VOS_NULL) {
            endPrintf(connMng, &curCtx);
        }
    } else {
        if (isNewConn) {
            curCtx.connCnt++;
            if (printfOneConn != VOS_NULL) {
                printfOneConn(connMng, conn, &curCtx);
            }
        }
        BkfDispTempCtx lastSessTempCtx;
        BkfDispTempCtx curSessTempCtx;
        BkfDispTempCtx *lastSessCtx = BkfSuberConnDispGetSessCtx(disp, &lastSessTempCtx, &curSessTempCtx, &curCtx);
        uint32_t sessCtxLen = 0;
        if (sessPrintf != VOS_NULL) {
            sessCtxLen = sessPrintf(conn->sessMng, disp, lastSessCtx, &curSessTempCtx);
        }
        curCtx.puberUrl = conn->puberUrl;
        curCtx.localUrl = conn->localUrl;
        curCtx.sessCtxLen = sessCtxLen;
        BKF_DISP_SAVE_CTX(disp, &curCtx, sizeof(curCtx), &curSessTempCtx, sessCtxLen);
    }
    return;
}

void BkfSuberConnDispOneConnSummary(BkfSuberConnMng *connMng, BkfSuberConn *conn, BkfSuberConnDispConnCtx *curCtx)
{
    BkfDisp *disp = connMng->env->disp;
    uint8_t buf[BKF_1K / 4] = {0};
    BKF_DISP_PRINTF(disp, "============================================\n");
    BKF_DISP_PRINTF(disp,
        "+conn[%d], puberUrl(%s), connId(%#x)/state(%u, %s), rcv_sendBuf(%#x, %d/%#x, %d), "
        "jobId(%#x), instCnt(%u), lockdel(%u), trigConn(%u), sessMng(%#x)\n",
        curCtx->connCnt, BkfUrlGetStr(&conn->puberUrl, buf, sizeof(buf)), BKF_MASK_ADDR(conn->connId),
        BKF_FSM_GET_STATE(&conn->fsm), BkfFsmGetStateStr(&conn->fsm), BKF_MASK_ADDR(conn->rcvDataBuf),
        BkfBufqGetUsedLen(conn->rcvDataBuf), BKF_MASK_ADDR(conn->lastSendLeftDataBuf),
        BkfBufqGetUsedLen(conn->lastSendLeftDataBuf), BKF_MASK_ADDR(conn->jobId), conn->obInstCnt, conn->lockDel,
        conn->isTrigConn, BKF_MASK_ADDR(conn->sessMng));
    return;
}

void BkfSuberConnDispAllConnAndSessSummary(BkfSuberConnMng *connMng)
{
    BkfSuberConnDispAllConnAndSessSched(connMng, VOS_NULL, BkfSuberConnDispOneConnSummary,
        BkfSuberConnDispAllConnEndPrintf, BkfSuberSessDispOneSessSummary);
    return;
}

void BkfSuberConnDispOneConnUrl(BkfSuberConnMng *connMng, BkfSuberConn *conn, BkfSuberConnDispConnCtx *curCtx)
{
    BkfDisp *disp = connMng->env->disp;
    uint8_t buf[BKF_DISP_PRINTF_BUF_LEN_MAX] = {0};
    BKF_DISP_PRINTF(disp, "============================================\n");
    BKF_DISP_PRINTF(disp, "+conn[%d], puberUrl(%s)\n", curCtx->connCnt,
        BkfUrlGetStr(&conn->puberUrl, buf, sizeof(buf)));
    return;
}

void BkfSuberConnDispAllSessFsm(BkfSuberConnMng *connMng)
{
    BkfSuberConnDispAllConnAndSessSched(connMng, VOS_NULL, BkfSuberConnDispOneConnUrl, BkfSuberConnDispAllConnEndPrintf,
        BkfSuberSessDispOneSessFsm);
    return;
}

void BkfSuberConnDispCloseBatchTimeout(BkfSuberConnMng *connMng)
{
    BkfDisp *disp = connMng->env->disp;
    BkfSuberConnUrlTypeMng *urlTypeMng = VOS_NULL;
    BkfSuberConn *conn = VOS_NULL;
    uint32_t i;
    uint32_t closeCnt = 0;
    for (i = BKF_URL_TYPE_INVALID + 1; i < BKF_GET_ARY_COUNT(connMng->urlTypeSet); i++) {
        urlTypeMng = connMng->urlTypeSet[i];
        if (urlTypeMng != VOS_NULL) {
            conn = BkfSuberConnDataGetFirstByUrlType(connMng, urlTypeMng->urlType, VOS_NULL);
            if (conn != VOS_NULL) {
                uint32_t connCloseCnt = BkfSuberSessDispCloseBatchTimeout(conn->sessMng, disp);
                closeCnt += connCloseCnt;
            }
        }
    }
    BKF_DISP_PRINTF(disp, " ***total %u conn(s) ***\n", closeCnt);
}

void BkfSuberConnDispBatchTimeoutTest(BkfSuberConnMng *connMng)
{
    BkfDisp *disp = connMng->env->disp;
    BkfSuberConnUrlTypeMng *urlTypeMng = VOS_NULL;
    BkfSuberConn *conn = VOS_NULL;
    uint32_t i;
    uint32_t testCnt = 0;
    for (i = BKF_URL_TYPE_INVALID + 1; i < BKF_GET_ARY_COUNT(connMng->urlTypeSet); i++) {
        urlTypeMng = connMng->urlTypeSet[i];
        if (urlTypeMng != VOS_NULL) {
            conn = BkfSuberConnDataGetFirstByUrlType(connMng, urlTypeMng->urlType, VOS_NULL);
            if (conn != VOS_NULL) {
                uint32_t connTestCnt = BkfSuberSessDispBatchTimeoutTest(conn->sessMng, disp);
                testCnt += connTestCnt;
            }
        }
    }
    BKF_DISP_PRINTF(disp, " ***total %u conn(s) ***\n", testCnt);
}

#endif

#if BKF_BLOCK("公有函数定义")
void BkfSuberConnDispInit(BkfSuberConnMng *connMng)
{
    BkfDisp *disp = connMng->env->disp;
    char *objName = connMng->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, connMng);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfSuberConnDispSummary, "disp suber conn summary", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfSuberConnDispAllConnAndSessSummary, "disp suber conn and sess summary", objName,
        0);
    (void)BKF_DISP_REG_FUNC(disp, BkfSuberConnDispAllConnFsm, "disp suber conn and sess summary", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfSuberConnDispAllSessFsm, "disp suber conn and sess summary", objName, 0);

    (void)BKF_DISP_REG_FUNC(disp, BkfSuberConnDispCloseBatchTimeout, "suber batch timeout chk close", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfSuberConnDispBatchTimeoutTest, "suber batch timeout chk test", objName, 0);
    return;
}

void BkfSuberConnDispUninit(BkfSuberConnMng *connMng)
{
    BkfDispUnregObj(connMng->env->disp, connMng->name);
    return;
}
#endif

#ifdef __cplusplus
}
#endif