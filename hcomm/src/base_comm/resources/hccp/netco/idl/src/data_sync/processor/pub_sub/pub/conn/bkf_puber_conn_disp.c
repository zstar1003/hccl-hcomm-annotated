/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_puber_conn_disp.h"
#include "bkf_puber_conn_data.h"
#include "bkf_str.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
typedef struct {
    int32_t connCnt;
    BkfChSerConnId *connId;
    int32_t sessCtxUsedLen;
} BkfPuberConnDispConnCtx;
#pragma pack()

STATIC int32_t BkfPuberConnDispGetConnCnt(BkfPuberConnMng *connMng)
{
    BkfPuberConn *conn = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t connCnt = 0;
    for (conn = BkfPuberConnGetFirst(connMng, &itor); conn != VOS_NULL; conn = BkfPuberConnGetNext(connMng, &itor)) {
        connCnt++;
    }

    return connCnt;
}

void BkfPuberConnDispSummary(BkfPuberConnMng *connMng, BkfDisp *disp)
{
    BKF_DISP_PRINTF(disp, "----conn----\n");
    BKF_DISP_PRINTF(disp, "name(%s/%d)/connCnt(%d)\n", connMng->name, BkfStrGetMemUsedLen(connMng->name),
        BkfPuberConnDispGetConnCnt(connMng));
    BKF_DISP_PRINTF(disp, "tableTypeMng(%#x)/chMng(%#x)/connFsmTmpl(%#x)/connCntSaved(%d)\n",
        BKF_MASK_ADDR(connMng->tableTypeMng), BKF_MASK_ADDR(connMng->argInit->chMng),
        BKF_MASK_ADDR(connMng->connFsmTmpl), connMng->connCnt);
    BkfDispTempCtx sessCtx;
    BKF_DISP_TEMP_CTX_INIT(&sessCtx);

    BkfPuberConn *conn = VOS_NULL;
    void *itor = VOS_NULL;
    for (conn = BkfPuberConnGetFirst(connMng, &itor); conn != VOS_NULL; conn = BkfPuberConnGetNext(connMng, &itor)) {
        BkfPuberSessDispGetSummary(conn->sessMng, &sessCtx);
    }
    BkfPuberSessDispPrintfSummary(disp, &sessCtx);
}

STATIC BkfPuberConnDispConnCtx *BkfPuberConnDispGetConnAndCtx(BkfPuberConnMng *connMng, BkfPuberConn **conn,
    BOOL *isNewConn, BkfDispTempCtx *lastSessCtx)
{
    *conn = VOS_NULL;
    *isNewConn = VOS_FALSE;
    BKF_DISP_TEMP_CTX_INIT(lastSessCtx);

    BkfPuberConnDispConnCtx *lastConnCtx = BKF_DISP_GET_LAST_CTX(connMng->argInit->disp, lastSessCtx);
    if (lastConnCtx == VOS_NULL) {
        *isNewConn = VOS_TRUE;
        *conn = BkfPuberConnGetFirst(connMng, VOS_NULL);
        return VOS_NULL;
    }

    if (lastConnCtx->sessCtxUsedLen > 0) {
        *conn = BkfPuberConnFind(connMng, lastConnCtx->connId);
    }

    if (*conn == VOS_NULL) {
        *isNewConn = VOS_TRUE;
        *conn = BkfPuberConnFindNext(connMng, lastConnCtx->connId);
    }

    return lastConnCtx;
}

void BkfPuberConnDispConnAndSess(BkfPuberConnMng *connMng)
{
    BkfDisp *disp = connMng->argInit->disp;
    BkfPuberConn *conn = VOS_NULL;
    BOOL isNewConn = VOS_FALSE;
    BkfDispTempCtx lastSessCtx = {{0}};
    BkfPuberConnDispConnCtx curConnCtx = {0};
    BkfPuberConnDispConnCtx *lastConnCtx = BkfPuberConnDispGetConnAndCtx(connMng, &conn, &isNewConn, &lastSessCtx);
    if (lastConnCtx == VOS_NULL) {
        BkfPuberConnDispSummary(connMng, disp);
        BKF_DISP_PRINTF(disp, "--------\n");
    } else {
        curConnCtx = *lastConnCtx;
    }

    if (conn == VOS_NULL) {
        BKF_DISP_PRINTF(disp, " ***total %d conn(s) ***\n", curConnCtx.connCnt);
        return;
    }

    BkfDispTempCtx curSessCtx;
    BKF_DISP_TEMP_CTX_INIT(&curSessCtx);
    if (isNewConn) {
        BKF_DISP_PRINTF(disp,
            "conn[%d], connId(%#x)/suber(%s)/state(%u, %s), rcv_sendBuf(%#x, %d/%#x, %d), "
            "jobId(%#x)/tmrId(%#x), lockDel(%u), slowSchedItor(%d), sessMng(%#x)\n",
            curConnCtx.connCnt, BKF_MASK_ADDR(conn->connId), conn->suberName, BKF_FSM_GET_STATE(&conn->fsm),
            BkfFsmGetStateStr(&conn->fsm), BKF_MASK_ADDR(conn->rcvDataBuf), BkfBufqGetUsedLen(conn->rcvDataBuf),
            BKF_MASK_ADDR(conn->lastSendLeftDataBuf), BkfBufqGetUsedLen(conn->lastSendLeftDataBuf),
            BKF_MASK_ADDR(conn->jobId), BKF_MASK_ADDR(conn->tmrId), conn->lockDel, conn->slowSchedGenProcCntItor,
            BKF_MASK_ADDR(conn->sessMng));
        BKF_DISP_PRINTF(disp, "--------\n");
        curConnCtx.connCnt++;
        curConnCtx.sessCtxUsedLen = BkfPuberSessDispSess(conn->sessMng, disp, VOS_NULL, &curSessCtx);
    } else {
        curConnCtx.sessCtxUsedLen = BkfPuberSessDispSess(conn->sessMng, disp, &lastSessCtx, &curSessCtx);
    }
    curConnCtx.connId = conn->connId;
    BKF_DISP_SAVE_CTX(disp, &curConnCtx, sizeof(curConnCtx), &curSessCtx, curConnCtx.sessCtxUsedLen);
}

STATIC void BkfPuberConnDispOneConnFsm(BkfPuberConn *conn, BkfPuberConnDispConnCtx *curConnCtx)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BkfDisp *disp = connMng->argInit->disp;
    uint8_t buf[BKF_DISP_PRINTF_BUF_LEN_MAX];
    BKF_DISP_PRINTF(disp, "conn[%d], connId(%#x), fsmInfo:\n", curConnCtx->connCnt, BKF_MASK_ADDR(conn->connId));
    BKF_DISP_PRINTF(disp, "%s\n", BkfFsmGetStr(&conn->fsm, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "--------\n");

    curConnCtx->connCnt++;
    curConnCtx->connId = conn->connId;
    BKF_DISP_SAVE_CTX(disp, curConnCtx, sizeof(BkfPuberConnDispConnCtx), VOS_NULL, 0);
}

void BkfPuberConnDispConnFsm(BkfPuberConnMng *connMng)
{
    BkfDisp *disp = connMng->argInit->disp;
    BkfPuberConnDispConnCtx curConnCtx = {0};
    BkfPuberConn *conn = VOS_NULL;

    BkfPuberConnDispConnCtx *lastConnCtx = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (lastConnCtx == VOS_NULL) {
        conn = BkfPuberConnGetFirst(connMng, VOS_NULL);
    } else {
        curConnCtx = *lastConnCtx;
        conn = BkfPuberConnFindNext(connMng, curConnCtx.connId);
    }

    if (conn == VOS_NULL) {
        BKF_DISP_PRINTF(disp, " ***total %d conn(s) ***\n", curConnCtx.connCnt);
        return;
    }

    BkfPuberConnDispOneConnFsm(conn, &curConnCtx);
}

STATIC void BkfPuberConnDispConnAndOneSessFsm(BkfPuberConn *conn, BOOL isNewConn, BkfDispTempCtx *lastSessCtx,
    BkfPuberConnDispConnCtx *curConnCtx)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BkfDisp *disp = connMng->argInit->disp;
    BkfDispTempCtx curSessCtx;
    BKF_DISP_TEMP_CTX_INIT(&curSessCtx);

    if (isNewConn) {
        BKF_DISP_PRINTF(disp, "conn[%d], connId(%#x):\n", curConnCtx->connCnt, BKF_MASK_ADDR(conn->connId));
        BKF_DISP_PRINTF(disp, "--------\n");
        curConnCtx->connCnt++;
        curConnCtx->sessCtxUsedLen = BkfPuberSessDispSessFsm(conn->sessMng, disp, VOS_NULL, &curSessCtx);
    } else {
        curConnCtx->sessCtxUsedLen = BkfPuberSessDispSessFsm(conn->sessMng, disp, lastSessCtx, &curSessCtx);
    }

    curConnCtx->connId = conn->connId;
    BKF_DISP_SAVE_CTX(disp, curConnCtx, sizeof(BkfPuberConnDispConnCtx), &curSessCtx, curConnCtx->sessCtxUsedLen);
}

void BkfPuberConnDispSessFsm(BkfPuberConnMng *connMng)
{
    BkfDisp *disp = connMng->argInit->disp;
    BkfPuberConn *conn = VOS_NULL;
    BOOL isNewConn = VOS_FALSE;
    BkfDispTempCtx lastSessCtx = {{0}};
    BkfPuberConnDispConnCtx curConnCtx = {0};
    BkfPuberConnDispConnCtx *lastConnCtx = BkfPuberConnDispGetConnAndCtx(connMng, &conn, &isNewConn, &lastSessCtx);
    if (lastConnCtx != VOS_NULL) {
        curConnCtx = *lastConnCtx;
    }

    if (conn == VOS_NULL) {
        BKF_DISP_PRINTF(disp, " ***total %d conn(s) ***\n", curConnCtx.connCnt);
        return;
    }

    BkfPuberConnDispConnAndOneSessFsm(conn, isNewConn, &lastSessCtx, &curConnCtx);
}

void BkfPuberConnDispCloseBatchTimeout(BkfPuberConnMng *connMng)
{
    BkfDisp *disp = connMng->argInit->disp;

    BkfPuberConn *conn = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t closeCnt = 0;
    for (conn = BkfPuberConnGetFirst(connMng, &itor); conn != VOS_NULL; conn = BkfPuberConnGetNext(connMng, &itor)) {
        uint32_t oneConnCnt = BkfPuberSessDispCloseBatchTimeout(conn->sessMng, disp);
        closeCnt += oneConnCnt;
    }
    BKF_DISP_PRINTF(disp, " ***total %u conn(s) ***\n", closeCnt);
}

void BkfPuberConnDispSessBatchTimeoutTest(BkfPuberConnMng *connMng)
{
    BkfDisp *disp = connMng->argInit->disp;
    BkfPuberConn *conn = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t testCnt = 0;
    for (conn = BkfPuberConnGetFirst(connMng, &itor); conn != VOS_NULL; conn = BkfPuberConnGetNext(connMng, &itor)) {
        uint32_t oneConnCnt = BkfPuberSessDispBatchTimeoutTest(conn->sessMng, disp);
        testCnt += oneConnCnt;
    }
    BKF_DISP_PRINTF(disp, " ***total %u conn(s) ***\n", testCnt);
}

void BkfPuberConnDispInit(BkfPuberConnMng *connMng)
{
    BkfDisp *disp = connMng->argInit->disp;
    char *objName = connMng->name;
    uint32_t ret = BkfDispRegObj(disp, objName, connMng);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfPuberConnDispConnAndSess, "disp puber conn info", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfPuberConnDispConnFsm, "disp puber conn fsm info", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfPuberConnDispSessFsm, "disp puber sess fsm info", objName, 0);

    (void)BKF_DISP_REG_FUNC(disp, BkfPuberConnDispCloseBatchTimeout, "puber sess close timeout chk", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfPuberConnDispSessBatchTimeoutTest, "puber sess timeout chk test", objName, 0);
    connMng->dispInitOk = VOS_TRUE;
}

void BkfPuberConnDispUninit(BkfPuberConnMng *connMng)
{
    if (connMng->dispInitOk) {
        BkfDispUnregObj(connMng->argInit->disp, connMng->name);
    }
}

#ifdef __cplusplus
}
#endif
