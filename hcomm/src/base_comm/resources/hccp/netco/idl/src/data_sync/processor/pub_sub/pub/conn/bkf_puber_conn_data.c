/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_puber_conn_data.h"
#include "bkf_ch_ser.h"
#include "bkf_str.h"
#include "bkf_prog.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

BkfPuberConnMng *BkfPuberConnDataInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng)
{
    uint32_t len = sizeof(BkfPuberConnMng);
    BkfPuberConnMng *connMng = BKF_MALLOC(arg->memMng, len);
    if (connMng == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(connMng, len, 0, len);
    connMng->argInit = arg;
    connMng->name = BkfStrNew(arg->memMng, "%s_conn", arg->name);
    if (connMng->name == VOS_NULL) {
        BKF_FREE(arg->memMng, connMng);
        return VOS_NULL;
    }
    connMng->log = arg->log;

    connMng->tableTypeMng = tableTypeMng;
    VOS_AVLL_INIT_TREE(connMng->connSet, (AVLL_COMPARE)BkfCmpPtrAddr, BKF_OFFSET(BkfPuberConn, connId),
        BKF_OFFSET(BkfPuberConn, avlNode));
    return connMng;
}

void BkfPuberConnDataUninit(BkfPuberConnMng *connMng)
{
    BkfPuberConnDelAll(connMng);
    BkfStrDel(connMng->name);
    BKF_FREE(connMng->argInit->memMng, connMng);
}

STATIC void BkfPuberConnAddProcError(BkfPuberConnMng *connMng, BkfPuberConn *conn, BOOL fsmInitOk)
{
    if (conn->sessMng != VOS_NULL) {
        BkfPuberSessUninit(conn->sessMng);
    }

    if (fsmInitOk) {
        BkfFsmUninit(&conn->fsm);
    }

    BKF_FREE(connMng->argInit->memMng, conn);
}

BkfPuberConn *BkfPuberConnAdd(BkfPuberConnMng *connMng, BkfChSerConnId *connId,
    F_BKF_PUBER_SESS_TRIG_SCHED_SELF trigSchedSess, F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF trigSlowSchedSess)
{
    uint32_t len = sizeof(BkfPuberConn);
    BkfPuberConn *conn = BKF_MALLOC(connMng->argInit->memMng, len);
    if (conn == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(conn, len, 0, len);
    conn->connMng = connMng;
    conn->connId = connId;
    if (BkfFsmInit(&conn->fsm, connMng->connFsmTmpl) != BKF_OK) {
        BkfPuberConnAddProcError(connMng, conn, VOS_FALSE);
        return VOS_NULL;
    }

    BKF_PROG_INIT_ITOR_FIRST(&conn->slowSchedGenProcCntItor);
    conn->sessMng = BkfPuberSessInit(connMng->argInit, connMng->tableTypeMng, trigSchedSess, trigSlowSchedSess, conn);
    if (conn->sessMng == VOS_NULL) {
        BkfPuberConnAddProcError(connMng, conn, VOS_TRUE);
        return VOS_NULL;
    }

    VOS_AVLL_INIT_NODE(conn->avlNode);
    if (!VOS_AVLL_INSERT(connMng->connSet, conn->avlNode)) {
        BkfPuberConnAddProcError(connMng, conn, VOS_TRUE);
        return VOS_NULL;
    }

    connMng->connCnt++;
    return conn;
}

void BkfPuberConnDel(BkfPuberConn *conn, BOOL ntfCh)
{
    BkfPuberConnMng *connMng = conn->connMng;
    if (BkfPuberConnIsLockDel(conn)) {
        BKF_ASSERT(0);
        return;
    }

    /* sess */
    BkfPuberSessUninit(conn->sessMng);
    conn->sessMng = VOS_NULL;

    /* self */
    if (ntfCh) {
        BkfChSerClose(connMng->argInit->chMng, conn->connId);
    }

    conn->connId = VOS_NULL;
    if (conn->rcvDataBuf != VOS_NULL) {
        BkfBufqUninit(conn->rcvDataBuf);
        conn->rcvDataBuf = VOS_NULL;
    }

    if (conn->lastSendLeftDataBuf != VOS_NULL) {
        BkfBufqUninit(conn->lastSendLeftDataBuf);
        conn->lastSendLeftDataBuf = VOS_NULL;
    }

    BkfPuberConnStopJob(conn);
    BkfPuberConnStopTmr(conn);
    BkfFsmUninit(&conn->fsm);
    VOS_AVLL_DELETE(connMng->connSet, conn->avlNode);
    BKF_FREE(connMng->argInit->memMng, conn);
    connMng->connCnt--;
}

void BkfPuberConnDelAll(BkfPuberConnMng *connMng)
{
    BkfPuberConn *conn = VOS_NULL;
    while ((conn = VOS_AVLL_FIRST(connMng->connSet)) != VOS_NULL) {
        BkfPuberConnDel(conn, VOS_TRUE);
    }
}

BkfPuberConn *BkfPuberConnFind(BkfPuberConnMng *connMng, BkfChSerConnId *connId)
{
    return VOS_AVLL_FIND(connMng->connSet, &connId);
}

BkfPuberConn *BkfPuberConnFindNext(BkfPuberConnMng *connMng, BkfChSerConnId *connId)
{
    return VOS_AVLL_FIND_NEXT(connMng->connSet, &connId);
}

BkfPuberConn *BkfPuberConnGetFirst(BkfPuberConnMng *connMng, void **itorOutOrNull)
{
    BkfPuberConn *conn = VOS_AVLL_FIRST(connMng->connSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (conn != VOS_NULL) ? VOS_AVLL_NEXT(connMng->connSet, conn->avlNode) : VOS_NULL;
    }
    return conn;
}

BkfPuberConn *BkfPuberConnGetNext(BkfPuberConnMng *connMng, void **itorInOut)
{
    BkfPuberConn *conn = (*itorInOut);
    *itorInOut = (conn != VOS_NULL) ? VOS_AVLL_NEXT(connMng->connSet, conn->avlNode) : VOS_NULL;
    return conn;
}

uint32_t BkfPuberConnStartJob(BkfPuberConn *conn)
{
    if (conn->jobId == VOS_NULL) {
        BkfPuberConnMng *connMng = conn->connMng;
        conn->jobId = BkfJobCreate(connMng->argInit->jobMng, connMng->argInit->jobTypeId, "jobIdSched", conn);
        if (conn->jobId == VOS_NULL) {
            BKF_ASSERT(0);
            return BKF_ERR;
        }
    }
    return BKF_OK;
}

void BkfPuberConnStopJob(BkfPuberConn *conn)
{
    if (conn->jobId != VOS_NULL) {
        BkfPuberConnMng *connMng = conn->connMng;
        BkfJobDelete(connMng->argInit->jobMng, conn->jobId);
        conn->jobId = VOS_NULL;
    }
}

uint32_t BkfPuberConnStartTmr(BkfPuberConn *conn, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs)
{
    if (conn->tmrId == VOS_NULL) {
        BkfPuberConnMng *connMng = conn->connMng;
        conn->tmrId = BkfTmrStartLoop(connMng->argInit->tmrMng, proc, intervalMs, conn);
    }
    return (conn->tmrId != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void BkfPuberConnStopTmr(BkfPuberConn *conn)
{
    if (conn->tmrId != VOS_NULL) {
        BkfPuberConnMng *connMng = conn->connMng;
        BkfTmrStop(connMng->argInit->tmrMng, conn->tmrId);
        conn->tmrId = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
