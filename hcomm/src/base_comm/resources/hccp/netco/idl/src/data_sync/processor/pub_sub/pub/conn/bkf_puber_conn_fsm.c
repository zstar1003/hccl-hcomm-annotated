/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((connMng)->log)
#define BKF_MOD_NAME ((connMng)->name)
#define BKF_LINE (__LINE__ + 10000)

#include "bkf_puber_conn_fsm.h"
#include "bkf_puber_conn_utils.h"
#include "bkf_ch_ser.h"
#include "bkf_prog.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
#define BKF_PUBER_CONN_SLOW_SCHED_PROC_CNT_MAX (1024 * 1)

/* fsm proc */
STATIC uint32_t BkfPuberConnProcHello(BkfPuberConn *conn, BkfMsgDecoder *decoder, void *unused2);
STATIC uint32_t BkfPuberConnTry2SendHelloAck(BkfPuberConn *conn, void *unused1, void *unused2);
STATIC uint32_t BkfPuberConnJobSchedSess(BkfPuberConn *conn, void *unused1, void *unused2);
STATIC uint32_t BkfPuberConnTmrSchedSess(BkfPuberConn *conn, void *unused1, void *unused2);

/* other proc */
STATIC void BkfPuberConnInitSendBufAndCoder(BkfPuberConn *conn, uint8_t **sendBuf, BkfMsgCoder *coder);
STATIC uint32_t BkfPuberConnWriteSendBufByCoder(BkfPuberConn *conn, uint8_t **sendBuf, BkfMsgCoder *coder);
STATIC uint32_t BkfPuberConnWriteLastLeftData(BkfPuberConn *conn);
STATIC uint32_t BkfPuberConnDoSchedSess(BkfPuberConn *conn, BOOL isSlowSched, void *ctx);

#endif

#if BKF_BLOCK("表")
const char *g_BkfPuberConnFsmStateStrTbl[] = {
    "BKF_PUBER_CONN_STATE_WAIT_HELLO",
    "BKF_PUBER_CONN_STATE_WAIT_SEND_HELLO_ACK",
    "BKF_PUBER_CONN_STATE_CONN",
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfPuberConnFsmStateStrTbl) == BKF_PUBER_CONN_STATE_CNT);

const char *g_BkfPuberConnFsmEvtStrTbl[] = {
    "BKF_PUBER_CONN_EVT_HELLO",
    "BKF_PUBER_CONN_EVT_UNDERLAY_UNBLOCK",
    "BKF_PUBER_CONN_EVT_JOB",
    "BKF_PUBER_CONN_EVT_TMR",
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfPuberConnFsmEvtStrTbl) == BKF_PUBER_CONN_EVT_CNT);

const BkfFsmProcItem g_BkfPuberConnFsm[][BKF_PUBER_CONN_EVT_CNT] = {
    /* BKF_PUBER_CONN_STATE_WAIT_HELLO */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberConnProcHello)},   /* BKF_PUBER_CONN_EVT_HELLO */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BKF_PUBER_CONN_EVT_UNDERLAY_UNBLOCK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)}, /* BKF_PUBER_CONN_EVT_JOB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)}, /* BKF_PUBER_CONN_EVT_TMR */
    },
    /* BKF_PUBER_CONN_STATE_WAIT_SEND_HELLO_ACK */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},      /* BKF_PUBER_CONN_EVT_HELLO */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberConnTry2SendHelloAck)}, /* BKF_PUBER_CONN_EVT_UNDERLAY_UNBLOCK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberConnTry2SendHelloAck)}, /* BKF_PUBER_CONN_EVT_JOB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},      /* BKF_PUBER_CONN_EVT_TMR */
    },
    /* BKF_PUBER_CONN_STATE_CONN */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BKF_PUBER_CONN_EVT_HELLO */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberConnJobSchedSess)}, /* BKF_PUBER_CONN_EVT_UNDERLAY_UNBLOCK ，快调度优先 */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberConnJobSchedSess)}, /* BKF_PUBER_CONN_EVT_JOB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberConnTmrSchedSess)}, /* BKF_PUBER_CONN_EVT_TMR */
    },
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfPuberConnFsm) == BKF_PUBER_CONN_STATE_CNT);

#endif

#if BKF_BLOCK("公有函数定义")
BkfFsmTmpl *BkfPuberConnFsmTmplInit(BkfPuberConnMng *connMng)
{
    char tempName[BKF_NAME_LEN_MAX + 1];
    int32_t err = snprintf_truncated_s(tempName, sizeof(tempName), "%s_ConnFsmTmpl", connMng->argInit->name);
    if (err <= 0) {
        return VOS_NULL;
    }

    BkfFsmTmplInitArg arg = {0};
    arg.name = tempName;
    arg.dbgOn = connMng->argInit->dbgOn;
    arg.memMng = connMng->argInit->memMng;
    arg.disp = connMng->argInit->disp;
    arg.log = connMng->log;
    arg.stateCnt = BKF_PUBER_CONN_STATE_CNT;
    arg.stateInit = BKF_PUBER_CONN_STATE_WAIT_HELLO;
    arg.evtCnt = BKF_PUBER_CONN_EVT_CNT;
    arg.stateStrTbl = g_BkfPuberConnFsmStateStrTbl;
    arg.evtStrTbl = g_BkfPuberConnFsmEvtStrTbl;
    arg.stateEvtProcItemMtrx = (const BkfFsmProcItem *)g_BkfPuberConnFsm;
    arg.getAppDataStrOrNull = VOS_NULL;
    return BkfFsmTmplInit(&arg);
}

uint32_t BkfPuberConnFsmProc(BkfPuberConn *conn, uint8_t evtId, void *param)
{
    BkfPuberConnMng *connMng = conn->connMng;
    uint32_t ret = BKF_FSM_DISPATCH(&conn->fsm, evtId, conn, param, VOS_NULL);
    if ((ret == BKF_PUBER_CONN_NEED_DELETE) || (ret == BKF_FSM_DISPATCH_RET_IMPOSSIBLE)) {
        BKF_LOG_WARN(BKF_LOG_HND, "conn(%#x)/evtId(%u, %s), ret(%u), 2del conn\n", BKF_MASK_ADDR(conn), evtId,
            BkfFsmTmplGetEvtStr(connMng->connFsmTmpl, evtId), ret);
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "conn(%#x)/evtId(%u, %s), ret(%u)\n", BKF_MASK_ADDR(conn), evtId,
        BkfFsmTmplGetEvtStr(connMng->connFsmTmpl, evtId), ret);
    return ret;
}

BOOL BkfPuberMaySchedSess(void *puberConn)
{
    (void)puberConn;
    return VOS_TRUE; /* 可测试性考虑 */
}

#endif

#if BKF_BLOCK("私有函数定义")
STATIC uint32_t BkfPuberConnProcHello(BkfPuberConn *conn, BkfMsgDecoder *decoder, void *unused2)
{
    (void)unused2;
    BkfPuberConnMng *connMng = conn->connMng;
    BkfWrapMsgHello hello = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint32_t ret = BkfPuberConnParseHello(conn, decoder, &hello);
    ret |= BkfPuberConnChkHello(conn, &hello);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    errno_t err = strcpy_s(conn->suberName, sizeof(conn->suberName), hello.suberName->name);
    if (err != EOK) {
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    if (BkfPuberConnStartJob(conn) != BKF_OK) {
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    if (hello.suberLsnUrl) {
        conn->cliLsnUrl = hello.suberLsnUrl->lsnUrl;
        if (conn->cliLsnUrl.type == BKF_URL_TYPE_V8TLS) {
            BkfUrlTlsSet(&conn->cliLsnUrl, hello.suberLsnUrl->lsnUrl.ip.addrH, hello.suberLsnUrl->lsnUrl.ip.port);
        } else if (conn->cliLsnUrl.type == BKF_URL_TYPE_V8TCP || conn->cliLsnUrl.type == BKF_URL_TYPE_LETCP) {
            BkfUrlV8TcpSet(&conn->cliLsnUrl, hello.suberLsnUrl->lsnUrl.ip.addrH, hello.suberLsnUrl->lsnUrl.ip.port);
        }
        BKF_LOG_DEBUG(BKF_LOG_HND, "hello has lsnUrl, type %u\n", hello.suberLsnUrl->lsnUrl.type);
    }

    if (connMng->argInit->onConnect) {
        uint8_t cliStr[BKF_URL_STR_LEN_MAX];
        uint8_t lsnUrl[BKF_URL_STR_LEN_MAX];
        BKF_LOG_DEBUG(BKF_LOG_HND, "onConnect App: localUrl %s, cliUrl %s cliLsnUrl %s\n",
            BkfUrlGetStr(&conn->localUrl, urlStr, sizeof(urlStr)), BkfUrlGetStr(&conn->cliUrl, cliStr, sizeof(cliStr)),
            BkfUrlGetStr(&conn->cliLsnUrl, lsnUrl, sizeof(lsnUrl)));
        connMng->argInit->onConnect(connMng->argInit->cookie, &conn->localUrl, &conn->cliUrl, &conn->cliLsnUrl);
    }

    (void)BKF_FSM_CHG_STATE(&conn->fsm, BKF_PUBER_CONN_STATE_WAIT_SEND_HELLO_ACK);
    return BKF_OK;
}

STATIC uint32_t BkfPuberConnTry2SendHelloAck(BkfPuberConn *conn, void *unused1, void *unused2)
{
    (void)unused1;
    (void)unused2;
    uint8_t *sendBuf = VOS_NULL;
    BkfMsgCoder coder = {};
    BkfPuberConnInitSendBufAndCoder(conn, &sendBuf, &coder);
    if (sendBuf == VOS_NULL) {
        return BKF_ERR; /* 等待解阻塞 */
    }

    uint32_t ret = BkfPuberConnPackHelloAck(conn, &coder);
    if (ret != BKF_OK) {
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    /*
    已经正常组包。即使部分写成功，都已经缓冲在发送缓冲中。
    因此，后续只需要处理need del异常，其余正常切换状态
    */
    ret = BkfPuberConnWriteSendBufByCoder(conn, &sendBuf, &coder);
    if (ret == BKF_PUBER_CONN_NEED_DELETE) {
        return ret;
    }

    (void)BKF_FSM_CHG_STATE(&conn->fsm, BKF_PUBER_CONN_STATE_CONN);
    if (BkfPuberConnStartJob(conn) != BKF_OK) { /* 启动job，无条件调度一次 */
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    return BKF_OK;
}

STATIC uint32_t BkfPuberConnJobSchedSess(BkfPuberConn *conn, void *unused1, void *unused2)
{
    (void)unused1;
    (void)unused2;
    /* 尽力发送，直到阻塞、发完或者错误。对于第一种情况，底层会上报解阻塞以继续发送 */
    return BkfPuberConnDoSchedSess(conn, VOS_FALSE, VOS_NULL);
}

STATIC uint32_t BkfPuberConnTmrSchedSess(BkfPuberConn *conn, void *unused1, void *unused2)
{
    (void)unused1;
    (void)unused2;
    BkfPuberConnMng *connMng = conn->connMng;
    BkfPuberSessSlowSchedCtx ctx = {0};

    /* 控制发送，直到阻塞、发完、让权或者错误 */
    if (connMng->argInit->verifyMayAccelerate(connMng->argInit->cookie)) {
        ctx.procCntMax = BkfFibo32GetNextValid(&conn->slowSchedGenProcCntItor, BKF_PUBER_CONN_SLOW_SCHED_PROC_CNT_MAX);
    } else {
        ctx.procCntMax = BkfFibo32GetPrevValid(&conn->slowSchedGenProcCntItor, BKF_PUBER_CONN_SLOW_SCHED_PROC_CNT_MAX);
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "cntMax(%d)\n", ctx.procCntMax);

    uint32_t ret = BkfPuberConnDoSchedSess(conn, VOS_TRUE, &ctx);
    if (ret == BKF_PUBER_SESS_SCHED_FINISH) {
        BkfPuberConnStopTmr(conn);
    }
    return ret;
}

STATIC void BkfPuberConnInitSendBufAndCoder(BkfPuberConn *conn, uint8_t **sendBuf, BkfMsgCoder *coder)
{
    BkfPuberConnMng *connMng = conn->connMng;
    *sendBuf = BkfChSerMallocDataBuf(connMng->argInit->chMng, conn->connId, BKF_CH_SER_MALLOC_DATA_BUF_LEN_MAX);
    if (*sendBuf != VOS_NULL) {
        BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(connMng->argInit->dc);
        (void)BkfMsgCodeInit(coder, connMng->argInit->name, *sendBuf, BKF_CH_SER_MALLOC_DATA_BUF_LEN_MAX,
            sliceVTbl->keyLen, sliceVTbl->keyCodec, sliceVTbl->keyGetStrOrNull, BKF_LOG_HND);
    }
}

STATIC uint32_t BkfPuberConnSaveWriteLeftData(BkfPuberConn *conn, uint8_t **sendBuf, int32_t dataLen, int32_t writeLen)
{
    BkfPuberConnMng *connMng = conn->connMng;
    int32_t leftLen = dataLen - writeLen;
    conn->lastSendLeftDataBuf = BkfBufqInit(connMng->argInit->memMng, leftLen);
    if (conn->lastSendLeftDataBuf == VOS_NULL) {
        BkfChSerFreeDataBuf(connMng->argInit->chMng, conn->connId, *sendBuf);
        *sendBuf = VOS_NULL;
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    (void)BkfBufqEn(conn->lastSendLeftDataBuf, *sendBuf + writeLen, leftLen);
    BkfChSerFreeDataBuf(connMng->argInit->chMng, conn->connId, *sendBuf);
    *sendBuf = VOS_NULL;
    return BKF_ERR;
}

STATIC uint32_t BkfPuberConnWriteSendBufByCoder(BkfPuberConn *conn, uint8_t **sendBuf, BkfMsgCoder *coder)
{
    BkfPuberConnMng *connMng = conn->connMng;
    if (conn->lastSendLeftDataBuf != VOS_NULL) {
        /* 忘记先发送上次未写成功的数据就发送新数据，错误，断言 */
        BKF_ASSERT(0);
        BkfChSerFreeDataBuf(connMng->argInit->chMng, conn->connId, *sendBuf);
        *sendBuf = VOS_NULL;
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    int32_t dataLen = BkfMsgCodeGetValidMsgLen(coder);
    int32_t writeLen = BkfChSerWrite(connMng->argInit->chMng, conn->connId, *sendBuf, dataLen);
    if (writeLen == dataLen) {
        *sendBuf = VOS_NULL;
        return BKF_OK;
    }

    if ((writeLen >= 0) && (writeLen < dataLen)) {
        return BkfPuberConnSaveWriteLeftData(conn, sendBuf, dataLen, writeLen);
    }

    BKF_ASSERT(0);
    *sendBuf = VOS_NULL;
    return BKF_PUBER_CONN_NEED_DELETE;
}

STATIC uint32_t BkfPuberConnWriteLastLeftData(BkfPuberConn *conn)
{
    if (conn->lastSendLeftDataBuf == VOS_NULL) {
        return BKF_OK;
    }

    int32_t leftLen;
    while ((leftLen = BkfBufqGetUsedLen(conn->lastSendLeftDataBuf)) > 0) {
        BkfPuberConnMng *connMng = conn->connMng;
        int32_t dataLen = BKF_GET_MIN(leftLen, BKF_CH_SER_MALLOC_DATA_BUF_LEN_MAX);
        uint8_t *sendBuf = BkfChSerMallocDataBuf(connMng->argInit->chMng, conn->connId, dataLen);
        if (sendBuf == VOS_NULL) {
            return BKF_ERR;
        }

        (void)memcpy_s(sendBuf, dataLen, BkfBufqGetUsedBegin(conn->lastSendLeftDataBuf), dataLen);
        int32_t writeLen = BkfChSerWrite(connMng->argInit->chMng, conn->connId, sendBuf, dataLen);
        if (writeLen == dataLen) {
            (void)BkfBufqDe(conn->lastSendLeftDataBuf, writeLen);
            sendBuf = VOS_NULL;
            continue; /* 继续从缓冲中读取数据&发送 */
        }

        if ((writeLen >= 0) && (writeLen < dataLen)) {
            (void)BkfBufqDe(conn->lastSendLeftDataBuf, writeLen);
            BkfChSerFreeDataBuf(connMng->argInit->chMng, conn->connId, sendBuf);
            sendBuf = VOS_NULL;
            return BKF_ERR; /* 无法发送更多 */
        }

        BKF_ASSERT(0);
        return BKF_PUBER_CONN_NEED_DELETE; /* 异常保护，进行删除 */
    }

    BkfBufqUninit(conn->lastSendLeftDataBuf);
    conn->lastSendLeftDataBuf = VOS_NULL;
    return BKF_OK;
}

/*
1. ok: 当前buf发完，conn申请新sndBuf发送数据
2. err: 当前buf未发完 ，conn等待解阻塞
3. connNeedDel: 外部删除conn
*/
static inline uint32_t BkfPuberConnProcSessSchedRetSndBufNotEnough(BkfPuberConn *conn, uint8_t **sendBuf,
    BkfMsgCoder *coder)
{
    return BkfPuberConnWriteSendBufByCoder(conn, sendBuf, coder);
}

/*
1. ok: 当前buf发完，所有sess调度完成，conn停止发送，返回转成finish
2. err: 当前buf未发完 ，conn等待解阻塞
3. connNeedDel: 外部删除conn
*/
static inline uint32_t BkfPuberConnProcSessSchedRetFinish(BkfPuberConn *conn, uint8_t **sendBuf, BkfMsgCoder *coder)
{
    uint32_t ret = BkfPuberConnWriteSendBufByCoder(conn, sendBuf, coder);
    return (ret == BKF_OK) ? BKF_PUBER_SESS_SCHED_FINISH : ret;
}

static inline uint32_t BkfPuberConnProcSessSchedRetFatalErr(BkfPuberConn *conn, uint8_t **sendBuf)
{
    /* 释放sendBuf & 返回转成connNeeddel */
    BkfPuberConnMng *connMng = conn->connMng;
    BkfChSerFreeDataBuf(connMng->argInit->chMng, conn->connId, *sendBuf);
    *sendBuf = VOS_NULL;
    return BKF_PUBER_CONN_NEED_DELETE;
}

/*
1. ok: 当前buf发完，conn放权等待定时器超时再次发送，返回转成yield
2. err: 当前buf未发完 ，conn等待解阻塞
3. connNeedDel: 外部删除conn
*/
static inline uint32_t BkfPuberConnProcSessSchedRetYield(BkfPuberConn *conn, uint8_t **sendBuf, BkfMsgCoder *coder,
    BOOL isSlowSched)
{
    BkfPuberConnMng *connMng = conn->connMng;
    if (!isSlowSched) { /* 只有慢调度才会返回yield */
        BKF_ASSERT(0);
        BkfChSerFreeDataBuf(connMng->argInit->chMng, conn->connId, *sendBuf);
        *sendBuf = VOS_NULL;
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    uint32_t ret = BkfPuberConnWriteSendBufByCoder(conn, sendBuf, coder);
    return (ret == BKF_OK) ? BKF_PUBER_SESS_SCHED_YIELD : ret;
}

STATIC uint32_t BkfPuberConnProcSessSchedRet(BkfPuberConn *conn, uint8_t **sendBuf, BkfMsgCoder *coder,
    BOOL isSlowSched, uint32_t retSchedSess)
{
    switch (retSchedSess) {
        case BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH: {
            return BkfPuberConnProcSessSchedRetSndBufNotEnough(conn, sendBuf, coder);
        }
        case BKF_PUBER_SESS_SCHED_FINISH: {
            return BkfPuberConnProcSessSchedRetFinish(conn, sendBuf, coder);
        }
        case BKF_PUBER_SESS_FATAL_ERR: {
            return BkfPuberConnProcSessSchedRetFatalErr(conn, sendBuf);
        }
        case BKF_PUBER_SESS_SCHED_YIELD: {
            return BkfPuberConnProcSessSchedRetYield(conn, sendBuf, coder, isSlowSched);
        }
        default: {
            BkfPuberConnMng *connMng = conn->connMng;
            BKF_LOG_DEBUG(BKF_LOG_HND, "retSchedSess(%u)\n", retSchedSess);
            return BKF_OK;
        }
    }
}

STATIC uint32_t BkfPuberConnDoSchedSess(BkfPuberConn *conn, BOOL isSlowSched, void *ctx)
{
    uint32_t ret = BkfPuberConnWriteLastLeftData(conn);
    if (ret != BKF_OK) {
        return ret;
    }

    BkfPuberConnMng *connMng = conn->connMng;
    uint8_t *sendBuf = VOS_NULL;
    BkfMsgCoder coder = {};
    do {
        BOOL isFastSchedInTest = !isSlowSched && !BkfPuberMaySchedSess(conn);
        if (isFastSchedInTest) {
            BKF_LOG_DEBUG(BKF_LOG_HND, "stop sched for test\n");
            (void)BkfPuberConnStartJob(conn);
            ret = (sendBuf != VOS_NULL) ? BkfPuberConnWriteSendBufByCoder(conn, &sendBuf, &coder) : BKF_OK;
            break;
        }

        if (sendBuf == VOS_NULL) {
            BkfPuberConnInitSendBufAndCoder(conn, &sendBuf, &coder);
            if (sendBuf == VOS_NULL) { /* 等待解阻塞 */
                ret = BKF_ERR;
                break;
            }
        }

        ret = BkfPuberSessProcSched(conn->sessMng, &coder, isSlowSched, ctx);
        ret = BkfPuberConnProcSessSchedRet(conn, &sendBuf, &coder, isSlowSched, ret);
    } while (ret == BKF_OK);

    /* 由此退出，sendBuf一定为空 */
    if (sendBuf != VOS_NULL) {
        BKF_ASSERT(0);
        BkfChSerFreeDataBuf(connMng->argInit->chMng, conn->connId, sendBuf);
        sendBuf = VOS_NULL;
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    return ret;
}

#endif

#ifdef __cplusplus
}
#endif
