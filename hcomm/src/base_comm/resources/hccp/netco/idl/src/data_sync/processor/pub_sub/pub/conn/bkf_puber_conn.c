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

#include "bkf_puber_conn.h"
#include "bkf_puber_conn_data.h"
#include "bkf_puber_conn_fsm.h"
#include "bkf_puber_conn_disp.h"
#include "bkf_puber_conn_sys_log.h"
#include "bkf_ch_ser.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
#define BKF_PUBER_CONN_TMR_INTERVAL 1002

/* on msg */
STATIC BOOL BkfPuberConnOnMayAccept(BkfPuberConnMng *connMng, BkfUrl *urlCli);
STATIC void BkfPuberConnOnRcvData(BkfPuberConnMng *connMng, BkfChSerConnId *connId, uint8_t *rcvData, int32_t dataLen);
STATIC void BkfPuberConnOnRcvDataEvent(BkfPuberConnMng *connMng, BkfChSerConnId *connId);
STATIC void BkfPuberConnOnUnblock(BkfPuberConnMng *connMng, BkfChSerConnId *connId);
STATIC void BkfPuberConnOnDisconn(BkfPuberConnMng *connMng, BkfChSerConnId *connId);
STATIC void BkfPuberConnOnTrigSchedSess(BkfPuberConn *conn);
STATIC uint32_t BkfPuberConnOnJob(BkfPuberConn *conn, uint32_t *runCost);
STATIC void BkfPuberConnOnTrigSlowSchedSess(BkfPuberConn *conn);
STATIC uint32_t BkfPuberConnOnTmrTO(BkfPuberConn *conn, void *paramTmrLibUnknown);

/* proc */
STATIC BkfPuberConn *BkfPuberConnDelIfRetNeedDel(BkfPuberConn *conn, uint32_t ret);
STATIC uint32_t BkfPuberConnProcRcvData(BkfPuberConn *conn, uint8_t *rcvData, int32_t dataLen);
STATIC uint32_t BkfPuberConnInitDecoderForRcvData(BkfPuberConn *conn, uint8_t *rcvData, int32_t dataLen,
    BkfMsgDecoder *decoder);
STATIC uint32_t BkfPuberConnProcDataInDecoder(BkfPuberConn *conn, BkfMsgDecoder *decoder);
STATIC uint32_t BkfPuberConnSaveDecoderLeftData(BkfPuberConn *conn, BkfMsgDecoder *decoder, BOOL delBufqIfNoLeftData);
STATIC uint32_t BkfPuberConnProcRcvDataEvent(BkfPuberConn *conn);
STATIC uint32_t BkfPuberConnInitDecoderForRcvDataEvent(BkfPuberConn *conn, BkfMsgDecoder *decoder);
STATIC int32_t BkfPuberConnReadData2Bufq(BkfPuberConn *conn);

#endif
#if BKF_BLOCK("私有函数定义")
STATIC BOOL BkfPuberConnOnMayAccept(BkfPuberConnMng *connMng, BkfUrl *urlCli)
{
    if (connMng->connCnt < connMng->argInit->connCntMax) {
        return VOS_TRUE;
    }

    if (connMng->argInit->onConnectOver) {
        connMng->argInit->onConnectOver(connMng->argInit->cookie, urlCli);
    }
    uint8_t buf[BKF_LOG_LEN] = {0};
    BKF_LOG_WARN(BKF_LOG_HND, "connMng(%#x), %d/%d, urlCli(%s), can not accept more conn\n", BKF_MASK_ADDR(connMng),
        connMng->connCnt, connMng->argInit->connCntMax, BkfUrlGetStr(urlCli, buf, sizeof(buf)));
    (void)BkfPuberConnLimitSysLog(connMng, urlCli);
    return VOS_FALSE;
}

STATIC void BkfPuberConnOnConn(BkfPuberConnMng *connMng, BkfChSerConnId *connId, BkfUrl *cliUrl)
{
    BKF_LOG_DEBUG(BKF_LOG_HND, "connMng(%#x), connId(%#x)\n", BKF_MASK_ADDR(connMng), BKF_MASK_ADDR(connId));

    BkfPuberConn *conn = BkfPuberConnFind(connMng, connId);
    if (conn != VOS_NULL) {
        /* 重复通知，错误 */
        BKF_LOG_ERROR(BKF_LOG_HND, "connId(%#x)/conn(%#x)\n", BKF_MASK_ADDR(connId), BKF_MASK_ADDR(conn));
        BkfPuberConnDel(conn, VOS_TRUE);
        return;
    }

    conn = BkfPuberConnAdd(connMng, connId, (F_BKF_PUBER_SESS_TRIG_SCHED_SELF)BkfPuberConnOnTrigSchedSess,
        (F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF)BkfPuberConnOnTrigSlowSchedSess);
    if (conn) {
        conn->cliUrl = *cliUrl;
    }
}

STATIC void BkfPuberConnOnConnEx(BkfPuberConnMng *connMng, BkfChSerConnId *connId, BkfUrl *cliUrl, BkfUrl *localUrl)
{
    BKF_LOG_DEBUG(BKF_LOG_HND, "connEx connMng(%#x), connId(%#x)\n", BKF_MASK_ADDR(connMng), BKF_MASK_ADDR(connId));

    BkfPuberConn *conn = BkfPuberConnFind(connMng, connId);
    if (conn != VOS_NULL) {
        /* 重复通知，错误 */
        BKF_LOG_ERROR(BKF_LOG_HND, "connEx connId(%#x)/conn(%#x)\n", BKF_MASK_ADDR(connId), BKF_MASK_ADDR(conn));
        if (connMng->argInit->onDisConnect) {
            connMng->argInit->onDisConnect(connMng->argInit->cookie, &conn->localUrl, &conn->cliUrl);
        }
        BkfPuberConnDel(conn, VOS_TRUE);
        return;
    }

    conn = BkfPuberConnAdd(connMng, connId, (F_BKF_PUBER_SESS_TRIG_SCHED_SELF)BkfPuberConnOnTrigSchedSess,
        (F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF)BkfPuberConnOnTrigSlowSchedSess);
    if (conn) {
        conn->localUrl = *localUrl;
        conn->cliUrl = *cliUrl;
    }
}

STATIC void BkfPuberConnOnRcvData(BkfPuberConnMng *connMng, BkfChSerConnId *connId, uint8_t *rcvData, int32_t dataLen)
{
    BkfPuberConn *conn = BkfPuberConnFind(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_WARN(BKF_LOG_HND, "connId(%#x)/rcvData(%#x)/dataLen(%u), conn(%#x)\n", BKF_MASK_ADDR(connId),
            BKF_MASK_ADDR(rcvData), dataLen, BKF_MASK_ADDR(conn));
        return;
    }

    BkfPuberConnLockDel(conn);
    uint32_t ret = BkfPuberConnProcRcvData(conn, rcvData, dataLen);
    BkfPuberConnUnlockDel(conn);
    (void)BkfPuberConnDelIfRetNeedDel(conn, ret);
}

STATIC void BkfPuberConnOnRcvDataEvent(BkfPuberConnMng *connMng, BkfChSerConnId *connId)
{
    BkfPuberConn *conn = BkfPuberConnFind(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_WARN(BKF_LOG_HND, "connId(%#x)/conn(%#x)\n", BKF_MASK_ADDR(connId), BKF_MASK_ADDR(conn));
        return;
    }

    BkfPuberConnLockDel(conn);
    uint32_t ret = BkfPuberConnProcRcvDataEvent(conn);
    BkfPuberConnUnlockDel(conn);
    (void)BkfPuberConnDelIfRetNeedDel(conn, ret);
}

STATIC void BkfPuberConnOnUnblock(BkfPuberConnMng *connMng, BkfChSerConnId *connId)
{
    BkfPuberConn *conn = BkfPuberConnFind(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_WARN(BKF_LOG_HND, "connId(%#x)/conn(%#x)\n", BKF_MASK_ADDR(connId), BKF_MASK_ADDR(conn));
        return;
    }

    BkfPuberConnLockDel(conn);
    uint32_t ret = BkfPuberConnFsmProc(conn, BKF_PUBER_CONN_EVT_UNDERLAY_UNBLOCK, VOS_NULL);
    BkfPuberConnUnlockDel(conn);
    (void)BkfPuberConnDelIfRetNeedDel(conn, ret);
}

STATIC void BkfPuberConnOnDisconn(BkfPuberConnMng *connMng, BkfChSerConnId *connId)
{
    BKF_LOG_DEBUG(BKF_LOG_HND, "connMng(%#x), connId(%#x)\n", BKF_MASK_ADDR(connMng), BKF_MASK_ADDR(connId));

    BkfPuberConn *conn = BkfPuberConnFind(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_WARN(BKF_LOG_HND, "connId(%#x)/conn(%#x)\n", BKF_MASK_ADDR(connId), BKF_MASK_ADDR(conn));
        return;
    }
    if (connMng->argInit->onDisConnect) {
        connMng->argInit->onDisConnect(connMng->argInit->cookie, &conn->localUrl, &conn->cliUrl);
    }
    BkfPuberConnDel(conn, VOS_FALSE);
}

STATIC void BkfPuberConnOnTrigSchedSess(BkfPuberConn *conn)
{
    (void)BkfPuberConnStartJob(conn);
}

STATIC uint32_t BkfPuberConnOnJob(BkfPuberConn *conn, uint32_t *runCost)
{
    BkfPuberConnMng *connMng = conn->connMng;
    *runCost = BkfJobGetRunCostMax(conn->connMng->argInit->jobMng);
    conn->jobId = VOS_NULL;

    BkfPuberConnLockDel(conn);
    uint32_t ret = BkfPuberConnFsmProc(conn, BKF_PUBER_CONN_EVT_JOB, VOS_NULL);
    BkfPuberConnUnlockDel(conn);
    (void)BkfPuberConnDelIfRetNeedDel(conn, ret);
    BKF_LOG_ERROR(BKF_LOG_HND, "BkfPuberConnOnJob proc\n");
    return BKF_JOB_FINSIH;
}

STATIC void BkfPuberConnOnTrigSlowSchedSess(BkfPuberConn *conn)
{
    (void)BkfPuberConnStartTmr(conn, (F_BKF_TMR_TIMEOUT_PROC)BkfPuberConnOnTmrTO, BKF_PUBER_CONN_TMR_INTERVAL);
}

STATIC uint32_t BkfPuberConnOnTmrTO(BkfPuberConn *conn, void *paramTmrLibUnknown)
{
    (void)paramTmrLibUnknown;
    BkfPuberConnLockDel(conn);
    uint32_t ret = BkfPuberConnFsmProc(conn, BKF_PUBER_CONN_EVT_TMR, VOS_NULL);
    BkfPuberConnUnlockDel(conn);
    (void)BkfPuberConnDelIfRetNeedDel(conn, ret);
    return BKF_OK;
}

STATIC BkfPuberConn *BkfPuberConnDelIfRetNeedDel(BkfPuberConn *conn, uint32_t ret)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BKF_LOG_DEBUG(BKF_LOG_HND, "ret(%u)\n", ret);
    if (ret == BKF_PUBER_CONN_NEED_DELETE) {
        if (connMng->argInit->onDisConnect) {
            connMng->argInit->onDisConnect(connMng->argInit->cookie, &conn->localUrl, &conn->cliUrl);
        }
        BkfPuberConnDel(conn, VOS_TRUE);
        return VOS_NULL;
    }

    return conn;
}

STATIC uint32_t BkfPuberConnProcRcvData(BkfPuberConn *conn, uint8_t *rcvData, int32_t dataLen)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BKF_LOG_DEBUG(BKF_LOG_HND, "conn(%#x), rcvData(%#x)/dataLen(%d)\n", BKF_MASK_ADDR(conn), BKF_MASK_ADDR(rcvData),
        dataLen);

    BkfMsgDecoder decoder;
    uint32_t ret = BkfPuberConnInitDecoderForRcvData(conn, rcvData, dataLen, &decoder);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    ret = BkfPuberConnProcDataInDecoder(conn, &decoder);
    if (ret == BKF_PUBER_CONN_NEED_DELETE) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    ret = BkfPuberConnSaveDecoderLeftData(conn, &decoder, VOS_TRUE);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_PUBER_CONN_NEED_DELETE;
    }

    return BKF_OK;
}

STATIC uint32_t BkfPuberConnInitDecoderForRcvData(BkfPuberConn *conn, uint8_t *rcvData, int32_t dataLen,
    BkfMsgDecoder *decoder)
{
    /*
    1. 如果bufq为空，根据rcvData直接初始化decoder
    2. 否则，将rcvData附加到bufq中，然后根据bufq初始化decoder
    */
    BkfPuberConnMng *connMng = conn->connMng;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(connMng->argInit->dc);
    if (conn->rcvDataBuf == VOS_NULL) {
        return BkfMsgDecodeInit(decoder, connMng->argInit->name, rcvData, dataLen, sliceVTbl->keyLen,
            sliceVTbl->keyCodec, sliceVTbl->keyGetStrOrNull, BKF_LOG_HND);
    }

    int32_t enqLen = BkfBufqEn(conn->rcvDataBuf, rcvData, dataLen);
    if ((enqLen < 0) || (enqLen != dataLen)) {
        return BKF_ERR;
    }

    return BkfMsgDecodeInit(decoder, connMng->argInit->name, BkfBufqGetUsedBegin(conn->rcvDataBuf),
        BkfBufqGetUsedLen(conn->rcvDataBuf), sliceVTbl->keyLen, sliceVTbl->keyCodec, sliceVTbl->keyGetStrOrNull,
        BKF_LOG_HND);
}

STATIC uint32_t BkfPuberConnProcOneMsgInDecoder(BkfPuberConn *conn, BkfMsgDecoder *decoder, BkfMsgHead *msgHead)
{
    BkfPuberConnMng *connMng = conn->connMng;
    if (msgHead->msgId == BKF_MSG_HELLO) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "proc conn hello msg\n");
        return BkfPuberConnFsmProc(conn, BKF_PUBER_CONN_EVT_HELLO, decoder);
    }

    if (BKF_MSG_IS_SESS(msgHead->msgId)) {
        uint8_t connState = BkfFsmGetState(&conn->fsm);
        BOOL mayProcSessData = (connState == BKF_PUBER_CONN_STATE_CONN);
        if (!mayProcSessData) {
            BKF_LOG_WARN(BKF_LOG_HND, "rcv sess msg before conn ok, ng\n");
            return BKF_ERR;
        }

        BKF_LOG_DEBUG(BKF_LOG_HND, "proc sess msg\n");
        uint32_t ret = BkfPuberSessProcRcvData(conn->sessMng, msgHead, decoder);
        return (ret == BKF_PUBER_SESS_FATAL_ERR) ? BKF_PUBER_CONN_NEED_DELETE : ret;
    }

    /* 到此处是异常报文，记录错误 */
    BKF_LOG_WARN(BKF_LOG_HND, "msgId(%u), unknown msg & ng\n", msgHead->msgId);
    return BKF_ERR;
}

STATIC uint32_t BkfPuberConnProcDataInDecoder(BkfPuberConn *conn, BkfMsgDecoder *decoder)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BkfMsgHead *msgHead = VOS_NULL;
    uint32_t errCode = BKF_OK;
    while ((msgHead = BkfMsgDecodeMsgHead(decoder, &errCode)) != VOS_NULL) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "msgId(%u, %s)/bodyLen(%u), errCode(%u)\n", msgHead->msgId,
            BkfMsgGetIdStr(msgHead->msgId), msgHead->bodyLen, errCode);

        uint32_t ret = BkfPuberConnProcOneMsgInDecoder(conn, decoder, msgHead);
        BKF_LOG_DEBUG(BKF_LOG_HND, "ret(%u)\n", ret);
        if (ret == BKF_PUBER_CONN_NEED_DELETE) {
            return BKF_PUBER_CONN_NEED_DELETE;
        }
    }
    return (errCode == BKF_OK) ? BKF_OK : BKF_PUBER_CONN_NEED_DELETE;
}

STATIC uint32_t BkfPuberConnSaveDecoderLeftData(BkfPuberConn *conn, BkfMsgDecoder *decoder, BOOL delBufqIfNoLeftData)
{
    /* 解决拆包粘包后接收数据问题 */
    int32_t leftLen = 0;
    uint8_t *leftBegin = BkfMsgDecodeGetLeft(decoder, &leftLen);
    if (leftLen < 0) {
        return BKF_ERR;
    }

    if (leftLen == 0) {
        BOOL delBufq = (conn->rcvDataBuf != VOS_NULL) && delBufqIfNoLeftData;
        BkfBufqReset(conn->rcvDataBuf);
        if (delBufq) {
            BkfBufqUninit(conn->rcvDataBuf);
            conn->rcvDataBuf = VOS_NULL;
        }
        return BKF_OK;
    }

    if (conn->rcvDataBuf != VOS_NULL) {
        int32_t bufqLeftLen = BkfBufqDe2Left(conn->rcvDataBuf, leftLen);
        return (bufqLeftLen == leftLen) ? BKF_OK : BKF_ERR;
    }

    BkfPuberConnMng *connMng = conn->connMng;
    conn->rcvDataBuf = BkfBufqInit(connMng->argInit->memMng, BKF_CH_SER_MALLOC_DATA_BUF_LEN_MAX);
    if (conn->rcvDataBuf == VOS_NULL) {
        return BKF_ERR;
    }

    int32_t bufqEnLen = BkfBufqEn(conn->rcvDataBuf, leftBegin, leftLen);
    return (bufqEnLen == leftLen) ? BKF_OK : BKF_ERR;
}

STATIC uint32_t BkfPuberConnProcRcvDataEvent(BkfPuberConn *conn)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BKF_LOG_DEBUG(BKF_LOG_HND, "conn(%#x)\n", BKF_MASK_ADDR(conn));
    if (conn->rcvDataBuf == VOS_NULL) {
        conn->rcvDataBuf = BkfBufqInit(connMng->argInit->memMng, BKF_CH_SER_MALLOC_DATA_BUF_LEN_MAX);
        if (conn->rcvDataBuf == VOS_NULL) {
            return BKF_PUBER_CONN_NEED_DELETE;
        }
    }

    for (;;) {
        int32_t readLen = BkfPuberConnReadData2Bufq(conn);
        if (readLen < 0) {
            return BKF_PUBER_CONN_NEED_DELETE;
        }

        if (readLen == 0) {
            break;
        }

        BkfMsgDecoder decoder;
        BOOL needDelConn = (BkfPuberConnInitDecoderForRcvDataEvent(conn, &decoder) != BKF_OK) ||
                           (BkfPuberConnProcDataInDecoder(conn, &decoder) == BKF_PUBER_CONN_NEED_DELETE) ||
                           (BkfPuberConnSaveDecoderLeftData(conn, &decoder, VOS_FALSE) != BKF_OK);
        if (needDelConn) {
            return BKF_PUBER_CONN_NEED_DELETE;
        }
    }

    if (BkfBufqIsEmpty(conn->rcvDataBuf)) {
        BkfBufqUninit(conn->rcvDataBuf);
        conn->rcvDataBuf = VOS_NULL;
    }
    return BKF_OK;
}

STATIC uint32_t BkfPuberConnInitDecoderForRcvDataEvent(BkfPuberConn *conn, BkfMsgDecoder *decoder)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BkfDcSliceVTbl *sliceVTbl = BKF_DC_GET_SLICE_VTBL(connMng->argInit->dc);
    uint32_t ret = BkfMsgDecodeInit(decoder, connMng->argInit->name, BkfBufqGetUsedBegin(conn->rcvDataBuf),
        BkfBufqGetUsedLen(conn->rcvDataBuf), sliceVTbl->keyLen, sliceVTbl->keyCodec, sliceVTbl->keyGetStrOrNull,
        BKF_LOG_HND);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
    }

    return ret;
}

STATIC int32_t BkfPuberConnReadData2Bufq(BkfPuberConn *conn)
{
    BkfPuberConnMng *connMng = conn->connMng;
    uint8_t *freeBegin = BkfBufqGetFreeBegin(conn->rcvDataBuf);
    int32_t freeLen = BkfBufqGetFreeLen(conn->rcvDataBuf);
    if ((freeBegin == VOS_NULL) || (freeLen <= 0)) {
        BKF_LOG_ERROR(BKF_LOG_HND, "bufq_freeBegin(%#x)/freeLen(%d)\n", BKF_MASK_ADDR(freeBegin), freeLen);
        return -1;
    }

    int32_t readLen = BkfChSerRead(connMng->argInit->chMng, conn->connId, freeBegin, freeLen);
    BKF_LOG_DEBUG(BKF_LOG_HND, "readLen(%d)\n", readLen);
    if (readLen <= 0) {
        return 0; /* 无剩余数据，正常 */
    }

    int32_t enqLen = BkfBufqEnDataLen(conn->rcvDataBuf, readLen);
    if (enqLen != readLen) {
        BKF_LOG_ERROR(BKF_LOG_HND, "enqLen(%d)/readLen(%d)\n", enqLen, readLen);
        return -1;
    }

    return readLen;
}

#endif
#if BKF_BLOCK("公有函数定义")
BkfPuberConnMng *BkfPuberConnInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng)
{
    BkfPuberConnMng *connMng = BkfPuberConnDataInit(arg, tableTypeMng);
    if (connMng == VOS_NULL) {
        return VOS_NULL;
    }

    connMng->connFsmTmpl = BkfPuberConnFsmTmplInit(connMng);
    if (connMng->connFsmTmpl == VOS_NULL) {
        BkfPuberConnUninit(connMng);
        return VOS_NULL;
    }

    uint32_t ret = BkfJobRegType(arg->jobMng, arg->jobTypeId, (F_BKF_JOB_PROC)BkfPuberConnOnJob, arg->jobPrio);
    if (ret != BKF_OK) {
        BkfPuberConnUninit(connMng);
        return VOS_NULL;
    }

    ret = BkfPuberConnSysLogInit(connMng);
    if (ret != BKF_OK) {
        BkfPuberConnUninit(connMng);
        return VOS_NULL;
    }

    BkfPuberConnDispInit(connMng);
    BKF_LOG_INFO(BKF_LOG_HND, "connMng(%#x, %s), init ok\n", BKF_MASK_ADDR(connMng), arg->name);
    return connMng;
}

void BkfPuberConnUninit(BkfPuberConnMng *connMng)
{
    if (connMng == VOS_NULL) {
        return;
    }

    BKF_LOG_INFO(BKF_LOG_HND, "connMng(%#x, %s), uninit\n", BKF_MASK_ADDR(connMng), connMng->argInit->name);
    BkfPuberConnDispUninit(connMng);
    BkfPuberConnSysLogUninit(connMng);
    BkfFsmTmpl *tmpl = connMng->connFsmTmpl;
    BkfPuberConnDataUninit(connMng);
    BkfFsmTmplUninit(tmpl);
}

uint32_t BkfPuberConnEnable(BkfPuberConnMng *connMng)
{
    BkfChSerEnableArg arg = {.cookie = connMng,
        .onMayAccept = (F_BKF_CH_SER_ON_MAY_ACCEPT)BkfPuberConnOnMayAccept,
        .onConn = (F_BKF_CH_SER_ON_CONN)BkfPuberConnOnConn,
        .onRcvData = (F_BKF_CH_SER_ON_RCV_DATA)BkfPuberConnOnRcvData,
        .onRcvDataEvent = (F_BKF_CH_SER_ON_RCV_DATA_EVENT)BkfPuberConnOnRcvDataEvent,
        .onUnblock = (F_BKF_CH_SER_ON_UNBLOCK)BkfPuberConnOnUnblock,
        .onDisconn = (F_BKF_CH_SER_ON_DISCONN)BkfPuberConnOnDisconn,
        .onConnEx = (F_BKF_CH_SER_ON_CONNEX)BkfPuberConnOnConnEx};
    return BkfChSerEnable(connMng->argInit->chMng, &arg);
}

uint32_t BkfPuberConnSetSelfUrl(BkfPuberConnMng *connMng, BkfUrl *url)
{
    return BkfChSerListen(connMng->argInit->chMng, url);
}

void BkfPuberConnUnsetSelfUrl(BkfPuberConnMng *connMng, BkfUrl *url)
{
    BkfChSerUnlisten(connMng->argInit->chMng, url);
}

#endif

#ifdef __cplusplus
}
#endif
