/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vos_typdef.h"
#include "bkf_suber_data.h"
#include "bkf_suber_conn.h"
#include "bkf_suber_conn_data.h"
#include "bkf_suber_conn_fsm.h"
#include "bkf_suber_conn_pri.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有函数定义")
static inline void BkfSuberConnBeforeProc(BkfSuberConn *conn)
{
    conn->lockDel = VOS_TRUE;
}

static inline BOOL BkfSuberSessMayProcRcvData(BkfSuberConn *conn)
{
    uint8_t state = BkfFsmGetState(&conn->fsm);
    return (state == BACKFIN_SUB_CONN_STATE_UP);
}

void BkfSuberConnAfterProc(BkfSuberConn *conn, uint32_t ret)
{
    // 解锁，删除conn
    conn->lockDel = VOS_FALSE;
    if (conn->softDel) {
        BkfSuberConnProcDelConn(conn);
        return;
    }

    if (ret == BKF_SUBER_CONN_NEED_DELETE) {
        if (conn->connId != VOS_NULL) {
            BkfChCliDisConn(conn->urlTypeMng->connMng->env->chCliMng, conn->connId);
        }
        BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_TCPDISCONN, VOS_NULL, VOS_NULL);
    }
    return;
}

uint32_t BkfSuberConnJobProc(BkfSuberConn *conn, uint32_t *runCost)
{
    *runCost = BkfJobGetRunCostMax(conn->urlTypeMng->connMng->env->jobMng);
    BkfSuberConnBeforeProc(conn);
    uint32_t ret = BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_JOB, VOS_NULL, VOS_NULL);
    uint32_t jobRet;
    if (ret == BKF_SUBER_SESS_SEND_BUF_NOT_ENOUGH) {
        jobRet = BKF_JOB_CONTINUE;
    } else {
        conn->jobId = VOS_NULL;
        jobRet = BKF_JOB_FINSIH;
    }
    BkfSuberConnAfterProc(conn, ret);
    return jobRet;
}

static inline BkfSuberConn *BkfSuberConnGetConnByChannel(BkfSuberConnMng *connMng, BkfChCliConnId *connId)
{
    return BkfSuberConnDataFindByConnId(connMng, connId);
}

uint32_t BkfSuberConnProcRcvDataDoBefore(BkfSuberConn *conn, uint8_t *rcvData, int32_t dataLen, BkfMsgDecoder *decoder)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    BkfSuberEnv *env = connMng->env;
    uint32_t ret;
    int32_t enqLen;

    /*
    1. 如果bufq为空，根据rcvData直接初始化decoder
    2. 否则，将rcvData附加到bufq中，然后根据bufq初始化decoder
    */

    if (conn->rcvDataBuf == VOS_NULL) {
        ret = BkfMsgDecodeInit(decoder, env->name, rcvData, dataLen, env->sliceVTbl.keyLen, env->sliceVTbl.keyCodec,
            env->sliceVTbl.keyGetStrOrNull, connMng->log);
        if (ret != BKF_OK) {
            return BKF_ERR;
        }
    } else {
        enqLen = BkfBufqEn(conn->rcvDataBuf, rcvData, dataLen);
        if ((enqLen < 0) || (enqLen != dataLen)) {
            return BKF_ERR;
        }
        ret = BkfMsgDecodeInit(decoder, env->name, BkfBufqGetUsedBegin(conn->rcvDataBuf),
            BkfBufqGetUsedLen(conn->rcvDataBuf), env->sliceVTbl.keyLen, env->sliceVTbl.keyCodec,
            env->sliceVTbl.keyGetStrOrNull, connMng->log);
        if (ret != BKF_OK) {
            return BKF_ERR;
        }
    }

    return BKF_OK;
}

uint32_t BkfSuberConnProcRcvDataDoAfter(BkfSuberConn *conn, BkfMsgDecoder *decoder)
{
    /*
    1. 如果bufq非空，更新其到decoder剩余长度
    2. 否则，如果decoder有剩余长度，缓冲到bufq中
    */
    int32_t leftLen;
    uint8_t *leftBegin = BkfMsgDecodeGetLeft(decoder, &leftLen);
    if (leftLen < 0) {
        return BKF_ERR;
    }
    if (conn->rcvDataBuf != VOS_NULL) {
        if (leftLen > 0) {
            if (BkfBufqDe2Left(conn->rcvDataBuf, leftLen) != leftLen) {
                return BKF_ERR;
            }
        } else {
            BkfBufqUninit(conn->rcvDataBuf);
            conn->rcvDataBuf = VOS_NULL;
        }
    } else {
        if (leftLen > 0) {
            BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
            conn->rcvDataBuf = BkfBufqInit(connMng->env->memMng, BKF_CH_CLI_MALLOC_DATA_BUF_LEN_MAX);
            if (conn->rcvDataBuf == VOS_NULL) {
                return BKF_ERR;
            }
            int32_t enqLen = BkfBufqEn(conn->rcvDataBuf, leftBegin, leftLen);
            if (enqLen != leftLen) {
                return BKF_ERR;
            }
        }
    }

    return BKF_OK;
}

uint32_t BkfSuberConnProcOneMsgInDecoder(BkfSuberConn *conn, BkfMsgDecoder *decoder, BkfMsgHead *msgHead)
{
    if (msgHead->msgId == BKF_MSG_HELLO_ACK) {
        return BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_RCVHELLOACK, decoder, VOS_NULL);
    } else if (BKF_MSG_IS_SESS(msgHead->msgId)) {
        if (!BkfSuberSessMayProcRcvData(conn)) {
            return BKF_ERR;
        }
        return BkfSuberSessProcRcvData(conn->sessMng, decoder, msgHead);
    } else {
        return BKF_SUBER_CONN_NEED_DELETE;
    }
}

uint32_t BkfSuberConnProcDataInDecoder(BkfSuberConn *conn, BkfMsgDecoder *decoder)
{
    BkfMsgHead *msgHead = VOS_NULL;
    uint32_t errCode = BKF_OK;
    uint32_t ret;

    while ((msgHead = BkfMsgDecodeMsgHead(decoder, &errCode)) != VOS_NULL) {
        BKF_ASSERT(errCode == BKF_OK);
        BKF_LOG_DEBUG(conn->urlTypeMng->connMng->log, "msgId(%u, %s)/bodyLen(%u), errCode(%u)\n", msgHead->msgId,
            BkfMsgGetIdStr(msgHead->msgId), msgHead->bodyLen, errCode);
        ret = BkfSuberConnProcOneMsgInDecoder(conn, decoder, msgHead);
        if (ret == BKF_SUBER_CONN_NEED_DELETE) {
            return ret;
        }
    }

    return BKF_OK;
}

uint32_t BkfSuberConnProcRcvData(BkfSuberConn *conn, uint8_t *rcvData, int32_t dataLen)
{
    BkfMsgDecoder decoder;
    uint32_t ret = BkfSuberConnProcRcvDataDoBefore(conn, rcvData, dataLen, &decoder);
    if (ret != BKF_OK) {
        BKF_LOG_DEBUG(conn->urlTypeMng->connMng->log, "onProcRcvDataBefore: retCode %u\n", ret);
        return BKF_SUBER_CONN_NEED_DELETE;
    }

    ret = BkfSuberConnProcDataInDecoder(conn, &decoder);
    if (ret == BKF_SUBER_CONN_NEED_DELETE) {
        BKF_LOG_DEBUG(conn->urlTypeMng->connMng->log, "onProcRcvDataInDecoder: retCode %u\n", ret);
        return BKF_SUBER_CONN_NEED_DELETE;
    }

    ret = BkfSuberConnProcRcvDataDoAfter(conn, &decoder);
    if (ret != BKF_OK) {
        BKF_LOG_DEBUG(conn->urlTypeMng->connMng->log, "onProcRcvDataAfter: retCode %u\n", ret);
        return BKF_SUBER_CONN_NEED_DELETE;
    }

    return BKF_OK;
}

uint32_t BkfSuberConnProcRcvDataEventDo(BkfSuberConn *conn)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    BkfSuberEnv *env = connMng->env;
    uint8_t *bufFreeBegin = VOS_NULL;
    int32_t bufFreeLen = 0;
    int32_t readLen = 0;
    int32_t enqLen = 0;
    BkfMsgDecoder decoder = {};
    uint32_t ret = 0;
    int32_t leftLen = 0;

    for (;;) {
        bufFreeBegin = BkfBufqGetFreeBegin(conn->rcvDataBuf);
        bufFreeLen = BkfBufqGetFreeLen(conn->rcvDataBuf);
        if ((bufFreeBegin == VOS_NULL) || (bufFreeLen <= 0)) {
            BKF_LOG_ERROR(connMng->log, "bufFreeBegin(%#x)/bufFreeLen(%d)\n", BKF_MASK_ADDR(bufFreeBegin), bufFreeLen);
            return BKF_ERR;
        }
        readLen = BkfChCliRead(env->chCliMng, conn->connId, bufFreeBegin, bufFreeLen);
        BKF_LOG_DEBUG(connMng->log, "readLen(%d)\n", readLen);
        if (readLen <= 0) {
            return BKF_OK; /* 无剩余数据，正常 */
        }

        enqLen = BkfBufqEnDataLen(conn->rcvDataBuf, readLen);
        if (enqLen != readLen) {
            BKF_LOG_ERROR(connMng->log, "enqLen(%d)/readLen(%d)\n", enqLen, readLen);
            return BKF_ERR;
        }

        ret = BkfMsgDecodeInit(&decoder, env->name, BkfBufqGetUsedBegin(conn->rcvDataBuf),
            BkfBufqGetUsedLen(conn->rcvDataBuf), env->sliceVTbl.keyLen, connMng->env->sliceVTbl.keyCodec,
            env->sliceVTbl.keyGetStrOrNull, connMng->log);
        if (ret != BKF_OK) {
            return ret;
        }

        ret = BkfSuberConnProcDataInDecoder(conn, &decoder);
        if (ret == BKF_SUBER_CONN_NEED_DELETE) {
            return ret;
        }
        (void)BkfMsgDecodeGetLeft(&decoder, &leftLen);
        if ((leftLen < 0) || (BkfBufqDe2Left(conn->rcvDataBuf, leftLen) != leftLen)) {
            return BKF_ERR;
        }
    }
}

uint32_t BkfSuberConnProcRcvDataEvent(BkfSuberConn *conn)
{
    if (conn->rcvDataBuf == VOS_NULL) {
        conn->rcvDataBuf = BkfBufqInit(conn->urlTypeMng->connMng->env->memMng, BKF_CH_CLI_MALLOC_DATA_BUF_LEN_MAX);
        if (conn->rcvDataBuf == VOS_NULL) {
            return BKF_SUBER_CONN_NEED_DELETE;
        }
    }

    uint32_t ret = BkfSuberConnProcRcvDataEventDo(conn);
    if (ret != BKF_OK) {
        return BKF_SUBER_CONN_NEED_DELETE;
    }

    if (BkfBufqIsEmpty(conn->rcvDataBuf)) {
        BkfBufqUninit(conn->rcvDataBuf);
        conn->rcvDataBuf = VOS_NULL;
    }

    return BKF_OK;
}
#endif

#if BKF_BLOCK("公有函数定义")
uint32_t BkfSuberConnCreateSess(BkfSuberConnMng *connMng, BkfSuberConnSessArg *arg)
{
    BkfSuberConn *conn = BkfSuberConnDataFind(connMng, arg->puberUrl, arg->localUrl);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't find suber conn data.\n");
        return BKF_ERR;
    }

    BkfSuberSess *sess = BkfSuberSessFind(conn->sessMng, arg->sliceKey, arg->tableTypeId);
    if (sess == VOS_NULL) {
        sess = BkfSuberSessCreate(conn->sessMng, arg->sliceKey, arg->tableTypeId, arg->isVerify);
        if (sess == VOS_NULL) {
            return BKF_ERR;
        }
    } else {
        BkfSuberSessReLoad(sess, arg->isVerify);
    }

    return BKF_OK;
}

void BkfSuberConnDelSess(BkfSuberConnMng *connMng, BkfSuberConnSessArg *arg)
{
    BkfSuberConn *conn = BkfSuberConnDataFind(connMng, arg->puberUrl, arg->localUrl);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't find suber conn data.\n");
        return;
    }

    BkfSuberSess *sess = BkfSuberSessFind(conn->sessMng, arg->sliceKey, arg->tableTypeId);
    if (sess == VOS_NULL) {
        return;
    }

    if (arg->isVerify) {
        BkfSuberSessUnVerify(sess);
    } else {
        BkfSuberSessDel(sess);
    }

    return;
}

BkfSuberConn *BkfSuberConnCreateConn(BkfSuberConnMng *connMng, BkfSuberInst *inst)
{
    uint8_t urlStr[BKF_URL_STR_LEN_MAX] = {0};
    BKF_LOG_DEBUG(connMng->log, "add conn url %s.\n", BkfUrlGetStr(&inst->puberUrl, urlStr, sizeof(urlStr)));

    BkfSuberConn *conn = BkfSuberConnDataAdd(connMng, &inst->puberUrl, &inst->localUrl);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "add conn data fail.\n");
        return VOS_NULL;
    }

    if (conn->sessMng == VOS_NULL) {
        BkfSuberSessMngInitArg arg = {.env = connMng->env,
            .cookie = conn,
            .trigSchedSelf = (F_BKF_SUBER_SESS_TRIG_SCHED_SELF)BkfSuberConnTrigSched,
            .dataMng = inst->dataMng,
            .pubUrl = conn->puberUrl,
            .locUrl = conn->localUrl};
        conn->sessMng = BkfSuberSessMngInit(&arg);
        if (conn->sessMng == VOS_NULL) {
            BkfSuberConnDataDel(conn);
            return VOS_NULL;
        }
    }

    return conn;
}

void BkfSuberConnProcConn(BkfSuberConn *conn)
{
    BkfSuberConnUrlTypeMng *urlTypeMng = conn->urlTypeMng;
    BkfSuberConnMng *connMng = urlTypeMng->connMng;

    uint8_t urlStr[BKF_URL_STR_LEN_MAX] = {0};
    BKF_LOG_DEBUG(connMng->log, "connect url %s.\n", BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));

    if (!connMng->isEnable) {
        BKF_LOG_ERROR(connMng->log, "connect url %s fail. not enable.\n",
            BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));
        return;
    }

    BkfSuberConnBeforeProc(conn);
    conn->isTrigConn = VOS_TRUE;
    BkfSuberSessResetDisconnectReason(conn->sessMng);
    uint32_t ret = BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_STARTINST, VOS_NULL, VOS_NULL);
    BkfSuberConnAfterProc(conn, ret);
    return;
}

void BkfSuberConnTrigSched(BkfSuberConn *conn)
{
    (void)BkfSuberConnStartJob(conn);
    return;
}

void BkfSuberConnUnBlock(BkfSuberConnMng *connMng, BkfChCliConnId *connId)
{
    BkfSuberConn *conn = BkfSuberConnGetConnByChannel(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't get conn.\n");
        return;
    }

    BkfSuberConnBeforeProc(conn);
    BkfSuberSessSetUnBlockFlag(conn->sessMng);
    uint32_t ret = BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_UNBLOCK, VOS_NULL, VOS_NULL);
    BkfSuberSessResetUnBlockFlag(conn->sessMng);
    BkfSuberConnAfterProc(conn, ret);
    return;
}

void BkfSuberConnDisConn(BkfSuberConnMng *connMng, BkfChCliConnId *connId)
{
    BkfSuberConn *conn = BkfSuberConnGetConnByChannel(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't get conn.\n");
        return;
    }
    if (connMng->env->onDisConn && BKF_FSM_GET_STATE(&conn->fsm) == BACKFIN_SUB_CONN_STATE_UP) {
        connMng->env->onDisConn(connMng->env->cookie, &conn->puberUrl, &conn->localUrl,
            BkfSuberSessGetDisconnectReason(conn->sessMng));
    }
    BkfSuberConnBeforeProc(conn);
    uint32_t ret = BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_TCPDISCONN, VOS_NULL, VOS_NULL);
    BkfSuberConnAfterProc(conn, ret);
    return;
}

void BkfSuberConnDisConnEx(BkfSuberConnMng *connMng, BkfChCliConnId *connId, BOOL isPeerFin)
{
    BkfSuberConn *conn = BkfSuberConnGetConnByChannel(connMng, connId);
    BKF_LOG_INFO(connMng->log, "SuberConnDisConnEx: connId: %#x\n", BKF_MASK_ADDR(connId));
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't get conn.\n");
        return;
    }
    BkfSuberSessSetDisconnectReason(conn->sessMng,
        isPeerFin ? BKF_SUBER_DISCONNECT_REASON_PEERFIN : BKF_SUBER_DISCONNECT_REASON_PEERERROR);
    if (connMng->env->onDisConn && BKF_FSM_GET_STATE(&conn->fsm) == BACKFIN_SUB_CONN_STATE_UP) {
        connMng->env->onDisConn(connMng->env->cookie, &conn->puberUrl, &conn->localUrl,
            BkfSuberSessGetDisconnectReason(conn->sessMng));
    }
    BkfSuberConnBeforeProc(conn);
    uint32_t ret = BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_TCPDISCONN, VOS_NULL, VOS_NULL);
    BkfSuberConnAfterProc(conn, ret);
    return;
}

void BkfSuberConnOnConn(BkfSuberConnMng *connMng, BkfChCliConnId *connId, uint16_t srcPort)
{
    BkfSuberConn *conn = BkfSuberConnGetConnByChannel(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't get conn.\n");
        return;
    }
    conn->srcPort = srcPort;
    return;
}

void BkfSuberConnRcvDataEvent(BkfSuberConnMng *connMng, BkfChCliConnId *connId)
{
    BkfSuberConn *conn = BkfSuberConnGetConnByChannel(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't find suber conn.\n");
        return;
    }

    BkfSuberConnBeforeProc(conn);
    uint32_t ret = BkfSuberConnProcRcvDataEvent(conn);
    BkfSuberConnAfterProc(conn, ret);
    return;
}

void BkfSuberConnRcvData(BkfSuberConnMng *connMng, BkfChCliConnId *connId, uint8_t *data, int32_t dataLen)
{
    BkfSuberConn *conn = BkfSuberConnGetConnByChannel(connMng, connId);
    if (conn == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "can't find suber conn data.\n");
        return;
    }

    BkfSuberConnBeforeProc(conn);
    uint32_t ret = BkfSuberConnProcRcvData(conn, data, dataLen);
    BkfSuberConnAfterProc(conn, ret);
    return;
}

void BkfSuberConnProcDelConn(BkfSuberConn *conn)
{
    if (conn->lockDel) {
        conn->softDel = VOS_TRUE;
        return;
    }

    BkfSuberConnProcDisConn(conn);
    if (conn->sessMng != VOS_NULL) {
        BkfSuberSessMngUnInit(conn->sessMng);
        conn->sessMng = VOS_NULL;
    }

    BkfDlNode *temp = VOS_NULL;
    BkfDlNode *tempNext = VOS_NULL;
    for (temp = BKF_DL_GET_FIRST(&conn->obInst); temp != VOS_NULL; temp = tempNext) {
        tempNext = BKF_DL_GET_NEXT(&conn->obInst, temp);
        BKF_DL_REMOVE(temp);
    }

    BkfSuberConnStopJob(conn);
    BkfSuberConnDataDel(conn);
}

void BkfSuberConnProcDisConn(BkfSuberConn *conn)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;

    uint8_t urlStr[BKF_URL_STR_LEN_MAX] = {0};
    BKF_LOG_DEBUG(connMng->log, "disconn url %s.\n", BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));

    conn->isTrigConn = VOS_FALSE;
    if (connMng->env->onDisConn && BKF_FSM_GET_STATE(&conn->fsm) == BACKFIN_SUB_CONN_STATE_UP) {
        connMng->env->onDisConn(connMng->env->cookie, &conn->puberUrl, &conn->localUrl,
            BkfSuberSessGetDisconnectReason(conn->sessMng));
    }
    (void)BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_STOPINST, VOS_NULL, VOS_NULL);
    return;
}

uint32_t BkfSuberConnJobRegType(BkfSuberConnMng *connMng)
{
    BkfSuberEnv *env = connMng->env;
    uint32_t ret = BkfJobRegType(env->jobMng, env->jobTypeId1, (F_BKF_JOB_PROC)BkfSuberConnJobProc, env->jobPrioL);
    if (unlikely(ret != BKF_OK)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    return BKF_OK;
}
#endif

#ifdef __cplusplus
}
#endif