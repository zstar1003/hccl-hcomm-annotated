/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_SUBER_CONN_RECONN_TIMER_INTERVAL (5000)
#define BKF_SUBER_CONN_RECONN_TIMER_TLS_INTERVAL (30000)
#define BKF_SUBER_CONN_RECONN_TIMER_INTERVALFORDISC (2000)
#include "v_stringlib.h"
#include "securec.h"
#include "bkf_msg.h"
#include "bkf_msg_codec.h"
#include "bkf_msg_wrap.h"
#include "bkf_suber_conn.h"
#include "bkf_suber_conn_data.h"
#include "bkf_suber_conn_fsm.h"

#ifdef __cplusplus
extern "C" {
#endif
#if BKF_BLOCK("私有函数定义")
uint32_t BkfSubConnReConnTmrProc(void *paramTmrStart, void *noUse)
{
    (void)noUse;
    BkfSuberConn *conn = (BkfSuberConn *)paramTmrStart;
    BkfSuberSessSetDisconnectReason(conn->sessMng, BKF_SUBER_DISCONNECT_REASON_CONNECTFAIL);
    if (conn->urlTypeMng->connMng->env->onDisConn) {
        conn->urlTypeMng->connMng->env->onDisConn(conn->urlTypeMng->connMng->env->cookie, &conn->puberUrl,
            &conn->localUrl, BkfSuberSessGetDisconnectReason(conn->sessMng));
    }
    BkfSuberSessResetDisconnectReason(conn->sessMng);
    BKF_LOG_DEBUG(conn->urlTypeMng->connMng->log, "SubConnReConnTmrProc_: con %x\n", BKF_MASK_ADDR(conn));
    return BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_STARTINST, VOS_NULL, VOS_NULL);
}

uint32_t BkfSubConnReConnTmrProcOnce(void *paramTmrStart, void *noUse)
{
    (void)noUse;
    BkfSuberConn *conn = (BkfSuberConn *)paramTmrStart;
    conn->reConnTmrId = VOS_NULL;
    BkfSuberSessResetDisconnectReason(conn->sessMng);
    return BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_STARTINST, VOS_NULL, VOS_NULL);
}

void BkfSuberConnStartReConnTmr(BkfSuberConn *conn)
{
    if (conn->reConnTmrId != VOS_NULL) {
        return;
    }
    uint32_t intervalMs = (BKF_URL_TYPE_IS_TCP(conn->puberUrl.type)) ? BKF_SUBER_CONN_RECONN_TIMER_TLS_INTERVAL
                                                                     : BKF_SUBER_CONN_RECONN_TIMER_INTERVAL;
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    BKF_LOG_DEBUG(connMng->log, "start reConnTmr_: conn %x, tmrval %u\n", BKF_MASK_ADDR(conn), intervalMs);
    conn->reConnTmrId = BkfTmrStartLoop(connMng->env->tmrMng, (F_BKF_TMR_TIMEOUT_PROC)BkfSubConnReConnTmrProc,
        intervalMs, conn);
    if (conn->reConnTmrId == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "Conn Create Timer failed.\n");
    }
    return;
}

void BkfSuberConnStartReConnTmrForDisc(BkfSuberConn *conn)
{
    if (conn->reConnTmrId != VOS_NULL) {
        return;
    }
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    conn->reConnTmrId = BkfTmrStartOnce(connMng->env->tmrMng, (F_BKF_TMR_TIMEOUT_PROC)BkfSubConnReConnTmrProcOnce,
        BKF_SUBER_CONN_RECONN_TIMER_INTERVALFORDISC, conn);
    if (conn->reConnTmrId == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "Conn Create Timer failed.\n");
    }
    return;
}

void BkfSuberConnStopReConnTmr(BkfSuberConn *conn)
{
    if (conn->reConnTmrId == VOS_NULL) {
        return;
    }
    BkfTmrStop(conn->urlTypeMng->connMng->env->tmrMng, conn->reConnTmrId);
    conn->reConnTmrId = VOS_NULL;
    return;
}

uint32_t BkfSuberConnWriteLastLeftData(BkfSuberConn *conn)
{
    if (conn->lastSendLeftDataBuf == VOS_NULL) {
        return BKF_OK;
    }

    int32_t leftLen = 0;
    int32_t dataLen = 0;
    errno_t err = EOK;
    int32_t writeLen = 0;
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    BkfSuberEnv *env = connMng->env;
    uint8_t *sendBuf = VOS_NULL;

    while ((leftLen = BkfBufqGetUsedLen(conn->lastSendLeftDataBuf)) > 0) {
        dataLen = BKF_GET_MIN(leftLen, BKF_CH_CLI_MALLOC_DATA_BUF_LEN_MAX);
        sendBuf = BkfChCliMallocDataBuf(env->chCliMng, conn->connId, dataLen);
        if (sendBuf == VOS_NULL) {
            return BKF_ERR;
        }

        err = memcpy_s(sendBuf, dataLen, BkfBufqGetUsedBegin(conn->lastSendLeftDataBuf), dataLen);
        BKF_ASSERT(err == EOK);
        writeLen = BkfChCliWrite(env->chCliMng, conn->connId, sendBuf, dataLen);
        if (writeLen == dataLen) {
            (void)BkfBufqDe(conn->lastSendLeftDataBuf, writeLen);
            sendBuf = VOS_NULL;
        } else if ((writeLen >= 0) && (writeLen < dataLen)) { /* 继续从缓冲中读取数据&发送 */
            (void)BkfBufqDe(conn->lastSendLeftDataBuf, writeLen);
            BkfChCliFreeDataBuf(env->chCliMng, conn->connId, sendBuf);
            sendBuf = VOS_NULL;
            /* 无法发送更多 */
            return BKF_ERR;
        } else {
            BKF_ASSERT(0);
            return BKF_SUBER_CONN_NEED_DELETE;
        }
    }

    BkfBufqUninit(conn->lastSendLeftDataBuf);
    conn->lastSendLeftDataBuf = VOS_NULL;
    return BKF_OK;
}

uint32_t BkfSuberConnMallocSendBuf(BkfSuberConn *conn, uint8_t **sendBuf, int32_t len)
{
    if (unlikely(*sendBuf != VOS_NULL)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    uint32_t ret = BkfSuberConnWriteLastLeftData(conn);
    if (ret != BKF_OK) {
        return ret;
    }

    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    BkfSuberEnv *env = connMng->env;
    *sendBuf = BkfChCliMallocDataBuf(env->chCliMng, conn->connId, len);
    if (*sendBuf == VOS_NULL) {
        BKF_LOG_INFO(connMng->log, "CreateMem Failed.\n");
        return BKF_ERR;
    }
    return BKF_OK;
}

uint32_t BkfSuberConnProcAfterWriteSendBuf(BkfSuberConn *conn, uint8_t *sendBuf, int32_t writeLen, int32_t needLen)
{
    if (writeLen == needLen) {
        return BKF_OK;
    }

    if ((writeLen < 0) || (writeLen > needLen)) {
        BKF_ASSERT(0);
        return BKF_SUBER_CONN_NEED_DELETE;
    }

    int32_t leftLen = needLen - writeLen;
    BkfSuberEnv *env = conn->urlTypeMng->connMng->env;
    conn->lastSendLeftDataBuf = BkfBufqInit(env->memMng, leftLen);
    if (conn->lastSendLeftDataBuf == VOS_NULL) {
        BKF_ASSERT(0);
        BkfChCliFreeDataBuf(env->chCliMng, conn->connId, sendBuf);
        return BKF_SUBER_CONN_NEED_DELETE;
    }

    (void)BkfBufqEn(conn->lastSendLeftDataBuf, sendBuf + writeLen, leftLen);
    BkfChCliFreeDataBuf(env->chCliMng, conn->connId, sendBuf);
    return BKF_OK;
}

uint32_t BkfSuberConnWriteSendBuf(BkfSuberConn *conn, uint8_t *sendBuf, int32_t len, BOOL ignoreSndRst)
{
    BkfSuberEnv *env = conn->urlTypeMng->connMng->env;

    if (conn->lastSendLeftDataBuf != VOS_NULL) {
        /* 忘记先发送上次未写成功的数据就发送新数据，错误，断言 */
        BKF_ASSERT(0);
        BkfChCliFreeDataBuf(env->chCliMng, conn->connId, sendBuf);
        return BKF_SUBER_CONN_NEED_DELETE;
    }

    int32_t writeLen = BkfChCliWrite(env->chCliMng, conn->connId, sendBuf, len);
    if (ignoreSndRst && writeLen == 0) {
        /* send fail,need free sendbuf */
        BKF_LOG_DEBUG(conn->urlTypeMng->connMng->log, "Connect Send Hello fail: conn %x\n", BKF_MASK_ADDR(conn));
        BkfChCliFreeDataBuf(env->chCliMng, conn->connId, sendBuf);
        return BKF_ERR;
    }
    return BkfSuberConnProcAfterWriteSendBuf(conn, sendBuf, writeLen, len);
}

uint32_t BkfSuberConnCreateChConn(BkfSuberConn *conn)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    BkfSuberEnv *env = connMng->env;
    /* chCli进行创建连接 */
    if (env->chCliMng == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "Connect Create Conn But ChCliMng VOS_NULL.\n");
        return BKF_ERR;
    }

    if (conn->connId != VOS_NULL) {
        return BKF_OK;
    }
    /* 两者互斥，传输层要么支持conn接口，要么支持connEx接口 */
    if (BkfUrlIsValid(&conn->localUrl)) {
        conn->connId = BkfChCliConnEx(env->chCliMng, &conn->puberUrl, &conn->localUrl);
    } else {
        conn->connId = BkfChCliConn(env->chCliMng, &(conn->puberUrl));
    }
    if (conn->connId == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "Connect Create ConnId Failed.\n");
        return BKF_ERR;
    }

    if (BkfSuberConnDataAddToConnIdSet(conn, conn->connId) != BKF_OK) {
        BKF_LOG_ERROR(connMng->log, "Add connid set Failed.\n");
        return BKF_SUBER_CONN_NEED_DELETE;
    }
    return BKF_OK;
}

void BkfSuberConnDelChConn(BkfSuberConn *conn)
{
    if (conn->connId == VOS_NULL) {
        return;
    }
    BkfChCliDisConn(conn->urlTypeMng->connMng->env->chCliMng, conn->connId);
    BkfSuberConnDataDelFromConnIdSet(conn);
    return;
}

uint32_t BkfSubConnActCreateConn(BkfSuberConn *conn, void *noUse1, void *noUse2)
{
    (void)noUse1;
    (void)noUse2;
    BkfSuberConnStartReConnTmr(conn);

    uint32_t ret = BkfSuberConnCreateChConn(conn);
    if (ret != BKF_OK) {
        return ret;
    }
    (void)BKF_FSM_CHG_STATE(&conn->fsm, BACKFIN_SUB_CONN_STATE_WAIT_SND_HELLO);
    return BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_UNBLOCK, VOS_NULL, VOS_NULL);
}

uint32_t BkfSubConnParseHelloAck(BkfSuberConn *conn, BkfMsgDecoder *decoder, BkfWrapMsgHelloAck *helloAck)
{
    BkfTL *tl = VOS_NULL;
    uint32_t errCode = BKF_OK;
    while ((tl = BkfMsgDecodeTLV(decoder, &errCode)) != VOS_NULL) {
        BKF_ASSERT(errCode == BKF_OK);

        if (tl->typeId == BKF_TLV_PUBER_NAME) {
            helloAck->puberName = (BkfTlvProjectName *)tl;
        } else if (tl->typeId == BKF_TLV_REASON_CODE) {
            helloAck->reasonCode = (BkfTlvReasonCode *)tl;
        } else if (tl->typeId == BKF_TLV_IDL_VERSION) {
            helloAck->puberIdlVersion = (BkfTlvIdlVersion *)tl;
        } else {
            BKF_LOG_ERROR(conn->urlTypeMng->connMng->log, "HelloACK tlv type(%u, %s)\n", tl->typeId,
                BkfTlvGetTypeStr(tl->typeId));
            errCode = BKF_ERR;
        }
    }
    return errCode;
}

uint32_t BkfSuberConnSndHello(BkfSuberConn *conn)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX] = {0};
    BKF_LOG_DEBUG(connMng->log, "Connect Send Hello.url %s\n", BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));

    uint8_t *sendBuf = VOS_NULL;
    uint32_t ret = BkfSuberConnMallocSendBuf(conn, &sendBuf, BKF_CH_CLI_MALLOC_DATA_BUF_LEN_MAX);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(connMng->log, "SendHello mem alloc failed. url %s\n",
            BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));
        return BKF_ERR;
    }
    BkfMsgCoder coder = {};
    BkfSuberEnv *env = connMng->env;
    ret = BkfMsgCodeInit(&coder, env->name, sendBuf, env->msgLenMax, env->sliceVTbl.keyLen, env->sliceVTbl.keyCodec,
        env->sliceVTbl.keyGetStrOrNull, connMng->log);
    ret |= BkfMsgCodeMsgHead(&coder, BKF_MSG_HELLO, 0);

    char *projectName = BKF_PROJECT_NAME;
    BkfTlvProjectVersion projectVersion = {.version = BKF_PROJECT_VERSION};
    ret |= BkfMsgCodeRawTLV(&coder, BKF_TLV_PROJECT_NAME, 0, projectName, VOS_StrLen((const char *)projectName) + 1,
        VOS_FALSE);
    ret |= BkfMsgCodeTLV(&coder, BKF_TLV_PROJECT_VERSION, 0, &projectVersion.tl + 1, VOS_FALSE);
    ret |= BkfMsgCodeRawTLV(&coder, BKF_TLV_SERVICE_NAME, 0, (void *)env->svcName, VOS_StrLen(env->svcName) + 1,
        VOS_FALSE);

    BkfTlvIdlVersion idlVersion = {.major = env->idlVersionMajor, .minor = env->idlVersionMinor};
    ret |= BkfMsgCodeRawTLV(&coder, BKF_TLV_SUBER_NAME, 0, (void *)env->name, VOS_StrLen(env->name) + 1, VOS_FALSE);
    if (BkfUrlIsValid(&env->lsnUrl)) {
        ret |= BkfMsgCodeTLV(&coder, BKF_TLV_IDL_VERSION, 0, &idlVersion.tl + 1, VOS_FALSE);
        ret |= BkfMsgCodeTLV(&coder, BKF_TLV_SUBER_LSNURL, 0, &env->lsnUrl, VOS_TRUE);
    } else {
        ret |= BkfMsgCodeTLV(&coder, BKF_TLV_IDL_VERSION, 0, &idlVersion.tl + 1, VOS_TRUE);
    }

    if (ret != BKF_OK) {
        BkfChCliFreeDataBuf(env->chCliMng, conn->connId, (void *)sendBuf);
        BKF_LOG_DEBUG(connMng->log, "Connect Send Hello encap fail. url %s\n",
            BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));
        return BKF_SUBER_SESS_SEND_BUF_NOT_ENOUGH;
    }

    int32_t needLen = BkfMsgCodeGetValidMsgLen(&coder);
    if (needLen == 0) {
        BkfChCliFreeDataBuf(conn->urlTypeMng->connMng->env->chCliMng, conn->connId, (void *)sendBuf);
        BKF_LOG_DEBUG(connMng->log, "Connect Send Hello needlen is zero.url %s\n",
            BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));
        return BKF_ERR;
    }

    return BkfSuberConnWriteSendBuf(conn, sendBuf, needLen, TRUE);
}

uint32_t BkfSubConnActSndHello(BkfSuberConn *conn, BkfMsgDecoder *decoder, void *param)
{
    (void)decoder;
    (void)param;
    uint32_t ret = BkfSuberConnSndHello(conn);
    if (ret == BKF_OK) {
        (void)BKF_FSM_CHG_STATE(&conn->fsm, BACKFIN_SUB_CONN_STATE_WAIT_HELLO_ACK);
    }
    return ret;
}

uint32_t BkfSubConnChkHelloAck(BkfSuberConn *conn, BkfWrapMsgHelloAck *helloAck)
{
    (void)conn;
    BOOL isValid = (helloAck->reasonCode != VOS_NULL) && (helloAck->reasonCode->reasonCode == BKF_RC_OK);
    if (!isValid) {
        return BKF_ERR;
    }

    return BKF_OK;
}

uint32_t BkfSubConnActRcvHelloAck(BkfSuberConn *conn, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    BkfWrapMsgHelloAck helloAck = {0};
    uint32_t ret = BkfSubConnParseHelloAck(conn, decoder, &helloAck);
    ret |= BkfSubConnChkHelloAck(conn, &helloAck);
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX] = {0};
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(connMng->log, "rcv hello ack, check fail.url %s\n",
            BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));
        (void)BKF_FSM_CHG_STATE(&conn->fsm, BACKFIN_SUB_CONN_STATE_WAIT_SND_HELLO);
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(connMng->log, "rcv hello ack.url %s\n", BkfUrlGetStr(&conn->puberUrl, urlStr, sizeof(urlStr)));
    BkfSuberConnStopReConnTmr(conn);
    (void)BKF_FSM_CHG_STATE(&conn->fsm, BACKFIN_SUB_CONN_STATE_UP);
    if (connMng->env->onConn) {
        connMng->env->onConn(connMng->env->cookie, &conn->puberUrl, &conn->localUrl, conn->srcPort);
    }
    return BkfSuberConnStartJob(conn); /* 建立连接后强制调度一次，避免sess之前发起的调度请求被忽略 */
}

uint32_t BkfSuberConnProcAfterSchedSess(BkfSuberConn *conn, BkfMsgCoder *coder, uint8_t *sendBuf, uint32_t retSchedSess)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    BkfSuberEnv *env = connMng->env;

    if (retSchedSess != BKF_OK && retSchedSess != BKF_SUBER_SESS_SEND_BUF_NOT_ENOUGH) {
        BkfChCliFreeDataBuf(env->chCliMng, conn->connId, (void *)sendBuf);
        return retSchedSess;
    }

    int32_t needLen = BkfMsgCodeGetValidMsgLen(coder);
    if (needLen == 0) {
        BkfChCliFreeDataBuf(env->chCliMng, conn->connId, sendBuf);
        return retSchedSess;
    }

    uint32_t ret = BkfSuberConnWriteSendBuf(conn, sendBuf, needLen, FALSE);
    if (ret == BKF_OK) {
        return retSchedSess;
    }

    return ret;
}

uint32_t BkfSubConnActSchedSess(BkfSuberConn *conn, BkfMsgDecoder *decoder, void *param)
{
    (void)decoder;
    (void)param;
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;
    uint8_t *sendBuf = VOS_NULL;
    uint32_t ret = BkfSuberConnMallocSendBuf(conn, &sendBuf, BKF_CH_CLI_MALLOC_DATA_BUF_LEN_MAX);
    if (ret != BKF_OK) {
        BKF_LOG_INFO(connMng->log, "Schedule CreateMem Failed.\n");
        return BKF_OK;
    }

    BkfSuberEnv *env = connMng->env;
    BkfMsgCoder coder = {};
    (void)BkfMsgCodeInit(&coder, env->name, sendBuf, env->msgLenMax, env->sliceVTbl.keyLen, env->sliceVTbl.keyCodec,
        env->sliceVTbl.keyGetStrOrNull, connMng->log);
    /* 驱动处于SUB/UNSUB状态的session进行sub、verify/unsub操作,unsub操作只发送一次 */
    ret = BkfSuberSessProcSched(conn->sessMng, &coder);
    return BkfSuberConnProcAfterSchedSess(conn, &coder, sendBuf, ret);
}

uint32_t BkfSubConnActTcpDisconn(BkfSuberConn *conn, BkfMsgDecoder *decoder, void *param)
{
    (void)decoder;
    (void)param;
    /* 通知所有会话回退状态，并更新序列号 */
    BKF_LOG_DEBUG(conn->urlTypeMng->connMng->log, "Act disconnect\n");
    uint32_t ret = BkfSuberSessProcDisconn(conn->sessMng);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(conn->urlTypeMng->connMng->log, "sess proc disc fail, ret %u.\n", ret);
        return ret;
    }
    (void)BKF_FSM_CHG_STATE(&conn->fsm, BACKFIN_SUB_CONN_STATE_CRTINGFD);
    BkfSuberConnDataDelFromConnIdSet(conn);

    // 启动定时器发起重连
    if (conn->isTrigConn) {
        if (!BKF_URL_TYPE_IS_TCP(conn->puberUrl.type)) {
            BkfSuberSessResetDisconnectReason(conn->sessMng);
            return BkfSuberConnFsmInputEvt(conn, BACKFIN_SUB_CONN_INPUT_STARTINST, VOS_NULL, VOS_NULL);
        }
        BkfSuberConnStartReConnTmrForDisc(conn);
    }
    return BKF_OK;
}

uint32_t BkfSubConnActReleaseFd(BkfSuberConn *conn, BkfMsgDecoder *decoder, void *param)
{
    (void)decoder;
    (void)param;
    BKF_LOG_ERROR(conn->urlTypeMng->connMng->log, "Rcv Release FD Msg.\n"); // 记录里程碑
    BkfSuberConnDelChConn(conn);
    (void)BKF_FSM_CHG_STATE(&conn->fsm, BACKFIN_SUB_CONN_STATE_CRTINGFD); // 回退到初始状态
    /* 通知所有会话回退状态，并更新序列号 */
    return BkfSuberSessProcDisconn(conn->sessMng);
}
#endif

#if BKF_BLOCK("表定义")
const char *subConnFsmStr = "SubscribeConnectionFsm";
const char *subConnTrsStr = "SubscribeConnectionTrs";
const char *subConnStateStrTbl[] = {"BACKFIN_SUB_CONN_STATE_CRTINGFD", "BACKFIN_SUB_CONN_STATE_WAIT_SND_HELLO",
    "BACKFIN_SUB_CONN_STATE_WAIT_HELLO_ACK", "BACKFIN_SUB_CONN_STATE_UP"};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(subConnStateStrTbl) == BACKFIN_SUB_CONN_STATE_MAX);

const char *subConnEvtStrTbl[] = {"BACKFIN_SUB_CONN_INPUT_STARTINST", "BACKFIN_SUB_CONN_INPUT_UNBLOCK",
    "BACKFIN_SUB_CONN_INPUT_RCVHELLOACK", "BACKFIN_SUB_CONN_INPUT_TCPDISCONN", "BACKFIN_SUB_CONN_INPUT_STOPINST",
    "BACKFIN_SUB_CONN_INPUT_JOB"};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(subConnEvtStrTbl) == BACKFIN_SUB_CONN_INPUT_MAX);

/*
N   BACKFIN_SUB_CONN_ACTION_NULL            空操作
Z   BACKFIN_SUB_CONN_ACTION_ASSERT          不应该进入的状态，打印断言
A   BACKFIN_SUB_CONN_ACTION_CREATEFD        创建连接，成功则等待发送hello，否则就定时重试
B   BACKFIN_SUB_CONN_ACTION_SNDHELLO        发送HELLO握手，成功就等待ACK，否则就定时尝试发送
C   BACKFIN_SUB_CONN_ACTION_RCVHELLOACK     收到HELLO-ACK
D   BACKFIN_SUB_CONN_ACTION_SCHED           调度连接上的所有会话
E   BACKFIN_SUB_CONN_ACTION_DISCONN         拆除连接，轮询connection
F   BACKFIN_SUB_CONN_ACTION_RELEASEFD       关闭连接，重新发起连接

---------------------------------------------------------------------------------------------------------
                                                  CRTINGFD    WAIT_SND_HELLO  WAIT_HELLO_ACK    UP    |
---------------------------------------------------------------------------------------------------------
 Inputs                                 states   |    0     |      1       |      2        |     3    |
---------------------------------------------------------------------------------------------------------
00   BACKFIN_SUB_CONN_INPUT_STARTINST(includeTmr)|   0  A   |    -  B      |   -  D        |   -  N   |
02   BACKFIN_SUB_CONN_INPUT_UNBLOCK              |   -  N   |    -  B      |   -  D        |   -  D   |
03   BACKFIN_SUB_CONN_INPUT_RCVHELLOACK          |   -  N   |    -  N      |   3  C        |      N   |
04   BACKFIN_SUB_CONN_INPUT_TCPDISCONN           |   -  F   |    0  F      |   0  F        |   0  F   |
05   BACKFIN_SUB_CONN_INPUT_STOPINST             |   -  E   |       E      |      E        |      E   |
06   BACKFIN_SUB_CONN_INPUT_JOB                  |   -  N   |    -  N      |   -  N        |   -  D   |
---------------------------------------------------------------------------------------------------------
*/
BkfFsmProcItem subConnStateEvtProcItemMtrx[BACKFIN_SUB_CONN_STATE_MAX][BACKFIN_SUB_CONN_INPUT_MAX] = {
    /* BACKFIN_SUB_CONN_STATE_CRTINGFD */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActCreateConn)}, /* BACKFIN_SUB_CONN_INPUT_STARTINST */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_CONN_INPUT_UNBLOCK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_CONN_INPUT_RCVHELLOACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_CONN_INPUT_TCPDISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActReleaseFd)},  /* BACKFIN_SUB_CONN_INPUT_STOPINST */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)}      /* BACKFIN_SUB_CONN_INPUT_JOB */
    },
    /* BACKFIN_SUB_CONN_STATE_WAIT_SND_HELLO */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActSndHello)},   /* BACKFIN_SUB_CONN_INPUT_STARTINST */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActSndHello)},   /* BACKFIN_SUB_CONN_INPUT_UNBLOCK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_CONN_INPUT_RCVHELLOACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActTcpDisconn)}, /* BACKFIN_SUB_CONN_INPUT_TCPDISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActReleaseFd)},  /* BACKFIN_SUB_CONN_INPUT_STOPINST */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)}      /* BACKFIN_SUB_CONN_INPUT_JOB */
    },
    /* BACKFIN_SUB_CONN_STATE_WAIT_HELLO_ACK */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActSndHello)},    /* BACKFIN_SUB_CONN_INPUT_STARTINST */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},      /* BACKFIN_SUB_CONN_INPUT_UNBLOCK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActRcvHelloAck)}, /* BACKFIN_SUB_CONN_INPUT_RCVHELLOACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActTcpDisconn)},  /* BACKFIN_SUB_CONN_INPUT_TCPDISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActReleaseFd)},   /* BACKFIN_SUB_CONN_INPUT_STOPINST */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)}       /* BACKFIN_SUB_CONN_INPUT_JOB */
    },
    /* BACKFIN_SUB_CONN_STATE_UP */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_CONN_INPUT_STARTINST */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActSchedSess)},  /* BACKFIN_SUB_CONN_INPUT_UNBLOCK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_CONN_INPUT_RCVHELLOACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActTcpDisconn)}, /* BACKFIN_SUB_CONN_INPUT_TCPDISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActReleaseFd)},  /* BACKFIN_SUB_CONN_INPUT_STOPINST */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubConnActSchedSess)}   /* BACKFIN_SUB_CONN_INPUT_JOB */
    },
};
#endif

#if BKF_BLOCK("公有函数定义")
uint32_t BkfSuberConnFsmInitFsmTmp(BkfSuberConnMng *connMng)
{
    BkfSuberEnv *env = connMng->env;
    int32_t err = snprintf_truncated_s(connMng->connFsmTmplName, sizeof(connMng->connFsmTmplName), "%s_ConnFsmTmpl",
        env->name);
    if (err <= 0) {
        return BKF_ERR;
    }

    BkfFsmTmplInitArg fsmArg = {0};
    fsmArg.evtCnt = BACKFIN_SUB_CONN_INPUT_MAX;
    fsmArg.stateInit = BACKFIN_SUB_CONN_STATE_CRTINGFD;
    fsmArg.stateCnt = BACKFIN_SUB_CONN_STATE_MAX;
    fsmArg.dbgOn = env->dbgOn;
    fsmArg.disp = env->disp;
    fsmArg.log = env->log;
    fsmArg.memMng = env->memMng;
    fsmArg.name = connMng->connFsmTmplName;
    fsmArg.stateStrTbl = subConnStateStrTbl;
    fsmArg.evtStrTbl = subConnEvtStrTbl;
    fsmArg.stateEvtProcItemMtrx = (const BkfFsmProcItem *)subConnStateEvtProcItemMtrx;
    fsmArg.getAppDataStrOrNull = VOS_NULL;

    connMng->connFsmTmpl = BkfFsmTmplInit(&fsmArg);
    if (connMng->connFsmTmpl == VOS_NULL) {
        return BKF_ERR;
    }
    return BKF_OK;
}

void BkfSuberConnFsmUnInitFsmTmp(BkfSuberConnMng *connMng)
{
    if (connMng->connFsmTmpl != VOS_NULL) {
        BkfFsmTmplUninit(connMng->connFsmTmpl);
    }
    return;
}
#endif

#ifdef __cplusplus
}
#endif
