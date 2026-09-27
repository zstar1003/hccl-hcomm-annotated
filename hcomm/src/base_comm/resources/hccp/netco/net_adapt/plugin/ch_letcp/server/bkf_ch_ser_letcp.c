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

#include <fcntl.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include "bkf_ch_ser_letcp_data.h"
#include "bkf_ch_ser_letcp_disp.h"
#include "bkf_url.h"
#include "securec.h"
#include "vos_base.h"
#include "vos_errno.h"
#include "bkf_ch_ser_letcp.h"

#ifdef __cplusplus
extern "C" {
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
/* common */
#define BKF_CH_SER_LETCP_KA_IDLE (60)
#define BKF_CH_SER_LETCP_KA_INTVL (20)
#define BKF_CH_SER_LETCP_KA_CNT (3)
#define BKF_CH_SER_LETCP_MAX_BUF (32 * 1024)

#define BKF_CH_SER_LETCP_ACCEPT_CONN_CNT_MAX (BKF_1K * 1)

#define BKF_CH_SER_LETCP_WRITE_ERR_RETRY_INTVL (1001)

#define BKF_SER_LETCPSTATE_INIT 0
#define BKF_SER_LETCPSTATE_ESTABLISHING 1
#define BKF_SER_LETCPSTATE_ESTABLISHED 2

#define BKF_SER_LETCP_RELISTEN_TIMEINTERVAL 5000 /* 5s  */

/* vtbl */
STATIC BkfChSer *BkfChSerLetcpInit(BkfChSerInitArg *arg);
STATIC void BkfChSerLetcpUninit(BkfChSer *ch);
STATIC uint32_t BkfChSerLetcpEnable(BkfChSer *ch, BkfChSerEnableArg *arg);

STATIC uint32_t BkfChSerLetcpListen(BkfChSer *ch, BkfUrl *urlSelf);
STATIC void BkfChSerLetcpUnlisten(BkfChSer *ch, BkfUrl *urlSelf);
STATIC void *BkfChSerLetcpMallocDataBuf(BkfChSerConnId *connId, int32_t dataBufLen);
STATIC void BkfChSerLetcpFreeDataBuf(BkfChSerConnId *connId, void *dataBuf);
STATIC int32_t BkfChSerLetcphRead(BkfChSerConnId *connId, void *dataBuf, int32_t bufLen);
STATIC int32_t BkfChSerLetcpWrite(BkfChSerConnId *connId, void *dataBuf, int32_t dataLen);
STATIC void BkfChSerLetcpClose(BkfChSerConnId *connId);

/* on msg */
STATIC void BkfChSerLetcpOnLsnFdEvents(int fd, uint32_t curEvents, BkfChSerLetcpLsn *lsn);
STATIC void BkfChSerLetcpOnConnFdEvents(int fd, uint32_t curEvents, BkfChSerConnId *connId);
STATIC uint32_t BkfChSerLetcpOnConnTmrWriteErrTO(BkfChSerConnId *connId, void *paramTmrLibUnknown);

/* proc */
STATIC void BkfChSerLetcpTry2StartOneLsn(BkfChSer *ch, BkfChSerLetcpLsn *lsn);
STATIC void BkfChSerLetcpTry2StartAllLsn(BkfChSer *ch);

STATIC void BkfChSerLetcpTry2WriteMore(BkfChSer *ch, BkfChSerConnId *connId);

STATIC uint32_t BkfChSerLetcpAttachConnFd(BkfChSer *ch, BkfChSerConnId *connId);
STATIC void BkfChSerLetcpProcLsnFdNewCli(BkfChSer *ch, BkfChSerLetcpLsn *lsn);
STATIC void BkfChSerLetcpProcConnFdCliDisc(BkfChSer *ch, BkfChSerConnId *connId);

STATIC void BkfChSerLetcpProcConnFdCliData(BkfChSer *ch, BkfChSerConnId *connId);

#pragma pack()
#endif

#if BKF_BLOCK("私有函数定义")

uint32_t BkfChSerLetcpStartConnTmrWriteErr(BkfChSer *ch, BkfChSerConnId *connId)
{
    return BkfChSerLetcpStartConnOnceTmrWriteErr(ch, connId, (F_BKF_TMR_TIMEOUT_PROC)BkfChSerLetcpOnConnTmrWriteErrTO,
        BKF_CH_SER_LETCP_WRITE_ERR_RETRY_INTVL);
}
/* vtbl */
STATIC BkfChSer *BkfChSerLetcpInit(BkfChSerInitArg *arg)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSer *ch = VOS_NULL;

    argIsInvalid = (arg == VOS_NULL) || (arg->base == VOS_NULL) || (arg->base->memMng == VOS_NULL) ||
                   (arg->base->disp == VOS_NULL) || (arg->base->tmrMng == VOS_NULL) ||
                   (arg->base->jobMng == VOS_NULL) || (arg->base->mux == VOS_NULL) || (arg->name == VOS_NULL);
    if (argIsInvalid) {
        return VOS_NULL;
    }

    ch = BkfChSerLetcpDataInit(arg);
    if (ch == VOS_NULL) {
        goto error;
    }
    BkfChSerLetcpDispInit(ch);

    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), init ok\n", BKF_MASK_ADDR(ch));
    return ch;

error:

    BkfChSerLetcpUninit(ch);
    return VOS_NULL;
}

STATIC void BkfChSerLetcpUninit(BkfChSer *ch)
{
    if (ch == VOS_NULL) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), 2uninit\n", BKF_MASK_ADDR(ch));

    BkfChSerLetcpDispUninit(ch);
    BkfChSerLetcpDataUninit(ch);
    return;
}

STATIC uint32_t BkfChSerLetcpEnable(BkfChSer *ch, BkfChSerEnableArg *arg)
{
    BOOL argIsInvalid = VOS_FALSE;

    argIsInvalid = (ch == VOS_NULL) || (arg == VOS_NULL) || (arg->onMayAccept == VOS_NULL) ||
                   (arg->onConn == VOS_NULL) || (arg->onRcvDataEvent == VOS_NULL) || (arg->onUnblock == VOS_NULL) ||
                   (arg->onDisconn == VOS_NULL);
    if (argIsInvalid) {
        return BKF_ERR;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), arg(%#x)/hasEnable(%u)\n", BKF_MASK_ADDR(ch), BKF_MASK_ADDR(arg),
        ch->hasEnable);

    if (ch->hasEnable) {
        return BKF_ERR;
    }
    ch->argEnable = *arg;
    ch->hasEnable = VOS_TRUE;
    BkfChSerLetcpTry2StartAllLsn(ch);
    return BKF_OK;
}

STATIC uint32_t BkfChSerLetcpListen(BkfChSer *ch, BkfUrl *urlSelf)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    if ((ch == VOS_NULL) || (urlSelf == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), urlSelf(%s)\n", BKF_MASK_ADDR(ch), BkfUrlGetStr(urlSelf, buf, sizeof(buf)));

    lsn = BkfChSerLetcpFindLsn(ch, urlSelf);
    if (lsn != VOS_NULL) {
        /* 对lsn下已经存在conn不做特殊处理 */
        return BKF_OK;
    }

    lsn = BkfChSerLetcpAddLsn(ch, urlSelf);
    if (lsn == VOS_NULL) {
        return BKF_ERR;
    }
    BkfChSerLetcpTry2StartOneLsn(ch, lsn);
    return BKF_OK;
}

STATIC void BkfChSerLetcpUnlisten(BkfChSer *ch, BkfUrl *urlSelf)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    if ((ch == VOS_NULL) || (urlSelf == VOS_NULL)) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), urlSelf(%s)\n", BKF_MASK_ADDR(ch), BkfUrlGetStr(urlSelf, buf, sizeof(buf)));

    lsn = BkfChSerLetcpFindLsn(ch, urlSelf);
    if (lsn != VOS_NULL) {
        BkfChSerLetcpDelLsn(ch, lsn);
    }
}

STATIC void *BkfChSerLetcpMallocDataBuf(BkfChSerConnId *connId, int32_t dataBufLen)
{
    BkfChSer *ch = VOS_NULL;
    BkfChSerLetcpMsgHead *chMsgHead = VOS_NULL;

    if ((connId == VOS_NULL) || (dataBufLen < 0)) {
        return VOS_NULL;
    }
    ch = connId->lsn->ch;
    chMsgHead = BkfChSerLetcpMsgMalloc(ch, dataBufLen);
    return (chMsgHead != VOS_NULL) ? chMsgHead->data : VOS_NULL;
}

STATIC void BkfChSerLetcpFreeDataBuf(BkfChSerConnId *connId, void *dataBuf)
{
    BkfChSer *ch = VOS_NULL;
    BkfChSerLetcpMsgHead *chMsgHead = VOS_NULL;

    if ((connId == VOS_NULL) || (dataBuf == VOS_NULL)) {
        return;
    }
    ch = connId->lsn->ch;

    chMsgHead = BKF_CH_SER_LETCP_APP_DATA_2CH_MSG_HEAD(dataBuf);
    if ((chMsgHead == VOS_NULL) || !BKF_SIGN_IS_VALID(chMsgHead->sign, BKF_CH_SER_LETCP_MSG_SIGN)) {
        BKF_ASSERT(0);
        return;
    }
    BkfChSerLetcpMsgFree(ch, chMsgHead);
}

STATIC int32_t BkfChSerLetcpRead(BkfChSerConnId *connId, void *dataBuf, int32_t bufLen)
{
    BkfChSer *ch = VOS_NULL;
    int32_t ret;

    if ((connId == VOS_NULL) || (dataBuf == VOS_NULL) || (bufLen < 0)) {
        return -1;
    }
    ch = connId->lsn->ch;
    ret = recv(connId->keyConnFd, dataBuf, bufLen, 0);
    if (ret == 0) { /* 上层判断断开依赖事件通知，不依赖读返回0 */
        BKF_LOG_WARN(BKF_LOG_HND, "connId(%d), ret(%d), ng\n", connId->keyConnFd, ret);
    } else if (ret < 0) {
        BKF_LOG_INFO(BKF_LOG_HND, "connId(%d), ret(%d)/errno(%u)\n", connId->keyConnFd, ret, errno);
    }
    return ret;
}

STATIC int32_t BkfChSerLetcpWriteChkArg(BkfChSerConnId *connId, void *dataBuf, int32_t dataLen,
    BkfChSerLetcpMsgHead **outChMsgHead)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerLetcpMsgHead *chMsgHead = VOS_NULL;

    argIsInvalid = (connId == VOS_NULL) || (dataBuf == VOS_NULL) || (dataLen < 0);
    if (argIsInvalid) {
        return -1;
    }

    chMsgHead = BKF_CH_SER_LETCP_APP_DATA_2CH_MSG_HEAD(dataBuf);
    argIsInvalid = (chMsgHead == VOS_NULL) || !BKF_SIGN_IS_VALID(chMsgHead->sign, BKF_CH_SER_LETCP_MSG_SIGN);
    if (argIsInvalid) {
        BKF_ASSERT(0);
        return -1;
    }

    *outChMsgHead = chMsgHead;
    return 0;
}
STATIC int32_t BkfChSerLetcpWrite(BkfChSerConnId *connId, void *dataBuf, int32_t dataLen)
{
    BkfChSer *ch = VOS_NULL;
    BkfChSerLetcpMsgHead *chMsgHead = VOS_NULL;
    int32_t ret;

    ret = BkfChSerLetcpWriteChkArg(connId, dataBuf, dataLen, &chMsgHead);
    if (ret < 0) {
        return ret;
    }
    ch = connId->lsn->ch;
    if (dataLen == 0) {
        BkfChSerLetcpMsgFree(ch, chMsgHead);
        return 0;
    }
    ret = send(connId->keyConnFd, dataBuf, dataLen, 0);
    if (ret == 0) {
        BKF_LOG_WARN(BKF_LOG_HND, "connId(%d), ret(%d), ng\n", connId->keyConnFd, ret);
        return 0;
    } else if (ret < 0) {
        (void)BkfChSerLetcpStartConnTmrWriteErr(ch, connId);
        return 0;
    } else if (ret == dataLen) {
        BkfChSerLetcpMsgFree(ch, chMsgHead);
    }
    return ret;
}

STATIC void BkfChSerLetcpClose(BkfChSerConnId *connId)
{
    BkfChSer *ch = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    if (connId == VOS_NULL) {
        return;
    }
    ch = connId->lsn->ch;
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x)/url(%s)\n", BKF_MASK_ADDR(ch), BkfUrlGetStr(&connId->urlCli, buf, sizeof(buf)));
    BkfChSerLetcpDelConnId(ch, connId);
}

/* on msg */
STATIC void BkfChSerLetcpOnLsnFdEvents(int fd, uint32_t curEvents, BkfChSerLetcpLsn *lsn)
{
    (void)fd;
    BkfChSer *ch = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    if ((lsn == VOS_NULL) || !BKF_SIGN_IS_VALID(lsn->sign, BKF_CH_SER_LETCP_LSN_SIGN)) {
        BKF_ASSERT(0); /* mux的错误 */
        return;
    }
    ch = lsn->ch;
    BKF_LOG_DEBUG(BKF_LOG_HND, "keyUrlSelf(%s), lsnFd(%d)/curEvents(%#x)\n",
        BkfUrlGetStr(&lsn->keyUrlSelf, buf, sizeof(buf)), lsn->lsnFd, curEvents);
    if (BKF_BIT_TEST(curEvents, EPOLLIN)) {
        BkfChSerLetcpProcLsnFdNewCli(ch, lsn);
    }
}

void BkfChSerLetcpOnConnFdEpollout(BkfChSerConnId *connId)
{
    BkfChSer *ch = connId->lsn->ch;
    uint8_t buf[BKF_1K / 8];
    if (connId->tcpState == BKF_SER_LETCPSTATE_ESTABLISHED) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "OnConnFdEpollout connId(%#x), Fd(%d)/cliurl(%s)\n", BKF_MASK_ADDR(connId),
        connId->keyConnFd, BkfUrlGetStr(&connId->urlCli, buf, sizeof(buf)));
    BKF_LOG_DEBUG(BKF_LOG_HND, "OnConnFdEpollout sockFd[%d] tcp success\n", connId->keyConnFd);
    connId->tcpState = BKF_SER_LETCPSTATE_ESTABLISHED;
    if (ch->argEnable.onConnEx) {
        struct sockaddr_in addr;
        socklen_t addrlen = sizeof(addr);
        int32_t ret = getsockname(connId->keyConnFd, (struct sockaddr *)&addr, &addrlen);
        (void)BkfUrlLeTcpSet(&connId->urlLocal, VOS_NTOHL(addr.sin_addr.s_addr), VOS_NTOHS(addr.sin_port));
        BKF_LOG_DEBUG(BKF_LOG_HND, "onConnEx: localAddr 0x%x, port %u ret %d, err %d\n",
            VOS_NTOHL(addr.sin_addr.s_addr), VOS_NTOHS(addr.sin_port), ret, errno);
        ch->argEnable.onConnEx(ch->argEnable.cookie, connId, &connId->urlCli, &connId->urlLocal);
    }
    if (ch->argEnable.onUnblock) {
        ch->argEnable.onUnblock(ch->argEnable.cookie, connId);
    }
    return;
}

STATIC void BkfChSerLetcpOnConnFdEvents(int fd, uint32_t curEvents, BkfChSerConnId *connId)
{
    (void)fd;
    BkfChSer *ch = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    if ((connId == VOS_NULL) || !BKF_SIGN_IS_VALID(connId->sign, BKF_CH_SER_LETCP_CONN_ID_SIGN)) {
        BKF_ASSERT(0); /* mux的错误 */
        return;
    }
    ch = connId->lsn->ch;
    BKF_LOG_DEBUG(BKF_LOG_HND, "keyConnFd(%d), urlCli(%s)/curEvents(%#x)\n", connId->keyConnFd,
        BkfUrlGetStr(&connId->urlCli, buf, sizeof(buf)), curEvents);
    if (BKF_BIT_TEST(curEvents, EPOLLERR | EPOLLHUP)) {
        BkfChSerLetcpProcConnFdCliDisc(ch, connId);
    } else {
        if (BKF_BIT_TEST(curEvents, EPOLLOUT)) {
            BkfChSerLetcpOnConnFdEpollout(connId);
        }
        if (BKF_BIT_TEST(curEvents, EPOLLIN)) {
            BkfChSerLetcpProcConnFdCliData(ch, connId);
        }
    }
}

STATIC uint32_t BkfChSerLetcpOnConnTmrWriteErrTO(BkfChSerConnId *connId, void *paramTmrLibUnknown)
{
    (void)paramTmrLibUnknown;
    BkfChSer *ch = connId->lsn->ch;

    connId->tmrIdWriteErr = VOS_NULL;
    BkfChSerLetcpTry2WriteMore(ch, connId);
    return BKF_OK;
}

/* proc */
uint32_t BkfChSerLetcpCreateLsnFd(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    if (lsn->lsnFd != -1) {
        return BKF_OK;
    }
    lsn->lsnFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (lsn->lsnFd == -1) {
        BKF_LOG_WARN(BKF_LOG_HND, "lsnFd(%d), errno(%d)\n", lsn->lsnFd, errno);
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "fd(%d), ok\n", lsn->lsnFd);
    return BKF_OK;
}

// 设置非阻塞式
STATIC uint32_t BkfChSerLetcpSetFdNonBlock(BkfChSer *ch, int32_t fd)
{
    int32_t flags = fcntl(fd, F_GETFL);
    if (flags == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), flags(%d)/errno(%d), ng\n", fd, flags, errno);
        return BKF_ERR;
    }

    int32_t ret = fcntl(fd, F_SETFL, (uint32_t)flags | O_NONBLOCK);
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%d), ng\n", fd, ret, errno);
        return BKF_ERR;
    }

    return BKF_OK;
}

uint32_t BkfChSerLetcpSetFdOptKa(BkfChSer *ch, int32_t fd)
{
    int32_t optval = 1;
    int32_t ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t recvBuf = BKF_CH_SER_LETCP_MAX_BUF;
    ret = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &recvBuf, sizeof(recvBuf));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t sendBuf = BKF_CH_SER_LETCP_MAX_BUF;
    ret = setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sendBuf, sizeof(sendBuf));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t keepSwitch = 1;
    ret = setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepSwitch, sizeof(keepSwitch));
    if (ret < 0) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t keepIdle = BKF_CH_SER_LETCP_KA_IDLE;
    ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &keepIdle, sizeof(keepIdle));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t keepIntvl = BKF_CH_SER_LETCP_KA_INTVL;
    ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &keepIntvl, sizeof(keepIntvl));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t keepCnt = BKF_CH_SER_LETCP_KA_CNT;
    ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &keepCnt, sizeof(keepCnt));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    return BKF_OK;
}

uint32_t BkfChSerLetcpProcReListenCallback(void *cookie, void *paramTmrLibUnknown)
{
    (void)paramTmrLibUnknown;
    BkfChSerLetcpLsn *lsn = (BkfChSerLetcpLsn *)cookie;
    if (lsn == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    BkfChSer *ch = lsn->ch;
    lsn->tmrIdReLsn = VOS_NULL;
    BKF_LOG_INFO(BKF_LOG_HND, "lsn %#x fd %d\n", BKF_MASK_ADDR(lsn), lsn->lsnFd);
    if (lsn->lsnFd == -1) {
        return BKF_ERR;
    }
    int32_t ret;
    if (lsn->lsnBindOk == VOS_FALSE) {
        struct sockaddr_in stAddr = {0};
        stAddr.sin_family = AF_INET;
        stAddr.sin_addr.s_addr = VOS_HTONL(lsn->keyUrlSelf.ip.addrH);
        stAddr.sin_port = VOS_HTONS(lsn->keyUrlSelf.ip.port);
        ret = bind(lsn->lsnFd, (struct sockaddr *)&stAddr, sizeof(stAddr));
        if (ret == -1) {
            (void)BkfChSerLetcpStartOnceTmrReListen(ch, lsn, BkfChSerLetcpProcReListenCallback,
                BKF_SER_LETCP_RELISTEN_TIMEINTERVAL);
            BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%d), bind err ng\n", lsn->lsnFd, ret, errno);
            return BKF_OK;
        }
        lsn->lsnBindOk = VOS_TRUE;
    }
    if (lsn->lsnOk == VOS_FALSE) {
        ret = listen(lsn->lsnFd, BKF_CH_SER_LETCP_ACCEPT_CONN_CNT_MAX);
        if (ret == -1) {
            (void)BkfChSerLetcpStartOnceTmrReListen(ch, lsn, BkfChSerLetcpProcReListenCallback,
                BKF_SER_LETCP_RELISTEN_TIMEINTERVAL);
            BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%d),lsn err ng\n", lsn->lsnFd, ret, errno);
            return BKF_OK;
        }
        lsn->lsnOk = VOS_TRUE;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "fd(%d), ok\n", lsn->lsnFd);
    return BKF_OK;
}

STATIC uint32_t BkfChSerLetcpStartLsnFd(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    int32_t ret;
    struct sockaddr_in stAddr = {0};
    stAddr.sin_family = AF_INET;
    stAddr.sin_addr.s_addr = VOS_HTONL(lsn->keyUrlSelf.ip.addrH);
    stAddr.sin_port = VOS_HTONS(lsn->keyUrlSelf.ip.port);

    ret = bind(lsn->lsnFd, (struct sockaddr *)&stAddr, sizeof(stAddr));
    if (ret == -1) {
        (void)BkfChSerLetcpStartOnceTmrReListen(ch, lsn, BkfChSerLetcpProcReListenCallback,
            BKF_SER_LETCP_RELISTEN_TIMEINTERVAL);
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%d), bind ng\n", lsn->lsnFd, ret, errno);
        return BKF_ERR;
    }
    lsn->lsnBindOk = VOS_TRUE;
    ret = listen(lsn->lsnFd, BKF_CH_SER_LETCP_ACCEPT_CONN_CNT_MAX);
    if (ret == -1) {
        (void)BkfChSerLetcpStartOnceTmrReListen(ch, lsn, BkfChSerLetcpProcReListenCallback,
            BKF_SER_LETCP_RELISTEN_TIMEINTERVAL);
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%d),lsn ng\n", lsn->lsnFd, ret, errno);
        return BKF_ERR;
    }
    lsn->lsnOk = VOS_TRUE;
    BKF_LOG_DEBUG(BKF_LOG_HND, "fd(%d), ok\n", lsn->lsnFd);
    return BKF_OK;
}

STATIC uint32_t BkfChSerLetcpAttachLsnFd(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    int ret;
    ret = WfwMuxAttachFd(ch->argInit.base->mux, lsn->lsnFd, EPOLLIN | EPOLLET,
        (F_WFW_MUX_FD_PROC)BkfChSerLetcpOnLsnFdEvents, lsn, VOS_NULL);
    if (ret == -1) {
        BKF_LOG_WARN(BKF_LOG_HND, "lsnFd(%d), errno(%u), attach fd ng\n", lsn->lsnFd, errno);
        return BKF_ERR;
    }

    lsn->lsnFdAttachOk = VOS_TRUE;
    BKF_LOG_DEBUG(BKF_LOG_HND, "fd(%d), ok\n", lsn->lsnFd);
    return BKF_OK;
}

STATIC void BkfChSerLetcpTry2StartOneLsn(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    uint32_t ret;
    uint8_t buf[BKF_1K / 8];

    BKF_LOG_DEBUG(BKF_LOG_HND, "ch(%#x), lsnUrl(%s)\n", BKF_MASK_ADDR(ch),
        BkfUrlGetStr(&lsn->keyUrlSelf, buf, sizeof(buf)));

    if (!ch->hasEnable) {
        return;
    }
    ret = BkfChSerLetcpCreateLsnFd(ch, lsn);
    if (ret != BKF_OK) {
        return;
    }
    ret = BkfChSerLetcpSetFdNonBlock(ch, lsn->lsnFd);
    ret |= BkfChSerLetcpSetFdOptKa(ch, lsn->lsnFd);
    ret |= BkfChSerLetcpStartLsnFd(ch, lsn);
    ret |= BkfChSerLetcpAttachLsnFd(ch, lsn);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
    }
}

STATIC void BkfChSerLetcpTry2StartAllLsn(BkfChSer *ch)
{
    BkfChSerLetcpLsn *lsn = VOS_NULL;
    void *itor = VOS_NULL;

    for (lsn = BkfChSerLetcpGetFirstLsn(ch, &itor); lsn != VOS_NULL; lsn = BkfChSerLetcpGetNextLsn(ch, &itor)) {
        BkfChSerLetcpTry2StartOneLsn(ch, lsn);
    }
}

STATIC void BkfChSerLetcpTry2WriteMore(BkfChSer *ch, BkfChSerConnId *connId)
{
    ch->argEnable.onUnblock(ch->argEnable.cookie, connId);
}

int32_t BkfChSerLetcpAccept(BkfChSer *ch, BkfChSerLetcpLsn *lsn, BkfUrl *cliUrl)
{
    int32_t connFd = -1;
    uint8_t buf[BKF_1K / 8];
    struct sockaddr_in stAddr = {0};
    stAddr.sin_family = AF_INET;
    stAddr.sin_addr.s_addr = VOS_HTONL(cliUrl->ip.addrH);
    stAddr.sin_port = VOS_HTONS(cliUrl->ip.port);
    socklen_t cli_addr_len = sizeof(stAddr);

    connFd = accept(lsn->lsnFd, (struct sockaddr *)&stAddr, &cli_addr_len);
    if (connFd == -1) {
        BKF_LOG_WARN(BKF_LOG_HND, "connFd(%d)/errno(%d), ng\n", connFd, errno);
    }

    (void)BkfUrlLeTcpSet(cliUrl, VOS_NTOHL(cliUrl->ip.addrH), VOS_NTOHS(cliUrl->ip.port));
    BKF_LOG_DEBUG(BKF_LOG_HND, "connFd(%d), cliUrl(%s)\n", connFd, BkfUrlGetStr(cliUrl, buf, sizeof(buf)));
    return connFd;
}

STATIC uint32_t BkfChSerLetcpAttachConnFd(BkfChSer *ch, BkfChSerConnId *connId)
{
    int ret;
    /*
    1. linux epoll 默认不需要加hup/err，但mesh需要
    2. DECONGEST 是边沿触发方式
    */
    uint32_t events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLOUT | EPOLLET;

    if (!connId->connFdAttachOk) {
        ret = WfwMuxAttachFd(ch->argInit.base->mux, connId->keyConnFd, events,
            (F_WFW_MUX_FD_PROC)BkfChSerLetcpOnConnFdEvents, connId, VOS_NULL);
        if (ret == 0) {
            connId->connFdAttachOk = VOS_TRUE;
        }
    } else {
        ret = WfwMuxReattachFd(ch->argInit.base->mux, connId->keyConnFd, events,
            (F_WFW_MUX_FD_PROC)BkfChSerLetcpOnConnFdEvents, connId, VOS_NULL);
    }
    if (ret < 0) {
        BKF_LOG_WARN(BKF_LOG_HND, "connFd(%d), errno(%u), attach fd ng\n", connId->keyConnFd, errno);
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "connFd(%d), ok\n", connId->keyConnFd);
    return BKF_OK;
}

STATIC void BkfChSerLetcpProcLsnFdNewCli(BkfChSer *ch, BkfChSerLetcpLsn *lsn)
{
    int32_t connFd = -1;
    BkfUrl cliUrl;
    uint32_t ret;
    BkfChSerConnId *connId = VOS_NULL;
    BOOL mayAccept = VOS_FALSE;
    uint8_t buf[BKF_1K / 8];

    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), keyLsnUrl(%s)/lsnFd(%d)\n", BKF_MASK_ADDR(ch),
        BkfUrlGetStr(&lsn->keyUrlSelf, buf, sizeof(buf)), lsn->lsnFd);

    connFd = BkfChSerLetcpAccept(ch, lsn, &cliUrl);
    if (connFd == -1) {
        goto error;
    }
    mayAccept = ch->argEnable.onMayAccept(ch->argEnable.cookie, &cliUrl);
    if (!mayAccept) {
        goto error;
    }
    connId = BkfChSerLetcpAddConnId(ch, lsn, connFd, &cliUrl);
    if (connId == VOS_NULL) {
        goto error;
    }
    ret = BkfChSerLetcpSetFdNonBlock(ch, connFd);
    ret |= BkfChSerLetcpSetFdOptKa(ch, connFd);
    ret |= BkfChSerLetcpAttachConnFd(ch, connId);
    if (ret != BKF_OK) {
        goto error;
    }
    return;
error:

    if (connId != VOS_NULL) {
        BkfChSerLetcpDelConnId(ch, connId);
    } else if (connFd != -1) {
        (void)close(connFd);
    }
    return;
}

STATIC void BkfChSerLetcpProcConnFdCliDisc(BkfChSer *ch, BkfChSerConnId *connId)
{
    uint8_t buf[BKF_1K / 8];

    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), connFd(%d)/url(%s)\n", BKF_MASK_ADDR(ch), connId->keyConnFd,
        BkfUrlGetStr(&connId->urlCli, buf, sizeof(buf)));

    connId->lockDel = VOS_TRUE;
    if (ch->argEnable.onDisconn) {
        ch->argEnable.onDisconn(ch->argEnable.cookie, connId);
    }
    connId->lockDel = VOS_FALSE;
    BkfChSerLetcpDelConnId(ch, connId);
}

STATIC void BkfChSerLetcpProcConnFdCliData(BkfChSer *ch, BkfChSerConnId *connId)
{
    connId->lockDel = VOS_TRUE;
    if (ch->argEnable.onRcvDataEvent) {
        ch->argEnable.onRcvDataEvent(ch->argEnable.cookie, connId);
    }
    connId->lockDel = VOS_FALSE;
    if (connId->softDel) {
        BkfChSerLetcpDelConnId(ch, connId);
    }
}
#endif

#if BKF_BLOCK("公有函数定义")
/* func */
uint32_t BkfChSerLetcpBuildVTbl(char *name, BkfChSerTypeVTbl *temp, uint32_t nameLen)
{
    (void)nameLen;
    if ((name == VOS_NULL) || (temp == VOS_NULL)) {
        return BKF_ERR;
    }

    temp->name = name;
    temp->typeId = BKF_URL_TYPE_LETCP;
    temp->init = BkfChSerLetcpInit;
    temp->uninit = BkfChSerLetcpUninit;
    temp->enable = BkfChSerLetcpEnable;
    temp->setSelfCid = VOS_NULL;
    temp->listen = BkfChSerLetcpListen;
    temp->unlisten = BkfChSerLetcpUnlisten;
    temp->mallocDataBuf = BkfChSerLetcpMallocDataBuf;
    temp->freeDataBuf = BkfChSerLetcpFreeDataBuf;
    temp->read = BkfChSerLetcpRead;
    temp->write = BkfChSerLetcpWrite;
    temp->close = BkfChSerLetcpClose;
    return BKF_OK;
}
#endif

#ifdef __cplusplus
}
#endif
