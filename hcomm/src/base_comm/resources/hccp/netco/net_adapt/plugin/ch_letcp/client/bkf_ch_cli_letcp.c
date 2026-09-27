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
#include <errno.h>
#include "bkf_ch_cli_letcp_data.h"
#include "bkf_ch_cli_letcp_disp.h"
#include "bkf_url.h"
#include "vos_base.h"
#include "vos_errno.h"
#include "bkf_ch_cli_letcp.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
/* common */
#define BKF_CH_CLI_LETCP_KA_IDLE (60)
#define BKF_CH_CLI_LETCP_KA_INTVL (20)
#define BKF_CH_CLI_LETCP_KA_CNT (3)
#define BKF_CH_CLI_LETCP_MAX_BUF (32 * 1024)

#define BKF_CLI_LETCPSTATE_INIT 0
#define BKF_CLI_LETCPSTATE_ESTABLISHING 1
#define BKF_CLI_LETCPSTATE_ESTABLISHED 2

#define BKF_CH_CLI_LETCP_WRITE_ERR_RETRY_INTVL (1001)

/* vtbl */
STATIC BkfChCli *BkfChCliLetcpInit(BkfChCliInitArg *arg);
STATIC void BkfChCliLetcpUninit(BkfChCli *ch);
STATIC uint32_t BkfChCliLetcpEnable(BkfChCli *ch, BkfChCliEnableArg *arg);
STATIC uint32_t BkfChCliLetcpSetSelfUrl(BkfChCli *ch, BkfUrl *selfUrl);

STATIC BkfChCliConnId *BkfChCliLetcpConnEx(BkfChCli *ch, BkfUrl *urlSer, BkfUrl *selfUrl);
STATIC void BkfChCliLetcpDisconn(BkfChCli *ch, BkfChCliConnId *connId);
STATIC void *BkfChCliLetcpMallocDataBuf(BkfChCliConnId *connId, int32_t dataBufLen);
STATIC void BkfChCliLetcpFreeDataBuf(BkfChCliConnId *connId, void *dataBuf);
STATIC int32_t BkfChCliLetcpRead(BkfChCliConnId *connId, void *dataBuf, int32_t bufLen);
STATIC int32_t BkfChCliLetcpWrite(BkfChCliConnId *connId, void *dataBuf, int32_t dataLen);

/* om msg */
STATIC void BkfChCliLetcpOnConnFdEvents(int32_t fd, uint32_t curEvents, BkfChCliConnId *connId);
STATIC uint32_t BkfChCliLetcpOnConnTmrWriteErrTO(BkfChCliConnId *connId, void *paramTmrLibUnknown);

/* proc */
STATIC BkfChCliConnId *BkfChCliLetcpCreateOneConn(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlSelf);
STATIC BkfChCliConnId *BkfChCliLetcpReconnToSer(BkfChCliConnId *connId);

STATIC void BkfChCliLetcpDelOneConn(BkfChCliConnId *connId);
STATIC void BkfChCliLetcpDelAllConn(BkfChCli *ch);

STATIC uint32_t BkfChCliLetcpConnect(BkfChCliConnId *connId);
STATIC void BkfChCliLetcpTry2WriteMore(BkfChCliConnId *connId);

STATIC uint32_t BkfChCliLetcpAttachConnFd(BkfChCliConnId *connId, uint32_t events);
void BkfChCliLetcpDetachConnFd(BkfChCliConnId *connId);

STATIC void BkfChCliLetcpProcConnFdSerDisc(BkfChCliConnId *connId);
STATIC void BkfChCliLetcpProcConnFdSerData(BkfChCliConnId *connId);

STATIC void BkfChCliLetcpSetFdBlock(int32_t fd);

static inline void BkfChCliLetcpClearSoftDel(BkfChCliConnId *connId)
{
    if (connId->lockDel) {
        connId->softDel = VOS_FALSE;
    }
    return;
}

static inline uint32_t BkfChCliLetcpSoftDel(BkfChCliConnId *connId)
{
    if (connId->lockDel) {
        connId->softDel = VOS_TRUE;
        return BKF_OK;
    }
    return BKF_ERR;
}

uint32_t BkfChCliLetcpStartConnTmrWriteErr(BkfChCliConnId *connId)
{
    return BkfChCliLetcpStartConnOnceTmrWriteErr(connId, (F_BKF_TMR_TIMEOUT_PROC)BkfChCliLetcpOnConnTmrWriteErrTO,
        BKF_CH_CLI_LETCP_WRITE_ERR_RETRY_INTVL);
}

#endif

#if BKF_BLOCK("公有函数定义")
/* func */
uint32_t BkfChCliLetcpBuildVTbl(char *name, BkfChCliTypeVTbl *cliVTbl, uint32_t nameLen)
{
    (void)nameLen;
    if ((name == VOS_NULL) || (cliVTbl == VOS_NULL)) {
        return BKF_ERR;
    }

    cliVTbl->name = name;
    cliVTbl->typeId = BKF_URL_TYPE_LETCP;
    cliVTbl->init = BkfChCliLetcpInit;
    cliVTbl->uninit = BkfChCliLetcpUninit;
    cliVTbl->enable = BkfChCliLetcpEnable;
    cliVTbl->setSelfCid = VOS_NULL;
    cliVTbl->setSelfUrl = BkfChCliLetcpSetSelfUrl;
    cliVTbl->connEx = BkfChCliLetcpConnEx;
    cliVTbl->disConn = BkfChCliLetcpDisconn;
    cliVTbl->mallocDataBuf = BkfChCliLetcpMallocDataBuf;
    cliVTbl->freeDataBuf = BkfChCliLetcpFreeDataBuf;
    cliVTbl->read = BkfChCliLetcpRead;
    cliVTbl->write = BkfChCliLetcpWrite;
    return BKF_OK;
}

#endif

#if BKF_BLOCK("私有函数定义")
/* vtbl */
STATIC BkfChCli *BkfChCliLetcpInit(BkfChCliInitArg *arg)
{
    BOOL argIsInvalid = (arg == VOS_NULL) || (arg->base == VOS_NULL) || (arg->base->memMng == VOS_NULL) ||
                        (arg->base->disp == VOS_NULL) || (arg->base->mux == VOS_NULL) ||
                        (arg->base->tmrMng == VOS_NULL) || (arg->name == VOS_NULL);
    if (argIsInvalid) {
        return VOS_NULL;
    }

    BkfChCli *ch = BkfChCliLetcpDataInit(arg);
    if (ch == VOS_NULL) {
        return VOS_NULL;
    }
    BkfChCliLetcpDispInit(ch);

    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), init ok\n", BKF_MASK_ADDR(ch));
    return ch;
}

STATIC void BkfChCliLetcpUninit(BkfChCli *ch)
{
    if (unlikely(ch == VOS_NULL)) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), 2uninit\n", BKF_MASK_ADDR(ch));
    BkfChCliLetcpDelAllConn(ch);
    BkfChCliLetcpDispUninit(ch);
    BkfChCliLetcpDataUninit(ch);
    return;
}

STATIC uint32_t BkfChCliLetcpEnable(BkfChCli *ch, BkfChCliEnableArg *arg)
{
    BOOL argIsInvalid = VOS_FALSE;

    argIsInvalid = (ch == VOS_NULL) || (arg == VOS_NULL) || (arg->onRcvDataEvent == VOS_NULL) ||
                   (arg->onUnblock == VOS_NULL) || (arg->onDisconn == VOS_NULL);
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
    return BKF_OK;
}

/*
    1. 业务设置了selfUrl后，传输层不会将以前的连接断开。仅对后续新的连接生效。
    2. 如需重新建连，业务需要调用disconn后，再重新建立连接。
*/
STATIC uint32_t BkfChCliLetcpSetSelfUrl(BkfChCli *ch, BkfUrl *selfUrl)
{
    BKF_RETURNVAL_IF(ch == VOS_NULL, BKF_ERR);
    BKF_RETURNVAL_IF(selfUrl == VOS_NULL, BKF_ERR);
    BKF_RETURNVAL_IF(selfUrl->type != BKF_URL_TYPE_LETCP, BKF_ERR);
    uint8_t buf[BKF_1K / 8];

    ch->selfUrl = *selfUrl;
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), selfUrl(%s)\n", BKF_MASK_ADDR(ch), BkfUrlGetStr(selfUrl, buf, sizeof(buf)));
    return BKF_OK;
}

STATIC BkfChCliConnId *BkfChCliLetcpConnEx(BkfChCli *ch, BkfUrl *urlSer, BkfUrl *selfUrl)
{
    BKF_RETURNNULL_IF(ch == VOS_NULL);
    BKF_RETURNNULL_IF(urlSer == VOS_NULL);
    BKF_RETURNNULL_IF(urlSer->type != BKF_URL_TYPE_LETCP);
    if (!ch->hasEnable) {
        BKF_LOG_WARN(BKF_LOG_HND, "connId create failed, enable(%u)\n", ch->hasEnable);
        return VOS_NULL;
    }

    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x), urlSer(%s), selfUrl(%s)\n", BKF_MASK_ADDR(ch),
        BkfUrlGetStr(urlSer, buf, sizeof(buf)), BkfUrlGetStr(selfUrl, buf2, sizeof(buf2)));

    BkfChCliConnId *connId = BkfChCliLetcpFindConnId(ch, urlSer, selfUrl);
    if (connId != VOS_NULL) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "connId(%#x), urlSer(%s) not null\n", BKF_MASK_ADDR(connId),
            BkfUrlGetStr(urlSer, buf, sizeof(buf)));
        return BkfChCliLetcpReconnToSer(connId);
    }

    return BkfChCliLetcpCreateOneConn(ch, urlSer, selfUrl);
}

STATIC void BkfChCliLetcpDisconn(BkfChCli *ch, BkfChCliConnId *connId)
{
    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];

    BKF_RETURNvoid_IF(ch == VOS_NULL);
    BKF_RETURNvoid_IF(connId == VOS_NULL);
    BKF_RETURNvoid_IF(connId->ch == VOS_NULL);
    BKF_LOG_INFO(BKF_LOG_HND, "ch(%#x)/serUrl(%s)/localUrl(%s), fd %d, connId %#x\n", BKF_MASK_ADDR(ch),
        BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)), BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)),
        connId->connFd, BKF_MASK_ADDR(connId));

    BkfChCliLetcpDelOneConn(connId);
    return;
}

STATIC void *BkfChCliLetcpMallocDataBuf(BkfChCliConnId *connId, int32_t dataBufLen)
{
    if ((connId == VOS_NULL) || (dataBufLen < 0)) {
        return VOS_NULL;
    }

    BkfChCli *ch = connId->ch;

    BkfChCliLetcpMsgHead *chMsgHead = BkfChCliLetcpMsgMalloc(ch, dataBufLen);
    return (chMsgHead != VOS_NULL) ? chMsgHead->data : VOS_NULL;
}

STATIC void BkfChCliLetcpFreeDataBuf(BkfChCliConnId *connId, void *dataBuf)
{
    if ((connId == VOS_NULL) || (dataBuf == VOS_NULL)) {
        return;
    }
    BkfChCli *ch = connId->ch;
    BkfChCliLetcpMsgHead *chMsgHead = BKF_CH_CLI_LETCP_APP_DATA_2CH_MSG_HEAD(dataBuf);

    if ((chMsgHead == VOS_NULL) || !BKF_SIGN_IS_VALID(chMsgHead->sign, BKF_CH_CLI_LETCP_MSG_SIGN)) {
        BKF_ASSERT(0);
        return;
    }
    BkfChCliLetcpMsgFree(ch, chMsgHead);
    return;
}

STATIC int32_t BkfChCliLetcpRead(BkfChCliConnId *connId, void *dataBuf, int32_t bufLen)
{
    BkfChCli *ch = VOS_NULL;

    if ((connId == VOS_NULL) || (dataBuf == VOS_NULL) || (bufLen < 0)) {
        return -1;
    }
    ch = connId->ch;

    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];

    BKF_LOG_INFO(BKF_LOG_HND, "LetcpRead serUrl(%s)/localUrl(%s), connId %#x\n",
        BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)), BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)),
        BKF_MASK_ADDR(connId));
    int32_t ret = recv(connId->connFd, dataBuf, bufLen, 0);
    if (ret == 0) { /* 上层判断断开依赖事件通知，不依赖读返回0 */
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d), ng\n", connId->connFd, ret);
    } else if (ret < 0) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u)\n", connId->connFd, ret, errno);
    }
    return ret;
}

STATIC int32_t BkfChCliLetcpSend(BkfChCliConnId *connId, void *dataBuf, int32_t dataLen)
{
    BkfChCli *ch = connId->ch;
    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];

    BKF_LOG_INFO(BKF_LOG_HND, "LetcpSend serUrl(%s)/localUrl(%s), connId %#x\n",
        BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)), BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)),
        BKF_MASK_ADDR(connId));
    int32_t ret = send(connId->connFd, dataBuf, dataLen, 0);
    if (ret == 0) { /* 上层判断断开依赖事件通知，不依赖写返回0 */
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d), ng\n", connId->connFd, ret);
        return 0;
    } else if (ret < 0) {
        BKF_LOG_INFO(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%d)\n", connId->connFd, ret, errno);
        (void)BkfChCliLetcpStartConnTmrWriteErr(connId);
        return 0;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "LetcpSend succ dataLen %d", dataLen);
    return dataLen;
}

STATIC int32_t BkfChCliLetcpWrite(BkfChCliConnId *connId, void *dataBuf, int32_t dataLen)
{
    BOOL argIsInvalid = (connId == VOS_NULL) || (dataBuf == VOS_NULL) || (dataLen < 0);
    if (argIsInvalid) {
        return -1;
    }
    BkfChCli *ch = connId->ch;
    BkfChCliLetcpMsgHead *chMsgHead = VOS_NULL;
    chMsgHead = BKF_CH_CLI_LETCP_APP_DATA_2CH_MSG_HEAD(dataBuf);
    argIsInvalid = (chMsgHead == VOS_NULL) || !BKF_SIGN_IS_VALID(chMsgHead->sign, BKF_CH_CLI_LETCP_MSG_SIGN);
    if (argIsInvalid) {
        BKF_ASSERT(0);
        return -1;
    }
    if (dataLen == 0) {
        BKF_ASSERT(0);
        return 0;
    }
    if (connId->tcpState != BKF_CLI_LETCPSTATE_ESTABLISHED) {
        BKF_LOG_INFO(BKF_LOG_HND, "ChCliLetcpWrite: fd %d, tcp not ready", connId->connFd);
        return 0;
    }
    int32_t lenRet = BkfChCliLetcpSend(connId, dataBuf, dataLen);
    if (lenRet == dataLen) {
        BkfChCliLetcpMsgFree(ch, chMsgHead);
    }
    return lenRet;
}

void BkfChCliLetcpProcConnFdEpollout(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;
    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];
    if (connId->tcpState == BKF_CLI_LETCPSTATE_ESTABLISHED) {
        return;
    }
    BkfChCliLetcpAttachConnFd(connId, EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLOUT | EPOLLET);
    BKF_LOG_INFO(BKF_LOG_HND, "ProcConnFdEpollout_ connId(%#x), Fd(%d)/serurl(%s) cliurl(%s)\n", BKF_MASK_ADDR(connId),
        connId->connFd, BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)),
        BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)));
    connId->tcpState = BKF_CLI_LETCPSTATE_ESTABLISHED;
    if (ch->argEnable.onUnblock) {
        ch->argEnable.onUnblock(ch->argEnable.cookie, connId);
    }
    return;
}

/* on msg */
STATIC void BkfChCliLetcpOnConnFdEvents(int32_t fd, uint32_t curEvents, BkfChCliConnId *connId)
{
    (void)fd;
    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];
    if ((connId == VOS_NULL) || (!BKF_SIGN_IS_VALID(connId->sign, BKF_CH_CLI_LETCP_CONN_ID_SIGN))) {
        BKF_ASSERT(0); /* mux的错误 */
        return;
    }
    BkfChCli *ch = connId->ch;
    BKF_LOG_DEBUG(BKF_LOG_HND, "fd(%d), keyUrlSer(%s) cliurl(%s)/curEvents(%#x)\n", connId->connFd,
        BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)), BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)),
        curEvents);

    if (BKF_BIT_TEST(curEvents, EPOLLERR | EPOLLHUP)) {
        BkfChCliLetcpProcConnFdSerDisc(connId);
    } else {
        if (BKF_BIT_TEST(curEvents, EPOLLOUT)) {
            BkfChCliLetcpProcConnFdEpollout(connId);
        }
        if (BKF_BIT_TEST(curEvents, EPOLLIN)) {
            BkfChCliLetcpProcConnFdSerData(connId);
        }
    }
    return;
}

STATIC uint32_t BkfChCliLetcpOnConnTmrWriteErrTO(BkfChCliConnId *connId, void *paramTmrLibUnknown)
{
    (void)paramTmrLibUnknown;
    connId->tmrIdWriteErr = VOS_NULL;
    BkfChCliLetcpTry2WriteMore(connId);
    return BKF_OK;
}

/* proc */
uint32_t BkfChCliLetcpCreateConnFd(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;
    if (connId->connFd != -1) {
        return BKF_OK;
    }
    connId->connFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (connId->connFd == -1) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), errno(%d), ng\n", connId->connFd, errno);
        return BKF_ERR;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "create newConn fd(%d)\n", connId->connFd);
    return BKF_OK;
}

void BkfChCliLetcpCloseFd(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;

    int32_t ret = close(connId->connFd);
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), errno(%d), ng\n", connId->connFd, errno);
    }
    connId->connFd = -1;
    return;
}

static inline uint32_t BkfChCliLetcpBind(BkfChCliConnId *connId, BkfUrl *urlCli)
{
    struct sockaddr_in stAddr = {0};
    stAddr.sin_family = AF_INET;
    stAddr.sin_addr.s_addr = VOS_HTONL(urlCli->ip.addrH);
    stAddr.sin_port = VOS_HTONS(urlCli->ip.port);
    BkfChCli *ch = connId->ch;
    BKF_LOG_INFO(BKF_LOG_HND, "ChCliLetcpBind:fd(%d), cli addr %x, port %u\n", connId->connFd, urlCli->ip.addrH,
        urlCli->ip.port);

    int32_t ret = bind(connId->connFd, (struct sockaddr *)&stAddr, sizeof(stAddr));
    return ret == VOS_ERROR ? BKF_ERR : BKF_OK;
}

STATIC uint32_t BkfChCliLetcpAttachConnFd(BkfChCliConnId *connId, uint32_t events)
{
    BkfChCli *ch = connId->ch;
    int32_t ret = 0;
    // 判断Fd是否附加到mux上，没有附加，执行WfwMuxAttachFd，并修改标识符；已经附加，执行WfwMuxReattachFd，并重新附加到mux上
    if (!connId->connFdAttachOk) {
        ret = WfwMuxAttachFd(ch->argInit.base->mux, connId->connFd, events,
            (F_WFW_MUX_FD_PROC)BkfChCliLetcpOnConnFdEvents, connId, VOS_NULL);
        if (ret == 0) {
            connId->connFdAttachOk = VOS_TRUE;
        }
    } else {
        ret = WfwMuxReattachFd(ch->argInit.base->mux, connId->connFd, events,
            (F_WFW_MUX_FD_PROC)BkfChCliLetcpOnConnFdEvents, connId, VOS_NULL);
    }

    if (ret < 0) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fd(%d), errno(%d), ng\n", connId->connFd, errno);
        return BKF_ERR;
    }
    return BKF_OK;
}

void BkfChCliLetcpDetachConnFd(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;

    if (connId->connFdAttachOk) {
        (void)WfwMuxDetachFd(ch->argInit.base->mux, connId->connFd);
        connId->connFdAttachOk = VOS_FALSE;
    }

    return;
}

// 设置非阻塞式
STATIC uint32_t BkfChCliLetcpSetFdNonBlock(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;
    int32_t fd = connId->connFd;

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

// 设置阻塞式
STATIC void BkfChCliLetcpSetFdBlock(int32_t fd)
{
    if (fd < 0) {
        return;
    }
    int32_t flags = fcntl(fd, F_GETFL, 0);
    (void)fcntl(fd, F_SETFL, (uint32_t)flags & ~O_NONBLOCK);
}

uint32_t BkfChCliLetcpSetFdOptKa(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;
    int32_t fd = connId->connFd;
    int32_t optval = 1;
    int32_t ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t recvBuf = BKF_CH_CLI_LETCP_MAX_BUF;
    ret = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &recvBuf, sizeof(recvBuf));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t sendBuf = BKF_CH_CLI_LETCP_MAX_BUF;
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
    int32_t keepIdle = BKF_CH_CLI_LETCP_KA_IDLE;
    ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &keepIdle, sizeof(keepIdle));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t keepIntvl = BKF_CH_CLI_LETCP_KA_INTVL;
    ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &keepIntvl, sizeof(keepIntvl));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    int32_t keepCnt = BKF_CH_CLI_LETCP_KA_CNT;
    ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &keepCnt, sizeof(keepCnt));
    if (ret == VOS_ERROR) {
        BKF_LOG_WARN(BKF_LOG_HND, "fd(%d), ret(%d)/errno(%u), ng\n", fd, ret, errno);
        return BKF_ERR;
    }
    return BKF_OK;
}

STATIC uint32_t BkfChCliLetcpConnect(BkfChCliConnId *connId)
{
    uint8_t buf[BKF_1K / 8];
    BkfChCli *ch = connId->ch;
    BkfUrl *urlSer = &(connId->keyUrlSer);
    BkfUrl *urlCli = &(ch->selfUrl);
    if (BkfUrlIsValid(urlCli)) {
        if (BkfChCliLetcpBind(connId, urlCli) != BKF_OK) {
            return BKF_ERR;
        }
    }
    struct sockaddr_in stAddr = {0};
    stAddr.sin_family = AF_INET;
    stAddr.sin_addr.s_addr = VOS_HTONL(urlSer->ip.addrH);
    stAddr.sin_port = VOS_HTONS(urlSer->ip.port);
    int32_t ret = connect(connId->connFd, (struct sockaddr *)&stAddr, sizeof(stAddr));
    if (ret < 0 && errno != EINPROGRESS) {
        BKF_LOG_WARN(BKF_LOG_HND, "Connect to server fail, urlSer(%s), fd(%d), ret(%d)/errno(%d), ng\n",
            BkfUrlGetStr(urlSer, buf, sizeof(buf)), connId->connFd, ret, errno);
        return BKF_ERR;
    }

    if (ret == 0) {
        connId->tcpState = BKF_CLI_LETCPSTATE_ESTABLISHED;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "Connect to server success, urlSer(%s), fd(%d) ret %d\n",
        BkfUrlGetStr(urlSer, buf, sizeof(buf)), connId->connFd, ret);
    return BKF_OK;
}

STATIC void BkfChCliLetcpDelConnFd(BkfChCliConnId *connId)
{
    BkfChCliLetcpDetachConnFd(connId);
    BkfChCliLetcpCloseFd(connId);
    return;
}

STATIC uint32_t BkfChCliLetcpNewConnFd(BkfChCliConnId *connId)
{
    if (BkfChCliLetcpCreateConnFd(connId) != BKF_OK) {
        return BKF_ERR;
    }

    uint32_t ret = BkfChCliLetcpSetFdOptKa(connId);
    if (ret != BKF_OK) {
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC uint32_t BkfChCliLetcpConnToSer(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;

    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];

    BKF_LOG_INFO(BKF_LOG_HND, "ConnToSer serUrl(%s)/localUrl(%s), connId %#x\n",
        BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)), BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)),
        BKF_MASK_ADDR(connId));

    if (BkfChCliLetcpNewConnFd(connId) != BKF_OK) {
        return BKF_ERR;
    }
    /* 设置非阻塞,可以删除     */
    if (BkfChCliLetcpSetFdNonBlock(connId) != BKF_OK) {
        return BKF_ERR;
    }
    if (BkfChCliLetcpAttachConnFd(connId, EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLOUT) != BKF_OK) {
        return BKF_ERR;
    }
    if (BkfChCliLetcpConnect(connId) != BKF_OK) {
        return BKF_ERR;
    }
    /* connId，清除删除标记 */
    BkfChCliLetcpClearSoftDel(connId);
    return BKF_OK;
}
/* 重新连接 */
STATIC BkfChCliConnId *BkfChCliLetcpReconnToSer(BkfChCliConnId *connId)
{
    BkfChCliLetcpDelConnFd(connId);

    if (BkfChCliLetcpConnToSer(connId) != BKF_OK) {
        BkfChCliLetcpDelOneConn(connId);
        return VOS_NULL;
    }

    return connId;
}

STATIC BkfChCliConnId *BkfChCliLetcpCreateOneConn(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlSelf)
{
    BkfChCliConnId *connId = BkfChCliLetcpNewConnId(ch, urlServer, urlSelf);
    if (connId == VOS_NULL) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "new connId failed\n");
        return VOS_NULL;
    }

    if (BkfChCliLetcpConnToSer(connId) != BKF_OK) {
        BkfChCliLetcpDelOneConn(connId);
        return VOS_NULL;
    }

    return connId;
}

STATIC void BkfChCliLetcpDelOneConn(BkfChCliConnId *connId)
{
    if (BkfChCliLetcpSoftDel(connId) == BKF_OK) {
        return;
    }

    BkfChCliLetcpDelConnFd(connId);
    BkfChCliLetcpFreeConnId(connId);
    return;
}

STATIC void BkfChCliLetcpDelAllConn(BkfChCli *ch)
{
    void *itor = VOS_NULL;
    BkfChCliConnId *connId = VOS_NULL;
    for (connId = BkfChCliLetcpGetFirstConnId(ch, &itor); connId != VOS_NULL;
         connId = BkfChCliLetcpGetNextConnId(ch, &itor)) {
        /* uninit ch, need block mod */
        BkfChCliLetcpSetFdBlock(connId->connFd);
        BkfChCliLetcpDelOneConn(connId);
    }
    return;
}

STATIC void BkfChCliLetcpProcConnFdSerDisc(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;

    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];
    BKF_LOG_INFO(BKF_LOG_HND, "ConnFdSerDisconnect connId(%#x), Fd(%d)/serurl(%s) cliurl(%s)\n", BKF_MASK_ADDR(connId),
        connId->connFd, BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)),
        BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)));

    connId->lockDel = VOS_TRUE;
    (void)BkfChCliLetcpSoftDel(connId);
    if (ch->argEnable.onDisconn) {
        ch->argEnable.onDisconn(ch->argEnable.cookie, connId);
    }
    connId->lockDel = VOS_FALSE;

    /* 如果重新建连成功，则不需要删除连接 */
    if (connId->softDel) {
        BkfChCliLetcpDelOneConn(connId);
    }
    return;
}

STATIC void BkfChCliLetcpProcConnFdSerData(BkfChCliConnId *connId)
{
    uint8_t buf[BKF_1K / 8];
    uint8_t buf2[BKF_1K / 8];

    BkfChCli *ch = connId->ch;
    BKF_LOG_INFO(BKF_LOG_HND, "ConnFdSerData connId(%#x), Fd(%d)/serurl(%s) cliurl(%s)\n", BKF_MASK_ADDR(connId),
        connId->connFd, BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)),
        BkfUrlGetStr(&connId->keyUrlCli, buf2, sizeof(buf2)));
    connId->tcpState = BKF_CLI_LETCPSTATE_ESTABLISHED;
    connId->lockDel = VOS_TRUE;
    if (ch->argEnable.onRcvDataEvent) {
        ch->argEnable.onRcvDataEvent(ch->argEnable.cookie, connId);
    }
    connId->lockDel = VOS_FALSE;
    if (connId->softDel) {
        BkfChCliLetcpDelOneConn(connId);
    }
    return;
}

STATIC void BkfChCliLetcpTry2WriteMore(BkfChCliConnId *connId)
{
    BkfChCli *ch = connId->ch;

    ch->argEnable.onUnblock(ch->argEnable.cookie, connId);
    return;
}

#endif
#ifdef __cplusplus
}
#endif
