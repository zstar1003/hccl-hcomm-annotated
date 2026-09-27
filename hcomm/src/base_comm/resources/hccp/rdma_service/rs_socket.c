/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <errno.h>
#include <netinet/tcp.h>
#include "user_log.h"
#include "config_log.h"
#include "rs_tls.h"
#include "ssl_adp.h"
#include "securec.h"
#include "rs.h"
#include "ra_rs_err.h"
#include "rs_epoll.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "dl_hal_function.h"
#include "rs_drv_socket.h"
#include "rs_socket.h"

static unsigned int gVnics[RS_VNIC_MAX] = {0};

RS_ATTRI_VISI_DEF int RsSocketInit(const unsigned int *vnicIp, unsigned int num)
{
    int ret;

    // vnic_ip max num  is RA_MAX_VNIC_NUM(16) RS_MAX_VNIC_NUM is also 16
    CHK_PRT_RETURN(num > RS_MAX_VNIC_NUM || num == 0 || vnicIp == NULL,
        hccp_err("param error, num:%u is 0 or bigger than %d, or vnicIp is NULL", num, RS_MAX_VNIC_NUM), -EINVAL);

    ret = memcpy_s(&(gVnics), sizeof(gVnics), vnicIp, sizeof(unsigned int) * num);
    CHK_PRT_RETURN(ret != 0, hccp_err("memcpy_s for vnic_ip failed ret[%d]", ret), -ESAFEFUNC);

    return 0;
}

int RsSocketNodeid2vnic(uint32_t nodeId, uint32_t *ipAddr)
{
    if (nodeId >= RS_VNIC_MAX) {
        return -1; /* it means real nic */
    }

    CHK_PRT_RETURN(ipAddr == NULL, hccp_err("ip_addr is NULL, invalid"), -EINVAL);

    *ipAddr = gVnics[nodeId];

    return RS_VNIC_FLAG;
}

STATIC uint32_t RsSocketVnic2nodeid(uint32_t ipAddr)
{
    uint32_t nodeId;

    if (ipAddr < RS_VNIC_MAX) { /* ip_addr is actually dev_id for vnic */
        return ipAddr;
    }

    for (nodeId = 0; nodeId < RS_VNIC_MAX; nodeId++) {
        if (gVnics[nodeId] == ipAddr) {
            break;
        }
    }

    if (nodeId == RS_VNIC_MAX) {
        return ipAddr;
    }

    return nodeId; /* it means virtual nic */
}

STATIC int RsServerSendWlistCheckResult(struct RsConnInfo *conn, bool flag)
{
    int ret;
    char invalid[] = "5a5a5";
    char valid[] = "a5a5a";

    if (flag == 0) {
        if ((gRsCb->sslEnable == RS_SSL_ENABLE) && (conn->ssl != NULL)) {
            ret = ssl_adp_write(conn->ssl, valid, sizeof(valid));
        } else {
            ret = RsSocketSend(conn->connfd, valid, sizeof(valid));
        }
        CHK_PRT_RETURN(ret != sizeof(valid),
            hccp_warn_socket("white list server send valid flag failed! fd[%d], ret[%d]", conn->connfd, ret), -1);
    } else {
        if ((gRsCb->sslEnable == RS_SSL_ENABLE) && (conn->ssl != NULL)) {
            ret = ssl_adp_write(conn->ssl, invalid, sizeof(invalid));
        } else {
            ret = RsSocketSend(conn->connfd, invalid, sizeof(invalid));
        }
        CHK_PRT_RETURN(ret != sizeof(invalid),
            hccp_warn_socket("white list server send invalid flag failed! fd[%d], ret[%d]", conn->connfd, ret), -1);
    }
    return 0;
}

STATIC int rs_socket_fill_wlist_by_phyID(unsigned int chipId, struct SocketWlistInfoT *whiteListNode,
    struct RsConnInfo *rsConn)
{
    unsigned int vnicIp = 0;
    int64_t deviceInfo = 0;
    char *tagTemp = NULL;
    unsigned int phyId;
    int ret;

    ret = memcpy_s(whiteListNode->tag, SOCK_CONN_TAG_SIZE, rsConn->tag, SOCK_CONN_TAG_SIZE);
    CHK_PRT_RETURN(ret, hccp_err("memcpy_s failed, ret[%d]", ret), -ESAFEFUNC);

    if (rsConn->clientIp.family == AF_INET) {
        // compare server_ip with current vnic_ip: use client_ip as remote_ip if it has bound or not vnic ip
        if (!RsSocketIsVnicIp(chipId, rsConn->serverIp.binAddr.addr.s_addr)) {
            // NIC IPv4
            whiteListNode->remoteIp.addr.s_addr = rsConn->clientIp.binAddr.addr.s_addr;
            return 0;
        }
    } else {
        // NIC IPv6
        whiteListNode->remoteIp = rsConn->clientIp.binAddr;
        return 0;
    }

    tagTemp = rsConn->tag + SOCK_CONN_TAG_SIZE;
    tagTemp[SOCK_CONN_DEV_ID_SIZE - 1] = '\0';
    RS_CHECK_POINTER_NULL_RETURN_INT(tagTemp);
    if (rsConn->clientIp.family == AF_INET) {
        // VNIC
        phyId = (unsigned int)strtol(tagTemp, NULL, 10); // Decimal(10)
        ret = DlHalGetDeviceInfo(phyId, MODULE_TYPE_SYSTEM, INFO_TYPE_VNIC_IP, &deviceInfo);
        CHK_PRT_RETURN(ret, hccp_err("dl_hal_get_device_info failed, ret(%d) tagTemp phyId(%u)", ret, phyId), ret);
        vnicIp = (unsigned int)deviceInfo;
        hccp_dbg("chip_id:%u phyId:%u vnic_ip:%u", chipId, phyId, vnicIp);
        whiteListNode->remoteIp.addr.s_addr = vnicIp;
    }
    return 0;
}

STATIC int RsServerValidAsyncInit(unsigned int chipId, struct RsConnInfo *conn,
    struct SocketWlistInfoT *whiteListExpect)
{
    int ret;

    ret = memset_s(whiteListExpect, sizeof(struct SocketWlistInfoT), 0, sizeof(struct SocketWlistInfoT));
    CHK_PRT_RETURN(ret, hccp_err("memset_s socket_wlist_info_t wlist failed, ret:%d", ret), -ESAFEFUNC);

    CHK_PRT_RETURN(conn->state != RS_CONN_STATE_TAG_SYNC,
        hccp_err("conn state is not RS_CONN_STATE_TAG_SYNC,"
                 "state[%u]. ",
            conn->state),
        -1);

    ret = rs_socket_fill_wlist_by_phyID(chipId, whiteListExpect, conn);
    CHK_PRT_RETURN(ret, hccp_err("rs_socket_fill_wlist_by_phyID failed, ret[%d]. ", ret), ret);

    return 0;
}

STATIC int RsServerValidAsync(unsigned int chipId, struct RsConnCb *connCb, struct RsConnInfo *conn)
{
    int ret;
    struct RsWhiteList *whiteListTmp = NULL;
    struct RsWhiteListInfo *whiteListNodeTmp = NULL;
    struct SocketWlistInfoT whiteListExpect;

    ret = RsServerValidAsyncInit(chipId, conn, &whiteListExpect);
    CHK_PRT_RETURN(ret, hccp_err("rs server valid async init failed, ret:%d", ret), -1);

    ret = RsFindWhiteList(connCb, &conn->serverIp, &whiteListTmp);
    if (ret) {
        ret = RsServerSendWlistCheckResult(conn, 1);
        CHK_PRT_RETURN(ret,
            hccp_err("rs server send wlist check invalid result failed, connfd[%d], ret[%d]", conn->connfd, ret), -1);
        hccp_info_socket("white list can not be found, connfd[%d], serverIp[%s], ret[%d]", conn->connfd,
            conn->serverIp.readAddr, ret);
        return -1;
    }

    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    ret = RsFindWhiteListNode(whiteListTmp, &whiteListExpect, (int)conn->clientIp.family, &whiteListNodeTmp);
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
    if (ret) {
        ret = RsServerSendWlistCheckResult(conn, 1);
        CHK_PRT_RETURN(ret,
            hccp_err("rs server send wlist check invalid result failed, connfd[%d], ret[%d]", conn->connfd, ret), -1);
        hccp_info_socket("white list node can not be found, connfd[%d], ret[%d]", conn->connfd, ret);
        return -1;
    }

    if (whiteListNodeTmp->connLimit < 1) {
        ret = RsServerSendWlistCheckResult(conn, 1);
        CHK_PRT_RETURN(ret,
            hccp_err("rs_server_send_wlist_check_result failed, connfd[%d], connLimit[%u], ret[%d]", conn->connfd,
                whiteListNodeTmp->connLimit, ret),
            -1);
        hccp_info_socket("white list node limit has less than 1, connfd[%d], ret[%d]", conn->connfd, ret);
        return -1;
    }

    ret = RsServerSendWlistCheckResult(conn, 0);
    CHK_PRT_RETURN(ret,
        hccp_warn_socket("rs server send wlist check valid result failed, connfd[%d], ret[%d]", conn->connfd, ret), -1);
    whiteListNodeTmp->connLimit--;
    return 0;
}

int RsSocketCopyConnInfo(struct RsConnInfo *connTmp, struct RsConnInfo *conn)
{
    int ret;

    conn->serverIp = connTmp->serverIp;
    conn->clientIp = connTmp->clientIp;
    conn->connfd = connTmp->connfd;
    conn->state = connTmp->state;
    conn->port = connTmp->port;
    conn->ssl = connTmp->ssl;
    ret = memcpy_s(conn->tag, SOCK_CONN_TAG_SIZE + SOCK_CONN_DEV_ID_SIZE, connTmp->tag, sizeof(connTmp->tag));
    if (ret) {
        hccp_err("rs_conn_info tag copy failed, ret[%d]", ret);
    }
    conn->isGot = false;
    return ret;
}

int RsWhiteListCheckValid(unsigned int chipId, struct RsConnCb *connCb, struct RsConnInfo *conn)
{
    int ret;

    ret = RsServerValidAsync(chipId, connCb, conn);
    if (ret) {
        RS_CLOSE_RETRY_FOR_EINTR(ret, conn->connfd);
        hccp_info_socket("rs_server_valid_async, white list doesn't exist, ret[%d]", ret);
        return -1;
    } else {
        conn->state = RS_CONN_STATE_VALID_SYNC;
    }
    return 0;
}

STATIC int RsSetFdNonblock(int connfd)
{
    int flags, ret;

    flags = fcntl(connfd, F_GETFL, 0);
    CHK_PRT_RETURN(flags < 0, hccp_err("fcntl connfd %d GETFL errno %d flags %d", connfd, errno, flags), -EFILEOPER);

    ret = fcntl(connfd, F_SETFL, (unsigned int)flags | O_NONBLOCK);
    if (ret < 0) {
        ret = -EFILEOPER;
        hccp_err("fcntl connfd %d nonblock errno %d ret %d", connfd, errno, ret);
    }

    return ret;
}

STATIC int RsSocketSetFdTimeoutUsec(int connfd, unsigned int tvUsec)
{
    struct timeval tv = {0};
    int ret = 0;

    tv.tv_usec = tvUsec;
    ret = setsockopt(connfd, SOL_SOCKET, SO_SNDTIMEO, (char *)&tv, sizeof(tv));
    CHK_PRT_RETURN(ret < 0, hccp_err("setsockopt connfd %d SO_SNDTIMEO tv_usec %u failed %d", connfd, tvUsec, ret),
        -EFILEOPER);

    ret = setsockopt(connfd, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv, sizeof(tv));
    CHK_PRT_RETURN(ret < 0, hccp_err("setsockopt connfd %d SO_RCVTIMEO tv_usec %u failed %d", connfd, tvUsec, ret),
        -EFILEOPER);

    return 0;
}

STATIC void RsEpollEventSslListenInHandle(struct rs_cb *rsCb, struct RsListenInfo *listenInfo, int connfd,
    struct RsIpAddrInfo *remoteIp)
{
    /*lint -e593*/
    int ret;
    struct RsAcceptInfo *acceptInfo = NULL;
    struct RsListHead *listHead = NULL;

    ret = RsEpollCtl(rsCb->connCb.epollfd, EPOLL_CTL_ADD, connfd, EPOLLIN | EPOLLRDHUP);
    if (ret) {
        hccp_err("epoll ctl add fd %d failed", connfd);
        goto out;
    }

    hccp_info_socket("epoll ctl add fd %d success", connfd);
    acceptInfo = calloc(1, sizeof(struct RsAcceptInfo));
    if (acceptInfo == NULL) {
        hccp_err("alloc mem for socket conn info failed!");
        goto out;
    }

    acceptInfo->sockPort = listenInfo->sockPort;
    acceptInfo->serverIpAddr = listenInfo->serverIpAddr;
    acceptInfo->clientIpAddr = *remoteIp;
    acceptInfo->connFd = connfd;
    RS_PTHREAD_MUTEX_LOCK(&rsCb->connCb.connMutex);
    listHead = &rsCb->connCb.serverAcceptList;
    RsListAddTail(&acceptInfo->list, listHead);
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->connCb.connMutex);

    return;

out:
    RS_CLOSE_RETRY_FOR_EINTR(ret, connfd);
    return;
    /*lint +e593*/
}

STATIC int RsTcpRecvTagInHandle(struct RsListenInfo *listenInfo, int connfd, struct RsConnInfo *connTmp,
    struct RsIpAddrInfo *remoteIp)
{
    int expSize = SOCK_CONN_TAG_SIZE + SOCK_CONN_DEV_ID_SIZE;
    char *recvBuff = connTmp->tag;
    struct timeval startTime, now;
    float timeCost = 0.0;
    int size = expSize;

    RsGetCurTime(&startTime);
    while (expSize > 0 && size != 0) {
        connTmp->tagSyncTimes++;
        size = recv(connfd, recvBuff, expSize, 0);
        if ((size < 0) && (errno == EINTR)) {
            connTmp->tagEintrTimes++;
            continue;
        }
        // peer socket session has been closed
        if (size == 0) {
            hccp_run_info("session has been closed, server:{%s:%u} client:%s tagSyncTimes:%u tagEintrTimes:%u",
                listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, remoteIp->readAddr, connTmp->tagSyncTimes,
                connTmp->tagEintrTimes);
            return -ESOCKCLOSED;
        }

        expSize -= size;
        recvBuff += size;
        RsGetCurTime(&now);
        HccpTimeInterval(&now, &startTime, &timeCost);
        // enlarge the timeout threshold to make sure the connection can be established successfully
        if (timeCost >= RS_RECV_TAG_MAX_TIME) {
            hccp_run_info("recv tag time out, server:{%s:%u} client:%s tagSyncTimes:%u tagEintrTimes:%u",
                listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, remoteIp->readAddr, connTmp->tagSyncTimes,
                connTmp->tagEintrTimes);
            return -ETIME;
        }

        if (timeCost <= 0) {
            RsGetCurTime(&startTime);
        }
    }

    connTmp->serverIp = listenInfo->serverIpAddr;
    connTmp->clientIp = *remoteIp;
    connTmp->connfd = connfd;
    connTmp->state = RS_CONN_STATE_TAG_SYNC;
    connTmp->port = listenInfo->sockPort;
    if (timeCost >= RS_RECV_MAX_TIME) {
        hccp_run_info("recv tag success, server:{%s:%u} client:%s timeCost:%fms tagSyncTimes:%u tagEintrTimes:%u",
            listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, remoteIp->readAddr, timeCost,
            connTmp->tagSyncTimes, connTmp->tagEintrTimes);
        return 0;
    }

    hccp_run_info("recv tag success, server:{%s:%u} client:%s timeCost:%fms tagSyncTimes:%u tagEintrTimes:%u",
        listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, remoteIp->readAddr, timeCost, connTmp->tagSyncTimes,
        connTmp->tagEintrTimes);
    return 0;
}

STATIC void RsEpollEventTcpListenInHandle(struct rs_cb *rsCb, struct RsListenInfo *listenInfo, int connfd,
    struct RsIpAddrInfo *remoteIp)
{
    struct RsListenInfo listenInfoTmp = {0};
    struct RsConnInfo connTmp = {0};
    int ret;

    (void)memcpy_s(&listenInfoTmp, sizeof(struct RsListenInfo), listenInfo, sizeof(struct RsListenInfo));
    // unlock mutex to prevent RsTcpRecvTagInHandle from blocking
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);
    ret = RsTcpRecvTagInHandle(&listenInfoTmp, connfd, &connTmp, remoteIp);
    RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
    if (ret != 0) {
        hccp_warn_socket("rs_tcp_recv_tag_in_handle unsuccessful, ret:%d", ret);
        RS_CLOSE_RETRY_FOR_EINTR(ret, connfd);
        return;
    }

    ret = RsWlistCheckConnAdd(rsCb, &connTmp);
    if (ret != 0) {
        hccp_warn_socket("rs_wlist_check_conn_add unsuccessful, ret %d", ret);
        return;
    }

    return;
}

void RsSocketSaveErrInfo(int action, int errNo, struct SocketErrInfo *errInfo)
{
    // Only record the first occurrence of err information
    if (errInfo->errNo != 0) {
        return;
    }

    if (errNo == -EAGAIN || errNo == -EINTR) {
        return;
    }

    RsGetCurTime(&errInfo->time);
    errInfo->action = action;
    errInfo->errNo = errNo;
}

STATIC int RsSocketCheckCredit(struct RsConnCb *connCb, struct RsListenInfo *listenInfo)
{
    // not using accept_credit, no need to check
    if (!listenInfo->acceptCreditFlag) {
        return 0;
    }

    // accept_credit is exhausted, check failed
    if (listenInfo->acceptCreditLimit == 0) {
        return -EINVAL;
    }

    RS_PTHREAD_MUTEX_LOCK(&listenInfo->acceptCreditMutex);
    listenInfo->acceptCreditLimit--;
    RS_PTHREAD_MUTEX_ULOCK(&listenInfo->acceptCreditMutex);

    // accept_credit is exhausted, ignore return value to delete from epoll
    if (listenInfo->acceptCreditLimit == 0) {
        (void)RsSocketListenDelFromEpoll(connCb, listenInfo);
    }

    return 0;
}

int RsEpollEventListenInHandle(struct rs_cb *rsCb, int fd)
{
    struct RsListenInfo *listenInfo2 = NULL;
    struct RsListenInfo *listenInfo = NULL;
    struct RsSocketaddrInfo remoteSAddr;
    struct RsIpAddrInfo remoteIp;
    int connfd = RS_FD_INVALID;
    int tcpNodelayFlag = 1;
    int ret, retClose;
    socklen_t ipLen;

    /* Server event: Connection accept */
    RS_LIST_GET_HEAD_ENTRY(listenInfo, listenInfo2, &rsCb->connCb.listenList, list, struct RsListenInfo);
    for (; (&listenInfo->list) != &rsCb->connCb.listenList;
         listenInfo = listenInfo2, listenInfo2 = list_entry(listenInfo2->list.next, struct RsListenInfo, list)) {
        /* connection request for Server */
        if (fd == listenInfo->listenFd) {
            ret = RsSocketCheckCredit(&rsCb->connCb, listenInfo);
            CHK_PRT_RETURN(ret != 0,
                hccp_warn_socket("[server]rs_socket_check_credit unsuccessful, serverIp:%s serverPort:%u ret:%d",
                    listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, ret),
                -EINVAL);

            remoteSAddr.family = (int)listenInfo->serverIpAddr.family;
            ipLen = (remoteSAddr.family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
            do {
                connfd = accept(fd, (struct sockaddr *)&remoteSAddr.addr, &ipLen);
            } while ((connfd < 0) && (errno == EINTR));

            // accept failed and errno is the same with the last time, avoid log flush
            ret = errno;
            if (connfd < 0 && listenInfo->lastAcceptErrno == ret) {
                hccp_warn_socket("[server]server_ip:%s server_port:%u accept() unsuccessful! errno:%d",
                    listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, ret);
                return -EINVAL;
            }
            listenInfo->lastAcceptErrno = ret;

            if (connfd < 0) {
                hccp_err("[server]server_ip:%s server_port:%u accept() failed! errno:%d",
                    listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, ret);
                goto err_accept;
            }

            hccp_info_socket("[server]server_ip:%s server_port:%u accept ok, listen_fd:%d, new fd:%d",
                listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, fd, connfd);

            remoteIp.family = (uint32_t)remoteSAddr.family;
            if (remoteIp.family == AF_INET) {
                remoteIp.binAddr.addr = remoteSAddr.addr.sAddr.sin_addr;
            } else {
                remoteIp.binAddr.addr6 = remoteSAddr.addr.sAddr6.sin6_addr;
            }

            ret = RsInetNtop(remoteIp.family, &remoteIp.binAddr, remoteIp.readAddr, sizeof(remoteIp.readAddr));
            if (ret) {
                hccp_err("[server]convert(ntop) ip failed, remoteIp.family:%d, remoteIp:%d, ret:%d, serverIp:%s "
                         "serverPort:%u",
                    remoteIp.family, remoteIp.binAddr.addr.s_addr, ret, listenInfo->serverIpAddr.readAddr,
                    listenInfo->sockPort);
                goto err_event_listen;
            }

            if (rsCb->sslEnable == RS_SSL_ENABLE) {
                ret = RsSetFdNonblock(connfd);
                if (ret) {
                    hccp_err("[server]fcntl connfd %d nonblock failed %d, serverIp:%s serverPort:%u", connfd, ret,
                        listenInfo->serverIpAddr.readAddr, listenInfo->sockPort);
                    goto err_event_listen;
                }
            }

            /* set tcp socket tos RS_TCP_DSCP_0 */
            int tosLocal = (RS_TCP_DSCP_0 & RS_DSCP_MASK) << RS_DSCP_OFF;
            ret = setsockopt(connfd, IPPROTO_IP, IP_TOS, (void *)&tosLocal, sizeof(tosLocal));
            if (ret) {
                hccp_err("[server]setsockopt(IP_TOS) failed, ret:%d, errno:%d, serverIp:%s serverPort:%u", ret, errno,
                    listenInfo->serverIpAddr.readAddr, listenInfo->sockPort);
                goto err_socket_option;
            }

            ret = setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, (void *)&tcpNodelayFlag, sizeof(int));
            if (ret < 0) {
                hccp_err("[server]setsockopt(TCP_NODELAY) failed, ret:%d, errno:%d, serverIp:%s serverPort:%u", ret,
                    errno, listenInfo->serverIpAddr.readAddr, listenInfo->sockPort);
                goto err_socket_option;
            }

            if (rsCb->sslEnable == RS_SSL_ENABLE) {
                RsEpollEventSslListenInHandle(rsCb, listenInfo, connfd, &remoteIp);
            } else {
                RsEpollEventTcpListenInHandle(rsCb, listenInfo, connfd, &remoteIp);
            }
            return 0;
        }
    }

    return -ENODEV;

err_socket_option:
    ret = -errno;
err_event_listen:
    RS_CLOSE_RETRY_FOR_EINTR(retClose, connfd);
err_accept:
    RsSocketSaveErrInfo((int)listenInfo->state, ret, &listenInfo->errInfo);
    return -ESYSFUNC;
}

STATIC int RsSocketListenBindListen(int listenFd, struct RsConnCb *connCb, struct SocketListenInfo *conn,
    struct RsListenInfo *listenInfo, uint32_t serverPort)
{
    int isReuseAddr = 1;
    int ret, errNo;

    ret = setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &isReuseAddr, sizeof(isReuseAddr));
    if (ret) {
        errNo = errno;
        hccp_err("set socket op failed! IP:%s, port:%u, sock:%d, ret:0x%x, error:%d", listenInfo->serverIpAddr.readAddr,
            serverPort, listenFd, ret, errNo);
        conn->phase = LISTEN_BIND_ERR;
        return -ESYSFUNC;
    }

    listenInfo->state = RS_CONN_STATE_INIT;

    hccp_info_socket("listen state:%d, then bind for (IP %s : port %u)", listenInfo->state,
        listenInfo->serverIpAddr.readAddr, serverPort);

    hccp_run_info("socket bind: family %d, addr %s, port %u", conn->family, listenInfo->serverIpAddr.readAddr,
        serverPort);
    if (conn->family == AF_INET) {
        struct sockaddr_in addr = {0};
        addr.sin_family = conn->family;
        addr.sin_port = htons(serverPort);
        addr.sin_addr.s_addr = listenInfo->serverIpAddr.binAddr.addr.s_addr;
        hccp_info_socket("socket bind: family %d, port %d, addr 0x%08x", addr.sin_family, addr.sin_port,
            addr.sin_addr.s_addr);
        ret = bind(listenFd, &addr, sizeof(addr));
    } else {
        struct sockaddr_in6 addr = {0};
        addr.sin6_family = conn->family;
        addr.sin6_port = htons(serverPort);
        addr.sin6_addr = listenInfo->serverIpAddr.binAddr.addr6;
        addr.sin6_scope_id = (uint32_t)connCb->scopeId;
        hccp_info_socket("socket bind: family %d, port %d, scopeId %d", addr.sin6_family, addr.sin6_port,
            addr.sin6_scope_id);
        for (unsigned long i = 0; i < sizeof(addr.sin6_addr.s6_addr); i++) {
            hccp_info_socket("socket bind: addr[%lu] 0x%02x", i, addr.sin6_addr.s6_addr[i]);
        }
        ret = bind(listenFd, &addr, sizeof(addr));
    }

    if (ret) {
        errNo = errno;
        if (errNo == EADDRINUSE) {
            hccp_run_warn("bind unsuccessful! family:%d, IP:%s, port:%u, sock:%d, ret:0x%x, error:%d, Possible Cause: "
                          "the IP address and port have been bound already",
                conn->family, listenInfo->serverIpAddr.readAddr, serverPort, listenFd, ret, errNo);
        } else {
            hccp_err("bind failed! family:%d, IP:%s, port:%u, sock:%d, ret:0x%x, error:%d", conn->family,
                listenInfo->serverIpAddr.readAddr, serverPort, listenFd, ret, errNo);
        }
        conn->phase = LISTEN_BIND_ERR;
        return errNo;
    }

    listenInfo->state = RS_CONN_STATE_BIND;

    hccp_info_socket("IP %s : port %u begin listen, fd:%d !", listenInfo->serverIpAddr.readAddr, serverPort, listenFd);
    ret = listen(listenFd, RS_SOCK_LISTEN_PARALLEL_NUM);
    if (ret) {
        errNo = errno;
        if (errNo == EADDRINUSE) {
            hccp_run_warn("listen unsuccessful! IP:%s, port:%u, sock:%d, ret:0x%x, errno:%d",
                listenInfo->serverIpAddr.readAddr, serverPort, listenFd, ret, errNo);
        } else {
            hccp_err("listen failed! IP:%s, port:%u, sock:%d, ret:0x%x, errno:%d", listenInfo->serverIpAddr.readAddr,
                serverPort, listenFd, ret, errNo);
        }
        conn->phase = LISTEN_BEGIN_ERR;
        return errNo;
    }

    return 0;
}

static int RsSocketInitListen(struct SocketListenInfo *conn, uint32_t i, struct RsConnCb **connCb, uint32_t serverPort,
    struct RsListenInfo **listenInfo)
{
    int ret;
    unsigned int chipId;

    CHK_PRT_RETURN(((conn[i].family != AF_INET) && (conn[i].family != AF_INET6)) || conn[i].phyId >= RS_MAX_DEV_NUM,
        hccp_err("family[%d] invalid, or phyId[%u] invalid, i:%u", conn[i].family, conn[i].phyId, i), -EINVAL);

    if (conn[i].family == AF_INET) {
        uint32_t *localIp = NULL;
        localIp = &(conn[i].localIp.addr.s_addr);
        ret = RsSocketNodeid2vnic(*localIp, localIp);
        hccp_info_socket("listen [%u] IP 0x%llx, ret_vnic %d", i, *localIp, ret);
    }

    ret = rsGetLocalDevIDByHostDevID(conn[i].phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId invalid, ret %d", ret), ret);

    ret = RsDev2conncb(chipId, connCb);
    CHK_PRT_RETURN(ret, hccp_err("get conncb from dev failed, ret:%d", ret), ret);

    struct RsIpAddrInfo ipInfo = {0};
    ret = RsConvertIpAddr(conn[i].family, &conn[i].localIp, &ipInfo);
    CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

    struct RsListenInfo *tmpListenInfo;
    ret = RsFindListenNode(*connCb, &ipInfo, serverPort, &tmpListenInfo);
    if (ret == 0) {
        int counter = __sync_fetch_and_add(&(tmpListenInfo->counter), 1);
        if (counter > 0) {
            hccp_info_socket("find listen node successful, counter:%d, IP:%s, port:%u", counter, ipInfo.readAddr,
                serverPort);
            return -EEXIST;
        }
    }

    ret = RsListenNodeAlloc(*connCb, &ipInfo, serverPort, listenInfo);
    // listen node found, degrade log level make it consistent with inner call
    if (ret == -EEXIST) {
        hccp_info_socket("alloc listen info node unsuccessful, ret:%d, IP:%s, port:%u", ret, ipInfo.readAddr,
            serverPort);
    } else if (ret != 0) {
        hccp_err("alloc listen info node failed, ret:%d, IP:%s, port:%u", ret, ipInfo.readAddr, serverPort);
    }
    if (ret != 0) {
        conn[i].err = ENOMEM;
        return ret;
    }

    return 0;
}

static void RsSocketSetConnListenInfo(struct RsListenInfo *listenInfo, int listenFd, uint32_t serverPort,
    struct SocketListenInfo *conn)
{
    listenInfo->listenFd = listenFd;
    listenInfo->sockPort = serverPort;
    listenInfo->state = RS_CONN_STATE_LISTENING;

    if (conn->family == AF_INET) {
        conn->localIp.addr.s_addr = RsSocketVnic2nodeid(conn->localIp.addr.s_addr);
    }
    conn->err = 0;
    conn->port = serverPort;
    conn->phase = LISTEN_OK;
}

static void RsSocketHandleListenNodeErr(uint32_t i, struct RsConnCb *connCb, struct SocketListenInfo conn[],
    uint32_t serverPort)
{
    uint32_t j;
    int ret;
    struct RsListenInfo *listenInfo = NULL;

    for (j = 0; j < i; j++) {
        struct RsIpAddrInfo ipInfo = {0};
        ret = RsConvertIpAddr(conn[j].family, &conn[j].localIp, &ipInfo);
        if (ret) {
            hccp_err("convert(ntop) ip failed");
            continue;
        }
        ret = RsFindListenNode(connCb, &ipInfo, serverPort, &listenInfo);
        if (ret) {
            hccp_dbg("not find listen node, ret %d", ret);
        } else {
            ret = RsEpollCtl(connCb->epollfd, EPOLL_CTL_DEL, listenInfo->listenFd, EPOLLIN);
            if (ret) {
                hccp_err("delete from epoll failed, ret:%d, epollfd:%d, listenFd:%d", ret, connCb->epollfd,
                    listenInfo->listenFd);
            }
            RS_CLOSE_RETRY_FOR_EINTR(ret, listenInfo->listenFd);
            RsListenNodeFree(connCb, listenInfo);
        }
    }
}

RS_ATTRI_VISI_DEF int RsSocketListenStart(struct SocketListenInfo conn[], uint32_t num)
{
    struct RsListenInfo *listenInfo = NULL;
    union RsSocketaddr serverAddr = {0};
    struct RsConnCb *connCb = NULL;
    socklen_t serverAddrLen = 0;
    unsigned int serverPort = 0;
    int listenFd = 0;
    int scopeId = 0;
    int errNo = 0;
    int ret, flag;
    uint32_t i;

    RS_SOCKET_PARA_CHECK(num, conn);
    if (conn[0].family == AF_INET6) {
        scopeId = RsGetIpv6ScopeId(conn[0].localIp.addr6);
        CHK_PRT_RETURN(scopeId < 0, hccp_err("scope_id[%d] is invalid", scopeId), -EINVAL);
    }

    for (i = 0; i < num; i++) {
        serverPort = conn[i].port;
        ret = RsSocketInitListen(conn, i, &connCb, serverPort, &listenInfo);
        if (ret == -EEXIST) {
            continue;
        }
        if (ret) {
            flag = -ENOMEM;
            hccp_err("listen init failed, ret:%d", ret);
            goto listen_node_err_handle;
        }

        /* socket */
        listenFd = socket(conn[i].family, SOCK_STREAM, 0);
        if (listenFd < 0) {
            errNo = errno;
            hccp_err("create socket for (IP %s : port %u) failed, family %d, errno %d",
                listenInfo->serverIpAddr.readAddr, serverPort, conn[i].family, errNo);
            conn[i].phase = LISTEN_CREATE_FD_ERR;
            goto listen_err_handle;
        }

        /* bind and listen */
        connCb->scopeId = scopeId;
        ret = RsSocketListenBindListen(listenFd, connCb, conn + i, listenInfo, serverPort);
        errNo = ret;
        if (ret == EADDRINUSE) {
            hccp_run_warn("bind and listen unsuccessful, errNo:%d, listenFd:%d, state:%u, IP(%s) serverPort:%u", errNo,
                listenFd, listenInfo->state, listenInfo->serverIpAddr.readAddr, serverPort);
            goto bind_err_handle;
        } else if (ret != 0) {
            hccp_err("bind and listen failed, errNo:%d, listenFd:%d, listen state:%u, IP(%s) serverPort:%u", errNo,
                listenFd, listenInfo->state, listenInfo->serverIpAddr.readAddr, serverPort);
            goto bind_err_handle;
        }

        ret = RsEpollCtl(connCb->epollfd, EPOLL_CTL_ADD, listenFd, EPOLLIN);
        if (ret) {
            errNo = ret;
            hccp_err("RsEpollCtl for epollfd[%d] listen_fd[%d]failed, errno:%d", connCb->epollfd, listenFd, errNo);
            goto bind_err_handle;
        }

        serverAddrLen = (conn->family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
        getsockname(listenFd, (struct sockaddr *)&serverAddr, &serverAddrLen);
        serverPort = (conn->family == AF_INET) ? ntohs(serverAddr.sAddr.sin_port) : ntohs(serverAddr.sAddr6.sin6_port);
        RsSocketSetConnListenInfo(listenInfo, listenFd, serverPort, &conn[i]);
    }

    return 0;

bind_err_handle:
    RS_CLOSE_RETRY_FOR_EINTR(ret, listenFd);
listen_err_handle:
    RsListenNodeFree(connCb, listenInfo);
    conn[i].err = (unsigned int)errNo;
    flag = -errNo;
listen_node_err_handle:
    RsSocketHandleListenNodeErr(i, connCb, conn, serverPort);
    return flag;
}

RS_ATTRI_VISI_DEF int RsSocketAcceptCreditAdd(struct SocketListenInfo conn[], uint32_t num, unsigned int creditLimit)
{
    struct RsListenInfo *listenInfo = NULL;
    struct RsIpAddrInfo ipInfo = {0};
    struct RsConnCb *connCb = NULL;
    unsigned int tmpCreditLimit;
    int ret = 0;
    uint32_t i;

    RS_SOCKET_PARA_CHECK(num, conn);
    for (i = 0; i < num; i++) {
        ret = RsConvertIpAddr(conn[i].family, &conn[i].localIp, &ipInfo);
        CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, i:%d, ret:%d", i, ret), ret);

        RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
        connCb = &gRsCb->connCb;
        ret = RsFindListenNode(connCb, &ipInfo, conn[i].port, &listenInfo);
        if (ret != 0) {
            hccp_err("rs_find_listen_node failed, i:%u, IP:%s serverPort:%u, ret:%d", i, ipInfo.readAddr, conn[i].port,
                ret);
            RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
            return ret;
        }
        RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);

        // prevent accept_credit_limit from overflow
        tmpCreditLimit = listenInfo->acceptCreditLimit + creditLimit;
        if (tmpCreditLimit < creditLimit) {
            hccp_err("credit_limit overflow, IP:%s serverPort:%u tmpCreditLimit:%u, creditLimit:%u", ipInfo.readAddr,
                conn[i].port, tmpCreditLimit, creditLimit);
            return -EINVAL;
        }
        RS_PTHREAD_MUTEX_LOCK(&listenInfo->acceptCreditMutex);
        listenInfo->acceptCreditLimit += creditLimit;
        RS_PTHREAD_MUTEX_ULOCK(&listenInfo->acceptCreditMutex);
        RsSocketListenAddToEpoll(connCb, listenInfo);
        listenInfo->acceptCreditFlag = true;
    }

    return ret;
}

RS_ATTRI_VISI_DEF int RsSocketListenStop(struct SocketListenInfo conn[], uint32_t num)
{
    struct RsListenInfo *listenInfo = NULL;
    struct RsConnCb *connCb = NULL;
    unsigned int chipId;
    uint32_t i;
    int ret;

    RS_SOCKET_PARA_CHECK(num, conn);
    for (i = 0; i < num; i++) {
        CHK_PRT_RETURN(((conn[i].family != AF_INET) && (conn[i].family != AF_INET6)) || conn[i].phyId >= RS_MAX_DEV_NUM,
            hccp_err("family[%d] invalid, or phyId[%u] invalid, i:%u", conn[i].family, conn[i].phyId, i), -EINVAL);

        if (conn[i].family == AF_INET) {
            uint32_t *localIp = NULL;
            localIp = &(conn[i].localIp.addr.s_addr);
            ret = RsSocketNodeid2vnic(*localIp, localIp);
            hccp_info_socket("listen [%d] IP 0x%llx, ret_vnic %d", i, *localIp, ret);
        }
        ret = rsGetLocalDevIDByHostDevID(conn[i].phyId, &chipId);
        CHK_PRT_RETURN(ret, hccp_err("phyId invalid, ret %d", ret), ret);
        ret = RsDev2conncb(chipId, &connCb);
        // degrade log level, make it consistent with inner call
        CHK_PRT_RETURN(ret != 0, hccp_warn_socket("get conncb from dev unsuccessful(%d)!", ret), -ENODEV);

        struct RsIpAddrInfo ipInfo = {0};
        ret = RsConvertIpAddr(conn[i].family, &conn[i].localIp, &ipInfo);
        CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

        ret = RsFindListenNode(connCb, &ipInfo, conn[i].port, &listenInfo);
        if (ret == 0 && __sync_fetch_and_sub(&(listenInfo->counter), 1) > 1) {
            continue;
        }
        // listen node not found, degrade log level due to this is non-fatal error
        if (ret != 0) {
            hccp_warn_socket("get listen info unsuccessful(%d), IP(%s)!", ret, ipInfo.readAddr);
            conn[i].err = ENODEV;
            continue;
        }

        ret = RsSocketListenDelFromEpoll(connCb, listenInfo);
        CHK_PRT_RETURN(ret,
            hccp_err("delete from epoll failed, ret:%d, epollfd:%d, listenFd:%d", ret, connCb->epollfd,
                listenInfo->listenFd),
            ret);

        /* close socket */
        RS_CLOSE_RETRY_FOR_EINTR(ret, listenInfo->listenFd);
        hccp_info_socket("IP(%s) close listen fd:%d !", ipInfo.readAddr, listenInfo->listenFd);

        listenInfo->listenFd = RS_FD_INVALID;
        listenInfo->state = RS_CONN_STATE_RESET;

        RsListenNodeFree(connCb, listenInfo);
    }

    return 0;
}

STATIC int RsAllocClientConnNode(struct RsConnCb *connCb, enum RsConnRole role, struct RsConnInfo **conn,
    struct SocketConnectInfo *socketConn, struct RsIpAddrInfo *clientIp, struct RsIpAddrInfo *serverIp, int serverPort)
{
    struct RsListHead *listHead = NULL;
    struct RsConnInfo *connInfo;
    int ret;

    connInfo = calloc(1, sizeof(struct RsConnInfo));
    CHK_PRT_RETURN(connInfo == NULL, hccp_err("alloc mem for socket conn info failed!"), -ENOMEM);

    connInfo->port = serverPort;
    connInfo->connfd = RS_FD_INVALID;
    connInfo->state = RS_CONN_STATE_RESET;
    connInfo->serverIp = *serverIp;
    connInfo->clientIp = *clientIp;
    connInfo->scopeId = connCb->scopeId;

    ret = strcpy_s(connInfo->tag, SOCK_CONN_TAG_SIZE, socketConn->tag);
    if (ret) {
        hccp_err("strcpy_s err, ret:%d, size of dest:%u, size of src:%u", ret, sizeof(connInfo->tag),
            sizeof(socketConn->tag));
        goto out;
    }
    ret = sprintf_s(connInfo->tag + SOCK_CONN_TAG_SIZE, SOCK_CONN_DEV_ID_SIZE, "%u", socketConn->phyId);
    if (ret < 0) {
        hccp_err("sprintf_s err, ret:%d, phyId:%u", ret, socketConn->phyId);
        goto out;
    }

    RsGetCurTime(&connInfo->startTime);

    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    listHead = (role == RS_CONN_ROLE_SERVER) ? (&connCb->serverConnList) : (&connCb->clientConnList);
    RsListAddTail(&connInfo->list, listHead);
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);

    *conn = connInfo;

    return 0;

out:
    free(connInfo);
    connInfo = NULL;
    return -ESAFEFUNC;
}

STATIC void RsSocketClientValidSync(struct RsConnInfo *conn)
{
    char isvalid[RS_WLIST_VALID_FLAG_SIZE] = {0};
    int ret, retClose;

    do {
        ret = RsSocketRecv(conn->connfd, isvalid, RS_WLIST_VALID_FLAG_SIZE);
        if (ret == RS_WLIST_VALID_FLAG_SIZE && (strncmp(isvalid, "a5a5a", strlen("a5a5a")) == 0)) {
            hccp_info_socket("[client]client is valid, ret:%d, clientIp:%s serverIp:%s serverPort:%u", ret,
                conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port);
            conn->state = RS_CONN_STATE_VALID_SYNC;
            return;
        } else if (ret == RS_WLIST_VALID_FLAG_SIZE && (strncmp(isvalid, "5a5a5", strlen("5a5a5")) == 0)) {
            hccp_info_socket("[client]client is invalid, errNo:%d, clientIp:%s serverIp:%s serverPort:%u", errno,
                conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port);
            goto out;
        } else if (ret == -EAGAIN) {
            return;
        }
    } while ((ret < 0) && (errno == EINTR));

    // ret is -EFILEOPER or recv unexpected data. state machine will connect again
    hccp_run_warn("[client]recv isvalid unsuccessful, ret:%d errNo:%d, clientIp:%s serverIp:%s serverPort:%u fd:%d."
                  " retry connect",
        ret, errno, conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->connfd);
out:
    if (gRsCb->sslEnable == RS_SSL_ENABLE) {
        ssl_adp_shutdown(conn->ssl);
        ssl_adp_free(conn->ssl);
        conn->ssl = NULL;
    }
    RS_CLOSE_RETRY_FOR_EINTR(retClose, conn->connfd);
    conn->connfd = RS_FD_INVALID;
    conn->state = RS_CONN_STATE_RESET;
    conn->tagSyncTimes = 0;
    return;
}

STATIC void RsSocketTagSync(struct RsConnInfo *conn)
{
    int ret;

    /* sync tag to server */
    conn->tagSyncTimes++;
    ret = RsDrvSocketSend(conn->connfd, conn->tag, SOCK_CONN_TAG_SIZE + SOCK_CONN_DEV_ID_SIZE, 0);
    if (ret == SOCK_CONN_TAG_SIZE + SOCK_CONN_DEV_ID_SIZE) {
        conn->state = RS_CONN_STATE_TAG_SYNC;
        hccp_info_socket(
            "[client]send tag success! ret:%d, tagSyncTimes:%u, clientIp:%s serverIp:%s serverPort:%u tag:%s", ret,
            conn->tagSyncTimes, conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag);
    } else if (ret == -EAGAIN) {
        conn->state = RS_CONN_STATE_TIMEOUT;
        hccp_info_socket("[client]send tag incomplete! ret:%d, tagSyncTimes:%u, clientIp:%s serverIp:%s serverPort:%u "
                         "tag:%s",
            ret, conn->tagSyncTimes, conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag);
    } else {
        hccp_run_info("[client]send tag unsuccessful, ret:%d, tagSyncTimes:%u, retry connect, clientIp:%s "
                      "serverIp:%s serverPort:%u tag:%s",
            ret, conn->tagSyncTimes, conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag);

        if (gRsCb->sslEnable == RS_SSL_ENABLE) {
            ssl_adp_shutdown(conn->ssl);
            ssl_adp_free(conn->ssl);
            conn->ssl = NULL;
        }
        RS_CLOSE_RETRY_FOR_EINTR(ret, conn->connfd);
        conn->connfd = RS_FD_INVALID;
        conn->state = RS_CONN_STATE_RESET;
        conn->tagSyncTimes = 0;
    }

    return;
}

/* ssl will connect again and again, HCCL get socket timeout after period time */
STATIC int RsSocketSslConnect(struct RsConnInfo *conn, struct rs_cb *rscb)
{
    int ret, err;

    ret = ssl_adp_do_handshake(conn->ssl);
    if (ret != 1) {
        err = ssl_adp_get_error(conn->ssl, ret);
        if (err == SSL_ERROR_WANT_WRITE) {
            hccp_dbg("ssl fd %d return want write", conn->connfd);
        } else if (err == SSL_ERROR_WANT_READ) {
            hccp_dbg("ssl fd %d return want read", conn->connfd);
        } else {
            rs_ssl_err_string(conn->connfd, err);
        }

        return -EAGAIN;
    }
    ret = rs_tls_peer_cert_verify(conn->ssl, rscb);
    CHK_PRT_RETURN(ret, hccp_err("verify peer cert failed ret %d", ret), ret);

    return 0;
}

STATIC int RsSocketStateSslFdBind(struct RsConnInfo *conn, uint32_t sslEnable, struct rs_cb *rscb)
{
    int ret;

    if (sslEnable == RS_SSL_ENABLE) {
        ret = RsSocketSslConnect(conn, rscb);
        if (ret) {
            return ret;
        }
        conn->state = RS_CONN_STATE_SSL_CONNECTED;
    }

    RsConnCostTime(conn);
    RsSocketTagSync(conn);
    return 0;
}

STATIC int RsSocketStateConnected(struct RsConnInfo *conn, uint32_t sslEnable, struct rs_cb *rscb)
{
    int ret;

    if (sslEnable == RS_SSL_ENABLE) {
        ret = RsDrvSslBindFd(conn, conn->connfd);
        if (ret != 0) {
            RsSocketSaveErrInfo(RS_CONN_STATE_CONNECTED, ret, &conn->errInfo);
            hccp_err("[client]ssl bind failed, connfd:%d, ret:%d, clientIp:%s serverIp:%s serverPort:%u tag:%s",
                conn->connfd, ret, conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag);
            return ret;
        }
        conn->state = RS_CONN_STATE_SSL_BIND_FD;
    }

    return RsSocketStateSslFdBind(conn, sslEnable, rscb);
}

STATIC int RsSocketStateInit(unsigned int chipId, struct RsConnInfo *conn, uint32_t sslEnable, struct rs_cb *rscb)
{
    (void)chipId;
    int ret;

    conn->tag[SOCK_CONN_TAG_SIZE + SOCK_CONN_DEV_ID_SIZE - 1] = '\0';

    ret = RsDrvConnect(conn->connfd, &conn->serverIp, &conn->clientIp, conn->port);
    if (ret != 0) {
        RsSocketSaveErrInfo(RS_CONN_STATE_INIT, ret, &conn->errInfo);
        hccp_warn_socket(
            "[client]rs_socket_state_init conn unsuccessful! client_ip:%s server_ip:%s server_port:%u tag:%s, "
            "fd:%d, ret:%d",
            conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag, conn->connfd, ret);
        return ret;
    }

    // should set back tcp socket send/recv timeout to OS default when ssl is disabled
    if (sslEnable == RS_SSL_DISABLE) {
        ret = RsSocketSetFdTimeoutUsec(conn->connfd, 0);
        if (ret != 0) {
            hccp_warn_socket("[client]rs_socket_set_fd_timeout_usec conn unsuccessful!, clientIp:%s serverIp:%s "
                             "serverPort:%u tag:%s, fd:%d, ret:%d",
                conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag, conn->connfd, ret);
        }
    }

    conn->state = RS_CONN_STATE_CONNECTED;
    /*
     * ssl will connect again and again, HCCL get socket timeout after period time,
     * so there is no log info to prevent over log
     */
    ret = RsSocketStateConnected(conn, sslEnable, rscb);
    if (ret) {
        return ret;
    }

    return 0;
}

STATIC int RsConnectBindClient(int fd, struct RsConnInfo *conn)
{
    int errNo;
    int ret;

    if (conn->clientIp.family == AF_INET) {
        struct sockaddr_in clientAddr = {0};
        clientAddr.sin_family = conn->clientIp.family;
        clientAddr.sin_addr = conn->clientIp.binAddr.addr;

        hccp_dbg("socket bind: family %d, port %d, addr 0x%08x", clientAddr.sin_family, clientAddr.sin_port,
            clientAddr.sin_addr.s_addr);
        ret = bind(fd, &clientAddr, sizeof(clientAddr));
    } else {
        struct sockaddr_in6 clientAddr = {0};
        clientAddr.sin6_family = conn->clientIp.family;
        clientAddr.sin6_addr = conn->clientIp.binAddr.addr6;
        clientAddr.sin6_scope_id = (uint32_t)conn->scopeId;

        hccp_dbg("socket bind: family %d, port %d, scopeId %d", clientAddr.sin6_family, clientAddr.sin6_port,
            clientAddr.sin6_scope_id);
        for (unsigned long i = 0; i < sizeof(struct in6_addr); i++) {
            hccp_dbg("socket bind: addr[%lu] 0x%02x", i, clientAddr.sin6_addr.s6_addr[i]);
        }

        ret = bind(fd, &clientAddr, sizeof(clientAddr));
    }
    if (ret) {
        errNo = errno;
        hccp_err("client bind failed! IP:%s, sock:%d, ret:%d, error:%d", conn->clientIp.readAddr, fd, ret, errNo);
        return -errNo;
    }
    union RsSocketaddr clientAddr = {0};
    socklen_t clientAddrLen = (conn->clientIp.family == AF_INET) ? sizeof(struct sockaddr_in)
                                                                 : sizeof(struct sockaddr_in6);
    getsockname(fd, (struct sockaddr *)&clientAddr, &clientAddrLen);
    uint16_t clientPort = (conn->clientIp.family == AF_INET) ? ntohs(clientAddr.sAddr.sin_port)
                                                             : ntohs(clientAddr.sAddr6.sin6_port);
    if ((clientPort < 60000) || (clientPort > 60015)) { // HCCL默认监听60000-60015端口,如client使用该端口，记录EVENT日志
        hccp_run_info("client bind success. client family %d addr %s:%u, fd:%d", conn->clientIp.family,
            conn->clientIp.readAddr, clientPort, fd);
    } else {
        hccp_run_info("client bind success. client family %d addr %s:%u, fd:%d", conn->clientIp.family,
            conn->clientIp.readAddr, clientPort, fd);
    }
    return 0;
}

STATIC int RsSocketBindClient(unsigned int chipId, int connFd, struct RsConnInfo *conn, int hccpMode)
{
    bool bindIp = true;

    if (conn->clientIp.family == AF_INET && hccpMode == NETWORK_OFFLINE) {
        // compare client_ip with current vnic_ip for compatibility issues, 910A & 910B no need to bind vnic ip
        bindIp = RsSocketIsVnicIp(chipId, conn->clientIp.binAddr.addr.s_addr) ? false : true;
    }

    // chip force to bind: 310P & 910_93
    if (!bindIp) {
        RsSocketGetBindByChip(chipId, &bindIp);
    }

    // no need to bind ip
    if (!bindIp) {
        return 0;
    }

    return RsConnectBindClient(connFd, conn);
}

STATIC int RsSocketStateReset(unsigned int chipId, struct RsConnInfo *conn, uint32_t sslEnable, struct rs_cb *rscb)
{
#define RS_SOCKET_CONNECT_TIMEOUT_USECS 100000
    int connFd, retClose, hccpMode;
    int tcpNodelayFlag = 1;
    int ret = 0;

    hccpMode = RsGetHccpMode(chipId);

    connFd = socket(conn->clientIp.family, SOCK_STREAM, 0);
    if (connFd < 0) {
        ret = -errno;
        hccp_err("[client]create socket failed, errno:%d", ret);
        goto err_socket_create;
    }

    ret = RsSocketBindClient(chipId, connFd, conn, hccpMode);
    if (ret != 0) {
        hccp_err("[client]rs_socket_bind_client failed, ret:%d", ret);
        goto err_connect_reset;
    }

    if (sslEnable == RS_SSL_ENABLE) {
        ret = RsSetFdNonblock(connFd);
        if (ret) {
            goto err_connect_reset;
        }
    }

    /* set tcp socket tos RS_TCP_DSCP_0 */
    int tosLocal = (RS_TCP_DSCP_0 & RS_DSCP_MASK) << RS_DSCP_OFF;
    ret = setsockopt(connFd, IPPROTO_IP, IP_TOS, (void *)&tosLocal, sizeof(tosLocal));
    if (ret) {
        hccp_err("[client]setsockopt(IP_TOS) failed, connFd:%d, ret:%d, errno:%d", connFd, ret, errno);
        goto err_socket_option;
    }

    ret = setsockopt(connFd, IPPROTO_TCP, TCP_NODELAY, (void *)&tcpNodelayFlag, sizeof(int));
    if (ret < 0) {
        hccp_err("[client]setsockopt(TCP_NODELAY) failed, connFd:%d, ret:%d, errno:%d", connFd, ret, errno);
        goto err_socket_option;
    }

    // should set tcp socket send/recv timeout when ssl is disabled
    if (sslEnable == RS_SSL_DISABLE) {
        ret = RsSocketSetFdTimeoutUsec(connFd, RS_SOCKET_CONNECT_TIMEOUT_USECS);
        if (ret != 0) {
            goto err_connect_reset;
        }
    }

    conn->connfd = connFd;
    conn->state = RS_CONN_STATE_INIT;
    /*
     * ssl will connect again and again, HCCL get socket timeout after period time,
     * so there is no log info to prevent over log
     */
    ret = RsSocketStateInit(chipId, conn, sslEnable, rscb);
    if (ret) {
        return ret;
    }

    return 0;

err_socket_option:
    ret = -errno;
err_connect_reset:
    RS_CLOSE_RETRY_FOR_EINTR(retClose, connFd);
err_socket_create:
    RsSocketSaveErrInfo(RS_CONN_STATE_RESET, ret, &conn->errInfo);
    return -ESYSFUNC;
}

int RsSocketConnectAsync(struct RsConnInfo *conn, struct rs_cb *rscb)
{
    uint32_t sslEnable = rscb->sslEnable;
    unsigned int chipId = rscb->chipId;
    int ret = 0;

    RS_CHECK_POINTER_NULL_WITH_RET(conn);
    switch (conn->state) {
        case RS_CONN_STATE_RESET:
            /* create socket for client */
            ret = RsSocketStateReset(chipId, conn, sslEnable, rscb);
            break;

        case RS_CONN_STATE_INIT:
            ret = RsSocketStateInit(chipId, conn, sslEnable, rscb);
            break;

        case RS_CONN_STATE_CONNECTED:
            ret = RsSocketStateConnected(conn, sslEnable, rscb);
            break;

        case RS_CONN_STATE_SSL_BIND_FD:
            ret = RsSocketStateSslFdBind(conn, sslEnable, rscb);
            break;

        case RS_CONN_STATE_SSL_CONNECTED:
            hccp_info_socket("[client]IP(%s) connect port %d, fd:%d OK!", conn->serverIp.readAddr, conn->port,
                conn->connfd);
            RsSocketTagSync(conn);
            break;

        case RS_CONN_STATE_TAG_SYNC:
            if (gRsCb->connCb.wlistEnable == 1) {
                RsSocketClientValidSync(conn);
            }
            break;

        case RS_CONN_STATE_TIMEOUT:
            hccp_info_socket("[client]!send tag again! local_ip:%s server_ip:%s server_port:%u, tag:%s, fd:%d!",
                conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag, conn->connfd);
            RsSocketTagSync(conn);
            break;

        case RS_CONN_STATE_VALID_SYNC:
            break;

        case RS_CONN_STATE_TX_TO_HCCL:
            break;

        case RS_CONN_STATE_ERR:
            break;

        default:
            hccp_err("[client]Unknown state:%u, localIp:%s serverIp:%s serverPort:%u, tag:%s, fd:%d", conn->state,
                conn->clientIp.readAddr, conn->serverIp.readAddr, conn->port, conn->tag, conn->connfd);
            return -EINVAL;
    }

    return ret;
}

// 获取socket connect状态；返回值 0:connect中，1:connect完成
int RsGetSocketConnectState(struct RsConnInfo *conn)
{
    if ((conn->state == RS_CONN_STATE_TX_TO_HCCL) ||
        ((gRsCb->connCb.wlistEnable == 1) && (conn->state == RS_CONN_STATE_VALID_SYNC)) ||
        ((gRsCb->connCb.wlistEnable == 0) && (conn->state == RS_CONN_STATE_TAG_SYNC))) {
        return 1;
    } else {
        return 0;
    }
}

STATIC void RsSocketsIpAddrConverter(struct SocketConnectInfo conn[], int num)
{
    int j;

    for (j = 0; j < num; j++) {
        if (conn[j].family == AF_INET) {
            conn[j].localIp.addr.s_addr = RsSocketVnic2nodeid(conn[j].localIp.addr.s_addr);
            conn[j].remoteIp.addr.s_addr = RsSocketVnic2nodeid(conn[j].remoteIp.addr.s_addr);
        }
    }
}

static void RsSocketHandleConnNodeErr(uint32_t i, struct RsConnCb *connCb, struct SocketConnectInfo conn[],
    uint32_t serverPort)
{
    struct RsConnInfo *connInfo = NULL;
    uint32_t j;
    int ret;

    for (j = 0; j < i; j++) {
        ret = RsGetConnInfo(connCb, conn + j, &connInfo, serverPort);
        if (ret) {
            hccp_dbg("not find conn node, ret %d", ret);
        } else {
            RS_PTHREAD_MUTEX_LOCK(&connCb->rscb->mutex);
            RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
            RsListDel(&connInfo->list);
            free(connInfo);
            connInfo = NULL;
            RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
            RS_PTHREAD_MUTEX_ULOCK(&connCb->rscb->mutex);
        }
    }

    return;
}

STATIC int RsSocketConnectCheckPara(struct SocketConnectInfo *connInfo)
{
    if (((connInfo->family != AF_INET) && (connInfo->family != AF_INET6)) || connInfo->phyId >= RS_MAX_DEV_NUM ||
        strlen(connInfo->tag) >= SOCK_CONN_TAG_SIZE) {
        hccp_err("family[%d] invalid, or phyId[%u] invalid, or conn tag len:%u more than max len:%d", connInfo->family,
            connInfo->phyId, strlen(connInfo->tag), SOCK_CONN_TAG_SIZE);
        return -EINVAL;
    }

    return 0;
}

STATIC int rs_socket_IP_convert(struct SocketConnectInfo *connInfo, struct RsIpAddrInfo *remoteIp,
    struct RsIpAddrInfo *localIp)
{
    int retVal = 0;
    int ret = 0;

    if (connInfo->family == AF_INET) {
        uint32_t *remoteIpTmp = &(connInfo->remoteIp.addr.s_addr);
        uint32_t *localIpTmp = &(connInfo->localIp.addr.s_addr);
        retVal = RsSocketNodeid2vnic(*remoteIpTmp, remoteIpTmp);
        ret = RsSocketNodeid2vnic(*localIpTmp, localIpTmp);
        hccp_info_socket("local IP[0x%llx], ret:%d, remote IP[0x%llx], ret:%d", *localIpTmp, ret, *remoteIpTmp, retVal);
    }

    ret = RsConvertIpAddr(connInfo->family, &connInfo->remoteIp, remoteIp);
    CHK_PRT_RETURN(ret, hccp_err("convert(ntop) remote ip failed, ret:%d", ret), ret);

    ret = RsConvertIpAddr(connInfo->family, &connInfo->localIp, localIp);
    CHK_PRT_RETURN(ret, hccp_err("convert(ntop) local ip failed, ret:%d", ret), ret);

    hccp_info_socket("local IP[%s], ret:%d, remote IP[%s], ret:%d", localIp->readAddr, ret, remoteIp->readAddr, retVal);
    return 0;
}

RS_ATTRI_VISI_DEF int RsSocketBatchConnect(struct SocketConnectInfo conn[], uint32_t num)
{
    struct RsConnInfo *connInfo = NULL;
    struct RsConnCb *connCb = NULL;
    unsigned int chipId, serverPort;
    struct RsIpAddrInfo remoteIp;
    struct RsIpAddrInfo localIp;
    unsigned int i;
    int ret;

    RS_SOCKET_PARA_CHECK(num, conn);
    for (i = 0; i < num; i++) {
        serverPort = conn[i].port;
        ret = RsSocketConnectCheckPara(&conn[i]);
        if (ret) {
            hccp_err("rs_socket_connect_check_para for failed, ret:%d, i:%u", ret, i);
            goto conn_node_err_handle;
        }

        ret = rs_socket_IP_convert(&conn[i], &remoteIp, &localIp);
        if (ret) {
            hccp_err("convert ip invalid, ret %d", ret);
            goto conn_node_err_handle;
        }
        ret = rsGetLocalDevIDByHostDevID(conn[i].phyId, &chipId);
        if (ret) {
            hccp_err("phyId invalid, ret %d", ret);
            goto conn_node_err_handle;
        }

        ret = RsDev2conncb(chipId, &connCb);
        if (ret) {
            hccp_err("get conncb from dev failed(%d)!", ret);
            goto conn_node_err_handle;
        }

        if (conn[i].family == AF_INET6) {
            connCb->scopeId = RsGetIpv6ScopeId(conn[i].localIp.addr6);
            if (connCb->scopeId < 0) {
                hccp_err("scope_id[%d] is invalid", connCb->scopeId);
                connCb->scopeId = 0;
                goto conn_node_err_handle;
            }
        }

        ret = RsGetConnInfo(connCb, conn + i, &connInfo, serverPort);
        if (ret) {
            ret = RsAllocClientConnNode(connCb, RS_CONN_ROLE_CLIENT, &connInfo, &conn[i], &localIp, &remoteIp,
                serverPort);
            if (ret) {
                hccp_err("rs_alloc_client_conn_node failed, ret:%d, role:%d, localIp:%s, remoteIp:%s, serverPort:%u,"
                         " tag:%s",
                    ret, RS_CONN_ROLE_CLIENT, localIp.readAddr, remoteIp.readAddr, serverPort, conn[i].tag);
                goto conn_node_err_handle;
            }

            hccp_info_socket("create conn node for {remote_ip(%s), serverPort(%u), tag(%s)}!", remoteIp.readAddr,
                serverPort, connInfo->tag);
        } else {
            hccp_info_socket("conn node for {remote_ip(%s), serverPort(%u), tag(%s)} exist! state:%u",
                remoteIp.readAddr, serverPort, connInfo->tag, connInfo->state);
        }
    }
    sem_post(&gRsCb->connectTrigSem);
    RsSocketsIpAddrConverter(conn, num);
    return 0;

conn_node_err_handle:
    RsSocketHandleConnNodeErr(i, connCb, conn, serverPort);
    return ret;
}

STATIC int RsSocketCloseFd(int fd)
{
    int errNo = -1;
    int ret;

    do {
        ret = close(fd);
        if (ret < 0) {
            errNo = errno;
            CHK_PRT_RETURN(errNo != EINTR, hccp_err("close fd[%d] failed, ret:%d, errNo[%d]", fd, ret, errNo), -errNo);
        }
    } while ((ret < 0) && (errNo == EINTR));

    return 0;
}

RS_ATTRI_VISI_DEF int RsSocketBatchClose(int disuseLinger, struct RsSocketCloseInfoT conn[], uint32_t num)
{
    struct RsConnInfo *connInfo = NULL;
    struct linger soLinger;
    int fd = RS_FD_INVALID;
    int retVal = 0;
    unsigned int i;
    int ret;

    RS_SOCKET_PARA_CHECK(num, conn);

    for (i = 0; i < num; i++) {
        fd = conn[i].fd;
        CHK_PRT_RETURN(fd < 0, hccp_err("param error ! fd:%d, i:%d, num:%d", fd, i, num), -EINVAL);

        // strict mutex lock before find to make sure conn_info is valid on concurrent scenario
        RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
        ret = RsFd2conn(fd, &connInfo);
        if (ret != 0) {
            hccp_err("get conn failed! ret:%d", ret);
            RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
            return ret;
        }

        hccp_info_socket("conn node of IP(%s) fd:%d, state:%d", connInfo->serverIp.readAddr, connInfo->connfd,
            connInfo->state);

        RS_PTHREAD_MUTEX_LOCK(&gRsCb->connCb.connMutex);
        RsListDel(&connInfo->list);
        RS_PTHREAD_MUTEX_ULOCK(&gRsCb->connCb.connMutex);
        if (gRsCb->sslEnable == RS_SSL_ENABLE) {
            ssl_adp_shutdown(connInfo->ssl);
            ssl_adp_free(connInfo->ssl);
            connInfo->ssl = NULL;
        }
        RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);

        if (connInfo->state > RS_CONN_STATE_RESET) {
            soLinger.l_onoff = 1;
            soLinger.l_linger = disuseLinger == 0 ? RS_CLOSE_TIMEOUT : 0;
            ret = setsockopt(connInfo->connfd, SOL_SOCKET, SO_LINGER, &soLinger, sizeof(soLinger));
            if (ret) {
                hccp_err("setsockopt l_onoff:%d l_linger:%d failed err:%d", soLinger.l_onoff, soLinger.l_linger, errno);
                retVal = ret;
            }

            ret = RsSocketCloseFd(connInfo->connfd);
            if (ret) {
                hccp_err("rs_socket_close_fd for fd[%d] failed, ret[%d]", connInfo->connfd, ret);
                retVal = ret;
            }
        }

        free(connInfo);
        connInfo = NULL;
    }

    return retVal;
}

RS_ATTRI_VISI_DEF int RsSocketBatchAbort(struct SocketConnectInfo conn[], uint32_t num)
{
    struct RsConnInfo *connInfo = NULL;
    struct linger soLinger = {0};
    int retVal = 0;
    unsigned int i;
    int ret;

    RS_SOCKET_PARA_CHECK(num, conn);

    for (i = 0; i < num; i++) {
        // strict mutex lock before find to make sure conn_info is valid on concurrent scenario
        RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
        ret = RsGetConnInfo(&gRsCb->connCb, &conn[i], &connInfo, conn[i].port);
        if (ret != 0) {
            hccp_err("rs_get_conn_info conn:%u failed! ret:%d", i, ret);
            RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
            return ret;
        }

        hccp_info_socket("abort conn node of IP(%s) fd:%d, state:%d", connInfo->serverIp.readAddr, connInfo->connfd,
            connInfo->state);

        RS_PTHREAD_MUTEX_LOCK(&gRsCb->connCb.connMutex);
        RsListDel(&connInfo->list);
        RS_PTHREAD_MUTEX_ULOCK(&gRsCb->connCb.connMutex);
        if (gRsCb->sslEnable == RS_SSL_ENABLE && connInfo->ssl != NULL) {
            ssl_adp_shutdown(connInfo->ssl);
            ssl_adp_free(connInfo->ssl);
            connInfo->ssl = NULL;
            ssl_adp_clear_error();
        }
        RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);

        if (connInfo->state > RS_CONN_STATE_RESET && connInfo->connfd != RS_FD_INVALID) {
            // force to close fd
            soLinger.l_onoff = 1;
            ret = setsockopt(connInfo->connfd, SOL_SOCKET, SO_LINGER, &soLinger, sizeof(soLinger));
            if (ret) {
                hccp_err("setsockopt l_onoff:%d l_linger:%d failed err:%d", soLinger.l_onoff, soLinger.l_linger, errno);
                retVal = ret;
            }

            ret = RsSocketCloseFd(connInfo->connfd);
            if (ret) {
                hccp_err("rs_socket_close_fd for fd[%d] failed, ret[%d]", connInfo->connfd, ret);
                retVal = ret;
            }
        }

        free(connInfo);
        connInfo = NULL;
    }

    return retVal;
}

STATIC void RsSocketsBackfill(struct SocketFdData conn[], int sockNum, struct RsConnInfo *connTmp,
    struct RsVnicInfo vnicInfo)
{
    conn[sockNum].fd = connTmp->connfd;

    if (vnicInfo.role == RS_CONN_ROLE_SERVER) {
        conn[sockNum].remoteIp = connTmp->clientIp.binAddr;
    } else {
        conn[sockNum].remoteIp = connTmp->serverIp.binAddr;
    }

    conn[sockNum].status = RS_SOCK_STATUS_OK;
    connTmp->state = RS_CONN_STATE_TX_TO_HCCL;
    connTmp->isGot = true;
}

STATIC void RsSocketsServeripConverter(struct SocketFdData conn[], int num, uint32_t vnicFlag)
{
    int j;

    if (vnicFlag) {
        for (j = 0; j < num; j++) {
            if (conn[j].family == AF_INET) {
                conn[j].localIp.addr.s_addr = RsSocketVnic2nodeid(conn[j].localIp.addr.s_addr);
                conn[j].remoteIp.addr.s_addr = RsSocketVnic2nodeid(conn[j].remoteIp.addr.s_addr);
            }
        }
    }
}

STATIC int RsFindSockets(struct RsConnInfo *connTmp, struct SocketFdData conn[], int num, int role)
{
    int ret, i;

    /* normal process, no record log */
    if (gRsCb->connCb.wlistEnable == 1) {
        if (connTmp->state != RS_CONN_STATE_VALID_SYNC) {
            return -EINVAL;
        }
    } else {
        if (connTmp->state != RS_CONN_STATE_TAG_SYNC) {
            return -EINVAL;
        }
    }

    // server skip to get current socket once socket already been got
    if (role == RS_CONN_ROLE_SERVER && connTmp->isGot) {
        return -EINVAL;
    }

    if (role == RS_CONN_ROLE_SERVER) {
        i = 0;
        struct RsIpAddrInfo localIp;
        ret = RsConvertIpAddr(conn->family, &conn->localIp, &localIp);
        CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

        CHK_PRT_RETURN(RsCompareIpAddr(&connTmp->serverIp, &localIp),
            hccp_warn_socket("server_ip[%s] != local_ip[%s]", connTmp->serverIp.readAddr, localIp.readAddr), -EINVAL);
    } else {
        for (i = 0; i < num; i++) {
            if (conn[i].status == RS_SOCK_STATUS_OK) {
                continue;
            }

            struct RsIpAddrInfo remoteIp;
            remoteIp.family = (uint32_t)conn[i].family;
            remoteIp.binAddr = conn[i].remoteIp;
            struct RsIpAddrInfo localIp;
            localIp.family = (uint32_t)conn[i].family;
            localIp.binAddr = conn[i].localIp;
            if ((!RsCompareIpAddr(&connTmp->serverIp, &remoteIp)) && (!RsCompareIpAddr(&connTmp->clientIp, &localIp))) {
                break;
            }
        }
    }

    CHK_PRT_RETURN(i == num, hccp_warn_socket("i == num %d, not find serverIp[%s]", num, connTmp->serverIp.readAddr),
        -EINVAL);

    conn[i].tag[SOCK_CONN_TAG_SIZE - 1] = '\0';
    ret = strcmp(conn[i].tag, connTmp->tag);
    CHK_PRT_RETURN(ret,
        hccp_warn_socket("The %dth conn tag[%s] is different from conn_tmp_tag [%s]", i, conn[i].tag, connTmp->tag),
        -EINVAL);

    return i;
}

/* find it */
STATIC int RsSocketsCompare(struct RsListHead *listHead, struct SocketFdData conn[], uint32_t num,
    struct RsVnicInfo vnicInfo, struct RsConnCb *connCb)
{
    struct RsConnInfo *connTmp = NULL;
    struct RsConnInfo *connTmp2 = NULL;
    int sockNum = 0;
    int i;
    int sockIndex;

    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    connTmp = list_entry((listHead)->next, struct RsConnInfo, list);
    connTmp2 = list_entry(connTmp->list.next, struct RsConnInfo, list);
    for (; &connTmp->list != (listHead);) {
        i = RsFindSockets(connTmp, conn, num, vnicInfo.role);
        if (i < 0) {
            goto renew_conn;
        }
        sockIndex = (vnicInfo.role == RS_CONN_ROLE_SERVER) ? sockNum : i;
        RsSocketsBackfill(conn, sockIndex, connTmp, vnicInfo);

        sockNum++;
        if ((unsigned int)sockNum >= num) {
            break;
        }
    renew_conn:
        connTmp = connTmp2;
        connTmp2 = list_entry(connTmp2->list.next, struct RsConnInfo, list);
    }
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
    RsSocketsServeripConverter(conn, num, vnicInfo.vnicFlag);
    return sockNum;
}

STATIC int RsGetVnicFlag(uint32_t role, uint32_t *localIp, uint32_t *remoteIp)
{
    int vnicFlag = 0;

    if (role == RS_CONN_ROLE_SERVER) {
        if (RsSocketNodeid2vnic(*localIp, localIp) == RS_VNIC_FLAG) {
            vnicFlag = 1;
        }
    } else {
        if ((RsSocketNodeid2vnic(*remoteIp, remoteIp) == RS_VNIC_FLAG) &&
            (RsSocketNodeid2vnic(*localIp, localIp) == RS_VNIC_FLAG)) {
            vnicFlag = 1;
        }
    }
    return vnicFlag;
}

RS_ATTRI_VISI_DEF int RsGetSockets(uint32_t role, struct SocketFdData conn[], uint32_t num)
{
    struct RsListHead *listHead = NULL;
    struct RsVnicInfo vnicInfo = {0};
    struct RsConnCb *connCb = NULL;
    unsigned int chipId;
    uint32_t j;
    int ret;

    vnicInfo.role = role;

    RS_SOCKET_PARA_CHECK(num, conn);
    CHK_PRT_RETURN(role > RS_CONN_ROLE_CLIENT, hccp_err("para invalid. role[%u]", role), -EINVAL);

    /* set conn status to NA */
    for (j = 0; j < num; j++) {
        conn[j].status = 0;
        CHK_PRT_RETURN(((conn[j].family != AF_INET) && (conn[j].family != AF_INET6)) || conn[j].phyId >= RS_MAX_DEV_NUM,
            hccp_err("family[%d] invalid, or phyId[%u] invalid, j:%u", conn[j].family, conn[j].phyId, j), -EINVAL);

        CHK_PRT_RETURN(strlen(conn[j].tag) >= SOCK_CONN_TAG_SIZE,
            hccp_err("conn tag len:%u more than max len:%d", strlen(conn[j].tag), SOCK_CONN_TAG_SIZE), -EINVAL);

        if (conn[j].family == AF_INET) {
            uint32_t *localIp = &(conn[j].localIp.addr.s_addr);
            uint32_t *remoteIp = &(conn[j].remoteIp.addr.s_addr);
            vnicInfo.vnicFlag = (uint32_t)RsGetVnicFlag(role, localIp, remoteIp);
        }
    }

    ret = rsGetLocalDevIDByHostDevID(conn->phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId invalid, ret %d", ret), ret);

    ret = RsDev2conncb(chipId, &connCb);
    CHK_PRT_RETURN(ret, hccp_err("get conncb from dev failed! ret(%d)", ret), -ENODEV);

    listHead = (role == RS_CONN_ROLE_SERVER) ? (&connCb->serverConnList) : (&connCb->clientConnList);
    return RsSocketsCompare(listHead, conn, num, vnicInfo, connCb);
}

RS_ATTRI_VISI_DEF int RsGetSslEnable(uint32_t *sslEnable)
{
    CHK_PRT_RETURN(gRsCb == NULL, hccp_err("param error, gRsCb is NULL"), -ENODEV);
    CHK_PRT_RETURN(sslEnable == NULL, hccp_err("param error, sslEnable is NULL"), -EINVAL);

    *sslEnable = gRsCb->sslEnable;
    return 0;
}

RS_ATTRI_VISI_DEF int RsSocketSend(int fd, const void *data, uint64_t size)
{
    int ret;

    ret = RsDrvSocketSend(fd, data, size, MSG_DONTWAIT | MSG_NOSIGNAL);

    hccp_dbg("send fd:%d, size:%llu, send %dB", fd, size, ret);
    return ret;
}

RS_ATTRI_VISI_DEF int RsHdcSocketSend(int fd, const void *data, uint64_t size)
{
    static __thread char buf[RS_BUF_SIZE] = {0};
    const void *sendData = data;
    uint64_t sendSize = size;
    int ret;

    if (gRsCb->sslEnable == RS_SSL_ENABLE) {
        // reuse buf to fix SSL_write retry issue
        sendSize = (size > RS_BUF_SIZE) ? RS_BUF_SIZE : size;
        (void)memcpy_s(buf, sendSize, data, sendSize);
        sendData = buf;
    }

    ret = RsDrvSocketSend(fd, sendData, sendSize, MSG_DONTWAIT | MSG_NOSIGNAL);

    hccp_dbg("send fd:%d, sendSize:%llu, send %dB", fd, sendSize, ret);
    return ret;
}

RS_ATTRI_VISI_DEF int RsPeerSocketSend(uint32_t sslEnable, int fd, const void *data, uint64_t size)
{
    struct RsConnInfo *conn = NULL;
    int ret = 0;
    int errNo;

    CHK_PRT_RETURN(fd < 0 || size == 0 || data == NULL,
        hccp_err("param error ! fd:%d < 0, size:%llu or data is NULL", fd, size), -EINVAL);
    if (sslEnable != RS_SSL_DISABLE) {
        ret = RsFd2conn(fd, &conn);
        CHK_PRT_RETURN(ret, hccp_err("fd to conn failed, ret:%d", ret), ret);
        ret = ssl_adp_write(conn->ssl, data, (int)size);
        if (ret <= 0) {
            ret = RsSslWriteInnerCheck(conn, ret, size);
        }
    } else {
        ret = (int)send(fd, data, size, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (ret < 0) {
            errNo = errno;
            if (errNo == EAGAIN || errNo == EINTR) {
                hccp_dbg("send to fd:%d need retry, send size:%llu, ret:%d, errno:%d", fd, size, ret, errNo);
                ret = -EAGAIN;
            } else {
                hccp_run_info("send to fd:%d not success, send size:%llu, ret:%d, errno:%d", fd, size, ret, errNo);
                ret = -EFILEOPER;
            }
        }
    }

    return ret;
}

RS_ATTRI_VISI_DEF int RsSocketRecv(int fd, void *data, uint64_t size)
{
    int ret;

    ret = RsDrvSocketRecv(fd, data, size, MSG_DONTWAIT);

    return ret;
}

RS_ATTRI_VISI_DEF int RsPeerSocketRecv(uint32_t sslEnable, int fd, void *data, uint64_t size)
{
    struct RsConnInfo *conn = NULL;
    int ret = 0;
    int errNo;

    CHK_PRT_RETURN(fd < 0 || data == NULL || size == 0,
        hccp_err("param error ! fd:%d < 0 or data is NULL, size:%llu", fd, size), -EINVAL);
    if (sslEnable != RS_SSL_DISABLE) {
        ret = RsFd2conn(fd, &conn);
        CHK_PRT_RETURN(ret,
            hccp_warn_socket("can not find conn for fd[%d], ret:%d, the local fd may have been closed ", fd, ret), ret);
        ret = ssl_adp_read(conn->ssl, data, (int)size);
        if (ret <= 0) {
            ret = RsSslReadInnerCheck(conn->ssl, conn->connfd, ret, size);
        }
    } else {
        ret = (int)recv(fd, data, size, MSG_DONTWAIT);
        if (ret < 0) {
            errNo = errno;
            // not to print to avoid log flush
            if (errNo == EAGAIN || errNo == EINTR) {
                ret = -EAGAIN;
            } else {
                hccp_run_info("recv for fd:%d not success, recv size:%llu, ret:%d, errNo:%d", fd, size, ret, errNo);
                ret = -EFILEOPER;
            }
        }
    }

    return ret;
}

RS_ATTRI_VISI_DEF int RsSocketGetClientSocketErrInfo(struct SocketConnectInfo conn[], struct SocketErrInfo err[],
    unsigned int num)
{
    struct RsConnInfo *connInfo = NULL;
    unsigned int i, serverPort;
    int ret;

    RS_SOCKET_PARA_CHECK(num, conn);
    RS_CHECK_POINTER_NULL_WITH_RET(err);
    for (i = 0; i < num; i++) {
        serverPort = conn[i].port;
        RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
        ret = RsGetConnInfo(&gRsCb->connCb, &conn[i], &connInfo, serverPort);
        if (ret != 0) {
            hccp_err("rs_get_conn_info failed, i:%u ret:%d", i, ret);
            RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
            return ret;
        }

        (void)memcpy_s(&err[i], sizeof(struct SocketErrInfo), &connInfo->errInfo, sizeof(struct SocketErrInfo));

        // clear the singer socket connect err info
        (void)memset_s(&connInfo->errInfo, sizeof(struct SocketErrInfo), 0, sizeof(struct SocketErrInfo));
        RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsSocketGetServerSocketErrInfo(struct SocketListenInfo conn[], struct ServerSocketErrInfo err[],
    unsigned int num)
{
    struct RsListenInfo *listenInfo = NULL;
    struct RsIpAddrInfo ipInfo = {0};
    struct RsConnCb *connCb = NULL;
    unsigned int i, serverPort;
    int ret;

    RS_SOCKET_PARA_CHECK(num, conn);
    RS_CHECK_POINTER_NULL_WITH_RET(err);
    for (i = 0; i < num; i++) {
        ret = RsConvertIpAddr(conn[i].family, &conn[i].localIp, &ipInfo);
        CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, i:%u, ret:%d", i, ret), ret);

        serverPort = conn[i].port;
        RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
        connCb = &gRsCb->connCb;
        ret = RsFindListenNode(connCb, &ipInfo, serverPort, &listenInfo);
        if (ret != 0) {
            hccp_err("rs_find_listen_node failed, i:%u, ip:%s, serverPort:%u, ret:%d", i, ipInfo.readAddr, serverPort,
                ret);
            RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
            return ret;
        }

        (void)memcpy_s(&err[i].epollWait, sizeof(struct SocketErrInfo), &connCb->epollErrInfo,
            sizeof(struct SocketErrInfo));
        (void)memcpy_s(&err[i].accept, sizeof(struct SocketErrInfo), &listenInfo->errInfo,
            sizeof(struct SocketErrInfo));

        // clear the single socket listen err info
        (void)memset_s(&listenInfo->errInfo, sizeof(struct SocketErrInfo), 0, sizeof(struct SocketErrInfo));
        RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
    }

    return 0;
}

static void RsSocketGetIpInfo(unsigned int *serverIp, unsigned int *clientIp)
{
    uint32_t serverNodeId = *serverIp;
    uint32_t clientNodeId = *clientIp;
    int ret;

    ret = RsSocketNodeid2vnic(serverNodeId, serverIp);
    hccp_info_socket("white list listen IP 0x%llx, ret_vnic %d", *serverIp, ret);

    ret = RsSocketNodeid2vnic(clientNodeId, clientIp);
    hccp_info_socket("white list client IP 0x%llx, ret_vnic %d", *clientIp, ret);

    return;
}

STATIC int RsSocketWhiteListAlloc(struct RsConnCb *connCb, struct SocketWlistInfoT *whiteList,
    struct RsIpAddrInfo *serverIp)
{
    int ret;
    /*lint -e429*/
    struct RsWhiteListInfo *whiteListNodeTmp = NULL;
    struct RsWhiteList *whiteListTmp = NULL;
    struct SocketWlistInfoT wlist;
    struct RsIpAddrInfo clientIp;
    ret = memcpy_s(&wlist, sizeof(struct SocketWlistInfoT), whiteList, sizeof(struct SocketWlistInfoT));
    CHK_PRT_RETURN(ret, hccp_err("memcpy socket_wlist_info_t wlist failed, ret[%d]!", ret), -ESAFEFUNC);

    if (serverIp->family == AF_INET) {
        RsSocketGetIpInfo(&serverIp->binAddr.addr.s_addr, &(wlist.remoteIp.addr.s_addr));
        RsInetNtop(serverIp->family, &serverIp->binAddr, (char *)&serverIp->readAddr, sizeof(serverIp->readAddr));
    }

    ret = RsConvertIpAddr(serverIp->family, &wlist.remoteIp, &clientIp);
    CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

    ret = RsFindWhiteList(connCb, serverIp, &whiteListTmp);
    if (ret) {
        whiteListTmp = calloc(1, sizeof(struct RsWhiteList));
        CHK_PRT_RETURN(whiteListTmp == NULL, hccp_err("alloc mem for rs_white_list failed!"), -ENOMEM);
        whiteListTmp->serverIp = *serverIp;
        RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
        RS_INIT_LIST_HEAD(&whiteListTmp->whiteList);
        RsListAddTail(&whiteListTmp->list, &connCb->whiteList);
        RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
    }

    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    ret = RsFindWhiteListNode(whiteListTmp, &wlist, (int)serverIp->family, &whiteListNodeTmp);
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
    if (ret == 0) {
        whiteListNodeTmp->connLimit += wlist.connLimit;
        return 0;
    }

    whiteListNodeTmp = calloc(1, sizeof(struct RsWhiteListInfo));
    CHK_PRT_RETURN(whiteListNodeTmp == NULL, hccp_err("alloc mem for socket_wlist_info_t failed!"), -ENOMEM);

    whiteListNodeTmp->clientIp = clientIp;
    whiteListNodeTmp->connLimit = wlist.connLimit;
    ret = memcpy_s(whiteListNodeTmp->tag, SOCK_CONN_TAG_SIZE, wlist.tag, sizeof(wlist.tag));
    if (ret) {
        hccp_err("memcpy_s failed, ret[%d]. ", ret);
        free(whiteListNodeTmp);
        whiteListNodeTmp = NULL;
        return -ESAFEFUNC;
    }

    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    RsListAddTail(&whiteListNodeTmp->list, &whiteListTmp->whiteList);
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
    return 0;
    /*lint +e429*/
}

RS_ATTRI_VISI_DEF int RsSocketWhiteListSwitch(unsigned int phyId, unsigned int enable)
{
    struct RsConnCb *connCb = NULL;
    int ret;

    ret = RsDev2conncb(phyId, &connCb);
    CHK_PRT_RETURN(ret, hccp_err("get conncb from dev failed, ret:%d", ret), -1);
    connCb->wlistEnable = enable;
    return 0;
}

RS_ATTRI_VISI_DEF int RsSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    struct RsConnCb *connCb = &(gRsCb->connCb);
    struct RsIpAddrInfo serverIp;
    unsigned int i, chipId;
    int ret;

    ret = RsConvertIpAddr(rdevInfo.family, &rdevInfo.localIp, &serverIp);
    CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), -EINVAL);

    CHK_PRT_RETURN(num <= 0 || whiteList == NULL || num > RS_MAX_WLIST_NUM ||
                       ((rdevInfo.family != AF_INET) && (rdevInfo.family != AF_INET6)) ||
                       rdevInfo.phyId >= RS_MAX_DEV_NUM,
        hccp_err("white list add param error, phyId[%u], server ip[%s], num[%u], family[%d]", rdevInfo.phyId,
            serverIp.readAddr, num, rdevInfo.family),
        -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(rdevInfo.phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId invalid, ret %d", ret), ret);

    for (i = 0; i < num; ++i) {
        CHK_PRT_RETURN(strnlen(whiteList[i].tag, SOCK_CONN_TAG_SIZE) >= SOCK_CONN_TAG_SIZE,
            hccp_err("white_list tag len:%u more than max len:%d", strlen(whiteList[i].tag), SOCK_CONN_TAG_SIZE),
            -EINVAL);
        ret = RsSocketWhiteListAlloc(connCb, &whiteList[i], &serverIp);
        if (ret) {
            struct RsIpAddrInfo clientIp;
            ret = RsConvertIpAddr(serverIp.family, &whiteList->remoteIp, &clientIp);
            CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);
            hccp_err("add white list node failed, server ip[%s], client ip[%s], tag[%s], ret:%d", serverIp.readAddr,
                clientIp.readAddr, whiteList[i].tag, ret);
        }
    }
    return 0;
}

STATIC int RsSocketWhiteListNodeDestroy(struct RsConnCb *connCb, struct SocketWlistInfoT *whiteList,
    struct RsIpAddrInfo *serverIp)
{
    struct RsWhiteListInfo *whiteListNodeTmp = NULL;
    struct RsWhiteList *whiteListTmp = NULL;
    struct SocketWlistInfoT wlist;
    struct RsIpAddrInfo clientIp;
    int ret;

    ret = RsConvertIpAddr((int)serverIp->family, &whiteList->remoteIp, &clientIp);
    CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

    ret = memset_s(&wlist, sizeof(struct SocketWlistInfoT), 0, sizeof(struct SocketWlistInfoT));
    CHK_PRT_RETURN(ret, hccp_err("memset_s socket_wlist_info_t wlist failed, ret:%d", ret), -ESAFEFUNC);
    ret = memcpy_s(&wlist, sizeof(struct SocketWlistInfoT), whiteList, sizeof(struct SocketWlistInfoT));
    CHK_PRT_RETURN(ret, hccp_err("memcpy socket_wlist_info_t wlist failed!"), -ESAFEFUNC);

    if (serverIp->family == AF_INET) {
        ret = RsSocketNodeid2vnic(serverIp->binAddr.addr.s_addr, &serverIp->binAddr.addr.s_addr);
        hccp_info_socket("listen IP 0x%llx, ret_vnic %d", serverIp->binAddr.addr.s_addr, ret);
        ret = RsSocketNodeid2vnic(wlist.remoteIp.addr.s_addr, &(wlist.remoteIp.addr.s_addr));
        hccp_info_socket("client IP 0x%llx, ret_vnic %d", wlist.remoteIp.addr.s_addr, ret);
    }

    ret = RsFindWhiteList(connCb, serverIp, &whiteListTmp);
    CHK_PRT_RETURN(ret != 0, hccp_err("white list for IP(%s) doesn't exist! state:%d", serverIp->readAddr, ret), ret);
    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    ret = RsFindWhiteListNode(whiteListTmp, &wlist, (int)serverIp->family, &whiteListNodeTmp);
    if (ret == 0) {
        RsListDel(&whiteListNodeTmp->list);
        free(whiteListNodeTmp);
        whiteListNodeTmp = NULL;
        RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
        return 0;
    }
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
    hccp_info_socket("can not find white list node: client ip[%s], tag[%s], ret:%d", clientIp.readAddr, wlist.tag, ret);
    return ret;
}

RS_ATTRI_VISI_DEF int RsSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    struct RsConnCb *connCb = &(gRsCb->connCb);
    unsigned int i, chipId;
    struct RsIpAddrInfo serverIp;
    int ret;

    ret = RsConvertIpAddr(rdevInfo.family, &rdevInfo.localIp, &serverIp);
    CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

    CHK_PRT_RETURN(num <= 0 || whiteList == NULL || num > RS_MAX_WLIST_NUM ||
                       ((rdevInfo.family != AF_INET) && (rdevInfo.family != AF_INET6)) ||
                       rdevInfo.phyId >= RS_MAX_DEV_NUM,
        hccp_err("white list del param error, phyId[%u], server ip[%s], num[%u] family[%d]", rdevInfo.phyId,
            serverIp.readAddr, num, rdevInfo.family),
        -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(rdevInfo.phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId invalid, ret %d", ret), ret);

    for (i = 0; i < num; ++i) {
        CHK_PRT_RETURN(strlen(whiteList[i].tag) >= SOCK_CONN_TAG_SIZE,
            hccp_err("white_list tag len:%u more than "
                     "max len:%d",
                strlen(whiteList[i].tag), SOCK_CONN_TAG_SIZE),
            -EINVAL);
        ret = RsSocketWhiteListNodeDestroy(connCb, &whiteList[i], &serverIp);
        if (ret) {
            struct RsIpAddrInfo clientIp;
            ret = RsConvertIpAddr(serverIp.family, &whiteList->remoteIp, &clientIp);
            CHK_PRT_RETURN(ret, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);
            hccp_info_socket("white list node wait to delete, server ip[%s], client ip[%s], tag[%s], ret:%d",
                serverIp.readAddr, clientIp.readAddr, whiteList[i].tag, ret);
        }
    }
    return 0;
}

// 获取device网卡信息，当前device网卡只支持IPv4
STATIC int RsFillIfaddrInfos(struct IfaddrInfo ifaddrInfos[], unsigned int *num, unsigned int phyId)
{
    struct ifaddrs *ifaddr = NULL;
    struct ifaddrs *ifa = NULL;
    int family, ret;
    unsigned int numBak = *num;
    *num = 0;
    enum RsHardwareType type;

    type = RsGetDeviceType(phyId);
    CHK_PRT_RETURN(type == RS_HARDWARE_UNKNOWN, hccp_err("rs_get_device_type failed, type[%d]", type), -EINVAL);
    ret = getifaddrs(&ifaddr);
    CHK_PRT_RETURN(ret == -1, hccp_err("get ifaddrs failed, ret[%d]", ret), -ESYSFUNC);
    /* Walk through linked list, maintaining head pointer so we can free list later */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_netmask == NULL) {
            continue;
        }
        family = ifa->ifa_addr->sa_family;
        /* If not an AF_INET/AF_INET6 interface address, continue */
        if (family != AF_INET) {
            continue;
        }
        ret = RsCheckDstInterface(phyId, ifa->ifa_name, type, false);
        if (ret < 0) {
            hccp_err("rs_check_dst_interface failed, ret[%d]", ret);
            goto out;
        }
        if (ret) {
            (*num)++;
            if ((*num) > numBak) {
                hccp_err("num of interfaces found is more than expect, expect[%u], actual[%u]", numBak, *num);
                goto out;
            }
            ifaddrInfos[*num - 1].ip.addr = ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            ifaddrInfos[*num - 1].mask = ((struct sockaddr_in *)ifa->ifa_netmask)->sin_addr;
        }
    }

    freeifaddrs(ifaddr);
    ifaddr = NULL;
    return 0;
out:
    freeifaddrs(ifaddr);
    ifaddr = NULL;
    return -EAGAIN;
}

// 获取device网卡信息，支持IPv4/IPV6
STATIC int RsFillIfaddrInfosV2(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId, bool isAll)
{
    struct ifaddrs *ifaddr = NULL;
    struct ifaddrs *ifa = NULL;
    enum RsHardwareType type;
    unsigned int numBak;
    int family, ret;

    numBak = *num;
    *num = 0;
    type = RsGetDeviceType(phyId);
    CHK_PRT_RETURN(type == RS_HARDWARE_UNKNOWN, hccp_err("rs_get_device_type failed, type[%d]", type), -EINVAL);
    ret = getifaddrs(&ifaddr);
    CHK_PRT_RETURN(ret != 0, hccp_err("get ifaddrs failed, ret[%d]", ret), -ESYSFUNC);
    /* Walk through linked list, maintaining head pointer so we can free list later */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_netmask == NULL) {
            continue;
        }

        /* If not an AF_INET/AF_INET6 interface address, continue */
        family = ifa->ifa_addr->sa_family;
        if ((family != AF_INET) && (family != AF_INET6)) {
            continue;
        }

        ret = RsCheckDstInterface(phyId, ifa->ifa_name, type, isAll);
        if (ret < 0) {
            hccp_err("rs_check_dst_interface failed, ret[%d]", ret);
            ret = -EAGAIN;
            break;
        }
        if (ret) {
            (*num)++;
            if ((*num) > numBak) {
                hccp_err("num of interfaces found is more than expect, expect[%u], actual[%u]", numBak, *num);
                ret = -EAGAIN;
                break;
            }

            ret = strcpy_s(interfaceInfos[*num - 1].ifname, MAX_INTERFACE_NAME_LEN, ifa->ifa_name);
            if (ret) {
                hccp_err("strcpy interface name failed, ret[%d]", ret);
                ret = -EAGAIN;
                break;
            }
            interfaceInfos[*num - 1].scopeId = 0;
            if (family == AF_INET) {
                interfaceInfos[*num - 1].ifaddr.ip.addr = ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
                interfaceInfos[*num - 1].ifaddr.mask = ((struct sockaddr_in *)ifa->ifa_netmask)->sin_addr;
            } else {
                interfaceInfos[*num - 1].ifaddr.ip.addr6 = ((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
                interfaceInfos[*num - 1].scopeId = (int)((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_scope_id;
            }
            interfaceInfos[*num - 1].family = family;
        }
    }

    freeifaddrs(ifaddr);
    ifaddr = NULL;
    return ret;
}

STATIC int RsFillIfnum(unsigned int phyId, bool isAll, unsigned int *num, unsigned int isPeer)
{
    struct ifaddrs *ifaddr = NULL;
    struct ifaddrs *ifa = NULL;
    enum RsHardwareType type = RS_HARDWARE_UNKNOWN;
    int family, ret;
    *num = 0;

    if (isPeer == 0) {
        type = RsGetDeviceType(phyId);
        CHK_PRT_RETURN(type == RS_HARDWARE_UNKNOWN, hccp_err("rs_get_device_type failed, type[%d]", type), -EINVAL);
    }
    ret = getifaddrs(&ifaddr);
    CHK_PRT_RETURN(ret == -1, hccp_err("get ifaddrs failed, ret[%d]", ret), -ESYSFUNC);
    /* Walk through linked list, maintaining head pointer so we can free list later */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_netmask == NULL) {
            continue;
        }
        family = ifa->ifa_addr->sa_family;
        /* If not an AF_INET/AF_INET6 interface address, continue */
        if ((family != AF_INET) && (family != AF_INET6)) {
            continue;
        }
        if (isPeer == 0) {
            ret = RsCheckDstInterface(phyId, ifa->ifa_name, type, isAll);
            if (ret < 0) {
                hccp_err("rs_check_dst_interface failed, ret[%d]", ret);
                goto out;
            }
            if (ret) {
                (*num)++;
            }
        } else {
            (*num)++;
        }
    }

    freeifaddrs(ifaddr);
    ifaddr = NULL;
    return 0;
out:
    freeifaddrs(ifaddr);
    ifaddr = NULL;
    return -EAGAIN;
}

RS_ATTRI_VISI_DEF int RsPeerGetIfnum(unsigned int phyId, unsigned int *num)
{
    int ret;
    CHK_PRT_RETURN(num == NULL, hccp_err("param error, num is NULL"), -EINVAL);
    CHK_PRT_RETURN(gRsCb == NULL, hccp_err("param error, gRsCb is NULL"), -EINVAL);
    ret = RsPeerFillIfnum(phyId, num, gRsCb->ifaddrList);
    CHK_PRT_RETURN(ret, hccp_err("rs_peer_fill_ifnum failed, ret[%d]", ret), ret);
    return ret;
}

RS_ATTRI_VISI_DEF int RsGetIfnum(unsigned int phyId, bool isAll, unsigned int *num)
{
    int ret;
    CHK_PRT_RETURN(num == NULL, hccp_err("rs_get_ifaddrs param error, num is NULL"), -EINVAL);
    ret = RsFillIfnum(phyId, isAll, num, 0);
    CHK_PRT_RETURN(ret, hccp_err("rs_fill_ifnum failed, ret[%d]", ret), ret);
    return ret;
}

RS_ATTRI_VISI_DEF int RsPeerGetIfaddrs(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId)
{
    int ret;
    CHK_PRT_RETURN(interfaceInfos == NULL || num == NULL, hccp_err("param error, interfaceInfos or num is NULL"),
        -EINVAL);
    CHK_PRT_RETURN(gRsCb == NULL, hccp_err("param error, gRsCb is NULL"), -EINVAL);
    ret = RsPeerFillIfaddrInfos(interfaceInfos, num, phyId, gRsCb->ifaddrList);
    CHK_PRT_RETURN(ret, hccp_err("rs_peer_fill_ifaddr_infos failed, ret[%d]", ret), ret);
    return ret;
}

RS_ATTRI_VISI_DEF int RsGetIfaddrs(struct IfaddrInfo ifaddrInfos[], unsigned int *num, unsigned int phyId)
{
    int ret;

    CHK_PRT_RETURN(ifaddrInfos == NULL || num == NULL,
        hccp_err("rs_get_ifaddrs param error, "
                 "ifaddrInfos or num is NULL"),
        -EINVAL);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM || *num > MAX_INTERFACE_NUM,
        hccp_err("rs_get_ifaddrs param error,"
                 "phyId[%u], num[%u]",
            phyId, *num),
        -EINVAL);

    ret = RsFillIfaddrInfos(ifaddrInfos, num, phyId);
    CHK_PRT_RETURN(ret, hccp_err("rs_fill_ifaddr_infos failed, ret[%d]", ret), ret);

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetIfaddrsV2(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId,
    bool isAll)
{
    int ret;

    CHK_PRT_RETURN(interfaceInfos == NULL || num == NULL,
        hccp_err("rs_get_ifaddrs_v2 param error,"
                 "interfaceInfos or num is NULL"),
        -EINVAL);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM || *num > MAX_INTERFACE_NUM,
        hccp_err("rs_get_ifaddrs_v2 param error,"
                 "phyId[%u], num[%u]",
            phyId, *num),
        -EINVAL);

    ret = RsFillIfaddrInfosV2(interfaceInfos, num, phyId, isAll);
    CHK_PRT_RETURN(ret, hccp_err("rs_fill_ifaddr_infos_v2 failed, ret[%d]", ret), ret);

    return 0;
}

RS_ATTRI_VISI_DEF int RsSocketSetScopeId(unsigned int devId, int scopeId)
{
    int ret;
    unsigned int chipId;
    struct RsConnCb *connCb = NULL;
    ret = rsGetLocalDevIDByHostDevID(devId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId invalid, ret %d", ret), ret);

    ret = RsDev2conncb(chipId, &connCb);
    CHK_PRT_RETURN(ret, hccp_err("get conncb from dev failed, ret:%d", ret), ret);

    connCb->scopeId = scopeId;
    return 0;
}
