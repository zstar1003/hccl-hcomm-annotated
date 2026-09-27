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
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>

#include "securec.h"
#include "dl_hal_function.h"
#include "ssl_adp.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "rs.h"
#include "config_log.h"
#include "rs_inner.h"
#include "rs_epoll.h"
#include "rs_tls.h"
#include "rs_drv_socket.h"

int RsInetNtop(int family, union HccpIpAddr *ip, char readAddr[], unsigned int len)
{
    // IPv4/IPv6 二进制转字符串
    const char *str = NULL;
    str = inet_ntop(family, ip, readAddr, len);
    CHK_PRT_RETURN(str == NULL, hccp_err("[rs][inet_ntop]ip is an invalid address, err(%d), family %d", errno, family),
        -EINVAL);
    return 0;
}

int RsConvertIpAddr(int family, union HccpIpAddr *ipAddr, struct RsIpAddrInfo *ip)
{
    // IPv4/IPv6 二进制转内部IP数据格式（含二进制、字符串）
    ip->family = (uint32_t)family;
    ip->binAddr = *ipAddr;
    return RsInetNtop((int)ip->family, &ip->binAddr, (char *)&ip->readAddr, sizeof(ip->readAddr));
}

bool RsCompareIpAddr(struct RsIpAddrInfo *a, struct RsIpAddrInfo *b)
{
    // return: true(IP不同), false(IP相同)
    if (a->family != b->family) {
        return true;
    }
    if (a->family == AF_INET) {
        return (a->binAddr.addr.s_addr != b->binAddr.addr.s_addr);
    } else {
        return memcmp(&a->binAddr.addr6, &b->binAddr.addr6, sizeof(b->binAddr.addr6));
    }
}

int RsGetIpv6ScopeId(struct in6_addr localIp)
{
    struct in6_addr ipv6Addr = {0};
    struct ifaddrs *ifaddr = NULL;
    struct ifaddrs *ifa = NULL;
    int scopeId = 0;
    int ret, i;

    ret = getifaddrs(&ifaddr);
    CHK_PRT_RETURN(ret == -1, hccp_err("get ifaddrs failed, ret[%d]", ret), -ESYSFUNC);
    /* Walk through linked list, maintaining head pointer so we can free list later */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_INET6) {
            continue;
        }
        ipv6Addr = ((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
        for (i = 0; i < IPV6_S6_ADDR_SIZE; i++) {
            if (ipv6Addr.s6_addr[i] != localIp.s6_addr[i]) {
                break;
            }
        }
        if (i == IPV6_S6_ADDR_SIZE) { /* all 16 u6_addr8 in ipv6 are equal */
            scopeId = (int)((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_scope_id;
            freeifaddrs(ifaddr);
            ifaddr = NULL;
            return scopeId;
        }
    }

    hccp_err("get scope id failed");
    freeifaddrs(ifaddr);
    ifaddr = NULL;
    return -EINVAL;
}

enum RsHardwareType RsGetDeviceType(unsigned int phyId)
{
    int64_t deviceInfo = 0;
    unsigned int boardType;
    unsigned int logicId;
    unsigned int chipId;
    int64_t boardId;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("invalid param phy_id[%u]", phyId), RS_HARDWARE_UNKNOWN);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret != 0, hccp_err("phy_id[%u] invalid, ret %d", phyId, ret), RS_HARDWARE_UNKNOWN);
    ret = DlDrvDeviceGetIndexByPhyId(chipId, &logicId);
    CHK_PRT_RETURN(ret != 0, hccp_err("dl_drv_device_get_index_by_phy_id failed, ret(%d), chipId(%u)", ret, chipId),
        RS_HARDWARE_UNKNOWN);

    ret = DlHalGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_BOARD_ID, &boardId);
    CHK_PRT_RETURN(ret != 0, hccp_err("dl_hal_get_device_info board_id failed, ret[%d]", ret), RS_HARDWARE_UNKNOWN);
    hccp_info_others("board_id is (0x%llx)", boardId);
    boardType = (unsigned int)((uint64_t)boardId & (0xfff0));
    ret = DlHalGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_VERSION, &deviceInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("dl_hal_get_device_info device_info failed, ret(%d), phyId(%u)", ret, phyId),
        RS_HARDWARE_UNKNOWN);

    // 910A场景判断逻辑
    if (DlHalPlatGetChip((uint64_t)deviceInfo) == CHIP_TYPE_910A) {
        if (((boardType & RS_BOARDID_PCIE_CARD_MASK) == RS_BOARDID_PCIE_CARD_MASK_VALUE) &&
            (boardType != RS_BOARDID_AI_SERVER_MODULE) && (boardType != RS_BOARDID_ARM_SERVER_AG)) {
            return RS_HARDWARE_PCIE;
        }
        return RS_HARDWARE_SERVER;
    }

    if (((boardType & RS_BOARDID_PCIE_CARD_MASK) == RS_BOARDID_PCIE_CARD_MASK_VALUE) &&
        (boardType != RS_BOARDID_AI_SERVER_MODULE) && (boardType != RS_BOARDID_ARM_SERVER_AG) &&
        (boardType != RS_BOARDID_ARM_POD) && (boardType != RS_BOARDID_X86_16P) &&
        (boardType != RS_BOARDID_ARM_SERVER_2DIE)) {
        return RS_HARDWARE_PCIE;
    }

    if ((boardType == RS_BOARDID_ARM_SERVER_2DIE)) {
        return RS_HARDWARE_2DIE;
    }
    return RS_HARDWARE_SERVER;
}

int RsCheckDstInterface(unsigned int phyId, const char *ifaName, enum RsHardwareType type, bool isAll)
{
    char dstIfaBondName[RS_INTERFACE_BOND_LEN + 1] = {0};
    char dstIfaName[RS_INTERFACE_LEN + 1] = {0};
    int ret, bondRet;

    if (isAll) {
        /* get information of all device with eth or bond prefix */
        if (strncmp("eth", ifaName, RS_INTERFACE_ETH_PREFIX_LEN) != 0 &&
            strncmp("bond", ifaName, RS_INTERFACE_BOND_PREFIX_LEN) != 0) {
            return 0;
        }
        return 1;
    }

    // 标卡场景910B和910A device网卡固定为eth0,处理标卡场景
    if (type == RS_HARDWARE_PCIE) {
        if (strncmp("eth0", ifaName, RS_INTERFACE_LEN) && strncmp("eth1", ifaName, RS_INTERFACE_LEN)) {
            return 0;
        }
        return 1;
    } else if (type == RS_HARDWARE_2DIE) {
        /* 1. For RoH mode, only "bondx" port is supported when binding groups,
         * and "ethx" port is used when unbinding ;
         * 2. For eth mode, only the eth port is supported
         */
        ret = snprintf_s(dstIfaName, RS_INTERFACE_LEN + 1, RS_INTERFACE_LEN, "eth%u", phyId);
        bondRet = snprintf_s(dstIfaBondName, RS_INTERFACE_BOND_LEN + 1, RS_INTERFACE_BOND_LEN, "bond%u", phyId);
        if (ret <= 0 || bondRet <= 0) {
            hccp_err("copy eth or bond name failed, ret(%d), bondRet(%d)", ret, bondRet);
            return -EAGAIN;
        }

        if (strncmp(dstIfaName, ifaName, RS_INTERFACE_LEN) && strncmp(dstIfaBondName, ifaName, RS_INTERFACE_BOND_LEN)) {
            return 0;
        }
    } else {
        ret = snprintf_s(dstIfaName, RS_INTERFACE_LEN + 1, RS_INTERFACE_LEN, "eth%u", phyId);
        CHK_PRT_RETURN(ret <= 0, hccp_err("copy eth name failed, %d", ret), -EAGAIN);

        if (strncmp(dstIfaName, ifaName, RS_INTERFACE_LEN)) {
            return 0;
        }
    }
    return 1;
}

int RsPeerFillIfnum(unsigned int phyId, unsigned int *num, struct ifaddrs *ifaddrList)
{
    struct ifaddrs *ifaddr = ifaddrList;
    struct ifaddrs *ifa = NULL;
    int family;

    *num = 0;
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_netmask == NULL) {
            continue;
        }
        family = ifa->ifa_addr->sa_family;
        /* If not an AF_INET/AF_INET6 interface address, continue */
        if ((family != AF_INET) && (family != AF_INET6)) {
            continue;
        }
        (*num)++;
    }

    hccp_dbg("phy_id:%u got interface num:%u", phyId, *num);
    return 0;
}

int RsPeerFillIfaddrInfos(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId,
    struct ifaddrs *ifaddrList)
{
    struct ifaddrs *ifaddr = ifaddrList;
    unsigned int numBak = *num;
    struct ifaddrs *ifa = NULL;
    int family, ret;
    *num = 0;

    /* Walk through linked list, maintaining head pointer so we can free list later */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_netmask == NULL) {
            continue;
        }
        family = ifa->ifa_addr->sa_family;
        // /* If not an AF_INET/AF_INET6 interface address, continue */
        if ((family != AF_INET) && (family != AF_INET6)) {
            continue;
        }

        (*num)++;
        if ((*num) > numBak) {
            hccp_err("num of interfaces found is more than expect, expect[%u], actual[%u]", numBak, *num);
            goto out;
        }
        ret = strcpy_s(interfaceInfos[*num - 1].ifname, MAX_INTERFACE_NAME_LEN, ifa->ifa_name);
        if (ret != 0) {
            hccp_err("strcpy interface name failed, ret[%d]", ret);
            goto out;
        }
        interfaceInfos[*num - 1].scopeId = 0;
        if (family == AF_INET) {
            interfaceInfos[*num - 1].ifaddr.ip.addr = ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            interfaceInfos[*num - 1].ifaddr.mask = ((struct sockaddr_in *)ifa->ifa_netmask)->sin_addr;
            hccp_info_socket("ifname[%s] addr[0x%08x]", ifa->ifa_name,
                ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr.s_addr);
        } else {
            interfaceInfos[*num - 1].ifaddr.ip.addr6 = ((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
            interfaceInfos[*num - 1].scopeId = (int)((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_scope_id;
            hccp_info_socket("ifname[%s] scope_id[%u] flowinfo[%u]", ifa->ifa_name,
                ((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_scope_id,
                ((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_flowinfo);
            for (unsigned long i = 0; i < sizeof(struct in6_addr); i++) {
                hccp_info_socket("addr[%lu] 0x%02x", i, ((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr.s6_addr[i]);
            }
        }
        interfaceInfos[*num - 1].family = family;
    }

    hccp_dbg("phy_id:%u got interface num:%u", phyId, *num);
    return 0;
out:
    hccp_dbg("phy_id:%u got interface num:%u", phyId, *num);
    return -EAGAIN;
}

int RsDrvSslBindFd(struct RsConnInfo *conn, int fd)
{
    int ret;
    if (conn->ssl == NULL) {
        conn->ssl = ssl_adp_new(gRsCb->clientSslCtx);
        CHK_PRT_RETURN(conn->ssl == NULL, hccp_err("server ssl ctx alloc failed"), -ENOMEM);
    }

    ssl_adp_set_mode(conn->ssl, SSL_MODE_AUTO_RETRY);
    ret = ssl_adp_set_fd(conn->ssl, fd);
    if (ret != 1) {
        hccp_err("bind connfd and ssl failed, ret %d", ret);
        goto out;
    }

    ssl_adp_set_connect_state(conn->ssl);

    return 0;
out:
    ssl_adp_shutdown(conn->ssl);
    ssl_adp_free(conn->ssl);
    conn->ssl = NULL;
    return -EINVAL;
}

int RsDrvConnect(int fd, struct RsIpAddrInfo *serverIp, struct RsIpAddrInfo *clientIp, uint16_t port)
{
    union RsSocketaddr clientAddr = {0};
    socklen_t clientAddrLen = 0;
    uint16_t clientPort = 0;
    int errNo;
    int ret;

    hccp_info_socket("IP(%s) port %d family %d fd:%d begin", serverIp->readAddr, port, clientIp->family, fd);
    if (clientIp->family == AF_INET) {
        struct sockaddr_in addr = {0};
        addr.sin_family = clientIp->family;
        addr.sin_port = htons(port);
        addr.sin_addr = serverIp->binAddr.addr;
        ret = connect(fd, &addr, sizeof(addr));
    } else {
        struct sockaddr_in6 addr = {0};
        addr.sin6_family = clientIp->family;
        addr.sin6_port = htons(port);
        addr.sin6_addr = serverIp->binAddr.addr6;
        ret = connect(fd, &addr, sizeof(addr));
    }

    if (ret) {
        errNo = errno;
        if (errNo == -EISCONN) {
            goto out;
        }

        /*
         * if the errno is EINTR, it can not retry directly,
         * otherwise it will directly return an error
         */
        hccp_warn_socket("connect not success, need to try again! server IP:%s, port:%d, fd:%d, ret:%d, errNo:%d",
            serverIp->readAddr, port, fd, ret, errNo);

        return -errNo;
    }

out:
    clientAddrLen = (clientIp->family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    getsockname(fd, (struct sockaddr *)&clientAddr, &clientAddrLen);
    clientPort = (clientIp->family == AF_INET) ? ntohs(clientAddr.sAddr.sin_port) : ntohs(clientAddr.sAddr6.sin6_port);

    if ((clientPort < 60000) || (clientPort > 60015)) { // HCCL默认监听60000-60015端口,如client使用该端口，记录EVENT日志
        hccp_run_info("client connect success. client family %d addr %s:%u, server addr %s:%u, fd:%d", clientIp->family,
            clientIp->readAddr, clientPort, serverIp->readAddr, port, fd);
    } else {
        hccp_run_info("client connect success. client family %d addr %s:%u, server addr %s:%u, fd:%d", clientIp->family,
            clientIp->readAddr, clientPort, serverIp->readAddr, port, fd);
    }

    return 0;
}

int RsFd2conn(int fd, struct RsConnInfo **conn)
{
    struct RsConnInfo *connTmp = NULL;
    struct RsConnInfo *connTmp2 = NULL;
    struct RsListHead *head = NULL;
    struct rs_cb *rsCb = NULL;

    if (gRsCb != NULL) {
        rsCb = gRsCb;
    } else {
        hccp_err("g_rs_cb is NULL");
        return -ENODEV;
    }

    RS_PTHREAD_MUTEX_LOCK(&rsCb->connCb.connMutex);
    head = &rsCb->connCb.serverConnList;
    RS_LIST_GET_HEAD_ENTRY(connTmp, connTmp2, head, list, struct RsConnInfo);
    for (; &connTmp->list != head;
         connTmp = connTmp2, connTmp2 = list_entry(connTmp2->list.next, struct RsConnInfo, list)) {
        if (connTmp->connfd == fd) {
            *conn = connTmp;
            RS_PTHREAD_MUTEX_ULOCK(&rsCb->connCb.connMutex);
            return 0;
        }
    }

    head = &rsCb->connCb.clientConnList;
    RS_LIST_GET_HEAD_ENTRY(connTmp, connTmp2, head, list, struct RsConnInfo);
    for (; &connTmp->list != head;
         connTmp = connTmp2, connTmp2 = list_entry(connTmp2->list.next, struct RsConnInfo, list)) {
        if (connTmp->connfd == fd) {
            *conn = connTmp;
            RS_PTHREAD_MUTEX_ULOCK(&rsCb->connCb.connMutex);
            return 0;
        }
    }

    RS_PTHREAD_MUTEX_ULOCK(&rsCb->connCb.connMutex);

    hccp_warn_socket("cannot find conn node for fd:%d!", fd);
    *conn = NULL;

    return -ENODEV;
}

int RsSslWriteInnerCheck(struct RsConnInfo *conn, int sslRet, uint64_t size)
{
    int err = ssl_adp_get_error(conn->ssl, sslRet);
    int fd = conn->connfd;
    int errNo = errno;

    rs_ssl_err_string(fd, err);
    CHK_PRT_RETURN((err == SSL_ERROR_WANT_WRITE) || (err == SSL_ERROR_WANT_READ),
        hccp_info_socket("ssl_adp_write fd:%d need to retry, err:%d errno:%d", fd, err, errNo), -EAGAIN);
    CHK_PRT_RETURN((err == SSL_ERROR_SYSCALL) && (errNo == EAGAIN || errNo == EWOULDBLOCK || errNo == EINTR),
        hccp_info_socket("ssl_adp_write fd:%d need to retry, err:%d errno:%d", fd, err, errNo), -EAGAIN);

    // degrade log level to prevent false alarms and log flooding in heartbeat monitor scenario
    hccp_warn_socket("ssl_adp_write fd:%d ret:%d, size:%llu err:%d errno:%d", fd, sslRet, size, err, errNo);
    return sslRet;
}

int RsDrvSocketSend(int fd, const void *data, uint64_t size, int flags)
{
    struct RsConnInfo *conn = NULL;
    int ret = 0;
    int errNo;

    CHK_PRT_RETURN(fd < 0 || size == 0 || data == NULL,
        hccp_err("param error ! fd:%d < 0, size:%llu or data is NULL", fd, size), -EINVAL);

    if (gRsCb->sslEnable == RS_SSL_ENABLE) {
        ret = RsFd2conn(fd, &conn);
        CHK_PRT_RETURN(ret != 0, hccp_err("fd:%d to conn failed, ret:%d", fd, ret), ret);
        ret = ssl_adp_write(conn->ssl, data, size);
        if (ret <= 0) {
            ret = RsSslWriteInnerCheck(conn, ret, size);
        }
    } else {
        ret = send(fd, data, size, flags);
        if (ret < 0) {
            errNo = errno;
            if (errNo == EAGAIN || errNo == EINTR) {
                hccp_dbg("send to fd:%d need retry, send size:%llu, ret:%d, errno:%d", fd, size, ret, errNo);
                ret = -EAGAIN;
            } else {
                hccp_warn_socket("send to fd:%d not success, send size:%llu, ret:%d, errno:%d", fd, size, ret, errNo);
                ret = -EFILEOPER;
            }
        }
    }

    return ret;
}

int RsSslReadInnerCheck(SSL *ssl, int fd, int sslRet, uint64_t size)
{
    int err = ssl_adp_get_error(ssl, sslRet);
    int errNo = errno;

    rs_ssl_err_string(fd, err);
    CHK_PRT_RETURN((err == SSL_ERROR_WANT_WRITE) || (err == SSL_ERROR_WANT_READ),
        hccp_dbg("ssl_adp_read fd:%d need to retry, err:%d errno:%d", fd, err, errNo), -EAGAIN);
    CHK_PRT_RETURN((err == SSL_ERROR_SYSCALL) && (errNo == EAGAIN || errNo == EWOULDBLOCK || errNo == EINTR),
        hccp_dbg("ssl_adp_read fd:%d need to retry, err:%d errno:%d", fd, err, errNo), -EAGAIN);

    // degrade log level to prevent false alarms and log flooding in heartbeat monitor scenario
    hccp_warn_socket("ssl_adp_read fd:%d ret:%d, size:%llu err:%d errno:%d", fd, sslRet, size, err, errNo);
    return sslRet;
}

int RsDrvSocketRecv(int fd, void *data, uint64_t size, int flags)
{
    struct RsConnInfo *conn = NULL;
    int ret = 0;
    int errNo;

    CHK_PRT_RETURN(fd < 0 || data == NULL || size == 0,
        hccp_err("param error ! fd:%d < 0 or data is NULL, size:%llu", fd, size), -EINVAL);

    if (gRsCb->sslEnable == RS_SSL_ENABLE) {
        ret = RsFd2conn(fd, &conn);
        CHK_PRT_RETURN(ret,
            hccp_warn_socket("can not find conn for fd[%d], ret:%d, the local fd may have been closed ", fd, ret), ret);
        ret = ssl_adp_read(conn->ssl, data, size);
        if (ret <= 0) {
            ret = RsSslReadInnerCheck(conn->ssl, conn->connfd, ret, size);
        }
    } else {
        ret = recv(fd, data, size, flags);
        if (ret < 0) {
            errNo = errno;
            // not to print to avoid log flush
            if (errNo == EAGAIN || errNo == EINTR) {
                ret = -EAGAIN;
            } else {
                hccp_warn_socket("recv for fd:%d not success, recv size:%llu, ret:%d, errNo:%d", fd, size, ret, errNo);
                ret = -EFILEOPER;
            }
        }
    }

    return ret;
}

void ShowConnNode(struct RsListHead *listHead)
{
    struct RsConnInfo *connTmp2 = NULL;
    struct RsConnInfo *connTmp = NULL;

    RS_LIST_GET_HEAD_ENTRY(connTmp, connTmp2, listHead, list, struct RsConnInfo);
    for (; (&connTmp->list) != listHead;
         connTmp = connTmp2, connTmp2 = list_entry(connTmp2->list.next, struct RsConnInfo, list)) {
        hccp_info_socket("current server ip: %s, client ip:%s, fd:%d, state:%d, tag:%s", connTmp->serverIp.readAddr,
            connTmp->clientIp.readAddr, connTmp->connfd, connTmp->state, connTmp->tag);
    }
}

int RsGetConnInfo(struct RsConnCb *connCb, struct SocketConnectInfo *conn, struct RsConnInfo **connInfo,
    unsigned int serverPort)
{
    struct RsConnInfo *connTmp2 = NULL;
    struct RsConnInfo *connTmp = NULL;
    struct RsIpAddrInfo ipAddr;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(connCb);
    RS_CHECK_POINTER_NULL_RETURN_INT(conn);

    ret = RsConvertIpAddr(conn->family, &conn->remoteIp, &ipAddr);
    CHK_PRT_RETURN(ret != 0, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    RS_LIST_GET_HEAD_ENTRY(connTmp, connTmp2, &connCb->clientConnList, list, struct RsConnInfo);
    for (; (&connTmp->list) != &connCb->clientConnList;
         connTmp = connTmp2, connTmp2 = list_entry(connTmp2->list.next, struct RsConnInfo, list)) {
        if ((!RsCompareIpAddr(&connTmp->serverIp, &ipAddr)) && connTmp->port == serverPort) {
            ret = strcmp(connTmp->tag, conn->tag);
            if (ret == 0) {
                *connInfo = connTmp;
                RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
                return 0;
            }
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);

    conn->tag[SOCK_CONN_TAG_SIZE - 1] = '\0';
    hccp_warn_socket("conn node for IP(%s) server_port(%u) tag(%s) not found", ipAddr.readAddr, serverPort, conn->tag);
    return -ENODEV;
}

int RsFindListenNode(struct RsConnCb *connCb, struct RsIpAddrInfo *ipAddr, uint32_t serverPort,
    struct RsListenInfo **listenInfo)
{
    struct RsListenInfo *listenTmp2 = NULL;
    struct RsListenInfo *listenTmp = NULL;

    RS_CHECK_POINTER_NULL_WITH_RET(connCb);
    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    RS_LIST_GET_HEAD_ENTRY(listenTmp, listenTmp2, &connCb->listenList, list, struct RsListenInfo);
    for (; (&listenTmp->list) != &connCb->listenList;
         listenTmp = listenTmp2, listenTmp2 = list_entry(listenTmp2->list.next, struct RsListenInfo, list)) {
        if ((!RsCompareIpAddr(&listenTmp->serverIpAddr, ipAddr)) && (listenTmp->sockPort == serverPort)) {
            *listenInfo = listenTmp;
            RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
            return 0;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);

    hccp_info_socket("listen node for IP(%s), serverPort(%u) is not listen!", ipAddr->readAddr, serverPort);
    return -ENODEV;
}

int RsSocketListenAddToEpoll(struct RsConnCb *connCb, struct RsListenInfo *listenInfo)
{
    int ret = 0;

    RS_PTHREAD_MUTEX_LOCK(&listenInfo->acceptCreditMutex);
    if (listenInfo->fdState == LISTEN_FD_STATE_ADDED) {
        goto out;
    }

    // should ctl_add to make sure epoll event can be triggered
    hccp_run_info("IP:%s server_port:%u listen_fd:%d add to epoll:%d", listenInfo->serverIpAddr.readAddr,
        listenInfo->sockPort, listenInfo->listenFd, connCb->epollfd);
    ret = RsEpollCtl(connCb->epollfd, EPOLL_CTL_ADD, listenInfo->listenFd, EPOLLIN);
    if (ret != 0) {
        hccp_err("IP:%s server_port:%u listen_fd:%d rs_epoll_ctl failed, ret:%d errno:%d",
            listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, listenInfo->listenFd, ret, errno);
        goto out;
    }

    listenInfo->fdState = LISTEN_FD_STATE_ADDED;

out:
    RS_PTHREAD_MUTEX_ULOCK(&listenInfo->acceptCreditMutex);
    return ret;
}

STATIC int RsListenCreditLimitInit(struct RsListenInfo *listenInfo)
{
    int ret;

    ret = pthread_mutex_init(&listenInfo->acceptCreditMutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("mutex_init accept_credit_mutex failed, ret:%d", ret), -ESYSFUNC);
    return 0;
}

STATIC void RsListenCreditLimitDeinit(struct RsListenInfo *listenInfo)
{
    (void)pthread_mutex_destroy(&listenInfo->acceptCreditMutex);
}

int RsListenNodeAlloc(struct RsConnCb *connCb, struct RsIpAddrInfo *ipAddr, uint32_t serverPort,
    struct RsListenInfo **node)
{
    struct RsListenInfo *listenInfo = NULL;
    int ret;

    ret = RsFindListenNode(connCb, ipAddr, serverPort, &listenInfo);
    CHK_PRT_RETURN(ret == 0,
        hccp_info_socket("listen node for IP(%s) exist! state:%u", ipAddr->readAddr, listenInfo->state), -EEXIST);

    listenInfo = calloc(1, sizeof(struct RsListenInfo));
    CHK_PRT_RETURN(listenInfo == NULL, hccp_err("alloc mem for socket listen info failed!"), -ENOMEM);

    hccp_info_socket("create listen node for IP(%s)!", ipAddr->readAddr);
    listenInfo->serverIpAddr = *ipAddr;
    listenInfo->state = RS_CONN_STATE_RESET;
    ret = RsListenCreditLimitInit(listenInfo);
    if (ret != 0) {
        hccp_err("rs_listen_credit_limit_init failed, ret:%d", ret);
        free(listenInfo);
        listenInfo = NULL;
        return ret;
    }

    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    RsListAddTail(&listenInfo->list, &connCb->listenList);
    (void)__sync_fetch_and_add(&(listenInfo->counter), 1);
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);

    *node = listenInfo;

    return 0;
}

int RsSocketListenDelFromEpoll(struct RsConnCb *connCb, struct RsListenInfo *listenInfo)
{
    int ret = 0;

    RS_PTHREAD_MUTEX_LOCK(&listenInfo->acceptCreditMutex);
    if (listenInfo->fdState == LISTEN_FD_STATE_DELETED) {
        goto out;
    }

    hccp_run_info("IP:%s server_port:%u listen_fd:%d del from epoll:%d", listenInfo->serverIpAddr.readAddr,
        listenInfo->sockPort, listenInfo->listenFd, connCb->epollfd);
    ret = RsEpollCtl(connCb->epollfd, EPOLL_CTL_DEL, listenInfo->listenFd, EPOLLIN);
    if (ret != 0) {
        hccp_err("IP:%s server_port:%u listen_fd:%d rs_epoll_ctl failed, ret:%d errno:%d",
            listenInfo->serverIpAddr.readAddr, listenInfo->sockPort, listenInfo->listenFd, ret, errno);
        goto out;
    }

    listenInfo->fdState = LISTEN_FD_STATE_DELETED;

out:
    RS_PTHREAD_MUTEX_ULOCK(&listenInfo->acceptCreditMutex);
    return ret;
}

void RsListenNodeFree(struct RsConnCb *connCb, struct RsListenInfo *node)
{
    RS_CHECK_POINTER_NULL_RETURN_VOID(connCb);
    RS_CHECK_POINTER_NULL_RETURN_VOID(node);

    hccp_dbg("delete listen node for (IP %s : port %u)!", node->serverIpAddr.readAddr, node->sockPort);

    RS_PTHREAD_MUTEX_LOCK(&connCb->rscb->mutex);
    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    RsListDel(&node->list);
    RsListenCreditLimitDeinit(node);
    free(node);
    node = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
    RS_PTHREAD_MUTEX_ULOCK(&connCb->rscb->mutex);

    return;
}

int RsAllocConnNode(struct RsConnInfo **conn, unsigned short serverPort)
{
    struct RsConnInfo *connInfo;

    connInfo = calloc(1, sizeof(struct RsConnInfo));
    CHK_PRT_RETURN(connInfo == NULL, hccp_err("alloc mem for socket conn info failed!"), -ENOMEM);

    connInfo->port = serverPort;
    connInfo->connfd = RS_FD_INVALID;
    connInfo->state = RS_CONN_STATE_RESET;

    *conn = connInfo;

    return 0;
}

int RsFindWhiteListNode(struct RsWhiteList *rsSocketWhiteList, struct SocketWlistInfoT *whiteListExpect, int family,
    struct RsWhiteListInfo **whiteListNode)
{
    struct RsWhiteListInfo *whiteListTmp2 = NULL;
    struct RsWhiteListInfo *whiteListTmp = NULL;
    struct RsIpAddrInfo expectIp;
    int ret;

    ret = RsConvertIpAddr(family, &whiteListExpect->remoteIp, &expectIp);
    CHK_PRT_RETURN(ret != 0, hccp_err("convert(ntop) ip failed, ret:%d", ret), ret);

    RS_CHECK_POINTER_NULL_WITH_RET(rsSocketWhiteList);
    RS_LIST_GET_HEAD_ENTRY(whiteListTmp, whiteListTmp2, &rsSocketWhiteList->whiteList, list, struct RsWhiteListInfo);
    for (; (&whiteListTmp->list) != &rsSocketWhiteList->whiteList;
         whiteListTmp = whiteListTmp2,
         whiteListTmp2 = list_entry(whiteListTmp2->list.next, struct RsWhiteListInfo, list)) {
        hccp_info_socket("client_ip %s 0x%08x, expectIp %s 0x%08x", whiteListTmp->clientIp.readAddr,
            whiteListTmp->clientIp.binAddr.addr.s_addr, expectIp.readAddr, expectIp.binAddr.addr.s_addr);
        if ((!RsCompareIpAddr(&whiteListTmp->clientIp, &expectIp)) &&
            (strncmp(whiteListTmp->tag, whiteListExpect->tag, SOCK_CONN_TAG_SIZE) == 0)) {
            *whiteListNode = whiteListTmp;
            return 0;
        }
    }

    whiteListExpect->tag[SOCK_CONN_TAG_SIZE - 1] = '\0';
    hccp_info_socket("white list node for IP(%s), tag(%s) doesn't exist!", expectIp.readAddr, whiteListExpect->tag);
    return -ENODEV;
}

int RsFindWhiteList(struct RsConnCb *connCb, struct RsIpAddrInfo *serverIp, struct RsWhiteList **whiteList)
{
    struct RsWhiteList *whiteListTmp2 = NULL;
    struct RsWhiteList *whiteListTmp = NULL;

    RS_CHECK_POINTER_NULL_WITH_RET(connCb);
    RS_PTHREAD_MUTEX_LOCK(&connCb->connMutex);
    RS_LIST_GET_HEAD_ENTRY(whiteListTmp, whiteListTmp2, &connCb->whiteList, list, struct RsWhiteList);
    for (; (&whiteListTmp->list) != &connCb->whiteList;
         whiteListTmp = whiteListTmp2, whiteListTmp2 = list_entry(whiteListTmp2->list.next, struct RsWhiteList, list)) {
        if (!RsCompareIpAddr(serverIp, &whiteListTmp->serverIp)) {
            *whiteList = whiteListTmp;
            RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);
            return 0;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&connCb->connMutex);

    hccp_info_socket("white list for IP(%s) doesn't exist!", serverIp->readAddr);
    return -ENODEV;
}

void RsSocketGetBindByChip(unsigned int chipId, bool *bindIp)
{
#define CHIP_NAME_910_93 "910_93"
    halChipInfo chipInfo = {0};
    int64_t deviceInfo = 0;
    unsigned int logicId;
    int ret;

    // get chip info failed, return directly to avoid exit from batch connect
    ret = DlDrvDeviceGetIndexByPhyId(chipId, &logicId);
    if (ret != 0) {
        hccp_warn_others("dl_drv_device_get_index_by_phy_id unsuccessful, ret(%d), chipId(%u)", ret, chipId);
        return;
    }
    ret = DlHalGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_VERSION, &deviceInfo);
    if (ret != 0) {
        hccp_warn_others("dl_hal_get_device_info unsuccessful, ret(%d), logicId(%u)", ret, logicId);
        return;
    }

    // chip force to bind: 310P & 910_93
    if ((DlHalPlatGetChip((uint64_t)deviceInfo) == CHIP_TYPE_310P) ||
        ((DlHalPlatGetChip((uint64_t)deviceInfo) == CHIP_TYPE_910B_910_93) &&
            (DlHalPlatGetVer((uint64_t)deviceInfo) >= VER_BIN5) &&
            (DlHalPlatGetVer((uint64_t)deviceInfo) <= VER_BIN8))) {
        *bindIp = true;
        return;
    }

    // get chip info, chip force to bind: 910_93
    ret = DlHalGetChipInfo(logicId, &chipInfo);
    if (ret != 0) {
        hccp_warn_others("dl_hal_get_chip_info unsuccessful, ret(%d), logicId(%u)", ret, logicId);
        return;
    }
    if (strncmp((char *)chipInfo.name, CHIP_NAME_910_93, sizeof(CHIP_NAME_910_93) - 1) == 0) {
        *bindIp = true;
    }

    return;
}

bool RsSocketIsVnicIp(unsigned int chipId, unsigned int ipAddr)
{
    unsigned int vnicIp = 0;
    int64_t deviceInfo = 0;
    unsigned int phyId = 0;
    bool bindIp = false;
    int hccpMode;
    int ret;

    // no need to handle other mode, only need to handle HDC mode
    hccpMode = RsGetHccpMode(chipId);
    if (hccpMode != NETWORK_OFFLINE) {
        return false;
    }

    // check chip info: 310P & 910_93 will force to bind, no need to compare ip_addr with vnic ip
    RsSocketGetBindByChip(chipId, &bindIp);
    if (bindIp) {
        return false;
    }

    // compare ip_addr with current vnic_ip
    ret = rsGetDevIDByLocalDevID(chipId, &phyId);
    if (ret != 0) {
        hccp_warn_others("rsGetDevIDByLocalDevID unsuccessful, ret(%d), chipId(%u)", ret, chipId);
        return false;
    }

    ret = DlHalGetDeviceInfo(phyId, MODULE_TYPE_SYSTEM, INFO_TYPE_VNIC_IP, &deviceInfo);
    if (ret != 0) {
        hccp_warn_others("dl_hal_get_device_info unsuccessful, ret(%d), chipId(%u), phyId(%u)", ret, chipId, phyId);
        return false;
    }

    vnicIp = (unsigned int)deviceInfo;
    hccp_dbg("chip_id:%u phy_id:%u vnic_ip:%u ip_addr:%u", chipId, phyId, vnicIp, ipAddr);
    if (vnicIp == ipAddr) {
        return true;
    }

    return false;
}

void RsConnCostTime(struct RsConnInfo *conn)
{
    float timeCost = 0.0;

    RsGetCurTime(&conn->endTime);
    HccpTimeInterval(&conn->endTime, &conn->startTime, &timeCost);
    if (timeCost > RS_EXPECT_TIME_MAX) {
        hccp_warn_socket("socket [%d] connect success cost [%f] ms more than[%f]ms!", conn->connfd, timeCost,
            RS_EXPECT_TIME_MAX);
    } else {
        hccp_info_socket("socket [%d] connect success! cost [%f] ms", conn->connfd, timeCost);
    }

    return;
}
