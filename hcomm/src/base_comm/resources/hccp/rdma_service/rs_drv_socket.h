/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_DRV_SOCKET_H
#define RS_DRV_SOCKET_H

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>

#include "securec.h"
#include "hccp_common.h"
#include "ra_rs_comm.h"
#include "rs.h"
#include "rs_common_inner.h"
#include "rs_inner.h"

#define RS_INTERFACE_LEN 5
#define RS_INTERFACE_BOND_LEN 6
#define RS_INTERFACE_ETH_PREFIX_LEN 3
#define RS_INTERFACE_BOND_PREFIX_LEN 4
/* pcie card boardid rule: GPIO[75:73]=0x000 */
#define RS_BOARDID_PCIE_CARD_MASK 0xE00
#define RS_BOARDID_PCIE_CARD_MASK_VALUE 0x0
#define RS_BOARDID_AI_SERVER_MODULE 0x0
#define RS_BOARDID_ARM_SERVER_AG 0x20
#define RS_BOARDID_ARM_POD 0x30
#define RS_BOARDID_X86_16P 0x50
#define RS_BOARDID_ARM_SERVER_2DIE 0xB0

enum RsHardwareType {
    RS_HARDWARE_SERVER,
    RS_HARDWARE_PCIE,
    RS_HARDWARE_2DIE,
    RS_HARDWARE_UNKNOWN,
};

union RsSocketaddr {
    struct sockaddr_in sAddr;
    struct sockaddr_in6 sAddr6;
};

struct RsSocketaddrInfo {
    int family;
    union RsSocketaddr addr;
};

int RsInetNtop(int family, union HccpIpAddr *ip, char readAddr[], unsigned int len);
int RsConvertIpAddr(int family, union HccpIpAddr *ipAddr, struct RsIpAddrInfo *ip);
bool RsCompareIpAddr(struct RsIpAddrInfo *a, struct RsIpAddrInfo *b);
int RsGetIpv6ScopeId(struct in6_addr localIp);
enum RsHardwareType RsGetDeviceType(unsigned int phyId);
int RsCheckDstInterface(unsigned int phyId, const char *ifaName, enum RsHardwareType type, bool isAll);
int RsPeerFillIfnum(unsigned int phyId, unsigned int *num, struct ifaddrs *ifaddrList);
int RsPeerFillIfaddrInfos(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId,
    struct ifaddrs *ifaddrList);
int RsDrvConnect(int fd, struct RsIpAddrInfo *serverIp, struct RsIpAddrInfo *clientIp, uint16_t port);
int RsDrvSocketSend(int fd, const void *data, uint64_t size, int flags);
int RsDrvSocketRecv(int fd, void *data, uint64_t size, int flags);
int RsDrvSslBindFd(struct RsConnInfo *conn, int fd);
void ShowConnNode(struct RsListHead *listHead);
int RsGetConnInfo(struct RsConnCb *connCb, struct SocketConnectInfo *conn, struct RsConnInfo **connInfo,
    unsigned int serverPort);
int RsFindListenNode(struct RsConnCb *connCb, struct RsIpAddrInfo *ipAddr, uint32_t serverPort,
    struct RsListenInfo **listenInfo);
int RsSocketListenAddToEpoll(struct RsConnCb *connCb, struct RsListenInfo *listenInfo);
int RsListenNodeAlloc(struct RsConnCb *connCb, struct RsIpAddrInfo *ipAddr, uint32_t serverPort,
    struct RsListenInfo **node);
int RsSocketListenDelFromEpoll(struct RsConnCb *connCb, struct RsListenInfo *listenInfo);
void RsListenNodeFree(struct RsConnCb *connCb, struct RsListenInfo *node);
int RsAllocConnNode(struct RsConnInfo **conn, unsigned short serverPort);
int RsFindWhiteListNode(struct RsWhiteList *rsSocketWhiteList, struct SocketWlistInfoT *whiteListExpect, int family,
    struct RsWhiteListInfo **whiteListNode);
int RsFindWhiteList(struct RsConnCb *connCb, struct RsIpAddrInfo *serverIp, struct RsWhiteList **whiteList);
void RsSocketGetBindByChip(unsigned int chipId, bool *bindIp);
bool RsSocketIsVnicIp(unsigned int chipId, unsigned int ipAddr);
void RsConnCostTime(struct RsConnInfo *conn);
int RsFd2conn(int fd, struct RsConnInfo **conn);
int RsSslWriteInnerCheck(struct RsConnInfo *conn, int sslRet, uint64_t size);
int RsSslReadInnerCheck(SSL *ssl, int fd, int sslRet, uint64_t size);
#endif // RS_DRV_SOCKET_H