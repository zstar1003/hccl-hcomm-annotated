/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_ADP_SOCKET_H
#define RA_ADP_SOCKET_H

#include "ra_rs_comm.h"
#include "rs.h"

struct RsSocketOps {
    int (*socketBatchConnect)(struct SocketConnectInfo conn[], unsigned int num);
    int (*socketBatchClose)(int disuseLinger, struct RsSocketCloseInfoT conn[], unsigned int num);
    int (*socketBatchAbort)(struct SocketConnectInfo conn[], unsigned int num);
    int (*socketListenStart)(struct SocketListenInfo conn[], unsigned int num);
    int (*socketListenStop)(struct SocketListenInfo conn[], unsigned int num);
    int (*getSockets)(unsigned int role, struct SocketFdData conn[], unsigned int num);
    int (*socketSend)(int fd, const void *data, uint64_t size);
    int (*socketRecv)(int fd, void *data, uint64_t size);
    int (*socketInit)(const unsigned int *vnicIp, unsigned int num);
    int (*socketDeinit)(struct rdev rdevInfo);
    int (*whiteListAdd)(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
    int (*whiteListDel)(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
    int (*acceptCreditAdd)(struct SocketListenInfo conn[], uint32_t num, unsigned int creditLimit);
    int (*getIfnum)(unsigned int phyId, bool isAll, unsigned int *num);
    int (*getIfaddrs)(struct IfaddrInfo ifaddrInfos[], unsigned int *num, unsigned int phyId);
    int (*getIfaddrsV2)(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId, bool isAll);
    int (*getVnicIp)(unsigned int phyId, unsigned int *vnicIp);
    int (*getVnicIpInfos)(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
        struct IpInfo infos[]);
};

int RaRsSocketBatchConnect(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketBatchClose(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketBatchAbort(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketListenStart(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketListenStop(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetSockets(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketRecv(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketSend(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketWhiteListAdd(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketWhiteListAddV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketWhiteListDel(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketWhiteListDelV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSocketCreditAdd(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetIfnum(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetIfaddrs(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetIfaddrsV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetVnicIp(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetVnicIpInfosV1(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetVnicIpInfos(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
#endif // RA_ADP_SOCKET_H
