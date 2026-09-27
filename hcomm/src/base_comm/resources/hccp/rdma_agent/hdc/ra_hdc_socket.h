/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_SOCKET_H
#define RA_HDC_SOCKET_H
#include "ascend_hal.h"
#include "hccp_common.h"
#include "ra.h"
#include "ra_hdc.h"
#include "ra_rs_comm.h"

#define VNIC_IP_TYPE 0
#define RA_MAX_VNIC_NUM 16

struct CloseFdData {
    unsigned int phyId;
    int closeFd;
};

union OpSocketInitData {
    struct {
        unsigned int vnicIp[RA_MAX_VNIC_NUM];
        unsigned int num;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpSocketDeinitData {
    struct {
        struct rdev rdevInfo;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpSocketConnectData {
    struct {
        unsigned int num; // resv bit 31 for use_port on HDC, for compatibility issue
        struct SocketConnectInfo conn[MAX_SOCKET_NUM];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_801];
    } rxData;
};

union OpSocketCloseData {
    struct {
        unsigned int num; // resv bit 31 for disuse_linger on HDC, for compatibility issue
        struct CloseFdData conn[MAX_SOCKET_NUM];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_33];
    } rxData;
};

union OpSocketListenData {
    struct {
        unsigned int phyId;
        unsigned int num; // resv bit 31 for use_port on HDC, for compatibility issue
        struct SocketListenInfo conn[MAX_SOCKET_NUM];
    } txData;

    struct {
        unsigned int rsvd;
        struct SocketListenInfo conn[MAX_SOCKET_NUM];
    } rxData;
};

union OpSocketInfoData {
    struct {
        unsigned int num;
        unsigned int role;
        struct SocketFdData conn[MAX_SOCKET_NUM];
    } txData;

    struct {
        int num;
        unsigned int rsvd;
        struct SocketFdData conn[MAX_SOCKET_NUM];
    } rxData;
};

union OpSocketSendData {
    struct {
        unsigned int fd;
        unsigned int rsvd;
        unsigned long long sendSize;
        char dataSend[SOCKET_SEND_MAXLEN];
    } txData;

    struct {
        unsigned long long realSendSize;
        unsigned int rsvd[RA_RSVD_NUM_2];
    } rxData;
};

union OpSocketRecvData {
    struct {
        unsigned int fd;
        unsigned int rsvd;
        unsigned long long recvSize;
    } txData;

    struct {
        unsigned long long realRecvSize;
        unsigned int rsvd[RA_RSVD_NUM_2];
    } rxData;
};

union OpWlistData {
    struct {
        struct rdev rdevInfo;
        struct SocketWlistInfoT wlist[MAX_WLIST_NUM_V1];
        unsigned int num;
    } txData;

    struct {
    } rxData;
};

union OpWlistDataV2 {
    struct {
        struct rdev rdevInfo;
        unsigned int num;
        struct SocketWlistInfoT wlist[MAX_WLIST_NUM];
    } txData;

    struct {
    } rxData;
};

union OpAcceptCreditData {
    struct {
        unsigned int phyId;
        unsigned int creditLimit;
        unsigned int num;
        struct SocketListenInfo conn[MAX_SOCKET_NUM];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_64];
    } rxData;
};

union OpIfaddrData {
    struct {
        unsigned int phyId;
        struct IfaddrInfo ifaddrInfos[MAX_INTERFACE_NUM];
        unsigned int num;
    } txData;

    struct {
        struct IfaddrInfo ifaddrInfos[MAX_INTERFACE_NUM];
        unsigned int num;
    } rxData;
};

// support IPV4/IPV6
union OpIfaddrDataV2 {
    struct {
        unsigned int phyId;
        struct InterfaceInfo interfaceInfos[MAX_INTERFACE_NUM];
        unsigned int num;
    } txData;

    struct {
        struct InterfaceInfo interfaceInfos[MAX_INTERFACE_NUM];
        unsigned int num;
    } rxData;
};

union OpGetVnicIpData {
    struct {
        unsigned int phyId;
    } txData;

    struct {
        unsigned int vnicIp;
    } rxData;
};

union OpGetVnicIpInfosDataV1 {
    struct {
        unsigned int phyId;
        enum IdType type;
        unsigned int ids[MAX_IP_INFO_NUM_V1];
        unsigned int num;
        unsigned int rsv;
    } txData;

    struct {
        struct IpInfo infos[MAX_IP_INFO_NUM_V1];
        unsigned int rsv;
    } rxData;
};

union OpGetVnicIpInfosData {
    struct {
        unsigned int phyId;
        enum IdType type;
        unsigned int ids[MAX_IP_INFO_NUM];
        unsigned int num;
        unsigned int rsv;
    } txData;

    struct {
        struct IpInfo infos[MAX_IP_INFO_NUM];
        unsigned int rsv;
    } rxData;
};

int RaHdcSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
int RaHdcSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
int RaHdcSocketAcceptCreditAdd(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num,
    unsigned int creditLimit);
int RaHdcSocketBatchConnect(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num);
int RaHdcSocketBatchClose(unsigned int phyId, struct SocketCloseInfoT conn[], unsigned int num);
int RaHdcSocketBatchAbort(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num);
int RaHdcSocketListenStart(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num);
int RaHdcSocketListenStop(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num);
int RaHdcGetSockets(unsigned int phyId, unsigned int role, struct SocketInfoT conn[], unsigned int num);
int RaHdcSocketSend(unsigned int phyId, const void *handle, const void *data, unsigned long long size);
int RaHdcSocketRecv(unsigned int phyId, const void *handle, void *data, unsigned long long size);
int RaHdcSocketInit(struct rdev rdevInfo);
int RaHdcSocketDeinit(struct rdev rdevInfo);
int RaHdcGetIfnum(unsigned int phyId, bool isAll, unsigned int *num);
int RaHdcGetIfaddrs(unsigned int phyId, struct IfaddrInfo ifaddrInfos[], unsigned int *num);
int RaHdcGetIfaddrsV2(unsigned int phyId, bool isAll, struct InterfaceInfo interfaceInfos[], unsigned int *num);
int RaHdcGetVnicIpInfosV1(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
    struct IpInfo infos[]);
int RaHdcGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
    struct IpInfo infos[]);
#endif // RA_HDC_SOCKET_H
