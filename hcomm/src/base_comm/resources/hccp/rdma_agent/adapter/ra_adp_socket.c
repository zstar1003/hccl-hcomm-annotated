/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <errno.h>
#include <sys/prctl.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "ra_comm.h"
#include "ra_hdc_socket.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "rs.h"
#include "ra_adp.h"
#include "ra_adp_socket.h"

struct RsSocketOps gSocketOps = {
    .socketBatchConnect = RsSocketBatchConnect,
    .socketBatchClose = RsSocketBatchClose,
    .socketBatchAbort = RsSocketBatchAbort,
    .socketListenStart = RsSocketListenStart,
    .socketListenStop = RsSocketListenStop,
    .getSockets = RsGetSockets,
    .socketSend = RsHdcSocketSend,
    .socketRecv = RsSocketRecv,
    .socketInit = RsSocketInit,
    .socketDeinit = RsSocketDeinit,
    .whiteListAdd = RsSocketWhiteListAdd,
    .whiteListDel = RsSocketWhiteListDel,
    .acceptCreditAdd = RsSocketAcceptCreditAdd,
    .getIfnum = RsGetIfnum,
    .getIfaddrs = RsGetIfaddrs,
    .getIfaddrsV2 = RsGetIfaddrsV2,
    .getVnicIp = RsGetVnicIp,
    .getVnicIpInfos = RsGetVnicIpInfos,
};

int RaRsSocketBatchConnect(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSocketConnectData *socketConnectData = (union OpSocketConnectData *)(inBuf + sizeof(struct MsgHead));
    unsigned int usePort = 0;
    unsigned int i;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketConnectData), sizeof(struct MsgHead), rcvBufLen, opResult);

    // clear resv bit 31 use_port, for compatibility issue
    usePort = socketConnectData->txData.num >> SOCKET_USE_PORT_BIT;
    socketConnectData->txData.num &= ~(1U << SOCKET_USE_PORT_BIT);
    HCCP_CHECK_PARAM_NUM(socketConnectData->txData.num, MAX_SOCKET_NUM);

    for (i = 0; i < (socketConnectData->txData).num; i++) {
        // use_port flag not specify, use default port for compatibility issue
        if (usePort == 0) {
            (socketConnectData->txData).conn[i].port = RS_SOCK_PORT_DEF;
        } else if ((socketConnectData->txData).conn[i].port > MAX_PORT_NUM) {
            hccp_err("[batch_connect]conn[%u].port=%u invalid", i, (socketConnectData->txData).conn[i].port);
            return -EINVAL;
        }
    }

    *opResult = gSocketOps.socketBatchConnect((socketConnectData->txData).conn, (socketConnectData->txData).num);
    if (*opResult != 0) {
        hccp_err("socket batch connect failed ret[%d].", *opResult);
    }

    return 0;
}

int RaRsSocketBatchClose(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSocketCloseData *socketCloseData = (union OpSocketCloseData *)(inBuf + sizeof(struct MsgHead));
    int disuseLinger = 0;
    unsigned int i;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketCloseData), sizeof(struct MsgHead), rcvBufLen, opResult);

    // clear resv bit 31 disuse_linger, for compatibility issue(0 by default)
    disuseLinger = socketCloseData->txData.num >> SOCKET_DISUSE_LINGER_BIT;
    socketCloseData->txData.num &= ~(1U << SOCKET_DISUSE_LINGER_BIT);
    HCCP_CHECK_PARAM_NUM(socketCloseData->txData.num, MAX_SOCKET_NUM);

    struct RsSocketCloseInfoT closeConn[MAX_SOCKET_NUM] = {0};
    for (i = 0; i < socketCloseData->txData.num; i++) {
        closeConn[i].fd = ((socketCloseData->txData).conn[i]).closeFd;
    }
    *opResult = gSocketOps.socketBatchClose(disuseLinger, closeConn, (socketCloseData->txData).num);
    if (*opResult != 0) {
        hccp_err("socket batch close failed ret[%d].", *opResult);
    }

    return 0;
}

int RaRsSocketBatchAbort(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSocketConnectData *socketConnectData = (union OpSocketConnectData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketConnectData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_NUM(socketConnectData->txData.num, MAX_SOCKET_NUM);

    *opResult = gSocketOps.socketBatchAbort((socketConnectData->txData).conn, (socketConnectData->txData).num);
    if (*opResult != 0) {
        hccp_err("socket batch abort failed ret[%d]", *opResult);
    }

    return 0;
}

int RaRsSocketListenStart(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpSocketListenData *socketListenData = (union OpSocketListenData *)(inBuf + sizeof(struct MsgHead));
    union OpSocketListenData *socketListenDataOut = NULL;
    unsigned int usePort = 0;
    unsigned int i;
    int ret;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketListenData), sizeof(struct MsgHead), rcvBufLen, opResult);

    // clear resv bit 31 use_port, for compatibility issue
    usePort = socketListenData->txData.num >> SOCKET_USE_PORT_BIT;
    socketListenData->txData.num &= ~(1U << SOCKET_USE_PORT_BIT);
    HCCP_CHECK_PARAM_LEN_RET_HOST(socketListenData->txData.num, 0, MAX_SOCKET_NUM, opResult);

    for (i = 0; i < (socketListenData->txData).num; i++) {
        // use_port flag not specify, use default port for compatibility issue
        if (usePort == 0) {
            (socketListenData->txData).conn[i].port = RS_SOCK_PORT_DEF;
        } else if ((socketListenData->txData).conn[i].port > MAX_PORT_NUM) {
            hccp_err("[listen_start]conn[%u].port=%u invalid", i, (socketListenData->txData).conn[i].port);
            return -EINVAL;
        }
    }
    *opResult = gSocketOps.socketListenStart((socketListenData->txData).conn, (socketListenData->txData).num);
    if (*opResult == -EADDRINUSE) {
        hccp_warn_socket("socket listen start unsuccessful ret[%d]", *opResult);
        return 0;
    } else if (*opResult != 0) {
        hccp_err("socket listen start failed ret[%d]", *opResult);
        return 0;
    }

    socketListenDataOut = (union OpSocketListenData *)(outBuf + sizeof(struct MsgHead));
    ret = memcpy_s((socketListenDataOut->rxData).conn, sizeof(struct SocketListenInfo) * MAX_SOCKET_NUM,
        (socketListenData->txData).conn, sizeof(struct SocketListenInfo) * (socketListenData->txData).num);
    CHK_PRT_RETURN(ret, hccp_err("memcpy_s socket_listen_info failed, ret[%d]", ret), -ESAFEFUNC);
    return 0;
}

int RaRsSocketListenStop(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSocketListenData *socketListenData = (union OpSocketListenData *)(inBuf + sizeof(struct MsgHead));
    unsigned int usePort = 0;
    unsigned int i;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketListenData), sizeof(struct MsgHead), rcvBufLen, opResult);

    // clear resv bit 31 use_port, for compatibility issue
    usePort = socketListenData->txData.num >> SOCKET_USE_PORT_BIT;
    socketListenData->txData.num &= ~(1U << SOCKET_USE_PORT_BIT);
    HCCP_CHECK_PARAM_LEN_RET_HOST(socketListenData->txData.num, 0, MAX_SOCKET_NUM, opResult);

    for (i = 0; i < (socketListenData->txData).num; i++) {
        // use_port flag not specify, use default port for compatibility issue
        if (usePort == 0) {
            (socketListenData->txData).conn[i].port = RS_SOCK_PORT_DEF;
        } else if ((socketListenData->txData).conn[i].port > MAX_PORT_NUM) {
            hccp_err("[listen_stop]conn[%u].port=%u invalid", i, (socketListenData->txData).conn[i].port);
            return -EINVAL;
        }
    }

    *opResult = gSocketOps.socketListenStop((socketListenData->txData).conn, (socketListenData->txData).num);
    if (*opResult != 0) {
        hccp_err("socket listen stop failed ret[%d].", *opResult);
    }

    return 0;
}

int RaRsGetSockets(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int ret;
    union OpSocketInfoData *socketInfoDataOut = NULL;
    union OpSocketInfoData *socketInfoData = (union OpSocketInfoData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(socketInfoData->txData.num, 0, MAX_SOCKET_NUM, opResult);

    *opResult = gSocketOps.getSockets(socketInfoData->txData.role, socketInfoData->txData.conn,
        socketInfoData->txData.num);
    if (*opResult < 0) {
        hccp_err("socket info get failed ret[%d].", *opResult);
        return 0;
    }

    socketInfoDataOut = (union OpSocketInfoData *)(outBuf + sizeof(struct MsgHead));

    (socketInfoDataOut->rxData).num = *opResult;
    ret = memcpy_s((socketInfoDataOut->rxData).conn, sizeof(struct SocketFdData) * MAX_SOCKET_NUM,
        (socketInfoData->txData).conn, sizeof(struct SocketFdData) * (socketInfoData->txData).num);
    CHK_PRT_RETURN(ret, hccp_err("ra_rs_get_sockets memcpy_s failed, ret[%d]. ", ret), -ESAFEFUNC);

    return 0;
}

int RaRsSocketSend(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int sendLen;
    union OpSocketSendData *sendData = (union OpSocketSendData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketSendData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(sendData->txData.sendSize, 0, SOCKET_SEND_MAXLEN, opResult);

    sendLen = gSocketOps.socketSend(sendData->txData.fd, sendData->txData.dataSend, sendData->txData.sendSize);
    if (sendLen <= 0) {
        if (sendLen == -EAGAIN) {
            hccp_dbg("socket send need retry, ret[%d]", sendLen);
        } else {
            hccp_warn_socket("send unsuccessful, sendLen[%d] expect greater than 0.", sendLen);
        }
    }

    *opResult = sendLen;
    sendData = (union OpSocketSendData *)(outBuf + sizeof(struct MsgHead));
    sendData->rxData.realSendSize = sendLen;

    return 0;
}

int RaRsSocketRecv(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int recvLen;
    union OpSocketRecvData *recvData = (union OpSocketRecvData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketRecvData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN(sizeof(union OpSocketRecvData) + recvData->txData.recvSize, sizeof(struct MsgHead), rcvBufLen);

    recvLen = gSocketOps.socketRecv(recvData->txData.fd,
        outBuf + sizeof(struct MsgHead) + sizeof(union OpSocketRecvData), recvData->txData.recvSize);
    *opResult = recvLen;

    recvData = (union OpSocketRecvData *)(outBuf + sizeof(struct MsgHead));
    recvData->rxData.realRecvSize = recvLen;

    return 0;
}

int RaRsSocketInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSocketInitData *socketInitData = (union OpSocketInitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketInitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gSocketOps.socketInit(socketInitData->txData.vnicIp, socketInitData->txData.num);
    if (*opResult != 0) {
        hccp_err("socket init failed ret[%d].", *opResult);
    }

    return 0;
}

int RaRsSocketDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSocketDeinitData *socketDeinitData = (union OpSocketDeinitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSocketDeinitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gSocketOps.socketDeinit(socketDeinitData->txData.rdevInfo);
    if (*opResult != 0) {
        hccp_err("socket deinit failed ret[%d].", *opResult);
    }

    return 0;
}

int RaRsSocketWhiteListAdd(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpWlistData *wlistData = (union OpWlistData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpWlistData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_NUM(wlistData->txData.num, MAX_WLIST_NUM);

    *opResult = gSocketOps.whiteListAdd(wlistData->txData.rdevInfo, wlistData->txData.wlist, wlistData->txData.num);
    if (*opResult != 0) {
        hccp_err("white_list_add failed, ret[%d]", *opResult);
    }
    return 0;
}

int RaRsSocketWhiteListAddV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpWlistDataV2 *wlistData = (union OpWlistDataV2 *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpWlistDataV2), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_NUM(wlistData->txData.num, MAX_WLIST_NUM);

    *opResult = gSocketOps.whiteListAdd(wlistData->txData.rdevInfo, wlistData->txData.wlist, wlistData->txData.num);
    if (*opResult != 0) {
        hccp_err("white_list_add failed, ret[%d]", *opResult);
    }
    return 0;
}

int RaRsSocketWhiteListDel(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpWlistData *wlistData = (union OpWlistData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpWlistData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_NUM(wlistData->txData.num, MAX_WLIST_NUM);

    *opResult = gSocketOps.whiteListDel(wlistData->txData.rdevInfo, wlistData->txData.wlist, wlistData->txData.num);
    if (*opResult != 0) {
        hccp_err("white_list_del failed, ret[%d]", *opResult);
    }
    return 0;
}

int RaRsSocketWhiteListDelV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpWlistDataV2 *wlistData = (union OpWlistDataV2 *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpWlistDataV2), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_NUM(wlistData->txData.num, MAX_WLIST_NUM);

    *opResult = gSocketOps.whiteListDel(wlistData->txData.rdevInfo, wlistData->txData.wlist, wlistData->txData.num);
    if (*opResult != 0) {
        hccp_err("white_list_del failed, ret[%d]", *opResult);
    }
    return 0;
}

int RaRsSocketCreditAdd(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpAcceptCreditData *opData = (union OpAcceptCreditData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpAcceptCreditData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_NUM(opData->txData.num, MAX_SOCKET_NUM);

    *opResult = gSocketOps.acceptCreditAdd(opData->txData.conn, opData->txData.num, opData->txData.creditLimit);
    if (*opResult != 0) {
        hccp_err("accept_credit_add failed, ret[%d]", *opResult);
    }
    return 0;
}

int RaRsGetIfnum(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpIfnumData *ifnumData = (union OpIfnumData *)(inBuf + sizeof(struct MsgHead));
    union OpIfnumData *ifnumDataOut = NULL;
    unsigned int num = 0;
    bool isAll = false;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpIfnumData), sizeof(struct MsgHead), rcvBufLen, opResult);
    /* resv bit 31 for is_all for compatibility issue */
    if ((ifnumData->txData.num & RA_RS_GET_ALL_IP_BIT_MASK) != 0) {
        isAll = true;
    }
    *opResult = gSocketOps.getIfnum(ifnumData->txData.phyId, isAll, &num);
    if (*opResult != 0) {
        hccp_err("ra_rs_get_ifnum result ret[%d].", *opResult);
        return 0;
    }

    ifnumDataOut = (union OpIfnumData *)(outBuf + sizeof(struct MsgHead));
    (ifnumDataOut->rxData).num = num;

    return 0;
}

int RaRsGetIfaddrs(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int ret;
    union OpIfaddrData *ifaddrDataOut = NULL;
    union OpIfaddrData *ifaddrData = (union OpIfaddrData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpIfaddrData), sizeof(struct MsgHead), rcvBufLen, opResult);
    CHK_PRT_RETURN(ifaddrData->txData.num > MAX_INTERFACE_NUM || ifaddrData->txData.num == 0,
        hccp_err("interface number is invalid, num[%u], valid range[1, %u]", ifaddrData->txData.num, MAX_INTERFACE_NUM),
        -EINVAL);

    *opResult = gSocketOps.getIfaddrs(ifaddrData->txData.ifaddrInfos, &(ifaddrData->txData.num),
        ifaddrData->txData.phyId);
    if (*opResult != 0) {
        hccp_err("ra_rs_get_ifaddrs result ret[%d].", *opResult);
        return 0;
    }

    ifaddrDataOut = (union OpIfaddrData *)(outBuf + sizeof(struct MsgHead));

    (ifaddrDataOut->rxData).num = ifaddrData->txData.num;
    ret = memcpy_s((ifaddrDataOut->rxData).ifaddrInfos, sizeof(struct IfaddrInfo) * MAX_INTERFACE_NUM,
        (ifaddrData->txData).ifaddrInfos, sizeof(struct IfaddrInfo) * (ifaddrData->txData).num);
    CHK_PRT_RETURN(ret, hccp_err("ra_rs_get_sockets memcpy_s failed, ret[%d]. ", ret), -ESAFEFUNC);

    return 0;
}

int RaRsGetIfaddrsV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpIfaddrDataV2 *ifaddrData = (union OpIfaddrDataV2 *)(inBuf + sizeof(struct MsgHead));
    union OpIfaddrDataV2 *ifaddrDataOut = NULL;
    bool isAll = false;
    int ret;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpIfaddrDataV2), sizeof(struct MsgHead), rcvBufLen, opResult);

    /* resv bit 31 for is_all for compatibility issue */
    if ((ifaddrData->txData.num & RA_RS_GET_ALL_IP_BIT_MASK) != 0) {
        isAll = true;
    }
    ifaddrData->txData.num = ifaddrData->txData.num & (~RA_RS_GET_ALL_IP_BIT_MASK);
    CHK_PRT_RETURN(ifaddrData->txData.num > MAX_INTERFACE_NUM || ifaddrData->txData.num == 0,
        hccp_err("interface number of op_ifaddr_data_v2 is invalid, num[%u], valid range[1, %u]",
            ifaddrData->txData.num, MAX_INTERFACE_NUM),
        -EINVAL);

    *opResult = gSocketOps.getIfaddrsV2(ifaddrData->txData.interfaceInfos, &(ifaddrData->txData.num),
        ifaddrData->txData.phyId, isAll);
    if (*opResult != 0) {
        hccp_err("ra_rs_get_ifaddrs_v2 result ret[%d].", *opResult);
        return 0;
    }

    ifaddrDataOut = (union OpIfaddrDataV2 *)(outBuf + sizeof(struct MsgHead));

    (ifaddrDataOut->rxData).num = ifaddrData->txData.num;
    ret = memcpy_s((ifaddrDataOut->rxData).interfaceInfos, sizeof(struct InterfaceInfo) * MAX_INTERFACE_NUM,
        (ifaddrData->txData).interfaceInfos, sizeof(struct InterfaceInfo) * (ifaddrData->txData).num);
    CHK_PRT_RETURN(ret, hccp_err("ra_rs_get_ifaddrs_v2 memcpy_s failed, ret[%d].", ret), -ESAFEFUNC);

    return 0;
}

int RaRsGetVnicIp(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    unsigned int vnicIp = 0;
    union OpGetVnicIpData *vnicIpDataRet = NULL;
    union OpGetVnicIpData *vnicIpData = (union OpGetVnicIpData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetVnicIpData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gSocketOps.getVnicIp(vnicIpData->txData.phyId, &vnicIp);
    if (*opResult != 0) {
        hccp_err("rs get vnic ip failed, phyId %d, ret %d", vnicIpData->txData.phyId, *opResult);
        return 0;
    }

    vnicIpDataRet = (union OpGetVnicIpData *)(outBuf + sizeof(struct MsgHead));
    hccp_info_socket("rs get vnic_ip, phyId %d, vnicIp 0x%x", vnicIpData->txData.phyId, vnicIp);
    vnicIpDataRet->rxData.vnicIp = vnicIp;
    return 0;
}

int RaRsGetVnicIpInfosV1(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetVnicIpInfosDataV1 *vnicIpData = (union OpGetVnicIpInfosDataV1 *)(inBuf + sizeof(struct MsgHead));
    union OpGetVnicIpInfosDataV1 *vnicIpOut = (union OpGetVnicIpInfosDataV1 *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetVnicIpInfosDataV1), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(vnicIpData->txData.num, 0, MAX_IP_INFO_NUM_V1, opResult);

    *opResult = gSocketOps.getVnicIpInfos(vnicIpData->txData.phyId, vnicIpData->txData.type, vnicIpData->txData.ids,
        vnicIpData->txData.num, vnicIpOut->rxData.infos);

    if (*opResult != 0) {
        hccp_err("rs get vnic ip infos failed, ret %d", *opResult);
    }

    return 0;
}

int RaRsGetVnicIpInfos(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetVnicIpInfosData *vnicIpData = (union OpGetVnicIpInfosData *)(inBuf + sizeof(struct MsgHead));
    union OpGetVnicIpInfosData *vnicIpOut = (union OpGetVnicIpInfosData *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetVnicIpInfosData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(vnicIpData->txData.num, 0, MAX_IP_INFO_NUM, opResult);

    *opResult = gSocketOps.getVnicIpInfos(vnicIpData->txData.phyId, vnicIpData->txData.type, vnicIpData->txData.ids,
        vnicIpData->txData.num, vnicIpOut->rxData.infos);

    if (*opResult != 0) {
        hccp_err("rs get vnic ip infos failed, ret %d", *opResult);
    }

    return 0;
}
