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
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include "user_log.h"
#include "config_log.h"
#include "ra_hdc.h"
#include "securec.h"
#include "ra.h"
#include "ra_comm.h"
#include "ra_rs_err.h"
#include "dl_hal_function.h"
#include "ra_hdc_socket.h"

int RaHdcSocketBatchConnect(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num)
{
    union OpSocketConnectData *socketConnectData = NULL;
    unsigned int interfaceVersion = 0;
    int ret;

    socketConnectData = (union OpSocketConnectData *)calloc(sizeof(union OpSocketConnectData), sizeof(char));
    CHK_PRT_RETURN(socketConnectData == NULL,
        hccp_err("[batch_connect][ra_hdc_socket]calloc socket_connect_data "
                 "failed, phyId(%u).",
            phyId),
        -ENOMEM);

    socketConnectData->txData.num = num;

    ret = RaGetSocketConnectInfo(conn, num, socketConnectData->txData.conn, MAX_SOCKET_NUM);
    if (ret != 0) {
        hccp_err("[batch_connect][ra_hdc_socket]ra_get_socket_connect_info failed, ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    // check opcode version, use port by default
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_SOCKET_CONN, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_SOCKET_CONN_VERSION) {
        socketConnectData->txData.num |= (1U << SOCKET_USE_PORT_BIT);
    }

    ret = RaHdcProcessMsg(RA_RS_SOCKET_CONN, phyId, (char *)socketConnectData, sizeof(union OpSocketConnectData));
    if (ret != 0) {
        hccp_err("[batch_connect][ra_hdc_socket]ra hdc message process failed, ret(%d) phyId(%u)", ret, phyId);
    }
out:
    free(socketConnectData);
    socketConnectData = NULL;
    return ret;
}

int RaHdcSocketListenStart(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num)
{
    union OpSocketListenData socketListenData = {0};
    unsigned int interfaceVersion = 0;
    int ret;

    socketListenData.txData.num = num;
    ret = RaGetSocketListenInfo(conn, num, socketListenData.txData.conn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_start][ra_hdc_socket]ra_get_socket_listen_info failed, "
                 "ret(%d) phyId(%u)",
            ret, phyId),
        -EINVAL);

    // check opcode version, use port by default
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_SOCKET_LISTEN_START, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_SOCKET_LISTEN_VERSION) {
        socketListenData.txData.num |= (1U << SOCKET_USE_PORT_BIT);
    }

    ret = RaHdcProcessMsg(RA_RS_SOCKET_LISTEN_START, phyId, (char *)&socketListenData,
        sizeof(union OpSocketListenData));
    CHK_PRT_RETURN(ret == -EADDRINUSE,
        hccp_warn_socket("[listen_start][ra_hdc_socket]ra hdc message process unsuccessful,"
                         " ret(%d) phyId(%u)",
            ret, phyId),
        ret);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_start][ra_hdc_socket]ra hdc message process failed, ret(%d) phyId(%u)", ret, phyId), ret);

    ret = RaGetSocketListenResult(socketListenData.rxData.conn, num, conn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_start][ra_hdc_socket]ra_get_socket_listen_result failed, ret(%d) "
                 "phyId(%u)",
            ret, phyId),
        -EINVAL);

    return ret;
}

int RaHdcSocketBatchClose(unsigned int phyId, struct SocketCloseInfoT conn[], unsigned int num)
{
    union OpSocketCloseData socketCloseData = {0};
    unsigned int interfaceVersion = 0;
    unsigned int i;
    int ret;

    socketCloseData.txData.num = num;
    for (i = 0; i < num; i++) {
        if (conn[i].fdHandle == NULL) {
            hccp_err("[batch_close][ra_hdc_socket]i(%u), conn fdHandle is null", i);
            ret = -EINVAL;
            goto out;
        }
        socketCloseData.txData.conn[i].phyId = phyId;
        socketCloseData.txData.conn[i].closeFd = ((struct SocketHdcInfo *)conn[i].fdHandle)->fd;
    }

    // check opcode version, use RA_RS_GET_SOCKET opcode due to compatibility issue
    // use attr disuse_linger of the fist conn as the common attr for all(0 by default)
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_GET_SOCKET, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_GET_SOCKET_VERSION && conn[0].disuseLinger != 0) {
        socketCloseData.txData.num |= (1U << SOCKET_DISUSE_LINGER_BIT);
    }

    ret = RaHdcProcessMsg(RA_RS_SOCKET_CLOSE, phyId, (char *)&socketCloseData, sizeof(union OpSocketCloseData));
    if (ret) {
        hccp_err("[batch_close][ra_hdc_socket]ra hdc message process failed, ret(%d) phyId(%u).", ret, phyId);
        goto out;
    }

out:
    if (ret != (-EAGAIN)) {
        for (i = 0; i < num; i++) {
            if (conn[i].fdHandle != NULL) {
                free(conn[i].fdHandle);
                conn[i].fdHandle = NULL;
            }
        }
    }
    return ret;
}

int RaHdcSocketBatchAbort(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num)
{
    union OpSocketConnectData *socketConnectData = NULL;
    int ret;

    socketConnectData = (union OpSocketConnectData *)calloc(sizeof(union OpSocketConnectData), sizeof(char));
    CHK_PRT_RETURN(socketConnectData == NULL,
        hccp_err("[batch_abort][ra_hdc_socket]calloc socket_connect_data "
                 "failed. phyId(%u).",
            phyId),
        -ENOMEM);

    socketConnectData->txData.num = num;
    ret = RaGetSocketConnectInfo(conn, num, socketConnectData->txData.conn, MAX_SOCKET_NUM);
    if (ret != 0) {
        hccp_err("[batch_abort][ra_hdc_socket]ra_get_socket_connect_info failed, ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    ret = RaHdcProcessMsg(RA_RS_SOCKET_ABORT, phyId, (char *)socketConnectData, sizeof(union OpSocketConnectData));
    if (ret != 0) {
        hccp_err("[batch_abort][ra_hdc_socket]ra hdc message process failed, ret(%d) phyId(%u)", ret, phyId);
    }
out:
    free(socketConnectData);
    socketConnectData = NULL;
    return ret;
}

int RaHdcSocketListenStop(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num)
{
    union OpSocketListenData socketListenData = {0};
    unsigned int interfaceVersion = 0;
    int ret;

    socketListenData.txData.num = num;

    ret = RaGetSocketListenInfo(conn, num, socketListenData.txData.conn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret,
        hccp_err("[listen_stop][ra_hdc_socket]ra_hdc_socket_listen_stop memcpy_s failed, ret(%d)"
                 " phyId(%u).",
            ret, phyId),
        -EINVAL);

    // check opcode version, use port by default
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_SOCKET_LISTEN_STOP, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_SOCKET_LISTEN_VERSION) {
        socketListenData.txData.num |= (1U << SOCKET_USE_PORT_BIT);
    }

    ret = RaHdcProcessMsg(RA_RS_SOCKET_LISTEN_STOP, phyId, (char *)&socketListenData, sizeof(union OpSocketListenData));
    CHK_PRT_RETURN(ret,
        hccp_err("[listen_stop][ra_hdc_socket]ra hdc message process failed ret(%d) phyId(%u).", ret, phyId), ret);

    return 0;
}

STATIC int RaGetIpAndTagInfo(union OpSocketInfoData *socketInfoData, struct RaSocketHandle *socketHandle,
    struct SocketInfoT conn[], unsigned int index)
{
    int ret;

    ret = memcpy_s(&(socketInfoData->txData.conn[index].localIp), sizeof(union HccpIpAddr),
        &(socketHandle->rdevInfo.localIp), sizeof(union HccpIpAddr));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_ip_and_tag_info]memcpy_s local_ip failed, ret(%d), index(%u)", ret, index),
        -ESAFEFUNC);
    ret = memcpy_s(&(socketInfoData->txData.conn[index].remoteIp), sizeof(union HccpIpAddr), &(conn[index].remoteIp),
        sizeof(union HccpIpAddr));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_ip_and_tag_info]memcpy_s remote_ip failed, ret(%d), index(%u)", ret, index),
        -ESAFEFUNC);
    ret = memcpy_s(socketInfoData->txData.conn[index].tag, sizeof(socketInfoData->txData.conn[index].tag),
        conn[index].tag, sizeof(conn[index].tag));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_ip_and_tag_info]memcpy_s tag failed, ret(%d), index(%u)", ret, index),
        -ESAFEFUNC);

    return 0;
}

STATIC int RaAssembleSockets(union OpSocketInfoData *socketInfoData, struct SocketInfoT *conn, unsigned int num,
    const int sockFd[], size_t sockFdLen)
{
    unsigned int i;
    int ret;
    struct RaSocketHandle *socketHandle = NULL;

    for (i = 0; (i < num) && (i < sockFdLen); i++) {
        if (conn[i].fdHandle == NULL) {
            conn[i].fdHandle = (struct SocketHdcInfo *)calloc(1, sizeof(struct SocketHdcInfo));
            if (conn[i].fdHandle == NULL) {
                hccp_err("[assemble][ra_sockets]fd handle calloc failed.");
                ret = -ENOMEM;
                goto calloc_err;
            }
        }
        ((socketInfoData->txData).conn[i]).fd = sockFd[i];
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        socketInfoData->txData.conn[i].phyId = socketHandle->rdevInfo.phyId;
        socketInfoData->txData.conn[i].family = socketHandle->rdevInfo.family;
        ret = RaGetIpAndTagInfo(socketInfoData, socketHandle, conn, i);
        if (ret) {
            hccp_err("[assemble][ra_sockets]ra_get_ip_and_tag_info failed, ret(%d)", ret);
            ret = -EINVAL;
            goto mem_err;
        }
    }

    return 0;

mem_err:
    i++;
calloc_err:
    for (; i > 0;) {
        i--;
        if (conn[i].fdHandle != NULL) {
            free(conn[i].fdHandle);
            conn[i].fdHandle = NULL;
        }
    }

    return ret;
}

static void FreeAssembleSockets(struct SocketInfoT conn[], unsigned int num)
{
    unsigned int i;

    for (i = 0; i < num; i++) {
        if (conn[i].fdHandle != NULL) {
            free(conn[i].fdHandle);
            conn[i].fdHandle = NULL;
        }
    }
}

int RaHdcSocketSend(unsigned int phyId, const void *handle, const void *data, unsigned long long size)
{
    union OpSocketSendData *sendDataHead = NULL;
    int realSendSize;
    int ret;

    if (size > SOCKET_SEND_MAXLEN) {
        size = SOCKET_SEND_MAXLEN;
    }
    sendDataHead = (union OpSocketSendData *)calloc(sizeof(union OpSocketSendData), sizeof(char));
    CHK_PRT_RETURN(sendDataHead == NULL, hccp_err("[send][ra_hdc_socket]calloc failed, phyId(%u)", phyId), -ENOMEM);

    sendDataHead->txData.fd = (unsigned int)((const struct SocketHdcInfo *)handle)->fd;
    sendDataHead->txData.sendSize = size;

    ret = memcpy_s(sendDataHead->txData.dataSend, SOCKET_SEND_MAXLEN, data, size);
    if (ret) {
        hccp_err("[send][ra_hdc_socket]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId);
        realSendSize = -ESAFEFUNC;
        goto out;
    }

    ret = RaHdcProcessMsg(RA_RS_SOCKET_SEND, phyId, (char *)sendDataHead, sizeof(union OpSocketSendData));
    if (ret) {
        if (ret > 0) {
            ret = -EINVAL; /* 0:success; ret > 0:failed maybe drv interface return; ret < 0:failed maybe rs return */
        }
        if (ret != -EAGAIN) {
            hccp_warn_socket("[send][ra_hdc_socket]ra hdc message process unsuccessful, ret(%d) phyId(%u)", ret, phyId);
        }
        realSendSize = ret;
        goto out;
    }

    realSendSize = (int)sendDataHead->rxData.realSendSize;

out:
    free(sendDataHead);
    sendDataHead = NULL;
    return realSendSize;
}

int RaHdcSocketRecv(unsigned int phyId, const void *handle, void *data, unsigned long long size)
{
    union OpSocketRecvData *recvDataHead = NULL;
    int realRecvSize;
    int ret;

    if (size > SOCKET_SEND_MAXLEN) {
        size = SOCKET_SEND_MAXLEN;
    }

    recvDataHead = (union OpSocketRecvData *)calloc(size + sizeof(union OpSocketRecvData), sizeof(char));
    CHK_PRT_RETURN(recvDataHead == NULL, hccp_err("[recv][ra_hdc_socket]calloc failed. phyId(%u)", phyId), -ENOMEM);
    recvDataHead->txData.fd = (unsigned int)((const struct SocketHdcInfo *)handle)->fd;
    recvDataHead->txData.recvSize = size;

    ret = RaHdcProcessMsg(RA_RS_SOCKET_RECV, phyId, (char *)recvDataHead, sizeof(union OpSocketRecvData) + size);
    if (ret) {
        if (ret > 0) {
            ret = -EINVAL; /* 0:success; ret > 0:failed maybe drv interface return; ret < 0:failed maybe rs return */
        }
        if (ret != -EAGAIN) {
            hccp_warn_socket("[recv][ra_hdc_socket]ra hdc message process unsuccessful, ret(%d) phyId(%u)", ret, phyId);
        }
        realRecvSize = ret;
        goto out;
    }

    realRecvSize = (int)recvDataHead->rxData.realRecvSize;
    if (realRecvSize <= 0) {
        goto out;
    } else {
        ret = memcpy_s(data, size, (char *)recvDataHead + sizeof(union OpSocketRecvData), realRecvSize);
        if (ret) {
            hccp_err("[recv][ra_hdc_socket]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId);
            realRecvSize = -ESAFEFUNC;
            goto out;
        }
    }

out:
    free(recvDataHead);
    recvDataHead = NULL;
    return realRecvSize;
}

STATIC int RaGetRecvSockets(union OpSocketInfoData *socketInfoData, struct SocketInfoT conn[], unsigned int num)
{
    unsigned int i;
    int realNum;
    int ret;
    struct RaSocketHandle *socketHandle = NULL;

    for (i = 0; i < num; i++) {
        ((struct SocketHdcInfo *)conn[i].fdHandle)->phyId = socketInfoData->rxData.conn[i].phyId;
        ((struct SocketHdcInfo *)conn[i].fdHandle)->fd = socketInfoData->rxData.conn[i].fd;
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        ((struct SocketHdcInfo *)conn[i].fdHandle)->socketHandle = socketHandle;
        ret = memcpy_s(&(socketHandle->rdevInfo.localIp), sizeof(union HccpIpAddr),
            &(socketInfoData->rxData.conn[i].localIp), sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_recv_sockets]memcpy_s local_ip failed, ret(%d)", ret), -ESAFEFUNC);
        ret = memcpy_s(&(conn[i].remoteIp), sizeof(union HccpIpAddr), &(socketInfoData->rxData.conn[i].remoteIp),
            sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_recv_sockets]memcpy_s remote_ip failed, ret(%d)", ret), -ESAFEFUNC);
        conn[i].status = socketInfoData->rxData.conn[i].status;
    }

    realNum = socketInfoData->rxData.num;
    return realNum;
}

int RaHdcGetSockets(unsigned int phyId, unsigned int role, struct SocketInfoT conn[], unsigned int num)
{
    int ret;
    int sockFd[MAX_SOCKET_NUM] = {0};
    union OpSocketInfoData *socketInfoData;

    socketInfoData = (union OpSocketInfoData *)calloc(sizeof(union OpSocketInfoData), sizeof(char));
    CHK_PRT_RETURN(socketInfoData == NULL,
        hccp_err("[get][ra_hdc_sockets]socket info data"
                 " calloc failed phyId(%u)",
            phyId),
        -ENOMEM);
    socketInfoData->txData.num = num;
    socketInfoData->txData.role = role;

    ret = RaAssembleSockets(socketInfoData, conn, num, sockFd, sizeof(sockFd) / sizeof(sockFd[0]));
    if (ret) {
        hccp_err("[get][ra_hdc_sockets]assemble sockets error ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    ret = RaHdcProcessMsg(RA_RS_GET_SOCKET, phyId, (char *)socketInfoData, sizeof(union OpSocketInfoData));
    if (ret) {
        hccp_err("[get][ra_hdc_sockets]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        ret = -EINVAL; /* !=0 is error situation return negative value, function normal return >=0 */
        goto err;
    }

    ret = RaGetRecvSockets(socketInfoData, conn, num);
    // no sockets get, free socket info(fd_handle)
    if (ret == 0) {
        goto err;
    }
    free(socketInfoData);
    socketInfoData = NULL;
    return ret;

err:
    FreeAssembleSockets(conn, num);
out:
    free(socketInfoData);
    socketInfoData = NULL;
    return ret;
}

STATIC int RaHdcGetAllVnic(unsigned int curPhyId, unsigned int *vnicIp, unsigned int num)
{
    int ret;
    unsigned int logicId, phyId;
    unsigned int devNum;
    union OpGetVnicIpData vnicIpData;

    ret = DlDrvGetDevNum(&devNum);
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_hdc_all_vnic]get dev num failed, ret(%d)", ret), ret);

    for (logicId = 0; logicId < devNum; logicId++) {
        ret = DlDrvDeviceGetPhyIdByIndex(logicId, &phyId);
        CHK_PRT_RETURN(ret != 0 || phyId >= RA_MAX_PHY_ID_NUM || phyId >= num,
            hccp_err("[get][ra_hdc_all_vnic]get phy"
                     "id failed, logicId(%u) ret(%d) phyId(%u) >= %d or >= %u invalid",
                logicId, ret, phyId, RA_MAX_PHY_ID_NUM, num),
            -ENODEV);

        vnicIpData.txData.phyId = phyId;
        ret = RaHdcProcessMsg(RA_RS_GET_VNIC_IP, curPhyId, (char *)&vnicIpData, sizeof(union OpGetVnicIpData));
        CHK_PRT_RETURN(ret,
            hccp_err("[get][ra_hdc_all_vnic]ra hdc message process failed ret(%d), phyId(%d),"
                     "logicId(%d)",
                ret, phyId, logicId),
            ret);

        vnicIp[phyId] = vnicIpData.rxData.vnicIp;
        hccp_info_socket("vnic ipaddr:0x%x, get vnicIp:0x%x, phyId:%u, logicId:%u", vnicIpData.rxData.vnicIp,
            vnicIp[phyId], phyId, logicId);
    }

    return 0;
}

int RaHdcGetVnicIpInfosV1(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
    struct IpInfo infos[])
{
    union OpGetVnicIpInfosDataV1 vnicIpData = {0};
    unsigned int completeCnt = 0;
    unsigned int sendNum = 0;
    int ret;

    while (completeCnt < num) {
        vnicIpData.txData.phyId = phyId;
        sendNum = ((num - completeCnt) >= MAX_IP_INFO_NUM_V1) ? MAX_IP_INFO_NUM_V1 : (num - completeCnt);
        ret = memcpy_s(vnicIpData.txData.ids, sizeof(unsigned int) * MAX_IP_INFO_NUM_V1, &ids[completeCnt],
            sizeof(unsigned int) * sendNum);
        CHK_PRT_RETURN(ret != 0, hccp_err("[get][ip_infos]memcpy_s for ids failed ret(%d)", ret), -ESAFEFUNC);
        vnicIpData.txData.num = sendNum;
        vnicIpData.txData.type = type;

        ret = RaHdcProcessMsg(RA_RS_GET_VNIC_IP_INFOS_V1, phyId, (char *)&vnicIpData,
            sizeof(union OpGetVnicIpInfosDataV1));
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[get][ip_infos]ra hdc message process failed ret(%d), phyId(%u)", ret, phyId), ret);

        ret = memcpy_s(&infos[completeCnt], sizeof(struct IpInfo) * (num - completeCnt), vnicIpData.rxData.infos,
            sizeof(struct IpInfo) * sendNum);
        CHK_PRT_RETURN(ret != 0, hccp_err("[get][ip_infos]memcpy_s for ids failed ret(%d)", ret), -ESAFEFUNC);
        completeCnt += sendNum;
    }

    return 0;
}

int RaHdcGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
    struct IpInfo infos[])
{
    union OpGetVnicIpInfosData vnicIpData = {0};
    unsigned int interfaceVersion = 0;
    unsigned int completeCnt = 0;
    unsigned int sendNum = 0;
    int ret;

    // origin procedure for compatibility issue
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_GET_VNIC_IP_INFOS, &interfaceVersion);
    if (ret != 0 || interfaceVersion < RA_RS_GET_VNIC_IP_INFOS_VERSION) {
        return RaHdcGetVnicIpInfosV1(phyId, type, ids, num, infos);
    }

    while (completeCnt < num) {
        vnicIpData.txData.phyId = phyId;
        sendNum = ((num - completeCnt) >= MAX_IP_INFO_NUM) ? MAX_IP_INFO_NUM : (num - completeCnt);
        ret = memcpy_s(vnicIpData.txData.ids, sizeof(unsigned int) * MAX_IP_INFO_NUM, &ids[completeCnt],
            sizeof(unsigned int) * sendNum);
        CHK_PRT_RETURN(ret != 0, hccp_err("[get][ip_infos]memcpy_s for ids failed ret(%d)", ret), -ESAFEFUNC);
        vnicIpData.txData.num = sendNum;
        vnicIpData.txData.type = type;

        ret = RaHdcProcessMsg(RA_RS_GET_VNIC_IP_INFOS, phyId, (char *)&vnicIpData, sizeof(union OpGetVnicIpInfosData));
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[get][ip_infos]ra hdc message process failed ret(%d), phyId(%u)", ret, phyId), ret);

        ret = memcpy_s(&infos[completeCnt], sizeof(struct IpInfo) * (num - completeCnt), vnicIpData.rxData.infos,
            sizeof(struct IpInfo) * sendNum);
        CHK_PRT_RETURN(ret != 0, hccp_err("[get][ip_infos]memcpy_s for ids failed ret(%d)", ret), -ESAFEFUNC);
        completeCnt += sendNum;
    }

    return 0;
}

int RaHdcSocketInit(struct rdev rdevInfo)
{
    unsigned int vnicIp[RA_MAX_VNIC_NUM] = {0};
    union OpSocketInitData socketInitData;
    unsigned int interfaceVersion = 0;
    int ret;

    ret = memset_s(&socketInitData, sizeof(socketInitData), 0, sizeof(socketInitData));
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_socket]memset_s failed ret(%d) phyId(%u)", ret, rdevInfo.phyId),
        -ESAFEFUNC);

    // check opcode version, init g_vnics with invalid ip mask 0xFFFFFFFF
    ret = RaHdcGetInterfaceVersion(rdevInfo.phyId, RA_RS_GET_VNIC_IP_INFOS_V1, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_GET_VNIC_IP_INFOS_VERSION) {
        ret = memset_s(vnicIp, sizeof(vnicIp), 0xFFFFFFFF, sizeof(vnicIp));
        CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_socket]memset_s failed ret(%d) phyId(%u)", ret, rdevInfo.phyId),
            -ESAFEFUNC);
    } else {
        // origin procedure: init g_vnics with vnic_ip get by phyId
        ret = RaHdcGetAllVnic(rdevInfo.phyId, vnicIp, RA_MAX_VNIC_NUM);
        CHK_PRT_RETURN(ret,
            hccp_err("[init][ra_hdc_socket]ra_hdc_get_all_vnic failed ret(%d) phyId(%u)", ret, rdevInfo.phyId), ret);
    }

    ret = memcpy_s(&(socketInitData.txData.vnicIp), sizeof(socketInitData.txData.vnicIp), &vnicIp, sizeof(vnicIp));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_socket]memcpy_s for vnic_ip failed ret(%d) phyId(%u)", ret, rdevInfo.phyId),
        -ESAFEFUNC);

    socketInitData.txData.num = RA_MAX_VNIC_NUM;
    ret = RaHdcProcessMsg(RA_RS_SOCKET_INIT, rdevInfo.phyId, (char *)&socketInitData, sizeof(union OpSocketInitData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_socket]ra hdc message process failed ret(%d) phyId(%u)", ret, rdevInfo.phyId), ret);

    return 0;
}

int RaHdcSocketDeinit(struct rdev rdevInfo)
{
    int ret;
    union OpSocketDeinitData socketDeinitData;

    ret = memset_s(&socketDeinitData, sizeof(socketDeinitData), 0, sizeof(socketDeinitData));
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_hdc_socket]memset_s failed ret(%d) phyId(%u)", ret, rdevInfo.phyId),
        -ESAFEFUNC);

    ret = memcpy_s(&(socketDeinitData.txData.rdevInfo), sizeof(struct rdev), &rdevInfo, sizeof(struct rdev));
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_hdc_socket]memcpy_s failed ret(%d) phyId(%u)", ret, rdevInfo.phyId),
        -ESAFEFUNC);

    ret = RaHdcProcessMsg(RA_RS_SOCKET_DEINIT, rdevInfo.phyId, (char *)&socketDeinitData,
        sizeof(union OpSocketDeinitData));
    CHK_PRT_RETURN(ret,
        hccp_err("[deinit][ra_hdc_socket]ra hdc message process failed ret(%d) phyId(%u)", ret, rdevInfo.phyId), ret);

    return 0;
}

int RaHdcGetIfnum(unsigned int phyId, bool isAll, unsigned int *num)
{
    union OpIfnumData ifnumData = {0};
    int ret;

    ifnumData.txData.num = isAll ? RA_RS_GET_ALL_IP_BIT_MASK : 0;
    ifnumData.txData.phyId = phyId;
    ret = RaHdcProcessMsg(RA_RS_GET_IFNUM, phyId, (char *)&ifnumData, sizeof(union OpIfnumData));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_hdc_ifnum]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    *num = ifnumData.rxData.num;

    return 0;
}

int RaHdcGetIfaddrs(unsigned int phyId, struct IfaddrInfo ifaddrInfos[], unsigned int *num)
{
    union OpIfaddrData ifaddrData = {0};
    int ret;

    ifaddrData.txData.num = *num;
    ret = memcpy_s(ifaddrData.txData.ifaddrInfos, sizeof(struct IfaddrInfo) * MAX_INTERFACE_NUM, ifaddrInfos,
        sizeof(struct IfaddrInfo) * (*num));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_hdc_ifaddrs]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId), -ESAFEFUNC);

    ifaddrData.txData.phyId = phyId;
    ret = RaHdcProcessMsg(RA_RS_GET_IFADDRS, phyId, (char *)&ifaddrData, sizeof(union OpIfaddrData));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_hdc_ifaddrs]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    ret = memcpy_s(ifaddrInfos, sizeof(struct IfaddrInfo) * (*num), ifaddrData.rxData.ifaddrInfos,
        sizeof(struct IfaddrInfo) * (ifaddrData.rxData.num));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_hdc_ifaddrs]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId), -ESAFEFUNC);

    *num = ifaddrData.rxData.num;
    return 0;
}

int RaHdcGetIfaddrsV2(unsigned int phyId, bool isAll, struct InterfaceInfo interfaceInfos[], unsigned int *num)
{
    union OpIfaddrDataV2 ifaddrData = {0};
    int ret;

    ifaddrData.txData.num = isAll ? (*num | RA_RS_GET_ALL_IP_BIT_MASK) : *num;
    ret = memcpy_s(ifaddrData.txData.interfaceInfos, sizeof(struct InterfaceInfo) * MAX_INTERFACE_NUM, interfaceInfos,
        sizeof(struct InterfaceInfo) * (*num));
    CHK_PRT_RETURN(ret,
        hccp_err("[get][ra_hdc_ifaddrs_v2]memcpy_s tx interface infos failed, ret(%d) phyId(%u)", ret, phyId),
        -ESAFEFUNC);

    ifaddrData.txData.phyId = phyId;
    ret = RaHdcProcessMsg(RA_RS_GET_IFADDRS_V2, phyId, (char *)&ifaddrData, sizeof(union OpIfaddrDataV2));
    CHK_PRT_RETURN(ret, hccp_err("[get][ra_hdc_ifaddrs_v2]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    ret = memcpy_s(interfaceInfos, sizeof(struct InterfaceInfo) * (*num), ifaddrData.rxData.interfaceInfos,
        sizeof(struct InterfaceInfo) * (ifaddrData.rxData.num));
    CHK_PRT_RETURN(ret,
        hccp_err("[get][ra_hdc_ifaddrs_v2]memcpy_s rx interface infos failed, ret(%d) phyId(%u)", ret, phyId),
        -ESAFEFUNC);

    *num = ifaddrData.rxData.num;
    return 0;
}

static int RaHdcSocketWhiteListOpV1(unsigned int opcode, struct rdev rdevInfo, struct SocketWlistInfoT whiteList[],
    unsigned int num)
{
    union OpWlistData *wlistData = NULL;
    int ret;
    wlistData = (union OpWlistData *)calloc(sizeof(union OpWlistData), sizeof(char));
    CHK_PRT_RETURN(wlistData == NULL,
        hccp_err("[op][ra_hdc_socket_white_list]calloc wlist data failed! phyId(%u)", rdevInfo.phyId), -ENOMEM);
    wlistData->txData.num = num;
    ret = memcpy_s(&(wlistData->txData.rdevInfo), sizeof(struct rdev), &(rdevInfo), sizeof(struct rdev));
    if (ret) {
        hccp_err("[op][ra_hdc_socket_white_list]memcpy_s for rdev_info failed, ret(%d) phyId(%u)", ret, rdevInfo.phyId);
        ret = -ESAFEFUNC;
        goto out;
    }

    ret = memcpy_s(wlistData->txData.wlist, sizeof(struct SocketWlistInfoT) * MAX_WLIST_NUM_V1, whiteList,
        sizeof(struct SocketWlistInfoT) * num);
    if (ret) {
        hccp_err("[op][ra_hdc_socket_white_list]memcpy_s for wlist failed, ret(%d) phyId(%u)", ret, rdevInfo.phyId);
        ret = -ESAFEFUNC;
        goto out;
    }

    ret = RaHdcProcessMsg(opcode, rdevInfo.phyId, (char *)wlistData, sizeof(union OpWlistData));
    if (ret) {
        hccp_err("[op][ra_hdc_socket_white_list]ra hdc process msg failed ret(%d) phyId(%u)", ret, rdevInfo.phyId);
        goto out;
    }

out:
    free(wlistData);
    wlistData = NULL;
    return ret;
}

static int RaHdcSocketWhiteListOpV2(unsigned int opcode, struct rdev rdevInfo, struct SocketWlistInfoT whiteList[],
    unsigned int num)
{
    int ret;
    union OpWlistDataV2 *wlistData = NULL;

    wlistData = (union OpWlistDataV2 *)calloc(sizeof(union OpWlistDataV2), sizeof(char));
    CHK_PRT_RETURN(wlistData == NULL,
        hccp_err("[op][ra_hdc_socket_white_list]calloc wlist data failed! phyId(%u)", rdevInfo.phyId), -ENOMEM);
    wlistData->txData.num = num;
    ret = memcpy_s(&(wlistData->txData.rdevInfo), sizeof(struct rdev), &(rdevInfo), sizeof(struct rdev));
    if (ret) {
        hccp_err("[op][ra_hdc_socket_white_list]memcpy_s for rdev_info failed, ret(%d) phyId(%u)", ret, rdevInfo.phyId);
        ret = -ESAFEFUNC;
        goto out;
    }

    ret = memcpy_s(wlistData->txData.wlist, sizeof(struct SocketWlistInfoT) * MAX_WLIST_NUM, whiteList,
        sizeof(struct SocketWlistInfoT) * num);
    if (ret) {
        hccp_err("[op][ra_hdc_socket_white_list]memcpy_s for wlist failed, ret(%d) phyId(%u)", ret, rdevInfo.phyId);
        ret = -ESAFEFUNC;
        goto out;
    }

    ret = RaHdcProcessMsg(opcode, rdevInfo.phyId, (char *)wlistData, sizeof(union OpWlistDataV2));
    if (ret) {
        hccp_err("[op][ra_hdc_socket_white_list]ra hdc process msg failed ret(%d) phyId(%u)", ret, rdevInfo.phyId);
        goto out;
    }

out:
    free(wlistData);
    wlistData = NULL;
    return ret;
}

int RaHdcSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    int ret;
    unsigned int interfaceVersion = 0;

    ret = RaHdcGetInterfaceVersion(rdevInfo.phyId, RA_RS_WLIST_ADD_V2, &interfaceVersion);
    if (ret != 0 || interfaceVersion != RA_RS_WLIST_ADD_V2_VERSION) {
        return RaHdcSocketWhiteListOpV1(RA_RS_WLIST_ADD, rdevInfo, whiteList, num);
    }

    return RaHdcSocketWhiteListOpV2(RA_RS_WLIST_ADD_V2, rdevInfo, whiteList, num);
}

int RaHdcSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    int ret;
    unsigned int interfaceVersion = 0;

    ret = RaHdcGetInterfaceVersion(rdevInfo.phyId, RA_RS_WLIST_DEL_V2, &interfaceVersion);
    if (ret != 0 || interfaceVersion != RA_RS_WLIST_DEL_V2_VERSION) {
        return RaHdcSocketWhiteListOpV1(RA_RS_WLIST_DEL, rdevInfo, whiteList, num);
    }

    return RaHdcSocketWhiteListOpV2(RA_RS_WLIST_DEL_V2, rdevInfo, whiteList, num);
}

int RaHdcSocketAcceptCreditAdd(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num,
    unsigned int creditLimit)
{
    union OpAcceptCreditData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.creditLimit = creditLimit;
    opData.txData.num = num;
    ret = RaGetSocketListenInfo(conn, num, opData.txData.conn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[set][ra_hdc_socket]ra_get_socket_listen_info failed, ret(%d) phyId(%u)", ret, phyId), -EINVAL);

    ret = RaHdcProcessMsg(RA_RS_ACCEPT_CREDIT_ADD, phyId, (char *)&opData, sizeof(union OpAcceptCreditData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[set][ra_hdc_socket]ra hdc message process failed, ret(%d) phyId(%u)", ret, phyId), ret);

    return ret;
}
