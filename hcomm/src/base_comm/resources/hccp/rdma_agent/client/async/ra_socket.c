/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_log.h"
#include "config_log.h"
#include "hccp_common.h"
#include "hccp_async.h"
#include "ra.h"
#include "ra_hdc_async_socket.h"
#include "ra_client_host.h"

HCCP_ATTRI_VISI_DEF int RaSocketBatchConnectAsync(struct SocketConnectInfoT conn[], unsigned int num, void **reqHandle)
{
    struct RaSocketHandle *socketHandle = NULL;
    char remoteIp[MAX_IP_LEN] = {0};
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i = 0;
    int ret = 0;

    CHK_PRT_RETURN(conn == NULL || reqHandle == NULL,
        hccp_err("[batch_connect][ra_socket]conn or "
                 "req_handle is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    CHK_PRT_RETURN(num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[batch_connect][ra_socket]num[%u] invalid, "
                 "must in range of (0, %u]",
            num, MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)(conn[i].socketHandle);
        CHK_PRT_RETURN(socketHandle == NULL, hccp_err("[batch_connect][ra_socket]socket_handle is NULL"),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[batch_connect][ra_socket]phyId[%u] invalid, "
                     "must in range of [0, %u)",
                phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        CHK_PRT_RETURN(strlen(conn[i].tag) >= SOCK_CONN_TAG_SIZE,
            hccp_err("[batch_connect][ra_socket]conn tag len(%d) more than max len(%u)", strlen(conn[i].tag),
                SOCK_CONN_TAG_SIZE),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[batch_connect][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        ret = RaInetPton(socketHandle->rdevInfo.family, conn[i].remoteIp, remoteIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[batch_connect][ra_socket]ra_inet_pton for remote_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s], remoteIp[%s], port[%u], tag[%s], cnt[%u]", i,
            socketHandle->rdevInfo.phyId, localIp, remoteIp, conn[i].port, conn[i].tag, socketHandle->connectCnt);
    }

    socketHandle->connectCnt++;
    ret = RaHdcSocketBatchConnectAsync(phyId, conn, num, reqHandle);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketListenStartAsync(struct SocketListenInfoT conn[], unsigned int num, void **reqHandle)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i = 0;
    int ret = 0;

    CHK_PRT_RETURN(conn == NULL || reqHandle == NULL, hccp_err("[listen_start][ra_socket]conn or req_handle is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    CHK_PRT_RETURN(num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[listen_start][ra_socket]num[%u] invalid, "
                 "must in range of (0, %u]",
            num, MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)(conn[i].socketHandle);
        CHK_PRT_RETURN(socketHandle == NULL, hccp_err("[listen_start][ra_socket]socket_handle is NULL"),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[listen_start][ra_socket]phyId[%u] invalid, "
                     "must in range of [0, %u)",
                phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[listen_start][ra_socket]ra_inet_pton for server_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s], port[%u]", i, phyId, localIp, conn[i].port);
    }

    ret = RaHdcSocketListenStartAsync(phyId, conn, num, reqHandle);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketListenStopAsync(struct SocketListenInfoT conn[], unsigned int num, void **reqHandle)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i = 0;
    int ret = 0;

    CHK_PRT_RETURN(conn == NULL || reqHandle == NULL, hccp_err("[listen_stop][ra_socket]conn or req_handle is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    CHK_PRT_RETURN(num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[listen_stop][ra_socket]num[%u] invalid, "
                 "must in range of (0, %u]",
            num, MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)(conn[i].socketHandle);
        CHK_PRT_RETURN(socketHandle == NULL, hccp_err("[listen_stop][ra_socket]socket_handle is NULL"),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[listen_stop][ra_socket]phyId[%u] invalid, "
                     "must in range of [0, %u)",
                phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[listen_stop][ra_socket]ra_inet_pton for server_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s]", i, phyId, localIp);
    }

    ret = RaHdcSocketListenStopAsync(phyId, conn, num, reqHandle);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketBatchCloseAsync(struct SocketCloseInfoT conn[], unsigned int num, void **reqHandle)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i = 0;
    int ret = 0;

    CHK_PRT_RETURN(conn == NULL || reqHandle == NULL, hccp_err("[batch_close][ra_socket]conn or req_handle is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    CHK_PRT_RETURN(num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[batch_close][ra_socket]num[%u] invalid, "
                 "must in range of (0, %u]",
            num, MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)(conn[i].socketHandle);
        CHK_PRT_RETURN(socketHandle == NULL, hccp_err("[batch_close][ra_socket]socket_handle is NULL"),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[batch_close][ra_socket]phyId[%u] invalid, "
                     "must in range of [0, %u)",
                phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[batch_close][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s], cnt[%u]", i, phyId, localIp,
            socketHandle->closeCnt);
    }

    socketHandle->closeCnt++;
    ret = RaHdcSocketBatchCloseAsync(phyId, conn, num, reqHandle);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketSendAsync(const void *fdHandle, const void *data, unsigned long long size,
    unsigned long long *sentSize, void **reqHandle)
{
    int ret = 0;

    CHK_PRT_RETURN(fdHandle == NULL || data == NULL || sentSize == NULL || size == 0 || reqHandle == NULL,
        hccp_err("[send][ra_socket]fd_handle or data or sent_size or req_handle is NULL or size[%llu] is 0", size),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    ret = RaHdcSocketSendAsync((const struct SocketHdcInfo *)fdHandle, data, size, sentSize, reqHandle);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketRecvAsync(const void *fdHandle, void *data, unsigned long long size,
    unsigned long long *receivedSize, void **reqHandle)
{
    int ret = 0;

    CHK_PRT_RETURN(fdHandle == NULL || data == NULL || receivedSize == NULL || size == 0 || reqHandle == NULL,
        hccp_err("[recv][ra_socket]fd_handle or data or received_size or req_handle is NULL or size[%llu] is 0", size),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    ret = RaHdcSocketRecvAsync((const struct SocketHdcInfo *)fdHandle, data, size, receivedSize, reqHandle);
    return ConverReturnCode(SOCKET_OP, ret);
}
