/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ra_comm.h"
#include <errno.h>
#include "securec.h"
#include "ra_rs_err.h"

int RaGetSocketConnectInfo(const struct SocketConnectInfoT conn[], unsigned int num, struct SocketConnectInfo rsConn[],
    unsigned int rsNum)
{
    struct RaSocketHandle *socketHandle = NULL;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(num > rsNum || num > MAX_SOCKET_NUM || num == 0 || conn == NULL || rsConn == NULL,
        hccp_err("[get]"
                 "[ra_socket_connect_info]num(%u) > rs_num(%u) or num == 0 or conn or rs_conn is NULL, invalid",
            num, rsNum),
        -EINVAL);
    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        CHK_PRT_RETURN(socketHandle == NULL, hccp_err("[get][ra_socket_connect_info] conn[%u].socketHandle is null", i),
            -EINVAL);
        rsConn[i].phyId = socketHandle->rdevInfo.phyId;
        rsConn[i].family = socketHandle->rdevInfo.family;
        rsConn[i].port = conn[i].port;
        ret = memcpy_s(&(rsConn[i].localIp), sizeof(union HccpIpAddr), &(socketHandle->rdevInfo.localIp),
            sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket_connect_info]memcpy_s for local_ip failed, ret(%d)", ret),
            -ESAFEFUNC);
        ret = memcpy_s(&(rsConn[i].remoteIp), sizeof(union HccpIpAddr), &(conn[i].remoteIp), sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket_connect_info]memcpy_s for remote_ip failed, ret(%d)", ret),
            -ESAFEFUNC);
        ret = memcpy_s(rsConn[i].tag, SOCK_CONN_TAG_SIZE, conn[i].tag, SOCK_CONN_TAG_SIZE);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket_connect_info]memcpy_s for tag failed, ret(%d)", ret), -ESAFEFUNC);
    }
    return 0;
}

int RaGetSocketListenInfo(const struct SocketListenInfoT conn[], unsigned int num, struct SocketListenInfo rsConn[],
    unsigned int rsNum)
{
    unsigned int i;
    int ret;
    struct RaSocketHandle *socketHandle = NULL;

    CHK_PRT_RETURN(num > rsNum || num > MAX_SOCKET_NUM || num == 0 || conn == NULL || rsConn == NULL,
        hccp_err("[get]"
                 "[ra_socket_listen_info]num(%u) > rs_num(%u) or num == 0 or conn or rsConn is NULL, invalid",
            num, rsNum),
        -EINVAL);

    for (i = 0; i < num; i++) {
        rsConn[i].phase = conn[i].phase;
        rsConn[i].err = conn[i].err;
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        rsConn[i].phyId = socketHandle->rdevInfo.phyId;
        rsConn[i].family = socketHandle->rdevInfo.family;
        rsConn[i].port = conn[i].port;
        ret = memcpy_s(&(rsConn[i].localIp), sizeof(union HccpIpAddr), &(socketHandle->rdevInfo.localIp),
            sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket_listen_info]memcpy_s for local_ip failed, ret(%d)", ret),
            -ESAFEFUNC);
    }
    return 0;
}

int RaGetSocketListenResult(const struct SocketListenInfo rsConn[], unsigned int rsNum, struct SocketListenInfoT conn[],
    unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(rsNum > num || rsNum > MAX_SOCKET_NUM || num == 0 || conn == NULL || rsConn == NULL,
        hccp_err("[get]"
                 "[ra_socket_listen_result]rs_num(%u) > num(%u) or num == 0 or conn or rs_conn is NULL, invalid",
            rsNum, num),
        -EINVAL);

    for (i = 0; i < rsNum; i++) {
        conn[i].phase = rsConn[i].phase;
        conn[i].err = rsConn[i].err;
        conn[i].port = rsConn[i].port;
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        socketHandle->rdevInfo.phyId = rsConn[i].phyId;
        socketHandle->rdevInfo.family = rsConn[i].family;
        ret = memcpy_s(&(socketHandle->rdevInfo.localIp), sizeof(union HccpIpAddr), &(rsConn[i].localIp),
            sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket_listen_result]memcpy_s for local_ip failed, ret(%d)", ret),
            -ESAFEFUNC);
    }
    return 0;
}
