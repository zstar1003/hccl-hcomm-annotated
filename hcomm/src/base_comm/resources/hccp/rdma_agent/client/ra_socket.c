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
#include "ra.h"
#include "ra_rs_comm.h"
#include "ra_client_host.h"
#include "hccp.h"

HCCP_ATTRI_VISI_DEF int RaGetClientSocketErrInfo(struct SocketConnectInfoT conn[], struct SocketErrInfo err[],
    unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    char remoteIp[MAX_IP_LEN] = {0};
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || err == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[get][ra_socket]conn is NULL or err is NULL or num[%u] is zero or num is greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        CHK_PRT_RETURN(socketHandle == NULL || socketHandle->socketOps == NULL ||
                           socketHandle->socketOps->raGetClientSocketErrInfo == NULL,
            hccp_err("[get][ra_socket]socket_handle or func is NULL"), ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[get][ra_socket]RaGetClientSocketErrInfo get phyid failed, phyid[%u] should < max[%u]", phyId,
                RA_MAX_PHY_ID_NUM),
            -EINVAL);

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[get][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        ret = RaInetPton(socketHandle->rdevInfo.family, conn[i].remoteIp, remoteIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[get][ra_socket]ra_inet_pton for remote_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_info_rma("Input parameters: [%u]th, phyId[%u], localIp[%s], remoteIp[%s], port[%u], tag[%s]", i, phyId,
            localIp, remoteIp, conn[i].port, conn[i].tag);
    }

    ret = socketHandle->socketOps->raGetClientSocketErrInfo(phyId, conn, err, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetServerSocketErrInfo(struct SocketListenInfoT conn[], struct ServerSocketErrInfo err[],
    unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || err == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[get][ra_socket]conn is NULL or err is NULL or num[%u] is zero or num is greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        CHK_PRT_RETURN(socketHandle == NULL || socketHandle->socketOps == NULL ||
                           socketHandle->socketOps->raGetServerSocketErrInfo == NULL,
            hccp_err("[get][ra_socket]socket_handle or func is NULL"), ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[get][ra_socket]RaGetServerSocketErrInfo get phyid failed, phyid[%u] should < max[%u]", phyId,
                RA_MAX_PHY_ID_NUM),
            -EINVAL);

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket]ra_inet_pton for server_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_info_rma("Input parameters: [%u]th, phyId[%u], localIp[%s], port[%u]", i, phyId, localIp, conn[i].port);
    }

    ret = socketHandle->socketOps->raGetServerSocketErrInfo(phyId, conn, err, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], unsigned int num,
    unsigned int creditLimit)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || num == 0 || num > MAX_SOCKET_NUM || creditLimit == 0,
        hccp_err("[set][ra_socket]conn is NULL or num[%u] is 0 or greater than %d, or creditLimit[%u] is 0", num,
            MAX_SOCKET_NUM, creditLimit),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        CHK_PRT_RETURN(socketHandle == NULL || socketHandle->socketOps == NULL ||
                           socketHandle->socketOps->raSocketAcceptCreditAdd == NULL,
            hccp_err("[set][ra_socket]socket_handle or func is NULL"), ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[set][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[set][ra_socket]ra_inet_pton for server_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s] port[%u] creditLimit[%u]", i, phyId, localIp,
            conn[i].port, creditLimit);
    }

    ret = socketHandle->socketOps->raSocketAcceptCreditAdd(phyId, conn, num, creditLimit);
    return ConverReturnCode(SOCKET_OP, ret);
}
