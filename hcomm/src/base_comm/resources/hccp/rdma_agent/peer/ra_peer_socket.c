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
#include "ra_comm.h"
#include "rs.h"
#include "ra_peer.h"
#include "ra_peer_socket.h"

int RaPeerGetClientSocketErrInfo(unsigned int phyId, struct SocketConnectInfoT conn[], struct SocketErrInfo err[],
    unsigned int num)
{
    struct SocketConnectInfo connOut[MAX_SOCKET_NUM] = {0};
    int ret;

    ret = RaGetSocketConnectInfo(conn, num, connOut, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][ra_peer_socket]ra_get_socket_connect_info failed, "
                 "ret(%d), phyId(%u)",
            ret, phyId),
        ret);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsSocketGetClientSocketErrInfo(connOut, err, num);
    if (ret != 0) {
        hccp_err("[get][ra_peer_socket]ra client get socket info failed, ret(%d)", ret);
    }
    RaPeerMutexUnlock(phyId);

    return ret;
}

int RaPeerGetServerSocketErrInfo(unsigned int phyId, struct SocketListenInfoT conn[], struct ServerSocketErrInfo err[],
    unsigned int num)
{
    struct SocketListenInfo connOut[MAX_SOCKET_NUM] = {0};
    int ret;

    ret = RaGetSocketListenInfo(conn, num, connOut, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][ra_peer_socket]ra_get_socket_listen_info failed "
                 "ret(%d), phyId(%u)",
            ret, phyId),
        ret);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsSocketGetServerSocketErrInfo(connOut, err, num);
    if (ret != 0) {
        hccp_err("[get][ra_peer_socket]ra server get socket info failed, ret(%d)", ret);
    }
    RaPeerMutexUnlock(phyId);

    return ret;
}

int RaPeerSocketAcceptCreditAdd(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num,
    unsigned int creditLimit)
{
    struct SocketListenInfo rsConn[MAX_SOCKET_NUM] = {0};
    int ret;

    ret = RaGetSocketListenInfo(conn, num, rsConn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[set][ra_peer_socket]ra_peer_get_socket_listen_info failed ret(%d) phyId(%u)", ret, phyId), ret);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsSocketAcceptCreditAdd(rsConn, num, creditLimit);
    if (ret == -ENODEV) {
        hccp_warn_socket("[set][ra_peer_socket]rs_socket_accept_credit_add unsuccessful ret(%d) phyId(%u)", ret, phyId);
    } else if (ret != 0) {
        hccp_err("[set][ra_peer_socket]rs_socket_accept_credit_add failed ret(%d) phyId(%u)", ret, phyId);
    }
    RaPeerMutexUnlock(phyId);
    return ret;
}
