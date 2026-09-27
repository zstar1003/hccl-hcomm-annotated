/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "config_log.h"
#include "ra_comm.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "rs.h"
#include "ra_peer_nda.h"
#include "ra_peer.h"

#define PAGE_SHIFT 12
int gNotifyFd = -1;

static pthread_mutex_t gRaPeerMutex[RA_MAX_PHY_ID_NUM];
int gRaInitCounter[RA_MAX_PHY_ID_NUM] = {0};

void RaPeerMutexLock(unsigned int phyId)
{
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
}

void RaPeerMutexUnlock(unsigned int phyId)
{
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
}

void RaPeerSetDebugConfig(uint64_t debugConfig)
{
    HccpSetDebugConfig(debugConfig);
    RsSetDebugConfig(debugConfig);
}

int RaPeerSocketBatchClose(unsigned int devId, struct SocketCloseInfoT conn[], unsigned int num)
{
    int ret;
    unsigned int i;
    int disuseLinger;
    unsigned int index = 0;
    unsigned int closeNum = 0;
    struct RsSocketCloseInfoT closeInfo[MAX_SOCKET_NUM];

    ret = memset_s(closeInfo, sizeof(struct RsSocketCloseInfoT) * MAX_SOCKET_NUM, 0,
        sizeof(struct RsSocketCloseInfoT) * MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[batch_close][ra_peer_socket]memset_s close_info failed, ret(%d), phyId(%u)", ret, devId),
        -ESAFEFUNC);

    for (i = 0; i < num; i++) {
        if (conn[i].fdHandle != NULL) {
            closeInfo[closeNum].fd = ((struct SocketPeerInfo *)(conn[i].fdHandle))->fd;
            ++closeNum;
        }
    }

    // use attr disuse_linger of the fist conn as the common attr for all(0 by default)
    disuseLinger = conn[0].disuseLinger;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[devId]);
    RsSetCtx(devId);
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
    ret = RsSocketBatchClose(disuseLinger, &closeInfo[index], closeNum);
    if (ret != 0) {
        hccp_err("[batch_close][ra_peer_socket]ra close failed ret(%d), phyId(%u)", ret, devId);
    }

    for (i = 0; i < num; i++) {
        if (conn[i].fdHandle != NULL) {
            free(conn[i].fdHandle);
            conn[i].fdHandle = NULL;
        }
    }
    return ret;
}

int RaPeerSocketBatchAbort(unsigned int devId, struct SocketConnectInfoT conn[], unsigned int num)
{
    struct SocketConnectInfo connOut[MAX_SOCKET_NUM];
    int ret = 0;

    ret = RaGetSocketConnectInfo(conn, num, connOut, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[batch_abort][ra_peer_socket]ra_get_socket_connect_info failed, ret(%d), phyId(%u)", ret, devId),
        ret);

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[devId]);
    RsSetCtx(devId);
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
    ret = RsSocketBatchAbort(connOut, num);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[batch_abort][ra_peer_socket]abort failed ret(%d), phyId(%u), num(%u)", ret, devId, num), ret);

    return ret;
}

int RaPeerSocketBatchConnect(unsigned int devId, struct SocketConnectInfoT conn[], unsigned int num)
{
    int ret;
    struct SocketConnectInfo connOut[MAX_SOCKET_NUM];

    ret = RaGetSocketConnectInfo(conn, num, connOut, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[batch_connect][ra_peer_socket]RaGetSocketConnectInfo failed,"
                 " ret(%d), phyId(%u)",
            ret, devId),
        ret);

    /* In peer online mode the server port number is user-defined */
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[devId]);
    RsSetCtx(devId);
    ret = RsSocketSetScopeId(devId, ((struct RaSocketHandle *)conn[0].socketHandle)->scopeId);
    if (ret != 0) {
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
        hccp_err("[set scope id][ra_peer_socket]ra_peer_socket_set_scope_id failed, ret(%d), phyId(%u)", ret, devId);
        return ret;
    }

    ret = RsSocketBatchConnect(connOut, num);
    if (ret) {
        hccp_err("[batch_connect][ra_peer_socket]ra client connect failed ret(%d), phyId(%u)", ret, devId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
    return ret;
}

int RaPeerSocketListenStart(unsigned int devId, struct SocketListenInfoT conn[], unsigned int num)
{
    struct SocketListenInfo rsConn[MAX_SOCKET_NUM] = {0};
    unsigned int i;
    int ret;

    for (i = 0; i < num; i++) {
        CHK_PRT_RETURN(conn[i].port > MAX_PORT_NUM,
            hccp_err("[listen_start][ra_peer_socket]port(%u) of "
                     "conn(%u) is invalid, max port(%u), "
                     "phyId(%u)",
                conn[i].port, i, MAX_PORT_NUM, devId),
            -EINVAL);
    }

    ret = RaGetSocketListenInfo(conn, num, rsConn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_start][ra_peer_socket]ra_get_socket_listen_info failed "
                 "ret(%d), phyId(%u)",
            ret, devId),
        ret);

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[devId]);
    RsSetCtx(devId);
    ret = RsSocketSetScopeId(devId, ((struct RaSocketHandle *)conn[0].socketHandle)->scopeId);
    if (ret != 0) {
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
        hccp_err("[set scope id][ra_peer_socket]ra_peer_socket_set_scope_id failed ret(%d), phyId(%u)", ret, devId);
        return ret;
    }

    ret = RsSocketListenStart(rsConn, num);
    // listen node found, degrade log level make it consistent with inner call
    if (ret == -EEXIST) {
        hccp_info_socket("[listen_start][ra_peer_socket]ra listen start unsuccessful ret(%d), phyId(%u)", ret, devId);
    } else if (ret == -EADDRINUSE) {
        hccp_warn_socket("[listen_start][ra_peer_socket]ra listen start unsuccessful ret(%d), phyId(%u)", ret, devId);
    } else if (ret != 0) {
        hccp_err("[listen_start][ra_peer_socket]ra listen start failed ret(%d), phyId(%u)", ret, devId);
    }
    if (ret != 0) {
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
        return ret;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);

    ret = RaGetSocketListenResult(rsConn, num, conn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_start][ra_peer_socket]ra_get_socket_listen_result failed ret(%d), phyId(%u)", ret, devId),
        ret);

    return ret;
}

int RaPeerSocketListenStop(unsigned int devId, struct SocketListenInfoT conn[], unsigned int num)
{
    struct SocketListenInfo rsConn[MAX_SOCKET_NUM] = {0};
    unsigned int i;
    int ret;

    for (i = 0; i < num; i++) {
        CHK_PRT_RETURN(conn[i].port > MAX_PORT_NUM,
            hccp_err("[listen_stop][ra_peer_socket]port(%u) of "
                     "conn(%u) is invalid, max port(%u), "
                     "phyId(%u)",
                conn[i].port, i, MAX_PORT_NUM, devId),
            -EINVAL);
    }

    ret = RaGetSocketListenInfo(conn, num, rsConn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_stop][ra_peer_socket]ra_peer_get_socket_listen_info failed ret(%d), phyId(%u)", ret, devId),
        ret);

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[devId]);
    RsSetCtx(devId);
    ret = RsSocketListenStop(rsConn, num);
    if (ret == -ENODEV) {
        hccp_warn_socket("[listen_stop][ra_peer_socket]ra socket listen stop unsuccessful ret(%d), phyId(%u)", ret,
            devId);
    } else if (ret != 0) {
        hccp_err("[listen_stop][ra_peer_socket]ra socket listen stop failed ret(%d), phyId(%u)", ret, devId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
    return ret;
}

STATIC int RaPeerSetRsConnParam(struct SocketInfoT conn[], unsigned int num, struct SocketFdData rsConn[],
    unsigned int rsNum)
{
    int ret;
    unsigned int i;
    struct RaSocketHandle *socketHandle = NULL;

    CHK_PRT_RETURN(num > rsNum,
        hccp_err("[set][ra_peer_rs_conn_param]num(%u) must be smaller than rs_num(%u)", num, rsNum), -EINVAL);

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        rsConn[i].phyId = socketHandle->rdevInfo.phyId;
        rsConn[i].family = socketHandle->rdevInfo.family;
        rsConn[i].status = conn[i].status;
        ret = memcpy_s(&(rsConn[i].localIp), sizeof(union HccpIpAddr), &(socketHandle->rdevInfo.localIp),
            sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[set][ra_peer_rs_conn_param]memcpy_s local_ip failed, ret(%d), phyId(%u)", ret,
                socketHandle->rdevInfo.phyId),
            -ESAFEFUNC);
        ret = memcpy_s(&(rsConn[i].remoteIp), sizeof(union HccpIpAddr), &(conn[i].remoteIp), sizeof(union HccpIpAddr));
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[set][ra_peer_rs_conn_param]memcpy_s remote_ip failed, ret(%d), phyId(%u)", ret,
                socketHandle->rdevInfo.phyId),
            ret);
        ret = memcpy_s(rsConn[i].tag, sizeof(rsConn[i].tag), conn[i].tag, sizeof(conn[i].tag));
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[set][ra_peer_rs_conn_param]memcpy_s tag failed, ret(%d), phyId(%u)", ret,
                socketHandle->rdevInfo.phyId),
            -ESAFEFUNC);
    }
    return 0;
}

STATIC int RaPeerSetConnParam(struct SocketInfoT conn[], struct SocketFdData rsConn[], unsigned int i,
    unsigned int sslEnable)
{
    int ret;
    struct RaSocketHandle *socketHandle = NULL;

    socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
    socketHandle->rdevInfo.phyId = rsConn[i].phyId;

    ret = memcpy_s(&(socketHandle->rdevInfo.localIp), sizeof(union HccpIpAddr), &(rsConn[i].localIp),
        sizeof(union HccpIpAddr));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[set][ra_peer_conn_param]memcpy_s local_ip failed, ret(%d), phyId(%u)", ret, rsConn[i].phyId),
        -ESAFEFUNC);
    ret = memcpy_s(&(conn[i].remoteIp), sizeof(union HccpIpAddr), &(rsConn[i].remoteIp), sizeof(union HccpIpAddr));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[set][ra_peer_conn_param]memcpy_s remote_ip failed, ret(%d), phyId(%u)", ret, rsConn[i].phyId),
        -ESAFEFUNC);

    if (conn[i].fdHandle != NULL) {
        ((struct SocketPeerInfo *)conn[i].fdHandle)->phyId = (int)rsConn[i].phyId;
        ((struct SocketPeerInfo *)conn[i].fdHandle)->fd = rsConn[i].fd;
        ((struct SocketPeerInfo *)conn[i].fdHandle)->socketHandle = socketHandle;
        ((struct SocketPeerInfo *)conn[i].fdHandle)->sslEnable = sslEnable;
    }
    conn[i].status = rsConn[i].status;
    return 0;
}

int RaPeerGetSockets(unsigned int phyId, unsigned int role, struct SocketInfoT conn[], unsigned int num)
{
    struct SocketFdData rsConn[MAX_SOCKET_NUM] = {0};
    unsigned int sslEnable;
    int connectedNum;
    unsigned int i;
    unsigned int j;
    int ret;

    ret = RaPeerSetRsConnParam(conn, num, rsConn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][ra_peer_sockets]ra_peer_set_rs_conn_param failed, ret(%d), phyId(%u)", ret, phyId), ret);

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    connectedNum = RsGetSockets(role, rsConn, num);
    if (connectedNum < 0) {
        hccp_err("[get][ra_peer_sockets]ra get socket failed ret(%d), phyId(%u)", connectedNum, phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
        return connectedNum;
    }
    ret = RsGetSslEnable(&sslEnable);
    if (ret < 0) {
        hccp_err("[get][ra_peer_sockets]rs_get_ssl_enable failed ret(%d), phyId(%u)", ret, phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
        return ret;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);

    for (i = 0; i < num; i++) {
        if (rsConn[i].status == RS_SOCK_STATUS_OK) {
            conn[i].fdHandle = (struct SocketPeerInfo *)calloc(1, sizeof(struct SocketPeerInfo));
            if (conn[i].fdHandle == NULL) {
                hccp_err("[get][ra_peer_sockets]socket handle calloc failed, phyId(%u)", phyId);
                ret = -ENOMEM;
                goto err_out;
            }
        } else {
            conn[i].fdHandle = NULL;
        }

        ret = RaPeerSetConnParam(conn, rsConn, i, sslEnable);
        if (ret) {
            hccp_err("[get][ra_peer_sockets]ra_peer_set_conn_param failed, ret(%d), phyId(%u)", ret, phyId);
            goto err_out;
        }
        if (memcpy_s(conn[i].tag, sizeof(conn[i].tag), rsConn[i].tag, sizeof(rsConn[i].tag))) {
            hccp_err("[get][ra_peer_sockets]memcpy_s tag failed, phyId(%u)", phyId);
            ret = -ESAFEFUNC;
            goto err_out;
        }
    }

    return connectedNum;

err_out:
    for (j = 0; j <= i; j++) {
        if (conn[j].fdHandle != NULL) {
            free(conn[j].fdHandle);
            conn[j].fdHandle = NULL;
        }
    }

    return ret;
}

int RaPeerSocketSend(unsigned int devId, const void *handle, const void *data, unsigned long long size)
{
    int fd;
    int ret;
    unsigned int sslEnable;

    fd = ((const struct SocketPeerInfo *)handle)->fd;
    sslEnable = ((const struct SocketPeerInfo *)handle)->sslEnable;
    if (sslEnable != RA_SSL_DISABLE) {
        PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[devId]);
        RsSetCtx(devId);
    }
    ret = RsPeerSocketSend(sslEnable, fd, data, size);
    if (sslEnable != RA_SSL_DISABLE) {
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
    }
    return ret;
}

int RaPeerSocketRecv(unsigned int devId, const void *handle, void *data, unsigned long long size)
{
    int fd;
    int ret;
    unsigned int sslEnable;

    fd = ((const struct SocketPeerInfo *)handle)->fd;
    sslEnable = ((const struct SocketPeerInfo *)handle)->sslEnable;
    if (sslEnable != RA_SSL_DISABLE) {
        PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[devId]);
        RsSetCtx(devId);
    }
    ret = RsPeerSocketRecv(sslEnable, fd, data, size);
    if (sslEnable != RA_SSL_DISABLE) {
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[devId]);
    }
    return ret;
}

int RaPeerSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    int ret;
    unsigned int i;
    char netAddr[MAX_IP_LEN] = {0};

    for (i = 0; i < num; i++) {
        CHK_PRT_RETURN(inet_ntop(rdevInfo.family, &whiteList[i].remoteIp, netAddr, sizeof(netAddr)) == NULL,
            hccp_err("[add][ra_peer_socket_white_list]remote ip is invalid! i(%u), phyId(%u)", i, rdevInfo.phyId),
            -EINVAL);
    }
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[rdevInfo.phyId]);
    RsSetCtx(rdevInfo.phyId);
    ret = RsSocketWhiteListAdd(rdevInfo, whiteList, num);
    if (ret) {
        hccp_err("[add][ra_peer_socket_white_list]rs_socket_white_list_add failed ret(%d), phyId(%u)", ret,
            rdevInfo.phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdevInfo.phyId]);
        return ret;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdevInfo.phyId]);
    return ret;
}

int RaPeerEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event)
{
    int ret;

    ret = RsEpollCtlAdd(fdHandle, event);
    if (ret) {
        hccp_err("[ra_peer_epoll_ctl_add]rs_epoll_ctl_add failed ret(%d)", ret);
    }
    return ret;
}

int RaPeerEpollCtlMod(const void *fdHandle, enum RaEpollEvent event)
{
    int ret;

    ret = RsEpollCtlMod(fdHandle, event);
    if (ret) {
        hccp_err("[ra_peer_epoll_ctl_mod]rs_epoll_ctl_mod failed ret(%d)", ret);
    }
    return ret;
}

int RaPeerEpollCtlDel(const void *fdHandle)
{
    int fd = -1;
    int ret;

    fd = ((const struct SocketPeerInfo *)fdHandle)->fd;
    ret = RsEpollCtlDel(fd);
    if (ret) {
        hccp_err("[ra_peer_epoll_ctl_del]rs_epoll_ctl_del failed ret(%d), phyId(%d)", ret,
            ((const struct SocketPeerInfo *)fdHandle)->phyId);
    }
    return ret;
}

void RaPeerSetTcpRecvCallback(unsigned int phyId, const void *callback)
{
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    RsSetTcpRecvCallback(callback);
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
}

int RaPeerSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    int ret;
    unsigned int i;
    char netAddr[MAX_IP_LEN] = {0};

    for (i = 0; i < num; i++) {
        CHK_PRT_RETURN(inet_ntop(rdevInfo.family, &whiteList[i].remoteIp, netAddr, sizeof(netAddr)) == NULL,
            hccp_err("[del][ra_peer_socket_white_list]remote ip is invalid! i(%u), phyId(%u)", i, rdevInfo.phyId),
            -EINVAL);
    }

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[rdevInfo.phyId]);
    RsSetCtx(rdevInfo.phyId);
    ret = RsSocketWhiteListDel(rdevInfo, whiteList, num);
    if (ret) {
        hccp_err("[del][ra_peer_socket_white_list]rs_socket_white_list_del failed ret(%d), phyId(%u)", ret,
            rdevInfo.phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdevInfo.phyId]);
    return ret;
}

int RaPeerSocketDeinit(struct rdev rdevInfo)
{
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[rdevInfo.phyId]);
    RsSetCtx(rdevInfo.phyId);
    ret = RsSocketDeinit(rdevInfo);
    if (ret) {
        hccp_err("[deinit][ra_peer_socket]rs_socket_deinit failed, ret(%d), phyId(%u)", ret, rdevInfo.phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdevInfo.phyId]);
        return ret;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdevInfo.phyId]);
    return 0;
}

int RaPeerQpCreate(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, void **qpHandle)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpPeer = NULL;
    struct RsQpResp qpResp = {0};
    struct RsQpNorm qpNorm = {0};
    int ret;

    qpPeer = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpPeer == NULL, hccp_err("[create][ra_peer_qp]qp_peer calloc failed, phyId[%u]", phyId), -ENOMEM);

    qpNorm.flag = flag;
    qpNorm.isExp = 1;
    qpNorm.isExt = 0;
    qpNorm.qpMode = qpMode;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsQpCreate(phyId, rdmaHandle->rdevIndex, qpNorm, &qpResp);
    if (ret) {
        hccp_err("[create][ra_peer_qp]RsQpCreate failed ret[%d], phyId[%u]", ret, phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
        goto calloc_err;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
    qpPeer->phyId = phyId;
    qpPeer->qpn = qpResp.qpn;
    qpPeer->psn = qpResp.psn;
    qpPeer->gidIdx = qpResp.gidIdx;
    qpPeer->flag = flag;
    qpPeer->qpMode = qpMode;
    qpPeer->rdevIndex = rdmaHandle->rdevIndex;
    qpPeer->rdmaHandle = rdmaHandle;
    qpPeer->rdmaOps = rdmaHandle->rdmaOps;

    *qpHandle = qpPeer;
    return ret;

calloc_err:
    free(qpPeer);
    qpPeer = NULL;
    return ret;
}

int RaPeerQpCreateWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, void **qpHandle)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RsQpNormWithAttrs qpNorm = {0};
    struct RsQpRespWithAttrs qpResp = {0};
    struct RaQpHandle *qpPeer = NULL;
    int ret;

    qpPeer = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpPeer == NULL, hccp_err("[create][ra_peer_qp_with_attrs]qp_peer calloc failed, phyId[%u]", phyId),
        -ENOMEM);

    qpNorm.isExp = 1;
    qpNorm.isExt = 0;
    ret = memcpy_s(&qpNorm.extAttrs, sizeof(struct QpExtAttrs), extAttrs, sizeof(struct QpExtAttrs));
    if (ret) {
        hccp_err("[create][ra_peer_qp_with_attrs]memcpy_s for ext_attrs failed ret[%d], phyId[%u]", ret, phyId);
        ret = -ESAFEFUNC;
        goto calloc_err;
    }

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsQpCreateWithAttrs(phyId, rdmaHandle->rdevIndex, &qpNorm, &qpResp);
    if (ret) {
        hccp_err("[create][ra_peer_qp_with_attrs]RsQpCreateWithAttrs failed ret[%d], phyId[%u]", ret, phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
        goto calloc_err;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
    qpPeer->phyId = phyId;
    qpPeer->qpn = qpResp.qpn;
    qpPeer->psn = qpResp.psn;
    qpPeer->gidIdx = qpResp.gidIdx;
    qpPeer->flag = extAttrs->qpAttr.qp_type == IBV_QPT_RC ? 0 : 1;
    qpPeer->qpMode = extAttrs->qpMode;
    qpPeer->rdevIndex = rdmaHandle->rdevIndex;
    qpPeer->rdmaHandle = rdmaHandle;
    qpPeer->rdmaOps = rdmaHandle->rdmaOps;
    qpPeer->typicalQpAttr.udpSport = extAttrs->udpSport;

    *qpHandle = qpPeer;
    return ret;

calloc_err:
    free(qpPeer);
    qpPeer = NULL;
    return ret;
}

int RaPeerMrReg(struct RaQpHandle *qpPeer, struct MrInfoT *info)
{
    int ret;
    struct RdmaMrRegInfo mrRegInfo = {0};

    mrRegInfo.addr = info->addr;
    mrRegInfo.len = info->size;
    mrRegInfo.access = info->access;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsMrReg(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, &mrRegInfo);
    if (ret) {
        hccp_err("[reg][ra_peer_mr]ra_reg_mr failed ret(%d), phyId(%u)", ret, qpPeer->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);
    info->lkey = mrRegInfo.lkey;
    info->rkey = mrRegInfo.rkey;
    return ret;
}

int RaPeerMrDereg(struct RaQpHandle *qpPeer, struct MrInfoT *info)
{
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsMrDereg(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, (char *)info->addr);
    if (ret) {
        hccp_err("[dereg][ra_peer_mr]ra_de_reg_mr failed ret(%d), phyId(%u)", ret, qpPeer->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);
    return ret;
}

int RaPeerRegisterMr(struct RaRdmaHandle *rdmaPeer, struct MrInfoT *info, void **mrHandle)
{
    struct RdmaMrRegInfo mrRegInfo = {0};
    int ret;

    mrRegInfo.addr = info->addr;
    mrRegInfo.len = info->size;
    mrRegInfo.access = info->access;

    RsSetCtx(rdmaPeer->rdevInfo.phyId);
    ret = RsRegisterMr(rdmaPeer->rdevInfo.phyId, rdmaPeer->rdevIndex, &mrRegInfo, mrHandle);
    if (ret) {
        hccp_err("[ra_peer_register_mr]rs_register_mr failed ret(%d), phyId(%u)", ret, rdmaPeer->rdevInfo.phyId);
    }
    info->lkey = mrRegInfo.lkey;
    info->rkey = mrRegInfo.rkey;
    return ret;
}

int RaPeerDeregisterMr(struct RaRdmaHandle *rdmaPeer, void *mrHandle)
{
    int ret;

    RsSetCtx(rdmaPeer->rdevInfo.phyId);
    ret = RsDeregisterMr(rdmaPeer->rdevInfo.phyId, rdmaPeer->rdevIndex, mrHandle);
    if (ret != 0) {
        hccp_err("[ra_peer_deregister_mr]rs_deregister_mr failed ret(%d), phyId(%u)", ret, rdmaPeer->rdevInfo.phyId);
    }
    return ret;
}

int RaPeerTypicalQpModify(struct RaQpHandle *qpPeer, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo)
{
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsTypicalQpModify(qpPeer->phyId, qpPeer->rdevIndex, *localQpInfo, *remoteQpInfo, &(qpPeer->typicalQpAttr));
    if (ret != 0) {
        hccp_err("[modify][ra_peer_qp]rs_typical_qp_modify failed ret(%d) phyId(%u) qpn(%u)", ret, qpPeer->phyId,
            qpPeer->qpn);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);

    return ret;
}

int RaPeerSetQpLbValue(struct RaQpHandle *qpHandle, int lbValue)
{
    int ret = 0;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpHandle->phyId]);
    RsSetCtx(qpHandle->phyId);
    ret = RsSetQpLbValue(qpHandle->phyId, qpHandle->rdevIndex, qpHandle->qpn, lbValue);
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpHandle->phyId]);
    if (ret != 0) {
        if (ret == -ENOTSUPP) {
            hccp_run_warn("[set][lbValue]RsSetQpLbValue unsuccessful ret:%d", ret);
        } else {
            hccp_err("[set][lbValue]RsSetQpLbValue failed ret:%d, phyId:%u", ret, qpHandle->phyId);
        }
    }
    return ret;
}

int RaPeerGetQpLbValue(struct RaQpHandle *qpHandle, int *lbValue)
{
    int ret = 0;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpHandle->phyId]);
    RsSetCtx(qpHandle->phyId);
    ret = RsGetQpLbValue(qpHandle->phyId, qpHandle->rdevIndex, qpHandle->qpn, lbValue);
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpHandle->phyId]);
    if (ret != 0) {
        hccp_err("[get][lbValue]RsGetQpLbValue failed ret:%d, phyId:%u", ret, qpHandle->phyId);
    }
    return ret;
}

int RaPeerQpConnectAsync(struct RaQpHandle *qpPeer, const void *sockHandle)
{
    int ret;
    int fd = ((const struct SocketPeerInfo *)sockHandle)->fd;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsQpConnectAsync(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, fd);
    if (ret) {
        hccp_err("[connect_async][ra_peer_qp]ra qp info sync failed socket fd(%d) ret(%d), phyId(%u)", fd, ret,
            qpPeer->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);
    return ret;
}

int RaPeerGetQpStatus(struct RaQpHandle *qpPeer, int *status)
{
    struct RsQpStatusInfo qpInfo = {0};
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsGetQpStatus(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, &qpInfo);
    if (ret) {
        hccp_err("[get][ra_peer_qp_status]ra get qp status failed ret(%d), phyId(%u)", ret, qpPeer->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);
    *status = qpInfo.status;
    return ret;
}

STATIC int RaPeerLoopbackQpModifyPrepare(struct RaQpHandle *qpHandle, struct TypicalQp *qpInfo)
{
    int ret = 0;

    qpInfo->qpn = qpHandle->qpn;
    qpInfo->psn = qpHandle->psn;
    qpInfo->gidIdx = qpHandle->gidIdx;
    qpInfo->retryCnt = QP_DEFAULT_MAX_ATTR_RETRY_CNT;
    qpInfo->retryTime = QP_DEFAULT_MAX_ATTR_TIMEOUT;
    ret = memcpy_s(qpInfo->gid, sizeof(qpInfo->gid), qpHandle->rdmaHandle->gid, sizeof(qpHandle->rdmaHandle->gid));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("memcpy_s gid failed, ret:%d, dst_len:%u, src_len:%d, phyId:%u", ret, sizeof(qpInfo->gid),
            qpHandle->rdmaHandle->gid, qpHandle->phyId),
        -ESAFEFUNC);

    return ret;
}

STATIC int RaPeerLoopbackQpModify(struct RaQpHandle *qpHandle0, struct RaQpHandle *qpHandle1)
{
    struct TypicalQp qp0Info = {0};
    struct TypicalQp qp1Info = {0};
    int ret = 0;

    ret = RaPeerLoopbackQpModifyPrepare(qpHandle0, &qp0Info);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("ra_peer_loopback_qp_modify_prepare qp0 failed, ret:%d, phyId:%u", ret, qpHandle0->phyId), ret);
    ret = RaPeerLoopbackQpModifyPrepare(qpHandle1, &qp1Info);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("ra_peer_loopback_qp_modify_prepare qp1 failed, ret:%d, phyId:%u", ret, qpHandle1->phyId), ret);

    ret = RaPeerTypicalQpModify(qpHandle0, &qp0Info, &qp1Info);
    CHK_PRT_RETURN(ret != 0, hccp_err("ra_peer_typical_qp_modify qp0 failed, ret:%d, phyId:%u", ret, qpHandle0->phyId),
        ret);
    ret = RaPeerTypicalQpModify(qpHandle1, &qp1Info, &qp0Info);
    CHK_PRT_RETURN(ret != 0, hccp_err("ra_peer_typical_qp_modify qp1 failed, ret:%d, phyId:%u", ret, qpHandle1->phyId),
        ret);

    return ret;
}

STATIC void RaPeerLoopbackSingleQpDestroy(struct RaQpHandle *qpHandle)
{
    struct RaLoopbackInfo *loopbackInfo = qpHandle->loopbackInfo;
    struct RaRdmaHandle *rdmaHandle = qpHandle->rdmaHandle;
    struct CqAttr attr = {0};

    attr.qpContext = &(loopbackInfo->cqContext);
    attr.ibSendCq = &(loopbackInfo->ibSendCq);
    attr.ibRecvCq = &(loopbackInfo->ibRecvCq);

    (void)RaPeerNormalQpDestroy(qpHandle);
    (void)RaPeerCqDestroy(rdmaHandle, &attr);
    (void)RaPeerDestroyCompChannel((void *)loopbackInfo->compChannel);

    free(loopbackInfo);
    loopbackInfo = NULL;
}

STATIC void RaPeerLoopbackQpCreatePrepare(struct CqAttr *cqAttr, struct ibv_qp_init_attr *qpInitAttr)
{
    qpInitAttr->qp_context = *(cqAttr->qpContext);
    qpInitAttr->send_cq = *(cqAttr->ibSendCq);
    qpInitAttr->recv_cq = *(cqAttr->ibRecvCq);
    qpInitAttr->qp_type = IBV_QPT_RC;
    qpInitAttr->cap.max_send_wr = QP_DEFAULT_MIN_CAP_SEND_WR;
    qpInitAttr->cap.max_recv_wr = QP_DEFAULT_MIN_CAP_RECV_WR;
    qpInitAttr->cap.max_send_sge = QP_DEFAULT_MIN_CAP_SEND_SGE;
    qpInitAttr->cap.max_recv_sge = QP_DEFAULT_MIN_CAP_RECV_SGE;
    qpInitAttr->cap.max_inline_data = QP_DEFAULT_MAX_CAP_INLINE_DATA;
}

STATIC int RaPeerLoopbackSingleQpCreate(struct RaRdmaHandle *rdmaHandle, struct RaQpHandle **qpHandle,
    struct ibv_qp **qp)
{
    struct RaLoopbackInfo *loopbackInfo = NULL;
    struct ibv_qp_init_attr qpInitAttr = {0};
    struct CqAttr cqAttr = {0};
    int ret = 0;

    loopbackInfo = (struct RaLoopbackInfo *)calloc(1, sizeof(struct RaLoopbackInfo));
    CHK_PRT_RETURN(loopbackInfo == NULL, hccp_err("loopback_info calloc failed, phyId:%u", rdmaHandle->rdevInfo.phyId),
        -ENOMEM);

    ret = RaPeerCreateCompChannel(rdmaHandle, (void **)&loopbackInfo->compChannel);
    if (ret != 0) {
        hccp_err("RaPeerCreateCompChannel failed, ret:%d, phyId:%u", ret, rdmaHandle->rdevInfo.phyId);
        goto channel_create_err;
    }

    cqAttr.qpContext = &(loopbackInfo->cqContext);
    cqAttr.ibSendCq = &(loopbackInfo->ibSendCq);
    cqAttr.ibRecvCq = &(loopbackInfo->ibRecvCq);
    cqAttr.sendChannel = loopbackInfo->compChannel;
    cqAttr.recvChannel = loopbackInfo->compChannel;
    cqAttr.sendCqDepth = CQ_DEFAULT_MIN_SEND_DEPTH;
    cqAttr.recvCqDepth = CQ_DEFAULT_MIN_RECV_DEPTH;
    ret = RaPeerCqCreate(rdmaHandle, &cqAttr);
    if (ret != 0) {
        hccp_err("ra_peer_cq_create failed, ret:%d, phyId:%u", ret, rdmaHandle->rdevInfo.phyId);
        goto cq_create_err;
    }

    RaPeerLoopbackQpCreatePrepare(&cqAttr, &qpInitAttr);
    ret = RaPeerNormalQpCreate(rdmaHandle, &qpInitAttr, (void **)qpHandle, (void **)qp);
    if (ret != 0) {
        hccp_err("ra_peer_normal_qp_create failed, ret:%d, phyId:%u", ret, rdmaHandle->rdevInfo.phyId);
        goto qp_create_err;
    }
    (*qpHandle)->loopbackInfo = loopbackInfo;
    return ret;

qp_create_err:
    (void)RaPeerCqDestroy(rdmaHandle, &cqAttr);
cq_create_err:
    (void)RsDestroyCompChannel((void *)loopbackInfo->compChannel);
channel_create_err:
    free(loopbackInfo);
    loopbackInfo = NULL;
    return ret;
}

int RaPeerLoopbackQpCreate(struct RaRdmaHandle *rdmaHandle, struct LoopbackQpPair *qpPair, void **qpHandle)
{
    struct RaQpHandle *qpHandle0 = NULL;
    struct RaQpHandle *qpHandle1 = NULL;
    struct ibv_qp *qp0 = NULL;
    struct ibv_qp *qp1 = NULL;
    int ret;

    ret = RaPeerLoopbackSingleQpCreate(rdmaHandle, &qpHandle0, &qp0);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("ra_peer_loopback_single_qp_create qp0 failed, ret:%d, phyId:%u", ret, rdmaHandle->rdevInfo.phyId),
        ret);

    ret = RaPeerLoopbackSingleQpCreate(rdmaHandle, &qpHandle1, &qp1);
    if (ret != 0) {
        hccp_err("ra_peer_loopback_single_qp_create qp1 failed, ret:%d, phyId:%u", ret, rdmaHandle->rdevInfo.phyId);
        goto qp1_create_err;
    }

    ret = RaPeerLoopbackQpModify(qpHandle0, qpHandle1);
    if (ret != 0) {
        hccp_err("ra_peer_loopback_qp_modify failed, ret:%d, phyId:%u", ret, rdmaHandle->rdevInfo.phyId);
        goto qp_modify_err;
    }

    qpPair->ibvQp0 = qp0;
    qpPair->ibvQp1 = qp1;
    qpHandle0->loopbackQpHandle = qpHandle1;
    qpHandle1->loopbackQpHandle = qpHandle0;
    *qpHandle = qpHandle0;
    return ret;

qp_modify_err:
    RaPeerLoopbackSingleQpDestroy(qpHandle1);
qp1_create_err:
    RaPeerLoopbackSingleQpDestroy(qpHandle0);
    return ret;
}

STATIC void RaPeerLoopbackQpDestroy(struct RaQpHandle *qpHandle0)
{
    struct RaQpHandle *qpHandle1 = qpHandle0->loopbackQpHandle;

    RaPeerLoopbackSingleQpDestroy(qpHandle1);
    RaPeerLoopbackSingleQpDestroy(qpHandle0);
}

STATIC int RaPeerSingleQpDestroy(struct RaQpHandle *qpPeer)
{
    int ret = 0;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsQpDestroy(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn);
    if (ret != 0) {
        hccp_err("[destroy][ra_peer_qp]destroy failed ret(%d), phyId(%u)", ret, qpPeer->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);
    free(qpPeer);
    qpPeer = NULL;
    return ret;
}

int RaPeerQpDestroy(struct RaQpHandle *qpPeer)
{
    if (qpPeer->loopbackQpHandle != NULL) {
        RaPeerLoopbackQpDestroy(qpPeer);
        return 0;
    } else if (qpPeer->directFlag != DIRECT_FLAG_NOTSUPP) {
        return RaPeerNdaQpDestroy(qpPeer);
    } else {
        return RaPeerSingleQpDestroy(qpPeer);
    }
}

int RaPeerSendWr(struct RaQpHandle *qpPeer, struct SendWr *wr, struct SendWrRsp *wrRsp)
{
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsSendWr(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, wr, wrRsp);
    if (ret) {
        hccp_err("[send][ra_peer_wr]ra_send_wr failed ret(%d), phyId(%u)", ret, qpPeer->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);
    return ret;
}

STATIC void RaInitWrlistBaseInfo(struct RsWrlistBaseInfo *baseInfo, struct RaQpHandle *qpHandle)
{
    baseInfo->phyId = qpHandle->phyId;
    baseInfo->rdevIndex = qpHandle->rdevIndex;
    baseInfo->qpn = qpHandle->qpn;
    baseInfo->keyFlag = 0;
}

int RaPeerSendWrlist(struct RaQpHandle *qpHandle, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret = 0;
    unsigned int completeCnt = 0;
    unsigned int sendCnt = 0;
    struct RsWrlistBaseInfo baseInfo;
    struct WrlistSendCompleteNum wrlistOnce;
    unsigned int compeletOnceCnt, i;
    struct WrInfo *wrList = NULL;

    RaInitWrlistBaseInfo(&baseInfo, qpHandle);
    CHK_PRT_RETURN(wrlistNum.sendNum > SIZE_MAX / sizeof(struct WrInfo),
        hccp_err("Sendnum is invalid, phyId[%u]", baseInfo.phyId), -EINVAL);
    wrList = calloc(wrlistNum.sendNum, sizeof(struct WrInfo));
    CHK_PRT_RETURN(wrList == NULL, hccp_err("wr_list calloc failed, phyId[%u]", baseInfo.phyId), -ENOMEM);

    for (i = 0; i < wrlistNum.sendNum; i++) {
        wrList[i].op = wr[i].op;
        wrList[i].sendFlags = wr[i].sendFlags;
        wrList[i].dstAddr = wr[i].dstAddr;
        wrList[i].memList.addr = wr[i].memList.addr;
        wrList[i].memList.len = wr[i].memList.len;
        wrList[i].memList.lkey = wr[i].memList.lkey;
    }

    while (sendCnt < wrlistNum.sendNum) {
        wrlistOnce.sendNum = (wrlistNum.sendNum - sendCnt) > MAX_WR_NUM ? MAX_WR_NUM : (wrlistNum.sendNum - sendCnt);

        PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[baseInfo.phyId]);
        RsSetCtx(baseInfo.phyId);
        ret = RsSendWrlist(baseInfo, &wrList[sendCnt], wrlistOnce.sendNum, &opRsp[sendCnt], &compeletOnceCnt);
        if (ret) {
            hccp_err("[send][ra_peer_wrlist]ra_peer_send_wrlist failed ret[%d], sendNum[%u], sendCnt[%u], phyId[%u]",
                ret, wrlistNum.sendNum, sendCnt, baseInfo.phyId);
            PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[baseInfo.phyId]);
            goto alloc_wr_list_fail;
        }
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[baseInfo.phyId]);
        sendCnt += wrlistOnce.sendNum;
        completeCnt += compeletOnceCnt;
    }

    if (sendCnt != completeCnt) {
        hccp_err("[send][ra_peer_wrlist]complete_cnt[%u] != send_cnt[%u], phyId[%u]", completeCnt, sendCnt,
            baseInfo.phyId);
        ret = -EINVAL;
    } else {
        *(wrlistNum.completeNum) = completeCnt;
    }

alloc_wr_list_fail:
    free(wrList);
    wrList = NULL;
    return ret;
}

int RaPeerGetNotifyBaseAddr(struct RaRdmaHandle *handle, unsigned long long *va, unsigned long long *size)
{
    struct MrInfoT info = {0};
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[handle->rdevInfo.phyId]);
    RsSetCtx(handle->rdevInfo.phyId);
    ret = RsGetNotifyMrInfo(handle->rdevInfo.phyId, handle->rdevIndex, &info);
    if (ret) {
        hccp_err("[get][ra_peer_notify_base_addr]rs_get_notify_mr_info failed ret(%d), phyId(%u)", ret,
            handle->rdevInfo.phyId);
    }
    *va = (unsigned long long)(uintptr_t)info.addr;
    *size = info.size;
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[handle->rdevInfo.phyId]);
    return ret;
}

int RaPeerInit(struct RaInitConfig *cfg, unsigned int whiteListStatus)
{
    int ret;

    hccp_info("[init][ra_peer]ra_peer_init phyId[%d] start", cfg->phyId);

    /* In peer online mode chip id equals to phy id */
    struct RsInitConfig rsPeerOnlineCfg = {
        .chipId = cfg->phyId,
        .hccpMode = cfg->nicPosition,
        .whiteListStatus = whiteListStatus,
    };
    ret = DlHalInit();
    if (ret) {
        hccp_err("[init][ra_peer]dl_hal_init failed, ret = %d, phyId = %u", ret, cfg->phyId);
        return ret;
    }

    int counter = __sync_fetch_and_add(&(gRaInitCounter[cfg->phyId]), 1);
    if (counter > 0) {
        PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[cfg->phyId]);
        RsSetCtx(cfg->phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[cfg->phyId]);
        hccp_warn("ra peer has been init for device %u!", cfg->phyId);
        return 0;
    }

    ret = pthread_mutex_init(&gRaPeerMutex[cfg->phyId], NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_peer]pthread_mutex_init failed, ret(%d) phyId(%u)", ret, cfg->phyId),
        -ESYSFUNC);

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[cfg->phyId]);
    RsSetCtx(cfg->phyId);
    ret = RsInit(&rsPeerOnlineCfg);
    if (ret) {
        hccp_err("[init][ra_peer]rs init failed(%d), phyId(%u)", ret, cfg->phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[cfg->phyId]);
        pthread_mutex_destroy(&gRaPeerMutex[cfg->phyId]);
        return ret;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[cfg->phyId]);
    hccp_info("[init][ra_peer]ra_peer_init phyId[%d] succ", cfg->phyId);
    return ret;
}

int RaPeerGetTlsEnable(unsigned int phyId, bool *tlsEnable)
{
    int ret;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetTlsEnable(phyId, tlsEnable);
    if (ret != 0) {
        hccp_err("[get][tls_enable]rs_get_tls_enable failed, ret(%d) phyId(%u)", ret, phyId);
    }
    RaPeerMutexUnlock(phyId);
    return ret;
}

int RaPeerGetSecRandom(unsigned int *value)
{
    int ret;

    ret = RsGetSecRandom(value);
    if (ret != 0) {
        hccp_run_warn("[get_random] unsuccessful, ret(%d)", ret);
    }
    return ret;
}

int RaPeerGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    unsigned int phyId = info->phyId;
    int ret;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetHccnCfg(info, key, value, valueLen);
    if (ret != 0) {
        hccp_err("[get][hccn_cfg]RsGetHccnCfg failed, ret(%d) phyId(%u)", ret, phyId);
    }
    RaPeerMutexUnlock(phyId);
    return ret;
}

int RaPeerDeinit(struct RaInitConfig *cfg)
{
    int ret = 0;

    hccp_info_init("[deinit][ra_peer]ra_peer_deinit phyId[%d] start", cfg->phyId);

    /* In peer online mode chip id equals to phy id */
    struct RsInitConfig rsPeerOnlineCfg = {
        .chipId = cfg->phyId,
        .hccpMode = cfg->nicPosition,
        .whiteListStatus = WHITE_LIST_ENABLE,
    };

    if (__sync_fetch_and_sub(&(gRaInitCounter[cfg->phyId]), 1) > 1) {
        goto dl_deinit;
    }

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[cfg->phyId]);
    RsSetCtx(cfg->phyId);
    ret = RsDeinit(&rsPeerOnlineCfg);
    // no need to destroy lock & return immediately for retry
    if (ret == -EAGAIN) {
        hccp_warn_init("[deinit][ra_peer]rs deinit unsuccessful(%d), phyId(%u)", ret, cfg->phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[cfg->phyId]);
        return ret;
    }

    if (ret) {
        hccp_err("[deinit][ra_peer]rs deinit failed(%d), phyId(%u)", ret, cfg->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[cfg->phyId]);
    pthread_mutex_destroy(&gRaPeerMutex[cfg->phyId]);

dl_deinit:
    DlHalDeinit();
    hccp_info_init("[deinit][ra_peer]ra_peer_deinit phyId[%d] succ", cfg->phyId);
    return ret;
}

int RaPeerGetIfnum(unsigned int phyId, unsigned int *num)
{
    int ret;

    hccp_info_socket("[get][ra_peer_ifnum]ra_peer_get_ifnum phyId[%u] start", phyId);
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsPeerGetIfnum(phyId, num);
    if (ret) {
        hccp_err("[get][ra_peer_ifnum]rs_peer_get_ifnum failed(%d) phyId[%u]", ret, phyId);
    } else {
        hccp_info_socket("[get][ra_peer_ifnum]ra_peer_get_ifnum phyId[%u] succ", phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);

    return ret;
}

int RaPeerGetIfaddrs(unsigned int phyId, struct InterfaceInfo interfaceInfos[], unsigned int *num)
{
    int ret;
    hccp_info_socket("[get][ra_peer_ifaddrs] ra_peer_get_ifaddrs phyId[%u] start", phyId);
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsPeerGetIfaddrs(interfaceInfos, num, phyId);
    if (ret) {
        hccp_err("[get][ra_peer_ifaddrs]rs_peer_get_ifaddrs failed(%d), phyId(%u)", ret, phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
    hccp_info_socket("[get][ra_peer_ifaddrs] ra_peer_get_ifaddrs phyId[%u] succ", phyId);
    return ret;
}

int HostNotifyBaseAddrInit(unsigned int phyId)
{
    int ret, retVal;
    unsigned int notifySize = 0;
    unsigned long long *notifyVa = NULL;
    unsigned int logicId = 0;

    ret = DlDrvDeviceGetIndexByPhyId(phyId, &logicId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][base_addr]drvDeviceGetIndexByPhyId failed, ret(%d), phyId(%u)", ret, phyId), ret);

    ret = DlHalNotifyGetInfo(logicId, 0, RA_NOTIFY_TYPE_TOTAL_SIZE, &notifySize);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][base_addr]halNotifyGetInfo failed, ret(%d), logicId(%u), phyId(%u)", ret, logicId, phyId),
        ret);

    gNotifyFd = open(HOST_DEVICE_NAME, O_RDWR);
    CHK_PRT_RETURN(gNotifyFd < 0,
        hccp_err("[init][base_addr]Failed to open file_path[%s], err_code[%d], phyId[%u]", HOST_DEVICE_NAME, errno,
            phyId),
        -ENOENT);

    notifyVa = mmap(NULL, notifySize, PROT_READ | PROT_WRITE, MAP_SHARED, gNotifyFd,
        (unsigned long long)logicId << PAGE_SHIFT);
    if (notifyVa == MAP_FAILED) {
        hccp_err("[init][base_addr]failed to mmap recv buf, fd[%d], err_code[%d], phyId[%u]", gNotifyFd, errno, phyId);
        ret = -ENOMEM;
        goto close_fd;
    }

    ret = RsNotifyCfgSet(phyId, (uintptr_t)notifyVa, notifySize);
    if (ret) {
        hccp_err("[init][base_addr]ra_hdc_notify_cfg_set failed, ret(%d), phyId(%u)", ret, phyId);
        goto unmmap_mem;
    }
    return 0;

unmmap_mem:
    retVal = munmap((void *)notifyVa, notifySize);
    if (retVal) {
        hccp_err("[init][base_addr]munmap buf munmap error, length:%lu, ret:%d, phyId:%u", notifySize, retVal, phyId);
    }
close_fd:
    HCCP_CLOSE_RETRY_FOR_EINTR(gNotifyFd);
    return ret;
}

int RaPeerNotifyBaseAddrInit(unsigned int notifyType, unsigned int phyId)
{
    switch (notifyType) {
        case NOTIFY:
            return HostNotifyBaseAddrInit(phyId);
        case EVENTID:
            return 0;
        case NO_USE:
            return 0;
        default: {
            hccp_err("[init][base_addr]notify_type[%u] is invalid, valid value[0-2], phyId[%u]", notifyType, phyId);
            return -EINVAL;
        }
    }
}

int HostNotifyBaseAddrUninit(unsigned int phyId)
{
    int ret;
    unsigned long long va, size;
    unsigned int logicId = 0;
    struct HostRoceNotifyInfo notifyNode = {0};

    ret = DlDrvDeviceGetIndexByPhyId(phyId, &logicId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[uninit][base_addr]drvDeviceGetIndexByPhyId failed, ret(%d), phyId(%u)", ret, phyId), ret);

    ret = RsNotifyCfgGet(phyId, &va, &size);
    CHK_PRT_RETURN(ret != 0, hccp_err("[uninit][base_addr]rs_notify_cfg_get failed, ret(%d), phyId(%u)", ret, phyId),
        ret);
    notifyNode.logicId = logicId;
    notifyNode.va = va;
    notifyNode.sz = size;

    CHK_PRT_RETURN(gNotifyFd < 0,
        hccp_err("[uninit][base_addr]file_path[%s] has closed, phyId[%u]", HOST_DEVICE_NAME, phyId), -ENOENT);

    ret = ioctl(gNotifyFd, HOST_CDEV_IOC_FREE_NOTIFY, &notifyNode);
    if (ret < 0) {
        hccp_err("[uninit][base_addr]Failed to run ioctl, ret[%d], err_code[%d], phyId[%u]", ret, errno, phyId);
        HCCP_CLOSE_RETRY_FOR_EINTR(gNotifyFd);
        return ret;
    }

    ret = munmap((void *)(uintptr_t)va, size);
    if (ret) {
        hccp_err("[uninit][base_addr]munmap buf munmap error, *size:%lu, ret:%d, phyId:%u", size, ret, phyId);
        HCCP_CLOSE_RETRY_FOR_EINTR(gNotifyFd);
        return ret;
    }

    HCCP_CLOSE_RETRY_FOR_EINTR(gNotifyFd);
    return 0;
}

int NotifyBaseAddrUninit(unsigned int notifyType, unsigned int phyId)
{
    switch (notifyType) {
        case NOTIFY:
            return HostNotifyBaseAddrUninit(phyId);
        case EVENTID:
            return 0;
        case NO_USE:
            return 0;
        default: {
            hccp_err("[uninit][base_addr]notify_type[%u] is invalid, valid value[0-2], phyId[%u]", notifyType, phyId);
            return -EINVAL;
        }
    }
}

int RaPeerRdevInit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType, struct rdev rdevInfo,
    unsigned int *rdevIndex)
{
    int ret, retVal;

    hccp_run_info("[init][ra_peer_rdev]ra_peer_rdev_init phyId[%d] notify_type[%u] physical device id[%u]",
        rdevInfo.phyId, notifyType, rdmaHandle->rdevInfo.phyId);

    RsSetCtx(rdevInfo.phyId);
    ret = RaPeerNotifyBaseAddrInit(notifyType, rdevInfo.phyId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_peer_rdev] ra_peer_notify_base_addr_init failed[%d], phyId[%u]", ret, rdevInfo.phyId), ret);

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[rdevInfo.phyId]);
    ret = RsRdevInit(rdevInfo, notifyType, rdevIndex);
    if (ret) {
        hccp_err("[init][ra_peer_rdev] rs_rdev_init failed[%d], phyId[%u]", ret, rdevInfo.phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdevInfo.phyId]);
        goto notify_base_addr_uninit;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdevInfo.phyId]);

    return 0;
notify_base_addr_uninit:
    retVal = NotifyBaseAddrUninit(notifyType, rdevInfo.phyId);
    CHK_PRT_RETURN(retVal,
        hccp_err("[init][ra_peer_rdev] notify_base_addr_uninit failed, ret(%d), phyId(%u)", retVal, rdevInfo.phyId),
        retVal);
    return ret;
}

int RaPeerRdevGetPortStatus(struct RaRdmaHandle *rdmaHandle, enum PortStatus *status)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsRdevGetPortStatus(phyId, rdmaHandle->rdevIndex, status);
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
    if (ret != 0) {
        hccp_err("[get][ra_peer_port_status]RsRdevGetPortStatus failed ret(%d) phyId(%u)", ret, phyId);
    }
    return ret;
}

int RaPeerGetLbMax(struct RaRdmaHandle *rdmaHandle, int *lbMax)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret = 0;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsGetLbMax(phyId, rdmaHandle->rdevIndex, lbMax);
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
    if (ret != 0) {
        hccp_err("[get][lbMax]RsGetLbMax failed ret:%d, phyId:%u", ret, phyId);
    }
    return ret;
}

int RaPeerRdevDeinit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType)
{
    int ret;

    hccp_info_init("[deinit][ra_peer_rdev]ra_peer_rdev_deinit phyId[%d]", rdmaHandle->rdevInfo.phyId);
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
    RsSetCtx(rdmaHandle->rdevInfo.phyId);
    ret = RsRdevDeinit(rdmaHandle->rdevInfo.phyId, notifyType, rdmaHandle->rdevIndex);
    if (ret) {
        hccp_err("[deinit][ra_peer_rdev] rs_rdev_deinit failed[%d], phyId[%u]", ret, rdmaHandle->rdevInfo.phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
        return ret;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);

    ret = NotifyBaseAddrUninit(notifyType, rdmaHandle->rdevInfo.phyId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_peer_rdev] notify_base_addr_uninit failed, ret(%d), phyId(%u)", ret,
            rdmaHandle->rdevInfo.phyId),
        ret);

    return 0;
}

int RaPeerSetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int tempDepth, unsigned int *qpNum)
{
    int ret;
    hccp_info_rma("[set][peer_set_tsqp_depth]ra_peer_set_tsqp_depth phyId[%d]", rdmaHandle->rdevInfo.phyId);
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
    RsSetCtx(rdmaHandle->rdevInfo.phyId);
    ret = RsSetTsqpDepth(rdmaHandle->rdevInfo.phyId, rdmaHandle->rdevIndex, tempDepth, qpNum);
    if (ret) {
        hccp_err("[set][peer_set_tsqp_depth] rs_set_tsqp_depth failed[%d], phyId[%u]", ret, rdmaHandle->rdevInfo.phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
        return ret;
    }

    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
    return 0;
}

int RaPeerGetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int *tempDepth, unsigned int *qpNum)
{
    int ret;

    hccp_info_rma("[get][peer_get_tsqp_depth]ra_peer_get_tsqp_depth phyId[%d]", rdmaHandle->rdevInfo.phyId);
    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
    RsSetCtx(rdmaHandle->rdevInfo.phyId);
    ret = RsGetTsqpDepth(rdmaHandle->rdevInfo.phyId, rdmaHandle->rdevIndex, tempDepth, qpNum);
    if (ret) {
        hccp_err("[get][peer_get_tsqp_depth]rs_get_tsqp_depth failed[%d], phyId[%u]", ret, rdmaHandle->rdevInfo.phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
        return ret;
    }

    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[rdmaHandle->rdevInfo.phyId]);
    return ret;
}

int RaPeerRecvWrlist(struct RaQpHandle *qpHandle, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum)
{
    int ret;
    struct RsWrlistBaseInfo baseInfo = {0};
    unsigned int completeCnt = 0;
    unsigned int recvCnt = 0;
    unsigned int recvNumPer;
    unsigned int compeletOnceCnt;

    RaInitWrlistBaseInfo(&baseInfo, qpHandle);

    while (recvCnt < recvNum) {
        recvNumPer = (recvNum - recvCnt) > MAX_WR_NUM ? MAX_WR_NUM : (recvNum - recvCnt);

        PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[baseInfo.phyId]);
        RsSetCtx(baseInfo.phyId);
        ret = RsRecvWrlist(baseInfo, &wr[recvCnt], recvNumPer, &compeletOnceCnt);
        if (ret) {
            hccp_err("[recv][peer_recv_wrlist]ra_peer_recv_wrlist failed ret[%d], recvCnt[%u], recvNumPer[%u],"
                     " phyId[%u]",
                ret, recvCnt, recvNumPer, baseInfo.phyId);
            PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[baseInfo.phyId]);
            return ret;
        }
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[baseInfo.phyId]);
        recvCnt += recvNumPer;
        completeCnt += compeletOnceCnt;
    }

    CHK_PRT_RETURN(recvCnt != completeCnt,
        hccp_err("[recv][peer_recv_wrlist]complete_cnt[%u] != recv_cnt[%u],"
                 " phyId[%u]",
            completeCnt, recvCnt, baseInfo.phyId),
        -EINVAL);

    *completeNum = completeCnt;
    return 0;
}

int RaPeerGetQpContext(struct RaQpHandle *qpPeer, void **qp, void **sendCq, void **recvCq)
{
    int ret;

    RsSetCtx(qpPeer->phyId);
    ret = RsGetQpContext(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, qp, sendCq, recvCq);
    if (ret) {
        hccp_err("[get][rs_get_qp_context]ra_peer_get_qp_context failed ret(%d), phyId(%u)", ret, qpPeer->phyId);
    }
    return ret;
}

int RaPeerCqCreate(struct RaRdmaHandle *rdmaHandle, struct CqAttr *attr)
{
    int ret;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsCqCreate(phyId, rdmaHandle->rdevIndex, attr);
    if (ret) {
        hccp_err("[create][ra_peer_cq_create]rs_cq_create failed ret[%d], phyId[%u]", ret, phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);

    return ret;
}

int RaPeerCqDestroy(struct RaRdmaHandle *rdmaHandle, struct CqAttr *attr)
{
    int ret;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsCqDestroy(phyId, rdmaHandle->rdevIndex, attr);
    if (ret) {
        hccp_err("[destroy][ra_peer_cq_destroy]rs_cq_destroy failed ret[%d], phyId[%u]", ret, phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);

    return ret;
}

int RaPeerNormalQpCreate(struct RaRdmaHandle *rdmaHandle, struct ibv_qp_init_attr *qpInitAttr, void **qpHandle,
    void **qp)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpPeer = NULL;
    struct RsQpResp qpResp = {0};
    int ret;

    qpPeer = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpPeer == NULL,
        hccp_err("[create][ra_normal_peer_qp]normal_qp_peer calloc failed, phyId[%u]", phyId), -ENOMEM);

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[phyId]);
    RsSetCtx(phyId);
    ret = RsNormalQpCreate(phyId, rdmaHandle->rdevIndex, qpInitAttr, &qpResp, qp);
    if (ret) {
        hccp_err("[create][ra_normal_peer_qp]rs_normal_qp_create failed ret[%d], phyId[%u]", ret, phyId);
        PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
        goto calloc_err;
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[phyId]);
    qpPeer->phyId = phyId;
    qpPeer->qpn = qpResp.qpn;
    qpPeer->psn = qpResp.psn;
    qpPeer->gidIdx = qpResp.gidIdx;
    qpPeer->rdevIndex = rdmaHandle->rdevIndex;
    qpPeer->rdmaHandle = rdmaHandle;
    qpPeer->rdmaOps = rdmaHandle->rdmaOps;

    *qpHandle = qpPeer;
    return ret;

calloc_err:
    free(qpPeer);
    qpPeer = NULL;
    return ret;
}

int RaPeerNormalQpDestroy(struct RaQpHandle *qpPeer)
{
    int ret;

    PEER_PTHREAD_MUTEX_LOCK(&gRaPeerMutex[qpPeer->phyId]);
    RsSetCtx(qpPeer->phyId);
    ret = RsNormalQpDestroy(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn);
    if (ret) {
        hccp_err("[destroy][ra_peer_normal_qp]ra close failed ret(%d), phyId(%u)", ret, qpPeer->phyId);
    }
    PEER_PTHREAD_MUTEX_UNLOCK(&gRaPeerMutex[qpPeer->phyId]);
    free(qpPeer);
    qpPeer = NULL;
    return ret;
}

int RaPeerSetQpAttrQos(struct RaQpHandle *qpPeer, struct QosAttr *attr)
{
    int ret;

    RsSetCtx(qpPeer->phyId);
    ret = RsSetQpAttrQos(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, attr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[ra_peer_set_qp_attr_qos]rs_set_qp_attr_qos failed ret(%d), phyId(%u)", ret, qpPeer->phyId), ret);
    return ret;
}

int RaPeerSetQpAttrTimeout(struct RaQpHandle *qpPeer, unsigned int *timeout)
{
    int ret;

    RsSetCtx(qpPeer->phyId);
    ret = RsSetQpAttrTimeout(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, timeout);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[ra_peer_set_qp_attr_timeout]rs_set_qp_attr_timeout failed ret(%d), phyId(%u)", ret, qpPeer->phyId),
        ret);
    return ret;
}

int RaPeerSetQpAttrRetryCnt(struct RaQpHandle *qpPeer, unsigned int *retryCnt)
{
    int ret;
    RsSetCtx(qpPeer->phyId);
    ret = RsSetQpAttrRetryCnt(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, retryCnt);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[ra_peer_set_qp_attr_retry_cnt]rs_set_qp_attr_retry_cnt failed ret(%d), phyId(%u)", ret,
            qpPeer->phyId),
        ret);
    return ret;
}

int RaPeerCreateCompChannel(struct RaRdmaHandle *rdmaHandle, void **compChannel)
{
    int ret;
    RsSetCtx(rdmaHandle->rdevInfo.phyId);
    ret = RsCreateCompChannel(rdmaHandle->rdevInfo.phyId, rdmaHandle->rdevIndex, compChannel);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[ra_peer_create_comp_channel]rs_create_comp_channel failed ret(%d), phyId(%u)", ret,
            rdmaHandle->rdevInfo.phyId),
        ret);

    return ret;
}

int RaPeerDestroyCompChannel(void *compChannel)
{
    int ret;

    ret = RsDestroyCompChannel(compChannel);
    CHK_PRT_RETURN(ret != 0, hccp_err("[ra_peer_destroy_comp_channel]rs_destroy_comp_channel failed ret(%d)", ret),
        ret);

    return ret;
}

int RaPeerCreateSrq(struct RaRdmaHandle *rdmaHandle, struct SrqAttr *attr)
{
    int ret;

    // 创建srq&srq cq
    RsSetCtx(rdmaHandle->rdevInfo.phyId);
    ret = RsCreateSrq(rdmaHandle->rdevInfo.phyId, rdmaHandle->rdevIndex, attr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[ra_peer_create_srq]rs_create_srq failed ret(%d), phyId(%u)", ret, rdmaHandle->rdevInfo.phyId), ret);

    return ret;
}

int RaPeerDestroySrq(struct RaRdmaHandle *rdmaHandle, struct SrqAttr *attr)
{
    int ret;

    // 销毁srq&srq cq
    RsSetCtx(rdmaHandle->rdevInfo.phyId);
    ret = RsDestroySrq(rdmaHandle->rdevInfo.phyId, rdmaHandle->rdevIndex, attr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[ra_peer_destroy_srq]rs_destroy_srq failed ret(%d), phyId(%u)", ret, rdmaHandle->rdevInfo.phyId),
        ret);

    return ret;
}

int RaPeerCreateEventHandle(int *eventHandle)
{
    int ret;

    ret = RsCreateEventHandle(eventHandle);
    CHK_PRT_RETURN(ret != 0, hccp_err("[ra_peer_create_event_handle]rs_create_event_handle failed ret(%d)", ret), ret);

    return ret;
}

int RaPeerCtlEventHandle(int eventHandle, const void *fdHandle, int opcode, enum RaEpollEvent event)
{
    int ret;

    ret = RsCtlEventHandle(eventHandle, fdHandle, opcode, event);
    CHK_PRT_RETURN(ret != 0, hccp_err("[ra_peer_ctl_event_handle]rs_ctl_event_handle failed ret(%d)", ret), ret);

    return ret;
}

int RaPeerWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout, unsigned int maxevents,
    unsigned int *eventsNum)
{
    int ret;

    ret = RsWaitEventHandle(eventHandle, eventInfos, timeout, maxevents, eventsNum);
    CHK_PRT_RETURN(ret != 0, hccp_err("[ra_peer_wait_event_handle]rs_wait_event_handle failed ret(%d)", ret), ret);

    return ret;
}

int RaPeerDestroyEventHandle(int *eventHandle)
{
    int ret;

    ret = RsDestroyEventHandle(eventHandle);
    CHK_PRT_RETURN(ret != 0, hccp_err("[ra_peer_destroy_event_handle]rs_destroy_event_handle failed ret(%d)", ret),
        ret);

    return ret;
}
