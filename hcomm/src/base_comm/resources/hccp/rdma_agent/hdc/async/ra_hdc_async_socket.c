/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "ra.h"
#include "ra_comm.h"
#include "ra_async.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "ra_hdc.h"
#include "ra_hdc_socket.h"
#include "ra_hdc_async.h"
#include "ra_hdc_async_socket.h"

int RaHdcSocketSendAsync(const struct SocketHdcInfo *fdHandle, const void *data, unsigned long long size,
    unsigned long long *sentSize, void **reqHandle)
{
    unsigned long long sendSize = (size > SOCKET_SEND_MAXLEN) ? SOCKET_SEND_MAXLEN : size;
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpSocketSendData *asyncData = NULL;
    unsigned int phyId = fdHandle->phyId;
    int ret = 0;

    asyncData = (union OpSocketSendData *)calloc(sizeof(union OpSocketSendData), sizeof(char));
    CHK_PRT_RETURN(asyncData == NULL, hccp_err("[send][ra_hdc_socket]calloc async_data failed, phyId(%u)", phyId),
        -ENOMEM);

    asyncData->txData.fd = (unsigned int)fdHandle->fd;
    asyncData->txData.sendSize = sendSize;
    ret = memcpy_s(asyncData->txData.dataSend, SOCKET_SEND_MAXLEN, data, sendSize);
    if (ret != 0) {
        hccp_err("[send][ra_hdc_socket]memcpy_s data failed, ret(%d) sendSize(%llu) phyId(%u)", ret, sendSize, phyId);
        ret = -ESAFEFUNC;
        goto out;
    }

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[send][ra_hdc_socket]calloc req_handle_tmp failed, phyId[%u]", phyId);
        ret = -ENOMEM;
        goto out;
    }
    *sentSize = 0;
    reqHandleTmp->privData = (void *)sentSize;

    ret = RaHdcSendMsgAsync(RA_RS_SOCKET_SEND, phyId, (char *)asyncData, sizeof(union OpSocketSendData), reqHandleTmp);
    if (ret != 0) {
        hccp_err("[send][ra_hdc_socket]hdc async send message process failed ret(%d) phyId(%u)", ret, phyId);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }

    *reqHandle = (void *)reqHandleTmp;

out:
    free(asyncData);
    asyncData = NULL;
    return ret;
}

void RaHdcAsyncHandleSocketSend(struct RaRequestHandle *reqHandle)
{
    union OpSocketSendData *asyncData = NULL;

    if (reqHandle->opRet > 0) {
        asyncData = (union OpSocketSendData *)reqHandle->recvBuf;
        *(unsigned long long *)reqHandle->privData = asyncData->rxData.realSendSize;
        reqHandle->opRet = 0;
    } else if (reqHandle->opRet == 0) {
        hccp_warn_socket("[send][ra_hdc_socket]socket has been closed. sent_size is 0");
        *(unsigned long long *)reqHandle->privData = 0;
        reqHandle->opRet = -ESOCKCLOSED;
    } else {
        if (reqHandle->opRet != -EAGAIN) {
            hccp_warn_socket("[send][ra_hdc_socket]socket send unsuccessful ret(%d) phyId(%u)", reqHandle->opRet,
                reqHandle->phyId);
        }
        *(unsigned long long *)reqHandle->privData = 0;
    }

    return;
}

STATIC void RaHdcSocketPrepareRecvRsp(struct RaResponseSocketRecv *recvRsp, void *data, unsigned long long size,
    unsigned long long *receivedSize)
{
    recvRsp->data = data;
    recvRsp->size = size;
    *receivedSize = 0;
    recvRsp->receivedSize = receivedSize;
}

int RaHdcSocketRecvAsync(const struct SocketHdcInfo *fdHandle, void *data, unsigned long long size,
    unsigned long long *receivedSize, void **reqHandle)
{
    unsigned long long recvSize = (size > SOCKET_SEND_MAXLEN) ? SOCKET_SEND_MAXLEN : size;
    struct RaResponseSocketRecv *recvRsp = NULL;
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpSocketRecvData *asyncData = NULL;
    unsigned int phyId = fdHandle->phyId;
    int ret = 0;

    recvRsp = (struct RaResponseSocketRecv *)calloc(1, sizeof(struct RaResponseSocketRecv));
    CHK_PRT_RETURN(recvRsp == NULL, hccp_err("[recv][ra_hdc_socket]calloc recv_rsp failed, phyId(%u)", phyId), -ENOMEM);
    RaHdcSocketPrepareRecvRsp(recvRsp, data, recvSize, receivedSize);

    asyncData = (union OpSocketRecvData *)calloc(sizeof(union OpSocketRecvData) + recvSize, sizeof(char));
    if (asyncData == NULL) {
        hccp_err("[recv][ra_hdc_socket]calloc async_data failed, phyId(%u)", phyId);
        ret = -ENOMEM;
        goto free_recv_rsp;
    }

    asyncData->txData.fd = (unsigned int)fdHandle->fd;
    asyncData->txData.recvSize = recvSize;
    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[recv][ra_hdc_socket]calloc req_handle_tmp failed, phyId[%u]", phyId);
        ret = -ENOMEM;
        goto out;
    }
    reqHandleTmp->privData = (void *)recvRsp;
    ret = RaHdcSendMsgAsync(RA_RS_SOCKET_RECV, phyId, (char *)asyncData,
        (unsigned int)(sizeof(union OpSocketRecvData) + recvSize), reqHandleTmp);
    if (ret != 0) {
        hccp_err("[recv][ra_hdc_socket]hdc async send message process failed ret(%d) phyId(%u)", ret, phyId);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }

    free(asyncData);
    asyncData = NULL;
    *reqHandle = (void *)reqHandleTmp;
    return 0;

out:
    free(asyncData);
    asyncData = NULL;
free_recv_rsp:
    free(recvRsp);
    recvRsp = NULL;
    return ret;
}

void RaHdcAsyncHandleSocketRecv(struct RaRequestHandle *reqHandle)
{
    struct RaResponseSocketRecv *recvRsp = NULL;
    union OpSocketRecvData *asyncData = NULL;
    unsigned long long realRecvSize = 0;
    unsigned int phyId = 0;
    int ret = 0;

    phyId = reqHandle->phyId;
    if (reqHandle->opRet == 0) {
        hccp_warn_socket("[recv][ra_hdc_socket]socket has been closed. received_size is 0");
        reqHandle->opRet = -ESOCKCLOSED;
        goto out;
    } else if (reqHandle->opRet < 0) {
        if (reqHandle->opRet != -EAGAIN) {
            hccp_warn_socket("[recv][ra_hdc_socket]socket recv ret(%d) phyId(%u)", reqHandle->opRet, phyId);
        }
        goto out;
    }

    asyncData = (union OpSocketRecvData *)reqHandle->recvBuf;
    realRecvSize = asyncData->rxData.realRecvSize;
    if (realRecvSize > SOCKET_SEND_MAXLEN) {
        hccp_err("[recv][ra_hdc_socket]real_recv_size:%llu invalid, phyId(%u)", realRecvSize, phyId);
        reqHandle->opRet = -EINVAL;
        goto out;
    }

    recvRsp = (struct RaResponseSocketRecv *)reqHandle->privData;
    ret = memcpy_s(recvRsp->data, recvRsp->size, (char *)asyncData + sizeof(union OpSocketRecvData), realRecvSize);
    if (ret != 0) {
        hccp_err("[recv][ra_hdc_socket]memcpy_s failed, ret(%d) phyId(%u) size(%llu) realRecvSize(%llu)", ret, phyId,
            recvRsp->size, realRecvSize);
        reqHandle->opRet = -ESAFEFUNC;
        goto out;
    }

    reqHandle->opRet = 0;
    *recvRsp->receivedSize = realRecvSize;

out:
    free(reqHandle->privData);
    reqHandle->privData = NULL;
    return;
}

int RaHdcSocketListenStartAsync(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num, void **reqHandle)
{
    struct RaResponseSocketListen *asyncRsp = NULL;
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpSocketListenData asyncData = {0};
    int ret = 0;

    ret = RaGetSocketListenInfo(conn, num, asyncData.txData.conn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_start][ra_hdc_socket]get_socket_listen_info failed, ret(%d) phyId(%u)", ret, phyId), -EINVAL);
    asyncData.txData.num = num | (1U << SOCKET_USE_PORT_BIT);

    asyncRsp = (struct RaResponseSocketListen *)calloc(1, sizeof(struct RaResponseSocketListen));
    CHK_PRT_RETURN(asyncRsp == NULL, hccp_err("[listen_start][ra_hdc_socket]calloc async_rsp failed, phyId(%u)", phyId),
        -ENOMEM);
    asyncRsp->conn = conn;
    asyncRsp->num = num;
    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[listen_start][ra_hdc_socket]calloc RaRequestHandle failed, phyId[%u]", phyId);
        ret = -ENOMEM;
        goto out;
    }
    reqHandleTmp->privData = (void *)asyncRsp;

    ret = RaHdcSendMsgAsync(RA_RS_SOCKET_LISTEN_START, phyId, (char *)&asyncData, sizeof(union OpSocketListenData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[listen_start][ra_hdc_socket]hdc async send message process failed ret(%d) phyId(%u)", ret, phyId);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }
    *reqHandle = (void *)reqHandleTmp;
    return 0;

out:
    free(asyncRsp);
    asyncRsp = NULL;
    return ret;
}

void RaHdcAsyncHandleSocketListenStart(struct RaRequestHandle *reqHandle)
{
    struct RaResponseSocketListen *asyncRsp = NULL;
    union OpSocketListenData *asyncData = NULL;
    unsigned int phyId = reqHandle->phyId;
    int ret = 0;

    asyncData = (union OpSocketListenData *)reqHandle->recvBuf;
    asyncRsp = (struct RaResponseSocketListen *)reqHandle->privData;
    ret = RaGetSocketListenResult(asyncData->rxData.conn, asyncRsp->num, asyncRsp->conn, MAX_SOCKET_NUM);
    if (ret != 0) {
        hccp_err("[listen_start][ra_hdc_socket]ra_get_socket_listen_result failed, ret(%d) phyId(%u)", ret, phyId);
        reqHandle->opRet = -EINVAL;
        goto out;
    }
    return;

out:
    free(reqHandle->privData);
    reqHandle->privData = NULL;
    return;
}

int RaHdcSocketListenStopAsync(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num, void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpSocketListenData asyncData = {0};
    int ret = 0;

    ret = RaGetSocketListenInfo(conn, num, asyncData.txData.conn, MAX_SOCKET_NUM);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[listen_stop][ra_hdc_socket]get_socket_listen_info failed, ret(%d) phyId(%u)", ret, phyId), -EINVAL);
    asyncData.txData.num = num | (1U << SOCKET_USE_PORT_BIT);

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[listen_stop][ra_hdc_socket]calloc RaRequestHandle failed, phyId[%u]", phyId), -ENOMEM);

    ret = RaHdcSendMsgAsync(RA_RS_SOCKET_LISTEN_STOP, phyId, (char *)&asyncData, sizeof(union OpSocketListenData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[listen_stop][ra_hdc_socket]hdc async send message process failed ret(%d) phyId(%u)", ret, phyId);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

int RaHdcSocketBatchConnectAsync(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num,
    void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpSocketConnectData *asyncData = NULL;
    int ret = 0;

    asyncData = (union OpSocketConnectData *)calloc(sizeof(union OpSocketConnectData), sizeof(char));
    CHK_PRT_RETURN(asyncData == NULL,
        hccp_err("[batch_connect][ra_hdc_socket]calloc async_data failed, phyId(%u)", phyId), -ENOMEM);

    asyncData->txData.num = num | (1U << SOCKET_USE_PORT_BIT);
    ret = RaGetSocketConnectInfo(conn, num, asyncData->txData.conn, MAX_SOCKET_NUM);
    if (ret != 0) {
        hccp_err("[batch_connect][ra_hdc_socket]ra_get_socket_connect_info failed, ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[batch_connect][ra_hdc_socket]calloc RaRequestHandle failed, phyId[%u]", phyId);
        ret = -ENOMEM;
        goto out;
    }

    ret = RaHdcSendMsgAsync(RA_RS_SOCKET_CONN, phyId, (char *)asyncData, sizeof(union OpSocketConnectData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[batch_connect][ra_hdc_socket]hdc async send message process failed ret(%d) phyId(%u)", ret, phyId);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }

    *reqHandle = (void *)reqHandleTmp;

out:
    free(asyncData);
    asyncData = NULL;
    return ret;
}

int RaHdcSocketBatchCloseAsync(unsigned int phyId, struct SocketCloseInfoT conn[], unsigned int num, void **reqHandle)
{
    struct RaResponseSocketBatchClose *asyncRsp = NULL;
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpSocketCloseData asyncData = {0};
    unsigned int i;
    int ret = 0;

    for (i = 0; i < num; i++) {
        if (conn[i].fdHandle == NULL) {
            hccp_err("[batch_close][ra_hdc_socket]i(%u), conn fdHandle is NULL", i);
            ret = -EINVAL;
            goto out;
        }
        asyncData.txData.conn[i].phyId = phyId;
        asyncData.txData.conn[i].closeFd = ((struct SocketHdcInfo *)conn[i].fdHandle)->fd;
    }
    // use attr disuse_linger of the fist conn as the common attr for all(0 by default)
    asyncData.txData.num = (conn[0].disuseLinger != 0) ? (num | (1U << SOCKET_DISUSE_LINGER_BIT)) : num;

    asyncRsp = (struct RaResponseSocketBatchClose *)calloc(1, sizeof(struct RaResponseSocketBatchClose));
    CHK_PRT_RETURN(asyncRsp == NULL, hccp_err("[batch_close][ra_hdc_socket]calloc async_rsp failed, phyId(%u)", phyId),
        -ENOMEM);
    asyncRsp->conn = conn;
    asyncRsp->num = num;

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[batch_close][ra_hdc_socket]calloc RaRequestHandle failed, phyId[%u]", phyId);
        ret = -ENOMEM;
        goto out;
    }

    reqHandleTmp->privData = (void *)asyncRsp;

    ret = RaHdcSendMsgAsync(RA_RS_SOCKET_CLOSE, phyId, (char *)&asyncData, sizeof(union OpSocketCloseData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[batch_close][ra_hdc_socket]hdc async send message process failed, ret(%d) phyId(%u)", ret, phyId);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;

out:
    free(asyncRsp);
    asyncRsp = NULL;
    return ret;
}

void RaHdcAsyncHandleSocketBatchClose(struct RaRequestHandle *reqHandle)
{
    struct RaResponseSocketBatchClose *asyncRsp = NULL;
    unsigned int i;

    // should free fd_handle when op_ret is not EAGAIN, otherwise caller will retry
    if (reqHandle->opRet == -EAGAIN) {
        return;
    }

    asyncRsp = (struct RaResponseSocketBatchClose *)reqHandle->privData;
    for (i = 0; i < asyncRsp->num; i++) {
        if (asyncRsp->conn[i].fdHandle != NULL) {
            free(asyncRsp->conn[i].fdHandle);
            asyncRsp->conn[i].fdHandle = NULL;
        }
    }
    return;
}