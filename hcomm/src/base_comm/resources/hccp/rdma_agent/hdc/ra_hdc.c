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
#include "ra_hdc.h"
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "ra.h"
#include "ra_comm.h"
#include "ra_hdc_lite.h"
#include "ra_rs_err.h"
#include "ra_rs_comm.h"
#include "ra_hdc_async.h"

struct HdcInfo gRaHdc[RA_MAX_PHY_ID_NUM] = {0};

struct HdcOps gRaHdcOpsHost = {
    .getCapacity = DlDrvHdcGetCapacity,
    .clientCreate = DlDrvHdcClientCreate,
    .clientDestroy = DlDrvHdcClientDestroy,
    .sessionConnect = DlDrvHdcSessionConnect,
    .sessionConnectEx = DlHalHdcSessionConnectEx,
    .serverCreate = DlDrvHdcServerCreate,
    .serverDestroy = DlDrvHdcServerDestroy,
    .sessionAccept = DlDrvHdcSessionAccept,
    .sessionClose = DlDrvHdcSessionClose,
    .freeMsg = DlDrvHdcFreeMsg,
    .reuseMsg = DlDrvHdcReuseMsg,
    .addMsgBuffer = DlDrvHdcAddMsgBuffer,
    .getMsgBuffer = DlDrvHdcGetMsgBuffer,
    .recv = DlHalHdcRecv,
    .send = DlHalHdcSend,
    .allocMsg = DlDrvHdcAllocMsg,
    .setSessionReference = DlDrvHdcSetSessionReference,
};

#define RA_HDC_OPS gRaHdcOpsHost

struct OpcodeInterfaceInfo gRaInterfaceInfoList[] = {
    // outer opcode version: 1.0
    {RA_RS_SOCKET_CONN, 0},
    {RA_RS_SOCKET_CLOSE, 0},
    {RA_RS_SOCKET_ABORT, 0},
    {RA_RS_SOCKET_LISTEN_START, 0},
    {RA_RS_SOCKET_LISTEN_STOP, 0},
    {RA_RS_GET_SOCKET, 0},
    {RA_RS_SOCKET_SEND, 0},
    {RA_RS_SOCKET_RECV, 0},
    {RA_RS_QP_CREATE, 0},
    {RA_RS_QP_CREATE_WITH_ATTRS, 0},
    {RA_RS_AI_QP_CREATE, 0},
    {RA_RS_AI_QP_CREATE_WITH_ATTRS, 0},
    {RA_RS_TYPICAL_QP_CREATE, 0},
    {RA_RS_QP_DESTROY, 0},
    {RA_RS_QP_DESTROY_WITHOUT_CQ, 0},
    {RA_RS_QP_CONNECT, 0},
    {RA_RS_TYPICAL_QP_MODIFY, 0},
    {RA_RS_QP_STATUS, 0},
    {RA_RS_QP_INFO, 0},
    {RA_RS_QP_BATCH_MODIFY, 0},
    {RA_RS_MR_REG, 0},
    {RA_RS_MR_DEREG, 0},
    {RA_RS_TYPICAL_MR_REG_V1, 0},
    {RA_RS_TYPICAL_MR_REG, 0},
    {RA_RS_REMAP_MR, 0},
    {RA_RS_TYPICAL_MR_DEREG, 0},
    {RA_RS_SEND_WR, 0},
    {RA_RS_GET_NOTIFY_BA, 0},
    {RA_RS_INIT, 0},
    {RA_RS_DEINIT, 0},
    {RA_RS_SOCKET_INIT, 0},
    {RA_RS_SOCKET_DEINIT, 0},
    {RA_RS_RDEV_INIT, 0},
    {RA_RS_RDEV_INIT_WITH_BACKUP, 0},
    {RA_RS_RDEV_GET_PORT_STATUS, 0},
    {RA_RS_RDEV_DEINIT, 0},
    {RA_RS_WLIST_ADD, 0},
    {RA_RS_WLIST_ADD_V2, 0},
    {RA_RS_WLIST_DEL, 0},
    {RA_RS_WLIST_DEL_V2, 0},
    {RA_RS_ACCEPT_CREDIT_ADD, 0},
    {RA_RS_GET_IFADDRS, 0},
    {RA_RS_GET_IFADDRS_V2, 0},
    {RA_RS_GET_INTERFACE_VERSION, 0},
    {RA_RS_SEND_WRLIST, 0},
    {RA_RS_SEND_WRLIST_V2, 0},
    {RA_RS_SEND_WRLIST_EXT, 0},
    {RA_RS_SEND_WRLIST_EXT_V2, 0},
    {RA_RS_SEND_NORMAL_WRLIST, 0},
    {RA_RS_SET_TSQP_DEPTH, 0},
    {RA_RS_GET_TSQP_DEPTH, 0},
    {RA_RS_SET_QP_ATTR_QOS, 0},
    {RA_RS_SET_QP_ATTR_TIMEOUT, 0},
    {RA_RS_SET_QP_ATTR_RETRY_CNT, 0},
    {RA_RS_GET_CQE_ERR_INFO, 0},
    {RA_RS_GET_CQE_ERR_INFO_NUM, 0},
    {RA_RS_GET_CQE_ERR_INFO_LIST, 0},
    {RA_RS_GET_LITE_SUPPORT, 0},
    {RA_RS_GET_LITE_RDEV_CAP, 0},
    {RA_RS_GET_LITE_QP_CQ_ATTR, 0},
    {RA_RS_GET_LITE_QP_ATTR, 0},
    {RA_RS_GET_LITE_CONNECTED_INFO, 0},
    {RA_RS_GET_LITE_MEM_ATTR, 0},
    {RA_RS_PING_INIT, 0},
    {RA_RS_PING_ADD, 0},
    {RA_RS_PING_START, 0},
    {RA_RS_PING_GET_RESULTS, 0},
    {RA_RS_PING_STOP, 0},
    {RA_RS_PING_DEL, 0},
    {RA_RS_PING_DEINIT, 0},
    {RA_RS_GET_VNIC_IP_INFOS_V1, 0},
    {RA_RS_GET_VNIC_IP_INFOS, 0},
    {RA_RS_TLV_INIT_V1, 0},
    {RA_RS_TLV_INIT, 0},
    {RA_RS_TLV_DEINIT, 0},
    {RA_RS_TLV_REQUEST, 0},
    {RA_RS_TLV_REQUEST_V2, 0},
    {RA_RS_GET_TLS_ENABLE, 0},
    {RA_RS_GET_SEC_RANDOM, 0},
    {RA_RS_GET_HCCN_CFG, 0},
    {RA_RS_GET_ROCE_API_VERSION, 0},
    {RA_RS_TYPICAL_CQ_CREATE, 0},
    {RA_RS_GET_LITE_CQ_ATTR, 0},
    {RA_RS_QP_CREATE_WITH_CQ_WITH_ATTRS, 0},
    {RA_RS_TYPICAL_CQ_DESTROY, 0},

    // outer opcode version: 2.0
    {RA_RS_GET_DEV_EID_INFO_NUM, 0},
    {RA_RS_GET_DEV_EID_INFO_LIST, 0},
    {RA_RS_CTX_INIT, 0},
    {RA_RS_CTX_GET_ASYNC_EVENTS, 0},
    {RA_RS_CTX_DEINIT, 0},
    {RA_RS_GET_EID_BY_IP, 0},
    {RA_RS_GET_TP_INFO_LIST, 0},
    {RA_RS_GET_TP_ATTR, 0},
    {RA_RS_SET_TP_ATTR, 0},
    {RA_RS_CTX_TOKEN_ID_ALLOC, 0},
    {RA_RS_CTX_TOKEN_ID_FREE, 0},
    {RA_RS_LMEM_REG, 0},
    {RA_RS_LMEM_UNREG, 0},
    {RA_RS_RMEM_IMPORT, 0},
    {RA_RS_RMEM_UNIMPORT, 0},
    {RA_RS_CTX_CHAN_CREATE, 0},
    {RA_RS_CTX_CHAN_DESTROY, 0},
    {RA_RS_CTX_CQ_CREATE, 0},
    {RA_RS_CTX_CQ_DESTROY, 0},
    {RA_RS_CTX_QP_CREATE, 0},
    {RA_RS_CTX_QUERY_QP_BATCH, 0},
    {RA_RS_CTX_QP_DESTROY, 0},
    {RA_RS_CTX_QP_DESTROY_BATCH, 0},
    {RA_RS_CTX_QP_IMPORT, 0},
    {RA_RS_CTX_QP_UNIMPORT, 0},
    {RA_RS_CTX_QP_BIND, 0},
    {RA_RS_CTX_QP_UNBIND, 0},
    {RA_RS_CTX_BATCH_SEND_WR, 0},
    {RA_RS_CTX_UPDATE_CI, 0},
    {RA_RS_CTX_GET_AUX_INFO, 0},
    {RA_RS_CTX_GET_CR_ERR_INFO_LIST, 0},
    {RA_RS_CTX_GET_UB_CONTEXT, 0},
    {RA_RS_CTX_NOTIFY_EVENT, 0},
    {RA_RS_GET_NET_API_VERSION, 0},

    // inner opcode version
    {RA_RS_HDC_SESSION_CLOSE, 0},
    {RA_RS_GET_VNIC_IP, 0},
    {RA_RS_NOTIFY_CFG_SET, 0},
    {RA_RS_NOTIFY_CFG_GET, 0},
    {RA_RS_SET_PID, 0},
    {RA_RS_ASYNC_HDC_SESSION_CONNECT, 0},
    {RA_RS_ASYNC_HDC_SESSION_CLOSE, 0},
    {RA_RS_SET_DEBUG_CONFIG, 0},
};

STATIC int MsgHeadCheck(struct MsgHead *sendRcvHead, unsigned int opcode, int rsRet, unsigned int msgDataLen);

static int HdcSendRecvPktSend(struct drvHdcMsg *pMsgSnd, char *sendRcvBuf, unsigned int inBufLen, HDC_SESSION session,
    struct drvHdcMsg **pMsgRcv)
{
    int ret;
    ret = RA_HDC_OPS.addMsgBuffer(pMsgSnd, sendRcvBuf, inBufLen);
    CHK_PRT_RETURN(ret != 0, hccp_err("[send][hdc_send_recv_pkt]HDC add msg buffer err ret(%d)", ret), ret);

    ret = RA_HDC_OPS.send(session, pMsgSnd, RA_HDC_WAIT_TIMEOUT, RA_HDC_RECV_SEND_TIMEOUT);
    CHK_PRT_RETURN(ret != 0, hccp_err("[send][hdc_send_recv_pkt]HDC send err ret(%d)", ret), ret);

    ret = RA_HDC_OPS.reuseMsg(pMsgSnd);
    CHK_PRT_RETURN(ret != 0, hccp_err("[send][hdc_send_recv_pkt]HDC reuse msg err ret(%d)", ret), ret);

    *pMsgRcv = pMsgSnd;
    return 0;
}

#ifndef HNS_ROCE_LLT
STATIC int HdcSendRetryPkt(unsigned int phyId, void *sendRcvBuf, unsigned int inBufLen, struct drvHdcMsg **pMsgRcv)
{
    int ret;
    struct drvHdcMsg *pMsgSnd = NULL;

    HDC_SESSION session = gRaHdc[phyId].session;
    if (session == NULL) {
        hccp_err("[send_recv][pkt]session is NULL!, phyId(%u)", phyId);
        return -EINVAL;
    }

    ret = RA_HDC_OPS.allocMsg(session, &pMsgSnd, 1);
    if (ret != 0) {
        hccp_err("[send_recv][pkt]HDC alloc msg err ret(%d) phyId(%u)", ret, phyId);
        return ret;
    }
    ret = HdcSendRecvPktSend(pMsgSnd, (char *)sendRcvBuf, inBufLen, session, pMsgRcv);
    if (ret != 0) {
        hccp_err("[send_recv][pkt]HDC pkt send err ret(%d) phyId(%u)", ret, phyId);
        goto msg_err;
    }

    return 0;

msg_err:
    RA_HDC_OPS.freeMsg(pMsgSnd);
    return ret;
}

STATIC int RaHdcSendRetryMsg(unsigned int phyId, struct drvHdcMsg **pMsgRcv)
{
    int ret;
    void *sendRcvBuf = NULL;
    unsigned int sendRcvLen;
    union OpIfnumData ifnumData;
    ifnumData.txData.phyId = phyId;

    pid_t hostTgid = DlDrvDeviceGetBareTgid();
    unsigned int dataSize = sizeof(union OpIfnumData);
    sendRcvLen = sizeof(struct MsgHead) + dataSize;
    sendRcvBuf = (void *)calloc(sendRcvLen, sizeof(char));
    CHK_PRT_RETURN(sendRcvBuf == NULL, hccp_err("[process][ra_hdc_msg]send_rcv_buf calloc failed. phyId(%u)", phyId),
        -ENOMEM);
    MsgHeadBuildUp(sendRcvBuf, RA_RS_GET_IFNUM, 0, dataSize, hostTgid);

    ret = memcpy_s(sendRcvBuf + sizeof(struct MsgHead), sendRcvLen - sizeof(struct MsgHead), &ifnumData, dataSize);
    if (ret) {
        hccp_err("[process][ra_hdc_msg]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId);
        ret = -ESAFEFUNC;
        goto out;
    }

    ret = HdcSendRetryPkt(phyId, sendRcvBuf, sendRcvLen, pMsgRcv);
    if (ret) {
        hccp_err("[process][ra_hdc_msg]hdc_send_recv_pkt failed ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

out:
    free(sendRcvBuf);
    sendRcvBuf = NULL;
    return ret;
}

static int RaHdcRecvRetryMsg(HDC_SESSION session, struct drvHdcMsg *pMsgRcv)
{
    int outBufLen = sizeof(struct MsgHead) + sizeof(union OpIfnumData);
    char *recvBuf = NULL;
    int recvBufCnt = 0;
    int rcvBufLen = 0;
    int ret;

    ret = RA_HDC_OPS.recv(session, pMsgRcv, MAX_HDC_DATA, RA_HDC_WAIT_TIMEOUT, &recvBufCnt, RA_HDC_RETRY_SEND_TIMEOUT);
    if (ret) {
        hccp_err("[recv][ra_hdc_recv_retry_msg]HDC get retry recv msg failed(%d)", ret);
        return ret;
    }

    ret = RA_HDC_OPS.getMsgBuffer(pMsgRcv, 0, &recvBuf, &rcvBufLen);
    if (ret) {
        hccp_err("[recv][ra_hdc_recv_retry_msg]HDC get retry msg buffer failed(%d)", ret);
        return ret;
    }

    ret = MsgHeadCheck((struct MsgHead *)recvBuf, RA_RS_GET_IFNUM, 0, sizeof(union OpIfnumData));
    if (rcvBufLen != outBufLen || ret != 0) {
        hccp_err("[recv][ra_hdc_recv_retry_msg]HDC get retry recv msg failed, ret(%d), rcvBufLen:%d, outBufLen:%d", ret,
            rcvBufLen, outBufLen);
        if (rcvBufLen != outBufLen) {
            ret = -EPIPE;
        }
        return ret;
    }

    return 0;
}
#endif

static int HdcSendRecvPktRecv(HDC_SESSION session, unsigned int phyId, struct drvHdcMsg *pMsgRcv, char **recvBuf,
    int *rcvBufLen)
{
    struct drvHdcMsg *pRetryRcv = NULL;
    int recvBufCnt = 0;
    int ret;

    ret = RA_HDC_OPS.recv(session, pMsgRcv, MAX_HDC_DATA, RA_HDC_WAIT_TIMEOUT, &recvBufCnt, RA_HDC_RECV_SEND_TIMEOUT);
#ifndef HNS_ROCE_LLT
    /* if timeout, start retry */
    if (gRaHdc[phyId].startDeinit == 0 && ret == -DRV_ERROR_WAIT_TIMEOUT) {
        hccp_run_info("[recv][hdc_send_recv_pkt_recv]HDC recv timeout, start retry");
        ret = RaHdcSendRetryMsg(phyId, &pRetryRcv);
        CHK_PRT_RETURN(ret != 0, hccp_err("[recv][hdc_send_recv_pkt_recv]HDC get msg by first retry failed(%d)", ret),
            ret);

        ret = RA_HDC_OPS.recv(session, pMsgRcv, MAX_HDC_DATA, RA_HDC_WAIT_TIMEOUT, &recvBufCnt,
            RA_HDC_RECV_SEND_TIMEOUT);
        if (ret) {
            hccp_err("[recv][hdc_send_recv_pkt_recv]HDC get msg by first retry failed(%d)", ret);
            RA_HDC_OPS.freeMsg(pRetryRcv);
            return ret;
        }

        ret = RA_HDC_OPS.getMsgBuffer(pMsgRcv, 0, (char **)recvBuf, rcvBufLen);
        if (ret) {
            hccp_err("[recv][hdc_send_recv_pkt]HDC get_msg_buffer msg err ret(%d), rcvBufLen(%d)", ret, *rcvBufLen);
            RA_HDC_OPS.freeMsg(pRetryRcv);
            return ret;
        }

        ret = RaHdcRecvRetryMsg(session, pRetryRcv);
        if (ret) {
            hccp_err("[recv][hdc_send_recv_pkt_recv]HDC recv first retry msg failed(%d)", ret);
        }

        RA_HDC_OPS.freeMsg(pRetryRcv);
        return ret;
    }
#endif
    CHK_PRT_RETURN(ret != 0, hccp_err("[recv][hdc_send_recv_pkt]HDC recv msg err ret(%d)", ret), ret);

    ret = RA_HDC_OPS.getMsgBuffer(pMsgRcv, 0, (char **)recvBuf, rcvBufLen);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[recv][hdc_send_recv_pkt]HDC get_msg_buffer msg err ret(%d), rcvBufLen(%d)", ret, *rcvBufLen), ret);

    return 0;
}

STATIC int HdcSendRecvPktRecvCheck(int rcvBufLen, unsigned int outDataLen, struct MsgHead *recvMsgHead,
    struct drvHdcMsg *pMsgRcv)
{
    unsigned int rcvBufLenTmp;

    rcvBufLenTmp = (unsigned int)rcvBufLen;
    if (outDataLen != rcvBufLenTmp) {
        if (recvMsgHead->ret == -EACCES) {
            hccp_warn_init("exceed the speed limit, need try again, ret:%d", recvMsgHead->ret);
            RA_HDC_OPS.freeMsg(pMsgRcv);
            return -EAGAIN;
        } else if (recvMsgHead->ret == -EPROTONOSUPPORT) {
            hccp_err("[check][hdc_send_recv_pkt_recv]unsupported opcode, ret(%d)", recvMsgHead->ret);
            RA_HDC_OPS.freeMsg(pMsgRcv);
            return -EPROTONOSUPPORT;
        } else if (recvMsgHead->ret == -EPERM) {
            hccp_err("[check][hdc_send_recv_pkt_recv]host pid is invalid, ret(%d)", recvMsgHead->ret);
            RA_HDC_OPS.freeMsg(pMsgRcv);
            return -EPERM;
        }
        hccp_err("[check][hdc_send_recv_pkt_recv]data len err out_data_len(%d) != rcv_buf_len(%d) ", outDataLen,
            rcvBufLen);
        RA_HDC_OPS.freeMsg(pMsgRcv);
        return -EPIPE;
    }
    return 0;
}

STATIC int HdcSendRecvPkt(unsigned int phyId, void *sendRcvBuf, unsigned int inBufLen, unsigned int outBufLen)
{
    int ret;
    char *recvBuf = NULL;
    struct drvHdcMsg *pMsgRcv = NULL, *pMsgSnd = NULL;
    int rcvBufLen = 0;

    struct MsgHead *recvMsgHead = NULL;

    RA_PTHREAD_MUTEX_LOCK(&gRaHdc[phyId].lock);
    HDC_SESSION session = gRaHdc[phyId].session;
    if (session == NULL) {
        hccp_err("[send_recv][pkt]session is NULL!, phyId(%u)", phyId);
        RA_PTHREAD_MUTEX_UNLOCK(&gRaHdc[phyId].lock);
        return -EINVAL;
    }

    // check last recv status
    if (gRaHdc[phyId].lastRecvStatus == -DRV_ERROR_WAIT_TIMEOUT) {
        ret = -DRV_ERROR_WAIT_TIMEOUT;
        goto alloc_msg_err;
    }

    ret = RA_HDC_OPS.allocMsg(session, &pMsgSnd, 1);
    if (ret != 0) {
        hccp_err("[send_recv][pkt]HDC alloc msg err ret(%d) phyId(%u)", ret, phyId);
        goto alloc_msg_err;
    }
    ret = HdcSendRecvPktSend(pMsgSnd, (char *)sendRcvBuf, inBufLen, session, &pMsgRcv);
    if (ret) {
        hccp_err("[send_recv][pkt]HDC pkt send err ret(%d) phyId(%u)", ret, phyId);
        goto msg_err;
    }
    ret = HdcSendRecvPktRecv(session, phyId, pMsgRcv, &recvBuf, &rcvBufLen);
    if (ret == -DRV_ERROR_WAIT_TIMEOUT) {
        hccp_err("[send_recv][pkt]HDC broken, pkt recv err ret(%d) phyId(%u)", ret, phyId);
        gRaHdc[phyId].lastRecvStatus = ret;
        goto msg_err;
    }
    if (ret) {
        hccp_err("[send_recv][pkt]HDC pkt recv err ret(%d) phyId(%u)", ret, phyId);
        goto msg_err;
    }
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdc[phyId].lock);
    recvMsgHead = (struct MsgHead *)recvBuf;
    ret = HdcSendRecvPktRecvCheck(rcvBufLen, outBufLen, recvMsgHead, pMsgRcv);
    CHK_PRT_RETURN(ret, hccp_err("[send_recv][pkt]HDC pkt recv check ret(%d) phyId(%u)", ret, phyId), ret);

    ret = memcpy_s(sendRcvBuf, outBufLen, recvBuf, rcvBufLen);
    if (ret) {
        hccp_err("[send_recv][pkt]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId);
        RA_HDC_OPS.freeMsg(pMsgRcv);
        return -ESAFEFUNC;
    }

    RA_HDC_OPS.freeMsg(pMsgRcv);
    return 0;
msg_err:
    RA_HDC_OPS.freeMsg(pMsgSnd);
alloc_msg_err:
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdc[phyId].lock);
    return ret;
}

void MsgHeadBuildUp(struct MsgHead *pSendRcvHead, unsigned int opcode, unsigned int reqId, unsigned int msgDataLen,
    pid_t hostTgid)
{
    pSendRcvHead->opcode = opcode;
    pSendRcvHead->ret = 0;
    pSendRcvHead->asyncReqId = reqId;
    pSendRcvHead->msgDataLen = msgDataLen;
    pSendRcvHead->hostTgid = hostTgid;

    return;
}

STATIC int MsgHeadCheck(struct MsgHead *sendRcvHead, unsigned int opcode, int rsRet, unsigned int msgDataLen)
{
    unsigned int ret;

    /* return rs real return value */
    if (sendRcvHead->ret < rsRet) {
        return sendRcvHead->ret;
    }

    ret = (sendRcvHead->opcode != opcode) || (sendRcvHead->msgDataLen != msgDataLen);

    return (ret != 0 ? -EPERM : 0);
}

int RaHdcProcessMsg(unsigned int opcode, unsigned int phyId, char *data, unsigned int dataSize)
{
    pid_t hostTgid = DlDrvDeviceGetBareTgid();
    void *sendRcvBuf = NULL;
    unsigned int sendRcvLen;
    int ret;

    if (gRaHdc[phyId].restoreFlag != 0) {
        return 0;
    }

    sendRcvLen = sizeof(struct MsgHead) + dataSize;
    CHK_PRT_RETURN(data == NULL, hccp_err("[process][ra_hdc_msg]data is NULL. phyId(%u)", phyId), -EINVAL);
    sendRcvBuf = (void *)calloc(sendRcvLen, sizeof(char));
    CHK_PRT_RETURN(sendRcvBuf == NULL, hccp_err("[process][ra_hdc_msg]send_rcv_buf calloc failed. phyId(%u)", phyId),
        -ENOMEM);
    MsgHeadBuildUp(sendRcvBuf, opcode, 0, dataSize, hostTgid);
    ret = memcpy_s(sendRcvBuf + sizeof(struct MsgHead), sendRcvLen - sizeof(struct MsgHead), data, dataSize);
    if (ret) {
        hccp_err("[process][ra_hdc_msg]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId);
        ret = -ESAFEFUNC;
        goto out;
    }

    ret = HdcSendRecvPkt(phyId, sendRcvBuf, sendRcvLen, sendRcvLen);
    if (ret) {
        hccp_err("[process][ra_hdc_msg]hdc_send_recv_pkt opcode(%u) failed ret(%d) phyId(%u)", opcode, ret, phyId);
        goto out;
    }

    ret = MsgHeadCheck(sendRcvBuf, opcode, 0, dataSize);
    // opcode RA_RS_SOCKET_RECV not to print EAGAIN to avoid log flush
    if ((ret != 0) && (ret != -EAGAIN || opcode != RA_RS_SOCKET_RECV)) {
        /* maybe has retry return value, record warning log */
        hccp_warn_socket("message head check unsuccessful, ret[%d] phyId[%u]", ret, phyId);
    }

    if (memcpy_s(data, dataSize, sendRcvBuf + sizeof(struct MsgHead), dataSize)) {
        hccp_err("[process][ra_hdc_msg]memcpy_s failed. dest size(%d) ret(%d) phyId(%u)", dataSize, ret, phyId);
        ret = -ESAFEFUNC;
    }
out:
    free(sendRcvBuf);
    sendRcvBuf = NULL;
    return ret;
}

int HdcAsyncSendPkt(struct HdcAsyncInfo *asyncInfo, unsigned int phyId, void *sendBuf, unsigned int sendLen)
{
    struct drvHdcMsg *pMsgSnd = NULL;
    HDC_SESSION session = NULL;
    int ret = 0;

    RA_PTHREAD_MUTEX_LOCK(&asyncInfo->sendMutex);
    session = asyncInfo->session;
    if (session == NULL) {
        RA_PTHREAD_MUTEX_UNLOCK(&asyncInfo->sendMutex);
        hccp_err("[async][send_pkt]session is NULL!, phyId(%u)", phyId);
        return -EINVAL;
    }

    ret = RA_HDC_OPS.allocMsg(session, &pMsgSnd, 1);
    if (ret != 0) {
        hccp_err("[async][send_pkt]HDC alloc msg err ret(%d) phyId(%u)", ret, phyId);
        goto alloc_msg_err;
    }

    ret = RA_HDC_OPS.addMsgBuffer(pMsgSnd, sendBuf, (int)sendLen);
    if (ret != 0) {
        hccp_err("[async][send_pkt]HDC add msg buffer err ret(%d) phyId(%u)", ret, phyId);
        goto msg_err;
    }

    ret = RA_HDC_OPS.send(session, pMsgSnd, RA_HDC_WAIT_TIMEOUT, RA_HDC_RECV_SEND_TIMEOUT);
    if (ret != 0) {
        hccp_err("[async][send_pkt]HDC send err ret(%d) phyId(%u)", ret, phyId);
        goto msg_err;
    }

msg_err:
    RA_HDC_OPS.freeMsg(pMsgSnd);
alloc_msg_err:
    RA_PTHREAD_MUTEX_UNLOCK(&asyncInfo->sendMutex);
    return ret;
}

STATIC int HdcAsyncPrepareRecvPkt(struct HdcAsyncInfo *asyncInfo, unsigned int phyId, HDC_SESSION *session,
    struct drvHdcMsg **pMsgRcv)
{
    int ret = 0;

    *session = asyncInfo->session;
    if (*session == NULL) {
        hccp_err("[async][recv_pkt]session is NULL!, phyId(%u)", phyId);
        return -EINVAL;
    }

    // check last recv status
    if (RaHdcIsBroken(asyncInfo->lastRecvStatus)) {
        return asyncInfo->lastRecvStatus;
    }

    ret = RA_HDC_OPS.allocMsg(*session, pMsgRcv, 1);
    if (ret != 0) {
        hccp_err("[async][recv_pkt]HDC alloc msg err ret(%d) phyId(%u)", ret, phyId);
        return ret;
    }

    return 0;
}

int HdcAsyncRecvPkt(struct HdcAsyncInfo *asyncInfo, unsigned int phyId, void *recvBuf, unsigned int *recvLen)
{
    struct drvHdcMsg *pMsgRcv = NULL;
    HDC_SESSION session = NULL;
    int recvBufCnt = 0;
    char *rcvBuf = NULL;
    int rcvLen = 0;
    int ret = 0;

    ret = HdcAsyncPrepareRecvPkt(asyncInfo, phyId, &session, &pMsgRcv);
    if (ret != 0) {
        goto session_err;
    }

    ret = RA_HDC_OPS.recv(session, pMsgRcv, MAX_HDC_DATA, RA_HDC_WAIT_TIMEOUT, &recvBufCnt, RA_HDC_RECV_SEND_TIMEOUT);
    // occur hdc time out when async session was closed before async request done
    if (ret == -DRV_ERROR_WAIT_TIMEOUT) {
        hccp_run_warn("[async][recv_pkt]HDC recv timeout, phyId(%u)", phyId);
        goto msg_err;
    }

    if (ret != 0) {
        hccp_err("[async][recv_pkt]HDC recv err ret(%d) phyId(%u)", ret, phyId);
        asyncInfo->lastRecvStatus = ret;
        goto msg_err;
    }

    ret = RA_HDC_OPS.getMsgBuffer(pMsgRcv, 0, (char **)&rcvBuf, &rcvLen);
    if (ret != 0 || rcvLen <= 0) {
        hccp_err("[async][recv_pkt]HDC get_msg_buffer err ret(%d) phyId(%u) rcv_len(%d)", ret, phyId, rcvLen);
        goto msg_err;
    }

    ret = memcpy_s(recvBuf, *recvLen, rcvBuf, (unsigned int)rcvLen);
    if (ret != 0) {
        hccp_err("[async][recv_pkt]memcpy_s failed, ret(%d) phyId(%u)", ret, phyId);
        RA_HDC_OPS.freeMsg(pMsgRcv);
        return -ESAFEFUNC;
    }

    *recvLen = (unsigned int)rcvLen;
    RA_HDC_OPS.freeMsg(pMsgRcv);
    return 0;

msg_err:
    RA_HDC_OPS.freeMsg(pMsgRcv);
session_err:
    return ret;
}

int RaHdcGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int *interfaceVersion)
{
    int num = sizeof(gRaInterfaceInfoList) / sizeof(gRaInterfaceInfoList[0]);
    int i;

    CHK_PRT_RETURN(interfaceVersion == NULL || phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][ra_interface_version]para invalid! interface_version is NULL or phyId(%u) >= [%u]", phyId,
            RA_MAX_PHY_ID_NUM),
        -EINVAL);

    *interfaceVersion = 0;
    for (i = 0; i < num; i++) {
        if (gRaInterfaceInfoList[i].opcode == interfaceOpcode) {
            *interfaceVersion = gRaInterfaceInfoList[i].version;
            break;
        }
    }
    return 0;
}

STATIC int RaHdcGetOpcodeVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int *interfaceVersion)
{
    union OpGetVersionData versionInfo = {0};
    int ret;

    versionInfo.txData.opcode = interfaceOpcode;

    ret = RaHdcProcessMsg(RA_RS_GET_INTERFACE_VERSION, phyId, (char *)&versionInfo, sizeof(union OpGetVersionData));
    CHK_PRT_RETURN(ret,
        hccp_err("[get][ra_hdc_interface_version]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    *interfaceVersion = versionInfo.rxData.version;
    return 0;
}

void RaHdcGetAllOpcodeVersion(unsigned int phyId)
{
    int num = sizeof(gRaInterfaceInfoList) / sizeof(gRaInterfaceInfoList[0]);
    int ret;
    int i;

    for (i = 0; i < num; i++) {
        ret = RaHdcGetOpcodeVersion(phyId, gRaInterfaceInfoList[i].opcode, &gRaInterfaceInfoList[i].version);
        if (ret != 0) {
            hccp_warn_rma("ra_hdc_get_opcode_version unsuccessful, ret[%d], opcode[%d]", ret,
                gRaInterfaceInfoList[i].opcode);
            continue;
        }
    }

    return;
}

bool RaHdcHasCapability(unsigned int phyId, unsigned int capability)
{
    unsigned int roceVersion = 0;
    unsigned int netVersion = 0;
    bool hasCapability = false;

    // return value is ignored as this function is guaranteed to succeed
    (void)RaHdcGetInterfaceVersion(phyId, RA_RS_GET_ROCE_API_VERSION, &roceVersion);
    (void)RaHdcGetInterfaceVersion(phyId, RA_RS_GET_NET_API_VERSION, &netVersion);

    hasCapability = RaRsHasCapability(capability, roceVersion, netVersion);
    if (!hasCapability) {
        hccp_run_info("phy_id:%u capability:%u roceVersion:0x%x netVersion:0x%x hasCapability:%d RA_CAP_INVALID:%u",
            phyId, capability, roceVersion, netVersion, hasCapability, RA_CAP_INVALID);
    }
    return hasCapability;
}

STATIC int RaHdcSendPid(unsigned int phyId, struct ProcessRaSign pRaSign)
{
    union OpSetPidData setPidData = {0};
    int ret;

    setPidData.txData.pid = pRaSign.tgid;
    setPidData.txData.phyId = phyId;

    ret = strcpy_s(setPidData.txData.pidSign, PROCESS_RA_SIGN_LENGTH, pRaSign.sign);
    CHK_PRT_RETURN(ret, hccp_err("[send][ra_hdc_pid]Invalid pid sign, ret(%d)", ret), -ESAFEFUNC);

    ret = RaHdcProcessMsg(RA_RS_SET_PID, phyId, (char *)&setPidData, sizeof(union OpSetPidData));
    CHK_PRT_RETURN(ret, hccp_err("[send][ra_hdc_pid]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    return 0;
}

STATIC int RaHdcInitApart(unsigned int phyId, unsigned int *logicId)
{
    int ret;
    ret = DlDrvDeviceGetIndexByPhyId(phyId, logicId);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_apart]get logic id failed(%d), phyId(%u)", ret, phyId), -ENODEV);

    ret = pthread_mutex_init(&gRaHdc[phyId].lock, NULL);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_apart]pthread_mutex_init failed, ret(%d) phyId(%u)", ret, phyId),
        -ESYSFUNC);
    return 0;
}

STATIC int RaHdcInitSessionConnectEx(int peerNode, int peerDevid, unsigned int phyId, HDC_SESSION *session)
{
    struct halQueryDevpidInfo info = {0};
    pid_t devPid;
    int ret;

    info.hostpid = getpid();
    info.devid = (unsigned int)peerDevid;
    info.proc_type = DEVDRV_PROCESS_HCCP;
    ret = DlHalQueryDevPid(info, &devPid);
    if (ret != 0) {
        hccp_err("[init][ra_hdc]hdc dl_hal_query_dev_pid failed ret(%d) peer_devid(%d) phyId(%u)", ret, peerDevid,
            phyId);
        return ret;
    }

    return RA_HDC_OPS.sessionConnectEx(peerNode, peerDevid, devPid, gRaHdc[phyId].client, session);
}

int RaHdcInitSession(int peerNode, int peerDevid, unsigned int phyId, int hdcType, HDC_SESSION *session)
{
    if (hdcType == HDC_SERVICE_TYPE_RDMA_V2) {
        return RaHdcInitSessionConnectEx(peerNode, peerDevid, phyId, session);
    }

    // default hdc type: HDC_SERVICE_TYPE_RDMA
    return RA_HDC_OPS.sessionConnect(peerNode, peerDevid, gRaHdc[phyId].client, session);
}

void RaHdcDeinitSession(HDC_SESSION *session)
{
    if (*session == NULL) {
        return;
    }

    (void)RA_HDC_OPS.sessionClose(*session);
    *session = NULL;
    return;
}

int RaHdcSetSessionReference(HDC_SESSION *session)
{
    return RA_HDC_OPS.setSessionReference(*session);
}

int RaHdcInit(struct RaInitConfig *cfg, struct ProcessRaSign pRaSign)
{
    unsigned int logicId = RA_MAX_PHY_ID_NUM;
    unsigned int phyId = cfg->phyId;
    int hdcType = cfg->hdcType;
    int ret;

    ret = DlHalInit();
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc]dl_hal_init failed, ret = %d", ret), ret);

    ret = RaHdcInitApart(phyId, &logicId);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc]ra_hdc_init_apart failed, ret(%d)", ret), ret);

    hccp_run_info("hdc init start! logic id is %u, phy id is %u, hdcType is %d", logicId, phyId, hdcType);

    if (gRaHdc[phyId].session == NULL) {
        // maxSessionNum 2U include: sync & async session
        ret = RA_HDC_OPS.clientCreate(&gRaHdc[phyId].client, 2U, hdcType, 0);
        if (ret != 0) {
            hccp_err("[init][ra_hdc]hdc client create failed, hccp not up ret(%d) phyId(%u)", ret, phyId);
            goto HDC_ERR;
        }
        ret = RaHdcInitSession(0, (int)logicId, phyId, hdcType, &gRaHdc[phyId].session);
        if (ret != 0) {
            hccp_err("[init][ra_hdc]hdc session_connect failed ret(%d) logic_id(%u) phyId(%u)", ret, logicId, phyId);
            goto CONN_ERR;
        }
        ret = RA_HDC_OPS.setSessionReference(gRaHdc[phyId].session);
        if (ret != 0) {
            hccp_err("[init][ra_hdc]hdc set_session_reference failed, ret(%d) phyId(%u)", ret, phyId);
            goto SESS_ERR;
        }
    } else {
        hccp_warn_socket("hdc session for phyId[%u] already existed", phyId);
        return -EEXIST;
    }

    ret = RaHdcSendPid(phyId, pRaSign);
    if (ret) {
        hccp_err("[init][ra_hdc]set pid for phyId(%u) failed, ret(%d), hostTgid(%d)", phyId, ret, pRaSign.tgid);
        goto SESS_ERR;
    }

    ret = RaHdcLiteInitCqeErrInfo(phyId);
    if (ret) {
        hccp_err("[init][ra_hdc]ra_hdc_lite_init_cqe_err_info failed, ret(%d)", ret);
        goto SESS_ERR;
    }

    hccp_run_info("hdc init OK! phyId[%u]", phyId);

    return 0;
SESS_ERR:
    RA_HDC_OPS.sessionClose(gRaHdc[phyId].session);
    gRaHdc[phyId].session = NULL;
CONN_ERR:
    RA_HDC_OPS.clientDestroy(gRaHdc[phyId].client);
    gRaHdc[phyId].client = NULL;
HDC_ERR:
    pthread_mutex_destroy(&gRaHdc[phyId].lock);
    return -EPERM;
}

int RaHdcGetTlsEnable(unsigned int phyId, bool *tlsEnable)
{
    union OpGetTlsEnableData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    ret = RaHdcProcessMsg(RA_RS_GET_TLS_ENABLE, phyId, (char *)&opData, sizeof(union OpGetTlsEnableData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[get][tls_enable]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    *tlsEnable = opData.rxData.tlsEnable;
    return ret;
}

STATIC int RaHdcSessionClose(unsigned int phyId)
{
    union OpHdcCloseData hdcCloseData = {0};
    int ret;

    hdcCloseData.txData.phyId = phyId;

    ret = RaHdcProcessMsg(RA_RS_HDC_SESSION_CLOSE, phyId, (char *)&hdcCloseData, sizeof(union OpHdcCloseData));
    CHK_PRT_RETURN(ret, hccp_err("[close][ra_hdc_session]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    return 0;
}

STATIC int RaHdcClientDeinit(unsigned int phyId)
{
    int ret;

    gRaHdc[phyId].startDeinit = 1;
    ret = RaHdcSessionClose(phyId);
    if (ret) {
        hccp_err("[deinit][ra_hdc]close hdc session failed, ret(%d) phyId(%u)", ret, phyId);
    }
    RaHdcDeinitSession(&gRaHdc[phyId].snapshotSession);
    RaHdcDeinitSession(&gRaHdc[phyId].session);

    ret = RA_HDC_OPS.clientDestroy(gRaHdc[phyId].client);
    if (ret) {
        hccp_err("[deinit][ra_hdc]hdc client_destroy failed, ret(%d) phyId(%u)", ret, phyId);
    }
    gRaHdc[phyId].client = NULL;
    gRaHdc[phyId].startDeinit = 0;

    return ret;
}

int RaHdcDeinit(struct RaInitConfig *cfg)
{
    unsigned int phyId = cfg->phyId;
    int ret;

    hccp_run_info("hdc deinit start! phyId[%u] restore_flag[%u]", phyId, gRaHdc[phyId].restoreFlag);

    RaHdcLiteDeinitCqeErrInfo(phyId);

    ret = RaHdcClientDeinit(phyId);
    if (ret != 0) {
        hccp_err("[deinit][ra_hdc]client deinit failed! ret(%d) phyId(%u)", ret, phyId);
    }

    ret = pthread_mutex_destroy(&gRaHdc[phyId].lock);
    if (ret != 0) {
        hccp_err("[deinit][ra_hdc]pthread_mutex_destroy failed! ret(%d) phyId(%u)", ret, phyId);
        ret = -ESYSFUNC;
    }

    CHK_PRT_RETURN(ret != 0, hccp_err("[deinit][ra_hdc]hdc deinit failed! phyId(%u)", phyId), ret);
    DlHalDeinit();

    (void)memset_s(&gRaHdc[phyId], sizeof(gRaHdc[phyId]), 0, sizeof(gRaHdc[phyId]));

    hccp_run_info("hdc deinit OK! phyId[%u]", phyId);
    return 0;
}

STATIC int RaHdcGetValidCqeErrInfo(struct CqeErrInfo *outInfo, struct CqeErrInfo info0, struct CqeErrInfo info1)
{
    int ret;

    if (info0.status != 0 && info1.status != 0) {
        if (timercmp((&info0.time), (&info1.time), <)) {
            ret = memcpy_s(outInfo, sizeof(struct CqeErrInfo), &info0, sizeof(struct CqeErrInfo));
        } else {
            ret = memcpy_s(outInfo, sizeof(struct CqeErrInfo), &info1, sizeof(struct CqeErrInfo));
        }
    } else {
        if (info0.status == 0) {
            ret = memcpy_s(outInfo, sizeof(struct CqeErrInfo), &info1, sizeof(struct CqeErrInfo));
        } else {
            ret = memcpy_s(outInfo, sizeof(struct CqeErrInfo), &info0, sizeof(struct CqeErrInfo));
        }
    }

    if (ret) {
        hccp_err("memcpy_s failed, ret(%d)", ret);
        return -ESAFEFUNC;
    }

    return 0;
}

int RaHdcGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info)
{
    int ret;
    struct CqeErrInfo opCqeInfo = {0};
    union OpGetCqeErrInfoData cqeErrInfoData;

    RaHdcLiteGetCqeErrInfo(phyId, &opCqeInfo);

    RA_PTHREAD_MUTEX_LOCK(&gRaHdc[phyId].lock);
    HDC_SESSION session = gRaHdc[phyId].session;
    if (session == NULL) {
        RA_PTHREAD_MUTEX_UNLOCK(&gRaHdc[phyId].lock);
        return 0;
    }
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdc[phyId].lock);

    ret = memset_s(&cqeErrInfoData, sizeof(cqeErrInfoData), 0, sizeof(cqeErrInfoData));
    CHK_PRT_RETURN(ret, hccp_err("[init]memset_s failed ret(%d)", ret), -ESAFEFUNC);
    ret = RaHdcProcessMsg(RA_RS_GET_CQE_ERR_INFO, phyId, (char *)&cqeErrInfoData, sizeof(union OpGetCqeErrInfoData));
    CHK_PRT_RETURN(ret, hccp_err("ra hdc message process failed ret(%d)", ret), ret);

    return RaHdcGetValidCqeErrInfo(info, opCqeInfo, cqeErrInfoData.rxData.info);
}

STATIC int RaHdcSessionSaveSnapshot(unsigned int phyId, enum SaveSnapshotAction action)
{
    int ret = 0;

    RA_PTHREAD_MUTEX_LOCK(&gRaHdc[phyId].lock);
    if (action == SAVE_SNAPSHOT_ACTION_PRE_PROCESSING && gRaHdc[phyId].session != NULL) {
        gRaHdc[phyId].snapshotSession = gRaHdc[phyId].session;
        gRaHdc[phyId].session = NULL;
    } else if (action == SAVE_SNAPSHOT_ACTION_POST_PROCESSING && gRaHdc[phyId].session == NULL) {
        gRaHdc[phyId].session = gRaHdc[phyId].snapshotSession;
        gRaHdc[phyId].snapshotSession = NULL;
    } else {
        hccp_err("duplicate or incorrect order calls are not allowed, phyId[%u] action[%d]", phyId, action);
        ret = -EPERM;
    }
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdc[phyId].lock);

    return ret;
}

int RaHdcSaveSnapshot(unsigned int phyId, enum SaveSnapshotAction action)
{
    int ret;

    ret = RaHdcAsyncSaveSnapshot(phyId, action);
    CHK_PRT_RETURN(ret != 0, hccp_err("ra_hdc_async_save_snapshot failed ret[%d]", ret), ret);
    ret = RaHdcSessionSaveSnapshot(phyId, action);
    CHK_PRT_RETURN(ret != 0, hccp_err("ra_hdc_session_save_snapshot failed ret[%d]", ret), ret);

    return ret;
}

STATIC void RaHdcSessionRestoreSnapshot(unsigned int phyId)
{
    RA_PTHREAD_MUTEX_LOCK(&gRaHdc[phyId].lock);
    gRaHdc[phyId].restoreFlag = 1;
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdc[phyId].lock);
}

int RaHdcRestoreSnapshot(unsigned int phyId)
{
    int ret;

    ret = RaHdcAsyncRestoreSnapshot(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("ra_hdc_async_restore_snapshot failed ret[%d]", ret), ret);
    RaHdcSessionRestoreSnapshot(phyId);

    return ret;
}

int RaHdcGetSecRandom(unsigned int phyId, unsigned int *value)
{
    union OpGetSecRandomData opData = {0};
    int ret;

    ret = RaHdcProcessMsg(RA_RS_GET_SEC_RANDOM, phyId, (char *)&opData, sizeof(union OpGetSecRandomData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[get][sec_random]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    *value = opData.rxData.value;
    return ret;
}

int RaHdcGetHccnCfg(unsigned int phyId, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    union OpGetHccnCfgData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.key = key;
    ret = RaHdcProcessMsg(RA_RS_GET_HCCN_CFG, phyId, (char *)&opData, sizeof(union OpGetHccnCfgData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[get][hccn_cfg]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    ret = memcpy_s(value, *valueLen, opData.rxData.value, opData.rxData.valueLen);
    CHK_PRT_RETURN(ret != 0, hccp_err("[get][hccn_cfg]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    *valueLen = opData.rxData.valueLen;
    return ret;
}

int RaHdcSetDebugConfig(unsigned int phyId, u64 debugConfig)
{
    union OpSetDebugConfigData opData = {0};
    unsigned int interfaceVersion = 0;
    int ret;

    HccpSetDebugConfig(debugConfig);
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_SET_DEBUG_CONFIG, &interfaceVersion);
    CHK_PRT_RETURN(ret != 0 || interfaceVersion < RA_RS_OPCODE_BASE_VERSION,
        hccp_run_warn("[set][debug_config]not support to set debug config, ret(%d), interfaceVersion(%u), phyId(%u)",
            ret, interfaceVersion, phyId),
        -EPROTONOSUPPORT);

    opData.txData.debugConfig = debugConfig;
    ret = RaHdcProcessMsg(RA_RS_SET_DEBUG_CONFIG, phyId, (char *)&opData, sizeof(union OpSetDebugConfigData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[set][debug_config]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);
    return ret;
}
