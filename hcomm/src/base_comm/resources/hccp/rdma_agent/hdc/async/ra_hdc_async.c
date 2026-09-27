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
#include <sys/prctl.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "dl_hal_function.h"
#include "ra.h"
#include "ra_async.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "ra_hdc.h"
#include "ra_hdc_socket.h"
#include "ra_hdc_async_socket.h"
#include "ra_hdc_ctx.h"
#include "ra_hdc_async_ctx.h"
#include "ra_hdc_async.h"

struct HdcAsyncInfo gRaHdcAsync[RA_MAX_PHY_ID_NUM] = {0};

struct RaAsyncOpHandle gRaAsyncOpHandle[] = {
    {RA_RS_GET_EID_BY_IP, RDMA_OP, RaHdcAsyncHandleGetEidByIp, sizeof(union OpGetEidByIpData)},
    {RA_RS_GET_IP_BY_EID, RDMA_OP, RaHdcAsyncHandleGetIpByEid, sizeof(union OpGetIpByEidData)},
    {RA_RS_LMEM_REG, RDMA_OP, RaHdcAsyncHandleLmemRegister, sizeof(union OpLmemRegInfoData)},
    {RA_RS_LMEM_UNREG, RDMA_OP, NULL, sizeof(union OpLmemUnregInfoData)},
    {RA_RS_CTX_QP_CREATE, RDMA_OP, RaHdcAsyncHandleQpCreate, sizeof(union OpCtxQpCreateData)},
    {RA_RS_CTX_QP_DESTROY, RDMA_OP, NULL, sizeof(union OpCtxQpDestroyData)},
    {RA_RS_CTX_QP_IMPORT, RDMA_OP, RaHdcAsyncHandleQpImport, sizeof(union OpCtxQpImportData)},
    {RA_RS_CTX_QP_UNIMPORT, RDMA_OP, NULL, sizeof(union OpCtxQpUnimportData)},
    {RA_RS_GET_TP_INFO_LIST, RDMA_OP, RaHdcAsyncHandleTpInfoList, sizeof(union OpGetTpInfoListData)},
    {RA_RS_GET_TP_ATTR, RDMA_OP, RaHdcAsyncHandleGetTpAttr, sizeof(union OpGetTpAttrData)},
    {RA_RS_SET_TP_ATTR, RDMA_OP, NULL, sizeof(union OpSetTpAttrData)},
    {RA_RS_CTX_QP_DESTROY_BATCH, RDMA_OP, RaHdcAsyncHandleQpDestroyBatch, sizeof(union OpCtxQpDestroyBatchData)},
    {RA_RS_SOCKET_SEND, SOCKET_OP, RaHdcAsyncHandleSocketSend, sizeof(union OpSocketSendData)},
    {RA_RS_SOCKET_RECV, SOCKET_OP, RaHdcAsyncHandleSocketRecv, sizeof(union OpSocketRecvData)},
    {RA_RS_SOCKET_LISTEN_START, SOCKET_OP, RaHdcAsyncHandleSocketListenStart, sizeof(union OpSocketListenData)},
    {RA_RS_SOCKET_LISTEN_STOP, SOCKET_OP, NULL, sizeof(union OpSocketListenData)},
    {RA_RS_SOCKET_CONN, SOCKET_OP, NULL, sizeof(union OpSocketConnectData)},
    {RA_RS_SOCKET_CLOSE, SOCKET_OP, RaHdcAsyncHandleSocketBatchClose, sizeof(union OpSocketCloseData)},
    {RA_RS_HDC_SESSION_CLOSE, OTHERS, NULL, sizeof(union OpHdcCloseData)},
};

STATIC struct RaAsyncOpHandle *RaHdcIsAsyncOp(unsigned int opcode)
{
    int num = sizeof(gRaAsyncOpHandle) / sizeof(gRaAsyncOpHandle[0]);
    int i;

    for (i = 0; i < num; i++) {
        if (gRaAsyncOpHandle[i].opcode == (enum OpType)opcode) {
            return &gRaAsyncOpHandle[i];
        }
    }
    return NULL;
}

STATIC void HdcAsyncHandlePrivData(struct RaRequestHandle *reqHandle)
{
    if (reqHandle->opHandle->privDataHandle == NULL) {
        return;
    }

    reqHandle->opHandle->privDataHandle(reqHandle);
}

STATIC void HdcAsyncSetRequest(struct RaRequestHandle *reqHandle, unsigned int reqId, struct RaAsyncOpHandle *opHandle,
    unsigned int phyId, unsigned int dataSize)
{
    reqHandle->reqId = reqId;
    reqHandle->opHandle = opHandle;
    reqHandle->phyId = phyId;
    reqHandle->dataSize = dataSize;
}

STATIC int HdcAsyncGetRequest(struct HdcAsyncInfo *asyncInfo, unsigned int reqId, struct RaRequestHandle **reqHandle)
{
    struct RaRequestHandle *reqTmp2 = NULL;
    struct RaRequestHandle *reqTmp = NULL;

    // no need to use lock: req_id always exist in current req_list(the data is always sent before it is received)
    RA_LIST_GET_HEAD_ENTRY(reqTmp, reqTmp2, &asyncInfo->reqList, list, struct RaRequestHandle);
    for (; (&reqTmp->list) != &asyncInfo->reqList;
         reqTmp = reqTmp2, reqTmp2 = list_entry(reqTmp2->list.next, struct RaRequestHandle, list)) {
        if (reqTmp->reqId == reqId) {
            *reqHandle = reqTmp;
            return 0;
        }
    }
    *reqHandle = NULL;
    return -ENODEV;
}

STATIC void HdcAsyncSetReqDone(struct RaRequestHandle *reqHandle, unsigned int phyId, int ret)
{
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].rspMutex);
    RaListAddTail(&reqHandle->list, &gRaHdcAsync[phyId].rspList);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].rspMutex);
    reqHandle->opRet = (ret != 0) ? ret : reqHandle->opRet;
    reqHandle->isDone = true;
}

STATIC void RaHwAsyncSetConnectStatus(unsigned int phyId, unsigned int connectStatus)
{
    gRaHdcAsync[phyId].connectStatus = connectStatus;
}

STATIC bool HdcAsyncIsMsgValid(unsigned int phyId, struct MsgHead *recvMsgHead, unsigned int recvLen,
    struct RaRequestHandle **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    int ret;

    // check recv_len and get req_handle
    CHK_PRT_RETURN(recvLen < sizeof(struct MsgHead),
        hccp_run_warn("[async][ra_hdc_recv]recv_len[%u] < [%lu] is invalid", recvLen, sizeof(struct MsgHead)), false);
    ret = HdcAsyncGetRequest(&gRaHdcAsync[phyId], recvMsgHead->asyncReqId, &reqHandleTmp);
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_run_warn("[async][ra_hdc_recv]req_id[%u] invalid, ret[%d], opcode[%u]", recvMsgHead->asyncReqId, ret,
            recvMsgHead->opcode),
        false);

    // del req_handle from req_list
    RaListDel(&reqHandleTmp->list);

    // opcode RA_RS_HDC_SESSION_CLOSE
    if (recvMsgHead->opcode == RA_RS_HDC_SESSION_CLOSE) {
        RaHwAsyncSetConnectStatus(phyId, HDC_UNCONNECTED);
        hccp_dbg("opcode[%u] req_id[%u] phyId[%u]", recvMsgHead->opcode, reqHandleTmp->reqId, phyId);
        reqHandleTmp->isDone = true;
        return false;
    }

    // need to check op_data size and recv data size
    if ((reqHandleTmp->dataSize != recvMsgHead->msgDataLen) ||
        (recvMsgHead->msgDataLen + (unsigned int)sizeof(struct MsgHead)) != recvLen) {
        hccp_run_warn("[async][ra_hdc_recv]opcode[%u] data_size[%u] msg_data_len[%u] mismatch or recv_len[%u] mismatch",
            recvMsgHead->opcode, reqHandleTmp->dataSize, recvMsgHead->msgDataLen, recvLen);
        HdcAsyncSetReqDone(reqHandleTmp, phyId, -EINVAL);
        return false;
    }

    *reqHandle = reqHandleTmp;
    return true;
}

STATIC int HdcAsyncAddResponse(unsigned int phyId, void *recvBuf, unsigned int recvLen)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    struct MsgHead *recvMsgHead = NULL;
    int ret = 0;

    recvMsgHead = (struct MsgHead *)recvBuf;
    // check recv msg: req_id, opcode, msg_data_len and get req_handle
    if (!HdcAsyncIsMsgValid(phyId, recvMsgHead, recvLen, &reqHandleTmp)) {
        return -EINVAL;
    }

    //  handle recv msg
    reqHandleTmp->recvBuf = (void *)calloc(recvMsgHead->msgDataLen, sizeof(char));
    if (reqHandleTmp->recvBuf == NULL) {
        hccp_err("[async][ra_hdc_recv]calloc recv_buf failed, msgDataLen[%u] reqId[%u] opcode[%u]",
            recvMsgHead->msgDataLen, recvMsgHead->asyncReqId, recvMsgHead->opcode);
        ret = -ENOMEM;
        goto out;
    }
    (void)memcpy_s(reqHandleTmp->recvBuf, recvMsgHead->msgDataLen, recvBuf + sizeof(struct MsgHead),
        recvMsgHead->msgDataLen);
    reqHandleTmp->recvLen = recvMsgHead->msgDataLen;
    reqHandleTmp->opRet = recvMsgHead->ret;
    HdcAsyncHandlePrivData(reqHandleTmp);

out:
    HdcAsyncSetReqDone(reqHandleTmp, phyId, ret);
    return ret;
}

static void HdcAsyncDelReqHandle(struct RaRequestHandle *reqHandle, pthread_mutex_t *mutex)
{
    RA_PTHREAD_MUTEX_LOCK(mutex);
    RaListDel(&reqHandle->list);
    RA_PTHREAD_MUTEX_UNLOCK(mutex);
    if (reqHandle->recvBuf != NULL && reqHandle->recvLen != 0) {
        free(reqHandle->recvBuf);
        reqHandle->recvBuf = NULL;
        reqHandle->recvLen = 0;
    }
    // async api return failed, free corresponding handle
    if (reqHandle->opRet != 0 && reqHandle->privHandle != NULL) {
        free(reqHandle->privHandle);
        reqHandle->privHandle = NULL;
    }
    free(reqHandle);
    reqHandle = NULL;
    return;
}

void HdcAsyncDelResponse(struct RaRequestHandle *reqHandle)
{
    HdcAsyncDelReqHandle(reqHandle, &gRaHdcAsync[reqHandle->phyId].rspMutex);
}

int RaHdcSendMsgAsync(unsigned int opcode, unsigned int phyId, char *data, unsigned int dataSize,
    struct RaRequestHandle *reqHandle)
{
    struct RaAsyncOpHandle *opHandleTmp = NULL;
    unsigned int asyncReqId = 0;
    void *sendBuf = NULL;
    unsigned int sendLen;
    pid_t hostTgid;
    int ret;

    if (gRaHdcAsync[phyId].restoreFlag != 0) {
        return 0;
    }

    CHK_PRT_RETURN(RaHdcIsBroken(gRaHdcAsync[phyId].lastRecvStatus),
        hccp_err("[async][ra_hdc_send]HDC broken, phyId(%u)", phyId), gRaHdcAsync[phyId].lastRecvStatus);
    opHandleTmp = RaHdcIsAsyncOp(opcode);
    CHK_PRT_RETURN(opHandleTmp == NULL, hccp_err("[async][ra_hdc_send]opcode[%u] invalid", opcode), -EINVAL);

    hostTgid = gRaHdcAsync[phyId].hostTgid;
    sendLen = (unsigned int)sizeof(struct MsgHead) + dataSize;
    sendBuf = (void *)calloc(sendLen, sizeof(char));
    CHK_PRT_RETURN(sendBuf == NULL,
        hccp_err("[async][ra_hdc_send]calloc send_buf failed. phyId(%u) opcode(%u)", phyId, opcode), -ENOMEM);

    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].reqMutex);
    asyncReqId = gRaHdcAsync[phyId].reqId;
    gRaHdcAsync[phyId].reqId++;
    HdcAsyncSetRequest(reqHandle, asyncReqId, opHandleTmp, phyId, dataSize);
    // make sure request has been added to req_list
    RaListAddTail(&reqHandle->list, &gRaHdcAsync[phyId].reqList);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].reqMutex);

    MsgHeadBuildUp(sendBuf, opcode, asyncReqId, dataSize, hostTgid);
    ret = memcpy_s(sendBuf + sizeof(struct MsgHead), sendLen - sizeof(struct MsgHead), data, dataSize);
    if (ret != 0) {
        hccp_err("[async][ra_hdc_send]memcpy_s failed, ret(%d) phyId(%u) opcode(%u)", ret, phyId, opcode);
        ret = -ESAFEFUNC;
        goto out;
    }

    ret = HdcAsyncSendPkt(&gRaHdcAsync[phyId], phyId, sendBuf, sendLen);
    if (ret != 0) {
        hccp_err("[async][ra_hdc_send]hdc_async_send_pkt opcode(%u) failed ret(%d) phyId(%u)", opcode, ret, phyId);
        goto out;
    }

    free(sendBuf);
    sendBuf = NULL;
    return 0;

out:
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].reqMutex);
    RaListDel(&reqHandle->list);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].reqMutex);
    free(sendBuf);
    sendBuf = NULL;
    return ret;
}

STATIC int RaHdcAsyncSessionConnect(struct RaInitConfig *cfg)
{
    union OpAsyncHdcConnectData asyncData = {0};
    unsigned int interfaceVersion = 0;
    int ret;

    ret = RaHdcGetInterfaceVersion(cfg->phyId, RA_RS_ASYNC_HDC_SESSION_CONNECT, &interfaceVersion);
    // compatibility issue: ignore return value and set base version queue size to MAX_POOL_QUEUE_SIZE_V1
    if (ret != 0 || interfaceVersion <= RA_RS_OPCODE_BASE_VERSION) {
        asyncData.txData.queueSize = MAX_POOL_QUEUE_SIZE_V1;
    } else {
        asyncData.txData.queueSize = MAX_POOL_QUEUE_SIZE;
    }
    asyncData.txData.phyId = cfg->phyId;
    asyncData.txData.threadNum = RA_POOL_THREAD_NUM;
    ret = RaHdcProcessMsg(RA_RS_ASYNC_HDC_SESSION_CONNECT, cfg->phyId, (char *)&asyncData,
        sizeof(union OpAsyncHdcConnectData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_async]ra hdc message process failed ret[%d] phyId[%u]", ret, cfg->phyId), ret);
    return ret;
}

STATIC int RaHdcAsyncSessionClose(unsigned int phyId)
{
    union OpAsyncHdcCloseData asyncData = {0};
    struct RaRequestHandle *reqHandle = NULL;
    union OpHdcCloseData opData = {0};
    int timeout = RA_THREAD_TRY_TIME;
    int ret;

    if (gRaHdcAsync[phyId].restoreFlag != 0) {
        return 0;
    }

    // close async session
    opData.txData.phyId = phyId;
    reqHandle = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandle == NULL, hccp_err("[deinit][ra_hdc_async]calloc req_handle failed, phyId[%u]", phyId),
        -ENOMEM);
    ret = RaHdcSendMsgAsync(RA_RS_HDC_SESSION_CLOSE, phyId, (char *)&opData, sizeof(union OpHdcCloseData), reqHandle);
    if (ret != 0) {
        hccp_err("[deinit][ra_hdc_async]hdc async send message failed ret[%d] phyId[%u]", ret, phyId);
        free(reqHandle);
        reqHandle = NULL;
        return ret;
    }

    // wait request done until time out: RA_THREAD_TRY_TIME * RA_THREAD_SLEEP_TIME us
    while (!reqHandle->isDone && timeout > 0) {
        usleep(RA_THREAD_SLEEP_TIME);
        timeout--;
    }
    if (timeout <= 0) {
        hccp_warn_init("[deinit][ra_hdc_async]hdc async session close timeout:%d phyId[%u]", timeout, phyId);
    }
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].reqMutex);
    HdcAsyncDelResponse(reqHandle);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].reqMutex);

    // destroy async recv thread and work thread pool
    ret = RaHdcProcessMsg(RA_RS_ASYNC_HDC_SESSION_CLOSE, phyId, (char *)&asyncData, sizeof(union OpAsyncHdcCloseData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_hdc_async]ra hdc message process failed ret[%d] phyId[%u]", ret, phyId), ret);
    return ret;
}

STATIC void RaHwAsyncHdcServerInit(void *arg)
{
    struct RaInitConfig cfg = {0};
    int ret;

    if (arg == NULL) {
        hccp_err("[init][ra_hdc_async]arg is NULL");
        return;
    }

    cfg = *(struct RaInitConfig *)arg;
    ret = pthread_detach(pthread_self());
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]pthread detach failed ret %d", ret);
        return;
    }

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_async_server", 0, 0, 0);

    // trigger server to connect session
    ret = RaHdcAsyncSessionConnect(&cfg);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]ra_hdc_async_session_connect failed ret[%d] phyId[%u]", ret, cfg.phyId);
        return;
    }
    return;
}

STATIC void RaHwAsyncHdcClientInit(void *arg)
{
    struct RaInitConfig cfg = {0};
    unsigned int logicId = 0;
    unsigned int phyId = 0;
    int ret;

    if (arg == NULL) {
        hccp_err("[init][ra_hdc_async]arg is NULL");
        return;
    }

    cfg = *(struct RaInitConfig *)arg;
    ret = pthread_detach(pthread_self());
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]pthread detach failed ret %d", ret);
        return;
    }

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_async_client", 0, 0, 0);

    phyId = cfg.phyId;
    ret = DlDrvDeviceGetIndexByPhyId(phyId, &logicId);
    if (ret != 0) {
        hccp_err("get logic id failed(%d), phyId(%u)", ret, phyId);
        return;
    }

    ret = RaHdcInitSession(0, (int)logicId, phyId, cfg.hdcType, &gRaHdcAsync[phyId].session);
    if (ret != 0) {
        hccp_err("hdc session_connect failed ret(%d) phyId(%u)", ret, phyId);
        return;
    }

    ret = RaHdcSetSessionReference(&gRaHdcAsync[phyId].session);
    if (ret != 0) {
        goto set_ref_err;
    }

    RaHwAsyncSetConnectStatus(phyId, HDC_CONNECTED);
    return;

set_ref_err:
    RaHdcDeinitSession(&gRaHdcAsync[phyId].session);
    return;
}

STATIC void RaHwAsyncSetThreadStatus(unsigned int phyId, unsigned int threadStatus)
{
    gRaHdcAsync[phyId].threadStatus = threadStatus;
}

STATIC void RaHwAsyncDelList(struct RaListHead *head, pthread_mutex_t *mutex)
{
    struct RaRequestHandle *reqNext = NULL;
    struct RaRequestHandle *reqCur = NULL;

    RA_LIST_GET_HEAD_ENTRY(reqCur, reqNext, head, list, struct RaRequestHandle);
    for (; (&reqCur->list) != head;
         reqCur = reqNext, reqNext = list_entry(reqNext->list.next, struct RaRequestHandle, list)) {
        HdcAsyncDelReqHandle(reqCur, mutex);
    }
}

STATIC void RaHwAsyncHdcClientDeinit(unsigned int phyId)
{
    int tryAgain = HDC_TRY_TIME;

    // destroy thread
    RaHwAsyncSetThreadStatus(phyId, THREAD_DESTROYING);
    while ((gRaHdcAsync[phyId].threadStatus != THREAD_HALT) && (tryAgain != 0)) {
        usleep(HDC_USLEEP_TIME);
        tryAgain--;
    }
    if (tryAgain <= 0) {
        hccp_warn_others("hdc async message thread quit timeout");
    }

    // close session
    RaHwAsyncSetConnectStatus(phyId, HDC_UNCONNECTED);
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].sendMutex);
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].recvMutex);
    RaHdcDeinitSession(&gRaHdcAsync[phyId].snapshotSession);
    RaHdcDeinitSession(&gRaHdcAsync[phyId].session);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].recvMutex);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].sendMutex);

    RaHwAsyncDelList(&gRaHdcAsync[phyId].reqList, &gRaHdcAsync[phyId].reqMutex);
    RaHwAsyncDelList(&gRaHdcAsync[phyId].rspList, &gRaHdcAsync[phyId].rspMutex);
}

STATIC int RaHdcAsyncMutexInit(unsigned int phyId)
{
    int ret = 0;

    ret = pthread_mutex_init(&gRaHdcAsync[phyId].sendMutex, NULL);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]pthread_mutex_init send_mutex failed ret(%d) phyId(%u)", ret, phyId);
        return -ESYSFUNC;
    }
    ret = pthread_mutex_init(&gRaHdcAsync[phyId].recvMutex, NULL);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]pthread_mutex_init recv_mutex failed ret(%d) phyId(%u)", ret, phyId);
        ret = -ESYSFUNC;
        goto recv_mutex_fail;
    }
    ret = pthread_mutex_init(&gRaHdcAsync[phyId].reqMutex, NULL);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]pthread_mutex_init req_mutex failed ret(%d) phyId(%u)", ret, phyId);
        ret = -ESYSFUNC;
        goto req_mutex_fail;
    }
    ret = pthread_mutex_init(&gRaHdcAsync[phyId].rspMutex, NULL);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]pthread_mutex_init rsp_mutex failed ret(%d) phyId(%u)", ret, phyId);
        ret = -ESYSFUNC;
        goto rsp_mutex_fail;
    }

    return 0;

rsp_mutex_fail:
    (void)pthread_mutex_destroy(&gRaHdcAsync[phyId].reqMutex);
req_mutex_fail:
    (void)pthread_mutex_destroy(&gRaHdcAsync[phyId].recvMutex);
recv_mutex_fail:
    (void)pthread_mutex_destroy(&gRaHdcAsync[phyId].sendMutex);
    return ret;
}

STATIC void RaHdcAsyncMutexDeinit(unsigned int phyId)
{
    (void)pthread_mutex_destroy(&gRaHdcAsync[phyId].rspMutex);
    (void)pthread_mutex_destroy(&gRaHdcAsync[phyId].reqMutex);
    (void)pthread_mutex_destroy(&gRaHdcAsync[phyId].recvMutex);
    (void)pthread_mutex_destroy(&gRaHdcAsync[phyId].sendMutex);
}

STATIC int RaHdcAsyncInitSession(struct RaInitConfig *cfg)
{
    unsigned int phyId = cfg->phyId;
    int timeout = RA_THREAD_TRY_TIME;
    pthread_t serverTidp;
    pthread_t clientTidp;
    int ret = 0;

    CHK_PRT_RETURN(gRaHdcAsync[phyId].session != NULL,
        hccp_warn_init("hdc async session for phyId[%u] already existed", phyId), -EEXIST);

    // server will be blocked, use a thread to trigger server to accept
    ret = pthread_create(&serverTidp, NULL, (void *)RaHwAsyncHdcServerInit, cfg);
    CHK_PRT_RETURN(ret != 0, hccp_err("Create async_hdc_server_init pthread failed, ret(%d)", ret), -ESYSFUNC);

    // client will be blocked, use a thread to trigger client to connect
    ret = pthread_create(&clientTidp, NULL, (void *)RaHwAsyncHdcClientInit, cfg);
    CHK_PRT_RETURN(ret != 0, hccp_err("Create async_hdc_client_init pthread failed, ret(%d)", ret), -ESYSFUNC);

    // will block until time out: RA_CONNECT_TRY_TIME * RA_THREAD_SLEEP_TIME us
    timeout = RA_CONNECT_TRY_TIME;
    while (gRaHdcAsync[phyId].connectStatus != HDC_CONNECTED && timeout > 0) {
        usleep(RA_THREAD_SLEEP_TIME);
        timeout--;
    }
    if (gRaHdcAsync[phyId].connectStatus == HDC_UNCONNECTED || timeout <= 0) {
        hccp_err("HDC async connect timeout, connectStatus %d, timeout %d, total_timeout %d(us)",
            gRaHdcAsync[phyId].connectStatus, timeout, RA_CONNECT_TRY_TIME * RA_THREAD_SLEEP_TIME);
        return -ETIMEDOUT;
    }

    gRaHdcAsync[phyId].hostTgid = DlDrvDeviceGetBareTgid();
    ret = RaHdcAsyncMutexInit(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("ra_hdc_async_mutex_init failed, ret(%d), phyId(%u)", ret, phyId), ret);

    RA_INIT_LIST_HEAD(&gRaHdcAsync[phyId].reqList);
    RA_INIT_LIST_HEAD(&gRaHdcAsync[phyId].rspList);
    return 0;
}

STATIC void HdcAsyncHandleRecvBroken(struct HdcAsyncInfo *asyncInfo)
{
    struct RaRequestHandle *reqNext = NULL;
    struct RaRequestHandle *reqCurr = NULL;

    if (!RaHdcIsBroken(asyncInfo->lastRecvStatus)) {
        return;
    }

    RA_PTHREAD_MUTEX_LOCK(&asyncInfo->reqMutex);
    RA_LIST_GET_HEAD_ENTRY(reqCurr, reqNext, &asyncInfo->reqList, list, struct RaRequestHandle);
    for (; (&reqCurr->list) != &asyncInfo->reqList;
         reqCurr = reqNext, reqNext = list_entry(reqNext->list.next, struct RaRequestHandle, list)) {
        RaListDel(&reqCurr->list);
        HdcAsyncSetReqDone(reqCurr, reqCurr->phyId, asyncInfo->lastRecvStatus);
    }
    RA_PTHREAD_MUTEX_UNLOCK(&asyncInfo->reqMutex);
}

STATIC void *RaHdcRecvMsgAsync(void *arg)
{
    unsigned int phyId = *(unsigned int *)arg;
    unsigned int recvLen = MAX_HDC_MSG_DATA;
    void *recvBuf = NULL;
    int ret;

    // free memory after using arg
    free(arg);
    arg = NULL;

    ret = pthread_detach(pthread_self());
    CHK_PRT_RETURN(ret, hccp_err("pthread detach failed ret %d, phyId %u", ret, phyId), NULL);

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_ra_async", 0, 0, 0);

    hccp_info("[async][ra_hdc_recv]thread[%d] phyId[%u] enter", getpid(), phyId);
    RaHwAsyncSetThreadStatus(phyId, THREAD_RUNNING);
    recvBuf = (void *)calloc(recvLen, sizeof(char));
    CHK_PRT_RETURN(recvBuf == NULL, hccp_err("[async][ra_hdc_recv]calloc recv_buf failed. phyId(%u)", phyId), NULL);

    while (1) {
        if (gRaHdcAsync[phyId].threadStatus == THREAD_DESTROYING) {
            break;
        }
        RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].recvMutex);
        if (gRaHdcAsync[phyId].connectStatus != HDC_CONNECTED) {
            RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].recvMutex);
            usleep(THREAD_SLEEP_TIME);
            continue;
        }

        if (RaListEmpty(&gRaHdcAsync[phyId].reqList)) {
            RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].recvMutex);
            usleep(THREAD_SLEEP_TIME);
            continue;
        }

        recvLen = MAX_HDC_MSG_DATA;
        ret = HdcAsyncRecvPkt(&gRaHdcAsync[phyId], phyId, recvBuf, &recvLen);
        if (ret != 0) {
            RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].recvMutex);
            HdcAsyncHandleRecvBroken(&gRaHdcAsync[phyId]);
            continue;
        }
        RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].recvMutex);

        RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].reqMutex);
        (void)HdcAsyncAddResponse(phyId, recvBuf, recvLen);
        RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].reqMutex);
    }

    hccp_info("[async][ra_hdc_recv]thread[%d] phyId[%u] is out", getpid(), phyId);
    RaHwAsyncSetThreadStatus(phyId, THREAD_HALT);
    free(recvBuf);
    recvBuf = NULL;
    return NULL;
}

STATIC int RaHdcAsyncInitRecvThread(unsigned int phyId)
{
    unsigned int *phyIdTmp = NULL;
    int ret = 0;

    phyIdTmp = (unsigned int *)calloc(1, sizeof(unsigned int));
    CHK_PRT_RETURN(phyIdTmp == NULL, hccp_err("calloc phy_id_tmp failed, errno(%d)", errno), -ENOMEM);
    *phyIdTmp = phyId;

    // create a thread to recv msg from server
    ret = pthread_create(&gRaHdcAsync[phyId].tid, NULL, RaHdcRecvMsgAsync, (void *)phyIdTmp);
    if (ret != 0) {
        hccp_err("Create ra_hdc_recv_msg_async pthread failed, ret(%d)", ret);
        goto err;
    }

    return 0;

err:
    free(phyIdTmp);
    phyIdTmp = NULL;
    return ret;
}

int RaHdcInitAsync(struct RaInitConfig *cfg)
{
    unsigned int interfaceVersion = 0;
    int ret = 0;

    CHK_PRT_RETURN(!cfg->enableHdcAsync, hccp_info_init("[init][ra_hdc_async]no need to init async hdc session"), 0);

    ret = RaHdcGetInterfaceVersion(cfg->phyId, RA_RS_ASYNC_HDC_SESSION_CONNECT, &interfaceVersion);
    // normal case: driver not support to or no need to init async hdc session
    CHK_PRT_RETURN(ret != 0 || interfaceVersion < RA_RS_OPCODE_BASE_VERSION,
        hccp_run_warn("[init][ra_hdc_async]not support to init async hdc session, ret(%d), interfaceVersion(%u)", ret,
            interfaceVersion),
        0);

    ret = RaHdcAsyncInitSession(cfg);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_async]ra_hdc_async_init_session failed, ret(%d) phyId(%u)", ret, cfg->phyId), ret);

    ret = RaHdcAsyncInitRecvThread(cfg->phyId);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_async]ra_hdc_async_init_recv_thread failed, ret(%d) phyId(%u)", ret, cfg->phyId);
        goto err;
    }

    return 0;

err:
    RaHdcAsyncMutexDeinit(cfg->phyId);
    return -ESRCH;
}

int RaHdcDeinitAsync(unsigned int phyId)
{
    int ret;

    hccp_run_info("hdc deinit async start! phyId[%u] restore_flag[%u]", phyId, gRaHdcAsync[phyId].restoreFlag);

    CHK_PRT_RETURN(gRaHdcAsync[phyId].session == NULL && gRaHdcAsync[phyId].restoreFlag == 0,
        hccp_warn_socket("hdc async session for phyId[%u] is NULL", phyId), -ENODEV);

    // close server session
    ret = RaHdcAsyncSessionClose(phyId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_hdc_async]ra_hdc_async_session_close failed ret[%d] phyId[%u]", ret, phyId), ret);

    // close client session & deinit client resources
    RaHwAsyncHdcClientDeinit(phyId);

    RaHdcAsyncMutexDeinit(phyId);

    (void)memset_s(&gRaHdcAsync[phyId], sizeof(gRaHdcAsync[phyId]), 0, sizeof(gRaHdcAsync[phyId]));

    return 0;
}

int RaHdcAsyncSaveSnapshot(unsigned int phyId, enum SaveSnapshotAction action)
{
    int ret = 0;

    if (gRaHdcAsync[phyId].threadStatus == THREAD_HALT) {
        return 0;
    }

#ifndef HNS_ROCE_LLT
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].sendMutex);
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].recvMutex);
    if (action == SAVE_SNAPSHOT_ACTION_PRE_PROCESSING && gRaHdcAsync[phyId].session != NULL) {
        RaHwAsyncSetConnectStatus(phyId, HDC_UNCONNECTED);
        gRaHdcAsync[phyId].snapshotSession = gRaHdcAsync[phyId].session;
        gRaHdcAsync[phyId].session = NULL;
    } else if (action == SAVE_SNAPSHOT_ACTION_POST_PROCESSING && gRaHdcAsync[phyId].session == NULL) {
        RaHwAsyncSetConnectStatus(phyId, HDC_CONNECTED);
        gRaHdcAsync[phyId].session = gRaHdcAsync[phyId].snapshotSession;
        gRaHdcAsync[phyId].snapshotSession = NULL;
    } else {
        hccp_err("duplicate or incorrect order calls are not allowed, phyId[%u] action[%d]", phyId, action);
        ret = -EPERM;
    }
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].recvMutex);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].sendMutex);
#endif
    return ret;
}

int RaHdcAsyncRestoreSnapshot(unsigned int phyId)
{
    int ret = 0;

    if (gRaHdcAsync[phyId].threadStatus == THREAD_HALT) {
        return 0;
    }

#ifndef HNS_ROCE_LLT
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].sendMutex);
    RA_PTHREAD_MUTEX_LOCK(&gRaHdcAsync[phyId].recvMutex);
    if (gRaHdcAsync[phyId].connectStatus != HDC_UNCONNECTED) {
        hccp_err("incorrect order calls are not allowed, phyId[%u] connectStatus[%u]", phyId,
            gRaHdcAsync[phyId].connectStatus);
        ret = -EPERM;
    } else {
        gRaHdcAsync[phyId].restoreFlag = 1;
    }
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].recvMutex);
    RA_PTHREAD_MUTEX_UNLOCK(&gRaHdcAsync[phyId].sendMutex);
#endif
    return ret;
}

void RaHdcDeinitAsyncAll(void)
{
    unsigned int phyId = 0;

    for (phyId = 0; phyId < RA_MAX_PHY_ID_NUM; phyId++) {
        if (gRaHdcAsync[phyId].session == NULL || gRaHdcAsync[phyId].threadStatus != THREAD_RUNNING) {
            continue;
        }

        (void)RaHdcDeinitAsync(phyId);
    }
}
